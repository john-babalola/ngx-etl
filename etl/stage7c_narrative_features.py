"""
Stage 7c: narrative features requiring BERTopic/AfriSenti outputs.

  f9  topic entropy            (Shannon entropy over BERTopic cluster
                                 proportions, per trading day)
  f10 dominant cluster share    (max proportion in a single topic)
  f12 sentiment skew            (skewness of the bullish-probability
                                 distribution, per trading day)

Corpus: FULL Nigeria-relevant, in-window set (duplicates INCLUDED).
This matches BERTopic's transform step, which deliberately ran on the
full set so that f9/f10 reflect topic concentration including amplified
volume - a viral, heavily-retweeted announcement should visibly pull a
day's topic distribution toward one cluster, not be invisible to it.

Both tweet_topics and tweet_sentiment (77,111 rows each) draw from the
same eligible population - Nigeria-relevant, in-window, non-truncated -
so no additional corpus decision is needed here; it inherits the split
already established in stages 6/7a/7b.

Writes a standalone narrative_features_f9_f10_f12 table, then merges
into ngx.premarket_features (built by stage7b), producing one complete
table with f1-f12.

Instrumented with timing prints for consistency with stage 7b, though
this stage is lighter (no per-tweet pairwise computation, no graph
construction) and should complete quickly.
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import entropy, skew
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()


def load_topics_by_day():
    """Topic assignments joined to their pre-market trading day."""
    query = f"""
    SELECT
      w.trading_day,
      t.tweet_id,
      t.topic_id
    FROM `{PROJECT}.{DATASET}.tweet_topics` t
    JOIN `{PROJECT}.{DATASET}.tweet_premarket_window` w USING (tweet_id)
    """
    print("Submitting topics query to BigQuery...", flush=True)
    t0 = time.time()
    job = client.query(query)
    result = job.result()
    print(f"Query completed in {time.time() - t0:.1f}s, downloading rows...", flush=True)

    t1 = time.time()
    df = result.to_dataframe(bqstorage_client=bqstorage_client)
    print(
        f"Downloaded {len(df):,} topic assignments in {time.time() - t1:.1f}s "
        f"across {df['trading_day'].nunique()} trading days",
        flush=True,
    )
    return df


def load_sentiment_by_day():
    """Bullish-probability scores joined to their pre-market trading day."""
    query = f"""
    SELECT
      w.trading_day,
      s.tweet_id,
      s.bullish_prob
    FROM `{PROJECT}.{DATASET}.tweet_sentiment` s
    JOIN `{PROJECT}.{DATASET}.tweet_premarket_window` w USING (tweet_id)
    """
    print("Submitting sentiment query to BigQuery...", flush=True)
    t0 = time.time()
    job = client.query(query)
    result = job.result()
    print(f"Query completed in {time.time() - t0:.1f}s, downloading rows...", flush=True)

    t1 = time.time()
    df = result.to_dataframe(bqstorage_client=bqstorage_client)
    print(
        f"Downloaded {len(df):,} sentiment scores in {time.time() - t1:.1f}s "
        f"across {df['trading_day'].nunique()} trading days",
        flush=True,
    )
    return df


def compute_topic_features(day, day_df):
    """f9: Shannon entropy over topic proportions (BERTopic label -1 is
    the outlier/no-topic bucket; excluded from the proportion calculation,
    treating it as 'no assignable narrative' rather than a topic).
    f10: share of tweets in the single largest assigned topic."""
    assigned = day_df[day_df["topic_id"] != -1]

    if len(assigned) < 5:
        return {
            "trading_day": day,
            "f9_topic_entropy": np.nan,
            "f10_dominant_cluster_share": np.nan,
            "n_topic_assigned": len(assigned),
            "n_topic_total": len(day_df),
        }

    counts = assigned["topic_id"].value_counts()
    proportions = counts / counts.sum()

    f9 = float(entropy(proportions, base=np.e))
    f10 = float(proportions.max())

    return {
        "trading_day": day,
        "f9_topic_entropy": f9,
        "f10_dominant_cluster_share": f10,
        "n_topic_assigned": len(assigned),
        "n_topic_total": len(day_df),
    }


def compute_sentiment_skew(day, day_df):
    """f12: skewness of the bullish-probability distribution. scipy
    returns 0.0 for a degenerate (constant) distribution rather than
    NaN; n_sentiment_scored is carried alongside so this is auditable
    in Ch4 if it occurs on a thin day."""
    probs = day_df["bullish_prob"].dropna()

    if len(probs) < 3:
        return {
            "trading_day": day,
            "f12_sentiment_skew": np.nan,
            "n_sentiment_scored": len(probs),
        }

    f12 = float(skew(probs))
    return {
        "trading_day": day,
        "f12_sentiment_skew": f12,
        "n_sentiment_scored": len(probs),
    }


def main():
    topics_df = load_topics_by_day()
    sentiment_df = load_sentiment_by_day()

    print("Computing f9/f10 per trading day...", flush=True)
    topic_rows = [
        compute_topic_features(str(day), day_df)
        for day, day_df in topics_df.groupby("trading_day")
    ]

    print("Computing f12 per trading day...", flush=True)
    sentiment_rows = [
        compute_sentiment_skew(str(day), day_df)
        for day, day_df in sentiment_df.groupby("trading_day")
    ]

    topic_result = pd.DataFrame(topic_rows)
    sentiment_result = pd.DataFrame(sentiment_rows)

    narrative_features = topic_result.merge(sentiment_result, on="trading_day", how="outer")

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    standalone_table = PROJECT + "." + DATASET + ".narrative_features_f9_f10_f12"
    client.load_table_from_dataframe(
        narrative_features, standalone_table, job_config=job_config
    ).result()
    print("Loaded " + str(len(narrative_features)) + " rows to " + standalone_table, flush=True)

    print("Merging into premarket_features...", flush=True)
    existing = client.query(
        "SELECT * FROM `" + PROJECT + "." + DATASET + ".premarket_features`"
    ).result().to_dataframe(bqstorage_client=bqstorage_client)

    existing["trading_day"] = existing["trading_day"].astype(str)
    narrative_features["trading_day"] = narrative_features["trading_day"].astype(str)

    complete = existing.merge(narrative_features, on="trading_day", how="left")

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    combined_table = PROJECT + "." + DATASET + ".premarket_features"
    client.load_table_from_dataframe(complete, combined_table, job_config=job_config).result()

    print(
        "Merged into "
        + combined_table
        + ": "
        + str(len(complete))
        + " rows, "
        + str(complete.shape[1])
        + " columns",
        flush=True,
    )

    print("\n--- Feature summary (f1-f12) ---")
    feature_cols = [c for c in complete.columns if c.startswith("f")]
    print(complete[feature_cols].describe())

    print("\n--- Topic assignment coverage ---")
    print(complete[["trading_day", "n_topic_assigned", "n_topic_total"]].to_string())


if __name__ == "__main__":
    main()
