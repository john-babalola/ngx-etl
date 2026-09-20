"""
Stage 7d: rolling intraday features (g1-g6).

  g1  rolling degree centralisation   (Freeman, 60-min trailing window)
  g2  rolling mean pairwise cosine    (60-min trailing, deduplicated)
  g3  rolling tweet volume            (log1p of tweet count, 60-min trailing)
  g4  lagged interval log return      r(t-1)
  g5  lagged interval log return      r(t-2)
  g6  lagged interval log return      r(t-3)

Corpus split mirrors the pre-market feature design (stage 7b), applied
per-interval rather than per-day:

  g1, g3 : FULL Nigeria-relevant set, duplicates INCLUDED. Retweet edges
           and raw volume are what these measure; excluding duplicates
           would remove the amplification/activity signal.

  g2     : DEDUPLICATED set only, for the same reason f7/f8 are
           deduplicated - measuring independent-voice convergence, not
           repetition of one viral tweet.

Missingness handling follows Ch3 3.6.4: windows below MIN_TWEETS are
marked missing (NaN) with an explicit indicator column, rather than
silently computed on an unstably small sample.

g4-g6 require no tweet data at all - they are lagged values of the
already-computed interval_log_return in interval_spine, and are
included here for completeness of the g1-g6 family in one output
table, joined against the trading-day-ordered interval sequence.

Writes ngx.intraday_features, one row per interval (matches
interval_spine's grain), ready to join to premarket_features (which is
constant within a trading day) to build the full model-ready table in
stage 8.

Instrumented with timing prints and periodic checkpointing, consistent
with stage 7b/7c, since this involves ~1,029 per-interval computations.
"""

import time
import numpy as np
import pandas as pd
import networkx as nx
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"
RANDOM_SEED = 42
SAMPLE_SIZE = 150  # smaller cap than f7/f8 - intraday windows are much smaller
MIN_TWEETS = 5     # below this, mark missing rather than compute an unstable value
CHECKPOINT_PATH = "intraday_features_checkpoint.parquet"
CHECKPOINT_EVERY = 100  # intervals

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()


def load_interval_tweets():
    """Every (interval, tweet) pair from the rolling 60-minute lookback,
    restricted to the same eligible population as the pre-market
    features: Nigeria-relevant, non-truncated. Duplicate flag is carried
    through rather than filtered here, same pattern as stage 7b."""
    query = f"""
    SELECT
      w.interval_start,
      w.trading_day,
      w.lookback_minutes_available,
      t.tweet_id,
      t.author_hash,
      t.reply_to_hash,
      t.retweet_of_id,
      t.quote_of_id,
      t.retweet_count,
      t.reply_count,
      t.like_count,
      COALESCE(d.is_near_duplicate, FALSE) AS is_near_duplicate,
      e.embedding
    FROM `{PROJECT}.{DATASET}.tweet_intraday_window` w
    JOIN `{PROJECT}.{DATASET}.tweets_nigeria_relevant` t USING (tweet_id)
    LEFT JOIN `{PROJECT}.{DATASET}.tweet_duplicates` d USING (tweet_id)
    LEFT JOIN `{PROJECT}.{DATASET}.tweet_embeddings` e USING (tweet_id)
    WHERE t.is_nigeria_relevant
      AND t.text_status != 'truncated'
    """
    print("Submitting interval-tweets query to BigQuery...", flush=True)
    t0 = time.time()
    job = client.query(query)
    result = job.result()
    print(f"Query completed in {time.time() - t0:.1f}s, downloading rows...", flush=True)

    t1 = time.time()
    df = result.to_dataframe(bqstorage_client=bqstorage_client, progress_bar_type="tqdm")
    print(
        f"Downloaded {len(df):,} interval-tweet pairs in {time.time() - t1:.1f}s "
        f"across {df['interval_start'].nunique():,} intervals",
        flush=True,
    )
    return df


def load_all_intervals():
    """Full interval spine (including intervals with zero tweets, so
    every row gets a g4-g6 value even where g1-g3 are missing)."""
    query = f"""
    SELECT trading_day, interval_start, interval_log_return
    FROM `{PROJECT}.{DATASET}.interval_spine`
    ORDER BY trading_day, interval_start
    """
    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorage_client)
    print(f"Loaded {len(df):,} rows from interval_spine for lag construction", flush=True)
    return df


def build_graph(window_df):
    G = nx.DiGraph()
    G.add_nodes_from(window_df["author_hash"].dropna().unique())

    for _, row in window_df.iterrows():
        src = row["author_hash"]
        weight = np.log1p((row["retweet_count"] or 0) + (row["like_count"] or 0))

        targets = []
        if pd.notna(row["retweet_of_id"]):
            targets.append(str(row["retweet_of_id"]))
        if pd.notna(row["quote_of_id"]):
            targets.append(str(row["quote_of_id"]))
        if pd.notna(row["reply_to_hash"]):
            targets.append(row["reply_to_hash"])

        for dst in targets:
            if dst != src:
                G.add_node(dst)
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += weight
                else:
                    G.add_edge(src, dst, weight=weight)

    return G


def freeman_degree_centralisation(G):
    n = G.number_of_nodes()
    if n < 3:
        return np.nan
    in_deg = dict(G.in_degree())
    c_max = max(in_deg.values()) if in_deg else 0
    numerator = sum(c_max - c for c in in_deg.values())
    denominator = (n - 1) * (n - 2)
    return numerator / denominator if denominator > 0 else np.nan


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def compute_g1_g3(window_df):
    """g1: rolling degree centralisation, full set (duplicates included).
    g3: rolling log1p tweet volume, full set."""
    n_tweets = len(window_df)
    g3 = float(np.log1p(n_tweets))

    if n_tweets < MIN_TWEETS:
        return np.nan, g3, True  # g1, g3, is_missing_g1

    G = build_graph(window_df)
    g1 = freeman_degree_centralisation(G)
    return g1, g3, False


def compute_g2(window_df, rng):
    """g2: rolling mean pairwise cosine similarity, deduplicated subset."""
    dedup = window_df[~window_df["is_near_duplicate"]].dropna(subset=["embedding"])

    if len(dedup) < 2:
        return np.nan, True  # g2, is_missing_g2

    if len(dedup) > SAMPLE_SIZE:
        idx = rng.choice(len(dedup), size=SAMPLE_SIZE, replace=False)
        sample = dedup.iloc[idx]
    else:
        sample = dedup

    vecs = np.stack(sample["embedding"].apply(np.array).values)
    n = len(vecs)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            sims.append(cosine_sim(vecs[i], vecs[j]))

    if not sims:
        return np.nan, True

    return float(np.mean(sims)), False


def main():
    tweets_df = load_interval_tweets()
    spine_df = load_all_intervals()
    rng = np.random.default_rng(RANDOM_SEED)

    # --- g4-g6: lagged interval log returns, computed via pandas groupby-shift,
    # ordered within trading_day, mirroring the SQL LAG() window function
    # but avoiding a second round trip to BigQuery ---
    spine_df = spine_df.sort_values(["trading_day", "interval_start"]).reset_index(drop=True)
    spine_df["g4_lag1_return"] = spine_df.groupby("trading_day")["interval_log_return"].shift(1)
    spine_df["g5_lag2_return"] = spine_df.groupby("trading_day")["interval_log_return"].shift(2)
    spine_df["g6_lag3_return"] = spine_df.groupby("trading_day")["interval_log_return"].shift(3)

    # --- Resume support ---
    try:
        done_df = pd.read_parquet(CHECKPOINT_PATH)
        done_intervals = set(done_df["interval_start"].astype(str))
        print(f"Resuming: {len(done_intervals)} intervals already checkpointed", flush=True)
    except FileNotFoundError:
        done_df = pd.DataFrame()
        done_intervals = set()

    all_rows = done_df.to_dict("records") if not done_df.empty else []

    grouped = list(tweets_df.groupby("interval_start"))
    intervals_with_tweets = {str(k) for k, _ in grouped}
    print(f"Processing {len(grouped)} intervals with at least one tweet...", flush=True)

    t_start = time.time()
    for i, (interval_start, window_df) in enumerate(grouped):
        interval_str = str(interval_start)
        if interval_str in done_intervals:
            continue

        g1, g3, missing_g1 = compute_g1_g3(window_df)
        g2, missing_g2 = compute_g2(window_df, rng)

        all_rows.append({
            "interval_start": interval_str,
            "n_tweets_in_window": len(window_df),
            "g1_rolling_degree_centralisation": g1,
            "g2_rolling_mean_cosine": g2,
            "g3_rolling_log_volume": g3,
            "g1_missing": missing_g1,
            "g2_missing": missing_g2,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0 or (i + 1) == len(grouped):
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(grouped) - (i + 1)) / rate if rate > 0 else 0
            print(
                f"  [{i + 1}/{len(grouped)}] checkpoint - "
                f"{rate:.1f} intervals/sec, ~{remaining:.0f}s remaining",
                flush=True,
            )
            pd.DataFrame(all_rows).to_parquet(CHECKPOINT_PATH)

    tweet_features = pd.DataFrame(all_rows)
    tweet_features["interval_start"] = pd.to_datetime(tweet_features["interval_start"], utc=True)

    # --- Merge onto the FULL interval spine, so intervals with zero
    # eligible tweets still get a row (g1-g3 NaN + missing flags TRUE,
    # g4-g6 populated from price data alone) ---
    spine_df["interval_start"] = pd.to_datetime(spine_df["interval_start"], utc=True)
    complete = spine_df.merge(tweet_features, on="interval_start", how="left")

    complete["g1_missing"] = complete["g1_missing"].fillna(True)
    complete["g2_missing"] = complete["g2_missing"].fillna(True)
    complete["g3_rolling_log_volume"] = complete["g3_rolling_log_volume"].fillna(0.0)
    complete["n_tweets_in_window"] = complete["n_tweets_in_window"].fillna(0)

    zero_tweet_intervals = (~complete["interval_start"].astype(str).isin(intervals_with_tweets)).sum()
    print(
        f"\n{zero_tweet_intervals} of {len(complete)} intervals had zero eligible "
        f"tweets in their 60-minute lookback (g1/g2 marked missing, g3=0)",
        flush=True,
    )

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    destination_table = PROJECT + "." + DATASET + ".intraday_features"
    client.load_table_from_dataframe(complete, destination_table, job_config=job_config).result()

    print("Loaded " + str(len(complete)) + " rows to " + destination_table, flush=True)
    print("\n--- Feature summary (g1-g6) ---")
    g_cols = [c for c in complete.columns if c.startswith("g")]
    print(complete[g_cols].describe())
    print("\n--- Missingness ---")
    print("g1 missing:", complete["g1_missing"].sum(), "/", len(complete))
    print("g2 missing:", complete["g2_missing"].sum(), "/", len(complete))


if __name__ == "__main__":
    main()
