"""
Stage 7b: pre-market network features (f1-f6) and narrative/engagement
features not requiring BERTopic/AfriSenti (f7, f8, f11).

Corpus split, resolved deliberately rather than applying one filter
uniformly:

  f1-f6 (network)   : FULL Nigeria-relevant set, duplicates INCLUDED.
                       Retweets are the edges this family measures -
                       excluding them would remove the amplification
                       signal the network features exist to capture.

  f7, f8 (narrative) : DEDUPLICATED set only (one representative per
                       near-duplicate cluster). These measure whether
                       INDEPENDENT voices converge; including raw
                       repetition would saturate similarity toward 1.0
                       whenever a single tweet goes viral, masking
                       everything else that day.

  f11 (engagement)   : FULL Nigeria-relevant set, duplicates INCLUDED.
                       This feature specifically targets top-decile
                       engagement content - excluding retweets would
                       remove exactly the tweets it is designed to find
                       (e.g. a viral CBN rate announcement).

BERTopic's fit (in Colab) uses the deduplicated set for the same reason
as f7/f8. Its transform (stage 7c) runs on the FULL set, so f9/f10
correctly reflect topic concentration including amplified volume.

Instrumented with timing prints and per-day checkpointing (to a local
parquet file) so a slow or interrupted run can be diagnosed and resumed
rather than rerun blind.
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
SAMPLE_SIZE = 300  # cap per window for f7/f8 pairwise cost
CHECKPOINT_PATH = "premarket_features_checkpoint.parquet"

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()


def load_tweets() -> pd.DataFrame:
    """All Nigeria-relevant, non-truncated tweets in the analysis window,
    joined to their pre-market window and embedding vector, carrying the
    near-duplicate flag through rather than filtering on it here."""
    query = f"""
    SELECT
      w.trading_day,
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
    FROM `{PROJECT}.{DATASET}.tweets_nigeria_relevant` t
    JOIN `{PROJECT}.{DATASET}.tweet_premarket_window` w USING (tweet_id)
    LEFT JOIN `{PROJECT}.{DATASET}.tweet_duplicates` d USING (tweet_id)
    LEFT JOIN `{PROJECT}.{DATASET}.tweet_embeddings` e USING (tweet_id)
    WHERE t.in_analysis_window
      AND t.is_nigeria_relevant
      AND t.text_status != 'truncated'
    """
    print("Submitting query to BigQuery...", flush=True)
    t0 = time.time()
    job = client.query(query)
    print(f"Query job started ({job.job_id}), waiting for completion...", flush=True)
    result = job.result()
    print(f"Query completed in {time.time() - t0:.1f}s, downloading rows...", flush=True)

    t1 = time.time()
    df = result.to_dataframe(bqstorage_client=bqstorage_client, progress_bar_type="tqdm")
    elapsed_dl = time.time() - t1
    print(
        f"Downloaded {len(df):,} rows in {elapsed_dl:.1f}s "
        f"({df['trading_day'].nunique()} trading days, "
        f"{df['is_near_duplicate'].sum():,} flagged near-duplicate)",
        flush=True,
    )
    return df


def build_graph(day_df: pd.DataFrame) -> nx.DiGraph:
    """Directed weighted graph for one trading day's pre-market window.
    Uses the FULL tweet set (duplicates/retweets included) since retweet
    edges are what this feature family measures."""
    G = nx.DiGraph()
    G.add_nodes_from(day_df["author_hash"].dropna().unique())

    for _, row in day_df.iterrows():
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


def freeman_centralisation(centrality, n):
    """General Freeman centralisation formula. Star-graph normaliser
    (n-1)*(n-2) for the undirected/general case."""
    if n < 3:
        return np.nan
    c_max = max(centrality.values()) if centrality else 0
    numerator = sum(c_max - c for c in centrality.values())
    denominator = (n - 1) * (n - 2)
    return numerator / denominator if denominator > 0 else np.nan


def gini_coefficient(values):
    if len(values) == 0 or np.mean(values) == 0:
        return np.nan
    n = len(values)
    diffs = np.abs(values[:, None] - values[None, :])
    return diffs.sum() / (2 * n**2 * np.mean(values))


def compute_graph_features(day, G):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_nodes < 3:
        return {
            "trading_day": day,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "f1_degree_centralisation": np.nan,
            "f2_betweenness_centralisation": np.nan,
            "f3_eigenvector_dispersion": np.nan,
            "f4_gini_indegree": np.nan,
            "f5_density": np.nan,
            "f6_lcc_ratio": np.nan,
        }

    in_deg = dict(G.in_degree())
    f1 = freeman_centralisation(in_deg, n_nodes)

    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    f2 = freeman_centralisation(betweenness, n_nodes)

    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
        f3 = float(np.var(list(eig.values())))
    except (nx.NetworkXException, np.linalg.LinAlgError):
        f3 = np.nan

    f4 = gini_coefficient(np.array(list(in_deg.values()), dtype=float))
    f5 = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else np.nan

    components = list(nx.weakly_connected_components(G))
    f6 = max(len(c) for c in components) / n_nodes if components else np.nan

    return {
        "trading_day": day,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "f1_degree_centralisation": f1,
        "f2_betweenness_centralisation": f2,
        "f3_eigenvector_dispersion": f3,
        "f4_gini_indegree": f4,
        "f5_density": f5,
        "f6_lcc_ratio": f6,
    }


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else np.nan


def compute_narrative_features(day, day_df, rng):
    """f7, f8: computed on the DEDUPLICATED subset only (one representative
    per near-duplicate cluster), so raw repetition of a viral tweet cannot
    mechanically saturate pairwise similarity toward 1.0."""
    dedup = day_df[~day_df["is_near_duplicate"]].dropna(subset=["embedding"])

    if len(dedup) < 2:
        return {
            "trading_day": day,
            "f7_mean_pairwise_cosine": np.nan,
            "f8_mean_embedding_variance": np.nan,
        }

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
    f7 = float(np.mean(sims)) if sims else np.nan
    f8 = float(np.mean(np.var(vecs, axis=0)))

    return {
        "trading_day": day,
        "f7_mean_pairwise_cosine": f7,
        "f8_mean_embedding_variance": f8,
    }


def compute_engagement_alignment(day, day_df):
    """f11: computed on the FULL tweet set (duplicates/retweets included).
    This feature specifically targets top-decile engagement, which is
    where viral/amplified content lives - excluding retweets here would
    remove the exact signal this feature is meant to capture."""
    valid = day_df.dropna(subset=["embedding"])
    if len(valid) < 10:
        return {"trading_day": day, "f11_engagement_alignment": np.nan}

    vecs = np.stack(valid["embedding"].apply(np.array).values)
    centroid = vecs.mean(axis=0)

    engagement = (
        valid["retweet_count"].fillna(0)
        + valid["like_count"].fillna(0)
        + valid["reply_count"].fillna(0)
    )
    threshold = engagement.quantile(0.9)
    top_mask = (engagement >= threshold).values

    if top_mask.sum() == 0:
        return {"trading_day": day, "f11_engagement_alignment": np.nan}

    top_vecs = vecs[top_mask]
    f11 = float(np.mean([cosine_sim(v, centroid) for v in top_vecs]))
    return {"trading_day": day, "f11_engagement_alignment": f11}


def main():
    df = load_tweets()
    rng = np.random.default_rng(RANDOM_SEED)

    try:
        done_df = pd.read_parquet(CHECKPOINT_PATH)
        done_days = set(done_df["trading_day"].astype(str))
        print(f"Resuming: {len(done_days)} days already checkpointed", flush=True)
    except FileNotFoundError:
        done_df = pd.DataFrame()
        done_days = set()

    all_rows = done_df.to_dict("records") if not done_df.empty else []
    grouped = list(df.groupby("trading_day"))
    print(f"Processing {len(grouped)} trading days...", flush=True)

    for i, (day, day_df) in enumerate(grouped):
        day_str = str(day)
        if day_str in done_days:
            continue

        t0 = time.time()
        G = build_graph(day_df)
        graph_feats = compute_graph_features(day_str, G)
        narrative_feats = compute_narrative_features(day_str, day_df, rng)
        engagement_feats = compute_engagement_alignment(day_str, day_df)

        row = {**graph_feats, **narrative_feats, **engagement_feats}
        all_rows.append(row)

        n_dedup = (~day_df["is_near_duplicate"]).sum()
        elapsed = time.time() - t0
        print(
            f"  [{i + 1}/{len(grouped)}] {day_str}: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges ({len(day_df)} tweets, {n_dedup} dedup) "
            f"- {elapsed:.1f}s",
            flush=True,
        )

        if (i + 1) % 5 == 0:
            pd.DataFrame(all_rows).to_parquet(CHECKPOINT_PATH)
            print(f"    checkpoint saved ({len(all_rows)} days)", flush=True)

    result = pd.DataFrame(all_rows)
    result.to_parquet(CHECKPOINT_PATH)

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    destination_table = PROJECT + "." + DATASET + ".premarket_features"
    client.load_table_from_dataframe(result, destination_table, job_config=job_config).result()

    print("Loaded " + str(len(result)) + " rows to " + destination_table, flush=True)
    print(result.describe())


if __name__ == "__main__":
    main()
