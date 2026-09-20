"""
Stage 9: Chapter 4 visualisations.

Produces five charts directly from the BigQuery results tables already
populated by stages 8a/8b/8c, addressing the supervisor's request to
visualise relationships between features and results rather than rely
on tables alone.

  1. Feature ablation comparison (bar chart)   - RQ1/RQ2
  2. SHAP attribution summary (bar chart)       - interpretability
  3. Granger causality p-value heatmap          - RQ3
  4. Feature correlation matrix (heatmap)       - "relationships
                                                    between features"
  5. f1/f4 centralisation over time, coloured
     by daily direction (scatter/line)          - exploratory

All charts save as PNG to a local ./figures/ directory, ready to embed
directly in Chapter 4. Uses matplotlib's non-interactive Agg backend
since this runs headless in Cloud Shell.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"
FIGURES_DIR = "figures"

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()

NETWORK_COLS = [
    "f1_degree_centralisation", "f2_betweenness_centralisation",
    "f3_eigenvector_dispersion", "f4_gini_indegree",
    "f5_density", "f6_lcc_ratio",
]
NARRATIVE_COLS = [
    "f7_mean_pairwise_cosine", "f8_mean_embedding_variance",
    "f9_topic_entropy", "f10_dominant_cluster_share",
    "f11_engagement_alignment", "f12_sentiment_skew",
]

FAMILY_COLORS = {"network": "#2C6E49", "narrative": "#C1666B", "other": "#8A8A8A"}


def query(sql):
    return client.query(sql).result().to_dataframe(bqstorage_client=bqstorage_client)


def ensure_dir():
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------
# Chart 1: feature ablation comparison
# ---------------------------------------------------------------
def plot_ablation(ablation_df, primary_df):
    """Grouped bar chart: ROC-AUC, PR-AUC, macro-F1 for majority,
    AR baseline, network-only, narrative-only, both. Pulls majority/AR
    from primary_df and ablation subsets from ablation_df so all five
    specifications appear on one chart."""
    baseline_rows = primary_df[primary_df["model"].isin(["majority", "ar_baseline"])]
    combined = pd.concat([baseline_rows, ablation_df], ignore_index=True)

    order = ["majority", "ar_baseline", "network_only", "narrative_only", "both"]
    combined["model"] = pd.Categorical(combined["model"], categories=order, ordered=True)
    combined = combined.sort_values("model").reset_index(drop=True)

    metrics = ["roc_auc", "pr_auc", "macro_f1"]
    x = np.arange(len(combined))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, combined[metric], width, label=metric.upper().replace("_", "-"))

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, label="Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Majority", "AR baseline", "Network only", "Narrative only", "Both (combined)"],
        rotation=15, ha="right"
    )
    ax.set_ylabel("Score")
    ax.set_title("Intraday classification performance by feature subset")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(0, max(0.75, combined[metrics].values.max() + 0.05))
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig1_ablation_comparison.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/fig1_ablation_comparison.png")


# ---------------------------------------------------------------
# Chart 2: SHAP attribution summary
# ---------------------------------------------------------------
def plot_shap(shap_df):
    """Horizontal bar chart, features ranked by mean |SHAP|, coloured
    by feature family - the standard presentation for this analysis."""
    df = shap_df.sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    colors = [FAMILY_COLORS.get(f, "#8A8A8A") for f in df["family"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(df["feature"], df["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature attribution (XGBoost, full-feature intraday specification)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, FAMILY_COLORS.keys(), loc="lower right", frameon=False, title="Family")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig2_shap_attribution.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/fig2_shap_attribution.png")


# ---------------------------------------------------------------
# Chart 3: Granger causality heatmap
# ---------------------------------------------------------------
def plot_granger(granger_df):
    """Heatmap of -log10(p-value) across feature x lag. Higher values
    (darker) would indicate stronger evidence; annotates the
    significance threshold explicitly since the expected finding here
    is a null result across the board."""
    df = granger_df.copy()
    df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-10))

    pivot = df.pivot_table(index="feature", columns="lag", values="neg_log10_p")
    # order rows by family then feature name for readability
    family_map = df.drop_duplicates("feature").set_index("feature")["family"]
    pivot = pivot.loc[sorted(pivot.index, key=lambda f: (family_map.get(f, "z"), f))]

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Lag {l}" for l in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Granger causality: -log10(p-value) by feature and lag\n(threshold for p<0.05 is 1.30)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("-log10(p-value)")
    # mark the significance threshold on the colorbar
    cbar.ax.axhline(1.30, color="blue", linewidth=1.5)

    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig3_granger_heatmap.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/fig3_granger_heatmap.png")


# ---------------------------------------------------------------
# Chart 4: feature correlation matrix
# ---------------------------------------------------------------
def plot_correlation_matrix(features_df):
    """Correlation heatmap across f1-f12, directly addressing the
    supervisor's request to visualise relationships between features -
    also substantiates the elastic-net regularisation justification in
    Ch3 3.7.2 (features expected to be correlated)."""
    cols = NETWORK_COLS + NARRATIVE_COLS
    corr = features_df[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [c.replace("_", " ").replace("f1 ", "f1: ") for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if abs(val) >= 0.4:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(val) > 0.6 else "black")

    # dividing lines between network and narrative blocks
    n_network = len(NETWORK_COLS)
    ax.axhline(n_network - 0.5, color="black", linewidth=1)
    ax.axvline(n_network - 0.5, color="black", linewidth=1)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson correlation")
    ax.set_title("Pre-market feature correlation matrix (f1-f12)")
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig4_correlation_matrix.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/fig4_correlation_matrix.png")


# ---------------------------------------------------------------
# Chart 5: centralisation over time, coloured by outcome
# ---------------------------------------------------------------
def plot_centralisation_over_time(features_with_outcome_df):
    """Scatter/line of f1 (degree centralisation) across trading days,
    coloured by whether that day closed up or down - exploratory chart
    showing what the key predictor looks like day to day."""
    df = features_with_outcome_df.sort_values("trading_day").reset_index(drop=True)
    df["trading_day"] = pd.to_datetime(df["trading_day"])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["trading_day"], df["f1_degree_centralisation"], color="gray", linewidth=0.8, alpha=0.5, zorder=1)

    up = df[df["daily_direction_label"] == 1]
    down = df[df["daily_direction_label"] == 0]
    ax.scatter(up["trading_day"], up["f1_degree_centralisation"], color="#2C6E49", label="Close up", s=35, zorder=2)
    ax.scatter(down["trading_day"], down["f1_degree_centralisation"], color="#C1666B", label="Close down/flat", s=35, zorder=2)

    ax.set_ylabel("Degree centralisation (f1)")
    ax.set_xlabel("Trading day")
    ax.set_title("Pre-market network centralisation across the sample, by daily outcome")
    ax.legend(loc="upper left", frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig5_centralisation_over_time.png", dpi=200)
    plt.close(fig)
    print(f"Saved {FIGURES_DIR}/fig5_centralisation_over_time.png")


# ---------------------------------------------------------------
def main():
    ensure_dir()

    print("Loading ablation_results and model_results_primary...", flush=True)
    ablation_df = query(f"SELECT * FROM `{PROJECT}.{DATASET}.ablation_results`")
    primary_df = query(f"SELECT * FROM `{PROJECT}.{DATASET}.model_results_primary`")
    plot_ablation(ablation_df, primary_df)

    print("Loading shap_attribution...", flush=True)
    shap_df = query(f"SELECT * FROM `{PROJECT}.{DATASET}.shap_attribution`")
    plot_shap(shap_df)

    print("Loading granger_causality_results...", flush=True)
    granger_df = query(f"SELECT * FROM `{PROJECT}.{DATASET}.granger_causality_results`")
    plot_granger(granger_df)

    print("Loading premarket_features for correlation matrix...", flush=True)
    features_df = query(f"SELECT * FROM `{PROJECT}.{DATASET}.premarket_features`")
    plot_correlation_matrix(features_df)

    print("Loading premarket_features + daily outcome for time series chart...", flush=True)
    daily_query = f"""
    WITH daily AS (
      SELECT
        trading_day,
        ARRAY_AGG(open_value ORDER BY interval_start LIMIT 1)[OFFSET(0)] AS day_open,
        ARRAY_AGG(close_value ORDER BY interval_start DESC LIMIT 1)[OFFSET(0)] AS day_close
      FROM `{PROJECT}.{DATASET}.interval_spine`
      WHERE direction_label IS NOT NULL
      GROUP BY trading_day
    ),
    labeled AS (
      SELECT trading_day,
             CASE WHEN day_close > day_open THEN 1 ELSE 0 END AS daily_direction_label
      FROM daily
    )
    SELECT l.trading_day, l.daily_direction_label, p.f1_degree_centralisation
    FROM labeled l
    JOIN `{PROJECT}.{DATASET}.premarket_features` p
      ON CAST(l.trading_day AS STRING) = p.trading_day
    ORDER BY l.trading_day
    """
    time_series_df = query(daily_query)
    plot_centralisation_over_time(time_series_df)

    print("\nAll five figures saved to ./figures/", flush=True)


if __name__ == "__main__":
    main()
