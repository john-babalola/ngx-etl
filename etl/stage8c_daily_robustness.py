"""
Stage 8c: daily-resolution supplementary analysis.

Per Ch3 3.8.3, the PRIMARY analysis and target of this study is the
intraday interval (already estimated in stage 8a/8b - see
model_results_primary, ablation_results, shap_attribution,
granger_causality_results). This script extends the secondary,
exploratory daily-resolution analysis Ch3 3.8.3 already commits to,
which stage 8b began with a single majority-vs-XGBoost comparison
(daily_aggregation_results, n=25).

The question this script answers: using the IDENTICAL f1-f12 feature
set already used for the primary intraday result, does the daily
directional target (close vs open) show stronger or weaker predictive
power than the intraday target? This is a direct, like-for-like
comparison - same features, comparable model specifications and
metrics - not a re-evaluation of which resolution is "correct." Per
Ch3 3.8.3, the two specifications are reported and discussed
separately in Chapter Four, and no claim of temporal-resolution
superiority is made from either analysis in isolation.

To make this comparison complete rather than partial, this script adds
what stage 8b's single-model daily test did not have: a daily AR
baseline (so daily discourse features can be compared against daily
autocorrelation, mirroring stage 8a's AR baseline), and network/
narrative feature ablation at daily grain (mirroring stage 8b's
intraday ablation_results), plus SHAP attribution on the daily model.

Five specifications, mirroring stage 8a's intraday structure exactly:

  1. Majority-class baseline
  2. Daily AR baseline     (lagged daily direction only - there is no
                             g4-g6 equivalent at daily grain, so this
                             is constructed fresh from interval_spine's
                             daily closes rather than reused from the
                             intraday feature set)
  3. Network-only          (f1-f6, nested hyperparameter search)
  4. Narrative-only        (f7-f12, nested hyperparameter search)
  5. Both                  (f1-f12, nested hyperparameter search)
  6. SHAP attribution on the "both" specification

Sample size: n=49 trading days. This is small enough that every result
here is reported with the raw prediction count alongside each metric,
so precision can be judged directly rather than assumed. No claim of
statistical significance is made from this analysis in isolation.

Validation: walk-forward, expanding window, same logic as stage 8a/8b
but at daily grain.
"""

import time
import warnings
import itertools
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"
RANDOM_SEED = 42
INITIAL_TRAIN_DAYS = 30   # of 49 total; leaves 19 test folds
INNER_VAL_DAYS = 5        # trailing days of each fold's training data, held out for inner search

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()

NETWORK_COLS = [
    "f1_degree_centralisation",
    "f2_betweenness_centralisation",
    "f3_eigenvector_dispersion",
    "f4_gini_indegree",
    "f5_density",
    "f6_lcc_ratio",
]
NARRATIVE_COLS = [
    "f7_mean_pairwise_cosine",
    "f8_mean_embedding_variance",
    "f9_topic_entropy",
    "f10_dominant_cluster_share",
    "f11_engagement_alignment",
    "f12_sentiment_skew",
]
ALL_FEATURE_COLS = NETWORK_COLS + NARRATIVE_COLS
AR_COLS = ["daily_lag1_direction", "daily_lag2_direction"]

XGB_GRID = {"max_depth": [2, 3, 4], "n_estimators": [50, 100, 150]}


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_daily_data():
    """Daily direction target (close vs open, per the original proposal
    formulation) joined to premarket_features (already daily-resolution
    by construction), with lagged daily direction constructed for the
    AR baseline.

    IMPORTANT: verify this query against your actual interval_spine /
    premarket_features schema before running. The join condition below
    assumes premarket_features.trading_day is stored as STRING (per the
    earlier pandas-merge side effect discovered during this project) -
    if that has since been corrected, drop the CAST."""
    query = f"""
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
      SELECT
        trading_day,
        CASE WHEN day_close > day_open THEN 1 ELSE 0 END AS daily_direction_label
      FROM daily
    )
    SELECT
      l.trading_day,
      l.daily_direction_label,
      p.* EXCEPT (trading_day, n_nodes, n_edges, n_topic_assigned, n_topic_total, n_sentiment_scored)
    FROM labeled l
    JOIN `{PROJECT}.{DATASET}.premarket_features` p
      ON CAST(l.trading_day AS STRING) = p.trading_day
    ORDER BY l.trading_day
    """
    print("Submitting daily-data query to BigQuery...", flush=True)
    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorage_client)
    df = df.sort_values("trading_day").reset_index(drop=True)

    df["daily_lag1_direction"] = df["daily_direction_label"].shift(1)
    df["daily_lag2_direction"] = df["daily_direction_label"].shift(2)

    print(f"Loaded {len(df):,} daily rows ({df['trading_day'].min()} to {df['trading_day'].max()})", flush=True)
    print(f"Class balance: {df['daily_direction_label'].value_counts().to_dict()}", flush=True)
    return df


def add_missing_indicators(df):
    out = df.copy()
    if "f3_eigenvector_dispersion" in out.columns:
        out["f3_eigenvector_dispersion_isnan"] = out["f3_eigenvector_dispersion"].isna().astype(int)
    return out


# ---------------------------------------------------------------
# Shared helpers (same logic as stage 8a/8b)
# ---------------------------------------------------------------
def expanding_window_folds(n_rows, initial_train):
    for i in range(initial_train, n_rows):
        yield list(range(i)), i


def standardize(train_df, test_df, cols):
    means = train_df[cols].mean()
    stds = train_df[cols].std().replace(0, 1).fillna(1)
    return (train_df[cols] - means) / stds, (test_df[cols] - means) / stds


def median_impute(train_df, test_df):
    medians = train_df.median()
    medians = medians.fillna(0.0)
    return train_df.fillna(medians), test_df.fillna(medians)


def pooled_classification_metrics(name, y_true, y_pred, y_prob):
    y_true, y_pred, y_prob = np.array(y_true), np.array(y_pred), np.array(y_prob)
    metrics = {"model": name, "n_predictions": len(y_true)}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
    return metrics


def search_xgb_params(train_df, cols, label_col):
    if len(train_df) <= INNER_VAL_DAYS + 5:
        return {"max_depth": 3, "n_estimators": 100}

    inner_train = train_df.iloc[:-INNER_VAL_DAYS]
    inner_val = train_df.iloc[-INNER_VAL_DAYS:]

    if inner_train[label_col].nunique() < 2 or inner_val[label_col].nunique() < 2:
        return {"max_depth": 3, "n_estimators": 100}

    best_score, best_params = -np.inf, {"max_depth": 3, "n_estimators": 100}
    for max_depth, n_estimators in itertools.product(XGB_GRID["max_depth"], XGB_GRID["n_estimators"]):
        model = XGBClassifier(
            max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
        model.fit(inner_train[cols], inner_train[label_col].astype(int))
        probs = model.predict_proba(inner_val[cols])[:, 1]
        try:
            score = roc_auc_score(inner_val[label_col].astype(int), probs)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_params = score, {"max_depth": max_depth, "n_estimators": n_estimators}
    return best_params


# ---------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------
def run_majority(train_df, label_col):
    majority_class = int(train_df[label_col].mode().iloc[0])
    return majority_class, float(majority_class)


def run_ar_baseline(train_df, test_row):
    cols = AR_COLS
    train_valid = train_df.dropna(subset=cols + ["daily_direction_label"])
    if len(train_valid) < 5 or train_valid["daily_direction_label"].nunique() < 2:
        majority_class = int(train_df["daily_direction_label"].mode().iloc[0])
        return majority_class, float(majority_class)

    x_train, x_test = standardize(train_valid, test_row, cols)
    x_train_i, x_test_i = median_impute(x_train, x_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(x_train_i, train_valid["daily_direction_label"].astype(int))
    prob = model.predict_proba(x_test_i)[:, 1][0]
    pred = int(prob >= 0.5)
    return pred, prob


def run_xgb_subset(train_df, test_row, cols, label_col):
    y_train = train_df[label_col].astype(int)
    if y_train.nunique() < 2:
        majority_class = int(y_train.mode().iloc[0])
        return majority_class, float(majority_class)

    params = search_xgb_params(train_df, cols, label_col)
    model = XGBClassifier(
        max_depth=params["max_depth"], n_estimators=params["n_estimators"],
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_SEED,
    )
    model.fit(train_df[cols], y_train)
    prob = model.predict_proba(test_row[cols])[:, 1][0]
    pred = int(prob >= 0.5)
    return pred, prob


# ---------------------------------------------------------------
# SHAP on the full daily model
# ---------------------------------------------------------------
def run_shap(df):
    import shap

    y = df["daily_direction_label"].astype(int)
    X = df[ALL_FEATURE_COLS].fillna(df[ALL_FEATURE_COLS].median())

    model = XGBClassifier(
        max_depth=3, n_estimators=100, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature": ALL_FEATURE_COLS,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df["family"] = shap_df["feature"].apply(
        lambda f: "network" if f in NETWORK_COLS else "narrative"
    )
    return shap_df


# ---------------------------------------------------------------
def main():
    df = load_daily_data()
    df = add_missing_indicators(df)
    n_rows = len(df)

    print(f"\n{n_rows} total days; {INITIAL_TRAIN_DAYS} used for initial training, "
          f"{n_rows - INITIAL_TRAIN_DAYS} expanding-window test folds", flush=True)

    results = {
        "majority": {"y_true": [], "y_pred": [], "y_prob": []},
        "ar_baseline": {"y_true": [], "y_pred": [], "y_prob": []},
        "network_only": {"y_true": [], "y_pred": [], "y_prob": []},
        "narrative_only": {"y_true": [], "y_pred": [], "y_prob": []},
        "both": {"y_true": [], "y_pred": [], "y_prob": []},
    }

    t0 = time.time()
    n_folds = 0
    for train_idx, test_idx in expanding_window_folds(n_rows, INITIAL_TRAIN_DAYS):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_row = df.iloc[[test_idx]].reset_index(drop=True)
        n_folds += 1

        y_test = test_row["daily_direction_label"].astype(int).values

        pred, prob = run_majority(train_df, "daily_direction_label")
        results["majority"]["y_true"].extend(y_test)
        results["majority"]["y_pred"].append(pred)
        results["majority"]["y_prob"].append(prob)

        pred, prob = run_ar_baseline(train_df, test_row)
        results["ar_baseline"]["y_true"].extend(y_test)
        results["ar_baseline"]["y_pred"].append(pred)
        results["ar_baseline"]["y_prob"].append(prob)

        pred, prob = run_xgb_subset(train_df, test_row, NETWORK_COLS, "daily_direction_label")
        results["network_only"]["y_true"].extend(y_test)
        results["network_only"]["y_pred"].append(pred)
        results["network_only"]["y_prob"].append(prob)

        pred, prob = run_xgb_subset(train_df, test_row, NARRATIVE_COLS, "daily_direction_label")
        results["narrative_only"]["y_true"].extend(y_test)
        results["narrative_only"]["y_pred"].append(pred)
        results["narrative_only"]["y_prob"].append(prob)

        pred, prob = run_xgb_subset(train_df, test_row, ALL_FEATURE_COLS, "daily_direction_label")
        results["both"]["y_true"].extend(y_test)
        results["both"]["y_pred"].append(pred)
        results["both"]["y_prob"].append(prob)

        if n_folds % 5 == 0:
            print(f"  fold {n_folds}: {time.time() - t0:.1f}s elapsed", flush=True)

    print(f"\nCompleted {n_folds} folds in {time.time() - t0:.1f}s\n", flush=True)

    summary_rows = [
        pooled_classification_metrics(name, r["y_true"], r["y_pred"], r["y_prob"])
        for name, r in results.items()
    ]
    summary_df = pd.DataFrame(summary_rows)
    print("--- Daily-resolution results (n=%d test predictions per model) ---" % n_folds)
    print(summary_df.to_string(index=False))

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    destination = PROJECT + "." + DATASET + ".daily_robustness_results"
    client.load_table_from_dataframe(summary_df, destination, job_config=job_config).result()
    print("\nLoaded results to " + destination, flush=True)

    print("\n--- SHAP attribution (daily, full-sample fit) ---")
    shap_df = run_shap(df)
    print(shap_df.to_string(index=False))
    shap_destination = PROJECT + "." + DATASET + ".daily_shap_attribution"
    client.load_table_from_dataframe(shap_df, shap_destination, job_config=job_config).result()
    print("Loaded SHAP results to " + shap_destination, flush=True)

    print("\nStage 8c complete.", flush=True)
    print("\nCompare this table's roc_auc/pr_auc/macro_f1 for 'both' directly against", flush=True)
    print("stage 8a's XGBoost row (intraday, all 18 features) to answer: does the", flush=True)
    print("daily formulation predict BETTER or WORSE than the intraday one?", flush=True)


if __name__ == "__main__":
    main()
