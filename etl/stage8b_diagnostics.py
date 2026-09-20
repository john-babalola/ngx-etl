"""
Stage 8b: diagnostics and secondary analyses.

Extends stage 8a with the analyses Ch3 commits to beyond the primary
walk-forward evaluation:

  1. Nested hyperparameter search (Ch3 3.7.4) for XGBoost and elastic
     net, using an inner time-respecting split within each outer
     training fold (never a random k-fold, which would violate the
     same temporal-leakage principle the whole design is built on).

  2. Feature ablation (Ch3 3.8.3): network-only vs narrative-only vs
     both, quantifying each family's marginal contribution.

  3. Daily-aggregation analysis (Ch3 3.8.3): an aggregated daily
     directional target, reported as its own finding rather than a
     robustness check on the intraday result, given the much smaller
     number of independent observations at daily resolution.

  4. SHAP attribution (Ch3 3.8.4) on the full-feature XGBoost
     specification, fitted once on the complete sample for
     interpretation purposes (not part of the walk-forward evaluation
     itself - SHAP here characterises which features the model relies
     on across the whole sample, matching standard practice for this
     kind of post-hoc attribution).

  5. Granger causality (Ch3 3.8.5), lag-limited to 1-5 intervals,
     testing whether each pre-market/intraday feature improves
     prediction of interval returns beyond lagged returns alone.

Deliberately NOT included, per the Ch3 3.8.3 disclosure: re-estimation
with automated/near-duplicate content retained, and re-computation
under alternative window-length definitions. Both are stated as scope
limitations rather than silently omitted.

Reads the same model-ready table as stage 8a (interval_spine joined to
premarket_features and intraday_features). Writes results to four
BigQuery tables: ablation_results, daily_aggregation_results,
shap_attribution, granger_causality_results.
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
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"
RANDOM_SEED = 42
INITIAL_TRAIN_DAYS = 25
INNER_VAL_DAYS = 5  # trailing days of each training fold held out for inner hyperparameter search

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()

NETWORK_COLS = [
    "f1_degree_centralisation",
    "f2_betweenness_centralisation",
    "f3_eigenvector_dispersion",
    "f4_gini_indegree",
    "f5_density",
    "f6_lcc_ratio",
    "g1_rolling_degree_centralisation",
]
NARRATIVE_COLS = [
    "f7_mean_pairwise_cosine",
    "f8_mean_embedding_variance",
    "f9_topic_entropy",
    "f10_dominant_cluster_share",
    "f11_engagement_alignment",
    "f12_sentiment_skew",
    "g2_rolling_mean_cosine",
]
OTHER_COLS = ["g3_rolling_log_volume", "g4_lag1_return", "g5_lag2_return", "g6_lag3_return"]
ALL_FEATURE_COLS = NETWORK_COLS + NARRATIVE_COLS + OTHER_COLS
DAILY_FEATURE_COLS = [c for c in NETWORK_COLS if not c.startswith("g")] + \
                     [c for c in NARRATIVE_COLS if not c.startswith("g")]

XGB_GRID = {
    "max_depth": [2, 3, 4],
    "n_estimators": [50, 100, 150],
}
ELASTIC_NET_GRID = {
    "l1_ratio": [0.3, 0.5, 0.7],
    "C": [0.1, 1.0, 10.0],
}

GRANGER_LAGS = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
def load_model_data():
    query = f"""
    SELECT
      s.trading_day,
      s.interval_start,
      s.direction_label,
      s.interval_log_return,
      p.* EXCEPT (trading_day, n_nodes, n_edges, n_topic_assigned, n_topic_total, n_sentiment_scored),
      i.* EXCEPT (interval_start, n_tweets_in_window, g1_missing, g2_missing)
    FROM `{PROJECT}.{DATASET}.interval_spine` s
    JOIN `{PROJECT}.{DATASET}.premarket_features` p
      ON CAST(s.trading_day AS STRING) = p.trading_day
    JOIN `{PROJECT}.{DATASET}.intraday_features` i USING (interval_start)
    WHERE s.direction_label IS NOT NULL
    ORDER BY s.trading_day, s.interval_start
    """
    print("Loading model data...", flush=True)
    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorage_client)
    print(f"Loaded {len(df):,} rows across {df['trading_day'].nunique()} trading days", flush=True)
    return df


def load_daily_data():
    """Daily direction target: close vs open on the trading day, joined
    to the same premarket_features (which are natively daily-resolution
    already)."""
    query = f"""
    WITH daily AS (
      SELECT
        trading_day,
        ARRAY_AGG(open_value ORDER BY interval_start LIMIT 1)[OFFSET(0)] AS day_open,
        ARRAY_AGG(close_value ORDER BY interval_start DESC LIMIT 1)[OFFSET(0)] AS day_close
      FROM `{PROJECT}.{DATASET}.interval_spine`
      WHERE direction_label IS NOT NULL
      GROUP BY trading_day
    )
    SELECT
      d.trading_day,
      CASE WHEN d.day_close > d.day_open THEN 1 ELSE 0 END AS daily_direction_label,
      p.* EXCEPT (trading_day, n_nodes, n_edges, n_topic_assigned, n_topic_total, n_sentiment_scored)
    FROM daily d
    JOIN `{PROJECT}.{DATASET}.premarket_features` p
      ON CAST(d.trading_day AS STRING) = p.trading_day
    ORDER BY d.trading_day
    """
    df = client.query(query).result().to_dataframe(bqstorage_client=bqstorage_client)
    print(f"Loaded {len(df):,} daily rows for daily-aggregation analysis", flush=True)
    return df


# ---------------------------------------------------------------
# Shared helpers (same logic as stage 8a)
# ---------------------------------------------------------------
def add_missing_indicators(df):
    out = df.copy()
    for col in ["f3_eigenvector_dispersion", "g1_rolling_degree_centralisation", "g2_rolling_mean_cosine"]:
        if col in out.columns:
            out[col + "_isnan"] = out[col].isna().astype(int)
    return out


def expanding_window_folds(trading_days, initial_train_days):
    for i in range(initial_train_days, len(trading_days)):
        yield trading_days[:i], trading_days[i]


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


# ---------------------------------------------------------------
# Nested hyperparameter search
# ---------------------------------------------------------------
def inner_split(train_df, trading_days_in_fold, inner_val_days):
    """Splits a fold's training data into an inner-train/inner-val split
    using the trailing `inner_val_days` trading days as validation -
    time-respecting, never a random split."""
    if len(trading_days_in_fold) <= inner_val_days:
        return train_df, train_df.iloc[0:0]  # degenerate: not enough days yet, skip search
    inner_val_day_set = set(trading_days_in_fold[-inner_val_days:])
    inner_train = train_df[~train_df["trading_day"].isin(inner_val_day_set)]
    inner_val = train_df[train_df["trading_day"].isin(inner_val_day_set)]
    return inner_train, inner_val


def search_xgb_params(train_df, trading_days_in_fold, feature_cols):
    inner_train, inner_val = inner_split(train_df, trading_days_in_fold, INNER_VAL_DAYS)
    if len(inner_val) == 0 or inner_train["direction_label"].nunique() < 2:
        return {"max_depth": 3, "n_estimators": 100}  # fall back to the stage 8a default

    best_score, best_params = -np.inf, {"max_depth": 3, "n_estimators": 100}
    for max_depth, n_estimators in itertools.product(XGB_GRID["max_depth"], XGB_GRID["n_estimators"]):
        model = XGBClassifier(
            max_depth=max_depth, n_estimators=n_estimators,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RANDOM_SEED,
        )
        model.fit(inner_train[feature_cols], inner_train["direction_label"].astype(int))
        if inner_val["direction_label"].nunique() < 2:
            continue
        probs = model.predict_proba(inner_val[feature_cols])[:, 1]
        try:
            score = roc_auc_score(inner_val["direction_label"].astype(int), probs)
        except ValueError:
            continue
        if score > best_score:
            best_score, best_params = score, {"max_depth": max_depth, "n_estimators": n_estimators}
    return best_params


def search_elastic_net_params(train_df, trading_days_in_fold, feature_cols):
    inner_train, inner_val = inner_split(train_df, trading_days_in_fold, INNER_VAL_DAYS)
    if len(inner_val) == 0 or inner_train["direction_label"].nunique() < 2:
        return {"l1_ratio": 0.5, "C": 1.0}

    x_train, x_val = standardize(inner_train, inner_val, feature_cols)
    x_train_i, x_val_i = median_impute(x_train, x_val)

    best_score, best_params = -np.inf, {"l1_ratio": 0.5, "C": 1.0}
    for l1_ratio, C in itertools.product(ELASTIC_NET_GRID["l1_ratio"], ELASTIC_NET_GRID["C"]):
        if inner_val["direction_label"].nunique() < 2:
            continue
        try:
            model = LogisticRegression(
                penalty="elasticnet", l1_ratio=l1_ratio, C=C,
                solver="saga", max_iter=5000, random_state=RANDOM_SEED,
            )
            model.fit(x_train_i, inner_train["direction_label"].astype(int))
            probs = model.predict_proba(x_val_i)[:, 1]
            score = roc_auc_score(inner_val["direction_label"].astype(int), probs)
        except (ValueError, ArithmeticError):
            continue
        if score > best_score:
            best_score, best_params = score, {"l1_ratio": l1_ratio, "C": C}
    return best_params


# ---------------------------------------------------------------
# Feature ablation
# ---------------------------------------------------------------
def run_ablation(df, trading_days):
    """Runs the same walk-forward XGBoost specification three times,
    once per feature subset, WITH nested hyperparameter search each
    time so the ablation comparison isn't confounded by fixed
    hyperparameters tuned implicitly for the full set."""
    subsets = {
        "network_only": NETWORK_COLS + OTHER_COLS,
        "narrative_only": NARRATIVE_COLS + OTHER_COLS,
        "both": ALL_FEATURE_COLS,
    }
    results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in subsets}

    t0 = time.time()
    for fold_i, (train_days, test_day) in enumerate(expanding_window_folds(trading_days, INITIAL_TRAIN_DAYS)):
        train_df = df[df["trading_day"].isin(train_days)].reset_index(drop=True)
        test_df = df[df["trading_day"] == test_day].reset_index(drop=True)
        if len(test_df) == 0 or len(train_df) == 0:
            continue
        y_test = test_df["direction_label"].astype(int).values

        for name, cols in subsets.items():
            y_train = train_df["direction_label"].astype(int)
            if y_train.nunique() < 2:
                majority_class = int(y_train.mode().iloc[0])
                preds = np.full(len(test_df), majority_class)
                probs = np.full(len(test_df), float(majority_class))
            else:
                params = search_xgb_params(train_df, train_days, cols)
                model = XGBClassifier(
                    max_depth=params["max_depth"], n_estimators=params["n_estimators"],
                    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss", random_state=RANDOM_SEED,
                )
                model.fit(train_df[cols], y_train)
                probs = model.predict_proba(test_df[cols])[:, 1]
                preds = (probs >= 0.5).astype(int)
            results[name]["y_true"].extend(y_test)
            results[name]["y_pred"].extend(preds)
            results[name]["y_prob"].extend(probs)

        if (fold_i + 1) % 5 == 0:
            print(f"  ablation fold {fold_i + 1}: {time.time() - t0:.1f}s elapsed", flush=True)

    rows = [pooled_classification_metrics(name, r["y_true"], r["y_pred"], r["y_prob"])
            for name, r in results.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------
# Daily-aggregation analysis
# ---------------------------------------------------------------
def run_daily_aggregation(daily_df):
    """Same expanding-window logic, but at daily grain. With ~49 total
    rows this uses a much smaller initial-train/test split; results are
    reported as an exploratory secondary finding, not a robustness
    check, per Ch3 3.8.3."""
    daily_df = daily_df.sort_values("trading_day").reset_index(drop=True)
    n_days = len(daily_df)
    initial_train = max(20, n_days // 2)  # smaller absolute floor given daily n is much smaller

    results = {"majority": {"y_true": [], "y_pred": [], "y_prob": []},
               "xgboost": {"y_true": [], "y_pred": [], "y_prob": []}}

    for i in range(initial_train, n_days):
        train_df = daily_df.iloc[:i]
        test_row = daily_df.iloc[[i]]
        y_train = train_df["daily_direction_label"].astype(int)
        y_test = test_row["daily_direction_label"].astype(int).values

        majority_class = int(y_train.mode().iloc[0])
        results["majority"]["y_true"].extend(y_test)
        results["majority"]["y_pred"].append(majority_class)
        results["majority"]["y_prob"].append(float(majority_class))

        if y_train.nunique() < 2:
            results["xgboost"]["y_true"].extend(y_test)
            results["xgboost"]["y_pred"].append(majority_class)
            results["xgboost"]["y_prob"].append(float(majority_class))
            continue

        model = XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
        model.fit(train_df[DAILY_FEATURE_COLS], y_train)
        prob = model.predict_proba(test_row[DAILY_FEATURE_COLS])[:, 1][0]
        pred = int(prob >= 0.5)
        results["xgboost"]["y_true"].extend(y_test)
        results["xgboost"]["y_pred"].append(pred)
        results["xgboost"]["y_prob"].append(prob)

    rows = [pooled_classification_metrics(name, r["y_true"], r["y_pred"], r["y_prob"])
            for name, r in results.items()]
    return pd.DataFrame(rows), n_days, initial_train


# ---------------------------------------------------------------
# SHAP attribution
# ---------------------------------------------------------------
def run_shap(df):
    """Fit once on the full sample (standard practice for post-hoc
    global attribution, distinct from the walk-forward predictive
    evaluation) and compute SHAP values for the full-feature XGBoost
    specification."""
    import shap

    y = df["direction_label"].astype(int)
    X = df[ALL_FEATURE_COLS]

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
        lambda f: "network" if f in NETWORK_COLS
        else "narrative" if f in NARRATIVE_COLS
        else "other"
    )
    return shap_df


# ---------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------
def run_granger(df):
    """Tests whether each pre-market/intraday feature Granger-causes
    interval_log_return, at lags 1-5, using the pooled time-ordered
    series. Reports all lags tested, not only significant ones, per
    Ch3 3.8.5."""
    df_sorted = df.sort_values(["trading_day", "interval_start"]).reset_index(drop=True)
    returns = df_sorted["interval_log_return"].fillna(0.0)

    rows = []
    for feature in ALL_FEATURE_COLS:
        series = df_sorted[feature].fillna(df_sorted[feature].median())
        test_data = pd.concat([returns, series], axis=1).dropna()
        if len(test_data) < max(GRANGER_LAGS) * 3:
            continue
        try:
            result = grangercausalitytests(test_data, maxlag=max(GRANGER_LAGS))
            for lag in GRANGER_LAGS:
                f_stat, p_value = result[lag][0]["ssr_ftest"][0], result[lag][0]["ssr_ftest"][1]
                rows.append({
                    "feature": feature,
                    "lag": lag,
                    "f_statistic": f_stat,
                    "p_value": p_value,
                })
        except Exception as e:
            print(f"    Granger test failed for {feature}: {e}", flush=True)

    granger_df = pd.DataFrame(rows)
    if not granger_df.empty:
        granger_df["family"] = granger_df["feature"].apply(
            lambda f: "network" if f in NETWORK_COLS
            else "narrative" if f in NARRATIVE_COLS
            else "other"
        )
    return granger_df


# ---------------------------------------------------------------
def main():
    df = load_model_data()
    df = add_missing_indicators(df)
    trading_days = sorted(df["trading_day"].unique())

    print("\n=== Feature ablation ===", flush=True)
    ablation_df = run_ablation(df, trading_days)
    print(ablation_df.to_string(index=False))
    client.load_table_from_dataframe(
        ablation_df, PROJECT + "." + DATASET + ".ablation_results",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()
    print("Loaded ablation results.", flush=True)

    print("\n=== Daily-aggregation analysis ===", flush=True)
    daily_df = load_daily_data()
    daily_df = add_missing_indicators(daily_df)
    daily_results_df, n_days, initial_train = run_daily_aggregation(daily_df)
    print(f"({n_days} total days, {initial_train} used for initial training, "
          f"{n_days - initial_train} test folds)")
    print(daily_results_df.to_string(index=False))
    client.load_table_from_dataframe(
        daily_results_df, PROJECT + "." + DATASET + ".daily_aggregation_results",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()
    print("Loaded daily-aggregation results.", flush=True)

    print("\n=== SHAP attribution ===", flush=True)
    shap_df = run_shap(df)
    print(shap_df.to_string(index=False))
    client.load_table_from_dataframe(
        shap_df, PROJECT + "." + DATASET + ".shap_attribution",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()
    print("Loaded SHAP attribution.", flush=True)

    print("\n=== Granger causality (lags 1-5) ===", flush=True)
    granger_df = run_granger(df)
    print(granger_df.to_string(index=False))
    client.load_table_from_dataframe(
        granger_df, PROJECT + "." + DATASET + ".granger_causality_results",
        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
    ).result()
    print("Loaded Granger causality results.", flush=True)

    print("\nStage 8b complete.", flush=True)


if __name__ == "__main__":
    main()
