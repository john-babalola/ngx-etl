"""
Stage 8a: primary model harness.

Joins interval_spine (target), premarket_features (f1-f12, constant
within a trading day), and intraday_features (g1-g6, varies per
interval) into one model-ready table, then evaluates five
specifications under walk-forward validation with day-aligned fold
boundaries, per Ch3 3.8.2:

  1. Majority-class baseline
  2. Autoregressive baseline    (g4-g6 only: lagged returns, no
                                  discourse features)
  3. Elastic-net logistic regression (all 18 features)
  4. XGBoost classifier              (all 18 features)
  5. ARIMAX                          (continuous log return, exogenous
                                       feature set, fixed order for
                                       this first pass)

Validation design: expanding window, day-aligned. An initial block of
trading days forms the first training set; the model is then evaluated
on the single next trading day, that day's intervals are folded into
the training set, and the process repeats through to the end of the
sample. This is what Ch3 3.8.2 specifies - fold boundaries aligned to
trading days rather than intervals, since pre-market features are
constant within a day and splitting a day across train/test would
otherwise leak information.

Standardisation: feature scaling (mean/std) is computed from the
training fold only at each step and applied to both train and test,
per Ch3 3.6.4 - this prevents evaluation-period information leaking
into the transform applied to training data.

Missingness: f3 (eigenvector dispersion) and g1/g2 (rolling network
features) can be NaN on thin windows. XGBoost handles this natively.
The elastic-net specification requires imputation, for which the
training fold's median is used (matching Ch3 3.6.4); missingness
indicators are retained as additional features so the model can
distinguish a genuine measurement from an imputed one.

Hyperparameters are FIXED for this first pass rather than searched via
the nested grid described in Ch3 3.7.4. This is a deliberate, disclosed
scope reduction given the project timeline; the nested search can be
layered in afterward (stage 8c) if time allows, and Ch4 should state
plainly which was actually used.
"""

import time
import warnings
import numpy as np
import pandas as pd
from google.cloud import bigquery
from google.cloud.bigquery_storage import BigQueryReadClient

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from xgboost import XGBClassifier
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

PROJECT = "ngx-discourse-2026"
DATASET = "ngx"
RANDOM_SEED = 42
INITIAL_TRAIN_DAYS = 25  # roughly half the 49 trading days

client = bigquery.Client(project=PROJECT)
bqstorage_client = BigQueryReadClient()

FEATURE_COLS = [
    "f1_degree_centralisation",
    "f2_betweenness_centralisation",
    "f3_eigenvector_dispersion",
    "f4_gini_indegree",
    "f5_density",
    "f6_lcc_ratio",
    "f7_mean_pairwise_cosine",
    "f8_mean_embedding_variance",
    "f9_topic_entropy",
    "f10_dominant_cluster_share",
    "f11_engagement_alignment",
    "f12_sentiment_skew",
    "g1_rolling_degree_centralisation",
    "g2_rolling_mean_cosine",
    "g3_rolling_log_volume",
    "g4_lag1_return",
    "g5_lag2_return",
    "g6_lag3_return",
]
AR_BASELINE_COLS = ["g4_lag1_return", "g5_lag2_return", "g6_lag3_return"]
MISSING_INDICATOR_SOURCE_COLS = ["f3_eigenvector_dispersion", "g1_rolling_degree_centralisation", "g2_rolling_mean_cosine"]


def load_model_data():
    """Joins spine + premarket + intraday features. Restricted to
    intervals with a non-null direction_label (i.e. neither endpoint
    was a stale API read), matching the exclusion already applied in
    stage 4.

    premarket_features.trading_day is stored as STRING (a side effect
    of a pandas merge during an earlier rebuild of that table), while
    interval_spine and intraday_features store it/interval_start as
    native DATE/TIMESTAMP. The join to premarket_features casts both
    sides to STRING explicitly rather than relying on implicit type
    coercion, which BigQuery does not perform across DATE/STRING."""
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
    print("Submitting model-data query to BigQuery...", flush=True)
    t0 = time.time()
    job = client.query(query)
    result = job.result()
    print(f"Query completed in {time.time() - t0:.1f}s, downloading rows...", flush=True)

    t1 = time.time()
    df = result.to_dataframe(bqstorage_client=bqstorage_client)
    print(
        f"Downloaded {len(df):,} model-ready rows in {time.time() - t1:.1f}s "
        f"across {df['trading_day'].nunique()} trading days",
        flush=True,
    )
    return df


def add_missing_indicators(df):
    out = df.copy()
    for col in MISSING_INDICATOR_SOURCE_COLS:
        out[col + "_isnan"] = out[col].isna().astype(int)
    return out


def expanding_window_folds(trading_days, initial_train_days):
    """Yields (train_days, test_day) pairs. train_days expands by one
    trading day after each step; test_day is always the single next
    trading day."""
    for i in range(initial_train_days, len(trading_days)):
        train_days = trading_days[:i]
        test_day = trading_days[i]
        yield train_days, test_day


def standardize(train_df, test_df, cols):
    """Z-score using training-fold mean/std only. Columns with zero
    training variance are left unscaled (subtract mean, divide by 1)
    rather than producing inf/NaN."""
    means = train_df[cols].mean()
    stds = train_df[cols].std().replace(0, 1).fillna(1)
    train_scaled = (train_df[cols] - means) / stds
    test_scaled = (test_df[cols] - means) / stds
    return train_scaled, test_scaled


def median_impute(train_df, test_df, cols):
    medians = train_df[cols].median()
    train_imputed = train_df[cols].fillna(medians)
    test_imputed = test_df[cols].fillna(medians)
    return train_imputed, test_imputed


def run_majority(train_y, test_y):
    majority_class = int(train_y.mode().iloc[0])
    preds = np.full(len(test_y), majority_class)
    probs = np.full(len(test_y), float(majority_class))
    return preds, probs


def run_ar_baseline(train_df, test_df):
    x_train, x_test = standardize(train_df, test_df, AR_BASELINE_COLS)
    x_train_i, x_test_i = median_impute(
        pd.DataFrame(x_train, columns=AR_BASELINE_COLS),
        pd.DataFrame(x_test, columns=AR_BASELINE_COLS),
        AR_BASELINE_COLS,
    )
    y_train = train_df["direction_label"].astype(int)

    if y_train.nunique() < 2:
        majority_class = int(y_train.mode().iloc[0])
        preds = np.full(len(test_df), majority_class)
        probs = np.full(len(test_df), float(majority_class))
        return preds, probs

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(x_train_i, y_train)
    probs = model.predict_proba(x_test_i)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def run_elastic_net(train_df, test_df, feature_cols):
    x_train, x_test = standardize(train_df, test_df, feature_cols)
    x_train_i, x_test_i = median_impute(
        pd.DataFrame(x_train, columns=feature_cols),
        pd.DataFrame(x_test, columns=feature_cols),
        feature_cols,
    )
    y_train = train_df["direction_label"].astype(int)

    if y_train.nunique() < 2:
        majority_class = int(y_train.mode().iloc[0])
        preds = np.full(len(test_df), majority_class)
        probs = np.full(len(test_df), float(majority_class))
        return preds, probs

    model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        C=1.0,
        solver="saga",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train_i, y_train)
    probs = model.predict_proba(x_test_i)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def run_xgboost(train_df, test_df, feature_cols):
    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]
    y_train = train_df["direction_label"].astype(int)

    if y_train.nunique() < 2:
        majority_class = int(y_train.mode().iloc[0])
        preds = np.full(len(test_df), majority_class)
        probs = np.full(len(test_df), float(majority_class))
        return preds, probs

    model = XGBClassifier(
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def run_arimax(train_df, test_df, feature_cols):
    """Fixed order (1,0,0) for this first pass - AIC-based order
    selection deferred to stage 8c per Ch3 3.7.4."""
    x_train, x_test = standardize(train_df, test_df, feature_cols)
    x_train_i, x_test_i = median_impute(
        pd.DataFrame(x_train, columns=feature_cols).reset_index(drop=True),
        pd.DataFrame(x_test, columns=feature_cols).reset_index(drop=True),
        feature_cols,
    )
    y_train = train_df["interval_log_return"].reset_index(drop=True)

    try:
        model = SARIMAX(
            y_train,
            exog=x_train_i,
            order=(1, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = model.get_forecast(steps=len(test_df), exog=x_test_i)
        pred_returns = forecast.predicted_mean.values
    except Exception as e:
        print(f"    ARIMAX fit failed for this fold ({e}); falling back to zero forecast", flush=True)
        pred_returns = np.zeros(len(test_df))

    pred_direction = (pred_returns > 0).astype(int)
    return pred_direction, pred_returns


def pooled_classification_metrics(name, y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

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


def main():
    df = load_model_data()
    df = add_missing_indicators(df)

    trading_days = sorted(df["trading_day"].unique())
    print(f"\n{len(trading_days)} trading days available; "
          f"{INITIAL_TRAIN_DAYS} used for initial training, "
          f"{len(trading_days) - INITIAL_TRAIN_DAYS} expanding-window test folds", flush=True)

    results = {
        "majority": {"y_true": [], "y_pred": [], "y_prob": []},
        "ar_baseline": {"y_true": [], "y_pred": [], "y_prob": []},
        "elastic_net": {"y_true": [], "y_pred": [], "y_prob": []},
        "xgboost": {"y_true": [], "y_pred": [], "y_prob": []},
    }
    arimax_results = {"y_true_return": [], "y_pred_return": [], "y_true_dir": [], "y_pred_dir": []}

    t_start = time.time()
    n_folds = 0

    for train_days, test_day in expanding_window_folds(trading_days, INITIAL_TRAIN_DAYS):
        train_df = df[df["trading_day"].isin(train_days)].reset_index(drop=True)
        test_df = df[df["trading_day"] == test_day].reset_index(drop=True)

        if len(test_df) == 0 or len(train_df) == 0:
            continue

        n_folds += 1
        y_test = test_df["direction_label"].astype(int).values

        preds, probs = run_majority(train_df["direction_label"], test_df["direction_label"])
        results["majority"]["y_true"].extend(y_test)
        results["majority"]["y_pred"].extend(preds)
        results["majority"]["y_prob"].extend(probs)

        preds, probs = run_ar_baseline(train_df, test_df)
        results["ar_baseline"]["y_true"].extend(y_test)
        results["ar_baseline"]["y_pred"].extend(preds)
        results["ar_baseline"]["y_prob"].extend(probs)

        preds, probs = run_elastic_net(train_df, test_df, FEATURE_COLS)
        results["elastic_net"]["y_true"].extend(y_test)
        results["elastic_net"]["y_pred"].extend(preds)
        results["elastic_net"]["y_prob"].extend(probs)

        preds, probs = run_xgboost(train_df, test_df, FEATURE_COLS)
        results["xgboost"]["y_true"].extend(y_test)
        results["xgboost"]["y_pred"].extend(preds)
        results["xgboost"]["y_prob"].extend(probs)

        pred_dir, pred_ret = run_arimax(train_df, test_df, FEATURE_COLS)
        arimax_results["y_true_return"].extend(test_df["interval_log_return"].values)
        arimax_results["y_pred_return"].extend(pred_ret)
        arimax_results["y_true_dir"].extend(y_test)
        arimax_results["y_pred_dir"].extend(pred_dir)

        if n_folds % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  fold {n_folds}: test_day={test_day}, train_n={len(train_df)}, "
                  f"test_n={len(test_df)} - {elapsed:.1f}s elapsed", flush=True)

    print(f"\nCompleted {n_folds} folds in {time.time() - t_start:.1f}s\n", flush=True)

    summary_rows = []
    for name, r in results.items():
        m = pooled_classification_metrics(name, r["y_true"], r["y_pred"], r["y_prob"])
        summary_rows.append(m)

    y_true_dir = np.array(arimax_results["y_true_dir"])
    y_pred_dir = np.array(arimax_results["y_pred_dir"])
    y_true_ret = np.array(arimax_results["y_true_return"])
    y_pred_ret = np.array(arimax_results["y_pred_return"])

    arimax_summary = {
        "model": "arimax",
        "n_predictions": len(y_true_dir),
        "accuracy": accuracy_score(y_true_dir, y_pred_dir),
        "macro_f1": f1_score(y_true_dir, y_pred_dir, average="macro", zero_division=0),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "mae": float(np.mean(np.abs(y_true_ret - y_pred_ret))),
        "rmse": float(np.sqrt(np.mean((y_true_ret - y_pred_ret) ** 2))),
    }
    summary_rows.append(arimax_summary)

    summary_df = pd.DataFrame(summary_rows)
    print("--- Pooled out-of-sample results ---")
    print(summary_df.to_string(index=False))

    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    destination_table = PROJECT + "." + DATASET + ".model_results_primary"
    client.load_table_from_dataframe(summary_df, destination_table, job_config=job_config).result()
    print("\nLoaded results to " + destination_table, flush=True)

    predictions_df = pd.DataFrame({
        "model": (
            ["majority"] * len(results["majority"]["y_true"])
            + ["ar_baseline"] * len(results["ar_baseline"]["y_true"])
            + ["elastic_net"] * len(results["elastic_net"]["y_true"])
            + ["xgboost"] * len(results["xgboost"]["y_true"])
        ),
        "y_true": (
            results["majority"]["y_true"]
            + results["ar_baseline"]["y_true"]
            + results["elastic_net"]["y_true"]
            + results["xgboost"]["y_true"]
        ),
        "y_pred": (
            results["majority"]["y_pred"]
            + results["ar_baseline"]["y_pred"]
            + results["elastic_net"]["y_pred"]
            + results["xgboost"]["y_pred"]
        ),
        "y_prob": (
            results["majority"]["y_prob"]
            + results["ar_baseline"]["y_prob"]
            + results["elastic_net"]["y_prob"]
            + results["xgboost"]["y_prob"]
        ),
    })
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    predictions_table = PROJECT + "." + DATASET + ".model_predictions_primary"
    client.load_table_from_dataframe(predictions_df, predictions_table, job_config=job_config).result()
    print("Loaded per-prediction detail to " + predictions_table, flush=True)


if __name__ == "__main__":
    main()
