# ngx-etl

ETL pipeline for an MSc Data Science dissertation investigating whether the structural properties of pre-market Nigerian financial discourse on X carry predictive information about intraday NGX All-Share Index movements.

This repository covers the **data engineering layer only**: moving raw collected data from Google Cloud Storage into BigQuery, cleaning it, and producing the analysis-ready interval table on which feature engineering and modelling are built.

---

## What this pipeline produces

The final output is `ngx.interval_spine` — one row per 20-minute trading interval, carrying the opening and closing index values, the log return, the binary direction label, and data-quality flags.

**Current state of the output:**

| Metric | Value |
|---|---|
| Analysis window | 10 July – 18 September 2026 |
| Trading days | 49 |
| Total interval rows | 1,029 |
| Usable intervals (labelled) | 758 |
| Class balance (of labelled) | 62.4% down/flat, 37.6% up |
| Tweets in analysis window | 171,246 |

---

## Prerequisites

- A Google Cloud project with BigQuery and Cloud Storage enabled
- Authenticated `gcloud` credentials with BigQuery and Storage read/write access
- Python 3.12+

Raw data is expected in Cloud Storage under:

```
gs://<BUCKET>/raw/tweets/dt=YYYY-MM-DD/*.jsonl
gs://<BUCKET>/raw/ngx/dt=YYYY-MM-DD/*.json
```

---

## Setup

```bash
git clone https://github.com/john-babalola/ngx-etl.git
cd ngx-etl

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the environment before running anything:

```bash
python3 -c "import pandas, pyarrow, google.cloud.bigquery; print('all ok')"
```

---

## Running the pipeline

```bash
gcloud config set project ngx-discourse-2026
python3 -m etl.run_etl
```

The pipeline is **idempotent**. Every stage uses `CREATE OR REPLACE` or `WRITE_TRUNCATE`, so it can be re-run as often as needed without duplicating data or compounding transformations.

Expected output:

```
BQ client ready: ngx-discourse-2026
Found 44 tweet day-folders
ngx.stg_tweets_raw: 360697 rows loaded from 44 day-folders
Found 55 ngx day-folders
ngx.stg_ngx_raw: 1195 rows loaded from 55 day-folders
OK: stage2_tweets_clean.sql
OK: stage2b_resolve_text.sql
OK: stage3_ngx_clean.sql
OK: stage4_interval_spine.sql
ETL complete
```

---

## Pipeline stages

### Stage 1 — Raw ingest to staging

`etl/stage1_load_staging.py` → `ngx.stg_tweets_raw`, `ngx.stg_ngx_raw`

Loads newline-delimited JSON from Cloud Storage directly into BigQuery staging tables. No transformation — staging exists so that a faulty downstream transform never requires re-reading from Cloud Storage.

**Implementation note.** BigQuery load jobs do not support multi-level wildcards such as `dt=*/*.jsonl`; such a URI resolves to zero files and fails with a 404. The loader therefore enumerates each `dt=` prefix via the Storage API and passes an explicit list of single-level glob URIs. A load job accepts up to 10,000 source URIs, well above the count in use here.

Loads run with `ignore_unknown_values=True` and `max_bad_records=50` to tolerate schema drift across collection days. Row-level errors are reported but do not abort the load.

---

### Stage 2 — Tweet deduplication

`etl/stage2_tweets_clean.sql` → `ngx.tweets_clean`

Deduplicates on `tweet_id` and derives the trading day. Three corner cases are handled here:

**Partition date is not tweet date.** The `dt=` folder reflects the *collection run* date, not the date a tweet was posted. Backfill runs place historical tweets in the folder for the day the backfill executed. The trading day is therefore derived from each tweet's own `created_at`, converted to `Africa/Lagos`. Grouping by the storage partition would badly misrepresent temporal coverage.

**Duplicates are guaranteed, not incidental.** The daily collector runs with a 26-hour lookback, deliberately overlapping roughly two hours with the previous run every day. Backfill and resume runs overlap further. Deduplication uses `ROW_NUMBER()` ordered by total engagement ascending, then `created_at` — selecting the observation with the *lowest* accumulated engagement, which most closely approximates the state visible at the time a prediction would have been made. The ordering is fully specified so the table is reproducible across runs.

**A tweet may match multiple query groups.** The collector writes one record per matching keyword group, so the same tweet can appear with different `backfill_tag` values. Rather than keeping the first tag seen, all matching groups are collected into a `matched_query_groups` array. This preserves information needed for per-group attribution in exploratory analysis.

**Analysis window guard.** Tweets collected between 1 and 7 July 2026 were gathered under an earlier, unanchored keyword dictionary that admitted large volumes of off-topic content. A boolean column `in_analysis_window` marks whether a tweet falls on or after 10 July 2026, when the corrected dictionary took effect.

```
in_analysis_window = FALSE →   109,346 tweets   (1–7 July)
in_analysis_window = TRUE  →   171,246 tweets   (10 July – 18 Sept)
```

> **Every downstream query must filter `WHERE in_analysis_window`.** The contaminated period remains in the table as an audit trail, not as analysis input.

---

### Stage 2b — Retweet text recovery

`etl/stage2b_resolve_text.sql` → `ngx.tweets_resolved`

The API returns retweet payloads containing a **truncated** copy of the source post, cut mid-sentence with an ellipsis. Within the analysis window, 38.9% of all text was affected.

This is harmless for the network features, which depend only on edge structure. It is a genuine measurement-validity problem for the narrative convergence features, because truncation is *systematic*: truncated texts cluster together in embedding space because they are truncated, not because they share meaning. Left unaddressed, this would artificially inflate the very convergence measures the study claims to observe.

Where the source post is itself present in the corpus, full text is restored by joining on `retweet_of_id`. Each row is labelled:

| `text_status` | Count | Share | Meaning |
|---|---|---|---|
| `complete` | 116,881 | 68.3% | Never truncated |
| `recovered` | 33,726 | 19.7% | Full text restored by reference |
| `truncated` | 20,639 | 12.1% | Source post absent; unrecoverable |

The residual 12.1% should be **excluded from semantic feature computation** and reported as a sensitivity check.

---

### Stage 3 — Index price cleaning

`etl/stage3_ngx_clean.sql` → `ngx.ngx_clean`

Deduplicates price snapshots and detects a provider defect.

**Initial deduplication** is on `price_last_updated`, ordered by `polled_at` — this removes re-polls that returned an unchanged reading.

**Stale record detection.** The provider intermittently returns an entire record from the previous trading day, carrying a freshly-stamped `price_last_updated`. Because the timestamp is current, timestamp-based deduplication cannot detect it.

The defect is identifiable because **both** `asi_value` and `prev_close` shift together. Since `prev_close` is invariant within a trading day by definition, any record whose `prev_close` deviates from the day's modal value is a stale response. This is an exact test, not a threshold.

```
Normal reading:   asi_value = 247654.92   prev_close = 246315.38
Stale reading:    asi_value = 246315.38   prev_close = 244791.79   ← both shifted
```

**15.0% of snapshots** were stale by this test. Affected readings are flagged via `stale_record` and the last good value within the trading day is carried forward into `current_value`. The original reading is retained in `current_value_raw` so the correction is auditable and the full analysis can be re-run on uncorrected data as a sensitivity check.

---

### Stage 4 — Interval spine

`etl/stage4_interval_spine.sql` → `ngx.interval_spine`

Constructs the analysis-ready table: one row per 20-minute interval, 21 intervals per trading day covering the 09:00–16:00 WAT session.

**The grid is generated first, then prices are joined onto it.** Building the spine from observed prices would hide missingness; generating it independently makes any gap explicit and countable.

Three exclusion rules apply:

**Weekends** are excluded by day-of-week.

**Days with fewer than three distinct index values** are excluded entirely. Two days qualified: 25 August (a single value across the full session — the index did not move at all) and 14 July (two values). These sessions contain no directional information and would contribute only majority-class weight.

**Intervals with a stale endpoint receive a null label.** Where either the opening or closing reading was a stale provider response, the true direction is unknown. Rather than infer direction from an imputed price, `direction_label` is set to `NULL`. This withholds 271 of 1,029 intervals (26.3%).

`interval_log_return` is still computed from the forward-filled series, so the continuous target remains available for time-series specifications that can carry a missingness indicator.

**Label encoding.** `close > open → 1`, otherwise `0`. Note that this collapses *down* and *unchanged* into a single class. 18.9% of cleanly observed intervals show no net movement — a genuine property of a thin frontier market, and the main driver of the 62/38 class imbalance.

---

## Output schema

`ngx.interval_spine`

| Column | Type | Description |
|---|---|---|
| `trading_day` | DATE | Trading day in `Africa/Lagos` |
| `interval_start` | TIMESTAMP | Interval opening timestamp |
| `open_value` | FLOAT | Index value at interval open (cleaned) |
| `close_value` | FLOAT | Index value at interval close (cleaned) |
| `open_stale` | BOOL | Opening reading was a stale provider response |
| `close_stale` | BOOL | Closing reading was a stale provider response |
| `open_flat` | BOOL | Opening value unchanged from prior reading |
| `close_flat` | BOOL | Closing value unchanged from prior reading |
| `missing_price` | BOOL | Either endpoint absent |
| `interval_log_return` | FLOAT | `ln(close / open)` |
| `direction_label` | INT | 1 if up, 0 if down or flat, NULL if either endpoint stale |

---

## Repository layout

```
etl/
  config.py                    Project, dataset, bucket and table constants
  stage1_load_staging.py       GCS → BigQuery staging loader
  stage2_tweets_clean.sql      Deduplication, trading day, analysis window guard
  stage2b_resolve_text.sql     Retweet text recovery
  stage3_ngx_clean.sql         Price deduplication and stale record handling
  stage4_interval_spine.sql    Interval grid, returns, direction labels
  run_etl.py                   Orchestrator
notebooks/
  colab_feasibility.ipynb      GPU throughput test for the embedding stage
  run_etl_colab.ipynb          Notebook harness for running the pipeline
requirements.txt
```

---

## Data quality summary

Three defects were identified and handled. Each would have introduced a systematic bias into downstream results had it gone undetected.

| Defect | Scale | Treatment |
|---|---|---|
| Unanchored keyword dictionary, 1–7 July | 109,346 tweets | Quarantined via `in_analysis_window` |
| Truncated retweet text | 38.9% of corpus | 62% recovered by join; residual 12.1% flagged |
| Stale provider price records | 15.0% of snapshots | Detected via `prev_close` inconsistency; labels withheld |

Original values are preserved in every case — `tweets_clean` retains the contaminated period, `text_status` records recovery provenance, and `current_value_raw` retains uncorrected prices. Each correction can therefore be reversed for sensitivity analysis.

---

## Known limitations

**Retrieval timestamp absent for tweets.** The collector does not record a retrieval time, so deduplication uses accumulated engagement as a proxy for recency. This is a reasonable ordering but not a direct measurement.

**Engagement captured at collection, not posting.** Retweet and like counts reflect the moment of collection. Backfilled tweets have had longer to accumulate engagement than daily-collected ones, which affects any engagement-weighted feature.

**Final interval close depends on a post-session poll.** The 16:00 WAT reading closes the 15:40–16:00 interval. On one day (8 September) this reading was stale and the last good value was carried forward, so the recorded close is approximate.

**Stale detection cannot distinguish a genuinely flat session.** On days with no price variation, the `prev_close` test may fire on readings that are not in fact stale. Only three days across the collection period exhibited this pattern, two of which are excluded by the distinct-value rule.

---

## Re-running and reproducibility

Every stage is deterministic and idempotent. Dedup ordering is fully specified, exclusion rules are explicit in SQL rather than applied ad hoc, and no stage depends on execution order beyond the sequence in `run_etl.py`.

To reproduce the reported figures, run the pipeline against the frozen raw data snapshot and execute the verification queries in `notebooks/run_etl_colab.ipynb`.

**Before submission:** extend the date bound in `stage4_interval_spine.sql` to the freeze date, snapshot `raw/tweets/` and `raw/ngx/` to a frozen prefix, and tag the repository. Collection continuing past the freeze must not alter the submitted analysis.
