CREATE OR REPLACE TABLE `{PROJECT}.{NGX_CLEAN}` AS
WITH ranked AS (
  SELECT
    *,
    DATE(DATETIME(price_last_updated, "Africa/Lagos")) AS trading_day,
    TIMESTAMP_SECONDS(
      DIV(UNIX_SECONDS(price_last_updated), 1200) * 1200
    ) AS interval_bucket,
    ROW_NUMBER() OVER (
      PARTITION BY price_last_updated
      ORDER BY polled_at ASC
    ) AS rn
  FROM `{PROJECT}.{NGX_RAW_STAGING}`
),
dedup AS (
  SELECT * EXCEPT(rn) FROM ranked WHERE rn = 1
),
modal AS (
  SELECT
    trading_day,
    APPROX_TOP_COUNT(prev_close, 1)[OFFSET(0)].value AS modal_prev_close
  FROM dedup
  GROUP BY trading_day
),
flagged AS (
  SELECT
    d.*,
    m.modal_prev_close,
    d.prev_close != m.modal_prev_close AS stale_record
  FROM dedup d
  JOIN modal m USING (trading_day)
),
filled AS (
  SELECT
    *,
    LAST_VALUE(IF(stale_record, NULL, asi_value) IGNORE NULLS) OVER (
      PARTITION BY trading_day
      ORDER BY interval_bucket
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS asi_value_clean
  FROM flagged
)
SELECT
  interval_bucket,
  trading_day,
  asi_value_clean AS current_value,
  asi_value       AS current_value_raw,
  stale_record,
  modal_prev_close AS prev_close,
  price_change,
  price_change_pct,
  price_last_updated,
  polled_at,
  LAG(asi_value_clean) OVER (ORDER BY interval_bucket) = asi_value_clean AS is_flat
FROM filled
ORDER BY interval_bucket;