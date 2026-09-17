CREATE OR REPLACE TABLE `{PROJECT}.{NGX_CLEAN}` AS
WITH ranked AS (
  SELECT
    *,
    -- bucket to the containing 20-min interval, aligned to 09:00 WAT
    TIMESTAMP_SECONDS(
      DIV(UNIX_SECONDS(price_last_updated), 1200) * 1200
    ) AS interval_bucket,
    ROW_NUMBER() OVER (
      PARTITION BY price_last_updated
      ORDER BY polled_at ASC
    ) AS rn
  FROM `{PROJECT}.{NGX_RAW_STAGING}`
)
SELECT
  interval_bucket,
  DATE(DATETIME(interval_bucket, "Africa/Lagos")) AS trading_day,
  asi_value AS current_value,
  prev_close,
  price_change,
  price_change_pct,
  price_last_updated,
  polled_at,
  LAG(asi_value) OVER (ORDER BY interval_bucket) = asi_value AS is_stale
FROM ranked
WHERE rn = 1
ORDER BY interval_bucket;