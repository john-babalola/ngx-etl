CREATE OR REPLACE TABLE `{PROJECT}.{NGX_CLEAN}` AS
WITH ranked AS (
  SELECT
    *,
    DATETIME(TIMESTAMP(price_last_updated), "Africa/Lagos") AS updated_wat,
    -- bucket to the containing 20-min interval, aligned to 09:00
    TIMESTAMP_SECONDS(
      DIV(UNIX_SECONDS(TIMESTAMP(price_last_updated)), 1200) * 1200
    ) AS interval_bucket,
    ROW_NUMBER() OVER (
      PARTITION BY price_last_updated
      ORDER BY retrieved_at ASC
    ) AS rn
  FROM `{PROJECT}.{NGX_RAW_STAGING}`
)
SELECT
  interval_bucket,
  DATE(DATETIME(interval_bucket, "Africa/Lagos")) AS trading_day,
  current_value,
  price_last_updated,
  -- staleness flag: did this reading actually refresh vs. the prior retrieval?
  LAG(current_value) OVER (ORDER BY interval_bucket) = current_value AS is_stale
FROM ranked
WHERE rn = 1
ORDER BY interval_bucket;