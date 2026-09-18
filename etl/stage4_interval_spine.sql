CREATE OR REPLACE TABLE `{PROJECT}.{INTERVAL_SPINE}` AS
WITH trading_days AS (
  -- weekdays only, within your actual collection window
  SELECT day
  FROM UNNEST(GENERATE_DATE_ARRAY('2026-07-10', '2026-09-18')) AS day
  WHERE EXTRACT(DAYOFWEEK FROM day) NOT IN (1, 7)  -- exclude Sat/Sun
  -- TODO: subtract Nigerian public holidays manually once confirmed
),
intervals AS (
  SELECT interval_start
  FROM UNNEST(
    GENERATE_TIMESTAMP_ARRAY(
      TIMESTAMP('1970-01-01 09:00:00'),
      TIMESTAMP('1970-01-01 15:40:00'),
      INTERVAL 20 MINUTE
    )
  ) AS interval_start
),
spine AS (
  SELECT
    d.day AS trading_day,
    TIMESTAMP(DATETIME(d.day, TIME(EXTRACT(HOUR FROM i.interval_start), EXTRACT(MINUTE FROM i.interval_start), 0)), "Africa/Lagos") AS interval_start
  FROM trading_days d
  CROSS JOIN intervals i
),
priced AS (
  SELECT
    s.trading_day,
    s.interval_start,
    p_open.current_value  AS open_value,
    p_close.current_value AS close_value,
    p_open.is_stale  AS open_stale,
    p_close.is_stale AS close_stale
  FROM spine s
  LEFT JOIN `{PROJECT}.{NGX_CLEAN}` p_open
    ON p_open.interval_bucket = s.interval_start
  LEFT JOIN `{PROJECT}.{NGX_CLEAN}` p_close
    ON p_close.interval_bucket = TIMESTAMP_ADD(s.interval_start, INTERVAL 20 MINUTE)
)
SELECT
  trading_day,
  interval_start,
  open_value,
  close_value,
  open_stale,
  close_stale,
  (open_value IS NULL OR close_value IS NULL) AS missing_price,
  SAFE.LN(close_value / open_value) AS interval_log_return,
  CASE WHEN close_value > open_value THEN 1
       WHEN close_value <= open_value THEN 0
       ELSE NULL END AS direction_label
FROM priced;