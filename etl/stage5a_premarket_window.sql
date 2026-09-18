CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.tweet_premarket_window` AS
WITH trading_calendar AS (
  SELECT DISTINCT trading_day
  FROM `{PROJECT}.{INTERVAL_SPINE}`
),
ordered_days AS (
  SELECT
    trading_day,
    LAG(trading_day) OVER (ORDER BY trading_day) AS prior_trading_day
  FROM trading_calendar
),
windows AS (
  SELECT
    trading_day,
    -- window opens at the prior trading day's session close (16:00 WAT).
    -- first day in the calendar has no prior trading day, so falls back
    -- to the previous calendar day as an edge-case approximation.
    TIMESTAMP(DATETIME(
      COALESCE(prior_trading_day, DATE_SUB(trading_day, INTERVAL 1 DAY)),
      TIME(16, 0, 0)
    ), "Africa/Lagos") AS window_start,
    -- window closes at this day's market open (09:00 WAT)
    TIMESTAMP(DATETIME(trading_day, TIME(9, 0, 0)), "Africa/Lagos") AS window_end,
    prior_trading_day IS NULL AS is_first_day_edge_case
  FROM ordered_days
)
SELECT
  t.tweet_id,
  w.trading_day,
  w.window_start,
  w.window_end,
  w.window_end AS feature_query_cutoff,
  w.is_first_day_edge_case,
  TIMESTAMP_DIFF(w.window_end, w.window_start, HOUR) AS window_span_hours
FROM `{PROJECT}.{TWEETS_RESOLVED}` t
JOIN windows w
  ON t.created_at >= w.window_start
 AND t.created_at <  w.window_end
WHERE t.in_analysis_window;