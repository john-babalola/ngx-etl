CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.tweet_intraday_window` AS
WITH day_open AS (
  SELECT
    trading_day,
    TIMESTAMP(DATETIME(trading_day, TIME(9, 0, 0)), "Africa/Lagos") AS market_open
  FROM (SELECT DISTINCT trading_day FROM `{PROJECT}.{INTERVAL_SPINE}`)
)
SELECT
  t.tweet_id,
  s.trading_day,
  s.interval_start,
  s.interval_start AS feature_query_cutoff,
  TIMESTAMP_DIFF(
    s.interval_start,
    GREATEST(TIMESTAMP_SUB(s.interval_start, INTERVAL 60 MINUTE), o.market_open),
    MINUTE
  ) AS lookback_minutes_available
FROM `{PROJECT}.{TWEETS_RESOLVED}` t
JOIN `{PROJECT}.{INTERVAL_SPINE}` s
  ON t.trading_day = s.trading_day
JOIN day_open o
  ON o.trading_day = s.trading_day
WHERE t.in_analysis_window
  AND t.created_at <  s.interval_start
  AND t.created_at >= GREATEST(TIMESTAMP_SUB(s.interval_start, INTERVAL 60 MINUTE), o.market_open);