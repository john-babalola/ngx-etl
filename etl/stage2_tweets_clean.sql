CREATE OR REPLACE TABLE `{PROJECT}.{TWEETS_CLEAN}` AS
WITH deduped AS (
  SELECT
    *,
    DATE(DATETIME(created_at, "Africa/Lagos")) AS trading_day,
    ROW_NUMBER() OVER (
      PARTITION BY tweet_id
      ORDER BY COALESCE(like_count, 0) + COALESCE(retweet_count, 0) ASC,
               created_at ASC
    ) AS rn
  FROM `{PROJECT}.{TWEETS_RAW_STAGING}`
),
grouped AS (
  SELECT
    tweet_id,
    ANY_VALUE(author_hash      HAVING MIN rn) AS author_hash,
    ANY_VALUE(created_at       HAVING MIN rn) AS created_at,
    ANY_VALUE(trading_day      HAVING MIN rn) AS trading_day,
    ANY_VALUE(text             HAVING MIN rn) AS text,
    ANY_VALUE(reply_to_hash    HAVING MIN rn) AS reply_to_hash,
    ANY_VALUE(retweet_of_id    HAVING MIN rn) AS retweet_of_id,
    ANY_VALUE(quote_of_id      HAVING MIN rn) AS quote_of_id,
    ANY_VALUE(retweet_count    HAVING MIN rn) AS retweet_count,
    ANY_VALUE(reply_count      HAVING MIN rn) AS reply_count,
    ANY_VALUE(like_count       HAVING MIN rn) AS like_count,
    ANY_VALUE(author_followers HAVING MIN rn) AS author_followers,
    ANY_VALUE(lang             HAVING MIN rn) AS lang,
    ARRAY_AGG(DISTINCT backfill_tag IGNORE NULLS) AS matched_query_groups
  FROM deduped
  GROUP BY tweet_id
)
SELECT
  *,
  trading_day >= '2026-07-10' AS in_analysis_window
FROM grouped;