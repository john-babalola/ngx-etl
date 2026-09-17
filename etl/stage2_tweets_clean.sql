CREATE OR REPLACE TABLE `{PROJECT}.{TWEETS_CLEAN}` AS
WITH deduped AS (
  SELECT
    *,
    DATE(DATETIME(created_at, "Africa/Lagos")) AS trading_day,
    ROW_NUMBER() OVER (PARTITION BY tweet_id) AS rn
  FROM `{PROJECT}.{TWEETS_RAW_STAGING}`
),
grouped AS (
  SELECT
    tweet_id,
    ANY_VALUE(author_hash)      AS author_hash,
    ANY_VALUE(created_at)       AS created_at,
    ANY_VALUE(trading_day)      AS trading_day,
    ANY_VALUE(text)             AS text,
    ANY_VALUE(reply_to_hash)    AS reply_to_hash,
    ANY_VALUE(retweet_of_id)    AS retweet_of_id,
    ANY_VALUE(quote_of_id)      AS quote_of_id,
    ANY_VALUE(retweet_count)    AS retweet_count,
    ANY_VALUE(reply_count)      AS reply_count,
    ANY_VALUE(like_count)       AS like_count,
    ANY_VALUE(author_followers) AS author_followers,
    ANY_VALUE(lang)             AS lang,
    ARRAY_AGG(DISTINCT backfill_tag IGNORE NULLS) AS matched_query_groups
  FROM deduped
  WHERE rn = 1
  GROUP BY tweet_id
)
SELECT * FROM grouped;