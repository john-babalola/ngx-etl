CREATE OR REPLACE TABLE `{PROJECT}.{TWEETS_CLEAN}` AS
WITH ranked AS (
  SELECT
    *,
    -- Corner case 1: trading_day from created_at, NEVER from the dt= partition
    DATE(DATETIME(TIMESTAMP(created_at), "Africa/Lagos")) AS trading_day_raw,
    -- Corner case 2: dedupe on tweet_id, keep earliest retrieval
    ROW_NUMBER() OVER (
      PARTITION BY post_id
      ORDER BY retrieved_at ASC
    ) AS rn
  FROM `{PROJECT}.{TWEETS_RAW_STAGING}`
),
deduped AS (
  SELECT * EXCEPT(rn) FROM ranked WHERE rn = 1
),
-- Corner case 3: collapse multi-group matches into an array rather than first-seen
grouped AS (
  SELECT
    post_id,
    ANY_VALUE(author_hash)      AS author_hash,
    ANY_VALUE(created_at)       AS created_at,
    ANY_VALUE(trading_day_raw)  AS trading_day,
    ANY_VALUE(text)             AS text,
    ANY_VALUE(reply_to_hash)    AS reply_to_hash,
    ANY_VALUE(repost_of_id)     AS repost_of_id,
    ANY_VALUE(quote_of_id)      AS quote_of_id,
    ANY_VALUE(repost_count)     AS repost_count,
    ANY_VALUE(reply_count)      AS reply_count,
    ANY_VALUE(like_count)       AS like_count,
    ANY_VALUE(author_followers) AS author_followers,
    ANY_VALUE(lang)             AS lang,
    ARRAY_AGG(DISTINCT backfill_tag IGNORE NULLS) AS matched_query_groups
  FROM deduped
  GROUP BY post_id
)
SELECT * FROM grouped;