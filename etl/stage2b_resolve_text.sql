CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.tweets_resolved` AS
WITH resolved AS (
  SELECT
    b.* EXCEPT(text),
    CASE
      WHEN ENDS_WITH(b.text, '…') AND o.text IS NOT NULL THEN o.text
      ELSE b.text
    END AS text,
    CASE
      WHEN ENDS_WITH(b.text, '…') AND o.text IS NOT NULL THEN 'recovered'
      WHEN ENDS_WITH(b.text, '…') THEN 'truncated'
      ELSE 'complete'
    END AS text_status
  FROM `{PROJECT}.{TWEETS_CLEAN}` b
  LEFT JOIN `{PROJECT}.{TWEETS_CLEAN}` o
    ON o.tweet_id = b.retweet_of_id
)
SELECT * FROM resolved;