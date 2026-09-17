from pathlib import Path
from google.cloud import bigquery
from etl.config import (
    PROJECT, TWEETS_RAW_STAGING, NGX_RAW_STAGING,
    TWEETS_CLEAN, NGX_CLEAN, INTERVAL_SPINE,
)

client = bigquery.Client(project=PROJECT)
SQL_DIR = Path(__file__).parent


def run_sql(filename: str, **params):
    sql = (SQL_DIR / filename).read_text().format(**params)
    job = client.query(sql)
    job.result()
    print(f"OK: {filename}")


if __name__ == "__main__":
    print("BQ client ready:", client.project)

    from etl.stage1_load_staging import load_all
    load_all(client)

    run_sql("stage2_tweets_clean.sql",
             PROJECT=PROJECT, TWEETS_CLEAN=TWEETS_CLEAN, TWEETS_RAW_STAGING=TWEETS_RAW_STAGING)
    run_sql("stage3_ngx_clean.sql",
             PROJECT=PROJECT, NGX_CLEAN=NGX_CLEAN, NGX_RAW_STAGING=NGX_RAW_STAGING)
    run_sql("stage4_interval_spine.sql",
             PROJECT=PROJECT, INTERVAL_SPINE=INTERVAL_SPINE, NGX_CLEAN=NGX_CLEAN)

    print("ETL complete")