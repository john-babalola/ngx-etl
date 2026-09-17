from google.cloud import bigquery
from etl.config import BUCKET, TWEETS_RAW_STAGING, NGX_RAW_STAGING


def load_staging(client, gcs_glob: str, table_id: str):
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
        ignore_unknown_values=True,
        max_bad_records=50,
    )
    load_job = client.load_table_from_uri(gcs_glob, table_id, job_config=job_config)
    load_job.result()
    tbl = client.get_table(table_id)
    print(f"{table_id}: {tbl.num_rows} rows loaded")
    if load_job.errors:
        print(f"  WARNING: {len(load_job.errors)} row-level errors")
    return load_job


def load_all(client):
    load_staging(client, f"gs://{BUCKET}/raw/tweets/dt=*/*.jsonl", TWEETS_RAW_STAGING)
    load_staging(client, f"gs://{BUCKET}/raw/ngx/dt=*/*.json", NGX_RAW_STAGING)