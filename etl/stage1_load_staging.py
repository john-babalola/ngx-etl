from google.cloud import bigquery
from google.cloud import storage
from etl.config import BUCKET, TWEETS_RAW_STAGING, NGX_RAW_STAGING


def list_day_uris(bucket_name: str, prefix: str, pattern: str) -> list:
    """List all dt=YYYY-MM-DD/ subfolders under prefix, return one glob URI per day."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    list(blobs)  # force iteration to populate .prefixes
    day_prefixes = sorted(blobs.prefixes)
    return [f"gs://{bucket_name}/{p}{pattern}" for p in day_prefixes]


def load_staging(client, uris: list, table_id: str):
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_TRUNCATE",
        autodetect=True,
        ignore_unknown_values=True,
        max_bad_records=50,
    )
    load_job = client.load_table_from_uri(uris, table_id, job_config=job_config)
    load_job.result()
    tbl = client.get_table(table_id)
    print(f"{table_id}: {tbl.num_rows} rows loaded from {len(uris)} day-folders")
    if load_job.errors:
        print(f"  WARNING: {len(load_job.errors)} row-level errors")
    return load_job


def load_all(client):
    tweet_uris = list_day_uris(BUCKET, "raw/tweets/", "*.jsonl")
    print(f"Found {len(tweet_uris)} tweet day-folders")
    load_staging(client, tweet_uris, TWEETS_RAW_STAGING)

    ngx_uris = list_day_uris(BUCKET, "raw/ngx/", "*.json")
    print(f"Found {len(ngx_uris)} ngx day-folders")
    load_staging(client, ngx_uris, NGX_RAW_STAGING)