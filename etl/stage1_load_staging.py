def load_staging(gcs_glob: str, table_id: str, source_format="NEWLINE_DELIMITED_JSON"):
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition="WRITE_TRUNCATE",   # staging is always a full rebuild
        autodetect=True,
        ignore_unknown_values=True,           # tolerate schema drift across days
        max_bad_records=50,                   # log and skip malformed lines, don't abort the whole load
    )
    load_job = client.load_table_from_uri(gcs_glob, table_id, job_config=job_config)
    load_job.result()  # blocks until done
    tbl = client.get_table(table_id)
    print(f"{table_id}: {tbl.num_rows} rows loaded")
    return load_job

# Tweets
load_staging(f"gs://{BUCKET}/raw/tweets/dt=*/*.jsonl", TWEETS_RAW_STAGING)

# NGX price snapshots
load_staging(f"gs://{BUCKET}/raw/ngx/dt=*/*.json", NGX_RAW_STAGING)