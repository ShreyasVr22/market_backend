"""Utilities to sync model files from S3/MinIO into a local directory.

This module performs a safe, limited download of objects under a prefix
into a target directory. It uses `boto3` and respects an optional
`endpoint_url` for MinIO or self-hosted S3-compatible stores.
"""
from typing import Optional
import os

def sync_models_from_s3(bucket: str, prefix: str, target_dir: str, endpoint_url: Optional[str] = None, max_files: int = 1000):
    """Download files from S3/MinIO under `prefix` into `target_dir`.

    - `bucket`: S3 bucket name
    - `prefix`: key prefix (e.g., 'trained_models/')
    - `target_dir`: local directory to download into (will be created)
    - `endpoint_url`: optional custom S3 endpoint (for MinIO)
    - `max_files`: safety cap on number of files to download
    """
    try:
        import boto3
    except Exception as e:
        raise RuntimeError("boto3 is required for S3 sync: add to requirements.txt") from e

    os.makedirs(target_dir, exist_ok=True)

    session = boto3.session.Session()
    client = session.client('s3', endpoint_url=endpoint_url) if endpoint_url else session.client('s3')

    paginator = client.get_paginator('list_objects_v2')
    downloaded = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []) or []:
            key = obj.get('Key')
            # keep relative path after prefix
            if not key or not key.startswith(prefix):
                continue
            rel = key[len(prefix):]
            if not rel:
                continue
            dest = os.path.join(target_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                client.download_file(bucket, key, dest)
                downloaded += 1
            except Exception:
                # ignore individual file failures
                continue
            if downloaded >= max_files:
                return downloaded

    return downloaded
