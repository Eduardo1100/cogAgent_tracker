import os

import boto3
from botocore.client import Config


def get_s3_client():
    """Creates and returns a connection to the MinIO object store."""
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT", "http://objectstore:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", "miniouser"),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", "miniopass"),
        # MinIO works best with s3v4 signatures
        config=Config(signature_version="s3v4"),
        # Region is required by boto3, even for local MinIO
        region_name="us-east-1",
    )


def upload_file(data: bytes, bucket: str, key: str) -> str:
    """Upload bytes to MinIO/S3 and return an s3:// URI."""
    client = get_s3_client()
    client.put_object(Body=data, Bucket=bucket, Key=key)
    return f"s3://{bucket}/{key}"
