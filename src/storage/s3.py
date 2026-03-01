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
        region_name="us-east-1" 
    )

def upload_file(file_data, bucket: str, object_name: str):
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=bucket)
    except:
        s3_client.create_bucket(Bucket=bucket)
    
    s3_client.put_object(Bucket=bucket, Key=object_name, Body=file_data)
    return f"s3://{bucket}/{object_name}"