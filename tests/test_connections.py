import os

import boto3
import psycopg
import redis
from botocore.config import Config


def test_all():
    print("🧪 Starting Infrastructure Health Check...\n")

    # 1. Test PostgreSQL
    try:
        conn_str = os.getenv("DATABASE_URL", "")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                row = cur.fetchone()
                print(f"✅ Postgres: Connected! ({row[0][:25] if row else 'unknown'}...)")
    except Exception as e:
        print(f"❌ Postgres: Failed! {e}")

    # 2. Test Redis
    try:
        r = redis.from_url(os.getenv("REDIS_URL", ""))
        r.ping()
        print("✅ Redis: Connected! (PONG)")
    except Exception as e:
        print(f"❌ Redis: Failed! {e}")

    # 3. Test MinIO (S3)
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT"),
            aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        s3.list_buckets()
        print("✅ MinIO: Connected! (S3 API is live)")
    except Exception as e:
        print(f"❌ MinIO: Failed! {e}")


if __name__ == "__main__":
    test_all()
