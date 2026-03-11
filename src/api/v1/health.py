import torch
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.cache import get_redis_client
from src.storage.database import get_db
from src.storage.s3 import get_s3_client

router = APIRouter()


@router.get("/ai")
def check_ai_env():
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    return {
        "status": "ready",
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "active_device": device_name,
        "message": f"PyTorch is initialized and processing tensors on the {device_name.upper()}!",
    }


@router.get("/db")
def check_db_env(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT version();")).scalar()
        return {
            "status": "connected",
            "postgres_version": result,
            "message": "Successfully connected to the PostgreSQL database!",
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the database.",
        }


@router.get("/storage")
def check_storage_env():
    s3 = get_s3_client()
    bucket_name = "test-bucket"
    file_name = "success.txt"
    file_content = b"Congratulations! Your FastAPI server can talk directly to the MinIO Object Store."

    try:
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError:
            s3.create_bucket(Bucket=bucket_name)

        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_content,
            ContentType="text/plain",
        )

        raw_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket_name, "Key": file_name},
            ExpiresIn=3600,
        )
        download_url = raw_url.replace(
            "http://objectstore:9000", "http://localhost:9000"
        )

        return {
            "status": "connected",
            "message": f"Successfully created bucket and uploaded '{file_name}'!",
            "download_url": download_url,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the object store.",
        }


@router.get("/cache")
def check_cache_env():
    try:
        r = get_redis_client()
        r.setex(
            "infrastructure_test", 60, "Redis is lightning fast and fully operational!"
        )
        cached_value = r.get("infrastructure_test")
        return {
            "status": "connected",
            "message": "Successfully read and wrote to the Redis cache.",
            "retrieved_value": cached_value,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the Redis cache.",
        }
