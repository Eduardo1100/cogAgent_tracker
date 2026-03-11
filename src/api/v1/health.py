import torch
from botocore.exceptions import ClientError
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config.schema_health import get_schema_revision_status
from src.storage.cache import get_redis_client
from src.storage.database import get_db
from src.storage.s3 import get_s3_client

router = APIRouter()


def _ai_health() -> dict:
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    return {
        "status": "ready",
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "active_device": device_name,
        "message": f"PyTorch is initialized and processing tensors on the {device_name.upper()}!",
    }


def _db_health(db: Session) -> dict:
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


def _schema_health() -> dict:
    try:
        schema_status = get_schema_revision_status()
        return {
            "status": "current" if schema_status["schema_ok"] else "out_of_date",
            "message": (
                "Database schema revision matches Alembic head."
                if schema_status["schema_ok"]
                else "Database schema revision does not match Alembic head."
            ),
            **schema_status,
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not determine database schema revision.",
        }


def _storage_health() -> dict:
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
        download_url = raw_url.replace("http://objectstore:9000", "http://localhost:9000")

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


def _cache_health() -> dict:
    try:
        r = get_redis_client()
        r.setex("infrastructure_test", 60, "Redis is lightning fast and fully operational!")
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


@router.get("")
def check_all_health(db: Session = Depends(get_db)):
    checks = {
        "ai": _ai_health(),
        "db": _db_health(db),
        "schema": _schema_health(),
        "storage": _storage_health(),
        "cache": _cache_health(),
    }
    overall_status = "healthy"
    if any(check["status"] == "failed" for check in checks.values()):
        overall_status = "degraded"
    elif any(check["status"] in {"out_of_date"} for check in checks.values()):
        overall_status = "warning"

    return {
        "status": overall_status,
        "checks": checks,
    }


@router.get("/ai")
def check_ai_env():
    return _ai_health()


@router.get("/db")
def check_db_env(db: Session = Depends(get_db)):
    return _db_health(db)


@router.get("/schema")
def check_schema_env():
    return _schema_health()


@router.get("/storage")
def check_storage_env():
    return _storage_health()


@router.get("/cache")
def check_cache_env():
    return _cache_health()
