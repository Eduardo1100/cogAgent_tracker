from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.storage.database import get_db
from src.storage import cache, s3
from src.models.db import Prediction

router = APIRouter()

@router.post("/test-stack")
async def run_integration_test(db: Session = Depends(get_db)):
    # 1. Save to Postgres
    new_pred = Prediction(filename="test.jpg", result="AI_SUCCESS", confidence=99)
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)

    # 2. Save to Redis (Cache the ID for 60 seconds)
    cache.set_cache(f"last_id", str(new_pred.id), expire=60)

    # 3. Save a "dummy log" to MinIO
    s3_uri = s3.upload_file(b"Test Log Data", "ai-logs", f"log_{new_pred.id}.txt")

    return {
        "database_id": new_pred.id,
        "cached_val": cache.get_cache("last_id"),
        "storage_uri": s3_uri,
        "status": "Integration Success"
    }