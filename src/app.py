from fastapi import FastAPI

from src.config.env_validation import require_env_vars
from src.config.schema_health import require_current_schema

require_env_vars(
    ["DATABASE_URL", "REDIS_URL", "S3_ENDPOINT", "S3_ACCESS_KEY", "S3_SECRET_KEY"],
    context="API startup",
)
require_current_schema(context="API startup")

app = FastAPI(title="cogAgent Tracker API")

from src.api.v1.endpoints import router as v1_router
from src.api.v1.health import router as health_router

app.include_router(v1_router, prefix="/api/v1")
app.include_router(health_router, prefix="/health")


@app.get("/")
def read_root():
    return {"message": "Welcome to the API! The engine is running."}
