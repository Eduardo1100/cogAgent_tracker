from fastapi import FastAPI

from src.api.v1.endpoints import router as v1_router
from src.api.v1.health import router as health_router
from src.storage.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="cogAgent Tracker API")

app.include_router(v1_router, prefix="/api/v1")
app.include_router(health_router, prefix="/health")


@app.get("/")
def read_root():
    return {"message": "Welcome to the API! The engine is running."}
