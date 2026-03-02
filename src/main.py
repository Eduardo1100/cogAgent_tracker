from fastapi import FastAPI, Depends
import torch
from sqlalchemy import text
from sqlalchemy.orm import Session
from src.storage.database import get_db
from botocore.exceptions import ClientError
from src.storage.s3 import get_s3_client
from src.storage.cache import get_redis_client
from src.storage.database import engine, Base
from src.storage.models import ExperimentRun, EpisodeRun

# Tell SQLAlchemy to create the tables if they don't exist yet
Base.metadata.create_all(bind=engine)

# Initialize the app
app = FastAPI(title="My Awesome AI API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the API! The engine is running."}

@app.get("/health/ai")
def check_ai_env():
    # 1. Check if an NVIDIA GPU is available
    cuda_available = torch.cuda.is_available()
    
    # 2. Determine which device PyTorch will use for tensor calculations
    device_name = "cuda" if cuda_available else "cpu"
    
    return {
        "status": "ready",
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "active_device": device_name,
        "message": f"PyTorch is initialized and processing tensors on the {device_name.upper()}!"
    }

@app.get("/health/db")
def check_db_env(db: Session = Depends(get_db)):
    try:
        # Execute a raw SQL query to ask Postgres for its exact version
        result = db.execute(text("SELECT version();")).scalar()
        
        return {
            "status": "connected",
            "postgres_version": result,
            "message": "Successfully connected to the PostgreSQL database!"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the database."
        }

@app.get("/health/storage")
def check_storage_env():
    s3 = get_s3_client()
    bucket_name = "test-bucket"
    file_name = "success.txt"
    file_content = b"Congratulations! Your FastAPI server can talk directly to the MinIO Object Store."

    try:
        # 1. Check if the bucket exists, create it if it doesn't
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError:
            s3.create_bucket(Bucket=bucket_name)

        # 2. Upload the file into the bucket
        s3.put_object(
            Bucket=bucket_name,
            Key=file_name,
            Body=file_content,
            ContentType="text/plain"
        )
        
        # 3. Generate a secure, 1-hour download link
        raw_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': file_name},
            ExpiresIn=3600
        )
        
        # 4. Swap the internal Docker hostname for localhost so your browser can reach it
        download_url = raw_url.replace("http://objectstore:9000", "http://localhost:9000")
        
        return {
            "status": "connected",
            "message": f"Successfully created bucket and uploaded '{file_name}'!",
            "download_url": download_url
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the object store."
        }
    
@app.get("/health/cache")
def check_cache_env():
    try:
        # 1. Connect to Redis
        r = get_redis_client()
        
        # 2. Set a test key with a 60-second expiration (setex = Set with Expiration)
        r.setex("infrastructure_test", 60, "Redis is lightning fast and fully operational! ⚡")
        
        # 3. Retrieve the value we just set
        cached_value = r.get("infrastructure_test")
        
        return {
            "status": "connected",
            "message": "Successfully read and wrote to the Redis cache.",
            "retrieved_value": cached_value
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "message": "Could not connect to the Redis cache."
        }