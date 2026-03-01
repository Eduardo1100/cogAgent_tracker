import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. Grab the URL from the Docker environment (fallback to a default just in case)
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://devuser:devpass@db:5432/devdb"
)

# 2. Create the engine
engine = create_engine(DATABASE_URL)

# 3. Create a Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Create the Base class for models
Base = declarative_base()

# 5. Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()