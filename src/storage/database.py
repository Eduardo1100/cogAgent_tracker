import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Manually load the .env file so os.getenv can see your variables
load_dotenv()

# 2. Grab the URL and apply "Auto-Fixes"
raw_url = os.getenv("DATABASE_URL")

if not raw_url:
    # Default to localhost and psycopg2 for local development (uv run)
    DATABASE_URL = "postgresql+psycopg2://devuser:devpass@localhost:5432/devdb"
else:
    # Fix the "postgres://" vs "postgresql://" issue automatically
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)

    # Ensure we use the psycopg2 driver we just installed
    if "psycopg2" not in raw_url:
        DATABASE_URL = raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    else:
        DATABASE_URL = raw_url

# 3. Create the engine
engine = create_engine(DATABASE_URL)

# 4. Create a Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 5. Create the Base class for models
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
