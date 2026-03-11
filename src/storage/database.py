import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# 1. Manually load the .env file so os.getenv can see your variables
load_dotenv()

# 2. Grab the URL and apply "Auto-Fixes"
def get_database_url() -> str:
    raw_url = os.getenv("DATABASE_URL")

    if not raw_url:
        return "postgresql+psycopg2://devuser:devpass@localhost:5432/devdb"

    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)

    if "psycopg2" not in raw_url:
        return raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    return raw_url


DATABASE_URL = get_database_url()

# 3. Create the engine
engine = create_engine(DATABASE_URL)

# 4. Create a Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
