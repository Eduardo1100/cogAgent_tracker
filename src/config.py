# src/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # These match the names in your docker-compose.yml environment
    DATABASE_URL: str
    REDIS_URL: str
    S3_ENDPOINT: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()  # type: ignore[call-arg]
