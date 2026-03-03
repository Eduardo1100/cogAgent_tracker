from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.sql import func

from src.storage.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)  # Name of the file in MinIO
    result = Column(String)  # AI Model Output (e.g., "Cat")
    confidence = Column(Integer)  # 0-100
    created_at = Column(DateTime(timezone=True), server_default=func.now())
