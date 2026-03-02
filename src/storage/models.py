from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, Float, Boolean, Text, JSON, DateTime, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 1. Base class for all models
class Base(DeclarativeBase):
    pass

# 2. The overarching Experiment
class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    
    # Configuration
    agent_name: Mapped[str] = mapped_column(String(100), index=True) # e.g., "GWTAutogenAgent"
    llm_model: Mapped[str] = mapped_column(String(50))               # e.g., "gpt-4o"
    eval_env_type: Mapped[str] = mapped_column(String(100))          # e.g., "AlfredThorEnv"
    long_term_guidance: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Global Limits
    max_actions_per_game: Mapped[int] = mapped_column(Integer)
    max_chat_rounds: Mapped[int] = mapped_column(Integer)
    
    # Metadata
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    total_runtime_minutes: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Relationship: One Experiment has Many Episodes
    episodes: Mapped[List["EpisodeRun"]] = relationship("EpisodeRun", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ExperimentRun {self.id}: {self.agent_name} on {self.eval_env_type}>"

# 3. The individual ALFWorld Game iteration
class EpisodeRun(Base):
    __tablename__ = "episode_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiment_runs.id"), index=True)
    
    # Game Identifiers
    game_number: Mapped[int] = mapped_column(Integer) # The 'i' in your script
    
    # Hard Metrics (Mapped directly from your wandb.log)
    success: Mapped[bool] = mapped_column(Boolean)
    actions_taken: Mapped[int] = mapped_column(Integer)
    chat_rounds: Mapped[int] = mapped_column(Integer)
    runtime_minutes: Mapped[float] = mapped_column(Float)
    
    # Agent Meta-Reasoning Data
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transitions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)     # Stores the from/to chat transitions
    belief_state: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)    # Stores episodic memory matches
    
    # S3 Artifact Link (The Vault)
    chat_history_s3_key: Mapped[Optional[str]] = mapped_column(String(255), nullable=True) # e.g., "experiments/run_1/game_124_chat.txt"

    # Relationship back to the parent Experiment
    experiment: Mapped["ExperimentRun"] = relationship("ExperimentRun", back_populates="episodes")

    def __repr__(self) -> str:
        return f"<EpisodeRun Game #{self.game_number} | Success: {self.success}>"