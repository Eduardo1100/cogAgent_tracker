from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# 1. Base class for all models
class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id = mapped_column(Integer, primary_key=True, index=True)
    filename = mapped_column(String)
    result = mapped_column(String)
    confidence = mapped_column(Integer)
    created_at = mapped_column(DateTime(timezone=True), server_default="now()")


# 2. The overarching Experiment
class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Configuration
    agent_name: Mapped[str] = mapped_column(
        String(100), index=True
    )  # e.g., "GWTAutogenAgent"
    llm_model: Mapped[str] = mapped_column(String(50))  # e.g., "gpt-4o"
    eval_env_type: Mapped[str] = mapped_column(String(100))  # e.g., "AlfredThorEnv"
    long_term_guidance: Mapped[bool] = mapped_column(Boolean, default=False)

    # Global Limits
    max_actions_per_game: Mapped[int] = mapped_column(Integer)
    max_chat_rounds: Mapped[int] = mapped_column(Integer)

    # Metadata
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_runtime_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Agent graph: prompts, descriptions, and allowed transitions per agent
    agents_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Git info at eval time (for reproducibility)
    git_commit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    git_branch: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Dataset split evaluated (e.g. "valid_seen", "valid_unseen")
    split: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Number of games selected for this run
    num_games: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Final outcome metrics (mirrors W&B logs)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_adjusted_success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    num_errors: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Token breakdown (total_tokens already exists)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Per-outcome averages (mirrors W&B logs)
    avg_actions_per_successful_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_chat_rounds_per_successful_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_runtime_per_successful_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_actions_per_failing_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_chat_rounds_per_failing_game: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_runtime_per_failing_game: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Relationship: One Experiment has Many Episodes
    episodes: Mapped[list["EpisodeRun"]] = relationship(
        "EpisodeRun", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ExperimentRun {self.id}: {self.agent_name} on {self.eval_env_type}>"


# 3. The individual ALFWorld Game iteration
class EpisodeRun(Base):
    __tablename__ = "episode_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_runs.id"), index=True
    )

    # Game Identifiers
    game_number: Mapped[int] = mapped_column(Integer)  # The 'i' in your script

    # Hard Metrics (Mapped directly from your wandb.log)
    success: Mapped[bool] = mapped_column(Boolean)
    actions_taken: Mapped[int] = mapped_column(Integer)
    chat_rounds: Mapped[int] = mapped_column(Integer)
    runtime_minutes: Mapped[float] = mapped_column(Float)

    # Agent Meta-Reasoning Data
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    transitions: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )  # Stores the from/to chat transitions
    belief_state: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )  # Stores episodic memory matches

    # Task description and type
    task: Mapped[str | None] = mapped_column(Text, nullable=True)
    task_type: Mapped[int | None] = mapped_column(Integer, nullable=True)  # ALFWorld types 1–6

    # Planning quality metrics
    inadmissible_action_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    concepts_learned: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Per-game token usage and cost (mirrors W&B game/*)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    episode_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Running success rates at time of game completion (mirrors W&B logs)
    success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_adjusted_success_rate: Mapped[float | None] = mapped_column(Float, nullable=True)

    # S3 Artifact Link (The Vault)
    chat_history_s3_key: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # e.g., "experiments/run_1/game_124_chat.txt"

    # Relationship back to the parent Experiment
    experiment: Mapped["ExperimentRun"] = relationship(
        "ExperimentRun", back_populates="episodes"
    )

    def __repr__(self) -> str:
        return f"<EpisodeRun Game #{self.game_number} | Success: {self.success}>"
