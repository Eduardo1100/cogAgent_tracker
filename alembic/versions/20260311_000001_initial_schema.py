"""Initial schema baseline.

Revision ID: 20260311_000001
Revises:
Create Date: 2026-03-11 00:00:01
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql


revision = "20260311_000001"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return column_name in {column["name"] for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not _has_table(inspector, "predictions"):
        op.create_table(
            "predictions",
            sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
            sa.Column("filename", sa.String(), nullable=True),
            sa.Column("result", sa.String(), nullable=True),
            sa.Column("confidence", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        )
        op.create_index("ix_predictions_id", "predictions", ["id"], unique=False)

    if not _has_table(inspector, "experiment_runs"):
        op.create_table(
            "experiment_runs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("agent_name", sa.String(length=100), nullable=False),
            sa.Column("llm_model", sa.String(length=50), nullable=False),
            sa.Column("eval_env_type", sa.String(length=100), nullable=False),
            sa.Column("max_actions_per_game", sa.Integer(), nullable=False),
            sa.Column("max_chat_rounds", sa.Integer(), nullable=False),
            sa.Column("start_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
            sa.Column("total_runtime_minutes", sa.Float(), nullable=True),
            sa.Column("total_tokens", sa.Integer(), nullable=True),
            sa.Column("total_cost", sa.Float(), nullable=True),
            sa.Column("agents_config", sa.JSON(), nullable=True),
            sa.Column("git_commit", sa.String(length=40), nullable=True),
            sa.Column("git_branch", sa.String(length=255), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=True),
            sa.Column("selected_games", sa.JSON(), nullable=True),
            sa.Column("selected_games_display", sa.Text(), nullable=True),
            sa.Column("current_game_number", sa.Integer(), nullable=True),
            sa.Column("current_game_label", sa.String(length=255), nullable=True),
            sa.Column("split", sa.String(length=50), nullable=True),
            sa.Column("num_games", sa.Integer(), nullable=True),
            sa.Column("success_rate", sa.Float(), nullable=True),
            sa.Column("error_adjusted_success_rate", sa.Float(), nullable=True),
            sa.Column("num_errors", sa.Integer(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("avg_actions_per_successful_game", sa.Float(), nullable=True),
            sa.Column("avg_chat_rounds_per_successful_game", sa.Float(), nullable=True),
            sa.Column("avg_runtime_per_successful_game", sa.Float(), nullable=True),
            sa.Column("avg_actions_per_failing_game", sa.Float(), nullable=True),
            sa.Column("avg_chat_rounds_per_failing_game", sa.Float(), nullable=True),
            sa.Column("avg_runtime_per_failing_game", sa.Float(), nullable=True),
        )
        op.create_index(
            "ix_experiment_runs_agent_name",
            "experiment_runs",
            ["agent_name"],
            unique=False,
        )
    else:
        experiment_columns: list[tuple[str, sa.types.TypeEngine, bool]] = [
            ("status", sa.String(length=20), True),
            ("selected_games", postgresql.JSON(astext_type=sa.Text()), True),
            ("selected_games_display", sa.Text(), True),
            ("current_game_number", sa.Integer(), True),
            ("current_game_label", sa.String(length=255), True),
        ]
        for column_name, column_type, nullable in experiment_columns:
            if not _has_column(inspector, "experiment_runs", column_name):
                op.add_column(
                    "experiment_runs",
                    sa.Column(column_name, column_type, nullable=nullable),
                )

    if not _has_table(inspector, "episode_runs"):
        op.create_table(
            "episode_runs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("experiment_id", sa.Integer(), nullable=False),
            sa.Column("game_number", sa.Integer(), nullable=False),
            sa.Column("success", sa.Boolean(), nullable=False),
            sa.Column("actions_taken", sa.Integer(), nullable=False),
            sa.Column("chat_rounds", sa.Integer(), nullable=False),
            sa.Column("runtime_minutes", sa.Float(), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("transitions", sa.JSON(), nullable=True),
            sa.Column("belief_state", sa.JSON(), nullable=True),
            sa.Column("task", sa.Text(), nullable=True),
            sa.Column("task_type", sa.Integer(), nullable=True),
            sa.Column("inadmissible_action_count", sa.Integer(), nullable=True),
            sa.Column("concepts_learned", sa.JSON(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("episode_cost", sa.Float(), nullable=True),
            sa.Column("success_rate", sa.Float(), nullable=True),
            sa.Column("error_adjusted_success_rate", sa.Float(), nullable=True),
            sa.Column("chat_history_s3_key", sa.String(length=255), nullable=True),
            sa.ForeignKeyConstraint(["experiment_id"], ["experiment_runs.id"]),
        )
        op.create_index(
            "ix_episode_runs_experiment_id",
            "episode_runs",
            ["experiment_id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _has_table(inspector, "episode_runs"):
        index_names = {index["name"] for index in inspector.get_indexes("episode_runs")}
        if "ix_episode_runs_experiment_id" in index_names:
            op.drop_index("ix_episode_runs_experiment_id", table_name="episode_runs")
        op.drop_table("episode_runs")

    if _has_table(inspector, "experiment_runs"):
        existing_columns = {column["name"] for column in inspector.get_columns("experiment_runs")}
        removable_columns = [
            "current_game_label",
            "current_game_number",
            "selected_games_display",
            "selected_games",
            "status",
        ]
        for column_name in removable_columns:
            if column_name in existing_columns:
                op.drop_column("experiment_runs", column_name)

    if _has_table(inspector, "predictions"):
        index_names = {index["name"] for index in inspector.get_indexes("predictions")}
        if "ix_predictions_id" in index_names:
            op.drop_index("ix_predictions_id", table_name="predictions")
        op.drop_table("predictions")
