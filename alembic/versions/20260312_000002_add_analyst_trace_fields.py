"""Add analyst trace fields for live and persisted run inspection.

Revision ID: 20260312_000002
Revises: 20260311_000001
Create Date: 2026-03-12 00:00:02
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "20260312_000002"
down_revision = "20260311_000001"
branch_labels = None
depends_on = None


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return column_name in {
        column["name"] for column in inspector.get_columns(table_name)
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not _has_column(inspector, "experiment_runs", "current_analyst_trace"):
        op.add_column(
            "experiment_runs",
            sa.Column("current_analyst_trace", sa.Text(), nullable=True),
        )

    if not _has_column(inspector, "episode_runs", "analyst_trace"):
        op.add_column(
            "episode_runs",
            sa.Column("analyst_trace", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _has_column(inspector, "episode_runs", "analyst_trace"):
        op.drop_column("episode_runs", "analyst_trace")

    if _has_column(inspector, "experiment_runs", "current_analyst_trace"):
        op.drop_column("experiment_runs", "current_analyst_trace")
