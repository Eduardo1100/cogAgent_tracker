"""Add architecture metrics fields for phase-0 runtime instrumentation.

Revision ID: 20260318_000003
Revises: 20260312_000002
Create Date: 2026-03-18 00:00:03
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import inspect

from alembic import op

revision = "20260318_000003"
down_revision = "20260312_000002"
branch_labels = None
depends_on = None


def _has_column(inspector, table_name: str, column_name: str) -> bool:
    return column_name in {
        column["name"] for column in inspector.get_columns(table_name)
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if not _has_column(inspector, "experiment_runs", "architecture_metrics"):
        op.add_column(
            "experiment_runs",
            sa.Column("architecture_metrics", sa.JSON(), nullable=True),
        )

    if not _has_column(inspector, "episode_runs", "architecture_metrics"):
        op.add_column(
            "episode_runs",
            sa.Column("architecture_metrics", sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    if _has_column(inspector, "episode_runs", "architecture_metrics"):
        op.drop_column("episode_runs", "architecture_metrics")

    if _has_column(inspector, "experiment_runs", "architecture_metrics"):
        op.drop_column("experiment_runs", "architecture_metrics")
