import os
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text

from src.storage.database import engine

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


def get_schema_revision_status() -> dict[str, object]:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    script = ScriptDirectory.from_config(config)
    head_revision = script.get_current_head()

    with engine.connect() as connection:
        has_version_table = connection.execute(
            text("SELECT to_regclass('public.alembic_version')")
        ).scalar()
        if has_version_table is None:
            return {
                "schema_ok": False,
                "current_revision": None,
                "expected_revision": head_revision,
                "versioned": False,
                "skipped": False,
            }

        current_revision = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar()

    return {
        "schema_ok": current_revision == head_revision,
        "current_revision": current_revision,
        "expected_revision": head_revision,
        "versioned": True,
        "skipped": False,
    }


def require_current_schema(*, context: str) -> None:
    if os.getenv("SKIP_SCHEMA_REVISION_CHECK") == "1":
        return

    status = get_schema_revision_status()
    if not status["versioned"]:
        raise RuntimeError(
            f"Database schema is not versioned for {context}. "
            "Run `make db-upgrade` before starting."
        )

    current_revision = status["current_revision"]
    head_revision = status["expected_revision"]
    if current_revision != head_revision:
        raise RuntimeError(
            f"Database schema is out of date for {context}: "
            f"current={current_revision}, expected={head_revision}. "
            "Run `make db-upgrade` before starting."
        )
