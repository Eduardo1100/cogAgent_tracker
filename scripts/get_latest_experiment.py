from __future__ import annotations
# ruff: noqa: E402, I001

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError

from src.automation.iteration import get_latest_experiment, summarize_experiment
from src.storage.database import DATABASE_URL, SessionLocal

RUNS_ROOT = REPO_ROOT / "runs"


def get_current_branch() -> str | None:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    return branch or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Return the latest experiment summary from Postgres."
    )
    parser.add_argument(
        "--env",
        default="scienceworld",
        help="Environment type stored in experiment_runs.eval_env_type.",
    )
    parser.add_argument(
        "--branch",
        default="current",
        help='Git branch filter. Use "current" to resolve the checked-out branch, or "any" to disable filtering.',
    )
    parser.add_argument(
        "--after-id",
        type=int,
        default=None,
        help="Only consider experiments with id strictly greater than this value.",
    )
    parser.add_argument(
        "--field",
        choices={"id", "json"},
        default="json",
        help="Select plain id output or a JSON summary.",
    )
    return parser


def describe_database_target(database_url: str) -> str:
    parsed = make_url(database_url)
    host = parsed.host or "localhost"
    port = parsed.port or 5432
    database = parsed.database or "<unknown-db>"
    return f"{host}:{port}/{database}"


def build_db_unavailable_message(database_url: str) -> str:
    return (
        "Could not reach Postgres at "
        f"{describe_database_target(database_url)}. "
        "This helper loads DATABASE_URL from `.env` inside Python, so the shell "
        "environment variable may still appear empty even when the script is configured."
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    git_branch = args.branch
    if git_branch == "current":
        git_branch = get_current_branch()
    elif git_branch == "any":
        git_branch = None

    db = SessionLocal()
    try:
        experiment = get_latest_experiment(
            db,
            env_type=args.env,
            git_branch=git_branch,
            after_id=args.after_id,
        )
        if experiment is None:
            return 1

        summary = summarize_experiment(db, experiment, runs_root=RUNS_ROOT)
        if args.field == "id":
            print(summary.experiment_id)
        else:
            print(summary.to_json())
        return 0
    except SQLAlchemyError:
        parser.exit(2, f"{build_db_unavailable_message(DATABASE_URL)}\n")
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
