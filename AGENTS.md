# AGENTS.md

## Purpose
- This repo is a Python 3.11 `uv` project for running cognitive-agent evaluations and exposing experiment data through a FastAPI service.
- The stack is local-first but container-oriented: Postgres, Redis, MinIO, and Grafana run via Docker Compose; the app and agent runner depend on those services.
- Treat this as an experiments + observability repo, not a generic CRUD API. Changes often affect runtime jobs, persisted metrics, and schema compatibility.

## Quick Start
- Read [README.md](/home/eduardo/Projects/cogAgent_tracker/README.md), [Makefile](/home/eduardo/Projects/cogAgent_tracker/Makefile), and [pyproject.toml](/home/eduardo/Projects/cogAgent_tracker/pyproject.toml) first.
- Preferred setup:
  - `cp .env.example .env`
  - `mise install`
  - `uv sync --frozen`
- Preferred service startup:
  - `make up` to start Postgres, Redis, MinIO, Grafana, and the app container.
  - `make dev` for local Marimo + FastAPI development.
- Preferred quality checks:
  - `make test`
  - `make lint`
  - `make db-current`

## Repo Map
- [src/app.py](/home/eduardo/Projects/cogAgent_tracker/src/app.py): FastAPI app construction. Startup hard-fails if required env vars are missing or Alembic is behind.
- [src/main.py](/home/eduardo/Projects/cogAgent_tracker/src/main.py): Uvicorn import target.
- [src/api/v1/endpoints.py](/home/eduardo/Projects/cogAgent_tracker/src/api/v1/endpoints.py): experiment/episode endpoints plus a stack smoke-test write path.
- [src/api/v1/health.py](/home/eduardo/Projects/cogAgent_tracker/src/api/v1/health.py): health checks for AI runtime, DB, schema, S3/MinIO, and Redis.
- [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py): SQLAlchemy models. Schema changes here usually require an Alembic revision.
- [src/storage/database.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/database.py): engine/session factory. Normalizes `DATABASE_URL`.
- [src/storage/cache.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/cache.py): Redis helpers.
- [src/storage/s3.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/s3.py): MinIO/S3 client and upload helper.
- [src/config/schema_health.py](/home/eduardo/Projects/cogAgent_tracker/src/config/schema_health.py): Alembic head/current revision enforcement.
- [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py): primary evaluation runner. It logs experiment metadata, token usage, cost, S3 artifacts, and W&B data.
- [scripts/backfill_experiment_metrics.py](/home/eduardo/Projects/cogAgent_tracker/scripts/backfill_experiment_metrics.py): repair utility for historical metrics and git metadata.
- [src/agent/configs/ALFworld.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/ALFworld.yaml), [src/agent/configs/scienceworld.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/scienceworld.yaml), [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml): agent/runtime configuration.
- [alembic/versions/20260311_000001_initial_schema.py](/home/eduardo/Projects/cogAgent_tracker/alembic/versions/20260311_000001_initial_schema.py): current schema baseline.
- [notebooks/exploration.py](/home/eduardo/Projects/cogAgent_tracker/notebooks/exploration.py): Marimo notebook entrypoint.

## Canonical Workflows
- API work:
  - Edit files under `src/api`, `src/app.py`, `src/config`, and `src/storage`.
  - If startup fails, check env vars first, then Alembic revision state.
- Schema work:
  - Update [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py).
  - Generate migration with `make db-revision MESSAGE="..."`.
  - Apply with `make db-upgrade`.
  - Keep schema health checks in sync if startup assumptions changed.
- Agent/eval work:
  - Main runner is [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py).
  - Default evaluation commands are `make train`, `make eval`, and `make debug`.
  - ALFWorld bootstrap is handled by [scripts/bootstrap_alfworld.sh](/home/eduardo/Projects/cogAgent_tracker/scripts/bootstrap_alfworld.sh) and is expected by Docker-based flows.
  - For Docker-run evals, preserve the `exec env ... uv run python ...` pattern in [Makefile](/home/eduardo/Projects/cogAgent_tracker/Makefile) so `Ctrl+C` reaches Python instead of stopping in the shell wrapper.
- Infra connectivity work:
  - Use [tests/test_connections.py](/home/eduardo/Projects/cogAgent_tracker/tests/test_connections.py) as a simple end-to-end dependency check.
  - Health endpoints also exercise DB/Redis/MinIO behavior.

## Environment And Secrets
- Copy [.env.example](/home/eduardo/Projects/cogAgent_tracker/.env.example) to `.env`; do not commit real secrets.
- Important env vars:
  - `DATABASE_URL`, `REDIS_URL`, `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`
  - provider keys such as `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `ANTHROPIC_API_KEY`
  - `ACTIVE_LLM_PROFILE`, `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_MODE`
  - `SKIP_SCHEMA_REVISION_CHECK=1` only as a temporary bypass for startup; do not normalize around it.
- Local app URLs in `.env.example` use `localhost`; containerized services use Compose hostnames like `db`, `cache`, and `objectstore`.

## Editing Rules For Agents
- Prefer minimal, targeted changes. This repo has real startup checks and persistence side effects.
- Do not remove schema validation or env validation just to make tests pass.
- If you change SQLAlchemy models, inspect Alembic impact immediately.
- Keep Docker and local env assumptions aligned. A change that works only for `localhost` or only for Compose hostnames is incomplete.
- Preserve `uv`/`mise` workflows; do not introduce ad hoc package-management paths unless explicitly requested.
- Keep Python compatible with `>=3.11,<3.12` even though Ruff targets `py313`.

## Verification Strategy
- Small Python changes:
  - `uv run pytest tests/`
  - `uvx ruff check .`
  - `uvx ruff format . --check`
  - If local `uv`/`uvx` is sandbox-blocked, use the installed `pytest` / `ruff` binaries to reproduce failures, but keep final verification aligned with the `uv` commands above when possible.
- API or storage changes:
  - verify `make up` stack health
  - hit `/health`, `/health/db`, `/health/storage`, `/health/cache`
- Schema changes:
  - `make db-upgrade`
  - `make db-current`
- Agent runner changes:
  - prefer `make debug`
  - use `WANDB_MODE=offline` when possible for local validation

## Known Gotchas
- [src/app.py](/home/eduardo/Projects/cogAgent_tracker/src/app.py) validates schema revision at import time, so stale DB state can break seemingly unrelated API work.
- [src/config/__init__.py](/home/eduardo/Projects/cogAgent_tracker/src/config/__init__.py) must stay lazy. Do not instantiate `Settings()` at module import time; tests and utility scripts may import `src.config` without app env vars present.
- [src/storage/database.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/database.py) force-rewrites `postgresql://` URLs to `postgresql+psycopg2://` unless `psycopg2` is already present.
- `make clean` and `make nuke` are destructive. Avoid them unless the user explicitly wants caches, volumes, or environments wiped.
- Docker startup runs `uv sync --frozen` and `scripts/bootstrap_alfworld.sh`; container boots may be slow by design.
- The README is sparse and the project metadata still uses the placeholder name `production-template`. Prefer the actual repo structure over marketing text.
- Pytest CI runs via `uv run pytest tests/`. Keep [tests/conftest.py](/home/eduardo/Projects/cogAgent_tracker/tests/conftest.py) in mind for repo-root import resolution, and make test stubs override `sys.modules` directly instead of relying on import order.
- [.pre-commit-config.yaml](/home/eduardo/Projects/cogAgent_tracker/.pre-commit-config.yaml) is intentionally tracked. Do not re-add it to `.gitignore`.

## When An Agent Should Pause
- Pause and ask before destructive cleanup, volume deletion, or broad git operations.
- Pause if a task requires new secrets, external accounts, or live provider billing.
- Pause if a schema change would require a migration strategy beyond a straightforward Alembic revision.
