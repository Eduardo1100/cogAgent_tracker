# CogAgentLab

[![CI](https://github.com/Eduardo1100/cogAgent_tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/Eduardo1100/cogAgent_tracker/actions/workflows/ci.yml)

`cogAgent_tracker` is a Python 3.11 project for running cognitive-agent evaluations and tracking experiment results through a FastAPI API backed by Postgres, Redis, and MinIO.

## What Is Here

- FastAPI service for experiment and episode data in `src/app.py`
- SQLAlchemy models and Alembic migrations for persistent run metadata
- Agent runner scripts for ALFWorld, ScienceWorld, and TextWorld (Tales) evaluations
- Docker Compose stack with Postgres, Redis, MinIO, Grafana, and the app container
- Marimo notebook for local exploration in `notebooks/exploration.py`

## Tooling

- Python: 3.11.15
- Package/runtime management: `uv`
- Tool version pinning: `mise`
- Lint/format: `ruff`
- Migrations: `alembic`
- Containers: Docker Compose

## Local Setup

1. Copy env defaults:

```bash
cp .env.example .env
```

2. Install toolchain and dependencies:

```bash
mise install
uv sync --frozen
```

3. Start the local stack:

```bash
make up
```

This starts:

- FastAPI app on `http://localhost:8000`
- Postgres on `localhost:5432`
- MinIO API on `http://localhost:9000`
- MinIO console on `http://localhost:9001`
- Grafana on `http://localhost:3000`

## Common Commands

```bash
make help
make dev
make test
make lint
make db-upgrade
make db-current
make debug
make eval
```

Notes:

- `make dev` starts Marimo plus the local FastAPI dev server.
- `make debug`, `make train`, and `make eval` run inside Docker and bootstrap ALFWorld as needed.
- API startup enforces required env vars and checks that the database schema matches Alembic head.

## Database Migrations

Schema changes should update both the SQLAlchemy models and Alembic history.

```bash
make db-revision MESSAGE="describe change"
make db-upgrade
make db-current
```

If startup fails with a schema revision error, run `make db-upgrade` unless you are intentionally bypassing the check for debugging.

## Health And Verification

Useful checks:

```bash
make test
make lint
uv run pytest tests/
uvx ruff check .
uvx ruff format . --check
```

Once the stack is running, the API exposes health endpoints:

- `GET /health`
- `GET /health/db`
- `GET /health/schema`
- `GET /health/storage`
- `GET /health/cache`

## CI

GitHub Actions runs `.github/workflows/ci.yml` on pushes to `main`, pull requests, and manual dispatch.

The workflow currently does four things:

- installs Python 3.11 and syncs the locked `uv` environment
- runs Ruff lint and format checks
- runs `pytest`
- builds the Docker image on `main` pushes to catch container regressions

## Environment Variables

Start from `.env.example`. The most important values are:

- `DATABASE_URL`
- `REDIS_URL`
- `S3_ENDPOINT`
- `S3_ACCESS_KEY`
- `S3_SECRET_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `DEEPSEEK_API_KEY`
- `ANTHROPIC_API_KEY`
- `ACTIVE_LLM_PROFILE`
- `HF_TOKEN`
- `WANDB_API_KEY`

Do not commit a populated `.env`.
