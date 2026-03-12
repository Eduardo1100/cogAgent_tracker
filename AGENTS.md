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
- [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py): primary evaluation runner. It logs experiment metadata, token usage, cost, S3 artifacts, and W&B data. On `Ctrl+C`, it now attempts to persist partial chat transcripts locally and save interrupted episode metadata to storage/DB before the experiment is marked `CANCELLED`.
- [scripts/get_latest_experiment.py](/home/eduardo/Projects/cogAgent_tracker/scripts/get_latest_experiment.py): local helper for resolving the latest experiment row and emitting a compact JSON summary from Postgres.
- [scripts/iterate_scienceworld.py](/home/eduardo/Projects/cogAgent_tracker/scripts/iterate_scienceworld.py): local iteration driver that can run `make debug ENV=scienceworld`, resolve the resulting experiment, render a Codex prompt, and optionally invoke `codex exec` to continue the agent-improvement loop. It pins Codex to `gpt-5.4` with `model_reasoning_effort="xhigh"`.
- [scripts/backfill_experiment_metrics.py](/home/eduardo/Projects/cogAgent_tracker/scripts/backfill_experiment_metrics.py): repair utility for historical metrics and git metadata.
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py): cognitive multi-agent runtime, including long-term memory path registration, task contracts, referent grounding, episode-local stage-aware progression plus lifecycle search-frontier control for ordered focus tasks, candidate-pivoting logic for search/placement tasks, state-change task control for substance search plus transformation, generic state-of-matter parsing for substance goals, measurement-and-branch task control with hidden-referent containment plus property-aware branch gating, conditional trait-branch inference with evidence-target retention plus branch gating, artifact-creation task control for ingredient search plus type-consistent product grounding, and inferred target-search plus relation-frontier control for local mechanism tasks.
- [src/agent/memory](/home/eduardo/Projects/cogAgent_tracker/src/agent/memory): tracked memory store split by environment (`alfworld/`, `scienceworld/`) with `memory1.txt` and `memory2.txt` per environment.
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
  - For the local ScienceWorld iteration loop, prefer `make iterate-scienceworld`. It runs a random `make debug ENV=scienceworld`, resolves the latest experiment for the current branch, renders the Codex iteration prompt, and by default invokes `codex exec --full-auto --model gpt-5.4 --config model_reasoning_effort="xhigh"`. Use `PROMPT_ONLY=1` to preview the prompt or `SKIP_DEBUG=1` to reuse the latest local experiment.
  - For consolidation / ablation passes, prefer `make ablate-scienceworld`. It reuses the latest local ScienceWorld experiment by default (`--skip-debug`) so you can remove redundant heuristics or shrink prompts without paying for a fresh random run. Use `RUN_DEBUG=1` only when you intentionally want a new regression anchor first.
  - Environment-specific agent memory now lives under [src/agent/memory](/home/eduardo/Projects/cogAgent_tracker/src/agent/memory); preserve the `alfworld/` and `scienceworld/` split when changing memory logic or moving files.
  - ALFWorld bootstrap is handled by [scripts/bootstrap_alfworld.sh](/home/eduardo/Projects/cogAgent_tracker/scripts/bootstrap_alfworld.sh) and is expected by Docker-based flows.
  - For Docker-run evals, preserve the `exec env ... uv run python ...` pattern in [Makefile](/home/eduardo/Projects/cogAgent_tracker/Makefile) so `Ctrl+C` reaches Python instead of stopping in the shell wrapper.
  - For iterative agent changes, prefer Graphite stacks with one focused branch per iteration. Current naming convention: `agent-iter-XX-<topic>`.
  - Keep only meaningful agent-behavior iterations in the stack. Fold doc-only, fixup, or cleanup-only changes into the nearest meaningful iteration or directly into `main`.
  - Consolidation / ablation work should still stay in the same linear `agent-iter-XX-<topic>` stack. Prefer names like `agent-iter-36-consolidate-shortlist` so rollback order stays obvious.
  - Use `main` for merged baseline and cross-cutting fixes such as persistence, observability, docs, or cleanup that should not stay as agent-iteration branches.
  - Current preferred review order for the agent stack:
    - `agent-iter-01-runtime-reasoning`
    - `agent-iter-02-task-grounding`
    - `agent-iter-03-referent-grounding`
    - `agent-iter-04-stage-aware-grounding`
    - `agent-iter-05-candidate-pivoting`
    - `agent-iter-06-state-change-contract`
    - `agent-iter-07-substance-grounding`
    - `agent-iter-08-artifact-creation-grounding`
    - `agent-iter-09-relation-frontier-grounding`
    - `agent-iter-10-generic-state-change-parsing`
    - `agent-iter-11-property-aware-measurement`
    - `agent-iter-12-trait-branch-inference`
    - `agent-iter-13-lifecycle-search-frontier`
    - `agent-iter-14-invalid-referent-guardrails`
- Infra connectivity work:
  - Use [tests/test_connections.py](/home/eduardo/Projects/cogAgent_tracker/tests/test_connections.py) as a simple end-to-end dependency check.
  - Health endpoints also exercise DB/Redis/MinIO behavior.

## ScienceWorld Policy
- Use ScienceWorld as a debugging surface, transcript source, and regression suite for the cognitive agent, not as the definition of intelligence or the sole optimization target.
- Treat ScienceWorld success rate as a development signal only. Improvements must also preserve token efficiency, task generalization, and environment agnosticism.
- Be cautious of benchmark adaptation. ScienceWorld rewards parser-specific phrasing, repeated ontologies, and environment-local task conventions that may not transfer to arbitrary environments.
- Do not turn ScienceWorld-specific regularities into persistent agent knowledge or hand-built policies unless they can be defended as environment-agnostic cognitive structure.
- Prefer runtime abstractions that generalize across environments:
  - task contracts
  - phase control
  - grounding / referent management
  - hypothesis retirement
  - local mechanism tracking
- Discount apparent gains that come mostly from:
  - exact-command benchmark gaming
  - memorizing recurrent task templates
  - overfitting to objects, rooms, or affordance quirks unique to ScienceWorld
- When evaluating agent changes, look at success rate together with:
  - invalid action rate
  - actions to success
  - chat rounds
  - tokens per successful episode
  - Prefer random-task evaluation for the main signal. Use hand-picked tasks mainly to reproduce a known failure mode and verify a targeted fix.
  - Every few agent iterations, prefer one explicit consolidation pass that removes overlapping heuristics, merges duplicate frontier logic, or trims model-facing context. Keep those consolidation passes tied to the same random-run workflow and evaluate them against the latest local experiment before keeping them.

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
  - If you touch the local iteration helpers under [src/automation](/home/eduardo/Projects/cogAgent_tracker/src/automation), [scripts/get_latest_experiment.py](/home/eduardo/Projects/cogAgent_tracker/scripts/get_latest_experiment.py), or [scripts/iterate_scienceworld.py](/home/eduardo/Projects/cogAgent_tracker/scripts/iterate_scienceworld.py), run `uv run pytest tests/test_iteration_workflow.py`.
- API or storage changes:
  - verify `make up` stack health
  - hit `/health`, `/health/db`, `/health/storage`, `/health/cache`
- Schema changes:
  - `make db-upgrade`
  - `make db-current`
- Agent runner changes:
  - prefer `make debug`
  - use `WANDB_MODE=offline` when possible for local validation
  - if you touch signal handling, interrupted runs, or per-episode persistence, run `uv run pytest tests/test_run_agent_status.py`
  - if you touch `src/agent/gwt_agent.py` runtime scoring, task grounding, referent resolution, stage progression, candidate pivoting, state-change task control, measurement/branch task control, artifact-creation task control, or relation-frontier / inferred-search logic, run `uv run pytest tests/test_gwt_agent_memory_paths.py`

## Known Gotchas
- [src/app.py](/home/eduardo/Projects/cogAgent_tracker/src/app.py) validates schema revision at import time, so stale DB state can break seemingly unrelated API work.
- [src/config/__init__.py](/home/eduardo/Projects/cogAgent_tracker/src/config/__init__.py) must stay lazy. Do not instantiate `Settings()` at module import time; tests and utility scripts may import `src.config` without app env vars present.
- [src/storage/database.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/database.py) force-rewrites `postgresql://` URLs to `postgresql+psycopg2://` unless `psycopg2` is already present.
- `make clean` and `make nuke` are destructive. Avoid them unless the user explicitly wants caches, volumes, or environments wiped.
- Docker startup runs `uv sync --frozen` and `scripts/bootstrap_alfworld.sh`; container boots may be slow by design.
- `Ctrl+C` now tries to preserve partial episode traces: `chat_history.txt`, `transition_log.json`, S3 chat upload, and a partial `EpisodeRun` row when the interrupt path has enough context. Hard kills such as `kill -9` still bypass this.
- Ordered lifecycle tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) rely on episode-local stage evidence, not just raw referents. When changing ordered focus behavior, keep progress semantics tied to stage-bearing evidence rather than container names or sibling object IDs.
- Lifecycle search in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now uses a room-frontier bias before any stage evidence is grounded. When changing this behavior, keep search broad across reachable rooms and exits before drilling into local containers, and avoid hard-coding ScienceWorld room names or benchmark-specific room priorities.
- Search-and-place tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate primary candidates from destination/support entities. When changing candidate search behavior, avoid counting support entities like destination rooms or containers as extra primary focus targets.
- State-change tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate target substances from desired transformations. Keep procedural cues like `first ... then ...` distinct from ordered-target progression, prefer locating a grounded substance before testing transformation mechanisms, and preserve generic goals like `change the state of matter of water` as state-change tasks instead of falling back to generic exploration.
- State-change probing in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now treats pseudo-admissible observations such as `No known action matches that input.` as invalid referent evidence. When changing probe-source behavior, preserve the episode-local penalty against retrying the same family on the same failed referent unless new grounded evidence makes that referent newly relevant.
- Measurement-and-branch tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate instrument, measured target, and branch targets, and distinguish stable threshold properties like melting point from instantaneous temperature readings. Keep proxy readings distinct from direct target measurements, do not activate branch targets before grounded property-resolution evidence exists, and route hidden-target actions through the enclosing referent when the target is no longer visible.
- Conditional trait-branch tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now retain an evidence target separately from final branch boxes. When changing this behavior, keep unresolved branch targets out of early shortlist pressure, preserve evidence-gathering on the trait-bearing referent, and only promote the final branch target after observation-level branch resolution.
- Artifact-creation tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now preserve artifact type separately from descriptor tokens like colors. When changing creation-task behavior, keep ingredient-gap search distinct from combine/transform steps, and do not let adjective-only distractors of the wrong type outrank grounded artifact actions.
- Relation-mechanism tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now infer target search from missing named targets and maintain an episode-local `relation_frontier`. When changing relation-task behavior, keep search broad only until the primary target/source are grounded, then prune relation and device-control reasoning to the local mechanism instead of the full combinatorial action space.
- [scripts/iterate_scienceworld.py](/home/eduardo/Projects/cogAgent_tracker/scripts/iterate_scienceworld.py) is intentionally local-first. It expects a reachable Postgres instance and a working `codex` CLI; if the worktree is dirty it stops by default unless `--allow-dirty` / `ALLOW_DIRTY=1` is set.
- The README is sparse and the project metadata still uses the placeholder name `production-template`. Prefer the actual repo structure over marketing text.
- Pytest CI runs via `uv run pytest tests/`. Keep [tests/conftest.py](/home/eduardo/Projects/cogAgent_tracker/tests/conftest.py) in mind for repo-root import resolution, and make test stubs override `sys.modules` directly instead of relying on import order.
- [.pre-commit-config.yaml](/home/eduardo/Projects/cogAgent_tracker/.pre-commit-config.yaml) is intentionally tracked. Do not re-add it to `.gitignore`.

## When An Agent Should Pause
- Pause and ask before destructive cleanup, volume deletion, or broad git operations.
- Pause if a task requires new secrets, external accounts, or live provider billing.
- Pause if a schema change would require a migration strategy beyond a straightforward Alembic revision.
