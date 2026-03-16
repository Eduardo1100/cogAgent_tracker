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
- [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py): primary evaluation runner. It logs experiment metadata, token usage, cost, S3 artifacts, and W&B data. It now also persists a Rich-rendered human-readable analyst trace for the current game into `experiment_runs.current_analyst_trace`, writes per-game `analyst_trace.txt`, writes a colored terminal companion `analyst_trace.ansi`, and saves the final trace on each `EpisodeRun`. Keep shared agent-facing percepts compact by pruning inactive task-contract fields and other empty runtime structure from model-facing JSON, but preserve the full execute-action observation/percept in the analyst trace alongside the compact agent-facing snapshot, plus the timestep-local belief-state and other agent outputs in a multiline hierarchical format with enough glossary/architecture context to interpret runtime terms without reading code. On `Ctrl+C`, it attempts to persist partial chat transcripts locally and save interrupted episode metadata to storage/DB before the experiment is marked `CANCELLED`.
- [scripts/get_latest_experiment.py](/home/eduardo/Projects/cogAgent_tracker/scripts/get_latest_experiment.py): local helper for resolving the latest experiment row and emitting a compact JSON summary from Postgres.
- [scripts/iterate_scienceworld.py](/home/eduardo/Projects/cogAgent_tracker/scripts/iterate_scienceworld.py): local iteration driver that runs `make debug ENV=<env>` (tales, scienceworld, or alfworld), resolves the resulting experiment, renders an agent-specific prompt, and optionally invokes the agent. Supports two agents: `claudecode` (default) which calls `claude --print` with ANTHROPIC_API_KEY stripped so it uses subscription auth, and `codex` which calls `codex exec --full-auto` pinned to `gpt-5.4` with `model_reasoning_effort="xhigh"`. Each agent has its own prompt templates under `prompts/` and its own branch-number counter (`cogfix-N` for claudecode, `agent-iter-N` for codex).
- [scripts/backfill_experiment_metrics.py](/home/eduardo/Projects/cogAgent_tracker/scripts/backfill_experiment_metrics.py): repair utility for historical metrics and git metadata.
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py): cognitive multi-agent runtime, including long-term memory path registration, task contracts, referent grounding, episode-local stage-aware progression plus lifecycle search-frontier control for ordered focus tasks, candidate-pivoting logic for search/placement tasks plus candidate last-seen-room / alias persistence plus post-goal candidate retirement / search-refresh reopening, state-change task control for substance search plus stalled-room frontier recovery and transformation, generic state-of-matter parsing for substance goals, measurement-and-branch task control with hidden-referent containment plus property-aware branch gating plus instrument-search frontier recovery, conditional trait-branch inference with evidence-target retention plus branch gating, artifact-creation task control for ingredient search plus type-consistent product grounding, inferred target-search plus relation-frontier control for local mechanism tasks, `action_executed` field in percept for BSA execution-confirmation grounding (prevents hallucinated completion claims when the environment rejects an action), dynamic task-contract injection from structured observations (extracts discovered targets and required families from recipe/instruction-format observations into the live task contract), extended `_DIRECT_EFFECT_RE` action-effect classifier covering cooking and manipulation verbs, exploration task detection via `_EXPLORATION_TASK_HINTS` (sets `exploration_task` flag in task contract, enables explicit search mode), and semantic entity scoring via `_compute_semantic_entity_score` (cosine similarity between action content embedding and target entity embeddings, cached per episode).
- [grafana/dashboards/cogagent_monitor.json](/home/eduardo/Projects/cogAgent_tracker/grafana/dashboards/cogagent_monitor.json): local Grafana dashboard for experiment metrics and health panels. Analyst traces are persisted to the DB and runs folder, but are no longer rendered directly in Grafana; inspect `analyst_trace.txt` or `analyst_trace.ansi` in the run directory instead.
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
  - For the local iteration loop, prefer `make iterate-tales`. It runs a random `make debug ENV=tales`, resolves the latest experiment for the current branch, renders the agent prompt, and by default invokes `claude --print` (claudecode). Use `AGENT=codex` to invoke Codex instead. Use `PROMPT_ONLY=1` to preview the rendered prompt without invoking the agent, or `SKIP_DEBUG=1` to reuse the latest local experiment.
  - For consolidation / ablation passes, prefer `make ablate-tales`. It reuses the latest local experiment by default (`--skip-debug`) so you can remove redundant heuristics or shrink prompts without a fresh run. Use `RUN_DEBUG=1` only when you intentionally want a new regression anchor first.
  - Environment-specific agent memory now lives under [src/agent/memory](/home/eduardo/Projects/cogAgent_tracker/src/agent/memory); preserve the `alfworld/` and `scienceworld/` split when changing memory logic or moving files.
  - ALFWorld bootstrap is handled by [scripts/bootstrap_alfworld.sh](/home/eduardo/Projects/cogAgent_tracker/scripts/bootstrap_alfworld.sh) and is expected by Docker-based flows.
  - For Docker-run evals, preserve the `exec env ... uv run python ...` pattern in [Makefile](/home/eduardo/Projects/cogAgent_tracker/Makefile) so `Ctrl+C` reaches Python instead of stopping in the shell wrapper.
  - For iterative agent changes, prefer Graphite stacks with one focused branch per iteration. Two naming conventions are in use:
    - `cogfix-NN-<topic>`: used by the Claude Code agent (`make iterate-tales AGENT=claudecode`). Targeted fixes and experiment-driven improvements.
    - `agent-iter-NN-<topic>`: used by the Codex agent (`make iterate-tales AGENT=codex`). Incremental feature iterations.
  - Keep only meaningful agent-behavior iterations in the stack. Fold doc-only, fixup, or cleanup-only changes into the nearest meaningful iteration or directly into `main`.
  - Consolidation / ablation work should stay in the same linear stack as the improvements it simplifies. For cogfix, prefer names like `cogfix-NN-consolidate-<topic>`.
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
    - `agent-iter-15-candidate-room-memory`
    - `agent-iter-16-state-change-target-grounding`
    - `agent-iter-17-state-change-room-frontier`
    - `agent-iter-18-measurement-instrument-search`
    - `agent-iter-19-artifact-room-frontier`
    - `agent-iter-20-room-navigation-frontier`
    - `agent-iter-21-growth-task-grounding`
    - `agent-iter-22-remote-room-entry`
    - `agent-iter-23-artifact-action-grounding`
    - `agent-iter-24-comparison-branch-resolution`
    - `agent-iter-25-comparison-search-frontier`
    - `agent-iter-26-relation-support-grounding`
    - `agent-iter-27-measurement-enclosure-activation`
    - `agent-iter-28-candidate-confirmation-pivot`
    - `agent-iter-29-conditional-transfer-branching`
    - `agent-iter-30-measurement-frontier-reopen`
    - `agent-iter-31-state-change-probe-confirmation`
    - `agent-iter-32-focus-first-candidate-grounding`
    - `agent-iter-33-measurement-heater-escalation`
    - `agent-iter-34-artifact-chemistry-guardrail`
    - `agent-iter-35-hybrid-growth-branching`
    - `agent-iter-36-consolidate-conditional-evidence`
    - `agent-iter-37-growth-frontier-recovery`
    - `agent-iter-38-measurement-proxy-guardrail`
    - `agent-iter-39-relation-commit-guardrail`
    - `agent-iter-40-consolidate-shared-task-contract`
    - `agent-iter-41-compact-model-facing-context`
  - `cogfix-<NN>-<topic>` branches are created by the Claude Code agent via `make iterate-tales` (default agent). They are targeted fixes tied to specific experiment post-mortems. The counter is tracked separately from `agent-iter-NN` and is computed automatically by `next_cogfix_number()` in `src/automation/iteration.py`.
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
- Keep Python compatible with `>=3.12,<3.13` (project requires Python 3.12; CI uses 3.12.10).

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
- Analyst traces are human-facing only. Keep them derived from runtime state and transcript artifacts; do not change agent prompts or message style just to make the trace prettier.
- Ordered lifecycle tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) rely on episode-local stage evidence, not just raw referents. When changing ordered focus behavior, keep progress semantics tied to stage-bearing evidence rather than container names or sibling object IDs.
- Search-and-place tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate primary candidates from destination/support entities. When changing candidate search behavior, avoid counting support entities like destination rooms or containers as extra primary focus targets.
- State-change tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate target substances from desired transformations. Keep procedural cues like `first ... then ...` distinct from ordered-target progression, prefer locating a grounded substance before testing transformation mechanisms, and preserve generic goals like `change the state of matter of water` as state-change tasks instead of falling back to generic exploration.
- Measurement-and-branch tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now separate instrument, measured target, and branch targets, and distinguish stable threshold properties like melting point from instantaneous temperature readings. Keep proxy readings distinct from direct target measurements, do not activate branch targets before grounded property-resolution evidence exists, and route hidden-target actions through the enclosing referent when the target is no longer visible.
- Artifact-creation tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now preserve artifact type separately from descriptor tokens like colors. When changing creation-task behavior, keep ingredient-gap search distinct from combine/transform steps, and do not let adjective-only distractors of the wrong type outrank grounded artifact actions.
- Artifact-creation grounding in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now retains artifact identity from successful action referents when ScienceWorld confirmations collapse to generic container names such as `You focus on the wood cup.` When changing this behavior, preserve the success-only / non-ambiguous guardrails and keep grounded labels normalized to artifact identity rather than transient room or container context.
- Prompt guidance in [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml) should stay abstract. Avoid benchmark-specific example objects like particular bulbs, boxes, or plants when a more general rule about object type, precursor setup, or evidence generation will do.
- Belief_State_Agent completion grounding in [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml) requires both `action_executed: true` AND a positive confirmation in `resulting_observation` before marking an action successful. The format rule ("BELIEF STATE:") is placed at the very top of the system message to prevent format drift. Do not move these constraints later in the prompt or weaken the two-condition grounding rule — it exists to prevent the hallucination pattern where BSA infers success from intent rather than from observation.
- Dynamic task-contract injection in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) (`_update_task_contract_from_recipe_observation`) mutates the cached `self._task_contract` dict in-place. This works because `_get_task_contract()` returns the same object for the duration of an episode (the task string never changes mid-episode). If you refactor the task-contract cache to copy-on-read, update this method to invalidate the source key or use a separate override field instead of in-place mutation.
- Artifact-creation search in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now reintroduces room-frontier pressure after repeated nonproductive local search before any base artifact is grounded. When changing this behavior, keep nearby exits and door-opening actions competitive before spending more actions on local containers or device toggles, and avoid hard-coding benchmark-specific room names or source locations.
- Relation-mechanism tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now infer target search from missing named targets, maintain an episode-local `relation_frontier`, and infer concrete support-source candidates for abstract support roles such as `renewable power source`. When changing relation-task behavior, keep abstract support grounding tied to observation-backed candidates rather than benchmark-specific name lists, and keep search broad only until the primary target/source are grounded before pruning relation and device-control reasoning to the local mechanism instead of the full combinatorial action space.
- Relation-mechanism shortlist control in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) now exposes `relation_frontier.commit_ready` plus exact commit candidates after a focused primary mechanism has already yielded component-level evidence. When changing this behavior, keep one grounded diagnostic step available before commitment, then suppress redundant sibling inspection on the same focused mechanism and push relation/device-control actions forward instead of spending extra turns on token-expensive reinspection.
- `_TASK_FAMILY_HINTS["relation"]` in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) was pruned to only `("connect", "link", "disconnect")` — ScienceWorld-specific terms like `"electrical circuit"`, `"wire"`, `"anode"`, `"cathode"` were removed. Similarly, `"electrically"`, `"powered"`, `"powering"`, `"powers"` were removed from `_TASK_STOPWORDS` and `_MEASUREMENT_TASK_STOPWORDS`. Do not re-add benchmark-specific terms to these hint lists.
- Exploration tasks in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) are detected by `_EXPLORATION_TASK_HINTS` and set `exploration_task=True` in the task contract plus `explicit_search_mode=True` and `inferred_search_mode=True`. Stall detection and novel-exit scoring for exploration tasks are handled by `_exploration_room_search_stalled` and `_update_exploration_search_tracking` (see cogfix-01 branch). When changing exploration task behavior, keep room-visit counts and tried-exit sets in `_search_location_states` rather than as separate per-episode fields.
- Semantic entity scoring in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py) (`_compute_semantic_entity_score`) uses cosine similarity between the action content embedding and target entity embeddings. Target embeddings are cached once per episode. The score is capped at 8 points and only fires for max_sim > 0.6. Do not lower the similarity threshold or raise the cap without running a multi-task eval — this signal supplements but must not override token-overlap scores.
- [scripts/iterate_scienceworld.py](/home/eduardo/Projects/cogAgent_tracker/scripts/iterate_scienceworld.py) is intentionally local-first. It expects a reachable Postgres instance; if the worktree is dirty it stops by default unless `--allow-dirty` / `ALLOW_DIRTY=1` is set. The `claudecode` agent (default) requires `claude` CLI authenticated via subscription — ANTHROPIC_API_KEY is intentionally stripped from the subprocess env so it does not accidentally use API credits. The `codex` agent requires a working `codex` CLI.
- Hard action/chat limits are still enforced in runtime, but explicit remaining-count numbers should stay out of model-facing context in [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py). If you change budget handling, prefer hidden runtime pressure over prompting the model with raw remaining counts.
- Full runtime bookkeeping belongs in analyst traces and persisted artifacts, not in model-facing percepts. When changing [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py), keep the agent-facing percept and focus-recovery payload compact and architecture-relevant so prompt size does not grow with every new controller.
- The README is sparse and the project metadata still uses the placeholder name `production-template`. Prefer the actual repo structure over marketing text.
- Pytest CI runs via `uv run pytest tests/`. Keep [tests/conftest.py](/home/eduardo/Projects/cogAgent_tracker/tests/conftest.py) in mind for repo-root import resolution, and make test stubs override `sys.modules` directly instead of relying on import order.
- [.pre-commit-config.yaml](/home/eduardo/Projects/cogAgent_tracker/.pre-commit-config.yaml) is intentionally tracked. Do not re-add it to `.gitignore`.

## When An Agent Should Pause
- Pause and ask before destructive cleanup, volume deletion, or broad git operations.
- Pause if a task requires new secrets, external accounts, or live provider billing.
- Pause if a schema change would require a migration strategy beyond a straightforward Alembic revision.
