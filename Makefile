# --- Variables ---
PYTHON := uv run python
SHELL  := /bin/zsh

.PHONY: help setup dev train eval debug iterate-agent ablate-agent test lint clean build-docker benchmark up down nuke sanity bootstrap-alfworld db-upgrade db-revision db-current

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:
	@echo "🚀 Setting up environment..."
	mise install
	uv sync --frozen
	@echo "✅ Setup complete. Environment is ready."

dev: ## Start the development environment (notebooks + local API)
	@echo "🛠️ Starting dev services..."
	uv run marimo edit notebooks/exploration.py &
	uv run fastapi dev src/main.py

train: ## Run eval on valid_seen split. GAMES=N to limit (default: all)
	docker compose run --rm app \
	sh -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_alfworld.sh; exec env PYTHONPATH=/app uv run python scripts/run_agent.py src/agent/configs/eval_config.yaml --gwt --splits valid_seen $(if $(GAMES),--num_games $(GAMES),)'

eval: ## Run eval on valid_unseen split. GAMES=N to limit (default: all)
	docker compose run --rm app \
	sh -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_alfworld.sh; exec env PYTHONPATH=/app uv run python scripts/run_agent.py src/agent/configs/eval_config.yaml --gwt --splits valid_unseen $(if $(GAMES),--num_games $(GAMES),)'

debug: ## Debug a single game. ENV=alfworld(default)|scienceworld|tales|nethack. ALFWorld: GAMES=1,2,3 | TASK=1-6 | N=k random. ScienceWorld: SW_TASKS SW_VARS. TALES: TALES_ENVS. NetHack: NH_VARIANT NH_SEEDS.
	docker compose run --rm app \
	sh -lc 'set -eux; uv sync --frozen; \
	$(if $(filter scienceworld tales nethack,$(ENV)),,bash scripts/bootstrap_alfworld.sh;) \
	exec env WANDB_MODE=offline PYTHONPATH=/app uv run python scripts/run_agent.py \
	$(if $(filter scienceworld,$(ENV)), \
	  src/agent/configs/scienceworld.yaml --env-type scienceworld --num_games 1 $(if $(SW_TASKS),--sw-tasks $(SW_TASKS),) $(if $(SW_VARS),--sw-variations $(SW_VARS),--sw-variations 1), \
	$(if $(filter tales,$(ENV)), \
	  src/agent/configs/tales.yaml --env-type tales --num_games 1 $(if $(TALES_ENVS),--tales-envs $(TALES_ENVS),), \
	$(if $(filter nethack,$(ENV)), \
	  src/agent/configs/nethack.yaml --env-type nethack --num_games 1 $(if $(NH_VARIANT),--nethack-variant $(NH_VARIANT),) $(if $(NH_SEEDS),--nethack-seeds $(NH_SEEDS),) $(if $(RENDER),--render,), \
	  src/agent/configs/ALFworld.yaml --splits valid_unseen --max_chat_rounds 150 $(if $(GAMES),--game_ids $(GAMES),$(if $(TASK),--task_type $(TASK),--num_games $(or $(N),1)))))) \
	--gwt'

iterate-agent: ## Run one debug episode then hand to agent. ENV=tales|nethack (default: random). GAMES=N AGENT=claudecode|codex SKIP_DEBUG=1 ALLOW_DIRTY=1 PROMPT_ONLY=1 DANGEROUS=1
	uv run python scripts/iterate_agent.py \
	  --env $(or $(ENV),$(shell python3 -c "import random; print(random.choice(['tales','nethack']))")) \
	  $(if $(GAMES),--games $(GAMES),) \
	  $(if $(AGENT),--agent $(AGENT),) \
	  $(if $(SKIP_DEBUG),--skip-debug,) \
	  $(if $(ALLOW_DIRTY),--allow-dirty,) \
	  $(if $(PROMPT_ONLY),--prompt-only,) \
	  $(if $(DANGEROUS),--dangerous,)

ablate-agent: ## Reuse the latest experiment for a consolidation/ablation pass. ENV=tales|nethack (default: random). GAMES=N AGENT=claudecode|codex RUN_DEBUG=1 ALLOW_DIRTY=1 PROMPT_ONLY=1 DANGEROUS=1
	uv run python scripts/iterate_agent.py --mode ablate \
	  --env $(or $(ENV),$(shell python3 -c "import random; print(random.choice(['tales','nethack']))")) \
	  $(if $(GAMES),--games $(GAMES),) \
	  $(if $(AGENT),--agent $(AGENT),) \
	  $(if $(RUN_DEBUG),,--skip-debug) \
	  $(if $(ALLOW_DIRTY),--allow-dirty,) \
	  $(if $(PROMPT_ONLY),--prompt-only,) \
	  $(if $(DANGEROUS),--dangerous,)

test: ## Run tests with pytest
	uv run pytest tests/

lint: ## Fix and lint code with Ruff
	uvx ruff check . --fix
	uvx ruff format .

clean: ## Wipe the environment and caches (The nuclear option)
	@echo "🧹 Cleaning up..."
	rm -rf .venv .ruff_cache .pytest_cache .uv .uv-cache .uv-python
	git clean -fdX
	@echo "✨ Workspace is pristine."

build-docker: ## Build the production Docker image locally
	docker build -t my-ai-app .

up: ## Start the full stack (App + DB + Redis)
	docker compose up -d

down: ## Stop containers and remove them (preserves volumes)
	docker compose down

nuke: ## Stop everything and destroy all volumes (use sparingly — wipes venv + dataset caches)
	docker compose down -v

sanity:
	docker compose run --rm app \
	sh -lc 'set -eux; python -V; uv --version; uv run python -V; echo $$ALFWORLD_DATA; ls -la $$ALFWORLD_DATA || true'

bootstrap-alfworld:
	docker compose run --rm app \
	sh -lc 'set -eux; bash scripts/bootstrap_alfworld.sh'

benchmark: eval ## Alias for eval (backwards compat)

db-upgrade: ## Apply Alembic migrations to the current DATABASE_URL
	uv run alembic upgrade head

db-current: ## Show the current Alembic revision
	uv run alembic current

db-revision: ## Create a new Alembic revision. MESSAGE="describe change"
	uv run alembic revision --autogenerate -m "$(or $(MESSAGE),migration)"
