# --- Variables ---
PYTHON := uv run python
SHELL  := /bin/zsh

.PHONY: help setup dev train eval debug iterate-agent ablate-agent format lint test ci clean build-docker benchmark up down nuke sanity bootstrap-alfworld bootstrap-webarena bootstrap-androidworld webarena-check webarena-up webarena-down webarena-status androidworld-check androidworld-smoke protocol-slice protocol-slice-contacts protocol-slice-input-method db-upgrade db-revision db-current

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

debug: ## Debug a single game. AGENT=gwt(default)|baseline. SHOW_RUNTIME_SUMMARY=1 prints a compact controller hint each step. ENV=alfworld(default)|scienceworld|tales|nethack|webarena|androidworld. ALFWorld: GAMES=1,2,3 | TASK=1-6 | N=k random. ScienceWorld: SW_TASKS SW_VARS. TALES: TALES_ENVS. NetHack: NH_VARIANT NH_SEEDS. WebArena: WEBARENA_TASK_IDS WEBARENA_ENV_ID. AndroidWorld: ANDROIDWORLD_TASKS ANDROIDWORLD_SMOKE_SUITE ANDROIDWORLD_SUITE_FAMILY ANDROIDWORLD_CONSOLE_PORT ANDROIDWORLD_ADB_INSTALL_TIMEOUT. MAX_ACTIONS=N MAX_CHATROUNDS=N.
	$(if $(filter androidworld,$(ENV)),\
	bash -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_androidworld.sh; exec env WANDB_MODE=offline PYTHONPATH=. uv run python scripts/run_agent.py \
	src/agent/configs/androidworld.yaml --env-type androidworld --num_games 1 \
	$(if $(ANDROIDWORLD_TASKS),--androidworld-tasks $(ANDROIDWORLD_TASKS),) \
	$(if $(ANDROIDWORLD_SMOKE_SUITE),--androidworld-smoke-suite $(ANDROIDWORLD_SMOKE_SUITE),) \
	$(if $(ANDROIDWORLD_SUITE_FAMILY),--androidworld-suite-family $(ANDROIDWORLD_SUITE_FAMILY),) \
	$(if $(ANDROIDWORLD_N_TASK_COMBINATIONS),--androidworld-n-task-combinations $(ANDROIDWORLD_N_TASK_COMBINATIONS),) \
	$(if $(ANDROIDWORLD_TASK_RANDOM_SEED),--androidworld-task-random-seed $(ANDROIDWORLD_TASK_RANDOM_SEED),) \
	$(if $(ANDROIDWORLD_PERFORM_EMULATOR_SETUP),--androidworld-perform-emulator-setup,) \
	$(if $(ANDROIDWORLD_CONSOLE_PORT),--androidworld-console-port $(ANDROIDWORLD_CONSOLE_PORT),) \
	$(if $(ANDROIDWORLD_GRPC_PORT),--androidworld-grpc-port $(ANDROIDWORLD_GRPC_PORT),) \
	$(if $(ANDROIDWORLD_ADB_PATH),--androidworld-adb-path $(ANDROIDWORLD_ADB_PATH),) \
	$(if $(ANDROIDWORLD_ADB_INSTALL_TIMEOUT),--androidworld-adb-install-timeout $(ANDROIDWORLD_ADB_INSTALL_TIMEOUT),) \
	$(if $(MAX_ACTIONS),--max_actions $(MAX_ACTIONS),) \
	$(if $(MAX_CHATROUNDS),--max_chat_rounds $(MAX_CHATROUNDS),) \
	$(if $(SHOW_RUNTIME_SUMMARY),--show_runtime_summary,) \
	$(if $(filter baseline,$(AGENT)),--baseline,--gwt)',\
	docker compose run --rm app \
	sh -lc 'set -eux; export NLTK_DATA=/app/.nltk_data; uv sync --frozen; \
	$(if $(filter scienceworld tales nethack webarena,$(ENV)),,bash scripts/bootstrap_alfworld.sh;) \
	$(if $(filter webarena,$(ENV)),bash scripts/bootstrap_webarena.sh;) \
	exec env WANDB_MODE=offline PYTHONPATH=/app NLTK_DATA=/app/.nltk_data uv run python scripts/run_agent.py \
	$(if $(filter scienceworld,$(ENV)),src/agent/configs/scienceworld.yaml --env-type scienceworld --num_games 1 $(if $(SW_TASKS),--sw-tasks $(SW_TASKS),) $(if $(SW_VARS),--sw-variations $(SW_VARS),--sw-variations 1),\
	  $(if $(filter tales,$(ENV)),src/agent/configs/tales.yaml --env-type tales --num_games 1 $(if $(TALES_ENVS),--tales-envs $(TALES_ENVS),),\
	    $(if $(filter nethack,$(ENV)),src/agent/configs/nethack.yaml --env-type nethack --num_games 1 $(if $(NH_VARIANT),--nethack-variant $(NH_VARIANT),) $(if $(NH_SEEDS),--nethack-seeds $(NH_SEEDS),) $(if $(RENDER),--render,),\
	      $(if $(filter webarena,$(ENV)),src/agent/configs/webarena.yaml --env-type webarena --num_games 1 $(if $(WEBARENA_TASK_IDS),--webarena-task-ids $(WEBARENA_TASK_IDS),) $(if $(WEBARENA_ENV_ID),--webarena-env-id $(WEBARENA_ENV_ID),) $(if $(RENDER),--render,),\
	        src/agent/configs/ALFworld.yaml --splits valid_unseen --max_chat_rounds 150 $(if $(GAMES),--game_ids $(GAMES),$(if $(TASK),--task_type $(TASK),--num_games $(or $(N),1)))\
	      )\
	    )\
	  )\
	) \
	$(if $(MAX_ACTIONS),--max_actions $(MAX_ACTIONS),) \
	$(if $(MAX_CHATROUNDS),--max_chat_rounds $(MAX_CHATROUNDS),) \
	$(if $(SHOW_RUNTIME_SUMMARY),--show_runtime_summary,$(if $(and $(filter nethack,$(ENV)),$(filter-out baseline,$(AGENT))),--show_runtime_summary,)) \
	$(if $(filter baseline,$(AGENT)),--baseline,--gwt)')

iterate-agent: ## Run one debug episode then hand to agent. ENV=tales|nethack (default: random for debug, latest for skip-debug). GAMES=N AGENT=claudecode|codex SKIP_DEBUG=1 ALLOW_DIRTY=1 PROMPT_ONLY=1 DANGEROUS=1 MAX_ACTIONS=N MAX_CHATROUNDS=N
	uv run python scripts/iterate_agent.py \
	  $(if $(ENV),--env $(ENV),) \
	  $(if $(GAMES),--games $(GAMES),) \
	  $(if $(AGENT),--agent $(AGENT),) \
	  $(if $(SKIP_DEBUG),--skip-debug,) \
	  $(if $(ALLOW_DIRTY),--allow-dirty,) \
	  $(if $(PROMPT_ONLY),--prompt-only,) \
	  $(if $(DANGEROUS),--dangerous,) \
	  $(if $(MAX_ACTIONS),--max-actions $(MAX_ACTIONS),) \
	  $(if $(MAX_CHATROUNDS),--max-chatrounds $(MAX_CHATROUNDS),)

ablate-agent: ## Reuse the latest experiment for a consolidation/ablation pass. ENV=tales|nethack (default: random for debug, latest for skip-debug). GAMES=N AGENT=claudecode|codex RUN_DEBUG=1 ALLOW_DIRTY=1 PROMPT_ONLY=1 DANGEROUS=1 MAX_ACTIONS=N MAX_CHATROUNDS=N
	uv run python scripts/iterate_agent.py --mode ablate \
	  $(if $(ENV),--env $(ENV),) \
	  $(if $(GAMES),--games $(GAMES),) \
	  $(if $(AGENT),--agent $(AGENT),) \
	  $(if $(RUN_DEBUG),,--skip-debug) \
	  $(if $(ALLOW_DIRTY),--allow-dirty,) \
	  $(if $(PROMPT_ONLY),--prompt-only,) \
	  $(if $(DANGEROUS),--dangerous,) \
	  $(if $(MAX_ACTIONS),--max-actions $(MAX_ACTIONS),) \
	  $(if $(MAX_CHATROUNDS),--max-chatrounds $(MAX_CHATROUNDS),)

format: ## Auto-fix and format code locally
	uvx ruff==0.15.4 check . --fix
	uvx ruff==0.15.4 format .

lint: ## Check formatting and lint without modifying files
	uvx ruff==0.15.4 format . --check
	uvx ruff==0.15.4 check .

test: ## Run tests (CI-style, fail fast)
	uv run pytest tests/ -x -q

ci: lint test ## Run CI-equivalent local checks

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

bootstrap-webarena:
	docker compose run --rm app \
	sh -lc 'set -eux; export NLTK_DATA=/app/.nltk_data; uv sync --frozen; bash scripts/bootstrap_webarena.sh'

bootstrap-androidworld:
	bash -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_androidworld.sh'

webarena-check: ## Validate WA_* URLs from inside the app container before a real WebArena run
	docker compose run --rm app \
	sh -lc 'set -eux; export NLTK_DATA=/app/.nltk_data; uv sync --frozen; exec env PYTHONPATH=/app NLTK_DATA=/app/.nltk_data uv run python scripts/check_webarena.py'

androidworld-check: ## Validate local AndroidWorld Python runtime plus adb/emulator visibility
	bash -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_androidworld.sh; env PYTHONPATH=. uv run python scripts/check_androidworld.py $(if $(ANDROIDWORLD_ADB_PATH),--adb-path $(ANDROIDWORLD_ADB_PATH),) $(if $(ANDROIDWORLD_CONSOLE_PORT),--console-port $(ANDROIDWORLD_CONSOLE_PORT),)'

androidworld-smoke: ## Run the small AndroidWorld transfer smoke suite (browser/text-entry/ui)
	bash -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_androidworld.sh; exec env WANDB_MODE=offline PYTHONPATH=. uv run python scripts/run_agent.py src/agent/configs/androidworld.yaml --env-type androidworld --androidworld-smoke-suite core --num_games 3 $(if $(ANDROIDWORLD_CONSOLE_PORT),--androidworld-console-port $(ANDROIDWORLD_CONSOLE_PORT),) $(if $(ANDROIDWORLD_GRPC_PORT),--androidworld-grpc-port $(ANDROIDWORLD_GRPC_PORT),) $(if $(ANDROIDWORLD_ADB_PATH),--androidworld-adb-path $(ANDROIDWORLD_ADB_PATH),) $(if $(ANDROIDWORLD_ADB_INSTALL_TIMEOUT),--androidworld-adb-install-timeout $(ANDROIDWORLD_ADB_INSTALL_TIMEOUT),) $(if $(MAX_ACTIONS),--max_actions $(MAX_ACTIONS),) $(if $(MAX_CHATROUNDS),--max_chat_rounds $(MAX_CHATROUNDS),) $(if $(SHOW_RUNTIME_SUMMARY),--show_runtime_summary,) $(if $(filter baseline,$(AGENT)),--baseline,--gwt)'

protocol-slice: ## Replay a JSON runtime trace into force_protocol vs auto cohorts. TRACE=fixtures/openclaw/contacts_focus_repair_trace.json FIELD=summary|json BASE_URL=http://localhost:8000
	uv run python scripts/run_protocol_mode_slice.py $(or $(TRACE),fixtures/openclaw/contacts_focus_repair_trace.json) $(if $(FIELD),--field $(FIELD),) $(if $(BASE_URL),--base-url $(BASE_URL),)

protocol-slice-contacts: ## Replay the checked-in contacts focus-repair fixture against the local runtime API
	uv run python scripts/run_protocol_mode_slice.py fixtures/openclaw/contacts_focus_repair_trace.json $(if $(FIELD),--field $(FIELD),) $(if $(BASE_URL),--base-url $(BASE_URL),)

protocol-slice-input-method: ## Replay the checked-in input-method detour fixture against the local runtime API
	uv run python scripts/run_protocol_mode_slice.py fixtures/openclaw/input_method_detour_trace.json $(if $(FIELD),--field $(FIELD),) $(if $(BASE_URL),--base-url $(BASE_URL),)

webarena-up: ## Start the upstream WebArena site stack on the host via the pinned source checkout
	bash scripts/manage_webarena_stack.sh up

webarena-down: ## Stop the upstream WebArena site stack on the host
	bash scripts/manage_webarena_stack.sh down

webarena-status: ## Show the upstream WebArena site stack status on the host
	bash scripts/manage_webarena_stack.sh status

benchmark: eval ## Alias for eval (backwards compat)

db-upgrade: ## Apply Alembic migrations to the current DATABASE_URL
	uv run alembic upgrade head

db-current: ## Show the current Alembic revision
	uv run alembic current

db-revision: ## Create a new Alembic revision. MESSAGE="describe change"
	uv run alembic revision --autogenerate -m "$(or $(MESSAGE),migration)"
