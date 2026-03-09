# --- Variables ---
PYTHON := uv run python
SHELL  := /bin/zsh

.PHONY: help setup dev train test lint clean build-docker benchmark

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

train: ## Run eval loop on the valid_seen split
	docker compose run --rm app \
	sh -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_alfworld.sh; PYTHONPATH=/app uv run python scripts/run_agent.py src/agent/configs/eval_config.yaml --gwt --splits valid_seen'

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

down: ## Stop everything and remove containers
	docker compose down -v

sanity:
	docker compose run --rm app \
	sh -lc 'set -eux; python -V; uv --version; uv run python -V; echo $$ALFWORLD_DATA; ls -la $$ALFWORLD_DATA || true'

bootstrap-alfworld:
	docker compose run --rm app \
	sh -lc 'set -eux; bash scripts/bootstrap_alfworld.sh'

benchmark: ## Run the full benchmark on valid_unseen
	docker compose run --rm app \
	sh -lc 'set -eux; uv sync --frozen; bash scripts/bootstrap_alfworld.sh; PYTHONPATH=/app uv run python scripts/run_agent.py src/agent/configs/eval_config.yaml --gwt --splits valid_unseen'