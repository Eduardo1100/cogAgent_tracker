# --- Variables ---
PYTHON := uv run python
SHELL  := /bin/zsh

.PHONY: help setup dev train test lint clean build-docker

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install all tools, Python versions, and dependencies
	@echo "🚀 Setting up environment..."
	mise install
	uv sync
	@echo "✅ Setup complete. Environment is ready."

dev: ## Start the development environment (notebooks + local API)
	@echo "🛠️ Starting dev services..."
	uv run marimo edit notebooks/exploration.py &
	uv run fastapi dev src/api.py

train: ## Run the model training pipeline
	@echo "🧠 Starting training..."
	uv run python src/train.py

test: ## Run tests with pytest
	uv run pytest tests/

lint: ## Fix and lint code with Ruff
	uvx ruff check . --fix
	uvx ruff format .

clean: ## Wipe the environment and caches (The nuclear option)
	@echo "🧹 Cleaning up..."
	rm -rf .venv .ruff_cache .pytest_cache .uv
	git clean -fdX
	@echo "✨ Workspace is pristine."

build-docker: ## Build the production Docker image locally
	docker build -t my-ai-app .

up: ## Start the full stack (App + DB + Redis)
	docker compose up -d

down: ## Stop everything and remove containers
	docker compose down