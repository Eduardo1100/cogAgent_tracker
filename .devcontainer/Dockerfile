FROM mcr.microsoft.com/vscode/devcontainers/python:3.11

# Install system dependencies for ALFWorld & Postgres
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' for fast package management (matching your compose command)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

USER vscode
WORKDIR /app