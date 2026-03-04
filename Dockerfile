# ==========================================
# Stage 1: Build stage (The Heavy Lifter)
# ==========================================
FROM mcr.microsoft.com/vscode/devcontainers/python:3.11
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

# 1. Bring the lightning-fast 'uv' tool into our builder
COPY --from=uv_bin /uv /uvx /bin/

# 2. INSTALL BUILD TOOLS HERE (Crucial for Jericho/ALFWorld)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Environment variables for uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/python

# 4. Copy dependency files
COPY pyproject.toml uv.lock ./

# 5. Install Python and sync dependencies
# We use --no-install-project because we haven't copied the src yet
RUN uv python install 3.11
ENV UV_PYTHON_PREFERENCE=only-managed
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --python 3.11

# ==========================================
# Stage 2: Final Run stage (The Lean Runner)
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# !!! CRUCIAL: Add make and gcc here because 'uv sync' runs at container startup !!!
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    g++ \
    libpq5 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring over uv and python toolchain
COPY --from=uv_bin /uv /uvx /bin/
COPY --from=builder /python /python

# We DON'T copy the .venv from builder because your docker-compose 
# uses a volume (venv_storage) which will overwrite it anyway.

ENV PATH="/app/.venv/bin:/python/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    PYTHONUNBUFFERED=1

# The command is handled by docker-compose.yml