# ==========================================
# Stage 1: uv binary
# ==========================================
FROM ghcr.io/astral-sh/uv:latest AS uv_bin

# ==========================================
# Stage 2: Builder
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

COPY --from=uv_bin /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    unzip \
    git \
    build-essential \
    cmake \
    gcc \
    g++ \
    make \
    libpq-dev \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed

COPY pyproject.toml uv.lock ./

RUN uv python install 3.11.15

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --python 3.11.15

# ==========================================
# Stage 3: Runtime
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

COPY --from=uv_bin /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    unzip \
    git \
    build-essential \
    gcc \
    g++ \
    make \
    libpq5 \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# bring in uv-managed python installs
COPY --from=builder /opt/uv-python /opt/uv-python

# make python available on PATH
RUN mkdir -p /usr/local/bin && \
    PY="$(find /opt/uv-python -type f -path '*/bin/python3.11' | head -n 1)" && \
    test -n "$PY" && test -x "$PY" && \
    ln -sf "$PY" /usr/local/bin/python3.11 && \
    ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.11 /usr/local/bin/python

# writable runtime dirs
RUN mkdir -p /opt/venv /opt/uv-cache /datasets/alfworld /wandb

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=only-managed \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:/usr/local/bin:$PATH"