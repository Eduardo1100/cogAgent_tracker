# ==========================================
# Stage 1: Build stage (The Heavy Lifter)
# ==========================================
FROM ghcr.io/astral-sh/uv:latest AS uv_bin
FROM nvidia/cuda:12.4.1-base-ubuntu22.04 AS builder

# 1. Bring the lightning-fast 'uv' tool into our builder
COPY --from=uv_bin /uv /uvx /bin/

WORKDIR /app

# 2. Crucial environment variables for multi-stage Docker + uv
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_PYTHON_INSTALL_DIR=/python

# 3. Explicitly copy dependency files so uv can see them
COPY pyproject.toml uv.lock ./

# 4. Install Python and sync dependencies into a dedicated .venv
RUN uv python install 3.13
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev


# ==========================================
# Stage 2: Final Run stage (The Lean Runner)
# ==========================================
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# 5. CUDA images need these to understand the local Python libs we are copying
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-lib2to3 libpython3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 6. Bring over the 'uv' binary (so we can run 'uv sync' at container runtime)
COPY --from=uv_bin /uv /uvx /bin/

# 7. Bring over the compiled Python toolchain and our populated virtual environment
COPY --from=builder /python /python
COPY --from=builder /app/.venv /app/.venv

# 8. Copy the actual application code
COPY src/ /app/src/

# 9. Put our virtual environment and Python toolchain at the absolute front of the line
ENV PATH="/app/.venv/bin:/python/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

# 10. Run the app as a module to prevent relative import path errors
CMD ["/app/.venv/bin/python", "-m", "src.main"]