# Multi-stage Dockerfile for ETL Pipeline
# Based on official uv-docker-example multistage pattern
# Used by both Cloud Run and VM instances

# ==============================================================================
# Stage 1: Builder - Install dependencies and build artifacts
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Set uv optimization flags
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

# Install build-time dependencies needed for compiling Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (better caching)
# This layer only rebuilds when pyproject.toml or uv.lock changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-dev --no-install-project --extra etl-lite

# Copy project code and install the project itself
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra etl-lite

# ==============================================================================
# Stage 2: Runtime - Minimal production image
# ==============================================================================
FROM python:3.12-slim-bookworm

# Install ONLY runtime dependencies (no build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    gnupg \
    ca-certificates \
    apt-transport-https \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user (security best practice)
RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

# Copy the application from the builder (including .venv)
COPY --from=builder --chown=nonroot:nonroot /app /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy entrypoint script and make it executable (as root, then switch user)
COPY deployment/cloud-run/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown nonroot:nonroot /entrypoint.sh

# Verify production config exists
RUN if [ ! -f "backend/etl/config.production.yaml" ]; then \
    echo "WARNING: Production config not found, will use default config.yaml"; \
    fi

# Create directories for logs and data with proper ownership
RUN mkdir -p /app/logs /app/data && chown -R nonroot:nonroot /app/logs /app/data

# Use the non-root user to run our application
USER nonroot

# Set entrypoint and default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--config", "backend/etl/config.production.yaml", "--log-level", "INFO"]
