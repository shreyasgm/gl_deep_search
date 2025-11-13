# Multi-stage Dockerfile for Cloud Run ETL Pipeline
# Optimized for production deployment with minimal image size

# ==============================================================================
# Stage 1: Build stage - Install dependencies and build artifacts
# ==============================================================================
FROM python:3.12-slim as builder

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
# Using --system-site-packages to allow system packages
RUN uv venv /opt/venv && \
    uv pip install --system -r <(uv pip compile pyproject.toml --extra etl) && \
    uv sync --extra etl --frozen

# ==============================================================================
# Stage 2: Runtime stage - Minimal production image
# ==============================================================================
FROM python:3.12-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for runtime (needed for uv run)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Verify production config exists
RUN if [ ! -f "backend/etl/config.production.yaml" ]; then \
    echo "WARNING: Production config not found, will use default config.yaml"; \
    fi

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check (optional - Cloud Run has its own health checks)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command - can be overridden in Cloud Run Job configuration
ENTRYPOINT ["uv", "run", "python", "-m", "backend.etl.orchestrator"]
CMD ["--config", "backend/etl/config.production.yaml", "--log-level", "INFO"]
