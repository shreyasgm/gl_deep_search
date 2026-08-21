# Dockerfile for PDF Processing (Marker + Docling fallback)
#
# Used locally via docker-compose and on FAS-RC via Singularity.
# Singularity's --nv flag handles GPU passthrough at runtime,
# so this image works for both CPU (local) and GPU (cluster).
#
# Build:
#   docker build -f deployment/pdf-processing/Dockerfile -t gl-pdf-processing .
#
# Convert to Singularity:
#   singularity pull gl-pdf-processing.sif docker-daemon://gl-pdf-processing:latest

# ==============================================================================
# Stage 1: Builder - Install Python dependencies
# ==============================================================================
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (cached unless pyproject.toml/uv.lock change).
# --frozen, not --locked: install exactly what uv.lock pins without
# re-validating it against pyproject.toml. A developer's global uv config
# (e.g. exclude-newer) can add an [options] block to uv.lock that a config-less
# uv inside the container rejects, which broke the build on 2026-08-21.
# CI still runs `uv sync --locked`, so the lock is validated there.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --frozen --no-dev --no-install-project --extra etl

# Copy project code and install
COPY . /app
# Normalise permissions from the build context. A repo checked out under a
# 0700 home or Dropbox folder yields /app/backend as drwx------, and since
# Singularity runs as the invoking user rather than root those directories
# become untraversable at runtime: `python -m backend.etl.orchestrator` fails
# with ModuleNotFoundError and relative --config paths cannot be read.
# Done here, in the throwaway builder stage, so the fix costs no layer in the
# final image (a chmod -R over the runtime /app would duplicate the ~4.6GB venv).
RUN chmod -R a+rX /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra etl

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.12-slim-bookworm

# Install system deps for PDF processing:
#   - tesseract-ocr: OCR engine (used by Docling's tesseract backend)
#   - poppler-utils: pdftotext/pdfinfo (used by Marker)
#   - libgl1: OpenCV dependency (used by Marker layout detection)
#   - libglib2.0-0: GLib (used by various image processing libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application from builder
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Create data and logs dirs
RUN mkdir -p /app/data /app/logs /app/reports

# Pre-download Marker/surya font files so they exist in the image.
# Singularity .sif images are read-only — without this, Marker fails with
# "[Errno 30] Read-only file system: .../site-packages/static"
RUN mkdir -p /app/.venv/lib/python3.12/site-packages/static/fonts && \
    curl -sL https://models.datalab.to/artifacts/GoNotoCurrent-Regular.ttf \
        -o /app/.venv/lib/python3.12/site-packages/static/fonts/GoNotoCurrent-Regular.ttf

# Record which commit this image was built from. The SLURM job compares this
# against the cluster checkout and refuses to run on a mismatch — a stale image
# silently missing new CLI flags cost a production run on 2026-02-25.
ARG GIT_SHA=unknown
RUN echo "$GIT_SHA" > /app/GIT_SHA
LABEL org.opencontainers.image.revision=$GIT_SHA

# Default: run the PDF processor script
ENTRYPOINT ["python", "-m"]
CMD ["backend.etl.scripts.run_pdf_processor"]
