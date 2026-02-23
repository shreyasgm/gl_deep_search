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

# Install dependencies (cached unless pyproject.toml/uv.lock change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-dev --no-install-project --extra etl

# Copy project code and install
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra etl

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
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Create data and logs dirs
RUN mkdir -p /app/data /app/logs /app/reports

# Pre-download Marker/surya font files so they exist in the image.
# Singularity .sif images are read-only — without this, Marker fails with
# "[Errno 30] Read-only file system: .../site-packages/static"
RUN mkdir -p /app/.venv/lib/python3.12/site-packages/static/fonts && \
    curl -sL https://models.datalab.to/artifacts/GoNotoCurrent-Regular.ttf \
        -o /app/.venv/lib/python3.12/site-packages/static/fonts/GoNotoCurrent-Regular.ttf

# Default: run the PDF processor script
ENTRYPOINT ["python", "-m"]
CMD ["backend.etl.scripts.run_pdf_processor"]
