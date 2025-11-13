#!/bin/bash
# VM Incremental Update Script for ETL Pipeline
#
# This script is similar to startup-script.sh but optimized for incremental updates.
# It limits scraping to recent publications and skips full reprocessing.

set -e

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/etl-incremental.log
}

log "=========================================="
log "ETL Pipeline Incremental Update Script"
log "=========================================="
log "Starting incremental ETL update..."

# Get configuration from metadata
GCS_BUCKET=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket 2>/dev/null || echo "")

GITHUB_REPO=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-repo 2>/dev/null || echo "")

GITHUB_BRANCH=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-branch 2>/dev/null || echo "main")

PROJECT_ID=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || echo "")

# Set defaults
GCS_BUCKET=${GCS_BUCKET:-"gl-deep-search-data"}
GITHUB_REPO=${GITHUB_REPO:-"https://github.com/YOUR_ORG/gl_deep_search.git"}
GITHUB_BRANCH=${GITHUB_BRANCH:-"main"}

log "Configuration:"
log "  GCS Bucket: $GCS_BUCKET"
log "  GitHub Repo: $GITHUB_REPO"
log "  GitHub Branch: $GITHUB_BRANCH"
log "  Project ID: $PROJECT_ID"

# Install system dependencies (same as startup-script.sh)
log "Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    > /dev/null 2>&1

# Install uv package manager
log "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:$PATH"

# Set up working directory
log "Setting up working directory..."
WORK_DIR="/opt/gl-deep-search"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Clone repository
log "Cloning repository..."
if [[ -d ".git" ]]; then
    log "Repository already exists, pulling latest changes..."
    git pull origin "$GITHUB_BRANCH" || log "WARNING: Failed to pull latest changes"
else
    git clone --branch "$GITHUB_BRANCH" "$GITHUB_REPO" . || {
        log "ERROR: Failed to clone repository"
        exit 1
    }
fi

# Get OpenAI API key from Secret Manager
log "Retrieving OpenAI API key from Secret Manager..."
export OPENAI_API_KEY=$(gcloud secrets versions access latest --secret="openai-api-key" --project="$PROJECT_ID" 2>/dev/null || echo "")

if [[ -z "$OPENAI_API_KEY" ]]; then
    log "ERROR: Failed to retrieve OpenAI API key from Secret Manager"
    exit 1
fi

# Set environment variables
export GCS_BUCKET="$GCS_BUCKET"
export ENVIRONMENT="production"

# Install Python dependencies
log "Installing Python dependencies..."
uv sync --extra etl --quiet || {
    log "ERROR: Failed to install Python dependencies"
    exit 1
}

# Create .env file
log "Creating environment configuration..."
cat > .env << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
GCS_BUCKET=$GCS_BUCKET
ENVIRONMENT=production
EOF

# Use production config
PROD_CONFIG="backend/etl/config.production.yaml"
if [[ ! -f "$PROD_CONFIG" ]]; then
    PROD_CONFIG="backend/etl/config.yaml"
fi

# Run ETL pipeline in incremental mode
log "=========================================="
log "Running incremental ETL update..."
log "=========================================="

# Incremental update: limit scraper to recent publications
# This assumes the orchestrator supports --scraper-limit flag
ETL_CMD="uv run python -m backend.etl.orchestrator --config $PROD_CONFIG --log-level INFO --scraper-limit 20"

log "Executing: $ETL_CMD"

EXIT_STATUS=0
if $ETL_CMD 2>&1 | tee -a /var/log/etl-incremental.log; then
    EXIT_STATUS=0
    log "Incremental update completed successfully!"
else
    EXIT_STATUS=$?
    log "Incremental update failed with exit code: $EXIT_STATUS"
fi

# Upload logs to GCS
log "Uploading logs to GCS..."
LOG_FILE="/var/log/etl-incremental.log"
LOG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_GCS_PATH="gs://$GCS_BUCKET/logs/etl-incremental-${LOG_TIMESTAMP}.log"

if gcloud storage cp "$LOG_FILE" "$LOG_GCS_PATH" --quiet 2>/dev/null; then
    log "Logs uploaded successfully to: $LOG_GCS_PATH"
else
    log "WARNING: Failed to upload logs to GCS"
fi

log "=========================================="
log "Incremental update completed. Shutting down..."
log "=========================================="

# Shutdown VM
shutdown -h now

exit $EXIT_STATUS
