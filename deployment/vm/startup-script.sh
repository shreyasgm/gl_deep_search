#!/bin/bash
# VM Startup Script for ETL Pipeline
#
# This script runs automatically when a VM instance starts.
# It installs dependencies, clones the repository, and runs the ETL pipeline.
#
# This script is uploaded to GCS and referenced in VM metadata.

set -e

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/etl-pipeline.log
}

log "=========================================="
log "ETL Pipeline VM Startup Script"
log "=========================================="
log "Starting ETL pipeline VM setup..."

# Get configuration from metadata
GCS_BUCKET=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket 2>/dev/null || echo "")

GITHUB_REPO=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-repo 2>/dev/null || echo "")

GITHUB_BRANCH=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/github-branch 2>/dev/null || echo "main")

ETL_ARGS=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/etl-args 2>/dev/null || echo "")

PROJECT_ID=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || echo "")

# Set defaults if not provided
GCS_BUCKET=${GCS_BUCKET:-"gl-deep-search-data"}
GITHUB_REPO=${GITHUB_REPO:-"https://github.com/YOUR_ORG/gl_deep_search.git"}
GITHUB_BRANCH=${GITHUB_BRANCH:-"main"}

log "Configuration:"
log "  GCS Bucket: $GCS_BUCKET"
log "  GitHub Repo: $GITHUB_REPO"
log "  GitHub Branch: $GITHUB_BRANCH"
log "  ETL Args: $ETL_ARGS"
log "  Project ID: $PROJECT_ID"

# Install system dependencies
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

# Verify uv installation
if ! command -v uv &> /dev/null; then
    log "ERROR: Failed to install uv"
    exit 1
fi

log "uv version: $(uv --version)"

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

log "Successfully retrieved OpenAI API key"

# Set GCS bucket environment variable
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

# Update config.yaml to use GCS storage if production config exists
PROD_CONFIG="backend/etl/config.production.yaml"
if [[ -f "$PROD_CONFIG" ]]; then
    log "Using production configuration: $PROD_CONFIG"
    # Replace GCS bucket placeholder if needed
    sed -i "s|\${GCS_BUCKET:-.*}|$GCS_BUCKET|g" "$PROD_CONFIG" || true
else
    log "WARNING: Production config not found, using default config.yaml"
    PROD_CONFIG="backend/etl/config.yaml"
    # Update default config for cloud storage
    sed -i "s|local_storage_path: \"data/\"|gcs_bucket: \"$GCS_BUCKET\"|" "$PROD_CONFIG" || true
    sed -i "s|sync_to_gcs: false|sync_to_gcs: true|" "$PROD_CONFIG" || true
fi

# Run ETL pipeline
log "=========================================="
log "Starting ETL pipeline..."
log "=========================================="

# Build command with arguments
ETL_CMD="uv run python -m backend.etl.orchestrator --config $PROD_CONFIG --log-level INFO"

if [[ -n "$ETL_ARGS" ]]; then
    ETL_CMD="$ETL_CMD $ETL_ARGS"
fi

log "Executing: $ETL_CMD"

# Run pipeline and capture exit status
EXIT_STATUS=0
if $ETL_CMD 2>&1 | tee -a /var/log/etl-pipeline.log; then
    EXIT_STATUS=0
    log "Pipeline completed successfully!"
else
    EXIT_STATUS=$?
    log "Pipeline failed with exit code: $EXIT_STATUS"
fi

# Upload logs to GCS
log "Uploading logs to GCS..."
LOG_FILE="/var/log/etl-pipeline.log"
LOG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_GCS_PATH="gs://$GCS_BUCKET/logs/etl-${LOG_TIMESTAMP}.log"

if gcloud storage cp "$LOG_FILE" "$LOG_GCS_PATH" --quiet 2>/dev/null; then
    log "Logs uploaded successfully to: $LOG_GCS_PATH"
else
    log "WARNING: Failed to upload logs to GCS"
fi

# Upload execution report if it exists
REPORT_FILE="$WORK_DIR/data/reports/etl_execution_report.json"
if [[ -f "$REPORT_FILE" ]]; then
    REPORT_GCS_PATH="gs://$GCS_BUCKET/reports/etl-execution-${LOG_TIMESTAMP}.json"
    if gcloud storage cp "$REPORT_FILE" "$REPORT_GCS_PATH" --quiet 2>/dev/null; then
        log "Execution report uploaded to: $REPORT_GCS_PATH"
    fi
fi

# Final status
log "=========================================="
if [[ $EXIT_STATUS -eq 0 ]]; then
    log "ETL Pipeline completed successfully!"
else
    log "ETL Pipeline failed with errors (exit code: $EXIT_STATUS)"
fi
log "=========================================="

# Shutdown VM after completion
log "Shutting down VM..."
shutdown -h now

exit $EXIT_STATUS
