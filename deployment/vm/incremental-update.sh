#!/bin/bash
# VM Incremental Update Script - Docker-based ETL Pipeline
#
# This script is similar to startup-script.sh but optimized for incremental updates.
# It uses Docker containers and limits scraping to recent publications.

set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/etl-incremental-startup.log
}

log "=========================================="
log "ETL Pipeline Incremental Update Script (Docker-based)"
log "=========================================="
log "Starting incremental ETL update..."

# Get metadata from VM instance
GCS_BUCKET=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket 2>/dev/null || echo "")

PROJECT_ID=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || echo "")

IMAGE_NAME=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/image-name 2>/dev/null || echo "")

REGION=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | sed 's|.*/||' | sed 's|-[a-z]$||' || echo "us-central1")

# Set defaults
GCS_BUCKET=${GCS_BUCKET:-"gl-deep-search-data"}
REGION=${REGION:-"us-central1"}

log "Configuration:"
log "  GCS Bucket: $GCS_BUCKET"
log "  Project ID: $PROJECT_ID"
log "  Image Name: $IMAGE_NAME"
log "  Region: $REGION"

# Validate required configuration
if [[ -z "$IMAGE_NAME" ]]; then
    log "ERROR: Image name not provided in metadata"
    exit 1
fi

if [[ -z "$PROJECT_ID" ]]; then
    log "ERROR: Project ID not provided"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    log "Installing Docker..."
    export DEBIAN_FRONTEND=noninteractive

    apt-get update -qq
    apt-get install -y -qq \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        > /dev/null 2>&1

    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null 2>&1

    if ! command -v docker &> /dev/null; then
        log "ERROR: Failed to install Docker"
        exit 1
    fi
    log "Docker installed successfully"
else
    log "Docker already installed"
fi

log "Docker version: $(docker --version)"

# Configure Docker authentication
log "Configuring Docker authentication for Artifact Registry..."

if ! command -v gcloud &> /dev/null; then
    log "Installing Google Cloud SDK..."
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
        tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    apt-get update -qq && apt-get install -y -qq google-cloud-cli > /dev/null 2>&1
    log "Google Cloud SDK installed"
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
log "Docker authentication configured"

# Pull container image
log "Pulling container image: $IMAGE_NAME"
if ! docker pull "$IMAGE_NAME"; then
    log "ERROR: Failed to pull container image"
    exit 1
fi

log "Successfully pulled container image"

# Run ETL pipeline in incremental mode
log "=========================================="
log "Running incremental ETL update in container..."
log "=========================================="

# Incremental update: limit scraper to recent publications (default: 20)
DOCKER_CMD=(
    docker run --rm
    --name etl-pipeline
    -e GCS_BUCKET="$GCS_BUCKET"
    -e ENVIRONMENT="production"
    -e GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
    "$IMAGE_NAME"
    --config backend/etl/config.production.yaml
    --log-level INFO
    --scraper-limit 20
)

log "Executing: ${DOCKER_CMD[*]}"

EXIT_STATUS=0
if "${DOCKER_CMD[@]}" 2>&1 | tee -a /var/log/etl-incremental.log; then
    EXIT_STATUS=0
    log "Incremental update completed successfully!"
else
    EXIT_STATUS=$?
    log "Incremental update failed with exit code: $EXIT_STATUS"
fi

# Upload logs to GCS
log "=========================================="
log "Uploading logs to GCS..."
log "=========================================="

LOG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
STARTUP_LOG="/var/log/etl-incremental-startup.log"
PIPELINE_LOG="/var/log/etl-incremental.log"

if [[ -f "$STARTUP_LOG" ]]; then
    LOG_GCS_PATH="gs://$GCS_BUCKET/logs/vm-incremental-startup-${LOG_TIMESTAMP}.log"
    if gcloud storage cp "$STARTUP_LOG" "$LOG_GCS_PATH" --quiet 2>/dev/null; then
        log "Startup log uploaded to: $LOG_GCS_PATH"
    else
        log "WARNING: Failed to upload startup log"
    fi
fi

if [[ -f "$PIPELINE_LOG" ]]; then
    LOG_GCS_PATH="gs://$GCS_BUCKET/logs/etl-incremental-${LOG_TIMESTAMP}.log"
    if gcloud storage cp "$PIPELINE_LOG" "$LOG_GCS_PATH" --quiet 2>/dev/null; then
        log "Pipeline log uploaded to: $LOG_GCS_PATH"
    else
        log "WARNING: Failed to upload pipeline log"
    fi
fi

log "=========================================="
log "Incremental update script completed"
log "Exit status: $EXIT_STATUS"
log "=========================================="

# Shut down the VM
log "Shutting down VM..."
shutdown -h now

exit $EXIT_STATUS
