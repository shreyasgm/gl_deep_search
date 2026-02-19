#!/bin/bash
# VM Startup Script - Docker-based ETL Pipeline
#
# This script runs automatically when a VM instance starts.
# It installs Docker, pulls the pre-built container image, and runs the ETL pipeline.
#
# This script is uploaded to GCS and referenced in VM metadata.

set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a /var/log/etl-startup.log
}

log "=========================================="
log "ETL Pipeline VM Startup Script (Docker-based)"
log "=========================================="
log "Starting ETL pipeline VM setup..."

# Get metadata from VM instance
GCS_BUCKET=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket 2>/dev/null || echo "")

PROJECT_ID=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/project/project-id 2>/dev/null || echo "")

IMAGE_NAME=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/image-name 2>/dev/null || echo "")

ETL_ARGS=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/etl-args 2>/dev/null || echo "")

REGION=$(curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | sed 's|.*/||' | sed 's|-[a-z]$||' || echo "us-central1")

# Set defaults if not provided
GCS_BUCKET=${GCS_BUCKET:-"gl-deep-search-data"}
REGION=${REGION:-"us-central1"}

log "Configuration:"
log "  GCS Bucket: $GCS_BUCKET"
log "  Project ID: $PROJECT_ID"
log "  Image Name: $IMAGE_NAME"
log "  ETL Args: $ETL_ARGS"
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

    # Update package list
    apt-get update -qq

    # Install prerequisites
    apt-get install -y -qq \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        > /dev/null 2>&1

    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    apt-get update -qq
    apt-get install -y -qq \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin \
        > /dev/null 2>&1

    # Verify Docker installation
    if ! command -v docker &> /dev/null; then
        log "ERROR: Failed to install Docker"
        exit 1
    fi
    log "Docker installed successfully"
else
    log "Docker already installed"
fi

log "Docker version: $(docker --version)"

# Configure Docker to authenticate with Artifact Registry
# The VM's service account will be used for authentication
log "Configuring Docker authentication for Artifact Registry..."

# Install google-cloud-sdk if not present (for gcloud command)
if ! command -v gcloud &> /dev/null; then
    log "Installing Google Cloud SDK..."
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
        tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
    apt-get update -qq && apt-get install -y -qq google-cloud-cli > /dev/null 2>&1
    log "Google Cloud SDK installed"
fi

# Configure Docker to use gcloud as credential helper
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

log "Docker authentication configured"

# Pull container image
log "Pulling container image: $IMAGE_NAME"
if ! docker pull "$IMAGE_NAME"; then
    log "ERROR: Failed to pull container image"
    log "Check that the image exists in Artifact Registry:"
    log "  gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/*"
    exit 1
fi

log "Successfully pulled container image"

# Run ETL pipeline in container
log "=========================================="
log "Starting ETL pipeline in container..."
log "=========================================="

# Prepare host directory for pipeline data (bind-mounted into container)
ETL_DATA_DIR="/tmp/etl-data"
mkdir -p "$ETL_DATA_DIR"

# Restore existing data from GCS to enable incremental processing
# Components (downloader, PDF processor, chunker, embeddings) all have
# skip logic that checks if output files already exist
log "=========================================="
log "Restoring existing data from GCS..."
log "=========================================="

for dir in raw intermediate processed; do
    SRC="gs://${GCS_BUCKET}/${dir}"
    DEST="${ETL_DATA_DIR}/${dir}"
    mkdir -p "$DEST"
    if gcloud storage rsync -r "$SRC/" "$DEST/" --quiet 2>/dev/null; then
        FILE_COUNT=$(find "$DEST" -type f | wc -l)
        log "  Restored ${dir}/ from GCS ($FILE_COUNT files)"
    else
        log "  No existing ${dir}/ in GCS (starting fresh)"
    fi
done

log "Data restore complete"

# Container runs as nonroot (UID 999), so set ownership accordingly
chown -R 999:999 "$ETL_DATA_DIR"

# Build docker run command
# Bind-mount host dir so pipeline output persists after container exits
DOCKER_CMD=(
    docker run --rm
    --name etl-pipeline
    -v "${ETL_DATA_DIR}:/app/data"
    -e GCS_BUCKET="$GCS_BUCKET"
    -e ENVIRONMENT="production"
    -e GOOGLE_CLOUD_PROJECT="$PROJECT_ID"
    "$IMAGE_NAME"
)

# Always include config and log-level (required args)
# Then append any additional ETL args if provided
DOCKER_CMD+=(--config backend/etl/config.production.yaml --log-level INFO)

# Add ETL args if provided (e.g., --scraper-limit)
if [[ -n "$ETL_ARGS" ]]; then
    # Split ETL_ARGS into array and append to DOCKER_CMD
    read -ra ARGS_ARRAY <<< "$ETL_ARGS"
    DOCKER_CMD+=("${ARGS_ARRAY[@]}")
fi

log "Executing: ${DOCKER_CMD[*]}"

# Run pipeline and capture exit status
# Logs go to stdout/stderr and are captured by Docker
EXIT_STATUS=0
if "${DOCKER_CMD[@]}" 2>&1 | tee -a /var/log/etl-pipeline.log; then
    EXIT_STATUS=0
    log "Pipeline completed successfully!"
else
    EXIT_STATUS=$?
    log "Pipeline failed with exit code: $EXIT_STATUS"
fi

# Upload pipeline data to GCS
log "=========================================="
log "Uploading pipeline data to GCS..."
log "=========================================="

DATA_DIRS=("raw" "intermediate" "processed" "reports")
for dir in "${DATA_DIRS[@]}"; do
    SRC="${ETL_DATA_DIR}/${dir}"
    if [[ -d "$SRC" ]]; then
        DEST="gs://${GCS_BUCKET}/${dir}"
        FILE_COUNT=$(find "$SRC" -type f | wc -l)
        log "Uploading ${dir}/ to ${DEST}/ (${FILE_COUNT} files)"
        if gcloud storage rsync -r "$SRC" "$DEST" --quiet 2>/dev/null; then
            log "  Uploaded ${dir}/ successfully"
        else
            log "WARNING: Failed to upload ${dir}/"
        fi
    else
        log "  Skipping ${dir}/ (not found)"
    fi
done

log "Data upload complete"

# Upload logs to GCS
log "=========================================="
log "Uploading logs to GCS..."
log "=========================================="

LOG_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
STARTUP_LOG="/var/log/etl-startup.log"
PIPELINE_LOG="/var/log/etl-pipeline.log"

# Upload startup log
if [[ -f "$STARTUP_LOG" ]]; then
    STARTUP_LOG_GCS="gs://$GCS_BUCKET/logs/vm-startup-${LOG_TIMESTAMP}.log"
    if gcloud storage cp "$STARTUP_LOG" "$STARTUP_LOG_GCS" --quiet 2>/dev/null; then
        log "Startup log uploaded to: $STARTUP_LOG_GCS"
    else
        log "WARNING: Failed to upload startup log"
    fi
fi

# Upload pipeline log
if [[ -f "$PIPELINE_LOG" ]]; then
    PIPELINE_LOG_GCS="gs://$GCS_BUCKET/logs/etl-${LOG_TIMESTAMP}.log"
    if gcloud storage cp "$PIPELINE_LOG" "$PIPELINE_LOG_GCS" --quiet 2>/dev/null; then
        log "Pipeline log uploaded to: $PIPELINE_LOG_GCS"
    else
        log "WARNING: Failed to upload pipeline log"
    fi
fi

log "=========================================="
log "Startup script completed"
log "Exit status: $EXIT_STATUS"
log "=========================================="

# Shut down the VM (the orchestrator will delete it)
log "Shutting down VM..."
shutdown -h now

exit $EXIT_STATUS
