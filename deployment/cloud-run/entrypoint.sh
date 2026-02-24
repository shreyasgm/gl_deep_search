#!/bin/bash
# Container Entrypoint Script for ETL Pipeline
#
# This script runs inside the container and handles:
# 1. Fetching secrets from Google Secret Manager
# 2. Setting up environment variables
# 3. Syncing the tracking DB from/to GCS
# 4. Launching the ETL pipeline with provided arguments
#
# Environment variables expected:
#   - GOOGLE_CLOUD_PROJECT: GCP project ID (required)
#   - GCS_BUCKET: Cloud Storage bucket name (required)
#   - ENVIRONMENT: Environment name (default: production)
#   - OPENAI_API_KEY: Can be pre-set (Cloud Run) or fetched from Secret Manager (VM)

set -e

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "=========================================="
log "ETL Pipeline Container Starting"
log "=========================================="

# Validate required environment variables
if [[ -z "${GOOGLE_CLOUD_PROJECT}" ]]; then
    log "ERROR: GOOGLE_CLOUD_PROJECT environment variable not set"
    exit 1
fi

if [[ -z "${GCS_BUCKET}" ]]; then
    log "ERROR: GCS_BUCKET environment variable not set"
    exit 1
fi

log "Configuration:"
log "  Project: ${GOOGLE_CLOUD_PROJECT}"
log "  GCS Bucket: ${GCS_BUCKET}"
log "  Environment: ${ENVIRONMENT:-production}"

# Fetch OpenAI API key from Secret Manager if not already set
# Cloud Run sets this via --set-secrets, but VMs need to fetch it
if [[ -z "${OPENAI_API_KEY}" ]]; then
    log "Fetching OpenAI API key from Secret Manager..."

    # Try to fetch the secret (will use VM's service account credentials)
    if OPENAI_API_KEY=$(gcloud secrets versions access latest \
        --secret="openai-api-key" \
        --project="${GOOGLE_CLOUD_PROJECT}" 2>&1); then
        export OPENAI_API_KEY
        log "Successfully retrieved OpenAI API key"
    else
        log "ERROR: Failed to retrieve OpenAI API key from Secret Manager"
        log "Error: $OPENAI_API_KEY"
        exit 1
    fi
else
    log "OpenAI API key already set (likely Cloud Run environment)"
fi

# Set environment variables for the ETL pipeline
export GCS_BUCKET="${GCS_BUCKET}"
export ENVIRONMENT="${ENVIRONMENT:-production}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}"

# Tell StorageFactory to use CloudStorage
export CLOUD_ENVIRONMENT="gcp"

# Additional environment setup
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# ------------------------------------------------------------------
# Sync tracking DB from GCS (lightweight — only the manifest DB)
# Raw/processed data is downloaded on-demand by each ETL component.
# ------------------------------------------------------------------
log "Syncing tracking DB from GCS..."
mkdir -p /app/data
gcloud storage cp "gs://${GCS_BUCKET}/etl_tracking.db" "/app/data/etl_tracking.db" 2>/dev/null \
    && log "  Tracking DB restored from GCS" \
    || log "  No existing tracking DB in GCS (starting fresh)"

log "=========================================="
log "Starting ETL Pipeline"
log "=========================================="
log "Command: python -m backend.etl.orchestrator $*"

# Execute the ETL pipeline — capture exit code instead of exec so we
# can upload the tracking DB afterwards.
EXIT_CODE=0
python -m backend.etl.orchestrator "$@" || EXIT_CODE=$?

# ------------------------------------------------------------------
# Upload tracking DB back to GCS
# ------------------------------------------------------------------
log "Uploading tracking DB to GCS..."
if [[ -f "/app/data/etl_tracking.db" ]]; then
    gcloud storage cp "/app/data/etl_tracking.db" "gs://${GCS_BUCKET}/etl_tracking.db" \
        && log "  Tracking DB uploaded to GCS" \
        || log "WARNING: Failed to upload tracking DB"
fi

log "Pipeline exited with code: $EXIT_CODE"
exit $EXIT_CODE
