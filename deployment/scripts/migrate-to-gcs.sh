#!/bin/bash
# One-time migration script: upload SLURM ETL outputs to GCS.
#
# Run this from the SLURM cluster after the initial batch processing
# is complete, so that Cloud Run incremental jobs can pick up where
# SLURM left off.
#
# Usage:
#   bash deployment/scripts/migrate-to-gcs.sh [--include-raw]
#
# By default only the tracking DB, processed data, and intermediate
# data are uploaded.  Pass --include-raw to also upload the raw PDFs
# (~3 GB).

set -euo pipefail

# Defaults
GCS_BUCKET="${GCS_BUCKET:-gl-deep-search-data}"
DATA_DIR="${DATA_DIR:-data}"
INCLUDE_RAW=false

for arg in "$@"; do
    case "$arg" in
        --include-raw) INCLUDE_RAW=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "=========================================="
log "Migrating ETL outputs to GCS"
log "  GCS Bucket : gs://${GCS_BUCKET}"
log "  Data dir   : ${DATA_DIR}"
log "  Include raw: ${INCLUDE_RAW}"
log "=========================================="

# 1. Upload tracking DB
if [[ -f "${DATA_DIR}/etl_tracking.db" ]]; then
    log "Uploading tracking DB..."
    gcloud storage cp "${DATA_DIR}/etl_tracking.db" "gs://${GCS_BUCKET}/etl_tracking.db"
    log "  Done."
else
    log "WARNING: No tracking DB found at ${DATA_DIR}/etl_tracking.db"
fi

# 2. Upload processed data (text, chunks, embeddings)
if [[ -d "${DATA_DIR}/processed" ]]; then
    log "Uploading processed/ ..."
    gcloud storage rsync -r "${DATA_DIR}/processed" "gs://${GCS_BUCKET}/processed" --quiet
    log "  Done."
else
    log "WARNING: No processed/ directory found"
fi

# 3. Upload intermediate data (CSV, etc.)
if [[ -d "${DATA_DIR}/intermediate" ]]; then
    log "Uploading intermediate/ ..."
    gcloud storage rsync -r "${DATA_DIR}/intermediate" "gs://${GCS_BUCKET}/intermediate" --quiet
    log "  Done."
else
    log "WARNING: No intermediate/ directory found"
fi

# 4. Optionally upload raw data
if [[ "$INCLUDE_RAW" == "true" ]]; then
    if [[ -d "${DATA_DIR}/raw" ]]; then
        log "Uploading raw/ (this may take a while)..."
        gcloud storage rsync -r "${DATA_DIR}/raw" "gs://${GCS_BUCKET}/raw" --quiet
        log "  Done."
    else
        log "WARNING: No raw/ directory found"
    fi
else
    log "Skipping raw/ upload (pass --include-raw to include)"
fi

log "=========================================="
log "Migration complete!"
log "=========================================="
