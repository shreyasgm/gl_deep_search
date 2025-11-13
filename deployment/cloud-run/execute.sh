#!/bin/bash
# Execute Cloud Run Job Manually
#
# This script manually triggers the Cloud Run Job execution.
# Useful for testing or running ad-hoc updates.
#
# Usage:
#   ./deployment/cloud-run/execute.sh [--full] [--scraper-limit N] [--wait]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse command line arguments
FULL_RUN=false
SCRAPER_LIMIT="20"
WAIT_FOR_COMPLETION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_RUN=true
            shift
            ;;
        --scraper-limit)
            SCRAPER_LIMIT="$2"
            shift 2
            ;;
        --wait)
            WAIT_FOR_COMPLETION=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--full] [--scraper-limit N] [--wait]"
            echo ""
            echo "Options:"
            echo "  --full              Run full pipeline (no scraper limit)"
            echo "  --scraper-limit N   Limit scraper to N publications (default: 20)"
            echo "  --wait              Wait for job completion and show logs"
            exit 1
            ;;
    esac
done

# Load GCP configuration
load_gcp_config

# Check prerequisites
log_step "Checking prerequisites"
check_gcp_auth
check_gcp_project "$PROJECT_ID"
check_command gcloud

# Check if Cloud Run Job exists
log_step "Checking Cloud Run Job"
if ! gcloud run jobs describe "$CLOUD_RUN_JOB_NAME" --region="$REGION" &> /dev/null; then
    error_exit "Cloud Run Job '$CLOUD_RUN_JOB_NAME' not found in region $REGION.
Please deploy it first: ./deployment/cloud-run/deploy.sh"
fi

# Build execution arguments
if [[ "$FULL_RUN" == "true" ]]; then
    EXEC_ARGS=(
        "--config"
        "backend/etl/config.production.yaml"
        "--log-level"
        "INFO"
    )
    log_info "Mode: Full pipeline run"
else
    EXEC_ARGS=(
        "--config"
        "backend/etl/config.production.yaml"
        "--scraper-limit"
        "$SCRAPER_LIMIT"
        "--log-level"
        "INFO"
    )
    log_info "Mode: Incremental update (scraper limit: $SCRAPER_LIMIT)"
fi

# Execute Cloud Run Job
log_step "Executing Cloud Run Job"
log_info "Job name: $CLOUD_RUN_JOB_NAME"
log_info "Region: $REGION"
log_info "Arguments: ${EXEC_ARGS[*]}"

EXECUTION_CMD=(
    gcloud run jobs execute "$CLOUD_RUN_JOB_NAME"
    --region="$REGION"
    --args="${EXEC_ARGS[*]}"
)

if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
    EXECUTION_CMD+=(--wait)
    log_info "Will wait for completion..."
fi

log_info "Executing job..."
if "${EXECUTION_CMD[@]}"; then
    log_success "Job execution started successfully"

    if [[ "$WAIT_FOR_COMPLETION" == "true" ]]; then
        log_info "Job completed!"
    else
        log_info "Job is running in the background"
        log_info "To view logs:"
        log_info "  gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=$CLOUD_RUN_JOB_NAME\" --limit 100"
        log_info ""
        log_info "To check execution status:"
        log_info "  gcloud run jobs executions list --job=$CLOUD_RUN_JOB_NAME --region=$REGION"
    fi
else
    error_exit "Failed to execute Cloud Run Job"
fi

# Get execution details if available
if [[ "$WAIT_FOR_COMPLETION" == "false" ]]; then
    log_step "Recent executions"
    gcloud run jobs executions list \
        --job="$CLOUD_RUN_JOB_NAME" \
        --region="$REGION" \
        --limit=5 \
        --format="table(name,status.conditions[0].type,status.conditions[0].status,status.startTime)"
fi
