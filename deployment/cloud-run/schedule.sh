#!/bin/bash
# Setup Cloud Scheduler for Weekly ETL Updates
#
# This script creates a Cloud Scheduler job that triggers the Cloud Run Job
# on a weekly schedule (default: Sundays at 2 AM).
#
# Usage:
#   ./deployment/cloud-run/schedule.sh [--schedule CRON] [--dry-run]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse command line arguments
DRY_RUN=false
CUSTOM_SCHEDULE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --schedule)
            CUSTOM_SCHEDULE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            export DRY_RUN
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--schedule CRON] [--dry-run]"
            exit 1
            ;;
    esac
done

# Load GCP configuration
load_gcp_config

# Use custom schedule if provided, otherwise use default
SCHEDULE="${CUSTOM_SCHEDULE:-$SCHEDULE_CRON}"

# Check prerequisites
log_step "Checking prerequisites"
check_gcp_auth
check_gcp_project "$PROJECT_ID"
check_command gcloud

# Enable Cloud Scheduler API
enable_gcp_api "cloudscheduler.googleapis.com" "Cloud Scheduler API"

# Build Cloud Run Job execution URL
JOB_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/jobs/${CLOUD_RUN_JOB_NAME}:run"

# Build request body for incremental update
# This passes --scraper-limit 20 to limit scraping to recent publications
# Note: Uses config.production.yaml if available, falls back to config.yaml
REQUEST_BODY=$(cat <<EOF
{
  "overrides": {
    "containerOverrides": [{
      "args": [
        "--config",
        "backend/etl/config.production.yaml",
        "--scraper-limit",
        "20",
        "--log-level",
        "INFO"
      ]
    }]
  }
}
EOF
)

# Check if scheduler job already exists
log_step "Checking if scheduler job exists"
if gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" --location="$REGION" &> /dev/null; then
    log_warning "Scheduler job '$SCHEDULER_JOB_NAME' already exists"
    if ! confirm_action "Do you want to update it?" "y"; then
        log_info "Exiting..."
        exit 0
    fi

    # Update existing job
    log_step "Updating Cloud Scheduler job"
    log_info "Schedule: $SCHEDULE"
    log_info "Timezone: $SCHEDULE_TIMEZONE"

    if is_dry_run; then
        log_info "[DRY RUN] Would update scheduler job"
    else
        if gcloud scheduler jobs update http "$SCHEDULER_JOB_NAME" \
            --location="$REGION" \
            --schedule="$SCHEDULE" \
            --uri="$JOB_URL" \
            --http-method=POST \
            --oauth-service-account-email="$SA_EMAIL" \
            --message-body="$REQUEST_BODY" \
            --time-zone="$SCHEDULE_TIMEZONE" \
            --quiet; then
            log_success "Scheduler job updated successfully"
        else
            error_exit "Failed to update scheduler job"
        fi
    fi
else
    # Create new job
    log_step "Creating Cloud Scheduler job"
    log_info "Job name: $SCHEDULER_JOB_NAME"
    log_info "Schedule: $SCHEDULE"
    log_info "Timezone: $SCHEDULE_TIMEZONE"
    log_info "Target: $CLOUD_RUN_JOB_NAME"

    if is_dry_run; then
        log_info "[DRY RUN] Would create scheduler job"
    else
        if gcloud scheduler jobs create http "$SCHEDULER_JOB_NAME" \
            --location="$REGION" \
            --schedule="$SCHEDULE" \
            --uri="$JOB_URL" \
            --http-method=POST \
            --oauth-service-account-email="$SA_EMAIL" \
            --message-body="$REQUEST_BODY" \
            --time-zone="$SCHEDULE_TIMEZONE" \
            --quiet; then
            log_success "Scheduler job created successfully"
        else
            error_exit "Failed to create scheduler job"
        fi
    fi
fi

# Summary
print_summary "Cloud Scheduler Setup Complete" \
    "Scheduler Job: $SCHEDULER_JOB_NAME" \
    "Schedule: $SCHEDULE" \
    "Timezone: $SCHEDULE_TIMEZONE" \
    "Target Job: $CLOUD_RUN_JOB_NAME" \
    "Mode: Incremental update (--scraper-limit 20)"

log_success "Cloud Scheduler setup completed!"
log_info "The ETL pipeline will run automatically on schedule."
log_info "To test the schedule manually:"
log_info "  ./deployment/cloud-run/execute.sh"
