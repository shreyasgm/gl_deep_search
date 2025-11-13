#!/bin/bash
# GCP Project Setup Script
#
# This script creates a new GCP project, enables required APIs, and links billing.
# Run this script first before any other deployment scripts.
#
# Usage:
#   ./deployment/scripts/01-setup-gcp-project.sh [--dry-run]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Parse command line arguments
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    export DRY_RUN
fi

# Load GCP configuration
load_gcp_config

# Check prerequisites
log_step "Checking prerequisites"
check_gcp_auth
check_command gcloud

# Validate configuration
log_step "Validating configuration"
validate_project_id "$PROJECT_ID"

if [[ -z "$PROJECT_NAME" ]]; then
    error_exit "PROJECT_NAME is not set in gcp-config.sh"
fi

if [[ -z "$BILLING_ACCOUNT_ID" ]] || [[ "$BILLING_ACCOUNT_ID" == "your-billing-account-id" ]]; then
    error_exit "BILLING_ACCOUNT_ID must be set in gcp-config.sh"
fi

# Check if project already exists
log_step "Checking if project exists"
if resource_exists "project" "$PROJECT_ID"; then
    log_warning "Project '$PROJECT_ID' already exists"
    if ! confirm_action "Do you want to continue with existing project?" "y"; then
        log_info "Exiting..."
        exit 0
    fi
    gcloud config set project "$PROJECT_ID" --quiet
else
    log_info "Creating new GCP project: $PROJECT_ID"

    if is_dry_run; then
        log_info "[DRY RUN] Would create project: $PROJECT_ID"
    else
        if gcloud projects create "$PROJECT_ID" --name="$PROJECT_NAME" --quiet; then
            log_success "Project created successfully"
        else
            error_exit "Failed to create project. It may already exist or the ID is taken."
        fi

        # Set as default project
        gcloud config set project "$PROJECT_ID" --quiet
    fi
fi

# Link billing account
log_step "Linking billing account"
if is_dry_run; then
    log_info "[DRY RUN] Would link billing account: $BILLING_ACCOUNT_ID"
else
    # Check if billing is already linked
    local current_billing
    current_billing=$(gcloud billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "")

    if [[ -n "$current_billing" ]]; then
        log_info "Billing account is already linked: $current_billing"
    else
        log_info "Linking billing account: $BILLING_ACCOUNT_ID"
        if gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID" --quiet; then
            log_success "Billing account linked successfully"
        else
            error_exit "Failed to link billing account. Please verify the billing account ID."
        fi
    fi
fi

# Enable required APIs
log_step "Enabling required GCP APIs"

APIS=(
    "compute.googleapis.com:Compute Engine API"
    "storage.googleapis.com:Cloud Storage API"
    "secretmanager.googleapis.com:Secret Manager API"
    "logging.googleapis.com:Cloud Logging API"
    "monitoring.googleapis.com:Cloud Monitoring API"
    "cloudscheduler.googleapis.com:Cloud Scheduler API"
    "run.googleapis.com:Cloud Run API"
    "artifactregistry.googleapis.com:Artifact Registry API"
    "cloudresourcemanager.googleapis.com:Cloud Resource Manager API"
)

for api_entry in "${APIS[@]}"; do
    IFS=':' read -r api_name api_display <<< "$api_entry"
    enable_gcp_api "$api_name" "$api_display"
done

# Set default region
log_step "Setting default region"
if is_dry_run; then
    log_info "[DRY RUN] Would set default region: $REGION"
else
    gcloud config set compute/region "$REGION" --quiet
    gcloud config set compute/zone "$ZONE" --quiet
    log_success "Default region set to $REGION, zone set to $ZONE"
fi

# Summary
print_summary "GCP Project Setup Complete" \
    "Project ID: $PROJECT_ID" \
    "Project Name: $PROJECT_NAME" \
    "Region: $REGION" \
    "Zone: $ZONE" \
    "Billing Account: $BILLING_ACCOUNT_ID"

log_success "GCP project setup completed successfully!"
log_info "Next step: Run ./deployment/scripts/02-setup-storage.sh"
