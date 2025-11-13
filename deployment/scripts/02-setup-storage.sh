#!/bin/bash
# GCS Storage Setup Script
#
# This script creates Cloud Storage buckets and applies lifecycle policies.
#
# Usage:
#   ./deployment/scripts/02-setup-storage.sh [--dry-run]

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
check_gcp_project "$PROJECT_ID"
check_command gcloud

# Validate bucket name
log_step "Validating bucket name"
validate_bucket_name "$BUCKET_NAME"

# Check if bucket already exists
log_step "Checking if bucket exists"
if resource_exists "bucket" "$BUCKET_NAME"; then
    log_warning "Bucket 'gs://$BUCKET_NAME' already exists"
    if ! confirm_action "Do you want to continue with existing bucket?" "y"; then
        log_info "Exiting..."
        exit 0
    fi
    # Try to enable Autoclass on existing bucket (may fail if bucket has incompatible settings)
    log_step "Enabling Autoclass on existing bucket"
    if is_dry_run; then
        log_info "[DRY RUN] Would enable Autoclass on existing bucket"
    else
        log_info "Attempting to enable Autoclass..."
        if gcloud storage buckets update "gs://$BUCKET_NAME" \
            --autoclass \
            --quiet 2>/dev/null; then
            log_success "Autoclass enabled on existing bucket"
        else
            log_warning "Could not enable Autoclass on existing bucket (may require bucket recreation)"
            log_info "Autoclass can only be enabled on buckets without certain lifecycle rules or configurations"
        fi
    fi
else
    # Create bucket
    log_step "Creating Cloud Storage bucket"
    log_info "Bucket name: $BUCKET_NAME"
    log_info "Region: $REGION"
    log_info "Storage class: $STORAGE_CLASS"

    if is_dry_run; then
        log_info "[DRY RUN] Would create bucket: gs://$BUCKET_NAME"
    else
        if gcloud storage buckets create "gs://$BUCKET_NAME" \
            --location="$REGION" \
            --default-storage-class="$STORAGE_CLASS" \
            --autoclass \
            --uniform-bucket-level-access \
            --public-access-prevention=enforced \
            --quiet; then
            log_success "Bucket created successfully with Autoclass enabled"
        else
            error_exit "Failed to create bucket. It may already exist or the name is taken."
        fi
    fi
fi

# Apply lifecycle policy
log_step "Applying lifecycle policy"
LIFECYCLE_POLICY_FILE="${SCRIPT_DIR}/../config/lifecycle-policy.json"

if [[ ! -f "$LIFECYCLE_POLICY_FILE" ]]; then
    log_warning "Lifecycle policy file not found: $LIFECYCLE_POLICY_FILE"
    log_info "Skipping lifecycle policy application"
else
    if is_dry_run; then
        log_info "[DRY RUN] Would apply lifecycle policy from: $LIFECYCLE_POLICY_FILE"
    else
        log_info "Applying lifecycle policy..."
        if gcloud storage buckets update "gs://$BUCKET_NAME" \
            --lifecycle-file="$LIFECYCLE_POLICY_FILE" \
            --quiet; then
            log_success "Lifecycle policy applied successfully"
        else
            log_warning "Failed to apply lifecycle policy (non-critical)"
        fi
    fi
fi

# Enable versioning (optional but recommended)
log_step "Configuring bucket versioning"
if is_dry_run; then
    log_info "[DRY RUN] Would enable versioning on bucket"
else
    log_info "Enabling versioning for data protection..."
    if gcloud storage buckets update "gs://$BUCKET_NAME" \
        --versioning \
        --quiet; then
        log_success "Versioning enabled"
    else
        log_warning "Failed to enable versioning (non-critical)"
    fi
fi

# Create initial folder structure (optional - folders are created automatically)
log_step "Creating initial folder structure"
FOLDERS=(
    "raw/documents/growthlab"
    "raw/documents/openalex"
    "processed/documents"
    "processed/chunks"
    "processed/embeddings"
    "intermediate"
    "logs"
    "reports"
)

if is_dry_run; then
    log_info "[DRY RUN] Would create folder structure"
else
    for folder in "${FOLDERS[@]}"; do
        # Create placeholder file to ensure folder exists
        echo "# Placeholder file" | gcloud storage cp - "gs://$BUCKET_NAME/$folder/.gitkeep" --quiet 2>/dev/null || true
    done
    log_success "Initial folder structure created"
fi

# Summary
print_summary "Storage Setup Complete" \
    "Bucket: gs://$BUCKET_NAME" \
    "Region: $REGION" \
    "Storage Class: $STORAGE_CLASS" \
    "Autoclass: Enabled (handles storage class transitions)" \
    "Lifecycle Policy: Applied (log deletion only)" \
    "Versioning: Enabled"

log_success "Storage setup completed successfully!"
log_info "Next step: Run ./deployment/scripts/03-setup-secrets.sh"
