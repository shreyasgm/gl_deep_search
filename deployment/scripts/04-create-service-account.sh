#!/bin/bash
# Service Account Setup Script
#
# This script creates a service account for the ETL pipeline and grants
# necessary permissions for GCS, Secret Manager, and logging.
#
# Usage:
#   ./deployment/scripts/04-create-service-account.sh [--dry-run]

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

# Create service account
log_step "Creating service account"
log_info "Service account name: $SA_NAME"
log_info "Service account email: $SA_EMAIL"

if resource_exists "service-account" "$SA_EMAIL"; then
    log_warning "Service account '$SA_EMAIL' already exists"
    if ! confirm_action "Do you want to continue with existing service account?" "y"; then
        log_info "Exiting..."
        exit 0
    fi
else
    if is_dry_run; then
        log_info "[DRY RUN] Would create service account: $SA_NAME"
    else
        log_info "Creating service account..."
        if gcloud iam service-accounts create "$SA_NAME" \
            --display-name="ETL Pipeline Service Account" \
            --description="Service account for Growth Lab Deep Search ETL pipeline" \
            --quiet; then
            log_success "Service account created successfully"
        else
            error_exit "Failed to create service account"
        fi
    fi
fi

# Grant IAM roles
log_step "Granting IAM roles to service account"

ROLES=(
    "roles/storage.objectAdmin:Full control over Cloud Storage objects"
    "roles/secretmanager.secretAccessor:Access to secrets in Secret Manager"
    "roles/logging.logWriter:Write logs to Cloud Logging"
    "roles/monitoring.metricWriter:Write metrics to Cloud Monitoring"
)

for role_entry in "${ROLES[@]}"; do
    IFS=':' read -r role role_description <<< "$role_entry"

    log_info "Granting role: $role"
    log_info "  Description: $role_description"

    if is_dry_run; then
        log_info "[DRY RUN] Would grant role $role to $SA_EMAIL"
    else
        # Check if role is already granted
        if gcloud projects get-iam-policy "$PROJECT_ID" \
            --flatten="bindings[].members" \
            --filter="bindings.members:serviceAccount:${SA_EMAIL} AND bindings.role:${role}" \
            --format="value(bindings.role)" | grep -q "^${role}$"; then
            log_info "  Role already granted, skipping..."
        else
            if gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                --member="serviceAccount:${SA_EMAIL}" \
                --role="$role" \
                --quiet; then
                log_success "  Role granted successfully"
            else
                log_error "  Failed to grant role"
            fi
        fi
    fi
done

# Grant access to specific secrets
log_step "Granting secret access"
log_info "Granting access to secret: $OPENAI_SECRET_NAME"

if is_dry_run; then
    log_info "[DRY RUN] Would grant secret access"
else
    # Check if access is already granted
    if gcloud secrets get-iam-policy "$OPENAI_SECRET_NAME" \
        --flatten="bindings[].members" \
        --filter="bindings.members:serviceAccount:${SA_EMAIL}" \
        --format="value(bindings.members)" 2>/dev/null | grep -q "${SA_EMAIL}"; then
        log_info "Secret access already granted"
    else
        log_info "Granting secret access..."

        # Try the standard approach first
        if gcloud secrets add-iam-policy-binding "$OPENAI_SECRET_NAME" \
            --member="serviceAccount:${SA_EMAIL}" \
            --role="roles/secretmanager.secretAccessor" \
            --quiet 2>/dev/null; then
            log_success "Secret access granted successfully"
        else
            log_warning "Failed to grant secret access via CLI (this may be a gcloud bug)"
            log_info "You can grant access manually via Cloud Console:"
            log_info "  1. Go to Secret Manager: https://console.cloud.google.com/security/secret-manager"
            log_info "  2. Click on '$OPENAI_SECRET_NAME'"
            log_info "  3. Click 'PERMISSIONS' tab"
            log_info "  4. Click 'GRANT ACCESS'"
            log_info "  5. Add '$SA_EMAIL' with role 'Secret Manager Secret Accessor'"

            if confirm_action "Do you want to try alternative method (grants access to all secrets)?" "n"; then
                # Alternative: grant at project level (less secure but works)
                log_warning "Granting project-level secret access (less secure)"
                gcloud projects add-iam-policy-binding "$PROJECT_ID" \
                    --member="serviceAccount:${SA_EMAIL}" \
                    --role="roles/secretmanager.secretAccessor" \
                    --quiet || log_error "Alternative method also failed"
            fi
        fi
    fi
fi

# Summary
print_summary "Service Account Setup Complete" \
    "Service Account: $SA_NAME" \
    "Email: $SA_EMAIL" \
    "Roles Granted: Storage Admin, Secret Accessor, Log Writer, Metric Writer" \
    "Secret Access: $OPENAI_SECRET_NAME"

log_success "Service account setup completed successfully!"
log_info "Setup complete! You can now proceed with deployment:"
log_info "  • For initial batch: ./deployment/vm/create-vm.sh"
log_info "  • For Cloud Run: ./deployment/cloud-run/deploy.sh"
