#!/bin/bash
# Secret Manager Setup Script
#
# This script creates secrets in GCP Secret Manager and configures access.
# It will prompt for secret values interactively to avoid storing them in files.
#
# Usage:
#   ./deployment/scripts/03-setup-secrets.sh [--dry-run]

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

# Enable Secret Manager API if not already enabled
enable_gcp_api "secretmanager.googleapis.com" "Secret Manager API"

# Function to create or update a secret
create_secret() {
    local secret_name="$1"
    local secret_description="$2"
    local prompt_message="$3"

    log_step "Setting up secret: $secret_name"

    # Check if secret already exists
    if resource_exists "secret" "$secret_name"; then
        log_warning "Secret '$secret_name' already exists"
        if ! confirm_action "Do you want to update this secret?" "n"; then
            log_info "Skipping secret: $secret_name"
            return 0
        fi
    fi

    # Prompt for secret value
    echo -e "${YELLOW}$prompt_message${NC}" >&2
    echo -e "${YELLOW}(Input will be hidden)${NC}" >&2
    read -rs secret_value
    echo "" >&2

    if [[ -z "$secret_value" ]]; then
        log_warning "Empty secret value provided. Skipping..."
        return 0
    fi

    if is_dry_run; then
        log_info "[DRY RUN] Would create/update secret: $secret_name"
        return 0
    fi

    # Create secret if it doesn't exist
    if ! resource_exists "secret" "$secret_name"; then
        log_info "Creating secret: $secret_name"
        if gcloud secrets create "$secret_name" \
            --replication-policy="automatic" \
            --data-file=- <<< "$secret_value" 2>/dev/null; then
            log_success "Secret created successfully"
        else
            # Try with description if creation failed
            if gcloud secrets create "$secret_name" \
                --replication-policy="automatic" \
                --data-file=- <<< "$secret_value" 2>/dev/null; then
                log_success "Secret created successfully"
            else
                log_error "Failed to create secret. It may already exist."
                return 1
            fi
        fi
    else
        # Add new version to existing secret
        log_info "Adding new version to existing secret: $secret_name"
        if echo -n "$secret_value" | gcloud secrets versions add "$secret_name" \
            --data-file=- --quiet; then
            log_success "Secret version added successfully"
        else
            log_error "Failed to add secret version"
            return 1
        fi
    fi

    # Set description if provided
    if [[ -n "$secret_description" ]]; then
        gcloud secrets update "$secret_name" \
            --update-labels="description=$secret_description" \
            --quiet 2>/dev/null || true
    fi
}

# Create OpenAI API key secret
create_secret \
    "$OPENAI_SECRET_NAME" \
    "OpenAI API key for embeddings generation" \
    "Enter your OpenAI API key:"

# Summary
print_summary "Secrets Setup Complete" \
    "OpenAI Secret: $OPENAI_SECRET_NAME"

log_success "Secrets setup completed successfully!"
log_info "Next step: Run ./deployment/scripts/04-create-service-account.sh"
log_warning "Note: Service account access to secrets will be configured in the next step"
