#!/bin/bash
# Deploy Cloud Run Job for ETL Pipeline
#
# This script builds the container image, pushes it to Artifact Registry,
# and creates/updates the Cloud Run Job.
#
# Usage:
#   ./deployment/cloud-run/deploy.sh [--dry-run]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

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
check_command docker

# Enable Artifact Registry API
enable_gcp_api "artifactregistry.googleapis.com" "Artifact Registry API"

# Create Artifact Registry repository if it doesn't exist
log_step "Setting up Artifact Registry"
if is_dry_run; then
    log_info "[DRY RUN] Would check/create Artifact Registry repository: $ARTIFACT_REGISTRY_REPO"
else
    if gcloud artifacts repositories describe "$ARTIFACT_REGISTRY_REPO" \
        --location="$REGION" &> /dev/null; then
        log_info "Artifact Registry repository already exists"
    else
        log_info "Creating Artifact Registry repository..."
        if gcloud artifacts repositories create "$ARTIFACT_REGISTRY_REPO" \
            --repository-format=docker \
            --location="$REGION" \
            --description="ETL pipeline container images" \
            --quiet; then
            log_success "Artifact Registry repository created"
        else
            error_exit "Failed to create Artifact Registry repository"
        fi
    fi
fi

# Configure Docker authentication
log_step "Configuring Docker authentication"
if is_dry_run; then
    log_info "[DRY RUN] Would configure Docker authentication"
else
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    log_success "Docker authentication configured"
fi

# Build container image
log_step "Building container image"
log_info "Image: $IMAGE_NAME"
log_info "Context: $PROJECT_ROOT"

if is_dry_run; then
    log_info "[DRY RUN] Would build Docker image"
else
    log_info "Building Docker image (this may take several minutes)..."
    cd "$PROJECT_ROOT"

    if docker build \
        -f "$SCRIPT_DIR/Dockerfile" \
        -t "$IMAGE_NAME" \
        .; then
        log_success "Docker image built successfully"
    else
        error_exit "Failed to build Docker image"
    fi
fi

# Push image to Artifact Registry
log_step "Pushing image to Artifact Registry"
if is_dry_run; then
    log_info "[DRY RUN] Would push image to: $IMAGE_NAME"
else
    log_info "Pushing image (this may take several minutes)..."
    if docker push "$IMAGE_NAME"; then
        log_success "Image pushed successfully"
    else
        error_exit "Failed to push image"
    fi
fi

# Create or update Cloud Run Job
log_step "Deploying Cloud Run Job"
log_info "Job name: $CLOUD_RUN_JOB_NAME"
log_info "Memory: $CLOUD_RUN_MEMORY"
log_info "CPU: $CLOUD_RUN_CPU"
log_info "Timeout: $CLOUD_RUN_TIMEOUT"

if is_dry_run; then
    log_info "[DRY RUN] Would create/update Cloud Run Job"
else
    # Check if job already exists
    if gcloud run jobs describe "$CLOUD_RUN_JOB_NAME" --region="$REGION" &> /dev/null; then
        log_info "Cloud Run Job already exists, updating..."
        UPDATE_CMD=(
            gcloud run jobs update "$CLOUD_RUN_JOB_NAME"
            --region="$REGION"
            --image="$IMAGE_NAME"
            --memory="$CLOUD_RUN_MEMORY"
            --cpu="$CLOUD_RUN_CPU"
            --task-timeout="$CLOUD_RUN_TIMEOUT"
            --max-retries="$CLOUD_RUN_MAX_RETRIES"
            --service-account="$SA_EMAIL"
            --set-env-vars="GCS_BUCKET=$BUCKET_NAME,ENVIRONMENT=production"
            --set-secrets="OPENAI_API_KEY=$OPENAI_SECRET_NAME:latest"
        )

        if "${UPDATE_CMD[@]}"; then
            log_success "Cloud Run Job updated successfully"
        else
            error_exit "Failed to update Cloud Run Job"
        fi
    else
        log_info "Creating Cloud Run Job..."
        CREATE_CMD=(
            gcloud run jobs create "$CLOUD_RUN_JOB_NAME"
            --region="$REGION"
            --image="$IMAGE_NAME"
            --memory="$CLOUD_RUN_MEMORY"
            --cpu="$CLOUD_RUN_CPU"
            --task-timeout="$CLOUD_RUN_TIMEOUT"
            --max-retries="$CLOUD_RUN_MAX_RETRIES"
            --service-account="$SA_EMAIL"
            --set-env-vars="GCS_BUCKET=$BUCKET_NAME,ENVIRONMENT=production"
            --set-secrets="OPENAI_API_KEY=$OPENAI_SECRET_NAME:latest"
        )

        if "${CREATE_CMD[@]}"; then
            log_success "Cloud Run Job created successfully"
        else
            error_exit "Failed to create Cloud Run Job"
        fi
    fi
fi

# Summary
print_summary "Cloud Run Job Deployment Complete" \
    "Job Name: $CLOUD_RUN_JOB_NAME" \
    "Image: $IMAGE_NAME" \
    "Region: $REGION" \
    "Memory: $CLOUD_RUN_MEMORY" \
    "CPU: $CLOUD_RUN_CPU"

log_success "Cloud Run Job deployment completed!"
log_info "Next steps:"
log_info "  • Schedule weekly updates: ./deployment/cloud-run/schedule.sh"
log_info "  • Execute manually: ./deployment/cloud-run/execute.sh"
