#!/bin/bash
# Deploy Cloud Run Job for ETL Pipeline
#
# This script builds the container image, pushes it to Artifact Registry,
# and creates/updates the Cloud Run Job.
#
# Usage:
#   ./deployment/cloud-run/deploy.sh [OPTIONS]
#
# Options:
#   --dry-run           Show what would be done without actually doing it
#   --skip-build        Skip building, use existing image from registry
#   --cloud-build       Use Cloud Build instead of local Docker build
#   --local-build       Use local Docker build (default)

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse command line arguments
DRY_RUN=false
SKIP_BUILD=false
USE_CLOUD_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            export DRY_RUN
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --cloud-build)
            USE_CLOUD_BUILD=true
            shift
            ;;
        --local-build)
            USE_CLOUD_BUILD=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run           Show what would be done without actually doing it"
            echo "  --skip-build        Skip building, use existing image from registry"
            echo "  --cloud-build       Use Cloud Build instead of local Docker build"
            echo "  --local-build       Use local Docker build (default)"
            echo ""
            echo "Examples:"
            echo "  $0                           # Local build and deploy"
            echo "  $0 --cloud-build             # Cloud Build and deploy"
            echo "  $0 --skip-build              # Deploy using existing image"
            echo "  $0 --dry-run --cloud-build   # Show what Cloud Build would do"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
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

# Build and push container image (unless --skip-build)
if [[ "$SKIP_BUILD" == true ]]; then
    log_step "Skipping build (using existing image)"
    log_info "Image: $IMAGE_NAME"
    log_info "Verifying image exists in registry..."

    if is_dry_run; then
        log_info "[DRY RUN] Would verify image exists: $IMAGE_NAME"
    else
        # Verify the image exists
        if gcloud artifacts docker images describe "$IMAGE_NAME" &> /dev/null; then
            log_success "Image found in registry"
        else
            error_exit "Image not found in registry: $IMAGE_NAME. Build the image first or use --cloud-build or --local-build"
        fi
    fi
elif [[ "$USE_CLOUD_BUILD" == true ]]; then
    # Use Cloud Build
    log_step "Building container image with Cloud Build"
    log_info "Image: $IMAGE_NAME"
    log_info "Config: $PROJECT_ROOT/deployment/cloudbuild.yaml"
    log_info "Platform: linux/amd64 (required for Cloud Run)"

    if is_dry_run; then
        log_info "[DRY RUN] Would submit Cloud Build"
    else
        cd "$PROJECT_ROOT"

        # Check if cloudbuild.yaml exists
        if [[ ! -f "deployment/cloudbuild.yaml" ]]; then
            error_exit "deployment/cloudbuild.yaml not found. Create it first."
        fi

        log_info "Submitting to Cloud Build (this may take 10-15 minutes)..."
        if gcloud builds submit --config deployment/cloudbuild.yaml --project="$PROJECT_ID"; then
            log_success "Cloud Build completed successfully"
        else
            error_exit "Cloud Build failed"
        fi
    fi
else
    # Use local Docker build
    log_step "Configuring Docker authentication"
    if is_dry_run; then
        log_info "[DRY RUN] Would configure Docker authentication"
    else
        gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
        log_success "Docker authentication configured"
    fi

    log_step "Building container image locally"
    log_info "Image: $IMAGE_NAME"
    log_info "Context: $PROJECT_ROOT"
    log_info "Platform: linux/amd64 (required for Cloud Run)"

    if is_dry_run; then
        log_info "[DRY RUN] Would build Docker image for linux/amd64"
    else
        # Check if Docker buildx is available
        if ! docker buildx version &> /dev/null; then
            error_exit "Docker buildx is not available. Please install Docker Desktop or enable buildx."
        fi

        # Create buildx builder if it doesn't exist
        BUILDER_NAME="multiarch-builder"
        if ! docker buildx inspect "$BUILDER_NAME" &> /dev/null; then
            log_info "Creating buildx builder: $BUILDER_NAME"
            docker buildx create --name "$BUILDER_NAME" --use --bootstrap
        else
            docker buildx use "$BUILDER_NAME"
        fi

        log_info "Building Docker image for linux/amd64 (this may take several minutes)..."
        cd "$PROJECT_ROOT"

        if docker buildx build \
            --platform linux/amd64 \
            --file "$SCRIPT_DIR/Dockerfile" \
            --tag "$IMAGE_NAME" \
            --load \
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
