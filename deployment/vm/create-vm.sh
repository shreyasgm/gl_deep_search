#!/bin/bash
# Create VM Instance for ETL Pipeline
#
# This script creates a VM instance that will automatically run the ETL pipeline
# and shut down when complete.
#
# Usage:
#   ./deployment/vm/create-vm.sh [--incremental] [--scraper-limit N] [--vm-name NAME] [--on-demand|--spot] [--dry-run]
#
# Options:
#   --incremental       Run incremental update instead of full batch
#   --scraper-limit N   Limit number of publications to process
#   --vm-name NAME      Custom VM name (default: etl-pipeline-vm)
#   --on-demand         Use on-demand instance (faster startup, higher cost)
#   --spot              Use spot instance (slower startup, 76% cost savings)
#   --dry-run           Show what would be done without executing

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse command line arguments
INCREMENTAL=false
SCRAPER_LIMIT=""
DRY_RUN=false
VM_NAME="etl-pipeline-vm"
USE_SPOT=""  # Empty means use config default

while [[ $# -gt 0 ]]; do
    case $1 in
        --incremental)
            INCREMENTAL=true
            shift
            ;;
        --scraper-limit)
            SCRAPER_LIMIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            export DRY_RUN
            shift
            ;;
        --vm-name)
            VM_NAME="$2"
            shift 2
            ;;
        --on-demand)
            USE_SPOT="false"
            shift
            ;;
        --spot)
            USE_SPOT="true"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--incremental] [--scraper-limit N] [--vm-name NAME] [--on-demand|--spot] [--dry-run]"
            exit 1
            ;;
    esac
done

# Load GCP configuration
load_gcp_config

# Override spot instance setting if specified via command line
if [[ -n "$USE_SPOT" ]]; then
    VM_USE_SPOT="$USE_SPOT"
    log_info "Overriding spot instance setting: VM_USE_SPOT=$VM_USE_SPOT"
fi

# Check prerequisites
log_step "Checking prerequisites"
check_gcp_auth
check_gcp_project "$PROJECT_ID"
check_command gcloud

# Determine startup script
if [[ "$INCREMENTAL" == "true" ]]; then
    STARTUP_SCRIPT_NAME="incremental-update.sh"
    log_info "Using incremental update script"
else
    STARTUP_SCRIPT_NAME="startup-script.sh"
    log_info "Using full batch processing script"
fi

STARTUP_SCRIPT_LOCAL="${SCRIPT_DIR}/${STARTUP_SCRIPT_NAME}"
STARTUP_SCRIPT_GCS="gs://${BUCKET_NAME}/scripts/${STARTUP_SCRIPT_NAME}"

# Upload startup script to GCS
log_step "Uploading startup script to GCS"
if [[ ! -f "$STARTUP_SCRIPT_LOCAL" ]]; then
    error_exit "Startup script not found: $STARTUP_SCRIPT_LOCAL"
fi

if is_dry_run; then
    log_info "[DRY RUN] Would upload startup script to: $STARTUP_SCRIPT_GCS"
else
    log_info "Uploading startup script..."
    if gcloud storage cp "$STARTUP_SCRIPT_LOCAL" "$STARTUP_SCRIPT_GCS" --quiet; then
        log_success "Startup script uploaded successfully"
    else
        error_exit "Failed to upload startup script"
    fi
fi

# Build ETL arguments
ETL_ARGS=""
if [[ -n "$SCRAPER_LIMIT" ]]; then
    ETL_ARGS="--scraper-limit $SCRAPER_LIMIT"
fi

# Check if VM already exists
log_step "Checking if VM already exists"
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
    log_warning "VM '$VM_NAME' already exists in zone $ZONE"
    if ! confirm_action "Do you want to delete and recreate it?" "n"; then
        log_info "Exiting..."
        exit 0
    fi

    log_info "Deleting existing VM..."
    if ! is_dry_run; then
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet || true
        log_info "Waiting for VM deletion to complete..."
        sleep 5
    fi
fi

# Create VM instance
log_step "Creating VM instance"
log_info "VM Name: $VM_NAME"
log_info "Machine Type: $VM_MACHINE_TYPE"
log_info "Zone: $ZONE"
log_info "Spot Instance: $VM_USE_SPOT"
log_info "Startup Script: $STARTUP_SCRIPT_NAME"

# Build gcloud command
# Use Ubuntu 22.04 LTS for Docker support and standard tooling
CREATE_CMD=(
    gcloud compute instances create "$VM_NAME"
    --zone="$ZONE"
    --machine-type="$VM_MACHINE_TYPE"
    --image-family="ubuntu-2204-lts"
    --image-project="ubuntu-os-cloud"
    --boot-disk-size="${VM_BOOT_DISK_SIZE}GB"
    --boot-disk-type="$VM_BOOT_DISK_TYPE"
    --service-account="$SA_EMAIL"
    --scopes="cloud-platform"
    --metadata="gcs-bucket=$BUCKET_NAME,image-name=$IMAGE_NAME,etl-args=$ETL_ARGS"
    --metadata-from-file="startup-script=$STARTUP_SCRIPT_LOCAL"
    --tags="etl-pipeline"
)

# Add spot instance configuration
if [[ "$VM_USE_SPOT" == "true" ]]; then
    CREATE_CMD+=(
        --provisioning-model=SPOT
        --instance-termination-action=DELETE
    )
fi

if is_dry_run; then
    log_info "[DRY RUN] Would execute:"
    log_info "${CREATE_CMD[*]}"
else
    log_info "Creating VM instance..."
    if "${CREATE_CMD[@]}"; then
        log_success "VM instance created successfully"
    else
        error_exit "Failed to create VM instance"
    fi
fi

# Summary
print_summary "VM Instance Created" \
    "VM Name: $VM_NAME" \
    "Zone: $ZONE" \
    "Machine Type: $VM_MACHINE_TYPE" \
    "Spot Instance: $VM_USE_SPOT" \
    "Startup Script: $STARTUP_SCRIPT_NAME" \
    "ETL Args: ${ETL_ARGS:-none}"

log_success "VM instance creation completed!"
log_info "The VM will automatically:"
log_info "  1. Install Docker and Google Cloud SDK"
log_info "  2. Authenticate with Artifact Registry"
log_info "  3. Pull container image from Artifact Registry"
log_info "  4. Run ETL pipeline in container"
log_info "  5. Upload logs and results to GCS"
log_info "  6. Shut down when complete"
log_info ""
log_info "Monitor progress with:"
log_info "  ./deployment/vm/monitor-vm.sh $VM_NAME"
