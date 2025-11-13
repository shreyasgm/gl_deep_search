#!/bin/bash
# Monitor VM Instance Execution
#
# This script monitors a running VM instance and displays logs in real-time.
#
# Usage:
#   ./deployment/vm/monitor-vm.sh [VM_NAME] [--follow]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse arguments
VM_NAME="${1:-etl-pipeline-vm}"
FOLLOW="${2:-}"

# Load GCP configuration
load_gcp_config

# Check prerequisites
check_gcp_auth
check_gcp_project "$PROJECT_ID"
check_command gcloud

# Check if VM exists
log_step "Checking VM status"
if ! gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &> /dev/null; then
    error_exit "VM '$VM_NAME' not found in zone $ZONE"
fi

# Get VM status
VM_STATUS=$(gcloud compute instances describe "$VM_NAME" --zone="$ZONE" \
    --format="value(status)")

log_info "VM Status: $VM_STATUS"

if [[ "$VM_STATUS" != "RUNNING" ]]; then
    log_warning "VM is not running. Current status: $VM_STATUS"
    log_info "To start the VM: gcloud compute instances start $VM_NAME --zone=$ZONE"
    exit 0
fi

# Display serial port output
log_step "Displaying VM serial port output"
log_info "Press Ctrl+C to stop monitoring"
log_info ""

if [[ "$FOLLOW" == "--follow" ]] || [[ "$FOLLOW" == "-f" ]]; then
    # Follow logs in real-time
    gcloud compute instances tail-serial-port-output "$VM_NAME" --zone="$ZONE"
else
    # Show recent logs
    gcloud compute instances get-serial-port-output "$VM_NAME" --zone="$ZONE" \
        --port=1 | tail -100

    log_info ""
    log_info "To follow logs in real-time, run:"
    log_info "  $0 $VM_NAME --follow"
fi
