#!/bin/bash
# Cleanup VM Instances Script
#
# This script lists and optionally deletes VM instances associated with the
# ETL pipeline project. Useful for cleaning up test VMs or orphaned instances.
#
# Usage:
#   ./deployment/vm/cleanup-vms.sh [--pattern PATTERN] [--all] [--dry-run] [--force]
#
# Options:
#   --pattern PATTERN   Filter VMs by name pattern (default: "test-phase")
#   --all               Show/delete all VMs (not just test VMs)
#   --dry-run           Show what would be deleted without actually deleting
#   --force             Skip confirmation prompt (use with caution)
#
# Examples:
#   # List test VMs
#   ./deployment/vm/cleanup-vms.sh --dry-run
#
#   # Delete all test VMs
#   ./deployment/vm/cleanup-vms.sh --pattern "test-phase"
#
#   # Delete all VMs (including production)
#   ./deployment/vm/cleanup-vms.sh --all --force

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=../scripts/utils.sh
source "${SCRIPT_DIR}/../scripts/utils.sh"

# Parse command line arguments
PATTERN="test-phase"
SHOW_ALL=false
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --all)
            SHOW_ALL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            export DRY_RUN
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            cat << EOF
Cleanup VM Instances Script

Usage: $0 [OPTIONS]

Options:
  --pattern PATTERN   Filter VMs by name pattern (default: "test-phase")
  --all               Show/delete all VMs (not just test VMs)
  --dry-run           Show what would be deleted without actually deleting
  --force             Skip confirmation prompt (use with caution)
  --help, -h          Show this help message

Examples:
  # List test VMs
  $0 --dry-run

  # Delete all test VMs
  $0 --pattern "test-phase"

  # Delete all VMs (including production)
  $0 --all --force
EOF
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
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

# List all VMs in the project
log_step "Listing VM instances"
log_info "Project: $PROJECT_ID"
if [[ "$SHOW_ALL" == "true" ]]; then
    log_info "Filter: All VMs"
else
    log_info "Filter: VMs matching pattern '$PATTERN'"
fi

# Get all VMs in the project
VM_LIST=$(gcloud compute instances list \
    --project="$PROJECT_ID" \
    --format="table(name,zone,status,machineType,creationTimestamp)" \
    --filter="name~'.*'" 2>&1 || true)

if [[ -z "$VM_LIST" ]] || echo "$VM_LIST" | grep -q "Listed 0 items"; then
    log_success "No VM instances found in project '$PROJECT_ID'"
    exit 0
fi

# Filter VMs based on pattern or show all
if [[ "$SHOW_ALL" == "true" ]]; then
    FILTERED_VMS=$(echo "$VM_LIST" | tail -n +2 | grep -v "^NAME" || true)
else
    FILTERED_VMS=$(echo "$VM_LIST" | tail -n +2 | grep -v "^NAME" | grep -i "$PATTERN" || true)
fi

if [[ -z "$FILTERED_VMS" ]]; then
    log_info "No VMs found matching pattern '$PATTERN'"
    log_info "All VMs in project:"
    echo "$VM_LIST"
    exit 0
fi

# Display found VMs
log_step "Found VM instances"
echo "$FILTERED_VMS"

# Extract VM names and zones
VM_NAMES=()
VM_ZONES=()

while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        # Parse the line: NAME ZONE STATUS MACHINE_TYPE CREATION_TIMESTAMP
        vm_name=$(echo "$line" | awk '{print $1}')
        vm_zone=$(echo "$line" | awk '{print $2}')
        vm_status=$(echo "$line" | awk '{print $3}')

        VM_NAMES+=("$vm_name")
        VM_ZONES+=("$vm_zone")

        log_info "  • $vm_name ($vm_zone) - Status: $vm_status"
    fi
done <<< "$FILTERED_VMS"

VM_COUNT=${#VM_NAMES[@]}

if [[ $VM_COUNT -eq 0 ]]; then
    log_info "No VMs to process"
    exit 0
fi

log_info "Total VMs found: $VM_COUNT"

# Confirm deletion
if is_dry_run; then
    log_step "DRY RUN MODE - No VMs will be deleted"
    log_info "Would delete the following $VM_COUNT VM(s):"
    for i in "${!VM_NAMES[@]}"; do
        log_info "  • ${VM_NAMES[$i]} (${VM_ZONES[$i]})"
    done
    exit 0
fi

if [[ "$FORCE" != "true" ]]; then
    echo ""
    log_warning "This will DELETE $VM_COUNT VM instance(s):"
    for i in "${!VM_NAMES[@]}"; do
        echo "  • ${VM_NAMES[$i]} (${VM_ZONES[$i]})"
    done
    echo ""

    if ! confirm_action "Are you sure you want to delete these VMs?" "n"; then
        log_info "Cancelled. No VMs were deleted."
        exit 0
    fi
fi

# Delete VMs
log_step "Deleting VM instances"
DELETED_COUNT=0
FAILED_COUNT=0
FAILED_VMS=()

for i in "${!VM_NAMES[@]}"; do
    vm_name="${VM_NAMES[$i]}"
    vm_zone="${VM_ZONES[$i]}"

    log_info "Deleting VM: $vm_name (zone: $vm_zone)..."

    if gcloud compute instances delete "$vm_name" \
        --zone="$vm_zone" \
        --project="$PROJECT_ID" \
        --quiet 2>&1; then
        log_success "Deleted: $vm_name"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    else
        log_error "Failed to delete: $vm_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_VMS+=("$vm_name")
    fi
done

# Summary
log_step "Cleanup Summary"
log_info "Total VMs processed: $VM_COUNT"
log_success "Successfully deleted: $DELETED_COUNT"

if [[ $FAILED_COUNT -gt 0 ]]; then
    log_error "Failed to delete: $FAILED_COUNT"
    log_error "Failed VMs:"
    for vm in "${FAILED_VMS[@]}"; do
        log_error "  • $vm"
    done
    exit 1
fi

log_success "All VMs cleaned up successfully!"
exit 0
