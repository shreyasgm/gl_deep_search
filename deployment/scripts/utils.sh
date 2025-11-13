#!/bin/bash
# Utility functions for GCP deployment scripts
#
# This file provides shared utility functions for logging, error handling,
# and validation used across all deployment scripts.

set -euo pipefail

# ==============================================================================
# Color codes for output
# ==============================================================================

readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# ==============================================================================
# Logging Functions
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $*" >&2
}

log_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
    echo -e "${BLUE}▶${NC} $*" >&2
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" >&2
}

# ==============================================================================
# Error Handling
# ==============================================================================

error_exit() {
    log_error "$*"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error_exit "Required command '$1' is not installed. Please install it first."
    fi
}

# ==============================================================================
# GCP Configuration Loading
# ==============================================================================

load_gcp_config() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local config_file="${script_dir}/../config/gcp-config.sh"

    if [[ ! -f "$config_file" ]]; then
        error_exit "GCP configuration file not found: $config_file
Please copy deployment/config/gcp-config.sh.template to deployment/config/gcp-config.sh
and fill in your values."
    fi

    # Source the config file
    # shellcheck source=/dev/null
    source "$config_file"

    log_info "Loaded GCP configuration from $config_file"
}

# ==============================================================================
# GCP Project Validation
# ==============================================================================

check_gcp_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        error_exit "No active GCP authentication found. Please run 'gcloud auth login' first."
    fi
    log_info "GCP authentication verified"
}

check_gcp_project() {
    local project_id="${1:-${PROJECT_ID:-}}"

    if [[ -z "$project_id" ]]; then
        error_exit "PROJECT_ID is not set. Please configure it in gcp-config.sh"
    fi

    # Check if project exists
    if ! gcloud projects describe "$project_id" &> /dev/null; then
        error_exit "GCP project '$project_id' does not exist or you don't have access to it."
    fi

    # Set as active project
    gcloud config set project "$project_id" --quiet

    log_info "Using GCP project: $project_id"
}

# ==============================================================================
# Confirmation Prompts
# ==============================================================================

confirm_action() {
    local message="${1:-Are you sure you want to continue?}"
    local default="${2:-n}"

    if [[ "$default" == "y" ]]; then
        local prompt="[Y/n]"
    else
        local prompt="[y/N]"
    fi

    echo -e "${YELLOW}$message $prompt${NC}" >&2
    read -r response

    if [[ "$default" == "y" ]]; then
        [[ "$response" =~ ^[Nn]$ ]] && return 1
    else
        [[ ! "$response" =~ ^[Yy]$ ]] && return 1
    fi

    return 0
}

# ==============================================================================
# Dry Run Support
# ==============================================================================

is_dry_run() {
    [[ "${DRY_RUN:-false}" == "true" ]]
}

dry_run_notice() {
    if is_dry_run; then
        log_warning "DRY RUN MODE - No actual changes will be made"
        return 0
    fi
    return 1
}

# ==============================================================================
# API Enablement
# ==============================================================================

enable_gcp_api() {
    local api_name="$1"
    local api_display_name="${2:-$api_name}"

    log_info "Enabling $api_display_name API..."

    if is_dry_run; then
        log_info "[DRY RUN] Would enable: $api_name"
        return 0
    fi

    if gcloud services list --enabled --filter="name:$api_name" --format="value(name)" | grep -q "^$api_name$"; then
        log_info "$api_display_name API is already enabled"
        return 0
    fi

    if gcloud services enable "$api_name" --quiet; then
        log_success "$api_display_name API enabled"
    else
        error_exit "Failed to enable $api_display_name API"
    fi
}

# ==============================================================================
# Wait Functions
# ==============================================================================

wait_for_operation() {
    local operation_type="$1"
    local operation_name="$2"
    local max_wait="${3:-300}"  # Default 5 minutes
    local interval="${4:-5}"    # Default 5 seconds

    log_info "Waiting for $operation_type '$operation_name' to complete..."

    local elapsed=0
    while [[ $elapsed -lt $max_wait ]]; do
        # This is a placeholder - actual implementation depends on operation type
        # For now, just wait
        sleep "$interval"
        elapsed=$((elapsed + interval))

        if [[ $((elapsed % 30)) -eq 0 ]]; then
            log_info "Still waiting... (${elapsed}s elapsed)"
        fi
    done

    log_warning "Wait timeout reached for $operation_type '$operation_name'"
}

# ==============================================================================
# Validation Functions
# ==============================================================================

validate_bucket_name() {
    local bucket_name="$1"

    # Bucket names must be 3-63 characters
    if [[ ${#bucket_name} -lt 3 ]] || [[ ${#bucket_name} -gt 63 ]]; then
        error_exit "Bucket name must be between 3 and 63 characters"
    fi

    # Bucket names can only contain lowercase letters, numbers, and hyphens
    if [[ ! "$bucket_name" =~ ^[a-z0-9][a-z0-9-]*[a-z0-9]$ ]] && [[ ! "$bucket_name" =~ ^[a-z0-9]$ ]]; then
        error_exit "Bucket name can only contain lowercase letters, numbers, and hyphens"
    fi

    # Cannot start or end with hyphen
    if [[ "$bucket_name" =~ ^- ]] || [[ "$bucket_name" =~ -$ ]]; then
        error_exit "Bucket name cannot start or end with a hyphen"
    fi
}

validate_project_id() {
    local project_id="$1"

    # Project IDs must be 6-30 characters
    if [[ ${#project_id} -lt 6 ]] || [[ ${#project_id} -gt 30 ]]; then
        error_exit "Project ID must be between 6 and 30 characters"
    fi

    # Project IDs can only contain lowercase letters, numbers, and hyphens
    if [[ ! "$project_id" =~ ^[a-z0-9][a-z0-9-]*[a-z0-9]$ ]] && [[ ! "$project_id" =~ ^[a-z0-9]$ ]]; then
        error_exit "Project ID can only contain lowercase letters, numbers, and hyphens"
    fi
}

# ==============================================================================
# Resource Existence Checks
# ==============================================================================

resource_exists() {
    local resource_type="$1"
    local resource_name="$2"

    case "$resource_type" in
        "bucket")
            gcloud storage buckets describe "gs://$resource_name" &> /dev/null
            ;;
        "service-account")
            gcloud iam service-accounts describe "$resource_name" &> /dev/null
            ;;
        "secret")
            gcloud secrets describe "$resource_name" &> /dev/null
            ;;
        "project")
            gcloud projects describe "$resource_name" &> /dev/null
            ;;
        *)
            log_warning "Unknown resource type: $resource_type"
            return 1
            ;;
    esac
}

# ==============================================================================
# Summary Output
# ==============================================================================

print_summary() {
    local title="$1"
    shift
    local items=("$@")

    echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
    echo -e "${GREEN}✓ $title${NC}" >&2
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" >&2

    for item in "${items[@]}"; do
        echo -e "  • $item" >&2
    done

    echo "" >&2
}
