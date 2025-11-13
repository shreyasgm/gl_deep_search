#!/bin/bash
# Setup Cost Monitoring and Budget Alerts
#
# This script configures budget alerts for the ETL pipeline to prevent cost overruns.
# It sets up budget alerts at 50%, 75%, and 100% of a monthly budget threshold.
#
# Note: This script uses the Google Cloud Billing Budgets REST API via curl
#       because the gcloud CLI has limited support for creating budgets with
#       threshold rules and notification configurations.
#
# Usage:
#   ./deployment/scripts/05-setup-cost-monitoring.sh [--budget AMOUNT] [--dry-run]

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source utility functions
# shellcheck source=utils.sh
source "${SCRIPT_DIR}/utils.sh"

# Parse command line arguments
BUDGET_AMOUNT="${BUDGET_AMOUNT:-20}"  # Default $20/month
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --budget)
            BUDGET_AMOUNT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            export DRY_RUN
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Usage: $0 [--budget AMOUNT] [--dry-run]"
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
check_command curl

# Validate budget amount
if ! [[ "$BUDGET_AMOUNT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    error_exit "Budget amount must be a positive number"
fi

# Check if amount is positive (using awk for portability)
if command -v awk &> /dev/null; then
    if awk "BEGIN {exit !($BUDGET_AMOUNT > 0)}"; then
        : # Amount is positive
    else
        error_exit "Budget amount must be greater than 0"
    fi
fi

# Budget configuration
BUDGET_DISPLAY_NAME="ETL Pipeline Budget"
BUDGET_ID="etl-pipeline-budget"

log_step "Setting up cost monitoring"
log_info "Budget amount: \$$BUDGET_AMOUNT/month"
log_info "Project: $PROJECT_ID"
log_info "Billing account: $BILLING_ACCOUNT_ID"
log_info ""
log_info "Note: Using Google Cloud Billing Budgets REST API"
log_info "      (gcloud CLI has limited support for budget threshold rules)"

# Check if billing budget API is enabled
if ! gcloud services list --enabled --filter="name:billingbudgets.googleapis.com" --format="value(name)" | grep -q "billingbudgets.googleapis.com"; then
    log_info "Enabling Cloud Billing Budget API..."
    if ! is_dry_run; then
        enable_gcp_api "billingbudgets.googleapis.com" "Cloud Billing Budget"
    fi
fi

# Create budget configuration file
BUDGET_CONFIG_FILE=$(mktemp)
cat > "$BUDGET_CONFIG_FILE" << EOF
{
  "displayName": "$BUDGET_DISPLAY_NAME",
  "budgetFilter": {
    "projects": ["projects/$PROJECT_ID"],
    "creditTypesTreatment": "INCLUDE_ALL_CREDITS"
  },
  "amount": {
    "specifiedAmount": {
      "currencyCode": "USD",
      "units": "$(echo "$BUDGET_AMOUNT" | cut -d. -f1)",
      "nanos": $(if echo "$BUDGET_AMOUNT" | grep -q "\."; then
        DECIMAL_PART=$(echo "$BUDGET_AMOUNT" | cut -d. -f2)
        # Convert decimal part to nanos (e.g., "50" -> 500000000, "5" -> 50000000)
        DECIMAL_LEN=${#DECIMAL_PART}
        awk "BEGIN {printf \"%.0f\", $DECIMAL_PART * (10 ^ (9 - $DECIMAL_LEN))}"
      else
        echo "0"
      fi)
    }
  },
  "thresholdRules": [
    {
      "thresholdPercent": 0.5,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 0.75,
      "spendBasis": "CURRENT_SPEND"
    },
    {
      "thresholdPercent": 1.0,
      "spendBasis": "CURRENT_SPEND"
    }
  ]
}
EOF

# Ensure nanos is properly formatted (already handled in heredoc)

# Check if budget already exists
log_step "Checking for existing budget"
EXISTING_BUDGETS=$(gcloud billing budgets list \
    --billing-account="$BILLING_ACCOUNT_ID" \
    --filter="displayName:\"$BUDGET_DISPLAY_NAME\"" \
    --format="value(name)" 2>/dev/null || echo "")

if [ -n "$EXISTING_BUDGETS" ]; then
    EXISTING_BUDGET_NAME=$(echo "$EXISTING_BUDGETS" | head -n1)
    log_warning "Budget '$BUDGET_DISPLAY_NAME' already exists"
    if ! confirm_action "Do you want to update it?" "n"; then
        log_info "Exiting..."
        rm -f "$BUDGET_CONFIG_FILE"
        exit 0
    fi

    if is_dry_run; then
        log_info "[DRY RUN] Would update budget: $EXISTING_BUDGET_NAME"
    else
        log_info "Updating existing budget using REST API..."
        # Get access token
        ACCESS_TOKEN=$(gcloud auth print-access-token)

        # Update budget using REST API (with quota project header)
        RESPONSE=$(curl -s -w "\n%{http_code}" -X PATCH \
            "https://billingbudgets.googleapis.com/v1/$EXISTING_BUDGET_NAME" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            -H "x-goog-user-project: $PROJECT_ID" \
            -d @"$BUDGET_CONFIG_FILE")

        # Extract HTTP code and body (portable approach for BSD/macOS and GNU/Linux)
        HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
        RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')  # Delete last line (portable)

        if [ "$HTTP_CODE" != "200" ]; then
            log_error "Failed to update budget. HTTP Code: $HTTP_CODE"
            log_error "API Response:"
            echo "$RESPONSE_BODY" | grep -o '"message":"[^"]*"' || echo "$RESPONSE_BODY"
            error_exit "Failed to update budget"
        fi
        log_success "Budget updated successfully"
    fi
else
    if is_dry_run; then
        log_info "[DRY RUN] Would create budget with display name: $BUDGET_DISPLAY_NAME"
    else
        log_info "Creating new budget using REST API..."
        # Get access token
        ACCESS_TOKEN=$(gcloud auth print-access-token)

        # Create budget using REST API (with quota project header)
        RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
            "https://billingbudgets.googleapis.com/v1/billingAccounts/$BILLING_ACCOUNT_ID/budgets" \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            -H "x-goog-user-project: $PROJECT_ID" \
            -d @"$BUDGET_CONFIG_FILE")

        # Extract HTTP code and body (portable approach for BSD/macOS and GNU/Linux)
        HTTP_CODE=$(echo "$RESPONSE" | tail -n 1)
        RESPONSE_BODY=$(echo "$RESPONSE" | sed '$d')  # Delete last line (portable)

        if [ "$HTTP_CODE" != "200" ]; then
            log_error "Failed to create budget. HTTP Code: $HTTP_CODE"
            log_error "API Response:"
            echo "$RESPONSE_BODY" | grep -o '"message":"[^"]*"' || echo "$RESPONSE_BODY"
            error_exit "Failed to create budget"
        fi
        log_success "Budget created successfully"
    fi
fi

# Clean up temp file
rm -f "$BUDGET_CONFIG_FILE"

# Set up email notifications (requires manual setup in console)
log_step "Notification setup"
# Calculate thresholds (using awk for portability)
THRESHOLD_50=$(awk "BEGIN {printf \"%.2f\", $BUDGET_AMOUNT * 0.5}")
THRESHOLD_75=$(awk "BEGIN {printf \"%.2f\", $BUDGET_AMOUNT * 0.75}")

log_info "Budget alerts are configured at:"
log_info "  • 50% threshold (\$$THRESHOLD_50)"
log_info "  • 75% threshold (\$$THRESHOLD_75)"
log_info "  • 100% threshold (\$$BUDGET_AMOUNT)"
log_info ""
log_warning "IMPORTANT: Email notifications are NOT automatically configured."
log_info "To receive email alerts, you must manually add notification channels:"
log_info ""
log_info "  1. Go to: https://console.cloud.google.com/billing/$BILLING_ACCOUNT_ID/budgets"
log_info "  2. Find and click on: $BUDGET_DISPLAY_NAME"
log_info "  3. Click 'Edit Budget'"
log_info "  4. In the 'Manage notifications' section:"
log_info "     • Connect a billing account (if not already connected)"
log_info "     • Add email recipients for budget alerts"
log_info "  5. Optionally: Set up Pub/Sub topic for programmatic alerts"
log_info ""
log_info "Without notification channels, budget thresholds will be tracked but"
log_info "you won't receive any alerts when they are exceeded."

# Summary
print_summary "Cost Monitoring Configured" \
    "Budget ID: $BUDGET_ID" \
    "Monthly Budget: \$$BUDGET_AMOUNT" \
    "Alerts: 50%, 75%, 100% thresholds" \
    "Project: $PROJECT_ID"

log_success "Cost monitoring setup completed!"
log_info "Budget alerts will trigger when spending reaches thresholds"
log_info "Monitor costs at: https://console.cloud.google.com/billing/budgets"
