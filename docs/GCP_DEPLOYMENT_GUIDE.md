# Growth Lab Deep Search - GCP Deployment Guide

**Version:** 1.0
**Last Updated:** November 2025
**Author:** Growth Lab Engineering

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Cost Analysis](#cost-analysis)
4. [Prerequisites](#prerequisites)
5. [Initial Setup](#initial-setup)
6. [Deployment Options](#deployment-options)
7. [Running the Pipeline](#running-the-pipeline)
8. [Automation & Scheduling](#automation--scheduling)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)
11. [Maintenance & Updates](#maintenance--updates)

---

## Executive Summary

This guide provides specifications and step-by-step instructions for deploying the Growth Lab Deep Search ETL pipeline on Google Cloud Platform (GCP).

**Key Benefits:**
- **Cost-effective**: ~$12/year for full operation
- **Low setup time**: ~1 hour vs 4-9 hours for SLURM
- **Zero maintenance**: No cluster management or queue systems
- **Scalable**: Easy to increase parallelization or add data sources

**Recommended Configuration:**
- **Compute**: n2-standard-4 (4 vCPUs, 16 GB RAM) spot instance
- **Storage**: Cloud Storage (Standard class)
- **Region**: us-central1 (Iowa) for lowest cost
- **Execution mode**: On-demand VM for initial batch, Cloud Run Jobs for weekly updates

---

## Architecture Overview

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                      GCP Project                             │
│                                                              │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Cloud Storage │◄────────┤  Compute Engine  │           │
│  │   (Raw Data)   │         │   VM Instance    │           │
│  └────────────────┘         │                  │           │
│         ▲                   │  - ETL Pipeline  │           │
│         │                   │  - Python 3.12   │           │
│         │                   │  - uv package    │           │
│         │                   │    manager       │           │
│         │                   └──────────────────┘           │
│         │                            │                      │
│         ▼                            ▼                      │
│  ┌────────────────┐         ┌──────────────────┐           │
│  │  Cloud Storage │         │   Cloud Logging  │           │
│  │ (Processed Data│         │   (Monitoring)   │           │
│  │  & Embeddings) │         └──────────────────┘           │
│  └────────────────┘                                         │
│                                                              │
│  ┌────────────────────────────────────────────┐            │
│  │         Cloud Scheduler (Optional)          │            │
│  │         Triggers weekly updates             │            │
│  └────────────────────────────────────────────┘            │
│                                                              │
│  ┌────────────────────────────────────────────┐            │
│  │            External APIs                    │            │
│  │  - OpenAI (Embeddings)                     │            │
│  │  - Growth Lab Website (Scraping)           │            │
│  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Scraping**: VM scrapes Growth Lab publications (~6s for 10 pubs)
2. **Download**: Downloads PDFs to Cloud Storage (~30s for 10 files)
3. **OCR**: Processes PDFs using docling (~32s for 7 PDFs)
4. **Chunking**: Creates text chunks (~0.4s for 7 docs)
5. **Embeddings**: Generates embeddings via OpenAI API (~46s for 7 docs)
6. **Storage**: Saves all artifacts to Cloud Storage
7. **Shutdown**: VM automatically terminates

**Total time per 7 PDFs**: ~115 seconds (~16s per PDF)

---

## Cost Analysis

### Compute Costs (n2-standard-4, us-central1)

| Scenario | Instance Type | Hours | Cost |
|----------|--------------|-------|------|
| Initial batch (1,400 PDFs) | Spot | ~1.6h | $0.07 |
| Initial batch (conservative) | Spot | ~6.2h | $0.29 |
| Weekly update (1 PDF) | Spot | ~1min | <$0.01 |

**Annual compute**: $0.29 (initial) + $0.50 (52 weeks) = **$0.79/year**

### API Costs (OpenAI text-embedding-3-small)

| Scenario | Embeddings | Cost |
|----------|-----------|------|
| Initial batch | ~333,200 | ~$8.00 |
| Weekly update | ~238 | ~$0.01 |

**Annual API**: $8.00 (initial) + $0.52 (52 weeks) = **$8.52/year**

### Storage Costs (GCS Standard, us-central1)

| Data Type | Size | Monthly | Annual |
|-----------|------|---------|--------|
| Raw PDFs | ~5.7 GB | $0.11 | $1.37 |
| Processed data | ~3 GB | $0.06 | $0.72 |
| Embeddings | ~1.5 GB | $0.03 | $0.36 |
| **Total** | **~10 GB** | **$0.20** | **$2.45** |

### Total First Year Cost

| Component | Cost |
|-----------|------|
| Compute | $0.79 |
| OpenAI API | $8.52 |
| Storage | $2.45 |
| **Total** | **$11.76** |

**Subsequent years**: ~$3.50/year (no initial batch costs)

---

## Prerequisites

### Required Accounts

1. **Google Cloud Platform account**
   - Credit card for billing (free tier available)
   - Recommended: Use institutional email if available

2. **OpenAI API account**
   - API key with billing enabled
   - Recommended: Set usage limits ($20/month)

### Local Development Environment

- macOS/Linux/WSL
- `gcloud` CLI installed
- Git repository cloned
- Python 3.12+ with uv package manager

### Knowledge Requirements

- Basic command-line proficiency
- Understanding of ETL pipelines (provided by this codebase)
- Basic cloud concepts (VMs, storage buckets)

---

## Initial Setup

### Step 1: GCP Project Setup (15 minutes)

#### 1.1 Create GCP Project

```bash
# Install gcloud CLI if not already installed
# macOS:
brew install google-cloud-sdk

# Linux:
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init
```

#### 1.2 Create New Project

```bash
# Set project variables
export PROJECT_ID="cid-hks-1537286359734"
export PROJECT_NAME="Growth Lab Deep Search"
export BILLING_ACCOUNT_ID="YOUR_BILLING_ACCOUNT_ID"  # Find in GCP Console > Billing

# Create project
gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"

# Set as default project
gcloud config set project $PROJECT_ID

# Link billing account
gcloud billing projects link $PROJECT_ID --billing-account=$BILLING_ACCOUNT_ID
```

#### 1.3 Enable Required APIs

```bash
# Enable necessary GCP services
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
gcloud services enable cloudscheduler.googleapis.com
gcloud services enable run.googleapis.com
```

### Step 2: Storage Setup (10 minutes)

#### 2.1 Create Cloud Storage Buckets

```bash
# Set variables
export REGION="us-east4"  # Cheapest region
export BUCKET_NAME="gl-deep-search-data"

# Create main data bucket
gcloud storage buckets create gs://$BUCKET_NAME \
  --location=$REGION \
  --default-storage-class=STANDARD \
  --uniform-bucket-level-access \
  --public-access-prevention=enforced

# Note: No need to create folders explicitly in Cloud Storage!
# Folders are virtual and automatically created when you upload objects with paths.
# For example, uploading to gs://bucket/raw/file.pdf automatically creates the "raw" folder.
```

#### 2.2 Set Lifecycle Policies (Optional Cost Optimization)

Create `lifecycle-config.json`:

```json
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 90,
          "matchesPrefix": ["raw/", "intermediate/"]
        }
      },
      {
        "action": {
          "type": "Delete"
        },
        "condition": {
          "age": 365,
          "matchesPrefix": ["logs/"]
        }
      }
    ]
  }
}
```

Apply lifecycle policy:

```bash
gcloud storage buckets update gs://$BUCKET_NAME \
  --lifecycle-file=lifecycle-config.json
```

### Step 3: Service Account Setup (5 minutes)

#### 3.1 Create Service Account

```bash
# Create service account for the ETL pipeline
export SA_NAME="etl-pipeline-service-account"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create $SA_NAME \
  --display-name="ETL Pipeline Service Account" \
  --description="Service account for Growth Lab Deep Search ETL pipeline"
```

#### 3.2 Grant Permissions

```bash
# Grant storage permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"

# Grant logging permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"

# Grant monitoring permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/monitoring.metricWriter"
```

### Step 4: Secrets Management (5 minutes)

#### 4.1 Enable Secret Manager

```bash
gcloud services enable secretmanager.googleapis.com
```

#### 4.2 Store OpenAI API Key

```bash
# Create secret
echo -n "your-openai-api-key" | gcloud secrets create openai-api-key \
  --data-file=- \
  --replication-policy="automatic"

# Grant service account access to secret
# Note: If you encounter a LayoutException error, try updating gcloud first:
# gcloud components update
# Or use the alternative command below

gcloud secrets add-iam-policy-binding openai-api-key \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

**Troubleshooting:** If you encounter a `LayoutException: Multiple definitions for release tracks` error:

1. **Update gcloud SDK** (recommended - fixes the bug):
```bash
gcloud components update
```
   Then retry the original command above.

2. **Use the Cloud Console** (most reliable alternative):
   - Go to [Secret Manager](https://console.cloud.google.com/security/secret-manager)
   - Click on the `openai-api-key` secret
   - Click "PERMISSIONS" tab
   - Click "GRANT ACCESS"
   - Add `${SA_EMAIL}` with the "Secret Manager Secret Accessor" role
   - Click "SAVE"

3. **Alternative CLI approach** (grants access to all secrets in project):
```bash
# Note: This grants access to ALL secrets in the project, not just openai-api-key
# Use only if the above options don't work
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/secretmanager.secretAccessor"
```

### Step 5: VM Image Preparation (15 minutes)

#### 5.1 Create Startup Script

Create `startup-script.sh`:

```bash
#!/bin/bash
set -e

# Logging function
log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting ETL pipeline VM setup..."

# Install system dependencies
log "Installing system dependencies..."
apt-get update
apt-get install -y \
  python3.12 \
  python3.12-venv \
  python3-pip \
  git \
  curl \
  wget

# Install uv package manager
log "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.cargo/bin:$PATH"

# Set up working directory
log "Setting up working directory..."
WORK_DIR="/opt/gl-deep-search"
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clone repository
log "Cloning repository..."
git clone https://github.com/YOUR_ORG/gl_deep_search.git .

# Get OpenAI API key from Secret Manager
log "Retrieving OpenAI API key..."
export OPENAI_API_KEY=$(gcloud secrets versions access latest --secret="openai-api-key")

# Get GCS bucket name from metadata
export GCS_BUCKET=$(curl -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-bucket)

# Install Python dependencies
log "Installing Python dependencies..."
uv sync --extra etl

# Create .env file
log "Creating environment configuration..."
cat > .env << EOF
OPENAI_API_KEY=$OPENAI_API_KEY
GCS_BUCKET=$GCS_BUCKET
ENVIRONMENT=production
EOF

# Update config.yaml to use GCS storage
log "Configuring for cloud storage..."
sed -i 's|local_storage_path: "data/"|gcs_bucket: "'$GCS_BUCKET'"|' backend/etl/config.yaml
sed -i 's|sync_to_gcs: false|sync_to_gcs: true|' backend/etl/config.yaml

# Run ETL pipeline
log "Starting ETL pipeline..."
uv run python -m backend.etl.orchestrator \
  --config backend/etl/config.yaml \
  --log-level INFO \
  2>&1 | tee /var/log/etl-pipeline.log

# Upload logs to GCS
log "Uploading logs to GCS..."
gcloud storage cp /var/log/etl-pipeline.log gs://$GCS_BUCKET/logs/etl-$(date +%Y%m%d-%H%M%S).log

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  log "Pipeline completed successfully!"
  EXIT_STATUS=0
else
  log "Pipeline failed with errors"
  EXIT_STATUS=1
fi

# Shutdown VM after completion
log "Shutting down VM..."
shutdown -h now

exit $EXIT_STATUS
```

Upload startup script to GCS:

```bash
gcloud storage cp startup-script.sh gs://$BUCKET_NAME/scripts/startup-script.sh
```

#### 5.2 Create Incremental Update Script

For weekly updates, create `incremental-update.sh`:

```bash
#!/bin/bash
set -e

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

log "Starting incremental ETL update..."

# Same setup as startup-script.sh (system deps, clone, etc.)
# ... (omitted for brevity, same as above until "Run ETL pipeline")

# Run ETL pipeline in incremental mode (no full scraping)
log "Running incremental update..."
uv run python -m backend.etl.orchestrator \
  --config backend/etl/config.yaml \
  --scraper-limit 20 \
  --log-level INFO \
  2>&1 | tee /var/log/etl-incremental.log

# Upload logs
gcloud storage cp /var/log/etl-incremental.log \
  gs://$GCS_BUCKET/logs/etl-incremental-$(date +%Y%m%d-%H%M%S).log

log "Incremental update completed. Shutting down..."
shutdown -h now
```

Upload incremental script:

```bash
gcloud storage cp incremental-update.sh gs://$BUCKET_NAME/scripts/incremental-update.sh
```

---

## Deployment Options

### Option A: Manual On-Demand Execution (Recommended for Initial Batch)

Use this for the initial batch processing or one-off runs.

#### Create VM Instance

```bash
# Set VM configuration
export VM_NAME="etl-pipeline-vm"
export MACHINE_TYPE="n2-standard-4"  # 4 vCPUs, 16 GB RAM
export ZONE="us-central1-a"
export IMAGE_FAMILY="ubuntu-2204-lts"
export IMAGE_PROJECT="ubuntu-os-cloud"
export BOOT_DISK_SIZE="50GB"

# Download startup script locally first (required for --metadata-from-file)
gcloud storage cp gs://$BUCKET_NAME/scripts/startup-script.sh ./startup-script.sh

# Create spot VM instance (76% cheaper)
gcloud compute instances create $VM_NAME \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --boot-disk-size=$BOOT_DISK_SIZE \
  --boot-disk-type=pd-balanced \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --service-account=$SA_EMAIL \
  --scopes=cloud-platform \
  --metadata=gcs-bucket=$BUCKET_NAME \
  --metadata-from-file=startup-script=./startup-script.sh \
  --tags=etl-pipeline
```

**Note**: VM will automatically:
1. Install dependencies
2. Clone repository
3. Run ETL pipeline
4. Upload results to GCS
5. Shut down when complete

#### Monitor Progress

```bash
# View logs in real-time
gcloud compute instances tail-serial-port-output $VM_NAME --zone=$ZONE

# Or via Cloud Logging
gcloud logging read "resource.type=gce_instance AND resource.labels.instance_id=$VM_NAME" \
  --limit 50 \
  --format json
```

#### Retrieve Results

```bash
# Download processed data
gcloud storage cp -r gs://$BUCKET_NAME/processed ./local-processed-data/

# Download logs
gcloud storage cp gs://$BUCKET_NAME/logs/etl-*.log ./logs/
```

### Option B: Cloud Run Jobs (Recommended for Weekly Updates)

Cloud Run Jobs provide serverless execution with automatic scaling and no VM management.

#### B.1 Create Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install Python dependencies
RUN uv sync --extra etl

# Set entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "backend.etl.orchestrator"]
CMD ["--config", "backend/etl/config.yaml", "--log-level", "INFO"]
```

Create `.dockerignore`:

```
.git
.env
data/
logs/
*.pyc
__pycache__/
.pytest_cache/
.ruff_cache/
*.log
```

#### B.2 Build and Push Container

```bash
# Enable Artifact Registry
gcloud services enable artifactregistry.googleapis.com

# Create Docker repository
gcloud artifacts repositories create etl-pipeline \
  --repository-format=docker \
  --location=$REGION \
  --description="ETL pipeline container images"

# Configure Docker authentication
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build container
export IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/etl-pipeline/etl-pipeline:latest"

docker build -t $IMAGE_NAME .

# Push to Artifact Registry
docker push $IMAGE_NAME
```

#### B.3 Create Cloud Run Job

```bash
# Create job
gcloud run jobs create etl-pipeline-job \
  --image=$IMAGE_NAME \
  --region=$REGION \
  --service-account=$SA_EMAIL \
  --set-env-vars=GCS_BUCKET=$BUCKET_NAME \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --memory=8Gi \
  --cpu=4 \
  --max-retries=2 \
  --task-timeout=2h
```

#### B.4 Execute Job Manually

```bash
# Run job on-demand
gcloud run jobs execute etl-pipeline-job --region=$REGION

# Monitor execution
gcloud run jobs executions list --job=etl-pipeline-job --region=$REGION

# View logs for specific execution
export EXECUTION_NAME="etl-pipeline-job-abc123"
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=etl-pipeline-job AND resource.labels.execution_name=$EXECUTION_NAME" \
  --limit 100 \
  --format json
```

---

## Automation & Scheduling

### Automated Weekly Updates with Cloud Scheduler

#### Step 1: Create Cloud Scheduler Job

```bash
# Enable Cloud Scheduler
gcloud services enable cloudscheduler.googleapis.com

# Create weekly schedule (every Sunday at 2 AM UTC)
gcloud scheduler jobs create http etl-weekly-update \
  --location=$REGION \
  --schedule="0 2 * * 0" \
  --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/jobs/etl-pipeline-job:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA_EMAIL \
  --message-body='{"overrides":{"containerOverrides":[{"args":["--config","backend/etl/config.yaml","--scraper-limit","20","--log-level","INFO"]}]}}' \
  --time-zone="America/New_York"
```

#### Step 2: Test Schedule

```bash
# Manually trigger scheduled job
gcloud scheduler jobs run etl-weekly-update --location=$REGION
```

#### Step 3: Monitor Scheduled Runs

```bash
# List recent executions
gcloud scheduler jobs describe etl-weekly-update --location=$REGION

# View scheduler logs
gcloud logging read "resource.type=cloud_scheduler_job AND resource.labels.job_id=etl-weekly-update" \
  --limit 10
```

### Alternative: GitHub Actions Workflow

Create `.github/workflows/etl-weekly.yml`:

```yaml
name: ETL Weekly Update

on:
  schedule:
    # Every Sunday at 2 AM UTC
    - cron: '0 2 * * 0'
  workflow_dispatch:  # Allow manual triggers

jobs:
  run-etl:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Trigger Cloud Run Job
        run: |
          gcloud run jobs execute etl-pipeline-job \
            --region=us-central1 \
            --wait

      - name: Check execution status
        if: failure()
        run: |
          echo "ETL pipeline failed!"
          # Send notification (e.g., Slack, email)
```

---

## Monitoring & Logging

### Cloud Logging

#### View Real-Time Logs

```bash
# Stream logs for VM instances
gcloud logging tail "resource.type=gce_instance AND resource.labels.instance_id=$VM_NAME"

# Stream logs for Cloud Run Jobs
gcloud logging tail "resource.type=cloud_run_job AND resource.labels.job_name=etl-pipeline-job"
```

#### Query Historical Logs

```bash
# View all ETL pipeline logs from last 7 days
gcloud logging read "
  (resource.type=gce_instance OR resource.type=cloud_run_job)
  AND (
    resource.labels.instance_id=$VM_NAME
    OR resource.labels.job_name=etl-pipeline-job
  )
  AND timestamp>\"$(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%S)Z\"
" --limit 1000 --format json > etl-logs.json
```

### Cloud Monitoring

#### Create Dashboard

1. Go to Cloud Console > Monitoring > Dashboards
2. Create new dashboard: "ETL Pipeline Monitoring"
3. Add widgets:
   - **Compute Instance CPU**: Filter by instance name `etl-pipeline-vm`
   - **Compute Instance Memory**: Filter by instance name
   - **Cloud Run Job Execution Count**: Filter by job name
   - **Cloud Storage Bucket Size**: Filter by bucket name
   - **Cloud Run Job Execution Duration**: Filter by job name

#### Set Up Alerts

```bash
# Create alert for pipeline failures
gcloud alpha monitoring policies create \
  --notification-channels=$NOTIFICATION_CHANNEL_ID \
  --display-name="ETL Pipeline Failure Alert" \
  --condition-display-name="Pipeline execution failed" \
  --condition-threshold-value=1 \
  --condition-threshold-duration=60s \
  --condition-filter='
    resource.type="cloud_run_job"
    AND resource.labels.job_name="etl-pipeline-job"
    AND metric.type="run.googleapis.com/job/completed_execution_count"
    AND metric.labels.result="failed"
  '
```

### Cost Monitoring

#### Set Budget Alerts

```bash
# Create budget alert at $20/month
gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT_ID \
  --display-name="ETL Pipeline Budget" \
  --budget-amount=20 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
```

Go to GCP Console > Billing > Budgets & alerts to configure email notifications.

### Application-Level Monitoring

The ETL pipeline already includes profiling via `backend/etl/utils/profiling.py`. Logs include:

- Component execution times
- Resource usage (memory, CPU)
- API call counts
- Success/failure metrics

Example log output:
```
[PROFILE] Download files: 30.34s
[METRICS] File Downloader:
  total_downloads_attempted: 13
  successful_downloads: 10
  total_size_mb: 28.59 MB
```

---

## Troubleshooting

### Common Issues

#### Issue 1: VM Fails to Start

**Symptoms**: VM is created but startup script doesn't execute

**Solution**:
```bash
# Check serial port output for errors
gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE

# SSH into VM for debugging
gcloud compute ssh $VM_NAME --zone=$ZONE

# Check startup script log
sudo journalctl -u google-startup-scripts.service
```

#### Issue 2: OpenAI API Rate Limits

**Symptoms**: `RateLimitError` in logs

**Solution**:
```python
# Already handled in backend/etl/utils/embeddings_generator.py
# Increase delays in backend/etl/config.yaml:
file_processing:
  embedding:
    rate_limit_delay: 0.5  # Increase from 0.1 to 0.5
    retry_delays: [2, 5, 10]  # Increase retry delays
```

#### Issue 3: Storage Permission Denied

**Symptoms**: `403 Forbidden` when accessing GCS

**Solution**:
```bash
# Verify service account has correct permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}"

# Re-grant storage permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

#### Issue 4: Pipeline Timeout

**Symptoms**: Cloud Run Job times out after 2 hours

**Solution**:
```bash
# Increase timeout (max 24 hours)
gcloud run jobs update etl-pipeline-job \
  --region=$REGION \
  --task-timeout=6h

# Or split pipeline into smaller jobs
uv run python -m backend.etl.orchestrator --scraper-limit 100
```

#### Issue 5: Out of Memory

**Symptoms**: VM or container crashes with OOM error

**Solution**:
```bash
# For VMs: Use larger machine type
gcloud compute instances set-machine-type $VM_NAME \
  --zone=$ZONE \
  --machine-type=n2-standard-8

# For Cloud Run Jobs: Increase memory
gcloud run jobs update etl-pipeline-job \
  --region=$REGION \
  --memory=16Gi
```

### Debugging Commands

```bash
# Check VM status
gcloud compute instances describe $VM_NAME --zone=$ZONE

# View Cloud Run Job executions
gcloud run jobs executions list \
  --job=etl-pipeline-job \
  --region=$REGION

# Check storage bucket contents
gcloud storage ls -r gs://$BUCKET_NAME/

# View recent error logs
gcloud logging read "severity>=ERROR" \
  --limit 50 \
  --format json

# Check OpenAI API key access
gcloud secrets versions access latest --secret="openai-api-key"
```

---

## Maintenance & Updates

### Updating the Pipeline Code

#### Option 1: Rebuild Container (Cloud Run Jobs)

```bash
# Pull latest code
git pull origin main

# Rebuild and push container
docker build -t $IMAGE_NAME .
docker push $IMAGE_NAME

# Update Cloud Run Job to use new image
gcloud run jobs update etl-pipeline-job \
  --region=$REGION \
  --image=$IMAGE_NAME
```

#### Option 2: Update Startup Script (VMs)

```bash
# Edit startup-script.sh locally
vim startup-script.sh

# Upload to GCS
gcloud storage cp startup-script.sh gs://$BUCKET_NAME/scripts/startup-script.sh

# Next VM creation will use updated script
```

### Updating Dependencies

If `pyproject.toml` changes:

```bash
# Update uv.lock
uv lock

# Commit changes
git add uv.lock pyproject.toml
git commit -m "Update dependencies"
git push

# Rebuild container or update startup script (as above)
```

### Updating Configuration

To modify ETL behavior without code changes:

```bash
# Edit config in GCS
gcloud storage cat gs://$BUCKET_NAME/config/etl-config.yaml > temp-config.yaml
vim temp-config.yaml
gcloud storage cp temp-config.yaml gs://$BUCKET_NAME/config/etl-config.yaml

# Update startup script to use GCS config
# In startup-script.sh, add:
gcloud storage cp gs://$GCS_BUCKET/config/etl-config.yaml backend/etl/config.yaml
```

### Scaling Considerations

#### Horizontal Scaling (Multiple VMs)

For very large batches, parallelize across multiple VMs:

```bash
# Create multiple VMs with different scraper ranges
for i in {1..4}; do
  START=$((($i - 1) * 500 + 1))
  END=$(($i * 500))

  gcloud compute instances create etl-pipeline-vm-$i \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --metadata=gcs-bucket=$BUCKET_NAME,scraper-start=$START,scraper-end=$END \
    # ... (other flags same as before)
done
```

Update startup script to use `scraper-start` and `scraper-end` metadata.

#### Vertical Scaling (Larger Instances)

For memory-intensive documents:

```bash
# Use high-memory machine type
export MACHINE_TYPE="n2-highmem-4"  # 4 vCPUs, 32 GB RAM
```

### Data Backup & Retention

#### Backup Strategy

```bash
# Create backup bucket
export BACKUP_BUCKET="${PROJECT_ID}-backup"
gcloud storage buckets create gs://$BACKUP_BUCKET \
  --location=$REGION \
  --storage-class=COLDLINE

# Schedule weekly backups via cron or Cloud Scheduler
gcloud storage cp -r gs://$BUCKET_NAME/processed gs://$BACKUP_BUCKET/processed-$(date +%Y%m%d)
```

#### Restore from Backup

```bash
# List available backups
gcloud storage ls gs://$BACKUP_BUCKET/

# Restore specific backup
gcloud storage cp -r gs://$BACKUP_BUCKET/processed-20250101 gs://$BUCKET_NAME/processed-restored
```

---

## Performance Optimization

### Compute Optimization

1. **Use Spot/Preemptible VMs**: 76% cost savings
2. **Right-size instances**: Start with n2-standard-4, adjust based on metrics
3. **Use SSDs for disk**: Faster I/O for OCR operations
4. **Enable CPU overcommit**: For I/O-bound workloads

### Storage Optimization

1. **Lifecycle policies**: Move old data to Nearline/Coldline
2. **Compress embeddings**: Use parquet with compression
3. **Delete intermediate files**: Keep only final outputs
4. **Use requester pays**: If sharing with external users

### API Optimization

1. **Batch embeddings**: Already implemented (batch_size=32)
2. **Use smaller model**: `text-embedding-3-small` vs `-large`
3. **Cache embeddings**: Skip documents already embedded
4. **Implement retry logic**: Already included in config

### Network Optimization

1. **Use regional resources**: All in same region (us-central1)
2. **Enable Private Google Access**: VMs access GCS without egress
3. **Use internal IPs**: Avoid external IP charges

---

## Security Best Practices

### Access Control

```bash
# Principle of least privilege
# Service account should only have necessary permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}"
```

### Secret Management

- Never commit API keys to Git
- Use Secret Manager for all sensitive data
- Rotate API keys quarterly
- Use workload identity for GKE (if scaling to Kubernetes)

### Network Security

```bash
# Create firewall rules for VMs (if needed)
gcloud compute firewall-rules create allow-internal-etl \
  --network=default \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --target-tags=etl-pipeline
```

### Data Security

```bash
# Enable encryption at rest (default)
# Enable versioning for critical buckets
gcloud storage buckets update gs://$BUCKET_NAME \
  --versioning

# Set retention policy
gcloud storage buckets update gs://$BUCKET_NAME \
  --retention-period=30d
```

---

## Appendix

### A. Cost Calculator Spreadsheet

Create a Google Sheet with:

| Parameter | Value | Formula |
|-----------|-------|---------|
| PDFs processed | 1400 | Input |
| Seconds per PDF | 16 | From profiling |
| Total hours | =B2*B3/3600 | |
| Spot price/hour | 0.046 | us-central1 |
| Compute cost | =B4*B5 | |
| Embeddings count | =B2*238 | 238 chunks/PDF avg |
| Embedding cost | =B7*0.00002/1000 | OpenAI pricing |
| Storage GB | 10 | Estimate |
| Storage cost/month | =B9*0.02 | GCS Standard |

### B. Monitoring Checklist

- [ ] Cloud Logging configured
- [ ] Error alerting set up
- [ ] Budget alerts configured ($20 threshold)
- [ ] Dashboard created for key metrics
- [ ] Weekly reports automated
- [ ] Backup strategy implemented

### C. Useful GCP Commands

```bash
# SSH into running VM
gcloud compute ssh $VM_NAME --zone=$ZONE

# Copy file to VM
gcloud compute scp local-file.txt $VM_NAME:~/remote-file.txt --zone=$ZONE

# List all VMs
gcloud compute instances list

# Delete VM
gcloud compute instances delete $VM_NAME --zone=$ZONE

# View project-wide logs
gcloud logging read "timestamp>\"$(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S)Z\"" --limit 100

# Check current costs
gcloud billing accounts list
gcloud billing projects describe $PROJECT_ID
```

### D. Configuration Reference

Key configuration files:

- `backend/etl/config.yaml` - ETL pipeline configuration
- `startup-script.sh` - VM initialization
- `Dockerfile` - Container image
- `.env` - Environment variables (not committed)

Important config values:

```yaml
# backend/etl/config.yaml
runtime:
  detect_automatically: true
  gcs_bucket: "gl-deep-search-data"
  sync_to_gcs: true

file_processing:
  embedding:
    batch_size: 32
    rate_limit_delay: 0.1
    max_retries: 3
```

### E. Support Resources

- **GCP Documentation**: https://cloud.google.com/docs
- **OpenAI API Docs**: https://platform.openai.com/docs
- **Project Repository**: (Your GitHub repo URL)
- **Issue Tracker**: (Your GitHub issues URL)

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-11-12 | Initial release | Growth Lab Engineering |

---

## Feedback & Contributions

For questions, issues, or improvements to this guide, please:

1. Open an issue in the GitHub repository
2. Submit a pull request with proposed changes
3. Contact the Growth Lab engineering team

---

**End of Guide**
