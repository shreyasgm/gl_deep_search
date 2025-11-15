# GCP Deployment Guide

This directory contains scripts and configurations for deploying the Growth Lab Deep Search ETL pipeline to Google Cloud Platform (GCP).

## Quick Start

1. **Configure GCP settings:**
   ```bash
   cp deployment/config/gcp-config.sh.template deployment/config/gcp-config.sh
   # Edit deployment/config/gcp-config.sh with your values
   ```

2. **Run setup scripts in order:**
   ```bash
   ./deployment/scripts/01-setup-gcp-project.sh
   ./deployment/scripts/02-setup-storage.sh
   ./deployment/scripts/03-setup-secrets.sh
   ./deployment/scripts/04-create-service-account.sh
   ```

3. **Build and deploy Docker image:**
   ```bash
   ./deployment/cloud-run/deploy.sh
   ```

4. **Schedule weekly updates:**
   ```bash
   ./deployment/cloud-run/schedule.sh
   ```

5. **For initial batch processing, create a VM:**
   ```bash
   ./deployment/vm/create-vm.sh
   ```

## Directory Structure

```
deployment/
├── README.md                    # This file
├── cloudbuild.yaml              # Cloud Build configuration (for --cloud-build option)
├── config/
│   ├── gcp-config.sh.template   # GCP configuration template
│   ├── gcp-config.sh            # GCP configuration (gitignored)
│   └── lifecycle-policy.json    # GCS lifecycle rules
├── scripts/
│   ├── 01-setup-gcp-project.sh  # GCP project setup
│   ├── 02-setup-storage.sh      # Storage bucket setup
│   ├── 03-setup-secrets.sh      # Secret Manager setup
│   ├── 04-create-service-account.sh  # Service account setup
│   ├── 05-setup-cost-monitoring.sh  # Budget alerts setup
│   ├── calculate-costs.sh       # Cost calculation utility
│   └── utils.sh                 # Shared utilities
├── vm/
│   ├── startup-script.sh        # VM initialization script
│   ├── incremental-update.sh    # Incremental update script
│   ├── create-vm.sh             # Create VM instance
│   ├── monitor-vm.sh            # Monitor VM execution
│   ├── cleanup-vms.sh           # Cleanup VM instances
│   └── test_batch_processing.py  # Test orchestration script
├── cloud-run/
│   ├── Dockerfile               # Container image definition
│   ├── build.sh                 # Build Docker image locally
│   ├── deploy.sh                # Deploy Cloud Run Job
│   ├── schedule.sh              # Setup Cloud Scheduler
│   ├── execute.sh               # Manual job execution
│   ├── entrypoint.sh            # Container entrypoint script
│   └── test-container-local.sh  # Test container locally
├── workflows/                   # (Optional) GitHub Actions workflows
├── TEST_RESULTS.md              # Test results template
└── TEST_RESULTS_PHASE*.md       # Generated test phase results
```

**Note**: The `.dockerignore` file is located in the repository root (not in `deployment/cloud-run/`). The `docker-compose.yml` file is also in the repository root for local testing.

## Prerequisites

- **GCP Account**: Active Google Cloud Platform account with billing enabled
- **gcloud CLI**: Installed and authenticated (`gcloud auth login`)
- **Docker**: Installed locally (for building container images)
- **Git**: Repository cloned locally
- **OpenAI API Key**: For embeddings generation

**Note**: Build the Docker image (`./deployment/cloud-run/deploy.sh`) before creating VMs for batch processing.

## Detailed Setup Instructions

### Step 1: Configure GCP Settings

Copy the configuration template and fill in your values:

```bash
cp deployment/config/gcp-config.sh.template deployment/config/gcp-config.sh
```

Edit `deployment/config/gcp-config.sh` and set:
- `PROJECT_ID`: Your GCP project ID (must be globally unique)
- `BILLING_ACCOUNT_ID`: Your billing account ID
- `BUCKET_NAME`: Cloud Storage bucket name (globally unique)
- `GITHUB_REPO_URL`: Your repository URL
- Other settings as needed

### Step 2: Initial GCP Setup

Run the setup scripts in order. Each script checks prerequisites and provides helpful error messages.

```bash
# 1. Create GCP project and enable APIs
./deployment/scripts/01-setup-gcp-project.sh

# 2. Create Cloud Storage bucket
./deployment/scripts/02-setup-storage.sh

# 3. Store secrets (will prompt for OpenAI API key)
./deployment/scripts/03-setup-secrets.sh

# 4. Create service account with permissions
./deployment/scripts/04-create-service-account.sh
```

### Step 3: Build and Deploy Docker Image

Build and deploy the containerized ETL pipeline:

```bash
./deployment/cloud-run/deploy.sh
```

This builds the Docker image, pushes it to Artifact Registry, and creates/updates the Cloud Run Job. The same image is used for both Cloud Run and VM batch processing.

**Build Options:**

The `deploy.sh` script supports multiple build methods:

```bash
# Local Docker build (default, requires Docker Desktop)
./deployment/cloud-run/deploy.sh --local-build

# Cloud Build (uses deployment/cloudbuild.yaml, no local Docker required)
./deployment/cloud-run/deploy.sh --cloud-build

# Skip build, use existing image from registry
./deployment/cloud-run/deploy.sh --skip-build

# Dry run (show what would be done)
./deployment/cloud-run/deploy.sh --dry-run
```

**Note**: For local testing, you can build the image without deploying:

```bash
# Build for local platform (ARM64 on Mac)
./deployment/cloud-run/build.sh --platform linux/arm64 --load

# Build for cloud deployment (AMD64)
./deployment/cloud-run/build.sh --platform linux/amd64 --push
```

### Step 4: Schedule Weekly Updates

Set up automatic weekly execution:

```bash
./deployment/cloud-run/schedule.sh
```

This creates a Cloud Scheduler job that runs every Sunday at 2 AM (configurable).

### Step 5: Test Batch Processing (Recommended)

Before running the full batch, test with a small subset to validate costs and performance:

```bash
# Set up cost monitoring first
./deployment/scripts/05-setup-cost-monitoring.sh --budget 20

# Phase 1: Test with 10 publications
python deployment/vm/test_batch_processing.py --limit 10 --phase 1

# Phase 2: Test with 100 publications (only if Phase 1 passes)
python deployment/vm/test_batch_processing.py --limit 100 --phase 2
```

Test reports will be generated in `deployment/TEST_RESULTS_PHASE*.md`.

### Step 6: Initial Batch Processing (Optional)

For processing a large initial batch of documents, use a VM:

```bash
# Full batch processing
./deployment/vm/create-vm.sh

# Incremental update (limited scraping)
./deployment/vm/create-vm.sh --incremental

# With custom scraper limit
./deployment/vm/create-vm.sh --scraper-limit 100
```

Monitor the VM execution:

```bash
./deployment/vm/monitor-vm.sh etl-pipeline-vm --follow
```

Clean up test VMs when done:

```bash
# List test VMs
./deployment/vm/cleanup-vms.sh --dry-run

# Delete test VMs matching pattern
./deployment/vm/cleanup-vms.sh --pattern "test-phase"

# Delete all VMs (use with caution)
./deployment/vm/cleanup-vms.sh --all --force
```

## Manual Execution

### Execute Cloud Run Job Manually

```bash
# Incremental update (default: 20 publications)
./deployment/cloud-run/execute.sh

# Full pipeline run
./deployment/cloud-run/execute.sh --full

# Custom scraper limit
./deployment/cloud-run/execute.sh --scraper-limit 50

# Wait for completion and show logs
./deployment/cloud-run/execute.sh --wait
```

### View Logs

```bash
# Cloud Run Job logs
gcloud logging read \
  "resource.type=cloud_run_job AND resource.labels.job_name=etl-pipeline-job" \
  --limit 100

# VM serial port output
gcloud compute instances get-serial-port-output etl-pipeline-vm --zone=us-central1-a
```

## Architecture

All deployments use the same Docker container image. Cloud Run runs containers directly; VMs pull and run containers from Artifact Registry.

### Container Image Build

The Docker image is built using a multi-stage build process:
- **Builder stage**: Uses `uv` to install dependencies and build the application
- **Runtime stage**: Minimal Python image with only runtime dependencies

The image is stored in Artifact Registry and can be built either:
- **Locally**: Using Docker Desktop with `buildx` (requires local Docker installation)
- **In Cloud**: Using Cloud Build (no local Docker required, uses `deployment/cloudbuild.yaml`)

Both methods produce the same image, which is then used by:
- Cloud Run Jobs (for scheduled and manual executions)
- VM instances (for batch processing)

## Configuration Files

### Production ETL Config

The production configuration is located at `backend/etl/config.production.yaml`. Key differences from development:

- Uses GCS bucket paths instead of local paths
- Cloud-optimized rate limits and timeouts
- Enhanced retry policies
- Production logging configuration

### Environment Variables

The following environment variables are automatically set:

- `GCS_BUCKET`: Cloud Storage bucket name
- `OPENAI_API_KEY`: Retrieved from Secret Manager
- `ENVIRONMENT`: Set to "production"
- `GOOGLE_CLOUD_PROJECT`: GCP project ID

## Cost Estimation

Based on the deployment guide:

- **Initial batch (1,400 PDFs)**: ~$0.29 (spot VM)
- **Weekly updates**: <$0.01 per run
- **Annual compute**: ~$0.79/year
- **OpenAI API**: ~$8.52/year (initial) + ~$0.52/year (weekly)
- **Storage**: ~$2.45/year (10 GB)

**Total first year**: ~$12/year

## Testing

### Cost Monitoring Setup

Before running tests, set up budget alerts:

```bash
./deployment/scripts/05-setup-cost-monitoring.sh --budget 20
```

This creates a monthly budget with alerts at 50%, 75%, and 100% thresholds.

### Running Tests

#### Local Container Testing

Test the container locally before deploying to GCP:

```bash
# Test locally with 10 documents
./deployment/cloud-run/test-container-local.sh
```

This runs the full ETL orchestrator locally in Docker, useful for:
- Validating code changes before deployment
- Testing without GCP costs
- Debugging container issues locally

#### GCP VM Testing

Build the Docker image before running GCP tests:
```bash
./deployment/cloud-run/deploy.sh
```

**Phase 1 (10 publications, cost threshold: $1.00):**
```bash
python deployment/vm/test_batch_processing.py --limit 10 --phase 1
```

**Phase 2 (100 publications, cost threshold: $10.00):**
```bash
python deployment/vm/test_batch_processing.py --limit 100 --phase 2
```

Tests include active cost monitoring and automatic VM termination if thresholds are exceeded.

Test reports are saved to `deployment/TEST_RESULTS_PHASE*.md`.

### Calculating Costs

Query costs for a specific time period:

```bash
# Last 24 hours
./deployment/scripts/calculate-costs.sh

# Last 7 days
./deployment/scripts/calculate-costs.sh --days 7

# Since specific timestamp
./deployment/scripts/calculate-costs.sh --since "2025-01-01T00:00:00Z"
```

## Troubleshooting

### Common Issues

#### 1. VM Fails to Start or Pull Docker Image

**Symptoms**: VM created but startup script fails, or Docker image pull fails

**Solution**:
```bash
# Check serial port output
gcloud compute instances get-serial-port-output etl-pipeline-vm --zone=us-central1-a

# Verify Docker image exists in Artifact Registry
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}

# Ensure image was built and pushed
./deployment/cloud-run/deploy.sh

# Check VM metadata includes image-name
gcloud compute instances describe etl-pipeline-vm --zone=us-central1-a --format="value(metadata.items[image-name])"

# SSH into VM for debugging (if using Ubuntu image)
gcloud compute ssh etl-pipeline-vm --zone=us-central1-a
```

#### 2. Secret Manager Access Denied

**Symptoms**: `403 Forbidden` when accessing secrets

**Solution**:
```bash
# Verify service account permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --flatten="bindings[].members" \
  --filter="bindings.members:serviceAccount:${SA_EMAIL}"

# Re-grant secret access (use Cloud Console if CLI fails)
# See: https://console.cloud.google.com/security/secret-manager
```

#### 3. Cloud Run Job Timeout

**Symptoms**: Job times out after 2 hours

**Solution**:
```bash
# Increase timeout (max 24 hours)
gcloud run jobs update etl-pipeline-job \
  --region=us-central1 \
  --task-timeout=6h
```

#### 4. Out of Memory

**Symptoms**: Container crashes with OOM error

**Solution**:
```bash
# Increase memory allocation
gcloud run jobs update etl-pipeline-job \
  --region=us-central1 \
  --memory=16Gi
```

### Debugging Commands

```bash
# Check VM status
gcloud compute instances describe etl-pipeline-vm --zone=us-central1-a

# View Cloud Run Job executions
gcloud run jobs executions list \
  --job=etl-pipeline-job \
  --region=us-central1

# Check storage bucket contents
gcloud storage ls -r gs://$BUCKET_NAME/

# View recent error logs
gcloud logging read "severity>=ERROR" --limit 50

# Docker-specific debugging
# List container images in Artifact Registry
gcloud artifacts docker images list ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}

# Test Docker image locally (if needed)
docker pull ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/etl-pipeline:latest
docker run --rm -e GCS_BUCKET=$BUCKET_NAME -e ENVIRONMENT=production \
  ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO}/etl-pipeline:latest \
  --config backend/etl/config.production.yaml --scraper-limit 1
```

## Local Testing

### Test Container Locally

Test the ETL pipeline container locally (without GCP):

```bash
# Test with 10 documents (builds image if needed)
./deployment/cloud-run/test-container-local.sh
```

This script:
- Builds the Docker image for your local platform (ARM64 on Mac, AMD64 on Linux)
- Runs the container with a 10-document limit
- Uses local data directory for outputs
- Requires `OPENAI_API_KEY` environment variable

**Prerequisites:**
- Docker installed
- `OPENAI_API_KEY` environment variable set
- GCP credentials configured (for GCS access if using cloud storage)

### Test with Docker Compose

Alternatively, test using Docker Compose (requires `docker-compose.yml` in the repository root):

```bash
# Run with default command
docker-compose up

# Run with custom arguments
docker-compose run etl-pipeline --scraper-limit 10

# Rebuild image and run
docker-compose up --build
```

The `docker-compose.yml` file in the repository root provides a convenient way to test the ETL pipeline locally using the same Docker container image that will be deployed to Cloud Run and VM instances. It mounts local data and logs directories for easy access to outputs.

## Security Best Practices

1. **Never commit secrets**: The `gcp-config.sh` file is gitignored
2. **Use Secret Manager**: All API keys stored in Secret Manager, not in code
3. **Principle of least privilege**: Service account has only necessary permissions
4. **Enable versioning**: Critical data buckets have versioning enabled
5. **Monitor access**: Regularly review IAM policies and access logs

## Maintenance

### Updating the Pipeline

When code changes:

```bash
# Rebuild and redeploy (default: local build)
./deployment/cloud-run/deploy.sh

# Or use Cloud Build (faster, no local Docker required)
./deployment/cloud-run/deploy.sh --cloud-build
```

VMs automatically pull the latest Docker image on next run. Cloud Run Jobs use the updated image immediately after deployment.

### Updating Dependencies

If `pyproject.toml` changes:

```bash
# Update lock file locally
uv lock

# Commit changes
git add uv.lock pyproject.toml
git commit -m "Update dependencies"
git push

# Rebuild container
./deployment/cloud-run/deploy.sh
```

### Updating Configuration

To modify ETL behavior without code changes:

1. Edit `backend/etl/config.production.yaml`
2. Commit and push changes
3. Redeploy: `./deployment/cloud-run/deploy.sh`

## Monitoring

### Cloud Logging

View logs in real-time:

```bash
# Stream Cloud Run Job logs
gcloud logging tail \
  "resource.type=cloud_run_job AND resource.labels.job_name=etl-pipeline-job"

# Stream VM logs
gcloud logging tail \
  "resource.type=gce_instance AND resource.labels.instance_id=etl-pipeline-vm"
```

### Cloud Monitoring

Create a dashboard in Cloud Console:
1. Go to Monitoring > Dashboards
2. Add widgets for:
   - Cloud Run Job execution count
   - Cloud Run Job execution duration
   - Cloud Storage bucket size
   - Error rates

### Cost Monitoring

Set up budget alerts using the provided script:

```bash
./deployment/scripts/05-setup-cost-monitoring.sh --budget 20
```

This automatically creates a budget with threshold alerts at 50%, 75%, and 100% of your monthly budget. The script uses the Google Cloud Billing Budgets REST API for full configuration support.

To manually check your budget status:

```bash
gcloud billing budgets list --billing-account=$BILLING_ACCOUNT_ID
```

## Additional Resources

- [GCP Deployment Guide](../docs/GCP_DEPLOYMENT_GUIDE.md): Comprehensive deployment documentation
- [GCP Documentation](https://cloud.google.com/docs)
- [Cloud Run Jobs Documentation](https://cloud.google.com/run/docs/create-jobs)
- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs using the debugging commands
3. Open an issue in the GitHub repository
4. Contact the Growth Lab engineering team
