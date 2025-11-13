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

3. **Deploy Cloud Run Job:**
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
├── config/
│   ├── gcp-config.sh.template   # GCP configuration template
│   └── lifecycle-policy.json   # GCS lifecycle rules
├── scripts/
│   ├── 01-setup-gcp-project.sh  # GCP project setup
│   ├── 02-setup-storage.sh      # Storage bucket setup
│   ├── 03-setup-secrets.sh      # Secret Manager setup
│   ├── 04-create-service-account.sh  # Service account setup
│   └── utils.sh                 # Shared utilities
├── vm/
│   ├── startup-script.sh        # VM initialization script
│   ├── incremental-update.sh    # Weekly update script
│   ├── create-vm.sh             # Create VM instance
│   └── monitor-vm.sh            # Monitor VM execution
├── cloud-run/
│   ├── Dockerfile               # Container image definition
│   ├── .dockerignore            # Docker ignore rules
│   ├── deploy.sh                # Deploy Cloud Run Job
│   ├── schedule.sh              # Setup Cloud Scheduler
│   └── execute.sh               # Manual job execution
└── workflows/
    └── etl-weekly.yml           # GitHub Actions workflow (optional)
```

## Prerequisites

- **GCP Account**: Active Google Cloud Platform account with billing enabled
- **gcloud CLI**: Installed and authenticated (`gcloud auth login`)
- **Docker**: Installed (for Cloud Run deployments)
- **Git**: Repository cloned locally
- **OpenAI API Key**: For embeddings generation

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

### Step 3: Deploy Cloud Run Job

Build and deploy the containerized ETL pipeline:

```bash
./deployment/cloud-run/deploy.sh
```

This will:
- Build a Docker container image
- Push it to Artifact Registry
- Create/update a Cloud Run Job

### Step 4: Schedule Weekly Updates

Set up automatic weekly execution:

```bash
./deployment/cloud-run/schedule.sh
```

This creates a Cloud Scheduler job that runs every Sunday at 2 AM (configurable).

### Step 5: Initial Batch Processing (Optional)

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

## Cost Estimation

Based on the deployment guide:

- **Initial batch (1,400 PDFs)**: ~$0.29 (spot VM)
- **Weekly updates**: <$0.01 per run
- **Annual compute**: ~$0.79/year
- **OpenAI API**: ~$8.52/year (initial) + ~$0.52/year (weekly)
- **Storage**: ~$2.45/year (10 GB)

**Total first year**: ~$12/year

## Troubleshooting

### Common Issues

#### 1. VM Fails to Start

**Symptoms**: VM created but startup script doesn't execute

**Solution**:
```bash
# Check serial port output
gcloud compute instances get-serial-port-output etl-pipeline-vm --zone=us-central1-a

# SSH into VM for debugging
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
```

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
# Rebuild and redeploy Cloud Run Job
./deployment/cloud-run/deploy.sh

# VM will automatically pull latest code on next run
```

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

Set up budget alerts:

```bash
gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT_ID \
  --display-name="ETL Pipeline Budget" \
  --budget-amount=20 \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=90 \
  --threshold-rule=percent=100
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
