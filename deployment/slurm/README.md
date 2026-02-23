# Running the ETL Pipeline on FASRC (SLURM)

This guide covers deploying and running the full ETL pipeline (scrape → download → PDF extract → chunk → embed) on Harvard's FASRC cluster using Singularity containers and SLURM job scheduling.

## Prerequisites

- `gcloud` CLI installed and authenticated locally
- SSH access to `login.rc.fas.harvard.edu`
- An OpenAI API key (for the embeddings stage)

## One-Time Cluster Setup

### 1. Create directories on the cluster

```bash
bash deployment/slurm/setup_env.sh setup
```

### 2. Set up Artifact Registry authentication

The cluster needs a GCP service account key to pull container images from Artifact Registry.

```bash
# Local: create the key
gcloud iam service-accounts keys create sa-key.json \
  --iam-account=etl-pipeline-service-account@cid-hks-1537286359734.iam.gserviceaccount.com

# Local: copy to cluster
scp sa-key.json ${USER}@login.rc.fas.harvard.edu:/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search/.gcp-sa-key.json

# On cluster: secure the key
chmod 600 /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search/.gcp-sa-key.json

# Local: clean up
rm sa-key.json
```

### 3. Set your OpenAI API key

Add your key to the `.env` file on the cluster:

```bash
# On cluster:
echo 'OPENAI_API_KEY=sk-...' >> /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search/.env
```

## Deployment Workflow

### Step 1: Build the container image (local)

Submits to Google Cloud Build, which builds a `linux/amd64` image and pushes it to Artifact Registry.

```bash
bash deployment/slurm/setup_env.sh build
```

### Step 2: Update code on cluster (on cluster)

The cluster has a git clone of the repo. Pull to get the latest configs and scripts:

```bash
cd /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search
git pull
```

### Step 3: Pull the container image (on cluster)

Pull the image from Artifact Registry as a Singularity `.sif`:

```bash
bash deployment/slurm/setup_env.sh pull
```

## Running Jobs

```bash
cd /n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search

# Test run with 10 publications (default)
sbatch deployment/slurm/etl_pipeline.sbatch

# Override limits via environment variables
SCRAPER_LIMIT=50 DOWNLOAD_LIMIT=50 sbatch deployment/slurm/etl_pipeline.sbatch

# Skip scraping and reuse existing data
SKIP_SCRAPING=1 sbatch deployment/slurm/etl_pipeline.sbatch
```

### Environment variable overrides

| Variable | Default | Description |
|---|---|---|
| `SCRAPER_LIMIT` | 10 | Max publications to scrape |
| `DOWNLOAD_LIMIT` | 10 | Max publications to download files for |
| `SKIP_SCRAPING` | 0 | Set to `1` to skip scraping and use existing CSV |
| `LOG_LEVEL` | INFO | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f logs/etl_pipeline_<JOB_ID>.out

# Check for errors
cat logs/etl_pipeline_<JOB_ID>.err

# After completion, check the pipeline report
cat data/reports/etl_execution_report.json
```

## Resource Allocation

The `etl_pipeline.sbatch` script requests:

| Resource | Value | Why |
|---|---|---|
| Partition | `gpu` | A100 GPU for Marker PDF processing |
| GPU | 1 | Marker CUDA acceleration |
| CPUs | 8 | Parallel downloads and text extraction |
| Memory | 100 GB | Marker models + PDF processing headroom |
| Time limit | 4 hours | Plenty for small runs; increase for full corpus |

## Other SLURM Scripts

- **`pdf_processing.sbatch`** — Runs only the PDF extraction stage (useful for reprocessing)
- **`benchmark.sbatch`** — Benchmarks Marker vs Docling backends on a sample of PDFs

## Updating the Image

When code changes, rebuild and re-pull:

```bash
# Local: rebuild via Cloud Build
bash deployment/slurm/setup_env.sh build

# On cluster: get latest code + pull new image
git pull
bash deployment/slurm/setup_env.sh pull
```
