# Running the ETL Pipeline on FASRC (SLURM)

This guide covers deploying and running the full ETL pipeline (scrape → download → PDF extract → chunk → embed) on Harvard's FASRC cluster using Singularity containers and SLURM job scheduling.

## Prerequisites

- Docker installed on your local machine
- SSH access to `login.rc.fas.harvard.edu`
- An OpenAI API key (for the embeddings stage)

## Step 1: Build the Docker image locally

```bash
cd "/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search"

docker build \
  -f deployment/pdf-processing/Dockerfile \
  -t gl-pdf-processing:latest \
  .

# Export as a tar for transfer (~2.5 GB)
docker save gl-pdf-processing:latest -o deployment/slurm/gl-pdf-processing.tar
```

Or use the helper script:

```bash
bash deployment/slurm/setup_env.sh build
```

## Step 2: Transfer to the cluster

```bash
# Create directories on the cluster
ssh shg309@login.rc.fas.harvard.edu \
  "mkdir -p ~/gl_deep_search/{logs,reports,data,deployment/slurm,backend/etl}"

# Transfer the Docker tar (~2.5 GB)
scp deployment/slurm/gl-pdf-processing.tar \
  shg309@login.rc.fas.harvard.edu:~/gl_deep_search/deployment/slurm/

# Transfer the sbatch script and config
scp deployment/slurm/etl_pipeline.sbatch \
  shg309@login.rc.fas.harvard.edu:~/gl_deep_search/deployment/slurm/

scp backend/etl/config.yaml \
  shg309@login.rc.fas.harvard.edu:~/gl_deep_search/backend/etl/
```

Or use the helper script (does all of the above):

```bash
bash deployment/slurm/setup_env.sh push
bash deployment/slurm/setup_env.sh setup
```

## Step 3: Convert to Singularity on the cluster

SSH into the cluster and convert the Docker tar to a Singularity `.sif` image. This is a one-time step.

```bash
ssh shg309@login.rc.fas.harvard.edu
cd ~/gl_deep_search/deployment/slurm

module load singularity

# Convert Docker tar → Singularity .sif (~5-10 min)
singularity build gl-pdf-processing.sif docker-archive://gl-pdf-processing.tar

# Verify it works
singularity exec gl-pdf-processing.sif \
  python -c "from backend.etl.orchestrator import ETLOrchestrator; print('OK')"

# Clean up the tar to save disk space
rm gl-pdf-processing.tar
```

## Step 4: Set your OpenAI API key

The embeddings stage calls the OpenAI API. Add your key to `~/.bashrc` so SLURM jobs can access it:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

## Step 5: Submit a job

```bash
cd ~/gl_deep_search

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

## Step 6: Monitor the job

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

## Resource allocation

The `etl_pipeline.sbatch` script requests:

| Resource | Value | Why |
|---|---|---|
| Partition | `gpu` | A100 GPU for Marker PDF processing |
| GPU | 1 | Marker CUDA acceleration |
| CPUs | 8 | Parallel downloads and text extraction |
| Memory | 64 GB | Marker models + PDF processing headroom |
| Time limit | 4 hours | Plenty for small runs; increase for full corpus |

Marker auto-detects the GPU and uses optimal batch sizes for the available hardware. A 10-publication test run should complete in ~10-15 minutes.

## Other SLURM scripts

- **`pdf_processing.sbatch`** — Runs only the PDF extraction stage (useful for reprocessing)
- **`benchmark.sbatch`** — Benchmarks Marker vs Docling backends on a sample of PDFs

## Updating the image

When the code changes, rebuild and re-transfer:

```bash
# Local
bash deployment/slurm/setup_env.sh build
bash deployment/slurm/setup_env.sh push

# On cluster
cd ~/gl_deep_search/deployment/slurm
module load singularity
singularity build gl-pdf-processing.sif docker-archive://gl-pdf-processing.tar
rm gl-pdf-processing.tar
```
