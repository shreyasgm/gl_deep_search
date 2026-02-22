# Running the ETL Pipeline on FASRC (SLURM)

This guide covers deploying and running the full ETL pipeline (scrape → download → PDF extract → chunk → embed) on Harvard's FASRC cluster using Singularity containers and SLURM job scheduling.

## Prerequisites

- Docker installed on your local machine
- SSH access to `login.rc.fas.harvard.edu`
- An OpenAI API key (for the embeddings stage)

## Step 1: Build the Docker image locally

```bash
cd "/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search"

bash deployment/slurm/setup_env.sh build
```

This builds the Docker image and exports it as a `.tar` file (~2.5 GB) at `deployment/slurm/gl-pdf-processing.tar`.

## Step 2: Transfer to the cluster

```bash
# Create directories on the cluster (first time only)
bash deployment/slurm/setup_env.sh setup

# Transfer the .tar + configs + sbatch scripts
bash deployment/slurm/setup_env.sh push
```

Or do both build + push in one step:

```bash
bash deployment/slurm/setup_env.sh all
```

The `.tar` is automatically converted to a Singularity `.sif` image on the first `sbatch` run — no manual `singularity build` step needed.

## Step 3: Set your OpenAI API key

The embeddings stage calls the OpenAI API. Add your key to `~/.bashrc` so SLURM jobs can access it:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

## Step 4: Submit a job

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

## Step 5: Monitor the job

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
| Memory | 100 GB | Marker models + PDF processing headroom |
| Time limit | 4 hours | Plenty for small runs; increase for full corpus |

Marker auto-detects the GPU and uses optimal batch sizes for the available hardware. A 10-publication test run should complete in ~10-15 minutes.

## GPU and CUDA compatibility

The container uses `python:3.12-slim-bookworm` (not an NVIDIA CUDA base image). This works because:

- `singularity exec --nv` mounts the host's NVIDIA drivers into the container at runtime
- PyTorch bundles its own CUDA runtime (libcudart, cuDNN, cuBLAS)

The sbatch scripts log the host driver's max supported CUDA version. If you see CUDA errors, verify that the PyTorch CUDA version in the container is <= the host driver's reported CUDA version.

## Other SLURM scripts

- **`pdf_processing.sbatch`** — Runs only the PDF extraction stage (useful for reprocessing)
- **`benchmark.sbatch`** — Benchmarks Marker vs Docling backends on a sample of PDFs

## Updating the image

When the code changes, rebuild and re-transfer:

```bash
# Local: rebuild + push
bash deployment/slurm/setup_env.sh all

# On cluster: delete the old .sif so next sbatch auto-rebuilds
rm ~/gl_deep_search/deployment/slurm/gl-pdf-processing.sif
```
