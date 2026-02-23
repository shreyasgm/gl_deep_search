#!/bin/bash
#SBATCH --job-name=gl-etl-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/etl_pipeline_%j.out
#SBATCH --error=logs/etl_pipeline_%j.err
# Resource notes:
#   --mem is CPU RAM only (not GPU VRAM). GPU VRAM is determined by hardware.
#   On the gpu partition, all nodes have A100s (40-80GB VRAM), which is 3-5x
#   what Marker needs (~8-14GB peak). No need to constrain GPU type.
#   Increase --time if processing >50 publications.

# Growth Lab Deep Search - Full ETL Pipeline via Singularity
#
# Runs the complete pipeline: scrape → download → PDF extract → chunk → embed
# Uses Singularity with GPU passthrough for Marker PDF processing.
#
# Usage:
#   # Test run (10 publications):
#   sbatch deployment/slurm/etl_pipeline.sbatch
#
#   # Override limits via environment variables:
#   SCRAPER_LIMIT=50 DOWNLOAD_LIMIT=50 sbatch deployment/slurm/etl_pipeline.sbatch
#
#   # Skip scraping (use existing data):
#   SKIP_SCRAPING=1 sbatch deployment/slurm/etl_pipeline.sbatch
#
# Prerequisites:
#   - Container image (.sif) at $PROJECT_DIR/deployment/slurm/
#     (run: bash deployment/slurm/setup_env.sh pull)
#   - OPENAI_API_KEY in $PROJECT_DIR/.env or shell environment

set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────
PROJECT_DIR="${PROJECT_DIR:-/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search}"
SIF_IMAGE="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.sif"
CONFIG_FILE="backend/etl/config.yaml"

SCRAPER_LIMIT="${SCRAPER_LIMIT:-10}"
DOWNLOAD_LIMIT="${DOWNLOAD_LIMIT:-10}"
SKIP_SCRAPING="${SKIP_SCRAPING:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ── Job info ─────────────────────────────────────────────────────────
echo "=== ETL Pipeline Job ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Scraper limit: $SCRAPER_LIMIT"
echo "Download limit:$DOWNLOAD_LIMIT"
echo "Skip scraping: $SKIP_SCRAPING"
echo "Start:         $(date)"
echo "========================"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
echo "=== CUDA Compatibility Check ==="
DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Host driver supports CUDA: $DRIVER_CUDA"
echo "================"

# ── Load secrets ─────────────────────────────────────────────────────
ENV_FILE="${PROJECT_DIR}/.env"
if [[ -f "$ENV_FILE" ]]; then
    echo "Loading environment from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Create $ENV_FILE with:  OPENAI_API_KEY=sk-..."
    exit 1
fi

# ── Ensure directories exist ─────────────────────────────────────────
mkdir -p "${PROJECT_DIR}"/{data,logs,reports}

# ── Load Singularity ─────────────────────────────────────────────────
module load singularity 2>/dev/null || true

# ── Container image ──────────────────────────────────────────────────
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "ERROR: Container image not found at $SIF_IMAGE"
    echo "Run on the cluster: bash deployment/slurm/setup_env.sh pull"
    exit 1
fi

# ── Stage to local scratch (avoids slow network filesystem I/O) ──────
# GPU nodes have ~396 GB of local scratch — much faster than holystore NFS.
LOCAL_WORK="/scratch/gl_${SLURM_JOB_ID}"
MODEL_CACHE="${PROJECT_DIR}/.model_cache"
mkdir -p "$LOCAL_WORK"/{data,logs,reports} "$MODEL_CACHE"

echo "Staging container image to local scratch..."
cp "$SIF_IMAGE" "$LOCAL_WORK/gl-pdf-processing.sif"
LOCAL_SIF="$LOCAL_WORK/gl-pdf-processing.sif"

echo "Staging data to local scratch..."
rsync -a "${PROJECT_DIR}/data/" "$LOCAL_WORK/data/"

# Stage model cache if it exists (avoids re-downloading 4.6 GB of models)
if [[ -d "$MODEL_CACHE" && "$(ls -A "$MODEL_CACHE" 2>/dev/null)" ]]; then
    echo "Staging model cache to local scratch..."
    rsync -a "$MODEL_CACHE/" "$LOCAL_WORK/.model_cache/"
else
    mkdir -p "$LOCAL_WORK/.model_cache"
fi
echo "Staging complete. Local scratch usage: $(du -sh "$LOCAL_WORK" | cut -f1)"

# Copy results back to persistent storage on exit (success or failure)
cleanup() {
    echo ""
    echo "=== Syncing results back to persistent storage ==="
    rsync -a "$LOCAL_WORK/data/" "${PROJECT_DIR}/data/"
    rsync -a "$LOCAL_WORK/logs/" "${PROJECT_DIR}/logs/"
    rsync -a "$LOCAL_WORK/reports/" "${PROJECT_DIR}/reports/"
    # Persist model cache so next job doesn't re-download
    rsync -a "$LOCAL_WORK/.model_cache/" "$MODEL_CACHE/"
    rm -rf "$LOCAL_WORK"
    echo "Scratch cleanup complete."
}
trap cleanup EXIT

# ── Build command args ───────────────────────────────────────────────
CMD_ARGS=(
    python -m backend.etl.orchestrator
    --config "$CONFIG_FILE"
    --storage-type local
    --scraper-limit "$SCRAPER_LIMIT"
    --download-limit "$DOWNLOAD_LIMIT"
    --log-level "$LOG_LEVEL"
)

if [[ "$SKIP_SCRAPING" == "1" ]]; then
    CMD_ARGS+=(--skip-scraping)
fi

echo "Command: ${CMD_ARGS[*]}"
echo ""

# ── Run pipeline ─────────────────────────────────────────────────────
singularity exec --nv --writable-tmpfs --pwd /app \
    --bind "$LOCAL_WORK/data:/app/data" \
    --bind "$LOCAL_WORK/logs:/app/logs" \
    --bind "$LOCAL_WORK/reports:/app/reports" \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --env PDF_DEVICE=cuda \
    --env MODEL_CACHE_DIR="$LOCAL_WORK/.model_cache" \
    --env OMP_NUM_THREADS=2 \
    --env MKL_NUM_THREADS=2 \
    --env OPENBLAS_NUM_THREADS=2 \
    "$LOCAL_SIF" \
    "${CMD_ARGS[@]}"

echo ""
echo "=== ETL Pipeline completed: $(date) ==="
echo "Report: ${PROJECT_DIR}/data/reports/etl_execution_report.json"
