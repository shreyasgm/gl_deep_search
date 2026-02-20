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
#   - Singularity image at $HOME/gl_deep_search/deployment/slurm/gl-pdf-processing.sif
#   - OPENAI_API_KEY in ~/.bashrc or passed via --export
#   - Run setup_env.sh first to build and push the image

set -euo pipefail

# ── Configuration (override via environment) ─────────────────────────
PROJECT_DIR="$HOME/gl_deep_search"
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
echo "================"

# ── Validation ───────────────────────────────────────────────────────
if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "ERROR: Singularity image not found at $SIF_IMAGE"
    echo "Run: bash deployment/slurm/setup_env.sh all"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY not set."
    echo "Add to ~/.bashrc:  export OPENAI_API_KEY='sk-...'"
    exit 1
fi

# ── Ensure directories exist ─────────────────────────────────────────
mkdir -p "${PROJECT_DIR}"/{data,logs,reports}

# ── Load Singularity ─────────────────────────────────────────────────
module load singularity 2>/dev/null || true

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
singularity exec --nv \
    --bind "${PROJECT_DIR}/data:/app/data" \
    --bind "${PROJECT_DIR}/logs:/app/logs" \
    --bind "${PROJECT_DIR}/reports:/app/reports" \
    --env OPENAI_API_KEY="$OPENAI_API_KEY" \
    --env PDF_DEVICE=cuda \
    --env OMP_NUM_THREADS=2 \
    --env MKL_NUM_THREADS=2 \
    --env OPENBLAS_NUM_THREADS=2 \
    "$SIF_IMAGE" \
    "${CMD_ARGS[@]}"

echo ""
echo "=== ETL Pipeline completed: $(date) ==="
echo "Report: ${PROJECT_DIR}/data/reports/etl_execution_report.json"
