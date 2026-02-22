#!/bin/bash
#SBATCH --job-name=gl-pdf-benchmark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

# Growth Lab Deep Search - PDF Backend Benchmark via Singularity
#
# Compares Marker and Docling backends on the cluster.
# Uses the same containerized environment as pdf_processing.sbatch.

set -euo pipefail

echo "=== Benchmark Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "=========================="

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
echo ""
echo "=== CUDA Compatibility Check ==="
DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
echo "Host driver supports CUDA: $DRIVER_CUDA"
echo "================"

PROJECT_DIR="$HOME/gl_deep_search"
SIF_IMAGE="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.sif"
TAR_IMAGE="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.tar"

# ── Container image (auto-convert tar → sif on first run) ────────────
module load singularity 2>/dev/null || true

if [[ ! -f "$SIF_IMAGE" ]]; then
    if [[ -f "$TAR_IMAGE" ]]; then
        echo "Converting Docker tar to Singularity SIF (one-time)..."
        singularity build "$SIF_IMAGE" "docker-archive://${TAR_IMAGE}"
        echo "SIF image created at $SIF_IMAGE"
    else
        echo "ERROR: No container image found (.sif or .tar) at ${PROJECT_DIR}/deployment/slurm/"
        echo "Run: bash deployment/slurm/setup_env.sh all"
        exit 1
    fi
fi

singularity exec --nv \
    --bind "${PROJECT_DIR}/data:/app/data" \
    --bind "${PROJECT_DIR}/logs:/app/logs" \
    --bind "${PROJECT_DIR}/reports:/app/reports" \
    --env PDF_DEVICE=cuda \
    "$SIF_IMAGE" \
    python -m backend.etl.scripts.benchmark_pdf_backends \
        --pdf-dir /app/data/raw/documents/growthlab \
        --sample-size 20 \
        --backends marker docling \
        --output "/app/reports/benchmark_${SLURM_JOB_ID}.json"

echo "=== Benchmark completed: $(date) ==="
