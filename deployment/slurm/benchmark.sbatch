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

PROJECT_DIR="${PROJECT_DIR:-/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search}"
SIF_IMAGE="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.sif"

# ── Container image ──────────────────────────────────────────────────
module load singularity 2>/dev/null || true

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
    rsync -a "$LOCAL_WORK/.model_cache/" "$MODEL_CACHE/"
    rm -rf "$LOCAL_WORK"
    echo "Scratch cleanup complete."
}
trap cleanup EXIT

singularity exec --nv --writable-tmpfs --pwd /app \
    --bind "$LOCAL_WORK/data:/app/data" \
    --bind "$LOCAL_WORK/logs:/app/logs" \
    --bind "$LOCAL_WORK/reports:/app/reports" \
    --env PDF_DEVICE=cuda \
    --env MODEL_CACHE_DIR="$LOCAL_WORK/.model_cache" \
    "$LOCAL_SIF" \
    python -m backend.etl.scripts.benchmark_pdf_backends \
        --pdf-dir /app/data/raw/documents/growthlab \
        --sample-size 20 \
        --backends marker docling \
        --output "/app/reports/benchmark_${SLURM_JOB_ID}.json"

echo "=== Benchmark completed: $(date) ==="
