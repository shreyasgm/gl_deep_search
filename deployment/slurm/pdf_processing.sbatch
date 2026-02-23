#!/bin/bash
#SBATCH --job-name=gl-pdf-process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=logs/pdf_processing_%j.out
#SBATCH --error=logs/pdf_processing_%j.err

# Growth Lab Deep Search - PDF Processing via Singularity
#
# Runs the full PDF extraction pipeline on FAS-RC with GPU acceleration.
# The container image includes Marker (primary) and Docling (fallback).
# Singularity --nv passes through host NVIDIA drivers for GPU access.
#
# Prerequisites:
#   - Container image (.sif) pulled via: bash deployment/slurm/setup_env.sh pull
#   - Data available at $PROJECT_DIR/data/

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "================"

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
mkdir -p "$LOCAL_WORK"/{data,logs,reports}

echo "Staging container image to local scratch..."
cp "$SIF_IMAGE" "$LOCAL_WORK/gl-pdf-processing.sif"
LOCAL_SIF="$LOCAL_WORK/gl-pdf-processing.sif"

echo "Staging data to local scratch..."
rsync -a "${PROJECT_DIR}/data/" "$LOCAL_WORK/data/"
echo "Staging complete. Local scratch usage: $(du -sh "$LOCAL_WORK" | cut -f1)"

# Copy results back to persistent storage on exit (success or failure)
cleanup() {
    echo ""
    echo "=== Syncing results back to persistent storage ==="
    rsync -a "$LOCAL_WORK/data/" "${PROJECT_DIR}/data/"
    rsync -a "$LOCAL_WORK/logs/" "${PROJECT_DIR}/logs/"
    rsync -a "$LOCAL_WORK/reports/" "${PROJECT_DIR}/reports/"
    rm -rf "$LOCAL_WORK"
    echo "Scratch cleanup complete."
}
trap cleanup EXIT

singularity exec --nv \
    --bind "$LOCAL_WORK/data:/app/data" \
    --bind "$LOCAL_WORK/logs:/app/logs" \
    --bind "$LOCAL_WORK/reports:/app/reports" \
    --env PDF_DEVICE=cuda \
    "$LOCAL_SIF" \
    python -m backend.etl.scripts.run_pdf_processor --force-reprocess

echo "=== Completed: $(date) ==="
