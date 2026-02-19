#!/bin/bash
#SBATCH --job-name=gl-pdf-process
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
#   - Singularity image built via: bash deployment/slurm/setup_env.sh all
#   - Data available at $HOME/gl_deep_search/data/

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start: $(date)"
echo "================"

PROJECT_DIR="$HOME/gl_deep_search"
SIF_IMAGE="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.sif"

if [[ ! -f "$SIF_IMAGE" ]]; then
    echo "ERROR: Singularity image not found at $SIF_IMAGE"
    echo "Run setup_env.sh first to build and push the image."
    exit 1
fi

module load singularity 2>/dev/null || true

singularity exec --nv \
    --bind "${PROJECT_DIR}/data:/app/data" \
    --bind "${PROJECT_DIR}/logs:/app/logs" \
    --bind "${PROJECT_DIR}/reports:/app/reports" \
    --env PDF_DEVICE=cuda \
    "$SIF_IMAGE" \
    python -m backend.etl.scripts.run_pdf_processor --force-reprocess

echo "=== Completed: $(date) ==="
