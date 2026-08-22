#!/bin/bash
#SBATCH --job-name=gl-etl-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/etl_pipeline_%j.out
#SBATCH --error=logs/etl_pipeline_%j.err
# Send USR1 to the batch script 15 min before the time limit so results
# can be synced off node-local scratch before SLURM kills the job.
#SBATCH --signal=B:USR1@900
# Resource notes:
#   --mem is CPU RAM only (not GPU VRAM). GPU VRAM is determined by hardware.
#   On the gpu partition, all nodes have A100s (40-80GB VRAM), which is 3-5x
#   what Marker needs (~8-14GB peak). No need to constrain GPU type.
#   Marker on an A100 runs ~145s/PDF, measured twice: the 2026-02-25 full run
#   (331 PDFs in 13.1h = 142s each) and the 2026-08-21 smoke test (4 PDFs in
#   11m13s including model load). With OpenAlex now in scope the corpus is
#   ~520 PDFs, so extraction alone is ~21h and the full run lands at ~23-24h.
#   24h was cutting it too close; 48h is still well inside the partition's
#   3-day ceiling. Jobs are billed for time used, not time requested.
#
#   --mem is deliberately left at 100G despite the smoke test peaking at 4.96G
#   MaxRSS. That evidence is not representative (4 small PDFs; max_pages is
#   250), and on this partition memory is not the binding constraint: each node
#   has 4 A100s, so at 1 GPU per job the GPU count caps packing long before
#   4x100G does. Re-check with `jobstats` after a full run before trimming.

# Growth Lab Deep Search - Full ETL Pipeline via Singularity
#
# Runs the complete pipeline: scrape → download → PDF extract → chunk → embed
# Uses Singularity with GPU passthrough for Marker PDF processing.
#
# Usage:
#   # Full production run (all sources, no limits):
#   sbatch deployment/slurm/etl_pipeline.sbatch
#
#   # Test run with limits:
#   SCRAPER_LIMIT=10 DOWNLOAD_LIMIT=10 sbatch deployment/slurm/etl_pipeline.sbatch
#
#   # Smoke test (also caps the PDFs processed, not just those downloaded):
#   SCRAPER_LIMIT=5 DOWNLOAD_LIMIT=5 PDF_LIMIT=5 TRANSCRIPTS_LIMIT=1 \
#       sbatch deployment/slurm/etl_pipeline.sbatch
#
#   # Run specific sources:
#   SOURCES=lectures sbatch deployment/slurm/etl_pipeline.sbatch
#   SOURCES="openalex lectures" sbatch deployment/slurm/etl_pipeline.sbatch
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

SOURCES="${SOURCES:-all}"
SCRAPER_LIMIT="${SCRAPER_LIMIT:-}"
DOWNLOAD_LIMIT="${DOWNLOAD_LIMIT:-}"
PDF_LIMIT="${PDF_LIMIT:-}"
TRANSCRIPTS_LIMIT="${TRANSCRIPTS_LIMIT:-}"
SKIP_SCRAPING="${SKIP_SCRAPING:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ── Job info ─────────────────────────────────────────────────────────
echo "=== ETL Pipeline Job ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "Sources:       $SOURCES"
echo "Scraper limit: ${SCRAPER_LIMIT:-unlimited}"
echo "Download limit:${DOWNLOAD_LIMIT:-unlimited}"
echo "PDF limit:     ${PDF_LIMIT:-unlimited}"
echo "Transcripts:   ${TRANSCRIPTS_LIMIT:-unlimited}"
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
    echo "WARNING: OPENAI_API_KEY not set. OpenAI-based steps will fail."
    echo "Embeddings will use local SentenceTransformer model (no API key needed)."
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

# ── Guard against a stale image ──────────────────────────────────────
# On 2026-02-25 a job died two minutes in because the .sif predated the commit
# that added --sources: the container's Python had never seen the flag. Compare
# the commit baked into the image against the cluster checkout and stop early.
# Override with ALLOW_STALE_IMAGE=1 when you knowingly want the older code.
REPO_SHA="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
IMAGE_SHA="$(singularity exec "$SIF_IMAGE" cat /app/GIT_SHA 2>/dev/null | tr -d '[:space:]' || echo unknown)"
echo "Repo commit:   $REPO_SHA"
echo "Image commit:  $IMAGE_SHA"

if [[ "$IMAGE_SHA" == "unknown" || -z "$IMAGE_SHA" ]]; then
    echo "WARNING: image carries no commit stamp (built before SHA provenance)."
    echo "         Rebuild with 'setup_env.sh build' to enable this check."
elif [[ "$REPO_SHA" == "unknown" ]]; then
    echo "WARNING: could not read the repo commit (is git on PATH?). Skipping"
    echo "         the staleness check rather than blocking the job."
elif [[ "$IMAGE_SHA" == "$REPO_SHA" ]]; then
    echo "Image matches checkout."
elif ! git -C "$PROJECT_DIR" cat-file -e "${IMAGE_SHA}^{commit}" 2>/dev/null; then
    echo "WARNING: the image's commit ($IMAGE_SHA) is not in this checkout's"
    echo "         history — cannot tell whether the baked code is current."
    echo "         Run 'git fetch', or rebuild the image."
else
    # Only the paths baked into the image matter. Changes to the sbatch script,
    # READMEs or docs do not require a rebuild, so don't block on them.
    IMAGE_PATHS=(backend pyproject.toml uv.lock deployment/pdf-processing)
    if git -C "$PROJECT_DIR" diff --quiet "$IMAGE_SHA" "$REPO_SHA" -- "${IMAGE_PATHS[@]}"; then
        echo "Image predates the checkout, but no image-relevant files changed."
        echo "Differences are confined to scripts/docs — safe to run."
    else
        echo "ERROR: container image is out of sync with the repository."
        echo "  Image was built from: $IMAGE_SHA"
        echo "  Checkout is at:       $REPO_SHA"
        echo ""
        echo "Image-relevant files that differ:"
        git -C "$PROJECT_DIR" diff --name-only "$IMAGE_SHA" "$REPO_SHA" \
            -- "${IMAGE_PATHS[@]}" | sed 's/^/    /'
        echo ""
        echo "Rebuild and re-pull:"
        echo "  (local)   bash deployment/slurm/setup_env.sh build"
        echo "  (cluster) bash deployment/slurm/setup_env.sh pull"
        echo ""
        echo "Set ALLOW_STALE_IMAGE=1 to run anyway."
        [[ "${ALLOW_STALE_IMAGE:-0}" == "1" ]] || exit 1
        echo "ALLOW_STALE_IMAGE=1 set — continuing with the stale image."
    fi
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

# ── Getting results off node-local scratch ───────────────────────────
# Scratch is fast but disappears with the job. Syncing only on exit means a
# TIMEOUT, scancel, node failure or OOM-kill discards everything — the
# 2026-08-22 run would have lost 17 hours of GPU time. So: sync incrementally
# while running, and sync again on any exit path, including signals.
SYNC_INTERVAL="${SYNC_INTERVAL:-900}"

sync_back() {
    # Never let a sync failure abort the job or mask the pipeline's exit code.
    rsync -a "$LOCAL_WORK/data/" "${PROJECT_DIR}/data/" || true
    rsync -a "$LOCAL_WORK/logs/" "${PROJECT_DIR}/logs/" || true
    rsync -a "$LOCAL_WORK/reports/" "${PROJECT_DIR}/reports/" || true
}

periodic_sync() {
    while true; do
        sleep "$SYNC_INTERVAL"
        sync_back
        echo "[$(date +%H:%M:%S)] incremental sync to persistent storage complete"
    done
}

CLEANUP_DONE=""
cleanup() {
    # EXIT fires after a signal trap has already run; only do this once.
    [[ -n "$CLEANUP_DONE" ]] && return 0
    CLEANUP_DONE=1
    if [[ -n "${SYNC_PID:-}" ]]; then kill "$SYNC_PID" 2>/dev/null || true; fi
    echo ""
    echo "=== Syncing results back to persistent storage ==="
    sync_back
    # Persist model cache so next job doesn't re-download
    rsync -a "$LOCAL_WORK/.model_cache/" "$MODEL_CACHE/" || true
    rm -rf "$LOCAL_WORK"
    echo "Scratch cleanup complete."
}

on_signal() {
    echo ""
    echo "!!! Termination signal received at $(date) - syncing before exit !!!"
    cleanup
    exit 143
}

trap cleanup EXIT
trap on_signal USR1 SIGTERM SIGINT

periodic_sync &
SYNC_PID=$!
echo "Incremental sync every ${SYNC_INTERVAL}s (pid $SYNC_PID)"

# ── Build command args ───────────────────────────────────────────────
CMD_ARGS=(
    python -m backend.etl.orchestrator
    --config "$CONFIG_FILE"
    --storage-type local
    --sources ${SOURCES}
    --log-level "$LOG_LEVEL"
)

if [[ -n "$SCRAPER_LIMIT" ]]; then CMD_ARGS+=(--scraper-limit "$SCRAPER_LIMIT"); fi
if [[ -n "$DOWNLOAD_LIMIT" ]]; then CMD_ARGS+=(--download-limit "$DOWNLOAD_LIMIT"); fi
if [[ -n "$PDF_LIMIT" ]]; then CMD_ARGS+=(--pdf-limit "$PDF_LIMIT"); fi
if [[ -n "$TRANSCRIPTS_LIMIT" ]]; then CMD_ARGS+=(--transcripts-limit "$TRANSCRIPTS_LIMIT"); fi

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
    --bind "$LOCAL_WORK/.model_cache:$LOCAL_WORK/.model_cache" \
    --env OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    --env PDF_DEVICE=cuda \
    --env MODEL_CACHE_DIR="$LOCAL_WORK/.model_cache" \
    --env HF_HOME="$LOCAL_WORK/.model_cache" \
    --env TRANSFORMERS_CACHE="$LOCAL_WORK/.model_cache" \
    --env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    --env OMP_NUM_THREADS=2 \
    --env MKL_NUM_THREADS=2 \
    --env OPENBLAS_NUM_THREADS=2 \
    "$LOCAL_SIF" \
    "${CMD_ARGS[@]}" &
PIPELINE_PID=$!

# `wait` is interruptible by traps; a foreground command is not, so signals
# would otherwise be handled only after the pipeline had already exited.
set +e
wait "$PIPELINE_PID"
PIPELINE_RC=$?
set -e
echo "Pipeline exited with code $PIPELINE_RC"

echo ""
echo "=== ETL Pipeline finished: $(date) (exit $PIPELINE_RC) ==="
echo "Report: ${PROJECT_DIR}/data/reports/etl_execution_report.json"

exit "$PIPELINE_RC"
