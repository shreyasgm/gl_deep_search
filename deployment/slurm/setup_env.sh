#!/bin/bash
# Growth Lab Deep Search - SLURM deployment helper
#
# Builds the Docker image via Google Cloud Build, pushes config files to the
# FASRC cluster, and pulls the container image from Artifact Registry.
#
# Usage (local machine):
#   bash deployment/slurm/setup_env.sh build   # Submit Cloud Build job
#   bash deployment/slurm/setup_env.sh push    # SCP configs + scripts to cluster
#
# Usage (on cluster):
#   bash deployment/slurm/setup_env.sh pull    # Pull image from Artifact Registry

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# ── GCP configuration ────────────────────────────────────────────────
GCP_CONFIG="${PROJECT_DIR}/deployment/config/gcp-config.sh"
if [[ -f "$GCP_CONFIG" ]]; then
    SKIP_VALIDATION=true source "$GCP_CONFIG"
fi
# Fallback defaults if gcp-config.sh doesn't exist or doesn't set these
SLURM_IMAGE_NAME="${SLURM_IMAGE_NAME:-us-east4-docker.pkg.dev/cid-hks-1537286359734/etl-pipeline/gl-pdf-processing:latest}"

# ── Cluster configuration ────────────────────────────────────────────
CLUSTER_HOST="${CLUSTER_HOST:-${USER}@login.rc.fas.harvard.edu}"
CLUSTER_DIR="${CLUSTER_DIR:-/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search}"

# ── Commands ─────────────────────────────────────────────────────────

build() {
    echo "=== Submitting Cloud Build job ==="
    cd "$PROJECT_DIR"
    gcloud builds submit \
        --config deployment/cloudbuild-slurm.yaml \
        .
    echo ""
    echo "Build complete. Image pushed to: ${SLURM_IMAGE_NAME}"
}

push() {
    echo "=== Pushing config and scripts to cluster ==="

    # Config file
    scp "${PROJECT_DIR}/backend/etl/config.yaml" \
        "${CLUSTER_HOST}:${CLUSTER_DIR}/backend/etl/config.yaml"

    # SLURM scripts
    scp "${PROJECT_DIR}/deployment/slurm/etl_pipeline.sbatch" \
        "${PROJECT_DIR}/deployment/slurm/pdf_processing.sbatch" \
        "${PROJECT_DIR}/deployment/slurm/benchmark.sbatch" \
        "${PROJECT_DIR}/deployment/slurm/setup_env.sh" \
        "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"

    echo "Configs and scripts pushed to cluster."
}

pull() {
    echo "=== Pulling container image from Artifact Registry ==="

    # This command runs ON the cluster
    local sif_path="${PROJECT_DIR}/deployment/slurm/gl-pdf-processing.sif"
    local sa_key="${PROJECT_DIR}/.gcp-sa-key.json"

    if [[ ! -f "$sa_key" ]]; then
        echo "ERROR: Service account key not found at $sa_key"
        echo "Copy your GCP service account key to the cluster:"
        echo "  scp sa-key.json <user>@login.rc.fas.harvard.edu:${CLUSTER_DIR}/.gcp-sa-key.json"
        exit 1
    fi

    module load singularity 2>/dev/null || true

    # Keep Singularity cache out of $HOME (100 GB quota)
    export SINGULARITY_CACHEDIR="${PROJECT_DIR}/.singularity_cache"
    mkdir -p "$SINGULARITY_CACHEDIR"

    # Remove stale image so we always get the latest
    if [[ -f "$sif_path" ]]; then
        echo "Removing old .sif image..."
        rm -f "$sif_path"
    fi

    # Authenticate with Artifact Registry via service account key
    export SINGULARITY_DOCKER_USERNAME="_json_key"
    export SINGULARITY_DOCKER_PASSWORD="$(cat "$sa_key")"

    echo "Pulling: docker://${SLURM_IMAGE_NAME}"
    singularity pull "$sif_path" "docker://${SLURM_IMAGE_NAME}"

    echo "Image saved to: $sif_path"
}

setup_cluster_dirs() {
    echo "=== Creating cluster directories ==="
    ssh "$CLUSTER_HOST" "mkdir -p ${CLUSTER_DIR}/{logs,reports,data,deployment/slurm,backend/etl}"
    echo "Cluster directories created."
}

case "${1:-}" in
    build)
        build
        ;;
    push)
        push
        ;;
    pull)
        pull
        ;;
    setup)
        setup_cluster_dirs
        ;;
    *)
        echo "Usage: $0 {build|push|pull|setup}"
        echo ""
        echo "  build  - Submit Cloud Build job (run locally)"
        echo "  push   - SCP configs and scripts to cluster (run locally)"
        echo "  pull   - Pull container image from Artifact Registry (run on cluster)"
        echo "  setup  - Create directory structure on cluster (run locally)"
        exit 1
        ;;
esac
