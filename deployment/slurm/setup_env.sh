#!/bin/bash
# Growth Lab Deep Search - One-time FAS-RC environment setup
#
# Builds the PDF processing Docker image locally and exports it as a .tar
# for transfer to the SLURM cluster. The sbatch scripts auto-convert the
# .tar to a Singularity .sif on first run — no manual conversion needed.
#
# Prerequisites:
#   - Docker installed on your local machine
#   - SSH/scp access to the cluster
#
# Usage (local machine):
#   bash deployment/slurm/setup_env.sh build   # Build Docker image + export .tar
#   bash deployment/slurm/setup_env.sh push    # Push .tar + configs to cluster

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_NAME="gl-pdf-processing"
IMAGE_TAG="latest"
DOCKER_TAR="${PROJECT_DIR}/deployment/slurm/${IMAGE_NAME}.tar"
# Update this to your cluster login node
CLUSTER_HOST="${CLUSTER_HOST:-${USER}@login.rc.fas.harvard.edu}"
CLUSTER_DIR="${CLUSTER_DIR:-/n/holystore01/LABS/hausmann_lab/users/shreyasgm/gl_deep_search}"

build() {
    echo "=== Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG} (linux/amd64) ==="
    cd "$PROJECT_DIR"
    docker buildx build \
        --platform linux/amd64 \
        -f deployment/pdf-processing/Dockerfile \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        --load \
        .
    echo "Docker image built successfully."

    echo "=== Exporting Docker image as tar ==="
    docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$DOCKER_TAR"
    echo "Docker image saved to: $DOCKER_TAR"
}

push() {
    echo "=== Pushing to cluster ==="

    if [[ ! -f "$DOCKER_TAR" ]]; then
        echo "No .tar file found. Run 'build' first."
        exit 1
    fi

    scp "$DOCKER_TAR" "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
    echo "Pushed .tar to cluster (auto-converts to .sif on first sbatch run)."

    # Also push config and sbatch scripts
    echo "=== Pushing config and sbatch scripts ==="
    scp "${PROJECT_DIR}/backend/etl/config.yaml" "${CLUSTER_HOST}:${CLUSTER_DIR}/backend/etl/config.yaml"
    scp "${PROJECT_DIR}/deployment/slurm/etl_pipeline.sbatch" "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
    scp "${PROJECT_DIR}/deployment/slurm/pdf_processing.sbatch" "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
    scp "${PROJECT_DIR}/deployment/slurm/benchmark.sbatch" "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
    echo "Config and scripts pushed."
}

setup_cluster_dirs() {
    echo "=== Creating cluster directories ==="
    ssh "$CLUSTER_HOST" "mkdir -p ${CLUSTER_DIR}/{logs,reports,data,deployment/slurm,backend/etl}"
    echo "Cluster directories created."
}

case "${1:-build}" in
    build)
        build
        ;;
    push)
        push
        ;;
    setup)
        setup_cluster_dirs
        ;;
    all)
        build
        push
        setup_cluster_dirs
        ;;
    *)
        echo "Usage: $0 {build|push|setup|all}"
        exit 1
        ;;
esac
