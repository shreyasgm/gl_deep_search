#!/bin/bash
# Growth Lab Deep Search - One-time FAS-RC environment setup
#
# Builds the PDF processing Docker image locally, then converts it
# to a Singularity .sif image for use on the SLURM cluster.
#
# Prerequisites:
#   - Docker installed on your local machine
#   - Singularity available on the cluster (module load singularity)
#   - SSH/scp access to the cluster
#
# Usage (local machine):
#   bash deployment/slurm/setup_env.sh build   # Build Docker image
#   bash deployment/slurm/setup_env.sh push     # Push .sif to cluster
#
# Usage (on the cluster, if Singularity can pull from a registry):
#   bash deployment/slurm/setup_env.sh pull     # Pull from registry

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_NAME="gl-pdf-processing"
IMAGE_TAG="latest"
SIF_FILE="${PROJECT_DIR}/deployment/slurm/${IMAGE_NAME}.sif"
# Update this to your cluster login node
CLUSTER_HOST="${CLUSTER_HOST:-${USER}@login.rc.fas.harvard.edu}"
CLUSTER_DIR="${CLUSTER_DIR:-\$HOME/gl_deep_search}"

build() {
    echo "=== Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG} ==="
    cd "$PROJECT_DIR"
    docker build \
        -f deployment/pdf-processing/Dockerfile \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        .
    echo "Docker image built successfully."
}

export_sif() {
    echo "=== Converting Docker image to Singularity .sif ==="
    # Save Docker image as a tar, then convert to .sif
    # This works on machines that have Singularity installed
    if command -v singularity &>/dev/null; then
        singularity build "$SIF_FILE" "docker-daemon://${IMAGE_NAME}:${IMAGE_TAG}"
    else
        echo "Singularity not available locally. Saving Docker image as tar..."
        DOCKER_TAR="${PROJECT_DIR}/deployment/slurm/${IMAGE_NAME}.tar"
        docker save "${IMAGE_NAME}:${IMAGE_TAG}" -o "$DOCKER_TAR"
        echo "Docker image saved to: $DOCKER_TAR"
        echo ""
        echo "Transfer to cluster and convert there:"
        echo "  scp $DOCKER_TAR ${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
        echo "  ssh ${CLUSTER_HOST}"
        echo "  module load singularity"
        echo "  singularity build ${IMAGE_NAME}.sif docker-archive://${IMAGE_NAME}.tar"
        return
    fi
    echo "Singularity image created: $SIF_FILE"
}

push() {
    echo "=== Pushing .sif to cluster ==="
    if [[ -f "$SIF_FILE" ]]; then
        scp "$SIF_FILE" "${CLUSTER_HOST}:${CLUSTER_DIR}/deployment/slurm/"
        echo "Pushed to cluster."
    else
        echo "No .sif file found. Run 'build' and 'export_sif' first."
        echo "Or transfer the .tar file manually (see export_sif output)."
        exit 1
    fi
}

setup_cluster_dirs() {
    echo "=== Creating cluster directories ==="
    ssh "$CLUSTER_HOST" "mkdir -p ${CLUSTER_DIR}/{logs,reports,data,deployment/slurm}"
    echo "Cluster directories created."
}

case "${1:-build}" in
    build)
        build
        export_sif
        ;;
    push)
        push
        ;;
    export)
        export_sif
        ;;
    setup)
        setup_cluster_dirs
        ;;
    all)
        build
        export_sif
        push
        setup_cluster_dirs
        ;;
    *)
        echo "Usage: $0 {build|push|export|setup|all}"
        exit 1
        ;;
esac
