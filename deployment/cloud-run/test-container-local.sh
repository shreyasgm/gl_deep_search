#!/bin/bash
# Test container locally with 10 documents
#
# This script builds the Docker image for your local platform (ARM64 on Mac)
# and runs it with a 10 document limit for testing.
#
# Usage:
#   ./deployment/cloud-run/test-container-local.sh
#
# Prerequisites:
#   - Docker installed
#   - OPENAI_API_KEY environment variable set
#   - GCP credentials configured (for GCS access)

set -e

echo "Testing container locally with 10 document limit..."

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/test-container"
IMAGE_TAG="gl-deep-search-etl:test"

# Create data directory
mkdir -p "$DATA_DIR"

# Build image for local platform if it doesn't exist or is outdated
echo "Building Docker image for local platform..."
if ! docker images "$IMAGE_TAG" --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_TAG}$"; then
    echo "Image not found, building..."
    "${SCRIPT_DIR}/build.sh" \
        --platform linux/arm64 \
        --load \
        --tag "$IMAGE_TAG"
else
    echo "Image exists, skipping build (use 'docker rmi $IMAGE_TAG' to force rebuild)"
fi

# Run container
echo "Running container..."
docker run --rm \
  -e GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-cid-hks-1537286359734}" \
  -e GCS_BUCKET="${GCS_BUCKET:-gl-deep-search-data}" \
  -e ENVIRONMENT=development \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -v "${DATA_DIR}:/app/data" \
  "$IMAGE_TAG" \
  --config backend/etl/config.yaml \
  --log-level INFO \
  --scraper-limit 10
