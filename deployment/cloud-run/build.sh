#!/bin/bash
# Build Docker image for multiple platforms
#
# This script builds Docker images for both local development (Mac ARM64)
# and cloud deployment (Linux AMD64).
#
# Usage:
#   ./deployment/cloud-run/build.sh [--platform PLATFORM] [--push] [--tag TAG]
#
# Options:
#   --platform PLATFORM    Build for specific platform (linux/amd64, linux/arm64, or both)
#   --push                 Push image to registry after building
#   --tag TAG              Tag for the image (default: gl-deep-search-etl:local)
#   --load                 Load image into local Docker (for single-platform builds)

set -euo pipefail

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
PLATFORM=""
PUSH=false
LOAD=false
TAG="gl-deep-search-etl:local"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --load)
            LOAD=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--platform PLATFORM] [--push] [--load] [--tag TAG]"
            echo ""
            echo "Options:"
            echo "  --platform PLATFORM    Build for specific platform (linux/amd64, linux/arm64, or both)"
            echo "  --push                 Push image to registry after building"
            echo "  --load                 Load image into local Docker (for single-platform builds)"
            echo "  --tag TAG              Tag for the image (default: gl-deep-search-etl:local)"
            echo ""
            echo "Examples:"
            echo "  $0 --platform linux/arm64 --load --tag gl-deep-search-etl:local"
            echo "  $0 --platform linux/amd64,linux/arm64 --push --tag my-registry/image:tag"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Docker buildx is available
if ! docker buildx version &> /dev/null; then
    echo "ERROR: Docker buildx is not available. Please install Docker Desktop or enable buildx."
    exit 1
fi

# Create buildx builder if it doesn't exist
BUILDER_NAME="multiarch-builder"
if ! docker buildx inspect "$BUILDER_NAME" &> /dev/null; then
    echo "Creating buildx builder: $BUILDER_NAME"
    docker buildx create --name "$BUILDER_NAME" --use --bootstrap
else
    echo "Using existing buildx builder: $BUILDER_NAME"
    docker buildx use "$BUILDER_NAME"
fi

# Determine platform(s)
if [[ -z "$PLATFORM" ]]; then
    # Detect local platform
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        PLATFORM="linux/arm64"
        echo "Detected ARM64 architecture, building for linux/arm64"
    else
        PLATFORM="linux/amd64"
        echo "Detected x86_64 architecture, building for linux/amd64"
    fi
fi

# Build arguments
BUILD_ARGS=(
    "buildx"
    "build"
    "--platform" "$PLATFORM"
    "--file" "$DOCKERFILE"
    "--tag" "$TAG"
)

# Add --load flag for single-platform builds (required for local testing)
if [[ "$LOAD" == true ]]; then
    if [[ "$PLATFORM" == *","* ]]; then
        echo "ERROR: --load flag cannot be used with multiple platforms"
        exit 1
    fi
    BUILD_ARGS+=("--load")
fi

# Add --push flag if requested
if [[ "$PUSH" == true ]]; then
    BUILD_ARGS+=("--push")
fi

# Add build context
BUILD_ARGS+=("$PROJECT_ROOT")

# Build the image
echo "Building Docker image..."
echo "  Platform(s): $PLATFORM"
echo "  Tag: $TAG"
echo "  Dockerfile: $DOCKERFILE"
echo "  Context: $PROJECT_ROOT"
echo ""

if docker "${BUILD_ARGS[@]}"; then
    echo ""
    echo "✓ Docker image built successfully!"
    if [[ "$LOAD" == true ]]; then
        echo "  Image loaded into local Docker: $TAG"
    fi
    if [[ "$PUSH" == true ]]; then
        echo "  Image pushed to registry: $TAG"
    fi
else
    echo ""
    echo "✗ Failed to build Docker image"
    exit 1
fi
