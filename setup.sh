#!/bin/bash
set -e

# To run this script:
#   chmod +x setup.sh
#   ./setup.sh

# Display banner
echo "=================================================="
echo "     Growth Lab Deep Search - Project Setup"
echo "=================================================="
echo

# Create base directories
echo "Creating project directory structure..."

# Root level directories
mkdir -p .github/workflows
mkdir -p backend/etl/scripts backend/etl/utils backend/etl/data/raw backend/etl/data/intermediate backend/etl/data/processed
mkdir -p backend/service/utils
mkdir -p backend/storage
mkdir -p backend/cloud
mkdir -p frontend

# Create basic .env files
echo "# Add your API keys and configuration here" > backend/etl/.env
echo "# Add your API keys and configuration here" > backend/service/.env
echo "# Add your API keys and configuration here" > frontend/.env

echo
echo "Project structure has been set up successfully!"
echo
