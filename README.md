# Growth Lab Deep Search

An agentic RAG system that helps users query Growth Lab-specific unstructured data.

## 🔍 Project Overview

Growth Lab Deep Search is an agentic AI system designed to answer complex questions about the Growth Lab's research and publications. The system incorporates:

**Key Features:**

- Automated ETL pipeline for harvesting Growth Lab publications and academic papers
- Advanced OCR processing of PDF documents using modern tools
- Vector embeddings with hybrid search
- Agentic RAG system based on LangGraph

## Project Architecture

### Directory structure

This is a rough outline of the intended directory structure. The actual structure might look different, but this should give an idea of the intended code organization.

```
gl_deep_search/
├── .github/
│   └── workflows/
│       ├── etl-pipeline.yml         # Scheduled ETL runs and deployment
│       ├── service-deploy.yml       # Service API deployment
│       └── frontend-deploy.yml      # Frontend deployment
├── .gitignore
├── README.md
├── pyproject.toml                   # Python project config for uv
├── docker-compose.yml               # Local development setup
├── docker-compose.prod.yml          # Production setup
│
├── backend/
│   ├── etl/
│   │   ├── Dockerfile                # ETL container configuration
│   │   ├── docker-compose.yml        # Local development setup
│   │   ├── config.yaml               # Default configuration
│   │   ├── .env.example              # Environment variables template
│   │   ├── pyproject.toml            # Python dependencies (uv)
│   │   ├── main.py                   # ETL orchestration entry point
│   │   ├── models.py                 # Pydantic data models
│   │   ├── config.py                 # Configuration management
│   │   ├── scrapers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Abstract scraper interface
│   │   │   ├── growthlab.py          # Growth Lab website scraper
│   │   │   └── openalex.py           # OpenAlex API client
│   │   ├── processors/
│   │   │   ├── __init__.py
│   │   │   ├── pdf_processor.py      # PDF processing and OCR
│   │   │   └── manifest.py           # Manifest management
│   │   ├── storage/
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Storage abstraction
│   │   │   ├── local.py              # Local filesystem adapter
│   │   │   └── gcs.py                # Google Cloud Storage adapter
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── id_utils.py           # ID generation utilities
│   │   │   ├── async_utils.py        # Async helpers & rate limiting
│   │   │   ├── ocr_utils.py          # OCR interface
│   │   │   └── logger.py             # Logging configuration
│   │   └── tests/                    # Unit and integration tests
│   │
│   ├── service/                      # Main backend service (replaces "agent")
│   │   ├── Dockerfile                # Service Docker configuration
│   │   ├── .env.example              # Example environment variables
│   │   ├── main.py                   # FastAPI entry point
│   │   ├── routes.py                 # API endpoints
│   │   ├── models.py                 # Data models
│   │   ├── config.py                 # Service configuration
│   │   ├── graph.py                  # LangGraph definition
│   │   ├── tools.py                  # Service tools
│   │   └── utils/
│   │       ├── retriever.py          # Vector retrieval
│   │       └── logger.py             # Logging and observability
│   │
│   ├── storage/                      # Storage configuration
│   │   ├── qdrant_config.yaml        # Qdrant vector DB config
│   │   └── metadata_schema.sql       # Metadata schema if needed
│   │
│   └── cloud/                        # Cloud deployment configs
│       ├── etl-cloudrun.yaml         # ETL Cloud Run config
│       └── service-cloudrun.yaml     # Service Cloud Run config
│
├── frontend/
│   ├── Dockerfile                    # Frontend Docker configuration
│   ├── .env.example                  # Example environment variables
│   ├── app.py                        # Single Streamlit application file
│   └── utils.py                      # Frontend utility functions
│
└── scripts/                          # Utility scripts
    ├── setup.sh                      # Project setup
    ├── deploy.sh                     # Deployment to GCP
    └── storage_switch.sh             # Script to switch between local/cloud storage
```



## Tech Stack

- **ETL Pipeline**: GitHub Actions, Modern OCR tools (Docling/Marker/Gemini Flash 2)
- **Vector Storage**: Qdrant for embeddings, with Cohere for reranking
- **Agent System**: LangGraph for agentic RAG workflows
- **Backend API**: FastAPI, Python 3.12+
- **Frontend**: Streamlit or Chainlit for MVP
- **Deployment**: Google Cloud Run
- **Package Management**: uv

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- `uv` for project management (check documentarion [here](https://docs.astral.sh/uv/))
- GCP account and credentials (for production)
- API keys for OpenAI, Anthropic, etc.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shreyasgm/gl_deep_search.git
   cd gl_deep_search
   ```
2. Run `uv` in the CLI to check that it is available. After this, run `uv sync` to install dependencies and create the virtual environment. This command will only install the core dependencies specified in the `pyproject.toml` file. To install dependencies that belong to a specific component (*i.e.*, optional dependencies) use:
   ```bash
   # For a single optional component
   uv sync --extra etl

   # For multiple optional components
   uv sync --extra etl, frontend, [other groups]
   ```

3. To add new packages to the project, use the following format:
   ```bash
   # Add a package to a specific group (etl, service, frontend, dev, prod)
   uv add package_name --optional group_name

   # Example: Add seaborn to the service group
   uv add seaborn --optional service
   ```

4. Create and configure environment files:
   ```bash
   cp backend/etl/.env.example backend/etl/.env
   cp backend/service/.env.example backend/service/.env
   cp frontend/.env.example frontend/.env
   ```

5. Add your API keys and configuration to the `.env` files

### Docker Development Environment

The project uses Docker for consistent development and deployment environments:

1. Start the complete development stack:
   ```bash
   docker-compose up
   ```

2. Access local services:
   - Frontend UI: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

3. Run individual components:
   ```bash
   # Run only the ETL service
   docker-compose up etl

   # Run only the backend service
   docker-compose up service

   # Run only the frontend
   docker-compose up frontend
   ```

### Running the ETL Pipeline

The ETL pipeline supports both development and production environments through containerized deployment. Initial data ingestion and processing of historical documents is executed on a High-Performance Computing (HPC) infrastructure using SLURM workload manager. Incremental updates for new documents are handled through Google Cloud Run.

```bash
# Development: Execute ETL pipeline in local environment
docker-compose run --rm etl python main.py

# Production: Initial bulk processing via HPC/SLURM
sbatch scripts/slurm_etl_initial.sh

# Component-specific execution
docker-compose run --rm etl python main.py --component scraper
docker-compose run --rm etl python main.py --component processor
docker-compose run --rm etl python main.py --component embedder
```

Post-initial processing, data is migrated to Google Cloud Storage. Subsequent ETL operations are orchestrated through automated GitHub Actions workflows and executed on Google Cloud Run.

## Deployment

### Local to Production Workflow

1. Development occurs in local Docker environment
2. Code is pushed to GitHub
3. GitHub Actions triggers:
   - Code testing
   - Building and publishing container images
   - Deploying to Cloud Run

### Production Infrastructure

- **ETL Pipeline**: Scheduled Cloud Run jobs triggered by GitHub Actions
- **Backend Service**: Cloud Run with autoscaling
- **Vector Database**: Managed Qdrant instance or Qdrant Cloud
- **Document Storage**: Cloud Storage
- **Frontend**: Streamlit or Chainlit

### Deployment Commands

```bash
# Deploy to development environment
./scripts/deploy.sh dev

# Deploy to production environment
./scripts/deploy.sh prod
```

## 🧪 Development Workflow

### Contributing Guidelines

1. Create a feature branch from `main`
2. Implement your changes with tests
3. Submit a pull request for review

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=backend
```

### Deployment

Development and production environments are managed through Docker and GitHub Actions:

```bash
# Deploy to development
./scripts/deploy.sh dev

# Deploy to production
./scripts/deploy.sh prod
```


## 🔒 Security & Configuration

- API keys and secrets are managed via `.env` files (not committed to GitHub)
- Production secrets are stored in GCP Secret Manager
- Access control is implemented at the API level

## License

This project is licensed under CC-BY-NC-SA 4.0. See the LICENSE file for details.
