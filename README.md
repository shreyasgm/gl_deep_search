# Growth Lab Deep Search

An agentic RAG system that helps users query Growth Lab-specific unstructured data.

## 🔍 Project Overview

Growth Lab Deep Search is an agentic AI system designed to answer complex questions about the Growth Lab's research and publications. The system incorporates:

**Key Features:**

- Automated ETL pipeline for harvesting Growth Lab publications and academic papers
- Advanced OCR processing of PDF documents using modern tools
- Vector embeddings with hybrid search
- Agentic RAG system based on LangGraph

## Ideas for where to go from here

These are some brainstorming ideas for applications of the data that we collect using the ETL pipeline.

- Growth Lab search: use embeddings + vector db, use full text of documents, maybe build a knowledge graph, and allow effective "deep search" using LLMs
- How was GL research changed over time? Who is involved? etc. Simple text clouds, simple topic modeling over time.
- What are the constraints identified in the applied projects? How has that changed over time? For a given type of constraint, what are the usual policy recommendations?
- How is GD done in the Growth Lab? How about Economic Complexity analysis? How has that changed over time? What is the modern way of doing it?


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
│
├── backend/
│   ├── etl/
│   │   ├── config.yaml               # Default ETL configuration
│   │   ├── config.production.yaml    # Production ETL configuration
│   │   ├── orchestrator.py           # ETL orchestration entry point
│   │   ├── models/
│   │   │   ├── publications.py       # Publication data models
│   │   │   └── tracking.py           # ETL tracking models
│   │   ├── scrapers/
│   │   │   ├── growthlab.py          # Growth Lab website scraper
│   │   │   └── openalex.py           # OpenAlex API client
│   │   ├── scripts/                  # ETL execution scripts
│   │   │   ├── run_growthlab_scraper.py
│   │   │   ├── run_openalex_scraper.py
│   │   │   ├── run_gl_file_downloader.py
│   │   │   ├── run_openalex_file_downloader.py
│   │   │   ├── run_pdf_processor.py
│   │   │   └── run_embeddings_generator.py
│   │   └── utils/
│   │       ├── pdf_processor.py      # PDF processing and OCR
│   │       ├── gl_file_downloader.py # Growth Lab file downloader
│   │       ├── oa_file_downloader.py # OpenAlex file downloader
│   │       ├── text_chunker.py       # Text chunking utilities
│   │       ├── embeddings_generator.py # Embedding generation
│   │       ├── publication_tracker.py # Publication tracking
│   │       └── retry.py              # Retry utilities
│   │
│   ├── service/                      # Main backend service (future)
│   │   ├── main.py                   # FastAPI entry point
│   │   ├── routes.py                 # API endpoints
│   │   ├── graph.py                  # LangGraph definition
│   │   └── utils/
│   │       └── retriever.py          # Vector retrieval
│   │
│   ├── storage/                      # Storage abstraction layer
│   │   ├── base.py                   # Storage interface
│   │   ├── local.py                  # Local filesystem adapter
│   │   ├── gcs.py                    # Google Cloud Storage adapter
│   │   ├── cloud.py                  # Cloud storage utilities
│   │   ├── database.py               # Database utilities
│   │   └── factory.py                # Storage factory
│   │
│   └── tests/                        # Unit and integration tests
│       ├── etl/
│       └── service/
│
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Raw scraped data
│   │   ├── documents/                # Raw documents by source
│   │   └── pdfs/                     # Downloaded PDF files
│   ├── intermediate/                 # Intermediate processing data
│   │   └── *.csv                     # Scraped publication metadata
│   ├── processed/                    # Processed data
│   │   ├── documents/                # Processed documents with text
│   │   ├── chunks/                   # Chunked documents
│   │   └── embeddings/               # Generated embeddings
│   └── reports/                      # ETL execution reports
│
├── deployment/                       # GCP deployment infrastructure
│   ├── cloud-run/                    # Cloud Run job scripts
│   ├── vm/                           # VM-based deployment scripts
│   ├── scripts/                      # Setup and utility scripts
│   └── config/                       # GCP configuration
│
└── frontend/                         # Frontend application (future)
    ├── app.py                        # Streamlit/Chainlit application
    └── utils.py                      # Frontend utility functions
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

### Running Locally

Start the backend and frontend services directly using `uv`:

```bash
# Start the backend API server
uv run uvicorn backend.service.main:app --host 0.0.0.0 --port 8000

# Start the frontend (in a separate terminal)
uv run streamlit run frontend/app.py --server.port 8501
```

- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Frontend UI: http://localhost:8501

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

### GCP Deployment Infrastructure

The project includes a complete deployment infrastructure for Google Cloud Platform (GCP). See the [deployment guide](deployment/README.md) for detailed instructions.

**Quick Start:**
1. Configure GCP settings: `cp deployment/config/gcp-config.sh.template deployment/config/gcp-config.sh`
2. Run setup scripts: `./deployment/scripts/01-setup-gcp-project.sh` (and 02-04)
3. Deploy Cloud Run Job: `./deployment/cloud-run/deploy.sh`
4. Schedule weekly updates: `./deployment/cloud-run/schedule.sh`

**Deployment Options:**
- **VM-based**: For initial batch processing (`./deployment/vm/create-vm.sh`)
- **Cloud Run Jobs**: For scheduled weekly updates (automated via Cloud Scheduler)
- **Manual execution**: Run on-demand (`./deployment/cloud-run/execute.sh`)

**Documentation:**
- [Deployment README](deployment/README.md) - Quick start and troubleshooting
- [GCP Deployment Guide](docs/GCP_DEPLOYMENT_GUIDE.md) - Comprehensive deployment documentation

### Production Infrastructure

- **ETL Pipeline**: Cloud Run Jobs (scheduled weekly) + VM instances (initial batch)
- **Backend Service**: Cloud Run with autoscaling (future)
- **Vector Database**: Managed Qdrant instance or Qdrant Cloud
- **Document Storage**: Cloud Storage (GCS)
- **Frontend**: Streamlit or Chainlit (future)
- **Scheduling**: Cloud Scheduler or GitHub Actions workflows

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
