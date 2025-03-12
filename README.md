# Growth Lab Deep Search

An agentic RAG system that helps users query Growth Lab-specific unstructured data.

## 🔍 Project Overview

Growth Lab Agent is an agentic AI system designed to answer complex questions about the Growth Lab's research and publications. The system incorporates:

**Key Features:**

- Automated ETL pipeline for harvesting Growth Lab publications and academic papers
- Advanced OCR processing of PDF documents using modern tools
- Vector embeddings with hybrid search
- Agentic RAG system based on LangGraph

## Project Architecture

### Directory structure

This is a rough outline of the intended directory structure. The actual structure might look different, but this should give an idea of the intended code organization.

```
growth-lab-agent/
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
│   ├── etl/                         # ETL Pipeline
│   │   ├── Dockerfile               # ETL Docker configuration
│   │   ├── config.yaml              # ETL configuration
│   │   ├── .env.example             # Example environment variables
│   │   ├── main.py                  # Main ETL orchestration script
│   │   ├── scripts/
│   │   │   ├── growth_lab_scraper.py # Growth Lab website scraper
│   │   │   ├── openAlex_client.py    # OpenAlex API client
│   │   │   ├── process_pdfs.py       # OCR and process PDFs
│   │   │   ├── embed_text.py         # Generate embeddings
│   │   │   └── upload_data.py        # Store in vector DB
│   │   ├── utils/
│   │   │   ├── ocr_utils.py          # OCR with modern tools
│   │   │   ├── text_utils.py         # Text processing and chunking
│   │   │   ├── embedding_utils.py    # Embedding generation
│   │   │   ├── storage_utils.py      # Vector DB interactions
│   │   │   └── cloud_utils.py        # Utilities for cloud storage
│   │   └── data/                     # Local data storage during development
│   │       ├── raw/                  # Raw downloaded files
│   │       ├── intermediate/         # Processed but not final data
│   │       └── processed/            # Final processed data
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

- **ETL Pipeline**: GitHub Actions, Modern OCR tools (Dockling/Marker/Gemini Flash 2)
- **Vector Storage**: Qdrant for embeddings, with Cohere for reranking
- **Agent System**: LangGraph for agentic RAG workflows
- **Backend API**: FastAPI, Python 3.11+
- **Frontend**: Streamlit or Chainlit for MVP
- **Deployment**: Google Cloud Run
- **Package Management**: uv

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- GCP account and credentials (for production)
- API keys for OpenAI, Anthropic, etc.

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/growth-lab-deep-search.git
   cd growth-lab-deep-search
   ```

2. Create and configure environment files:
   ```bash
   cp backend/etl/.env.example backend/etl/.env
   cp backend/service/.env.example backend/service/.env
   cp frontend/.env.example frontend/.env
   ```
   
3. Add your API keys and configuration to the `.env` files

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

The ETL pipeline can be run through Docker in both development and production:

```bash
# Development: Run the ETL pipeline locally
docker-compose run --rm etl python main.py

# Production: Initial data processing on SLURM (HPC environment)
sbatch scripts/slurm_etl_initial.sh

# Test specific ETL components
docker-compose run --rm etl python main.py --component scraper
docker-compose run --rm etl python main.py --component processor
docker-compose run --rm etl python main.py --component embedder
```

After the initial SLURM processing, data is transferred to GCP Cloud Storage, and subsequent ETL runs are automatically scheduled through GitHub Actions and Cloud Run.

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