# Publication Tracking API Service

This folder contains a FastAPI-based service for tracking and querying the processing status of academic publications. The service provides REST API endpoints to monitor publications as they move through various processing stages (download, processing, embedding, and ingestion).

## Overview

The Publication Tracking API allows users to:
1. Query the status of all publications with filtering, pagination, and sorting
2. Look up the status of individual publications by ID
3. Check the health status of the service

## File Structure

This folder contains the following files:

- **`database.py`**: Database connection and query utilities
- **`main.py`**: Main FastAPI application entry point
- **`models.py`**: Pydantic models for request and response schemas
- **`routes.py`**: API endpoint definitions

## Detailed Component Description

### 1. `database.py`

This module handles all database interactions using asynchronous SQLite connections. It provides functions for:

- Connecting to and querying the publication tracking database
- Building dynamic SQL queries based on filter parameters
- Retrieving publication status records with filtering, sorting, and pagination
- Fetching individual publications by ID

Key functions:
- `get_db_connection()`: Establishes an async connection to the SQLite database
- `get_publication_statuses(filters)`: Returns paginated, filtered publication status records
- `get_publication_by_id(publication_id)`: Retrieves a single publication by its ID

### 2. `models.py`

This module defines Pydantic data models for:

- Request parameters and filters
- Response structures
- Status enumerations for tracking publication progress

Key models and enums:
- Status enums: `DownloadStatus`, `ProcessingStatus`, `EmbeddingStatus`, `IngestionStatus`
- `PublicationStatusFilter`: Filter parameters for searching publications
- `PublicationStatus`: Complete representation of a publication's processing status
- `PublicationStatusResponse`: Paginated response containing publication status records

### 3. `routes.py`

This module defines the API endpoints using FastAPI's router functionality:

- `/publications/status`: Endpoint for querying multiple publications with filters
- `/publications/status/{publication_id}`: Endpoint for retrieving a specific publication

### 4. `main.py`

The application entry point that:
- Creates the FastAPI application
- Configures middleware, routers, and error handlers
- Provides health check endpoints

## API Usage Guide

### Running the API Server

To start the API server:

```bash
# From the project root directory
uv run backend/service/main.py
```

The API will be available at http://localhost:8000.

### API Endpoints

#### 1. Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy"
}
```

#### 2. Get All Publications Status (with filtering)

```
GET /publications/status
```

Query Parameters:
- `page` (int, default=1): Page number for pagination
- `page_size` (int, default=10): Number of results per page (max 100)
- `download_status` (string, optional): Filter by download status (options: "Pending", "In Progress", "Downloaded", "Failed")
- `processing_status` (string, optional): Filter by processing status
- `embedding_status` (string, optional): Filter by embedding status
- `ingestion_status` (string, optional): Filter by ingestion status
- `sort_by` (string, default="last_updated"): Field to sort by
- `sort_order` (string, default="desc"): Sort order ("asc" or "desc")
- `title_contains` (string, optional): Filter by substring match in title
- `year` (int, optional): Filter by publication year

Example Request:
```
GET /publications/status?page=1&page_size=10&download_status=Downloaded&sort_by=year&sort_order=desc
```

Response:
```json
{
  "total": 42,
  "page": 1,
  "page_size": 10,
  "items": [
    {
      "publication_id": "pub12345",
      "title": "Example Publication Title",
      "authors": "Author 1, Author 2",
      "year": 2023,
      "abstract": "This is an abstract...",
      "source_url": "https://example.com/paper",
      "file_urls": ["https://example.com/paper.pdf"],
      "download_status": "Downloaded",
      "download_timestamp": "2023-06-01T12:34:56",
      "download_attempt_count": 1,
      "processing_status": "Processed",
      "processing_timestamp": "2023-06-01T13:45:12",
      "processing_attempt_count": 1,
      "embedding_status": "Embedded",
      "embedding_timestamp": "2023-06-01T14:15:30",
      "embedding_attempt_count": 1,
      "ingestion_status": "Ingested",
      "ingestion_timestamp": "2023-06-01T14:45:22",
      "ingestion_attempt_count": 1,
      "discovery_timestamp": "2023-06-01T10:00:00",
      "last_updated": "2023-06-01T14:45:22",
      "error_message": null
    },
    // ... more publications
  ]
}
```

#### 3. Get Single Publication Status

```
GET /publications/status/{publication_id}
```

Example Request:
```
GET /publications/status/pub12345
```

Response:
```json
{
  "publication_id": "pub12345",
  "title": "Example Publication Title",
  "authors": "Author 1, Author 2",
  "year": 2023,
  "abstract": "This is an abstract...",
  "source_url": "https://example.com/paper",
  "file_urls": ["https://example.com/paper.pdf"],
  "download_status": "Downloaded",
  "download_timestamp": "2023-06-01T12:34:56",
  "download_attempt_count": 1,
  "processing_status": "Processed",
  "processing_timestamp": "2023-06-01T13:45:12",
  "processing_attempt_count": 1,
  "embedding_status": "Embedded",
  "embedding_timestamp": "2023-06-01T14:15:30",
  "embedding_attempt_count": 1,
  "ingestion_status": "Ingested",
  "ingestion_timestamp": "2023-06-01T14:45:22",
  "ingestion_attempt_count": 1,
  "discovery_timestamp": "2023-06-01T10:00:00",
  "last_updated": "2023-06-01T14:45:22",
  "error_message": null
}
```

### Error Handling

The API uses standard HTTP error codes:
- `404 Not Found`: When a requested publication doesn't exist
- `500 Internal Server Error`: For database connection or query errors

Error responses include a detail message explaining the issue.

## Database Structure

The service connects to an SQLite database (`data/processed/publication_tracking.db`) with the following structure:

**publication_tracking table:**
- `publication_id`: Unique identifier (TEXT)
- `title`: Publication title (TEXT)
- `authors`: Author names (TEXT)
- `year`: Publication year (INTEGER)
- `abstract`: Publication abstract (TEXT)
- `source_url`: URL where the publication was discovered (TEXT)
- `file_urls`: JSON array of file URLs (TEXT)
- `download_status`: Current download status (TEXT)
- `download_timestamp`: When download completed (TIMESTAMP)
- `download_attempt_count`: Number of download attempts (INTEGER)
- ... (similar fields for processing, embedding, and ingestion stages)
- `discovery_timestamp`: When the publication was discovered (TIMESTAMP)
- `last_updated`: When the record was last updated (TIMESTAMP)
- `error_message`: Latest error message if any (TEXT)

## Development Notes

- The API uses asynchronous database connections for improved performance
- SQLite is used for storage but can be replaced with other databases if needed
- All endpoints are documented with OpenAPI (accessible at `/docs` when running)

## API Documentation

When the server is running, full interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
