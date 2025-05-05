# Publication Tracking Manifest Schema

This repository contains the implementation of a manifest schema for tracking publications through the ETL pipeline for the Growth Lab Deep Search project.

## Components Created

1. **SQL Schema File**: `/backend/storage/etl_metadata_schema.sql`
   - Defines the table structure for tracking publications
   - Includes fields for all ETL stages: discovery, download, processing, embedding, ingestion
   - Defines appropriate data types and constraints
   - Includes indexes for efficient querying

2. **SQLModel Definition**: `/backend/etl/models/tracking.py`
   - Object-relational mapping using SQLModel
   - Defines Enum classes for status fields
   - Includes helper methods for updating publication status
   - Handles JSON serialization of file URLs

3. **Database Connection Utilities**: `/backend/storage/database.py`
   - Connection management with SQLAlchemy
   - Support for both SQLite (development) and PostgreSQL (production)
   - Automatic database initialization
   - Environment variable configuration

4. **Test Script**: `/backend/etl/scripts/test_publication_tracking.py`
   - Scrapes sample publications from Growth Lab website
   - Stores metadata in the tracking database
   - Simulates status updates through the ETL pipeline
   - Demonstrates querying and displaying stored data

5. **Environment Configuration**:
   - `.env.example` files for ETL and service components
   - Database connection configuration
   - Storage path configuration

6. **Docker Compose Setup**: `/docker-compose.yml`
   - PostgreSQL database service
   - API service with database connection
   - ETL service for scrapers and processors
   - Volume configuration for data persistence

## Testing the Implementation

To test this implementation, you'll need:

1. Required Python packages:
   ```
   pip install aiohttp sqlmodel pydantic python-dotenv sqlalchemy bs4 requests
   ```

2. Run the test script:
   ```
   cd /path/to/gl_deep_search
   python -m backend.etl.scripts.test_publication_tracking
   ```

This will:
- Initialize the database
- Scrape a few sample publications from the Growth Lab website
- Store their metadata in the tracking database
- Simulate updates to their download and processing status
- Display the stored data from the database

## Next Steps

1. **Integration with Scrapers**:
   - Modify existing scrapers to update tracking metadata
   - Add tracking to download processes

2. **API Endpoint Development**:
   - Create REST API endpoints for querying tracking status
   - Implement filtering and sorting options

3. **Dashboard Integration**:
   - Connect tracking database to a dashboard for visualization
   - Display ETL pipeline status and metrics

4. **Error Handling and Retry Logic**:
   - Implement automated retry for failed publications
   - Add error logging and notification

## Schema Details

The publication tracking schema includes the following key fields:

### Identification
- `publication_id`: Primary key, unique identifier
- `source_url`: URL where the publication was found
- `title`: Publication title
- `authors`: Publication authors
- `year`: Publication year
- `abstract`: Publication abstract
- `file_urls`: Associated file URLs (stored as JSON)
- `content_hash`: Hash for detecting content changes

### ETL Pipeline Stages
- Discovery: When the publication was first found
- Download: Status and timing of file downloads
- Processing: Status and timing of text extraction and chunking
- Embedding: Status and timing of vector embedding generation
- Ingestion: Status and timing of database ingestion

### Tracking and Diagnostics
- `last_updated`: Timestamp of the last modification
- `error_message`: Most recent error message
- Attempt counters for each pipeline stage