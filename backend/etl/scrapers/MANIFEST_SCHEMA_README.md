# Publication Tracking Manifest Schema

This document describes the manifest schema for tracking publications through the ETL pipeline for the Growth Lab Deep Search project, which has now been fully integrated into the pipeline.

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
   - Enhanced to properly handle multiple SQL statements

4. **ETL Integration**: `/backend/etl/utils/publication_tracker.py`
   - PublicationTracker utility class for managing tracking operations
   - Methods for adding and updating publications at each pipeline stage
   - Querying methods to find publications for each stage of processing

5. **FileDownloader Integration**: `/backend/etl/utils/gl_file_downloader.py`
   - Automatic tracking of publication download status
   - Integration with PublicationTracker to update database entries
   - Proper error handling and status updates

6. **Test Scripts**:
   - `/backend/etl/scripts/test_download_tracking.py`: Tests downloading with tracking
   - `/backend/etl/scripts/test_query_publications.py`: Queries tracked publications
   - `/backend/etl/scripts/reset_publication_status.py`: Utility to reset statuses for reprocessing

7. **Environment Configuration**:
   - `.env.example` files for ETL and service components
   - Database connection configuration
   - Storage path configuration

## Testing the Implementation

To test this implementation, you can run the following scripts:

1. Download publications with tracking:
   ```
   python backend/etl/scripts/test_download_tracking.py
   ```

2. Query tracked publications:
   ```
   python backend/etl/scripts/test_query_publications.py
   ```

3. Reset publication statuses:
   ```
   python backend/etl/scripts/reset_publication_status.py [--stage STAGE]
   ```
   Where STAGE can be: download, processing, embedding, ingestion, or all

## Integration Details

The publication tracking system is now fully integrated into the ETL pipeline:

1. **Scraper Integration**:
   - Publications are automatically added to tracking when discovered
   - The PublicationTracker.add_publication() method creates or updates tracking records

2. **Download Integration**:
   - The FileDownloader now tracks download status in the database
   - Status is updated to IN_PROGRESS at start and DOWNLOADED or FAILED on completion
   - Error messages are captured for failed downloads

3. **Future Processing Integration**:
   - The infrastructure is in place to track processing, embedding, and ingestion
   - Utility methods are available for each stage of the pipeline

## Schema Details

The publication tracking schema includes the following key fields:

### Identification
- `publication_id`: Primary key, unique identifier
- `source_url`: URL where the publication was found
- `title`: Publication title
- `authors`: Publication authors
- `year`: Publication year
- `abstract`: Publication abstract
- `file_urls_json`: Associated file URLs (stored as JSON)
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