# Publication Tracking System

This directory contains the PublicationTracker utility, which provides a convenient interface for tracking publications through the ETL pipeline stages (discovery, download, processing, embedding, ingestion).

## Overview

The publication tracking system keeps track of publications as they move through the ETL pipeline, recording status, timestamps, and errors at each stage. This information can be used to:

- Monitor pipeline progress
- Identify bottlenecks or failures
- Support retry logic for failed items
- Enable reporting and metrics

## Core Components

- **PublicationTracker**: Utility class for managing tracking operations
- **PublicationTracking**: SQLModel for database representation
- **Status Enums**: Defined states for each pipeline stage

## How to Use in ETL Components

### 1. Initialize the Tracker

```python
from backend.etl.utils.publication_tracker import PublicationTracker

# Create a tracker instance
tracker = PublicationTracker()
```

### 2. Add Publications to Tracking

```python
# Add a publication (typically from a scraper)
publication = scraper.extract_publication(...)
tracking_record = tracker.add_publication(publication)

# Or add multiple publications at once
publications = scraper.extract_publications()
tracking_records = tracker.add_publications(publications)
```

### 3. Update Status at Pipeline Stages

```python
# Update download status
tracker.update_download_status(
    publication_id="pub123",
    status=DownloadStatus.IN_PROGRESS
)

# Later, update with success or failure
tracker.update_download_status(
    publication_id="pub123",
    status=DownloadStatus.DOWNLOADED
)

# Or with failure and error message
tracker.update_download_status(
    publication_id="pub123",
    status=DownloadStatus.FAILED,
    error="Failed to download: 404 Not Found"
)

# Similarly for other stages
tracker.update_processing_status(...)
tracker.update_embedding_status(...)
tracker.update_ingestion_status(...)
```

### 4. Query Publications for Pipeline Stages

```python
# Get publications ready for download
publications_to_download = tracker.get_publications_for_download(limit=10)

# Get publications ready for processing
publications_to_process = tracker.get_publications_for_processing(limit=10)

# Similarly for other stages
publications_to_embed = tracker.get_publications_for_embedding(limit=10)
publications_to_ingest = tracker.get_publications_for_ingestion(limit=10)
```

### 5. Check Status of a Specific Publication

```python
# Get full status of a publication
status = tracker.get_publication_status("pub123")
if status:
    print(f"Download status: {status['download_status']}")
    print(f"Processing status: {status['processing_status']}")
    # ...
```

## Integration Examples

### FileDownloader Integration

The file downloader has been integrated with publication tracking:

```python
# Initialize with a tracker
downloader = FileDownloader(
    storage=storage,
    publication_tracker=tracker
)

# Download publications (will update tracking status automatically)
results = await downloader.download_publications(publications)
```

### Integration with Processing Components

When implementing document processing, follow this pattern:

```python
# Get publications ready for processing
publications = tracker.get_publications_for_processing(limit=batch_size)

for pub in publications:
    try:
        # Update status to IN_PROGRESS
        tracker.update_processing_status(
            pub.publication_id,
            ProcessingStatus.IN_PROGRESS
        )

        # Perform processing
        # ...

        # Update status to PROCESSED on success
        tracker.update_processing_status(
            pub.publication_id,
            ProcessingStatus.PROCESSED
        )
    except Exception as e:
        # Update status to FAILED on error
        tracker.update_processing_status(
            pub.publication_id,
            ProcessingStatus.FAILED,
            error=str(e)
        )
```

## Reset for Reprocessing

To reset publications for reprocessing, create a  the reset script (was eliminated for simplicity)
