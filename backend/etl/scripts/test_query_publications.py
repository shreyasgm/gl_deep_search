"""
Test script for querying publication tracking data.

This script demonstrates how to retrieve publication tracking data
from the database using the PublicationTracker class.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlmodel import Session, select

from backend.etl.models.tracking import (
    PublicationTracking,
    DownloadStatus,
    ProcessingStatus,
    EmbeddingStatus,
    IngestionStatus,
)
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.database import engine, ensure_db_initialized

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Query and display tracked publications."""
    # Ensure database is initialized
    ensure_db_initialized()
    
    # Create tracker
    tracker = PublicationTracker()
    
    # Get publications by status
    with Session(engine) as session:
        # Get all tracked publications
        stmt = select(PublicationTracking)
        all_publications = session.exec(stmt).all()
        
        print(f"\nTotal tracked publications: {len(all_publications)}")
        
        # Count publications by download status
        print("\nDownload Status Counts:")
        for status in DownloadStatus:
            stmt = select(PublicationTracking).where(
                PublicationTracking.download_status == status
            )
            count = len(session.exec(stmt).all())
            print(f"  {status.value}: {count}")
        
        # Count publications by processing status
        print("\nProcessing Status Counts:")
        for status in ProcessingStatus:
            stmt = select(PublicationTracking).where(
                PublicationTracking.processing_status == status
            )
            count = len(session.exec(stmt).all())
            print(f"  {status.value}: {count}")
        
        # Count publications by embedding status
        print("\nEmbedding Status Counts:")
        for status in EmbeddingStatus:
            stmt = select(PublicationTracking).where(
                PublicationTracking.embedding_status == status
            )
            count = len(session.exec(stmt).all())
            print(f"  {status.value}: {count}")
        
        # Count publications by ingestion status
        print("\nIngestion Status Counts:")
        for status in IngestionStatus:
            stmt = select(PublicationTracking).where(
                PublicationTracking.ingestion_status == status
            )
            count = len(session.exec(stmt).all())
            print(f"  {status.value}: {count}")
        
        # Show recently downloaded publications
        print("\nRecently Downloaded Publications:")
        stmt = select(PublicationTracking).where(
            PublicationTracking.download_status == DownloadStatus.DOWNLOADED
        ).order_by(PublicationTracking.download_timestamp.desc()).limit(5)
        
        recent_downloads = session.exec(stmt).all()
        
        if not recent_downloads:
            print("  No downloaded publications found")
        else:
            for pub in recent_downloads:
                print(f"\n  Title: {pub.title}")
                print(f"  ID: {pub.publication_id}")
                print(f"  Downloaded at: {pub.download_timestamp}")
                print(f"  Processing status: {pub.processing_status.value}")
                if pub.file_urls:
                    print(f"  File URL: {pub.file_urls[0]}")
                    if len(pub.file_urls) > 1:
                        print(f"    ... and {len(pub.file_urls) - 1} more")
                else:
                    print("  No file URLs")


if __name__ == "__main__":
    main() 