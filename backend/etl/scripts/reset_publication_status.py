"""
Reset publication tracking status.

This script demonstrates how to reset the status of publications for reprocessing.
"""

import argparse
import logging
import subprocess
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


def reset_download_status():
    """Reset all publications to PENDING download status."""
    with Session(engine) as session:
        stmt = select(PublicationTracking)
        publications = session.exec(stmt).all()
        
        count = 0
        for pub in publications:
            if pub.download_status != DownloadStatus.PENDING:
                pub.download_status = DownloadStatus.PENDING
                pub.download_timestamp = None
                pub.download_attempt_count = 0
                pub.error_message = None
                session.add(pub)
                count += 1
        
        session.commit()
        logger.info(f"Reset download status for {count} publications")


def reset_processing_status():
    """Reset all publications to PENDING processing status."""
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.download_status == DownloadStatus.DOWNLOADED
        )
        publications = session.exec(stmt).all()
        
        count = 0
        for pub in publications:
            if pub.processing_status != ProcessingStatus.PENDING:
                pub.processing_status = ProcessingStatus.PENDING
                pub.processing_timestamp = None
                pub.processing_attempt_count = 0
                pub.error_message = None
                session.add(pub)
                count += 1
        
        session.commit()
        logger.info(f"Reset processing status for {count} publications")


def reset_embedding_status():
    """Reset all publications to PENDING embedding status."""
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.processing_status == ProcessingStatus.PROCESSED
        )
        publications = session.exec(stmt).all()
        
        count = 0
        for pub in publications:
            if pub.embedding_status != EmbeddingStatus.PENDING:
                pub.embedding_status = EmbeddingStatus.PENDING
                pub.embedding_timestamp = None
                pub.embedding_attempt_count = 0
                pub.error_message = None
                session.add(pub)
                count += 1
        
        session.commit()
        logger.info(f"Reset embedding status for {count} publications")


def reset_ingestion_status():
    """Reset all publications to PENDING ingestion status."""
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.embedding_status == EmbeddingStatus.EMBEDDED
        )
        publications = session.exec(stmt).all()
        
        count = 0
        for pub in publications:
            if pub.ingestion_status != IngestionStatus.PENDING:
                pub.ingestion_status = IngestionStatus.PENDING
                pub.ingestion_timestamp = None
                pub.ingestion_attempt_count = 0
                pub.error_message = None
                session.add(pub)
                count += 1
        
        session.commit()
        logger.info(f"Reset ingestion status for {count} publications")


def reset_all_statuses():
    """Reset all publication statuses to PENDING."""
    reset_download_status()
    reset_processing_status()
    reset_embedding_status()
    reset_ingestion_status()
    logger.info("Reset all publication statuses to PENDING")


def main():
    """Process command line arguments and reset statuses."""
    parser = argparse.ArgumentParser(
        description="Reset publication tracking status for the ETL pipeline"
    )
    
    parser.add_argument(
        "--stage",
        choices=["download", "processing", "embedding", "ingestion", "all"],
        default="all",
        help="Pipeline stage to reset (default: all)",
    )
    
    args = parser.parse_args()
    
    # Ensure database is initialized
    ensure_db_initialized()
    
    # Reset status based on the specified stage
    if args.stage == "download":
        reset_download_status()
    elif args.stage == "processing":
        reset_processing_status()
    elif args.stage == "embedding":
        reset_embedding_status()
    elif args.stage == "ingestion":
        reset_ingestion_status()
    else:  # args.stage == "all"
        reset_all_statuses()
    
    # Print current status after reset
    print("Current publication status after reset:")
    subprocess.run([sys.executable, f"{Path(__file__).parent}/test_query_publications.py"])


if __name__ == "__main__":
    main() 