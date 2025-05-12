"""
Test script for querying publication tracking data.

This script demonstrates how to retrieve publication tracking data
from the database using the PublicationTracker class.
"""

import sys
from pathlib import Path

from models.tracking import DownloadStatus, ProcessingStatus
from storage.database import Database
from utils.publication_tracker import PublicationTracker

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def test_query_publications():
    """Test querying publications from the database"""
    # Initialize database and tracker
    db = Database()
    tracker = PublicationTracker(db)

    # Query publications
    publications = tracker.get_publications(
        download_status=DownloadStatus.DOWNLOADED,
        processing_status=ProcessingStatus.PROCESSED,
    )

    return publications


if __name__ == "__main__":
    test_query_publications()
