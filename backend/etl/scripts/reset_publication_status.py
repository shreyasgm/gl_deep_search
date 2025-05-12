"""
Reset publication tracking status.

This script resets the tracking status of all publications in the database
to a specified state. This is useful for testing or when you need to
reprocess publications from a certain state.
"""

import subprocess
import sys
from pathlib import Path

from models.tracking import DownloadStatus, ProcessingStatus
from storage.database import Database
from utils.publication_tracker import PublicationTracker

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def reset_publication_status(
    download_status: DownloadStatus = DownloadStatus.PENDING,
    processing_status: ProcessingStatus = ProcessingStatus.PENDING,
):
    """Reset the status of all publications in the database.

    Args:
        download_status: The download status to set for all publications
        processing_status: The processing status to set for all publications
    """
    # Initialize database and tracker
    db = Database()
    tracker = PublicationTracker(db)

    # Reset status for all publications
    tracker.reset_publication_status(download_status, processing_status)

    # Run query script to verify changes
    subprocess.run(
        [sys.executable, f"{Path(__file__).parent}/test_query_publications.py"]
    )


if __name__ == "__main__":
    reset_publication_status()
