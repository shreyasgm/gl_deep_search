"""
Test script for publication downloading with tracking.

This script tests the integration between the FileDownloader and
PublicationTracker components.
"""

import sys
from pathlib import Path

from storage.database import Database
from utils.gl_file_downloader import GLFileDownloader
from utils.publication_tracker import PublicationTracker

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


def test_download_tracking():
    """Test the download tracking functionality"""
    # Initialize database and tracker
    db = Database()
    tracker = PublicationTracker(db)

    # Test data
    test_pub = {
        "paper_id": "test_paper_001",
        "title": "Test Paper",
        "year": 2024,
        "source_url": "https://example.com/test.pdf",
        "file_urls": ["https://example.com/test.pdf"],
    }

    # Add test publication
    tracker.add_publication(test_pub)

    # Initialize downloader
    downloader = GLFileDownloader(tracker)

    # Test download
    result = downloader.download_file(test_pub["paper_id"], test_pub["file_urls"][0])

    # Check status
    status = tracker.get_publication_status(test_pub["paper_id"])
    return result and status


if __name__ == "__main__":
    test_download_tracking()
