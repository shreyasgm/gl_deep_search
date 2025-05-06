"""
Test script for publication downloading with tracking.

This script tests the integration between the FileDownloader and
PublicationTracker components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.etl.models.tracking import DownloadStatus
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.utils.gl_file_downloader import FileDownloader
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.database import ensure_db_initialized
from backend.storage.factory import get_storage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Test downloading publications with tracking."""
    # Ensure database is initialized
    ensure_db_initialized()
    
    # Create storage, tracker and downloader
    storage = get_storage()
    tracker = PublicationTracker()
    downloader = FileDownloader(
        storage=storage,
        concurrency_limit=2,
        publication_tracker=tracker
    )
    
    try:
        # Create scraper and get publications
        scraper = GrowthLabScraper()
        publications = await scraper.extract_publications()
        
        logger.info(f"Extracted {len(publications)} publications")
        
        # Limit to a few publications for testing
        test_publications = publications[:3]
        
        # Download publications (this will automatically track them)
        results = await downloader.download_publications(
            test_publications,
            overwrite=False,
            progress_bar=True
        )
        
        # Print download results
        print("\n=== Download Results ===")
        for pub_result in results:
            pub_id = pub_result["publication_id"]
            title = pub_result["title"]
            downloads = pub_result["downloads"]
            
            print(f"\nPublication: {title} (ID: {pub_id})")
            
            successful_downloads = sum(1 for d in downloads if d["success"])
            total_downloads = len(downloads)
            
            print(f"Download status: {successful_downloads}/{total_downloads} files successful")
            
            # Get tracking status from database
            status = tracker.get_publication_status(pub_id)
            if status:
                print(f"Tracked status: {status['download_status'].value}")
                print(f"Timestamp: {status['download_timestamp']}")
                if status['error_message']:
                    print(f"Error: {status['error_message']}")
            else:
                print("Publication not found in tracking database")
    
    finally:
        # Clean up resources
        if hasattr(scraper, '_session') and scraper._session:
            await scraper._session.close()
        
        if hasattr(downloader, '_session') and downloader._session:
            await downloader._session.close()


if __name__ == "__main__":
    asyncio.run(main()) 