"""
Test script for publication tracking metadata storage
"""

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from sqlmodel import Session, select

# Add parent directory to path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.etl.models.tracking import PublicationTracking, DownloadStatus, ProcessingStatus
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.storage.database import engine, ensure_db_initialized

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


async def sample_publications(scraper: GrowthLabScraper, count: int = 2):
    """Sample a few random publications from the Growth Lab website"""
    # Get all publications
    publications = await scraper.extract_publications()
    logger.info(f"Extracted {len(publications)} publications")
    
    # Randomly sample if we have more than requested
    if len(publications) > count:
        return random.sample(publications, count)
    return publications


async def main():
    """Main function to test publication tracking"""
    # Initialize the database
    ensure_db_initialized()
    
    # Create a scraper
    scraper = GrowthLabScraper()
    
    # Sample a few publications
    publications = await sample_publications(scraper)
    
    # Store publications in tracking database
    with Session(engine) as session:
        for pub in publications:
            # Convert to tracking model
            tracking = PublicationTracking(
                publication_id=pub.paper_id,
                source_url=str(pub.pub_url) if pub.pub_url else "",
                title=pub.title,
                authors=pub.authors,
                year=pub.year,
                abstract=pub.abstract,
                content_hash=pub.content_hash,
            )
            
            # Set file URLs
            tracking.file_urls = [str(url) for url in pub.file_urls]
            
            # Insert into database
            session.add(tracking)
            
            # Simulate download status (just for testing)
            tracking.update_download_status(DownloadStatus.DOWNLOADED)
            
            # Simulate processing (just for testing)
            if pub.file_urls:
                tracking.update_processing_status(ProcessingStatus.PROCESSED)
            else:
                tracking.update_processing_status(
                    ProcessingStatus.FAILED, 
                    error="No files available for processing"
                )
            
            logger.info(f"Added publication tracking for: {pub.title}")
        
        # Commit all changes
        session.commit()
    
    # Verify we can query the data
    with Session(engine) as session:
        # Query all tracked publications
        stmt = select(PublicationTracking)
        results = session.exec(stmt).all()
        
        print("\n=== Tracked Publications ===")
        for idx, pub in enumerate(results, 1):
            print(f"\n{idx}. {pub.title} ({pub.year if pub.year else 'Unknown Year'})")
            print(f"   ID: {pub.publication_id}")
            print(f"   URL: {pub.source_url}")
            print(f"   Download: {pub.download_status.value} at {pub.download_timestamp}")
            print(f"   Processing: {pub.processing_status.value} at {pub.processing_timestamp}")
            print(f"   Error: {pub.error_message or 'None'}")
            print(f"   File URLs: {', '.join(pub.file_urls) if pub.file_urls else 'None'}")


if __name__ == "__main__":
    asyncio.run(main())