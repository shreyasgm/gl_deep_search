#!/usr/bin/env python3
"""
Simple script to run the GrowthLab scraper
"""

import argparse
import asyncio
import logging

from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.storage.factory import get_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the scraper and save results"""
    parser = argparse.ArgumentParser(description="Run GrowthLab scraper")
    parser.add_argument(
        "--storage-type",
        choices=["local", "cloud"],
        help="Override storage type (local or cloud)",
    )
    args = parser.parse_args()

    # Get storage implementation based on environment or arguments
    storage = get_storage(
        storage_type=args.storage_type if "storage_type" in args else None
    )
    logger.info(f"Using storage type: {storage.__class__.__name__}")

    # Initialize the scraper
    scraper = GrowthLabScraper()
    logger.info("Starting GrowthLab scraper")

    # Get output path from storage abstraction
    output_path = storage.get_path("intermediate/growth_lab_publications.csv")
    # Ensure directory exists
    storage.ensure_dir(output_path.parent)

    # Extract and save publications
    try:
        # Use the update_publications method with the storage instance
        publications = await scraper.update_publications(storage=storage)
        logger.info(f"Processed {len(publications)} publications")
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
