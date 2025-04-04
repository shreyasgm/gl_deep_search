"""
Script to run the OpenAlex scraper
"""

import argparse
import asyncio
import logging

from backend.etl.scrapers.openalex import OpenAlexClient
from backend.storage.factory import get_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Run the OpenAlex scraper and save results"""
    parser = argparse.ArgumentParser(description="Run OpenAlex scraper")
    parser.add_argument(
        "--storage-type",
        choices=["local", "cloud"],
        help="Override storage type (local or cloud)",
    )
    args = parser.parse_args()

    # Get storage implementation based on environment or arguments
    storage = get_storage(
        storage_type=args.storage_type if hasattr(args, "storage_type") else None
    )
    logger.info(f"Using storage type: {storage.__class__.__name__}")

    # Initialize the OpenAlex client
    client = OpenAlexClient()
    logger.info("Starting OpenAlex scraper")

    # Get output path from storage abstraction
    output_path = storage.get_path("intermediate/openalex_publications.csv")

    # Ensure directory exists
    storage.ensure_dir(output_path.parent)

    # Extract, update and save publications
    try:
        # Use the update_publications method with the storage instance
        publications = await client.update_publications(storage=storage)
        logger.info(f"Processed {len(publications)} publications")
    except Exception as e:
        logger.error(f"Error running OpenAlex scraper: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
