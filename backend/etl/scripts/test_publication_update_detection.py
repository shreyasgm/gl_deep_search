"""
Test script for publication update detection.

This script tests the automatic detection of new and updated publications
with database integration using the PublicationTracker.
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pydantic import HttpUrl
from sqlmodel import Session, select

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import PublicationTracking, DownloadStatus
from backend.etl.utils.publication_tracker import PublicationTracker
from backend.storage.database import engine, ensure_db_initialized
from backend.storage.factory import get_storage

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_publication(pub_id, title, is_new=True):
    """Create a test publication for testing"""
    pub = GrowthLabPublication(
        paper_id=f"test_{pub_id}",
        title=f"Test Publication {title}",
        authors="Test Author",
        year=2023,
        pub_url=HttpUrl("https://example.com/publication"),
        file_urls=[HttpUrl("https://example.com/file1.pdf")],
        abstract="This is a test publication.",
        source="Test",
    )
    
    # Generate content hash
    pub.content_hash = pub.generate_content_hash()
    
    return pub


async def run_test_scenario():
    """Run a test scenario to validate update detection"""
    # Ensure database is initialized
    ensure_db_initialized()
    
    # Initialize tracker
    tracker = PublicationTracker()
    
    # Clear existing test publications (if any)
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.publication_id.like("test_%")
        )
        test_pubs = session.exec(stmt).all()
        if test_pubs:
            for pub in test_pubs:
                session.delete(pub)
            session.commit()
            logger.info(f"Cleared {len(test_pubs)} existing test publications")
    
    # Scenario 1: Adding brand new publications
    logger.info("SCENARIO 1: Adding new publications")
    
    # Create test publications
    new_publications = [
        create_test_publication("new1", "Brand New 1"),
        create_test_publication("new2", "Brand New 2"),
        create_test_publication("new3", "Brand New 3"),
    ]
    
    # Add to tracker
    add_results = tracker.add_publications(new_publications)
    
    # Verify results
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.publication_id.like("test_new%")
        )
        tracked_pubs = session.exec(stmt).all()
        
        logger.info(f"Added {len(tracked_pubs)} new publications")
        for pub in tracked_pubs:
            logger.info(f"  - {pub.publication_id}: {pub.title} (Discovery: {pub.discovery_timestamp})")
    
    # Scenario 2: Adding updated publications
    logger.info("\nSCENARIO 2: Updating existing publications")
    
    # Create updated versions of existing publications
    updated_publications = []
    for i, pub in enumerate(new_publications):
        if i < 2:  # Update only the first two
            # Change title to mark as updated
            updated_pub = GrowthLabPublication(**pub.model_dump())
            updated_pub.title = f"{pub.title} - UPDATED"
            updated_pub.abstract = f"{pub.abstract} With additional information."
            # Update content hash
            updated_pub.content_hash = updated_pub.generate_content_hash()
            updated_publications.append(updated_pub)
    
    # Add new publication
    new_pub = create_test_publication("new4", "Brand New 4")
    updated_publications.append(new_pub)
    
    # Add to tracker
    update_results = tracker.add_publications(updated_publications)
    
    # Verify results
    with Session(engine) as session:
        # Check all publications
        stmt = select(PublicationTracking).where(
            PublicationTracking.publication_id.like("test_%")
        )
        all_pubs = session.exec(stmt).all()
        
        # Count updated vs new
        updated_count = sum(1 for p in all_pubs if "UPDATED" in (p.title or ""))
        total_count = len(all_pubs)
        
        logger.info(f"Total publications: {total_count}")
        logger.info(f"Updated publications: {updated_count}")
        logger.info(f"New publication added: {'test_new4' in [p.publication_id for p in all_pubs]}")
        
        # Show details of each publication
        for pub in all_pubs:
            logger.info(f"  - {pub.publication_id}: {pub.title}")
            logger.info(f"    Discovery: {pub.discovery_timestamp}")
            logger.info(f"    Last Updated: {pub.last_updated}")
            is_updated = pub.discovery_timestamp != pub.last_updated
            logger.info(f"    Updated since discovery: {is_updated}")
    
    # Scenario 3: Re-adding unchanged publications
    logger.info("\nSCENARIO 3: Re-adding unchanged publications")
    
    # Re-add the same publications without changes
    unchanged_results = tracker.add_publications(updated_publications)
    
    # Verify no updates happened
    with Session(engine) as session:
        stmt = select(PublicationTracking).where(
            PublicationTracking.publication_id.like("test_%")
        ).order_by(PublicationTracking.last_updated.desc())
        
        latest_pubs = session.exec(stmt).all()
        last_updated_times = [pub.last_updated for pub in latest_pubs]
        
        # Calculate time since last update
        now = datetime.now()
        recent_updates = sum(1 for t in last_updated_times if (now - t) < timedelta(seconds=5))
        
        logger.info(f"Publications with recent updates: {recent_updates}")
        logger.info("No updates should have occurred in this scenario")
    
    return {
        "initial_count": len(new_publications),
        "final_count": total_count,
        "updated_count": updated_count
    }


async def test_automatic_detection():
    """Run end-to-end testing of publication detection logic"""
    storage = get_storage()
    tracker = PublicationTracker()
    
    # Run the test scenario
    results = await run_test_scenario()
    
    # Print summary
    print("\n=== Test Results ===")
    print(f"Initial publications: {results['initial_count']}")
    print(f"Final publication count: {results['final_count']}")
    print(f"Updated publications: {results['updated_count']}")
    
    # Pass/fail assessment
    expected_final = results['initial_count'] + 1  # One new publication added in scenario 2
    expected_updated = 2  # Two publications updated in scenario 2
    
    if results['final_count'] == expected_final and results['updated_count'] == expected_updated:
        print("\nTEST PASSED: Publication detection logic is working correctly!")
    else:
        print("\nTEST FAILED: Publication detection did not work as expected.")
        print(f"  Expected {expected_final} total publications, got {results['final_count']}")
        print(f"  Expected {expected_updated} updated publications, got {results['updated_count']}")


if __name__ == "__main__":
    asyncio.run(test_automatic_detection()) 