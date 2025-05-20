from unittest.mock import AsyncMock, Mock

import pytest
from sqlmodel import Session, select

from backend.etl.models.publications import GrowthLabPublication, OpenAlexPublication
from backend.etl.models.tracking import (
    DownloadStatus,
    EmbeddingStatus,
    IngestionStatus,
    ProcessingStatus,
    PublicationTracking,
)
from backend.etl.utils.publication_tracker import ProcessingPlan, PublicationTracker
from backend.storage.database import engine


@pytest.fixture(autouse=True)
def clear_publication_tracking():
    # Clear the publication_tracking table before each test for isolation
    with Session(engine) as session:
        session.exec("DELETE FROM publication_tracking")
        session.commit()


@pytest.fixture
def tracker():
    return PublicationTracker(ensure_db=False)


@pytest.fixture
def session():
    with Session(engine) as session:
        yield session


@pytest.fixture
def mock_growthlab_pub():
    return GrowthLabPublication(
        paper_id="test-glab-1",
        title="Test GrowthLab Paper",
        authors="Author 1, Author 2",
        year=2023,
        abstract="Test abstract",
        pub_url="https://growthlab.test/paper1",
        content_hash="hash1",
        file_urls=["https://growthlab.test/paper1.pdf"],
    )


@pytest.fixture
def mock_openalex_pub():
    return OpenAlexPublication(
        paper_id="test-oa-1",
        title="Test OpenAlex Paper",
        authors="Author 3, Author 4",
        year=2023,
        abstract="Test abstract",
        pub_url="https://openalex.test/paper1",
        content_hash="hash2",
        file_urls=["https://openalex.test/paper1.pdf"],
    )


@pytest.mark.asyncio
async def test_discover_publications(tracker):
    # Mock the scrapers
    tracker.growthlab_scraper.discover_publications = Mock(
        return_value=[
            GrowthLabPublication(
                paper_id="test-glab-1",
                title="Test GrowthLab Paper",
                authors="Author 1",
                year=2023,
                abstract="Test",
                content_hash="hash1",
                file_urls=[],
            )
        ]
    )

    tracker.openalex_client.fetch_publications = AsyncMock(
        return_value=[
            OpenAlexPublication(
                paper_id="test-oa-1",
                title="Test OpenAlex Paper",
                authors="Author 2",
                year=2023,
                abstract="Test",
                content_hash="hash2",
                file_urls=[],
            )
        ]
    )

    publications = await tracker.discover_publications()

    assert len(publications) == 2
    assert isinstance(publications[0][0], GrowthLabPublication)
    assert isinstance(publications[1][0], OpenAlexPublication)
    assert publications[0][1] == "growthlab"
    assert publications[1][1] == "openalex"


def test_generate_processing_plan_new_publication(tracker, session, mock_growthlab_pub):
    plan = tracker.generate_processing_plan(mock_growthlab_pub, session)

    assert isinstance(plan, ProcessingPlan)
    assert plan.publication_id == mock_growthlab_pub.paper_id
    assert plan.needs_download
    assert plan.needs_processing
    assert plan.needs_embedding
    assert plan.needs_ingestion
    assert plan.reason == "New publication"


def test_generate_processing_plan_content_changed(tracker, session, mock_growthlab_pub):
    # Add initial publication with all required fields
    tracking = PublicationTracking(
        publication_id=mock_growthlab_pub.paper_id,
        source_url=str(mock_growthlab_pub.pub_url),
        title=mock_growthlab_pub.title,
        authors=mock_growthlab_pub.authors,
        year=mock_growthlab_pub.year,
        abstract=mock_growthlab_pub.abstract,
        content_hash="old_hash",
    )
    tracking.file_urls = [str(url) for url in mock_growthlab_pub.file_urls]
    session.add(tracking)
    session.commit()

    # Generate plan with new content hash
    plan = tracker.generate_processing_plan(mock_growthlab_pub, session)

    assert plan.needs_download
    assert plan.needs_processing
    assert plan.needs_embedding
    assert plan.needs_ingestion
    assert plan.reason == "Content hash changed"


def test_add_publication_new(tracker, session, mock_growthlab_pub):
    tracking = tracker.add_publication(mock_growthlab_pub, session)
    # Query the object again to ensure it's attached to the session
    stmt = select(PublicationTracking).where(
        PublicationTracking.publication_id == mock_growthlab_pub.paper_id
    )
    tracking_db = session.exec(stmt).first()
    assert tracking_db.publication_id == mock_growthlab_pub.paper_id
    assert tracking_db.title == mock_growthlab_pub.title
    assert tracking_db.download_status == DownloadStatus.PENDING
    assert tracking_db.processing_status == ProcessingStatus.PENDING
    assert tracking_db.embedding_status == EmbeddingStatus.PENDING
    assert tracking_db.ingestion_status == IngestionStatus.PENDING


def test_add_publication_update(tracker, session, mock_growthlab_pub):
    # Add initial publication
    tracker.add_publication(mock_growthlab_pub, session)
    # Update with new content
    mock_growthlab_pub.content_hash = "new_hash"
    updated = tracker.add_publication(mock_growthlab_pub, session)
    stmt = select(PublicationTracking).where(
        PublicationTracking.publication_id == mock_growthlab_pub.paper_id
    )
    tracking_db = session.exec(stmt).first()
    assert updated.publication_id == tracking_db.publication_id
    assert updated.content_hash == "new_hash"
    assert updated.download_status == DownloadStatus.PENDING


def test_update_download_status(tracker, session, mock_growthlab_pub):
    # Add publication
    tracker.add_publication(mock_growthlab_pub, session)
    # Update status
    success = tracker.update_download_status(
        mock_growthlab_pub.paper_id, DownloadStatus.DOWNLOADED, session=session
    )
    assert success
    # Query the object again
    stmt = select(PublicationTracking).where(
        PublicationTracking.publication_id == mock_growthlab_pub.paper_id
    )
    tracking_db = session.exec(stmt).first()
    assert tracking_db.download_status == DownloadStatus.DOWNLOADED


def test_get_publications_for_download(tracker, session, mock_growthlab_pub):
    # Add publication
    tracker.add_publication(mock_growthlab_pub, session)
    # Get publications for download
    pubs = tracker.get_publications_for_download(session=session)
    assert len(pubs) == 1
    assert pubs[0].publication_id == mock_growthlab_pub.paper_id


def test_get_publication_status(tracker, session, mock_growthlab_pub):
    # Add publication
    tracker.add_publication(mock_growthlab_pub, session)
    # Get status
    status = tracker.get_publication_status(mock_growthlab_pub.paper_id, session)
    assert status is not None
    assert status["publication_id"] == mock_growthlab_pub.paper_id
    assert status["download_status"] == DownloadStatus.PENDING
    assert status["processing_status"] == ProcessingStatus.PENDING
    assert status["embedding_status"] == EmbeddingStatus.PENDING
    assert (
        status["ingestion_status"] == DownloadStatus.PENDING
        or status["ingestion_status"] == IngestionStatus.PENDING
    )
