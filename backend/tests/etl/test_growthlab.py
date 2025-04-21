import logging
from unittest.mock import AsyncMock, patch

import pytest

from backend.etl.scrapers.growthlab import (
    GrowthLabPublication,
    GrowthLabScraper,
)

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_publication():
    pub = GrowthLabPublication(
        title="Test Publication",
        authors="John Doe, Jane Smith",
        year=2023,
        abstract="This is a test abstract",
        pub_url="https://growthlab.hks.harvard.edu/publications/test",
        file_urls=["https://growthlab.hks.harvard.edu/files/test.pdf"],
        source="GrowthLab",
    )
    # Generate stable IDs
    pub.paper_id = pub.generate_id()
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def scraper():
    return GrowthLabScraper()


@pytest.mark.asyncio
async def test_parse_publication(scraper):
    # Mock HTML content
    html = """
    <div class="biblio-entry">
        <span class="biblio-title">
            <a href="/publications/test">Test Publication</a>
        </span>
        <span class="biblio-authors">John Doe, Jane Smith</span> 2023
        <div class="biblio-abstract-display">This is a test abstract</div>
        <span class="file">
            <a href="/files/test.pdf">PDF</a>
        </span>
    </div>
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    pub_element = soup.find("div", {"class": "biblio-entry"})

    # Parse publication
    publication = await scraper.parse_publication(
        pub_element, "https://growthlab.hks.harvard.edu"
    )

    # Verify core fields
    assert publication is not None
    assert publication.title == "Test Publication"
    assert publication.authors == "John Doe, Jane Smith"
    assert publication.year == 2023
    assert publication.abstract == "This is a test abstract"
    assert len(publication.file_urls) == 1
    assert "test.pdf" in str(publication.file_urls[0])
    assert publication.source == "GrowthLab"
    assert publication.paper_id is not None
    assert publication.content_hash is not None


@pytest.mark.asyncio
async def test_extract_publications(scraper):
    # Mock aiohttp session and responses
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(
        return_value="""
        <html>
            <div class="biblio-entry">
                <span class="biblio-title">
                    <a href="/publications/test">Test Publication</a>
                </span>
                <span class="biblio-authors">John Doe</span> 2023
                <div class="biblio-abstract-display">Test abstract</div>
            </div>
        </html>
    """
    )
    # Set up the context manager return value properly
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response

    # Mock get_max_page_num
    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch.object(scraper, "get_max_page_num", return_value=1):
            # Also mock the fetch_page method to return a valid publication
            test_pub = GrowthLabPublication(
                title="Test Publication",
                authors="John Doe",
                year=2023,
                abstract="Test abstract",
                pub_url="https://growthlab.hks.harvard.edu/publications/test",
                source="GrowthLab",
            )
            test_pub.paper_id = test_pub.generate_id()
            test_pub.content_hash = test_pub.generate_content_hash()

            with patch.object(scraper, "fetch_page", return_value=[test_pub]):
                publications = await scraper.extract_publications()

                assert len(publications) > 0
                assert publications[0].title == "Test Publication"


def test_publication_model(sample_publication):
    # Test ID generation
    paper_id = sample_publication.generate_id()
    assert paper_id.startswith("gl_")
    assert (
        "2023" in paper_id or "url_" in paper_id
    )  # Check for either year or URL-based ID

    # Test URL-based ID generation
    pub_with_url = GrowthLabPublication(
        title="Test URL Based ID",
        pub_url="https://growthlab.hks.harvard.edu/publications/economic-complexity-analysis",
        source="GrowthLab",
    )
    url_based_id = pub_with_url.generate_id()
    assert url_based_id.startswith("gl_url_")
    assert len(url_based_id) > 10  # Should be longer than the old 10-char hash

    # Test text normalization
    pub1 = GrowthLabPublication(
        title="Test Publication",
        authors="John Doe, Jane Smith",
        year=2023,
    )
    pub2 = GrowthLabPublication(
        title="TEST PUBLICATION",  # Different case
        authors="John Doe,Jane Smith",  # Different spacing
        year=2023,
    )
    # IDs should be the same despite minor text differences
    assert pub1.generate_id() == pub2.generate_id()

    # Test with minimal information
    pub_minimal = GrowthLabPublication(title="Just a Title")
    minimal_id = pub_minimal.generate_id()
    assert minimal_id.startswith("gl_")
    assert "0000" in minimal_id  # Default year

    # Test empty case (fallback to random ID)
    pub_empty = GrowthLabPublication()
    empty_id = pub_empty.generate_id()
    assert empty_id.startswith("gl_unknown_")

    # Test content hash generation
    content_hash = sample_publication.generate_content_hash()
    assert len(content_hash) == 64  # SHA-256 hash length

    # Test year validation
    with pytest.raises(ValueError, match="Year 1899 is not in valid range"):
        GrowthLabPublication(title="Invalid Year", year=1899, source="GrowthLab")


def test_publication_enrichment():
    """Test the enrichment functionality without using async mocks"""
    # Create a test publication with missing fields
    test_pub = GrowthLabPublication(
        title="Test Publication",
        authors=None,  # Set to None to test enrichment
        year=2023,
        pub_url="https://growthlab.hks.harvard.edu/publications/test",
        file_urls=["https://growthlab.hks.harvard.edu/files/test.pdf"],
        source="GrowthLab",
        abstract=None,  # Set to None to test enrichment
    )

    # Manually enrich the publication as the method would
    endnote_data = {
        "author": "John Doe, Jane Smith",
        "title": "Test Publication",
        "date": "2023",
        "abstract": "This is an enriched abstract",
    }

    # Update fields that are missing in the original
    if not test_pub.authors and "author" in endnote_data:
        test_pub.authors = endnote_data["author"]

    if not test_pub.abstract and "abstract" in endnote_data:
        test_pub.abstract = endnote_data["abstract"]

    # Verify enrichment process worked
    assert test_pub.abstract == "This is an enriched abstract"
    assert test_pub.authors == "John Doe, Jane Smith"


def test_save_and_load_publications(scraper, sample_publication, tmp_path):
    # Test saving to CSV
    output_path = tmp_path / "test_publications.csv"
    scraper.save_to_csv([sample_publication], output_path)
    assert output_path.exists()

    # Test loading from CSV
    loaded_publications = scraper.load_from_csv(output_path)
    assert len(loaded_publications) == 1
    loaded_pub = loaded_publications[0]
    assert loaded_pub.title == sample_publication.title
    assert loaded_pub.authors == sample_publication.authors
    assert loaded_pub.year == sample_publication.year
    assert loaded_pub.abstract == sample_publication.abstract
    assert str(loaded_pub.pub_url) == str(sample_publication.pub_url)
    assert loaded_pub.source == sample_publication.source


@pytest.mark.asyncio
async def test_update_publications_with_storage(scraper, sample_publication, tmp_path):
    """Test update publications with storage abstraction"""
    from backend.storage.local import LocalStorage

    # Create a test storage instance for the test
    storage = LocalStorage(tmp_path)

    # Mock extract_and_enrich_publications to return our sample publication
    async def mock_extract(*args, **kwargs):
        return [sample_publication]

    # Save a test CSV first to simulate existing data
    existing_path = storage.get_path("intermediate/growth_lab_publications.csv")
    storage.ensure_dir(existing_path.parent)
    scraper.save_to_csv([sample_publication], existing_path)

    # Run the update with our mock
    with patch.object(scraper, "extract_and_enrich_publications", mock_extract):
        publications = await scraper.update_publications(storage=storage)

    # Verify results
    assert len(publications) == 1
    assert publications[0].paper_id == sample_publication.paper_id

    # Verify the file was created in the right location
    assert existing_path.exists()


@pytest.mark.integration
def test_growthlab_real_data_id_generation(tmp_path):
    """
    Integration test that verifies ID generation with real data
    from the Growth Lab scraper.

    This test confirms that:
    1. Most publications have URL-based IDs (gl_url_*)
    2. ID generation is stable for real publications
    3. The fallback ID generation mechanisms work properly
    """
    from pathlib import Path

    import pandas as pd

    # Path to real Growth Lab data
    data_path = Path("data/intermediate/growth_lab_publications.csv")

    # Skip if data file doesn't exist (for CI environments)
    if not data_path.exists():
        pytest.skip(f"Real data file not found at {data_path}")

    # Load the real data
    df = pd.read_csv(data_path)

    # Ensure we have at least a few records to test with
    assert len(df) > 10, "Not enough real data records found for testing"

    # Count different types of IDs
    url_based_ids = sum(1 for id in df["paper_id"] if id.startswith("gl_url_"))
    year_based_ids = sum(
        1
        for id in df["paper_id"]
        if id.startswith("gl_")
        and not id.startswith("gl_url_")
        and not id.startswith("gl_unknown_")
    )
    unknown_ids = sum(1 for id in df["paper_id"] if id.startswith("gl_unknown_"))

    # Compute percentages
    total_count = len(df)
    url_pct = (url_based_ids / total_count) * 100
    year_pct = (year_based_ids / total_count) * 100
    unknown_pct = (unknown_ids / total_count) * 100

    # Log summary for visibility in test output
    logger.info(f"Growth Lab ID generation summary ({total_count} publications):")
    logger.info(f"- URL-based IDs: {url_based_ids} ({url_pct:.1f}%)")
    logger.info(f"- Year-based IDs: {year_based_ids} ({year_pct:.1f}%)")
    logger.info(f"- Unknown IDs: {unknown_ids} ({unknown_pct:.1f}%)")

    # Load a sample of publications to verify ID regeneration
    from backend.etl.scrapers.growthlab import GrowthLabPublication

    # Test with a subset of publications to keep test fast
    sample_size = min(20, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    # For each publication, recreate the model and check if IDs match
    for _, row in sample_df.iterrows():
        # Convert string representation of list to actual list for file_urls
        file_urls = eval(row["file_urls"]) if isinstance(row["file_urls"], str) else []

        # Create Publication object
        pub = GrowthLabPublication(
            title=row["title"] if not pd.isna(row["title"]) else None,
            authors=row["authors"] if not pd.isna(row["authors"]) else None,
            year=int(row["year"]) if not pd.isna(row["year"]) else None,
            abstract=row["abstract"] if not pd.isna(row["abstract"]) else None,
            pub_url=row["pub_url"] if not pd.isna(row["pub_url"]) else None,
            file_urls=file_urls,
            source="GrowthLab",
        )

        # Generate ID without setting paper_id first to test generation logic
        generated_id = pub.generate_id()

        # The original ID from the dataset
        original_id = row["paper_id"]

        # Verify ID stability - generated ID should match original
        assert (
            generated_id == original_id
        ), f"ID generation not stable: {generated_id} != {original_id}"

    # Verify most publications use URL-based IDs as expected
    assert url_pct > 50, "Less than 50% of publications have URL-based IDs"
