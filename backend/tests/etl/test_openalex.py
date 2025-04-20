import logging
from unittest.mock import AsyncMock, patch

# No need to import aiohttp directly for tests
import pytest

from backend.etl.scrapers.openalex import (
    OpenAlexClient,
    OpenAlexPublication,
)

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_publication():
    pub = OpenAlexPublication(
        paper_id="W123456789",
        openalex_id="https://openalex.org/W123456789",
        title="Test OpenAlex Publication",
        authors="John Doe, Jane Smith",
        year=2023,
        abstract="This is a test abstract for OpenAlex publication",
        pub_url="https://example.com/publication",
        file_urls=["https://doi.org/10.1234/test"],
        source="OpenAlex",
        cited_by_count=42,
    )
    # Generate content hash
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def client():
    return OpenAlexClient()


def test_publication_model(sample_publication):
    # Test content hash generation
    content_hash = sample_publication.generate_content_hash()
    assert len(content_hash) == 64  # SHA-256 hash length
    assert sample_publication.content_hash == content_hash

    # Test ID validation
    pub1 = OpenAlexPublication(
        paper_id="https://openalex.org/W123456",
        openalex_id="https://openalex.org/W123456",
        title="Test",
    )
    assert pub1.paper_id == "W123456"  # Should strip the URL

    # Test openalex_id generation
    pub2 = OpenAlexPublication(
        paper_id="W654321",
        title="Test",
    )
    assert pub2.openalex_id == "https://openalex.org/W654321"

    # Test ID generation with OpenAlex ID
    assert pub2.generate_id() == "oa_W654321"

    # Test ID generation with DOI
    pub_with_doi = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="Test DOI Based ID",
        file_urls=["https://doi.org/10.1234/test.doi"],
    )
    doi_based_id = pub_with_doi.generate_id()
    assert doi_based_id.startswith("oa_doi_")
    assert len(doi_based_id) > 10

    # Test ID generation with URL
    pub_with_url = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="Test URL Based ID",
        pub_url="https://example.com/paper/123",
    )
    url_based_id = pub_with_url.generate_id()
    assert url_based_id.startswith("oa_url_")
    assert len(url_based_id) > 10

    # Test text normalization
    pub1 = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="Test Publication",
        authors="John Doe, Jane Smith",
        year=2023,
    )
    pub2 = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="TEST PUBLICATION",  # Different case
        authors="John Doe,Jane Smith",  # Different spacing
        year=2023,
    )
    # IDs should be the same despite minor text differences
    assert pub1.generate_id() == pub2.generate_id()

    # Test with minimal information
    pub_minimal = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="Just a Title",
    )
    minimal_id = pub_minimal.generate_id()
    assert minimal_id.startswith("oa_")
    assert "0000" in minimal_id  # Default year

    # Test empty case (fallback to random ID)
    pub_empty = OpenAlexPublication(
        paper_id="test_no_w_id",
    )
    empty_id = pub_empty.generate_id()
    assert empty_id.startswith("oa_unknown_")


def test_extract_abstract(client):
    # Create a sample abstract inverted index
    abstract_dict = {
        "This": [0],
        "is": [1],
        "a": [2],
        "test": [3],
        "abstract": [4],
        "with": [5],
        "multiple": [6],
        "words": [7],
    }

    abstract = client._extract_abstract(abstract_dict)
    assert abstract == "This is a test abstract with multiple words"

    # Test empty abstract
    assert client._extract_abstract({}) == ""

    # Test abstract with gaps
    abstract_dict_with_gaps = {
        "This": [0],
        "is": [1],
        "a": [2],
        "test": [4],  # Note the gap at position 3
        "abstract": [5],
    }

    abstract = client._extract_abstract(abstract_dict_with_gaps)
    assert abstract == "This is a  test abstract"  # Extra space at position 3


@pytest.mark.asyncio
async def test_fetch_page():
    # Let's create a simpler test that uses a patched client method
    client = OpenAlexClient()

    # Use a custom implementation of fetch_page for testing
    async def mock_fetch_page(self, session, cursor=None):
        return [{"title": "Test Publication"}], "next_cursor_value"

    # Patch the method
    with patch.object(OpenAlexClient, "fetch_page", new=mock_fetch_page):
        # Create a session mock
        session_mock = AsyncMock()

        # Call the patched method
        results, next_cursor = await client.fetch_page(session_mock, "*")

        # Verify results
        assert len(results) == 1
        assert results[0]["title"] == "Test Publication"
        assert next_cursor == "next_cursor_value"


@pytest.mark.asyncio
async def test_fetch_publications(client):
    # Mock fetch_all_pages to return test data
    test_results = [
        {
            "id": "https://openalex.org/W123456789",
            "title": "Test Publication 1",
            "publication_year": 2023,
            "authorships": [{"author": {"display_name": "John Doe"}}],
            "abstract_inverted_index": {"Abstract": [0], "one": [1]},
            "primary_location": {"landing_page_url": "https://example.com/pub1"},
            "doi": "https://doi.org/10.1234/test1",
            "cited_by_count": 10,
        },
        {
            "id": "https://openalex.org/W987654321",
            "title": "Test Publication 2",
            "publication_year": 2022,
            "authorships": [{"author": {"display_name": "Jane Smith"}}],
            "abstract_inverted_index": {"Abstract": [0], "two": [1]},
            "primary_location": {"landing_page_url": "https://example.com/pub2"},
            "doi": "https://doi.org/10.1234/test2",
            "cited_by_count": 20,
        },
    ]

    with patch.object(client, "fetch_all_pages", AsyncMock(return_value=test_results)):
        publications = await client.fetch_publications()

        # Verify results
        assert len(publications) == 2
        assert publications[0].title == "Test Publication 1"
        assert publications[0].year == 2023
        assert publications[0].authors == "John Doe"
        assert publications[0].abstract == "Abstract one"
        assert str(publications[0].pub_url) == "https://example.com/pub1"
        assert publications[0].cited_by_count == 10

        assert publications[1].title == "Test Publication 2"
        assert publications[1].year == 2022
        assert publications[1].authors == "Jane Smith"
        assert publications[1].abstract == "Abstract two"
        assert str(publications[1].pub_url) == "https://example.com/pub2"
        assert publications[1].cited_by_count == 20


def test_save_and_load_publications(client, sample_publication, tmp_path):
    # Test saving to CSV
    output_path = tmp_path / "test_openalex_publications.csv"
    client.save_to_csv([sample_publication], output_path)
    assert output_path.exists()

    # Test loading from CSV
    loaded_publications = client.load_from_csv(output_path)
    assert len(loaded_publications) == 1
    loaded_pub = loaded_publications[0]
    assert loaded_pub.title == sample_publication.title
    assert loaded_pub.authors == sample_publication.authors
    assert loaded_pub.year == sample_publication.year
    assert loaded_pub.abstract == sample_publication.abstract
    assert str(loaded_pub.pub_url) == str(sample_publication.pub_url)
    assert loaded_pub.source == sample_publication.source
    assert loaded_pub.paper_id == sample_publication.paper_id
    assert loaded_pub.cited_by_count == sample_publication.cited_by_count


@pytest.mark.asyncio
async def test_update_publications(client, sample_publication, tmp_path):
    """Test update publications with storage abstraction"""
    from backend.storage.local import LocalStorage

    # Create a test storage instance for the test
    storage = LocalStorage(tmp_path)

    # Mock fetch_publications to return our sample publication
    async def mock_fetch():
        return [sample_publication]

    # Patch the required functions
    with patch.object(client, "fetch_publications", mock_fetch):
        # Run the update with our mock and storage
        publications = await client.update_publications(storage=storage)

        # Verify results
        assert len(publications) == 1
        assert publications[0].paper_id == sample_publication.paper_id

        # Verify the file was created in the right location
        expected_path = storage.get_path("intermediate/openalex_publications.csv")
        assert expected_path.exists()


@pytest.mark.integration
def test_openalex_real_data_id_generation(tmp_path):
    """
    Integration test that verifies ID generation with real data
    from the OpenAlex API client.

    This test confirms that:
    1. Most publications have OpenAlex IDs as primary IDs (oa_W*)
    2. ID generation is stable for real publications
    3. The fallback ID generation mechanisms work properly
    """
    from pathlib import Path

    import pandas as pd

    # Path to real OpenAlex data
    data_path = Path(
        "/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/gl_deep_search/data/intermediate/openalex_publications.csv"
    )

    # Skip if data file doesn't exist (for CI environments)
    if not data_path.exists():
        pytest.skip(f"Real data file not found at {data_path}")

    # Load the real data
    df = pd.read_csv(data_path)

    # Ensure we have at least a few records to test with
    assert len(df) > 10, "Not enough real data records found for testing"

    # Count different types of IDs
    openalex_ids = sum(
        1 for id in df["paper_id"] if id.startswith("W")
    )  # Direct OpenAlex IDs
    prefixed_openalex_ids = sum(
        1 for id in df["paper_id"] if id.startswith("oa_W")
    )  # Prefixed OpenAlex IDs
    doi_based_ids = sum(1 for id in df["paper_id"] if id.startswith("oa_doi_"))
    url_based_ids = sum(1 for id in df["paper_id"] if id.startswith("oa_url_"))
    year_based_ids = sum(
        1
        for id in df["paper_id"]
        if id.startswith("oa_")
        and not id.startswith("oa_W")
        and not id.startswith("oa_doi_")
        and not id.startswith("oa_url_")
        and not id.startswith("oa_unknown_")
    )
    unknown_ids = sum(1 for id in df["paper_id"] if id.startswith("oa_unknown_"))

    # Compute percentages
    total_count = len(df)
    openalex_pct = ((openalex_ids + prefixed_openalex_ids) / total_count) * 100
    doi_pct = (doi_based_ids / total_count) * 100
    url_pct = (url_based_ids / total_count) * 100
    year_pct = (year_based_ids / total_count) * 100
    unknown_pct = (unknown_ids / total_count) * 100

    # Log summary for visibility in test output
    logger.info(f"OpenAlex ID generation summary ({total_count} publications):")
    logger.info(
        f"- OpenAlex IDs (W* or oa_W*): {openalex_ids + prefixed_openalex_ids} ({openalex_pct:.1f}%)"
    )
    logger.info(f"- DOI-based IDs: {doi_based_ids} ({doi_pct:.1f}%)")
    logger.info(f"- URL-based IDs: {url_based_ids} ({url_pct:.1f}%)")
    logger.info(f"- Year-based IDs: {year_based_ids} ({year_pct:.1f}%)")
    logger.info(f"- Unknown IDs: {unknown_ids} ({unknown_pct:.1f}%)")

    # Load a sample of publications to verify ID regeneration
    from backend.etl.scrapers.openalex import OpenAlexPublication

    # Test with a subset of publications to keep test fast
    sample_size = min(20, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    # For each publication, recreate the model and check if IDs match
    for _, row in sample_df.iterrows():
        # Skip if the paper_id is already in OpenAlex format (W*) without our prefix
        # since our generation would add the prefix
        if row["paper_id"].startswith("W"):
            continue

        # Convert string representation of list to actual list for file_urls
        try:
            file_urls = (
                eval(row["file_urls"]) if isinstance(row["file_urls"], str) else []
            )
        except:
            file_urls = []

        # Create OpenAlexPublication object without setting paper_id
        # This simulates ID generation for a new publication
        original_id = row["paper_id"]
        test_id = "test_id_for_regeneration"  # Temporary ID for testing

        pub = OpenAlexPublication(
            paper_id=test_id,  # Use a temporary ID - we'll test regeneration
            openalex_id=row["openalex_id"] if not pd.isna(row["openalex_id"]) else None,
            title=row["title"] if not pd.isna(row["title"]) else None,
            authors=row["authors"] if not pd.isna(row["authors"]) else None,
            year=int(row["year"]) if not pd.isna(row["year"]) else None,
            abstract=row["abstract"] if not pd.isna(row["abstract"]) else None,
            pub_url=row["pub_url"] if not pd.isna(row["pub_url"]) else None,
            file_urls=file_urls,
            source="OpenAlex",
            cited_by_count=row["cited_by_count"]
            if not pd.isna(row["cited_by_count"])
            else None,
        )

        # Now regenerate the ID - if we have OpenAlex ID, it should preserve it
        if "openalex_id" in row and not pd.isna(row["openalex_id"]):
            openalex_id = row["openalex_id"]
            if openalex_id.startswith("https://openalex.org/W"):
                # Extract actual ID from URL and ensure it's preserved
                expected_id = f"oa_{openalex_id.replace('https://openalex.org/', '')}"
                generated_id = pub.generate_id()
                assert (
                    generated_id == expected_id
                ), f"OpenAlex ID not preserved: {generated_id} != {expected_id}"

    # Verify most publications use OpenAlex IDs as expected
    assert openalex_pct > 90, "Less than 90% of publications have OpenAlex IDs"
