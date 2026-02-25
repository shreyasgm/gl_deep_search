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
        authors=["John Doe", "Jane Smith"],
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
        authors=["John Doe", "Jane Smith"],
        year=2023,
    )
    pub2 = OpenAlexPublication(
        paper_id="test_no_w_id",
        title="TEST PUBLICATION",  # Different case
        authors=["John Doe", "Jane Smith"],
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
@pytest.mark.skip(reason="Skipping OpenAlex file downloader test - moving to MVP")
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
@pytest.mark.skip(reason="Skipping OpenAlex file downloader test - moving to MVP")
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
        assert publications[0].authors == ["John Doe"]
        assert publications[0].abstract == "Abstract one"
        assert str(publications[0].pub_url) == "https://example.com/pub1"
        assert publications[0].cited_by_count == 10

        assert publications[1].title == "Test Publication 2"
        assert publications[1].year == 2022
        assert publications[1].authors == ["Jane Smith"]
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
@pytest.mark.skip(reason="Skipping OpenAlex file downloader test - moving to MVP")
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


class TestLoadFromCsvSafety:
    """Test that load_from_csv uses safe parsing (not eval) for list fields."""

    def test_load_csv_with_list_strings(self, client, tmp_path):
        """CSV with list-like string values should load correctly."""
        import pandas as pd

        csv_path = tmp_path / "test_safe_load.csv"
        df = pd.DataFrame(
            [
                {
                    "paper_id": "W111",
                    "openalex_id": "https://openalex.org/W111",
                    "title": "Test Paper",
                    "authors": "['Alice', 'Bob']",
                    "year": 2023,
                    "abstract": "An abstract",
                    "pub_url": "https://example.com",
                    "file_urls": "['https://doi.org/10.1234/test']",
                    "source": "OpenAlex",
                    "cited_by_count": 5,
                    "content_hash": "abc123",
                }
            ]
        )
        df.to_csv(csv_path, index=False)
        pubs = client.load_from_csv(csv_path)
        assert len(pubs) == 1
        assert pubs[0].authors == ["Alice", "Bob"]
        assert [str(u) for u in pubs[0].file_urls] == ["https://doi.org/10.1234/test"]


class TestBuildUrl:
    """Tests for OpenAlexClient._build_url URL construction."""

    def test_normal_author_id(self, client):
        """Normal author ID like 'A5034550995'."""
        url = client._build_url()
        # The config default is "A5034550995"; after lstrip('A') we get "5034550995"
        assert "authorships.author.id:A5034550995" in url
        assert f"mailto={client.email}" in url

    def test_author_id_without_leading_a(self):
        """Author ID that doesn't start with 'A'."""
        client = OpenAlexClient()
        client.author_id = "12345"
        url = client._build_url()
        assert "authorships.author.id:A12345" in url

    def test_author_id_double_a_prefix(self):
        """CRITICAL: Author ID 'AAB123' should become 'AB123', not 'B123'.

        The production code uses lstrip('A') which strips ALL leading A's.
        This test demonstrates the bug.
        """
        client = OpenAlexClient()
        client.author_id = "AAB123"
        url = client._build_url()
        # After fix: should contain "AAB123" -> removeprefix('A') -> "AB123"
        assert "authorships.author.id:AAB123" in url

    def test_cursor_parameter(self, client):
        """Cursor should be appended to URL."""
        url = client._build_url(cursor="abc123")
        assert "cursor=abc123" in url
        assert "per-page=200" in url

    def test_no_cursor(self, client):
        """Without cursor, URL should still have per-page."""
        url = client._build_url()
        assert "cursor=" not in url
        assert "per-page=200" in url


class TestProcessResults:
    """Tests for OpenAlexClient.process_results data transformation."""

    def test_complete_api_response(self, client):
        """Process a complete API response dict with all fields."""
        results = [
            {
                "id": "https://openalex.org/W123456789",
                "title": "Economic Complexity",
                "publication_year": 2023,
                "authorships": [
                    {"author": {"display_name": "Ricardo Hausmann"}},
                    {"author": {"display_name": "Bailey Klinger"}},
                ],
                "abstract_inverted_index": {
                    "Economic": [0],
                    "complexity": [1],
                    "matters": [2],
                },
                "primary_location": {"landing_page_url": "https://example.com/paper"},
                "doi": "https://doi.org/10.1234/test",
                "cited_by_count": 100,
            }
        ]
        pubs = client.process_results(results)
        assert len(pubs) == 1
        pub = pubs[0]
        assert pub.paper_id == "W123456789"
        assert pub.title == "Economic Complexity"
        assert pub.year == 2023
        assert pub.authors == ["Ricardo Hausmann", "Bailey Klinger"]
        assert pub.abstract == "Economic complexity matters"
        assert str(pub.pub_url) == "https://example.com/paper"
        assert [str(u) for u in pub.file_urls] == ["https://doi.org/10.1234/test"]
        assert pub.cited_by_count == 100

    def test_missing_authorships(self, client):
        """Process result with missing authorships field."""
        results = [
            {
                "id": "https://openalex.org/W111",
                "title": "No Authors",
                "publication_year": 2020,
                "authorships": None,
                "abstract_inverted_index": {},
                "primary_location": None,
                "doi": None,
            }
        ]
        pubs = client.process_results(results)
        assert len(pubs) == 1
        assert pubs[0].authors == []

    def test_missing_primary_location(self, client):
        """Process result with None primary_location."""
        results = [
            {
                "id": "https://openalex.org/W222",
                "title": "No Location",
                "authorships": [],
                "primary_location": None,
            }
        ]
        pubs = client.process_results(results)
        assert len(pubs) == 1
        assert pubs[0].pub_url is None

    def test_inverted_abstract_reconstruction(self, client):
        """Test abstract reconstruction from inverted index with repeated positions."""
        results = [
            {
                "id": "https://openalex.org/W333",
                "title": "Abstract Test",
                "abstract_inverted_index": {
                    "The": [0, 5],
                    "model": [1],
                    "predicts": [2],
                    "that": [3],
                    "growth": [4],
                    "depends": [6],
                    "on": [7],
                    "complexity": [8],
                },
            }
        ]
        pubs = client.process_results(results)
        expected = "The model predicts that growth The depends on complexity"
        assert pubs[0].abstract == expected

    def test_empty_results_list(self, client):
        """Process empty results list."""
        pubs = client.process_results([])
        assert pubs == []
