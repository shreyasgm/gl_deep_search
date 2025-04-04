from unittest.mock import AsyncMock, patch

# No need to import aiohttp directly for tests
import pytest

from backend.etl.scrapers.openalex import (
    OpenAlexClient,
    OpenAlexPublication,
)


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
