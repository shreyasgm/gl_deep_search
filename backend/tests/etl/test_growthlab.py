import logging
from unittest.mock import AsyncMock, patch

import pytest

from backend.etl.scrapers.growthlab import (
    GrowthLabPublication,
    GrowthLabScraper,
    _parse_author_string,
)

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_publication():
    pub = GrowthLabPublication(
        title="Test Publication",
        authors=["John Doe", "Jane Smith"],
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
    # Full names without initials can't be reliably split, so kept as one entry
    assert publication.authors == ["John Doe, Jane Smith"]
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
                authors=["John Doe"],
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


@pytest.mark.asyncio
async def test_extract_publications_with_limit(scraper):
    """Test that extract_publications respects the limit parameter."""
    # Create multiple test publications
    test_publications = []
    for i in range(5):
        pub = GrowthLabPublication(
            title=f"Test Publication {i}",
            authors=["John Doe"],
            year=2023,
            abstract=f"Test abstract {i}",
            pub_url=f"https://growthlab.hks.harvard.edu/publications/test{i}",
            source="GrowthLab",
        )
        pub.paper_id = pub.generate_id()
        pub.content_hash = pub.generate_content_hash()
        test_publications.append(pub)

    # Mock aiohttp session
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response

    # Mock get_max_page_num to return multiple pages
    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch.object(scraper, "get_max_page_num", return_value=10):
            # Mock fetch_page to return publications (3 per page, 1-based pages)
            async def mock_fetch_page(session, page_num):
                # Return 3 publications per page (pages are 1-based)
                start_idx = (page_num - 1) * 3
                return test_publications[start_idx : start_idx + 3]

            with patch.object(scraper, "fetch_page", side_effect=mock_fetch_page):
                # Test with limit of 3
                publications = await scraper.extract_publications(limit=3)

                # Should return exactly 3 publications
                assert len(publications) == 3
                assert publications[0].title == "Test Publication 0"
                assert publications[1].title == "Test Publication 1"
                assert publications[2].title == "Test Publication 2"

                # Test with limit of 1 (should stop after first page)
                publications = await scraper.extract_publications(limit=1)
                assert len(publications) == 1
                assert publications[0].title == "Test Publication 0"


@pytest.mark.asyncio
async def test_update_publications_with_limit(scraper, sample_publication, tmp_path):
    """Test that update_publications passes limit through correctly."""
    from backend.storage.local import LocalStorage

    storage = LocalStorage(tmp_path)

    # Create multiple test publications
    test_publications = [sample_publication]
    for i in range(4):
        pub = GrowthLabPublication(
            title=f"Test Publication {i}",
            authors=["John Doe"],
            year=2023,
            abstract=f"Test abstract {i}",
            pub_url=f"https://growthlab.hks.harvard.edu/publications/test{i}",
            source="GrowthLab",
        )
        pub.paper_id = pub.generate_id()
        pub.content_hash = pub.generate_content_hash()
        test_publications.append(pub)

    # Mock extract_and_enrich_publications to return limited publications
    async def mock_extract_with_limit(limit=None):
        if limit:
            return test_publications[:limit]
        return test_publications

    with patch.object(
        scraper, "extract_and_enrich_publications", side_effect=mock_extract_with_limit
    ):
        # Test with limit of 3
        publications = await scraper.update_publications(storage=storage, limit=3)

        # Should return exactly 3 publications
        assert len(publications) == 3

        # Test without limit
        publications = await scraper.update_publications(storage=storage, limit=None)
        assert len(publications) == 5


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
        authors=["John Doe", "Jane Smith"],
        year=2023,
    )
    pub2 = GrowthLabPublication(
        title="TEST PUBLICATION",  # Different case
        authors=["John Doe", "Jane Smith"],
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


class TestParseAuthorString:
    """Tests for the _parse_author_string pure function."""

    def test_single_author(self):
        assert _parse_author_string("Hausmann, R.") == ["Hausmann, R."]

    def test_two_authors_with_ampersand(self):
        result = _parse_author_string("Hausmann, R. & Klinger, B.")
        assert result == ["Hausmann, R.", "Klinger, B."]

    def test_multiple_authors_comma_ampersand(self):
        result = _parse_author_string("Hausmann, R., Tyson, L.D. & Zahidi, S.")
        assert result == ["Hausmann, R.", "Tyson, L.D.", "Zahidi, S."]

    def test_empty_string(self):
        assert _parse_author_string("") == []

    def test_whitespace_only(self):
        assert _parse_author_string("   ") == []

    def test_none_input(self):
        assert _parse_author_string(None) == []

    def test_single_full_name_no_initials(self):
        # Full names without period-comma pattern stay as one entry
        result = _parse_author_string("John Doe")
        assert result == ["John Doe"]

    def test_parenthetical_initials(self):
        # Closing paren followed by comma should split
        result = _parse_author_string("Hausmann, R. (A.), Klinger, B.")
        # The regex splits on ")" followed by ", " and uppercase letter
        assert len(result) == 2
        assert "Hausmann, R. (A.)" in result[0]


class TestParseEndnoteContent:
    """Tests for parse_endnote_content method."""

    @pytest.mark.asyncio
    async def test_valid_endnote(self):
        scraper = GrowthLabScraper()
        content = (
            "%A Hausmann, Ricardo\n"
            "%A Klinger, Bailey\n"
            "%T Economic Complexity\n"
            "%D 2023\n"
            "%X <p>This is the abstract.</p>\n"
        )
        result = await scraper.parse_endnote_content(content)
        assert result["author"] == ["Ricardo Hausmann", "Bailey Klinger"]
        assert result["title"] == "Economic Complexity"
        assert result["date"] == "2023"
        assert result["abstract"] == "This is the abstract."

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        scraper = GrowthLabScraper()
        content = "%T Only a Title\n"
        result = await scraper.parse_endnote_content(content)
        assert result["title"] == "Only a Title"
        assert "author" not in result
        assert "abstract" not in result

    @pytest.mark.asyncio
    async def test_html_in_abstract(self):
        scraper = GrowthLabScraper()
        content = "%X <p><b>Bold</b> and <strong>strong</strong> text.</p>\n"
        result = await scraper.parse_endnote_content(content)
        # Bold/strong tags should be unwrapped, leaving just text
        assert "Bold" in result["abstract"]
        assert "strong" in result["abstract"]
        assert "<b>" not in result["abstract"]
        assert "<strong>" not in result["abstract"]


class TestParsePublicationNewFormat:
    """Tests for parse_publication with new cp-publication HTML structure."""

    @pytest.mark.asyncio
    async def test_cp_publication_format(self):
        scraper = GrowthLabScraper()
        html = """
        <div class="cp-publication">
            <h2 class="publication-title">
                <a href="/publications/test-new-format">New Format Publication</a>
            </h2>
            <p class="publication-authors">
                Hausmann, R. &amp; Klinger, B.
                <span class="publication-year">, 2024</span>
            </p>
            <div class="publication-excerpt">
                <div>This is the abstract for the new format.</div>
            </div>
            <div class="publication-links">
                <a href="/files/test-new.pdf">Download PDF</a>
                <a href="https://doi.org/10.1234/test">DOI</a>
            </div>
        </div>
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        pub_element = soup.find("div", {"class": "cp-publication"})

        pub = await scraper.parse_publication(
            pub_element,
            "https://growthlab.hks.harvard.edu/publications-home/repository",
        )

        assert pub is not None
        assert pub.title == "New Format Publication"
        assert pub.year == 2024
        assert pub.abstract == "This is the abstract for the new format."
        assert len(pub.file_urls) == 2
        assert pub.authors == ["Hausmann, R.", "Klinger, B."]
        assert pub.paper_id is not None
        assert pub.content_hash is not None


class TestLoadFromCsvSafety:
    """Test that load_from_csv uses safe parsing (not eval) for list fields."""

    def test_load_csv_with_list_strings(self, scraper, tmp_path):
        """CSV with list-like string values should load correctly."""
        import pandas as pd

        csv_path = tmp_path / "test_safe_load.csv"
        df = pd.DataFrame(
            [
                {
                    "paper_id": "gl_test_123",
                    "title": "Test Paper",
                    "authors": "['Alice', 'Bob']",
                    "year": 2023,
                    "abstract": "An abstract",
                    "pub_url": "https://growthlab.hks.harvard.edu/publications/test",
                    "file_urls": "['https://example.com/test.pdf']",
                    "source": "GrowthLab",
                    "content_hash": "abc123",
                }
            ]
        )
        df.to_csv(csv_path, index=False)
        pubs = scraper.load_from_csv(csv_path)
        assert len(pubs) == 1
        assert pubs[0].authors == ["Alice", "Bob"]
        assert [str(u) for u in pubs[0].file_urls] == ["https://example.com/test.pdf"]


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
    assert len(df) >= 10, "Not enough real data records found for testing"

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

    # Load a sample of publications to verify ID generation behavior
    from backend.etl.scrapers.growthlab import GrowthLabPublication

    # Test with a subset of publications to keep test fast
    sample_size = min(20, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)

    # Track publications with URLs to verify they generate URL-based IDs
    publications_with_urls = 0
    url_based_ids_generated = 0

    # For each publication, recreate the model and verify ID generation behavior
    for _, row in sample_df.iterrows():
        # Convert string representation of list to actual list for file_urls
        file_urls = eval(row["file_urls"]) if isinstance(row["file_urls"], str) else []

        # Convert authors from CSV (may be string repr of list or old-style string)
        authors_raw = row["authors"] if not pd.isna(row["authors"]) else None
        if isinstance(authors_raw, str) and authors_raw.startswith("["):
            authors_val = eval(authors_raw)
        elif isinstance(authors_raw, str):
            authors_val = [authors_raw] if authors_raw else []
        else:
            authors_val = []

        # Create Publication object WITHOUT paper_id to test generation logic
        pub = GrowthLabPublication(
            title=row["title"] if not pd.isna(row["title"]) else None,
            authors=authors_val,
            year=int(row["year"]) if not pd.isna(row["year"]) else None,
            abstract=row["abstract"] if not pd.isna(row["abstract"]) else None,
            pub_url=row["pub_url"] if not pd.isna(row["pub_url"]) else None,
            file_urls=file_urls,
            source="GrowthLab",
        )

        # Verify ID generation is deterministic - generate twice and compare
        generated_id_1 = pub.generate_id()
        generated_id_2 = pub.generate_id()
        assert generated_id_1 == generated_id_2, (
            f"ID generation not deterministic: {generated_id_1} != {generated_id_2}"
        )

        # Verify that publications with URLs generate URL-based IDs
        if pub.pub_url:
            publications_with_urls += 1
            assert generated_id_1.startswith("gl_url_"), (
                f"Publication with URL should generate URL-based ID, "
                f"but got: {generated_id_1}"
            )
            url_based_ids_generated += 1

    # Verify that publications with URLs are generating URL-based IDs
    if publications_with_urls > 0:
        logger.info(
            f"Of {publications_with_urls} publications with URLs, "
            f"{url_based_ids_generated} generated URL-based IDs"
        )
        assert url_based_ids_generated == publications_with_urls, (
            "All publications with URLs should generate URL-based IDs"
        )

    # Verify most publications use URL-based IDs as expected
    assert url_pct > 50, "Less than 50% of publications have URL-based IDs"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_growthlab_real_website_scraping(scraper):
    """
    Integration test that verifies the scraper can successfully connect to
    and scrape the real Growth Lab website.

    This test verifies:
    1. The website is accessible
    2. The HTML structure matches what the scraper expects
    3. At least one publication can be parsed from the first page
    """
    from curl_cffi.requests import AsyncSession

    from backend.etl.scrapers.growthlab import BROWSER_IMPERSONATE

    # Create a session with browser impersonation to bypass Cloudflare
    async with AsyncSession(impersonate=BROWSER_IMPERSONATE, timeout=30) as session:
        # Test 1: Verify we can get the max page number
        max_page = await scraper.get_max_page_num(session, scraper.base_url)
        assert max_page >= 0, "Failed to get max page number from website"
        logger.info(f"Found {max_page} pages on Growth Lab website")

        # Test 2: Verify we can fetch and parse the first page
        publications = await scraper.fetch_page(session, 1)
        assert len(publications) > 0, (
            "Failed to extract any publications from first page"
        )
        logger.info(
            f"Successfully extracted {len(publications)} publications from first page"
        )

        # Test 3: Verify at least one publication has required fields
        sample_pub = publications[0]
        assert sample_pub.title is not None and sample_pub.title.strip() != "", (
            "Sample publication missing title"
        )
        assert sample_pub.paper_id is not None and sample_pub.paper_id.startswith(
            "gl_"
        ), f"Sample publication has invalid paper_id: {sample_pub.paper_id}"
        assert sample_pub.pub_url is not None, "Sample publication missing pub_url"

        logger.info(
            f"Sample publication: {sample_pub.title[:50]}... "
            f"(ID: {sample_pub.paper_id})"
        )

        # Test 4: Verify HTML structure by checking for expected elements
        # We'll do a quick fetch to verify the structure
        url = scraper.base_url
        response = await session.get(url)
        assert response.status_code == 200, (
            f"Failed to fetch {url}: status {response.status_code}"
        )
        html = response.text
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Check for expected HTML structure (new or old)
        biblio_entries = soup.find_all("div", {"class": "biblio-entry"})
        cp_publications = soup.find_all("div", {"class": "cp-publication"})

        # Also check for publications in li.wp-block-post
        post_items = soup.find_all(
            "li", {"class": lambda x: x and "wp-block-post" in str(x)}
        )

        total_publications = (
            len(biblio_entries) + len(cp_publications) + len(post_items)
        )
        assert total_publications > 0, (
            "Website HTML structure changed: no publication elements found "
            "(checked biblio-entry, cp-publication, and wp-block-post)"
        )

        # Check for pagination if multiple pages exist (old or new structure)
        # Note: FacetWP pagination may be loaded via JavaScript, so it's OK
        # if not found in HTML
        if max_page > 0:
            old_pagination = soup.find("ul", {"class": "pager"})
            new_pagination = soup.find_all(
                "a",
                href=lambda x: x and ("fwp_paged" in str(x) or "/page/" in str(x)),
            )
            # If using discovery method, pagination links may not be in
            # initial HTML. This is acceptable - the scraper uses binary
            # search discovery in that case
            if old_pagination is None and len(new_pagination) == 0:
                logger.info(
                    "No pagination links found in HTML (likely "
                    "JavaScript-loaded), using discovery method"
                )

        logger.info(
            f"Verified HTML structure: found {len(biblio_entries)} "
            f"biblio-entry, {len(cp_publications)} cp-publication, and "
            f"{len(post_items)} wp-block-post elements"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pagination_returns_unique_publications_across_pages():
    """Integration test: verify that pages 1, 2, 3 return different publications.

    This is the critical pagination test. The Growth Lab site uses FacetWP
    which only paginates via AJAX POST. Page 1 is server-rendered (GET),
    pages 2+ come from a JSON template via POST. This test confirms that
    each page actually returns a distinct set of publications.
    """
    from curl_cffi.requests import AsyncSession

    from backend.etl.scrapers.growthlab import BROWSER_IMPERSONATE

    scraper = GrowthLabScraper()

    async with AsyncSession(impersonate=BROWSER_IMPERSONATE, timeout=30) as session:
        page_results: dict[int, list[GrowthLabPublication]] = {}

        # Fetch 3 pages
        for page_num in [1, 2, 3]:
            pubs = await scraper.fetch_page(session, page_num)
            page_results[page_num] = pubs
            logger.info(
                f"Page {page_num}: {len(pubs)} publications, "
                f"IDs: {[p.paper_id for p in pubs[:3]]}..."
            )

        # 1) Each page must have publications
        for page_num in [1, 2, 3]:
            assert len(page_results[page_num]) > 0, (
                f"Page {page_num} returned 0 publications"
            )

        # 2) Paper IDs from each page must be mutually exclusive
        ids_by_page = {
            p: {pub.paper_id for pub in pubs} for p, pubs in page_results.items()
        }

        overlap_1_2 = ids_by_page[1] & ids_by_page[2]
        overlap_1_3 = ids_by_page[1] & ids_by_page[3]
        overlap_2_3 = ids_by_page[2] & ids_by_page[3]

        assert len(overlap_1_2) == 0, (
            f"Pages 1 and 2 share {len(overlap_1_2)} paper_ids: {overlap_1_2}"
        )
        assert len(overlap_1_3) == 0, (
            f"Pages 1 and 3 share {len(overlap_1_3)} paper_ids: {overlap_1_3}"
        )
        assert len(overlap_2_3) == 0, (
            f"Pages 2 and 3 share {len(overlap_2_3)} paper_ids: {overlap_2_3}"
        )

        # 3) Titles should mostly differ (same title can appear with different
        # URLs if there are multiple editions/versions of a publication)
        titles_by_page = {
            p: {pub.title for pub in pubs} for p, pubs in page_results.items()
        }
        title_overlap_1_2 = titles_by_page[1] & titles_by_page[2]
        if title_overlap_1_2:
            logger.warning(
                f"Pages 1 and 2 share {len(title_overlap_1_2)} titles "
                f"(may be different editions): {title_overlap_1_2}"
            )
        # Allow a small number of shared titles (different editions), but
        # if most titles overlap the pagination is clearly broken
        assert len(title_overlap_1_2) <= 2, (
            f"Pages 1 and 2 share {len(title_overlap_1_2)} titles — "
            f"pagination is likely broken: {title_overlap_1_2}"
        )

        # 4) Combined unique count should equal sum of page counts
        all_ids = set()
        for ids in ids_by_page.values():
            all_ids.update(ids)
        total_pubs = sum(len(pubs) for pubs in page_results.values())
        assert len(all_ids) == total_pubs, (
            f"Expected {total_pubs} unique IDs but got {len(all_ids)} "
            f"({total_pubs - len(all_ids)} duplicates)"
        )

        logger.info(
            f"Pagination test passed: {total_pubs} publications across 3 pages, "
            f"all unique"
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_facetwp_max_page_detection():
    """Integration: FWP_JSON extraction returns a plausible page count."""
    from curl_cffi.requests import AsyncSession

    from backend.etl.scrapers.growthlab import BROWSER_IMPERSONATE

    scraper = GrowthLabScraper()

    async with AsyncSession(impersonate=BROWSER_IMPERSONATE, timeout=30) as session:
        max_page = await scraper.get_max_page_num(session, scraper.base_url)

        # Growth Lab has hundreds of publications, ~10 per page, so >10 pages
        assert max_page >= 10, (
            f"Expected at least 10 pages but got {max_page}. "
            f"FWP_JSON extraction may be broken."
        )
        # Sanity upper bound (they'd need 5000+ publications for 500 pages)
        assert max_page < 500, f"Got {max_page} pages which seems implausibly high"

        logger.info(f"FacetWP reports {max_page} total pages")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_extract_publications_with_real_limit():
    """Integration test: extract_publications(limit=25) returns ~25 unique pubs.

    This tests the full extraction pipeline end-to-end with a small limit,
    verifying deduplication, pagination, and the limit mechanism all work
    together against the real website.
    """
    scraper = GrowthLabScraper()
    publications = await scraper.extract_publications(limit=25)

    # Should have publications
    assert len(publications) > 0, "extract_publications returned empty list"

    # Should respect limit
    assert len(publications) <= 25, (
        f"extract_publications returned {len(publications)} pubs, expected <= 25"
    )

    # All paper_ids should be unique
    ids = [p.paper_id for p in publications]
    assert len(ids) == len(set(ids)), (
        f"Found duplicate paper_ids in results: {[x for x in ids if ids.count(x) > 1]}"
    )

    # Each publication should have basic fields
    for pub in publications:
        assert pub.title, f"Publication {pub.paper_id} has no title"
        assert pub.paper_id, "Publication has no paper_id"
        assert pub.paper_id.startswith("gl_"), (
            f"Unexpected paper_id format: {pub.paper_id}"
        )

    # Count publications with file URLs
    pubs_with_files = [p for p in publications if p.file_urls]
    total_urls = sum(len(p.file_urls) for p in pubs_with_files)
    logger.info(
        f"Extracted {len(publications)} publications: "
        f"{len(pubs_with_files)} have file URLs ({total_urls} total URLs)"
    )


@pytest.mark.asyncio
async def test_deduplication_in_extract_publications():
    """Unit test: verify that extract_publications deduplicates by paper_id."""
    scraper = GrowthLabScraper()

    # Create publications with duplicate paper_ids
    pubs = []
    for i in range(6):
        pub = GrowthLabPublication(
            title=f"Publication {i % 3}",  # 3 unique titles
            authors=["Author"],
            year=2023,
            pub_url=f"https://growthlab.hks.harvard.edu/publications/test{i % 3}",
            source="GrowthLab",
        )
        pub.paper_id = pub.generate_id()
        pub.content_hash = pub.generate_content_hash()
        pubs.append(pub)

    # Mock fetch_page to return all 6 (with 3 duplicate paper_ids)
    mock_session = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_session.get = AsyncMock()
    mock_session.get.return_value = mock_response

    with patch("aiohttp.ClientSession", return_value=mock_session):
        with patch.object(scraper, "get_max_page_num", return_value=1):
            with patch.object(scraper, "fetch_page", return_value=pubs):
                result = await scraper.extract_publications()

    # Should have only 3 unique publications
    assert len(result) == 3
    ids = [p.paper_id for p in result]
    assert len(ids) == len(set(ids))


class TestGrowthLabScraperTrackerIntegration:
    """Tests for GrowthLab scraper integration with publication tracker."""

    @pytest.mark.asyncio
    async def test_scraper_works_without_tracker(
        self, scraper, sample_publication, tmp_path
    ):
        """Test that scraper works correctly when tracker is None."""
        from unittest.mock import patch

        # Mock the extract_and_enrich_publications method
        with patch.object(scraper, "extract_and_enrich_publications") as mock_extract:
            mock_extract.return_value = [sample_publication]

            # Mock storage
            from backend.storage.local import LocalStorage

            storage = LocalStorage(base_path=tmp_path)

            # Call update_publications (should not fail)
            publications = await scraper.update_publications(storage=storage)

            # Should complete without errors
            assert len(publications) > 0
