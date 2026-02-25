"""
Tests for the Growth Lab file downloader module.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from curl_cffi.requests import AsyncSession

from backend.etl.scrapers.growthlab import GrowthLabPublication
from backend.etl.utils.gl_file_downloader import DownloadResult, FileDownloader
from backend.storage.local import LocalStorage

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_publication():
    """Create a sample publication for testing."""
    pub = GrowthLabPublication(
        title="Test Publication",
        authors=["John Doe", "Jane Smith"],
        year=2023,
        abstract="This is a test abstract",
        pub_url="https://growthlab.hks.harvard.edu/publications/test",
        file_urls=[
            "https://growthlab.hks.harvard.edu/files/test.pdf",
            "https://growthlab.hks.harvard.edu/files/test.docx",
        ],
        source="GrowthLab",
    )
    # Generate stable IDs
    pub.paper_id = pub.generate_id()
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def storage(tmp_path):
    """Create a temporary storage for testing."""
    return LocalStorage(tmp_path)


@pytest.fixture
def file_downloader(storage):
    """Create a file downloader instance for testing."""
    downloader = FileDownloader(storage=storage, concurrency_limit=2)
    # Set min_file_size to a very small value for testing
    downloader.min_file_size = 10
    return downloader


@pytest.mark.asyncio
async def test_get_file_path(file_downloader, sample_publication):
    """Test generating the correct file path for a publication."""
    # Get path for a PDF
    file_url = "https://example.com/files/document.pdf"
    file_path = file_downloader._get_file_path(sample_publication, file_url)

    # Check path structure
    assert file_path.parts[-3] == "growthlab"
    assert file_path.parts[-2] == sample_publication.paper_id
    assert file_path.parts[-1] == "document.pdf"

    # Test with URL without a filename
    file_url = "https://example.com/files/download?id=123"
    file_path = file_downloader._get_file_path(sample_publication, file_url)

    # Should generate a filename based on URL hash
    assert len(file_path.parts[-1]) > 5  # Should have a reasonable filename length

    # Test with a docx file
    file_url = "https://example.com/files/document.docx"
    file_path = file_downloader._get_file_path(sample_publication, file_url)

    # Check path structure
    assert file_path.parts[-3] == "growthlab"
    assert file_path.parts[-2] == sample_publication.paper_id
    assert file_path.parts[-1] == "document.docx"


@pytest.mark.asyncio
async def test_validate_downloaded_file(file_downloader, tmp_path):
    """Test real file validation with magic bytes for different file types."""
    file_downloader.min_file_size = 10

    # --- Valid PDF ---
    valid_pdf = tmp_path / "valid.pdf"
    valid_pdf.write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)
    result = await file_downloader._validate_downloaded_file(
        valid_pdf, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is True
    assert result["file_exists"] is True
    assert result["size_check"] is True
    assert result["format_check"] is True

    # --- Valid DOCX (PK zip signature) ---
    valid_docx = tmp_path / "valid.docx"
    valid_docx.write_bytes(b"PK\x03\x04" + b"\x00" * 100)
    _docx_ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    result = await file_downloader._validate_downloaded_file(
        valid_docx, expected_content_type=_docx_ct
    )
    assert result["is_valid"] is True
    assert result["format_check"] is True

    # --- Valid DOC (OLE compound document signature) ---
    valid_doc = tmp_path / "valid.doc"
    valid_doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100)
    result = await file_downloader._validate_downloaded_file(
        valid_doc, expected_content_type="application/msword"
    )
    assert result["is_valid"] is True
    assert result["format_check"] is True

    # --- Invalid file (random bytes, claimed PDF) ---
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_bytes(b"This is not a valid PDF file at all!!")
    result = await file_downloader._validate_downloaded_file(
        invalid_pdf, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is False
    assert result["file_exists"] is True
    assert result["format_check"] is False

    # --- Too-small file ---
    small_file = tmp_path / "small.pdf"
    small_file.write_bytes(b"%PDF")  # Only 4 bytes, below min_file_size=10
    result = await file_downloader._validate_downloaded_file(
        small_file, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is False
    assert result["size_check"] is False

    # --- Non-existent file ---
    result = await file_downloader._validate_downloaded_file(
        tmp_path / "nonexistent.pdf"
    )
    assert result["is_valid"] is False
    assert result["file_exists"] is False


@pytest.mark.asyncio
async def test_download_file_impl_status_codes(file_downloader, tmp_path):
    """Test _download_file_impl handles HTTP status codes correctly."""
    file_downloader.min_file_size = 10
    url = "https://example.com/test.pdf"

    # Helper: create a mock curl_cffi response
    def _make_mock_response(status_code, content=b"", headers=None):
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = headers or {"Content-Type": "application/pdf"}

        # aiter_content for streaming
        async def _aiter():
            yield content

        resp.aiter_content = _aiter
        return resp

    mock_session = MagicMock(spec=AsyncSession)

    # --- 200 OK: successful download ---
    dest_200 = tmp_path / "download_200.pdf"
    pdf_content = b"%PDF-1.5\n" + b"\x00" * 100
    mock_session.get = AsyncMock(
        return_value=_make_mock_response(200, content=pdf_content)
    )
    result = await file_downloader._download_file_impl(
        mock_session, url, dest_200, resume=False
    )
    assert result.success is True
    assert result.file_path.exists()
    assert result.file_size > 0

    # --- 206 Partial Content: resume ---
    dest_206 = tmp_path / "download_206.pdf"
    # Write initial partial content
    dest_206.write_bytes(b"%PDF-1.5\n")
    remaining = b"\x00" * 100
    mock_session.get = AsyncMock(
        return_value=_make_mock_response(206, content=remaining)
    )
    result = await file_downloader._download_file_impl(
        mock_session, url, dest_206, resume=True
    )
    assert result.success is True
    # File should contain original + remaining
    assert dest_206.stat().st_size == len(b"%PDF-1.5\n") + len(remaining)

    # --- 416 Range Not Satisfiable: file already complete ---
    dest_416 = tmp_path / "download_416.pdf"
    dest_416.write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)
    mock_session.get = AsyncMock(
        return_value=_make_mock_response(
            416, headers={"Content-Type": "application/pdf"}
        )
    )
    result = await file_downloader._download_file_impl(
        mock_session, url, dest_416, resume=True
    )
    assert result.cached is True
    assert result.success is True  # validation passes since file is valid

    # --- 404 Not Found: error ---
    dest_404 = tmp_path / "download_404.pdf"
    mock_session.get = AsyncMock(return_value=_make_mock_response(404))
    result = await file_downloader._download_file_impl(
        mock_session, url, dest_404, resume=False
    )
    assert result.success is False
    assert "404" in result.error


@pytest.mark.asyncio
async def test_cached_file(file_downloader, tmp_path):
    """Test behavior with cached files."""
    dest_path = tmp_path / "cached.pdf"
    url = "https://example.com/cached.pdf"

    # Create a "cached" PDF file
    with open(dest_path, "wb") as f:
        f.write(b"%PDF-1.5\nCached PDF content\n%%EOF")

    # Mock session but it shouldn't be used
    mock_session = MagicMock(spec=AsyncSession)
    mock_get = AsyncMock()
    mock_session.get = mock_get

    # Mock _validate_downloaded_file to return success
    async def mock_validate(file_path, expected_content_type=None):
        return {
            "is_valid": True,
            "file_exists": True,
            "size_check": True,
            "format_check": True,
            "file_size": 100,
        }

    # Test download with existing file and no overwrite
    with patch.object(file_downloader, "_validate_downloaded_file", mock_validate):
        with patch.object(file_downloader, "_get_session", return_value=mock_session):
            result = await file_downloader.download_file(
                url, dest_path, overwrite=False, resume=False
            )

    # Check the file was reused
    assert result.success
    assert result.cached
    assert result.file_path == dest_path

    # Session.get should not have been called (no download)
    mock_get.assert_not_called()


@pytest.mark.asyncio
async def test_download_publications(file_downloader, sample_publication, tmp_path):
    """Test downloading files for multiple publications."""
    # Create multiple test publications
    pub1 = sample_publication
    pub2 = GrowthLabPublication(
        title="Another Publication",
        authors=["Jane Smith"],
        year=2022,
        file_urls=["https://example.com/file2.pdf", "https://example.com/file2.docx"],
        paper_id="gl_test_123",
    )

    # Publications list
    publications = [pub1, pub2]

    # Mock download_file to avoid actual downloads
    async def mock_download_file(
        url, destination, referer=None, overwrite=False, resume=True
    ):
        return DownloadResult(
            url=url,
            success=True,
            file_path=destination,
            file_size=1000,
            content_type="application/octet-stream",
            cached=False,
        )

    # Test download_publications
    with patch.object(file_downloader, "download_file", mock_download_file):
        with patch.object(file_downloader, "_get_session", return_value=AsyncMock()):
            results = await file_downloader.download_publications(
                publications, overwrite=False, limit=None, progress_bar=False
            )

    # Check results - should have 4 files (2 from each publication)
    assert len(results) == 4
    assert results[0]["publication_id"] == pub1.paper_id
    assert results[2]["publication_id"] == pub2.paper_id
    assert results[0]["success"]
    assert results[2]["success"]


@pytest.mark.asyncio
async def test_download_growthlab_files(storage, sample_publication, tmp_path):
    """Test download_growthlab_files with real production function."""
    # Save sample publication to a real CSV so load_from_csv works
    from backend.etl.scrapers.growthlab import GrowthLabScraper
    from backend.etl.utils.gl_file_downloader import download_growthlab_files

    scraper = GrowthLabScraper()
    csv_path = tmp_path / "test_publications.csv"
    scraper.save_to_csv([sample_publication], csv_path)

    # Mock download_file to avoid real HTTP calls
    async def mock_download_file(
        url, destination, referer=None, overwrite=False, resume=True
    ):
        # Write a fake PDF so validation passes
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"%PDF-1.5\n" + b"\x00" * 100)
        return DownloadResult(
            url=str(url),
            success=True,
            file_path=destination,
            file_size=110,
            content_type="application/pdf",
            cached=False,
        )

    with patch.object(FileDownloader, "download_file", side_effect=mock_download_file):
        results = await download_growthlab_files(
            storage=storage,
            publication_data_path=csv_path,
            overwrite=False,
            limit=None,
            concurrency=1,
        )

    # Sample publication has 2 file URLs
    assert len(results) == 2
    assert results[0]["publication_id"] == sample_publication.paper_id
    assert results[0]["success"] is True
    assert results[1]["success"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_end_to_end(storage, tmp_path):
    """End-to-end integration test with actual files."""
    # Create test publications with known good Growth Lab PDF URLs
    pub = GrowthLabPublication(
        title="Test Publication",
        authors=["Integration Test"],
        year=2024,
        file_urls=[
            "https://growthlab.hks.harvard.edu/wp-content/uploads/2024/09/2024-09-glwp-235-global-trends-innovation-patterns.pdf",
            "https://growthlab.hks.harvard.edu/wp-content/uploads/2026/01/2025-12-glwp-259-new-mexico-economy.pdf",
        ],
        paper_id="gl_test_integration",
    )

    logger.info(f"Created test publication with URLs: {pub.file_urls}")

    # Create the downloader with minimal concurrency
    downloader = FileDownloader(storage=storage, concurrency_limit=1)

    # Set timeout to a more reasonable value
    downloader.config["retry_max_delay"] = 10.0

    try:
        # Download the files
        logger.info("Starting download...")
        results = await downloader.download_publications(
            [pub], overwrite=True, limit=None, progress_bar=False
        )
        logger.info("Download complete.")
        logger.debug(f"Download results: {results}")

        # Check results
        assert len(results) > 0
        logger.info(f"Total results: {len(results)}")

        # Count successful downloads
        successful = [r for r in results if r["success"]]
        logger.info(f"Successful downloads: {len(successful)}/{len(results)}")

        # Log all results for debugging
        for i, r in enumerate(results):
            logger.debug(f"Result {i + 1}:")
            logger.debug(f"  URL: {r['url']}")
            logger.debug(f"  Success: {r['success']}")
            logger.debug(f"  File path: {r.get('file_path')}")
            logger.debug(f"  Error: {r.get('error')}")

        if not successful:
            pytest.skip("All downloads failed, skipping validation")

        # Verify at least one file was downloaded successfully
        assert len(successful) > 0

        # For each successful download, verify the file exists
        for result in successful:
            # Verify the file was downloaded and exists
            assert result["file_path"].exists(), (
                f"File {result['file_path']} does not exist"
            )

            # Verify file size is reasonable
            assert result["file_size"] > 1000, (
                f"File {result['file_path']} is too small: {result['file_size']} bytes"
            )
    except Exception as e:
        logger.error(f"Exception during test: {e}")
        raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_pdf_url_ends_in_pdf(tmp_path):
    """Integration test: download a file whose URL clearly ends in .pdf.

    Verifies the saved file has a .pdf extension and starts with PDF magic bytes.
    """
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)
    downloader.min_file_size = 10

    pub = GrowthLabPublication(
        title="Explicit PDF URL test",
        file_urls=[
            "https://growthlab.hks.harvard.edu/wp-content/uploads/2024/09/2024-09-glwp-235-global-trends-innovation-patterns.pdf",
        ],
        paper_id="gl_test_explicit_pdf",
    )

    results = await downloader.download_publications(
        [pub], overwrite=True, progress_bar=False
    )
    successful = [r for r in results if r["success"]]
    if not successful:
        pytest.skip("Download failed (network issue), skipping")

    result = successful[0]
    file_path = result["file_path"]

    # File should exist
    assert file_path.exists(), f"Downloaded file not found at {file_path}"

    # Extension should be .pdf
    assert file_path.suffix.lower() == ".pdf", (
        f"Expected .pdf extension but got {file_path.suffix}"
    )

    # Should contain PDF magic bytes
    with open(file_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"File does not start with PDF magic bytes: {header!r}"

    # Size should be reasonable
    assert result["file_size"] > 100, (
        f"PDF is suspiciously small: {result['file_size']} bytes"
    )

    logger.info(
        f"Downloaded PDF: {file_path.name}, "
        f"{result['file_size']} bytes, "
        f"Content-Type: {result.get('content_type')}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_pdf_from_non_pdf_url(tmp_path):
    """Integration test: download a PDF from a URL that doesn't end in .pdf.

    Many Growth Lab publication URLs are DOI links or landing pages that
    don't have a .pdf extension. The downloader should detect the Content-Type
    from the server response and rename the file accordingly.
    This tests the core _correct_file_extension fix.
    """
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)
    downloader.min_file_size = 10

    # Use an arXiv PDF URL without .pdf extension — arXiv serves PDFs at
    # /pdf/<id> without an extension, returning Content-Type: application/pdf
    pub = GrowthLabPublication(
        title="Non-PDF URL test",
        file_urls=[
            "https://arxiv.org/pdf/2301.00774",
        ],
        paper_id="gl_test_non_pdf_url",
    )

    results = await downloader.download_publications(
        [pub], overwrite=True, progress_bar=False
    )
    successful = [r for r in results if r["success"]]
    if not successful:
        pytest.skip("Download failed (network issue), skipping")

    result = successful[0]
    file_path = result["file_path"]

    assert file_path.exists()
    assert result["file_size"] > 1000

    # Verify it's actually a PDF by checking magic bytes
    with open(file_path, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", f"Downloaded file is not a PDF (header: {header!r})"

    # The file should have been renamed to .pdf by _correct_file_extension
    # (original URL has no extension, so initial save would be .bin)
    assert file_path.suffix.lower() == ".pdf", (
        f"Expected .pdf extension after Content-Type correction, got {file_path.suffix}"
    )

    logger.info(
        f"Non-.pdf URL download: {file_path.name}, "
        f"extension={file_path.suffix}, "
        f"Content-Type={result.get('content_type')}, "
        f"size={result['file_size']}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_download_multiple_publications_unique_files(tmp_path):
    """Integration test: download files for multiple publications, verify uniqueness.

    Creates 2 publications with different real URLs and verifies:
    - Each publication's files land in the correct paper_id directory
    - Files from different publications don't overwrite each other
    - All downloaded files have non-zero size
    """
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)
    downloader.min_file_size = 10

    pub1 = GrowthLabPublication(
        title="Global Trends in Innovation Patterns",
        file_urls=[
            "https://growthlab.hks.harvard.edu/wp-content/uploads/2024/09/2024-09-glwp-235-global-trends-innovation-patterns.pdf",
        ],
        paper_id="gl_test_pub_one",
    )
    pub2 = GrowthLabPublication(
        title="New Mexico Economy",
        file_urls=[
            "https://growthlab.hks.harvard.edu/wp-content/uploads/2026/01/2025-12-glwp-259-new-mexico-economy.pdf",
        ],
        paper_id="gl_test_pub_two",
    )

    results = await downloader.download_publications(
        [pub1, pub2], overwrite=True, progress_bar=False
    )
    successful = [r for r in results if r["success"]]
    if len(successful) < 2:
        pytest.skip(f"Only {len(successful)}/2 downloads succeeded (network issue)")

    # Verify files are in different directories (by paper_id)
    paths = [r["file_path"] for r in successful]
    parent_dirs = [p.parent.name for p in paths]
    assert "gl_test_pub_one" in parent_dirs, (
        f"pub1's file not in expected directory. Dirs: {parent_dirs}"
    )
    assert "gl_test_pub_two" in parent_dirs, (
        f"pub2's file not in expected directory. Dirs: {parent_dirs}"
    )

    # Verify files are distinct (different content)
    sizes = [r["file_size"] for r in successful]
    # They should have different sizes (different documents)
    assert sizes[0] != sizes[1], (
        f"Both files have identical size ({sizes[0]}), "
        f"possibly same file downloaded twice"
    )

    # Each file should exist and have reasonable size
    for r in successful:
        assert r["file_path"].exists()
        assert r["file_size"] > 1000

    logger.info(
        f"Multi-publication download: {len(successful)} files, "
        f"sizes={sizes}, dirs={parent_dirs}"
    )


def test_correct_file_extension_renames_bin_to_pdf(tmp_path):
    """Unit test: _correct_file_extension renames .bin to .pdf for PDF content type."""
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)

    # Create a fake .bin file with PDF content
    bin_file = tmp_path / "test_doc.bin"
    bin_file.write_bytes(b"%PDF-1.5 fake pdf content")

    # Call the extension corrector
    result = downloader._correct_file_extension(bin_file, "application/pdf")

    assert result.suffix == ".pdf"
    assert result.name == "test_doc.pdf"
    assert result.exists()
    assert not bin_file.exists()  # Original should be gone


def test_correct_file_extension_no_op_when_already_correct(tmp_path):
    """Unit test: _correct_file_extension is a no-op when extension already matches."""
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)

    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.write_bytes(b"%PDF-1.5 fake pdf content")

    result = downloader._correct_file_extension(pdf_file, "application/pdf")

    assert result == pdf_file
    assert result.exists()


def test_correct_file_extension_handles_charset_in_content_type(tmp_path):
    """Unit test: _correct_file_extension strips charset from Content-Type."""
    storage = LocalStorage(tmp_path)
    downloader = FileDownloader(storage=storage, concurrency_limit=1)

    bin_file = tmp_path / "report.bin"
    bin_file.write_bytes(b"%PDF-1.5 fake")

    result = downloader._correct_file_extension(
        bin_file, "application/pdf; charset=utf-8"
    )

    assert result.suffix == ".pdf"
    assert result.exists()
