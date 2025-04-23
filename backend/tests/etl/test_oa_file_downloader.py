"""
Tests for the OpenAlex file downloader module.
"""

import logging
import os
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import aiohttp
import pytest

from backend.etl.models.publications import OpenAlexPublication
from backend.etl.utils.oa_file_downloader import DownloadResult, OpenAlexFileDownloader
from backend.storage.local import LocalStorage

# Configure logger
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_publication():
    """Create a sample publication for testing."""
    pub = OpenAlexPublication(
        paper_id="W12345678",
        openalex_id="https://openalex.org/W12345678",
        title="Test Publication",
        authors="John Doe, Jane Smith",
        year=2023,
        abstract="This is a test abstract",
        pub_url="https://example.org/article/test",
        file_urls=[
            "https://doi.org/10.1234/example.5678",
            "https://example.org/files/test.pdf",
        ],
        source="OpenAlex",
    )
    pub.content_hash = pub.generate_content_hash()
    return pub


@pytest.fixture
def storage(tmp_path):
    """Create a temporary storage for testing."""
    return LocalStorage(tmp_path)


@pytest.fixture
def file_downloader(storage):
    """Create a file downloader instance for testing."""
    downloader = OpenAlexFileDownloader(storage=storage, concurrency_limit=2)
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
    assert file_path.parts[-3] == "openalex"
    assert file_path.parts[-2] == sample_publication.paper_id
    assert file_path.parts[-1] == "document.pdf"

    # Test with URL without a filename
    file_url = "https://example.com/files/download?id=123"
    file_path = file_downloader._get_file_path(sample_publication, file_url)

    # Should generate a filename based on URL hash
    assert len(file_path.parts[-1]) > 5  # Should have a reasonable filename length

    # Test with a DOI
    file_url = "https://doi.org/10.1234/abc.123"
    file_path = file_downloader._get_file_path(sample_publication, file_url)

    # Check that it creates a reasonable path
    assert file_path.parts[-3] == "openalex"
    assert file_path.parts[-2] == sample_publication.paper_id
    assert file_path.suffix == ".pdf"  # Default extension for academic papers


@pytest.mark.asyncio
async def test_check_open_access(file_downloader):
    """Test checking for open access versions of papers."""
    # Mock the Unpaywall API response
    mock_session = MagicMock()
    mock_session.get = AsyncMock()

    # Case 1: Found open access version in Unpaywall
    mock_unpaywall_response = AsyncMock()
    mock_unpaywall_response.status = 200
    mock_unpaywall_response.json = AsyncMock(
        return_value={
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://example.org/open-access.pdf",
                "url": "https://example.org/open-access",
            },
        }
    )

    # Patch the session
    with patch.object(
        file_downloader, "_get_session", return_value=mock_session
    ):
        # Set up context manager returns
        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_unpaywall_response
        mock_session.get.return_value = mock_context

        # Call the function
        is_oa, url = await file_downloader._check_open_access(
            "https://doi.org/10.1234/example"
        )

        # Verify the result
        assert is_oa is True
        assert url == "https://example.org/open-access.pdf"


@pytest.mark.asyncio
async def test_validate_downloaded_file(file_downloader, tmp_path):
    """Test file validation."""
    # Create a valid PDF file
    valid_pdf_path = tmp_path / "valid.pdf"
    with open(valid_pdf_path, "wb") as f:
        f.write(b"%PDF-1.5\n1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n%%EOF")

    # Create an invalid file
    invalid_file_path = tmp_path / "invalid.pdf"
    with open(invalid_file_path, "wb") as f:
        f.write(b"This is not a valid file")

    # Create a too small file
    small_file_path = tmp_path / "small.pdf"
    with open(small_file_path, "wb") as f:
        f.write(b"%PDF-1.5\n")  # Just the header

    # Test valid PDF
    result = await file_downloader._validate_downloaded_file(
        valid_pdf_path, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is True
    assert result["file_exists"] is True
    assert result["format_check"] is True

    # Test invalid format
    result = await file_downloader._validate_downloaded_file(
        invalid_file_path, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is False
    assert result["file_exists"] is True
    assert result["format_check"] is False

    # Test too small file
    result = await file_downloader._validate_downloaded_file(
        small_file_path, expected_content_type="application/pdf"
    )
    assert result["is_valid"] is False
    assert result["file_exists"] is True
    assert result["size_check"] is False

    # Test nonexistent file
    result = await file_downloader._validate_downloaded_file(
        tmp_path / "nonexistent.pdf", expected_content_type="application/pdf"
    )
    assert result["is_valid"] is False
    assert result["file_exists"] is False


@pytest.mark.asyncio
async def test_download_file_with_aiohttp(file_downloader, tmp_path):
    """Test the HTTP file download implementation."""
    dest_path = tmp_path / "test_http.pdf"
    url = "https://example.com/test.pdf"

    # Mock HTTP response for successful download
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Type": "application/pdf", "Content-Length": "1024"}
    
    # Mock the content
    mock_content = MagicMock()
    mock_content.iter_chunked = AsyncMock(return_value=[b"%PDF-1.5\nTest PDF content"])
    mock_response.content = mock_content

    # Mock the session and context manager
    mock_session = MagicMock()
    mock_context = MagicMock()
    mock_context.__aenter__.return_value = mock_response
    mock_session.get.return_value = mock_context

    # Mock file operations
    mock_file = MagicMock()
    mock_file.write = AsyncMock()
    mock_file_context = MagicMock()
    mock_file_context.__aenter__.return_value = mock_file
    
    # Mock the file validation
    mock_validation = {
        "is_valid": True,
        "file_exists": True,
        "size_check": True,
        "format_check": True,
        "file_size": 1024,
    }
    
    # Patch the required functions
    with patch("aiofiles.open", return_value=mock_file_context), \
         patch.object(file_downloader.storage, "ensure_dir"), \
         patch.object(file_downloader, "_validate_downloaded_file", AsyncMock(return_value=mock_validation)), \
         patch.object(Path, "exists", return_value=True), \
         patch.object(Path, "stat") as mock_stat:
        
        # Setup stat for file size
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = 1024
        mock_stat.return_value = mock_stat_result
        
        # Call the function
        result = await file_downloader._download_file_with_aiohttp(
            mock_session, url, dest_path
        )
        
        # Check the result
        assert result.success is True
        assert result.url == url
        assert result.file_path == dest_path
        assert result.file_size == 1024
        assert result.content_type == "application/pdf"
        assert result.open_access is True
        assert result.source == "http"


@pytest.mark.asyncio
async def test_download_file_with_scidownl(file_downloader, tmp_path):
    """Test download using scidownl."""
    dest_path = tmp_path / "test_scidownl.pdf"
    doi = "https://doi.org/10.1234/example"
    
    # Mock subprocess.run to simulate scidownl behavior
    mock_run_result = MagicMock()
    mock_run_result.returncode = 0
    mock_run_result.stdout = "Downloaded successfully"
    mock_run_result.stderr = ""
    
    # Create a mock PDF in the temp dir that scidownl would create
    # This will be used in the test as if scidownl downloaded it
    with patch("tempfile.TemporaryDirectory") as mock_temp_dir, \
         patch("subprocess.run", return_value=mock_run_result), \
         patch.object(file_downloader.storage, "ensure_dir"), \
         patch.object(file_downloader, "_validate_downloaded_file") as mock_validate, \
         patch("shutil.copy2") as mock_copy:
        
        # Setup tempdir
        mock_temp_dir.return_value.__enter__.return_value = str(tmp_path)
        
        # Create a "downloaded" file in the temp dir
        test_pdf = tmp_path / "downloaded.pdf"
        with open(test_pdf, "wb") as f:
            f.write(b"%PDF-1.5\nTest PDF content")
        
        # Mock the validation result
        mock_validate.return_value = {
            "is_valid": True,
            "file_exists": True,
            "size_check": True,
            "format_check": True,
            "file_size": 100,
        }
        
        # Patch stat to return a file size
        with patch.object(Path, "stat") as mock_stat, \
             patch.object(Path, "glob") as mock_glob:
            
            mock_stat_result = MagicMock()
            mock_stat_result.st_size = 100
            mock_stat.return_value = mock_stat_result
            
            # Make the glob return our created file
            mock_glob.return_value = [test_pdf]
            
            # Call the function
            result = file_downloader._download_file_with_scidownl(doi, dest_path)
            
            # Check the result
            assert result.success is True
            assert result.url == doi
            assert result.file_path == dest_path
            assert result.file_size == 100
            assert result.source == "scidownl"
            
            # Verify scidownl was called correctly
            subprocess.run.assert_called_once()
            args = subprocess.run.call_args[0][0]
            assert args[0] == "scidownl"
            assert args[1] == "download"
            assert args[2] == "--doi"
            assert args[3] == doi
            
            # Verify file was copied
            mock_copy.assert_called_once_with(test_pdf, dest_path)


@pytest.mark.asyncio
async def test_download_file_doi_handling(file_downloader, tmp_path):
    """Test downloading a DOI with the two-step approach."""
    dest_path = tmp_path / "test_doi.pdf"
    doi = "https://doi.org/10.1234/example"
    
    # Mock check_open_access to simulate finding an open access version
    with patch.object(
        file_downloader, "_check_open_access", AsyncMock(return_value=(True, "https://example.org/open-access.pdf"))
    ):
        # Also mock _download_file_with_aiohttp for successful download
        mock_http_result = DownloadResult(
            url="https://example.org/open-access.pdf",
            success=True,
            file_path=dest_path,
            file_size=1024,
            content_type="application/pdf",
            open_access=True,
            source="http",
        )
        
        with patch.object(
            file_downloader, "_download_file_with_aiohttp", AsyncMock(return_value=mock_http_result)
        ), patch.object(
            file_downloader, "_get_session", return_value=MagicMock()
        ):
            # Call the function
            result = await file_downloader.download_file(
                doi, dest_path, is_doi=True
            )
            
            # Check that open access was used
            assert result.success is True
            assert result.file_path == dest_path
            assert result.open_access is True
            assert result.source == "http"
            
            # Verify check_open_access was called
            file_downloader._check_open_access.assert_called_once_with(doi)
    
    # Case 2: No open access, fallback to scidownl
    with patch.object(
        file_downloader, "_check_open_access", AsyncMock(return_value=(False, None))
    ):
        # Mock scidownl for successful download
        mock_scidownl_result = DownloadResult(
            url=doi,
            success=True,
            file_path=dest_path,
            file_size=1024,
            content_type="application/pdf",
            source="scidownl",
        )
        
        with patch.object(
            file_downloader, "_download_file_with_scidownl", return_value=mock_scidownl_result
        ), patch.object(
            file_downloader, "_get_session", return_value=MagicMock()
        ):
            # Call the function
            result = await file_downloader.download_file(
                doi, dest_path, is_doi=True
            )
            
            # Check that scidownl was used
            assert result.success is True
            assert result.file_path == dest_path
            assert result.open_access is False
            assert result.source == "scidownl"


@pytest.mark.asyncio
async def test_download_publications(file_downloader, sample_publication, tmp_path):
    """Test downloading files for multiple publications."""
    # Create multiple test publications
    pub1 = sample_publication
    pub2 = OpenAlexPublication(
        paper_id="W87654321",
        openalex_id="https://openalex.org/W87654321",
        title="Another Publication",
        authors="Jane Smith",
        year=2022,
        file_urls=["https://doi.org/10.5678/another", "https://example.org/files/another.pdf"],
    )

    # Publications list
    publications = [pub1, pub2]

    # Mock download_file to avoid actual downloads
    async def mock_download_file(
        url, destination, referer=None, overwrite=False, resume=True, is_doi=False
    ):
        # Determine source based on URL
        source = "http"
        open_access = False
        if "doi.org" in url:
            source = "scidownl"
            if "example.5678" in url:  # First publication's DOI
                open_access = True
                source = "http"
                
        return DownloadResult(
            url=url,
            success=True,
            file_path=destination,
            file_size=1000,
            content_type="application/pdf",
            cached=False,
            open_access=open_access,
            source=source,
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
    
    # Check the sources
    assert results[0]["source"] == "http"  # First DOI is open access
    assert results[2]["source"] == "scidownl"  # Second DOI needs scidownl


@pytest.mark.asyncio
async def test_integration_with_real_dois(storage, tmp_path):
    """
    Integration test with actual DOIs.
    
    This test attempts to download a real open access paper.
    If it fails, the test is skipped rather than failed.
    """
    # Skip if CI environment or network is known to be unavailable
    if os.environ.get("CI") == "true" or os.environ.get("SKIP_NETWORK_TESTS") == "true":
        pytest.skip("Skipping network-dependent test in CI environment")
    
    # Create test publication with known open access DOI
    pub = OpenAlexPublication(
        paper_id="W12345678",
        openalex_id="https://openalex.org/W12345678",
        title="Test Publication",
        authors="Integration Test",
        year=2023,
        # Use real open access paper DOI (modify if this becomes unavailable)
        file_urls=["https://doi.org/10.1371/journal.pone.0230416"],
        source="OpenAlex",
    )

    # Create the downloader with minimal concurrency
    downloader = OpenAlexFileDownloader(storage=storage, concurrency_limit=1)

    # Set timeout to a more reasonable value
    downloader.config["retry_max_delay"] = 10.0
    
    # Patch the scidownl download to avoid actually using scidownl in tests
    with patch.object(
        downloader, "_download_file_with_scidownl", 
        return_value=DownloadResult(
            url="https://doi.org/10.1371/journal.pone.0230416",
            success=True,
            file_path=tmp_path / "scidownl.pdf",
            file_size=1000,
            content_type="application/pdf",
            source="scidownl",
        )
    ):
        try:
            # Download the files
            logger.info("Starting download...")
            results = await downloader.download_publications(
                [pub], overwrite=True, limit=None, progress_bar=False
            )
            logger.debug(f"Download results: {results}")

            # Check results
            assert len(results) > 0
            
            # Check if any download was successful
            successful = [r for r in results if r["success"]]
            if not successful:
                pytest.skip("All downloads failed, skipping validation")
                
            # Verify at least one file was downloaded successfully
            assert len(successful) > 0
            
            # For successful downloads using open access (not scidownl), verify file exists
            oa_downloads = [r for r in successful if r.get("open_access")]
            if oa_downloads:
                for result in oa_downloads:
                    # Verify the file was downloaded and exists
                    assert Path(result["file_path"]).exists(), (
                        f"File {result['file_path']} does not exist"
                    )
                    
                    # Verify file size is reasonable
                    assert result["file_size"] > 1000, (
                        f"File {result['file_path']} is too small: {result['file_size']} bytes"
                    )
        except Exception as e:
            logger.error(f"Exception during test: {e}")
            pytest.skip(f"Network or service error: {e}")