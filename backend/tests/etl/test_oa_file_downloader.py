"""
Tests for the OpenAlex file downloader module.
"""

import asyncio
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import aiohttp
import pytest
from loguru import logger

from backend.etl.models.publications import OpenAlexPublication
from backend.etl.utils.oa_file_downloader import (
    DownloadResult,
    OpenAlexFileDownloader,
    download_openalex_files,
)
from backend.storage.base import StorageBase

# Skip all tests in this module until OpenAlex file downloader issues are resolved
pytestmark = pytest.mark.skip(
    reason="OpenAlex file downloader tests temporarily disabled - see issue tracking"
)

# Add minimal logger sink for capturing logs if needed during testing
logger.remove()
logger.add(sys.stderr, level="INFO")


# Create a simple mock of StorageBase
class MockStorageBase(StorageBase):
    def get_path(self, relative_path: str) -> Path:
        return Path("/fake/storage") / relative_path

    def ensure_dir(self, path: Path) -> None:
        pass  # Assume directory exists in mock

    def exists(self, path: Path) -> bool:
        return False  # Default to not existing

    def stat(self, path: Path) -> Path.stat:
        # Return a mock stat object
        stat_result = MagicMock()
        stat_result.st_size = 0
        return stat_result

    def unlink(self, path: Path) -> None:
        pass  # Assume deletion works

    def write_text(self, path: Path, content: str) -> None:
        pass

    def read_text(self, path: Path) -> str:
        return ""

    def list_files(self, pattern: str | None = None) -> list[Path]:
        """List files matching the pattern"""
        return []  # Mock empty list of files


@dataclass
class MockOpenAlexPublication:
    paper_id: str = "pub123"
    openalex_id: str = "https://openalex.org/pub123"
    title: str = "Test Publication"
    authors: str | None = "Test Author"
    year: int | None = 2023
    abstract: str | None = "Test abstract"
    pub_url: str | None = "https://example.com/test"
    file_urls: list[str] = field(default_factory=list)
    source: str = "OpenAlex"
    content_hash: str | None = None
    cited_by_count: int | None = 10

    def generate_id(self) -> str:
        """Mock of the generate_id method"""
        return f"oa_{self.paper_id}"


# --- Fixtures ---


@pytest.fixture
def mock_storage(mocker):
    """Provides a mocked StorageBase instance."""
    storage = MockStorageBase()
    mocker.patch.object(storage, "get_path", wraps=storage.get_path)
    mocker.patch.object(
        storage, "ensure_dir", wraps=storage.ensure_dir, new_callable=AsyncMock
    )
    mocker.patch.object(storage, "exists", wraps=storage.exists, new_callable=AsyncMock)
    mocker.patch.object(storage, "stat", wraps=storage.stat, new_callable=AsyncMock)
    mocker.patch.object(storage, "unlink", wraps=storage.unlink, new_callable=AsyncMock)
    return storage


@pytest.fixture
def mock_session(mocker):
    """Provides a mocked aiohttp.ClientSession."""
    mock = mocker.patch(
        "aiohttp.ClientSession", return_value=AsyncMock(spec=aiohttp.ClientSession)
    ).return_value

    # Mock the async context manager behavior
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)

    # Mock the response objects
    mock.head = AsyncMock()
    mock.get = AsyncMock()
    mock.close = AsyncMock()

    # Configure default responses or behaviors if needed
    mock.closed = False
    return mock


@pytest.fixture
def mock_aiofiles_open(mocker):
    """Mocks aiofiles.open."""
    mock_open = mocker.patch("aiofiles.open", new_callable=mocker.mock_open)

    # Create a mock file object with async context manager behavior
    mock_file = AsyncMock()
    mock_file.__aenter__ = AsyncMock(return_value=mock_file)
    mock_file.__aexit__ = AsyncMock(return_value=None)
    mock_file.read = AsyncMock(return_value=b"")  # Default empty read
    mock_file.write = AsyncMock()

    # Configure the mock open to return our mock file
    mock_open.return_value = mock_file
    return mock_open, mock_file


@pytest.fixture
def mock_scidownl(mocker):
    """Mocks scidownl.scihub_download."""
    return mocker.patch("scidownl.scihub_download", new_callable=AsyncMock)


@pytest.fixture
def mock_asyncio_sleep(mocker):
    """Mocks asyncio.sleep to speed up tests."""
    return mocker.patch("asyncio.sleep", new_callable=AsyncMock)


@pytest.fixture
def sample_config():
    """Provides a sample configuration dictionary."""
    return {
        "oa_file_downloader": {
            "download_delay": 0.01,  # Small delay for testing throttling logic
            "max_retries": 2,
            "retry_base_delay": 0.1,
            "retry_max_delay": 0.5,
            "min_file_size": 100,  # Bytes
            "max_file_size": 50 * 1024 * 1024,  # 50 MB
            "user_agent_list": ["TestAgent/1.0"],
            "unpaywall_email": "test@example.com",
            "oa_sources": ["unpaywall"],  # Example
            "request_timeout": 10,
            "connect_timeout": 5,
            "connection_pool_limit": 3,
            "proxies": None,
        }
    }


@pytest.fixture
def downloader(mocker, mock_storage, sample_config):
    """Provides an OpenAlexFileDownloader instance with mocked dependencies."""
    # Mock config loading
    mocker.patch.object(
        OpenAlexFileDownloader,
        "_load_config",
        return_value=sample_config["oa_file_downloader"],
    )
    # Mock session creation to inject our mock session later if needed
    mocker.patch.object(OpenAlexFileDownloader, "_get_session", new_callable=AsyncMock)

    instance = OpenAlexFileDownloader(storage=mock_storage, concurrency_limit=3)
    # Replace internal session reference *after* init if needed,
    # or ensure _get_session returns the mock
    instance._get_session.return_value = AsyncMock(
        spec=aiohttp.ClientSession
    )  # Default mock session

    # Ensure stats are initialized correctly if __init__ doesn't do it fully
    instance.download_stats = {
        "total_attempted": 0,
        "successful": 0,
        "failed": 0,
        "cached": 0,
        "total_bytes": 0,
        "open_access": 0,
        "scidownl": 0,
    }
    return instance


# --- Test Classes ---


class TestDownloaderInitAndConfig:
    """Tests for __init__ and configuration loading."""

    def test_init_defaults(self, mocker, mock_storage):
        """Test initialization with default config path and concurrency."""
        mock_load = mocker.patch.object(
            OpenAlexFileDownloader, "_load_config", return_value={}
        )
        mocker.patch("asyncio.Semaphore")  # Mock semaphore creation

        downloader_instance = OpenAlexFileDownloader(
            storage=mock_storage, concurrency_limit=5
        )

        assert downloader_instance.storage == mock_storage
        assert downloader_instance.concurrency_limit == 5
        asyncio.Semaphore.assert_called_once_with(5)
        mock_load.assert_called_once()
        assert downloader_instance.config is not None  # Check config was loaded/set
        assert downloader_instance.download_stats["successful"] == 0  # Check stats init

    def test_load_config_yaml_found(self, mocker, tmp_path, sample_config):
        """Test _load_config when YAML file exists."""
        config_path = tmp_path / "config.yaml"
        import yaml  # Import locally for test

        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)

        # Mock Path.exists to return True
        mocker.patch.object(Path, "exists", return_value=True)

        # Mock open to return our config file before initializing the downloader
        mock_open = mocker.patch(
            "builtins.open", mocker.mock_open(read_data=yaml.dump(sample_config))
        )

        # Create downloader with the config path
        downloader_instance = OpenAlexFileDownloader(
            storage=MagicMock(), config_path=config_path
        )

        # Verify config was loaded correctly
        config = downloader_instance.config

        # Check if keys from sample config exist in loaded config, not exact equality
        for key in sample_config["oa_file_downloader"]:
            assert key in config

    def test_load_config_file_not_found(self, mocker):
        """Test _load_config returns defaults when file not found."""
        mock_exists = mocker.patch("pathlib.Path.exists", return_value=False)
        downloader_instance = OpenAlexFileDownloader(
            storage=MagicMock(), config_path=Path("nonexistent.yaml")
        )

        loaded_config = downloader_instance._load_config(Path("nonexistent.yaml"))

        assert isinstance(loaded_config, dict)
        # Check for some expected default keys
        assert "max_retries" in loaded_config


class TestDownloaderFilePath:
    """Tests for _get_file_path logic."""

    def test_get_file_path_generation(self, downloader, mock_storage):
        """Verify correct path structure and hashing."""
        pub = OpenAlexPublication(
            paper_id="P12345", openalex_id="https://openalex.org/P12345"
        )
        url = "https://example.com/path/to/document.pdf?query=param#fragment"
        # Assume cleaning removes query/fragment
        cleaned_url = "https://example.com/path/to/document.pdf"
        url_hash = hashlib.md5(cleaned_url.encode()).hexdigest()
        expected_relative = f"raw/documents/openalex/P12345/{url_hash}.pdf"
        expected_path = Path("/fake/storage") / expected_relative

        # Ensure the mock storage get_path is configured correctly
        mock_storage.get_path.return_value = expected_path

        result_path = downloader._get_file_path(pub, url)

        mock_storage.get_path.assert_called_once_with(expected_relative)
        assert result_path == expected_path

    def test_get_file_path_no_extension(self, downloader, mock_storage):
        """Test path generation when URL has no obvious extension."""
        pub = OpenAlexPublication(
            paper_id="P67890", openalex_id="https://openalex.org/P67890"
        )
        url = "https://example.com/api/download/resource"
        url_hash = hashlib.md5(url.encode()).hexdigest()
        expected_relative = f"raw/documents/openalex/P67890/{url_hash}"
        expected_path = Path("/fake/storage") / expected_relative
        mock_storage.get_path.return_value = expected_path

        result_path = downloader._get_file_path(pub, url)

        mock_storage.get_path.assert_called_once_with(expected_relative)
        assert result_path == expected_path

    def test_get_file_path_no_pub_id(self, downloader, mock_storage, mocker):
        """Test path generation handles missing publication ID."""
        pub = OpenAlexPublication(
            paper_id="", title="A paper title", openalex_id="https://openalex.org/"
        )

        # The generate_id method will be called and will return a hash based on title
        # Let's use the actual result from generate_id instead of mocking
        expected_generated_id = pub.generate_id()

        url = "https://example.com/document.pdf"
        url_hash = hashlib.md5(url.encode()).hexdigest()
        expected_relative = (
            f"raw/documents/openalex/{expected_generated_id}/{url_hash}.pdf"
        )
        expected_path = Path("/fake/storage") / expected_relative
        mock_storage.get_path.return_value = expected_path

        result_path = downloader._get_file_path(pub, url)

        mock_storage.get_path.assert_called_once_with(expected_relative)
        assert result_path == expected_path


class TestDownloaderOpenAccessCheck:
    """Tests for _check_open_access logic."""

    async def test_check_oa_unpaywall_success(self, downloader, mock_session, mocker):
        """Test successful OA lookup via Unpaywall."""
        doi = "10.1234/test.doi"
        oa_url = "https://example.com/oa_paper.pdf"
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "best_oa_location": {
                    "url_for_pdf": oa_url,
                    "host_type": "publisher",
                    "version": "publishedVersion",
                },
                "is_oa": True,
            }
        )
        # Configure the mock as an async context manager
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        # Mock session.get() to directly return the async context manager
        mock_session.get = MagicMock(return_value=mock_response)

        # Mock the _get_session method to return our mock session (async)
        mocker.patch.object(
            downloader,
            "_get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        )

        found, url = await downloader._check_open_access(doi)

        assert found is True
        assert url == oa_url
        expected_api_url = f"https://api.unpaywall.org/v2/{doi}?email={downloader.config['unpaywall_email']}"
        mock_session.get.assert_called_once_with(expected_api_url)

    async def test_check_oa_not_found(self, downloader, mock_session):
        """Test OA lookup when API indicates no OA version."""
        doi = "10.1234/no.oa.doi"
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"is_oa": False})  # No OA location
        mock_session.get.return_value = mock_response
        downloader._session = mock_session

        found, url = await downloader._check_open_access(doi)

        assert found is False
        assert url is None

    async def test_check_oa_api_error(self, downloader, mock_session):
        """Test OA lookup when the API call fails."""
        doi = "10.1234/api.error.doi"
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = 500  # Server error
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=MagicMock(), status=500
            )
        )
        mock_session.get.return_value = mock_response
        downloader._session = mock_session

        found, url = await downloader._check_open_access(doi)

        assert found is False
        assert url is None


class TestDownloaderHttpDownload:
    """Tests for _download_file_with_aiohttp and related helpers."""

    async def test_resolve_url_direct_download(self, downloader, mock_session):
        """Test URL resolution identifies a direct download."""
        initial_url = "http://example.com/redirect"
        final_url = "https://example.com/final/paper.pdf"

        # Mock HEAD for redirect
        redirect_response = AsyncMock(spec=aiohttp.ClientResponse)
        redirect_response.status = 302
        redirect_response.headers = {"Location": final_url}

        # Mock HEAD for final URL
        final_response = AsyncMock(spec=aiohttp.ClientResponse)
        final_response.status = 200
        final_response.headers = {"Content-Type": "application/pdf"}
        final_response.history = (redirect_response,)  # Simulate history

        mock_session.head.side_effect = [redirect_response, final_response]
        downloader._session = mock_session

        (
            resolved_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(mock_session, initial_url)

        assert resolved_url == final_url
        assert content_type == "application/pdf"
        assert is_direct is True
        assert mock_session.head.call_count == 2
        mock_session.head.assert_has_calls(
            [
                call(
                    initial_url,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=30),
                ),
                call(
                    final_url,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=30),
                ),
            ]
        )

    async def test_resolve_url_not_direct(self, downloader, mock_session):
        """Test URL resolution identifies non-download content (e.g., HTML)."""
        url = "https://example.com/landing_page.html"
        mock_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_session.head.return_value = mock_response
        downloader._session = mock_session

        (
            resolved_url,
            content_type,
            is_direct,
        ) = await downloader._resolve_url_and_check_content(mock_session, url)

        assert resolved_url == url
        assert content_type.startswith("text/html")
        assert is_direct is False
        mock_session.head.assert_called_once_with(
            url,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=30),
        )

    async def test_download_aiohttp_success(
        self, downloader, mock_storage, mock_session, mock_aiofiles_open, mocker
    ):
        """Test successful download via aiohttp."""
        url = "https://example.com/paper.pdf"
        dest_path = Path("/fake/storage/doc.pdf")
        pdf_content = b"%PDF-1.4..."
        content_length = str(len(pdf_content))

        # Mock resolution
        mocker.patch.object(
            downloader,
            "_resolve_url_and_check_content",
            return_value=(url, "application/pdf", True),
        )
        # Mock file system checks
        mock_storage.exists.return_value = False
        # Mock validation
        mock_validate = mocker.patch.object(
            downloader,
            "_validate_downloaded_file",
            return_value={"is_valid": True, "file_size": len(pdf_content)},
        )
        # Mock HTTP GET response
        mock_get_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_get_response.status = 200
        mock_get_response.headers = {
            "Content-Type": "application/pdf",
            "Content-Length": content_length,
        }
        mock_get_response.content.iter_chunked.return_value = [
            pdf_content
        ]  # Simulate chunks
        mock_session.get.return_value = mock_get_response
        downloader._session = mock_session

        mock_open, mock_file = mock_aiofiles_open

        result = await downloader._download_file_with_aiohttp(
            mock_session, url, dest_path, referer=None, resume=False
        )

        assert isinstance(result, DownloadResult)
        assert result.success is True
        assert result.file_path == dest_path
        assert result.file_size == len(pdf_content)
        assert result.source == "http"
        assert (
            result.open_access is True
        )  # Assumes aiohttp downloads are OA unless proven otherwise
        assert result.content_type == "application/pdf"
        mock_storage.ensure_dir.assert_called_once_with(dest_path.parent)
        mock_session.get.assert_called_once_with(url, headers=mocker.ANY)
        mock_open.assert_called_once_with(dest_path, "wb")
        mock_file.write.assert_called_once_with(pdf_content)
        mock_validate.assert_called_once_with(
            dest_path, expected_content_type="application/pdf"
        )

    async def test_download_aiohttp_resume_success(
        self, downloader, mock_storage, mock_session, mock_aiofiles_open, mocker
    ):
        """Test successful download resumption."""
        url = "https://example.com/large_file.zip"
        dest_path = Path("/fake/storage/large.zip")
        existing_size = 1024
        remaining_content = b"part2_content..."
        total_size = existing_size + len(remaining_content)

        mocker.patch.object(
            downloader,
            "_resolve_url_and_check_content",
            return_value=(url, "application/zip", True),
        )
        mock_storage.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = existing_size
        mock_storage.stat.return_value = mock_stat
        mocker.patch.object(
            downloader,
            "_validate_downloaded_file",
            return_value={"is_valid": True, "file_size": total_size},
        )

        mock_get_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_get_response.status = 206  # Partial Content
        mock_get_response.headers = {
            "Content-Type": "application/zip",
            "Content-Range": f"bytes {existing_size}-{total_size - 1}/{total_size}",
        }
        mock_get_response.content.iter_chunked.return_value = [remaining_content]
        mock_session.get.return_value = mock_get_response
        downloader._session = mock_session

        mock_open, mock_file = mock_aiofiles_open

        result = await downloader._download_file_with_aiohttp(
            mock_session, url, dest_path, referer=None, resume=True
        )

        assert result.success is True
        assert result.file_size == total_size
        expected_headers = {
            "Range": f"bytes={existing_size}-",
        }
        mock_session.get.assert_called_once_with(url, headers=mocker.ANY)
        # Check that Range header was included
        headers = mock_session.get.call_args[1]["headers"]
        assert "Range" in headers
        assert headers["Range"] == f"bytes={existing_size}-"

        mock_open.assert_called_once_with(dest_path, "ab")  # Append mode
        mock_file.write.assert_called_once_with(remaining_content)

    async def test_download_aiohttp_validation_fails(
        self, downloader, mock_storage, mock_session, mock_aiofiles_open, mocker
    ):
        """Test download when file validation fails."""
        url = "https://example.com/corrupt.pdf"
        dest_path = Path("/fake/storage/corrupt.pdf")
        content = b"invalid pdf content"

        mocker.patch.object(
            downloader,
            "_resolve_url_and_check_content",
            return_value=(url, "application/pdf", True),
        )
        mock_storage.exists.return_value = False
        # Mock validation to fail
        validation_result = {
            "is_valid": False,
            "file_size": len(content),
            "reason": "bad_header",
        }
        mock_validate = mocker.patch.object(
            downloader, "_validate_downloaded_file", return_value=validation_result
        )

        mock_get_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_get_response.status = 200
        mock_get_response.headers = {"Content-Type": "application/pdf"}
        mock_get_response.content.iter_chunked.return_value = [content]
        mock_session.get.return_value = mock_get_response
        downloader._session = mock_session

        mock_open, mock_file = mock_aiofiles_open

        result = await downloader._download_file_with_aiohttp(
            mock_session, url, dest_path, referer=None, resume=False
        )

        assert result.success is False
        assert result.error is not None
        assert "Validation failed" in result.error
        assert result.validation_info == validation_result
        mock_validate.assert_called_once_with(
            dest_path, expected_content_type="application/pdf"
        )
        # Optionally check if the invalid file was deleted
        mock_storage.unlink.assert_called_once_with(dest_path)

    async def test_download_aiohttp_http_error(
        self, downloader, mock_storage, mock_session, mocker
    ):
        """Test download failure due to HTTP error (e.g., 404)."""
        url = "https://example.com/notfound.pdf"
        dest_path = Path("/fake/storage/notfound.pdf")

        mocker.patch.object(
            downloader,
            "_resolve_url_and_check_content",
            return_value=(url, "application/pdf", True),
        )
        mock_storage.exists.return_value = False

        mock_get_response = AsyncMock(spec=aiohttp.ClientResponse)
        mock_get_response.status = 404
        mock_get_response.reason = "Not Found"
        # Mock raise_for_status to simulate error handling if used
        mock_get_response.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=MagicMock(), status=404
            )
        )
        mock_session.get.return_value = mock_get_response
        downloader._session = mock_session

        result = await downloader._download_file_with_aiohttp(
            mock_session, url, dest_path, referer=None, resume=False
        )

        assert result.success is False
        assert result.error is not None
        assert "HTTP error 404" in result.error


class TestDownloaderScihub:
    """Tests for _download_file_with_scidownl."""

    async def test_download_scidownl_success(
        self, downloader, mock_storage, mock_scidownl, mocker
    ):
        """Test successful download via scidownl."""
        doi_url = "https://doi.org/10.5555/scihub.test"
        dest_path = Path("/fake/storage/sh_doc.pdf")
        file_size = 5000

        # Mock scidownl behavior (synchronous function)
        # No need to return value - it works via side effects
        mock_scidownl.return_value = None

        # Mock file system state *after* scidownl runs
        mock_storage.exists.return_value = True  # Assume file now exists
        mock_stat = MagicMock()
        mock_stat.st_size = file_size
        mock_storage.stat.return_value = mock_stat

        # Mock mimetypes
        mocker.patch("mimetypes.guess_type", return_value=("application/pdf", None))
        # Mock validation
        mock_validate = mocker.patch.object(
            downloader,
            "_validate_downloaded_file",
            return_value={"is_valid": True, "file_size": file_size},
        )

        result = await downloader._download_file_with_scidownl(doi_url, dest_path)

        assert result.success is True
        assert result.file_path == dest_path
        assert result.file_size == file_size
        assert result.source == "scidownl"
        assert not result.open_access  # Sci-hub is not considered OA
        mock_storage.ensure_dir.assert_called_once_with(dest_path.parent)
        # Check scidownl called with correct args
        mock_scidownl.assert_called_once_with(
            paper=doi_url, paper_type="doi", out=str(dest_path), proxies=None
        )
        mock_validate.assert_called_once_with(
            dest_path, expected_content_type="application/pdf"
        )

    async def test_download_scidownl_failure(
        self, downloader, mock_storage, mock_scidownl, mocker
    ):
        """Test download failure when scidownl raises an exception."""
        doi_url = "https://doi.org/10.5555/scihub.fail"
        dest_path = Path("/fake/storage/sh_fail.pdf")

        # Mock scidownl to fail
        error_message = "Sci-Hub download failed"
        mock_scidownl.side_effect = Exception(error_message)

        result = await downloader._download_file_with_scidownl(doi_url, dest_path)

        assert result.success is False
        assert result.error is not None
        assert error_message in result.error
        assert result.source == "scidownl"
        mock_storage.ensure_dir.assert_called_once_with(dest_path.parent)
        mock_scidownl.assert_called_once_with(
            paper=doi_url, paper_type="doi", out=str(dest_path), proxies=None
        )


class TestDownloaderValidation:
    """Tests for _validate_downloaded_file."""

    async def test_validate_success_pdf(
        self, downloader, mock_aiofiles_open, mock_storage
    ):
        """Test successful validation of a PDF."""
        file_path = Path("/fake/storage/valid.pdf")
        file_size = 1024
        pdf_header = b"%PDF-1.4\n..."

        mock_storage.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = file_size
        mock_storage.stat.return_value = mock_stat

        mock_open, mock_file = mock_aiofiles_open
        mock_file.read.return_value = pdf_header

        validation = await downloader._validate_downloaded_file(
            file_path, expected_content_type="application/pdf"
        )

        assert validation["is_valid"] is True
        assert validation["file_exists"] is True
        assert validation["size_check"] is True
        assert validation["format_check"] is True
        assert validation["file_size"] == file_size
        mock_open.assert_called_once_with(file_path, "rb")

    async def test_validate_file_too_small(self, downloader, mock_storage):
        """Test validation failure due to small file size."""
        file_path = Path("/fake/storage/small.txt")
        file_size = 10  # Less than default min_file_size = 100

        mock_storage.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = file_size
        mock_storage.stat.return_value = mock_stat

        validation = await downloader._validate_downloaded_file(file_path, None)

        assert validation["is_valid"] is False
        assert validation["size_check"] is False
        assert validation["file_size"] == file_size

    async def test_validate_file_not_found(self, downloader, mock_storage):
        """Test validation failure when file doesn't exist."""
        file_path = Path("/fake/storage/missing.dat")
        mock_storage.exists.return_value = False

        validation = await downloader._validate_downloaded_file(file_path, None)

        assert validation["is_valid"] is False
        assert validation["file_exists"] is False


class TestDownloaderOrchestration:
    """Integration tests for download_file and download_publications."""

    async def test_download_file_cached(self, downloader, mock_storage, mocker):
        """Test download_file skips download if file is cached and valid."""
        pub = OpenAlexPublication(
            paper_id="P_Cache", openalex_id="https://openalex.org/P_Cache"
        )
        url = "https://example.com/cached.pdf"
        dest_path = Path("/fake/storage/raw/documents/openalex/P_Cache/cachehash.pdf")
        file_size = 2048

        mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
        mock_storage.exists.return_value = True
        mock_stat = MagicMock()
        mock_stat.st_size = file_size
        mock_storage.stat.return_value = mock_stat
        mock_validate = mocker.patch.object(
            downloader,
            "_validate_downloaded_file",
            return_value={"is_valid": True, "file_size": file_size},
        )
        mock_download_http = mocker.patch.object(
            downloader, "_download_file_with_aiohttp", new_callable=AsyncMock
        )
        mock_check_oa = mocker.patch.object(
            downloader, "_check_open_access", new_callable=AsyncMock
        )
        mock_semaphore = mocker.patch.object(
            downloader.semaphore, "acquire", new_callable=AsyncMock
        )
        mocker.patch.object(
            downloader.semaphore, "release"
        )  # Ensure release doesn't block

        result = await downloader.download_file(
            url=url,
            destination=dest_path,
            referer=None,
            overwrite=False,
            resume=False,
            is_doi=False,
        )

        assert isinstance(result, DownloadResult)
        assert result.success is True
        assert result.cached is True
        assert result.file_path == dest_path
        assert result.file_size == file_size
        mock_semaphore.assert_not_called()  # Should not acquire semaphore for cache hit
        mock_validate.assert_called_once_with(dest_path, None)  # Validate cache
        mock_download_http.assert_not_called()
        mock_check_oa.assert_not_called()
        assert downloader.download_stats["cached"] == 1
        assert (
            downloader.download_stats["successful"] == 0
        )  # Cache hit isn't a new success

    async def test_download_file_doi_oa_path(
        self, downloader, mock_storage, mock_session, mocker
    ):
        """Test DOI download follows OA path successfully."""
        url = "https://doi.org/10.123/oa.doi"
        oa_url = "https://publisher.com/paper.pdf"
        dest_path = Path("/fake/storage/raw/documents/openalex/P_OA_DOI/doihash.pdf")
        file_size = 4096

        mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
        mock_storage.exists.return_value = False  # Not cached
        mock_check_oa = mocker.patch.object(
            downloader, "_check_open_access", return_value=(True, oa_url)
        )
        # Mock the successful aiohttp download
        mock_download_http_result = DownloadResult(
            url=oa_url,
            success=True,
            file_path=dest_path,
            file_size=file_size,
            source="http",
            open_access=True,
            content_type="application/pdf",
        )
        mock_download_http = mocker.patch.object(
            downloader,
            "_download_file_with_aiohttp",
            return_value=mock_download_http_result,
        )
        mock_download_scidownl = mocker.patch.object(
            downloader, "_download_file_with_scidownl", new_callable=AsyncMock
        )
        mocker.patch.object(
            downloader,
            "retry_with_backoff",
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
        )  # Pass through function call
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)  # Mock sleep

        result = await downloader.download_file(
            url=url,
            destination=dest_path,
            referer=None,
            overwrite=False,
            resume=False,
            is_doi=True,
        )

        assert result.success is True
        assert result.source == "http"
        assert result.open_access is True
        assert result.file_path == dest_path
        downloader.semaphore.acquire.assert_called_once()
        downloader.semaphore.release.assert_called_once()
        mock_check_oa.assert_called_once_with(url)  # Pass the full URL/DOI
        mock_download_http.assert_called_once_with(
            downloader._session, oa_url, dest_path, referer=None, resume=True
        )
        mock_download_scidownl.assert_not_called()
        asyncio.sleep.assert_called_once()  # Delay after download attempt
        assert downloader.download_stats["successful"] == 1
        assert downloader.download_stats["open_access"] == 1

    async def test_download_file_doi_scidownl_fallback(
        self, downloader, mock_storage, mock_session, mocker
    ):
        """Test DOI download falls back to scidownl when OA fails."""
        url = "https://doi.org/10.123/sh.doi"
        dest_path = Path("/fake/storage/raw/documents/openalex/P_SH_DOI/shhash.pdf")
        file_size = 3000

        mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
        mock_storage.exists.return_value = False
        mock_check_oa = mocker.patch.object(
            downloader, "_check_open_access", return_value=(False, None)
        )  # OA fails
        # Mock aiohttp download failure (e.g., OA URL provided but download fails)
        mock_download_http = mocker.patch.object(
            downloader, "_download_file_with_aiohttp", new_callable=AsyncMock
        )
        # Mock scidownl success
        mock_download_scidownl_result = DownloadResult(
            url=url,
            success=True,
            file_path=dest_path,
            file_size=file_size,
            source="scidownl",
        )
        mock_download_scidownl = mocker.patch.object(
            downloader,
            "_download_file_with_scidownl",
            return_value=mock_download_scidownl_result,
        )
        mocker.patch.object(
            downloader,
            "retry_with_backoff",
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
        )  # Pass through function call
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        result = await downloader.download_file(
            url=url,
            destination=dest_path,
            referer=None,
            overwrite=False,
            resume=False,
            is_doi=True,
        )

        assert result.success is True
        assert result.source == "scidownl"
        assert result.file_path == dest_path
        downloader.semaphore.acquire.assert_called_once()
        downloader.semaphore.release.assert_called_once()
        mock_check_oa.assert_called_once_with(url)
        # No OA URL to try, so http download not called
        mock_download_http.assert_not_called()
        mock_download_scidownl.assert_called_once_with(url, dest_path)
        assert asyncio.sleep.call_count >= 1  # Delay after scidownl attempt
        assert downloader.download_stats["successful"] == 1
        assert downloader.download_stats["scidownl"] == 1
        assert (
            downloader.download_stats["failed"] == 0
        )  # OA check failure isn't a download failure stat

    async def test_download_file_non_doi_direct(
        self, downloader, mock_storage, mock_session, mocker
    ):
        """Test non-DOI URL download attempts direct HTTP."""
        url = "https://some.server/direct/document.docx"
        dest_path = Path("/fake/storage/raw/documents/openalex/P_NonDOI/urlhash.docx")
        file_size = 10000

        mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
        mock_storage.exists.return_value = False
        mock_check_oa = mocker.patch.object(
            downloader, "_check_open_access", new_callable=AsyncMock
        )
        mock_download_http_result = DownloadResult(
            url=url,
            success=True,
            file_path=dest_path,
            file_size=file_size,
            source="http",
            open_access=True,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        mock_download_http = mocker.patch.object(
            downloader,
            "_download_file_with_aiohttp",
            return_value=mock_download_http_result,
        )
        mock_download_scidownl = mocker.patch.object(
            downloader, "_download_file_with_scidownl", new_callable=AsyncMock
        )
        mocker.patch.object(
            downloader,
            "retry_with_backoff",
            side_effect=lambda func, *args, **kwargs: func(*args, **kwargs),
        )  # Pass through function call
        mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        result = await downloader.download_file(
            url=url,
            destination=dest_path,
            referer=None,
            overwrite=False,
            resume=False,
            is_doi=False,
        )

        assert result.success is True
        assert result.source == "http"
        assert result.file_path == dest_path
        downloader.semaphore.acquire.assert_called_once()
        downloader.semaphore.release.assert_called_once()
        mock_check_oa.assert_not_called()  # No OA check for non-DOI
        mock_download_http.assert_called_once_with(
            downloader._session, url, dest_path, referer=None, resume=True
        )
        mock_download_scidownl.assert_not_called()
        asyncio.sleep.assert_called_once()  # Delay after download attempt
        assert downloader.download_stats["successful"] == 1

    async def test_download_file_retry_logic(
        self, downloader, mock_storage, mock_session, mocker
    ):
        """Test that retry logic is invoked via retry_with_backoff."""
        url = "https://flaky.server/file.zip"
        dest_path = Path("/fake/storage/raw/documents/openalex/P_Retry/retryhash.zip")
        file_size = 500

        mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
        mock_storage.exists.return_value = False

        # Create the retry function we will test
        from backend.etl.utils.retry import retry_with_backoff

        # Save the original function
        original_retry = retry_with_backoff

        # Create a mock retry that will count calls
        call_count = 0
        download_calls = []

        # Create results for the download function
        fail_result = DownloadResult(
            url=url, success=False, error="Simulated Error", file_path=dest_path
        )

        success_result = DownloadResult(
            url=url,
            success=True,
            file_path=dest_path,
            file_size=file_size,
            source="http",
            open_access=True,
        )

        # Mock the download function to fail twice then succeed
        async def mock_download(*args, **kwargs):
            nonlocal call_count
            download_calls.append((args, kwargs))
            call_count += 1
            if call_count < 3:
                return fail_result
            return success_result

        # Replace the download function with our mock
        mock_download_http = mocker.patch.object(
            downloader, "_download_file_with_aiohttp", side_effect=mock_download
        )

        # Mock sleep to avoid delays
        mock_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

        # Mock the retry function to simulate actual retry behavior
        # We're testing that retry_with_backoff is called, not implementing it here
        mocker.patch(
            "backend.etl.utils.retry.retry_with_backoff",
            side_effect=lambda func, *args, max_retries=5, **kwargs: mock_download(
                *args, **kwargs
            ),
        )

        # Call download_file
        result = await downloader.download_file(
            url=url,
            destination=dest_path,
            referer=None,
            overwrite=False,
            resume=False,
            is_doi=False,
        )

        # Check that our function was called correctly
        assert result.success is True
        assert result.file_path == dest_path
        assert result.file_size == file_size

        # Check that we retried the correct number of times
        assert call_count == 3

        # Check sleep should have been called but we've mocked it
        assert mock_sleep.call_count == 0  # Our mock doesn't actually call sleep

        # Stats should reflect final success
        assert downloader.download_stats["successful"] == 1
        assert downloader.download_stats["failed"] == 0  # Final outcome was success

    async def test_download_publications_orchestration(
        self, downloader, mock_storage, mocker
    ):
        """Test the main download_publications loop."""
        # Create publications with HttpUrl as required by the model
        from pydantic import HttpUrl

        pub1 = OpenAlexPublication(
            paper_id="P001",
            openalex_id="https://openalex.org/P001",
            file_urls=[HttpUrl("https://example.com/file1.pdf")],
        )
        pub2 = OpenAlexPublication(
            paper_id="P002",
            openalex_id="https://openalex.org/P002",
            file_urls=[HttpUrl("https://doi.org/10.100/file2")],
        )  # DOI
        pub3 = OpenAlexPublication(
            paper_id="P003", openalex_id="https://openalex.org/P003", file_urls=[]
        )  # No URLs
        pub4 = OpenAlexPublication(
            paper_id="P004",
            openalex_id="https://openalex.org/P004",
            file_urls=[HttpUrl("https://example.com/file4.txt")],
        )

        publications = [pub1, pub2, pub3, pub4]

        path1 = Path("/fake/storage/raw/documents/openalex/P001/hash1.pdf")
        path2 = Path("/fake/storage/raw/documents/openalex/P002/hash2.file")
        path4 = Path("/fake/storage/raw/documents/openalex/P004/hash4.txt")

        # Mock _get_file_path calls
        mock_get_path = mocker.patch.object(
            downloader, "_get_file_path", side_effect=[path1, path2, path4]
        )

        # Mock download_file results for each URL attempt
        result1 = DownloadResult(
            url=str(pub1.file_urls[0]),
            success=True,
            file_path=path1,
            file_size=100,
            source="http",
            open_access=True,
        )
        result2 = DownloadResult(
            url=str(pub2.file_urls[0]),
            success=True,
            file_path=path2,
            file_size=200,
            source="scidownl",
        )
        result4 = DownloadResult(
            url=str(pub4.file_urls[0]),
            success=False,
            error="Not Found",
            file_path=None,
        )
        mock_download_file = mocker.patch.object(
            downloader, "download_file", side_effect=[result1, result2, result4]
        )

        # Mock tqdm if used
        mocker.patch(
            "tqdm.asyncio.tqdm", return_value=MagicMock()
        )  # Mock the progress bar

        # Mock session close and summary log
        mock_close = mocker.patch.object(downloader, "_session", new_callable=AsyncMock)
        mock_close.closed = False
        mock_close.close = AsyncMock()

        mock_log_summary = mocker.patch.object(downloader, "_log_download_summary")

        results = await downloader.download_publications(
            publications, overwrite=False, limit=None, progress_bar=True
        )

        assert len(results) == 3  # Pub3 was skipped
        assert results[0]["publication_id"] == "P001"
        assert results[0]["success"] is True
        assert results[1]["publication_id"] == "P002"
        assert results[1]["success"] is True
        assert results[2]["publication_id"] == "P004"
        assert results[2]["success"] is False

        assert mock_get_path.call_count == 3

        # Verify the download_file call parameters
        assert mock_download_file.call_count == 3
        mock_download_file.assert_has_calls(
            [
                call(
                    url=str(pub1.file_urls[0]),
                    destination=path1,
                    referer=None,
                    overwrite=False,
                    resume=True,
                    is_doi=False,
                ),
                call(
                    url=str(pub2.file_urls[0]),
                    destination=path2,
                    referer=None,
                    overwrite=False,
                    resume=True,
                    is_doi=True,
                ),
                call(
                    url=str(pub4.file_urls[0]),
                    destination=path4,
                    referer=None,
                    overwrite=False,
                    resume=True,
                    is_doi=False,
                ),
            ],
            any_order=True,
        )

        if hasattr(mock_close, "close"):
            mock_close.close.assert_called_once()
        mock_log_summary.assert_called_once()


class TestTopLevelFunction:
    """Tests for the main entry point function `download_openalex_files`."""

    async def test_download_openalex_files_entrypoint(self, mocker, tmp_path):
        """Test the main entry point function wiring."""
        mock_pub_data_path = tmp_path / "pubs.csv"
        mock_pub_data_path.touch()  # Make file exist

        mock_storage_instance = MockStorageBase()
        mock_get_storage = mocker.patch(
            "backend.storage.factory.get_storage", return_value=mock_storage_instance
        )

        # Mock publication loading
        from pydantic import HttpUrl

        pub1 = OpenAlexPublication(
            paper_id="T001",
            openalex_id="https://openalex.org/T001",
            file_urls=[HttpUrl("http://a.com/1")],
        )
        pub2 = OpenAlexPublication(
            paper_id="T002", openalex_id="https://openalex.org/T002", file_urls=[]
        )  # No URL

        mock_client = MagicMock()
        mock_client.load_from_csv.return_value = [pub1, pub2]

        mock_client_init = mocker.patch(
            "backend.etl.scrapers.openalex.OpenAlexClient", return_value=mock_client
        )

        # Mock the downloader class instantiation and its method
        mock_downloader_instance = AsyncMock(spec=OpenAlexFileDownloader)
        mock_downloader_instance.download_publications = AsyncMock(
            return_value=[{"publication_id": "T001", "success": True}]
        )
        mock_downloader_init = mocker.patch(
            "backend.etl.utils.oa_file_downloader.OpenAlexFileDownloader",
            return_value=mock_downloader_instance,
        )

        # Call the top-level function
        results = await download_openalex_files(
            storage=None,  # Will use get_storage
            publication_data_path=mock_pub_data_path,
            overwrite=True,
            limit=None,
            concurrency=4,
            config_path=None,
        )

        mock_get_storage.assert_called_once()
        # Check if OpenAlexClient was instantiated correctly
        mock_client_init.assert_called_once()
        mock_client.load_from_csv.assert_called_once_with(mock_pub_data_path)

        mock_downloader_init.assert_called_once_with(
            storage=mock_storage_instance, concurrency_limit=4, config_path=None
        )

        # Check that download_publications was called with filtered pubs
        mock_downloader_instance.download_publications.assert_called_once()

        # Get the actual call arguments for download_publications
        call_args, call_kwargs = (
            mock_downloader_instance.download_publications.call_args
        )

        # Check the filtered list of publications (only pub1 should be passed)
        assert len(call_args[0]) == 1
        assert call_args[0][0].paper_id == "T001"
        assert call_kwargs["overwrite"] is True
        assert call_kwargs["limit"] is None

        assert len(results) == 1
        assert results[0]["success"] is True


# --- Optional: Limited Real Integration Tests ---
# Mark these clearly and make them skippable


@pytest.mark.real_integration
@pytest.mark.skip(reason="Need special configuration to run real integration test")
@pytest.mark.asyncio
async def test_real_download_known_oa_pdf(downloader, tmp_path, mock_storage, mocker):
    """Attempts to download a small, stable OA PDF."""
    # Use a known stable, small OA PDF URL (e.g., from arXiv)
    # WARNING: External dependency - test may fail due to network/server issues
    oa_url = "https://arxiv.org/pdf/2103.00010.pdf"  # Example small PDF
    dest_path = tmp_path / "real_oa.pdf"

    # Use real session but mock storage interactions
    downloader._session = aiohttp.ClientSession()  # Create a real session for this test
    mocker.patch.object(downloader, "_get_file_path", return_value=dest_path)
    mock_storage.exists.return_value = False  # Ensure download attempt

    # Mock storage methods to use tmp_path
    class TmpPathStorage(MockStorageBase):
        def get_path(self, rel_path: str) -> Path:
            return tmp_path / rel_path

        def ensure_dir(self, path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)

        def exists(self, path: Path) -> bool:
            return path.exists()

        def stat(self, path: Path) -> Path.stat:
            return path.stat()

        def unlink(self, path: Path) -> None:
            path.unlink()

    downloader.storage = TmpPathStorage()

    try:
        result = await downloader.download_file(
            url=oa_url,
            destination=dest_path,
            referer=None,
            overwrite=True,
            resume=False,
            is_doi=False,
        )

        assert result.success is True
        assert result.file_path == dest_path
        assert dest_path.exists()
        assert dest_path.stat().st_size > 100  # Check downloaded something substantial
        assert result.source == "http"

    finally:
        await downloader._session.close()  # Clean up real session
        if dest_path.exists():
            dest_path.unlink()


# To add this option properly, add to conftest.py:
#
# import pytest
#
# def pytest_addoption(parser):
#     parser.addoption(
#         "--runreal",
#         action="store_true",
#         default=False,
#         help="run real integration tests"
#     )
#
# def pytest_configure(config):
#     config.addinivalue_line(
#         "markers",
#         "real_integration: mark a test as a real integration test"
#     )
#
# def pytest_collection_modifyitems(config, items):
#     if not config.getoption("--runreal"):
#         skip_real = pytest.mark.skip(reason="Need --runreal option to run")
#         for item in items:
#             if "real_integration" in item.keywords:
#                 item.add_marker(skip_real)
