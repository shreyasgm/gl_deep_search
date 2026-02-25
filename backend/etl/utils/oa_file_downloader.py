"""
OpenAlex file downloader module for Deep Search.

Handles downloading of files from OpenAlex publication DOIs, with features like:
1. Checking for open access versions
2. Using scidownl as a fallback for closed-access papers
3. Retrying, rate limiting, and validation
"""

import asyncio
import hashlib
import logging
import mimetypes
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    TypeVar,
)
from urllib.parse import urlparse

import aiofiles
import aiohttp
import tqdm.asyncio
from scidownl import scihub_download

from backend.etl.models.publications import OpenAlexPublication
from backend.etl.scrapers.openalex import OpenAlexClient
from backend.etl.utils.retry import retry_with_backoff
from backend.storage.base import StorageBase
from backend.storage.factory import get_storage

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar("T")


@dataclass
class DownloadResult:
    """Result of a file download operation."""

    url: str
    success: bool
    file_path: Path | None = None
    error: str | None = None
    file_size: int | None = None
    content_type: str | None = None
    cached: bool = False
    validation_info: dict[str, Any] | None = None
    open_access: bool = False
    source: str | None = None


class OpenAlexFileDownloader:
    """
    Asynchronous file downloader for OpenAlex publications.

    Features:
    - Open access verification and download
    - Scidownl fallback for closed-access papers
    - Asynchronous downloads with concurrency limits
    - Rate limiting and retry logic
    - Downloaded file validation
    - Intelligent caching
    """

    def __init__(
        self,
        storage: StorageBase | None = None,
        concurrency_limit: int = 3,
        config_path: Path | None = None,
    ):
        """
        Initialize the file downloader.

        Args:
            storage: Storage backend to use (defaults to factory-configured storage)
            concurrency_limit: Maximum number of concurrent downloads
            config_path: Path to configuration file
        """
        # Storage configuration
        self.storage = storage or get_storage()

        # Load configuration or use defaults
        self.config = self._load_config(config_path)

        # Concurrency and rate limiting
        self.concurrency_limit = concurrency_limit
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.download_delay = self.config.get("download_delay", 1.0)

        # Retry configuration
        self.max_retries = self.config.get("max_retries", 5)
        self.base_delay = self.config.get("retry_base_delay", 1.0)
        self.max_delay = self.config.get("retry_max_delay", 60.0)

        # File validation
        self.min_file_size = self.config.get("min_file_size", 1024)  # 1KB
        self.max_file_size = self.config.get(
            "max_file_size", 100 * 1024 * 1024
        )  # 100MB

        # Download statistics
        self.download_stats = {
            "total_attempted": 0,
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "total_bytes": 0,
            "open_access": 0,
            "scidownl": 0,
        }

        # Session cache
        self._session: aiohttp.ClientSession | None = None

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml

        if config_path is None:
            # Look for config in standard location
            config_path = Path(__file__).parent.parent / "config.yaml"

        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    result = config.get("oa_file_downloader", {})
                    # Fall back to sources.openalex for unpaywall_email
                    if "unpaywall_email" not in result:
                        oa_config = config.get("sources", {}).get("openalex", {})
                        if "unpaywall_email" in oa_config:
                            result["unpaywall_email"] = oa_config["unpaywall_email"]
                    return result
            except Exception as e:
                logger.warning(
                    f"Error loading file downloader config: {e}. Using defaults."
                )

        # Default configuration
        return {
            "download_delay": 1.0,
            "max_retries": 5,
            "retry_base_delay": 1.0,
            "retry_max_delay": 60.0,
            "min_file_size": 1024,  # 1KB
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "user_agent_list": [
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",  # noqa: E501
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",  # noqa: E501
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",  # noqa: E501
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/120.0.0.0",  # noqa: E501
            ],
            "unpaywall_email": "example@example.com",
            "open_access_apis": [
                "unpaywall",  # Unpaywall API
                "oadoi",  # oaDOI service
                "core",  # CORE API
            ],
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session with proper configuration."""
        if self._session is None or self._session.closed:
            # Configure timeouts
            timeout = aiohttp.ClientTimeout(
                total=60,
                connect=20,
                sock_connect=20,
                sock_read=20,
            )

            # Set up connection pooling
            connector = aiohttp.TCPConnector(
                limit=self.concurrency_limit,
                ttl_dns_cache=300,
                force_close=False,
                enable_cleanup_closed=True,
            )

            # Random user agent
            user_agents = self.config.get(
                "user_agent_list",
                [
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"  # noqa: E501
                ],
            )
            user_agent = random.choice(user_agents)

            # Default headers
            headers = {
                "User-Agent": user_agent,
                "Accept": "application/pdf,application/octet-stream,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*",  # noqa: E501
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }

            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers,
                cookie_jar=aiohttp.CookieJar(unsafe=True),
            )

        return self._session

    def _get_file_path(self, publication: OpenAlexPublication, file_url: str) -> Path:
        """
        Determine the appropriate path for saving a file.

        Args:
            publication: Publication metadata
            file_url: URL of the file

        Returns:
            Path where the file should be saved
        """
        # Clean URL by removing query parameters and fragments
        clean_url = file_url.split("?")[0].split("#")[0]

        # Get the file extension from the URL path
        url_path = urlparse(clean_url).path
        ext = Path(url_path).suffix.lower()

        # If no extension found, use empty string
        if not ext:
            ext = ""

        # Generate MD5 hash of the cleaned file URL
        url_hash = hashlib.md5(clean_url.encode()).hexdigest()

        # Use the publication ID to create a directory
        pub_id = publication.paper_id
        if not pub_id:
            # Generate ID if not set
            pub_id = publication.generate_id()

        # Path structure: data/raw/documents/openalex/<publication_id>/<url_hash><ext>
        relative_path = f"raw/documents/openalex/{pub_id}/{url_hash}{ext}"

        return self.storage.get_path(relative_path)

    async def _check_open_access(self, doi: str) -> tuple[bool, str | None]:
        """
        Check if an open access version of the paper is available.

        Args:
            doi: DOI of the publication

        Returns:
            Tuple of (is_open_access, download_url)
        """
        if not doi.startswith("https://doi.org/"):
            doi = f"https://doi.org/{doi.lstrip('doi:').strip()}"

        # Try Unpaywall API
        unpaywall_email = self.config.get("unpaywall_email", "example@example.com")
        doi_id = doi.replace("https://doi.org/", "")
        unpaywall_url = f"https://api.unpaywall.org/v2/{doi_id}?email={unpaywall_email}"

        session = await self._get_session()
        try:
            async with session.get(unpaywall_url) as response:
                if response.status == 200:
                    data = await response.json()
                    best_oa_url = None

                    # Look for the best open access URL
                    if data.get("is_oa"):
                        # First check for a direct PDF link
                        if data.get("best_oa_location", {}).get("url_for_pdf"):
                            best_oa_url = data["best_oa_location"]["url_for_pdf"]
                        # Otherwise get the landing page
                        elif data.get("best_oa_location", {}).get("url"):
                            best_oa_url = data["best_oa_location"]["url"]

                    if best_oa_url:
                        logger.info(
                            f"Found open access URL via Unpaywall: {best_oa_url}"
                        )
                        return True, best_oa_url
        except Exception as e:
            logger.warning(f"Error checking Unpaywall for {doi}: {e}")

        # Try CORE API (simplified version without API key for demo)
        try:
            doi_id = doi.replace("https://doi.org/", "")
            core_url = f"https://core.ac.uk/api-v2/articles/search/doi:{doi_id}"
            async with session.get(core_url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        # Look for downloadUrl in the first result
                        if data["data"][0].get("downloadUrl"):
                            download_url = data["data"][0]["downloadUrl"]
                            logger.info(
                                f"Found open access URL via CORE: {download_url}"
                            )
                            return True, download_url
        except Exception as e:
            logger.warning(f"Error checking CORE for {doi}: {e}")

        # If we got here, we couldn't find an open access version
        return False, None

    async def _resolve_url_and_check_content(
        self,
        session: aiohttp.ClientSession,
        url: str,
        max_redirects: int = 5,
    ) -> tuple[str | None, str | None, bool]:
        """
        Resolve a URL by following redirects and check its content type.

        Args:
            session: aiohttp session to use
            url: URL to resolve
            max_redirects: Maximum number of redirects to follow

        Returns:
            Tuple of (final_url, content_type, is_direct_download)
            - final_url: The resolved URL after following redirects
            - content_type: The content type of the final resource
            - is_direct_download: Whether this appears to be a direct file download
        """
        current_url = url
        redirect_count = 0

        while redirect_count < max_redirects:
            try:
                # Make HEAD request to check content type and follow redirects
                async with session.head(
                    current_url,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    # Handle redirects
                    if response.status in (301, 302, 303, 307, 308):
                        redirect_count += 1
                        location = response.headers.get("Location")
                        if not location:
                            logger.warning(
                                f"No Location header in redirect from {current_url}"
                            )
                            return None, None, False

                        # Handle relative redirects
                        if not location.startswith(("http://", "https://")):
                            parsed = urlparse(current_url)
                            location = f"{parsed.scheme}://{parsed.netloc}{location}"

                        current_url = location
                        continue

                    # Check if this is a direct file download
                    content_type = response.headers.get("Content-Type", "").lower()
                    content_disposition = response.headers.get(
                        "Content-Disposition", ""
                    ).lower()

                    # Check for direct file indicators
                    is_direct_download = (
                        # Content type indicates a file
                        any(
                            ct in content_type
                            for ct in [
                                "application/pdf",
                                "application/octet-stream",
                                "application/msword",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            ]
                        )
                        or
                        # Content-Disposition indicates a file
                        "attachment" in content_disposition
                        or "filename=" in content_disposition
                    )

                    return current_url, content_type, is_direct_download

            except aiohttp.ClientError as e:
                logger.warning(f"Error resolving URL {current_url}: {e}")
                return None, None, False

        logger.warning(f"Too many redirects for URL {url}")
        return None, None, False

    async def _download_file_with_aiohttp(
        self,
        session: aiohttp.ClientSession,
        url: str,
        destination: Path,
        referer: str | None = None,
        resume: bool = True,
    ) -> DownloadResult:
        """
        Download file using aiohttp.

        Args:
            session: aiohttp session to use
            url: URL to download
            destination: Where to save the file
            referer: Optional referer header
            resume: Whether to attempt resuming partial downloads

        Returns:
            DownloadResult with information about the download
        """
        # First resolve the URL and check content type
        (
            final_url,
            content_type,
            is_direct_download,
        ) = await self._resolve_url_and_check_content(session, url)

        if not final_url:
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error="Failed to resolve URL or too many redirects",
                open_access=True,
                source="http",
            )

        if not is_direct_download:
            logger.warning(
                f"URL {final_url} appears to be a landing page or "
                f"non-downloadable content. "
                f"Content-Type: {content_type}"
            )
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error="URL points to a landing page or non-downloadable content",
                open_access=True,
                source="http",
            )

        # Check if file already exists and get its size for resume
        file_size = 0
        if destination.exists() and resume:
            file_size = destination.stat().st_size

        # Set up headers with range request if resuming
        headers = {}
        if referer:
            headers["Referer"] = referer

        if file_size > 0 and resume:
            headers["Range"] = f"bytes={file_size}-"
            logger.info(f"Resuming download of {url} from byte {file_size}")

        # Ensure directory exists
        self.storage.ensure_dir(destination.parent)

        # Download with progress tracking
        try:
            # Make range request if resuming, otherwise normal request
            async with session.get(final_url, headers=headers) as response:
                # Handle response
                if response.status == 416:  # Range Not Satisfiable
                    # File is already complete
                    logger.info(f"File {destination} appears to be already complete")

                    # Validate the existing file
                    validation_result = await self._validate_downloaded_file(
                        destination,
                        expected_content_type=response.headers.get("Content-Type"),
                    )

                    return DownloadResult(
                        url=url,
                        success=validation_result["is_valid"],
                        file_path=destination,
                        file_size=file_size,
                        content_type=response.headers.get(
                            "Content-Type"
                        ),  # Use reported content type
                        cached=True,
                        validation_info=validation_result,
                        open_access=True,
                        source="http",
                    )

                elif response.status == 206:  # Partial Content (for range requests)
                    # Resume download
                    content_length = int(response.headers.get("Content-Length", "0"))
                    total_size = file_size + content_length
                    mode = "ab"  # Append binary

                elif response.status == 200:  # OK
                    # New download or server doesn't support range requests
                    content_length = int(response.headers.get("Content-Length", "0"))
                    total_size = content_length
                    file_size = 0  # Start from beginning
                    mode = "wb"  # Write binary

                else:
                    # Other status codes are errors
                    error_msg = f"HTTP error {response.status} when downloading {url}"
                    logger.error(error_msg)
                    return DownloadResult(
                        url=url,
                        success=False,
                        file_path=destination,
                        error=error_msg,
                        open_access=True,
                        source="http",
                    )

                # Get content type
                content_type = response.headers.get(
                    "Content-Type", "application/octet-stream"
                )

                # Download the file with progress tracking
                try:
                    async with aiofiles.open(destination, mode) as f:
                        downloaded = file_size  # Bytes already downloaded
                        chunk_size = 64 * 1024  # 64KB chunks

                        async for chunk in response.content.iter_chunked(chunk_size):
                            await f.write(chunk)
                            downloaded += len(chunk)
                except Exception as e:
                    logger.error(f"Error writing to file {destination}: {e}")
                    return DownloadResult(
                        url=url,
                        success=False,
                        file_path=destination,
                        error=f"Failed to write file: {str(e)}",
                        open_access=True,
                        source="http",
                    )

                # Get final file size
                if destination.exists():
                    final_size = destination.stat().st_size
                else:
                    final_size = 0

                # Validate the downloaded file
                validation_result = await self._validate_downloaded_file(
                    destination, expected_content_type=content_type
                )

                if not validation_result["is_valid"]:
                    logger.error(
                        f"Downloaded file validation failed for {url}: "
                        f"{validation_result}"
                    )

                    # Delete invalid file
                    if destination.exists():
                        destination.unlink()

                    return DownloadResult(
                        url=url,
                        success=False,
                        file_path=destination,
                        error=f"Validation failed: {validation_result}",
                        file_size=final_size,
                        content_type=content_type,
                        validation_info=validation_result,
                        open_access=True,
                        source="http",
                    )

                return DownloadResult(
                    url=url,
                    success=True,
                    file_path=destination,
                    file_size=final_size,
                    content_type=content_type,
                    validation_info=validation_result,
                    open_access=True,
                    source="http",
                )

        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error when downloading {url}: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
                open_access=True,
                source="http",
            )

        except TimeoutError:
            error_msg = f"Timeout error when downloading {url}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
                open_access=True,
                source="http",
            )

        except Exception as e:
            error_msg = f"Unexpected error when downloading {url}: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
                open_access=True,
                source="http",
            )

    async def _download_file_with_scidownl(
        self, doi: str, destination: Path
    ) -> DownloadResult:
        """
        Download file using scidownl library.

        Args:
            doi: DOI URL to download
            destination: Where to save the file

        Returns:
            DownloadResult with information about the download
        """
        # Ensure destination directory exists
        self.storage.ensure_dir(destination.parent)

        # Make sure DOI is formatted correctly for scidownl
        if not doi.startswith("https://doi.org/"):
            doi = f"https://doi.org/{doi.lstrip('doi:').strip()}"

        try:
            # Configure proxies if needed
            proxies = self.config.get("proxies", None)

            # Download the paper
            logger.info(f"Downloading paper with scidownl: {doi}")
            scihub_download(
                keyword=doi, paper_type="doi", out=str(destination), proxies=proxies
            )

            # Validate the file
            file_size = destination.stat().st_size
            content_type = (
                mimetypes.guess_type(str(destination))[0] or "application/pdf"
            )

            # Validate the file
            validation_result = await self._validate_downloaded_file(
                destination, expected_content_type=content_type
            )

            if not validation_result["is_valid"]:
                logger.error(f"scidownl file validation failed: {validation_result}")
                if destination.exists():
                    destination.unlink()
                return DownloadResult(
                    url=doi,
                    success=False,
                    file_path=destination,
                    error=f"Validation failed: {validation_result}",
                    file_size=file_size,
                    content_type=content_type,
                    validation_info=validation_result,
                    source="scidownl",
                )

            return DownloadResult(
                url=doi,
                success=True,
                file_path=destination,
                file_size=file_size,
                content_type=content_type,
                validation_info=validation_result,
                source="scidownl",
            )

        except Exception as e:
            error_msg = f"Error using scidownl: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                url=doi,
                success=False,
                file_path=destination,
                error=error_msg,
                source="scidownl",
            )

    async def _validate_downloaded_file(
        self, file_path: Path, expected_content_type: str | None = None
    ) -> dict[str, Any]:
        """
        Validate a downloaded file.

        Performs checks like:
        - File exists
        - File size is reasonable
        - Basic format validation for common file types

        Args:
            file_path: Path to the file to validate
            expected_content_type: Expected content type (optional)

        Returns:
            Dict with validation results
        """
        result = {
            "is_valid": False,
            "file_exists": False,
            "size_check": False,
            "format_check": False,
            "file_size": 0,
        }

        # Check if file exists
        if not file_path.exists():
            result["is_valid"] = False
            return result

        result["file_exists"] = True

        # Check file size
        file_size = file_path.stat().st_size
        result["file_size"] = file_size

        if file_size < self.min_file_size:
            result["size_check"] = False
            result["is_valid"] = False
            logger.warning(f"File too small: {file_path} ({file_size} bytes)")
            return result

        if file_size > self.max_file_size:
            result["size_check"] = False
            result["is_valid"] = False
            logger.warning(f"File too large: {file_path} ({file_size} bytes)")
            return result

        result["size_check"] = True

        # Determine file type
        if not expected_content_type:
            expected_content_type = mimetypes.guess_type(str(file_path))[0]

        # File format check based on extension and magic bytes
        try:
            # Read the first few bytes to check for file signatures
            async with aiofiles.open(file_path, "rb") as f:
                header = await f.read(1024)  # Read first KB

                # Check for basic file validity based on file type
                if (
                    expected_content_type == "application/pdf"
                    or file_path.suffix.lower() == ".pdf"
                ):
                    # Check for PDF signature %PDF-
                    if header.startswith(b"%PDF-"):
                        result["format_check"] = True
                    else:
                        result["format_check"] = False
                        result["is_valid"] = False
                        logger.warning(f"File is not a valid PDF: {file_path}")
                        return result

                elif expected_content_type in [
                    "application/msword",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ] or file_path.suffix.lower() in [".doc", ".docx"]:
                    # Check for DOC signature (D0 CF 11 E0 A1 B1 1A E1)
                    # or DOCX signature (PK..)
                    if header.startswith(
                        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
                    ) or header.startswith(b"PK\x03\x04"):
                        result["format_check"] = True
                    else:
                        result["format_check"] = False
                        result["is_valid"] = False
                        logger.warning(
                            f"File is not a valid Word document: {file_path}"
                        )
                        return result

                # For other file types, we don't do specific validation
                else:
                    # Just verify it's not empty and has some content
                    if len(header) > 0:
                        result["format_check"] = True
                    else:
                        result["format_check"] = False
                        result["is_valid"] = False
                        logger.warning(f"File appears to be empty: {file_path}")
                        return result

        except Exception as e:
            result["format_check"] = False
            result["is_valid"] = False
            logger.error(f"Error validating file {file_path}: {e}")
            return result

        # All checks passed
        result["is_valid"] = True
        return result

    async def download_file(
        self,
        url: str,
        destination: Path,
        referer: str | None = None,
        overwrite: bool = False,
        resume: bool = True,
        is_doi: bool = False,
    ) -> DownloadResult:
        """
        Download a file with retry logic and validation.

        Args:
            url: URL to download
            destination: Where to save the file
            referer: Optional referer URL
            overwrite: Whether to overwrite existing files
            resume: Whether to attempt resuming partial downloads
            is_doi: Whether the URL is a DOI (for special handling)

        Returns:
            DownloadResult with information about the download
        """
        # Check if file already exists and skip if not overwriting
        if destination.exists() and not overwrite and not resume:
            logger.info(f"File {destination} already exists, skipping download")

            # Still validate the existing file
            content_type = (
                mimetypes.guess_type(str(destination))[0] or "application/octet-stream"
            )
            validation_result = await self._validate_downloaded_file(
                destination, expected_content_type=content_type
            )

            # Update statistics
            self.download_stats["total_attempted"] += 1
            self.download_stats["cached"] += 1

            return DownloadResult(
                url=url,
                success=validation_result["is_valid"],
                file_path=destination,
                file_size=destination.stat().st_size,
                content_type=content_type,
                cached=True,
                validation_info=validation_result,
            )

        # Get aiohttp session
        session = await self._get_session()

        # Use semaphore to limit concurrency
        async with self.semaphore:
            # Update statistics
            self.download_stats["total_attempted"] += 1

            # Special handling for DOI URLs
            if is_doi or "doi.org" in url:
                # Clean DOI format
                if not url.startswith("https://doi.org/"):
                    url = f"https://doi.org/{url.replace('doi:', '').strip()}"

                # Step 1: Check for open access version
                is_open_access, oa_url = await self._check_open_access(url)

                if is_open_access and oa_url:
                    logger.info(f"Found open access URL for {url}: {oa_url}")
                    # Use the open access URL for download
                    download_result = await retry_with_backoff(
                        self._download_file_with_aiohttp,
                        session,
                        oa_url,
                        destination,
                        referer,
                        resume,
                        max_retries=self.max_retries,
                        base_delay=self.base_delay,
                        max_delay=self.max_delay,
                        retry_on=(
                            aiohttp.ClientError,
                            TimeoutError,
                            asyncio.TimeoutError,
                        ),
                    )

                    if download_result.success:
                        self.download_stats["open_access"] += 1
                        self.download_stats["successful"] += 1
                        if download_result.file_size:
                            self.download_stats["total_bytes"] += (
                                download_result.file_size
                            )
                        return download_result

                # Step 2: If open access failed or not available, try scidownl
                logger.info(
                    "No open access version found or download failed, "
                    f"trying scidownl for {url}"
                )
                scidownl_result = await self._download_file_with_scidownl(
                    url, destination
                )

                # Update statistics
                if scidownl_result.success:
                    self.download_stats["scidownl"] += 1
                    self.download_stats["successful"] += 1
                    if scidownl_result.file_size:
                        self.download_stats["total_bytes"] += scidownl_result.file_size
                else:
                    self.download_stats["failed"] += 1

                # Add small delay after scidownl use
                await asyncio.sleep(self.download_delay * (0.5 + random.random()))

                return scidownl_result

            else:
                # Standard URL download (non-DOI)
                download_result = await retry_with_backoff(
                    self._download_file_with_aiohttp,
                    session,
                    url,
                    destination,
                    referer,
                    resume,
                    max_retries=self.max_retries,
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                    retry_on=(aiohttp.ClientError, TimeoutError, asyncio.TimeoutError),
                )

                # Add a small random delay to avoid overwhelming servers
                await asyncio.sleep(self.download_delay * (0.5 + random.random()))

                # Update statistics
                if download_result.success:
                    self.download_stats["successful"] += 1
                    if download_result.file_size:
                        self.download_stats["total_bytes"] += download_result.file_size
                elif download_result.cached:
                    self.download_stats["cached"] += 1
                else:
                    self.download_stats["failed"] += 1

                return download_result

    async def download_publications(
        self,
        publications: Sequence[OpenAlexPublication],
        overwrite: bool = False,
        limit: int | None = None,
        progress_bar: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Download files for a list of publications.

        Args:
            publications: List of publications to download files for
            overwrite: Whether to overwrite existing files
            limit: Maximum number of publications to process (for testing)
            progress_bar: Whether to show a progress bar

        Returns:
            List of download results by publication
        """
        # Start client session if not already created
        session = await self._get_session()

        # Apply limit if specified
        if limit and limit > 0:
            publications = publications[:limit]
            logger.info(f"Limited to downloading files for {limit} publications")

        # Find all file URLs to download
        all_downloads = []
        for pub in publications:
            if not pub.file_urls:
                continue

            for url in pub.file_urls:
                # Convert HttpUrl to string
                url_str = str(url)

                # Get destination path
                dest_path = self._get_file_path(pub, url_str)

                # Store the original HttpUrl object, not the string
                all_downloads.append((pub, url, dest_path))

        # Log download plan
        logger.info(f"Found {len(all_downloads)} files to download")

        # Download files with progress tracking
        results = []

        # Create tasks for all downloads
        tasks = []
        for pub, url, dest_path in all_downloads:
            # Cast URL to string to ensure compatibility
            url_str = str(url)
            is_doi = "doi.org" in url_str or url_str.startswith("10.")

            task = self.download_file(
                url=url_str,
                destination=dest_path,
                referer=str(pub.pub_url) if pub.pub_url else None,
                overwrite=overwrite,
                resume=True,
                is_doi=is_doi,
            )
            # Store publication, URL string, and task in tasks
            # We must use string to avoid HttpUrl type error
            tasks.append((pub, url, task))

        # Process downloads with progress bar
        with tqdm.asyncio.tqdm(
            total=len(tasks),
            desc="Downloading files",
            disable=not progress_bar,
        ) as pbar:
            for pub, url, download_task in tasks:
                try:
                    # Convert url to string for logging and results
                    url_str = str(url)

                    # Await the download task
                    result = await download_task

                    # Record the result with publication info
                    pub_result = {
                        "publication_id": pub.paper_id,
                        "publication_title": pub.title,
                        "url": url_str,
                        "success": result.success,
                        "file_path": result.file_path,
                        "file_size": result.file_size,
                        "cached": result.cached,
                        "error": result.error,
                        "open_access": result.open_access,
                        "source": result.source,
                    }
                    results.append(pub_result)

                    # Update progress
                    pbar.update(1)

                    # Success/failure message
                    if result.success:
                        status = (
                            "cached"
                            if result.cached
                            else f"downloaded via {result.source}"
                        )
                        pbar.set_postfix_str(f"Last: {status}")
                    else:
                        pbar.set_postfix_str(f"Last: failed - {result.error}")

                except Exception as e:
                    # Convert url to string for logging and results
                    url_str = str(url)

                    # Record unexpected errors
                    logger.error(f"Unexpected error downloading {url_str}: {e}")
                    pub_result = {
                        "publication_id": pub.paper_id,
                        "publication_title": pub.title,
                        "url": url_str,
                        "success": False,
                        "error": f"Unexpected error: {str(e)}",
                    }
                    results.append(pub_result)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Last: error - {str(e)}")

        # Close the session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        # Log summary
        self._log_download_summary()

        return results

    def _log_download_summary(self) -> None:
        """Log a summary of download statistics."""
        total = self.download_stats["total_attempted"]
        if total == 0:
            logger.info("No downloads were attempted")
            return

        successful = self.download_stats["successful"]
        cached = self.download_stats["cached"]
        failed = self.download_stats["failed"]
        bytes_downloaded = self.download_stats["total_bytes"]
        open_access = self.download_stats["open_access"]
        scidownl = self.download_stats["scidownl"]

        # Convert bytes to human-readable format
        if bytes_downloaded < 1024:
            size_str = f"{bytes_downloaded} bytes"
        elif bytes_downloaded < 1024 * 1024:
            size_str = f"{bytes_downloaded / 1024:.1f} KB"
        else:
            size_str = f"{bytes_downloaded / (1024 * 1024):.1f} MB"

        logger.info("-" * 50)
        logger.info("File Download Summary:")
        logger.info(f"Total attempts: {total}")
        logger.info(
            f"  - Successfully downloaded: {successful} "
            f"({successful / total * 100:.1f}%)"
        )
        logger.info(f"    - Via open access: {open_access}")
        logger.info(f"    - Via scidownl: {scidownl}")
        logger.info(f"  - Used cached files: {cached} ({cached / total * 100:.1f}%)")
        logger.info(f"  - Failed: {failed} ({failed / total * 100:.1f}%)")
        logger.info(f"Total data downloaded: {size_str}")
        logger.info("-" * 50)


async def download_openalex_files(
    storage: StorageBase | None = None,
    publication_data_path: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    concurrency: int = 3,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Download files for OpenAlex publications.

    Args:
        storage: Storage backend to use
        publication_data_path: Path to publication CSV (defaults to standard location)
        overwrite: Whether to overwrite existing files
        limit: Maximum number of publications to process (for testing)
        concurrency: Maximum concurrent downloads
        config_path: Path to configuration file

    Returns:
        List of download results
    """
    # Get storage
    storage = storage or get_storage()

    # Default path if not provided
    if not publication_data_path:
        publication_data_path = storage.get_path(
            "intermediate/openalex_publications.csv"
        )

    # Check if publication data exists
    if not publication_data_path.exists():
        logger.error(f"Publication data not found at {publication_data_path}")
        return []

    # Load publication data
    client = OpenAlexClient()
    publications = client.load_from_csv(publication_data_path)

    if not publications:
        logger.error("No publications found in the data file")
        return []

    # Filter to publications with file URLs
    publications_with_files = [p for p in publications if p.file_urls]

    logger.info(
        f"Found {len(publications_with_files)}/{len(publications)} "
        "publications with file URLs"
    )

    # Create downloader and download files
    downloader = OpenAlexFileDownloader(
        storage=storage,
        concurrency_limit=concurrency,
        config_path=config_path,
    )

    results = await downloader.download_publications(
        publications_with_files,
        overwrite=overwrite,
        limit=limit,
        progress_bar=True,
    )

    return results
