"""
Growth Lab file downloader module for Deep Search.

Handles asynchronous downloading of files from Growth Lab publication URLs, with
features like retrying, rate limiting, and validation.
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

import aiofiles
import aiohttp
import tqdm.asyncio

from backend.etl.models.publications import GrowthLabPublication
from backend.etl.models.tracking import DownloadStatus
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.utils.publication_tracker import PublicationTracker
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


class FileDownloader:
    """
    Asynchronous file downloader for Growth Lab publications.

    Features:
    - Asynchronous downloads with concurrency limits
    - Rate limiting and retry logic
    - Downloaded file validation
    - Resume of partial downloads
    - Intelligent caching (won't re-download existing files unless overwrite=True)
    - Progress tracking
    - Publication tracking in database
    """

    def __init__(
        self,
        storage: StorageBase | None = None,
        concurrency_limit: int = 3,
        config_path: Path | None = None,
        publication_tracker: PublicationTracker | None = None,
    ):
        """
        Initialize the file downloader.

        Args:
            storage: Storage backend to use (defaults to factory-configured storage)
            concurrency_limit: Maximum number of concurrent downloads
            config_path: Path to configuration file
            publication_tracker: tracking download status
        """
        # Storage configuration
        self.storage = storage or get_storage()

        # Publication tracking
        self.publication_tracker = publication_tracker or PublicationTracker()

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
        }

        # Session cache
        self._session: aiohttp.ClientSession | None = None

    def _load_config(self, config_path: Path | None) -> dict[str, Any]:
        """Load configuration from YAML file."""
        if not config_path:
            # Look for config in standard location
            import yaml

            config_path = Path(__file__).parent.parent / "config.yaml"

            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("gl_file_downloader", {})
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
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
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
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ],
            )
            user_agent = random.choice(user_agents)

            # Default headers
            headers = {
                "User-Agent": user_agent,
                "Accept": (
                    "application/pdf,application/octet-stream,"
                    "application/msword,application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document,*/*"
                ),
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

    def _get_file_path(self, publication: GrowthLabPublication, file_url: str) -> Path:
        """
        Determine the appropriate path for saving a file.

        Args:
            publication: Publication metadata
            file_url: URL of the file

        Returns:
            Path where the file should be saved
        """
        # Convert HttpUrl to string if needed
        if not isinstance(file_url, str):
            file_url = str(file_url)

        # Extract filename from URL or generate a safe filename
        url_path = file_url.split("?")[0].split("#")[0]  # Remove query params
        file_name = url_path.split("/")[-1]

        # If the URL doesn't contain a filename or extension, try to determine one
        if not file_name or "." not in file_name:
            # Generate a filename if none exists
            url_hash = hashlib.md5(file_url.encode()).hexdigest()[:8]

            # Try to guess extension from content type if available
            ext = ".bin"  # Default extension for unknown types
            content_type = mimetypes.guess_type(file_url)[0]
            if content_type:
                if content_type == "application/pdf":
                    ext = ".pdf"
                elif content_type == "application/msword":
                    ext = ".doc"
                elif (
                    content_type
                    == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # noqa: E501
                ):
                    ext = ".docx"
                elif content_type.startswith("text/"):
                    ext = ".txt"
                elif content_type.startswith("image/"):
                    ext = f".{content_type.split('/')[-1]}"

            file_name = f"{url_hash}{ext}"

        # Use the publication ID to create a directory
        pub_id = publication.paper_id

        if not pub_id:
            # Generate ID if not set
            pub_id = publication.generate_id()

        # Path structure: data/raw/documents/growthlab/<publication_id>/<filename>
        relative_path = f"raw/documents/growthlab/{pub_id}/{file_name}"

        return self.storage.get_path(relative_path)

    async def _download_file_impl(
        self,
        session: aiohttp.ClientSession,
        url: str,
        destination: Path,
        referer: str | None = None,
        resume: bool = True,
    ) -> DownloadResult:
        """
        Implementation of file download with resume capability.

        Args:
            session: aiohttp session to use
            url: URL to download
            destination: Where to save the file
            referer: Optional referer header
            resume: Whether to attempt resuming partial downloads

        Returns:
            DownloadResult with information about the download
        """
        # Convert HttpUrl to string if needed
        if not isinstance(url, str):
            url = str(url)

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
            async with session.get(url, headers=headers) as response:
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
                    )

                return DownloadResult(
                    url=url,
                    success=True,
                    file_path=destination,
                    file_size=final_size,
                    content_type=content_type,
                    validation_info=validation_result,
                )

        except aiohttp.ClientError as e:
            error_msg = f"HTTP client error when downloading {url}: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
            )

        except TimeoutError:
            error_msg = f"Timeout error when downloading {url}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
            )

        except Exception as e:
            error_msg = f"Unexpected error when downloading {url}: {str(e)}"
            logger.error(error_msg)
            return DownloadResult(
                url=url,
                success=False,
                file_path=destination,
                error=error_msg,
            )

    async def download_file(
        self,
        url: str,
        destination: Path,
        referer: str | None = None,
        overwrite: bool = False,
        resume: bool = True,
    ) -> DownloadResult:
        """
        Download a file with retry logic and validation.

        Args:
            url: URL to download
            destination: Where to save the file
            referer: Optional referer URL
            overwrite: Whether to overwrite existing files
            resume: Whether to attempt resuming partial downloads

        Returns:
            DownloadResult with information about the download
        """
        # Convert HttpUrl to string if needed
        if not isinstance(url, str):
            url = str(url)

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

            # Download with retry
            # Explicitly annotate the result as DownloadResult to help mypy
            result: DownloadResult = await retry_with_backoff(
                self._download_file_impl,
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
            # Note: result is already awaited, no need to await again
            if result.success:
                self.download_stats["successful"] += 1
                if result.file_size:
                    self.download_stats["total_bytes"] += result.file_size
            elif result.cached:
                self.download_stats["cached"] += 1
            else:
                self.download_stats["failed"] += 1

            return result

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
                    and file_path.suffix.lower() == ".pdf"
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

    async def download_publications(
        self,
        publications: Sequence[GrowthLabPublication],
        overwrite: bool = False,
        limit: int | None = None,
        progress_bar: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Download files for a list of publications.

        Args:
            publications: List of publications to download
            overwrite: Whether to overwrite existing files
            limit: Maximum number of publications to download (None for all)
            progress_bar: Whether to show progress bar

        Returns:
            List of download results by publication
        """
        results: list[dict] = []  # List of download results

        # Limit the number of publications if specified
        pub_list = list(publications)
        if limit is not None and limit > 0:
            pub_list = pub_list[:limit]

        # Track publications in database before downloading
        for pub in pub_list:
            self.publication_tracker.add_publication(pub)

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

        # Create tasks for all downloads
        tasks = []
        for pub, url, dest_path in all_downloads:
            # Cast URL to string to ensure compatibility
            url_str = str(url)
            task = self.download_file(
                url=url_str,
                destination=dest_path,
                referer=str(pub.pub_url) if pub.pub_url else None,
                overwrite=overwrite,
                resume=True,
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
            try:
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
                        }
                        results.append(pub_result)

                        # Update download status in tracker
                        if result.success:
                            self.publication_tracker.update_download_status(
                                pub.paper_id, DownloadStatus.DOWNLOADED
                            )
                        else:
                            self.publication_tracker.update_download_status(
                                pub.paper_id, DownloadStatus.FAILED, error=result.error
                            )

                        # Update progress
                        pbar.update(1)

                        # Success/failure message
                        if result.success:
                            status = "cached" if result.cached else "downloaded"
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

                        # Update download status to failed
                        self.publication_tracker.update_download_status(
                            pub.paper_id,
                            DownloadStatus.FAILED,
                            error=f"Unexpected error: {str(e)}",
                        )
                        pbar.update(1)

                    # Rate limiting delay
                    await asyncio.sleep(self.download_delay)

                return results
            finally:
                pbar.close()

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
        logger.info(f"  - Used cached files: {cached} ({cached / total * 100:.1f}%)")
        logger.info(f"  - Failed: {failed} ({failed / total * 100:.1f}%)")
        logger.info(f"Total data downloaded: {size_str}")
        logger.info("-" * 50)


async def download_growthlab_files(
    storage: StorageBase | None = None,
    publication_data_path: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    concurrency: int = 3,
    config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Download files for all Growth Lab publications.

    Args:
        storage: Storage backend to use
        publication_data_path: Path to publication data CSV file
        overwrite: Whether to overwrite existing files
        limit: Maximum number of publications to download
        concurrency: Maximum number of concurrent downloads
        config_path: Path to configuration file

    Returns:
        List of download results by publication
    """
    # Initialize publication tracker
    publication_tracker = PublicationTracker()

    # Initialize downloader
    downloader = FileDownloader(
        storage=storage,
        concurrency_limit=concurrency,
        config_path=config_path,
        publication_tracker=publication_tracker,
    )

    # Get storage
    storage = storage or get_storage()

    # Default path if not provided
    if not publication_data_path:
        publication_data_path = storage.get_path(
            "intermediate/growth_lab_publications.csv"
        )

    # Check if publication data exists
    if not publication_data_path.exists():
        logger.error(f"Publication data not found at {publication_data_path}")
        return []

    # Load publication data
    scraper = GrowthLabScraper()
    publications = scraper.load_from_csv(publication_data_path)

    if not publications:
        logger.error("No publications found in the data file")
        return []

    # Filter to publications with file URLs
    publications_with_files = [p for p in publications if p.file_urls]

    logger.info(
        f"Found {len(publications_with_files)}/{len(publications)} "
        "publications with file URLs"
    )

    results = await downloader.download_publications(
        publications_with_files,
        overwrite=overwrite,
        limit=limit,
        progress_bar=True,
    )

    return results
