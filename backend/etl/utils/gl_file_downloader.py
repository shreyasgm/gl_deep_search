"""
Growth Lab file downloader module for Deep Search.

Handles asynchronous downloading of files from Growth Lab publication URLs, with
features like retrying, rate limiting, and validation.

Uses curl_cffi for HTTP requests to bypass Cloudflare TLS fingerprinting.
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
import tqdm.asyncio
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.errors import RequestsError

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

# Browser to impersonate for Cloudflare bypass
BROWSER_IMPERSONATE = "chrome120"


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
        self._session: AsyncSession | None = None

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
        }

    async def _get_session(self) -> AsyncSession:
        """Get or create a curl_cffi session with browser impersonation."""
        if self._session is None:
            # Default headers for file downloads
            headers = {
                "Accept": (
                    "application/pdf,application/octet-stream,"
                    "application/msword,application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document,*/*"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }

            self._session = AsyncSession(
                impersonate=BROWSER_IMPERSONATE,
                timeout=60,
                headers=headers,
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

    def _correct_file_extension(self, file_path: Path, content_type: str) -> Path:
        """
        Rename file to match actual Content-Type if extension is wrong.

        Many Growth Lab publication URLs are DOI links or landing pages that
        don't end in .pdf, so the initial filename gets a .bin extension.
        After download, we know the real Content-Type from the server response
        and can fix the extension.

        Args:
            file_path: Current path of the downloaded file
            content_type: Content-Type header from the server response

        Returns:
            The (possibly renamed) file path
        """
        if not content_type or not file_path.exists():
            return file_path

        # Map Content-Type to expected extension
        _docx = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        _xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ct_to_ext = {
            "application/pdf": ".pdf",
            "application/msword": ".doc",
            _docx: ".docx",
            "application/vnd.ms-excel": ".xls",
            _xlsx: ".xlsx",
        }

        base_ct = content_type.split(";")[0].strip().lower()
        expected_ext = ct_to_ext.get(base_ct)

        if expected_ext and file_path.suffix.lower() != expected_ext:
            new_path = file_path.with_suffix(expected_ext)
            file_path.rename(new_path)
            logger.info(
                f"Renamed {file_path.name} -> {new_path.name} (Content-Type: {base_ct})"
            )
            return new_path

        return file_path

    async def _download_file_impl(
        self,
        session: AsyncSession,
        url: str,
        destination: Path,
        referer: str | None = None,
        resume: bool = True,
    ) -> DownloadResult:
        """
        Implementation of file download with resume capability.

        Uses curl_cffi with browser impersonation to bypass Cloudflare TLS
        fingerprinting.

        Args:
            session: curl_cffi async session to use
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
            response = await session.get(url, headers=headers, stream=True, timeout=60)

            # Handle response
            if response.status_code == 416:  # Range Not Satisfiable
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
                    content_type=response.headers.get("Content-Type"),
                    cached=True,
                    validation_info=validation_result,
                )

            elif response.status_code == 206:  # Partial Content (range requests)
                # Resume download
                mode = "ab"  # Append binary

            elif response.status_code == 200:  # OK
                # New download or server doesn't support range requests
                file_size = 0  # Start from beginning
                mode = "wb"  # Write binary

            else:
                # Other status codes are errors
                error_msg = f"HTTP error {response.status_code} when downloading {url}"
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

            # Download the file using streaming
            try:
                async with aiofiles.open(destination, mode) as f:
                    async for chunk in response.aiter_content():
                        await f.write(chunk)
            except Exception as e:
                logger.error(f"Error writing to file {destination}: {e}")
                return DownloadResult(
                    url=url,
                    success=False,
                    file_path=destination,
                    error=f"Failed to write file: {str(e)}",
                )

            # Fix file extension based on actual Content-Type from server
            destination = self._correct_file_extension(destination, content_type)

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
                    f"Downloaded file validation failed for {url}: {validation_result}"
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

            # Upload to remote storage (no-op for local)
            try:
                rel = str(destination.relative_to(self.storage.get_path("")))
                self.storage.upload(rel)
            except (ValueError, TypeError, Exception) as upload_err:
                logger.debug(f"Could not upload after download: {upload_err}")

            return DownloadResult(
                url=url,
                success=True,
                file_path=destination,
                file_size=final_size,
                content_type=content_type,
                validation_info=validation_result,
            )

        except RequestsError as e:
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

        # Check if file already exists (use storage for cloud-aware check)
        # Compute storage-relative path for exists() check
        try:
            relative_path = str(destination.relative_to(self.storage.get_path("")))
            file_exists_in_storage = self.storage.exists(relative_path)
        except (ValueError, TypeError):
            file_exists_in_storage = destination.exists()

        # When resume=True, still skip if the file looks complete (>min size)
        if file_exists_in_storage and not overwrite:
            # Download locally if needed (for validation / size check)
            if not destination.exists():
                try:
                    self.storage.download(relative_path)
                except Exception:
                    pass

            if destination.exists():
                file_size = destination.stat().st_size
                if not resume or file_size >= self.min_file_size:
                    logger.info(
                        f"File {destination} already exists "
                        f"({file_size} bytes), skipping download"
                    )

                    content_type = (
                        mimetypes.guess_type(str(destination))[0]
                        or "application/octet-stream"
                    )
                    validation_result = await self._validate_downloaded_file(
                        destination, expected_content_type=content_type
                    )

                    self.download_stats["total_attempted"] += 1
                    self.download_stats["cached"] += 1

                    return DownloadResult(
                        url=url,
                        success=validation_result["is_valid"],
                        file_path=destination,
                        file_size=file_size,
                        content_type=content_type,
                        cached=True,
                        validation_info=validation_result,
                    )

        # Get curl_cffi session
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
                retry_on=(RequestsError, TimeoutError, asyncio.TimeoutError),
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

        # Track download results per publication to determine final status
        publication_results: dict[str, list[dict[str, Any]]] = {}

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

                        # Track results per publication
                        if pub.paper_id not in publication_results:
                            publication_results[pub.paper_id] = []
                        publication_results[pub.paper_id].append(pub_result)

                        # Update progress
                        pbar.update(1)

                        # Success/failure message
                        if result.success:
                            status = "cached" if result.cached else "downloaded"
                            pbar.set_postfix_str(f"Last: {status}")
                        else:
                            # Ensure error is a string for display
                            error_display = (
                                str(result.error)
                                if result.error is not None
                                else "Download failed"
                            )
                            pbar.set_postfix_str(f"Last: failed - {error_display}")

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

                        # Track results per publication
                        if pub.paper_id not in publication_results:
                            publication_results[pub.paper_id] = []
                        publication_results[pub.paper_id].append(pub_result)

                        pbar.update(1)

                    # Rate limiting delay
                    await asyncio.sleep(self.download_delay)

                # Update download status per publication after all files processed
                for pub_id, file_results in publication_results.items():
                    # Check if all files succeeded
                    all_succeeded = all(r.get("success", False) for r in file_results)
                    any_failed = any(not r.get("success", False) for r in file_results)

                    if all_succeeded:
                        # All files downloaded successfully
                        self.publication_tracker.update_download_status(
                            pub_id, DownloadStatus.DOWNLOADED
                        )
                    elif any_failed:
                        # At least one file failed - collect error messages
                        error_messages = [
                            str(r.get("error", "Unknown error"))
                            for r in file_results
                            if not r.get("success", False) and r.get("error")
                        ]
                        # Ensure error is a string
                        error_msg = (
                            "; ".join(error_messages)
                            if error_messages
                            else "One or more files failed to download"
                        )
                        self.publication_tracker.update_download_status(
                            pub_id, DownloadStatus.FAILED, error=error_msg
                        )

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
