"""
Cloud storage implementation using Google Cloud Storage.

Provides a local cache directory so that ETL components can continue
to use normal ``Path`` / ``open()`` operations, while ``exists()``
checks the remote bucket and ``download()`` / ``upload()`` sync
individual files between the local cache and GCS.
"""

import fnmatch
import logging
import tempfile
from pathlib import Path

from google.cloud import storage

from backend.storage.base import StorageBase

logger = logging.getLogger(__name__)


class CloudStorage(StorageBase):
    """Google Cloud Storage implementation with local cache."""

    def __init__(self, bucket_name: str, base_prefix: str = ""):
        """
        Initialize cloud storage.

        Args:
            bucket_name: GCS bucket name
            base_prefix: Optional prefix within the bucket
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.base_prefix = base_prefix.rstrip("/") if base_prefix else ""

        # Local cache directory for working with files
        self.local_cache = Path(tempfile.mkdtemp())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _blob_name(self, filename: str) -> str:
        """Build the full blob name from a storage-relative filename."""
        if self.base_prefix:
            return f"{self.base_prefix}/{filename}"
        return filename

    # ------------------------------------------------------------------
    # StorageBase interface
    # ------------------------------------------------------------------

    def get_path(self, filename: str) -> Path:
        """Return a local cache path for *filename*.

        The file may not exist on disk yet — call ``download()`` first
        if you need the actual content.
        """
        return self.local_cache / filename

    def ensure_dir(self, path: Path) -> None:
        """Ensure a local directory exists."""
        path.mkdir(parents=True, exist_ok=True)

    def list_files(self, pattern: str | None = None) -> list[Path]:
        """List files in the bucket, returning local-cache ``Path`` objects."""
        filenames = self.glob(pattern or "*")
        return [self.local_cache / fn for fn in filenames]

    # ------------------------------------------------------------------
    # New abstract method implementations
    # ------------------------------------------------------------------

    def exists(self, filename: str) -> bool:
        """Check if *filename* exists in the GCS bucket.

        For directories (prefix paths), checks whether any blob exists
        under that prefix.
        """
        blob_name = self._blob_name(filename)

        # First try exact blob match
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            return True

        # Check as a directory prefix (at least one blob under it)
        prefix = blob_name.rstrip("/") + "/"
        blobs = self.bucket.list_blobs(prefix=prefix, max_results=1)
        return any(True for _ in blobs)

    def download(self, filename: str) -> Path:
        """Download *filename* from GCS to the local cache and return the local path.

        If *filename* refers to a directory prefix, all blobs under it
        are downloaded recursively.  Files already present in the cache
        are skipped.
        """
        blob_name = self._blob_name(filename)
        local_path = self.local_cache / filename

        # Try as a single file first
        blob = self.bucket.blob(blob_name)
        if blob.exists():
            if not local_path.exists():
                local_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(local_path))
                logger.debug(
                    "Downloaded gs://%s/%s -> %s",
                    self.bucket.name,
                    blob_name,
                    local_path,
                )
            return local_path

        # Try as a directory prefix
        prefix = blob_name.rstrip("/") + "/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        if blobs:
            for b in blobs:
                if b.name.endswith("/"):
                    continue
                # Strip base_prefix to get the storage-relative name
                if self.base_prefix:
                    rel = b.name[len(self.base_prefix) + 1 :]
                else:
                    rel = b.name
                dest = self.local_cache / rel
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    b.download_to_filename(str(dest))
                    logger.debug(
                        "Downloaded gs://%s/%s -> %s",
                        self.bucket.name,
                        b.name,
                        dest,
                    )
            return local_path

        logger.warning(f"Nothing found in GCS for: {blob_name}")
        return local_path

    def upload(self, filename: str) -> None:
        """Upload *filename* from the local cache to GCS.

        If *filename* is a directory, all files under it are uploaded
        recursively.
        """
        local_path = self.local_cache / filename

        if not local_path.exists():
            logger.warning(f"Cannot upload — local path does not exist: {local_path}")
            return

        if local_path.is_file():
            blob_name = self._blob_name(filename)
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            logger.debug(
                f"Uploaded {local_path} -> gs://{self.bucket.name}/{blob_name}"
            )

        elif local_path.is_dir():
            for child in local_path.rglob("*"):
                if child.is_file():
                    rel = str(child.relative_to(self.local_cache))
                    blob_name = self._blob_name(rel)
                    blob = self.bucket.blob(blob_name)
                    blob.upload_from_filename(str(child))
                    logger.debug(
                        f"Uploaded {child} -> gs://{self.bucket.name}/{blob_name}"
                    )

    def glob(self, pattern: str) -> list[str]:
        """Glob for files in GCS matching *pattern*.

        Returns storage-relative filenames (not full blob names).
        """
        prefix = self.base_prefix + "/" if self.base_prefix else ""

        # Optimise: extract a static prefix from the pattern to narrow
        # the GCS listing (everything before the first wildcard).
        parts = pattern.split("/")
        static_parts = []
        for p in parts:
            if "*" in p or "?" in p or "[" in p:
                break
            static_parts.append(p)
        listing_prefix = prefix + "/".join(static_parts)
        if static_parts:
            listing_prefix += "/"

        blobs = self.bucket.list_blobs(prefix=listing_prefix)

        results: list[str] = []
        prefix_len = len(prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = blob.name[prefix_len:] if prefix_len else blob.name
            if not rel:
                continue
            if fnmatch.fnmatch(rel, pattern):
                results.append(rel)
        return results
