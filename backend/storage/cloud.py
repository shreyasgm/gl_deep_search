"""
Cloud storage implementation.
"""

import tempfile
from pathlib import Path

from google.cloud import storage

from backend.storage.base import StorageBase


class CloudStorage(StorageBase):
    """Cloud Storage implementation."""

    def __init__(self, bucket_name: str, base_prefix: str = ""):
        """
        Initialize cloud storage with bucket name.

        Args:
            bucket_name: Name of the cloud bucket
            base_prefix: Base prefix within the bucket (optional)
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.base_prefix = base_prefix.rstrip("/") if base_prefix else ""

        # Create temp directory for local operations
        self.temp_dir = Path(tempfile.mkdtemp())

    def get_path(self, filename: str) -> Path:
        """
        Get the appropriate path for a given filename.

        Note: For cloud storage, we return a special Path that represents
        the cloud location but can be used for temporary local operations.

        Args:
            filename: Name of the file

        Returns:
            Path representing the cloud location
        """
        # Return a temporary local path for operations
        # Note: This doesn't actually download anything yet
        local_path = self.temp_dir
        self.ensure_dir(local_path)
        return local_path / filename

    def ensure_dir(self, path: Path) -> None:
        """
        Ensure a directory exists.

        For cloud storage, this creates the local temp directory.

        Args:
            path: Directory path to ensure
        """
        path.mkdir(parents=True, exist_ok=True)

    def list_files(self, pattern: str | None = None) -> list[Path]:
        """
        List files, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths
        """
        prefix = self.base_prefix
        if not prefix.endswith("/") and prefix:
            prefix += "/"

        blobs = list(self.bucket.list_blobs(prefix=prefix))

        # Convert to local paths for consistency
        local_paths = []
        prefix_len = len(prefix) if prefix else 0

        for blob in blobs:
            # Skip directories (represented as blobs ending with /)
            if blob.name.endswith("/"):
                continue

            # Get filename without prefix
            filename = blob.name[prefix_len:] if prefix_len > 0 else blob.name
            if not filename:  # Skip the directory blob itself
                continue

            # Create local path reference
            local_path = self.temp_dir / filename
            local_paths.append(local_path)

        # Filter by pattern if needed
        if pattern:
            import fnmatch

            return [p for p in local_paths if fnmatch.fnmatch(p.name, pattern)]

        return local_paths
