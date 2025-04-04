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

        # Define data type paths
        self.data_types = {
            "raw": f"{self.base_prefix}/raw" if self.base_prefix else "raw",
            "intermediate": f"{self.base_prefix}/intermediate"
            if self.base_prefix
            else "intermediate",
            "processed": f"{self.base_prefix}/processed"
            if self.base_prefix
            else "processed",
        }

    def get_path(self, data_type: str, filename: str) -> Path:
        """
        Get the appropriate path for a given data type and filename.

        Note: For cloud storage, we return a special Path that represents
        the cloud location but can be used for temporary local operations.

        Args:
            data_type: Type of data (raw, intermediate, processed)
            filename: Name of the file

        Returns:
            Path representing the cloud location
        """
        if data_type not in self.data_types:
            raise ValueError(f"Unknown data type: {data_type}")

        # Return temp path for local operations if needed
        _ = self.data_types[data_type]  # Validate data_type exists

        # Return a temporary local path for operations
        # Note: This doesn't actually download anything yet
        local_path = self.temp_dir / data_type
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

    def list_files(self, data_type: str, pattern: str | None = None) -> list[Path]:
        """
        List files of a specific data type, optionally filtered by pattern.

        Args:
            data_type: Type of data (raw, intermediate, processed)
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths
        """
        if data_type not in self.data_types:
            raise ValueError(f"Unknown data type: {data_type}")

        prefix = self.data_types[data_type]
        if not prefix.endswith("/"):
            prefix += "/"

        blobs = list(self.bucket.list_blobs(prefix=prefix))

        # Convert to local paths for consistency
        local_paths = []
        prefix_len = len(prefix)

        for blob in blobs:
            # Skip directories (represented as blobs ending with /)
            if blob.name.endswith("/"):
                continue

            # Get filename without prefix
            filename = blob.name[prefix_len:]
            if not filename:  # Skip the directory blob itself
                continue

            # Create local path reference
            local_path = self.temp_dir / data_type / filename
            local_paths.append(local_path)

        # Filter by pattern if needed
        if pattern:
            import fnmatch

            return [p for p in local_paths if fnmatch.fnmatch(p.name, pattern)]

        return local_paths
