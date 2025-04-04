"""
Google Cloud Storage implementation.
"""

import tempfile
from pathlib import Path

from google.cloud import storage

from backend.storage.base import StorageBase


class GCSStorage(StorageBase):
    """Google Cloud Storage implementation."""

    def __init__(self, bucket_name: str, base_prefix: str = ""):
        """
        Initialize GCS storage with bucket name.

        Args:
            bucket_name: Name of the GCS bucket
            base_prefix: Base prefix within the bucket (optional)
        """
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        self.base_prefix = base_prefix

        # Create temp directory for local operations
        self.temp_dir = Path(tempfile.mkdtemp())

    def _get_blob_name(self, path: str) -> str:
        """Convert path to blob name within base prefix."""
        path = path.lstrip("/")
        if self.base_prefix:
            return f"{self.base_prefix.rstrip('/')}/{path}"
        return path

    def read_text(self, path: str) -> str:
        """Read text content from the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        return blob.download_as_text()

    def write_text(self, path: str, content: str) -> None:
        """Write text content to the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(content, content_type="text/plain")

    def read_binary(self, path: str) -> bytes:
        """Read binary content from the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        return blob.download_as_bytes()

    def write_binary(self, path: str, content: bytes) -> None:
        """Write binary content to the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(content)

    def exists(self, path: str) -> bool:
        """Check if a path exists in the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        return blob.exists()

    def list_files(self, data_type: str, pattern: str | None = None) -> list[Path]:
        """List files of a specific data type, optionally filtered by pattern."""
        # Map data_type to appropriate path prefix
        if data_type == "raw":
            path_prefix = "raw"
        elif data_type == "intermediate":
            path_prefix = "intermediate"
        elif data_type == "processed":
            path_prefix = "processed"
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        prefix = self._get_blob_name(path_prefix)
        if not prefix.endswith("/"):
            prefix += "/"

        blobs = self.bucket.list_blobs(prefix=prefix)

        # Convert to local paths for consistency
        local_paths = []

        for blob in blobs:
            # Skip directories (blobs that end with /)
            if blob.name.endswith("/"):
                continue

            # Get just the filename part
            filename = Path(blob.name).name

            # Create a local path reference
            local_path = self.temp_dir / data_type / filename
            local_paths.append(local_path)

        # Filter by pattern if needed
        if pattern:
            import fnmatch

            return [p for p in local_paths if fnmatch.fnmatch(p.name, pattern)]

        return local_paths

    def copy(self, src_path: str, dst_path: str) -> None:
        """Copy a file from source to destination."""
        src_blob_name = self._get_blob_name(src_path)
        dst_blob_name = self._get_blob_name(dst_path)

        src_blob = self.bucket.blob(src_blob_name)
        self.bucket.copy_blob(src_blob, self.bucket, dst_blob_name)

    def move(self, src_path: str, dst_path: str) -> None:
        """Move a file from source to destination."""
        # In GCS, we copy then delete
        self.copy(src_path, dst_path)
        self.remove(src_path)

    def remove(self, path: str) -> None:
        """Remove a file from the storage."""
        blob_name = self._get_blob_name(path)
        blob = self.bucket.blob(blob_name)
        blob.delete()

    def get_full_path(self, path: str) -> str:
        """Get full path in GCS."""
        blob_name = self._get_blob_name(path)
        return f"gs://{self.bucket.name}/{blob_name}"

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up temporary directory on exit."""
        import shutil

        shutil.rmtree(self.temp_dir)
