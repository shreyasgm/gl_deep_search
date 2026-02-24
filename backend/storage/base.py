"""
Base storage module providing abstract interfaces for data storage.
"""

import abc
from pathlib import Path


class StorageBase(abc.ABC):
    """Abstract base class for storage implementations."""

    @abc.abstractmethod
    def get_path(self, filename: str) -> Path:
        """
        Get the appropriate path for a given filename.

        Args:
            filename: Name of the file

        Returns:
            Path to the file
        """
        pass

    @abc.abstractmethod
    def ensure_dir(self, path: Path) -> None:
        """
        Ensure a directory exists.

        Args:
            path: Directory path to ensure
        """
        pass

    @abc.abstractmethod
    def list_files(self, pattern: str | None = None) -> list[Path]:
        """
        List files, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths
        """
        pass

    @abc.abstractmethod
    def exists(self, filename: str) -> bool:
        """
        Check if a file or directory exists in storage.

        Args:
            filename: Storage-relative path to check

        Returns:
            True if the file/directory exists
        """
        pass

    @abc.abstractmethod
    def download(self, filename: str) -> Path:
        """
        Ensure a file is available locally. Returns the local path.

        For local storage this is a no-op. For cloud storage this
        downloads the file from the remote to a local cache directory.

        Args:
            filename: Storage-relative path to download

        Returns:
            Local filesystem path to the file
        """
        pass

    @abc.abstractmethod
    def upload(self, filename: str) -> None:
        """
        Upload a local file to remote storage.

        For local storage this is a no-op. For cloud storage this
        uploads the file from the local cache to the remote bucket.

        Args:
            filename: Storage-relative path to upload
        """
        pass

    @abc.abstractmethod
    def glob(self, pattern: str) -> list[str]:
        """
        Glob for files in storage.

        Args:
            pattern: Glob pattern (e.g. "processed/chunks/**/*.json")

        Returns:
            List of storage-relative filenames matching the pattern
        """
        pass
