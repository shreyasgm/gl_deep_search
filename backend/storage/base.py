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
