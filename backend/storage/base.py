"""
Base storage module providing abstract interfaces for data storage.
"""

import abc
from pathlib import Path


class StorageBase(abc.ABC):
    """Abstract base class for storage implementations."""

    @abc.abstractmethod
    def get_path(self, data_type: str, filename: str) -> Path:
        """
        Get the appropriate path for a given data type and filename.

        Args:
            data_type: Type of data (raw, intermediate, processed)
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
    def list_files(self, data_type: str, pattern: str | None = None) -> list[Path]:
        """
        List files of a specific data type, optionally filtered by pattern.

        Args:
            data_type: Type of data (raw, intermediate, processed)
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths
        """
        pass
