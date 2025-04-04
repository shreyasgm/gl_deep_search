"""
Local filesystem storage implementation.
"""

from pathlib import Path

from backend.storage.base import StorageBase


class LocalStorage(StorageBase):
    """Local filesystem storage implementation."""

    def __init__(self, base_path: Path):
        """
        Initialize local storage with base path.

        Args:
            base_path: Base path for storage
        """
        self.base_path = base_path

        # Ensure base directory exists
        self.ensure_dir(self.base_path)

    def get_path(self, filename: str) -> Path:
        """
        Get the appropriate path for a given filename.

        Args:
            filename: Name of the file

        Returns:
            Path to the file
        """
        return self.base_path / filename

    def ensure_dir(self, path: Path) -> None:
        """
        Ensure a directory exists.

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
        if not self.base_path.exists():
            return []

        if pattern:
            return list(self.base_path.glob(pattern))
        else:
            return [p for p in self.base_path.iterdir() if p.is_file()]
