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

        # Create standard subdirectories
        self.raw_path = self.base_path / "raw"
        self.intermediate_path = self.base_path / "intermediate"
        self.processed_path = self.base_path / "processed"

        # Ensure directories exist
        for path in [
            self.base_path,
            self.raw_path,
            self.intermediate_path,
            self.processed_path,
        ]:
            self.ensure_dir(path)

    def get_path(self, data_type: str, filename: str) -> Path:
        """
        Get the appropriate path for a given data type and filename.

        Args:
            data_type: Type of data (raw, intermediate, processed)
            filename: Name of the file

        Returns:
            Path to the file
        """
        if data_type == "raw":
            return self.raw_path / filename
        elif data_type == "intermediate":
            return self.intermediate_path / filename
        elif data_type == "processed":
            return self.processed_path / filename
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def ensure_dir(self, path: Path) -> None:
        """
        Ensure a directory exists.

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
        base_dir = self.get_path(data_type, "")

        if not base_dir.exists():
            return []

        if pattern:
            return list(base_dir.glob(pattern))
        else:
            return [p for p in base_dir.iterdir() if p.is_file()]
