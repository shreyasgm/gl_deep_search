"""
Factory module for creating storage instances based on configuration.
"""

import os
from pathlib import Path

import yaml

from backend.storage.base import StorageBase
from backend.storage.cloud import CloudStorage
from backend.storage.local import LocalStorage


class StorageFactory:
    """Factory for creating storage instances."""

    @staticmethod
    def load_config(config_path: str | Path | None = None) -> dict:
        """Load configuration from YAML file."""
        if not config_path:
            config_path = Path(__file__).parent.parent / "etl" / "config.yaml"

        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            import logging

            logging.warning(f"Error loading config: {e}. Using defaults.")
            return {
                "runtime": {
                    "detect_automatically": True,
                    "local_storage_path": "data/",
                    "gcs_bucket": "growth-lab-deep-search",
                }
            }

    @staticmethod
    def create_storage(
        config_path: str | Path | None = None,
        storage_type: str | None = None,
    ) -> StorageBase:
        """
        Create storage instance based on configuration or environment.

        Args:
            config_path: Path to configuration YAML
            storage_type: Force specific storage type (local, cloud)

        Returns:
            Storage instance
        """
        config = StorageFactory.load_config(config_path)
        runtime_config = config.get("runtime", {})

        # Auto-detect storage type if not specified
        if not storage_type:
            # Auto-detect environment if configured
            if runtime_config.get("detect_automatically", True):
                # Check for cloud environment (GCP, AWS, etc.)
                if os.environ.get("CLOUD_ENVIRONMENT"):
                    storage_type = "cloud"
                else:
                    storage_type = "local"
            else:
                storage_type = config.get("storage", {}).get("type", "local")

        # Create appropriate storage instance
        if storage_type == "cloud":
            bucket_name = runtime_config.get("gcs_bucket", "growth-lab-deep-search")
            return CloudStorage(bucket_name)
        else:
            # Default to local storage
            base_path = runtime_config.get("local_storage_path", "data/")

            # Convert to absolute path if relative
            base_path = Path(base_path)
            if not base_path.is_absolute():
                # Use project root as base
                project_root = Path(__file__).parent.parent.parent
                base_path = project_root / base_path

            return LocalStorage(base_path)


# Create a global function for easy access
def get_storage(
    config_path: str | Path | None = None, storage_type: str | None = None
) -> StorageBase:
    """
    Get configured storage instance.

    Args:
        config_path: Optional path to config file
        storage_type: Optional storage type override

    Returns:
        Configured storage instance
    """
    return StorageFactory.create_storage(config_path, storage_type)
