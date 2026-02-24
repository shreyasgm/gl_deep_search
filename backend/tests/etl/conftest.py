"""Shared fixtures for ETL tests."""

from pathlib import Path

import pytest
import yaml

from backend.etl.orchestrator import _deep_merge

# Paths to config files
_ETL_DIR = Path(__file__).parents[2] / "etl"
_BASE_CONFIG = _ETL_DIR / "config.yaml"
_DEV_OVERLAY = _ETL_DIR / "config.dev.yaml"


@pytest.fixture
def dev_config_path(tmp_path: Path) -> Path:
    """Return a temp config file with dev overrides merged on top of the base config.

    This gives tests the lightweight dev settings (unstructured PDF backend,
    all-MiniLM-L6-v2 embeddings) without touching the production config files.
    """
    with open(_BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    with open(_DEV_OVERLAY) as f:
        overlay = yaml.safe_load(f)

    merged = _deep_merge(base, overlay)

    config_file = tmp_path / "config_dev_resolved.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False)

    return config_file
