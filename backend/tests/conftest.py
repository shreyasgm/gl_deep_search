"""Configure pytest for the test suite"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Enable asyncio for all async tests
pytest.register_assert_rewrite("pytest_asyncio")

# Load .env so integration tests can find API keys
_ENV_PATH = Path(__file__).parents[1] / "etl" / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# Env vars required by integration tests that hit external APIs
REQUIRED_INTEGRATION_ENV_VARS = [
    "EMBEDDING_API_KEY",  # OpenRouter API key for Qwen3 embeddings
]


# Register custom markers to avoid warnings
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests that may require real data",
    )


@pytest.fixture
def require_api_keys():
    """Fixture that skips the test if required API keys are missing.

    Use in integration tests that call external APIs:

        def test_something(self, require_api_keys):
            ...
    """
    missing = [v for v in REQUIRED_INTEGRATION_ENV_VARS if not os.environ.get(v)]
    if missing:
        pytest.skip(f"Missing required env vars: {', '.join(missing)}")
