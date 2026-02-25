"""Smoke tests that verify required environment variables are available.

These run as integration tests so they don't block normal CI, but they
catch misconfigured environments before heavier integration tests fail
with cryptic auth errors.
"""

import os

import pytest


@pytest.mark.integration
class TestRequiredEnvVars:
    """Verify that env vars needed by integration tests are set and non-empty."""

    def test_embedding_api_key_is_set(self):
        """EMBEDDING_API_KEY is required for OpenRouter embedding calls."""
        val = os.environ.get("EMBEDDING_API_KEY")
        assert val and len(val) > 0, (
            "EMBEDDING_API_KEY is not set. "
            "Add it to backend/etl/.env (it should contain your OpenRouter API key)."
        )

    def test_embedding_api_key_looks_valid(self):
        """EMBEDDING_API_KEY should look like a real key, not a placeholder."""
        val = os.environ.get("EMBEDDING_API_KEY", "")
        placeholders = {"", "test-key", "your-key-here", "CHANGEME", "xxx"}
        assert val not in placeholders, (
            f"EMBEDDING_API_KEY appears to be a placeholder: {val!r}"
        )
