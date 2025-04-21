"""Configure pytest for the test suite"""

import pytest

# Enable asyncio for all async tests
pytest.register_assert_rewrite("pytest_asyncio")


# Register custom markers to avoid warnings
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests that may require real data",
    )
