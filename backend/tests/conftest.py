"""Configure pytest for the test suite"""

import pytest

# Enable asyncio for all async tests
pytest.register_assert_rewrite("pytest_asyncio")
