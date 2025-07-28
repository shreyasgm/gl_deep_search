"""Configure pytest for the test suite"""

import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Enable asyncio for all async tests
pytest.register_assert_rewrite("pytest_asyncio")


# Register custom markers to avoid warnings
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests that may require real data",
    )


# Fixtures for API service tests
@pytest.fixture(scope="session")
def test_client():
    """
    Create a test client for the FastAPI application.

    Returns:
        TestClient: A client for testing the application.
    """
    from backend.service.main import app

    client = TestClient(app)
    return client


@pytest.fixture(scope="session")
def db_path():
    """
    Get the path to the test database.

    Returns:
        Path: Path to the test database.
    """
    # Use the actual DB for integration tests
    # In a real setup, you might want to create a test-specific database
    project_root = Path(__file__).parent.parent.parent
    return project_root / "data" / "processed" / "publication_tracking.db"


@pytest.fixture(scope="session")
def db_connection(db_path):
    """
    Create a connection to the test database.

    Args:
        db_path: Path to the test database.

    Returns:
        sqlite3.Connection: Connection to the SQLite database.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an asyncio event loop for the test session.

    Returns:
        asyncio.AbstractEventLoop: Event loop for tests.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
