"""
Integration tests for publication status API endpoints.
"""

from http import HTTPStatus

import pytest


def test_health_endpoint(test_client):
    """
    Test the health check endpoint.

    Args:
        test_client: FastAPI test client.
    """
    response = test_client.get("/api/health")

    # Check response status code and content
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"status": "healthy"}


def test_get_publications_status_default(test_client):
    """
    Test the publications status endpoint with default parameters.

    Args:
        test_client: FastAPI test client.
    """
    response = test_client.get("/api/v1/publications/status")

    # Check response status code and structure
    assert response.status_code == HTTPStatus.OK
    data = response.json()

    # Verify response structure
    assert "total" in data
    assert "page" in data
    assert "page_size" in data
    assert "items" in data

    # Check default pagination values
    assert data["page"] == 1
    assert data["page_size"] == 10
    assert isinstance(data["items"], list)

    # Ensure we have publications in the response
    assert len(data["items"]) > 0

    # Check structure of the first publication
    first_pub = data["items"][0]
    assert "publication_id" in first_pub
    assert "title" in first_pub
    assert "download_status" in first_pub
    assert "processing_status" in first_pub
    assert "embedding_status" in first_pub
    assert "ingestion_status" in first_pub


def test_get_publications_status_with_filters(test_client):
    """
    Test the publications status endpoint with filters.

    Args:
        test_client: FastAPI test client.
    """
    # Test with download_status filter
    response = test_client.get(
        "/api/v1/publications/status", params={"download_status": "Downloaded"}
    )

    assert response.status_code == HTTPStatus.OK
    data = response.json()

    # Check that all returned items have the requested status
    assert all(item["download_status"] == "Downloaded" for item in data["items"])


def test_get_publications_status_pagination(test_client):
    """
    Test pagination in the publications status endpoint.

    Args:
        test_client: FastAPI test client.
    """
    # Request first page with 5 items
    response1 = test_client.get(
        "/api/v1/publications/status", params={"page": 1, "page_size": 5}
    )
    assert response1.status_code == HTTPStatus.OK
    data1 = response1.json()
    assert len(data1["items"]) == 5

    # Request second page with 5 items
    response2 = test_client.get(
        "/api/v1/publications/status", params={"page": 2, "page_size": 5}
    )
    assert response2.status_code == HTTPStatus.OK
    data2 = response2.json()

    # Check that we get different items on different pages
    if data1["total"] > 5:  # Only check if we have more than 5 items in total
        first_page_ids = [item["publication_id"] for item in data1["items"]]
        second_page_ids = [item["publication_id"] for item in data2["items"]]
        # Ensure no overlap between pages
        assert not any(pid in second_page_ids for pid in first_page_ids)


def test_get_publication_by_id(test_client, db_connection):
    """
    Test retrieving a single publication by ID.

    Args:
        test_client: FastAPI test client.
        db_connection: SQLite database connection.
    """
    # First, get an actual publication ID from the database
    cursor = db_connection.cursor()
    cursor.execute("SELECT publication_id FROM publication_tracking LIMIT 1")
    row = cursor.fetchone()

    if row:
        publication_id = row["publication_id"]

        # Test retrieving the publication
        response = test_client.get(f"/api/v1/publications/status/{publication_id}")

        assert response.status_code == HTTPStatus.OK
        data = response.json()

        # Check that we got the requested publication
        assert data["publication_id"] == publication_id
    else:
        pytest.skip("No publications in the database to test with")


def test_get_nonexistent_publication(test_client):
    """
    Test retrieving a publication that doesn't exist.

    Args:
        test_client: FastAPI test client.
    """
    response = test_client.get("/api/v1/publications/status/non_existent_id")

    # Should return 404 Not Found
    assert response.status_code == HTTPStatus.NOT_FOUND


def test_get_publications_with_sorting(test_client):
    """
    Test sorting in the publications status endpoint.

    Args:
        test_client: FastAPI test client.
    """
    # Test ascending sort by year
    response_asc = test_client.get(
        "/api/v1/publications/status", params={"sort_by": "year", "sort_order": "asc"}
    )
    assert response_asc.status_code == HTTPStatus.OK
    data_asc = response_asc.json()

    # Get the years from the response
    if data_asc["items"]:
        years_asc = [item["year"] for item in data_asc["items"]]
        # Check that years are in ascending order
        assert years_asc == sorted(years_asc)

    # Test descending sort by year
    response_desc = test_client.get(
        "/api/v1/publications/status", params={"sort_by": "year", "sort_order": "desc"}
    )
    assert response_desc.status_code == HTTPStatus.OK
    data_desc = response_desc.json()

    # Get the years from the response
    if data_desc["items"]:
        years_desc = [item["year"] for item in data_desc["items"]]
        # Check that years are in descending order
        assert years_desc == sorted(years_desc, reverse=True)
