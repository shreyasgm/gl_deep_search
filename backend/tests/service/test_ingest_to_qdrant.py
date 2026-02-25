"""Tests for the Qdrant ingestion script."""

import json
import uuid
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.service.scripts.ingest_to_qdrant import (
    build_points_for_document,
    deterministic_uuid,
)

# ---------------------------------------------------------------------------
# deterministic_uuid() tests
# ---------------------------------------------------------------------------


class TestDeterministicUuid:
    """Tests for the deterministic_uuid() helper."""

    def test_same_input_same_output(self):
        result1 = deterministic_uuid("chunk-abc-123")
        result2 = deterministic_uuid("chunk-abc-123")
        assert result1 == result2

    def test_different_inputs_different_outputs(self):
        result1 = deterministic_uuid("chunk-A")
        result2 = deterministic_uuid("chunk-B")
        assert result1 != result2

    def test_output_is_valid_uuid(self):
        result = deterministic_uuid("some-chunk-id")
        # Should not raise ValueError if it's a valid UUID
        parsed = uuid.UUID(result)
        assert str(parsed) == result

    def test_uuid5_version(self):
        result = deterministic_uuid("test-chunk")
        parsed = uuid.UUID(result)
        assert parsed.version == 5


# ---------------------------------------------------------------------------
# build_points_for_document() tests
# ---------------------------------------------------------------------------


class TestBuildPointsForDocument:
    """Tests for build_points_for_document() with fixture data."""

    @pytest.fixture
    def doc_dir(self, tmp_path):
        """Create a temporary document directory with parquet + metadata.json."""
        doc = tmp_path / "doc-test-001"
        doc.mkdir()

        # Create embeddings parquet with 2 chunks
        chunk_ids = ["chunk-1", "chunk-2"]
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        embedding_type = pa.list_(pa.float32())
        table = pa.table(
            {
                "chunk_id": pa.array(chunk_ids, type=pa.string()),
                "embedding": pa.array(embeddings, type=embedding_type),
            }
        )
        pq.write_table(table, doc / "embeddings.parquet")

        # Create metadata.json
        metadata = {
            "document_id": "doc-test-001",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "text_content": "Economic growth is important.",
                    "page_numbers": [1, 2],
                    "section_title": "Introduction",
                    "chunk_index": 0,
                    "token_count": 50,
                },
                {
                    "chunk_id": "chunk-2",
                    "text_content": "Policy implications are vast.",
                    "page_numbers": [3],
                    "section_title": "Conclusion",
                    "chunk_index": 1,
                    "token_count": 40,
                },
            ],
        }
        with open(doc / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        return doc

    @pytest.fixture
    def mock_pub(self):
        """Mock PublicationTracking record."""
        pub = MagicMock()
        pub.title = "Test Publication Title"
        pub.authors = ["Author One", "Author Two"]
        pub.year = 2023
        pub.abstract = "This is an abstract."
        pub.source_url = "http://example.com/paper"
        return pub

    @pytest.fixture
    def mock_sparse_model(self):
        """Mock sparse text embedding model."""
        model = MagicMock()
        # Return sparse vectors for 2 chunks
        sparse_vec_1 = MagicMock()
        sparse_vec_1.indices = np.array([0, 5, 10])
        sparse_vec_1.values = np.array([0.5, 0.3, 0.1])

        sparse_vec_2 = MagicMock()
        sparse_vec_2.indices = np.array([1, 7])
        sparse_vec_2.values = np.array([0.4, 0.2])

        model.embed.return_value = [sparse_vec_1, sparse_vec_2]
        return model

    def test_builds_correct_number_of_points(
        self, doc_dir, mock_pub, mock_sparse_model
    ):
        points = build_points_for_document(doc_dir, mock_pub, mock_sparse_model)
        assert len(points) == 2

    def test_point_payload_contains_expected_fields(
        self, doc_dir, mock_pub, mock_sparse_model
    ):
        points = build_points_for_document(doc_dir, mock_pub, mock_sparse_model)
        payload = points[0].payload

        assert payload["chunk_id"] == "chunk-1"
        assert payload["document_id"] == "doc-test-001"
        assert payload["text_content"] == "Economic growth is important."
        assert payload["page_numbers"] == [1, 2]
        assert payload["section_title"] == "Introduction"
        assert payload["chunk_index"] == 0
        assert payload["token_count"] == 50

        # Publication metadata
        assert payload["document_title"] == "Test Publication Title"
        assert payload["document_authors"] == ["Author One", "Author Two"]
        assert payload["document_year"] == 2023
        assert payload["document_abstract"] == "This is an abstract."
        assert payload["document_url"] == "http://example.com/paper"

    def test_point_has_dense_and_sparse_vectors(
        self, doc_dir, mock_pub, mock_sparse_model
    ):
        points = build_points_for_document(doc_dir, mock_pub, mock_sparse_model)
        point = points[0]

        # Vector should be a dict with "dense" and "bm25" keys
        assert "dense" in point.vector
        assert "bm25" in point.vector
        assert len(point.vector["dense"]) == 1024

        # BM25 sparse vector
        bm25 = point.vector["bm25"]
        assert hasattr(bm25, "indices")
        assert hasattr(bm25, "values")

    def test_point_id_is_deterministic_uuid(self, doc_dir, mock_pub, mock_sparse_model):
        points = build_points_for_document(doc_dir, mock_pub, mock_sparse_model)
        expected_id = deterministic_uuid("chunk-1")
        assert points[0].id == expected_id

    def test_no_pub_metadata_sets_none_defaults(self, doc_dir, mock_sparse_model):
        """When pub is None, document metadata fields should be None/empty."""
        points = build_points_for_document(doc_dir, None, mock_sparse_model)
        payload = points[0].payload

        assert payload["document_title"] is None
        assert payload["document_authors"] == []
        assert payload["document_year"] is None
        assert payload["document_abstract"] is None
        assert payload["document_url"] is None

    def test_sparse_model_called_with_correct_texts(
        self, doc_dir, mock_pub, mock_sparse_model
    ):
        build_points_for_document(doc_dir, mock_pub, mock_sparse_model)
        mock_sparse_model.embed.assert_called_once()
        texts_arg = mock_sparse_model.embed.call_args[0][0]
        assert texts_arg == [
            "Economic growth is important.",
            "Policy implications are vast.",
        ]
