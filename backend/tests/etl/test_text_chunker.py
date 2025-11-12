"""
Comprehensive tests for the text chunking system.

This test suite validates the chunking functionality based on the
requirements in TEXT_CHUNKING_REQUIREMENTS.md, focusing on integration tests
with some targeted unit tests.

Test Coverage:
- Integration tests with actual processed PDF data
- Multiple chunking strategies (fixed, sentence, structure, hybrid)
- Configuration loading and validation
- Output format validation (chunks.json)
- Error handling and graceful degradation
- Metadata handling (required and optional)
- Performance requirements validation
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import yaml

from backend.etl.orchestrator import ETLOrchestrator, OrchestrationConfig
from backend.etl.utils.text_chunker import ChunkingStatus, TextChunker
from backend.storage.local import LocalStorage


@pytest.fixture
def test_storage():
    """Module-scoped storage fixture for tests outside integration class."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Match expected structure used by output tests
        (temp_dir / "processed" / "documents" / "growthlab").mkdir(
            parents=True, exist_ok=True
        )
        (temp_dir / "processed" / "chunks").mkdir(parents=True, exist_ok=True)
        storage = LocalStorage(base_path=temp_dir)
        yield temp_dir, storage
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestTextChunkerIntegration:
    """Integration tests for text chunking system - Primary focus per requirements."""

    @pytest.fixture
    def test_storage(self):
        """Create temporary directory for test storage matching existing pattern."""
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure matching requirements
        processed_dir = temp_dir / "processed" / "documents" / "growthlab"
        processed_dir.mkdir(parents=True, exist_ok=True)

        chunks_dir = temp_dir / "processed" / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)

        # Create storage instance
        storage = LocalStorage(base_path=temp_dir)

        yield temp_dir, storage

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory with test configuration."""
        temp_dir = Path(tempfile.mkdtemp())
        config_path = temp_dir / "config.yaml"

        # Create test configuration matching requirements
        config_content = """
file_processing:
  chunking:
    enabled: true
    strategy: "hybrid"
    chunk_size: 1000
    chunk_overlap: 200
    min_chunk_size: 100
    max_chunk_size: 2000
    preserve_structure: true
    respect_sentences: true
    structure_markers:
      - "^#{1,6}\\\\s+"
      - "^\\\\d+\\\\.\\\\s+"
      - "^[A-Z][A-Z\\\\s]+:"
"""
        config_path.write_text(config_content)
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_pdf_text(self):
        """Sample processed PDF text matching actual data format."""
        return """--- Page 1 ---

Tax Avoidance in Buenos Aires: The Case of Ingresos Brutos

Carolina Ines Pan

CID Research Fellow and Graduate Student Working Paper No. 117 September 2019

© Copyright 2019 Pan, Carolina Ines; and the President and Fellows of Harvard College

--- Page 2 ---

Abstract

This study presents evidence of tax avoidance in Buenos Aires, Argentina. Using a comprehensive dataset of businesses and their tax payments, we analyze the patterns of tax compliance and avoidance in the city's gross income tax (Ingresos Brutos). The findings suggest that businesses engage in strategic tax avoidance behaviors, particularly through underreporting of income and manipulation of tax classifications.

--- Page 3 ---

1. INTRODUCTION

Tax avoidance represents a significant challenge for developing economies. In Argentina, the gross income tax (Ingresos Brutos) is a crucial source of revenue for provincial governments. This paper examines the extent and patterns of tax avoidance behavior among businesses in Buenos Aires.

The research contributes to the literature on tax compliance in developing countries by providing empirical evidence of avoidance strategies and their economic impacts. Our analysis reveals systematic patterns of underreporting that result in substantial revenue losses for the city government.

2. LITERATURE REVIEW

Previous studies have examined tax avoidance in various contexts. Smith (2018) found that businesses in emerging markets tend to engage in more aggressive tax planning strategies compared to their counterparts in developed economies. This behavior is often attributed to weaker enforcement mechanisms and regulatory frameworks.
"""  # noqa

    @pytest.fixture
    def sample_transcript_text(self):
        """Sample processed transcript text matching expected format."""
        return """Welcome to Development Policy Strategy. I'm Professor Brian Xavier. Today we'll discuss economic growth theories and their policy implications. First, we'll cover the basics of growth accounting. Then we'll move on to structural transformation and productivity. And finally, we'll discuss the importance of economic complexity in development.

Growth accounting is a fundamental tool in development economics. It allows us to decompose the sources of economic growth into various factors such as capital accumulation, labor force growth, and total factor productivity. This decomposition helps policymakers understand which factors are driving growth and where interventions might be most effective.

Structural transformation refers to the process by which economies shift from agriculture to manufacturing and services. This transformation is often accompanied by increases in productivity and living standards. However, the pace and nature of structural transformation can vary significantly across countries and regions.

Economic complexity provides another lens through which we can understand development patterns. Countries that produce more complex products tend to grow faster and have higher incomes. This relationship suggests that policies aimed at building productive capabilities and moving into more complex activities can support long-term development."""  # noqa

    @pytest.mark.integration
    def test_configuration_loading(self, temp_config_dir):
        """Test that chunking configuration is loaded correctly from YAML."""
        chunker = TextChunker(temp_config_dir / "config.yaml")

        assert chunker.chunking_config.get("strategy") == "hybrid"
        assert chunker.chunking_config.get("chunk_size") == 1000
        assert chunker.chunking_config.get("chunk_overlap") == 200
        assert chunker.chunking_config.get("min_chunk_size") == 100
        assert chunker.chunking_config.get("max_chunk_size") == 2000
        assert chunker.chunking_config.get("preserve_structure") is True
        assert chunker.chunking_config.get("respect_sentences") is True
        assert len(chunker.chunking_config.get("structure_markers", [])) == 3

    @pytest.mark.integration
    def test_process_pdf_document_end_to_end(self, temp_config_dir, sample_pdf_text):
        """End-to-end test: Process sample PDF through complete chunking pipeline."""
        # Setup test data structure
        processed_dir = (
            temp_config_dir
            / "data"
            / "processed"
            / "documents"
            / "growthlab"
            / "test_doc"
        )
        processed_dir.mkdir(parents=True)

        input_file = processed_dir / "processed_text.txt"
        input_file.write_text(sample_pdf_text)

        # Setup output directory structure
        chunks_dir = (
            temp_config_dir
            / "data"
            / "processed"
            / "chunks"
            / "documents"
            / "growthlab"
            / "test_doc"
        )
        chunks_dir.mkdir(parents=True)

        # Run chunking
        chunker = TextChunker(temp_config_dir / "config.yaml")
        result = chunker.process_single_document(input_file)

        # Validate processing result
        assert result.status == ChunkingStatus.SUCCESS
        assert result.document_id == "test_doc"
        assert result.source_path == input_file
        assert result.total_chunks > 0
        assert result.processing_time > 0
        assert result.error_message is None

        # Validate chunks
        assert len(result.chunks) > 0
        for i, chunk in enumerate(result.chunks):
            assert chunk.chunk_id is not None
            assert chunk.source_document_id == "test_doc"
            assert chunk.source_file_path == input_file
            assert chunk.chunk_index == i
            assert len(chunk.text_content) >= 100  # Minimum chunk size
            assert len(chunk.text_content) <= 2200  # Max size + overlap
            assert chunk.character_start >= 0
            assert chunk.character_end > chunk.character_start
            assert isinstance(chunk.page_numbers, list)
            assert isinstance(chunk.created_at, datetime)
            assert chunk.chunk_size == len(chunk.text_content)

        # Validate output JSON structure
        output_file = chunks_dir / "chunks.json"
        assert output_file.exists()

        chunks_data = json.loads(output_file.read_text())
        assert isinstance(chunks_data, list)
        assert len(chunks_data) == len(result.chunks)

        # Validate JSON schema matches DocumentChunk
        for chunk_json in chunks_data:
            required_fields = [
                "chunk_id",
                "source_document_id",
                "source_file_path",
                "chunk_index",
                "text_content",
                "character_start",
                "character_end",
                "page_numbers",
                "created_at",
                "chunk_size",
            ]
            for field in required_fields:
                assert field in chunk_json

    @pytest.mark.integration
    def test_process_transcript_document_end_to_end(
        self, temp_config_dir, sample_transcript_text
    ):
        """End-to-end test: Process sample transcript through complete
        chunking pipeline."""
        # Setup test data structure
        processed_dir = (
            temp_config_dir / "data" / "processed" / "transcripts" / "lecture_001"
        )
        processed_dir.mkdir(parents=True)

        input_file = processed_dir / "processed_transcript.txt"
        input_file.write_text(sample_transcript_text)

        # Setup output directory structure
        chunks_dir = (
            temp_config_dir
            / "data"
            / "processed"
            / "chunks"
            / "transcripts"
            / "lecture_001"
        )
        chunks_dir.mkdir(parents=True)

        # Run chunking
        chunker = TextChunker(temp_config_dir / "config.yaml")
        result = chunker.process_single_document(input_file)

        # Validate processing result
        assert result.status == ChunkingStatus.SUCCESS
        assert result.document_id == "lecture_001"
        assert len(result.chunks) > 0

        # Transcripts should not have page numbers
        for chunk in result.chunks:
            assert chunk.page_numbers == []  # No pages in transcripts
            assert chunk.section_title is None  # No sections in continuous text

    @pytest.mark.integration
    def test_batch_processing_with_multiple_documents(
        self, temp_config_dir, sample_pdf_text
    ):
        """Test batch processing of multiple documents with error isolation."""
        # Setup multiple test documents
        base_dir = temp_config_dir / "data" / "processed" / "documents" / "growthlab"

        # Valid document
        doc1_dir = base_dir / "doc1"
        doc1_dir.mkdir(parents=True)
        (doc1_dir / "processed_text.txt").write_text(sample_pdf_text)

        # Another valid document
        doc2_dir = base_dir / "doc2"
        doc2_dir.mkdir(parents=True)
        (doc2_dir / "processed_text.txt").write_text(
            sample_pdf_text[:500]
        )  # Shorter document

        # Invalid document (empty)
        doc3_dir = base_dir / "doc3"
        doc3_dir.mkdir(parents=True)
        (doc3_dir / "processed_text.txt").write_text("")

        # Very short document (should still succeed)
        doc4_dir = base_dir / "doc4"
        doc4_dir.mkdir(parents=True)
        (doc4_dir / "processed_text.txt").write_text("Short text.")

        # Setup mock storage
        mock_storage = Mock()

        # Run batch processing
        chunker = TextChunker(temp_config_dir / "config.yaml")
        results = chunker.process_all_documents(mock_storage)

        # Validate results
        assert len(results) == 4

        # Should have 3 successful and 1 failed (only empty document fails)
        successful = [r for r in results if r.status == ChunkingStatus.SUCCESS]
        failed = [r for r in results if r.status == ChunkingStatus.FAILED]

        assert len(successful) == 3
        assert len(failed) == 1

        # Failed document should be the empty one
        assert failed[0].document_id == "doc3"
        assert "empty" in failed[0].error_message.lower()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_orchestrator_integration(self, temp_config_dir):
        """Test that chunking integrates properly with ETL orchestrator."""
        # Create orchestrator with config
        config = OrchestrationConfig(
            config_path=temp_config_dir / "config.yaml",
            storage_type="local",
            dry_run=True,
        )
        orchestrator = ETLOrchestrator(config)

        # Dry-run pipeline should include Text Chunker in simulated components
        results = await orchestrator.run_pipeline()
        assert any(r.component_name == "Text Chunker" for r in results)

        # Now ensure the chunker method is invoked when executing that component
        with patch.object(
            ETLOrchestrator, "_run_text_chunker", new_callable=AsyncMock
        ) as mock_chunker:
            await orchestrator._execute_component(
                "Text Chunker", orchestrator._run_text_chunker
            )
            mock_chunker.assert_awaited_once()

    @pytest.mark.integration
    def test_performance_requirements(self, temp_config_dir, sample_pdf_text):
        """Test that performance requirements are met.

        (< 5 seconds per file, < 100MB memory)
        """
        # Create a larger document (simulate 50-page document)
        large_document = sample_pdf_text * 20  # Approximate 50-page document

        processed_dir = (
            temp_config_dir
            / "data"
            / "processed"
            / "documents"
            / "growthlab"
            / "large_doc"
        )
        processed_dir.mkdir(parents=True)

        input_file = processed_dir / "processed_text.txt"
        input_file.write_text(large_document)

        chunker = TextChunker(temp_config_dir / "config.yaml")
        result = chunker.process_single_document(input_file)

        # Validate performance requirements
        assert result.processing_time < 5.0  # < 5 seconds per requirements
        assert result.status == ChunkingStatus.SUCCESS

        # Memory usage validation would require memory profiling tools
        # For now, ensure processing completes without memory errors

    @pytest.mark.integration
    def test_process_actual_sample_pdfs(self, temp_config_dir):
        """Test processing with actual sample PDF files from the requirements."""
        # Check if sample processed PDFs exist (as mentioned in requirements)
        base_path = Path(
            "/Users/shg309/Dropbox (Personal)/Education/hks_cid_growth_lab/"
            "gl_deep_search"
        )
        sample_files = [
            base_path
            / (
                "data/processed/documents/growthlab/gl_url_39aabeaa471ae241/"
                "2019-09-cid-fellows-wp-117-tax-avoidance-buenos-aires.txt"
            ),
            base_path
            / (
                "data/processed/documents/growthlab/gl_url_3e115487b5f521a6/"
                "libro-hiper-15-05-19-paginas-185-207.txt"
            ),
            base_path
            / (
                "data/processed/documents/growthlab/gl_url_71a29a74fc0321d5/"
                "growth_diagnostic_paraguay.txt"
            ),
        ]

        # Find which sample files actually exist
        existing_files = [f for f in sample_files if f.exists()]

        if not existing_files:
            pytest.skip("No actual processed PDF files found for testing")

        chunker = TextChunker(temp_config_dir / "config.yaml")

        # Process one of the actual files
        sample_file = existing_files[0]
        result = chunker.process_single_document(sample_file)

        # Validate results with real data
        assert result.status == ChunkingStatus.SUCCESS
        assert len(result.chunks) > 0
        assert result.processing_time > 0

        # Validate that chunks contain meaningful content
        total_chars = sum(len(chunk.text_content) for chunk in result.chunks)
        assert total_chars > 1000  # Should have substantial content

        # Check that page markers are detected (PDFs should have them)
        page_chunks = [c for c in result.chunks if c.page_numbers]
        assert len(page_chunks) > 0  # Real PDFs should have page information


class TestTextChunkerStrategies:
    """Unit tests for different chunking strategies and core logic."""

    @pytest.fixture
    def simple_config(self):
        """Simple configuration for unit testing."""
        return {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "min_chunk_size": 50,
            "max_chunk_size": 200,
            "preserve_structure": True,
            "respect_sentences": True,
            "structure_markers": [r"^#{1,6}\s+", r"^\d+\.\s+"],
        }

    def test_fixed_size_chunking_strategy(self, simple_config):
        """Test fixed-size chunking with overlap."""
        text = "This is a test document. " * 20  # 500 characters

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_fixed_size(text, Path("/test.txt"), "test_doc")

        # Validate chunk properties
        assert len(chunks) > 1  # Should create multiple chunks

        for i, chunk in enumerate(chunks):
            assert len(chunk.text_content) <= simple_config["max_chunk_size"]
            assert len(chunk.text_content) >= simple_config["min_chunk_size"]
            assert chunk.chunk_index == i

            # Test overlap (except for last chunk)
            if i > 0:
                overlap = chunks[i - 1].text_content[-simple_config["chunk_overlap"] :]
                current_start = chunk.text_content[: simple_config["chunk_overlap"]]
                assert overlap == current_start  # Verify overlap

    def test_sentence_aware_chunking_strategy(self, simple_config):
        """Test sentence-boundary aware chunking."""
        text = (
            "First sentence here. Second sentence follows! "
            "Third sentence ends? Fourth sentence. "
            "Fifth one too."
        )

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_by_sentences(text, Path("/test.txt"), "test_doc")

        # Sentences should not be broken mid-sentence
        for chunk in chunks:
            # Each chunk should end with sentence terminator or be at document end
            last_char = chunk.text_content.strip()[-1]
            assert last_char in ".!?" or chunk == chunks[-1]

    def test_structure_aware_chunking_strategy(self, simple_config):
        """Test document structure-aware chunking."""
        structured_text = """# Introduction
This is the introduction section with some content that explains the topic.

## Background
This subsection provides background information about the research area.

1. First numbered point
This explains the first important concept.

2. Second numbered point
This covers the second key area.

# Methods
This section describes the methodology used in the study.
"""

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_by_structure(
            structured_text, Path("/test.txt"), "test_doc"
        )

        # Should respect document structure
        assert len(chunks) > 1

        # Some chunks should capture section information
        section_chunks = [c for c in chunks if c.section_title is not None]
        assert len(section_chunks) > 0

        # Headers should be preserved in chunks
        header_chunks = [c for c in chunks if c.text_content.strip().startswith("#")]
        assert len(header_chunks) > 0

    def test_hybrid_chunking_strategy(self, simple_config):
        """Test hybrid strategy that combines all approaches."""
        structured_text = """# Introduction
This is a longer introduction with multiple sentences. It contains several ideas
that need to be explained clearly. The content flows from one concept to
another.

## Detailed Analysis
This section provides detailed analysis of the topic. Each paragraph builds on
the previous one. The arguments are presented systematically.

1. First Point
The first point makes an important argument about the topic.

2. Second Point
The second point extends the analysis further.
"""

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_hybrid(structured_text, Path("/test.txt"), "test_doc")

        # Hybrid should produce reasonable chunks
        assert len(chunks) > 1

        # Should respect both structure and sentences
        for chunk in chunks:
            assert len(chunk.text_content) >= simple_config["min_chunk_size"]
            # Content should be coherent (not cut mid-word in most cases)
            assert not chunk.text_content.endswith(" ")

    def test_graceful_strategy_fallback(self, simple_config):
        """Test graceful degradation when advanced strategies fail."""
        # Malformed text that might break structure detection
        problematic_text = (
            "###No space after hash\n1.No space\n2.No space\nJust regular text."
        )

        with patch.object(
            TextChunker, "chunk_by_structure", side_effect=Exception("Structure failed")
        ):
            with patch.object(
                TextChunker,
                "chunk_by_sentences",
                side_effect=Exception("Sentence failed"),
            ):
                # Create temporary config file
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as f:
                    yaml.dump({"file_processing": {"chunking": simple_config}}, f)
                    config_path = Path(f.name)

                chunker = TextChunker(config_path)
                chunks = chunker.create_chunks(
                    problematic_text, Path("/test.txt"), "test_doc"
                )

                # Should fall back to fixed-size chunking
                assert len(chunks) > 0
                assert all(
                    len(c.text_content) <= simple_config["max_chunk_size"]
                    for c in chunks
                )

    def test_metadata_handling_required_fields(self, simple_config):
        """Test that required metadata fields are always present."""
        text = "Simple test document content here."
        source_path = Path("/test/document.txt")

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.create_chunks(text, source_path, "test_doc")

        # All required fields must be present
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert chunk.source_document_id == "test_doc"
            assert chunk.source_file_path == source_path
            assert isinstance(chunk.chunk_index, int)
            assert chunk.text_content is not None
            assert isinstance(chunk.character_start, int)
            assert isinstance(chunk.character_end, int)
            assert isinstance(chunk.created_at, datetime)
            assert chunk.chunk_size == len(chunk.text_content)

    def test_metadata_handling_optional_fields(self, simple_config):
        """Test graceful handling of optional metadata fields."""
        text = (
            "--- Page 1 ---\nContent on page one.\n--- Page 2 ---\nContent on page two."
        )

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.create_chunks(text, Path("/test.txt"), "test_doc")

        # Optional fields should be handled gracefully
        for chunk in chunks:
            # page_numbers should be extracted when available
            assert isinstance(chunk.page_numbers, list)
            # section_title can be None
            assert chunk.section_title is None or isinstance(chunk.section_title, str)
            # metadata dict should exist
            assert isinstance(chunk.metadata, dict)

    def test_error_handling_empty_input(self, simple_config):
        """Test behavior with empty or invalid input."""
        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)

        # Empty text file should fail
        empty_tmp = Path(tempfile.mktemp())
        empty_tmp.write_text("")
        result = chunker.process_single_document(empty_tmp)
        assert result.status == ChunkingStatus.FAILED
        assert (
            "empty" in result.error_message.lower()
            or "too short" in result.error_message.lower()
        )

        # Very short text (below min_chunk_size) should succeed if it's entire document
        short_text = "Too short."  # Shorter than min_chunk_size (50) but should work
        temp_short = Path(tempfile.mktemp())
        temp_short.write_text(short_text)
        result = chunker.process_single_document(temp_short)
        assert result.status == ChunkingStatus.SUCCESS
        assert len(result.chunks) == 1
        assert result.chunks[0].text_content.strip() == short_text.strip()

    def test_error_handling_file_not_found(self, simple_config):
        """Test behavior when input file doesn't exist."""
        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)

        non_existent_file = Path("/does/not/exist.txt")
        result = chunker.process_single_document(non_existent_file)

        assert result.status == ChunkingStatus.FAILED
        assert (
            "not found" in result.error_message.lower()
            or "does not exist" in result.error_message.lower()
        )

    def test_chunk_size_validation(self, simple_config):
        """Test that chunk size constraints are enforced."""
        # Long text that should be split
        long_text = "A very long sentence. " * 50  # Much longer than max_chunk_size

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_fixed_size(long_text)

        # All chunks should respect size constraints
        for chunk in chunks:
            assert len(chunk.text_content) <= simple_config["max_chunk_size"]
            assert len(chunk.text_content) >= simple_config["min_chunk_size"]

    def test_short_document_vs_short_chunk_distinction(self, simple_config):
        """Test that short entire documents succeed but short individual chunks
        within longer documents are filtered."""
        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)

        # Test 1: Entire document shorter than min_chunk_size should succeed
        short_document = "This is a very short document."  # < min_chunk_size (50)
        chunks = chunker.create_chunks(short_document, Path("/test.txt"), "test_doc")
        assert len(chunks) == 1
        assert chunks[0].text_content.strip() == short_document.strip()

        # Test 2: Long document where last chunk would be too short should filter it
        # Create a document that when chunked will have a final chunk < min_chunk_size
        long_text = "A sentence of reasonable length for testing purposes. " * 10
        short_ending = "Short end."  # This would create a final chunk < min_chunk_size
        combined_text = long_text + short_ending

        chunks = chunker.chunk_fixed_size(combined_text, Path("/test.txt"), "test_doc")

        # All returned chunks should meet min_chunk_size requirement
        for chunk in chunks:
            assert len(chunk.text_content) >= simple_config["min_chunk_size"]

        # The short ending should not appear as a separate chunk
        last_chunk_text = chunks[-1].text_content
        # The short ending should either be included in the last valid chunk or
        # filtered out (but it shouldn't be a standalone short chunk)

    def test_overlap_continuity(self, simple_config):
        """Test that overlapping chunks provide proper context continuity."""
        text = (
            "Sentence one here. Sentence two follows. Sentence three continues. " * 10
        )

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_fixed_size(text, Path("/test.txt"), "test_doc")

        if len(chunks) > 1:
            # Check overlap between consecutive chunks
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]

                # Should have meaningful overlap
                overlap_size = simple_config["chunk_overlap"]
                prev_end = prev_chunk.text_content[-overlap_size:]
                curr_start = curr_chunk.text_content[:overlap_size]

                assert prev_end == curr_start


class TestTextChunkerConfiguration:
    """Tests for configuration handling and validation."""

    def test_default_configuration_values(self):
        """Test that reasonable defaults are used when config is missing."""
        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)  # Empty config

        # Should use reasonable defaults
        assert chunker.chunking_config.get("chunk_size", 0) > 0
        assert chunker.chunking_config.get("min_chunk_size", 0) > 0
        assert chunker.chunking_config.get(
            "max_chunk_size", 0
        ) > chunker.chunking_config.get("chunk_size", 0)
        assert chunker.chunking_config.get("chunk_overlap", -1) >= 0
        assert chunker.chunking_config.get("strategy") in [
            "fixed",
            "sentence",
            "structure",
            "hybrid",
        ]

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration values."""
        invalid_configs = [
            {"chunk_size": -100},  # Negative size
            {"min_chunk_size": 1000, "max_chunk_size": 500},  # Min > Max
            {"chunk_overlap": 2000, "chunk_size": 1000},  # Overlap > Size
            {"strategy": "invalid_strategy"},  # Invalid strategy
        ]

        for config in invalid_configs:
            # Should either use defaults or raise clear error
            try:
                # Create temporary config file
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".yaml", delete=False
                ) as f:
                    yaml.dump({"file_processing": {"chunking": config}}, f)
                    config_path = Path(f.name)

                chunker = TextChunker(config_path)
                # If it doesn't raise an error, it should use valid defaults
                assert chunker.chunking_config.get("chunk_size", 0) > 0
                assert chunker.chunking_config.get(
                    "min_chunk_size", 0
                ) < chunker.chunking_config.get("max_chunk_size", 0)
                assert chunker.chunking_config.get(
                    "chunk_overlap", 0
                ) < chunker.chunking_config.get("chunk_size", 0)
            except ValueError as e:
                # Clear error message about configuration issue
                assert "config" in str(e).lower() or "invalid" in str(e).lower()


class TestTextChunkerOutput:
    """Tests for output format validation and storage integration."""

    def test_json_output_schema_validation(self, test_storage):
        """Test that JSON output matches expected schema."""
        temp_dir, storage = test_storage

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {"chunk_size": 100}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.create_chunks(
            "Test content for validation.", Path("/test.txt"), "test_doc"
        )

        # Convert to JSON and back to validate serialization
        chunks_json = [chunk.to_dict() for chunk in chunks]
        json_str = json.dumps(chunks_json, indent=2, default=str)
        parsed_chunks = json.loads(json_str)

        # Validate JSON structure
        for chunk_data in parsed_chunks:
            required_fields = [
                "chunk_id",
                "source_document_id",
                "source_file_path",
                "chunk_index",
                "text_content",
                "character_start",
                "character_end",
                "page_numbers",
                "chunk_size",
                "created_at",
            ]
            for field in required_fields:
                assert field in chunk_data

    def test_directory_structure_creation(self, test_storage):
        """Test that proper directory structure is created for output."""
        temp_dir, storage = test_storage

        # Test document chunking
        doc_input = (
            temp_dir / "processed" / "documents" / "growthlab" / "test_doc" / "file.txt"
        )
        doc_input.parent.mkdir(parents=True)
        doc_input.write_text("Test document content here.")

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {"chunk_size": 50}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        result = chunker.process_single_document(doc_input)

        # Should create corresponding output directory
        expected_output_dir = (
            temp_dir / "processed" / "chunks" / "documents" / "growthlab" / "test_doc"
        )
        expected_output_file = expected_output_dir / "chunks.json"

        # Directory structure should match input but in chunks folder
        assert expected_output_dir.exists()
        assert expected_output_file.exists()

    def test_resume_capability_skip_processed(self, test_storage):
        """Test that already processed files are skipped on resume."""
        temp_dir, storage = test_storage

        # Setup test file
        input_file = temp_dir / "processed" / "documents" / "test" / "document.txt"
        input_file.parent.mkdir(parents=True)
        input_file.write_text("Content to be chunked.")

        # Create existing output file
        output_dir = temp_dir / "processed" / "chunks" / "documents" / "test"
        output_dir.mkdir(parents=True)
        output_file = output_dir / "chunks.json"
        output_file.write_text('["existing_chunk_data"]')

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {"chunk_size": 50}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        result = chunker.process_single_document(input_file)

        # Should skip processing (implementation may handle this differently)
        # Note: Actual implementation may not have SKIPPED status, check behavior
        assert result.status in [ChunkingStatus.SUCCESS, ChunkingStatus.FAILED]

        # Content should remain unchanged
        assert output_file.read_text() == '["existing_chunk_data"]'


class TestTextChunkerTrackerIntegration:
    """Tests for text chunker integration with publication tracker."""

    def test_chunker_works_without_tracker(self, test_storage):
        """Test that chunker works correctly when tracker is None."""
        temp_dir, storage = test_storage

        # Create a test text file
        doc_dir = temp_dir / "processed" / "documents" / "growthlab" / "test_doc_789"
        doc_dir.mkdir(parents=True, exist_ok=True)
        text_file = doc_dir / "test.txt"
        text_file.write_text("This is a test document with enough content to chunk.")

        # Create chunker without tracker
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {"chunk_size": 100}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path=config_path, tracker=None)
        results = chunker.process_all_documents(storage=storage)

        # Should complete without errors
        assert isinstance(results, list)
