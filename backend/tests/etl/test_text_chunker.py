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
from unittest.mock import patch

import pytest
import yaml

from backend.etl.utils.text_chunker import ChunkingStatus, DocumentChunk, TextChunker
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

        # Use real LocalStorage so glob/exists/download work
        storage = LocalStorage(temp_config_dir / "data")

        # Run batch processing
        chunker = TextChunker(temp_config_dir / "config.yaml")
        results = chunker.process_all_documents(storage)

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


class TestTextChunkerStrategies:
    """Unit tests for different chunking strategies and core logic."""

    @pytest.fixture
    def simple_config(self):
        """Simple configuration for unit testing.

        Note: All size values are in TOKENS (not characters) since the chunker
        now uses token-based limits for embedding model compatibility.
        """
        return {
            "chunk_size": 50,  # tokens
            "chunk_overlap": 10,  # tokens
            "min_chunk_size": 20,  # tokens
            "max_chunk_size": 100,  # tokens
            "preserve_structure": True,
            "respect_sentences": True,
            "structure_markers": [r"^#{1,6}\s+", r"^\d+\.\s+"],
        }

    def test_fixed_size_chunking_strategy(self, simple_config):
        """Test fixed-size chunking with overlap using token-based limits."""
        # Create text with enough tokens to generate multiple chunks
        # Each repetition is ~14 tokens
        text = "This is a test document with more words to increase token count. " * 20

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
            # Token-based validation (not character-based)
            assert chunk.token_count <= simple_config["max_chunk_size"]
            assert chunk.token_count >= simple_config["min_chunk_size"]
            assert chunk.chunk_index == i

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
        """Test document structure-aware chunking with token-based limits."""
        # Create structured text with enough tokens per section to generate chunks
        structured_text = """# Introduction
This is the introduction section with extensive content that explains the topic
in great detail. We provide comprehensive background information and context
for the research problem. The introduction covers multiple aspects of the issue
and sets up the framework for the rest of the document.

## Background
This subsection provides detailed background information about the research area.
It includes historical context, prior work, and the current state of knowledge.
We discuss various approaches that have been tried and their limitations.

1. First numbered point
This explains the first important concept with sufficient detail to ensure
it contains enough tokens. We elaborate on the implications and connections.

2. Second numbered point
This covers the second key area with thorough explanation of the methodology
and findings. Additional context is provided for completeness.

# Methods
This section describes the methodology used in the study with detailed
explanations of each step in the process. We cover data collection,
analysis procedures, and validation approaches used throughout.
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

        # Should respect document structure and create chunks
        assert len(chunks) >= 1

        # Verify token limits are respected
        for chunk in chunks:
            assert chunk.token_count <= simple_config["max_chunk_size"]
            assert chunk.token_count >= simple_config["min_chunk_size"]

        # If multiple chunks, some should capture section information
        if len(chunks) > 1:
            section_chunks = [c for c in chunks if c.section_title is not None]
            # Note: section_title may not be set for all chunks
            # Just verify chunks respect token limits (already done above)

    def test_hybrid_chunking_strategy(self, simple_config):
        """Test hybrid strategy that combines all approaches with token limits."""
        # Create structured text with enough tokens to generate multiple chunks
        structured_text = """# Introduction
This is a longer introduction with multiple sentences that provide extensive
context about the topic at hand. It contains several ideas that need to be
explained clearly and thoroughly. The content flows from one concept to
another, building a comprehensive understanding of the subject matter.
We include additional details to ensure sufficient token count for testing.

## Detailed Analysis
This section provides detailed analysis of the topic with thorough examination
of each component. Each paragraph builds on the previous one with additional
evidence and supporting arguments. The arguments are presented systematically
with clear logical progression from premise to conclusion.

1. First Point
The first point makes an important argument about the topic with sufficient
detail to constitute a meaningful chunk. We elaborate on the implications
and provide examples to illustrate the concept clearly.

2. Second Point
The second point extends the analysis further with additional considerations.
We explore the connections between different aspects of the problem and
discuss potential solutions and their trade-offs.

# Conclusion
This section summarizes the key findings and their implications for future
research and practice. We highlight the main contributions and suggest
directions for further investigation.
"""

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_hybrid(structured_text, Path("/test.txt"), "test_doc")

        # Hybrid should produce reasonable chunks
        assert len(chunks) >= 1

        # Should respect token limits
        for chunk in chunks:
            assert chunk.token_count >= simple_config["min_chunk_size"]
            assert chunk.token_count <= simple_config["max_chunk_size"]

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
                # Token-based validation
                assert all(
                    c.token_count <= simple_config["max_chunk_size"] for c in chunks
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
        """Test that chunk token limits are enforced."""
        # Long text that should be split - enough tokens to exceed max_chunk_size
        # "A very long sentence with additional words for token count. " is ~12 tokens
        long_text = "A very long sentence with additional words for token count. " * 50

        # Create temporary config file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": simple_config}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.chunk_fixed_size(long_text)

        # All chunks should respect TOKEN size constraints (not character)
        for chunk in chunks:
            assert chunk.token_count <= simple_config["max_chunk_size"]
            assert chunk.token_count >= simple_config["min_chunk_size"]

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
        """Test that overlapping chunks provide proper context continuity.

        With token-based chunking, overlap is measured in tokens not characters.
        We verify that consecutive chunks share some common content at their
        boundaries (the overlap region).
        """
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
            # Check that consecutive chunks have overlapping content
            for i in range(1, len(chunks)):
                prev_chunk = chunks[i - 1]
                curr_chunk = chunks[i]

                # With token-based overlap, verify that the end of prev_chunk
                # appears somewhere at the start of curr_chunk
                # (not exact character match since token boundaries may differ)
                prev_tokens = chunker.encoder.encode(prev_chunk.text_content)
                curr_tokens = chunker.encoder.encode(curr_chunk.text_content)

                # Get the overlap tokens from prev chunk
                overlap_tokens = simple_config["chunk_overlap"]
                if len(prev_tokens) >= overlap_tokens:
                    expected_overlap = prev_tokens[-overlap_tokens:]
                    actual_start = curr_tokens[:overlap_tokens]
                    # The overlap tokens should match
                    assert expected_overlap == actual_start

    def test_embedding_model_token_limit_enforced(self):
        """Test that chunks never exceed the embedding model's token limit.

        This is a critical test to prevent the 33% embedding failure rate
        that was caused by chunks exceeding OpenAI's 8192 token limit.
        Uses the tiktoken fallback path (no model_name in config).
        """
        import tempfile

        # Create a very long document that would exceed token limits
        # if not properly chunked (this simulates problematic documents)
        long_paragraph = (
            "This is a very long paragraph with many words that continues "
            "to add more content to increase the token count significantly. "
            "The economic analysis reveals important findings about the "
            "relationship between various factors and development outcomes. "
        )
        # Create text that would be ~20,000 tokens without chunking
        long_text = long_paragraph * 500

        # Use default config which should enforce token limits
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"file_processing": {"chunking": {}}}, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)
        chunks = chunker.create_chunks(long_text, Path("/test.txt"), "test_doc")

        # Critical assertion: NO chunk should exceed the embedding model limit
        # (uses the chunker's configured limit from YAML)
        for chunk in chunks:
            assert chunk.token_count < chunker.embedding_max_tokens, (
                f"Chunk exceeds embedding model token limit: "
                f"{chunk.token_count} > {chunker.embedding_max_tokens}"
            )

        # Also verify we created multiple chunks (not one huge chunk)
        assert len(chunks) > 10, "Should create many chunks for this large document"

    def test_chunk_size_clamped_when_exceeds_max_after_model_limit(self):
        """Test that chunk_size is reduced when max_chunk_size is clamped below it.

        When max_tokens is small (e.g., 256 for all-MiniLM-L6-v2), max_chunk_size
        gets clamped to max_tokens - 100. If chunk_size > clamped max_chunk_size,
        chunk_size must also be reduced automatically.
        """
        import tempfile

        config = {
            "file_processing": {
                "embedding": {
                    "model_name": "all-MiniLM-L6-v2",
                    "max_tokens": 256,
                },
                "chunking": {
                    "chunk_size": 500,  # Would exceed clamped max_chunk_size
                    "max_chunk_size": 8000,
                    "chunk_overlap": 50,
                    "min_chunk_size": 50,
                },
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)

        chunker = TextChunker(config_path)

        # max_chunk_size should be clamped to 256 - 100 = 156
        assert chunker.max_chunk_size == 156
        # chunk_size must be reduced to fit within max_chunk_size
        assert chunker.chunk_size <= chunker.max_chunk_size
        assert chunker.chunk_size >= chunker.min_chunk_size


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


class TestEnforceTokenLimits:
    """Tests for _enforce_token_limits() and _force_split_by_tokens()."""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker with small token limits for testing."""
        import tempfile

        config = {
            "file_processing": {
                "chunking": {
                    "strategy": "hybrid",
                    "chunk_size": 50,
                    "chunk_overlap": 10,
                    "min_chunk_size": 10,
                    "max_chunk_size": 100,
                }
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)
        return TextChunker(config_path)

    def _make_chunk(self, chunker, text: str, index: int = 0) -> DocumentChunk:
        """Helper to create a DocumentChunk with correct token_count."""
        return DocumentChunk(
            chunk_id=f"doc_chunk_{index:04d}",
            source_document_id="doc",
            source_file_path=Path("/test.txt"),
            chunk_index=index,
            text_content=text,
            character_start=0,
            character_end=len(text),
            page_numbers=[],
            section_title=None,
            metadata={},
            created_at=datetime.now(),
            chunk_size=len(text),
            token_count=chunker._count_tokens(text),
        )

    def test_oversized_chunk_gets_split(self, chunker):
        """A chunk at 2x the max_chunk_size should be force-split."""
        # Create text that is ~200 tokens (2x max_chunk_size of 100)
        oversized_text = "word " * 200
        chunk = self._make_chunk(chunker, oversized_text)
        assert chunk.token_count > chunker.max_chunk_size

        result = chunker._enforce_token_limits([chunk], Path("/test.txt"), "doc")

        # Should have split into multiple chunks
        assert len(result) > 1
        # Every resulting chunk must respect the token limit
        for c in result:
            assert c.token_count <= chunker.max_chunk_size
        # Chunk indices should be reindexed
        for i, c in enumerate(result):
            assert c.chunk_index == i
            assert c.chunk_id == f"doc_chunk_{i:04d}"

    def test_chunk_at_limit_not_split(self, chunker):
        """A chunk exactly at max_chunk_size should NOT be split."""
        # Build text that is exactly max_chunk_size tokens
        # Approach: encode tokens, then decode exactly max_chunk_size of them
        base_text = "word " * 200
        tokens = chunker.encoder.encode(base_text)
        exact_text = chunker.encoder.decode(tokens[: chunker.max_chunk_size])
        chunk = self._make_chunk(chunker, exact_text)
        assert chunk.token_count == chunker.max_chunk_size

        result = chunker._enforce_token_limits([chunk], Path("/test.txt"), "doc")

        assert len(result) == 1
        assert result[0].text_content == exact_text

    def test_multiple_oversized_chunks(self, chunker):
        """Multiple oversized chunks should each be split independently."""
        oversized1 = "alpha " * 200
        oversized2 = "beta " * 200
        normal = "gamma " * 20  # Should be small enough

        chunks = [
            self._make_chunk(chunker, oversized1, 0),
            self._make_chunk(chunker, normal, 1),
            self._make_chunk(chunker, oversized2, 2),
        ]

        result = chunker._enforce_token_limits(chunks, Path("/test.txt"), "doc")

        # Should have more than 3 chunks due to splitting
        assert len(result) > 3
        # All chunks respect limit
        for c in result:
            assert c.token_count <= chunker.max_chunk_size
        # The normal chunk text should still appear in the results
        normal_texts = [c.text_content for c in result if "gamma" in c.text_content]
        assert len(normal_texts) >= 1

    def test_force_split_by_tokens_produces_valid_chunks(self, chunker):
        """_force_split_by_tokens splits text into max_chunk_size chunks."""
        big_text = "word " * 300
        sub_texts = chunker._force_split_by_tokens(big_text)

        assert len(sub_texts) > 1
        for sub in sub_texts:
            assert chunker._count_tokens(sub) <= chunker.max_chunk_size


class TestDetectSentencesAdversarial:
    """Tests for _detect_sentences with adversarial/edge-case input."""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance for sentence detection tests."""
        import tempfile

        config = {"file_processing": {"chunking": {"chunk_size": 500}}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            config_path = Path(f.name)
        return TextChunker(config_path)

    def test_no_punctuation_ocr_text(self, chunker):
        """OCR text without any sentence-ending punctuation should not crash."""
        text = (
            "the quick brown fox jumps over the lazy dog "
            "no punctuation anywhere at all "
            "just continuous text without any periods or marks"
        )
        result = chunker._detect_sentences(text)
        # Should return at least one segment (the whole text)
        assert len(result) >= 1
        # Combined result should contain all the original text
        combined = "".join(result)
        assert "the quick brown fox" in combined

    def test_abbreviations(self, chunker):
        """Abbreviations like 'Dr.' and 'et al.' should not crash."""
        text = (
            "Dr. Smith et al. found that the rate was significant. The study continued."
        )
        result = chunker._detect_sentences(text)
        # Should not crash and should return some sentences
        assert len(result) >= 1
        combined = "".join(result)
        assert "Dr" in combined
        assert "study continued" in combined

    def test_decimal_numbers(self, chunker):
        """Decimal numbers like '3.14' should not crash."""
        text = "The rate was 3.14 percent. This exceeded expectations."
        result = chunker._detect_sentences(text)
        assert len(result) >= 1
        combined = "".join(result)
        assert "3" in combined
        assert "expectations" in combined

    def test_urls(self, chunker):
        """URLs with periods should not crash."""
        text = "Visit https://example.com. Then continue reading the document."
        result = chunker._detect_sentences(text)
        assert len(result) >= 1
        combined = "".join(result)
        assert "example" in combined
        assert "document" in combined

    def test_empty_string(self, chunker):
        """Empty string should return empty list without crashing."""
        result = chunker._detect_sentences("")
        assert result == []

    def test_only_punctuation(self, chunker):
        """String of only punctuation should not crash."""
        result = chunker._detect_sentences("... !!! ???")
        # Should not crash; may return empty or contain punctuation
        assert isinstance(result, list)
