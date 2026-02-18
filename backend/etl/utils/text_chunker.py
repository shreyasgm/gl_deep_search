"""
Text Chunking System for Growth Lab Deep Search.

This module provides text chunking functionality that transforms processed text
documents into semantically meaningful chunks for vector embeddings and retrieval.

All chunk size limits are enforced using token counts (not character counts) to
ensure compatibility with embedding model token limits (e.g., OpenAI's
text-embedding-3-small has an 8,192 token limit).
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import tiktoken
import yaml
from loguru import logger

from backend.etl.models.tracking import ProcessingStatus
from backend.etl.utils.publication_tracker import PublicationTracker

# Fallback defaults if not specified in config
DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"
DEFAULT_EMBEDDING_MAX_TOKENS = 8192


class ChunkingStatus(Enum):
    """Status of text chunking operation."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class DocumentChunk:
    """Represents a single text chunk from a document."""

    chunk_id: str  # Unique identifier
    source_document_id: str  # Reference to original document
    source_file_path: Path  # Path to processed text file
    chunk_index: int  # Sequential position in document
    text_content: str  # The actual chunk text
    character_start: int  # Start position in original text
    character_end: int  # End position in original text
    page_numbers: list[int]  # Pages covered by this chunk
    section_title: str | None  # Parent section if available
    metadata: dict[str, Any]  # Additional metadata
    created_at: datetime  # Processing timestamp
    chunk_size: int  # Actual chunk size in characters (for reference)
    token_count: int  # Actual chunk size in tokens (primary metric)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert Path to string
        result["source_file_path"] = str(self.source_file_path)
        # Convert datetime to ISO string
        result["created_at"] = self.created_at.isoformat()
        return result


@dataclass
class ChunkingResult:
    """Result of chunking operation for a single document."""

    document_id: str
    source_path: Path
    chunks: list[DocumentChunk]
    total_chunks: int
    processing_time: float
    status: ChunkingStatus
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert Path to string
        result["source_path"] = str(self.source_path)
        # Convert chunks to dictionaries
        result["chunks"] = [chunk.to_dict() for chunk in self.chunks]
        # Convert enum to value
        result["status"] = self.status.value
        return result


class TextChunker:
    """Main text chunking system with multiple strategies.

    All size limits are enforced using token counts to ensure compatibility
    with embedding model limits (e.g., OpenAI text-embedding-3-small: 8192 tokens).
    """

    def __init__(self, config_path: Path, tracker: PublicationTracker | None = None):
        """
        Initialize text chunker with configuration.

        Args:
            config_path: Path to configuration file
            tracker: Optional PublicationTracker instance for updating processing status

        Note:
            All chunk size parameters (chunk_size, chunk_overlap, min_chunk_size,
            max_chunk_size) are measured in TOKENS, not characters. This ensures
            chunks fit within embedding model token limits.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.tracker = tracker

        # Get embedding config for tokenizer settings
        embedding_config = self.config.get("file_processing", {}).get("embedding", {})
        tiktoken_encoding = embedding_config.get(
            "tiktoken_encoding", DEFAULT_TIKTOKEN_ENCODING
        )
        self.embedding_max_tokens = embedding_config.get(
            "max_tokens", DEFAULT_EMBEDDING_MAX_TOKENS
        )

        # Initialize tiktoken encoder for token counting
        self.encoder = tiktoken.get_encoding(tiktoken_encoding)
        logger.debug(
            f"Using tiktoken encoding: {tiktoken_encoding}, "
            f"embedding max tokens: {self.embedding_max_tokens}"
        )

        raw_config = self.config.get("file_processing", {}).get("chunking", {})

        # Defaults (all sizes in TOKENS, not characters)
        # These are conservative defaults that work well with text-embedding-3-small
        defaults = {
            "strategy": "hybrid",
            "chunk_size": 500,  # target tokens per chunk
            "chunk_overlap": 50,  # tokens of overlap between chunks
            "min_chunk_size": 50,  # minimum viable chunk size in tokens
            "max_chunk_size": 8000,  # max tokens (embedding model limit is 8192)
            "preserve_structure": True,
            "respect_sentences": True,
            "structure_markers": [
                r"^#{1,6}\s+",
                r"^\d+\.\s+",
                r"^[A-Z][A-Z\s]+:",
            ],
        }

        # Merge and normalize
        merged = {**defaults, **(raw_config or {})}

        # Normalize/validate values; fall back to defaults on invalid
        allowed_strategies = {"fixed", "sentence", "structure", "hybrid"}
        if merged.get("strategy") not in allowed_strategies:
            merged["strategy"] = defaults["strategy"]

        if not isinstance(merged.get("chunk_size"), int) or merged["chunk_size"] <= 0:
            merged["chunk_size"] = defaults["chunk_size"]

        if (
            not isinstance(merged.get("min_chunk_size"), int)
            or not isinstance(merged.get("max_chunk_size"), int)
            or merged["min_chunk_size"] <= 0
            or merged["max_chunk_size"] <= merged["min_chunk_size"]
        ):
            merged["min_chunk_size"] = defaults["min_chunk_size"]
            merged["max_chunk_size"] = defaults["max_chunk_size"]

        # Enforce max_chunk_size doesn't exceed embedding model limit
        if merged["max_chunk_size"] > self.embedding_max_tokens:
            clamped_value = self.embedding_max_tokens - 100
            logger.warning(
                f"max_chunk_size ({merged['max_chunk_size']}) exceeds embedding "
                f"model limit ({self.embedding_max_tokens}). Clamping to "
                f"{clamped_value}."
            )
            merged["max_chunk_size"] = self.embedding_max_tokens - 100  # Leave buffer

        if (
            not isinstance(merged.get("chunk_overlap"), int)
            or merged["chunk_overlap"] < 0
            or merged["chunk_overlap"] >= merged["chunk_size"]
        ):
            # ensure reasonable overlap strictly less than chunk_size
            merged["chunk_overlap"] = min(
                defaults["chunk_overlap"],
                merged["chunk_size"] - 1,
            )

        # Persist normalized config for tests expecting defaults present
        self.chunking_config = merged

        # Extract configuration parameters from normalized config
        self.strategy = merged["strategy"]
        self.chunk_size = merged["chunk_size"]  # in tokens
        self.chunk_overlap = merged["chunk_overlap"]  # in tokens
        self.min_chunk_size = merged["min_chunk_size"]  # in tokens
        self.max_chunk_size = merged["max_chunk_size"]  # in tokens
        self.preserve_structure = merged["preserve_structure"]
        self.respect_sentences = merged["respect_sentences"]
        self.structure_markers = merged["structure_markers"]

        logger.info(
            f"TextChunker initialized with strategy: {self.strategy}, "
            f"chunk_size: {self.chunk_size} tokens, max: {self.max_chunk_size} tokens"
        )

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoder.encode(text))

    def _split_text_by_tokens(
        self, text: str, max_tokens: int, overlap_tokens: int = 0
    ) -> list[tuple[str, int, int]]:
        """
        Split text into chunks that respect token limits.

        Args:
            text: The text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks

        Returns:
            List of tuples: (chunk_text, char_start, char_end)
        """
        tokens = self.encoder.encode(text)
        chunks = []

        if len(tokens) <= max_tokens:
            return [(text, 0, len(text))]

        start_token = 0
        char_position = 0

        while start_token < len(tokens):
            end_token = min(start_token + max_tokens, len(tokens))
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.encoder.decode(chunk_tokens)

            # Calculate character positions
            char_start = char_position
            char_end = char_start + len(chunk_text)

            chunks.append((chunk_text, char_start, char_end))

            # Move to next chunk with overlap
            if end_token >= len(tokens):
                break

            advance = max_tokens - overlap_tokens
            if advance <= 0:
                advance = max_tokens  # Prevent infinite loop
            start_token += advance
            # Update char position (approximate - overlap makes this complex)
            char_position = (
                char_end
                - len(
                    self.encoder.decode(
                        tokens[start_token - overlap_tokens : start_token]
                    )
                )
                if overlap_tokens > 0
                else char_end
            )

        return chunks

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise

    def process_all_documents(self, storage) -> list[ChunkingResult]:
        """Process all documents in the processed directory."""
        processed_dir = self._resolve_processed_documents_dir(storage)
        results: list[ChunkingResult] = []

        if not processed_dir.exists():
            logger.warning("No processed documents directory found")
            return results

        # Find all processed text files
        text_files = list(processed_dir.rglob("*.txt"))
        logger.info(f"Found {len(text_files)} text files to process")

        for text_file in text_files:
            try:
                result = self.process_single_document(text_file)
                results.append(result)

                # Save chunks to JSON file
                self._save_chunks(result, storage)

                # Update tracker status based on result
                if self.tracker:
                    try:
                        if result.status == ChunkingStatus.SUCCESS:
                            self.tracker.update_processing_status(
                                result.document_id, ProcessingStatus.PROCESSED
                            )
                            logger.debug(
                                f"Updated processing status to PROCESSED for "
                                f"{result.document_id}"
                            )
                        elif result.status == ChunkingStatus.FAILED:
                            self.tracker.update_processing_status(
                                result.document_id,
                                ProcessingStatus.FAILED,
                                error=result.error_message,
                            )
                            logger.debug(
                                f"Updated processing status to FAILED for "
                                f"{result.document_id}"
                            )
                    except Exception as tracker_error:
                        logger.warning(
                            f"Failed to update processing status for "
                            f"{result.document_id}: {tracker_error}"
                        )

            except Exception as e:
                logger.error(f"Failed to process {text_file}: {e}")
                error_result = ChunkingResult(
                    document_id=text_file.stem,
                    source_path=text_file,
                    chunks=[],
                    total_chunks=0,
                    processing_time=0.0,
                    status=ChunkingStatus.FAILED,
                    error_message=str(e),
                )
                results.append(error_result)

                # Update tracker status for exception case
                if self.tracker:
                    try:
                        self.tracker.update_processing_status(
                            error_result.document_id,
                            ProcessingStatus.FAILED,
                            error=str(e),
                        )
                    except Exception as tracker_error:
                        logger.warning(
                            f"Failed to update processing status for "
                            f"{error_result.document_id}: {tracker_error}"
                        )

        return results

    def process_single_document(self, text_file_path: Path) -> ChunkingResult:
        """Process a single document and return chunking result."""
        start_time = time.time()
        # Prefer directory name as document ID
        # (e.g., processed/documents/<doc_id>/file.txt)
        try:
            parent_name = text_file_path.parent.name
        except Exception:
            parent_name = ""
        document_id = parent_name if parent_name else text_file_path.stem

        logger.info(f"Processing document: {document_id}")

        try:
            # Read the text file
            with open(text_file_path, encoding="utf-8") as f:
                text_content = f.read()

            if len(text_content.strip()) == 0:
                raise ValueError("Empty text file")

            # Note: Do not hard-fail for short documents here. Strategy
            # implementations will decide whether a document is chunkable.

            # Create chunks using the configured strategy
            chunks = self.create_chunks(text_content, text_file_path, document_id)

            processing_time = time.time() - start_time

            result = ChunkingResult(
                document_id=document_id,
                source_path=text_file_path,
                chunks=chunks,
                total_chunks=len(chunks),
                processing_time=processing_time,
                status=ChunkingStatus.SUCCESS if chunks else ChunkingStatus.FAILED,
            )

            # Persist chunks to disk for single-document processing as well
            try:
                self._save_chunks(result, storage=None)
            except Exception as save_error:
                logger.warning(
                    (
                        "Failed to save chunks for %s during single-document "
                        "processing: %s"
                    ),
                    document_id,
                    save_error,
                )

            logger.info(
                f"Successfully chunked {document_id}: {len(chunks)} chunks "
                f"in {processing_time:.2f}s"
            )
            return result

        except FileNotFoundError:
            processing_time = time.time() - start_time
            error_msg = f"File not found: {text_file_path}"
            logger.error(f"Failed to process {document_id}: {error_msg}")

            return ChunkingResult(
                document_id=document_id,
                source_path=text_file_path,
                chunks=[],
                total_chunks=0,
                processing_time=processing_time,
                status=ChunkingStatus.FAILED,
                error_message=error_msg,
            )
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process {document_id}: {e}")

            return ChunkingResult(
                document_id=document_id,
                source_path=text_file_path,
                chunks=[],
                total_chunks=0,
                processing_time=processing_time,
                status=ChunkingStatus.FAILED,
                error_message=str(e),
            )

    def create_chunks(
        self, text: str, source_path: Path, document_id: str
    ) -> list[DocumentChunk]:
        """Create chunks using the configured strategy with fallback."""
        strategies: dict[
            str, Callable[[str, Path | None, str | None], list[DocumentChunk]]
        ] = {
            "fixed": self.chunk_fixed_size,
            "sentence": self.chunk_by_sentences,
            "structure": self.chunk_by_structure,
            "hybrid": self.chunk_hybrid,
        }

        # Try the configured strategy first
        strategy_order = [self.strategy]

        # Add fallback strategies if the main one fails
        if self.strategy != "hybrid":
            strategy_order.append("hybrid")
        if "sentence" not in strategy_order:
            strategy_order.append("sentence")
        if "fixed" not in strategy_order:
            strategy_order.append("fixed")

        for strategy_name in strategy_order:
            try:
                logger.debug(f"Attempting chunking with strategy: {strategy_name}")
                strategy_func = strategies.get(strategy_name)
                if strategy_func is None:
                    logger.warning(f"Strategy {strategy_name} not found")
                    continue
                chunks = strategy_func(text, source_path, document_id)

                if chunks:
                    logger.info(f"Successfully chunked with strategy: {strategy_name}")
                    # Safety net: split any chunks that exceed embedding token limit
                    chunks = self._enforce_token_limits(
                        chunks, source_path, document_id
                    )
                    return chunks
                else:
                    logger.warning(f"Strategy {strategy_name} produced no chunks")

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # If all strategies fail, return empty list
        logger.error("All chunking strategies failed")
        return []

    def _enforce_token_limits(
        self,
        chunks: list[DocumentChunk],
        source_path: Path | None,
        document_id: str | None,
    ) -> list[DocumentChunk]:
        """Final safety net: split any chunks exceeding the embedding model token limit.

        This catches oversized chunks that slip through any strategy (e.g., when
        sentence detection fails on OCR output with missing punctuation).
        """
        safe_chunks: list[DocumentChunk] = []
        reindex_needed = False

        for chunk in chunks:
            if chunk.token_count <= self.max_chunk_size:
                safe_chunks.append(chunk)
            else:
                reindex_needed = True
                logger.warning(
                    f"Chunk {chunk.chunk_id} exceeds token limit "
                    f"({chunk.token_count} > {self.max_chunk_size}). "
                    f"Force-splitting."
                )
                # Force-split using token-based splitting
                sub_texts = self._force_split_by_tokens(chunk.text_content)
                for sub_text in sub_texts:
                    token_count = self._count_tokens(sub_text)
                    sub_chunk = DocumentChunk(
                        chunk_id="",  # Will be reindexed below
                        source_document_id=chunk.source_document_id,
                        source_file_path=chunk.source_file_path,
                        chunk_index=0,  # Will be reindexed below
                        text_content=sub_text,
                        character_start=chunk.character_start,
                        character_end=chunk.character_start + len(sub_text),
                        page_numbers=chunk.page_numbers,
                        section_title=chunk.section_title,
                        metadata={
                            **chunk.metadata,
                            "force_split": True,
                        },
                        created_at=chunk.created_at,
                        chunk_size=len(sub_text),
                        token_count=token_count,
                    )
                    safe_chunks.append(sub_chunk)

        # Reindex chunk IDs and indices if any splitting occurred
        if reindex_needed:
            doc_id = document_id or "document"
            for i, chunk in enumerate(safe_chunks):
                chunk.chunk_index = i
                chunk.chunk_id = f"{doc_id}_chunk_{i:04d}"

        return safe_chunks

    def chunk_fixed_size(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create fixed-size chunks with overlap, using token-based limits."""
        chunks = []
        chunk_index = 0

        # Remove page markers for cleaner chunking
        clean_text = self._clean_text_for_chunking(text)
        page_info = self._extract_page_info(text)

        # Defaults for optional parameters
        if source_path is None:
            source_path = Path("/tmp/unknown.txt")
        if document_id is None:
            document_id = source_path.stem or "document"

        # Encode the full text into tokens
        tokens = self.encoder.encode(clean_text)
        total_tokens = len(tokens)

        # Check if entire text is below minimum
        if total_tokens < self.min_chunk_size:
            # Create single chunk for small documents
            token_count = total_tokens
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                source_document_id=document_id,
                source_file_path=source_path,
                chunk_index=chunk_index,
                text_content=clean_text,
                character_start=0,
                character_end=len(clean_text),
                page_numbers=self._find_pages_for_position(
                    0, len(clean_text), page_info
                ),
                section_title=None,
                metadata={"strategy": "fixed"},
                created_at=datetime.now(),
                chunk_size=len(clean_text),
                token_count=token_count,
            )
            return [chunk]

        start_token = 0
        while start_token < total_tokens:
            end_token = min(start_token + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.encoder.decode(chunk_tokens)
            token_count = len(chunk_tokens)

            # Skip chunks that are too small (in tokens)
            if token_count < self.min_chunk_size and start_token > 0:
                break

            # Calculate approximate character positions for page mapping
            # (This is approximate since token boundaries don't align with characters)
            char_start = len(self.encoder.decode(tokens[:start_token]))
            char_end = char_start + len(chunk_text)

            # Find which pages this chunk covers
            chunk_pages = self._find_pages_for_position(char_start, char_end, page_info)

            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                source_document_id=document_id,
                source_file_path=source_path,
                chunk_index=chunk_index,
                text_content=chunk_text,
                character_start=char_start,
                character_end=char_end,
                page_numbers=chunk_pages,
                section_title=None,
                metadata={"strategy": "fixed"},
                created_at=datetime.now(),
                chunk_size=len(chunk_text),
                token_count=token_count,
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap (in tokens)
            advance = self.chunk_size - self.chunk_overlap
            if advance <= 0:
                advance = self.chunk_size  # Prevent infinite loop
            start_token += advance

        return chunks

    def chunk_by_sentences(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create chunks respecting sentence boundaries, using token-based limits."""
        chunks = []
        chunk_index = 0

        # Defaults for optional parameters
        if source_path is None:
            source_path = Path("/tmp/unknown.txt")
        if document_id is None:
            document_id = source_path.stem or "document"

        clean_text = self._clean_text_for_chunking(text)
        page_info = self._extract_page_info(text)
        sentences = self._detect_sentences(clean_text)

        current_chunk = ""
        current_chunk_tokens = 0
        current_start = 0
        sentence_start = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence would exceed chunk size (in tokens)
            if (
                current_chunk_tokens + sentence_tokens > self.chunk_size
                and current_chunk
            ):
                # Save current chunk
                chunk_text = current_chunk.strip()
                token_count = self._count_tokens(chunk_text)

                if token_count >= self.min_chunk_size:
                    chunk_pages = self._find_pages_for_position(
                        current_start, current_start + len(chunk_text), page_info
                    )

                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                        source_document_id=document_id,
                        source_file_path=source_path,
                        chunk_index=chunk_index,
                        text_content=chunk_text,
                        character_start=current_start,
                        character_end=current_start + len(chunk_text),
                        page_numbers=chunk_pages,
                        section_title=None,
                        metadata={"strategy": "sentence"},
                        created_at=datetime.now(),
                        chunk_size=len(chunk_text),
                        token_count=token_count,
                    )

                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap if needed (in tokens)
                if self.chunk_overlap > 0 and chunks:
                    # Take last sentences that fit within overlap token budget
                    overlap_text = self._get_overlap_text(
                        chunk_text, self.chunk_overlap
                    )
                    current_chunk = overlap_text + " " + sentence
                    current_chunk_tokens = self._count_tokens(current_chunk)
                    current_start = current_start + len(chunk_text) - len(overlap_text)
                else:
                    current_chunk = sentence
                    current_chunk_tokens = sentence_tokens
                    current_start = sentence_start
            else:
                current_chunk += sentence
                current_chunk_tokens += sentence_tokens
                if not current_chunk.strip():
                    current_start = sentence_start

            sentence_start += len(sentence)

        # Handle final chunk
        if current_chunk.strip():
            chunk_text = current_chunk.strip()
            token_count = self._count_tokens(chunk_text)

            if token_count >= self.min_chunk_size:
                chunk_pages = self._find_pages_for_position(
                    current_start, current_start + len(chunk_text), page_info
                )

                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    source_document_id=document_id,
                    source_file_path=source_path,
                    chunk_index=chunk_index,
                    text_content=chunk_text,
                    character_start=current_start,
                    character_end=current_start + len(chunk_text),
                    page_numbers=chunk_pages,
                    section_title=None,
                    metadata={"strategy": "sentence"},
                    created_at=datetime.now(),
                    chunk_size=len(chunk_text),
                    token_count=token_count,
                )

                chunks.append(chunk)

        return chunks

    def _get_overlap_text(self, text: str, target_tokens: int) -> str:
        """Get the last portion of text that fits within target token count."""
        tokens = self.encoder.encode(text)
        if len(tokens) <= target_tokens:
            return text
        overlap_tokens = tokens[-target_tokens:]
        return self.encoder.decode(overlap_tokens)

    def chunk_by_structure(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create chunks based on document structure, using token-based limits."""
        chunks = []
        chunk_index = 0

        # Defaults for optional parameters
        if source_path is None:
            source_path = Path("/tmp/unknown.txt")
        if document_id is None:
            document_id = source_path.stem or "document"

        clean_text = self._clean_text_for_chunking(text)
        page_info = self._extract_page_info(text)
        sections = self._detect_structure(clean_text)

        for section in sections:
            section_text = section["text"].strip()
            section_title = section.get("title")
            section_tokens = self._count_tokens(section_text)

            if section_tokens < self.min_chunk_size:
                continue

            # If section is too large (in tokens), split it further
            if section_tokens > self.max_chunk_size:
                # Fall back to sentence-based chunking for this section
                sub_chunks = self._split_large_section(section_text, section_title)
                for sub_chunk_text in sub_chunks:
                    sub_chunk_text_stripped = sub_chunk_text.strip()
                    token_count = self._count_tokens(sub_chunk_text_stripped)

                    if token_count >= self.min_chunk_size:
                        chunk_pages = self._find_pages_for_text(
                            sub_chunk_text_stripped, page_info
                        )

                        chunk = DocumentChunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                            source_document_id=document_id,
                            source_file_path=source_path,
                            chunk_index=chunk_index,
                            text_content=sub_chunk_text_stripped,
                            character_start=section["start"],
                            character_end=section["end"],
                            page_numbers=chunk_pages,
                            section_title=section_title,
                            metadata={
                                "strategy": "structure",
                                "large_section_split": True,
                            },
                            created_at=datetime.now(),
                            chunk_size=len(sub_chunk_text_stripped),
                            token_count=token_count,
                        )

                        chunks.append(chunk)
                        chunk_index += 1
            else:
                # Use section as single chunk
                chunk_pages = self._find_pages_for_position(
                    section["start"], section["end"], page_info
                )

                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                    source_document_id=document_id,
                    source_file_path=source_path,
                    chunk_index=chunk_index,
                    text_content=section_text,
                    character_start=section["start"],
                    character_end=section["end"],
                    page_numbers=chunk_pages,
                    section_title=section_title,
                    metadata={"strategy": "structure"},
                    created_at=datetime.now(),
                    chunk_size=len(section_text),
                    token_count=section_tokens,
                )

                chunks.append(chunk)
                chunk_index += 1

        return chunks

    def chunk_hybrid(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create chunks using hybrid approach combining structure, sentence,
        and size constraints."""
        chunks = []
        chunk_index = 0

        # Defaults for optional parameters
        if source_path is None:
            source_path = Path("/tmp/unknown.txt")
        if document_id is None:
            document_id = source_path.stem or "document"

        clean_text = self._clean_text_for_chunking(text)
        page_info = self._extract_page_info(text)

        # Try structure-based chunking first
        try:
            sections = self._detect_structure(clean_text)
            if sections and len(sections) > 1:
                # Use structure-based approach but with size constraints
                for section in sections:
                    section_chunks = self._chunk_section_with_constraints(
                        section, source_path, document_id, chunk_index, page_info
                    )
                    chunks.extend(section_chunks)
                    chunk_index += len(section_chunks)

                if chunks:
                    logger.debug("Hybrid: Used structure-based chunking")
                    return chunks
        except Exception as e:
            logger.debug(f"Hybrid: Structure detection failed, falling back: {e}")

        # Fall back to sentence-based chunking with size constraints
        try:
            return self.chunk_by_sentences(text, source_path, document_id)
        except Exception as e:
            logger.debug(f"Hybrid: Sentence chunking failed, falling back: {e}")

        # Final fallback to fixed-size chunking
        return self.chunk_fixed_size(text, source_path, document_id)

    def _chunk_section_with_constraints(
        self,
        section: dict[str, Any],
        source_path: Path,
        document_id: str,
        start_index: int,
        page_info: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Chunk a section while respecting token limits and sentence constraints."""
        chunks: list[DocumentChunk] = []
        section_text = section["text"].strip()
        section_title = section.get("title")
        section_tokens = self._count_tokens(section_text)

        if section_tokens < self.min_chunk_size:
            return chunks

        if section_tokens <= self.chunk_size:
            # Section fits in one chunk (token-wise)
            chunk_pages = self._find_pages_for_position(
                section["start"], section["end"], page_info
            )

            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{start_index:04d}",
                source_document_id=document_id,
                source_file_path=source_path,
                chunk_index=start_index,
                text_content=section_text,
                character_start=section["start"],
                character_end=section["end"],
                page_numbers=chunk_pages,
                section_title=section_title,
                metadata={"strategy": "hybrid", "approach": "single_section"},
                created_at=datetime.now(),
                chunk_size=len(section_text),
                token_count=section_tokens,
            )

            chunks.append(chunk)
        else:
            # Split section using sentence boundaries with token limits
            sentences = self._detect_sentences(section_text)
            current_chunk = ""
            current_chunk_tokens = 0
            current_start = section["start"]
            chunk_idx = start_index

            for sentence in sentences:
                sentence_tokens = self._count_tokens(sentence)

                if (
                    current_chunk_tokens + sentence_tokens > self.chunk_size
                    and current_chunk
                ):
                    # Save current chunk
                    chunk_text = current_chunk.strip()
                    token_count = self._count_tokens(chunk_text)

                    if token_count >= self.min_chunk_size:
                        chunk_pages = self._find_pages_for_position(
                            current_start, current_start + len(chunk_text), page_info
                        )

                        chunk = DocumentChunk(
                            chunk_id=f"{document_id}_chunk_{chunk_idx:04d}",
                            source_document_id=document_id,
                            source_file_path=source_path,
                            chunk_index=chunk_idx,
                            text_content=chunk_text,
                            character_start=current_start,
                            character_end=current_start + len(chunk_text),
                            page_numbers=chunk_pages,
                            section_title=section_title,
                            metadata={
                                "strategy": "hybrid",
                                "approach": "section_sentences",
                            },
                            created_at=datetime.now(),
                            chunk_size=len(chunk_text),
                            token_count=token_count,
                        )

                        chunks.append(chunk)
                        chunk_idx += 1

                    # Start new chunk with overlap (in tokens)
                    if self.chunk_overlap > 0:
                        overlap_text = self._get_overlap_text(
                            chunk_text, self.chunk_overlap
                        )
                        current_chunk = overlap_text + " " + sentence
                        current_chunk_tokens = self._count_tokens(current_chunk)
                        current_start = (
                            current_start + len(chunk_text) - len(overlap_text)
                        )
                    else:
                        current_chunk = sentence
                        current_chunk_tokens = sentence_tokens
                        current_start = current_start + len(chunk_text)
                else:
                    current_chunk += sentence
                    current_chunk_tokens += sentence_tokens

            # Handle final chunk
            if current_chunk.strip():
                chunk_text = current_chunk.strip()
                token_count = self._count_tokens(chunk_text)

                if token_count >= self.min_chunk_size:
                    chunk_pages = self._find_pages_for_position(
                        current_start, current_start + len(chunk_text), page_info
                    )

                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_idx:04d}",
                        source_document_id=document_id,
                        source_file_path=source_path,
                        chunk_index=chunk_idx,
                        text_content=chunk_text,
                        character_start=current_start,
                        character_end=current_start + len(chunk_text),
                        page_numbers=chunk_pages,
                        section_title=section_title,
                        metadata={
                            "strategy": "hybrid",
                            "approach": "section_sentences",
                        },
                        created_at=datetime.now(),
                        chunk_size=len(chunk_text),
                        token_count=token_count,
                    )

                    chunks.append(chunk)

        return chunks

    def _clean_text_for_chunking(self, text: str) -> str:
        """Clean text by removing page markers and normalizing whitespace."""
        # Remove page markers
        cleaned = re.sub(r"^--- Page \d+ ---\s*$", "", text, flags=re.MULTILINE)
        # Collapse excessive blank lines but preserve line breaks
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        # Normalize spaces and tabs within lines (preserve newlines)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned.strip()

    def _extract_page_info(self, text: str) -> list[dict[str, Any]]:
        """Extract page information from text with page markers."""
        page_info = []
        lines = text.split("\n")
        current_page = None
        char_position = 0

        for line in lines:
            page_match = re.match(r"^--- Page (\d+) ---\s*$", line)
            if page_match:
                current_page = int(page_match.group(1))
                page_info.append(
                    {"page_number": current_page, "start_position": char_position}
                )

            char_position += len(line) + 1  # +1 for newline

        # Set end positions
        for i in range(len(page_info) - 1):
            page_info[i]["end_position"] = page_info[i + 1]["start_position"]

        if page_info:
            page_info[-1]["end_position"] = char_position

        return page_info

    def _find_pages_for_position(
        self, start: int, end: int, page_info: list[dict[str, Any]]
    ) -> list[int]:
        """Find which pages cover a given character range."""
        pages = set()

        for page in page_info:
            page_start = page.get("start_position", 0)
            page_end = page.get("end_position", float("inf"))

            # Check if there's any overlap between chunk and page
            if start < page_end and end > page_start:
                pages.add(page["page_number"])

        return sorted(list(pages))

    def _find_pages_for_text(
        self, text: str, page_info: list[dict[str, Any]]
    ) -> list[int]:
        """Find pages for a text snippet (fallback method)."""
        if not page_info:
            return []

        # For now, return first page as fallback
        return [page_info[0]["page_number"]] if page_info else []

    def _detect_sentences(self, text: str) -> list[str]:
        """Detect sentences using simple regex patterns."""
        # Split on sentence endings, keeping the delimiter
        sentences = re.split(r"([.!?;]+\s+)", text)

        result: list[str] = []
        current = ""

        for i, part in enumerate(sentences):
            current += part
            # If this is a delimiter and we have content, finish the sentence
            if i % 2 == 1 and current.strip():
                result.append(current)
                current = ""

        # Add any remaining text
        if current.strip():
            result.append(current)

        return [s for s in result if s.strip()]

    def _detect_structure(self, text: str) -> list[dict[str, Any]]:
        """Detect document structure using configured markers."""
        sections: list[dict[str, Any]] = []

        # Split text into lines for analysis
        lines = text.split("\n")
        current_section: dict[str, Any] = {"text": "", "start": 0, "title": None}
        char_position = 0

        for line in lines:
            line_stripped = line.strip()

            # Check if line matches any structure marker
            is_header = False
            for marker_pattern in self.structure_markers:
                if re.match(marker_pattern, line_stripped):
                    is_header = True
                    break

            if is_header and current_section["text"].strip():
                # Save previous section
                current_section["end"] = char_position
                current_section["text"] = current_section["text"].strip()
                if len(current_section["text"]) >= self.min_chunk_size:
                    sections.append(current_section.copy())

                # Start new section
                current_section = {
                    # Include the header line in the section text for preservation
                    "text": line + "\n",
                    "start": char_position,
                    "title": line_stripped,
                }
            else:
                current_section["text"] += line + "\n"

            char_position += len(line) + 1

        # Add final section
        if current_section["text"].strip():
            current_section["end"] = char_position
            current_section["text"] = current_section["text"].strip()
            if len(current_section["text"]) >= self.min_chunk_size:
                sections.append(current_section)

        # If no structure detected, create one big section
        if not sections:
            sections = [{"text": text, "start": 0, "end": len(text), "title": None}]
        else:
            # Propagate section titles into created chunks later via section_title field
            for s in sections:
                if "title" not in s:
                    s["title"] = None

        return sections

    def _split_large_section(self, text: str, section_title: str | None) -> list[str]:
        """Split a large section into smaller chunks.

        Uses sentence boundaries and token limits.
        """
        sentences = self._detect_sentences(text)
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # Check token limit
            if (
                current_chunk_tokens + sentence_tokens > self.chunk_size
                and current_chunk
            ):
                chunks.append(current_chunk.strip())

                # Add overlap if configured (in tokens)
                if self.chunk_overlap > 0:
                    overlap = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap + " " + sentence
                    current_chunk_tokens = self._count_tokens(current_chunk)
                else:
                    current_chunk = sentence
                    current_chunk_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_chunk_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Final safety check: split any chunks that still exceed max_chunk_size
        safe_chunks = []
        for chunk in chunks:
            if self._count_tokens(chunk) > self.max_chunk_size:
                # Force split using token-based splitting
                sub_chunks = self._force_split_by_tokens(chunk)
                safe_chunks.extend(sub_chunks)
            else:
                safe_chunks.append(chunk)

        return safe_chunks

    def _force_split_by_tokens(self, text: str) -> list[str]:
        """Force split text into chunks that fit within max_chunk_size tokens.

        This is a last-resort method for text that can't be split at sentence
        boundaries while staying within token limits.
        """
        tokens = self.encoder.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + self.max_chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)

            # Advance with overlap
            advance = self.max_chunk_size - self.chunk_overlap
            if advance <= 0:
                advance = self.max_chunk_size
            start += advance

        return chunks

    def _save_chunks(self, result: ChunkingResult, storage) -> None:
        """Save chunking result to JSON file."""
        if result.status != ChunkingStatus.SUCCESS or not result.chunks:
            logger.warning(f"Skipping save for failed result: {result.document_id}")
            return

        # Create output directory structure
        output_dir = self._resolve_output_dir(result, storage)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save chunks as JSON (array of DocumentChunk dicts per requirements)
        output_file = output_dir / "chunks.json"

        # Resume capability: if chunks already exist, skip writing
        if output_file.exists():
            logger.info(
                ("Chunks already exist for %s at %s. Skipping save."),
                result.document_id,
                output_file,
            )
            return

        try:
            chunks_data = [chunk.to_dict() for chunk in result.chunks]

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(result.chunks)} chunks to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save chunks for {result.document_id}: {e}")
            raise

    def _resolve_processed_documents_dir(self, storage) -> Path:
        """Resolve the path to processed/documents, using storage if available."""
        # Use storage if it provides get_path and returns a real Path
        if storage and hasattr(storage, "get_path") and callable(storage.get_path):
            try:
                candidate = storage.get_path("processed/documents")
                if isinstance(candidate, Path):
                    return candidate
            except Exception:
                pass

        # Fallback to config runtime.local_storage_path relative to config file
        base_dir = self.config.get("runtime", {}).get("local_storage_path", "data/")
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = (self.config_path.parent / base_path).resolve()
        return base_path / "processed" / "documents"

    def _resolve_output_dir(self, result: ChunkingResult, storage) -> Path:
        """Resolve output directory for chunks.json, using storage if available.

        Tries to mirror input path under processed/chunks/. If storage is not
        provided, compute root by finding the 'processed' segment in the
        source path; otherwise, fall back to runtime.local_storage_path.
        """
        # If storage is available, use it
        if storage and hasattr(storage, "get_path") and callable(storage.get_path):
            try:
                processed_root = storage.get_path("processed")
                if isinstance(processed_root, Path):
                    relative_path = result.source_path.relative_to(processed_root)
                    return storage.get_path("processed/chunks") / relative_path.parent
            except Exception:
                pass

        src_abs = result.source_path.resolve()
        parts = list(src_abs.parts)
        if "processed" in parts:
            idx = parts.index("processed")
            root = Path(*parts[:idx]) if idx > 0 else Path("/")
            # After 'processed', exclude filename
            relative_parent = Path(*parts[idx + 1 : -1])
            return (root / "processed" / "chunks" / relative_parent).resolve()

        # Fallback to runtime.local_storage_path
        base_dir = self.config.get("runtime", {}).get("local_storage_path", "data/")
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = (self.config_path.parent / base_path).resolve()
        # Try to mirror the documents path structure
        return (base_path / "processed" / "chunks").resolve()
