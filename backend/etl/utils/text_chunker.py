"""
Text Chunking System for Growth Lab Deep Search.

This module provides text chunking functionality that transforms processed text
documents into semantically meaningful chunks for vector embeddings and retrieval.
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

import yaml
from loguru import logger


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
    chunk_size: int  # Actual chunk size in characters

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
    """Main text chunking system with multiple strategies."""

    def __init__(self, config_path: Path):
        """Initialize text chunker with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        raw_config = self.config.get("file_processing", {}).get("chunking", {})

        # Defaults
        defaults = {
            "strategy": "hybrid",
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "min_chunk_size": 100,
            "max_chunk_size": 2000,
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
        self.chunk_size = merged["chunk_size"]
        self.chunk_overlap = merged["chunk_overlap"]
        self.min_chunk_size = merged["min_chunk_size"]
        self.max_chunk_size = merged["max_chunk_size"]
        self.preserve_structure = merged["preserve_structure"]
        self.respect_sentences = merged["respect_sentences"]
        self.structure_markers = merged["structure_markers"]

        logger.info(f"TextChunker initialized with strategy: {self.strategy}")

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
                    return chunks
                else:
                    logger.warning(f"Strategy {strategy_name} produced no chunks")

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {e}")
                continue

        # If all strategies fail, return empty list
        logger.error("All chunking strategies failed")
        return []

    def chunk_fixed_size(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create fixed-size chunks with overlap."""
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

        start = 0
        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))

            # Extract chunk text without stripping to preserve exact overlap
            chunk_text = clean_text[start:end]

            # Skip chunks that are too small, but always include first chunk if short
            if (
                len(chunk_text) < self.min_chunk_size
                and len(clean_text) >= self.min_chunk_size
            ):
                break

            # Find which pages this chunk covers
            chunk_pages = self._find_pages_for_position(start, end, page_info)

            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                source_document_id=document_id,
                source_file_path=source_path,
                chunk_index=chunk_index,
                text_content=chunk_text,
                character_start=start,
                character_end=end,
                page_numbers=chunk_pages,
                section_title=None,
                metadata={"strategy": "fixed"},
                created_at=datetime.now(),
                chunk_size=len(chunk_text),
            )

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            next_start = start + self.chunk_size - self.chunk_overlap
            # Ensure exactly chunk_overlap characters of overlap when possible
            if next_start < end:
                start = next_start
            else:
                start = end

        return chunks

    def chunk_by_sentences(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create chunks respecting sentence boundaries."""
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
        current_start = 0
        sentence_start = 0

        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = current_chunk.strip()
                if len(chunk_text) >= self.min_chunk_size:
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
                    )

                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with overlap if needed
                if self.chunk_overlap > 0 and chunks:
                    # Take last part of previous chunk as overlap
                    overlap_text = (
                        chunk_text[-self.chunk_overlap :]
                        if len(chunk_text) >= self.chunk_overlap
                        else chunk_text
                    )
                    current_chunk = overlap_text + " " + sentence
                    current_start = current_start + len(chunk_text) - len(overlap_text)
                else:
                    current_chunk = sentence
                    current_start = sentence_start
            else:
                current_chunk += sentence
                if not current_chunk.strip():
                    current_start = sentence_start

            sentence_start += len(sentence)

        # Handle final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunk_text = current_chunk.strip()
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
            )

            chunks.append(chunk)

        return chunks

    def chunk_by_structure(
        self, text: str, source_path: Path | None = None, document_id: str | None = None
    ) -> list[DocumentChunk]:
        """Create chunks based on document structure."""
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

            if len(section_text) < self.min_chunk_size:
                continue

            # If section is too large, split it further
            if len(section_text) > self.max_chunk_size:
                # Fall back to sentence-based chunking for this section
                sub_chunks = self._split_large_section(section_text, section_title)
                for sub_chunk_text in sub_chunks:
                    if len(sub_chunk_text.strip()) >= self.min_chunk_size:
                        chunk_pages = self._find_pages_for_text(
                            sub_chunk_text, page_info
                        )

                        chunk = DocumentChunk(
                            chunk_id=f"{document_id}_chunk_{chunk_index:04d}",
                            source_document_id=document_id,
                            source_file_path=source_path,
                            chunk_index=chunk_index,
                            text_content=sub_chunk_text.strip(),
                            character_start=section["start"],
                            character_end=section["end"],
                            page_numbers=chunk_pages,
                            section_title=section_title,
                            metadata={
                                "strategy": "structure",
                                "large_section_split": True,
                            },
                            created_at=datetime.now(),
                            chunk_size=len(sub_chunk_text.strip()),
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
        """Chunk a section while respecting size and sentence constraints."""
        chunks: list[DocumentChunk] = []
        section_text = section["text"].strip()
        section_title = section.get("title")

        if len(section_text) < self.min_chunk_size:
            return chunks

        if len(section_text) <= self.chunk_size:
            # Section fits in one chunk
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
            )

            chunks.append(chunk)
        else:
            # Split section using sentence boundaries
            sentences = self._detect_sentences(section_text)
            current_chunk = ""
            current_start = section["start"]
            chunk_idx = start_index

            for sentence in sentences:
                if (
                    len(current_chunk) + len(sentence) > self.chunk_size
                    and current_chunk
                ):
                    # Save current chunk
                    chunk_text = current_chunk.strip()
                    if len(chunk_text) >= self.min_chunk_size:
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
                        )

                        chunks.append(chunk)
                        chunk_idx += 1

                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = (
                            chunk_text[-self.chunk_overlap :]
                            if len(chunk_text) >= self.chunk_overlap
                            else chunk_text
                        )
                        current_chunk = overlap_text + " " + sentence
                        current_start = (
                            current_start + len(chunk_text) - len(overlap_text)
                        )
                    else:
                        current_chunk = sentence
                        current_start = current_start + len(chunk_text)
                else:
                    current_chunk += sentence

            # Handle final chunk
            if (
                current_chunk.strip()
                and len(current_chunk.strip()) >= self.min_chunk_size
            ):
                chunk_text = current_chunk.strip()
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
                    metadata={"strategy": "hybrid", "approach": "section_sentences"},
                    created_at=datetime.now(),
                    chunk_size=len(chunk_text),
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
        """Split a large section into smaller chunks using sentence boundaries."""
        sentences = self._detect_sentences(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Add overlap if configured
                if self.chunk_overlap > 0:
                    overlap = (
                        current_chunk[-self.chunk_overlap :]
                        if len(current_chunk) >= self.chunk_overlap
                        else current_chunk
                    )
                    current_chunk = overlap + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

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
