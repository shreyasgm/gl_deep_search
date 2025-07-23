# Text Chunking System Requirements - MVP Version

## Overview

The text chunking system transforms processed text documents into semantically meaningful chunks for vector embeddings and retrieval. This MVP focuses on reliable, simple chunking that works across different document types (PDFs, lecture transcripts, etc.) with graceful failure handling.

## System Context & Integration Points

### Current ETL Pipeline (orchestrator.py)
The chunking system will be integrated as a new component in the existing pipeline sequence:
```python
components = [
    ("Growth Lab Scraper", self._run_scraper),
    ("Growth Lab File Downloader", self._run_file_downloader),
    ("PDF Processor", self._run_pdf_processor),
    ("Lecture Transcripts Processor", self._run_lecture_transcripts),
    ("Text Chunker", self._run_text_chunker),  # NEW COMPONENT
]
```

### Input Sources
The system must handle processed text from multiple sources:

1. **PDF Processing Output** (`backend/etl/utils/pdf_processor.py`):
   - **Location**: `data/processed/documents/growthlab/{document_id}/processed_text.txt`
   - **Format**: Markdown-like text with structure preservation
   - **Metadata**: May include page numbers, section headers
   - **Current data**: 3 sample processed PDFs available

2. **Lecture Transcripts** (`backend/etl/scripts/run_lecture_transcripts.py`):
   - **Location**: `data/processed/transcripts/{lecture_id}/processed_transcript.txt`
   - **Format**: Plain text or structured text output from LLM processing
   - **Metadata**: No page numbers, may have lecture titles/topics
   - **Content**: Long-form transcribed lectures

3. **Future Text Sources**: Any processed text file that follows the pattern:
   - Plain text format
   - Optional metadata in accompanying JSON files
   - UTF-8 encoding

## MVP Functional Requirements

### Core Functionality

#### FR1: Multiple Chunking Strategies
- **Input**: Any processed text file (PDF, transcript, etc.)
- **Output**: JSON file with chunks and metadata
- **Available strategies**:
  - `fixed`: Fixed-size character-based chunking
  - `sentence`: Sentence boundary-aware chunking
  - `structure`: Document structure-aware chunking (respects headers, sections)
  - `hybrid`: Combines fixed-size with sentence and structure awareness (recommended)
- **Configuration**: All parameters configurable via `config.yaml`
- **Fallback hierarchy**: hybrid → sentence → fixed (graceful degradation)

#### FR2: Graceful Metadata Handling
- **Required metadata**: `source_file_path`, `chunk_index`, `text_content`, `created_at`
- **Optional metadata**: `page_numbers`, `section_title` (only if available in source)
- **Failure mode**: System continues if optional metadata extraction fails
- **Source tracking**: Always preserve link to original processed text file

#### FR3: Simple Batch Processing
- **Input discovery**: Scan `data/processed/` directories for text files
- **Sequential processing**: Process one document at a time (avoid concurrency complexity)
- **Error isolation**: Continue processing if individual files fail
- **Resume capability**: Skip files that already have chunks output

#### FR4: Comprehensive Configuration
- **Config source**: Read from existing `backend/etl/config.yaml`
- **Strategy selection**: Choose chunking approach via config
- **Size parameters**: Configurable chunk size, overlap, min/max limits
- **Behavior options**: Structure preservation, sentence boundary respect
- **Override capability**: Allow command-line parameter override for testing

### Data Models

#### DocumentChunk
```python
@dataclass
class DocumentChunk:
    chunk_id: str                    # Unique identifier
    source_document_id: str          # Reference to original PDF
    source_file_path: Path           # Path to processed text file
    chunk_index: int                 # Sequential position in document
    text_content: str                # The actual chunk text
    character_start: int             # Start position in original text
    character_end: int               # End position in original text
    page_numbers: List[int]          # Pages covered by this chunk
    section_title: Optional[str]     # Parent section if available
    metadata: Dict[str, Any]         # Additional metadata
    created_at: datetime             # Processing timestamp
    chunk_size: int                  # Actual chunk size in characters
```

#### ChunkingResult
```python
@dataclass
class ChunkingResult:
    document_id: str
    source_path: Path
    chunks: List[DocumentChunk]
    total_chunks: int
    processing_time: float
    status: ChunkingStatus
    error_message: Optional[str] = None
```

## Technical Requirements

### TR1: Implementation Architecture
- **Module location**: `backend/etl/utils/text_chunker.py`
- **Class structure**: `TextChunker` class with configurable chunking strategies
- **Integration point**: Called from ETL orchestrator after PDF processing
- **Dependencies**: Leverage existing utilities (`retry.py`, logging, config management)

### TR2: Multiple Chunking Strategies Implementation
- **Fixed strategy**: Pure character-based chunking with configurable overlap
- **Sentence strategy**: Sentence boundary detection using simple regex patterns (`.`, `!`, `?`, `;`)
- **Structure strategy**: Detect headers/sections via markdown-like patterns and formatting
- **Hybrid strategy**: Combine all approaches with intelligent fallbacks (recommended default)
- **Simple implementation**: Use regex and string operations (no heavy NLP dependencies)
- **Graceful degradation**: Fall back to simpler strategies if complex ones fail

### TR3: Performance Requirements
- **Processing speed**: Handle 100+ page documents efficiently
- **Memory usage**: Stream processing for large documents
- **Concurrency**: Parallel processing of multiple documents
- **Scalability**: Support for batch processing 1000+ documents

### TR4: Error Handling
- **Graceful degradation**: Fall back to simpler chunking if advanced strategies fail
- **Comprehensive logging**: Detailed error reporting with context
- **Recovery mechanisms**: Retry logic for transient failures
- **Validation**: Ensure chunk quality and completeness

### TR5: Storage Integration
- **Output format**: JSON files with chunk metadata and text
- **File organization**: Structured directory layout matching input
- **Storage backend**: Compatible with existing local/GCS storage system
- **Compression**: Optional compression for large chunk datasets

## Integration Requirements

### IR1: ETL Orchestrator Integration
- **Pipeline step**: Add chunking step between PDF processing and embeddings
- **Configuration**: Read settings from `backend/etl/config.yaml`
- **Progress reporting**: Integrate with orchestrator's monitoring system
- **Error propagation**: Report failures to orchestrator

### IR2: Comprehensive Configuration Schema
Add to `backend/etl/config.yaml`:
```yaml
file_processing:
  # ... existing ocr and embedding config ...
  chunking:
    enabled: true
    strategy: "hybrid"             # Options: fixed, sentence, structure, hybrid
    chunk_size: 1000               # target characters per chunk
    chunk_overlap: 200             # characters of overlap between chunks
    min_chunk_size: 100            # minimum viable chunk size
    max_chunk_size: 2000           # maximum chunk size before forced split
    preserve_structure: true       # respect document hierarchy (headers, sections)
    respect_sentences: true        # prefer sentence boundaries for splits
    structure_markers:             # patterns for detecting document structure
      - "^#{1,6}\\s+"              # markdown headers
      - "^\\d+\\.\\s+"             # numbered lists
      - "^[A-Z][A-Z\\s]+:"         # section labels like "INTRODUCTION:"
```

### IR3: Simple Directory Structure
```
data/processed/chunks/
├── documents/
│   └── growthlab/
│       ├── gl_url_39aabeaa471ae241/
│       │   └── chunks.json      # Array of DocumentChunk objects
│       └── gl_url_3e115487b5f521a6/
│           └── chunks.json
└── transcripts/
    └── lecture_001/
        └── chunks.json
```

## MVP Quality Requirements

### QR1: Text Quality (Minimal Viable)
- **Sentence completion**: Avoid breaking mid-sentence when possible
- **Minimum chunk size**: No chunks smaller than 50 characters
- **Maximum chunk size**: Respect configured limit (1000 chars + overlap)
- **Overlap continuity**: Ensure overlapping text provides context

### QR2: Metadata Reliability
- **Source tracking**: Always preserve accurate source file path
- **Chunk ordering**: Maintain sequential chunk indices
- **Optional metadata**: Gracefully handle missing page numbers/sections
- **Error transparency**: Clear error messages when processing fails

### QR3: MVP Performance Standards
- **Processing time**: < 5 seconds per processed text file
- **Memory usage**: < 100MB for single document processing
- **Error tolerance**: System continues if individual files fail
- **Completion rate**: Process 90%+ of valid input files successfully

## MVP Testing Strategy

### Integration Tests (Primary Focus)
- **End-to-end test**: Process existing 3 sample PDFs through chunking
- **Orchestrator integration**: Test that chunking step runs in pipeline
- **Output validation**: Verify chunks.json files are created correctly
- **Configuration test**: Test reading config from YAML

### Minimal Unit Tests
- **Basic chunking**: Test chunk creation with simple text input
- **Sentence boundary**: Test sentence-aware splitting vs character splitting
- **Error handling**: Test behavior with empty/invalid input
- **Metadata handling**: Test optional metadata extraction

### Test Data & Scenarios
- **Existing data**: Use 3 processed PDFs in `data/raw/documents/growthlab/`
- **Transcript simulation**: Create mock transcript text for testing
- **Edge cases**: Very short text (< 100 chars), very long text (> 10k chars)
- **Error cases**: Missing files, malformed text, permission issues

### Testing Philosophy
- **Real data over mocks**: Test with actual processed text files
- **Integration over isolation**: Focus on end-to-end functionality
- **Manual verification**: Inspect a few chunks.json files manually
- **Performance observation**: Time the processing of sample documents

## MVP Implementation Plan

### Week 1: Core Implementation
1. **Day 1**: Create `backend/etl/utils/text_chunker.py` with fixed-size chunking
2. **Day 2**: Add sentence boundary detection and sentence-based chunking
3. **Day 3**: Implement structure-aware chunking with section detection
4. **Day 4**: Create hybrid strategy that combines all approaches
5. **Day 5**: Add comprehensive configuration and orchestrator integration

### Week 2: Testing & Refinement
1. **Day 1-2**: Write integration tests with sample data
2. **Day 3**: Test with lecture transcripts (create mock data if needed)
3. **Day 4**: Handle edge cases and error scenarios
4. **Day 5**: Performance testing and optimization

### Week 3: Production Readiness
1. **Day 1-2**: End-to-end pipeline testing
2. **Day 3**: Documentation and code review
3. **Day 4-5**: Ready for embeddings system integration

## MVP Success Criteria

### Development Success
- [ ] `TextChunker` class processes existing 3 sample PDFs successfully
- [ ] `chunks.json` files created in correct directory structure
- [ ] Chunking step integrated into orchestrator pipeline
- [ ] Configuration loaded from `config.yaml` without errors

### Quality Success
- [ ] Chunks are readable and coherent
- [ ] No chunks smaller than min_chunk_size
- [ ] Source file paths correctly preserved
- [ ] Processing completes without crashing on sample data

### Integration Success
- [ ] JSON output format ready for embeddings system
- [ ] Local storage works (no cloud complexity for MVP)
- [ ] Logging follows existing project patterns
- [ ] Pipeline can be run end-to-end with chunking enabled

## Implementation Details

### Required Files to Create
1. **`backend/etl/utils/text_chunker.py`** - Main chunking implementation
2. **Modify `backend/etl/orchestrator.py`** - Add `_run_text_chunker` method
3. **Update `backend/etl/config.yaml`** - Add chunking configuration section
4. **`backend/tests/etl/test_text_chunker.py`** - Integration tests

### External Dependencies
- **No new dependencies required** for MVP
- Use standard library: `json`, `pathlib`, `dataclasses`, `datetime`
- Existing project tools: `loguru`, `yaml`, `pydantic`

### Key Implementation Functions
```python
class TextChunker:
    def __init__(self, config_path: Path)
    def process_all_documents(self, storage) -> List[ChunkingResult]
    def process_single_document(self, text_file_path: Path) -> ChunkingResult
    def create_chunks(self, text: str, strategy: str) -> List[DocumentChunk]

    # Strategy implementations
    def chunk_fixed_size(self, text: str) -> List[DocumentChunk]
    def chunk_by_sentences(self, text: str) -> List[DocumentChunk]
    def chunk_by_structure(self, text: str) -> List[DocumentChunk]
    def chunk_hybrid(self, text: str) -> List[DocumentChunk]

    # Utility methods
    def detect_sentences(self, text: str) -> List[str]
    def detect_structure(self, text: str) -> List[dict]  # sections with positions
    def apply_overlap(self, chunks: List[str]) -> List[str]
    def validate_chunk_size(self, chunk: str) -> bool
```

### Failure Points & Mitigation
1. **No processed text files**: Log warning, return empty results
2. **Malformed text files**: Skip file, log error, continue with others
3. **Very short documents**: Create single chunk if above min_chunk_size
4. **Strategy failure**: Fall back to simpler strategy (hybrid → sentence → fixed)
5. **Structure detection failure**: Fall back to sentence or fixed chunking
6. **Sentence detection failure**: Fall back to fixed-size chunking
7. **Missing optional metadata**: Set to None, continue processing
8. **Storage write failure**: Log error, mark document as failed
9. **Configuration errors**: Use default values, log warning

### Future Extensions (Post-MVP)
- Advanced NLP-based sentence detection (spaCy, NLTK)
- Semantic chunking with embeddings similarity
- Parallel processing for large document batches
- Cloud storage integration (GCS backend)
- Incremental updates for changed documents
- Custom chunking strategies via plugin architecture
- Real-time chunking APIs for dynamic content
