# Growth Lab Deep Search ETL Pipeline Orchestration Requirements

## Overview
The orchestration system should manage the complete end-to-end pipeline from scraping publication metadata to producing processed text files. The pipeline consists of four sequential components that must be executed in order with proper error handling and data validation.

## Component Analysis

### 1. Growth Lab Scraper (`backend/etl/scrapers/growthlab.py`)
**Purpose**: Scrapes publication metadata from the Growth Lab website
**Input**: None (scrapes from https://growthlab.hks.harvard.edu/publications)
**Output**: CSV file with publication metadata (`data/intermediate/growth_lab_publications.csv`)
**Key Features**:
- Handles pagination automatically
- Enriches publications with EndNote metadata
- Maintains publication versioning via content hashes
- Outputs ~400+ publications with file URLs

### 2. Growth Lab File Downloader (`backend/etl/utils/gl_file_downloader.py`)
**Purpose**: Downloads PDF and document files from publication URLs
**Input**: Publication CSV from scraper
**Output**: Downloaded files in `data/raw/documents/growthlab/<publication_id>/`
**Key Features**:
- Concurrent downloads with rate limiting
- File validation and resume capability
- Processes ~1000+ file URLs from publications

### 3. PDF Processor (`backend/etl/utils/pdf_processor.py`)
**Purpose**: Extracts text from downloaded PDF files using OCR
**Input**: PDF files from file downloader
**Output**: Processed text files in `data/processed/documents/growthlab/<publication_id>/`
**Key Features**:
- Uses unstructured library for OCR
- Preserves document structure and metadata
- Handles various PDF formats and languages

### 4. Lecture Transcripts Processor (`backend/etl/scripts/run_lecture_transcripts.py`)
**Purpose**: Processes lecture transcript files using OpenAI API
**Input**: Raw transcript files from `data/raw/lecture_transcripts/`
**Output**: Structured JSON files in `data/processed/lecture_transcripts/`
**Key Features**:
- Cleans and structures transcript content
- Extracts metadata using LLM
- Handles multiple lecture files

## Orchestration Requirements

### 1. Sequential Execution
The pipeline must execute components in strict order:
1. **Growth Lab Scraper** → generates publication metadata
2. **Growth Lab File Downloader** → downloads files based on metadata
3. **PDF Processor** → processes downloaded PDFs
4. **Lecture Transcripts Processor** → processes transcript files (independent)

### 2. Configuration Parameters
The orchestration system should accept the following parameters:

#### Global Parameters:
- `--config`: Path to configuration file (`backend/etl/config.yaml`)
- `--storage-type`: Storage backend (`local` or `cloud`)
- `--log-level`: Logging verbosity (`INFO`, `DEBUG`, `WARNING`)
- `--dry-run`: Preview actions without execution

#### Scraper Parameters:
- `--skip-scraping`: Skip scraping and use existing publication data
- `--scraper-concurrency`: Concurrent requests limit (default: 2)
- `--scraper-delay`: Delay between requests (default: 2.0s)

#### File Downloader Parameters:
- `--download-concurrency`: Concurrent downloads (default: 3)
- `--download-limit`: Max files to download (for testing)
- `--overwrite-files`: Overwrite existing downloaded files
- `--min-file-size`: Minimum file size threshold (default: 1KB)
- `--max-file-size`: Maximum file size threshold (default: 100MB)

#### PDF Processor Parameters:
- `--force-reprocess`: Reprocess existing PDF files
- `--ocr-language`: OCR language codes (default: ["eng"])
- `--extract-images`: Extract images from PDFs
- `--min-chars-per-page`: Minimum characters for valid extraction

#### Lecture Transcripts Parameters:
- `--transcripts-input`: Input directory for raw transcripts
- `--transcripts-limit`: Limit number of transcripts to process
- `--max-tokens`: Token limit for transcript processing

### 3. Data Flow Management
- **Publication Data**: ~400 publications with metadata
- **File URLs**: ~1000+ file URLs to download
- **Downloaded Files**: PDF, DOC, DOCX files (estimated 1-10GB total)
- **Processed Files**: Text files with extracted content

### 4. Error Handling & Recovery
- **Component Failures**: Continue pipeline if non-critical components fail
- **Partial Failures**: Track failed items and allow reprocessing
- **Retry Logic**: Implement exponential backoff for network operations
- **Checkpointing**: Save progress between components for recovery

### 5. Monitoring & Logging
- **Progress Tracking**: Show progress for each component
- **Statistics**: Success/failure counts, processing times, data volumes
- **Error Reporting**: Detailed error logs with context
- **Performance Metrics**: Processing rates, memory usage, storage usage

### 6. Storage Integration
- **Local Storage**: Default data directory structure
- **Cloud Storage**: GCS integration for SLURM environments
- **Path Management**: Consistent path handling across components
- **Space Management**: Monitor disk usage and cleanup temporary files

### 7. Validation & Quality Control
- **Data Validation**: Verify output files exist and are valid
- **Content Checks**: Ensure extracted text meets quality thresholds
- **Consistency Checks**: Validate data flow between components
- **Output Verification**: Confirm expected file counts and sizes

## Orchestration Command Interface

```bash
# Full pipeline execution
python -m backend.etl.orchestrator --config config.yaml

# With specific parameters
python -m backend.etl.orchestrator \
  --config config.yaml \
  --storage-type local \
  --download-concurrency 5 \
  --download-limit 100 \
  --force-reprocess

# Skip scraping and use existing data
python -m backend.etl.orchestrator \
  --skip-scraping \
  --overwrite-files

# Dry run to preview actions
python -m backend.etl.orchestrator --dry-run
```

## Output Requirements
At completion, the orchestration should produce:
- **Metadata**: Updated publication CSV with processing status
- **Downloaded Files**: Organized in publication-specific directories
- **Processed Text**: Extracted text files ready for indexing
- **Transcripts**: Structured lecture transcript data
- **Logs**: Comprehensive execution logs and statistics
- **Report**: Summary of pipeline execution with metrics

## Exclusions
- **OpenAlex Components**: Explicitly excluded due to current issues
- **Vector Database**: Embedding and indexing handled separately
- **Frontend Integration**: UI components not part of ETL orchestration

## Code Structure for Testing

### Core Data Types

#### ComponentStatus (Enum)
```python
class ComponentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
```

#### ComponentResult (dataclass)
**Purpose**: Represents the result of executing a single pipeline component

**Fields**:
- `component_name: str` - Name of the component that was executed
- `status: ComponentStatus` - Current status of the component
- `start_time: float | None` - Timestamp when component started (None if not started)
- `end_time: float | None` - Timestamp when component finished (None if not finished)
- `error: str | None` - Error message if component failed (None if no error)
- `metrics: dict[str, Any]` - Component-specific metrics and statistics
- `output_files: list[Path]` - List of files created by the component

**Properties**:
- `duration: float | None` - Calculated execution duration in seconds (None if start/end times missing)

#### OrchestrationConfig (dataclass)
**Purpose**: Configuration object containing all orchestration parameters

**Key Fields**:
- `config_path: Path` - Path to YAML configuration file
- `storage_type: str | None` - Storage backend ("local" or "cloud")
- `log_level: str` - Logging verbosity level (default: "INFO")
- `dry_run: bool` - Preview mode flag (default: False)
- `skip_scraping: bool` - Skip scraper component flag (default: False)
- `scraper_concurrency: int` - Concurrent requests for scraper (default: 2)
- `scraper_delay: float` - Delay between scraper requests (default: 2.0)
- `download_concurrency: int` - Concurrent downloads (default: 3)
- `download_limit: int | None` - Maximum files to download (None for unlimited)
- `overwrite_files: bool` - Overwrite existing files flag (default: False)
- `min_file_size: int` - Minimum file size threshold (default: 1024)
- `max_file_size: int` - Maximum file size threshold (default: 100_000_000)
- `force_reprocess: bool` - Force reprocessing of existing files (default: False)
- `ocr_language: list[str]` - OCR language codes (default: ["eng"])
- `extract_images: bool` - Extract images from PDFs (default: False)
- `min_chars_per_page: int` - Minimum characters for valid extraction (default: 100)
- `transcripts_input: Path | None` - Input directory for transcripts (None uses default)
- `transcripts_limit: int | None` - Limit number of transcripts to process (None for all)
- `max_tokens: int | None` - Token limit for transcript processing (None for unlimited)

### Core Classes

#### ETLOrchestrator
**Purpose**: Main orchestration class that manages the complete ETL pipeline

**Constructor**:
```python
def __init__(self, config: OrchestrationConfig) -> None
```
**Expected Behavior**: Initializes orchestrator with configuration, sets up logging, loads ETL config from YAML file, initializes storage backend

**Key Methods**:

##### `run_pipeline(self) -> list[ComponentResult]`
**Purpose**: Execute the complete ETL pipeline in sequence
**Returns**: List of ComponentResult objects for each component
**Expected Behavior**:
- Executes components in order: scraper → file downloader → PDF processor → lecture transcripts
- Handles component failures and continues pipeline where possible
- Stops pipeline if critical components fail
- Generates final execution report
- Returns results for all attempted components

##### `_execute_component(self, name: str, func) -> ComponentResult`
**Purpose**: Execute a single component with error handling and monitoring
**Parameters**:
- `name: str` - Human-readable component name
- `func` - Async function to execute for the component
**Returns**: ComponentResult with execution details
**Expected Behavior**:
- Creates ComponentResult with PENDING status
- Sets start_time and changes status to RUNNING
- Calls the component function
- Handles exceptions and sets FAILED status with error message
- Sets end_time and calculates duration
- Logs execution progress and results

##### `_run_scraper(self, result: ComponentResult) -> None`
**Purpose**: Execute the Growth Lab scraper component
**Parameters**: `result: ComponentResult` - Result object to update
**Expected Behavior**:
- Skip if skip_scraping is True (sets SKIPPED status)
- Create GrowthLabScraper instance with config
- Call scraper.update_publications()
- Update result.metrics with publication counts and file URLs
- Set result.output_files to publications CSV path

##### `_run_file_downloader(self, result: ComponentResult) -> None`
**Purpose**: Execute the file downloader component
**Parameters**: `result: ComponentResult` - Result object to update
**Expected Behavior**:
- Create FileDownloader instance with config
- Check for publications CSV from scraper
- Download files based on publication URLs
- Update result.metrics with download statistics
- Set result.output_files to downloaded file paths

##### `_run_pdf_processor(self, result: ComponentResult) -> None`
**Purpose**: Execute the PDF processor component
**Parameters**: `result: ComponentResult` - Result object to update
**Expected Behavior**:
- Create PDFProcessor instance with config
- Find downloaded PDF files in storage
- Process each PDF file to extract text
- Update result.metrics with processing statistics
- Set result.output_files to processed text file paths

##### `_run_lecture_transcripts(self, result: ComponentResult) -> None`
**Purpose**: Execute the lecture transcripts processor component
**Parameters**: `result: ComponentResult` - Result object to update
**Expected Behavior**:
- Check for transcript input directory
- Process transcript files (currently placeholder implementation)
- Update result.metrics with processing statistics
- Set result.output_files to processed transcript paths

##### `_simulate_pipeline(self) -> list[ComponentResult]`
**Purpose**: Simulate pipeline execution for dry run mode
**Returns**: List of ComponentResult objects with simulated results
**Expected Behavior**:
- Create ComponentResult for each component with COMPLETED status
- Set simulated metrics and timing
- Log dry run actions without executing components

##### `_generate_report(self) -> None`
**Purpose**: Generate and save final execution report
**Expected Behavior**:
- Log summary of all component results
- Calculate total execution time
- Save detailed JSON report to storage
- Include component status, metrics, and output files

### Standalone Functions

#### `create_argument_parser() -> argparse.ArgumentParser`
**Purpose**: Create command-line argument parser for the orchestrator
**Returns**: Configured ArgumentParser instance
**Expected Behavior**:
- Define all CLI arguments matching OrchestrationConfig fields
- Include help text and examples
- Set appropriate defaults and validation

#### `main() -> None` (async)
**Purpose**: Main entry point for the orchestrator
**Expected Behavior**:
- Parse command-line arguments
- Create OrchestrationConfig from arguments
- Initialize ETLOrchestrator and run pipeline
- Exit with appropriate code based on results

## Implementation Notes
- The orchestration system should be implemented as a new module: `backend/etl/orchestrator.py`
- Configuration should extend the existing `backend/etl/config.yaml` structure
- Progress tracking should use existing logging infrastructure with loguru
- Error handling should be comprehensive but allow partial pipeline completion
- The system should be designed to run both locally and in SLURM environments
