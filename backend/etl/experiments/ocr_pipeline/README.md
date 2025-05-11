# OCR Pipeline

This directory contains scripts for processing PDFs from the GrowthLab dataset, using OCR when necessary.

## Overview

The OCR pipeline processes PDFs through several steps:

1. Download PDFs from a list of URLs
2. Check if each PDF is text-based or requires OCR
3. Perform OCR on documents that need it
4. Save extracted text for further processing

## Modular Script Approach

The entire pipeline is now contained in a single modular script (`ocr_pipeline.py`) that allows you to:

1. Run individual components separately for testing
2. Execute the complete pipeline in one command
3. Reuse code between different parts of the pipeline

## Usage

### Complete Pipeline

Run the entire pipeline with a single command:

```bash
python ocr_pipeline.py pipeline --csv data/publevel.csv --output-dir processed_papers
```

### Individual Components

You can also run individual components separately:

#### 1. Download Documents

```bash
python ocr_pipeline.py download --csv data/publevel.csv --output-dir downloaded_papers
```

Options:
- `--csv`: Path to CSV file with document data (default: data/publevel.csv)
- `--output-dir`: Directory to store downloaded files (default: downloaded_papers)
- `--concurrency`: Number of concurrent downloads (default: 3)
- `--all`: Download all files, not just samples

#### 2. Analyze PDFs

```bash
python ocr_pipeline.py analyze downloaded_papers --output analysis_results.json
```

Options:
- `--output`: Path to save analysis results (default: analysis_results.json)

#### 3. Run OCR

```bash
python ocr_pipeline.py ocr --analysis-file analysis_results.json --engine tesseract --output-dir ocr_results
```

Options:
- `--analysis-file`: JSON file with analysis results
- `--engine`: OCR engine to use (default: tesseract)
- `--output-dir`: Directory to store OCR results (default: ocr_results)

### Pipeline Options

When running the full pipeline:

```bash
python ocr_pipeline.py pipeline [options]
```

Options:
- `--csv`: Path to CSV file with document data (default: data/publevel.csv)
- `--output-dir`: Directory to store processed files (default: processed_papers)
- `--engine`: OCR engine to use (default: tesseract)
- `--concurrency`: Number of concurrent operations (default: 3)
- `--all`: Process all files, not just samples

## Output

The pipeline generates the following outputs:

1. Downloaded PDF files in the output directory, organized by paper_id
2. OCR results in JSON format for non-text-based PDFs
3. A summary JSON file (`pipeline_results.json`) with information about each processed document

## Development

To add new OCR engines or extend the pipeline functionality, modify the `ocr_pipeline.py` script.

### Key Components in the Module

The script is organized into functional modules:

- **Download Module**: Functions for downloading PDFs from URLs
- **PDF Analysis Module**: Functions for analyzing whether PDFs are text-based
- **OCR Module**: Functions for performing OCR on PDFs
- **Full Pipeline**: Functions that orchestrate the entire workflow

### Adding New OCR Engines

To add a new OCR engine, extend the `ocr_document()` function in the OCR module section of the script.
