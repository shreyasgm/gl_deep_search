# OCR Pipeline for Academic Papers

This pipeline is designed to process academic papers, with a focus on handling both text-based and image-based PDFs. It includes tools for document preprocessing, language detection, OCR processing, and text extraction.

## Project Evolution

The project has evolved through several key phases:

1. **Initial Setup** (354eebe)
   - Initial observations and setup of OCR pipeline
   - Established basic project structure

2. **Pipeline Structure and Logging** (6e4003d)
   - Switched to logging for better debugging
   - Established preferred pipeline structure
   - Fixed column handling in data processing

3. **Language Detection and File Management** (b00f736)
   - Implemented robust language detection system
   - Added support for multiple languages including Arabic and French
   - Created system to handle multiple files per publication
   - Improved handling of edge cases in language detection
   - Added language source tracking for debugging
   - Fixed issues with language detection in filenames and abstracts

4. **Document Processing and Selection** (8dc8ce0)
   - Developed main document selector
   - Implemented scoring system for document selection
   - Rewrote components using polars for better performance
   - Fixed issues with document selection logic

5. **OCR Implementation** (357f019)
   - Added run_ocr.py with wrapper for parse_marker
   - Implemented initial OCR processing capabilities
   - Set up testing framework for OCR engines
   - Added support for multiple OCR tools (Llamaparse, Marker, Docling, Mistral OCR API, Unstructured, Deepdoctection)

6. **PDF Analysis and Processing** (7d852ec)
   - Added preflight_pdf.py for PDF type detection
   - Implemented decision point for OCR vs. text parsing
   - Enhanced text-based parsing capabilities
   - Added support for handling multi-part documents
   - Improved handling of text-based PDFs

## Components

### Core Scripts

1. **run_gl_directory_preprocessing.py**
   - Handles initial document processing
   - Manages file organization and preparation

2. **run_download_sample.py**
   - Downloads sample documents for testing
   - Manages document acquisition

3. **run_ocr.py**
   - Implements OCR processing
   - Integrates with various OCR engines

4. **run_parser.py**
   - Handles text extraction and parsing
   - Processes both OCR and native text content

5. **preflight_pdf.py**
   - Analyzes PDFs to determine if OCR is needed
   - Checks PDF type and content

## Features

- **Language Detection**: Robust system for detecting document language
- **Multi-file Handling**: Support for publications with multiple associated files
- **OCR Processing**: Integration with multiple OCR engines
- **Text Extraction**: Capable of handling both text-based and image-based PDFs
- **PDF Analysis**: Tools for determining PDF type and processing requirements

## Structure

PDF → (preflight_pdf.py) → Text-based?
    ├── Yes → Marker / Llamaparse / Docling → Markdown or JSON
    └── No  → OCR (e.g., Mistral OCR API) → Parsed text → Llamaparse / Docling


## Dependencies

The project's dependencies are managed through pyproject.toml. Please refer to the project's root directory for the complete list of dependencies.

## Ongoing

- Need to verify that my checkpoint file (preflight_pdf.py) works well, maybe with downloading all pdfs
- I need to compare between parsers
- I also need to compare between OCR engines
- Set up as a more modular pipeline with parallel processing w/ dask
- More robust Support for additional languages and document types; filtering things like presentation decks early. I had a langdetect thing upstream, maybe need to fix that because the latest excel seems to have dropped that column
