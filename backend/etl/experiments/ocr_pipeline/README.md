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
   - Rewritten components using polars for better performance
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

7. **Preset-Based Parser System** (Current)
   - Implemented consistent preset-based interfaces for all parsers
   - Added configurable presets for different use cases (fast, accurate, multilingual, etc.)
   - Created modular parser system with standardized error handling
   - Added performance tracking and metadata collection across all parsers

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

### Parser Modules

The pipeline now includes a comprehensive preset-based parser system:

#### **Marker** (`module_marker.py`)
High-quality PDF parsing with optional LLM enhancement.

**Presets:** baseline, fast, use_llm, ocr_and_llm, ocr_and_llm_math  
**Models:** gpt-4o-mini (default), gpt-4.1 variants, gemini-flash-lite, claude variants

#### **Unstructured** (`module_unstructured.py`)
Layout-aware PDF parsing with multiple strategies.

**Presets:** baseline, informed_layout_base, informed_layout_tables, informed_layout_tables_and_images, ocr_base, ocr_tables, ocr_tables_and_images

#### **LlamaParse** (`module_llamaparse.py`)
Fast, efficient PDF parsing with multiple output formats.

**Presets:** baseline, fast, detailed, parallel, json_output, text_only

#### **Mistral OCR** (`module_mistral.py`)
Cloud-based OCR with high accuracy.

**Presets:** baseline, fast, detailed, multilingual, json_output, text_only

#### **Tesseract** (`module_tesseract.py`)
Local OCR with extensive configuration options.

**Presets:** baseline, fast, accurate, multilingual, table_focused, handwritten, math

#### **EasyOCR** (`module_easyocr.py`)
Deep learning-based OCR with confidence scoring.

**Presets:** baseline, fast, accurate, multilingual, table_focused, handwritten

#### **Docling** (`module_docling.py`)
Document intelligence with flexible extraction methods.

**Presets:** baseline, fast, detailed, multilingual, table_focused, text_only, ocr_heavy

## Features

- **Language Detection**: Robust system for detecting document language
- **Multi-file Handling**: Support for publications with multiple associated files
- **OCR Processing**: Integration with multiple OCR engines
- **Text Extraction**: Capable of handling both text-based and image-based PDFs
- **PDF Analysis**: Tools for determining PDF type and processing requirements
- **Preset System**: Configurable presets for different use cases and performance requirements
- **Consistent Interface**: Standardized API across all parsers with unified error handling
- **Performance Tracking**: Built-in timing and metadata collection

## Usage Examples

### Basic Parser Usage

```python
from pdf_module import PARSERS

# Use any parser with default preset
result = PARSERS["marker"](
    pdf_path="document.pdf",
    preset="baseline"
)

print(f"Success: {result['success']}")
print(f"Processing time: {result['processing_time']:.2f}s")
print(f"Output: {result['output_path']}")
```

### Advanced Usage with Custom Presets

```python
from pdf_module import marker_parser_adapter, unstructured_parser_adapter

# Marker with specific model
result = marker_parser_adapter(
    pdf_path="document.pdf",
    preset="use_llm",
    model="gpt-4o-mini",
    output_path="output.md"
)

# Unstructured with custom options
result = unstructured_parser_adapter(
    pdf_path="document.pdf",
    preset="informed_layout_tables",
    extra_opts={"languages": ["eng", "fra"]}
)
```

### Batch Processing

```python
from pdf_module import PARSERS
import glob

# Process multiple PDFs with different parsers
pdf_files = glob.glob("*.pdf")
parsers_to_test = ["marker", "unstructured", "tesseract"]

for pdf_file in pdf_files:
    for parser_name in parsers_to_test:
        result = PARSERS[parser_name](
            pdf_path=pdf_file,
            preset="baseline"
        )
        print(f"{pdf_file} - {parser_name}: {result['success']}")
```

## Structure

PDF → (preflight_pdf.py) → Text-based?
    ├── Yes → Marker / Llamaparse / Docling → Markdown or JSON
    └── No  → OCR (e.g., Mistral OCR API) → Parsed text → Llamaparse / Docling

### Preset System Architecture

```
pdf_module.py (main interface)
├── module_marker.py (preset configurations)
├── module_unstructured.py (preset configurations)
├── module_llamaparse.py (preset configurations)
├── module_mistral.py (preset configurations)
├── module_tesseract.py (preset configurations)
├── module_easyocr.py (preset configurations)
└── module_docling.py (preset configurations)
```

## Configuration

### Environment Variables

Some parsers require API keys:

```bash
# For Marker (OpenAI)
export OPENAI_API_KEY="your-openai-key"

# For Marker (Gemini)
export GOOGLE_API_KEY="your-google-key"

# For Marker (Claude)
export CLAUDE_API_KEY="your-claude-key"

# For Mistral
export MISTRAL_API_KEY="your-mistral-key"
```

### Performance Considerations

- **Fast presets**: Optimized for speed, may sacrifice accuracy
- **Accurate presets**: Optimized for quality, may be slower
- **Baseline presets**: Balanced approach
- **Local vs Cloud**: Tesseract/EasyOCR (local) vs Mistral/Marker (cloud)

## Dependencies

The project's dependencies are managed through pyproject.toml. Please refer to the project's root directory for the complete list of dependencies.

## Testing

Use the test script to validate all parsers:

```bash
python test_presets.py
```

This will list all available presets and test each parser with different configurations.

## Ongoing

- I need to see if my new modular setup works and start building the 'run' pipeline from scratch
- set up notes to compare between parsers seems we dont need ocr
- use the dask parallelisation setup you used before but on the running script
- More robust Support for additional languages and document types; filtering things like presentation decks early. I had a langdetect thing upstream, maybe need to fix that because the latest excel seems to have dropped that column
- **Integration of preset system into main pipeline**
- **Performance benchmarking across different presets**
- **Custom preset creation for specific document types**
