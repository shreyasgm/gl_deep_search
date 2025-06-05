#!/usr/bin/env python
# process_single_pdf.py

import argparse
import json
import logging
import os
import time
from pathlib import Path

from pdf_modules import is_pdf_text_based, PARSERS, OCR_ENGINES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_pdf(
    pdf_path: Path,
    text_parser: str = "marker",
    ocr_engine: str = "mistral",
    output_dir: Path = None,
    cache: bool = True
):
    """Process a single PDF file."""
    start_time = time.time()
    
    # Extract the parent folder name and PDF name
    parent_folder = pdf_path.parent.name
    pdf_name = pdf_path.stem
    
    # Create corresponding output directory
    if output_dir:
        doc_output_dir = output_dir / parent_folder
        os.makedirs(doc_output_dir, exist_ok=True)
    else:
        doc_output_dir = None
        
    # Generate output paths
    cache_path = doc_output_dir / f"{pdf_name}_result.json" if doc_output_dir else None
    
    # Check if results are cached
    if cache and cache_path and cache_path.exists():
        logger.info(f"Using cached results for {pdf_path.name}")
        with open(cache_path, "r") as f:
            return json.load(f)
    
    # Preflight check to determine if PDF is text-based
    logger.info(f"Running preflight check on {pdf_path.name}")
    preflight_result = is_pdf_text_based(pdf_path)
    is_text_based = preflight_result.get("text_based", False)
    
    result = {
        "path": str(pdf_path),
        "is_text_based": is_text_based,
        "processing_time": 0,
        "preflight_error": preflight_result.get("error"),
    }
    
    try:
        if is_text_based:
            # Process text-based PDF with selected parser
            logger.info(f"PDF {pdf_path.name} is text-based, using {text_parser} parser")
            if text_parser not in PARSERS:
                raise ValueError(f"Unknown text parser: {text_parser}")
            
            parser_func = PARSERS[text_parser]
            output_path = str(doc_output_dir / f"{pdf_name}_{text_parser}.md") if doc_output_dir else None
                        
            parser_start = time.time()
            parsing_result = parser_func(str(pdf_path), output_path)
            
            parser_time = time.time() - parser_start
            
            result.update({
                "extraction_method": "text_parser",
                "parser": text_parser,
                "text_length": len(parsing_result.get("text", "")),
                "extraction_success": True,
                "parser_time": parser_time
            })
            logger.info(f"{text_parser} parsing complete for {pdf_path.name}: {parser_time:.2f}s, text length: {len(parsing_result.get('text', ''))} chars")
        else:
            # Process image-based PDF with OCR
            logger.info(f"PDF {pdf_path.name} is image-based, using {ocr_engine} OCR")
            if ocr_engine not in OCR_ENGINES:
                raise ValueError(f"Unknown OCR engine: {ocr_engine}")
                
            ocr_func = OCR_ENGINES[ocr_engine]
            ocr_start = time.time()
            ocr_result = ocr_func(pdf_path)
            ocr_time = time.time() - ocr_start
            
            # Save OCR output
            if doc_output_dir:
                with open(doc_output_dir / f"{pdf_name}_{ocr_engine}_ocr.txt", "w", encoding="utf-8") as f:
                    f.write(ocr_result["text"])
                    
            result.update({
                "extraction_method": "ocr",
                "ocr_engine": ocr_engine,
                "text_length": len(ocr_result.get("text", "")),
                "pages_processed": ocr_result.get("pages", 0),
                "extraction_success": True,
                "ocr_time": ocr_time
            })
            logger.info(f"OCR complete for {pdf_path.name}: {ocr_time:.2f}s, text length: {len(ocr_result.get('text', ''))} chars, pages: {ocr_result.get('pages', 0)}")
                
    except Exception as e:
        logger.error(f"ERROR processing {pdf_path.name}: {str(e)}", exc_info=True)
        result.update({
            "extraction_success": False,
            "error": str(e)
        })
    
    # Calculate processing time
    total_time = time.time() - start_time
    result["processing_time"] = total_time
    
    # Cache results if requested
    if cache and doc_output_dir:
        with open(doc_output_dir / f"{pdf_name}_result.json", "w") as f:
            json.dump(result, f, indent=2)
    
    logger.info(f"Completed processing {pdf_path.name} in {total_time:.2f}s")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single PDF file")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--output_dir", type=str, default="extracted_texts", help="Directory to save output files")
    parser.add_argument("--text_parser", type=str, default="marker", 
                        choices=list(PARSERS.keys()), help="Parser for text-based PDFs")
    parser.add_argument("--ocr_engine", type=str, default="mistral", 
                        choices=list(OCR_ENGINES.keys()), help="OCR engine for image-based PDFs")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of results")
    
    args = parser.parse_args()
    
    # Process PDF
    result = process_pdf(
        Path(args.pdf_path),
        text_parser=args.text_parser,
        ocr_engine=args.ocr_engine,
        output_dir=Path(args.output_dir),
        cache=not args.no_cache
    )
    
    # Print result to stdout so it can be captured
    print(json.dumps(result))