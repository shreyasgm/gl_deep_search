#!/usr/bin/env python
# process_single_pdf.py

import argparse
import json
import logging
import os
import time
from pathlib import Path

from pdf_module import PARSERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_pdf(
    pdf_path: Path,
    parser: str = "marker",
    output_dir: Path = None,
    cache: bool = True
):
    """Process a single PDF file."""
    start_time = time.time()
    
    # Initialize result dictionary
    result = {
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.name,
        "parent_folder": pdf_path.parent.name,
    }
    
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
    
    try:
        # Process PDF with selected parser
        logger.info(f"Processing PDF {pdf_path.name} with {parser} parser")
        
        if parser not in PARSERS:
            raise ValueError(f"Unknown parser: {parser}")
        
        parser_func = PARSERS[parser]
        output_path = str(doc_output_dir / f"{pdf_name}_{parser}.md") if doc_output_dir else None
        
        parser_start = time.time()
        parsing_result = parser_func(str(pdf_path), output_path)
        parser_time = time.time() - parser_start
        
        result.update({
            "parser": parser,
            "text_length": len(parsing_result.get("text", "")),
            "extraction_success": True,
            "parser_time": parser_time,
            "pages_processed": parsing_result.get("pages", None)
        })
        
        logger.info(f"{parser} parsing complete for {pdf_path.name}: {parser_time:.2f}s, text length: {len(parsing_result.get('text', ''))} chars")
                
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
    parser.add_argument("--parser", type=str, default="marker", 
                        choices=list(PARSERS.keys()), help="Parser to use for PDF processing")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of results")
    
    args = parser.parse_args()
    
    # Process PDF
    result = process_pdf(
        Path(args.pdf_path),
        parser=args.parser,
        output_dir=Path(args.output_dir),
        cache=not args.no_cache
    )
    
    # Print result to stdout so it can be captured
    print(json.dumps(result))