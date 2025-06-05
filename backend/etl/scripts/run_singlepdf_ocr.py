#!/usr/bin/env python
# process_single_pdf.py

import argparse
import json
import logging
import os
import time
from pathlib import Path

from pdf_modules import parse_marker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def process_pdf(
    pdf_path: Path,
    output_dir: Path = None,
    cache: bool = True
):
    """Process a single PDF file using Marker parser only."""
    start_time = time.time()
    
    parent_folder = pdf_path.parent.name
    pdf_name = pdf_path.stem
    
    # Create output directory
    if output_dir:
        doc_output_dir = output_dir / parent_folder
        os.makedirs(doc_output_dir, exist_ok=True)
    else:
        doc_output_dir = None

    # Path for cache/result
    cache_path = doc_output_dir / f"{pdf_name}_result.json" if doc_output_dir else None

    if cache and cache_path and cache_path.exists():
        logger.info(f"Using cached results for {pdf_path.name}")
        with open(cache_path, "r") as f:
            return json.load(f)

    result = {
        "path": str(pdf_path),
        "processing_time": 0,
        "extraction_success": False
    }

    try:
        output_path = str(doc_output_dir / f"{pdf_name}_marker.md") if doc_output_dir else "marker_output.md"

        parsing_result = parse_marker(str(pdf_path), output_path)

        result.update({
            "extraction_method": "marker",
            "text_length": len(parsing_result.get("text", "")),
            "extraction_success": True,
            "output_path": parsing_result.get("output_path"),
            "metadata": parsing_result.get("metadata"),
            "images": parsing_result.get("images"),
        })
        logger.info(
            f"Marker parsing complete for {pdf_path.name}: "
            f"{result['text_length']} chars extracted"
        )
    except Exception as e:
        logger.error(f"ERROR processing {pdf_path.name}: {str(e)}", exc_info=True)
        result['error'] = str(e)

    total_time = time.time() - start_time
    result["processing_time"] = total_time

    if cache and doc_output_dir:
        with open(doc_output_dir / f"{pdf_name}_result.json", "w") as f:
            json.dump(result, f, indent=2)
    logger.info(f"Completed processing {pdf_path.name} in {total_time:.2f}s")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single PDF file with Marker")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--output_dir", type=str, default="extracted_texts", help="Directory to save output files")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of results")
    
    args = parser.parse_args()
    
    result = process_pdf(
        Path(args.pdf_path),
        output_dir=Path(args.output_dir),
        cache=not args.no_cache
    )
    print(json.dumps(result))