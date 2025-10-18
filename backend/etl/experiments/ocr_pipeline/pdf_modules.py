#!/usr/bin/env python3
# pdf_modules.py - All-in-one PDF processing module

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(
    pdf_path: str | Path,
    module: str = "marker",
    preset: str = "baseline", 
    model: str = "none",
    output_path: str | Path = None,
    output_dir: str | Path = None,
    cache: bool = True,
    organize_by_folder: bool = True,
    **kwargs
) -> dict:
    """Parse PDF using specified module and preset."""
    start_time = time.time()
    pdf_path = Path(pdf_path)
    
    # Create organized output structure
    if output_dir and not output_path:
        output_dir = Path(output_dir)
        if organize_by_folder:
            doc_output_dir = output_dir / pdf_path.parent.name
            doc_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            doc_output_dir = output_dir
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            
        ext = ".md" if preset != "json" else ".json"
        output_path = doc_output_dir / f"{pdf_path.stem}_{module}_{preset}{ext}"
        cache_path = doc_output_dir / f"{pdf_path.stem}_result.json"
    else:
        cache_path = None
    
    # Check cache first
    if cache and cache_path and cache_path.exists():
        logger.info(f"Using cached results for {pdf_path.name}")
        with open(cache_path) as f:
            cached_result = json.load(f)
            cached_result.update({"from_cache": True})
            return cached_result
    
    logger.info(f"Processing {pdf_path.name} with {module} (preset: {preset}, model: {model})")
    
    try:
        if module == "marker":
            from module_marker import parse_marker_preset
            result = parse_marker_preset(pdf_path, preset=preset, model=model, output_path=output_path)
            
        elif module == "unstructured":
            from module_unstructured import parse_unstructured_preset
            result = parse_unstructured_preset(pdf_path, preset=preset, output_path=output_path)
            
        elif module == "tesseract":
            from module_tesseract import parse_tesseract_preset
            result = parse_tesseract_preset(pdf_path, preset=preset, output_path=output_path)
            
        else:
            raise ValueError(f"Unknown module '{module}'. Available: marker, unstructured, tesseract")
        
        # Standardize result format
        result.update({
            "module": module,
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "parent_folder": pdf_path.parent.name,
            "text_length": len(result.get("text", "")),
            "extraction_success": result.get("success", False),
            "total_time": time.time() - start_time,
            "from_cache": False,
        })
        
        # Cache the result
        if cache and cache_path:
            with open(cache_path, 'w') as f:
                json.dump(result, f, indent=2)
                
        return result
            
    except Exception as e:
        logger.error(f"Parser {module} failed on {pdf_path}: {e}")
        error_result = {
            "success": False,
            "extraction_success": False,
            "error": str(e),
            "module": module,
            "preset": preset,
            "model": model,
            "pdf_path": str(pdf_path),
            "pdf_name": pdf_path.name,
            "parent_folder": pdf_path.parent.name,
            "output_path": str(output_path) if output_path else None,
            "total_time": time.time() - start_time,
            "from_cache": False,
        }
        
        if cache and cache_path:
            with open(cache_path, 'w') as f:
                json.dump(error_result, f, indent=2)
                
        return error_result

# CLI interface built into the same file
def main():
    parser = argparse.ArgumentParser(description="Process PDF with various modules")
    
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-m", "--module", default="marker", 
                       choices=["marker", "unstructured", "tesseract"],
                       help="Parser module to use")
    parser.add_argument("-p", "--preset", default="baseline",
                       help="Configuration preset")
    parser.add_argument("--model", default="none",
                       help="Model for LLM-enabled presets")
    parser.add_argument("--output_dir", default="extracted_texts",
                       help="Base output directory")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable caching")
    parser.add_argument("--no_organize", action="store_true",
                       help="Don't organize by parent folder")
    
    args = parser.parse_args()
    
    result = parse_pdf(
        pdf_path=args.pdf_path,
        module=args.module,
        preset=args.preset,
        model=args.model,
        output_dir=args.output_dir,
        cache=not args.no_cache,
        organize_by_folder=not args.no_organize
    )
    
    # Print JSON result for script consumption
    print(json.dumps(result))
    
    # Exit with error code if processing failed
    if not result.get("extraction_success", False):
        sys.exit(1)

if __name__ == "__main__":
    main()