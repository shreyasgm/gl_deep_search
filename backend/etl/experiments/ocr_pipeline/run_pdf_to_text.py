# run_parallel_pipeline.py

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import dask
import dask.distributed
from dask.distributed import Client, progress

from pdf_modules import is_pdf_text_based, PARSERS, OCR_ENGINES


def process_pdf(
    pdf_path: Path,
    text_parser: str = "marker",
    ocr_engine: str = "mistral",
    output_dir: Optional[Path] = None,
    cache: bool = True,
) -> Dict[str, Any]:
    """
    Process a PDF file through the pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        text_parser: Parser to use for text-based PDFs
        ocr_engine: OCR engine to use for image-based PDFs
        output_dir: Directory to save output files
        cache: Whether to cache results
        
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate output paths
    pdf_name = pdf_path.stem
    cache_path = Path(output_dir) / f"{pdf_name}_result.json" if output_dir else None
    
    # Check if results are cached
    if cache and cache_path and cache_path.exists():
        with open(cache_path, "r") as f:
            return json.load(f)
    
    # Preflight check to determine if PDF is text-based
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
            if text_parser not in PARSERS:
                raise ValueError(f"Unknown text parser: {text_parser}")
            
            parser_func = PARSERS[text_parser]
            output_path = str(Path(output_dir) / f"{pdf_name}_{text_parser}.md") if output_dir else None
            
            if text_parser in ["unstructured", "llamaparse", "mistral"]:
                parsing_result = parser_func(str(pdf_path), output_path)
            else:  # marker
                parsing_result = parser_func(pdf_path)
                
                # Save marker output if output_dir is specified
                if output_dir:
                    with open(Path(output_dir) / f"{pdf_name}_marker.md", "w", encoding="utf-8") as f:
                        f.write(parsing_result["text"])
            
            result.update({
                "extraction_method": "text_parser",
                "parser": text_parser,
                "text_length": len(parsing_result.get("text", "")),
                "extraction_success": True,
            })
        else:
            # Process image-based PDF with OCR
            if ocr_engine not in OCR_ENGINES:
                raise ValueError(f"Unknown OCR engine: {ocr_engine}")
                
            ocr_func = OCR_ENGINES[ocr_engine]
            ocr_result = ocr_func(pdf_path)
            
            # Save OCR output
            if output_dir:
                with open(Path(output_dir) / f"{pdf_name}_{ocr_engine}_ocr.txt", "w", encoding="utf-8") as f:
                    f.write(ocr_result["text"])
                    
            result.update({
                "extraction_method": "ocr",
                "ocr_engine": ocr_engine,
                "text_length": len(ocr_result.get("text", "")),
                "pages_processed": ocr_result.get("pages", 0),
                "extraction_success": True,
            })
                
    except Exception as e:
        result.update({
            "extraction_success": False,
            "error": str(e)
        })
    
    # Calculate processing time
    result["processing_time"] = time.time() - start_time
    
    # Cache results if requested
    if cache and output_dir:
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
    
    return result


def process_pdfs_parallel(
    pdf_dir: Path,
    output_dir: Path,
    text_parser: str = "marker",
    ocr_engine: str = "mistral",
    n_workers: int = 4,
    cache: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process multiple PDF files in parallel using Dask.
    
    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save output files
        text_parser: Parser to use for text-based PDFs
        ocr_engine: OCR engine to use for image-based PDFs
        n_workers: Number of Dask workers
        cache: Whether to cache results
    
    Returns:
        List of processing results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in directory and subdirectories
    pdf_files = []
    for root, _, files in os.walk(pdf_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(Path(root) / file)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Set up Dask client for parallel processing
    client = Client(n_workers=n_workers, threads_per_worker=1)
    print(f"Dask dashboard available at: {client.dashboard_link}")
    
    # Create tasks for each PDF
    tasks = []
    for pdf_path in pdf_files:
        task = dask.delayed(process_pdf)(
            pdf_path, 
            text_parser=text_parser,
            ocr_engine=ocr_engine,
            output_dir=output_dir,
            cache=cache
        )
        tasks.append(task)
    
    # Compute all tasks in parallel
    print(f"Processing {len(tasks)} PDFs in parallel...")
    results = dask.compute(*tasks)
    
    # Save summary report
    summary_path = Path(output_dir) / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Close the client
    client.close()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process PDFs in parallel with preflight check")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Directory containing PDF files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--text_parser", type=str, default="marker", 
                        choices=list(PARSERS.keys()), help="Parser for text-based PDFs")
    parser.add_argument("--ocr_engine", type=str, default="mistral", 
                        choices=list(OCR_ENGINES.keys()), help="OCR engine for image-based PDFs")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of results")
    
    args = parser.parse_args()
    
    print(f"Starting PDF processing pipeline with:")
    print(f"- PDF directory: {args.pdf_dir}")
    print(f"- Output directory: {args.output_dir}")
    print(f"- Text parser: {args.text_parser}")
    print(f"- OCR engine: {args.ocr_engine}")
    print(f"- Workers: {args.workers}")
    print(f"- Cache: {not args.no_cache}")
    
    # Process PDFs
    start_time = time.time()
    results = process_pdfs_parallel(
        Path(args.pdf_dir),
        Path(args.output_dir),
        text_parser=args.text_parser,
        ocr_engine=args.ocr_engine,
        n_workers=args.workers,
        cache=not args.no_cache
    )
    
    # Print summary
    total_time = time.time() - start_time
    total_pdfs = len(results)
    successful = sum(1 for r in results if r.get("extraction_success", False))
    text_based = sum(1 for r in results if r.get("is_text_based", False))
    
    print(f"\nProcessing completed in {total_time:.2f} seconds")
    print(f"Total PDFs: {total_pdfs}")
    print(f"Successful extractions: {successful} ({successful/total_pdfs*100:.1f}%)")
    print(f"Text-based PDFs: {text_based} ({text_based/total_pdfs*100:.1f}%)")
    print(f"Image-based PDFs: {total_pdfs - text_based} ({(total_pdfs - text_based)/total_pdfs*100:.1f}%)")
    print(f"Results saved to {args.output_dir}/processing_summary.json")