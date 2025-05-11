#!/usr/bin/env python
# ocr_pipeline.py

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import aiofiles
import aiohttp
import fitz  # PyMuPDF
import pandas as pd
import tqdm.asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Download Module
# =========================


async def download_file(
    session, url: str, destination_path: str
) -> tuple[bool, str | None]:
    """Download a single file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    try:
        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(destination_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        await f.write(chunk)
                return True, None
            else:
                return False, f"HTTP error {response.status}"
    except Exception as e:
        return False, str(e)


async def download_documents(
    csv_path: str, output_dir: str, concurrency: int = 3, filter_samples: bool = True
) -> list[dict[str, Any]]:
    """Download documents from URLs in a CSV file"""
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Filter files
    if filter_samples:
        files_to_download = df[df["sample"] & (~df["file_url"].isna())]
    else:
        files_to_download = df[~df["file_url"].isna()]

    logger.info(f"Found {len(files_to_download)} files to download")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []

        for _, row in files_to_download.iterrows():
            file_url = row["file_url"]
            paper_id = row["paper_id"]
            file_name = file_url.split("/")[-1]
            dest_path = os.path.join(output_dir, paper_id, file_name)

            result = {
                "paper_id": paper_id,
                "file_url": file_url,
                "local_path": dest_path,
                "downloaded": False,
                "error": None,
            }

            async def download_task(result=result, url=file_url, path=dest_path):
                async with semaphore:
                    if not os.path.exists(path):
                        success, error = await download_file(session, url, path)
                        result["downloaded"] = success
                        if not success:
                            result["error"] = f"Download failed: {error}"
                    else:
                        result["downloaded"] = True
                        result["already_existed"] = True
                    return result

            tasks.append(download_task())

        # Execute downloads with progress bar
        with tqdm.asyncio.tqdm(total=len(tasks), desc="Downloading files") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)
                status = "success"
                if result.get("error"):
                    status = f"error: {result['error']}"
                elif result.get("already_existed"):
                    status = "already existed"
                pbar.set_postfix_str(f"Last: {status}")

    # Log summary
    successful = sum(1 for r in results if r["downloaded"])
    failed = len(results) - successful
    logger.info(f"Download summary: {successful} successful, {failed} failed")

    return results


# =========================
# PDF Analysis Module
# =========================


def is_pdf_text_based(filepath: str | Path, min_text_threshold: int = 100) -> bool:
    """Check if a PDF contains actual text or is just scanned images"""
    try:
        doc = fitz.open(filepath)
        total_text = sum(len(page.get_text().strip()) for page in doc)
        doc.close()
        return total_text > min_text_threshold
    except Exception as e:
        logger.error(f"Error checking if PDF is text-based: {str(e)}")
        return False


def analyze_pdfs(
    directory: str | Path, output_path: str | None = None
) -> list[dict[str, Any]]:
    """Analyze PDFs in a directory to determine if they're text-based"""
    directory = Path(directory)
    pdf_paths = list(directory.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_paths)} PDFs to analyze")

    results = []
    with tqdm.tqdm(total=len(pdf_paths), desc="Analyzing PDFs") as pbar:
        for path in pdf_paths:
            try:
                is_text = is_pdf_text_based(path)
                results.append(
                    {
                        "path": str(path),
                        "text_based": is_text,
                        "ocr_needed": not is_text,
                        "error": None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "path": str(path),
                        "text_based": None,
                        "ocr_needed": None,
                        "error": str(e),
                    }
                )
            pbar.update(1)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Analysis results saved to {output_path}")

    return results


# =========================
# OCR Module
# =========================


def ocr_document(pdf_path: Path, engine: str = "tesseract") -> dict[str, Any]:
    """Process a PDF with OCR using the specified engine"""
    # Import the appropriate OCR module dynamically based on the selected engine
    if engine == "tesseract":
        try:
            import pytesseract
            from pdf2image import convert_from_path

            # Convert PDF to images
            images = convert_from_path(str(pdf_path))

            # Process each page
            all_text = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i + 1}/{len(images)} with Tesseract")
                text = pytesseract.image_to_string(image)
                all_text.append(text)

            return {
                "engine": "tesseract",
                "text": "\n\n".join(all_text),
                "pages": len(images),
            }
        except ImportError:
            logger.error(
                "Required packages not installed. Please install pytesseract and pdf2image"
            )
            raise
    else:
        logger.error(f"OCR engine '{engine}' not supported yet")
        return {
            "engine": engine,
            "text": "",
            "pages": 0,
            "error": "Engine not supported",
        }


def run_ocr_batch(
    pdf_list: list[dict[str, Any]],
    engine: str = "tesseract",
    output_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Run OCR on a batch of PDFs"""
    results = []

    with tqdm.tqdm(total=len(pdf_list), desc=f"Running OCR with {engine}") as pbar:
        for item in pdf_list:
            if not item.get("ocr_needed", True):
                # Skip files that don't need OCR
                item["ocr_performed"] = False
                results.append(item)
                pbar.update(1)
                continue

            pdf_path = item["path"]
            result = dict(item)  # Copy the original item

            try:
                ocr_result = ocr_document(Path(pdf_path), engine=engine)
                result["ocr_performed"] = True
                result["ocr_engine"] = engine
                result["ocr_result"] = ocr_result

                # Save OCR results to a separate file if output_dir is specified
                if output_dir:
                    out_filename = f"{Path(pdf_path).stem}_ocr.json"
                    out_path = os.path.join(output_dir, out_filename)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                    with open(out_path, "w") as f:
                        json.dump(ocr_result, f, indent=2)

                    result["ocr_output_path"] = out_path
            except Exception as e:
                result["ocr_performed"] = False
                result["error"] = str(e)

            results.append(result)
            pbar.update(1)

    return results


# =========================
# Full Pipeline
# =========================


async def run_pipeline(
    csv_path: str,
    output_dir: str,
    ocr_engine: str = "tesseract",
    concurrency: int = 3,
    filter_samples: bool = True,
) -> list[dict[str, Any]]:
    """Run the full pipeline: download, analyze, and OCR documents"""
    # Step 1: Download documents
    logger.info("Step 1: Downloading documents")
    download_results = await download_documents(
        csv_path, output_dir, concurrency, filter_samples
    )

    # Get all successfully downloaded documents
    downloaded_docs = [doc for doc in download_results if doc["downloaded"]]
    logger.info(f"Successfully downloaded {len(downloaded_docs)} documents")

    # Step 2: Analyze PDFs
    logger.info("Step 2: Analyzing PDFs")
    analysis_results = []
    for doc in tqdm.tqdm(downloaded_docs, desc="Analyzing PDFs"):
        path = doc["local_path"]
        is_text = is_pdf_text_based(path)
        analysis_results.append(
            {
                **doc,  # Include download info
                "text_based": is_text,
                "ocr_needed": not is_text,
            }
        )

    # Step 3: Run OCR on documents that need it
    logger.info("Step 3: Running OCR on non-text-based documents")
    non_text_docs = [doc for doc in analysis_results if doc.get("ocr_needed")]
    logger.info(f"Found {len(non_text_docs)} documents needing OCR")

    ocr_results = []
    for doc in tqdm.tqdm(non_text_docs, desc=f"Running OCR with {ocr_engine}"):
        result = dict(doc)  # Copy the document info

        try:
            pdf_path = doc["local_path"]
            ocr_result = ocr_document(Path(pdf_path), engine=ocr_engine)
            result["ocr_performed"] = True
            result["ocr_engine"] = ocr_engine

            # Save OCR results to a separate file
            ocr_output_path = pdf_path.rsplit(".", 1)[0] + "_ocr.json"
            with open(ocr_output_path, "w") as f:
                json.dump(ocr_result, f, indent=2)

            result["ocr_output_path"] = ocr_output_path
        except Exception as e:
            result["ocr_performed"] = False
            result["error"] = str(e)

        ocr_results.append(result)

    # Combine results from all steps
    text_docs = [doc for doc in analysis_results if not doc.get("ocr_needed")]
    final_results = text_docs + ocr_results

    # Save overall results
    results_path = os.path.join(output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Pipeline completed. Results saved to {results_path}")

    # Log summary
    downloaded = len(downloaded_docs)
    text_based = len(text_docs)
    ocr_needed = len(non_text_docs)
    ocr_performed = sum(1 for r in ocr_results if r.get("ocr_performed"))
    errors = sum(1 for r in final_results if r.get("error"))

    logger.info("Pipeline summary:")
    logger.info(f"  - Total documents: {len(final_results)}")
    logger.info(f"  - Downloaded: {downloaded}")
    logger.info(f"  - Text-based: {text_based}")
    logger.info(f"  - OCR needed: {ocr_needed}")
    logger.info(f"  - OCR performed: {ocr_performed}")
    logger.info(f"  - Errors: {errors}")

    return final_results


# =========================
# Command-line Interface
# =========================


def main():
    parser = argparse.ArgumentParser(description="Modular OCR Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser(
        "download", help="Download documents from URLs in a CSV file"
    )
    download_parser.add_argument(
        "--csv",
        type=str,
        default="data/publevel.csv",
        help="Path to CSV file with document data",
    )
    download_parser.add_argument(
        "--output-dir",
        type=str,
        default="downloaded_papers",
        help="Directory to store downloaded files",
    )
    download_parser.add_argument(
        "--concurrency", type=int, default=3, help="Number of concurrent downloads"
    )
    download_parser.add_argument(
        "--all", action="store_true", help="Download all files, not just samples"
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze PDFs to check if they're text-based"
    )
    analyze_parser.add_argument(
        "directory", help="Directory containing PDFs to analyze"
    )
    analyze_parser.add_argument(
        "--output",
        type=str,
        default="analysis_results.json",
        help="Path to save analysis results",
    )

    # OCR command
    ocr_parser = subparsers.add_parser("ocr", help="Run OCR on PDFs")
    ocr_parser.add_argument(
        "--analysis-file",
        type=str,
        required=True,
        help="JSON file with analysis results",
    )
    ocr_parser.add_argument(
        "--engine",
        type=str,
        default="tesseract",
        choices=["tesseract"],
        help="OCR engine to use",
    )
    ocr_parser.add_argument(
        "--output-dir",
        type=str,
        default="ocr_results",
        help="Directory to store OCR results",
    )

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the full pipeline")
    pipeline_parser.add_argument(
        "--csv",
        type=str,
        default="data/publevel.csv",
        help="Path to CSV file with document data",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_papers",
        help="Directory to store processed files",
    )
    pipeline_parser.add_argument(
        "--engine",
        type=str,
        default="tesseract",
        choices=["tesseract"],
        help="OCR engine to use",
    )
    pipeline_parser.add_argument(
        "--concurrency", type=int, default=3, help="Number of concurrent operations"
    )
    pipeline_parser.add_argument(
        "--all", action="store_true", help="Process all files, not just samples"
    )

    args = parser.parse_args()

    if args.command == "download":
        asyncio.run(
            download_documents(
                args.csv, args.output_dir, args.concurrency, not args.all
            )
        )

    elif args.command == "analyze":
        analyze_pdfs(args.directory, args.output)

    elif args.command == "ocr":
        # Load analysis results
        with open(args.analysis_file) as f:
            analysis_results = json.load(f)

        # Run OCR on documents that need it
        run_ocr_batch(
            [doc for doc in analysis_results if doc.get("ocr_needed")],
            args.engine,
            args.output_dir,
        )

    elif args.command == "pipeline":
        asyncio.run(
            run_pipeline(
                args.csv, args.output_dir, args.engine, args.concurrency, not args.all
            )
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
