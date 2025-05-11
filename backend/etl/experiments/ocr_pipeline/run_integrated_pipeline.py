#!/usr/bin/env python
# run_integrated_pipeline.py

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


async def process_document(
    session,
    row: pd.Series,
    output_dir: str,
    ocr_engine: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Process a single document: download, check if text-based, and OCR if needed"""
    result = {
        "paper_id": row["paper_id"],
        "file_url": row["file_url"],
        "downloaded": False,
        "text_based": None,
        "ocr_needed": None,
        "ocr_performed": False,
        "ocr_engine": None,
        "error": None,
    }

    file_url = row["file_url"]
    paper_id = row["paper_id"]
    file_name = file_url.split("/")[-1]
    dest_path = os.path.join(output_dir, paper_id, file_name)
    result["local_path"] = dest_path

    async with semaphore:
        # Step 1: Download the file if it doesn't exist
        if not os.path.exists(dest_path):
            success, error = await download_file(session, file_url, dest_path)
            result["downloaded"] = success
            if not success:
                result["error"] = f"Download failed: {error}"
                return result
        else:
            result["downloaded"] = True

        # Step 2: Check if the PDF is text-based
        is_text = is_pdf_text_based(dest_path)
        result["text_based"] = is_text
        result["ocr_needed"] = not is_text

        # Step 3: If not text-based, perform OCR
        if not is_text:
            try:
                ocr_results = ocr_document(Path(dest_path), engine=ocr_engine)
                result["ocr_performed"] = True
                result["ocr_engine"] = ocr_engine

                # Save OCR results to a separate file
                ocr_output_path = dest_path.rsplit(".", 1)[0] + "_ocr.json"
                async with aiofiles.open(ocr_output_path, "w") as f:
                    await f.write(json.dumps(ocr_results, indent=2))

                result["ocr_output_path"] = ocr_output_path
            except Exception as e:
                result["error"] = f"OCR failed: {str(e)}"

    return result


async def run_pipeline(
    csv_path: str, output_dir: str, ocr_engine: str = "tesseract", concurrency: int = 3
) -> list[dict[str, Any]]:
    """Run the full pipeline: download, check if text-based, and OCR if needed"""
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Filter for sample files with valid file URLs
    sample_files = df[df["sample"] & (~df["file_url"].isna())]
    logger.info(f"Found {len(sample_files)} sample files to process")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in sample_files.iterrows():
            task = process_document(session, row, output_dir, ocr_engine, semaphore)
            tasks.append(task)

        # Execute tasks with progress bar
        with tqdm.asyncio.tqdm(total=len(tasks), desc="Processing documents") as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                pbar.update(1)
                status = "success"
                if result.get("error"):
                    status = f"error: {result['error']}"
                elif result.get("ocr_performed"):
                    status = f"OCR performed with {result['ocr_engine']}"
                elif result.get("text_based"):
                    status = "text-based (no OCR needed)"
                pbar.set_postfix_str(f"Last: {status}")

    # Save overall results
    results_path = os.path.join(output_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Pipeline completed. Results saved to {results_path}")

    # Log summary
    downloaded = sum(1 for r in results if r["downloaded"])
    text_based = sum(1 for r in results if r["text_based"])
    ocr_needed = sum(1 for r in results if r["ocr_needed"])
    ocr_performed = sum(1 for r in results if r["ocr_performed"])
    errors = sum(1 for r in results if r["error"])

    logger.info("Pipeline summary:")
    logger.info(f"  - Total documents: {len(results)}")
    logger.info(f"  - Downloaded: {downloaded}")
    logger.info(f"  - Text-based: {text_based}")
    logger.info(f"  - OCR needed: {ocr_needed}")
    logger.info(f"  - OCR performed: {ocr_performed}")
    logger.info(f"  - Errors: {errors}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Integrated document processing pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/publevel.csv",
        help="Path to CSV file with document data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="processed_papers",
        help="Directory to store processed files",
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default="tesseract",
        choices=["tesseract"],
        help="OCR engine to use",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Number of concurrent operations"
    )

    args = parser.parse_args()

    # Run the pipeline
    asyncio.run(
        run_pipeline(args.csv, args.output_dir, args.ocr_engine, args.concurrency)
    )


if __name__ == "__main__":
    main()
