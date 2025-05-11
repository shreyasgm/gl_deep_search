import argparse
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF
from dask import bag as db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_pdf_text_based(
    filepath: str | Path, min_text_threshold: int = 100
) -> dict[str, str | bool | None]:
    try:
        doc = fitz.open(filepath)
        total_text = sum(len(page.get_text().strip()) for page in doc)
        doc.close()
        return {
            "path": str(filepath),
            "text_based": total_text > min_text_threshold,
            "error": None,
        }
    except Exception as e:
        return {"path": str(filepath), "text_based": None, "error": str(e)}


def main(
    directory: str | Path,
    output_path: str = "preflight_results.json",
    workers: int = 8,
) -> None:
    pdf_paths = list(Path(directory).rglob("*.pdf"))
    pdf_bag = db.from_sequence(pdf_paths, npartitions=workers)
    results = pdf_bag.map(is_pdf_text_based).compute()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Completed scan of %d PDFs", len(pdf_paths))
    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect if PDFs are text-based or scanned."
    )
    parser.add_argument("directory", help="Directory containing PDFs to scan")
    parser.add_argument(
        "--output", default="preflight_results.json", help="Path to save output JSON"
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of Dask workers")

    args = parser.parse_args()
    main(args.directory, args.output, args.workers)
