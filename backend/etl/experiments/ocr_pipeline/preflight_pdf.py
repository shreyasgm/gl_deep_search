import fitz  # PyMuPDF
import argparse
from pathlib import Path
from dask import bag as db
from tqdm import tqdm
import json


def is_pdf_text_based(filepath, min_text_threshold=100):
    try:
        doc = fitz.open(filepath)
        total_text = sum(len(page.get_text().strip()) for page in doc)
        doc.close()
        return {
            "path": str(filepath),
            "text_based": total_text > min_text_threshold,
            "error": None
        }
    except Exception as e:
        return {
            "path": str(filepath),
            "text_based": None,
            "error": str(e)
        }


def main(directory, output_path="preflight_results.json", workers=8):
    pdf_paths = list(Path(directory).rglob("*.pdf"))
    pdf_bag = db.from_sequence(pdf_paths, npartitions=workers)

    # Compute results (parallel), THEN loop through them with tqdm
    raw_results = pdf_bag.map(is_pdf_text_based).compute()

    # Display progress bar while iterating over the finished results
    results = []
    for r in tqdm(raw_results, total=len(pdf_paths), desc="Saving results"):
        results.append(r)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Completed scan of {len(pdf_paths)} PDFs.")
    print(f"📄 Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect if PDFs are text-based or scanned.")
    parser.add_argument("directory", help="Directory containing PDFs to scan")
    parser.add_argument("--output", default="preflight_results.json", help="Path to save output JSON")
    parser.add_argument("--workers", type=int, default=8, help="Number of Dask workers")

    args = parser.parse_args()
    main(args.directory, args.output, args.workers)
