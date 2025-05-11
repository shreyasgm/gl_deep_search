# parse.py

import argparse
import json
import pathlib

"""
This script is used to parse a PDF using a selected engine.

The engines are:
- marker with llm
- unstructured
- llamaparse
"""

# also here I realised that my excel seems to have lost the columns for language lol. but not neededfor now


# Mock functions to replace with actual tool wrappers
def parse_marker(pdf_path):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    # Match existing structure exactly
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    rendered = converter(str(pdf_path))
    text, _, images = text_from_rendered(rendered)

    return {"engine": "marker", "text": text}


def parse_unstructured(pdf_path):
    return {"engine": "unstructured", "text": "...unstructured output here..."}


def parse_llamaparse(pdf_path):
    return {"engine": "llamaparse", "text": "...llamaparse output here..."}


def parse_mistral(pdf_path):
    return {"engine": "mistral", "text": "...mistral output here..."}


PARSERS = {
    "marker": parse_marker,
    "unstructured": parse_unstructured,
    "llamaparse": parse_llamaparse,
    "mistral": parse_mistral,
}


def main():
    parser = argparse.ArgumentParser(description="Parse PDF using selected engine.")
    parser.add_argument(
        "--engine", choices=PARSERS.keys(), required=True, help="Parsing engine"
    )
    parser.add_argument(
        "--pdf", type=pathlib.Path, required=True, help="Path to input PDF"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default="output.json", help="Output file path"
    )

    args = parser.parse_args()
    engine_func = PARSERS[args.engine]

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    result = engine_func(args.pdf)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Parsed using {args.engine}, saved to {args.out}")


# run_ocr.py

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
This script is used to perform OCR on PDFs using various OCR engines.

The engines are:
- tesseract: Open source OCR engine
- easyocr: Another open source OCR engine with good language support
- mistral: Mistral AI's OCR API
- docling: Document processing with OCR capabilities
"""


def ocr_tesseract(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Process PDF using Tesseract OCR."""
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


def ocr_easyocr(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Process PDF using EasyOCR."""
    try:
        import easyocr
        from pdf2image import convert_from_path

        # Initialize EasyOCR reader
        reader = easyocr.Reader(["en"])  # Add more languages as needed

        # Convert PDF to images
        images = convert_from_path(str(pdf_path))

        # Process each page
        all_text = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)} with EasyOCR")
            results = reader.readtext(image)
            page_text = "\n".join([text for _, text, _ in results])
            all_text.append(page_text)

        return {
            "engine": "easyocr",
            "text": "\n\n".join(all_text),
            "pages": len(images),
        }
    except ImportError:
        logger.error(
            "Required packages not installed. Please install easyocr and pdf2image"
        )
        raise


def ocr_mistral(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Process PDF using Mistral AI's OCR API."""
    try:
        import base64
        from io import BytesIO

        import requests
        from pdf2image import convert_from_path

        # Convert PDF to images
        images = convert_from_path(str(pdf_path))

        # Process each page
        all_text = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)} with Mistral OCR")

            # Convert image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            # Call Mistral API
            # Note: You'll need to set up your API key and endpoint
            # This is a placeholder for the actual API call
            response = requests.post(
                "https://api.mistral.ai/v1/ocr",  # Replace with actual endpoint
                headers={
                    "Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={"image": img_str},
            )

            if response.status_code == 200:
                page_text = response.json()["text"]
                all_text.append(page_text)
            else:
                logger.error(f"Error processing page {i + 1}: {response.text}")

        return {
            "engine": "mistral",
            "text": "\n\n".join(all_text),
            "pages": len(images),
        }
    except ImportError:
        logger.error(
            "Required packages not installed. Please install requests and pdf2image"
        )
        raise


def ocr_docling(pdf_path: pathlib.Path) -> dict[str, Any]:
    """Process PDF using Docling."""
    try:
        from docling import Document

        # Process document
        doc = Document(str(pdf_path))
        text = doc.extract_text()

        return {"engine": "docling", "text": text, "pages": len(doc.pages)}
    except ImportError:
        logger.error("Required package not installed. Please install docling")
        raise


OCR_ENGINES = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "mistral": ocr_mistral,
    "docling": ocr_docling,
}


def main():
    parser = argparse.ArgumentParser(
        description="Perform OCR on PDF using selected engine."
    )
    parser.add_argument(
        "--engine", choices=OCR_ENGINES.keys(), required=True, help="OCR engine to use"
    )
    parser.add_argument(
        "--pdf", type=pathlib.Path, required=True, help="Path to input PDF"
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default="ocr_output.json", help="Output file path"
    )

    args = parser.parse_args()
    engine_func = OCR_ENGINES[args.engine]

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    logger.info(f"Starting OCR processing with {args.engine}")
    result = engine_func(args.pdf)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"OCR completed using {args.engine}, saved to {args.out}")


if __name__ == "__main__":
    main()
