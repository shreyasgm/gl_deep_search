# parse.py

import logging
import pathlib
from pathlib import Path
from typing import Any, Union
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# PDF Analysis Module
# ----------------------------

def is_pdf_text_based(
    filepath: Union[str, Path], min_text_threshold: int = 100
) -> dict[str, Union[str, bool, None]]:
    """Check if a PDF is primarily text-based or scanned."""
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

# ----------------------------
# Parsing Engines
# ----------------------------

def parse_marker(pdf_path: Path) -> dict[str, Any]:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(str(pdf_path))
    text, _, _ = text_from_rendered(rendered)

    return {"engine": "marker", "text": text}



def parse_unstructured(pdf_path: str, output_path: str = "unstructured_output.txt") -> dict:
    from unstructured.partition.pdf import partition_pdf
    elements = partition_pdf(filename=pdf_path)
    text = "\n\n".join(str(el) for el in elements)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "unstructured", "text": text}



def parse_llamaparse(pdf_path: Path) -> dict[str, Any]:
    raise NotImplementedError("Llamaparse parser integration not implemented yet.")


def parse_mistral(pdf_path: Path) -> dict[str, Any]:
    raise NotImplementedError("Mistral parser integration not implemented yet.")


PARSERS = {
    "marker": parse_marker,
    "unstructured": parse_unstructured,
    "llamaparse": parse_llamaparse,
    "mistral": parse_mistral,
}

# ----------------------------
# OCR Engines
# ----------------------------

def ocr_tesseract(pdf_path: Path) -> dict[str, Any]:
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        all_text = [pytesseract.image_to_string(image) for image in images]

        return {"engine": "tesseract", "text": "\n\n".join(all_text), "pages": len(images)}

    except ImportError:
        logger.error("Missing packages: pytesseract, pdf2image")
        raise


def ocr_easyocr(pdf_path: Path) -> dict[str, Any]:
    try:
        import easyocr
        from pdf2image import convert_from_path

        reader = easyocr.Reader(["en"])
        images = convert_from_path(str(pdf_path))

        all_text = []
        for image in images:
            results = reader.readtext(image)
            page_text = "\n".join([text for _, text, _ in results])
            all_text.append(page_text)

        return {"engine": "easyocr", "text": "\n\n".join(all_text), "pages": len(images)}

    except ImportError:
        logger.error("Missing packages: easyocr, pdf2image")
        raise


def ocr_mistral(pdf_path: Path) -> dict[str, Any]:
    try:
        import base64
        import os
        from io import BytesIO
        import requests
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        all_text = []

        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            response = requests.post(
                "https://api.mistral.ai/v1/ocr",  # Replace with real URL
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
                raise RuntimeError(f"Failed Mistral request: {response.text}")

        return {"engine": "mistral", "text": "\n\n".join(all_text), "pages": len(images)}

    except ImportError:
        logger.error("Missing packages: requests, pdf2image")
        raise


def ocr_docling(pdf_path: Path) -> dict[str, Any]:
    try:
        from docling import Document
        doc = Document(str(pdf_path))
        text = doc.extract_text()
        return {"engine": "docling", "text": text, "pages": len(doc.pages)}
    except ImportError:
        logger.error("Missing package: docling")
        raise


OCR_ENGINES = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "mistral": ocr_mistral,
    "docling": ocr_docling,
}
