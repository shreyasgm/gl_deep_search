# pdf_module.py

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from module_marker import parse_marker_preset

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# PDF MODULES
# ----------------------------


def marker_parser_adapter(
    pdf_path, output_path=None, marker_preset="baseline", marker_model="gpt-4o-mini"
):
    """
    Adapter for Marker with full presets/models support.
    """
    return parse_marker_preset(
        pdf_path=pdf_path,
        preset=marker_preset,
        model=marker_model,
        output_path=output_path,
    )


def parse_unstructured(
    pdf_path: str, output_path: str = "unstructured_output.txt"
) -> dict:
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(filename=pdf_path)
    text = "\n\n".join(str(el) for el in elements)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "unstructured", "text": text}


def parse_llamaparse(pdf_path: str, output_path: str = "llamaparse_output.md") -> dict:
    from llama_parse import LlamaParse

    parser = LlamaParse(result_type="markdown")  # or "text"
    with open(pdf_path, "rb") as f:
        documents = parser.load_data(f, extra_info={"file_name": pdf_path})
    text = "\n\n".join(doc.text for doc in documents)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "llamaparse", "text": text}


def parse_mistral(pdf_path: str, output_path: str = "mistral_output.md") -> dict:
    import base64

    from mistralai import Mistral

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise OSError("MISTRAL_API_KEY not found in environment")
    with open(pdf_path, "rb") as f:
        encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded_pdf}",
        },
    )
    text = ocr_response.markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"engine": "mistral", "text": text}


def ocr_tesseract(pdf_path: Path) -> dict[str, Any]:
    try:
        import pytesseract
        from pdf2image import convert_from_path

        images = convert_from_path(str(pdf_path))
        all_text = [pytesseract.image_to_string(image) for image in images]

        return {
            "engine": "tesseract",
            "text": "\n\n".join(all_text),
            "pages": len(images),
        }

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

        return {
            "engine": "easyocr",
            "text": "\n\n".join(all_text),
            "pages": len(images),
        }

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

        return {
            "engine": "mistral",
            "text": "\n\n".join(all_text),
            "pages": len(images),
        }

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


PARSERS = {
    "marker": marker_parser_adapter,
    "unstructured": parse_unstructured,
    "llamaparse": parse_llamaparse,
    "mistral": parse_mistral,
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "docling": ocr_docling,
}
