# parse.py

import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# load env
load_dotenv()

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# PDF Analysis Module
# ----------------------------

# def is_pdf_text_based(
#     filepath: Union[str, Path], min_text_threshold: int = 100
# ) -> dict[str, Union[str, bool, None]]:
#     """Check if a PDF is primarily text-based or scanned."""
#     try:
#         doc = fitz.open(filepath)
#         total_text = sum(len(page.get_text().strip()) for page in doc)
#         doc.close()
#         return {
#             "path": str(filepath),
#             "text_based": total_text > min_text_threshold,
#             "error": None,
#         }
#     except Exception as e:
#         return {"path": str(filepath), "text_based": None, "error": str(e)}

# ----------------------------
# Parsing Engines
# ----------------------------


def parse_marker(
    pdf_path: str | Path,
    output_path: str = "marker_output.md",
    openai_model: str = "gpt-4o",
    openai_base_url: str = "https://api.openai.com/v1",
) -> dict:
    """Parse a PDF using Marker with forced OCR and OpenAI LLM assistance"""
    import os

    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    # Check if OPENAI_API_KEY is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OSError(
            "OPENAI_API_KEY not found in environment. Required for OpenAI LLM mode."
        )

    # Configure Marker with OCR, line formatting, and OpenAI
    config = {
        "force_ocr": True,  # Force OCR on all pages
        "format_lines": True,  # Reformat all lines for better quality
        "use_llm": True,  # Use LLM to improve accuracy
        "output_format": "markdown",  # Output format (could be "json" or "html" as well)
        "llm_service": "marker.services.openai.OpenAIService",  # Use OpenAI service
        "openai_api_key": api_key,  # Pass the API key
        "openai_model": openai_model,  # Specify the OpenAI model to use
        "openai_base_url": openai_base_url,  # Specify the OpenAI API endpoint
    }

    # Create config parser
    config_parser = ConfigParser(config)

    # Create converter with our configuration
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    # Convert the document
    rendered = converter(pdf_path)

    # Extract text and images
    text, metadata, images = text_from_rendered(rendered)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return {
        "engine": "marker_with_ocr_openai",
        "text": text,
        "metadata": metadata,
        "images": images,
        "output_path": output_path,
    }


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
    import os

    from dotenv import load_dotenv
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


OCR_ENGINES = {
    "tesseract": ocr_tesseract,
    "easyocr": ocr_easyocr,
    "mistral": ocr_mistral,
    "docling": ocr_docling,
}
