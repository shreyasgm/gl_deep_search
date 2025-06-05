# pdf_modules.py

import logging
from pathlib import Path
from typing import Any, Union
import fitz
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def parse_marker(
    pdf_path: Union[str, Path],
    output_path: str = "marker_output.md",
    openai_model: str = "gpt-4o",
    openai_base_url: str = "https://api.openai.com/v1",
) -> dict:
    """Parse a PDF using Marker with forced OCR and OpenAI LLM assistance."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    from marker.config.parser import ConfigParser

    # Check if OPENAI_API_KEY is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in environment. Required for OpenAI LLM mode.")

    # Main config — EDIT HERE to change settings!
    config = {
        "force_ocr": True,                   # Force OCR on all pages
        "format_lines": True,                # Reformat lines for better quality
        "use_llm": True,                     # Use LLM to improve accuracy
        "output_format": "markdown",         # Other options: "json", "html"
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_api_key": api_key,
        "openai_model": openai_model,
        "openai_base_url": openai_base_url,
    }

    config_parser = ConfigParser(config)
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    rendered = converter(str(pdf_path))
    text, metadata, images = text_from_rendered(rendered)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return {
        "engine": "marker_with_ocr_openai",
        "text": text,
        "metadata": metadata,
        "images": images,
        "output_path": output_path
    }