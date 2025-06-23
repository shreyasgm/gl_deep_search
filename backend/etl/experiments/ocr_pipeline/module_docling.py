# module_docling.py

import logging
import os
import time
from typing import Union
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR DOCLING ===
docling_presets = {
    "baseline": {
        "extraction_method": "auto",
        "language": "en",
        "include_images": False,
        "include_tables": True,
        "include_headers": True,
        "include_footers": True,
        "ocr_enabled": True,
        "ocr_language": "eng",
        "ocr_config": None,
    },
    "fast": {
        "extraction_method": "text_only",
        "language": "en",
        "include_images": False,
        "include_tables": False,
        "include_headers": False,
        "include_footers": False,
        "ocr_enabled": False,
        "ocr_language": "eng",
        "ocr_config": None,
    },
    "detailed": {
        "extraction_method": "full",
        "language": "en",
        "include_images": True,
        "include_tables": True,
        "include_headers": True,
        "include_footers": True,
        "ocr_enabled": True,
        "ocr_language": "eng",
        "ocr_config": "--psm 3 --oem 3",
    },
    "multilingual": {
        "extraction_method": "auto",
        "language": "auto",
        "include_images": False,
        "include_tables": True,
        "include_headers": True,
        "include_footers": True,
        "ocr_enabled": True,
        "ocr_language": "eng+fra+deu+spa",
        "ocr_config": "--psm 3",
    },
    "table_focused": {
        "extraction_method": "tables",
        "language": "en",
        "include_images": False,
        "include_tables": True,
        "include_headers": False,
        "include_footers": False,
        "ocr_enabled": True,
        "ocr_language": "eng",
        "ocr_config": "--psm 6",
    },
    "text_only": {
        "extraction_method": "text_only",
        "language": "en",
        "include_images": False,
        "include_tables": False,
        "include_headers": True,
        "include_footers": True,
        "ocr_enabled": True,
        "ocr_language": "eng",
        "ocr_config": "--psm 3",
    },
    "ocr_heavy": {
        "extraction_method": "ocr_heavy",
        "language": "en",
        "include_images": False,
        "include_tables": True,
        "include_headers": True,
        "include_footers": True,
        "ocr_enabled": True,
        "ocr_language": "eng",
        "ocr_config": "--psm 3 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}",
    },
}

def parse_docling_preset(
    pdf_path: Union[str, Path],
    preset: str = "baseline",
    output_path: Union[str, Path, None] = None,
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with Docling and preset configuration.
    
    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        extra_opts (dict, optional): Extra kwargs to pass to Docling.
    
    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in docling_presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(docling_presets.keys())}")
    
    config = docling_presets[preset].copy()
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension
    if output_path is None:
        output_path = f"docling_{preset}.txt"

    start_time = time.time()
    try:
        from docling import Document
        
        # Create document with configuration
        doc = Document(
            str(pdf_path),
            language=config["language"],
            ocr_enabled=config["ocr_enabled"],
            ocr_language=config["ocr_language"],
            ocr_config=config["ocr_config"]
        )
        
        # Extract text based on method
        if config["extraction_method"] == "text_only":
            text = doc.extract_text()
        elif config["extraction_method"] == "tables":
            # Focus on table extraction
            text_parts = []
            for page in doc.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)
        elif config["extraction_method"] == "full":
            # Full extraction including metadata
            text_parts = []
            for page in doc.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)
        elif config["extraction_method"] == "ocr_heavy":
            # Force OCR-heavy extraction
            text_parts = []
            for page in doc.pages:
                page_text = page.extract_text(force_ocr=True)
                if page_text.strip():
                    text_parts.append(page_text)
            text = "\n\n".join(text_parts)
        else:  # auto
            text = doc.extract_text()

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        processing_time = time.time() - start_time
        
        # Gather metadata
        metadata = {
            "num_pages": len(doc.pages),
            "extraction_method": config["extraction_method"],
            "language": config["language"],
            "ocr_enabled": config["ocr_enabled"],
            "ocr_language": config["ocr_language"],
            "include_images": config["include_images"],
            "include_tables": config["include_tables"],
            "include_headers": config["include_headers"],
            "include_footers": config["include_footers"],
        }
        
        return {
            "preset": preset,
            "text": text,
            "output_path": str(output_path),
            "processing_time": processing_time,
            "success": True,
            "error": None,
            "metadata": metadata
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("Docling parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "metadata": {}
        } 