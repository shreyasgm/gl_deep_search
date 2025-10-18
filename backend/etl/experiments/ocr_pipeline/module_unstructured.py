# module_unstructured.py

import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import unstructured_pytesseract

unstructured_pytesseract.tesseract_cmd = r"/n/home04/kdaryanani/local/bin/tesseract"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR UNSTRUCTURED ===
unstructured_presets = {
    "baseline": {
        "strategy": "fast",
        "infer_table_structure": False,
        "extract_images_in_pdf": False,
    },
    "informed_layout_base": {
        "strategy": "hi_res",
        "infer_table_structure": False,
        "extract_images_in_pdf": False,
    },
    "informed_layout_tables": {
        "strategy": "hi_res",
        "infer_table_structure": True,
        "extract_images_in_pdf": False,
    },
    "informed_layout_tables_and_images": {
        "strategy": "hi_res",
        "infer_table_structure": True,
        "extract_images_in_pdf": True,
    },
    "ocr_base": {
        "strategy": "ocr_only",
        "infer_table_structure": False,
        "extract_images_in_pdf": False,
    },
    "ocr_tables": {
        "strategy": "ocr_only",
        "infer_table_structure": True,
        "extract_images_in_pdf": False,
    },
    "ocr_tables_and_images": {
        "strategy": "ocr_only",
        "infer_table_structure": True,
        "extract_images_in_pdf": True,
    },
}

def parse_unstructured_preset(
    pdf_path: str | Path,
    preset: str = "baseline",
    output_path: str | Path | None = None,
    languages: list = ["eng"],
    postprocess_mode: str = "basic",
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with unstructured and preset configuration.
    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        languages (list): List of OCR languages.
        postprocess_mode (str): 'basic', 'keep_structure', or 'plain_words'.
        extra_opts (dict, optional): Extra kwargs to pass to unstructured.
    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in unstructured_presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(unstructured_presets.keys())}"
        )

    config = unstructured_presets[preset].copy()
    config["filename"] = str(pdf_path)
    config["include_page_breaks"] = True
    config["languages"] = languages
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension by preset
    if output_path is None:
        ext = ".txt"
        if config["strategy"] == "hi_res" and (
            config.get("infer_table_structure") or config.get("extract_images_in_pdf")
        ):
            ext = ".json"
        output_path = f"unstructured_{preset}{ext}"

    start_time = time.time()
    try:
        from unstructured.cleaners.core import (
            clean,
            remove_punctuation,
            replace_unicode_quotes,
        )
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(**config)

        # Postprocessing
        for el in elements:
            if hasattr(el, "text") and isinstance(el.text, str):
                if postprocess_mode == "basic":
                    el.text = clean(
                        replace_unicode_quotes(el.text),
                        bullets=True,
                        extra_whitespace=True,
                    )
                elif postprocess_mode == "keep_structure":
                    el.text = replace_unicode_quotes(el.text)
                elif postprocess_mode == "plain_words":
                    el.text = clean(
                        el.text, bullets=True, extra_whitespace=True, lowercase=True
                    )
                    el.text = remove_punctuation(el.text)

        # Safe stringification
        try:
            text = "\n\n".join(str(el) for el in elements)
        except Exception:
            # Fallback if elements are not directly stringify-able
            import json

            text = json.dumps(
                [el.to_dict() for el in elements], ensure_ascii=False, indent=2
            )

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        processing_time = time.time() - start_time
        return {
            "preset": preset,
            "text": text,
            "output_path": str(output_path),
            "processing_time": processing_time,
            "success": True,
            "error": None,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("Unstructured parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
        }

