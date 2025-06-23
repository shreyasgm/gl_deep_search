# module_unstructured.py

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
        "extract_images_in_pdf": True
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

for el in elements:
    if el.category == "Table":
        html = el.metadata.text_as_html

def parse_unstructured_preset(
    pdf_path: Union[str, Path],
    preset: str = "baseline",
    output_path: Union[str, Path, None] = None,
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
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(unstructured_presets.keys())}")
    
    config = unstructured_presets[preset].copy()
    config["filename"] = str(pdf_path)
    config["include_page_breaks"] = True
    config["languages"] = languages
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension by preset
    if output_path is None:
        ext = ".txt"
        if config["strategy"] == "hi_res" and (config.get("infer_table_structure") or config.get("extract_images_in_pdf")):
            ext = ".json"
        output_path = f"unstructured_{preset}{ext}"

    start_time = time.time()
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.cleaners.core import clean, replace_unicode_quotes, remove_punctuation

        elements = partition_pdf(**config)

        # Postprocessing
        for el in elements:
            if hasattr(el, "text") and isinstance(el.text, str):
                if postprocess_mode == "basic":
                    el.text = clean(replace_unicode_quotes(el.text), bullets=True, extra_whitespace=True)
                elif postprocess_mode == "keep_structure":
                    el.text = replace_unicode_quotes(el.text)
                elif postprocess_mode == "plain_words":
                    el.text = clean(el.text, bullets=True, extra_whitespace=True, lowercase=True)
                    el.text = remove_punctuation(el.text)

        # Safe stringification
        try:
            text = "\n\n".join(str(el) for el in elements)
        except Exception:
            # Fallback if elements are not directly stringify-able
            import json
            text = json.dumps([el.to_dict() for el in elements], ensure_ascii=False, indent=2)

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
            "error": None
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
            "error": str(e)
        }


# %%
result = parse_unstructured_preset("/n/hausmann_lab/lab/kdaryanani/deeplearn/gl_deep_search/backend/etl/experiments/ocr_pipeline/downloaded_papers/gl_url_0ffdb26974b640b8/nostalgic_trade_albanian_americans.pdf", preset="informed_layout_tables_and_images")  # uses default 'baseline' preset
print("====== Results ======")
print("Success:", result["success"])
print("Output Path:", result["output_path"])
print("Processing Time:", result["processing_time"])
print("Error:", result["error"])
print("------ Text Preview ------\n", result["text"][:1000])  # Shows first 1000 chars

# %%
from collections import Counter

display(Counter(type(element) for element in elements))
print("")

# %%
display(*[(type(element), element.text) for element in elements[10:13]])

# %%
text = "\n\n".join(str(el) for el in elements)

print(text)


