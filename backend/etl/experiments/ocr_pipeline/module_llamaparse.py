# module_llamaparse.py

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

# === PRESET CONFIGURATIONS FOR LLAMAPARSE ===
llamaparse_presets = {
    "baseline": {
        "result_type": "markdown",
        "max_workers": 1,
        "show_progress": False,
        "verbose": False,
    },
    "fast": {
        "result_type": "text",
        "max_workers": 1,
        "show_progress": False,
        "verbose": False,
    },
    "detailed": {
        "result_type": "markdown",
        "max_workers": 1,
        "show_progress": True,
        "verbose": True,
    },
    "parallel": {
        "result_type": "markdown",
        "max_workers": 4,
        "show_progress": True,
        "verbose": False,
    },
    "json_output": {
        "result_type": "json",
        "max_workers": 1,
        "show_progress": False,
        "verbose": False,
    },
    "text_only": {
        "result_type": "text",
        "max_workers": 1,
        "show_progress": False,
        "verbose": False,
    },
}

def parse_llamaparse_preset(
    pdf_path: Union[str, Path],
    preset: str = "baseline",
    output_path: Union[str, Path, None] = None,
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with LlamaParse and preset configuration.
    
    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        extra_opts (dict, optional): Extra kwargs to pass to LlamaParse.
    
    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in llamaparse_presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(llamaparse_presets.keys())}")
    
    config = llamaparse_presets[preset].copy()
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension by preset
    if output_path is None:
        ext_map = {
            "markdown": ".md",
            "text": ".txt", 
            "json": ".json"
        }
        ext = ext_map.get(config["result_type"], ".txt")
        output_path = f"llamaparse_{preset}{ext}"

    start_time = time.time()
    try:
        from llama_parse import LlamaParse
        
        parser = LlamaParse(**config)
        
        with open(pdf_path, "rb") as f:
            documents = parser.load_data(f, extra_info={"file_name": str(pdf_path)})
        
        # Extract text based on result type
        if config["result_type"] == "json":
            import json
            text = json.dumps([doc.dict() for doc in documents], ensure_ascii=False, indent=2)
        else:
            text = "\n\n".join(doc.text for doc in documents)

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
            "num_documents": len(documents)
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("LlamaParse parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "num_documents": 0
        } 