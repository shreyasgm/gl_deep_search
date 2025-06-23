# module_mistral.py

import logging
import os
import time
import base64
from typing import Union
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR MISTRAL ===
mistral_presets = {
    "baseline": {
        "model": "mistral-ocr-latest",
        "output_format": "markdown",
        "language": "auto",
    },
    "fast": {
        "model": "mistral-ocr-latest",
        "output_format": "text",
        "language": "auto",
    },
    "detailed": {
        "model": "mistral-ocr-latest",
        "output_format": "markdown",
        "language": "auto",
        "include_confidence": True,
    },
    "multilingual": {
        "model": "mistral-ocr-latest",
        "output_format": "markdown",
        "language": "auto",
        "detect_language": True,
    },
    "json_output": {
        "model": "mistral-ocr-latest",
        "output_format": "json",
        "language": "auto",
    },
    "text_only": {
        "model": "mistral-ocr-latest",
        "output_format": "text",
        "language": "auto",
    },
}

def parse_mistral_preset(
    pdf_path: Union[str, Path],
    preset: str = "baseline",
    output_path: Union[str, Path, None] = None,
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with Mistral OCR and preset configuration.
    
    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        extra_opts (dict, optional): Extra kwargs to pass to Mistral OCR.
    
    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in mistral_presets:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(mistral_presets.keys())}")
    
    config = mistral_presets[preset].copy()
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension by preset
    if output_path is None:
        ext_map = {
            "markdown": ".md",
            "text": ".txt", 
            "json": ".json"
        }
        ext = ext_map.get(config["output_format"], ".txt")
        output_path = f"mistral_{preset}{ext}"

    # Check for API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError("MISTRAL_API_KEY not found in environment")

    start_time = time.time()
    try:
        from mistralai import Mistral
        
        client = Mistral(api_key=api_key)
        
        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode("utf-8")
        
        # Prepare document payload
        document_payload = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{encoded_pdf}"
        }
        
        # Add language if specified
        if config.get("language") and config["language"] != "auto":
            document_payload["language"] = config["language"]
        
        # Process OCR
        ocr_response = client.ocr.process(
            model=config["model"],
            document=document_payload
        )
        
        # Extract text based on output format
        if config["output_format"] == "markdown":
            text = ocr_response.markdown
        elif config["output_format"] == "text":
            text = ocr_response.text
        elif config["output_format"] == "json":
            import json
            text = json.dumps(ocr_response.model_dump(), ensure_ascii=False, indent=2)
        else:
            text = ocr_response.text

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
            "model": config["model"],
            "output_format": config["output_format"]
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("Mistral OCR parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "model": config.get("model"),
            "output_format": config.get("output_format")
        } 