# module_marker.py

import logging
import os
import time
from typing import Union
from pathlib import Path

# marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

from dotenv import load_dotenv
load_dotenv()

#configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR MARKER===

preset_configs = {
    "baseline": {
        "force_ocr": False,
        "use_llm": False,
        "format_lines": False,
        "output_format": "markdown",
        "disable_image_extraction": True
    },
    "force_ocr": {
        "force_ocr": True,
        "strip_existing_ocr": True,
        "use_llm": False,
        "format_lines": False,
        "output_format": "markdown",
        "disable_image_extraction": True
    },
    "use_llm": {
        "force_ocr": False,
        "use_llm": True,
        "format_lines": False,
        "output_format": "markdown",
        "disable_image_extraction": True
    },
    "ocr_and_llm": {
        "force_ocr": True,
        "strip_existing_ocr": True,
        "use_llm": True,
        "format_lines": False,
        "output_format": "markdown",
        "disable_image_extraction": True
    },
    "ocr_and_llm_math": {
        "force_ocr": True,
        "strip_existing_ocr": True,
        "use_llm": True,
        "format_lines": True,
        "redo_inline_math": True, #only works with llm
        "output_format": "markdown",
        "disable_image_extraction": True
    }
}

# === MODEL CONFIGURATIONS FOR USING LLM==

model_configs = {
    "gpt-4.1": {
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_model": "gpt-4.1",
        "openai_base_url": "https://api.openai.com/v1"
    },
    "gpt-4.1-mini": {
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_model": "gpt-4.1-mini",
        "openai_base_url": "https://api.openai.com/v1"
    },
    "gpt-4.1-nano": {
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_model": "gpt-4.1-nano",
        "openai_base_url": "https://api.openai.com/v1"
    },
    "gpt-4o-mini": {
        "llm_service": "marker.services.openai.OpenAIService",
        "openai_model": "gpt-4o-mini",
        "openai_base_url": "https://api.openai.com/v1"
    },
    # Gemini (cost-effective/fast, non-heavy-reasoning)
    "gemini-flash-lite": {
        "llm_service": "marker.services.gemini.GoogleGeminiService",
        "gemini_model": "models/gemini-2.5-flash-lite-preview-06-17"
    },
    # Claude family, latest recommended versions
    "claude-3-7-sonnet-latest": {
        "llm_service": "marker.services.claude.ClaudeService",
        "claude_model_name": "claude-3-7-sonnet-20250219"
    },
    "claude-3-5-haiku-latest": {
        "llm_service": "marker.services.claude.ClaudeService",
        "claude_model_name": "claude-3-5-haiku-20241022"
    },
    "none": {}
}

def parse_marker_preset(
    pdf_path: Union[str, Path],
    preset: str = "baseline",
    model: str = "gpt-4o-mini",
    output_path: Union[str, Path, None] = None,
) -> dict:
    """
    Parse PDF with marker preset/model configuration merged.

    Args:
        pdf_path (str or Path): Path to the input PDF.
        preset (str): Name of the preset to use.
        model (str): Name of the model (if preset uses LLM).
        output_path (str or Path, optional): Where to save the output. Default is auto-generated.

    Returns:
        dict: Details about extraction, possible error, metadata, timing, etc.
    """

    # ---- Validate and merge config ----
    if preset not in preset_configs:
        raise ValueError(f"Unknown preset '{preset}'. Available presets: {list(preset_configs.keys())}")
    if model not in model_configs:
        raise ValueError(f"Unknown model '{model}'. Available models: {list(model_configs.keys())}")
    config = preset_configs[preset].copy()
    if config.get("use_llm"):
        config.update(model_configs[model])

    # ---- Dynamic output extension based on output format ----
    if output_path is None:
        ext = ".md" if config.get("output_format", "markdown") == "markdown" else ".json"
        output_path = f"output_{preset}_{model}{ext}"

    # ---- API key resolution (OpenAI/Gemini/Claude etc) ----
    if config.get("use_llm"):
        llm_service = config.get("llm_service", "")
        if "openai" in llm_service.lower():
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY required for OpenAI service")
            config["openai_api_key"] = api_key

        elif "gemini" in llm_service.lower():
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini service")
            config["gemini_api_key"] = api_key

        elif "claude" in llm_service.lower():
            api_key = os.getenv("CLAUDE_API_KEY")
            if not api_key:
                raise EnvironmentError("CLAUDE_API_KEY required for Claude service")
            config["claude_api_key"] = api_key

    # ---- Main processing block ----
    start_time = time.time()

    try:
        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service() if config.get("use_llm") else None
        )
        rendered = converter(pdf_path)
        text, metadata, images = text_from_rendered(rendered)

        # Write output
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)

        processing_time = time.time() - start_time

        return {
            "preset": preset,
            "model": model,
            "text": text,
            "metadata": metadata,
            "images": images,
            "output_path": str(output_path),
            "processing_time": processing_time,
            "success": True,
            "error": None
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("Marker parsing failed.")
        return {
            "preset": preset,
            "model": model,
            "text": "",
            "metadata": {},
            "images": [],
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e)
        }








