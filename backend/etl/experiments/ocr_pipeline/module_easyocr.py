# module_easyocr.py

import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR EASYOCR ===
easyocr_presets = {
    "baseline": {
        "languages": ["en"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 10,
        "width_ths": 0.7,
        "height_ths": 0.7,
        "add_margin": 0.1,
        "contrast_ths": 0.1,
        "adjust_contrast": 0.5,
        "text_threshold": 0.7,
        "link_threshold": 0.4,
        "low_text": 0.4,
        "canvas_size": 2560,
        "mag_ratio": 1.0,
        "slope_ths": 0.1,
        "ycenter_ths": 0.5,
        "add_margin": 0.1,
        "x_ths": 1.0,
        "y_ths": 0.5,
    },
    "fast": {
        "languages": ["en"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 20,
        "width_ths": 0.8,
        "height_ths": 0.8,
        "add_margin": 0.05,
        "contrast_ths": 0.2,
        "adjust_contrast": 0.3,
        "text_threshold": 0.8,
        "link_threshold": 0.5,
        "low_text": 0.5,
        "canvas_size": 1280,
        "mag_ratio": 0.8,
        "slope_ths": 0.2,
        "ycenter_ths": 0.6,
        "add_margin": 0.05,
        "x_ths": 1.2,
        "y_ths": 0.6,
    },
    "accurate": {
        "languages": ["en"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 5,
        "width_ths": 0.6,
        "height_ths": 0.6,
        "add_margin": 0.15,
        "contrast_ths": 0.05,
        "adjust_contrast": 0.7,
        "text_threshold": 0.6,
        "link_threshold": 0.3,
        "low_text": 0.3,
        "canvas_size": 3200,
        "mag_ratio": 1.2,
        "slope_ths": 0.05,
        "ycenter_ths": 0.4,
        "add_margin": 0.15,
        "x_ths": 0.8,
        "y_ths": 0.4,
    },
    "multilingual": {
        "languages": ["en", "fr", "de", "es"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 10,
        "width_ths": 0.7,
        "height_ths": 0.7,
        "add_margin": 0.1,
        "contrast_ths": 0.1,
        "adjust_contrast": 0.5,
        "text_threshold": 0.7,
        "link_threshold": 0.4,
        "low_text": 0.4,
        "canvas_size": 2560,
        "mag_ratio": 1.0,
        "slope_ths": 0.1,
        "ycenter_ths": 0.5,
        "add_margin": 0.1,
        "x_ths": 1.0,
        "y_ths": 0.5,
    },
    "table_focused": {
        "languages": ["en"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 8,
        "width_ths": 0.5,
        "height_ths": 0.5,
        "add_margin": 0.05,
        "contrast_ths": 0.15,
        "adjust_contrast": 0.6,
        "text_threshold": 0.6,
        "link_threshold": 0.3,
        "low_text": 0.3,
        "canvas_size": 2560,
        "mag_ratio": 1.1,
        "slope_ths": 0.05,
        "ycenter_ths": 0.4,
        "add_margin": 0.05,
        "x_ths": 0.8,
        "y_ths": 0.4,
    },
    "handwritten": {
        "languages": ["en"],
        "model_storage_directory": None,
        "download_enabled": True,
        "recog_network": "standard",
        "detect_network": "craft",
        "gpu": False,
        "min_size": 15,
        "width_ths": 0.6,
        "height_ths": 0.6,
        "add_margin": 0.2,
        "contrast_ths": 0.05,
        "adjust_contrast": 0.8,
        "text_threshold": 0.5,
        "link_threshold": 0.2,
        "low_text": 0.2,
        "canvas_size": 3200,
        "mag_ratio": 1.3,
        "slope_ths": 0.05,
        "ycenter_ths": 0.3,
        "add_margin": 0.2,
        "x_ths": 0.7,
        "y_ths": 0.3,
    },
}


def parse_easyocr_preset(
    pdf_path: str | Path,
    preset: str = "baseline",
    output_path: str | Path | None = None,
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with EasyOCR and preset configuration.

    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        extra_opts (dict, optional): Extra kwargs to pass to EasyOCR.

    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in easyocr_presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(easyocr_presets.keys())}"
        )

    config = easyocr_presets[preset].copy()
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension
    if output_path is None:
        output_path = f"easyocr_{preset}.txt"

    start_time = time.time()
    try:
        import easyocr
        from pdf2image import convert_from_path

        # Initialize EasyOCR reader
        reader = easyocr.Reader(
            config["languages"],
            model_storage_directory=config["model_storage_directory"],
            download_enabled=config["download_enabled"],
            gpu=config["gpu"],
        )

        # Convert PDF to images
        images = convert_from_path(str(pdf_path))

        all_text = []
        page_metadata = []
        total_confidence = 0
        total_detections = 0

        for i, image in enumerate(images):
            # Perform OCR with detailed results
            results = reader.readtext(
                image,
                min_size=config["min_size"],
                width_ths=config["width_ths"],
                height_ths=config["height_ths"],
                add_margin=config["add_margin"],
                contrast_ths=config["contrast_ths"],
                adjust_contrast=config["adjust_contrast"],
                text_threshold=config["text_threshold"],
                link_threshold=config["link_threshold"],
                low_text=config["low_text"],
                canvas_size=config["canvas_size"],
                mag_ratio=config["mag_ratio"],
                slope_ths=config["slope_ths"],
                ycenter_ths=config["ycenter_ths"],
                x_ths=config["x_ths"],
                y_ths=config["y_ths"],
                detail=1,  # Get detailed results with confidence scores
            )

            # Extract text and calculate confidence
            page_text = []
            page_confidence = 0
            page_detections = len(results)

            for bbox, text, confidence in results:
                page_text.append(text)
                page_confidence += confidence

            page_text_combined = "\n".join(page_text)
            all_text.append(page_text_combined)

            total_confidence += page_confidence
            total_detections += page_detections

            page_metadata.append(
                {
                    "page": i + 1,
                    "detections": page_detections,
                    "avg_confidence": page_confidence / page_detections
                    if page_detections > 0
                    else 0,
                    "languages": config["languages"],
                }
            )

        # Combine all text
        text = "\n\n".join(all_text)

        # Write output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        processing_time = time.time() - start_time
        avg_confidence = (
            total_confidence / total_detections if total_detections > 0 else 0
        )

        return {
            "preset": preset,
            "text": text,
            "output_path": str(output_path),
            "processing_time": processing_time,
            "success": True,
            "error": None,
            "pages": len(images),
            "languages": config["languages"],
            "total_detections": total_detections,
            "avg_confidence": avg_confidence,
            "page_metadata": page_metadata,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("EasyOCR parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "pages": 0,
            "languages": config.get("languages", []),
            "total_detections": 0,
            "avg_confidence": 0,
            "page_metadata": [],
        }
