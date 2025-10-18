import logging
import time
import os
from pathlib import Path
from dotenv import load_dotenv
import pytesseract

load_dotenv()

# Set tesseract binary and tessdata paths
pytesseract.pytesseract.tesseract_cmd = "/n/home04/kdaryanani/local/bin/tesseract"
os.environ['TESSDATA_PREFIX'] = '/n/home04/kdaryanani/local/share/tessdata'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === PRESET CONFIGURATIONS FOR TESSERACT (full-page PDFs) ===
# Optimized for feeding academic paper PDFs directly, without prior column or region segmentation.

tesseract_presets = {
    # Default, balanced preset for typical academic paper pages.
    # Works well for most PDFs: clean scans, regular layouts, standard two-column pages.
    "default_full_page": {
        "languages": ["eng"],
        "config": "--psm 3 --oem 1",   # auto layout, no OSD
        "preprocessing": "none",
        "dpi": 300,
    },

    # Slightly stricter single-column assumption; can improve accuracy on PDFs
    # whose text is tightly packed or where column detection misfires.
    "column_like_page": {
        "languages": ["eng"],
        "config": "--psm 4 --oem 1",   # single column mode
        "preprocessing": "none",
        "dpi": 300,
    },

    # Treats each page as one uniform text block.
    # Useful for short reports, letters, abstracts, or pages with clean, centered text.
    "uniform_block": {
        "languages": ["eng"],
        "config": "--psm 6 --oem 1",   # one uniform block
        "preprocessing": "enhance",    # boosts contrast for faint scans
        "dpi": 300,
    },

    # “Catch-everything” mode for messy scans, PDFs with figures, tables,
    # or mixed formatting. Ignores layout, grabs all visible text.
    "messy_sparse": {
        "languages": ["eng"],
        "config": "--psm 11 --oem 3",  # sparse text, no order guaranteed
        "preprocessing": "enhance",
        "dpi": 300,
    },

    # Like messy_sparse, but automatically detects page rotation and script.
    # Best for camera-scanned or rotated pages; slightly slower.
    "rotated_sparse": {
        "languages": ["eng"],
        "config": "--psm 12 --oem 3",  # sparse text + OSD
        "preprocessing": "enhance",
        "dpi": 300,
    },
}



def preprocess_image(image, method: str):
    """Apply image preprocessing based on method."""
    if method == "none":
        return image
    elif method == "enhance":
        from PIL import ImageEnhance

        # Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.2)
        return image
    elif method == "deskew":
        # Simple deskewing (you might want to use more sophisticated methods)
        return image.rotate(0, expand=True)
    else:
        return image


def parse_tesseract_preset(
    pdf_path: str | Path,
    preset: str = "default_full_page",
    output_path: str | Path | None = None,
    extra_opts: dict = None,
) -> dict:
    """
    Parse PDF with Tesseract OCR and preset configuration.

    Args:
        pdf_path (str/Path): Path to input PDF.
        preset (str): Preset name (see above).
        output_path (str/Path, optional): Output file path; auto-names if None.
        extra_opts (dict, optional): Extra kwargs to pass to Tesseract.

    Returns:
        dict: Details about extraction, errors, metadata, timing, etc.
    """
    if preset not in tesseract_presets:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(tesseract_presets.keys())}"
        )

    config = tesseract_presets[preset].copy()
    if extra_opts:
        config.update(extra_opts)

    # Handle output extension
    if output_path is None:
        output_path = f"tesseract_{preset}.txt"

    start_time = time.time()
    try:
        import pytesseract
        from pdf2image import convert_from_path

        # Convert PDF to images
        images = convert_from_path(str(pdf_path), dpi=config["dpi"])

        all_text = []
        page_metadata = []

        for i, image in enumerate(images):
            # Apply preprocessing
            processed_image = preprocess_image(image, config["preprocessing"])

            # Perform OCR
            page_text = pytesseract.image_to_string(
                processed_image,
                lang="+".join(config["languages"]),
                config=config["config"],
            )

            all_text.append(page_text)
            page_metadata.append(
                {
                    "page": i + 1,
                    "dpi": config["dpi"],
                    "preprocessing": config["preprocessing"],
                    "languages": config["languages"],
                }
            )

        # Combine all text
        text = "\n\n".join(all_text)

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
            "pages": len(images),
            "languages": config["languages"],
            "dpi": config["dpi"],
            "preprocessing": config["preprocessing"],
            "page_metadata": page_metadata,
        }
    except Exception as e:
        processing_time = time.time() - start_time
        logger.exception("Tesseract OCR parsing failed.")
        return {
            "preset": preset,
            "text": "",
            "output_path": str(output_path) if output_path else None,
            "processing_time": processing_time,
            "success": False,
            "error": str(e),
            "pages": 0,
            "languages": config.get("languages", []),
            "dpi": config.get("dpi"),
            "preprocessing": config.get("preprocessing"),
            "page_metadata": [],
        }
