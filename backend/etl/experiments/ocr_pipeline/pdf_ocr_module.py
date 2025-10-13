# pdf_modules.py

import logging
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#combine the ocr module and the run_pdf script into one simple thing
#command line - just take pdf path, output path, module name, preset
#you can edit the batch processing to do the grid search seperately instead of the current submit-jobs


def parse_pdf(
    pdf_path: str | Path,
    module: str = "marker",
    preset: str = "baseline", 
    model: str = "none",
    output_path: str | Path = None,
    **kwargs
) -> dict:
    """
    Parse PDF using specified module and preset.
    
    Args:
        pdf_path: Path to PDF file
        module: Parser module ('marker', 'unstructured', 'tesseract', etc.)
        preset: Configuration preset for the module
        model: Model to use (for LLM-enabled presets)
        output_path: Where to save output
        
    Returns:
        dict: Parser result with text, metadata, timing, etc.
    """
    logger.info(f"Processing {Path(pdf_path).name} with {module} (preset: {preset}, model: {model})")
    
    try:
        if module == "marker":
            from module_marker import parse_marker_preset
            return parse_marker_preset(pdf_path, preset=preset, model=model, output_path=output_path)
            
        elif module == "unstructured":
            from module_unstructured import parse_unstructured_preset
            return parse_unstructured_preset(pdf_path, preset=preset, output_path=output_path)
            
        elif module == "tesseract":
            from module_tesseract import parse_tesseract_preset
            return parse_tesseract_preset(pdf_path, preset=preset, output_path=output_path)
            
        else:
            raise ValueError(f"Unknown module '{module}'. Available: marker, unstructured, tesseract")
            
    except Exception as e:
        logger.error(f"Parser {module} failed on {pdf_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "module": module,
            "preset": preset,
            "model": model,
            "pdf_path": str(pdf_path),
            "output_path": str(output_path) if output_path else None,
        }