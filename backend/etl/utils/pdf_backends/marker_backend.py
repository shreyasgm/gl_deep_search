"""
Marker-based PDF extraction backend.

Uses the marker-pdf library for PDF text extraction with layout detection,
OCR, and table recognition.

Supports both marker v0.3.x (convert_single_pdf API) and v1.x+ (PdfConverter API).
"""

from pathlib import Path

from loguru import logger

from backend.etl.utils.gpu_memory import release_gpu_memory
from backend.etl.utils.pdf_backends import register_backend
from backend.etl.utils.pdf_backends.base import ExtractionResult, PDFBackend
from backend.etl.utils.pdf_backends.device import DeviceType, detect_device


def _detect_marker_version() -> int:
    """Detect which major version of marker is installed."""
    try:
        from marker.converters.pdf import PdfConverter  # noqa: F401

        return 1  # v1.x+ API
    except ImportError:
        pass
    try:
        from marker.convert import convert_single_pdf  # noqa: F401

        return 0  # v0.3.x API
    except ImportError:
        pass
    raise ImportError("marker-pdf is not installed or has an unknown API version")


class MarkerBackend(PDFBackend):
    """PDF extraction backend using Marker."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._models = None
        self._converter = None
        self._api_version: int | None = None

    def _get_device_str(self) -> str:
        device = detect_device()
        return {
            DeviceType.CUDA: "cuda",
            DeviceType.MPS: "mps",
            DeviceType.CPU: "cpu",
        }.get(device, "cpu")

    def _load_models(self) -> None:
        """Lazy-load marker models (once, reused across extractions)."""
        if self._models is not None:
            return

        self._api_version = _detect_marker_version()
        device_str = self._get_device_str()

        if self._api_version >= 1:
            from marker.models import create_model_dict

            self._models = create_model_dict(device=device_str)
        else:
            import torch
            from marker.models import load_all_models

            dtype = torch.float32 if device_str == "cpu" else torch.float16
            self._models = load_all_models(device=device_str, dtype=dtype)

        logger.info(
            f"Initialized Marker backend v{self._api_version}.x "
            f"(device={device_str}, force_ocr={self.config.get('force_ocr', False)})"
        )

    def _extract_v0(self, pdf_path: Path) -> ExtractionResult:
        """Extract using marker v0.3.x API."""
        from marker.convert import convert_single_pdf

        text, images, metadata = convert_single_pdf(
            str(pdf_path),
            self._models,
            ocr_all_pages=self.config.get("force_ocr", False),
        )

        return ExtractionResult(
            text=text,
            pages=0,
            backend_name="marker",
            metadata={"marker_metadata": metadata} if metadata else {},
        )

    # Keys from our YAML config that map directly to Marker ConfigParser options.
    _MARKER_PASSTHROUGH_KEYS = frozenset(
        {
            "pdftext_workers",
            "layout_batch_size",
            "detection_batch_size",
            "recognition_batch_size",
            "ocr_error_batch_size",
            "equation_batch_size",
            "table_rec_batch_size",
            "extract_images",
            "disable_ocr",
            "lowres_image_dpi",
            "highres_image_dpi",
            "disable_tqdm",
        }
    )

    def _extract_v1(self, pdf_path: Path) -> ExtractionResult:
        """Extract using marker v1.x+ API."""
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter

        if self._converter is None:
            marker_config: dict = {"output_format": "markdown"}
            if self.config.get("force_ocr", False):
                marker_config["force_ocr"] = True

            # Forward memory/performance settings from our YAML config
            for key in self._MARKER_PASSTHROUGH_KEYS:
                if key in self.config:
                    marker_config[key] = self.config[key]

            logger.debug(f"Marker config: {marker_config}")
            config_parser = ConfigParser(marker_config)
            self._converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=self._models,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )

        rendered = self._converter(str(pdf_path))
        text = rendered.markdown
        metadata = {}
        if rendered.metadata:
            metadata["marker_metadata"] = rendered.metadata

        return ExtractionResult(
            text=text,
            pages=0,
            backend_name="marker",
            metadata=metadata,
        )

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from a PDF using Marker."""
        try:
            self._load_models()

            if self._api_version is not None and self._api_version >= 1:
                return self._extract_v1(pdf_path)
            else:
                return self._extract_v0(pdf_path)

        except Exception as e:
            logger.error(f"Marker extraction failed for {pdf_path}: {e}")
            return ExtractionResult(
                text="",
                pages=0,
                backend_name="marker",
                success=False,
                error=str(e),
            )

    def cleanup(self) -> None:
        """Release Marker models and GPU memory."""
        self._converter = None
        self._models = None
        self._api_version = None
        release_gpu_memory()
        logger.info("Marker backend cleaned up and GPU memory released")


register_backend("marker", MarkerBackend)
