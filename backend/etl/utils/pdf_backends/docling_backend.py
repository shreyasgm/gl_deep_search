"""
Docling-based PDF extraction backend.

Uses the Docling library for high-quality PDF text extraction with OCR,
table structure recognition, and optional picture description.
"""

from pathlib import Path

from loguru import logger

from backend.etl.utils.pdf_backends import register_backend
from backend.etl.utils.pdf_backends.base import ExtractionResult, PDFBackend
from backend.etl.utils.pdf_backends.device import DeviceType, detect_device


class DoclingBackend(PDFBackend):
    """PDF extraction backend using Docling."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config)
        self._converter = None

    def _get_converter(self):
        """Lazy-initialize the Docling DocumentConverter."""
        if self._converter is not None:
            return self._converter

        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.get("do_ocr", True)
        pipeline_options.do_table_structure = self.config.get(
            "do_table_structure", True
        )

        if self.config.get("table_mode_accurate", True):
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE,
            )

        pipeline_options.do_picture_description = self.config.get(
            "do_picture_description", False
        )

        # Map device detection to Docling's accelerator
        device = detect_device()
        device_map = {
            DeviceType.CUDA: AcceleratorDevice.CUDA,
            DeviceType.MPS: AcceleratorDevice.MPS,
            DeviceType.CPU: AcceleratorDevice.CPU,
        }
        accel_device = device_map.get(device, AcceleratorDevice.CPU)
        num_threads = self.config.get("num_threads", 4)

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=accel_device,
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        logger.info(
            f"Initialized Docling backend (device={device.value}, "
            f"ocr={pipeline_options.do_ocr}, "
            f"table_structure={pipeline_options.do_table_structure})"
        )
        return self._converter

    def extract(self, pdf_path: Path) -> ExtractionResult:
        """Extract text from a PDF using Docling."""
        try:
            from docling.datamodel.base_models import ConversionStatus

            converter = self._get_converter()
            result = converter.convert(pdf_path)

            if result.status not in (
                ConversionStatus.SUCCESS,
                ConversionStatus.PARTIAL_SUCCESS,
            ):
                return ExtractionResult(
                    text="",
                    pages=0,
                    backend_name="docling",
                    success=False,
                    error=f"Conversion failed with status: {result.status}",
                )

            markdown = result.document.export_to_markdown()
            num_pages = (
                len(result.document.pages) if hasattr(result.document, "pages") else 0
            )
            metadata = {}
            if hasattr(result.document, "tables"):
                metadata["num_tables"] = len(result.document.tables)
            if hasattr(result.document, "pictures"):
                metadata["num_pictures"] = len(result.document.pictures)

            return ExtractionResult(
                text=markdown,
                pages=num_pages,
                backend_name="docling",
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Docling extraction failed for {pdf_path}: {e}")
            return ExtractionResult(
                text="",
                pages=0,
                backend_name="docling",
                success=False,
                error=str(e),
            )

    def extract_batch(self, pdf_paths: list[Path]) -> list[ExtractionResult]:
        """Extract text from multiple PDFs using Docling's native batch API."""
        try:
            from docling.datamodel.base_models import ConversionStatus

            converter = self._get_converter()
            results = []

            for conv_result in converter.convert_all(
                source=pdf_paths,
                raises_on_error=False,
            ):
                if conv_result.status in (
                    ConversionStatus.SUCCESS,
                    ConversionStatus.PARTIAL_SUCCESS,
                ):
                    markdown = conv_result.document.export_to_markdown()
                    num_pages = (
                        len(conv_result.document.pages)
                        if hasattr(conv_result.document, "pages")
                        else 0
                    )
                    results.append(
                        ExtractionResult(
                            text=markdown,
                            pages=num_pages,
                            backend_name="docling",
                        )
                    )
                else:
                    source_name = getattr(
                        conv_result.input, "file", pdf_paths[len(results)]
                    )
                    logger.warning(f"Docling batch: failed for {source_name}")
                    results.append(
                        ExtractionResult(
                            text="",
                            pages=0,
                            backend_name="docling",
                            success=False,
                            error=f"Conversion status: {conv_result.status}",
                        )
                    )

            return results

        except Exception as e:
            logger.error(f"Docling batch extraction failed: {e}")
            return [
                ExtractionResult(
                    text="",
                    pages=0,
                    backend_name="docling",
                    success=False,
                    error=str(e),
                )
                for _ in pdf_paths
            ]

    def cleanup(self) -> None:
        """Release Docling converter and GPU memory."""
        self._converter = None
        logger.debug("Docling backend cleaned up")


register_backend("docling", DoclingBackend)
