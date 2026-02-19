"""
PDF extraction backend registry and factory.

Usage:
    from backend.etl.utils.pdf_backends import get_backend
    backend = get_backend("marker", config={"force_ocr": False})
    result = backend.extract(pdf_path)
"""

import importlib
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from backend.etl.utils.pdf_backends.base import PDFBackend

# Registry mapping backend names to their classes
BACKEND_REGISTRY: dict[str, type["PDFBackend"]] = {}

# Mapping from backend name to module path for lazy imports
_BACKEND_MODULES: dict[str, str] = {
    "marker": "backend.etl.utils.pdf_backends.marker_backend",
    "docling": "backend.etl.utils.pdf_backends.docling_backend",
}


def register_backend(name: str, cls: type["PDFBackend"]) -> None:
    """Register a PDF backend class under the given name."""
    BACKEND_REGISTRY[name] = cls
    logger.debug(f"Registered PDF backend: {name}")


def get_backend(name: str, config: dict | None = None) -> "PDFBackend":
    """
    Get an instance of the named PDF backend.

    Lazily imports the backend module to trigger registration, then
    instantiates the backend class with the provided config.

    Args:
        name: Backend name (marker, docling).
        config: Backend-specific configuration dict.

    Returns:
        An initialized PDFBackend instance.

    Raises:
        ValueError: If the backend name is not recognized.
        ImportError: If the backend's dependencies are not installed.
    """
    if name not in BACKEND_REGISTRY:
        if name in _BACKEND_MODULES:
            logger.debug(f"Lazy-importing backend module: {_BACKEND_MODULES[name]}")
            importlib.import_module(_BACKEND_MODULES[name])
        else:
            raise ValueError(
                f"Unknown PDF backend: {name!r}. "
                f"Available: {list(_BACKEND_MODULES.keys())}"
            )

    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Backend {name!r} module loaded but did not register itself. "
            f"Check that the module calls register_backend()."
        )

    return BACKEND_REGISTRY[name](config=config)
