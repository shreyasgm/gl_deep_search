"""Route standard-library logging through loguru.

Five ETL modules (`pdf_processor`, `gl_file_downloader`, `oa_file_downloader`,
`growthlab`, `publication_tracker`) log via the standard library, while the
orchestrator configures only loguru. With no root handler installed, stdlib
INFO records are discarded outright and WARNING/ERROR fall through to
``logging.lastResort`` as bare, unformatted stderr.

That is why the 2026-08-22 production run emitted nothing for the 16 hours it
spent in PDF extraction: every progress line in `pdf_processor` was dropped on
the floor. The standalone `run_*.py` scripts each installed their own
interceptor, so the gap only appeared on the orchestrated path.
"""

import logging
from types import FrameType

from loguru import logger


class InterceptHandler(logging.Handler):
    """Forward standard-library log records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Re-emit a stdlib record through loguru at the matching level.

        Args:
            record: The standard-library log record to forward.
        """
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk out of the logging module so loguru reports the true caller.
        frame: FrameType | None = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def install_stdlib_bridge(level: int = 0) -> None:
    """Send all standard-library logging through loguru.

    Safe to call repeatedly; ``force=True`` replaces any handlers a
    third-party import may have installed.

    Args:
        level: Root logger level. 0 lets loguru's own sink decide.
    """
    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)
