"""GPU memory management utilities for the ETL pipeline."""

import gc

from loguru import logger


def release_gpu_memory() -> None:
    """Force Python GC and release cached GPU memory (CUDA or MPS).

    Safe to call on any platform — silently skips if torch is not
    installed or no GPU is available.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Released CUDA cached memory")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            logger.debug("Released MPS cached memory")
    except ImportError:
        pass  # torch not installed
