"""
GPU and device auto-detection for PDF processing backends.
"""

import os
from enum import Enum

from loguru import logger


class DeviceType(Enum):
    """Available compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def detect_device() -> DeviceType:
    """
    Detect the best available compute device.

    Checks (in order):
    1. PDF_DEVICE environment variable override
    2. CUDA availability via torch
    3. MPS availability via torch (Apple Silicon)
    4. Falls back to CPU

    Returns:
        DeviceType indicating the detected device.
    """
    env_override = os.environ.get("PDF_DEVICE", "").lower()
    if env_override in ("cuda", "mps", "cpu"):
        device = DeviceType(env_override)
        logger.info(f"Using device from PDF_DEVICE env var: {device.value}")
        return device

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Detected CUDA device")
            return DeviceType.CUDA
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Detected MPS device (Apple Silicon)")
            return DeviceType.MPS
    except ImportError:
        logger.debug("torch not available; falling back to CPU")

    logger.info("Using CPU device")
    return DeviceType.CPU


def is_slurm_environment() -> bool:
    """Check if running inside a SLURM job."""
    return "SLURM_JOB_ID" in os.environ
