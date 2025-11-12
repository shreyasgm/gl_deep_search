"""
Profiling utilities for ETL pipeline components.

Provides timing and resource monitoring for ETL operations with minimal overhead.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

import psutil
from loguru import logger


@dataclass
class ProfileResult:
    """Result of a profiling operation."""

    operation: str
    duration: float  # seconds
    memory_delta_mb: float | None = None
    cpu_percent: float | None = None
    peak_memory_mb: float | None = None


@contextmanager
def profile_operation(
    operation_name: str,
    log_level: str = "INFO",
    include_resources: bool = False,
) -> Generator[ProfileResult, None, None]:
    """
    Context manager for profiling code blocks.

    Args:
        operation_name: Name of the operation being profiled
        log_level: Log level for timing info ("INFO" or "DEBUG")
        include_resources: Whether to collect resource metrics (memory/CPU)

    Yields:
        ProfileResult object that gets populated with metrics

    Example:
        with profile_operation("Download files", include_resources=True) as prof:
            # ... do work ...
            pass
        # Metrics are automatically logged when context exits
    """
    # Initialize result object
    result = ProfileResult(operation=operation_name, duration=0.0)

    # Capture initial state
    start_time = time.perf_counter()
    process = psutil.Process() if include_resources else None
    initial_memory = None
    initial_cpu_times = None

    if include_resources and process:
        try:
            # Get baseline memory usage
            mem_info = process.memory_info()
            initial_memory = mem_info.rss / (1024 * 1024)  # Convert to MB
            initial_cpu_times = process.cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.debug(f"Could not capture initial resource metrics: {e}")

    try:
        # Yield control to the code block
        yield result

    finally:
        # Capture final state
        end_time = time.perf_counter()
        result.duration = end_time - start_time

        # Collect resource metrics if requested
        if include_resources and process and initial_memory is not None:
            try:
                # Memory delta
                final_mem_info = process.memory_info()
                final_memory = final_mem_info.rss / (1024 * 1024)
                result.memory_delta_mb = final_memory - initial_memory
                result.peak_memory_mb = final_memory

                # CPU usage (approximate)
                final_cpu_times = process.cpu_times()
                if initial_cpu_times:
                    cpu_delta = (
                        final_cpu_times.user
                        + final_cpu_times.system
                        - initial_cpu_times.user
                        - initial_cpu_times.system
                    )
                    # Calculate percentage based on wall time
                    result.cpu_percent = (
                        (cpu_delta / result.duration * 100)
                        if result.duration > 0
                        else 0
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.debug(f"Could not capture final resource metrics: {e}")

        # Log results
        _log_profile_result(result, log_level, include_resources)


def _log_profile_result(
    result: ProfileResult, log_level: str, include_resources: bool
) -> None:
    """Log profiling results at appropriate level."""
    # Format timing message
    duration_str = _format_duration(result.duration)
    timing_msg = f"[PROFILE] {result.operation}: {duration_str}"

    # Log timing at requested level
    if log_level.upper() == "DEBUG":
        logger.debug(timing_msg)
    else:
        logger.info(timing_msg)

    # Log resource metrics at DEBUG level if available
    if include_resources and (
        result.memory_delta_mb is not None or result.cpu_percent is not None
    ):
        resource_parts = []
        if result.memory_delta_mb is not None:
            resource_parts.append(f"Memory Δ: {result.memory_delta_mb:+.1f} MB")
            if result.peak_memory_mb is not None:
                resource_parts.append(f"Peak: {result.peak_memory_mb:.1f} MB")
        if result.cpu_percent is not None:
            resource_parts.append(f"CPU: {result.cpu_percent:.1f}%")

        resource_msg = (
            f"[PROFILE] {result.operation} resources: {', '.join(resource_parts)}"
        )
        logger.debug(resource_msg)


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def log_component_metrics(component_name: str, metrics: dict) -> None:
    """
    Log component-specific metrics in a standardized format.

    Args:
        component_name: Name of the component
        metrics: Dictionary of metrics to log
    """
    logger.info(f"[METRICS] {component_name}:")
    for key, value in metrics.items():
        # Format value based on type
        if isinstance(value, float):
            if "time" in key.lower() or "duration" in key.lower():
                formatted_value = _format_duration(value)
            elif "size" in key.lower() and "mb" in key.lower():
                formatted_value = f"{value:.2f} MB"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)

        logger.info(f"  {key}: {formatted_value}")
