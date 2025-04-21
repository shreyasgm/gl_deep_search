"""
Retry utilities for Growth Lab Deep Search.

Provides retry mechanisms with exponential backoff for async operations.
"""

import asyncio
import logging
import random
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retry_on: tuple = (Exception,),
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: The async function to retry
        *args: Arguments to pass to the function
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        retry_on: Tuple of exceptions to retry on
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function

    Raises:
        The last exception encountered if max_retries is exceeded
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retry_on as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)  # Add 10% jitter
            total_delay = delay + jitter

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {total_delay:.1f} seconds..."
            )
            await asyncio.sleep(total_delay)

    # This should never be reached due to the raise in the loop
    raise last_exception  # type: ignore
