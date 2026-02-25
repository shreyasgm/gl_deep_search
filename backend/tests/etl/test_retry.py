"""Tests for the retry_with_backoff utility."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.etl.utils.retry import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for retry_with_backoff."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Function that succeeds immediately should return without retrying."""
        func = AsyncMock(return_value="ok")

        result = await retry_with_backoff(func, max_retries=3)

        assert result == "ok"
        func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Function that fails N-1 times then succeeds should return the result."""
        func = AsyncMock(
            side_effect=[ValueError("fail1"), ValueError("fail2"), "success"]
        )

        with patch("backend.etl.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            result = await retry_with_backoff(
                func, max_retries=3, retry_on=(ValueError,)
            )

        assert result == "success"
        assert func.await_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self):
        """When max retries are exceeded, the last exception should propagate."""
        func = AsyncMock(side_effect=ValueError("persistent failure"))

        with patch("backend.etl.utils.retry.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="persistent failure"):
                await retry_with_backoff(func, max_retries=2, retry_on=(ValueError,))

        # Initial attempt + 2 retries = 3 total calls
        assert func.await_count == 3

    @pytest.mark.asyncio
    async def test_non_retriable_exception_propagates_immediately(self):
        """Exception type NOT in retry_on should propagate without retrying."""
        func = AsyncMock(side_effect=TypeError("wrong type"))

        with patch(
            "backend.etl.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            with pytest.raises(TypeError, match="wrong type"):
                await retry_with_backoff(
                    func,
                    max_retries=5,
                    retry_on=(ValueError,),  # Only retry ValueError, not TypeError
                )

        # Should have been called exactly once (no retries)
        func.assert_awaited_once()
        mock_sleep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backoff_delay_capped_at_max_delay(self):
        """Verify that the computed delay never exceeds max_delay."""
        func = AsyncMock(
            side_effect=[
                ValueError("1"),
                ValueError("2"),
                ValueError("3"),
                ValueError("4"),
                ValueError("5"),
                "done",
            ]
        )
        max_delay = 10.0
        base_delay = 1.0

        with patch(
            "backend.etl.utils.retry.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep:
            # Patch random.uniform to return 0 jitter for deterministic testing
            with patch("backend.etl.utils.retry.random.uniform", return_value=0.0):
                result = await retry_with_backoff(
                    func,
                    max_retries=5,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    retry_on=(ValueError,),
                )

        assert result == "done"

        # Verify all sleep calls had delay <= max_delay
        for call in mock_sleep.call_args_list:
            delay = call[0][0]
            assert delay <= max_delay, f"Delay {delay} exceeded max_delay {max_delay}"

        # Expected delays: 1, 2, 4, 8, 10 (capped from 16)
        expected_delays = [1.0, 2.0, 4.0, 8.0, 10.0]
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        """Verify that positional and keyword arguments are forwarded."""
        func = AsyncMock(return_value="result")

        result = await retry_with_backoff(
            func, "arg1", "arg2", max_retries=1, key="val"
        )

        assert result == "result"
        func.assert_awaited_once_with("arg1", "arg2", key="val")
