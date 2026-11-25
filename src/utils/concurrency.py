"""Concurrency utilities for the NeurIPS pipeline."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class RateLimiter:
    """Simple rate limiter using a token bucket algorithm."""

    def __init__(self, rate: float, burst: int = 1):
        """Initialize rate limiter.

        Args:
            rate: Requests per second allowed.
            burst: Maximum burst size (concurrent requests).
        """
        self.rate = rate
        self.burst = burst
        self._tokens = burst
        self._last_update = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.rate
                logger.debug(f"Rate limiter waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    coro: Awaitable[T],
) -> T:
    """Run a coroutine with semaphore-based concurrency limiting.

    Args:
        semaphore: Semaphore to limit concurrency.
        coro: Coroutine to execute.

    Returns:
        Result of the coroutine.
    """
    async with semaphore:
        return await coro


async def gather_with_concurrency(
    limit: int,
    coros: list[Awaitable[T]],
) -> list[T]:
    """Run coroutines with a concurrency limit.

    Args:
        limit: Maximum concurrent coroutines.
        coros: List of coroutines to execute.

    Returns:
        List of results in the same order as input coroutines.
    """
    semaphore = asyncio.Semaphore(limit)
    return await asyncio.gather(*[run_with_semaphore(semaphore, c) for c in coros])


async def process_batch(
    items: list[T],
    processor: Callable[[T], Awaitable[T]],
    batch_size: int = 10,
    delay_between_batches: float = 0.0,
) -> list[T]:
    """Process items in batches with optional delay between batches.

    Args:
        items: Items to process.
        processor: Async function to process each item.
        batch_size: Number of items per batch.
        delay_between_batches: Delay in seconds between batches.

    Returns:
        List of processed items.
    """
    results: list[T] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)

        if delay_between_batches > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)

    return results
