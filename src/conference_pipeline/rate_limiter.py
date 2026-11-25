"""Rate limiter for API requests with exponential backoff."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import TypeVar, Callable, Awaitable

from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API requests.

    Implements a sliding window rate limiter that tracks requests
    and enforces a maximum requests per minute (RPM) limit.
    """

    max_rpm: int = 50  # Default to 50 requests per minute
    max_concurrent: int = 5  # Maximum concurrent requests

    _request_times: list[float] = field(default_factory=list)
    _semaphore: asyncio.Semaphore | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self._semaphore = asyncio.Semaphore(self.max_concurrent)

    async def acquire(self) -> None:
        """Acquire permission to make a request.

        Blocks if rate limit would be exceeded or max concurrent requests reached.
        """
        async with self._lock:
            now = time.time()
            # Remove requests older than 60 seconds
            self._request_times = [t for t in self._request_times if now - t < 60]

            # If at rate limit, wait until oldest request expires
            if len(self._request_times) >= self.max_rpm:
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 0.1  # Add small buffer
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    self._request_times = [t for t in self._request_times if now - t < 60]

            # Record this request
            self._request_times.append(time.time())

        # Also respect concurrent request limit
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release the concurrent request slot."""
        self._semaphore.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    rate_limit_codes: tuple[int, ...] = (429,),
) -> T:
    """Execute an async function with exponential backoff on rate limit errors.

    Args:
        func: Async function to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        rate_limit_codes: HTTP status codes that indicate rate limiting.

    Returns:
        Result of the function.

    Raises:
        The last exception if all retries fail.
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            # Check if it's a rate limit error
            is_rate_limit = False
            error_str = str(e).lower()

            if "rate" in error_str and "limit" in error_str:
                is_rate_limit = True
            elif any(str(code) in error_str for code in rate_limit_codes):
                is_rate_limit = True
            elif hasattr(e, "status_code") and e.status_code in rate_limit_codes:
                is_rate_limit = True
            elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                if e.response.status_code in rate_limit_codes:
                    is_rate_limit = True

            if not is_rate_limit or attempt == max_retries:
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), max_delay)
            # Add jitter (±25%)
            import random
            delay = delay * (0.75 + random.random() * 0.5)

            logger.warning(
                f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(delay)

    raise last_exception


# Global rate limiter instance for OpenAI Whisper API
# Can be configured at runtime
_whisper_rate_limiter: RateLimiter | None = None


def get_whisper_rate_limiter(max_rpm: int = 50, max_concurrent: int = 5) -> RateLimiter:
    """Get or create the global Whisper rate limiter.

    Args:
        max_rpm: Maximum requests per minute.
        max_concurrent: Maximum concurrent requests.

    Returns:
        RateLimiter instance.
    """
    global _whisper_rate_limiter

    if _whisper_rate_limiter is None:
        _whisper_rate_limiter = RateLimiter(
            max_rpm=max_rpm,
            max_concurrent=max_concurrent,
        )

    return _whisper_rate_limiter


def reset_whisper_rate_limiter() -> None:
    """Reset the global Whisper rate limiter."""
    global _whisper_rate_limiter
    _whisper_rate_limiter = None
