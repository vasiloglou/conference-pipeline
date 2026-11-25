"""HTTP utilities with retry logic and rate limiting."""

import asyncio
import random
from typing import Any
from urllib.parse import urlparse

import httpx

from utils.logging import get_logger

logger = get_logger(__name__)


class DomainRateLimiter:
    """Per-domain rate limiter using semaphores.

    Limits concurrent requests to each domain to prevent overloading servers
    and avoid being blocked.
    """

    _instance: "DomainRateLimiter | None" = None
    _lock = asyncio.Lock()

    def __init__(self, default_limit: int = 4):
        """Initialize rate limiter.

        Args:
            default_limit: Default concurrent request limit for unknown domains.
        """
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._domain_limits: dict[str, int] = {
            "neurips.cc": 3,
            "slideslive.com": 3,
            "arxiv.org": 3,
            "api.crossref.org": 5,
            "crossref.org": 5,
            "api.semanticscholar.org": 5,
            "semanticscholar.org": 5,
        }
        self._default_limit = default_limit
        self._local_lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, default_limit: int = 4) -> "DomainRateLimiter":
        """Get singleton instance of rate limiter.

        Args:
            default_limit: Default concurrent request limit.

        Returns:
            DomainRateLimiter instance.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(default_limit)
            return cls._instance

    def set_domain_limit(self, domain: str, limit: int) -> None:
        """Set rate limit for a specific domain.

        Args:
            domain: Domain name.
            limit: Max concurrent requests.
        """
        self._domain_limits[domain.lower()] = limit

    def _get_limit_for_domain(self, domain: str) -> int:
        """Get rate limit for a domain."""
        domain_lower = domain.lower()

        # Check exact match first
        if domain_lower in self._domain_limits:
            return self._domain_limits[domain_lower]

        # Check partial matches
        for key, limit in self._domain_limits.items():
            if key in domain_lower or domain_lower in key:
                return limit

        return self._default_limit

    async def _get_semaphore(self, domain: str) -> asyncio.Semaphore:
        """Get or create semaphore for a domain."""
        async with self._local_lock:
            if domain not in self._semaphores:
                limit = self._get_limit_for_domain(domain)
                self._semaphores[domain] = asyncio.Semaphore(limit)
                logger.debug(f"Created rate limiter for {domain} with limit {limit}")
            return self._semaphores[domain]

    async def acquire(self, url: str) -> "asyncio.Semaphore":
        """Acquire rate limit for a URL.

        Args:
            url: URL to acquire rate limit for.

        Returns:
            The semaphore that was acquired.
        """
        domain = urlparse(url).netloc
        semaphore = await self._get_semaphore(domain)
        await semaphore.acquire()
        return semaphore

    def release(self, url: str, semaphore: asyncio.Semaphore) -> None:
        """Release rate limit for a URL.

        Args:
            url: URL that was accessed.
            semaphore: The semaphore to release.
        """
        semaphore.release()


# Global rate limiter instance
_rate_limiter: DomainRateLimiter | None = None


async def get_rate_limiter() -> DomainRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = await DomainRateLimiter.get_instance()
    return _rate_limiter


class RetryConfig:
    """Configuration for HTTP retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_statuses: set[int] | None = None,
        jitter: float = 0.1,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.retry_statuses = retry_statuses or {429, 500, 502, 503, 504}
        self.jitter = jitter


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    retry_config: RetryConfig | None = None,
    use_rate_limiting: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    """Make an HTTP request with exponential backoff retry and rate limiting.

    Args:
        client: httpx async client.
        method: HTTP method (GET, POST, etc.).
        url: Request URL.
        retry_config: Retry configuration. Uses defaults if None.
        use_rate_limiting: Whether to apply per-domain rate limiting (default True).
        **kwargs: Additional arguments passed to client.request().

    Returns:
        HTTP response.

    Raises:
        httpx.HTTPStatusError: If request fails after all retries.
    """
    config = retry_config or RetryConfig()
    last_exception: Exception | None = None

    # Get rate limiter if needed
    rate_limiter = None
    semaphore = None
    if use_rate_limiting:
        rate_limiter = await get_rate_limiter()
        semaphore = await rate_limiter.acquire(url)

    try:
        for attempt in range(config.max_retries + 1):
            try:
                response = await client.request(method, url, **kwargs)

                # Check if we should retry based on status code
                if response.status_code in config.retry_statuses and attempt < config.max_retries:
                    wait_time = config.backoff_factor ** attempt
                    jitter = random.uniform(-config.jitter, config.jitter) * wait_time
                    wait_time += jitter
                    logger.warning(
                        f"Request to {url} returned {response.status_code}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{config.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response

            except httpx.TimeoutException as e:
                last_exception = e
                if attempt < config.max_retries:
                    wait_time = config.backoff_factor ** attempt
                    logger.warning(
                        f"Request to {url} timed out, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{config.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request to {url} failed after {config.max_retries} retries: {e}")
                    raise

            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code in config.retry_statuses and attempt < config.max_retries:
                    wait_time = config.backoff_factor ** attempt
                    jitter = random.uniform(-config.jitter, config.jitter) * wait_time
                    wait_time += jitter
                    logger.warning(
                        f"Request to {url} failed with {e.response.status_code}, "
                        f"retrying in {wait_time:.2f}s (attempt {attempt + 1}/{config.max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request to {url} failed: {e}")
                    raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError(f"Request to {url} failed unexpectedly")

    finally:
        # Always release the rate limit
        if rate_limiter and semaphore:
            rate_limiter.release(url, semaphore)


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Convenience wrapper for GET requests with retry."""
    return await request_with_retry(client, "GET", url, retry_config, **kwargs)


async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs: Any,
) -> httpx.Response:
    """Convenience wrapper for POST requests with retry."""
    return await request_with_retry(client, "POST", url, retry_config, **kwargs)


def create_client(
    timeout: float = 30.0,
    follow_redirects: bool = True,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Create a configured httpx async client.

    Args:
        timeout: Request timeout in seconds.
        follow_redirects: Whether to follow redirects.
        headers: Default headers to include in all requests.

    Returns:
        Configured httpx.AsyncClient.
    """
    default_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    if headers:
        default_headers.update(headers)

    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=follow_redirects,
        headers=default_headers,
    )
