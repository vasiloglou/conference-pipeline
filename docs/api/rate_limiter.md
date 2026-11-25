# rate_limiter.py - Rate Limiting + Backoff

Async rate limiting utilities used by transcription and OCR workflows.

## Location

`src/conference_pipeline/rate_limiter.py`

## Classes

### RateLimiter

Token bucket rate limiter with concurrency control.

```python
@dataclass
class RateLimiter:
    max_rpm: int = 50
    max_concurrent: int = 5

    async def acquire(self) -> None:
        """Block until rate and concurrency limits allow a request."""

    def release(self) -> None:
        """Release a concurrent slot."""

    async def __aenter__(self):
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...
```

## Functions

### retry_with_backoff

```python
async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    rate_limit_codes: tuple[int, ...] = (429,),
) -> T:
    """Retry async function with exponential backoff on rate-limit errors."""
```

### Whisper Rate Limiter Helpers

```python
def get_whisper_rate_limiter(max_rpm: int = 50, max_concurrent: int = 5) -> RateLimiter:
    """Get or create the global Whisper rate limiter."""


def reset_whisper_rate_limiter() -> None:
    """Reset the global Whisper rate limiter."""
```

## Usage Example

```python
from conference_pipeline.rate_limiter import RateLimiter, retry_with_backoff

rate_limiter = RateLimiter(max_rpm=30, max_concurrent=3)

async def call_api():
    async with rate_limiter:
        return await retry_with_backoff(lambda: client.request(...))
```

## Source References

- `src/conference_pipeline/rate_limiter.py`
