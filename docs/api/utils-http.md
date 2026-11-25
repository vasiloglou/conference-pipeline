# utils/http.py - HTTP Utilities

HTTP client utilities with retry logic and rate limiting.

## Location

`src/utils/http.py`

## Classes

### RetryConfig

Configuration for HTTP retry behavior.

```python
class RetryConfig:
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        retry_statuses: set[int] | None = None,
        jitter: float = 0.1,
    ):
        """
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff multiplier
            retry_statuses: HTTP status codes to retry (default: 429, 500-504)
            jitter: Random variation factor for backoff timing
        """
```

## Functions

### request_with_retry

Make an HTTP request with exponential backoff retry.

```python
async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> httpx.Response:
    """
    Make HTTP request with automatic retry on failure.

    Args:
        client: httpx async client
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        retry_config: Retry configuration
        **kwargs: Additional arguments for client.request()

    Returns:
        HTTP response

    Raises:
        httpx.HTTPStatusError: If request fails after all retries
    """
```

**Retry Behavior:**
- Retries on status codes: 429, 500, 502, 503, 504
- Retries on timeout exceptions
- Exponential backoff: `backoff_factor ** attempt`
- Random jitter to prevent thundering herd

### get_with_retry

Convenience wrapper for GET requests.

```python
async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> httpx.Response:
    """GET request with retry."""
```

### post_with_retry

Convenience wrapper for POST requests.

```python
async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    retry_config: RetryConfig | None = None,
    **kwargs,
) -> httpx.Response:
    """POST request with retry."""
```

### create_client

Create a configured httpx async client.

```python
def create_client(
    timeout: float = 30.0,
    follow_redirects: bool = True,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """
    Create configured HTTP client.

    Args:
        timeout: Request timeout in seconds
        follow_redirects: Whether to follow redirects
        headers: Additional default headers

    Returns:
        Configured httpx.AsyncClient
    """
```

**Default Headers:**
```python
{
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,...",
    "Accept-Language": "en-US,en;q=0.5",
}
```

## Usage Examples

### Basic Request with Retry

```python
from utils.http import create_client, get_with_retry

client = create_client()

try:
    response = await get_with_retry(client, "https://api.example.com/data")
    print(response.json())
finally:
    await client.aclose()
```

### Custom Retry Configuration

```python
from utils.http import RetryConfig, request_with_retry

config = RetryConfig(
    max_retries=5,
    backoff_factor=3.0,
    retry_statuses={429, 500, 502, 503, 504, 520},
    jitter=0.2,
)

response = await request_with_retry(
    client,
    "GET",
    "https://api.example.com/data",
    retry_config=config,
)
```

### POST with Retry

```python
from utils.http import post_with_retry

response = await post_with_retry(
    client,
    "https://api.example.com/submit",
    json={"key": "value"},
    headers={"Authorization": "Bearer token"},
)
```

### Using Context Manager

```python
async with httpx.AsyncClient() as client:
    response = await get_with_retry(client, url)
```

## Backoff Timing

For `max_retries=3` and `backoff_factor=2.0`:

| Attempt | Wait Time | With 10% Jitter |
|---------|-----------|-----------------|
| 1 | 1.0s | 0.9-1.1s |
| 2 | 2.0s | 1.8-2.2s |
| 3 | 4.0s | 3.6-4.4s |

## Error Handling

```python
import httpx
from utils.http import get_with_retry

try:
    response = await get_with_retry(client, url)
except httpx.TimeoutException:
    print("Request timed out after all retries")
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
```

## Source References

- `src/utils/http.py`
