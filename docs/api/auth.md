# auth.py - Authentication

Handles authentication with the NeurIPS website.

## Location

`src/conference_pipeline/auth.py`

## Classes

### NeuripsSession

HTTP-based session for NeurIPS pages.

```python
class NeuripsSession:
    def __init__(self, client: httpx.AsyncClient, settings: PipelineSettings):
        """Initialize with an httpx client and settings."""

    @classmethod
    async def login(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create authenticated session by logging into NeurIPS."""

    @classmethod
    def create_anonymous(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create an anonymous session (no login)."""

    @classmethod
    async def create(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create session, logging in if credentials are available."""

    async def get(self, path: str, **kwargs) -> httpx.Response:
        """GET request to a path under neurips_base_url."""

    async def get_full_url(self, url: str, **kwargs) -> httpx.Response:
        """GET request to a full URL."""

    async def close(self) -> None:
        """Close the HTTP client."""
```

#### Login Flow

1. Fetch login page to get CSRF token
2. Submit login form with email/password
3. Verify login by checking for "logout" link or redirect
4. Return session with authenticated cookies

### NeuripsBrowserSession

Playwright-based browser session for JS-heavy pages.

```python
class NeuripsBrowserSession:
    def __init__(
        self,
        settings: PipelineSettings,
        context: BrowserContext | None = None,
    ):
        """Initialize with settings and optional browser context."""

    async def initialize(self) -> None:
        """Launch browser + context if not already initialized."""

    async def login(self) -> bool:
        """Attempt cookie-based or credential-based login."""

    @classmethod
    async def create_and_login(cls, settings: PipelineSettings) -> "NeuripsBrowserSession":
        """Create and authenticate session (raises on failure)."""

    async def get_rendered_html(
        self,
        url: str,
        wait_selector: str | None = None,
        wait_time: float | None = None,
        force_reload: bool = False,
    ) -> str:
        """Return rendered HTML after JS execution."""

    @property
    def last_url(self) -> str | None:
        """Last navigated URL (after redirects)."""

    async def close(self) -> None:
        """Close browser context and cleanup."""
```

#### Browser Login Behavior

- Loads `cookies.json` if present and validates access before form login.
- Falls back to interactive login using `NEURIPS_EMAIL`/`NEURIPS_PASSWORD`.
- Captures diagnostic screenshots/HTML on failures.

### AuthenticationError

Exception raised when authentication fails.

```python
class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass
```

## Usage Examples

### HTTP Session Login (auto)

```python
from conference_pipeline.config import load_settings
from conference_pipeline.auth import NeuripsSession

settings = load_settings()
session = await NeuripsSession.create(settings)

response = await session.get("/virtual/2025")
print(response.text)

await session.close()
```

### Browser Session Login

```python
from conference_pipeline.config import load_settings
from conference_pipeline.auth import NeuripsBrowserSession

settings = load_settings()
session = await NeuripsBrowserSession.create_and_login(settings)

html = await session.get_rendered_html(
    "https://neurips.cc/virtual/2025/paper/12345",
    wait_selector=".video-player",
    wait_time=2.5,
)

await session.close()
```

### Error Handling

```python
from conference_pipeline.auth import NeuripsSession, AuthenticationError

try:
    session = await NeuripsSession.login(settings)
except AuthenticationError as e:
    print(f"Login failed: {e}")
```

## Authentication Notes

### Credentials

Credentials must be set in environment variables or `.env` file:

```env
NEURIPS_EMAIL=your-email@example.com
NEURIPS_PASSWORD=your-password
```

### Login URL

The login URL pattern used is:

```
https://neurips.cc/accounts/login?nextp=/virtual/{year}
```

### CSRF Tokens

The HTTP login attempts to find CSRF tokens in:
1. `<input name="csrf_token">`
2. `<meta name="csrf-token">`
3. JavaScript assignments

## Source References

- `src/conference_pipeline/auth.py`
- `src/conference_pipeline/config.py`
