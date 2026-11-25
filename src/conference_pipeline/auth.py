"""Authentication handling for NeurIPS website."""

import re
from typing import TYPE_CHECKING

import httpx
from bs4 import BeautifulSoup

from conference_pipeline.config import PipelineSettings
from utils.http import create_client, get_with_retry, post_with_retry
from utils.logging import get_logger

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext, Page

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class NeuripsSession:
    """Authenticated HTTP session for NeurIPS website."""

    def __init__(self, client: httpx.AsyncClient, settings: PipelineSettings):
        """Initialize session with an authenticated client.

        Args:
            client: httpx client with authentication cookies.
            settings: Pipeline settings.
        """
        self.client = client
        self.settings = settings

    @classmethod
    async def login(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create an authenticated session by logging into NeurIPS.

        Args:
            settings: Pipeline settings with credentials.

        Returns:
            Authenticated NeuripsSession.

        Raises:
            AuthenticationError: If login fails.
        """
        if not settings.neurips_email or not settings.neurips_password:
            raise AuthenticationError(
                "NeurIPS credentials not configured. "
                "Set NEURIPS_EMAIL and NEURIPS_PASSWORD environment variables."
            )

        client = create_client(timeout=60.0)

        try:
            # Step 1: Get login page to obtain CSRF token
            # Use /accounts/login with redirect to virtual site for proper cookie scope
            login_url = f"{settings.neurips_base_url}/accounts/login"
            redirect_target = f"/virtual/{settings.year}"
            full_login_url = f"{login_url}?nextp={redirect_target}"
            logger.info(f"Fetching login page: {full_login_url}")

            response = await get_with_retry(client, full_login_url)
            soup = BeautifulSoup(response.text, "lxml")

            # Look for CSRF token in various forms
            csrf_token = None

            # Try to find csrf_token input field
            csrf_input = soup.find("input", {"name": "csrf_token"})
            if csrf_input and csrf_input.get("value"):
                csrf_token = csrf_input["value"]

            # Try meta tag
            if not csrf_token:
                csrf_meta = soup.find("meta", {"name": "csrf-token"})
                if csrf_meta and csrf_meta.get("content"):
                    csrf_token = csrf_meta["content"]

            # Try finding it in JavaScript
            if not csrf_token:
                scripts = soup.find_all("script")
                for script in scripts:
                    if script.string:
                        match = re.search(r'csrf[_-]?token["\']?\s*[:=]\s*["\']([^"\']+)["\']', script.string, re.I)
                        if match:
                            csrf_token = match.group(1)
                            break

            logger.debug(f"Found CSRF token: {csrf_token is not None}")

            # Step 2: Submit login form
            login_data = {
                "email": settings.neurips_email,
                "password": settings.neurips_password,
            }
            if csrf_token:
                login_data["csrf_token"] = csrf_token

            logger.info("Submitting login credentials...")
            login_response = await post_with_retry(
                client,
                full_login_url,
                data=login_data,
                headers={"Referer": full_login_url},
            )

            # Step 3: Verify login success
            # Check for common indicators of successful login
            if "logout" in login_response.text.lower() or "sign out" in login_response.text.lower():
                logger.info("Login successful")
                return cls(client, settings)

            # Check if we were redirected to a dashboard or home page
            if login_response.url and "login" not in str(login_response.url):
                logger.info(f"Login successful, redirected to: {login_response.url}")
                return cls(client, settings)

            # Check for error messages
            error_soup = BeautifulSoup(login_response.text, "lxml")
            error_div = error_soup.find(class_=re.compile(r"error|alert|danger", re.I))
            if error_div:
                error_text = error_div.get_text(strip=True)
                raise AuthenticationError(f"Login failed: {error_text}")

            # If we can't determine status, assume failure
            raise AuthenticationError(
                "Login status unclear. Please verify credentials and try again."
            )

        except httpx.HTTPError as e:
            await client.aclose()
            raise AuthenticationError(f"HTTP error during login: {e}") from e

    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Make authenticated GET request to NeurIPS.

        Args:
            path: URL path (will be joined with base URL).
            **kwargs: Additional arguments for httpx.

        Returns:
            HTTP response.
        """
        url = f"{self.settings.neurips_base_url}{path}"
        return await get_with_retry(self.client, url, **kwargs)

    async def get_full_url(self, url: str, **kwargs) -> httpx.Response:
        """Make authenticated GET request to a full URL.

        Args:
            url: Complete URL.
            **kwargs: Additional arguments for httpx.

        Returns:
            HTTP response.
        """
        return await get_with_retry(self.client, url, **kwargs)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    @classmethod
    def create_anonymous(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create an anonymous (unauthenticated) session.

        Useful for scraping public pages that don't require login.

        Args:
            settings: Pipeline settings.

        Returns:
            Anonymous NeuripsSession.
        """
        client = create_client(timeout=60.0)
        return cls(client, settings)

    @classmethod
    async def create(cls, settings: PipelineSettings) -> "NeuripsSession":
        """Create a session, logging in if credentials are available.

        Args:
            settings: Pipeline settings.

        Returns:
            NeuripsSession (authenticated if credentials available).
        """
        if settings.neurips_email and settings.neurips_password:
            try:
                return await cls.login(settings)
            except AuthenticationError as e:
                logger.warning(f"Login failed, using anonymous session: {e}")

        return cls.create_anonymous(settings)


class NeuripsBrowserSession:
    """Browser-based session for JavaScript-heavy pages."""

    def __init__(
        self,
        settings: PipelineSettings,
        context: "BrowserContext | None" = None,
    ):
        """Initialize browser session.

        Args:
            settings: Pipeline settings.
            context: Playwright browser context (optional, can initialize later).
        """
        self.settings = settings
        self.context = context
        self._page: "Page | None" = None
        self._playwright = None
        self._browser = None
        self._initialized = False
        self._last_url: str | None = None

    async def initialize(self) -> None:
        """Initialize browser context if not already done."""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for browser-based authentication. "
                "Install it with: pip install playwright && playwright install"
            )

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.settings.headless)
        self.context = await self._browser.new_context()

        # Load cookies from cookies.json if it exists
        from pathlib import Path
        import json
        cookies_file = Path("cookies.json")
        if cookies_file.exists():
            logger.info("Loading cookies from cookies.json")
            try:
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                await self.context.add_cookies(cookies)
                logger.info(f"Loaded {len(cookies)} cookies")
            except Exception as e:
                logger.warning(f"Failed to load cookies: {e}")

        self._initialized = True
        logger.info("Browser session initialized")

    async def login(self) -> bool:
        """Attempt to login with stored credentials or verify cookie-based login.

        Returns:
            True if login successful, False otherwise.
        """
        if not self._initialized:
            await self.initialize()

        # Check if we're already logged in via cookies
        from pathlib import Path
        cookies_file = Path("cookies.json")
        if cookies_file.exists():
            logger.info("Cookies loaded - verifying authentication")
            # Try to access a protected page to verify cookies work
            try:
                test_page = await self.context.new_page()
                await test_page.goto(f"{self.settings.virtual_base_url}/calendar", timeout=self.settings.browser_timeout)
                await test_page.wait_for_load_state("domcontentloaded")
                content = await test_page.content()
                await test_page.close()

                # Check if we see login indicators (bad) or logout indicators (good)
                if "logout" in content.lower() or "sign out" in content.lower():
                    logger.info("Cookie-based authentication successful")
                    return True
                else:
                    logger.warning("Cookies present but authentication failed - will try form login")
            except Exception as e:
                logger.warning(f"Cookie verification failed: {e}")

        if not self.settings.neurips_email or not self.settings.neurips_password:
            logger.warning("No credentials configured for login")
            return False

        page = await self.context.new_page()

        try:
            # Navigate to login page with redirect to virtual site
            redirect_target = f"/virtual/{self.settings.year}"
            login_url = f"{self.settings.neurips_base_url}/accounts/login?nextp={redirect_target}"
            logger.info(f"Navigating to login page: {login_url}")
            await page.goto(login_url, timeout=self.settings.browser_timeout)

            # Wait for page to load
            await page.wait_for_load_state("domcontentloaded")

            # Add human-like delay
            import asyncio
            await asyncio.sleep(1)

            # Check if login form exists
            email_input = await page.query_selector('input[name="email"], input[type="email"], input#id_username')
            password_input = await page.query_selector('input[name="password"], input[type="password"], input#id_password')

            if not email_input or not password_input:
                logger.error("Could not find login form inputs")
                # Save screenshot for debugging
                await page.screenshot(path="login_page_error.png")
                logger.info("Saved screenshot to login_page_error.png")
                await page.close()
                return False

            # Fill in credentials with delays (appear more human)
            logger.info("Filling in credentials...")
            await email_input.click()
            await asyncio.sleep(0.5)
            await email_input.fill(self.settings.neurips_email)
            await asyncio.sleep(0.5)
            await password_input.click()
            await asyncio.sleep(0.5)
            await password_input.fill(self.settings.neurips_password)
            await asyncio.sleep(0.5)

            # Find and click submit button
            submit_button = await page.query_selector('button[type="submit"], input[type="submit"], button:has-text("Sign in"), button:has-text("Login"), button:has-text("Log In")')
            if not submit_button:
                logger.error("Could not find submit button")
                await page.screenshot(path="login_page_error.png")
                logger.info("Saved screenshot to login_page_error.png")
                await page.close()
                return False

            logger.info("Submitting login form...")

            # Wait for navigation after clicking submit
            try:
                async with page.expect_navigation(timeout=self.settings.browser_timeout, wait_until="networkidle"):
                    await submit_button.click()
            except Exception as nav_error:
                logger.warning(f"Navigation timeout or error: {nav_error}")
                # Continue anyway - might have succeeded but timed out

            # Take screenshot after login attempt
            await page.screenshot(path="after_login.png")
            logger.info("Saved screenshot to after_login.png")

            # Verify login
            content = await page.content()
            current_url = page.url
            logger.info(f"Current URL after login: {current_url}")

            # Save HTML for debugging
            with open("after_login.html", "w") as f:
                f.write(content)
            logger.info("Saved HTML to after_login.html")

            if "logout" in content.lower() or "sign out" in content.lower():
                logger.info("Browser login successful")
                self._page = page
                return True

            # Check for errors
            error_element = await page.query_selector(".error, .alert-danger, .alert-error, .errorlist")
            if error_element:
                error_text = await error_element.text_content()
                logger.warning(f"Login failed with error: {error_text}")
            else:
                logger.warning("Login failed - no logout link found and no error message")

            await page.close()
            return False

        except Exception as e:
            logger.warning(f"Browser login failed with exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            try:
                await page.screenshot(path="login_exception.png")
                logger.info("Saved screenshot to login_exception.png")
            except:
                pass
            await page.close()
            return False

    @classmethod
    async def create_and_login(cls, settings: PipelineSettings) -> "NeuripsBrowserSession":
        """Create and authenticate browser session.

        Args:
            settings: Pipeline settings with credentials.

        Returns:
            Authenticated browser session.

        Raises:
            AuthenticationError: If login fails.
        """
        session = cls(settings)
        await session.initialize()

        if not await session.login():
            raise AuthenticationError("Browser login failed")

        return session

    async def get_rendered_html(
        self,
        url: str,
        wait_selector: str | None = None,
        wait_time: float | None = None,
        force_reload: bool = False,
    ) -> str:
        """Get fully rendered HTML from a URL.

        Args:
            url: URL to load.
            wait_selector: Optional CSS selector to wait for before returning.
            wait_time: Optional seconds to sleep after load (for dynamic content).
            force_reload: If True, reload once after initial load (helps delayed embeds).

        Returns:
            Rendered HTML content.
        """
        if not self._initialized:
            await self.initialize()

        if self._page is None:
            self._page = await self.context.new_page()

        try:
            from playwright.async_api import TimeoutError as PlaywrightTimeout  # type: ignore
        except Exception:  # pragma: no cover
            PlaywrightTimeout = Exception  # fallback to catch-all if import fails

        try:
            await self._page.goto(url, timeout=self.settings.browser_timeout)
            await self._page.wait_for_load_state("networkidle", timeout=self.settings.browser_timeout)
            self._last_url = self._page.url
        except PlaywrightTimeout as e:
            logger.debug(f"Timeout loading {url}: {e}")
            self._last_url = self._page.url
        except Exception as e:  # pragma: no cover - keep going to return whatever we have
            logger.debug(f"Error loading {url}: {e}")
            self._last_url = self._page.url

        if wait_selector:
            try:
                await self._page.wait_for_selector(wait_selector, timeout=self.settings.browser_timeout)
            except PlaywrightTimeout as e:
                logger.debug(f"Timeout waiting for selector {wait_selector} on {url}: {e}")
            except Exception as e:  # pragma: no cover
                logger.debug(f"Error waiting for selector {wait_selector} on {url}: {e}")

        if force_reload:
            try:
                await self._page.reload(wait_until="networkidle", timeout=self.settings.browser_timeout)
                self._last_url = self._page.url
            except PlaywrightTimeout as e:
                logger.debug(f"Timeout reloading {url}: {e}")
                self._last_url = self._page.url
            except Exception as e:  # pragma: no cover
                logger.debug(f"Error reloading {url}: {e}")
                self._last_url = self._page.url

        if wait_time and wait_time > 0:
            import asyncio
            await asyncio.sleep(wait_time)

        return await self._page.content()

    @property
    def last_url(self) -> str | None:
        """Last successfully (or partially) navigated page URL."""
        return self._last_url

    async def close(self) -> None:
        """Close browser context and cleanup."""
        if self._page:
            await self._page.close()
        if self.context:
            await self.context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._initialized = False
