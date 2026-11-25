"""Client for interacting with SlidesLive to fetch metadata and assets."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from bs4 import BeautifulSoup

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import SlidesLiveMetadata
from utils.http import get_with_retry
from utils.logging import get_logger

logger = get_logger(__name__)


# SlidesLive migrated its public slide assets from the retired
# ``s.slideslive.com`` / ``rs.slideslive.com`` hosts to this Bunny CDN host.
# Keep metadata and image URLs together so the recovery lane cannot resolve
# one host while issuing image requests to the other.
SLIDESLIVE_SLIDES_CDN_HOST = "slideslive-slides.b-cdn.net"


@dataclass
class InterceptedUrls:
    """URLs intercepted during page load."""

    video_urls: list[str] = field(default_factory=list)
    manifest_urls: list[str] = field(default_factory=list)
    slide_urls: list[str] = field(default_factory=list)
    pdf_urls: list[str] = field(default_factory=list)
    api_responses: list[dict[str, Any]] = field(default_factory=list)


class SlidesLiveClient:
    """Client for SlidesLive pages and assets."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize SlidesLive client.

        Args:
            http_client: httpx async client.
            settings: Pipeline settings.
        """
        self.client = http_client
        self.settings = settings

    async def fetch_metadata(self, slideslive_id: str) -> SlidesLiveMetadata:
        """Fetch metadata for a SlidesLive presentation.

        Args:
            slideslive_id: SlidesLive presentation ID.

        Returns:
            SlidesLiveMetadata with discovered info.
        """
        metadata = SlidesLiveMetadata(slideslive_id=slideslive_id)

        try:
            # Fetch main page
            url = f"{self.settings.slideslive_base_url}/{slideslive_id}"
            response = await get_with_retry(self.client, url)
            html = response.text
            soup = BeautifulSoup(html, "lxml")

            # Extract title
            title_elem = soup.find("title")
            if title_elem:
                metadata.title = title_elem.get_text(strip=True)

            # Look for video manifest URL in page
            metadata = self._extract_video_urls(metadata, html, soup)

            # Look for slides
            metadata = self._extract_slides_info(metadata, html, soup)

            # Try API endpoint if available
            metadata = await self._fetch_api_metadata(metadata, slideslive_id)

        except Exception as e:
            logger.error(f"Error fetching SlidesLive metadata for {slideslive_id}: {e}")

        return metadata

    def _extract_video_urls(
        self,
        metadata: SlidesLiveMetadata,
        html: str,
        soup: BeautifulSoup,
    ) -> SlidesLiveMetadata:
        """Extract video URLs from page content.

        Args:
            metadata: Metadata to update.
            html: Raw HTML.
            soup: Parsed HTML.

        Returns:
            Updated metadata.
        """
        try:
            # Look for HLS manifest URL (.m3u8)
            m3u8_patterns = [
                r'["\']([^"\']+\.m3u8[^"\']*)["\']',
                r"src[=:]\s*[\"']([^\"']+\.m3u8[^\"']*)",
            ]

            for pattern in m3u8_patterns:
                match = re.search(pattern, html, re.I)
                if match:
                    metadata.video_manifest_url = match.group(1)
                    break

            # Look for direct MP4 URL
            mp4_patterns = [
                r'["\']([^"\']+\.mp4[^"\']*)["\']',
                r"src[=:]\s*[\"']([^\"']+\.mp4[^\"']*)",
            ]

            for pattern in mp4_patterns:
                match = re.search(pattern, html, re.I)
                if match:
                    metadata.video_url = match.group(1)
                    break

            # Look for video source elements
            video_elem = soup.find("video")
            if video_elem:
                source = video_elem.find("source", src=True)
                if source:
                    metadata.video_url = source["src"]

            # Look for JSON config in scripts
            for script in soup.find_all("script"):
                if script.string:
                    # Try to find JSON with video URLs
                    json_match = re.search(r'({[^{}]*"video"[^{}]*})', script.string)
                    if json_match:
                        try:
                            config = json.loads(json_match.group(1))
                            if "video" in config:
                                metadata.video_url = config["video"]
                        except json.JSONDecodeError:
                            pass

            return metadata

        except Exception as e:
            logger.debug(f"Error extracting video URLs: {e}")
            return metadata

    def _extract_slides_info(
        self,
        metadata: SlidesLiveMetadata,
        html: str,
        soup: BeautifulSoup,
    ) -> SlidesLiveMetadata:
        """Extract slide deck information.

        Args:
            metadata: Metadata to update.
            html: Raw HTML.
            soup: Parsed HTML.

        Returns:
            Updated metadata.
        """
        try:
            # Look for PDF download link
            pdf_patterns = [
                r'href[=:]\s*["\']([^"\']+\.pdf)["\']',
                r'download[^>]+href[=:]\s*["\']([^"\']+\.pdf)',
            ]

            for pattern in pdf_patterns:
                match = re.search(pattern, html, re.I)
                if match:
                    url = match.group(1)
                    if not url.startswith("http"):
                        url = f"{self.settings.slideslive_base_url}{url}"
                    metadata.slides_pdf_url = url
                    break

            # Look for download button
            download_link = soup.find("a", class_=re.compile(r"download", re.I))
            if download_link and download_link.get("href"):
                href = download_link["href"]
                if ".pdf" in href.lower():
                    if not href.startswith("http"):
                        href = f"{self.settings.slideslive_base_url}{href}"
                    metadata.slides_pdf_url = href

            # Look for slide images
            slide_images = []

            # Pattern 1: Slide image elements
            slide_elems = soup.find_all(class_=re.compile(r"slide", re.I))
            for elem in slide_elems:
                img = elem.find("img", src=True)
                if img:
                    slide_images.append(img["src"])

            # Pattern 2: Data attributes with slide URLs
            data_match = re.findall(r'"slides?"\s*:\s*\[([^\]]+)\]', html)
            if data_match:
                for match in data_match:
                    urls = re.findall(r'"([^"]+\.(jpg|jpeg|png))"', match, re.I)
                    slide_images.extend([url[0] for url in urls])

            # Pattern 3: Image URLs with slide patterns
            img_patterns = [
                r'["\']([^"\']+/slide[_-]?\d+\.(jpg|jpeg|png))["\']',
                r'["\']([^"\']+/\d+\.(jpg|jpeg|png))["\']',
            ]

            for pattern in img_patterns:
                matches = re.findall(pattern, html, re.I)
                slide_images.extend([m[0] for m in matches])

            # Deduplicate and filter
            seen = set()
            unique_images = []
            for url in slide_images:
                if url not in seen and "thumb" not in url.lower():
                    seen.add(url)
                    unique_images.append(url)

            metadata.slide_image_urls = unique_images

            return metadata

        except Exception as e:
            logger.debug(f"Error extracting slides info: {e}")
            return metadata

    async def _fetch_api_metadata(
        self,
        metadata: SlidesLiveMetadata,
        slideslive_id: str,
    ) -> SlidesLiveMetadata:
        """Try to fetch metadata from SlidesLive API.

        Args:
            metadata: Metadata to update.
            slideslive_id: Presentation ID.

        Returns:
            Updated metadata.
        """
        # Common SlidesLive API patterns
        api_urls = [
            f"{self.settings.slideslive_base_url}/api/v1/presentations/{slideslive_id}",
            f"{self.settings.slideslive_base_url}/api/presentations/{slideslive_id}",
            f"https://api.slideslive.com/v1/presentations/{slideslive_id}",
        ]

        for api_url in api_urls:
            try:
                response = await self.client.get(api_url)
                if response.status_code == 200:
                    data = response.json()
                    metadata = self._parse_api_response(metadata, data)
                    break
            except Exception:
                continue

        return metadata

    def _parse_api_response(
        self,
        metadata: SlidesLiveMetadata,
        data: dict[str, Any],
    ) -> SlidesLiveMetadata:
        """Parse API response and update metadata.

        Args:
            metadata: Metadata to update.
            data: API response data.

        Returns:
            Updated metadata.
        """
        try:
            if "title" in data and not metadata.title:
                metadata.title = data["title"]

            if "video_url" in data and not metadata.video_url:
                metadata.video_url = data["video_url"]

            if "video" in data and isinstance(data["video"], dict):
                if "url" in data["video"] and not metadata.video_url:
                    metadata.video_url = data["video"]["url"]
                if "manifest_url" in data["video"] and not metadata.video_manifest_url:
                    metadata.video_manifest_url = data["video"]["manifest_url"]

            if "slides_pdf_url" in data and not metadata.slides_pdf_url:
                metadata.slides_pdf_url = data["slides_pdf_url"]

            if "slides" in data and isinstance(data["slides"], list):
                if not metadata.slide_image_urls:
                    metadata.slide_image_urls = [
                        s.get("url") or s.get("image_url")
                        for s in data["slides"]
                        if s.get("url") or s.get("image_url")
                    ]

        except Exception as e:
            logger.debug(f"Error parsing API response: {e}")

        return metadata

    async def get_download_urls(self, slideslive_id: str) -> dict[str, Any]:
        """Get all downloadable asset URLs for a presentation.

        Args:
            slideslive_id: SlidesLive presentation ID.

        Returns:
            Dictionary with video_url, slides_pdf_url, slide_images keys.
        """
        metadata = await self.fetch_metadata(slideslive_id)

        return {
            "video_url": metadata.video_url,
            "video_manifest_url": metadata.video_manifest_url,
            "slides_pdf_url": metadata.slides_pdf_url,
            "slide_images": metadata.slide_image_urls,
        }

    def get_yt_dlp_url(self, slideslive_id: str) -> str:
        """Get URL suitable for yt-dlp download.

        Args:
            slideslive_id: SlidesLive presentation ID.

        Returns:
            URL for yt-dlp to process.
        """
        return f"{self.settings.slideslive_base_url}/{slideslive_id}"

    async def fetch_metadata_with_browser(
        self,
        slideslive_id: str,
        wait_time: float = 5.0,
    ) -> SlidesLiveMetadata:
        """Fetch metadata using Playwright browser with network interception.

        This method uses a headless browser to load the SlidesLive page and
        intercepts network requests to capture dynamically loaded URLs.

        Args:
            slideslive_id: SlidesLive presentation ID.
            wait_time: Time to wait for dynamic content to load.

        Returns:
            SlidesLiveMetadata with discovered info.

        Raises:
            ImportError: If playwright is not installed.
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            logger.warning(
                "Playwright not available, falling back to HTTP fetch. "
                "Install with: pip install playwright && playwright install"
            )
            return await self.fetch_metadata(slideslive_id)

        metadata = SlidesLiveMetadata(slideslive_id=slideslive_id)
        intercepted = InterceptedUrls()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.settings.headless)
            context = await browser.new_context()
            page = await context.new_page()

            # Set up request interception
            async def handle_route(route):
                """Intercept and analyze requests."""
                request = route.request
                url = request.url.lower()

                # Capture video URLs
                if any(ext in url for ext in [".mp4", ".webm", ".mov"]):
                    intercepted.video_urls.append(request.url)
                elif ".m3u8" in url:
                    intercepted.manifest_urls.append(request.url)
                # Capture slide/image URLs
                elif any(ext in url for ext in [".jpg", ".jpeg", ".png", ".gif"]):
                    if "slide" in url or "thumb" not in url:
                        intercepted.slide_urls.append(request.url)
                # Capture PDF URLs
                elif ".pdf" in url:
                    intercepted.pdf_urls.append(request.url)

                await route.continue_()

            # Set up response interception for API calls
            async def handle_response(response):
                """Capture API responses."""
                url = response.url.lower()
                if "api" in url and response.status == 200:
                    try:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            data = await response.json()
                            intercepted.api_responses.append(data)
                    except Exception:
                        pass

            # Enable interception
            await page.route("**/*", handle_route)
            page.on("response", handle_response)

            try:
                # Navigate to page
                url = f"{self.settings.slideslive_base_url}/{slideslive_id}"
                await page.goto(url, wait_until="networkidle")

                # Wait for dynamic content
                await asyncio.sleep(wait_time)

                # Extract title from page
                title = await page.title()
                if title:
                    metadata.title = title.replace(" | SlidesLive", "").strip()

                # Get page HTML for additional parsing
                html = await page.content()
                soup = BeautifulSoup(html, "lxml")

                # Parse HTML as before
                metadata = self._extract_video_urls(metadata, html, soup)
                metadata = self._extract_slides_info(metadata, html, soup)

            except Exception as e:
                logger.error(f"Browser fetch error for {slideslive_id}: {e}")

            finally:
                await browser.close()

        # Merge intercepted URLs into metadata
        metadata = self._merge_intercepted_urls(metadata, intercepted)

        return metadata

    def _merge_intercepted_urls(
        self,
        metadata: SlidesLiveMetadata,
        intercepted: InterceptedUrls,
    ) -> SlidesLiveMetadata:
        """Merge intercepted URLs into metadata.

        Args:
            metadata: Existing metadata.
            intercepted: URLs captured via network interception.

        Returns:
            Updated metadata.
        """
        # Prefer intercepted manifest URLs (more reliable for HLS)
        if intercepted.manifest_urls and not metadata.video_manifest_url:
            # Prefer the first quality-specific manifest or master
            for url in intercepted.manifest_urls:
                if "master" in url.lower() or "index" in url.lower():
                    metadata.video_manifest_url = url
                    break
            if not metadata.video_manifest_url:
                metadata.video_manifest_url = intercepted.manifest_urls[0]

        # Use intercepted video URLs
        if intercepted.video_urls and not metadata.video_url:
            # Prefer highest quality (often largest segment or main video)
            metadata.video_url = intercepted.video_urls[0]

        # Use intercepted PDF URLs
        if intercepted.pdf_urls and not metadata.slides_pdf_url:
            metadata.slides_pdf_url = intercepted.pdf_urls[0]

        # Merge slide URLs (deduplicate)
        if intercepted.slide_urls:
            existing = set(metadata.slide_image_urls)
            for url in intercepted.slide_urls:
                if url not in existing:
                    metadata.slide_image_urls.append(url)
                    existing.add(url)

        # Parse API responses for additional data
        for data in intercepted.api_responses:
            metadata = self._parse_api_response(metadata, data)

        logger.info(
            f"Intercepted URLs for {metadata.slideslive_id}: "
            f"{len(intercepted.video_urls)} videos, "
            f"{len(intercepted.manifest_urls)} manifests, "
            f"{len(intercepted.slide_urls)} slides, "
            f"{len(intercepted.pdf_urls)} PDFs"
        )

        return metadata

    async def fetch_metadata_auto(
        self,
        slideslive_id: str,
        use_browser: bool | None = None,
    ) -> SlidesLiveMetadata:
        """Fetch metadata using best available method.

        Args:
            slideslive_id: SlidesLive presentation ID.
            use_browser: Force browser mode (True), HTTP mode (False),
                        or auto-detect (None).

        Returns:
            SlidesLiveMetadata with discovered info.
        """
        if use_browser is None:
            use_browser = self.settings.use_playwright

        if use_browser:
            return await self.fetch_metadata_with_browser(slideslive_id)
        return await self.fetch_metadata(slideslive_id)

    async def fetch_slides_json(self, slideslive_id: str) -> list[dict] | None:
        """Fetch slide metadata from SlidesLive's slides.json API.

        This is the authoritative source for slide data, returning all slides
        with their names and timestamps.

        Args:
            slideslive_id: SlidesLive presentation ID.

        Returns:
            List of slide dictionaries with 'image' (name, extname) and 'time' keys,
            or None if fetch fails.
        """
        url = f"https://{SLIDESLIVE_SLIDES_CDN_HOST}/{slideslive_id}/v2/slides.json"
        try:
            response = await self.client.get(
                url,
                headers={"Referer": "https://slideslive.com/"},
            )
            if response.status_code == 200:
                data = response.json()
                slides = data.get("slides", [])
                logger.info(f"Fetched {len(slides)} slides from slides.json for {slideslive_id}")
                return slides
            else:
                logger.warning(f"Failed to fetch slides.json for {slideslive_id}: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error fetching slides.json for {slideslive_id}: {e}")
            return None

    def build_slide_image_url(
        self,
        slideslive_id: str,
        slide_name: str,
        extension: str = ".png",
        height: int = 1080,
        format: str = "jpeg",
    ) -> str:
        """Build the URL for a slide image from SlidesLive's resizing service.

        Args:
            slideslive_id: SlidesLive presentation ID.
            slide_name: Slide name from slides.json (e.g., 'LaCAzwJKSclnnsMS').
            extension: File extension from slides.json (e.g., '.png').
            height: Desired height (432, 540, 720, 1080, 1440, 2160).
            format: Output format ('jpeg' or 'webp').

        Returns:
            Full URL to the slide image.
        """
        # The Bunny CDN's current public contract uses an original-image path
        # plus a numeric resize class. It preserves the source image extension;
        # ``format`` remains an API-compatibility parameter for callers that
        # also support the legacy SlidesLive image service.
        path = f"{slideslive_id}/slides/original/{slide_name}{extension}"
        return f"https://{SLIDESLIVE_SLIDES_CDN_HOST}/{path}?class={height}"

    async def get_all_slide_urls(
        self,
        slideslive_id: str,
        height: int = 1080,
        format: str = "jpeg",
    ) -> list[str]:
        """Get URLs for all slides in a presentation.

        Fetches slide metadata from slides.json and constructs full URLs.

        Args:
            slideslive_id: SlidesLive presentation ID.
            height: Desired height for images.
            format: Output format ('jpeg' or 'webp').

        Returns:
            List of slide image URLs in presentation order.
        """
        slide_info = await self.get_all_slide_info(slideslive_id, height, format)
        return [info["url"] for info in slide_info]

    async def get_all_slide_info(
        self,
        slideslive_id: str,
        height: int = 1080,
        format: str = "jpeg",
    ) -> list[dict]:
        """Get URLs and timing info for all slides in a presentation.

        Fetches slide metadata from slides.json and constructs full URLs
        with synchronization timestamps.

        Args:
            slideslive_id: SlidesLive presentation ID.
            height: Desired height for images.
            format: Output format ('jpeg' or 'webp').

        Returns:
            List of dicts with 'url' and 'timestamp_ms' keys in presentation order.
        """
        slides_data = await self.fetch_slides_json(slideslive_id)
        if not slides_data:
            return []

        slide_info = []
        for slide in slides_data:
            if slide.get("type") == "image" and "image" in slide:
                name = slide["image"].get("name", "")
                ext = slide["image"].get("extname", ".png")
                if name:
                    url = self.build_slide_image_url(
                        slideslive_id, name, ext, height, format
                    )
                    # 'time' is in milliseconds, indicates when slide appears in video
                    timestamp_ms = slide.get("time")
                    slide_info.append({
                        "url": url,
                        "timestamp_ms": timestamp_ms,
                    })

        logger.info(f"Built {len(slide_info)} slide URLs with timing for {slideslive_id}")
        return slide_info
