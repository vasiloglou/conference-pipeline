"""Scrape slides by navigating through SlidesLive player."""

import asyncio
import json
from pathlib import Path

from playwright.async_api import Page, TimeoutError as PlaywrightTimeout

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import SlideImage
from utils.logging import get_logger

logger = get_logger(__name__)


class SlidesScraperError(Exception):
    """Raised when slide scraping fails."""
    pass


class SlidesScraper:
    """Scrapes slides by navigating through SlidesLive player."""

    def __init__(self, settings: PipelineSettings, browser_session):
        """Initialize scraper.

        Args:
            settings: Pipeline settings.
            browser_session: NeuripsBrowserSession instance.
        """
        self.settings = settings
        self.browser_session = browser_session

    async def scrape_slides_from_player(
        self,
        slideslive_id: str,
        out_dir: Path,
        presentation_id: str,
        neurips_url: str | None = None,
        max_slides: int = 500,
        crop_bottom_pixels: int = 80,
    ) -> list[SlideImage]:
        """Scrape slides by navigating through the SlidesLive player.

        Uses browser automation to click through slides and screenshot each one.

        Args:
            slideslive_id: SlidesLive presentation ID.
            out_dir: Output directory for slide images.
            presentation_id: Presentation ID for metadata.
            neurips_url: Optional NeurIPS presentation URL (preferred, already authenticated).
            max_slides: Maximum number of slides to scrape (safety limit).
            crop_bottom_pixels: Pixels to crop from bottom (player controls).

        Returns:
            List of SlideImage objects.

        Raises:
            SlidesScraperError: If scraping fails.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        slide_images = []

        # Prefer NeurIPS URL (already authenticated) over direct SlidesLive
        if neurips_url:
            url = neurips_url
            logger.info(f"Scraping slides from NeurIPS page: {url}")
        else:
            url = f"https://slideslive.com/{slideslive_id}"
            logger.info(f"Scraping slides from SlidesLive: {url}")

        try:
            # Create a new page
            page = await self.browser_session.context.new_page()

            # Navigate to presentation
            logger.debug(f"Loading {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await asyncio.sleep(3)  # Wait for player to load

            # Find the slide panel/iframe
            # SlidesLive uses an iframe for the slide panel
            slide_frame = None
            try:
                # Try to find iframe with slides
                frames = page.frames
                for frame in frames:
                    frame_url = frame.url
                    if 'embed' in frame_url or 'player' in frame_url:
                        slide_frame = frame
                        logger.debug(f"Found potential slide frame: {frame_url}")
                        break
            except Exception as e:
                logger.warning(f"Could not find slide frame: {e}")

            # If we found a frame, use it; otherwise use main page
            target = slide_frame if slide_frame else page

            # Wait for slide panel to be visible
            await asyncio.sleep(2)

            # Find the slide panel selector
            # Common SlidesLive selectors for the slide display area
            slide_selectors = [
                '[class*="slide-view"]',
                '[class*="slides"]',
                '[id*="slide"]',
                'canvas',
                'img[class*="slide"]',
            ]

            slide_element = None
            for selector in slide_selectors:
                try:
                    slide_element = await target.query_selector(selector)
                    if slide_element:
                        logger.debug(f"Found slide element with selector: {selector}")
                        break
                except:
                    continue

            if not slide_element:
                logger.warning("Could not find slide display element, using full page")
                slide_element = target

            # Find navigation buttons
            next_selectors = [
                'button[aria-label*="next" i]',
                'button[aria-label*="Next" i]',
                'button:has-text("Next")',
                '[class*="next"]',
                '[class*="arrow-right"]',
            ]

            slide_count = 0
            consecutive_failures = 0
            max_consecutive_failures = 3

            logger.info(f"Starting slide capture for {slideslive_id}")

            while slide_count < max_slides:
                slide_count += 1

                try:
                    # Take screenshot of slide area
                    output_path = out_dir / f"slide_{slide_count:03d}.png"

                    # Screenshot the specific element or full page
                    if slide_frame:
                        # Screenshot the frame's body (Frame has no screenshot API)
                        screenshot_bytes = await slide_frame.locator("body").screenshot(type="png")
                    elif slide_element and slide_element != target:
                        # Screenshot specific element
                        screenshot_bytes = await slide_element.screenshot(type="png")
                    else:
                        # Screenshot the viewport
                        screenshot_bytes = await page.screenshot(
                            type="png",
                            full_page=False,
                        )

                    # Save screenshot
                    with open(output_path, 'wb') as f:
                        f.write(screenshot_bytes)

                    # Optionally crop using PIL if needed
                    if crop_bottom_pixels > 0:
                        try:
                            from PIL import Image
                            img = Image.open(output_path)
                            width, height = img.size
                            # Crop bottom
                            cropped = img.crop((0, 0, width, height - crop_bottom_pixels))
                            cropped.save(output_path)
                        except ImportError:
                            logger.warning("PIL not available, skipping crop")
                        except Exception as e:
                            logger.warning(f"Failed to crop slide {slide_count}: {e}")

                    # Create slide image object
                    slide_image = SlideImage(
                        presentation_id=presentation_id,
                        slide_index=slide_count,
                        image_path=output_path,
                        source_type="browser_screenshot",
                    )
                    slide_images.append(slide_image)
                    logger.debug(f"Captured slide {slide_count}")

                    consecutive_failures = 0

                    # Try to click next button
                    next_button = None
                    for selector in next_selectors:
                        try:
                            next_button = await target.query_selector(selector)
                            if next_button:
                                # Check if button is visible and enabled
                                if await next_button.is_visible():
                                    break
                                else:
                                    next_button = None
                        except:
                            continue

                    if not next_button:
                        # Try keyboard navigation
                        logger.debug("No next button found, trying arrow key")
                        await page.keyboard.press("ArrowRight")
                    else:
                        # Click the button
                        try:
                            await next_button.click()
                        except Exception as e:
                            logger.debug(f"Click failed, trying keyboard: {e}")
                            await page.keyboard.press("ArrowRight")

                    # Wait for slide to change
                    await asyncio.sleep(0.5)

                except PlaywrightTimeout:
                    logger.warning(f"Timeout capturing slide {slide_count}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.info(f"Reached end of slides at slide {slide_count}")
                        break

                except Exception as e:
                    logger.warning(f"Error capturing slide {slide_count}: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        logger.info(f"Too many failures, stopping at slide {slide_count}")
                        break

            await page.close()

            logger.info(f"Scraped {len(slide_images)} slides from {slideslive_id}")
            return slide_images

        except Exception as e:
            raise SlidesScraperError(f"Failed to scrape slides from {slideslive_id}: {e}") from e
