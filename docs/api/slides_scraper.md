# slides_scraper.py - SlidesLive Player Scraper

Scrape slide images by automating the SlidesLive player in a browser.

## Location

`src/conference_pipeline/slides_scraper.py`

## Classes

### SlidesScraper

```python
class SlidesScraper:
    def __init__(self, settings: PipelineSettings, browser_session):
        """Initialize with PipelineSettings and NeuripsBrowserSession."""

    async def scrape_slides_from_player(
        self,
        slideslive_id: str,
        out_dir: Path,
        presentation_id: str,
        neurips_url: str | None = None,
        max_slides: int = 500,
        crop_bottom_pixels: int = 80,
    ) -> list[SlideImage]:
        """Navigate SlidesLive player and capture slide screenshots."""
```

## Behavior Notes

- Requires Playwright and an initialized browser session.
- Uses NeurIPS URL (if provided) to reuse authenticated session.
- Saves PNG images as `slide_###.png` and optionally crops the bottom bar.

## Usage Example

```python
from conference_pipeline.auth import NeuripsBrowserSession
from conference_pipeline.slides_scraper import SlidesScraper
from pathlib import Path

browser = await NeuripsBrowserSession.create_and_login(settings)
scraper = SlidesScraper(settings, browser)

slides = await scraper.scrape_slides_from_player(
    slideslive_id="39025946",
    out_dir=Path("slides/images"),
    presentation_id="paper-123",
    neurips_url="https://neurips.cc/virtual/2025/paper/12345",
)
```

## Source References

- `src/conference_pipeline/slides_scraper.py`
- `src/conference_pipeline/models.py`
