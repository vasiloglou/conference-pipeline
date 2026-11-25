# schedule_scraper.py - Schedule Scraper

Scrape NeurIPS schedule pages to discover presentations.

## Location

`src/conference_pipeline/schedule_scraper.py`

## Classes

### ScheduleScraper

```python
class ScheduleScraper:
    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize schedule scraper."""

    async def scrape_presentations_from_sessions(
        self,
        sessions: list[Session],
        storage=None,
    ) -> list[Presentation]:
        """Scrape presentations for chosen sessions."""

    async def scrape_all_presentations(self) -> list[Presentation]:
        """Legacy mode: scrape calendar pages directly."""
```

## Behavior Notes

- Uses `browser_session.get_rendered_html` for JS-heavy pages when available.
- Optional `storage` enables session checkpoints (`mark_session_started`/`mark_session_completed`).
- Parses session pages using multiple strategies: cards, tables, links.

## Usage Example

```python
from conference_pipeline.schedule_scraper import ScheduleScraper

scraper = ScheduleScraper(session, settings, browser_session)
presentations = await scraper.scrape_all_presentations()
```

## Source References

- `src/conference_pipeline/schedule_scraper.py`
- `src/conference_pipeline/models.py`
