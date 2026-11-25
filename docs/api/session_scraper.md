# session_scraper.py - Session Discovery

Discover NeurIPS sessions by type and export to JSON for manual selection.

## Location

`src/conference_pipeline/session_scraper.py`

## Classes

### SessionScraper

```python
class SessionScraper:
    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize session scraper."""

    async def discover_sessions(
        self,
        session_types: list[SessionType] | None = None,
    ) -> list[Session]:
        """Discover sessions for the target types."""

    async def save_sessions_json(
        self,
        sessions: list[Session],
        output_path: Path,
    ) -> None:
        """Write sessions JSON for manual filtering."""

    async def load_chosen_sessions(self, input_path: Path) -> list[Session]:
        """Load sessions with chosen='Y' from JSON."""
```

### Convenience Function

```python
async def discover_and_save_sessions(
    settings: PipelineSettings,
    output_path: Path,
    session_types: list[SessionType] | None = None,
) -> list[Session]:
    """Discover sessions and write JSON."""
```

## Output JSON Format

```json
{
  "discovered_at": "2025-01-01T00:00:00",
  "total_sessions": 123,
  "session_types": ["tutorial", "workshop"],
  "sessions": [
    {
      "id": "12345",
      "title": "Example Session",
      "session_type": "tutorial",
      "url": "https://neurips.cc/virtual/2025/12345",
      "date": "Dec 9",
      "time": "10:00 AM",
      "location": "Room A",
      "abstract": "Session description",
      "presentation_count": 12,
      "chosen": null
    }
  ]
}
```

## Usage Example

```python
from conference_pipeline.session_scraper import SessionScraper
from conference_pipeline.models import TARGET_SESSION_TYPES
from pathlib import Path

scraper = SessionScraper(session, settings, browser_session)
sessions = await scraper.discover_sessions(TARGET_SESSION_TYPES)
await scraper.save_sessions_json(sessions, Path("sessions.json"))
```

## Source References

- `src/conference_pipeline/session_scraper.py`
- `src/conference_pipeline/models.py`
