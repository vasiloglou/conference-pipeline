# presentation_parser.py - Presentation Page Parser

Parse individual NeurIPS presentation pages to discover SlidesLive embeds.

## Location

`src/conference_pipeline/presentation_parser.py`

## Classes

### PresentationPageParser

```python
class PresentationPageParser:
    def __init__(
        self,
        session: NeuripsSession,
        settings: PipelineSettings,
        browser_session: NeuripsBrowserSession | None = None,
    ):
        """Initialize parser."""

    async def enrich_presentation(self, pres: Presentation) -> Presentation:
        """Populate SlidesLive fields for a presentation."""
```

## Behavior Notes

- Uses browser rendering when available to detect delayed embeds.
- May follow nested presentation links (oral pages, location variants).
- Optional LLM-assisted video detection via `settings.use_llm_video_detection`.

## Usage Example

```python
from conference_pipeline.presentation_parser import PresentationPageParser

parser = PresentationPageParser(session, settings, browser_session)
pres = await parser.enrich_presentation(pres)
print(pres.slideslive_id, pres.slideslive_url)
```

## Source References

- `src/conference_pipeline/presentation_parser.py`
- `src/conference_pipeline/models.py`
