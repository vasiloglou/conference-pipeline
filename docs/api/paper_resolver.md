# paper_resolver.py - Reference Resolution + PDF Download

Resolve reference candidates to papers and download PDFs.

## Location

`src/conference_pipeline/paper_resolver.py`

## Classes

### PaperResolver

```python
class PaperResolver:
    def __init__(self, http_client: httpx.AsyncClient):
        """Initialize resolver."""

    async def resolve_reference(self, ref: ReferenceCandidate) -> Paper | None:
        """Resolve a single reference candidate."""

    async def resolve_many(
        self,
        refs: list[ReferenceCandidate],
    ) -> list[tuple[Paper, str]]:
        """Resolve and deduplicate references (returns paper + ref id)."""
```

Resolution order:
1. arXiv ID
2. DOI
3. URL (arXiv/DOI patterns)
4. Semantic Scholar bibliographic search

### PaperDownloader

```python
class PaperDownloader:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize downloader."""

    async def download_pdf(self, paper: Paper, dest_dir: Path) -> Paper:
        """Download PDF if likely open-access."""

    async def download_many(
        self,
        papers: list[Paper],
        dest_dir: Path,
        concurrency: int = 3,
    ) -> list[Paper]:
        """Download multiple PDFs with rate limiting."""
```

## Usage Example

```python
from conference_pipeline.paper_resolver import PaperResolver, PaperDownloader
from pathlib import Path

resolver = PaperResolver(http_client)
resolved = await resolver.resolve_many(candidates)

papers = [paper for paper, _ in resolved]

pdfs = await PaperDownloader(http_client, settings).download_many(
    papers,
    Path("papers"),
)
```

## Source References

- `src/conference_pipeline/paper_resolver.py`
- `src/conference_pipeline/models.py`
