"""Paper resolution and PDF download."""

import asyncio
import re
from pathlib import Path
from urllib.parse import quote

import httpx

from conference_pipeline.config import PipelineSettings
from conference_pipeline.models import Paper, ReferenceCandidate
from utils.http import RetryConfig, get_with_retry
from utils.logging import get_logger

logger = get_logger(__name__)


class PaperResolverError(Exception):
    """Raised when paper resolution fails."""

    pass


class PaperResolver:
    """Resolves reference candidates to canonical papers."""

    def __init__(self, http_client: httpx.AsyncClient):
        """Initialize paper resolver.

        Args:
            http_client: httpx async client.
        """
        self.client = http_client
        self.retry_config = RetryConfig(max_retries=2)

    async def resolve_reference(self, ref: ReferenceCandidate) -> Paper | None:
        """Resolve a reference candidate to a Paper.

        Args:
            ref: Reference candidate to resolve.

        Returns:
            Resolved Paper or None if resolution fails.
        """
        paper = None

        # Try resolution in order of reliability
        if ref.arxiv_id:
            paper = await self._resolve_arxiv(ref.arxiv_id)

        if not paper and ref.doi:
            paper = await self._resolve_doi(ref.doi)

        if not paper and ref.detected_url:
            paper = await self._resolve_url(ref.detected_url)

        if not paper and ref.title and ref.authors:
            paper = await self._resolve_bibliographic(ref.title, ref.authors)

        return paper

    async def resolve_many(
        self,
        refs: list[ReferenceCandidate],
    ) -> list[tuple[Paper, str]]:
        """Resolve multiple reference candidates.

        Args:
            refs: List of reference candidates.

        Returns:
            List of tuples (paper, reference_candidate_id) for tracking.
        """
        # Deduplicate refs first
        unique_refs = self._deduplicate_refs(refs)

        results: list[tuple[Paper, str]] = []
        seen_paper_ids: set[str] = set()

        for ref in unique_refs:
            try:
                paper = await self.resolve_reference(ref)
                if paper and paper.id not in seen_paper_ids:
                    results.append((paper, ref.id))
                    seen_paper_ids.add(paper.id)

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Failed to resolve reference {ref.id}: {e}")

        return results

    async def _resolve_arxiv(self, arxiv_id: str) -> Paper | None:
        """Resolve paper using arXiv API.

        Args:
            arxiv_id: arXiv identifier.

        Returns:
            Paper or None.
        """
        try:
            # Clean up arXiv ID
            arxiv_id = arxiv_id.replace("arxiv:", "").strip()

            url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = await get_with_retry(self.client, url, self.retry_config)

            # Parse Atom XML response
            from xml.etree import ElementTree as ET

            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return None

            # Extract metadata
            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None else "Unknown"

            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None:
                    authors.append(name_elem.text)

            # Get published year
            published = entry.find("atom:published", ns)
            year = None
            if published is not None and published.text:
                year_match = re.search(r"(\d{4})", published.text)
                if year_match:
                    year = int(year_match.group(1))

            # Get PDF URL
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            return Paper(
                id=f"arxiv:{arxiv_id}",
                title=title,
                authors=authors,
                year=year,
                arxiv_id=arxiv_id,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.warning(f"arXiv resolution failed for {arxiv_id}: {e}")
            return None

    async def _resolve_doi(self, doi: str) -> Paper | None:
        """Resolve paper using Crossref API.

        Args:
            doi: DOI identifier.

        Returns:
            Paper or None.
        """
        try:
            # Clean up DOI
            doi = doi.strip()
            if doi.startswith("https://doi.org/"):
                doi = doi[16:]

            url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
            response = await get_with_retry(
                self.client,
                url,
                self.retry_config,
                headers={"Accept": "application/json"},
            )

            data = response.json()
            message = data.get("message", {})

            # Extract metadata
            title = message.get("title", ["Unknown"])[0]

            authors = []
            for author in message.get("author", []):
                name_parts = []
                if author.get("given"):
                    name_parts.append(author["given"])
                if author.get("family"):
                    name_parts.append(author["family"])
                if name_parts:
                    authors.append(" ".join(name_parts))

            # Get year
            year = None
            published = message.get("published-print") or message.get("published-online")
            if published and published.get("date-parts"):
                year = published["date-parts"][0][0]

            # Get venue
            venue = None
            container = message.get("container-title", [])
            if container:
                venue = container[0]

            # Get PDF URL (if available via Unpaywall or direct link)
            pdf_url = None
            links = message.get("link", [])
            for link in links:
                if link.get("content-type") == "application/pdf":
                    pdf_url = link.get("URL")
                    break

            return Paper(
                id=f"doi:{doi}",
                title=title,
                authors=authors,
                venue=venue,
                year=year,
                doi=doi,
                url=f"https://doi.org/{doi}",
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.warning(f"DOI resolution failed for {doi}: {e}")
            return None

    async def _resolve_url(self, url: str) -> Paper | None:
        """Try to resolve paper from URL.

        Args:
            url: URL that might be a paper link.

        Returns:
            Paper or None.
        """
        # Check for arXiv URL
        arxiv_match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|\w+-\w+/\d+)", url)
        if arxiv_match:
            return await self._resolve_arxiv(arxiv_match.group(1))

        # Check for DOI URL
        doi_match = re.search(r"doi\.org/(10\.\d+/[^\s]+)", url)
        if doi_match:
            return await self._resolve_doi(doi_match.group(1))

        # Other URL patterns could be added here
        return None

    async def _resolve_bibliographic(
        self,
        title: str,
        authors: list[str],
    ) -> Paper | None:
        """Resolve paper using bibliographic search.

        Args:
            title: Paper title.
            authors: Author names.

        Returns:
            Paper or None.
        """
        try:
            # Try Semantic Scholar
            query = quote(title)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=1"

            response = await get_with_retry(self.client, url, self.retry_config)
            data = response.json()

            papers = data.get("data", [])
            if not papers:
                return None

            paper_data = papers[0]
            paper_id = paper_data.get("paperId")

            # Get detailed info
            detail_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields=title,authors,year,venue,externalIds,openAccessPdf"
            detail_response = await get_with_retry(self.client, detail_url, self.retry_config)
            detail = detail_response.json()

            # Extract metadata
            resolved_title = detail.get("title", title)
            resolved_authors = [a.get("name", "") for a in detail.get("authors", [])]
            year = detail.get("year")
            venue = detail.get("venue")

            # Get IDs
            external_ids = detail.get("externalIds", {})
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")

            # Get PDF URL
            pdf_info = detail.get("openAccessPdf", {})
            pdf_url = pdf_info.get("url") if pdf_info else None

            # Create canonical ID
            if arxiv_id:
                canonical_id = f"arxiv:{arxiv_id}"
            elif doi:
                canonical_id = f"doi:{doi}"
            else:
                canonical_id = f"s2:{paper_id}"

            return Paper(
                id=canonical_id,
                title=resolved_title,
                authors=resolved_authors,
                venue=venue,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                url=f"https://www.semanticscholar.org/paper/{paper_id}",
                pdf_url=pdf_url,
            )

        except Exception as e:
            logger.warning(f"Bibliographic resolution failed for '{title}': {e}")
            return None

    def _deduplicate_refs(
        self,
        refs: list[ReferenceCandidate],
    ) -> list[ReferenceCandidate]:
        """Deduplicate reference candidates."""
        seen = set()
        unique = []

        for ref in refs:
            key = None
            if ref.arxiv_id:
                key = f"arxiv:{ref.arxiv_id}"
            elif ref.doi:
                key = f"doi:{ref.doi}"
            elif ref.detected_url:
                key = ref.detected_url

            if key and key not in seen:
                seen.add(key)
                unique.append(ref)
            elif not key:
                unique.append(ref)

        return unique

    def _deduplicate_papers(self, papers: list[Paper]) -> list[Paper]:
        """Deduplicate papers."""
        seen = set()
        unique = []

        for paper in papers:
            if paper.id not in seen:
                seen.add(paper.id)
                unique.append(paper)

        return unique


class PaperDownloader:
    """Downloads PDFs for resolved papers."""

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        settings: PipelineSettings,
    ):
        """Initialize paper downloader.

        Args:
            http_client: httpx async client.
            settings: Pipeline settings.
        """
        self.client = http_client
        self.settings = settings

    async def download_pdf(self, paper: Paper, dest_dir: Path) -> Paper:
        """Download PDF for a paper if available and open-access.

        Args:
            paper: Paper to download.
            dest_dir: Destination directory.

        Returns:
            Paper with updated local_pdf_path.
        """
        if not paper.pdf_url:
            logger.debug(f"No PDF URL for {paper.id}")
            return paper

        # Check if PDF is likely open-access
        if not self._is_likely_open_access(paper):
            logger.info(f"Skipping potentially paywalled PDF: {paper.id}")
            return paper

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from paper ID
        safe_id = re.sub(r"[:/]", "_", paper.id)
        pdf_path = dest_dir / f"{safe_id}.pdf"

        if pdf_path.exists():
            logger.debug(f"PDF already exists: {pdf_path}")
            paper.local_pdf_path = pdf_path
            return paper

        try:
            logger.info(f"Downloading PDF: {paper.pdf_url}")

            async with self.client.stream("GET", paper.pdf_url) as response:
                # Check content type
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower() and response.status_code == 200:
                    # Might be HTML (paywall page)
                    logger.warning(f"Unexpected content type for {paper.id}: {content_type}")
                    return paper

                response.raise_for_status()

                with open(pdf_path, "wb") as f:
                    async for chunk in response.aiter_bytes(8192):
                        f.write(chunk)

            # Verify it's a valid PDF
            if self._is_valid_pdf(pdf_path):
                paper.local_pdf_path = pdf_path
                logger.info(f"Downloaded PDF: {pdf_path}")
            else:
                pdf_path.unlink()
                logger.warning(f"Downloaded file is not a valid PDF: {paper.id}")

        except Exception as e:
            logger.warning(f"Failed to download PDF for {paper.id}: {e}")
            if pdf_path.exists():
                pdf_path.unlink()

        return paper

    def _is_likely_open_access(self, paper: Paper) -> bool:
        """Check if paper is likely open-access.

        Args:
            paper: Paper to check.

        Returns:
            True if likely open-access.
        """
        if not paper.pdf_url:
            return False

        url_lower = paper.pdf_url.lower()

        # Known open-access sources
        open_access_patterns = [
            "arxiv.org",
            "openaccess",
            "open-access",
            "semanticscholar.org",
            "unpaywall",
            "core.ac.uk",
            "hal.science",
            "biorxiv.org",
            "medrxiv.org",
            "ssrn.com",
        ]

        for pattern in open_access_patterns:
            if pattern in url_lower:
                return True

        # arXiv papers are always open-access
        if paper.arxiv_id:
            return True

        return False

    def _is_valid_pdf(self, path: Path) -> bool:
        """Check if file is a valid PDF.

        Args:
            path: File path.

        Returns:
            True if valid PDF.
        """
        try:
            with open(path, "rb") as f:
                header = f.read(5)
                return header == b"%PDF-"
        except Exception:
            return False

    async def download_many(
        self,
        papers: list[Paper],
        dest_dir: Path,
        concurrency: int = 3,
    ) -> list[Paper]:
        """Download PDFs for multiple papers.

        Args:
            papers: Papers to download.
            dest_dir: Destination directory.
            concurrency: Maximum concurrent downloads.

        Returns:
            Papers with updated local_pdf_path.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def download_with_limit(paper: Paper) -> Paper:
            async with semaphore:
                result = await self.download_pdf(paper, dest_dir)
                await asyncio.sleep(0.5)  # Rate limiting
                return result

        return await asyncio.gather(*[download_with_limit(p) for p in papers])
