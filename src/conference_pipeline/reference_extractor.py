"""Reference extraction from OCR text and transcripts."""

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.models import OcrResult, ReferenceCandidate, TranscriptChunk
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_chat_completions_create,
)
from prompts.reference_normalization import format_normalization_prompt
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionPatterns:
    """Regex patterns for reference extraction."""

    # URL patterns
    url: re.Pattern = re.compile(
        r'https?://[^\s<>"\')\]]+',
        re.IGNORECASE,
    )

    # arXiv ID patterns
    arxiv_new: re.Pattern = re.compile(
        r'(?:arxiv[:\s]*)?(\d{4}\.\d{4,5}(?:v\d+)?)',
        re.IGNORECASE,
    )
    arxiv_old: re.Pattern = re.compile(
        r'arxiv[:\s/]*([a-z.-]+/\d{7})',
        re.IGNORECASE,
    )
    arxiv_url: re.Pattern = re.compile(
        r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|\w+-\w+/\d+)',
        re.IGNORECASE,
    )

    # DOI patterns
    doi: re.Pattern = re.compile(
        r'(?:doi[:\s]*)?10\.\d{4,}/[^\s<>"\')\]]+',
        re.IGNORECASE,
    )

    # Citation patterns (Author et al., Year)
    citation_etal: re.Pattern = re.compile(
        r'([A-Z][a-z]+)\s+et\s+al\.?,?\s*\(?(\d{4})\)?',
    )

    # Multiple authors citation
    citation_multi: re.Pattern = re.compile(
        r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)+),?\s*\(?(\d{4})\)?',
    )

    # Conference/venue references
    venue: re.Pattern = re.compile(
        r'(NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|ACL|EMNLP|NAACL|AAAI|IJCAI)\s*(\d{4})',
        re.IGNORECASE,
    )


class ReferenceExtractor:
    """Extracts reference candidates from text."""

    def __init__(self):
        """Initialize reference extractor."""
        self.patterns = ExtractionPatterns()

    def extract_from_ocr(self, ocr: OcrResult) -> list[ReferenceCandidate]:
        """Extract references from OCR result.

        Args:
            ocr: OCR result to process.

        Returns:
            List of reference candidates.
        """
        if not ocr.raw_text:
            return []

        candidates = []

        # Extract URLs
        candidates.extend(
            self._extract_urls(
                ocr.raw_text,
                ocr.presentation_id,
                ocr.slide_index,
                "slide_ocr",
            )
        )

        # Extract arXiv IDs
        candidates.extend(
            self._extract_arxiv_ids(
                ocr.raw_text,
                ocr.presentation_id,
                ocr.slide_index,
                "slide_ocr",
            )
        )

        # Extract DOIs
        candidates.extend(
            self._extract_dois(
                ocr.raw_text,
                ocr.presentation_id,
                ocr.slide_index,
                "slide_ocr",
            )
        )

        # Extract citation patterns
        candidates.extend(
            self._extract_citations(
                ocr.raw_text,
                ocr.presentation_id,
                ocr.slide_index,
                "slide_ocr",
            )
        )

        # Deduplicate
        candidates = self._deduplicate_candidates(candidates)

        logger.debug(
            f"Extracted {len(candidates)} references from slide {ocr.slide_index}"
        )
        return candidates

    def extract_from_transcript(
        self,
        transcript_chunks: list[TranscriptChunk],
        presentation_id: str,
    ) -> list[ReferenceCandidate]:
        """Extract references from transcript chunks.

        Args:
            transcript_chunks: Transcript chunks to process.
            presentation_id: Presentation ID.

        Returns:
            List of reference candidates.
        """
        candidates = []

        for chunk in transcript_chunks:
            if not chunk.text:
                continue

            # Extract URLs
            candidates.extend(
                self._extract_urls(
                    chunk.text,
                    presentation_id,
                    None,  # No slide index for transcripts
                    "transcript",
                )
            )

            # Extract arXiv IDs
            candidates.extend(
                self._extract_arxiv_ids(
                    chunk.text,
                    presentation_id,
                    None,
                    "transcript",
                )
            )

            # Extract DOIs
            candidates.extend(
                self._extract_dois(
                    chunk.text,
                    presentation_id,
                    None,
                    "transcript",
                )
            )

            # Extract citation patterns
            candidates.extend(
                self._extract_citations(
                    chunk.text,
                    presentation_id,
                    None,
                    "transcript",
                )
            )

        # Deduplicate
        candidates = self._deduplicate_candidates(candidates)

        logger.debug(
            f"Extracted {len(candidates)} references from transcript"
        )
        return candidates

    def _extract_urls(
        self,
        text: str,
        presentation_id: str,
        slide_index: int | None,
        source_type: str,
    ) -> list[ReferenceCandidate]:
        """Extract URLs from text."""
        candidates = []

        for match in self.patterns.url.finditer(text):
            url = match.group(0)

            # Clean up URL (remove trailing punctuation)
            url = re.sub(r'[.,;:!?]+$', '', url)

            # Skip common non-paper URLs
            if self._is_irrelevant_url(url):
                continue

            # Check if URL contains arXiv reference
            arxiv_match = self.patterns.arxiv_url.search(url)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None

            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    detected_url=url,
                    arxiv_id=arxiv_id,
                )
            )

        return candidates

    def _extract_arxiv_ids(
        self,
        text: str,
        presentation_id: str,
        slide_index: int | None,
        source_type: str,
    ) -> list[ReferenceCandidate]:
        """Extract arXiv IDs from text."""
        candidates = []

        # New format (YYMM.NNNNN)
        for match in self.patterns.arxiv_new.finditer(text):
            arxiv_id = match.group(1)
            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    arxiv_id=arxiv_id,
                )
            )

        # Old format (category/YYYYNNN)
        for match in self.patterns.arxiv_old.finditer(text):
            arxiv_id = match.group(1)
            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    arxiv_id=arxiv_id,
                )
            )

        return candidates

    def _extract_dois(
        self,
        text: str,
        presentation_id: str,
        slide_index: int | None,
        source_type: str,
    ) -> list[ReferenceCandidate]:
        """Extract DOIs from text."""
        candidates = []

        for match in self.patterns.doi.finditer(text):
            doi = match.group(0)

            # Clean up DOI prefix
            doi = re.sub(r'^doi[:\s]*', '', doi, flags=re.IGNORECASE)

            # Clean trailing punctuation
            doi = re.sub(r'[.,;:!?]+$', '', doi)

            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    doi=doi,
                )
            )

        return candidates

    def _extract_citations(
        self,
        text: str,
        presentation_id: str,
        slide_index: int | None,
        source_type: str,
    ) -> list[ReferenceCandidate]:
        """Extract citation patterns (Author et al., Year) from text."""
        candidates = []

        # Author et al., Year
        for match in self.patterns.citation_etal.finditer(text):
            author = match.group(1)
            year = match.group(2)

            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    authors=[f"{author} et al."],
                    title=None,  # Would need to resolve
                )
            )

        # Multi-author citations
        for match in self.patterns.citation_multi.finditer(text):
            authors_str = match.group(1)
            # year = match.group(2)  # Available if needed

            # Parse authors
            authors = re.split(r'\s+(?:and|&)\s+', authors_str)

            candidates.append(
                ReferenceCandidate(
                    id=str(uuid.uuid4()),
                    presentation_id=presentation_id,
                    slide_index=slide_index,
                    source_type=source_type,
                    raw_text=self._get_context(text, match.start(), match.end()),
                    authors=authors,
                )
            )

        return candidates

    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        context_chars: int = 100,
    ) -> str:
        """Get surrounding context for a match.

        Args:
            text: Full text.
            start: Match start position.
            end: Match end position.
            context_chars: Number of context characters on each side.

        Returns:
            Text snippet with context.
        """
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)

        snippet = text[context_start:context_end]

        # Add ellipsis if truncated
        if context_start > 0:
            snippet = "..." + snippet
        if context_end < len(text):
            snippet = snippet + "..."

        return snippet

    def _is_irrelevant_url(self, url: str) -> bool:
        """Check if URL is likely not a paper reference.

        Args:
            url: URL to check.

        Returns:
            True if URL is likely irrelevant.
        """
        irrelevant_domains = [
            "twitter.com",
            "x.com",
            "facebook.com",
            "linkedin.com",
            "youtube.com",
            "youtu.be",
            "instagram.com",
            "github.com",  # Sometimes relevant, but often not paper refs
            "google.com",
            "neurips.cc",  # Conference site itself  # conference-literal-ok: known non-paper domain filter; harmless to keep
            "slideslive.com",
        ]

        url_lower = url.lower()
        for domain in irrelevant_domains:
            if domain in url_lower:
                return True

        return False

    def _deduplicate_candidates(
        self,
        candidates: list[ReferenceCandidate],
    ) -> list[ReferenceCandidate]:
        """Deduplicate reference candidates.

        Args:
            candidates: List of candidates.

        Returns:
            Deduplicated list.
        """
        seen = set()
        unique = []

        for candidate in candidates:
            # Create a key for deduplication
            key_parts = []
            if candidate.arxiv_id:
                key_parts.append(f"arxiv:{candidate.arxiv_id}")
            if candidate.doi:
                key_parts.append(f"doi:{candidate.doi}")
            if candidate.detected_url:
                key_parts.append(f"url:{candidate.detected_url}")

            key = "|".join(key_parts) if key_parts else candidate.raw_text[:50]

            if key not in seen:
                seen.add(key)
                unique.append(candidate)

        return unique

    def merge_candidates(
        self,
        ocr_candidates: list[ReferenceCandidate],
        transcript_candidates: list[ReferenceCandidate],
    ) -> list[ReferenceCandidate]:
        """Merge and deduplicate candidates from multiple sources.

        Args:
            ocr_candidates: Candidates from OCR.
            transcript_candidates: Candidates from transcript.

        Returns:
            Merged and deduplicated list.
        """
        all_candidates = ocr_candidates + transcript_candidates
        return self._deduplicate_candidates(all_candidates)


class LLMReferenceNormalizer:
    """Normalizes reference candidates using an LLM.

    Uses Claude or GPT to clean up OCR errors and extract structured
    reference information from noisy text.
    """

    def __init__(self, settings: PipelineSettings):
        """Initialize LLM reference normalizer.

        Args:
            settings: Pipeline settings with API keys.
        """
        self.settings = settings
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self) -> Any:
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic

                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.settings.anthropic_api_key
                )
            except ImportError:
                raise ImportError(
                    "Anthropic client not available. Install with: pip install anthropic"
                )
        return self._anthropic_client

    def _get_openai_client(self) -> Any:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            try:
                import openai

                self._openai_client = openai.OpenAI(
                    api_key=self.settings.openai_api_key
                )
            except ImportError:
                raise ImportError(
                    "OpenAI client not available. Install with: pip install openai"
                )
        return self._openai_client

    async def normalize_candidate(
        self,
        candidate: ReferenceCandidate,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """Normalize a single reference candidate using LLM.

        Args:
            candidate: Reference candidate to normalize.
            provider: LLM provider ("anthropic" or "openai").

        Returns:
            Updated candidate with normalized fields.
        """
        prompt = format_normalization_prompt(
            raw_text=candidate.raw_text,
            arxiv_id=candidate.arxiv_id,
            doi=candidate.doi,
            url=candidate.detected_url,
            authors=candidate.authors,
        )

        try:
            if provider == "anthropic":
                result = await self._call_anthropic(prompt)
            else:
                result = await self._call_openai(prompt)

            # Parse JSON response
            normalized = self._parse_llm_response(result)

            # Update candidate with normalized values
            if normalized.get("normalized_title"):
                candidate.normalized_title = normalized["normalized_title"]
            if normalized.get("normalized_authors"):
                candidate.normalized_authors = normalized["normalized_authors"]
            if normalized.get("arxiv_id") and not candidate.arxiv_id:
                candidate.arxiv_id = normalized["arxiv_id"]
            if normalized.get("doi") and not candidate.doi:
                candidate.doi = normalized["doi"]
            if normalized.get("confidence"):
                candidate.confidence = normalized["confidence"]

            candidate.normalized = True
            logger.debug(f"Normalized reference: {candidate.normalized_title}")

        except Exception as e:
            logger.warning(f"Failed to normalize reference: {e}")
            candidate.normalized = False

        return candidate

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Claude API."""
        import asyncio

        client = self._get_anthropic_client()

        # Run in executor since anthropic client is synchronous
        def _call():
            response = anthropic_messages_create(client, caller=__file__,
                model=self.settings.anthropic_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return await asyncio.get_event_loop().run_in_executor(None, _call)

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        import asyncio

        client = self._get_openai_client()

        def _call():
            response = openai_chat_completions_create(client, caller=__file__,
                model=default_model_for_provider("analysis.source_reading", "openai"),
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        return await asyncio.get_event_loop().run_in_executor(None, _call)

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw LLM response.

        Returns:
            Parsed dictionary.
        """
        # Try to extract JSON from response
        try:
            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Try parsing entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {response[:100]}")
            return {}

    async def normalize_batch(
        self,
        candidates: list[ReferenceCandidate],
        provider: str = "anthropic",
        max_concurrent: int = 5,
    ) -> list[ReferenceCandidate]:
        """Normalize multiple candidates with rate limiting.

        Args:
            candidates: List of candidates to normalize.
            provider: LLM provider.
            max_concurrent: Maximum concurrent API calls.

        Returns:
            List of normalized candidates.
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def normalize_with_limit(candidate: ReferenceCandidate) -> ReferenceCandidate:
            async with semaphore:
                return await self.normalize_candidate(candidate, provider)

        results = await asyncio.gather(
            *[normalize_with_limit(c) for c in candidates],
            return_exceptions=True,
        )

        normalized = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Normalization failed for candidate: {result}")
                normalized.append(candidates[i])
            else:
                normalized.append(result)

        successful = sum(1 for c in normalized if c.normalized)
        logger.info(
            f"Normalized {successful}/{len(normalized)} references"
        )

        return normalized

    async def normalize_with_context(
        self,
        candidate: ReferenceCandidate,
        slide_text: str | None = None,
        surrounding_refs: list[ReferenceCandidate] | None = None,
        provider: str = "anthropic",
    ) -> ReferenceCandidate:
        """Normalize with additional context for better accuracy.

        Args:
            candidate: Reference candidate to normalize.
            slide_text: Full slide text for context.
            surrounding_refs: Other references on same slide.
            provider: LLM provider.

        Returns:
            Normalized candidate.
        """
        context_parts = []

        if slide_text:
            context_parts.append(f"Full slide text:\n{slide_text[:500]}")

        if surrounding_refs:
            other_refs = [
                r.raw_text[:100] for r in surrounding_refs[:3] if r.id != candidate.id
            ]
            if other_refs:
                context_parts.append(f"Other references on slide:\n" + "\n".join(other_refs))

        extra_context = "\n\n".join(context_parts) if context_parts else None

        prompt = format_normalization_prompt(
            raw_text=candidate.raw_text,
            arxiv_id=candidate.arxiv_id,
            doi=candidate.doi,
            url=candidate.detected_url,
            authors=candidate.authors,
            extra_context=extra_context,
        )

        try:
            if provider == "anthropic":
                result = await self._call_anthropic(prompt)
            else:
                result = await self._call_openai(prompt)

            normalized = self._parse_llm_response(result)

            if normalized.get("normalized_title"):
                candidate.normalized_title = normalized["normalized_title"]
            if normalized.get("normalized_authors"):
                candidate.normalized_authors = normalized["normalized_authors"]
            if normalized.get("arxiv_id") and not candidate.arxiv_id:
                candidate.arxiv_id = normalized["arxiv_id"]
            if normalized.get("doi") and not candidate.doi:
                candidate.doi = normalized["doi"]
            if normalized.get("confidence"):
                candidate.confidence = normalized["confidence"]

            candidate.normalized = True

        except Exception as e:
            logger.warning(f"Failed to normalize reference with context: {e}")
            candidate.normalized = False

        return candidate
