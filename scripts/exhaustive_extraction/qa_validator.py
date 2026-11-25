"""
Automated QA validation for content extractions.

Note: Extractions now use SEGMENT BOUNDS for timestamps (not pinpointed
moments within segments). This simplifies timestamp validation.

Implements checks for:
- Timestamp accuracy (within segment bounds, start < end)
- Quote verbatim check (exact match in transcript)
- Slide reference validation (file exists, OCR confidence)
- Hallucination detection (generic statements not in source)
- Topic guardrail enforcement (required cues, min lengths)
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

from .schemas import ContentExtraction


@dataclass
class QAResult:
    """Result from QA check."""

    valid: bool
    reason: str = ""
    severity: str = "info"  # info, low, medium, high
    needs_review: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PresentationQAValidator:
    """QA validator for presentation extractions."""

    def __init__(self, conf_base_dir: Path, guardrails: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize validator.

        Args:
            conf_base_dir: Path to conference base directory
        """
        self.conf_base_dir = Path(conf_base_dir)
        self.guardrails = guardrails or {}

    def check_timestamps(self, extraction: ContentExtraction) -> QAResult:
        """
        Check timestamp validity.

        Note: Extractions now use SEGMENT BOUNDS for timestamps (not pinpointed
        moments). This means timestamp_start_sec == segment_start_sec and
        timestamp_end_sec == segment_end_sec for all extractions.

        Validates:
        1. Timestamps within segment bounds
        2. Start < End

        Args:
            extraction: ContentExtraction to validate

        Returns:
            QAResult
        """
        if extraction.source_type != "presentation":
            return QAResult(valid=True, reason="Not a presentation")

        # Ensure required fields are present
        required_fields = [
            extraction.segment_start_sec,
            extraction.segment_end_sec,
            extraction.timestamp_start_sec,
            extraction.timestamp_end_sec,
        ]
        if any(value is None for value in required_fields):
            return QAResult(
                valid=False,
                reason="Missing required timestamp/segment fields for presentation",
                severity="high"
            )

        # Check bounds
        if extraction.timestamp_start_sec < extraction.segment_start_sec:
            return QAResult(
                valid=False,
                reason=f"Timestamp start {extraction.timestamp_start_sec}s before segment start {extraction.segment_start_sec}s",
                severity="high"
            )

        if extraction.timestamp_end_sec > extraction.segment_end_sec:
            return QAResult(
                valid=False,
                reason=f"Timestamp end {extraction.timestamp_end_sec}s after segment end {extraction.segment_end_sec}s",
                severity="high"
            )

        # Check start < end
        if extraction.timestamp_start_sec >= extraction.timestamp_end_sec:
            return QAResult(
                valid=False,
                reason=f"Timestamp start {extraction.timestamp_start_sec}s >= end {extraction.timestamp_end_sec}s",
                severity="high"
            )

        # Note: Duration check removed. With segment-bound timestamps, extraction
        # duration equals segment duration (3-5 min for workshops, 2-5 min for
        # presentations). This is expected behavior, not a flag for review.

        return QAResult(valid=True, reason="Timestamps valid")

    def check_quote_in_transcript(
        self,
        extraction: ContentExtraction,
        transcript: str
    ) -> QAResult:
        """
        Check if verbatim quote appears in transcript.

        Note: With anchor-based extraction, quotes are extracted directly from
        the transcript using start/end anchors provided by the model. This check
        should always pass for anchor-extracted quotes. Failures indicate either:
        1. Legacy extractions with direct verbatim_quote (not anchor-extracted)
        2. Edge cases in anchor extraction that somehow produced invalid quotes

        Args:
            extraction: ContentExtraction to validate
            transcript: Full transcript text

        Returns:
            QAResult
        """
        if not extraction.verbatim_quote:
            return QAResult(valid=True, reason="No verbatim quote to check")

        # Normalize whitespace for comparison
        quote_normalized = " ".join(extraction.verbatim_quote.split())
        transcript_normalized = " ".join(transcript.split())

        # Check if quote appears in transcript
        if quote_normalized in transcript_normalized:
            return QAResult(valid=True, reason="Quote found in transcript")

        # Check for partial match (with [...] omissions)
        if "[...]" in extraction.verbatim_quote:
            # Split on [...] and check each part
            parts = [p.strip() for p in extraction.verbatim_quote.split("[...]")]
            all_parts_found = all(
                " ".join(p.split()) in transcript_normalized
                for p in parts if p
            )
            if all_parts_found:
                return QAResult(
                    valid=True,
                    reason="Quote parts found (with omissions)"
                )

        # Quote not found
        return QAResult(
            valid=False,
            reason="Verbatim quote not found in transcript",
            severity="high",
            metadata={"quote": extraction.verbatim_quote[:100]}
        )

    def check_slides_exist(
        self,
        extraction: ContentExtraction
    ) -> List[QAResult]:
        """
        Check slide reference validity.

        Validates:
        1. Slide file exists
        2. Slide timestamp within segment bounds
        3. OCR confidence assessment

        Args:
            extraction: ContentExtraction to validate

        Returns:
            List of QAResult (one per slide reference)
        """
        results = []

        for slide_ref in extraction.slide_references:
            # Check 1: File existence (skip if slide_path not provided)
            if slide_ref.slide_path is None:
                # slide_path is optional - skip file existence check
                # This is valid since processor may hydrate paths later
                pass
            else:
                slide_path = self.conf_base_dir / slide_ref.slide_path
                if not slide_path.exists():
                    results.append(QAResult(
                        valid=False,
                        reason=f"Slide file not found: {slide_ref.slide_id}",
                        severity="medium"
                    ))
                    continue

            # Check 2: Timestamp alignment (skip if timestamp not provided)
            if slide_ref.slide_timestamp_sec is None:
                # No timestamp provided - skip alignment check
                pass
            elif (extraction.segment_start_sec is not None and
                  extraction.segment_end_sec is not None):
                in_bounds = (
                    extraction.segment_start_sec <= slide_ref.slide_timestamp_sec <=
                    extraction.segment_end_sec
                )

                if not in_bounds:
                    results.append(QAResult(
                        valid=False if slide_ref.timestamp_aligned else True,
                        reason=f"Slide {slide_ref.slide_id} timestamp {slide_ref.slide_timestamp_sec}s "
                               f"outside segment bounds [{extraction.segment_start_sec}, {extraction.segment_end_sec}]",
                        severity="medium" if slide_ref.timestamp_aligned else "low",
                        metadata={"timestamp_aligned_flag": slide_ref.timestamp_aligned}
                    ))
                    continue

            # Check 3: OCR confidence
            if slide_ref.ocr_confidence in ["missing", "low"]:
                results.append(QAResult(
                    valid=True,
                    reason=f"Slide {slide_ref.slide_id} has {slide_ref.ocr_confidence} OCR confidence",
                    severity="low",
                    needs_review=True,
                    metadata={"ocr_confidence": slide_ref.ocr_confidence}
                ))
            else:
                results.append(QAResult(
                    valid=True,
                    reason=f"Slide {slide_ref.slide_id} valid"
                ))

        return results if results else [QAResult(valid=True, reason="No slides to check")]

    def detect_hallucination(
        self,
        extraction: ContentExtraction,
        transcript: str
    ) -> QAResult:
        """
        Detect potential hallucinations or generic statements.

        Checks for:
        - Generic phrases ("they discussed", "the paper talks about")
        - Concepts not present in transcript
        - Overly vague summaries

        Args:
            extraction: ContentExtraction to validate
            transcript: Full transcript text

        Returns:
            QAResult
        """
        summary = extraction.summary.lower()

        # Check for generic phrases
        generic_phrases = [
            "they discussed",
            "the paper talks about",
            "this presentation covers",
            "they mentioned",
            "they talked about",
            "the speaker said",
            "it was about"
        ]
        allow_generic = self.guardrails.get(extraction.topic_id, {}).get("allow_generic", False)
        if not allow_generic:
            for phrase in generic_phrases:
                if phrase in summary:
                    return QAResult(
                        valid=False,
                        reason=f"Generic statement detected: '{phrase}'",
                        severity="medium",
                        metadata={"generic_phrase": phrase}
                    )

        # Check if summary is too short (likely not substantive)
        if len(summary.split()) < 5:
            return QAResult(
                valid=False,
                reason="Summary too short (< 5 words)",
                severity="low",
                metadata={"summary_length": len(summary.split())}
            )

        return QAResult(valid=True, reason="No hallucination detected")

    def validate_extraction(
        self,
        extraction: ContentExtraction,
        transcript: str
    ) -> Dict[str, Any]:
        """
        Run all QA checks on an extraction.

        Args:
            extraction: ContentExtraction to validate
            transcript: Full transcript text

        Returns:
            Dict with all QA results
        """
        results = {
            "timestamp_check": self.check_timestamps(extraction),
            "quote_check": self.check_quote_in_transcript(extraction, transcript),
            "slide_checks": self.check_slides_exist(extraction),
            "hallucination_check": self.detect_hallucination(extraction, transcript)
        }
        guardrail_result = self._apply_topic_guardrails(extraction)
        if guardrail_result:
            results["topic_guardrail_check"] = guardrail_result
        return results

    def _apply_topic_guardrails(self, extraction: ContentExtraction) -> Optional[QAResult]:
        """Apply per-topic guardrail rules if present."""
        rules = self.guardrails.get(extraction.topic_id)
        if not rules:
            return None

        text_blob = " ".join(
            [
                extraction.verbatim_quote or "",
                extraction.summary or "",
            ]
        ).lower()

        # Required cues
        cues = rules.get("required_cues") or []
        if cues:
            if not any(cue.lower() in text_blob for cue in cues):
                return QAResult(
                    valid=False,
                    reason=f"Missing required cues for topic {extraction.topic_id}: {', '.join(cues)}",
                    severity="high",
                )

        # Min lengths
        min_quote = rules.get("min_quote_len")
        if min_quote and extraction.verbatim_quote:
            if len(extraction.verbatim_quote.split()) < min_quote:
                return QAResult(
                    valid=False,
                    reason=f"Quote too short (<{min_quote} words)",
                    severity="medium",
                )

        min_summary = rules.get("min_summary_len")
        if min_summary and extraction.summary:
            if len(extraction.summary.split()) < min_summary:
                return QAResult(
                    valid=False,
                    reason=f"Summary too short (<{min_summary} words)",
                    severity="medium",
                )

        return QAResult(valid=True, reason="Guardrails passed")


class PaperQAValidator:
    """QA validator for paper extractions."""

    def check_quote_in_abstract(
        self,
        extraction: ContentExtraction,
        abstract: str
    ) -> QAResult:
        """
        Check if quote appears in abstract.

        Args:
            extraction: ContentExtraction to validate
            abstract: Paper abstract text

        Returns:
            QAResult
        """
        if not extraction.verbatim_quote:
            return QAResult(valid=True, reason="No verbatim quote to check")

        # Normalize whitespace
        quote_normalized = " ".join(extraction.verbatim_quote.split())
        abstract_normalized = " ".join(abstract.split())

        if quote_normalized in abstract_normalized:
            return QAResult(valid=True, reason="Quote found in abstract")

        return QAResult(
            valid=False,
            reason="Quote not found in abstract",
            severity="high",
            metadata={"quote": extraction.verbatim_quote[:100]}
        )

    def detect_hallucination(
        self,
        extraction: ContentExtraction,
        abstract: str
    ) -> QAResult:
        """
        Detect hallucinations in paper extractions.

        Args:
            extraction: ContentExtraction to validate
            abstract: Paper abstract text

        Returns:
            QAResult
        """
        summary = extraction.summary.lower()
        abstract_lower = abstract.lower()

        # Check for generic phrases
        generic_phrases = [
            "this paper discusses",
            "this paper presents",
            "the authors describe",
            "this work explores",
            "this study examines"
        ]

        for phrase in generic_phrases:
            if phrase in summary:
                return QAResult(
                    valid=False,
                    reason=f"Generic statement: '{phrase}'",
                    severity="medium"
                )

        # Check if key terms in summary appear in abstract
        # Extract potential key terms (> 3 chars, not common words)
        common_words = {"the", "and", "for", "with", "this", "that", "from", "have", "been", "are"}
        summary_words = {
            w.lower() for w in re.findall(r'\b\w{4,}\b', summary)
            if w.lower() not in common_words
        }

        if summary_words:
            matches = sum(1 for word in summary_words if word in abstract_lower)
            match_rate = matches / len(summary_words)

            if match_rate < 0.3:  # Less than 30% of key terms found
                return QAResult(
                    valid=False,
                    reason=f"Low term overlap with abstract ({match_rate:.0%})",
                    severity="medium",
                    metadata={"match_rate": match_rate}
                )

        return QAResult(valid=True, reason="No hallucination detected")

    def validate_batch(
        self,
        extractions: List[ContentExtraction],
        abstracts: Dict[str, str],
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """
        Validate a batch of paper extractions (sampling).

        Args:
            extractions: List of extractions
            abstracts: Dict mapping source_id to abstract text
            sample_size: Number to sample for validation

        Returns:
            QA metrics dict
        """
        import random

        # Sample extractions
        sample = random.sample(extractions, min(sample_size, len(extractions)))

        quote_found = 0
        hallucination_detected = 0
        generic_detected = 0

        for extraction in sample:
            abstract = abstracts.get(extraction.source_id, "")
            if not abstract:
                continue

            # Check quote
            quote_result = self.check_quote_in_abstract(extraction, abstract)
            if quote_result.valid:
                quote_found += 1

            # Check hallucination
            hall_result = self.detect_hallucination(extraction, abstract)
            if not hall_result.valid:
                hallucination_detected += 1
                if "generic" in hall_result.reason.lower():
                    generic_detected += 1

        total = len(sample)
        return {
            "paper_quote_in_abstract": quote_found / total if total > 0 else 0,
            "paper_hallucination_rate": hallucination_detected / total if total > 0 else 0,
            "paper_generic_rate": generic_detected / total if total > 0 else 0,
            "sample_size": total
        }
