"""
Main processing logic for exhaustive content extraction.

Handles:
- Segment-by-segment processing for presentations
- Batch processing for papers
- Integration with extraction runner, storage, and QA validation
- Progress tracking and cost estimation
- Resume capability at segment level
"""

import json
import os
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .schemas import ContentExtraction, ExtractionBatch, SlideReference
from .models import ExtractionRunner, estimate_tokens
from .storage import QueuedTopicStorage, ProgressTracker
from .qa_validator import PresentationQAValidator, PaperQAValidator
from .prompts import (
    build_presentation_extraction_prompt,
    build_paper_extraction_prompt,
    build_paper_batch_extraction_prompt,
    build_workshop_extraction_prompt,
    format_slides_list,
    seconds_to_hms,
    format_guardrails_for_prompt,
)
from .workshop_analyzer import (
    WorkshopAnalyzer,
    Talk,
    find_talks_for_batch,
    save_talks_json,
    load_talks_json,
)
from .quote_extractor import extract_quote_from_anchors, extract_quote_with_gpt52_fallback

try:
    from conference_pipeline.costs.estimator import estimate_text_model_cost
except ImportError:  # pragma: no cover - script may run outside editable installs
    _repo_src = Path(__file__).resolve().parents[2] / "src"
    if _repo_src.exists() and str(_repo_src) not in sys.path:
        sys.path.insert(0, str(_repo_src))
    try:
        from conference_pipeline.costs.estimator import estimate_text_model_cost
    except ImportError:
        estimate_text_model_cost = None


def include_slides_for_extraction() -> bool:
    """Return whether slide OCR should be included in extraction prompts."""

    raw = os.getenv("EXHAUSTIVE_EXTRACTION_INCLUDE_SLIDES", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


from conference_pipeline.model_governance.registry import default_model

DEFAULT_EXTRACTION_COST_MODEL = default_model("exhaustive_extraction.cost_estimation")


class PresentationProcessor:
    """Process presentation segments with extraction."""

    def __init__(
        self,
        conf_base_dir: Path,
        output_dir: Path,
        storage: QueuedTopicStorage,
        progress_tracker: ProgressTracker,
        qa_validator: PresentationQAValidator,
        guardrails: Optional[Dict[str, Any]] = None,
        max_workers: int = 3,
        batch_short_segments: bool = False,
        batch_token_limit: int = 3000,
        batch_duration_limit: int = 300,
    ):
        """
        Initialize presentation processor.

        Args:
            conf_base_dir: Path to conference base directory
            output_dir: Path to output directory
            storage: Storage manager
            progress_tracker: Progress tracker
            qa_validator: QA validator
        """
        self.conf_base_dir = Path(conf_base_dir)
        self.output_dir = Path(output_dir)
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.qa_validator = qa_validator
        self.guardrails = guardrails or {}
        self.extraction_runner = ExtractionRunner()
        self.topic_guardrail_text = format_guardrails_for_prompt(self.guardrails)
        self.max_workers = max_workers
        self.batch_short_segments = batch_short_segments
        self.batch_token_limit = batch_token_limit
        self.batch_duration_limit = batch_duration_limit

        # Initialize workshop analyzer for detecting and processing workshops
        self.workshop_analyzer = WorkshopAnalyzer()

    def load_presentation_metadata(self, pres_id: str) -> Optional[Dict[str, Any]]:
        """
        Load presentation metadata.

        Args:
            pres_id: Presentation ID

        Returns:
            Metadata dict or None if not found
        """
        metadata_path = self.conf_base_dir / "decks" / pres_id / "meta.json"
        if not metadata_path.exists():
            return None

        with open(metadata_path) as f:
            return json.load(f)

    def load_segments(self, pres_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load presentation segments.

        Args:
            pres_id: Presentation ID

        Returns:
            List of segment dicts or None if not found
        """
        segments_path = self.conf_base_dir / "decks" / pres_id / "transcripts" / "segments.json"
        if not segments_path.exists():
            return None

        with open(segments_path) as f:
            segments = json.load(f)
            # Normalize segment format
            normalized = []
            for idx, seg in enumerate(segments):
                normalized.append({
                    "segment_index": idx,
                    "start_sec": seg.get("start", 0),
                    "end_sec": seg.get("end", 0),
                    "transcript": seg.get("text", "")
                })
            return normalized

    def _load_all_slides(self, pres_id: str) -> List[Dict[str, Any]]:
        """
        Load all slides for a presentation once (for caching).

        Args:
            pres_id: Presentation ID

        Returns:
            List of slide dicts with OCR content
        """
        ocr_dir = self.conf_base_dir / "decks" / pres_id / "ocr"
        if not ocr_dir.exists():
            return []

        slides = []
        for slide_file in sorted(ocr_dir.glob("slide_*.json")):
            with open(slide_file) as f:
                slide_data = json.load(f)

            timestamp_sec = slide_data.get("timestamp_sec", 0)

            # Extract text from slide_text object
            slide_text_obj = slide_data.get("slide_text", {})
            slide_text_parts = []

            if slide_text_obj.get("title"):
                slide_text_parts.append(f"Title: {slide_text_obj['title']}")

            if slide_text_obj.get("bullet_points"):
                slide_text_parts.append("Bullet points:")
                slide_text_parts.extend([f"  - {bp}" for bp in slide_text_obj['bullet_points']])

            if slide_text_obj.get("other_text"):
                slide_text_parts.append(f"Other text: {slide_text_obj['other_text']}")

            slide_text_combined = "\n".join(slide_text_parts) if slide_text_parts else "[No OCR text available]"

            slides.append({
                "slide_id": f"slide_{slide_data.get('slide_index', 0):03d}",
                "timestamp_sec": timestamp_sec,
                "timestamp_hms": seconds_to_hms(timestamp_sec),
                "slide_text": slide_text_combined,
                "slide_path": str(slide_file.relative_to(self.conf_base_dir))
            })

        return slides

    def load_slides_for_segment(
        self,
        pres_id: str,
        segment_start_sec: float,
        segment_end_sec: float,
        slides_cache: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Load slides that appear during a segment.

        Args:
            pres_id: Presentation ID
            segment_start_sec: Segment start time
            segment_end_sec: Segment end time

        Returns:
            List of slide dicts with OCR content
        """
        slides_source = slides_cache if slides_cache is not None else self._load_all_slides(pres_id)
        return [
            slide for slide in slides_source
            if segment_start_sec <= slide.get("timestamp_sec", 0) <= segment_end_sec
        ]

    def process_segment(
        self,
        pres_id: str,
        pres_title: str,
        segment: Dict[str, Any],
        slides_cache: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, int, int, Dict[str, int]]:
        """
        Process a single segment.

        Args:
            pres_id: Presentation ID
            pres_title: Presentation title
            segment: Segment dict

        Returns:
            Tuple of (extractions_count, input_tokens, output_tokens, topic_distribution)
        """
        segment_index = segment.get("segment_index", 0)
        segment_start_sec = segment.get("start_sec", 0)
        segment_end_sec = segment.get("end_sec", 0)
        segment_duration_sec = segment_end_sec - segment_start_sec
        transcript_text = segment.get("transcript", "")

        # Convert to HH:MM:SS
        segment_start_hms = seconds_to_hms(segment_start_sec)
        segment_end_hms = seconds_to_hms(segment_end_sec)

        # Load slides for this segment unless the run explicitly separates
        # slide-description enrichment from authoritative semantic extraction.
        slides = (
            self.load_slides_for_segment(pres_id, segment_start_sec, segment_end_sec, slides_cache)
            if include_slides_for_extraction()
            else []
        )
        slides_list = format_slides_list(slides)

        # Build extraction prompt
        prompt = build_presentation_extraction_prompt(
            presentation_title=pres_title,
            segment_index=segment_index,
            segment_start_sec=segment_start_sec,
            segment_end_sec=segment_end_sec,
            segment_start_hms=segment_start_hms,
            segment_end_hms=segment_end_hms,
            segment_duration_sec=segment_duration_sec,
            transcript_text=transcript_text,
            slides_list_with_ocr=slides_list,
            topic_guardrails=self.topic_guardrail_text
        )

        print(f"    Segment {segment_index}: {segment_start_hms} - {segment_end_hms} ({segment_duration_sec:.0f}s)")

        # Run extraction
        try:
            parsed, model_used, input_tokens, output_tokens = self.extraction_runner.run_extraction(prompt)
        except Exception as e:
            print(f"      ❌ Extraction failed: {e}")
            return 0, 0, 0, {}

        # Parse extractions
        extraction_list = parsed.get("extractions", [])

        # Pre-filter: skip extractions with invalid extraction_type or timestamps
        # Models sometimes return "none" or other invalid values to signal "no content"
        VALID_EXTRACTION_TYPES = {"verbatim_quote", "summary"}
        filtered_list = []
        invalid_type_count = 0
        invalid_ts_count = 0
        for ext in extraction_list:
            ext_type = ext.get("extraction_type", "summary")
            if ext_type not in VALID_EXTRACTION_TYPES:
                invalid_type_count += 1
                continue
            # Also skip if timestamps are equal (invalid range)
            ts_start = ext.get("timestamp_start_sec")
            ts_end = ext.get("timestamp_end_sec")
            if ts_start is not None and ts_end is not None and ts_start >= ts_end:
                invalid_ts_count += 1
                continue
            filtered_list.append(ext)

        skipped_count = invalid_type_count + invalid_ts_count
        if skipped_count > 0:
            print(f"      ⚠️  Pre-filtered {skipped_count} invalid extractions (type={invalid_type_count}, timestamps={invalid_ts_count})")
            # Track stats for analysis
            self.progress_tracker.update_pre_filter_stats(
                invalid_extraction_type=invalid_type_count,
                invalid_timestamps=invalid_ts_count
            )

        extraction_list = filtered_list
        print(f"      Model: {model_used}, Extractions: {len(extraction_list)}")

        # Process each extraction
        valid_extractions = 0
        topic_distribution = {}

        for extraction_data in extraction_list:
            try:
                # Build ContentExtraction object (with anchor-based quote extraction)
                extraction = self._build_extraction(
                    extraction_data=extraction_data,
                    source_type="presentation",
                    source_id=pres_id,
                    source_title=pres_title,
                    segment_index=segment_index,
                    segment_start_sec=segment_start_sec,
                    segment_end_sec=segment_end_sec,
                    model_used=model_used,
                    source_text=transcript_text
                )

                # QA validation
                qa_results = self.qa_validator.validate_extraction(extraction, transcript_text)

                # Check if validation passed and collect failure reasons
                all_valid = True
                failure_reasons = []

                for check_name, result in qa_results.items():
                    if check_name == "slide_checks":
                        # slide_checks is a list
                        for slide_result in result:
                            if not slide_result.valid:
                                all_valid = False
                                failure_reasons.append(
                                    f"{check_name}: {slide_result.reason} (severity: {slide_result.severity})"
                                )
                    else:
                        if not result.valid:
                            all_valid = False
                            failure_reasons.append(
                                f"{check_name}: {result.reason} (severity: {result.severity})"
                            )

                if all_valid:
                    extraction.qa_validated = True
                    valid_extractions += 1

                    # Update topic distribution
                    topic_id = extraction.topic_id
                    topic_distribution[topic_id] = topic_distribution.get(topic_id, 0) + 1

                    # Store extraction
                    self.storage.append_extraction(extraction, topic_id)
                else:
                    # Report QA failures with specific reasons
                    print(f"      ⚠️  QA failed for extraction: {extraction.extraction_id}")
                    for reason in failure_reasons[:3]:  # Show first 3 failures to avoid log spam
                        print(f"         - {reason}")
                    if len(failure_reasons) > 3:
                        print(f"         ... and {len(failure_reasons) - 3} more")

                    # Record QA failure for later review
                    self.progress_tracker.record_qa_failure(
                        extraction_id=extraction.extraction_id,
                        pres_id=pres_id,
                        segment_index=segment_index,
                        reasons=failure_reasons,
                        extraction_data=extraction.model_dump()
                    )

            except Exception as e:
                print(f"      ❌ Failed to process extraction: {e}")
                continue

        print(f"      ✓ Valid extractions: {valid_extractions}/{len(extraction_list)}")

        return valid_extractions, input_tokens, output_tokens, topic_distribution

    def _build_extraction(
        self,
        extraction_data: Dict[str, Any],
        source_type: str,
        source_id: str,
        source_title: str,
        segment_index: Optional[int],
        segment_start_sec: Optional[float],
        segment_end_sec: Optional[float],
        model_used: str,
        source_text: Optional[str] = None
    ) -> ContentExtraction:
        """
        Build ContentExtraction object from extraction data.

        Args:
            extraction_data: Raw extraction dict from model
            source_type: "presentation" or "paper"
            source_id: Source ID
            source_title: Source title
            segment_index: Segment index (for presentations)
            segment_start_sec: Segment start (for presentations)
            segment_end_sec: Segment end (for presentations)
            model_used: Model ID
            source_text: Original transcript/abstract text for anchor-based quote extraction

        Returns:
            ContentExtraction object

        Raises:
            KeyError: If required fields are missing from extraction_data
            ValueError: If extraction_data is malformed or anchor extraction fails
        """
        # Validate required fields with explicit error reporting
        required_fields = ["topic_id", "summary"]
        missing_fields = [f for f in required_fields if f not in extraction_data]
        if missing_fields:
            raise KeyError(
                f"Missing required fields in extraction_data: {', '.join(missing_fields)}. "
                f"Available fields: {', '.join(extraction_data.keys())}"
            )

        topic_id = extraction_data["topic_id"]
        summary = extraction_data["summary"]

        # Handle verbatim quote extraction from anchors
        verbatim_quote = extraction_data.get("verbatim_quote")
        quote_start_anchor = extraction_data.get("quote_start_anchor")
        quote_end_anchor = extraction_data.get("quote_end_anchor")

        # If we have anchors and source text, extract the quote
        if quote_start_anchor and quote_end_anchor and source_text:
            # Use GPT-5.2 fallback for direct quote finding when anchor matching fails
            quote_result = extract_quote_with_gpt52_fallback(
                source_text=source_text,
                start_anchor=quote_start_anchor,
                end_anchor=quote_end_anchor,
                summary=summary,
                topic_name=extraction_data.get("topic_name"),
                timestamp_start_sec=extraction_data.get("timestamp_start_sec"),
                timestamp_end_sec=extraction_data.get("timestamp_end_sec"),
                relevance_score=extraction_data.get("relevance_score")
            )

            if quote_result.success:
                verbatim_quote = quote_result.verbatim_quote
                if quote_result.method == "gpt52_fallback":
                    print(f"      📎 Quote extracted via GPT-5.2 fallback")
            else:
                # Both anchor matching and GPT-5.2 failed - save summary without quote
                verbatim_quote = None
                print(f"      ⚠️  Quote extraction failed: {quote_result.error}")
                print(f"      💾 Saving extraction with summary only (no verbatim quote)")

        # Validate topic_name with explicit error
        topic_name = extraction_data.get("topic_name")
        if not topic_name:
            raise ValueError(
                f"topic_name is required but missing or empty in extraction_data. "
                f"topic_id={topic_id}, source_id={source_id}"
            )

        # Compute content hash
        content_hash = ContentExtraction.compute_content_hash(
            source_id=source_id,
            segment_index=segment_index,
            topic_id=topic_id,
            verbatim_quote=verbatim_quote,
            summary=summary
        )

        # Build extraction ID
        extraction_id = f"{source_id}_{topic_id}_{content_hash[:16]}"

        # Build extraction object
        extraction_dict = {
            "extraction_id": extraction_id,
            "content_hash": content_hash,
            "source_type": source_type,
            "source_id": source_id,
            "source_title": source_title,
            "topic_id": topic_id,
            "topic_name": topic_name,
            "extraction_type": extraction_data.get("extraction_type", "summary"),
            "verbatim_quote": verbatim_quote,
            "summary": summary,
            "relevance_score": extraction_data.get("relevance_score", 0.5),
            "model_used": model_used,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }

        # Add presentation-specific fields
        if source_type == "presentation":
            extraction_dict.update({
                "segment_index": segment_index,
                "segment_start_sec": segment_start_sec,
                "segment_end_sec": segment_end_sec,
                "timestamp_start_sec": extraction_data.get("timestamp_start_sec"),
                "timestamp_end_sec": extraction_data.get("timestamp_end_sec"),
                "timestamp_start": extraction_data.get("timestamp_start"),
                "timestamp_end": extraction_data.get("timestamp_end"),
                "slide_references": extraction_data.get("slide_references", []),
                "transcript_quality": extraction_data.get("transcript_quality")
            })
        else:
            # Paper-specific fields
            extraction_dict.update({
                "source_coverage": extraction_data.get("source_coverage", "abstract_only")
            })

        return ContentExtraction(**extraction_dict)

    def _should_process_segment(self, transcript: str) -> bool:
        """
        Lightweight pre-filter: if guardrails define cues, require at least one cue in transcript.
        Designed to avoid skipping in short transcripts; only skips when cues exist and transcript is non-empty.
        """
        if not self.guardrails:
            return True
        transcript_lower = transcript.lower()
        cue_union = set()
        for rules in self.guardrails.values():
            for cue in rules.get("required_cues", []) or []:
                cue_union.add(cue.lower())
        if not cue_union:
            return True
        # Only skip if transcript is reasonably long and no cues present
        if len(transcript_lower.split()) < 5:
            return True
        return any(cue in transcript_lower for cue in cue_union)

    def _batch_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optionally batch adjacent short segments to reduce LLM calls.
        Segments are combined while total duration and estimated tokens stay under limits.

        Batched segments include segment_index_start and segment_index_end to track
        the range of original segments that were combined.
        """
        if not self.batch_short_segments or not segments:
            return segments

        batched: List[Dict[str, Any]] = []
        current = None
        current_tokens = 0
        current_duration = 0.0

        for seg in segments:
            seg_duration = seg.get("end_sec", 0) - seg.get("start_sec", 0)
            transcript = seg.get("transcript", "")
            seg_tokens = estimate_tokens(transcript)
            seg_index = seg.get("segment_index", 0)

            if current is None:
                current = seg.copy()
                # Track segment range for QA validation
                current["segment_index_start"] = seg_index
                current["segment_index_end"] = seg_index
                current_tokens = seg_tokens
                current_duration = seg_duration
                continue

            combined_tokens = current_tokens + seg_tokens
            combined_duration = (seg.get("end_sec", 0) - current.get("start_sec", 0))

            if combined_tokens <= self.batch_token_limit and combined_duration <= self.batch_duration_limit:
                # merge into current
                current["end_sec"] = max(current.get("end_sec", 0), seg.get("end_sec", 0))
                current["transcript"] = current.get("transcript", "") + "\n\n" + transcript
                # Update segment range end
                current["segment_index_end"] = seg_index
                current_tokens = combined_tokens
                current_duration = combined_duration
            else:
                batched.append(current)
                current = seg.copy()
                # Track segment range for new batch
                current["segment_index_start"] = seg_index
                current["segment_index_end"] = seg_index
                current_tokens = seg_tokens
                current_duration = seg_duration

        if current is not None:
            batched.append(current)

        return batched

    def process_presentation(self, pres_id: str) -> bool:
        """
        Process all segments for a presentation.

        Automatically detects workshops and routes to workshop-specific processing.

        Args:
            pres_id: Presentation ID

        Returns:
            True if successful, False otherwise
        """
        print(f"\n  Processing presentation: {pres_id}")

        # Load metadata
        metadata = self.load_presentation_metadata(pres_id)
        if not metadata:
            print(f"    ❌ Metadata not found")
            return False

        pres_title = metadata.get("title", pres_id)

        # Load segments
        segments = self.load_segments(pres_id)
        if not segments:
            print(f"    ❌ Segments not found")
            return False

        # Check if this is a workshop - if so, use talk-aware processing
        if self.workshop_analyzer.is_workshop(pres_id, len(segments)):
            print(f"    📚 Detected WORKSHOP ({len(segments)} segments) - using talk-aware processing")
            return self.process_workshop(pres_id, pres_title, segments)

        # Optional batching to reduce calls for regular presentations
        segments = self._batch_segments(segments)
        print(f"    Segments: {len(segments)}")

        # Update progress with total segments
        # NOTE: This load/modify/save pattern can race in multi-process scenarios.
        # If parallelizing in the future, wrap all progress updates in a single
        # locked transaction to avoid lost updates.
        progress = self.progress_tracker.load_progress()
        progress["presentations"]["total_segments"] += len(segments)
        self.progress_tracker.save_progress(progress)

        # Cache slides once
        slides_cache = self._load_all_slides(pres_id)

        # Process each segment (optionally in parallel)
        total_extractions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        topic_distribution = {}
        segments_processed = 0
        segments_failed = 0

        def _process(seg: Dict[str, Any]):
            nonlocal slides_cache
            transcript_text = seg.get("transcript", "")
            if not self._should_process_segment(transcript_text):
                # Skip processing if no cues found
                return 0, 0, 0, {}
            return self.process_segment(
                pres_id=pres_id,
                pres_title=pres_title,
                segment=seg,
                slides_cache=slides_cache
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(_process, seg): seg for seg in segments}
            for future in as_completed(future_map):
                seg = future_map[future]
                seg_idx = seg.get("segment_index", 0)
                try:
                    extractions, input_tok, output_tok, topic_dist = future.result()
                    total_extractions += extractions
                    total_input_tokens += input_tok
                    total_output_tokens += output_tok
                    segments_processed += 1
                    for topic_id, count in topic_dist.items():
                        topic_distribution[topic_id] = topic_distribution.get(topic_id, 0) + count
                except Exception as e:
                    segments_failed += 1
                    print(f"    ❌ Segment {seg_idx} failed: {e}")
                    # Record failed segment for retry
                    self.progress_tracker.record_failed_segment(
                        pres_id=pres_id,
                        segment_index=seg_idx,
                        error=str(e)
                    )

        # Batch progress and cost updates
        self.progress_tracker.update_presentation_progress(
            presentations_processed=1,
            segments_processed=segments_processed,
            segments_failed=segments_failed,
            extractions=total_extractions,
            topic_distribution=topic_distribution
        )
        self.progress_tracker.update_cost_tracking(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=self._estimate_cost(total_input_tokens, total_output_tokens)
        )

        print(f"    ✓ Total extractions: {total_extractions}")
        print(f"    ✓ Topic distribution: {topic_distribution}")

        return True

    def process_workshop(
        self,
        pres_id: str,
        pres_title: str,
        segments: List[Dict[str, Any]]
    ) -> bool:
        """
        Process a workshop with talk-aware extraction.

        Phase 1: Analyze workshop to identify individual talks
        Phase 2: Process 5-minute batches with talk context

        Args:
            pres_id: Workshop presentation ID
            pres_title: Workshop title
            segments: List of segment dicts

        Returns:
            True if successful, False otherwise
        """
        # Check for existing talks analysis
        talks_path = self.output_dir / "workshops" / f"{pres_id}_talks.json"
        talks = load_talks_json(talks_path)

        if talks is None:
            # Phase 1: Analyze workshop to identify talks
            print(f"    🔍 Phase 1: Analyzing workshop structure...")

            # Concatenate all transcripts
            full_transcript = self._concatenate_transcripts(segments)

            # Load all slides
            slides_cache = self._load_all_slides(pres_id)

            # Convert to format expected by analyzer
            slides_for_analyzer = [
                {
                    "slide_id": s["slide_id"],
                    "timestamp_sec": s["timestamp_sec"],
                    "ocr_text": s.get("slide_text", "")
                }
                for s in slides_cache
            ]

            # Analyze workshop
            analysis = self.workshop_analyzer.analyze_workshop(
                pres_id=pres_id,
                transcript=full_transcript,
                slides=slides_for_analyzer,
                title=pres_title
            )

            talks = analysis.talks
            print(f"    ✓ Identified {len(talks)} talks in workshop")

            # Save talks analysis for resume capability
            save_talks_json(
                talks=talks,
                output_path=talks_path,
                workshop_info={
                    "workshop_id": pres_id,
                    "workshop_title": pres_title,
                    "total_duration_sec": analysis.total_duration_sec,
                    "analysis_chunks": analysis.analysis_chunks
                }
            )
            print(f"    ✓ Saved talks analysis to {talks_path}")
        else:
            print(f"    ✓ Loaded existing talks analysis: {len(talks)} talks")

        # Log identified talks
        for talk in talks[:5]:  # Show first 5 talks
            print(f"      - {talk.talk_id}: {talk.title[:50]} ({talk.speaker})")
        if len(talks) > 5:
            print(f"      ... and {len(talks) - 5} more talks")

        # Phase 2: Batch segments and process with talk context
        print(f"    🎯 Phase 2: Processing with talk context...")

        # Filter out empty/silent segments (breaks, music, etc.)
        meaningful_segments = self._filter_meaningful_segments(segments)
        filtered_count = len(segments) - len(meaningful_segments)
        if filtered_count > 0:
            print(f"    📊 Filtered out {filtered_count} empty/silent segments (breaks)")

        # Batch filtered segments for workshop (dynamic 5 or 10 minute batches)
        batched_segments = self._batch_segments_for_workshop(meaningful_segments)
        print(f"    Batched segments: {len(batched_segments)} (from {len(meaningful_segments)} meaningful, {len(segments)} original)")

        # Update progress with total segments
        progress = self.progress_tracker.load_progress()
        progress["presentations"]["total_segments"] += len(batched_segments)
        self.progress_tracker.save_progress(progress)

        # Cache slides
        slides_cache = self._load_all_slides(pres_id)

        # Process each batch
        total_extractions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        topic_distribution = {}
        segments_processed = 0
        segments_failed = 0

        for batch in batched_segments:
            batch_start = batch.get("start_sec", 0)
            batch_end = batch.get("end_sec", 0)

            # Find talks overlapping this batch
            overlapping_talks = find_talks_for_batch(talks, batch_start, batch_end)

            if not overlapping_talks:
                # No talk identified for this segment - use "Unknown Talk"
                overlapping_talks = [Talk(
                    talk_id="talk_unknown",
                    title="Unidentified Segment",
                    speaker="Unknown",
                    start_sec=batch_start,
                    end_sec=batch_end,
                    description=""
                )]

            # Process batch for each overlapping talk (duplicate extraction)
            for talk in overlapping_talks:
                batch_idx = batch.get("segment_index", 0)
                try:
                    extractions, input_tok, output_tok, topic_dist = self.process_workshop_segment(
                        workshop_id=pres_id,
                        workshop_title=pres_title,
                        segment=batch,
                        talk=talk,
                        slides_cache=slides_cache
                    )

                    total_extractions += extractions
                    total_input_tokens += input_tok
                    total_output_tokens += output_tok
                    for tid, cnt in topic_dist.items():
                        topic_distribution[tid] = topic_distribution.get(tid, 0) + cnt

                    segments_processed += 1

                except Exception as e:
                    print(f"    ❌ Failed to process batch {batch_idx} for {talk.talk_id}: {e}")
                    segments_failed += 1
                    # Record failed segment for retry
                    self.progress_tracker.record_failed_segment(
                        pres_id=pres_id,
                        segment_index=batch_idx,
                        error=f"{talk.talk_id}: {str(e)}"
                    )

        # Update progress
        self.progress_tracker.update_presentation_progress(
            presentations_processed=1,
            segments_processed=segments_processed,
            segments_failed=segments_failed,
            extractions=total_extractions,
            topic_distribution=topic_distribution
        )
        self.progress_tracker.update_cost_tracking(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=self._estimate_cost(total_input_tokens, total_output_tokens)
        )

        print(f"    ✓ Workshop total extractions: {total_extractions}")
        print(f"    ✓ Topic distribution: {topic_distribution}")

        return True

    def _concatenate_transcripts(self, segments: List[Dict[str, Any]]) -> str:
        """Concatenate all segment transcripts with timestamps.

        Args:
            segments: List of segment dicts

        Returns:
            Full transcript with timestamp markers
        """
        lines = []
        for seg in segments:
            start_sec = seg.get("start_sec", 0)
            start_hms = seconds_to_hms(start_sec)
            transcript = seg.get("transcript", "")
            lines.append(f"[{start_hms}] {transcript}")

        return "\n\n".join(lines)

    def _filter_meaningful_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter out empty/silent segments (breaks, music, etc.).

        Removes segments with minimal content to avoid wasting API calls
        on silent periods during workshop breaks.

        Args:
            segments: List of segment dicts

        Returns:
            Filtered list with only meaningful segments
        """
        MIN_TEXT_LENGTH = 20  # Minimum characters for meaningful content
        MIN_WORDS = 3  # Minimum word count

        filtered = []
        for seg in segments:
            transcript = seg.get("transcript", "").strip()

            # Skip segments with too little text
            if len(transcript) < MIN_TEXT_LENGTH:
                continue

            # Skip segments with too few words
            words = transcript.split()
            if len(words) < MIN_WORDS:
                continue

            # Skip segments that are just noise patterns
            noise_patterns = ["you", "okay", "um", "uh", "hmm"]
            lower_text = transcript.lower()
            if lower_text in noise_patterns or all(w.lower() in noise_patterns for w in words):
                continue

            filtered.append(seg)

        return filtered

    def _batch_segments_for_workshop(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Batch segments for workshop processing with dynamic duration.

        Uses 3-minute batches for workshops <= 4 hours (better timestamp precision).
        Uses 5-minute batches for workshops > 4 hours.

        Batched segments include segment_index_start and segment_index_end to track
        the range of original segments that were combined.

        Args:
            segments: List of segment dicts

        Returns:
            List of batched segment dicts
        """
        if not segments:
            return []

        # Calculate total workshop duration
        total_duration = segments[-1].get("end_sec", 0) - segments[0].get("start_sec", 0)

        # Dynamic batch duration: 5 min for long workshops, 3 min otherwise
        # Shorter batches improve timestamp precision for extraction
        FOUR_HOURS = 4 * 3600  # 14,400 seconds
        if total_duration > FOUR_HOURS:
            workshop_batch_duration = 300  # 5 minutes
            print(f"    📊 Long workshop ({total_duration/3600:.1f}h) - using 5-minute batches")
        else:
            workshop_batch_duration = 180  # 3 minutes

        batched: List[Dict[str, Any]] = []
        current = None

        for seg in segments:
            seg_index = seg.get("segment_index", 0)

            if current is None:
                current = seg.copy()
                # Track segment range for QA validation
                current["segment_index_start"] = seg_index
                current["segment_index_end"] = seg_index
                continue

            combined_duration = seg.get("end_sec", 0) - current.get("start_sec", 0)

            if combined_duration <= workshop_batch_duration:
                # Merge into current
                current["end_sec"] = max(current.get("end_sec", 0), seg.get("end_sec", 0))
                current["transcript"] = current.get("transcript", "") + "\n\n" + seg.get("transcript", "")
                # Update segment range end
                current["segment_index_end"] = seg_index
            else:
                batched.append(current)
                current = seg.copy()
                # Track segment range for new batch
                current["segment_index_start"] = seg_index
                current["segment_index_end"] = seg_index

        if current is not None:
            batched.append(current)

        return batched

    def process_workshop_segment(
        self,
        workshop_id: str,
        workshop_title: str,
        segment: Dict[str, Any],
        talk: Talk,
        slides_cache: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[int, int, int, Dict[str, int]]:
        """
        Process a single workshop segment with talk context.

        Args:
            workshop_id: Workshop presentation ID
            workshop_title: Workshop title
            segment: Segment dict
            talk: Talk that this segment belongs to
            slides_cache: Cached slides list

        Returns:
            Tuple of (extractions_count, input_tokens, output_tokens, topic_distribution)
        """
        segment_index = segment.get("segment_index", 0)
        segment_start_sec = segment.get("start_sec", 0)
        segment_end_sec = segment.get("end_sec", 0)
        segment_duration_sec = segment_end_sec - segment_start_sec
        transcript_text = segment.get("transcript", "")

        # Convert to HH:MM:SS
        segment_start_hms = seconds_to_hms(segment_start_sec)
        segment_end_hms = seconds_to_hms(segment_end_sec)

        # Load slides for this segment unless slide OCR is being held for the
        # separate presentation enrichment lane.
        slides = (
            self.load_slides_for_segment(workshop_id, segment_start_sec, segment_end_sec, slides_cache)
            if include_slides_for_extraction()
            else []
        )
        slides_list = format_slides_list(slides)

        # Build workshop-specific extraction prompt
        prompt = build_workshop_extraction_prompt(
            workshop_title=workshop_title,
            talk_id=talk.talk_id,
            talk_title=talk.title,
            talk_speaker=talk.speaker,
            segment_start_hms=segment_start_hms,
            segment_end_hms=segment_end_hms,
            segment_duration_sec=segment_duration_sec,
            transcript_text=transcript_text,
            slides_list_with_ocr=slides_list,
            topic_guardrails=self.topic_guardrail_text
        )

        print(f"    [{talk.talk_id}] {segment_start_hms} - {segment_end_hms} ({segment_duration_sec:.0f}s)")

        # Run extraction
        try:
            parsed, model_used, input_tokens, output_tokens = self.extraction_runner.run_extraction(prompt)
        except Exception as e:
            print(f"      ❌ Extraction failed: {e}")
            return 0, 0, 0, {}

        # Parse extractions
        extraction_list = parsed.get("extractions", [])

        # Pre-filter: skip extractions with invalid extraction_type or timestamps
        # Models sometimes return "none" or other invalid values to signal "no content"
        VALID_EXTRACTION_TYPES = {"verbatim_quote", "summary"}
        filtered_list = []
        invalid_type_count = 0
        invalid_ts_count = 0
        for ext in extraction_list:
            ext_type = ext.get("extraction_type", "summary")
            if ext_type not in VALID_EXTRACTION_TYPES:
                invalid_type_count += 1
                continue
            # Also skip if timestamps are equal (invalid range)
            ts_start = ext.get("timestamp_start_sec")
            ts_end = ext.get("timestamp_end_sec")
            if ts_start is not None and ts_end is not None and ts_start >= ts_end:
                invalid_ts_count += 1
                continue
            filtered_list.append(ext)

        skipped_count = invalid_type_count + invalid_ts_count
        if skipped_count > 0:
            print(f"      ⚠️  Pre-filtered {skipped_count} invalid extractions (type={invalid_type_count}, timestamps={invalid_ts_count})")
            # Track stats for analysis
            self.progress_tracker.update_pre_filter_stats(
                invalid_extraction_type=invalid_type_count,
                invalid_timestamps=invalid_ts_count
            )

        extraction_list = filtered_list
        print(f"      Model: {model_used}, Extractions: {len(extraction_list)}")

        # Process each extraction
        valid_extractions = 0
        topic_distribution = {}

        for extraction_data in extraction_list:
            try:
                # Build ContentExtraction object with workshop/talk fields (with anchor-based quote extraction)
                extraction = self._build_workshop_extraction(
                    extraction_data=extraction_data,
                    workshop_id=workshop_id,
                    workshop_title=workshop_title,
                    talk=talk,
                    segment_index=segment_index,
                    segment_start_sec=segment_start_sec,
                    segment_end_sec=segment_end_sec,
                    model_used=model_used,
                    source_text=transcript_text
                )

                # Hydrate slide paths
                if extraction.slide_references:
                    extraction.slide_references = self._hydrate_slide_paths(
                        extraction.slide_references, slides_cache or []
                    )

                # QA validation
                qa_results = self.qa_validator.validate_extraction(extraction, transcript_text)

                # Check if validation passed
                all_valid = True
                failure_reasons = []

                for check_name, result in qa_results.items():
                    if check_name == "slide_checks":
                        for slide_result in result:
                            if not slide_result.valid:
                                all_valid = False
                                failure_reasons.append(
                                    f"{check_name}: {slide_result.reason} (severity: {slide_result.severity})"
                                )
                    else:
                        if not result.valid:
                            all_valid = False
                            failure_reasons.append(
                                f"{check_name}: {result.reason} (severity: {result.severity})"
                            )

                if all_valid:
                    extraction.qa_validated = True
                    valid_extractions += 1

                    # Update topic distribution
                    topic_id = extraction.topic_id
                    topic_distribution[topic_id] = topic_distribution.get(topic_id, 0) + 1

                    # Store extraction
                    self.storage.append_extraction(extraction, topic_id)
                else:
                    print(f"      ⚠️  QA failed for extraction: {extraction.extraction_id}")
                    for reason in failure_reasons[:3]:
                        print(f"         - {reason}")
                    if len(failure_reasons) > 3:
                        print(f"         ... and {len(failure_reasons) - 3} more")

                    # Record QA failure for later review
                    self.progress_tracker.record_qa_failure(
                        extraction_id=extraction.extraction_id,
                        pres_id=workshop_id,
                        segment_index=segment_index,
                        reasons=failure_reasons,
                        extraction_data=extraction.model_dump()
                    )

            except Exception as e:
                print(f"      ❌ Failed to process extraction: {e}")
                continue

        print(f"      ✓ Valid extractions: {valid_extractions}/{len(extraction_list)}")

        return valid_extractions, input_tokens, output_tokens, topic_distribution

    def _build_workshop_extraction(
        self,
        extraction_data: Dict[str, Any],
        workshop_id: str,
        workshop_title: str,
        talk: Talk,
        segment_index: Optional[int],
        segment_start_sec: Optional[float],
        segment_end_sec: Optional[float],
        model_used: str,
        source_text: Optional[str] = None
    ) -> ContentExtraction:
        """
        Build ContentExtraction object from workshop extraction data.

        Args:
            extraction_data: Raw extraction dict from model
            workshop_id: Workshop ID
            workshop_title: Workshop title
            talk: Talk that this extraction belongs to
            segment_index: Segment index
            segment_start_sec: Segment start
            segment_end_sec: Segment end
            model_used: Model ID
            source_text: Original transcript text for anchor-based quote extraction

        Returns:
            ContentExtraction object
        """
        # Validate required fields
        required_fields = ["topic_id", "summary"]
        missing_fields = [f for f in required_fields if f not in extraction_data]
        if missing_fields:
            raise KeyError(
                f"Missing required fields in extraction_data: {', '.join(missing_fields)}. "
                f"Available fields: {', '.join(extraction_data.keys())}"
            )

        topic_id = extraction_data["topic_id"]
        topic_name = extraction_data.get("topic_name")
        if not topic_name:
            raise ValueError(
                f"topic_name is required but missing or empty in extraction_data. "
                f"topic_id={topic_id}, workshop_id={workshop_id}"
            )

        summary = extraction_data["summary"]

        # Handle verbatim quote extraction from anchors
        verbatim_quote = extraction_data.get("verbatim_quote")
        quote_start_anchor = extraction_data.get("quote_start_anchor")
        quote_end_anchor = extraction_data.get("quote_end_anchor")

        # If we have anchors and source text, extract the quote
        if quote_start_anchor and quote_end_anchor and source_text:
            # Use GPT-5.2 fallback for direct quote finding when anchor matching fails
            quote_result = extract_quote_with_gpt52_fallback(
                source_text=source_text,
                start_anchor=quote_start_anchor,
                end_anchor=quote_end_anchor,
                summary=summary,
                topic_name=extraction_data.get("topic_name"),
                timestamp_start_sec=extraction_data.get("timestamp_start_sec"),
                timestamp_end_sec=extraction_data.get("timestamp_end_sec"),
                relevance_score=extraction_data.get("relevance_score")
            )

            if quote_result.success:
                verbatim_quote = quote_result.verbatim_quote
                if quote_result.method == "gpt52_fallback":
                    print(f"      📎 Quote extracted via GPT-5.2 fallback")
            else:
                # Both anchor matching and GPT-5.2 failed - save summary without quote
                verbatim_quote = None
                print(f"      ⚠️  Quote extraction failed: {quote_result.error}")
                print(f"      💾 Saving extraction with summary only (no verbatim quote)")

        # Compute content hash (using talk_id in source to differentiate)
        source_id = f"{workshop_id}_{talk.talk_id}"
        content_hash = ContentExtraction.compute_content_hash(
            source_id=source_id,
            segment_index=segment_index,
            topic_id=topic_id,
            verbatim_quote=verbatim_quote,
            summary=summary
        )

        # Build extraction ID
        extraction_id = f"{workshop_id}_{talk.talk_id}_{topic_id}_{content_hash[:16]}"

        # Build source reference: "workshop-name/talk_001-Talk Title:HH:MM:SS"
        timestamp_start = extraction_data.get("timestamp_start", "00:00:00")
        source_reference = f"{workshop_id}/{talk.talk_id}-{talk.title}:{timestamp_start}"

        # Build extraction object
        extraction_dict = {
            "extraction_id": extraction_id,
            "content_hash": content_hash,
            "source_type": "presentation",
            "source_id": workshop_id,
            "source_title": workshop_title,
            "topic_id": topic_id,
            "topic_name": topic_name,

            # Workshop/talk fields
            "workshop_id": workshop_id,
            "talk_id": talk.talk_id,
            "talk_title": talk.title,
            "talk_speaker": talk.speaker,
            "source_reference": source_reference,

            # Segment info
            "segment_index": segment_index,
            "segment_start_sec": segment_start_sec,
            "segment_end_sec": segment_end_sec,

            # Timestamps
            "timestamp_start_sec": extraction_data.get("timestamp_start_sec"),
            "timestamp_end_sec": extraction_data.get("timestamp_end_sec"),
            "timestamp_start": extraction_data.get("timestamp_start"),
            "timestamp_end": extraction_data.get("timestamp_end"),

            # Content
            "extraction_type": extraction_data.get("extraction_type", "summary"),
            "verbatim_quote": verbatim_quote,
            "summary": summary,

            # Slide references
            "slide_references": extraction_data.get("slide_references", []),

            # Quality metadata
            "relevance_score": extraction_data.get("relevance_score", 0.5),
            "transcript_quality": extraction_data.get("transcript_quality"),
            "model_used": model_used,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }

        return ContentExtraction(**extraction_dict)

    def _hydrate_slide_paths(
        self,
        slide_refs: List[SlideReference],
        slides_cache: List[Dict[str, Any]]
    ) -> List[SlideReference]:
        """
        Hydrate missing slide_path fields from slides cache.

        Args:
            slide_refs: List of SlideReference objects
            slides_cache: Cached slides list with paths

        Returns:
            List of SlideReference objects with hydrated paths
        """
        # Build lookup by slide_id
        path_lookup = {s["slide_id"]: s.get("slide_path", "") for s in slides_cache}

        for ref in slide_refs:
            if not ref.slide_path:
                ref.slide_path = path_lookup.get(ref.slide_id, f"{ref.slide_id}.json")

        return slide_refs

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD from the canonical conference pricing registry."""

        if estimate_text_model_cost is None:
            return 0.0
        return estimate_text_model_cost(
            DEFAULT_EXTRACTION_COST_MODEL,
            input_tokens,
            output_tokens,
        )

    def retry_failed_segment(
        self,
        pres_id: str,
        segment_index: int
    ) -> bool:
        """
        Retry a specific failed segment.

        Args:
            pres_id: Presentation ID
            segment_index: Segment index to retry

        Returns:
            True if successful, False otherwise
        """
        print(f"  Retrying {pres_id} segment {segment_index}...")

        # Load metadata
        metadata = self.load_presentation_metadata(pres_id)
        if not metadata:
            print(f"    ❌ Metadata not found for {pres_id}")
            return False

        pres_title = metadata.get("title", pres_id)

        # Load all segments
        segments = self.load_segments(pres_id)
        if not segments:
            print(f"    ❌ Segments not found for {pres_id}")
            return False

        # Find the specific segment
        target_segment = None
        for seg in segments:
            if seg.get("segment_index") == segment_index:
                target_segment = seg
                break

        if target_segment is None:
            print(f"    ❌ Segment {segment_index} not found in {pres_id}")
            return False

        # Cache slides
        slides_cache = self._load_all_slides(pres_id)

        # Check if this is a workshop - if so, may need different processing
        if self.workshop_analyzer.is_workshop(pres_id, len(segments)):
            # For workshops, we'd need to reload talk context
            # For now, just process as a regular segment
            print(f"    ⚠️  Workshop segment - processing without talk context")

        # Process the segment
        try:
            extractions, input_tok, output_tok, topic_dist = self.process_segment(
                pres_id=pres_id,
                pres_title=pres_title,
                segment=target_segment,
                slides_cache=slides_cache
            )

            # Update cost tracking
            self.progress_tracker.update_cost_tracking(
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost_usd=self._estimate_cost(input_tok, output_tok)
            )

            # Clear from failed list on success
            if extractions > 0 or (input_tok > 0):  # Consider success if we got a response
                self.progress_tracker.clear_failed_segment(pres_id, segment_index)
                print(f"    ✓ Retry successful: {extractions} extractions")
                return True
            else:
                print(f"    ⚠️  Retry completed but no extractions")
                return True  # Still clear it - the segment was processed

        except Exception as e:
            print(f"    ❌ Retry failed: {e}")
            # Update the error message in failed segments
            self.progress_tracker.record_failed_segment(
                pres_id=pres_id,
                segment_index=segment_index,
                error=f"Retry failed: {str(e)}"
            )
            return False


class PaperProcessor:
    """Process paper batches with extraction."""

    def __init__(
        self,
        conf_base_dir: Path,
        output_dir: Path,
        storage: QueuedTopicStorage,
        progress_tracker: ProgressTracker,
        qa_validator: PaperQAValidator
    ):
        """
        Initialize paper processor.

        Args:
            conf_base_dir: Path to conference base directory
            output_dir: Path to output directory
            storage: Storage manager
            progress_tracker: Progress tracker
            qa_validator: QA validator
        """
        self.conf_base_dir = Path(conf_base_dir)
        self.output_dir = Path(output_dir)
        self.storage = storage
        self.progress_tracker = progress_tracker
        self.qa_validator = qa_validator
        self.extraction_runner = ExtractionRunner()

    def load_paper_metadata(self, paper_id: str) -> Optional[Dict[str, Any]]:
        """
        Load paper metadata.

        Args:
            paper_id: Paper ID

        Returns:
            Metadata dict or None if not found
        """
        papers_file = self.conf_base_dir / "data" / "papers.json"
        if not papers_file.exists():
            return None

        # Load papers.json and find the paper by ID
        with open(papers_file) as f:
            data = json.load(f)

        papers = data.get("papers", [])
        for paper in papers:
            if paper.get("id") == paper_id:
                return paper

        return None

    def process_paper(self, paper_id: str) -> Tuple[int, int, int]:
        """
        Process a single paper.

        Args:
            paper_id: Paper ID

        Returns:
            Tuple of (extractions_count, input_tokens, output_tokens)
        """
        # Load metadata
        metadata = self.load_paper_metadata(paper_id)
        if not metadata:
            print(f"    ❌ Metadata not found for {paper_id}")
            return 0, 0, 0

        paper_title = metadata.get("title", paper_id)
        paper_abstract = metadata.get("abstract", "")

        if not paper_abstract:
            print(f"    ⚠️  No abstract for {paper_id}")
            return 0, 0, 0

        # Build extraction prompt
        prompt = build_paper_extraction_prompt(
            paper_title=paper_title,
            paper_abstract=paper_abstract,
            paper_id=paper_id
        )

        # Run extraction
        try:
            parsed, model_used, input_tokens, output_tokens = self.extraction_runner.run_extraction(prompt)
        except Exception as e:
            print(f"    ❌ Extraction failed for {paper_id}: {e}")
            return 0, 0, 0

        # Parse extractions
        extraction_list = parsed.get("extractions", [])

        # Process each extraction
        valid_extractions = 0

        for extraction_data in extraction_list:
            try:
                # Build ContentExtraction object (with anchor-based quote extraction)
                extraction = self._build_extraction(
                    extraction_data=extraction_data,
                    source_id=paper_id,
                    source_title=paper_title,
                    model_used=model_used,
                    source_text=paper_abstract
                )

                # Store extraction (QA validation can be done in batch later)
                extraction.qa_validated = False
                self.storage.append_extraction(extraction, extraction.topic_id)
                valid_extractions += 1

            except Exception as e:
                print(f"      ❌ Failed to process extraction: {e}")
                continue

        return valid_extractions, input_tokens, output_tokens

    def _build_extraction(
        self,
        extraction_data: Dict[str, Any],
        source_id: str,
        source_title: str,
        model_used: str,
        source_text: Optional[str] = None
    ) -> ContentExtraction:
        """
        Build ContentExtraction object from paper extraction data.

        Args:
            extraction_data: Raw extraction dict from model
            source_id: Paper ID
            source_title: Paper title
            model_used: Model ID
            source_text: Paper abstract text for anchor-based quote extraction

        Returns:
            ContentExtraction object
        """
        topic_id = extraction_data["topic_id"]
        summary = extraction_data["summary"]

        # Handle verbatim quote extraction from anchors
        verbatim_quote = extraction_data.get("verbatim_quote")
        quote_start_anchor = extraction_data.get("quote_start_anchor")
        quote_end_anchor = extraction_data.get("quote_end_anchor")

        # If we have anchors and source text, extract the quote
        if quote_start_anchor and quote_end_anchor and source_text:
            quote_result = extract_quote_from_anchors(
                source_text=source_text,
                start_anchor=quote_start_anchor,
                end_anchor=quote_end_anchor
            )

            if quote_result.success:
                verbatim_quote = quote_result.verbatim_quote
            else:
                # For papers, skip LLM fallback (too expensive for batch)
                # Just use summary as the extraction without verbatim quote
                verbatim_quote = None
                print(f"      ⚠️  Anchor extraction failed for paper, using summary only")

        # Compute content hash
        content_hash = ContentExtraction.compute_content_hash(
            source_id=source_id,
            segment_index=None,
            topic_id=topic_id,
            verbatim_quote=verbatim_quote,
            summary=summary
        )

        # Build extraction ID
        extraction_id = f"{source_id}_{topic_id}_{content_hash[:16]}"

        # Build extraction object
        extraction_dict = {
            "extraction_id": extraction_id,
            "content_hash": content_hash,
            "source_type": "paper",
            "source_id": source_id,
            "source_title": source_title,
            "source_coverage": extraction_data.get("source_coverage", "abstract_only"),
            "topic_id": topic_id,
            "topic_name": extraction_data["topic_name"],
            "extraction_type": extraction_data.get("extraction_type", "summary"),
            "verbatim_quote": verbatim_quote,
            "summary": summary,
            "relevance_score": extraction_data.get("relevance_score", 0.5),
            "model_used": model_used,
            "extracted_at": datetime.now(timezone.utc).isoformat()
        }

        return ContentExtraction(**extraction_dict)

    def process_paper_mini_batch(
        self,
        papers_data: List[Dict[str, Any]],
        mini_batch_id: int
    ) -> Tuple[int, int, int, int, int]:
        """
        Process multiple papers in a single API call.

        Args:
            papers_data: List of paper dicts with paper_id, paper_title, paper_abstract
            mini_batch_id: Identifier for this mini-batch

        Returns:
            Tuple of (extractions_count, input_tokens, output_tokens, processed, failed)
        """
        if not papers_data:
            return 0, 0, 0, 0, 0

        # Build batched prompt
        prompt = build_paper_batch_extraction_prompt(papers_data)

        # Run extraction
        try:
            parsed, model_used, input_tokens, output_tokens = self.extraction_runner.run_extraction(prompt)
        except Exception as e:
            print(f"      ❌ Mini-batch {mini_batch_id} extraction failed: {e}")
            return 0, 0, 0, 0, len(papers_data)

        # Parse extractions - now grouped by paper_id
        extraction_list = parsed.get("extractions", [])

        # Create lookup for paper data by ID
        papers_by_id = {p["paper_id"]: p for p in papers_data}

        # Process each extraction
        valid_extractions = 0
        papers_with_extractions = set()

        for extraction_data in extraction_list:
            try:
                paper_id = extraction_data.get("paper_id")
                if not paper_id or paper_id not in papers_by_id:
                    continue

                paper_info = papers_by_id[paper_id]

                # Build ContentExtraction object
                extraction = self._build_extraction(
                    extraction_data=extraction_data,
                    source_id=paper_id,
                    source_title=paper_info["paper_title"],
                    model_used=model_used,
                    source_text=paper_info["paper_abstract"]
                )

                # Store extraction
                extraction.qa_validated = False
                self.storage.append_extraction(extraction, extraction.topic_id)
                valid_extractions += 1
                papers_with_extractions.add(paper_id)

            except Exception as e:
                # Silently skip individual extraction errors
                continue

        processed = len(papers_data)
        return valid_extractions, input_tokens, output_tokens, processed, 0

    def process_batch(
        self,
        paper_ids: List[str],
        batch_number: int,
        mini_batch_size: int = 10,
        max_workers: int = 3
    ) -> bool:
        """
        Process a batch of papers using mini-batches and parallel execution.

        Args:
            paper_ids: List of paper IDs
            batch_number: Batch number
            mini_batch_size: Number of papers per API call (default: 10)
            max_workers: Number of parallel workers (default: 3)

        Returns:
            True if successful
        """
        print(f"\n  Processing batch {batch_number}: {len(paper_ids)} papers ({mini_batch_size} per API call, {max_workers} workers)")

        # Load all paper metadata first
        papers_data = []
        for paper_id in paper_ids:
            metadata = self.load_paper_metadata(paper_id)
            if metadata and metadata.get("abstract"):
                papers_data.append({
                    "paper_id": paper_id,
                    "paper_title": metadata.get("title", paper_id),
                    "paper_abstract": metadata.get("abstract", "")
                })

        if not papers_data:
            print(f"    ⚠️  No papers with abstracts found")
            return True

        # Split into mini-batches
        mini_batches = []
        for i in range(0, len(papers_data), mini_batch_size):
            mini_batches.append(papers_data[i:i + mini_batch_size])

        print(f"    📦 Split into {len(mini_batches)} mini-batches")

        total_extractions = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_processed = 0
        total_failed = 0

        # Process mini-batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for idx, mini_batch in enumerate(mini_batches):
                future = executor.submit(
                    self.process_paper_mini_batch,
                    mini_batch,
                    idx
                )
                futures[future] = idx

            for future in as_completed(futures):
                mini_batch_idx = futures[future]
                try:
                    extractions, in_tok, out_tok, processed, failed = future.result()
                    total_extractions += extractions
                    total_input_tokens += in_tok
                    total_output_tokens += out_tok
                    total_processed += processed
                    total_failed += failed

                    if extractions > 0:
                        print(f"    ✓ Mini-batch {mini_batch_idx}: {extractions} extractions from {processed} papers")

                except Exception as e:
                    print(f"    ❌ Mini-batch {mini_batch_idx} failed: {e}")
                    total_failed += len(mini_batches[mini_batch_idx])

        # Update progress
        cost_usd = self._estimate_cost(total_input_tokens, total_output_tokens)

        self.progress_tracker.update_paper_progress(
            processed=total_processed,
            failed=total_failed,
            extractions=total_extractions,
            last_batch=batch_number
        )

        self.progress_tracker.update_cost_tracking(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=cost_usd
        )

        print(f"    ✓ Batch {batch_number}: {total_extractions} total extractions")
        print(f"    ✓ Processed: {total_processed}, Failed: {total_failed}")

        return True

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD from the canonical conference pricing registry."""

        if estimate_text_model_cost is None:
            return 0.0
        return estimate_text_model_cost(
            DEFAULT_EXTRACTION_COST_MODEL,
            input_tokens,
            output_tokens,
        )
