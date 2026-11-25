"""Pydantic schemas for content extraction with strict timestamp validation."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SlideReference(BaseModel):
    """Reference to a slide with OCR confidence and timestamp alignment.

    All fields except slide_id are optional since model output may not include them.
    Missing paths can be hydrated later via processor._hydrate_slide_paths().
    """

    slide_id: str = Field(..., description="Slide identifier (e.g., 'slide_042')")
    slide_path: Optional[str] = Field(None, description="Path to slide JSON file")
    slide_timestamp_sec: Optional[float] = Field(None, description="When slide was shown (seconds)")
    ocr_confidence: Optional[Literal["missing", "low", "medium", "high"]] = Field(
        None,
        description="OCR quality assessment"
    )
    slide_content_summary: Optional[str] = Field(
        None,
        description="Brief description of slide content"
    )
    timestamp_aligned: bool = Field(
        default=False,
        description="Whether slide timestamp is within segment bounds"
    )


class ContentExtraction(BaseModel):
    """
    A single content extraction with strict validation.

    IMPORTANT: Out-of-bounds timestamps cause ValidationError (no auto-clipping).
    """

    extraction_id: str = Field(..., description="Unique extraction identifier")
    content_hash: str = Field(..., description="SHA256 hash for deduplication")

    # Source information
    source_type: Literal["presentation", "paper"] = Field(..., description="Content source type")
    source_id: str = Field(..., description="Presentation or paper ID")
    source_title: str = Field(..., description="Full title")
    source_coverage: Optional[Literal["abstract_only", "full_text"]] = Field(
        None,
        description="Coverage level for papers"
    )

    # Segment information (for presentations)
    segment_index: Optional[int] = Field(None, description="Segment number")
    segment_start_sec: Optional[float] = Field(None, description="Segment start time")
    segment_end_sec: Optional[float] = Field(None, description="Segment end time")

    # Topic classification
    topic_id: str = Field(..., description="Topic ID (01-20)")
    topic_name: str = Field(..., description="Topic name")

    # Workshop/talk fields (optional, only for workshop presentations)
    workshop_id: Optional[str] = Field(None, description="Workshop ID if from workshop")
    talk_id: Optional[str] = Field(None, description="Talk ID within workshop (e.g., 'talk_001')")
    talk_title: Optional[str] = Field(None, description="Human-readable talk title")
    talk_speaker: Optional[str] = Field(None, description="Speaker name if known")
    source_reference: Optional[str] = Field(
        None,
        description="Formatted reference (e.g., 'workshop-name/talk_001-Talk Title:HH:MM:SS')"
    )

    # Timestamps (for presentations)
    timestamp_start_sec: Optional[float] = Field(None, description="Extraction start (seconds)")
    timestamp_end_sec: Optional[float] = Field(None, description="Extraction end (seconds)")
    timestamp_start: Optional[str] = Field(None, description="Start time (HH:MM:SS)")
    timestamp_end: Optional[str] = Field(None, description="End time (HH:MM:SS)")

    # Content
    extraction_type: Literal["verbatim_quote", "summary"] = Field(
        ...,
        description="Type of extraction"
    )
    verbatim_quote: Optional[str] = Field(None, description="Exact quote from transcript")
    summary: str = Field(..., description="Summary of key point")

    # Slide references (for presentations)
    slide_references: List[SlideReference] = Field(
        default_factory=list,
        description="Referenced slides"
    )

    # Quality metadata
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    transcript_quality: Optional[Literal["low", "medium", "high"]] = Field(
        None,
        description="Transcript quality assessment"
    )

    # Processing metadata
    model_used: str = Field(..., description="Model ID used for extraction")
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Extraction timestamp (ISO 8601, UTC)"
    )
    qa_validated: bool = Field(default=False, description="Whether QA validation passed")

    @model_validator(mode='after')
    def validate_timestamps(self):
        """Reject missing, out-of-bounds, or inverted presentation timestamps."""
        if self.source_type != "presentation":
            return self

        # Require timestamps/segment bounds for presentations
        required_fields = {
            "segment_start_sec": self.segment_start_sec,
            "segment_end_sec": self.segment_end_sec,
            "timestamp_start_sec": self.timestamp_start_sec,
            "timestamp_end_sec": self.timestamp_end_sec,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(
                f"Missing required fields for presentation: {', '.join(missing)}"
            )

        # Validate HH:MM:SS format
        if self.timestamp_start:
            self._validate_time_format(self.timestamp_start, "timestamp_start")
        if self.timestamp_end:
            self._validate_time_format(self.timestamp_end, "timestamp_end")

        if self.timestamp_start_sec < self.segment_start_sec:
            raise ValueError(
                f"timestamp_start_sec ({self.timestamp_start_sec}) before "
                f"segment_start_sec ({self.segment_start_sec})"
            )

        if self.timestamp_end_sec > self.segment_end_sec:
            raise ValueError(
                f"timestamp_end_sec ({self.timestamp_end_sec}) after "
                f"segment_end_sec ({self.segment_end_sec})"
            )

        # Validate start <= end.
        if (self.timestamp_start_sec is not None and
            self.timestamp_end_sec is not None):
            if self.timestamp_start_sec > self.timestamp_end_sec:
                raise ValueError(
                    f"timestamp_start_sec ({self.timestamp_start_sec}) > "
                    f"timestamp_end_sec ({self.timestamp_end_sec})"
                )

        for slide_ref in self.slide_references:
            if slide_ref.slide_timestamp_sec is None or not slide_ref.timestamp_aligned:
                continue
            if not (
                self.segment_start_sec
                <= slide_ref.slide_timestamp_sec
                <= self.segment_end_sec
            ):
                raise ValueError(
                    f"slide {slide_ref.slide_id} timestamp "
                    f"({slide_ref.slide_timestamp_sec}) outside segment bounds "
                    f"[{self.segment_start_sec}, {self.segment_end_sec}]"
                )

        return self

    @staticmethod
    def _validate_time_format(time_str: str, field_name: str):
        """Validate HH:MM:SS format."""
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"{field_name} must be HH:MM:SS format, got: {time_str}")
        try:
            hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
            if not (0 <= hours <= 23 and 0 <= mins <= 59 and 0 <= secs <= 59):
                raise ValueError(f"Invalid time values in {field_name}: {time_str}")
        except ValueError as e:
            raise ValueError(f"Invalid time format for {field_name}: {time_str}") from e

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS to seconds."""
        parts = time_str.split(':')
        hours, mins, secs = int(parts[0]), int(parts[1]), int(parts[2])
        return hours * 3600 + mins * 60 + secs

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        total_secs = int(seconds)
        hours = total_secs // 3600
        mins = (total_secs % 3600) // 60
        secs = total_secs % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    @staticmethod
    def compute_content_hash(
        source_id: str,
        segment_index: Optional[int],
        topic_id: str,
        verbatim_quote: Optional[str],
        summary: str
    ) -> str:
        """
        Compute full SHA256 content hash for deduplication.

        CRITICAL: Use FULL hash (64 chars), not truncated.
        Truncation = collision risk with 100K+ extractions.
        """
        # Use verbatim_quote if available, otherwise summary
        content = verbatim_quote or summary

        # Hash: source_id + segment_index + topic_id + content
        key = f"{source_id}_{segment_index}_{topic_id}_{content}"
        full_hash = hashlib.sha256(key.encode()).hexdigest()
        return f"sha256:{full_hash}"  # 71 chars total (7 + 64)

    @field_validator('content_hash')
    @classmethod
    def validate_content_hash(cls, v: str) -> str:
        """Validate hash format."""
        if not v.startswith('sha256:'):
            raise ValueError("content_hash must start with 'sha256:'")
        hash_part = v[7:]
        if len(hash_part) != 64:
            raise ValueError(
                f"content_hash must be full SHA256 (64 hex chars), got {len(hash_part)}"
            )
        return v


class ExtractionBatch(BaseModel):
    """Batch of extractions returned by model."""

    extractions: List[ContentExtraction] = Field(
        default_factory=list,
        description="List of content extractions"
    )

    model_used: str = Field(..., description="Model ID")
    processing_time_sec: Optional[float] = Field(None, description="Processing time")
    tokens_input: Optional[int] = Field(None, description="Input tokens")
    tokens_output: Optional[int] = Field(None, description="Output tokens")
