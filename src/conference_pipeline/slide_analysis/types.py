"""Typed result shapes for v1.61 slide analysis."""

from __future__ import annotations

from dataclasses import dataclass, field


_RAW_OUTPUT_TRUNCATION = 1024


@dataclass(frozen=True)
class SlideFigureRef:
    """A reference to a figure on the slide."""

    caption: str
    modality: str  # "chart" | "diagram" | "image" | "table" | "code" | "equation" | "other"
    axes: tuple[str, ...] | None = None  # for charts: ordered axis labels


@dataclass(frozen=True)
class SlideAnalysisResult:
    """Structured analysis of one slide image.

    Fields are intentionally tuple-of-string rather than list-of-string so the
    object is hashable and safe to share between threads / cache.
    """

    salient_claims: tuple[str, ...]
    figures: tuple[SlideFigureRef, ...]
    code_blocks: tuple[str, ...]
    equations: tuple[str, ...]
    headings: tuple[str, ...]
    body_text: str
    reading_order_hint: str
    confidence: float
    raw_model_output: str = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "salient_claims": list(self.salient_claims),
            "figures": [
                {
                    "caption": f.caption,
                    "modality": f.modality,
                    "axes": list(f.axes) if f.axes else None,
                }
                for f in self.figures
            ],
            "code_blocks": list(self.code_blocks),
            "equations": list(self.equations),
            "headings": list(self.headings),
            "body_text": self.body_text,
            "reading_order_hint": self.reading_order_hint,
            "confidence": self.confidence,
        }


class SlideAnalysisParseError(RuntimeError):
    """Raised when the raw model output cannot be parsed into a SlideAnalysisResult.

    Stores up to 1KB of the raw output for debugging via ``raw_output_truncated``.
    """

    def __init__(self, message: str, *, raw_output: str = "") -> None:
        super().__init__(message)
        self.raw_output_truncated = raw_output[:_RAW_OUTPUT_TRUNCATION]
