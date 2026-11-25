"""v1.61 SlideAnalysisEngine — multimodal slide analysis (distinct from text-only OCR).

Public surface:

* :class:`SlideAnalysisEngine` — provider-strategy engine
  (``openai`` / ``anthropic`` / ``local_vlm`` / ``local_fastvlm``).
* :class:`SlideAnalysisResult` — typed result with salient claims,
  figures, code blocks, equations, headings, body text, reading order.
* :class:`SlideAnalysisParseError` — raised when the model output
  cannot be parsed into the structured schema.
"""

from conference_pipeline.slide_analysis.engine import (
    SlideAnalysisEngine,
    create_slide_analysis_engine,
)
from conference_pipeline.slide_analysis.types import (
    SlideAnalysisParseError,
    SlideAnalysisResult,
    SlideFigureRef,
)

__all__ = (
    "SlideAnalysisEngine",
    "create_slide_analysis_engine",
    "SlideAnalysisParseError",
    "SlideAnalysisResult",
    "SlideFigureRef",
)
