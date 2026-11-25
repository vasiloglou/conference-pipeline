"""Prompts module for the NeurIPS pipeline.

This module contains all LLM prompts used throughout the pipeline,
organized by functionality. Using centralized prompts makes them
easier to maintain, test, and iterate on.
"""

from prompts.ocr import OCR_EXTRACTION_PROMPT
from prompts.reference_normalization import (
    REFERENCE_NORMALIZATION_PROMPT,
    format_normalization_prompt,
)
from prompts.transcription import (
    TRANSCRIPTION_PROMPT,
    TRANSCRIPTION_WITH_CONTEXT_PROMPT,
    format_transcription_prompt_with_context,
)

__all__ = [
    # OCR
    "OCR_EXTRACTION_PROMPT",
    # Reference normalization
    "REFERENCE_NORMALIZATION_PROMPT",
    "format_normalization_prompt",
    # Transcription
    "TRANSCRIPTION_PROMPT",
    "TRANSCRIPTION_WITH_CONTEXT_PROMPT",
    "format_transcription_prompt_with_context",
]
