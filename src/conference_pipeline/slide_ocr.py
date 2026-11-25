"""Slide OCR service using vision-capable LLMs.

This module provides OCR functionality for slide images using OpenAI,
Anthropic Claude, and Gemini vision models.
"""

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.models import OcrResult
from conference_pipeline.provider_receipts import (
    anthropic_messages_create_async,
    gemini_generate_content,
    openai_chat_completions_create_async,
)
from conference_pipeline.rate_limiter import RateLimiter, retry_with_backoff
from conference_pipeline.model_governance.model_families import uses_openai_reasoning_params
from prompts.ocr import format_slide_ocr_prompt
from utils.logging import get_logger

logger = get_logger(__name__)

# Default rate limiting for vision API calls
DEFAULT_VISION_RPM = 30  # requests per minute (more conservative for vision)
DEFAULT_VISION_CONCURRENT = 3  # max concurrent requests


def _observe_provider_receipt(
    receipt_observer: Callable[[object], None] | None,
    provider_result_or_error: object,
) -> None:
    """Forward a provider-call receipt to a receipt-owning workflow.

    The normal OCR API remains text-only, but r6 needs the wrapper receipt for
    every successful or failed SDK dispatch before it can settle its ledger.
    """

    if receipt_observer is None:
        return
    receipt = getattr(provider_result_or_error, "receipt", None) or getattr(
        provider_result_or_error, "provider_receipt", None
    )
    if receipt is not None:
        receipt_observer(receipt)


@dataclass
class SlideInfo:
    """Information about a slide for OCR processing."""

    index: int
    image_path: Path
    timestamp_sec: float | None = None


@dataclass
class TranscriptContext:
    """Transcript context for a slide."""

    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class OCRConfidence:
    """Confidence decision for a raw OCR model response."""

    accepted: bool
    reason: str
    extracted_text_chars: int = 0


@dataclass(frozen=True)
class OCRCallMetadata:
    """Provider/model metadata for a completed OCR call."""

    provider: str
    model: str
    fallback_used: bool = False
    fallback_provider: str | None = None
    fallback_model: str | None = None
    fallback_reason: str | None = None
    confidence_reason: str | None = None


def load_slide_sync(slides_dir: Path, include_skipped: bool = False) -> list[SlideInfo]:
    """Load slide synchronization data.

    Args:
        slides_dir: Path to slides directory containing sync.json.
        include_skipped: If False (default), exclude slides with skip_ocr=True.

    Returns:
        List of SlideInfo objects with timestamps.
    """
    sync_file = slides_dir / "sync.json"
    if not sync_file.exists():
        # Fall back to listing images without timestamps
        images_dir = slides_dir / "images"
        if not images_dir.exists():
            return []

        slides = []
        for img_path in sorted(images_dir.glob("slide_*.png")):
            # Extract index from filename (slide_001.png -> 1)
            try:
                idx = int(img_path.stem.split("_")[1])
                slides.append(SlideInfo(index=idx, image_path=img_path))
            except (ValueError, IndexError):
                continue
        return slides

    with open(sync_file) as f:
        sync_data = json.load(f)

    slides = []
    images_dir = slides_dir / "images"
    for slide in sync_data.get("slides", []):
        # Skip slides marked for skipping (unless include_skipped is True)
        if not include_skipped and slide.get("skip_ocr"):
            continue

        image_path = images_dir / slide["image"]
        if image_path.exists():
            slides.append(
                SlideInfo(
                    index=slide["index"],
                    image_path=image_path,
                    timestamp_sec=slide.get("timestamp_sec"),
                )
            )
    return slides


def load_transcript_segments(transcripts_dir: Path) -> list[dict]:
    """Load transcript segments with timestamps.

    Args:
        transcripts_dir: Path to transcripts directory.

    Returns:
        List of segment dictionaries with start, end, text.
    """
    segments_file = transcripts_dir / "segments.json"
    if not segments_file.exists():
        return []

    with open(segments_file) as f:
        return json.load(f)


def get_transcript_context(
    slide_timestamp: float,
    next_slide_timestamp: float | None,
    segments: list[dict],
    context_before: float = 30.0,
    context_after: float = 30.0,
) -> TranscriptContext | None:
    """Get transcript text around a slide timestamp.

    Args:
        slide_timestamp: The timestamp (in seconds) when the slide appears.
        next_slide_timestamp: Timestamp of the next slide (or None if last).
        segments: List of transcript segments.
        context_before: Seconds of context before slide appears.
        context_after: Seconds of context after slide appears.

    Returns:
        TranscriptContext with relevant text, or None if no segments.
    """
    if not segments or slide_timestamp is None:
        return None

    # Determine time window
    start_time = max(0, slide_timestamp - context_before)

    # Use next slide timestamp if available, otherwise use context_after
    if next_slide_timestamp is not None:
        end_time = next_slide_timestamp
    else:
        end_time = slide_timestamp + context_after

    # Find segments in the time window
    relevant_segments = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)

        # Check if segment overlaps with our window
        if seg_end >= start_time and seg_start <= end_time:
            relevant_segments.append(seg)

    if not relevant_segments:
        return None

    # Combine text from relevant segments
    text = " ".join(seg.get("text", "") for seg in relevant_segments)

    return TranscriptContext(
        text=text.strip(),
        start_time=start_time,
        end_time=end_time,
    )


def parse_ocr_response(response_text: str) -> dict:
    """Parse the JSON response from OCR.

    Args:
        response_text: Raw response text from the LLM.

    Returns:
        Parsed JSON dictionary, or a dict with raw_text if parsing fails.
    """
    # Try to extract JSON from the response
    text = response_text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse OCR response as JSON: {e}")
        # Return a minimal structure with the raw text
        return {
            "slide_text": {
                "title": "",
                "other_text": response_text,
            },
            "visual_elements": [],
            "key_concepts": [],
            "references": [],
            "parse_error": str(e),
        }


def _collect_text_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_collect_text_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_collect_text_values(item))
        return values
    return []


def evaluate_ocr_confidence(response_text: str, *, min_text_chars: int = 20) -> OCRConfidence:
    """Decide whether an OCR response is usable enough to avoid fallback."""

    stripped = response_text.strip()
    if not stripped:
        return OCRConfidence(False, "empty_response", 0)

    lower = stripped.lower()
    refusal_markers = (
        "i can't help",
        "i cannot help",
        "i'm unable",
        "i am unable",
        "cannot extract",
        "[ocr error:",
    )
    if any(marker in lower for marker in refusal_markers):
        return OCRConfidence(False, "refusal_or_error_response", len(stripped))

    data = parse_ocr_response(stripped)
    if data.get("parse_error"):
        return OCRConfidence(False, "invalid_json", len(stripped))

    relevant_values = []
    for key in ("slide_text", "visual_elements", "key_concepts", "references"):
        relevant_values.extend(_collect_text_values(data.get(key)))
    extracted_chars = len(" ".join(value.strip() for value in relevant_values if value.strip()))
    if extracted_chars < min_text_chars:
        return OCRConfidence(False, "low_extracted_content", extracted_chars)
    return OCRConfidence(True, "accepted", extracted_chars)


def encode_image_base64(image_path: Path) -> str:
    """Encode an image file to base64.

    Args:
        image_path: Path to image file.

    Returns:
        Base64-encoded string.
    """
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """Get the media type for an image based on extension.

    Args:
        image_path: Path to image file.

    Returns:
        Media type string (e.g., "image/png").
    """
    suffix = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/png")


@runtime_checkable
class SlideOCRService(Protocol):
    """Protocol for slide OCR services."""

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        """Perform OCR on a single slide image.

        Args:
            image_path: Path to slide image.
            title: Presentation title.
            transcript_context: Optional transcript text for context.

        Returns:
            Extracted text and descriptions.
        """
        ...


class OpenAISlideOCRService:
    """Slide OCR using OpenAI vision models."""

    def __init__(
        self,
        api_key: str,
        model: str = default_model_for_provider("ocr.slide_image_parse", "openai"),
        max_tokens: int = 4096,
        timeout_seconds: float | None = None,
        receipt_observer: Callable[[object], None] | None = None,
    ):
        """Initialize OpenAI slide OCR service.

        Args:
            api_key: OpenAI API key.
            model: Model to use.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.receipt_observer = receipt_observer
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        """Perform OCR on a single slide using OpenAI vision.

        Args:
            image_path: Path to slide image.
            title: Presentation title.
            transcript_context: Optional transcript text for context.

        Returns:
            Extracted text and descriptions.
        """
        prompt = format_slide_ocr_prompt(title, transcript_context)
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)

        request: dict[str, object] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        if uses_openai_reasoning_params(self.model):
            request["max_completion_tokens"] = self.max_tokens
        else:
            request["max_tokens"] = self.max_tokens

        try:
            response = await openai_chat_completions_create_async(self.client, caller=__file__, **request)
        except Exception as exc:
            _observe_provider_receipt(self.receipt_observer, exc)
            raise
        _observe_provider_receipt(self.receipt_observer, response)

        return response.choices[0].message.content or ""


class AnthropicSlideOCRService:
    """Slide OCR using Anthropic Claude vision."""

    def __init__(
        self,
        api_key: str,
        model: str = default_model_for_provider("ocr.slide_image_parse", "anthropic"),
        max_tokens: int = 4096,
        timeout_seconds: float | None = None,
        receipt_observer: Callable[[object], None] | None = None,
    ):
        """Initialize Anthropic slide OCR service.

        Args:
            api_key: Anthropic API key.
            model: Model to use.
            max_tokens: Maximum tokens in response.
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.receipt_observer = receipt_observer
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout_seconds)
        return self._client

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        """Perform OCR on a single slide using Claude vision.

        Args:
            image_path: Path to slide image.
            title: Presentation title.
            transcript_context: Optional transcript text for context.

        Returns:
            Extracted text and descriptions.
        """
        prompt = format_slide_ocr_prompt(title, transcript_context)
        image_data = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)

        try:
            response = await anthropic_messages_create_async(self.client, caller=__file__,
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
        except Exception as exc:
            _observe_provider_receipt(self.receipt_observer, exc)
            raise
        _observe_provider_receipt(self.receipt_observer, response)

        # Extract text from response
        text_parts = []
        for block in response.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)


class GeminiSlideOCRService:
    """Slide OCR using Gemini vision models."""

    def __init__(
        self,
        api_key: str,
        model: str = default_model_for_provider("ocr.slide_image_parse", "gemini"),
        max_tokens: int = 4096,
        timeout_seconds: float | None = None,
        receipt_observer: Callable[[object], None] | None = None,
    ):
        """Initialize Gemini slide OCR service."""
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.receipt_observer = receipt_observer
        self._client = None
        self._genai = None

    @property
    def genai(self):
        """Lazy import of the Google GenAI SDK."""
        if self._genai is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError(
                    "google-genai package required for Gemini OCR. Install with `pip install google-genai`."
                ) from exc
            self._genai = genai
        return self._genai

    @property
    def client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            self._client = self.genai.Client(api_key=self.api_key)
        return self._client

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        """Perform OCR on a single slide using Gemini vision."""
        prompt = format_slide_ocr_prompt(title, transcript_context)
        media_type = get_image_media_type(image_path)
        part = self.genai.types.Part.from_bytes(
            data=image_path.read_bytes(),
            mime_type=media_type,
        )

        def _generate() -> str:
            try:
                response = gemini_generate_content(self.client, caller=__file__,
                    model=self.model,
                    contents=[part, prompt],
                    config={
                        "max_output_tokens": self.max_tokens,
                        "temperature": 0.0,
                    },
                )
            except Exception as exc:
                _observe_provider_receipt(self.receipt_observer, exc)
                raise
            _observe_provider_receipt(self.receipt_observer, response)
            return getattr(response, "text", "") or ""

        return await asyncio.to_thread(_generate)


def service_provider_model(service: SlideOCRService) -> tuple[str, str]:
    """Return best-effort provider and model labels for an OCR service."""

    if isinstance(service, OpenAISlideOCRService):
        return "openai", service.model
    if isinstance(service, AnthropicSlideOCRService):
        return "anthropic", service.model
    if isinstance(service, GeminiSlideOCRService):
        return "gemini", service.model
    if isinstance(service, FallbackSlideOCRService):
        return service.primary_provider, service.primary_model
    return service.__class__.__name__.lower(), getattr(service, "model", "unknown")


def service_call_metadata(service: SlideOCRService, image_path: Path) -> OCRCallMetadata:
    """Return metadata for the latest call for an image."""

    metadata_for = getattr(service, "metadata_for", None)
    if callable(metadata_for):
        metadata = metadata_for(image_path)
        if isinstance(metadata, OCRCallMetadata):
            return metadata
    provider, model = service_provider_model(service)
    return OCRCallMetadata(provider=provider, model=model, confidence_reason="accepted")


class FallbackSlideOCRService:
    """OCR service that falls back when primary output is missing or low confidence."""

    def __init__(
        self,
        primary: SlideOCRService,
        fallback: SlideOCRService,
        *,
        min_text_chars: int = 20,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.min_text_chars = min_text_chars
        self.primary_provider, self.primary_model = service_provider_model(primary)
        self.fallback_provider, self.fallback_model = service_provider_model(fallback)
        self._metadata_by_image: dict[str, OCRCallMetadata] = {}

    def metadata_for(self, image_path: Path) -> OCRCallMetadata | None:
        return self._metadata_by_image.get(str(image_path))

    async def ocr_slide(
        self,
        image_path: Path,
        title: str,
        transcript_context: str | None = None,
    ) -> str:
        fallback_reason: str | None = None
        try:
            response = await self.primary.ocr_slide(image_path, title, transcript_context)
            confidence = evaluate_ocr_confidence(
                response,
                min_text_chars=self.min_text_chars,
            )
            if confidence.accepted:
                self._metadata_by_image[str(image_path)] = OCRCallMetadata(
                    provider=self.primary_provider,
                    model=self.primary_model,
                    fallback_used=False,
                    confidence_reason=confidence.reason,
                )
                return response
            fallback_reason = confidence.reason
        except Exception as exc:  # noqa: BLE001 - fallback receipts need the reason.
            fallback_reason = f"primary_error:{type(exc).__name__}"

        fallback_response = await self.fallback.ocr_slide(image_path, title, transcript_context)
        fallback_confidence = evaluate_ocr_confidence(
            fallback_response,
            min_text_chars=self.min_text_chars,
        )
        self._metadata_by_image[str(image_path)] = OCRCallMetadata(
            provider=self.fallback_provider,
            model=self.fallback_model,
            fallback_used=True,
            fallback_provider=self.fallback_provider,
            fallback_model=self.fallback_model,
            fallback_reason=fallback_reason,
            confidence_reason=fallback_confidence.reason,
        )
        return fallback_response


class SlideOCRPipeline:
    """Pipeline for OCR processing of presentation slides."""

    def __init__(
        self,
        ocr_service: SlideOCRService,
        max_concurrent: int = DEFAULT_VISION_CONCURRENT,
        max_rpm: int = DEFAULT_VISION_RPM,
    ):
        """Initialize slide OCR pipeline.

        Args:
            ocr_service: The OCR service to use.
            max_concurrent: Maximum concurrent API requests.
            max_rpm: Maximum requests per minute.
        """
        self.ocr_service = ocr_service
        self.max_concurrent = max_concurrent
        self.max_rpm = max_rpm

    async def process_presentation(
        self,
        pres_dir: Path,
        title: str,
        show_progress: bool = True,
        parallel: bool = True,
    ) -> list[OcrResult]:
        """Process all slides in a presentation.

        Args:
            pres_dir: Path to presentation directory.
            title: Presentation title.
            show_progress: Whether to show progress bar.
            parallel: Whether to process slides in parallel.

        Returns:
            List of OcrResult objects.
        """
        slides_dir = pres_dir / "slides"
        transcripts_dir = pres_dir / "transcripts"
        ocr_dir = pres_dir / "ocr"
        ocr_dir.mkdir(exist_ok=True)

        # Load slides and transcript segments
        slides = load_slide_sync(slides_dir)
        if not slides:
            logger.warning(f"No slides found in {slides_dir}")
            return []

        segments = load_transcript_segments(transcripts_dir)

        # Get presentation ID from meta.json
        meta_file = pres_dir / "meta.json"
        presentation_id = pres_dir.name
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                presentation_id = meta.get("id", pres_dir.name)

        if parallel:
            results = await self._process_parallel(
                slides, segments, title, presentation_id, ocr_dir, show_progress
            )
        else:
            results = await self._process_sequential(
                slides, segments, title, presentation_id, ocr_dir, show_progress
            )

        return results

    async def _process_sequential(
        self,
        slides: list[SlideInfo],
        segments: list[dict],
        title: str,
        presentation_id: str,
        ocr_dir: Path,
        show_progress: bool,
    ) -> list[OcrResult]:
        """Process slides sequentially."""
        try:
            from tqdm import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        results = []

        if show_progress and has_tqdm:
            slide_iter = tqdm(slides, desc="  OCR slides", unit="slide", leave=False)
        else:
            slide_iter = slides

        for i, slide in enumerate(slide_iter):
            # Get next slide timestamp for context window
            next_timestamp = None
            if i + 1 < len(slides):
                next_timestamp = slides[i + 1].timestamp_sec

            # Get transcript context
            context = get_transcript_context(
                slide.timestamp_sec, next_timestamp, segments
            )
            transcript_text = context.text if context else None

            try:
                ocr_text = await self.ocr_service.ocr_slide(
                    slide.image_path, title, transcript_text
                )

                # Parse JSON response
                ocr_data = parse_ocr_response(ocr_text)
                metadata = service_call_metadata(self.ocr_service, slide.image_path)

                result = OcrResult(
                    presentation_id=presentation_id,
                    slide_index=slide.index,
                    image_path=slide.image_path,
                    raw_text=ocr_text,
                    engine=metadata.provider,
                )
                results.append(result)

                # Save individual OCR result as JSON
                ocr_file = ocr_dir / f"slide_{slide.index:03d}.json"
                save_data = {
                    "slide_index": slide.index,
                    "timestamp_sec": slide.timestamp_sec,
                    "image_path": str(slide.image_path.name),
                    "provider": metadata.provider,
                    "provider_model": metadata.model,
                    "fallback_used": metadata.fallback_used,
                    "fallback_provider": metadata.fallback_provider,
                    "fallback_model": metadata.fallback_model,
                    "fallback_reason": metadata.fallback_reason,
                    "confidence_reason": metadata.confidence_reason,
                    **ocr_data,
                }
                with open(ocr_file, "w") as f:
                    json.dump(save_data, f, indent=2)

            except Exception as e:
                logger.error(f"Error processing slide {slide.index}: {e}")
                results.append(
                    OcrResult(
                        presentation_id=presentation_id,
                        slide_index=slide.index,
                        image_path=slide.image_path,
                        raw_text=f"[OCR Error: {e}]",
                        engine="error",
                    )
                )

            # Rate limiting delay
            await asyncio.sleep(0.5)

        return results

    async def _process_parallel(
        self,
        slides: list[SlideInfo],
        segments: list[dict],
        title: str,
        presentation_id: str,
        ocr_dir: Path,
        show_progress: bool,
    ) -> list[OcrResult]:
        """Process slides in parallel with rate limiting."""
        try:
            from tqdm import tqdm

            has_tqdm = True
        except ImportError:
            has_tqdm = False

        rate_limiter = RateLimiter(
            max_rpm=self.max_rpm, max_concurrent=self.max_concurrent
        )
        results: dict[int, OcrResult] = {}

        pbar = None
        if show_progress and has_tqdm:
            pbar = tqdm(total=len(slides), desc="  OCR slides", unit="slide", leave=False)

        async def process_slide(slide: SlideInfo, slide_idx: int):
            async with rate_limiter:
                # Get next slide timestamp for context window
                next_timestamp = None
                if slide_idx + 1 < len(slides):
                    next_timestamp = slides[slide_idx + 1].timestamp_sec

                # Get transcript context
                context = get_transcript_context(
                    slide.timestamp_sec, next_timestamp, segments
                )
                transcript_text = context.text if context else None

                try:
                    ocr_text = await retry_with_backoff(
                        lambda: self.ocr_service.ocr_slide(
                            slide.image_path, title, transcript_text
                        ),
                        max_retries=3,
                    )

                    # Parse JSON response
                    ocr_data = parse_ocr_response(ocr_text)
                    metadata = service_call_metadata(self.ocr_service, slide.image_path)

                    result = OcrResult(
                        presentation_id=presentation_id,
                        slide_index=slide.index,
                        image_path=slide.image_path,
                        raw_text=ocr_text,
                        engine=metadata.provider,
                    )
                    results[slide.index] = result

                    # Save individual OCR result as JSON
                    ocr_file = ocr_dir / f"slide_{slide.index:03d}.json"
                    save_data = {
                        "slide_index": slide.index,
                        "timestamp_sec": slide.timestamp_sec,
                        "image_path": str(slide.image_path.name),
                        "provider": metadata.provider,
                        "provider_model": metadata.model,
                        "fallback_used": metadata.fallback_used,
                        "fallback_provider": metadata.fallback_provider,
                        "fallback_model": metadata.fallback_model,
                        "fallback_reason": metadata.fallback_reason,
                        "confidence_reason": metadata.confidence_reason,
                        **ocr_data,
                    }
                    with open(ocr_file, "w") as f:
                        json.dump(save_data, f, indent=2)

                except Exception as e:
                    logger.error(f"Error processing slide {slide.index}: {e}")
                    results[slide.index] = OcrResult(
                        presentation_id=presentation_id,
                        slide_index=slide.index,
                        image_path=slide.image_path,
                        raw_text=f"[OCR Error: {e}]",
                        engine="error",
                    )

                if pbar:
                    pbar.update(1)

        # Create tasks for all slides
        tasks = [
            asyncio.create_task(process_slide(slide, i))
            for i, slide in enumerate(slides)
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        if pbar:
            pbar.close()

        # Return results in order
        return [results[slide.index] for slide in slides if slide.index in results]


def _create_single_slide_ocr_service(
    settings: PipelineSettings,
    provider: str,
    model: str,
    *,
    receipt_observer: Callable[[object], None] | None = None,
) -> SlideOCRService:
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key required for OCR")
        return OpenAISlideOCRService(
            api_key=settings.openai_api_key,
            model=model,
            receipt_observer=receipt_observer,
        )
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key required for OCR")
        return AnthropicSlideOCRService(
            api_key=settings.anthropic_api_key,
            model=model,
            receipt_observer=receipt_observer,
        )
    if provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY required for Gemini OCR")
        return GeminiSlideOCRService(
            api_key=settings.google_api_key,
            model=model,
            receipt_observer=receipt_observer,
        )
    raise ValueError(f"Unknown OCR provider: {provider}")


def create_slide_ocr_service(
    settings: PipelineSettings,
    *,
    receipt_observer: Callable[[object], None] | None = None,
) -> SlideOCRService:
    """Create a slide OCR service based on settings.

    Args:
        settings: Pipeline settings.

    Returns:
        SlideOCRService instance.
    """
    provider = getattr(settings, "ocr_provider", "openai")
    model = getattr(settings, "ocr_model", None) or default_model_for_provider(
        "ocr.slide_image_parse", provider
    )
    primary = _create_single_slide_ocr_service(
        settings, provider, model, receipt_observer=receipt_observer
    )

    fallback_provider = getattr(settings, "ocr_fallback_provider", "none")
    if fallback_provider == "none":
        return primary
    fallback_model = getattr(settings, "ocr_fallback_model", None) or default_model_for_provider(
        "ocr.slide_image_parse", fallback_provider
    )
    try:
        fallback = _create_single_slide_ocr_service(
            settings, fallback_provider, fallback_model, receipt_observer=receipt_observer
        )
    except ValueError as exc:
        logger.warning(f"OCR fallback disabled: {exc}")
        return primary
    return FallbackSlideOCRService(primary, fallback)
