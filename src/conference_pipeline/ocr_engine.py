"""OCR engines for extracting text from slide images."""

import asyncio
import base64
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance import default_model_for_provider
from conference_pipeline.models import OcrResult, SlideImage
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_chat_completions_create,
    record_local_vision_call,
)
from prompts.ocr import OCR_EXTRACTION_PROMPT
from utils.logging import get_logger

logger = get_logger(__name__)


class OcrError(Exception):
    """Raised when OCR fails."""

    pass


@runtime_checkable
class OcrEngine(Protocol):
    """Protocol for OCR engines."""

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run OCR on a slide image.

        Args:
            slide: SlideImage to process.

        Returns:
            OcrResult with extracted text.
        """
        ...


class BaseOcrEngine(ABC):
    """Base class for OCR engines."""

    @abstractmethod
    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run OCR on a slide image."""
        pass

    def run_ocr_batch(self, slides: list[SlideImage]) -> list[OcrResult]:
        """Run OCR on multiple slides.

        Args:
            slides: List of SlideImage objects.

        Returns:
            List of OcrResult objects.
        """
        results = []
        for slide in slides:
            try:
                result = self.run_ocr(slide)
                results.append(result)
            except Exception as e:
                logger.error(f"OCR failed for {slide.image_path}: {e}")
                results.append(
                    OcrResult(
                        presentation_id=slide.presentation_id,
                        slide_index=slide.slide_index,
                        image_path=slide.image_path,
                        raw_text=f"[OCR error: {e}]",
                    )
                )
        return results


class TesseractOcrEngine(BaseOcrEngine):
    """OCR using Tesseract."""

    def __init__(self, settings: PipelineSettings):
        """Initialize Tesseract OCR.

        Args:
            settings: Pipeline settings.
        """
        self.settings = settings
        self._check_tesseract()

    def _check_tesseract(self) -> None:
        """Check if Tesseract is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception:
            raise RuntimeError(
                "Tesseract not found. Install with: brew install tesseract (macOS) "
                "or apt-get install tesseract-ocr (Linux). "
                "Also install pytesseract: pip install pytesseract"
            )

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run Tesseract OCR on a slide.

        Args:
            slide: SlideImage to process.

        Returns:
            OcrResult with extracted text.
        """
        import pytesseract
        from PIL import Image

        if not slide.image_path.exists():
            raise OcrError(f"Image not found: {slide.image_path}")

        logger.debug(f"Running OCR on {slide.image_path}")

        try:
            image = Image.open(slide.image_path)

            # Preprocess image for better OCR
            image = self._preprocess_image(image)

            # Run OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.settings.ocr_language,
                config="--psm 1",  # Automatic page segmentation with OSD
            )

            return OcrResult(
                presentation_id=slide.presentation_id,
                slide_index=slide.slide_index,
                image_path=slide.image_path,
                raw_text=text.strip(),
            )

        except Exception as e:
            raise OcrError(f"Tesseract OCR failed: {e}") from e

    def _preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        from PIL import ImageEnhance, ImageFilter

        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)

        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)

        return image


class LLMOcrEngine(BaseOcrEngine):
    """OCR using LLM vision capabilities (OpenAI, Anthropic, or local vllm-mlx).

    The ``local`` provider (v1.61) routes through
    :class:`conference_pipeline.model_governance.mlx_vision_client.MlxVisionServerClient`
    on a loopback endpoint. The returned text is the model's raw OCR output;
    a :class:`LocalVisionCallReceipt` is written for each call.
    """

    def __init__(
        self,
        settings: PipelineSettings,
        provider: str = "openai",
        *,
        local_model_id: str | None = None,
        local_base_url: str | None = None,
        workspace: Path | None = None,
        run_id: str | None = None,
        stage_id: str | None = None,
    ):
        """Initialize LLM OCR.

        Args:
            settings: Pipeline settings.
            provider: LLM provider (``openai``, ``anthropic``, or ``local``).
            local_model_id: Required when ``provider='local'`` — registry
                ``model_id`` (e.g. ``mlx-community/GLM-OCR-0.9B-4bit``).
            local_base_url: Loopback URL of the running vision server. Falls
                back to ``GPTR_LOCAL_VISION_BASE_URL`` when not provided.
            workspace, run_id, stage_id: Optional receipt-write context for
                the ``local`` provider; ignored for paid providers (which
                already write their own receipts via ``provider_receipts``).
        """
        self.settings = settings
        self.provider = provider
        self.local_model_id = local_model_id
        self.local_base_url = local_base_url
        self.workspace = workspace
        self.run_id = run_id
        self.stage_id = stage_id

        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key required for LLM OCR")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.openai_api_key)
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                raise ValueError("Anthropic API key required for LLM OCR")
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        elif provider == "local":
            if not self.local_model_id:
                raise ValueError(
                    "LLMOcrEngine(provider='local') requires local_model_id "
                    "(e.g. mlx-community/GLM-OCR-0.9B-4bit)"
                )
            base_url = local_base_url or os.environ.get("GPTR_LOCAL_VISION_BASE_URL")
            if not base_url:
                raise ValueError(
                    "LLMOcrEngine(provider='local') requires local_base_url= or "
                    "GPTR_LOCAL_VISION_BASE_URL env var"
                )
            from conference_pipeline.model_governance.mlx_vision_client import (
                MlxVisionServerClient,
            )

            self.local_base_url = base_url
            self.client = MlxVisionServerClient(base_url)
        else:
            raise ValueError(
                f"LLMOcrEngine: unknown provider {provider!r}; "
                "expected one of 'openai', 'anthropic', 'local'"
            )

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run LLM-based OCR on a slide.

        Args:
            slide: SlideImage to process.

        Returns:
            OcrResult with extracted text.
        """
        if not slide.image_path.exists():
            raise OcrError(f"Image not found: {slide.image_path}")

        logger.debug(f"Running LLM OCR on {slide.image_path}")

        # Read and encode image
        image_data = self._encode_image(slide.image_path)

        try:
            if self.provider == "openai":
                text = self._run_openai_ocr(image_data, slide.image_path.suffix)
            elif self.provider == "anthropic":
                text = self._run_anthropic_ocr(image_data, slide.image_path.suffix)
            else:  # local
                text = self._run_local_ocr(slide.image_path)

            return OcrResult(
                presentation_id=slide.presentation_id,
                slide_index=slide.slide_index,
                image_path=slide.image_path,
                raw_text=text.strip(),
            )

        except Exception as e:
            raise OcrError(f"LLM OCR failed: {e}") from e

    def _encode_image(self, image_path: Path) -> str:
        """Encode image as base64."""
        with open(image_path, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")

    def _run_openai_ocr(self, image_data: str, suffix: str) -> str:
        """Run OCR using OpenAI Vision."""
        media_type = "image/jpeg" if suffix.lower() in [".jpg", ".jpeg"] else "image/png"

        response = openai_chat_completions_create(self.client, caller=__file__,
            model=default_model("ocr.slide_image_parse"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": OCR_EXTRACTION_PROMPT,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
        )

        return response.choices[0].message.content or ""

    def _run_anthropic_ocr(self, image_data: str, suffix: str) -> str:
        """Run OCR using Anthropic Claude Vision."""
        media_type = "image/jpeg" if suffix.lower() in [".jpg", ".jpeg"] else "image/png"

        response = anthropic_messages_create(self.client, caller=__file__,
            model=default_model_for_provider("ocr.slide_image_parse", "anthropic"),
            max_tokens=4096,
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
                        {
                            "type": "text",
                            "text": OCR_EXTRACTION_PROMPT,
                        },
                    ],
                }
            ],
        )

        return response.content[0].text if response.content else ""

    def _run_local_ocr(self, image_path: Path) -> str:
        """Run OCR through a local vllm-mlx / mlx-vlm vision server.

        Reads image bytes from disk, hands them to the Phase 66 vision
        message helper (image data never persists to disk anywhere downstream),
        POSTs to the loopback ``/chat/completions`` endpoint, and writes a
        :class:`LocalVisionCallReceipt`. Returns the model's raw OCR text.
        """
        from conference_pipeline.model_governance.mlx_vision_client import (
            build_vision_messages,
            extract_assistant_content,
            vision_endpoint_label,
        )

        suffix = image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            media_type = "image/jpeg"
        elif suffix == ".webp":
            media_type = "image/webp"
        else:
            media_type = "image/png"

        image_bytes = image_path.read_bytes()
        messages, dims = build_vision_messages(
            image_bytes=image_bytes,
            media_type=media_type,
            prompt=OCR_EXTRACTION_PROMPT,
        )
        start = time.monotonic()
        payload = self.client.chat_complete(
            messages,
            model=self.local_model_id,
            max_tokens=4096,
        )
        latency_ms = (time.monotonic() - start) * 1000.0
        text = extract_assistant_content(
            payload, base_url=self.client.base_url
        )
        record_local_vision_call(
            provider="local",
            runtime_kind="vllm_mlx",
            model=self.local_model_id,
            endpoint_label=vision_endpoint_label(self.local_model_id),
            vision_task="ocr.slide_image_parse",
            image_count=1,
            image_dimensions=[list(dims)],
            image_bytes_size=len(image_bytes),
            latency_ms=latency_ms,
            run_id=self.run_id,
            stage_id=self.stage_id,
            workspace=self.workspace,
        )
        return text


class CombinedOcrEngine(BaseOcrEngine):
    """Combines multiple OCR engines for better results."""

    def __init__(
        self,
        settings: PipelineSettings,
        use_llm_fallback: bool = True,
    ):
        """Initialize combined OCR engine.

        Args:
            settings: Pipeline settings.
            use_llm_fallback: Whether to use LLM for low-quality Tesseract results.
        """
        self.settings = settings
        self.tesseract = TesseractOcrEngine(settings)
        self.llm_engine = None

        if use_llm_fallback and (settings.openai_api_key or settings.anthropic_api_key):
            provider = "openai" if settings.openai_api_key else "anthropic"
            try:
                self.llm_engine = LLMOcrEngine(settings, provider)
            except Exception as e:
                logger.warning(f"Failed to initialize LLM OCR: {e}")

    def run_ocr(self, slide: SlideImage) -> OcrResult:
        """Run OCR with fallback to LLM for poor results.

        Args:
            slide: SlideImage to process.

        Returns:
            OcrResult with best extracted text.
        """
        # First try Tesseract
        tesseract_result = self.tesseract.run_ocr(slide)

        # Check quality of result
        if self._is_good_quality(tesseract_result.raw_text):
            return tesseract_result

        # Try LLM if available and Tesseract result is poor
        if self.llm_engine:
            logger.debug(f"Tesseract result poor, trying LLM for {slide.image_path}")
            try:
                llm_result = self.llm_engine.run_ocr(slide)
                # Use LLM result if it's better
                if len(llm_result.raw_text) > len(tesseract_result.raw_text):
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM OCR fallback failed: {e}")

        return tesseract_result

    def _is_good_quality(self, text: str, min_length: int = 50) -> bool:
        """Check if OCR result is good quality.

        Args:
            text: Extracted text.
            min_length: Minimum length for good quality.

        Returns:
            True if result appears good quality.
        """
        if len(text) < min_length:
            return False

        # Check for too many special characters (often indicates poor OCR)
        alpha_ratio = sum(c.isalnum() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return False

        return True


def create_ocr_engine(
    settings: PipelineSettings,
    *,
    local_first_enabled: bool | None = None,
    readiness_check: "Callable[[str], bool] | None" = None,
    local_base_url: str | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
) -> OcrEngine:
    """Factory function to create appropriate OCR engine.

    Local-first routing (Phase 10 / LOCAL-02): when LLM OCR is enabled, the
    provider choice for ``ocr.slide_image_parse`` flows through
    ``resolve_local_first_selection``. A local OCR engine is constructed ONLY
    when ``decision.used_local is True`` AND a loopback vision base URL is
    derivable; otherwise the existing paid ``LLMOcrEngine`` path runs unchanged.
    All routing kwargs are optional/additive so existing callers
    (``create_ocr_engine(settings)``) keep working.

    NOTE: this wires the GENERIC ``NeuripsPipeline`` OCR seam, which has a local
    executor. The ``iclr2025_live`` ``create_slide_ocr_service`` path is
    openai/anthropic/gemini-only (no local executor) and is intentionally left
    paid-only — see docs/reference/model-governance.md.

    Args:
        settings: Pipeline settings.
        local_first_enabled: Global switch; defaults to
            ``settings.local_first_enabled`` when omitted.
        readiness_check: Injectable vision-server readiness probe; defaults to
            the real shared vision-server probe.
        local_base_url: Loopback URL of the local vision server (used for both
            the readiness probe and the local engine). Falls back to
            ``GPTR_LOCAL_VISION_BASE_URL`` when omitted.
        workspace, run_id, stage_id: Run context; on a fallback the resolver
            writes a sanitized ``LocalFallbackReceipt`` when ``workspace`` is set.

    Returns:
        Configured OCR engine.
    """
    if not settings.use_llm_ocr:
        return CombinedOcrEngine(settings)

    from conference_pipeline.model_governance.local_readiness import (
        make_vision_server_probe,
    )
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )

    base_url = local_base_url or os.environ.get("GPTR_LOCAL_VISION_BASE_URL")
    switch = (
        settings.local_first_enabled
        if local_first_enabled is None
        else local_first_enabled
    )
    probe = (
        readiness_check
        if readiness_check is not None
        else make_vision_server_probe(base_url)
    )

    decision = resolve_local_first_selection(
        "ocr.slide_image_parse",
        local_first_enabled=switch,
        readiness_check=probe,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )

    # Construct a local OCR engine ONLY when the resolver actually selected an
    # approved local model (proof + readiness passed). Never branch on
    # decision.provider == "local". If no loopback vision URL is derivable, fall
    # through to the paid path gracefully (no crash).
    if decision.used_local and base_url:
        return LLMOcrEngine(
            settings,
            "local",
            local_model_id=decision.model,
            local_base_url=base_url,
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
        )

    # Paid path — unchanged provider selection.
    provider = "openai" if settings.openai_api_key else "anthropic"
    return LLMOcrEngine(
        settings,
        provider,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )
