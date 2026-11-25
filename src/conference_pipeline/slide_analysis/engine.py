"""v1.61 SlideAnalysisEngine — provider-strategy dispatch for slide-image analysis.

Distinct from :class:`conference_pipeline.ocr_engine.LLMOcrEngine`:

* ``LLMOcrEngine`` returns text-only OCR (its existing contract).
* ``SlideAnalysisEngine`` returns a structured :class:`SlideAnalysisResult`
  with salient claims, figures, code, equations, headings, reading order.

Both ship side-by-side. Receipts are written via
:func:`conference_pipeline.provider_receipts.record_local_vision_call`
for ``local_vlm`` / ``local_fastvlm`` and via the existing paid-provider
receipt machinery for ``openai`` / ``anthropic`` (out of scope for v1.61
which only extends the local lane; paid providers continue to write
their normal :class:`ProviderCallReceipt`).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from conference_pipeline.config import PipelineSettings
from conference_pipeline.model_governance.mlx_vision_client import (
    MlxVisionServerClient,
    build_vision_messages,
    extract_assistant_content,
    vision_endpoint_label,
)
from conference_pipeline.provider_receipts import record_local_vision_call
from conference_pipeline.slide_analysis.prompts import (
    SLIDE_ANALYSIS_JSON_SYSTEM,
    SLIDE_ANALYSIS_MARKDOWN_SYSTEM,
    build_user_prompt,
)
from conference_pipeline.slide_analysis.types import (
    SlideAnalysisParseError,
    SlideAnalysisResult,
    SlideFigureRef,
)


SlideProvider = Literal["openai", "anthropic", "local_vlm", "local_fastvlm"]


_MEDIA_TYPE_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def _resolve_image_bytes(image: Path | bytes) -> tuple[bytes, str]:
    """Return ``(bytes, media_type)`` from a Path or bytes input."""
    if isinstance(image, (bytes, bytearray)):
        # No extension info; assume PNG (the common slide-render output).
        return bytes(image), "image/png"
    path = Path(image)
    if not path.exists():
        raise FileNotFoundError(f"slide image not found: {path}")
    media_type = _MEDIA_TYPE_BY_SUFFIX.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(
            f"unsupported slide image extension: {path.suffix!r}; "
            f"supported: {sorted(_MEDIA_TYPE_BY_SUFFIX)}"
        )
    return path.read_bytes(), media_type


def _try_parse_json_result(raw: str) -> SlideAnalysisResult:
    """Parse the model output as a structured JSON SlideAnalysisResult."""
    candidate = raw.strip()
    # Strip optional ``` fences models sometimes emit despite instructions.
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        # Drop the opening fence line and a trailing ``` if present.
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        payload = json.loads(candidate)
    except (ValueError, TypeError) as exc:
        raise SlideAnalysisParseError(
            f"slide analysis output is not valid JSON: {exc}",
            raw_output=raw,
        )
    if not isinstance(payload, dict):
        raise SlideAnalysisParseError(
            "slide analysis JSON output must be an object",
            raw_output=raw,
        )

    figures_raw = payload.get("figures") or []
    if not isinstance(figures_raw, list):
        raise SlideAnalysisParseError(
            "slide analysis 'figures' must be a list",
            raw_output=raw,
        )
    figures: list[SlideFigureRef] = []
    for entry in figures_raw:
        if not isinstance(entry, dict):
            continue
        caption = str(entry.get("caption") or "").strip()
        modality = str(entry.get("modality") or "other").strip().lower() or "other"
        axes_raw = entry.get("axes")
        axes: tuple[str, ...] | None
        if isinstance(axes_raw, list):
            axes = tuple(str(a).strip() for a in axes_raw if str(a).strip())
        else:
            axes = None
        figures.append(SlideFigureRef(caption=caption, modality=modality, axes=axes))

    confidence_value = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    def _string_tuple(key: str) -> tuple[str, ...]:
        items = payload.get(key) or []
        if not isinstance(items, list):
            raise SlideAnalysisParseError(
                f"slide analysis '{key}' must be a list",
                raw_output=raw,
            )
        return tuple(str(item) for item in items if str(item).strip())

    return SlideAnalysisResult(
        salient_claims=_string_tuple("salient_claims"),
        figures=tuple(figures),
        code_blocks=_string_tuple("code_blocks"),
        equations=_string_tuple("equations"),
        headings=_string_tuple("headings"),
        body_text=str(payload.get("body_text") or "").strip(),
        reading_order_hint=str(payload.get("reading_order_hint") or "").strip(),
        confidence=confidence,
        raw_model_output=raw,
    )


_MARKDOWN_SECTION_RE = re.compile(r"^##\s+(?P<title>[A-Za-z][A-Za-z ]*)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^[-*]\s+(?P<item>.+?)\s*$", re.MULTILINE)
_FENCED_CODE_RE = re.compile(r"```(?:[a-zA-Z0-9_+.-]*\n)?(?P<body>.*?)```", re.DOTALL)


def _try_parse_markdown_result(raw: str) -> SlideAnalysisResult:
    """Fallback path: parse the structured markdown layout from prompts.py."""
    sections: dict[str, str] = {}
    matches = list(_MARKDOWN_SECTION_RE.finditer(raw))
    if not matches:
        raise SlideAnalysisParseError(
            "slide analysis markdown output has no '##' sections",
            raw_output=raw,
        )
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        sections[match.group("title").strip().lower()] = raw[start:end].strip()

    def _bullets(key: str) -> tuple[str, ...]:
        body = sections.get(key, "")
        return tuple(m.group("item").strip() for m in _BULLET_RE.finditer(body))

    figures_block = sections.get("figures", "")
    figures: list[SlideFigureRef] = []
    for line in _BULLET_RE.finditer(figures_block):
        text = line.group("item").strip()
        # Format: 'caption — modality [— axes]'
        parts = [p.strip() for p in re.split(r"\s+[—\-]\s+", text)]
        caption = parts[0] if parts else text
        modality = parts[1].lower() if len(parts) > 1 else "other"
        axes = (
            tuple(a.strip() for a in parts[2].split(",") if a.strip())
            if len(parts) > 2
            else None
        )
        figures.append(SlideFigureRef(caption=caption, modality=modality, axes=axes))

    code_blocks_body = sections.get("code blocks", "")
    code_blocks = tuple(
        match.group("body").strip()
        for match in _FENCED_CODE_RE.finditer(code_blocks_body)
        if match.group("body").strip()
    )

    confidence_body = sections.get("confidence", "0").strip()
    try:
        confidence = float(confidence_body.splitlines()[0].strip()) if confidence_body else 0.0
    except (ValueError, IndexError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return SlideAnalysisResult(
        salient_claims=_bullets("salient claims"),
        figures=tuple(figures),
        code_blocks=code_blocks,
        equations=_bullets("equations"),
        headings=_bullets("headings"),
        body_text=sections.get("body text", "").strip(),
        reading_order_hint=sections.get("reading order", "").strip(),
        confidence=confidence,
        raw_model_output=raw,
    )


def parse_slide_analysis(raw: str, *, prefer: str = "json") -> SlideAnalysisResult:
    """Parse a model's raw output, trying JSON first then markdown.

    Public so test fixtures and the benchmark harness can reuse it.
    """
    if prefer == "markdown":
        try:
            return _try_parse_markdown_result(raw)
        except SlideAnalysisParseError:
            return _try_parse_json_result(raw)
    try:
        return _try_parse_json_result(raw)
    except SlideAnalysisParseError:
        return _try_parse_markdown_result(raw)


@dataclass
class SlideAnalysisEngine:
    """Provider-strategy slide-analysis engine.

    Construction is dependency-injection-friendly: the four ``*_runner``
    callables each accept ``(image_bytes, media_type, system_prompt,
    user_prompt, model_id)`` and return the model's raw text output. Tests
    inject fakes for the runners; production wires them to real provider
    SDKs / loopback HTTP clients.
    """

    settings: PipelineSettings | None
    provider: SlideProvider
    model_id: str = "mlx-community/Qwen3-VL-8B-Instruct-4bit"
    output_format: str = "json"  # "json" | "markdown"
    workspace: Path | None = None
    run_id: str | None = None
    stage_id: str | None = None
    base_url: str | None = None  # local_vlm only
    timeout_s: float = 120.0
    runner: Callable[..., str] | None = None  # injected runner, optional

    def analyze(
        self,
        image: Path | bytes,
        *,
        task_id: str = "vision.slide_analysis",
        prompt_overrides: dict | None = None,
    ) -> SlideAnalysisResult:
        image_bytes, media_type = _resolve_image_bytes(image)
        system_prompt = (
            SLIDE_ANALYSIS_JSON_SYSTEM
            if self.output_format == "json"
            else SLIDE_ANALYSIS_MARKDOWN_SYSTEM
        )
        user_prompt = build_user_prompt(task=task_id, prompt_overrides=prompt_overrides)

        start = time.monotonic()
        raw_output = self._dispatch(
            image_bytes=image_bytes,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        latency_ms = (time.monotonic() - start) * 1000.0

        if self.provider in ("local_vlm", "local_fastvlm"):
            self._record_local_receipt(
                task_id=task_id,
                image_bytes_size=len(image_bytes),
                latency_ms=latency_ms,
            )

        try:
            return parse_slide_analysis(raw_output, prefer=self.output_format)
        except SlideAnalysisParseError:
            # Receipt is already written; re-raise with the truncated raw output.
            raise

    # --------------------------------------------------------- dispatch

    def _dispatch(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if self.runner is not None:
            return self.runner(
                image_bytes=image_bytes,
                media_type=media_type,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_id=self.model_id,
            )
        if self.provider == "openai":
            return self._dispatch_openai(image_bytes, media_type, system_prompt, user_prompt)
        if self.provider == "anthropic":
            return self._dispatch_anthropic(image_bytes, media_type, system_prompt, user_prompt)
        if self.provider == "local_vlm":
            return self._dispatch_local_vlm(image_bytes, media_type, system_prompt, user_prompt)
        if self.provider == "local_fastvlm":
            return self._dispatch_local_fastvlm(image_bytes, media_type, system_prompt, user_prompt)
        raise ValueError(f"unknown SlideAnalysisEngine provider: {self.provider!r}")

    def _dispatch_openai(
        self,
        image_bytes: bytes,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        from openai import OpenAI

        from conference_pipeline.provider_receipts import openai_chat_completions_create

        if self.settings is None or not self.settings.openai_api_key:
            raise ValueError("OpenAI API key required for SlideAnalysisEngine(provider='openai')")
        client = OpenAI(api_key=self.settings.openai_api_key)
        messages, _ = build_vision_messages(
            image_bytes=image_bytes,
            media_type=media_type,
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        response = openai_chat_completions_create(
            client,
            caller=__file__,
            model=self.model_id,
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def _dispatch_anthropic(
        self,
        image_bytes: bytes,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        import base64

        import anthropic

        from conference_pipeline.provider_receipts import anthropic_messages_create

        if self.settings is None or not self.settings.anthropic_api_key:
            raise ValueError(
                "Anthropic API key required for SlideAnalysisEngine(provider='anthropic')"
            )
        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = anthropic_messages_create(
            client,
            caller=__file__,
            model=self.model_id,
            max_tokens=2048,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        if not response.content:
            return ""
        return response.content[0].text.strip()

    def _dispatch_local_vlm(
        self,
        image_bytes: bytes,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        base_url = self.base_url or os.environ.get("GPTR_LOCAL_VISION_BASE_URL")
        if not base_url:
            raise ValueError(
                "SlideAnalysisEngine(provider='local_vlm') requires base_url= or "
                "GPTR_LOCAL_VISION_BASE_URL"
            )
        client = MlxVisionServerClient(base_url, timeout_s=self.timeout_s)
        messages, _ = build_vision_messages(
            image_bytes=image_bytes,
            media_type=media_type,
            prompt=user_prompt,
            system_prompt=system_prompt,
        )
        payload = client.chat_complete(messages, model=self.model_id, max_tokens=2048)
        return extract_assistant_content(payload, base_url=base_url)

    def _dispatch_local_fastvlm(
        self,
        image_bytes: bytes,
        media_type: str,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        from conference_pipeline.model_governance.fastvlm_native_client import (
            FastVLMNativeClient,
        )

        client = FastVLMNativeClient(model_id=self.model_id)
        return client.complete(
            image_bytes=image_bytes,
            media_type=media_type,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2048,
        )

    # --------------------------------------------------------- receipts

    def _record_local_receipt(
        self,
        *,
        task_id: str,
        image_bytes_size: int,
        latency_ms: float,
    ) -> None:
        runtime_kind = (
            "mlx_native_fastvlm" if self.provider == "local_fastvlm" else "vllm_mlx"
        )
        endpoint_label = (
            f"mlx-fastvlm://{self.model_id}"
            if self.provider == "local_fastvlm"
            else vision_endpoint_label(self.model_id)
        )
        # Image dimensions are recorded as best-effort by build_vision_messages
        # callers; SlideAnalysisEngine omits them from the receipt to keep the
        # write atomic (Phase 69 will join the dim metadata back in via the
        # benchmark harness path).
        record_local_vision_call(
            provider=self.provider,
            runtime_kind=runtime_kind,
            model=self.model_id,
            endpoint_label=endpoint_label,
            vision_task=task_id,
            image_count=1,
            image_dimensions=[],
            image_bytes_size=image_bytes_size,
            latency_ms=latency_ms,
            run_id=self.run_id,
            stage_id=self.stage_id,
            workspace=self.workspace,
        )


# ---------------------------------------------------------------------------
# Local-first routing factory (Phase 10 / LOCAL-02, plan 10-02 Task 3)
# ---------------------------------------------------------------------------

# Map a registry local candidate's runtime_kind to the engine's SlideProvider.
# Unknown/unsupported runtime kinds resolve to the paid path (never raise).
_RUNTIME_KIND_TO_SLIDE_PROVIDER: dict[str, SlideProvider] = {
    "vllm_mlx": "local_vlm",
    "mlx_vlm": "local_vlm",
    "mlx_native_fastvlm": "local_fastvlm",
}


def _slide_runtime_kind_for(policy, model_id: str) -> str | None:
    """Return the registry runtime_kind for a local candidate model_id."""
    for cand in getattr(policy, "local_candidates", ()):  # LocalCandidateRecord
        if cand.model_id == model_id:
            return cand.runtime_kind
    return None


def _paid_slide_selection(policy, decision) -> tuple[SlideProvider, str]:
    """Return a paid provider/model for the slide-analysis engine."""
    if decision.provider in ("openai", "anthropic"):
        return decision.provider, decision.model

    if decision.paid_fallback_provider in ("openai", "anthropic"):
        if decision.paid_fallback_model:
            return decision.paid_fallback_provider, decision.paid_fallback_model

    if policy.provider in ("openai", "anthropic"):
        return policy.provider, policy.model

    for selection in getattr(policy, "allowed_selections", ()):
        if selection.provider in ("openai", "anthropic"):
            return selection.provider, selection.model

    return "openai", policy.model


def create_slide_analysis_engine(
    settings: PipelineSettings,
    *,
    local_first_enabled: bool | None = None,
    readiness_check: Callable[[str], bool] | None = None,
    local_base_url: str | None = None,
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
    registry_path: str | Path | None = None,
) -> SlideAnalysisEngine:
    """Resolver-backed factory for the ``vision.slide_analysis`` engine.

    Local-first routing (Phase 10 / LOCAL-02): provider selection flows through
    ``resolve_local_first_selection``. A local SlideProvider
    (``local_vlm`` / ``local_fastvlm``, mapped from the registry candidate's
    ``runtime_kind``) is used ONLY when ``decision.used_local is True`` AND the
    runtime_kind is supported; otherwise the paid provider (``openai`` /
    ``anthropic``) is used. An unknown/unsupported runtime_kind falls back to
    paid rather than raising. Never branches on ``decision.provider == "local"``.

    INACTIVE SPINE: there is no live production call-site for the slide-analysis
    engine in the pipeline today; this factory wires the engine/factory surface
    and is proven offline by fixtures (see docs/reference/model-governance.md).

    Args:
        settings: Pipeline settings.
        local_first_enabled: Global switch; defaults to
            ``settings.local_first_enabled``.
        readiness_check: Injectable vision-server readiness probe; defaults to
            the real shared vision-server probe.
        local_base_url: Loopback vision-server URL (probe + local engine).
        workspace, run_id, stage_id: Run context for fallback receipts.
        registry_path: Optional registry override (testing/fixtures).
    """
    from conference_pipeline.model_governance.local_readiness import (
        make_vision_server_probe,
    )
    from conference_pipeline.model_governance.local_routing import (
        resolve_local_first_selection,
    )
    from conference_pipeline.model_governance.registry import load_registry

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
        "vision.slide_analysis",
        local_first_enabled=switch,
        readiness_check=probe,
        registry_path=registry_path,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )

    policy = load_registry(registry_path).get("vision.slide_analysis")
    if decision.used_local:
        runtime_kind = _slide_runtime_kind_for(policy, decision.model)
        slide_provider = _RUNTIME_KIND_TO_SLIDE_PROVIDER.get(runtime_kind or "")
        if slide_provider is not None:
            return SlideAnalysisEngine(
                settings=settings,
                provider=slide_provider,
                model_id=decision.model,
                base_url=base_url,
                workspace=workspace,
                run_id=run_id,
                stage_id=stage_id,
            )
        # Unknown/unsupported runtime_kind -> paid path (no raise).

    paid_provider, paid_model = _paid_slide_selection(policy, decision)
    return SlideAnalysisEngine(
        settings=settings,
        provider=paid_provider,
        model_id=paid_model,
        workspace=workspace,
        run_id=run_id,
        stage_id=stage_id,
    )
