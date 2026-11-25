"""Three-judge rendered-figure quality panel for v2.5 trial gating.

The module is intentionally import-light: paid/provider SDKs and local MLX
runtime helpers are imported only inside live adapters. Unit tests use injected
judge callables and never call providers.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from conference_pipeline.model_governance import default_model, default_model_for_provider
from conference_pipeline.model_governance.llm_client import complete, resolve_selection

PANEL_TASK_ID = "presentation.figure_quality_panel"
GEMINI_MODEL = default_model(PANEL_TASK_ID)
OPUS_MODEL = default_model_for_provider("presentation.figure_quality_panel", "anthropic")
QWEN_MODEL = default_model_for_provider("presentation.figure_quality_panel", "local")
QWEN_AVAILABLE = True
DEFAULT_JUDGES: tuple[str, ...] = ("gemini", "claude", "qwen")

FAILURE_MODE_TAG_ORDER: tuple[str, ...] = (
    "wrong-source",
    "missing-components",
    "invented-components",
    "legibility",
    "composition",
    "aesthetic-polish",
    "style-drift",
    "other",
)
FAILURE_MODE_TAGS: frozenset[str] = frozenset(FAILURE_MODE_TAG_ORDER)

Verdict = Literal["pass", "fail"]
JudgeStatus = Literal["ok", "parse_failed", "blocked"]


@dataclass(frozen=True)
class JudgeVerdict:
    verdict: Verdict
    failure_modes: list[str]
    rationale: str
    status: JudgeStatus
    receipt_path: Path | None
    faithfulness_verdict: Verdict
    faithfulness_tags: list[str]
    legibility_verdict: Verdict
    legibility_tags: list[str]
    style_verdict: Verdict
    style_tags: list[str]

    def is_pass(self) -> bool:
        return self.status == "ok" and self.verdict == "pass"


@dataclass(frozen=True)
class PanelVerdict:
    votes: dict[str, str]
    tags_by_model: dict[str, list[str]]
    majority_verdict: Verdict
    unanimous: bool
    failure_modes: list[str]
    per_judge_sub_verdicts: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "votes": self.votes,
            "tags_by_model": self.tags_by_model,
            "majority_verdict": self.majority_verdict,
            "failure_modes": self.failure_modes,
            "unanimous": self.unanimous,
            "per_judge_sub_verdicts": self.per_judge_sub_verdicts,
        }


def _mime_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "image/png"


def _image_block(image_path: Path, *, max_dim: int = 1280) -> dict[str, Any]:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    if max(img.size) > max_dim:
        ratio = max_dim / max(img.size)
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _panel_critique_prompt(*, has_source: bool) -> str:
    source_note = (
        "Image 1 is the source figure; image 2 is the rendered redraw. "
        if has_source
        else "The image is a rendered presentation figure. "
    )
    return f"""{source_note}Judge whether the rendered figure should pass a trial-quality gate.

Evaluate three lenses in one response:
1. faithfulness: source structure, components, labels, axes, flows, and data relationships.
2. legibility: clipped text, overlapping labels, tiny text, oversized arrows, and layout clarity.
3. style/composition: brand coherence, polish, visual hierarchy, whitespace, flow direction,
   connector routing, arrowhead restraint, and whether the diagram feels publication-ready.

Intentional redraw-policy deltas are NOT defects: attribution text, slide titles,
brand color/font restyling, and callout borders or highlight boxes. These
exclusions do not relax structural fidelity: dropped stages, invented boxes,
wrong connections, garbled labels, or altered data relationships must fail.

Return JSON only with exactly this shape:
{{
  "verdict": "pass" | "fail",
  "failure_modes": ["wrong-source" | "missing-components" | "invented-components" | "legibility" | "composition" | "aesthetic-polish" | "style-drift" | "other"],
  "rationale": "concise reason",
  "faithfulness": {{"verdict": "pass" | "fail", "tags": [], "notes": "concise"}},
  "legibility": {{"verdict": "pass" | "fail", "tags": [], "notes": "concise"}},
  "style": {{"verdict": "pass" | "fail", "tags": [], "notes": "concise"}}
}}

Overall verdict must be "fail" if any lens fails. Tags must use the fixed vocabulary."""


def _raw_text_from_result(result: Any) -> str:
    value = getattr(result, "value", None)
    if value is not None:
        text = getattr(value, "text", None)
        if isinstance(text, str):
            return text
    response = getattr(result, "response", result)
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if isinstance(content, list):
        return "".join(
            str(getattr(block, "text", ""))
            for block in content
            if getattr(block, "type", "") == "text"
        )
    if isinstance(result, str):
        return result
    return str(result)


def _receipt_path_from_result(result: Any, fallback: Path | None) -> Path | None:
    if fallback is not None:
        return fallback
    receipt = getattr(result, "receipt", None)
    raw = getattr(receipt, "receipt_path", None)
    return Path(raw) if raw else None


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("judge response was not a JSON object")
    return payload


def _normalize_verdict(value: Any) -> Verdict:
    return "pass" if str(value).strip().lower() == "pass" else "fail"


def _normalize_tags(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    normalized: set[str] = set()
    unknown = False
    for item in raw_items:
        tag = str(item).strip().lower()
        if not tag:
            continue
        if tag in FAILURE_MODE_TAGS:
            normalized.add(tag)
        else:
            unknown = True
    if unknown:
        normalized.add("other")
    return [tag for tag in FAILURE_MODE_TAG_ORDER if tag in normalized]


def _lens(payload: dict[str, Any], key: str) -> tuple[Verdict, list[str]]:
    raw = payload.get(key)
    if not isinstance(raw, dict):
        raw = {}
    return _normalize_verdict(raw.get("verdict", "pass")), _normalize_tags(raw.get("tags", []))


def _judge_from_payload(
    payload: dict[str, Any],
    *,
    status: JudgeStatus,
    receipt_path: Path | None,
) -> JudgeVerdict:
    faithfulness_verdict, faithfulness_tags = _lens(payload, "faithfulness")
    legibility_verdict, legibility_tags = _lens(payload, "legibility")
    style_verdict, style_tags = _lens(payload, "style")
    verdict = _normalize_verdict(payload.get("verdict", "fail"))
    if "fail" in {faithfulness_verdict, legibility_verdict, style_verdict}:
        verdict = "fail"
    failure_modes = _normalize_tags(payload.get("failure_modes", []))
    if verdict == "fail" and not failure_modes:
        failure_modes = _ordered_tag_union(
            [faithfulness_tags, legibility_tags, style_tags, ["other"]]
        )
    rationale = str(payload.get("rationale") or payload.get("feedback") or "").strip()
    return JudgeVerdict(
        verdict=verdict,
        failure_modes=failure_modes,
        rationale=rationale,
        status=status,
        receipt_path=receipt_path,
        faithfulness_verdict=faithfulness_verdict,
        faithfulness_tags=faithfulness_tags,
        legibility_verdict=legibility_verdict,
        legibility_tags=legibility_tags,
        style_verdict=style_verdict,
        style_tags=style_tags,
    )


def _parse_judge_text(text: str, *, receipt_path: Path | None) -> JudgeVerdict:
    try:
        return _judge_from_payload(
            _parse_json_object(text),
            status="ok",
            receipt_path=receipt_path,
        )
    except Exception as exc:
        return JudgeVerdict(
            verdict="fail",
            failure_modes=["other"],
            rationale=f"judge response parse failed: {exc}",
            status="parse_failed",
            receipt_path=receipt_path,
            faithfulness_verdict="fail",
            faithfulness_tags=["other"],
            legibility_verdict="fail",
            legibility_tags=["other"],
            style_verdict="fail",
            style_tags=["other"],
        )


def _blocked_verdict(exc: Exception, *, receipt_path: Path | None) -> JudgeVerdict:
    return JudgeVerdict(
        verdict="fail",
        failure_modes=["other"],
        rationale=f"judge blocked: {exc}",
        status="blocked",
        receipt_path=receipt_path,
        faithfulness_verdict="fail",
        faithfulness_tags=["other"],
        legibility_verdict="fail",
        legibility_tags=["other"],
        style_verdict="fail",
        style_tags=["other"],
    )


def _ordered_tag_union(groups: list[list[str]]) -> list[str]:
    tags = {tag for group in groups for tag in _normalize_tags(group)}
    return [tag for tag in FAILURE_MODE_TAG_ORDER if tag in tags]


def gemini_judge_adapter(
    rendered_png: Path,
    source_crop: Path | None,
    *,
    judge_callable: Any = None,
    receipt_path: Path | None = None,
) -> JudgeVerdict:
    prompt = _panel_critique_prompt(has_source=source_crop is not None)
    try:
        if judge_callable is not None:
            raw = judge_callable([source_crop, rendered_png, prompt])
            return _parse_judge_text(_raw_text_from_result(raw), receipt_path=receipt_path)
        from google import genai  # type: ignore

        from conference_pipeline.provider_receipts import gemini_generate_content

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        client = genai.Client(api_key=api_key)
        contents: list[Any] = []
        if source_crop is not None:
            contents.append(
                genai.types.Part.from_bytes(
                    data=source_crop.read_bytes(),
                    mime_type=_mime_type(source_crop),
                )
            )
        contents.append(
            genai.types.Part.from_bytes(
                data=rendered_png.read_bytes(),
                mime_type=_mime_type(rendered_png),
            )
        )
        contents.append(prompt)
        result = gemini_generate_content(
            client,
            caller=__file__,
            model=GEMINI_MODEL,
            contents=contents,
            config={"max_output_tokens": 1024, "temperature": 0.1},
        )
        return _parse_judge_text(
            _raw_text_from_result(result),
            receipt_path=_receipt_path_from_result(result, receipt_path),
        )
    except Exception as exc:
        return _blocked_verdict(exc, receipt_path=receipt_path)


def claude_judge_adapter(
    rendered_png: Path,
    source_crop: Path | None,
    *,
    judge_callable: Any = None,
    receipt_path: Path | None = None,
) -> JudgeVerdict:
    prompt = _panel_critique_prompt(has_source=source_crop is not None)
    try:
        if judge_callable is not None:
            raw = judge_callable([source_crop, rendered_png, prompt])
            return _parse_judge_text(_raw_text_from_result(raw), receipt_path=receipt_path)
        import anthropic  # type: ignore

        from conference_pipeline.provider_receipts import anthropic_messages_create

        content: list[dict[str, Any]] = []
        if source_crop is not None:
            content.append(_image_block(source_crop))
        content.append(_image_block(rendered_png))
        content.append({"type": "text", "text": prompt})
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        result = anthropic_messages_create(
            client,
            caller=__file__,
            model=OPUS_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        return _parse_judge_text(
            _raw_text_from_result(result),
            receipt_path=_receipt_path_from_result(result, receipt_path),
        )
    except Exception as exc:
        return _blocked_verdict(exc, receipt_path=receipt_path)


def qwen_judge_adapter(
    rendered_png: Path,
    source_crop: Path | None,
    *,
    judge_callable: Any = None,
    receipt_path: Path | None = None,
    base_url: str = "http://127.0.0.1:8090/v1",
) -> JudgeVerdict:
    prompt = _panel_critique_prompt(has_source=source_crop is not None)
    if judge_callable is not None:
        raw = judge_callable([source_crop, rendered_png, prompt])
        return _parse_judge_text(_raw_text_from_result(raw), receipt_path=receipt_path)

    from conference_pipeline.model_governance.mlx_vision_client import (
        MlxVisionServerClient,
        VisionRuntimeUnavailableError,
        build_vision_messages_multi,
        extract_assistant_content,
    )

    client = MlxVisionServerClient(base_url, timeout_s=120.0)
    if not client.health_check():
        raise VisionRuntimeUnavailableError(
            f"Qwen3-VL server at {base_url} is unavailable - HALT: never fall back to 2-judge panel"
        )
    try:
        images: list[tuple[bytes, str]] = []
        if source_crop is not None:
            images.append((source_crop.read_bytes(), _mime_type(source_crop)))
        images.append((rendered_png.read_bytes(), _mime_type(rendered_png)))
        messages, _dims = build_vision_messages_multi(images=images, prompt=prompt)
        payload = client.chat_complete(
            messages,
            model=QWEN_MODEL,
            temperature=0.1,
            max_tokens=1024,
        )
        text = extract_assistant_content(payload, base_url=base_url)
        return _parse_judge_text(text, receipt_path=receipt_path)
    except VisionRuntimeUnavailableError:
        raise
    except Exception as exc:
        return _blocked_verdict(exc, receipt_path=receipt_path)


def api_judge_adapter(
    rendered_png: Path,
    source_crop: Path | None,
    *,
    model: str,
    provider: str | None = None,
    judge_callable: Any = None,
    receipt_path: Path | None = None,
) -> JudgeVerdict:
    """A panel judge reached through the governed API (any provider the task allows).

    The same prompt and the same verdict parsing as the Gemini, Claude and Qwen
    judges, so a vote means the same thing whoever cast it. The model must be
    one ``presentation.figure_quality_panel`` allows; ``complete`` validates it.
    """

    prompt = _panel_critique_prompt(has_source=source_crop is not None)
    try:
        if judge_callable is not None:
            raw = judge_callable([source_crop, rendered_png, prompt])
            return _parse_judge_text(_raw_text_from_result(raw), receipt_path=receipt_path)
        images = []
        if source_crop is not None:
            images.append((source_crop.read_bytes(), _mime_type(source_crop)))
        images.append((rendered_png.read_bytes(), _mime_type(rendered_png)))
        result = complete(
            PANEL_TASK_ID,
            prompt=prompt,
            images=images,
            provider=provider,
            model=model,
            max_tokens=int(os.environ.get("FIGURE_PANEL_API_MAX_TOKENS", "4000")),
            temperature=0.1,
            caller=__file__,
        )
        return _parse_judge_text(
            result.text,
            receipt_path=_receipt_path_from_result(result, receipt_path),
        )
    except Exception as exc:
        return _blocked_verdict(exc, receipt_path=receipt_path)


def judge_name(spec: str) -> str:
    """The key a judge votes under: a built-in's own name, else the model slug."""

    return spec if spec in DEFAULT_JUDGES else spec.rsplit("/", 1)[-1]


def configured_judges(spec: str | None = None) -> tuple[str, ...]:
    """The panel's judges: the Gemini/Claude/Qwen trio unless ``FIGURE_PANEL_JUDGES`` says otherwise.

    A comma-separated list. A token is a built-in judge (``gemini``, ``claude``,
    ``qwen``) or a model the task allows (``openrouter/moonshotai/kimi-k3``),
    which judges through the governed API. Unknown models fail loudly (the
    registry refuses them when the judge first runs); a repeated judge is an
    error, never a double vote.
    """

    raw = os.environ.get("FIGURE_PANEL_JUDGES") if spec is None else spec
    if not raw or not raw.strip():
        return DEFAULT_JUDGES
    judges = tuple(part.strip() for part in raw.split(",") if part.strip())
    names = [judge_name(judge) for judge in judges]
    if len(set(names)) != len(names):
        raise ValueError(f"a judge is listed twice in the panel: {judges}")
    if not judges:
        raise ValueError("the figure panel needs at least one judge")
    return judges


def aggregate_panel(
    verdicts: list[JudgeVerdict],
    judge_names: list[str] | None = None,
) -> PanelVerdict:
    if judge_names is None:
        judge_names = list(DEFAULT_JUDGES)
    if len(verdicts) != len(judge_names):
        raise ValueError("verdicts and judge_names must have the same length")
    votes: dict[str, str] = {}
    tags_by_model: dict[str, list[str]] = {}
    per_judge_sub_verdicts: dict[str, dict[str, Any]] = {}
    failing_tag_groups: list[list[str]] = []
    pass_count = 0
    normalized_votes: list[str] = []
    for name, verdict in zip(judge_names, verdicts, strict=True):
        vote = "blocked" if verdict.status == "blocked" else verdict.verdict
        votes[name] = vote
        tags_by_model[name] = _normalize_tags(verdict.failure_modes)
        per_judge_sub_verdicts[name] = {
            "faithfulness_verdict": verdict.faithfulness_verdict,
            "faithfulness_tags": _normalize_tags(verdict.faithfulness_tags),
            "legibility_verdict": verdict.legibility_verdict,
            "legibility_tags": _normalize_tags(verdict.legibility_tags),
            "style_verdict": verdict.style_verdict,
            "style_tags": _normalize_tags(verdict.style_tags),
        }
        if verdict.is_pass():
            pass_count += 1
            normalized_votes.append("pass")
        else:
            normalized_votes.append("fail")
            failing_tag_groups.append(verdict.failure_modes)
    # A strict majority of the judges seated. For the three-judge panel that is
    # the original "two of three"; a four-judge panel needs three, so an even
    # split fails closed rather than passing a figure half the judges rejected.
    majority_verdict: Verdict = "pass" if pass_count * 2 > len(verdicts) else "fail"
    unanimous = len(set(normalized_votes)) == 1
    return PanelVerdict(
        votes=votes,
        tags_by_model=tags_by_model,
        majority_verdict=majority_verdict,
        unanimous=unanimous,
        failure_modes=_ordered_tag_union(failing_tag_groups),
        per_judge_sub_verdicts=per_judge_sub_verdicts,
    )


def run_panel(
    rendered_png: Path,
    source_crop: Path | None,
    *,
    judge_callables: dict[str, Any] | None = None,
    qwen_base_url: str = "http://127.0.0.1:8090/v1",
    judges: tuple[str, ...] | None = None,
) -> PanelVerdict:
    callables = judge_callables or {}
    seated = configured_judges() if judges is None else judges
    verdicts: list[JudgeVerdict] = []
    for judge in seated:
        if judge == "gemini":
            verdicts.append(gemini_judge_adapter(rendered_png, source_crop, judge_callable=callables.get("gemini")))
        elif judge == "claude":
            verdicts.append(claude_judge_adapter(rendered_png, source_crop, judge_callable=callables.get("claude")))
        elif judge == "qwen":
            verdicts.append(
                qwen_judge_adapter(
                    rendered_png,
                    source_crop,
                    judge_callable=callables.get("qwen"),
                    base_url=qwen_base_url,
                )
            )
        else:
            verdicts.append(
                api_judge_adapter(
                    rendered_png,
                    source_crop,
                    model=judge,
                    provider=resolve_selection(PANEL_TASK_ID, None, judge)[0],
                    judge_callable=callables.get(judge_name(judge)),
                )
            )
    return aggregate_panel(verdicts, [judge_name(judge) for judge in seated])
