"""Vision-LLM visual quality checks for rendered presentation screenshots."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import concurrent.futures
import contextvars
import queue
import threading
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.model_governance import validate_selection
from conference_pipeline.model_governance.llm_client import (
    complete,
    credential_names,
    has_credentials,
    openai_completion_options,
    resolve_selection,
)
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic


VISUAL_QC_WORKERS = max(1, int(os.environ.get('PRESENTATION_VISUAL_QC_WORKERS') or 6))
SCHEMA_VERSION = 1
MODEL_TASK_ID = "presentation.visual_qc"
DEFAULT_STAGE_ID = "visual_qc"
SUPPORTED_SCREENSHOT_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
BLOCKING_SEVERITIES = {"blocker", "fail", "failed", "error"}
COMPOSITION_BLOCKING_CHECK_IDS = {
    "text_overlap",
    "figure_overlap",
    "generic_placeholder_visual",
    "crowded_layout",
}
COMPOSITION_BLOCKING_TERMS = (
    "arrowhead",
    "connector",
    "flow",
    "clutter",
    "crowded",
    "overlap",
    "spaghetti",
    "generic",
    "placeholder",
)

# The check ids were an unglossed enum: ten names and not one definition, so
# the model inferred each from its own name. That is why unrelated complaints --
# a repeated thumbnail, an unlabelled abstraction, a two-point chart -- all piled
# into `generic_placeholder_visual`, and why every semantic objection landed in
# `other`. The enum is UNCHANGED, so counts stay comparable with earlier rounds;
# only the definitions are new.
CHECK_ID_GLOSSES = {
    "text_overflow": "a label runs past the edge of the box that contains it",
    "text_clipped": "a label is cut off by the slide or panel edge",
    "text_overlap": "two pieces of text, or a drawn line and a label, sit on top of each other",
    "figure_overlap": "shapes, connectors or arrowheads intersect where they should not",
    "illegible_text": "text too small, too faint, or broken into fragments to read",
    "low_contrast": "an element is too close in tone to what is behind it",
    "broken_svg": "something failed to render: a missing glyph, a stray marker, a blank area",
    "generic_placeholder_visual": (
        "imagery that does not depict the subject its caption names -- a stock-looking "
        "thumbnail standing in for a specific mechanism. NOT a figure that is labelled "
        "as schematic, and not an abstraction that carries a legend"
    ),
    "crowded_layout": (
        "elements packed tightly enough that reading order or connector routing is unclear"
    ),
    "other": (
        "the picture says something the claim or the takeaway does not: wrong order, "
        "alternatives drawn as a sequence, an element joined to nothing, an unlabelled branch"
    ),
}

# What the compiler does on purpose, so the judge does not report it as a defect.
COMPILER_CONVENTIONS = """THIS FIGURE WAS COMPILED, NOT DRAWN BY HAND. Some things are deliberate:
- The small grey line in the bottom corner ("schematic - illustrates the claim, not
  measured data", or "schematic layout, measured callouts") is a required honesty
  label, not a stray production note. It appears ONLY on charts and curve figures,
  which could otherwise be mistaken for measured data. Box-and-arrow diagrams,
  matrices, quadrant maps and picture strips deliberately carry no such label, and
  its absence there is not a defect.
- A schematic figure carries NO numbers anywhere, including on its axes. That is
  enforced: a number may only appear if a source states it.
- The amber line along the bottom is the slide's takeaway.
- Glyphs are drawn procedurally in the deck's own style, never photographs. Judge
  whether the drawing communicates the subject its caption names, not whether it
  looks photographic.
- Numbered marks with a list beside them are a deliberate treatment for a figure
  with more labels than fit beside their marks. Every name is present; that is not
  a missing-label defect."""

RUBRIC_VERSION_PLAIN = "visual-qc-rubric/1"
RUBRIC_VERSION_WITH_CONTEXT = "visual-qc-rubric/2"

SYSTEM_PROMPT = """You are a senior visual QA reviewer for technical conference decks.
Inspect one rendered slide screenshot. Return ONLY a JSON object with this schema:
{
  "status": "pass" | "warn" | "fail",
  "issues": [
    {
      "check_id": "text_overflow" | "text_clipped" | "text_overlap" | "figure_overlap" | "illegible_text" | "low_contrast" | "broken_svg" | "generic_placeholder_visual" | "crowded_layout" | "other",
      "severity": "warn" | "blocker",
      "description": "specific visible problem",
      "evidence": "where the issue appears on the slide",
      "suggested_fix": "specific repair"
    }
  ],
  "confidence": 0.0
}
Use severity "blocker" for any overflowing, clipped, overlapping, unreadable, broken,
generic placeholder, crowded-layout, connector-spaghetti, ambiguous-flow, or
arrowhead-over-text problem that should stop publication. Judge the figure as a
designed conference diagram, not only as parseable text: if connectors intrude into
labels, flow direction is unclear, or boxes/arrows feel crowded enough to make the
figure less publication-ready, return "blocker". Use "warn" only for minor polish
concerns. If the slide looks publication-ready, return status "pass" and an empty
issues array."""


class VisualQCTimeout(TimeoutError):
    """Raised when a single visual-QC provider call exceeds the local timeout."""


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _visual_qc_worker(
    *,
    provider: str,
    path: str,
    slide_number: int,
    model: str,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        raw = _visual_qc_call(provider, Path(path), slide_number=slide_number, model=model)
        result_queue.put({"ok": True, "raw": raw})
    except BaseException as exc:  # noqa: BLE001 - child process must report all failures.
        result_queue.put({"ok": False, "error": repr(exc)})


def _visual_qc_call_with_timeout(
    provider: str,
    path: Path,
    *,
    slide_number: int,
    model: str,
) -> str:
    timeout_seconds = _env_float("PRESENTATION_VISUAL_QC_TIMEOUT_SECONDS", 90.0)
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_visual_qc_worker,
        kwargs={
            "provider": provider,
            "path": str(path),
            "slide_number": slide_number,
            "model": model,
            "result_queue": result_queue,
        },
    )
    proc.start()
    proc.join(None if timeout_seconds <= 0 else timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise VisualQCTimeout(
            f"visual QC slide {slide_number} exceeded {timeout_seconds:g}s subprocess timeout"
        )
    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError(
            f"visual QC slide {slide_number} worker exited without returning a result "
            f"(exit code {proc.exitcode})"
        ) from exc
    if not result.get("ok"):
        raise RuntimeError(str(result.get("error", "unknown visual QC worker failure")))
    return str(result.get("raw") or "")


def inspect_screenshots_with_llm(
    screenshots_dir: Path,
    *,
    output_report: Path,
    workspace: Path,
    provider: str | None = None,
    model: str | None = None,
    run_id: str | None = None,
    stage_id: str = DEFAULT_STAGE_ID,
    max_slides: int | None = None,
    runner: Callable[[Path, int], str] | None = None,
    figure_context: dict[str, dict[str, object]] | None = None,
) -> dict[str, object]:
    """Inspect rendered slide screenshots with a vision-capable LLM.

    ``runner`` is an optional test seam that returns raw model JSON for a
    screenshot path and slide number. Production uses whichever provider the
    governed ``presentation.visual_qc`` task selects (``provider``/``model`` left
    unset mean its registered default); any provider in the task's
    ``allowed_selections`` with an adapter in ``llm_client`` can judge.

    ``figure_context`` maps a screenshot filename onto what the slide claims.
    When it is given the reviewer is also handed a definition for every check
    id and a statement of the compiler's own conventions, and the report records
    a different `rubric_version` -- the check ids themselves are unchanged, so
    counts stay comparable with rounds run without it.
    """

    context_by_name = figure_context or {}
    rubric_version = RUBRIC_VERSION_WITH_CONTEXT if context_by_name else RUBRIC_VERSION_PLAIN

    workspace = workspace.expanduser().resolve()
    screenshots_root = _resolve_inside(screenshots_dir, workspace, label="screenshots_dir")
    report_path = _resolve_inside(output_report, workspace, label="output_report")
    provider, selected_model = resolve_visual_qc_selection(provider, model)
    validate_selection(MODEL_TASK_ID, provider=provider, model=selected_model)

    screenshots = _screenshot_paths(screenshots_root)
    if max_slides is not None:
        screenshots = screenshots[: max(0, max_slides)]
    if not screenshots:
        return _write_report(
            report_path,
            _blocked_report(
                workspace=workspace,
                screenshots_dir=screenshots_root,
                provider=provider,
                model=selected_model,
                reason="no rendered slide screenshots found",
            ),
        )
    required_credentials = credential_names(provider)
    if required_credentials is None:
        return _write_report(
            report_path,
            _blocked_report(
                workspace=workspace,
                screenshots_dir=screenshots_root,
                provider=provider,
                model=selected_model,
                reason=(
                    f"no model-call adapter for provider {provider!r} "
                    "(see conference_pipeline.model_governance.llm_client)"
                ),
            ),
        )
    if runner is None and not has_credentials(provider):
        return _write_report(
            report_path,
            _blocked_report(
                workspace=workspace,
                screenshots_dir=screenshots_root,
                provider=provider,
                model=selected_model,
                reason=(
                    f"{' or '.join(required_credentials)} is required for "
                    f"vision-LLM visual QC via {provider}"
                ),
            ),
        )

    slides: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    provider_failures = 0
    max_provider_failures = _env_int("PRESENTATION_VISUAL_QC_MAX_PROVIDER_FAILURES", 3)
    provider_attempts = max(1, _env_int("PRESENTATION_VISUAL_QC_PROVIDER_ATTEMPTS", 2))
    retry_sleep_seconds = max(0.0, _env_float("PRESENTATION_VISUAL_QC_RETRY_SLEEP_SECONDS", 1.0))
    prior_slides = _load_reusable_prior_slides(
        report_path,
        workspace=workspace,
        provider=provider,
        model=selected_model,
    )
    reused_slide_count = 0
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage_id,
        caller=__file__,
    ):
        failures = {"n": 0}
        breaker_lock = threading.Lock()

        def judge_one(index: int, screenshot: Path) -> dict[str, object] | None:
            """One slide's verdict, or None when the breaker has already tripped.

            Runs on a worker thread: slides are independent, every call is a
            network round trip, and 27 of them in a row is the single longest
            block in building a deck. The provider-failure breaker is shared
            and locked, so a broken provider still stops the run rather than
            being paid for 27 times over -- the difference is that a task
            already in flight finishes instead of being cut off mid-call.
            """

            reused = False
            if runner is None and max_provider_failures > 0:
                with breaker_lock:
                    if failures["n"] >= max_provider_failures:
                        return None
            screenshot_digest = _file_sha256(screenshot)
            prior_slide = prior_slides.get(_relative(screenshot, workspace))
            try:
                if (
                    prior_slide is not None
                    and str(prior_slide.get("screenshot_sha256") or "") == screenshot_digest
                ):
                    reused = True
                    slide_issues = _prior_slide_issues(prior_slide)
                    slide_status = _prior_slide_status(prior_slide, slide_issues)
                    parsed = {"confidence": prior_slide.get("confidence")}
                else:
                    context = context_by_name.get(screenshot.name)
                    parsed = _inspect_screenshot_with_retries(
                        screenshot,
                        index,
                        provider=provider,
                        model=selected_model,
                        runner=runner,
                        attempts=provider_attempts,
                        retry_sleep_seconds=retry_sleep_seconds,
                        context=context,
                    )
                    slide_issues = _normalize_issues(
                        parsed.get("issues"), index, screenshot, workspace,
                        enum_only=bool(context_by_name),
                    )
                    slide_status = _slide_status(parsed.get("status"), slide_issues)
            except Exception as exc:  # noqa: BLE001 - fail closed with a durable report.
                with breaker_lock:
                    failures["n"] += 1
                slide_issues = [
                    {
                        "slide_number": index,
                        "check_id": "vision_llm_visual_qc_parseable",
                        "severity": "blocker",
                        "description": f"vision-LLM visual QC failed: {exc}",
                        "evidence": _relative(screenshot, workspace),
                        "suggested_fix": "rerun visual QC after checking provider output and screenshot validity",
                    }
                ]
                slide_status = "fail"
                parsed = {"status": "fail", "issues": slide_issues, "confidence": 0.0}
            return {
                "reused": reused,
                "slide": {
                    "slide_number": index,
                    "screenshot_path": _relative(screenshot, workspace),
                    "screenshot_sha256": screenshot_digest,
                    "status": slide_status,
                    "confidence": _confidence(parsed.get("confidence")),
                    "issues": slide_issues,
                },
                "issues": slide_issues,
            }

        results: list[dict[str, object] | None] = [None] * len(screenshots)
        with concurrent.futures.ThreadPoolExecutor(max_workers=VISUAL_QC_WORKERS) as pool:
            pending = {
                pool.submit(contextvars.copy_context().run, judge_one, index, screenshot): index - 1
                for index, screenshot in enumerate(screenshots, start=1)
            }
            for future in concurrent.futures.as_completed(pending):
                results[pending[future]] = future.result()

        for result in results:
            if result is None:
                continue
            if result["reused"]:
                reused_slide_count += 1
            for issue in result["issues"]:
                if str(issue.get("severity")).lower() in BLOCKING_SEVERITIES:
                    errors.append(issue)
                else:
                    warnings.append(issue)
            slides.append(result["slide"])
        provider_failures = failures["n"]

    status = "failed" if errors else "warnings" if warnings else "passed"
    return _write_report(
        report_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "generated_at": _now(),
            "model_task_id": MODEL_TASK_ID,
            "provider": provider,
            "model": selected_model,
            "screenshots_dir": _relative(screenshots_root, workspace),
            "reused_slide_count": reused_slide_count,
            "rubric_version": rubric_version,
            "promotion_policy": "enum_only" if context_by_name else "enum_and_prose",
            "slides": slides,
            "errors": errors,
            "warnings": warnings,
        },
    )


def _inspect_screenshot_with_retries(
    screenshot: Path,
    slide_number: int,
    *,
    provider: str = "openai",
    model: str,
    runner: Callable[[Path, int], str] | None,
    attempts: int,
    retry_sleep_seconds: float,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = (
                runner(screenshot, slide_number)
                if runner is not None
                else _visual_qc_call_with_timeout(
                    provider,
                    screenshot,
                    slide_number=slide_number,
                    model=model,
                )
                if _env_bool("PRESENTATION_VISUAL_QC_USE_SUBPROCESS")
                else _visual_qc_call(
                    provider,
                    screenshot,
                    slide_number=slide_number,
                    model=model,
                    figure_context=context,
                )
            )
            return _parse_model_json(raw)
        except Exception as exc:  # noqa: BLE001 - retry governed provider/parse failures.
            last_exc = exc
            if attempt < attempts and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
    assert last_exc is not None
    raise last_exc


def figure_context_block(context: dict[str, object] | None) -> str:
    """What this slide is claiming, for a judge that has only ever seen pixels.

    Without it the reviewer cannot tell a figure that omits a number from one
    that is forbidden to carry it, or a picture that fails to depict its subject
    from one whose subject it simply does not know.
    """

    if not context:
        return ""
    lines = [
        "WHAT THIS SLIDE CLAIMS",
        f"  claim: {context.get('claim') or ''}",
        f"  takeaway: {context.get('takeaway') or ''}",
        f"  figure type: {context.get('archetype') or ''} ({context.get('family') or ''})",
        f"  grounding: {context.get('grounding_kind') or ''}",
    ]
    glyphs = context.get("glyphs_used") or []
    if glyphs:
        lines.append(f"  drawings used: {', '.join(str(item) for item in glyphs)}")
    return "\n".join(lines) + "\n\n" + COMPILER_CONVENTIONS + "\n\n"


def _check_id_block() -> str:
    return "WHAT EACH check_id MEANS\n" + "\n".join(
        f"  {name}: {gloss}" for name, gloss in CHECK_ID_GLOSSES.items()
    )


def resolve_visual_qc_selection(provider: str | None, model: str | None) -> tuple[str, str]:
    """The (provider, model) pair a visual QC run uses (see ``llm_client.resolve_selection``)."""

    return resolve_selection(MODEL_TASK_ID, provider, model)


def _visual_qc_prompt(slide_number: int, figure_context: dict[str, object] | None) -> str:
    return (
        f"Review rendered screenshot sequence {slide_number}. Focus on text overflow, clipped text, "
        "text/figure overlap, unreadable labels, broken SVG rendering, low contrast, "
        "crowded layout, and generic placeholder visuals. Some screenshots are zoom "
        "companion views of a source slide, so a visible badge such as 'SLIDE 36' "
        "does not need to match the screenshot sequence number; only flag the badge "
        "when the badge itself is clipped, unreadable, or visually wrong."
        "\n\n" + figure_context_block(figure_context) + _check_id_block()
    )


def _max_completion_tokens() -> int:
    # Reasoning models (e.g. gpt-5.6-sol at high reasoning effort) spend
    # completion tokens on hidden reasoning before emitting the JSON verdict, so
    # the budget must cover reasoning + output. A 1200 cap let reasoning consume
    # the whole allowance and return an empty completion, which surfaced as a
    # reproducible "Expecting value: line 1 column 1" parse failure on denser
    # slides. Keep generous headroom; the visible verdict itself is small.
    return _env_int("PRESENTATION_VISUAL_QC_MAX_COMPLETION_TOKENS", 8000)


def _visual_qc_call(provider: str, path: Path, *, slide_number: int, model: str,
                    figure_context: dict[str, object] | None = None) -> str:
    """One screenshot, one governed model, the raw JSON text it answered with.

    The prompt, the system prompt and the parsing are identical for every
    provider (``llm_client`` carries only the transport), so a verdict means
    the same thing whoever gave it. Reasoning effort is opt-in for OpenRouter
    models via ``PRESENTATION_VISUAL_QC_REASONING_EFFORT``; OpenAI's comes from
    the registry.
    """

    effort = (os.getenv("PRESENTATION_VISUAL_QC_REASONING_EFFORT") or "").strip()
    result = complete(
        MODEL_TASK_ID,
        system=SYSTEM_PROMPT,
        prompt=_visual_qc_prompt(slide_number, figure_context),
        images=[(path.read_bytes(), _media_type(path))],
        provider=provider,
        model=model,
        max_tokens=_max_completion_tokens(),
        temperature=0.0,
        reasoning_effort=effort if effort and provider == "openrouter" else None,
        timeout=_env_float("PRESENTATION_VISUAL_QC_TIMEOUT_SECONDS", 90.0),
        max_retries=_env_int("PRESENTATION_VISUAL_QC_MAX_RETRIES", 1),
        caller=__file__,
        validate=False,  # the whole run was validated once, before the first slide
    )
    if not result.text and result.stop_reason == "max_tokens":
        raise RuntimeError(
            f"{provider} visual QC answer was cut off by the completion-token cap before "
            "any verdict was written; raise PRESENTATION_VISUAL_QC_MAX_COMPLETION_TOKENS"
        )
    return result.text


_openai_chat_completion_options = openai_completion_options


def _parse_model_json(raw: str) -> dict[str, object]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate).strip()
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("visual QC model output must be a JSON object")
    return payload


def _normalize_issues(
    issues: object,
    slide_number: int,
    screenshot: Path,
    workspace: Path,
    *,
    enum_only: bool = False,
) -> list[dict[str, object]]:
    if not isinstance(issues, list):
        raise ValueError("visual QC JSON must include issues as a list")
    normalized: list[dict[str, object]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        if _is_expected_slide_badge_sequence_warning(issue):
            continue
        normalized.append(
            {
                "slide_number": slide_number,
                "check_id": str(issue.get("check_id") or "other"),
                "severity": _issue_severity(issue, enum_only=enum_only),
                "description": str(issue.get("description") or "").strip(),
                "evidence": str(issue.get("evidence") or _relative(screenshot, workspace)),
                "suggested_fix": str(issue.get("suggested_fix") or "").strip(),
                "screenshot_path": _relative(screenshot, workspace),
            }
        )
    return normalized


def _is_expected_slide_badge_sequence_warning(issue: dict[str, object]) -> bool:
    text = " ".join(
        str(issue.get(key) or "")
        for key in ("check_id", "description", "evidence", "suggested_fix")
    ).lower()
    if "badge" not in text or "slide" not in text:
        return False
    return any(
        phrase in text
        for phrase in (
            "does not match",
            "mismatch",
            "while the prompt identifies",
            "while the slide under review",
            "requested slide number",
        )
    )


def _slide_status(status: object, issues: list[dict[str, object]]) -> str:
    if any(str(issue.get("severity")).lower() in BLOCKING_SEVERITIES for issue in issues):
        return "fail"
    normalized = str(status or "").strip().lower()
    if normalized in {"fail", "failed"}:
        return "fail"
    if normalized == "warn" or issues:
        return "warn"
    return "pass"


def _issue_severity(issue: dict[str, object], *, enum_only: bool = False) -> str:
    """The severity this finding blocks at.

    ``enum_only`` drops the substring promotion, and is used whenever the judge
    was given the check-id definitions. That list was calibrated for the
    hand-drawn route, where "flow" and "connector" in a description meant
    spaghetti. On the compiled route geometry is gated by construction, the
    model now has a definition for every id and assigns severity against it, and
    promoting on prose turned every advisory semantic remark -- any sentence
    containing the word "flow" -- into a deck blocker with no repair path.
    """

    check_id = str(issue.get("check_id") or "").strip().lower()
    if check_id in COMPOSITION_BLOCKING_CHECK_IDS:
        return "blocker"
    if not enum_only:
        text = " ".join(
            str(issue.get(key) or "")
            for key in ("check_id", "description", "evidence", "suggested_fix")
        ).lower()
        if any(term in text for term in COMPOSITION_BLOCKING_TERMS):
            return "blocker"
    return _severity(issue.get("severity"))


def _severity(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return "blocker" if normalized in BLOCKING_SEVERITIES else "warn"


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _blocked_report(
    *,
    workspace: Path,
    screenshots_dir: Path,
    provider: str,
    model: str,
    reason: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "generated_at": _now(),
        "model_task_id": MODEL_TASK_ID,
        "provider": provider,
        "model": model,
        "screenshots_dir": _relative(screenshots_dir, workspace),
        "slides": [],
        "errors": [
            {
                "slide_number": None,
                "check_id": "vision_llm_visual_qc_available",
                "severity": "blocker",
                "description": reason,
                "evidence": _relative(screenshots_dir, workspace),
                "suggested_fix": "configure the governed vision-LLM provider and rerun visual QC",
            }
        ],
        "warnings": [],
    }


def _write_report(path: Path, report: dict[str, object]) -> dict[str, object]:
    write_json_atomic(path, report)
    return report


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reusable_prior_slides(
    report_path: Path,
    *,
    workspace: Path,
    provider: str,
    model: str,
) -> dict[str, dict[str, object]]:
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(report, dict):
        return {}
    if report.get("schema_version") != SCHEMA_VERSION:
        return {}
    if report.get("provider") != provider or report.get("model") != model:
        return {}
    reusable: dict[str, dict[str, object]] = {}
    for slide in report.get("slides") or []:
        if not isinstance(slide, dict):
            continue
        screenshot_path = str(slide.get("screenshot_path") or "").strip()
        if not screenshot_path:
            continue
        if _has_provider_failure_issue(slide.get("issues")):
            continue
        reusable[screenshot_path] = slide
    return reusable


def _has_provider_failure_issue(issues: object) -> bool:
    if not isinstance(issues, list):
        return False
    return any(
        isinstance(issue, dict) and issue.get("check_id") == "vision_llm_visual_qc_parseable"
        for issue in issues
    )


def _prior_slide_issues(slide: dict[str, object]) -> list[dict[str, object]]:
    issues = slide.get("issues")
    if not isinstance(issues, list):
        return []
    return [
        issue
        for issue in issues
        if isinstance(issue, dict) and not _is_expected_slide_badge_sequence_warning(issue)
    ]


def _prior_slide_status(slide: dict[str, object], issues: list[dict[str, object]]) -> str:
    status = str(slide.get("status") or "").strip().lower()
    if not issues and status in {"warn", "fail", "failed"}:
        return "pass"
    if status in {"pass", "warn", "fail"}:
        return status
    return _slide_status(status, issues)


def _screenshot_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SCREENSHOT_SUFFIXES
    )


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"unsupported screenshot type: {path.suffix}")


def _resolve_inside(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {resolved}") from exc
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
