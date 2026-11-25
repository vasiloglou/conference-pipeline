"""Reusable narrative-content gate for presentation deck drafts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence


DEFAULT_POST_STAGE14_DECK_STALE_MARKERS: tuple[str, ...] = (
    "Gap: transcripts and OCR not produced",
    "virtual-session metadata remains a recorded source gap",
    "Audio package is script/audio-ready, not rendered waveform audio",
    "Video generation remains unsupported without conference-local proof",
)
DEFAULT_PUBLIC_LEDGER_HEADINGS: tuple[str, ...] = (
    "## Source Basis",
    "## Verbatim transcript/OCR evidence",
    "## Matched Paper Evidence",
    "## Source Gaps",
)
DEFAULT_DECK_STAGE_FILENAMES: tuple[str, ...] = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
)
DEFAULT_REQUIRED_DRAFT_MARKERS: tuple[str, ...] = (
    "## Slide 1: Executive Summary",
    "Traceable Production Boundary",
)
DELIVERY_REQUIRED_SOURCE_GROUNDING_CHECKS: tuple[str, ...] = (
    "what the audience sees",
    "why it matters",
    "named paper, named talk, speaker, author, or work contribution",
)

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_workspace_path(workspace_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace_root / path


def _relpath(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def _clean_heading(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned.strip(" -*\t\r\n\"")


def _heading_block(markdown: str, heading: str) -> str:
    lines = markdown.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("## ") and lines[index].strip() != heading:
            end = index
            break
    return "\n".join(lines[start:end]).strip()


def _bullet_items(block: str) -> list[str]:
    return [
        _clean_heading(line.removeprefix("- ").removeprefix("* "))
        for line in block.splitlines()
        if line.strip().startswith(("- ", "* "))
    ]


def _default_analysis_path(workspace_root: Path, presentation_id: str) -> Path:
    slide_brief = workspace_root / "analysis" / "slide_briefs" / f"{presentation_id}.md"
    if slide_brief.exists():
        return slide_brief
    return workspace_root / "analysis" / f"{presentation_id}.md"


def parse_narrative_analysis(markdown: str) -> dict[str, Any]:
    """Extract the public narrative sections needed by decks."""

    themes: list[dict[str, Any]] = []
    matches = list(
        re.finditer(r"^## Major Theme\s+\d+:\s*(?P<title>.+?)\s*$", markdown, flags=re.M)
    )
    if matches:
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
            block = markdown[start:end]
            concise = re.search(r"^## Concise References\s*$", block, flags=re.M)
            if concise is not None:
                block = block[: concise.start()]
            themes.append(
                {
                    "title": _clean_heading(match.group("title")),
                    "has_why_this_matters": bool(
                        re.search(r"^### Why This Matters\s*$", block, flags=re.M)
                    ),
                    "has_slide_seeds": bool(
                        re.search(r"^### Slide Seeds\s*$", block, flags=re.M)
                    ),
                }
            )
        return {
            "source_kind": "legacy-topic-report",
            "executive_summary": _heading_block(markdown, "## Executive Summary"),
            "themes": themes,
            "concise_references": _heading_block(markdown, "## Concise References"),
        }

    key_points = _bullet_items(_heading_block(markdown, "## Key Points"))
    if not key_points:
        key_points = _bullet_items(_heading_block(markdown, "## Key Findings"))
    slide_seeds = _bullet_items(_heading_block(markdown, "## Slide Seeds"))
    if not slide_seeds:
        slide_seeds = _bullet_items(_heading_block(markdown, "## Slide Brief"))
    anchors = _heading_block(markdown, "## Anchor Papers") or _heading_block(
        markdown, "## References And Sources"
    )
    themes = [
        {
            "title": _clean_heading(point),
            "has_why_this_matters": True,
            "has_slide_seeds": bool(slide_seeds),
        }
        for point in key_points
    ]
    return {
        "source_kind": "slide-brief",
        "executive_summary": "\n".join(f"- {point}" for point in key_points),
        "themes": themes,
        "concise_references": anchors,
    }


def _valid_identical_stage_waiver(deck_dir: Path, stage_filenames: Sequence[str]) -> dict[str, Any] | None:
    waiver_path = deck_dir / "stage_checkpoint_waiver.json"
    if not waiver_path.exists():
        return None
    try:
        waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(waiver, dict):
        return None
    if waiver.get("allow_identical_stage_files") is not True:
        return None
    reason = str(waiver.get("reason") or "").strip()
    if len(reason) < 20:
        return None
    applies_to = waiver.get("applies_to")
    if applies_to == "all":
        return waiver
    if isinstance(applies_to, list) and set(str(item) for item in applies_to) >= set(
        stage_filenames
    ):
        return waiver
    return None


def validate_presentation_narrative_contract(
    workspace_root: Path,
    *,
    presentation_id: str,
    analysis_path: str | Path | None = None,
    deck_dir: str | Path | None = None,
    stage_filenames: Sequence[str] = DEFAULT_DECK_STAGE_FILENAMES,
    stale_markers: Sequence[str] = DEFAULT_POST_STAGE14_DECK_STALE_MARKERS,
    public_ledger_headings: Sequence[str] = DEFAULT_PUBLIC_LEDGER_HEADINGS,
    required_draft_markers: Sequence[str] = DEFAULT_REQUIRED_DRAFT_MARKERS,
    required_theme_count: int = 3,
    require_analysis_sha_marker: bool = True,
    blocker_code_prefix: str = "presentation_deck",
    stage_id: str = "presentation",
) -> dict[str, Any]:
    """Validate that a deck is bound to the narrative analysis contract.

    The gate is intentionally content-focused: structural render/export checks
    remain in ``conference_pipeline.presentation.validation`` while this gate
    catches stale scaffold copies, public evidence-ledger leakage, and missing
    narrative-to-deck binding.
    """

    workspace_root = workspace_root.resolve()
    analysis = (
        _resolve_workspace_path(workspace_root, analysis_path)
        if analysis_path is not None
        else _default_analysis_path(workspace_root, presentation_id)
    )
    deck = _resolve_workspace_path(workspace_root, deck_dir or f"decks/{presentation_id}")
    blockers: list[dict[str, Any]] = []
    analysis_themes: list[str] = []
    analysis_sha256: str | None = None

    if not analysis.exists():
        blockers.append(
            {
                "code": f"{blocker_code_prefix}_missing_stage14_narrative",
                "stage_id": stage_id,
                "retryable": True,
                "evidence": _relpath(analysis, workspace_root),
            }
        )
    else:
        analysis_text = analysis.read_text(encoding="utf-8")
        analysis_sha256 = sha256_text(analysis_text)
        narrative = parse_narrative_analysis(analysis_text)
        themes = narrative["themes"]
        analysis_themes = [str(theme.get("title") or "") for theme in themes]
        missing_theme_contract = [
            str(theme.get("title") or f"theme-{index + 1}")
            for index, theme in enumerate(themes[:required_theme_count])
            if not theme.get("has_why_this_matters") or not theme.get("has_slide_seeds")
        ]
        if (
            not narrative.get("executive_summary")
            or len(analysis_themes) < required_theme_count
            or missing_theme_contract
        ):
            blockers.append(
                {
                    "code": f"{blocker_code_prefix}_missing_stage14_narrative",
                    "stage_id": stage_id,
                    "retryable": True,
                    "evidence": _relpath(analysis, workspace_root),
                    "reason": (
                        "Deck generation requires a v1.63 slide brief with "
                        f"{required_theme_count} Key Points and Slide Seeds, "
                        "or a legacy topic report with equivalent narrative sections."
                    ),
                    "themes_missing_required_subsections": missing_theme_contract,
                }
            )

    stage_paths = [deck / filename for filename in stage_filenames]
    missing_stage_files = [_relpath(path, workspace_root) for path in stage_paths if not path.exists()]
    if missing_stage_files:
        blockers.append(
            {
                "code": f"{blocker_code_prefix}_missing_stage_files",
                "stage_id": stage_id,
                "retryable": True,
                "missing_files": missing_stage_files,
            }
        )

    stage_hashes: dict[str, str] = {}
    stale_hits: list[dict[str, str]] = []
    public_ledger_hits: list[dict[str, str]] = []
    draft_text = ""
    for path in stage_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if path.name == "1_draft_slides.md":
            draft_text = text
        stage_hashes[_relpath(path, workspace_root)] = sha256_text(text)
        for marker in stale_markers:
            if marker in text:
                stale_hits.append({"path": _relpath(path, workspace_root), "marker": marker})
        for marker in public_ledger_headings:
            if marker in text:
                public_ledger_hits.append(
                    {"path": _relpath(path, workspace_root), "marker": marker}
                )

    if stale_hits:
        blockers.append(
            {
                "code": f"{blocker_code_prefix}_stale_post_stage14",
                "stage_id": stage_id,
                "retryable": True,
                "stale_markers": stale_hits,
            }
        )
    if public_ledger_hits:
        blockers.append(
            {
                "code": f"{blocker_code_prefix}_public_ledger_heading",
                "stage_id": stage_id,
                "retryable": True,
                "ledger_headings": public_ledger_hits,
            }
        )
    if stage_hashes and len(set(stage_hashes.values())) == 1 and len(stage_hashes) > 1:
        waiver = _valid_identical_stage_waiver(deck, stage_filenames)
        if waiver is None:
            blockers.append(
                {
                    "code": f"{blocker_code_prefix}_stage_files_identical",
                    "stage_id": stage_id,
                    "retryable": True,
                    "stage_hashes": stage_hashes,
                    "reason": (
                        "Post-Stage-14 deck checkpoints must record distinct stage "
                        "outputs or a valid explicit stage_checkpoint_waiver.json."
                    ),
                }
            )

    missing_draft_markers: list[str] = []
    if draft_text:
        for marker in required_draft_markers:
            if marker not in draft_text:
                missing_draft_markers.append(marker)
        for theme in [theme for theme in analysis_themes[:required_theme_count] if theme]:
            if theme not in draft_text:
                missing_draft_markers.append(theme)
        if require_analysis_sha_marker and analysis_sha256:
            digest_marker = f"ANALYSIS_SHA256: {analysis_sha256}"
            if digest_marker not in draft_text:
                missing_draft_markers.append(digest_marker)
    if missing_draft_markers:
        blockers.append(
            {
                "code": f"{blocker_code_prefix}_missing_stage14_narrative",
                "stage_id": stage_id,
                "retryable": True,
                "evidence": _relpath(deck / "1_draft_slides.md", workspace_root),
                "missing_markers": missing_draft_markers,
            }
        )

    return {
        "status": "blocked" if blockers else "pass",
        "blockers": blockers,
        "presentation_id": presentation_id,
        "analysis_path": _relpath(analysis, workspace_root),
        "analysis_sha256": analysis_sha256,
        "stage_hashes": stage_hashes,
        "analysis_theme_count": len(analysis_themes),
        "required_theme_count": required_theme_count,
    }


def validate_delivery_contracts_for_tts_video(
    single_speaker_contract: str | Path,
    notebooklm_contract: str | Path,
) -> dict[str, Any]:
    """Fail closed before TTS/video unless both delivery variants are approved.

    The delivery contracts are separate artifacts because a single-presenter
    read and a NotebookLM-style dialogue can shift timing, turn taking, and the
    way visual details are explained. This gate validates both contracts
    together so one approved variant cannot accidentally authorize the other.
    """

    contract_specs = (
        (Path(single_speaker_contract), "single_speaker"),
        (Path(notebooklm_contract), "notebooklm"),
    )
    blockers: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}

    for path, expected_variant in contract_specs:
        payload = _load_delivery_contract(path, expected_variant, blockers)
        if payload is None:
            continue
        payloads[expected_variant] = payload
        blockers.extend(_validate_delivery_contract_payload(path, payload, expected_variant))

    status = "blocked" if blockers else "pass"
    return {
        "status": status,
        "message": (
            "TTS/video blocked until both delivery variants have approved "
            "conference-narration-review receipts and named source contribution mentions."
            if blockers
            else "TTS/video ready: both delivery variants are approved and source-grounded."
        ),
        "blockers": blockers,
        "contracts": {
            variant: {
                "path": str(path),
                "approve_for_tts": payloads.get(variant, {}).get("approve_for_tts"),
            }
            for path, variant in contract_specs
        },
    }


def _load_delivery_contract(
    path: Path,
    expected_variant: str,
    blockers: list[dict[str, Any]],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_contract_missing",
                "path": str(path),
                "message": "delivery contract is missing",
            }
        )
        return None
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_contract_unreadable",
                "path": str(path),
                "message": f"delivery contract is unreadable: {type(exc).__name__}",
            }
        )
        return None
    if not isinstance(payload, dict):
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_contract_invalid",
                "path": str(path),
                "message": "delivery contract root must be a JSON object",
            }
        )
        return None
    return payload


def _validate_delivery_contract_payload(
    path: Path,
    payload: dict[str, Any],
    expected_variant: str,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    variant = str(payload.get("delivery_variant") or payload.get("style") or "")
    if variant != expected_variant:
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_variant_mismatch",
                "path": str(path),
                "message": f"expected delivery_variant={expected_variant}, found {variant or '<missing>'}",
            }
        )

    required_fields = (
        "presentation_deck_sha256",
        "source_binding_report_sha256",
        "visual_plan_sha256",
        "script_path",
        "script_json_path",
        "timing_map",
        "slide_state_map",
        "narration_review_paths",
        "source_grounding_checks",
    )
    missing = [field for field in required_fields if field not in payload]
    if missing:
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_contract_incomplete",
                "path": str(path),
                "missing_fields": missing,
            }
        )

    review_paths = payload.get("narration_review_paths")
    if not isinstance(review_paths, dict):
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_review_not_approved",
                "path": str(path),
                "message": "narration_review_paths must name deck and series review JSON files",
            }
        )
    else:
        for key in ("deck_review_json", "series_review_json"):
            review_path = _resolve_contract_path(path, review_paths.get(key))
            approved, reason = _review_json_approves_tts(review_path)
            if not approved:
                blockers.append(
                    {
                        "code": f"delivery_{expected_variant}_review_not_approved",
                        "path": str(path),
                        "review_path": str(review_path) if review_path is not None else None,
                        "review_key": key,
                        "message": reason,
                    }
                )

    source_checks = payload.get("source_grounding_checks")
    if not isinstance(source_checks, dict):
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_missing_source_grounding_checks",
                "path": str(path),
                "message": "source_grounding_checks must be an object",
            }
        )
    else:
        required_checks = " ".join(str(item) for item in source_checks.get("required_checks", []))
        for phrase in DELIVERY_REQUIRED_SOURCE_GROUNDING_CHECKS:
            if phrase not in required_checks:
                blockers.append(
                    {
                        "code": f"delivery_{expected_variant}_missing_source_grounding_checks",
                        "path": str(path),
                        "message": f"source_grounding_checks missing required rule: {phrase}",
                    }
                )
        segment_results = source_checks.get("segment_results")
        if not isinstance(segment_results, list) or not segment_results:
            blockers.append(
                {
                    "code": f"delivery_{expected_variant}_missing_named_source_contribution",
                    "path": str(path),
                    "message": "source_grounding_checks.segment_results must be non-empty",
                }
            )
        else:
            for index, result in enumerate(segment_results):
                if not isinstance(result, dict):
                    continue
                if not result.get("mentions_named_source_contribution"):
                    blockers.append(
                        {
                            "code": f"delivery_{expected_variant}_missing_named_source_contribution",
                            "path": str(path),
                            "segment_id": result.get("segment_id") or f"segment-{index + 1}",
                            "message": (
                                "narration must mention which named paper, talk, speaker, "
                                "author, or work contribution supports the claim"
                            ),
                        }
                    )
                if not result.get("mentions_what_the_audience_sees"):
                    blockers.append(
                        {
                            "code": f"delivery_{expected_variant}_missing_visual_narration",
                            "path": str(path),
                            "segment_id": result.get("segment_id") or f"segment-{index + 1}",
                            "message": "narration must mention what the audience sees",
                        }
                    )
                if not result.get("mentions_why_it_matters"):
                    blockers.append(
                        {
                            "code": f"delivery_{expected_variant}_missing_why_it_matters",
                            "path": str(path),
                            "segment_id": result.get("segment_id") or f"segment-{index + 1}",
                            "message": "narration must say why the visual/source claim matters",
                        }
                    )

    if payload.get("approve_for_tts") is not True:
        blockers.append(
            {
                "code": f"delivery_{expected_variant}_not_approved_for_tts",
                "path": str(path),
                "message": "approve_for_tts must be true before TTS/video",
            }
        )
    return blockers


def _resolve_contract_path(contract_path: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (contract_path.parent / candidate).resolve()


def _review_json_approves_tts(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return False, "review path missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "review JSON missing"
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"review JSON unreadable: {type(exc).__name__}"
    if not isinstance(payload, dict):
        return False, "review JSON root must be an object"
    if payload.get("approve_for_tts") is not True:
        return False, "review JSON does not set approve_for_tts=true"
    status = str(payload.get("status") or "").strip().lower()
    if status in {"fail", "failed", "blocked", "needs_revision", "needs-revision"}:
        return False, f"review status blocks TTS: {status}"
    counts = payload.get("finding_counts")
    if isinstance(counts, dict):
        fail_count = counts.get("fail", counts.get("fails", 0))
        try:
            if int(fail_count or 0) > 0:
                return False, "review JSON reports failures"
        except (TypeError, ValueError):
            return False, "review JSON failure count is invalid"
    return True, "approved"
