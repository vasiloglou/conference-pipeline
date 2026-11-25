"""Bounded repair loop receipts for presentation figure quality."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.programmatic_figures import generate_programmatic_figures
from conference_pipeline.presentation.rendered_qc import inspect_rendered_slides
from conference_pipeline.workflow_runs import write_json_atomic

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPAIR_FEEDBACK_FILENAME = "visual_repair_feedback.json"


def run_repair_loop(
    workspace_root: Path,
    presentation_id: str,
    *,
    slides_md: Path,
    generated_dir: Path,
    html_output: Path,
    slide_mapping: Path,
    validation_report: Path,
    visual_qc_report: Path,
    screenshots_dir: Path,
    receipt_output: Path,
    source_pack: Path | None = None,
    ranked_papers: Path | None = None,
    asset_inventory: Path | None = None,
    llm_qc_report: Path | None = None,
    repair_feedback_output: Path | None = None,
    target_svgs_per_slide: float = 0.0,
    max_attempts: int = 2,
    rerender_after_repair: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    """Run a bounded fail-closed validation/QC repair loop."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if target_svgs_per_slide < 0:
        raise ValueError("target_svgs_per_slide must be non-negative")

    slides_path = _resolve_inside(workspace, slides_md, label="slides_md")
    generated_root = _resolve_inside(workspace, generated_dir, label="generated_dir")
    html_path = _resolve_inside(workspace, html_output, label="html_output")
    mapping_path = _resolve_inside(workspace, slide_mapping, label="slide_mapping")
    validation_path = _resolve_inside(workspace, validation_report, label="validation_report")
    visual_qc_path = _resolve_inside(workspace, visual_qc_report, label="visual_qc_report")
    screenshot_root = _resolve_inside(workspace, screenshots_dir, label="screenshots_dir")
    receipt_path = _resolve_inside(workspace, receipt_output, label="receipt_output")
    deck_dir = slides_path.parent
    source_pack_path = _resolve_inside(
        workspace,
        source_pack or deck_dir / "source_pack_receipt.json",
        label="source_pack",
    )
    ranked_papers_path = _resolve_inside(
        workspace,
        ranked_papers or deck_dir / "ranked_papers.json",
        label="ranked_papers",
    )
    inventory_path = _resolve_inside(
        workspace,
        asset_inventory or deck_dir / "asset_inventory.json",
        label="asset_inventory",
    )
    llm_qc_path = _resolve_inside(
        workspace,
        llm_qc_report or generated_root / "visual_llm_qc_report.json",
        label="llm_qc_report",
    )
    repair_feedback_path = _resolve_inside(
        workspace,
        repair_feedback_output or generated_root / DEFAULT_REPAIR_FEEDBACK_FILENAME,
        label="repair_feedback_output",
    )

    attempts: list[dict[str, object]] = []
    final_status = "blocked"
    for attempt_number in range(1, max_attempts + 1):
        before_hashes = _svg_hashes(generated_root)
        validation_before = _validate_svg_files(generated_root)
        rendered_before = _load_rendered_qc_errors(visual_qc_path)
        llm_before = _load_llm_qc_errors(llm_qc_path)
        screenshot_paths = _screenshot_paths(screenshot_root, workspace)

        repair_feedback = _build_visual_repair_feedback(
            workspace,
            presentation_id=presentation_id,
            llm_qc_path=llm_qc_path,
            slide_mapping_path=mapping_path,
            asset_inventory_path=inventory_path,
            issues=llm_before,
        )
        regeneration_result: dict[str, object] | None = None
        repair_action = "none"
        if repair_feedback.get("repairs") and _figure_regeneration_inputs_ready(
            slides_path,
            source_pack_path,
            ranked_papers_path,
        ):
            repair_action = "regenerated_programmatic_figures"
            if not dry_run:
                repair_feedback_path.parent.mkdir(parents=True, exist_ok=True)
                write_json_atomic(repair_feedback_path, repair_feedback)
                regeneration_result = generate_programmatic_figures(
                    workspace,
                    presentation_id,
                    input_slides=slides_path,
                    source_pack=source_pack_path,
                    ranked_papers=ranked_papers_path,
                    asset_inventory=inventory_path,
                    output_slides=slides_path,
                    generated_dir=generated_root,
                    repair_feedback=repair_feedback_path,
                    target_svgs_per_slide=target_svgs_per_slide,
                    max_waivers=0,
                    dry_run=False,
                )
        elif not dry_run:
            repair_action = "conservative_svg_repair"
            _apply_conservative_repairs(generated_root)
        repaired_hashes = before_hashes if dry_run else _svg_hashes(generated_root)
        validation_after = _validate_svg_files(generated_root)
        rerendered = False
        if (
            not dry_run
            and rerender_after_repair
            and repair_action == "regenerated_programmatic_figures"
            and not validation_after
        ):
            _render_and_inspect(
                slides_path=slides_path,
                html_path=html_path,
                mapping_path=mapping_path,
                visual_qc_path=visual_qc_path,
                screenshot_root=screenshot_root,
            )
            rerendered = True
        rendered_after = _load_rendered_qc_errors(visual_qc_path)
        llm_qc_rerun_required = bool(llm_before)
        attempt_status = (
            "passed"
            if not validation_after and not rendered_after and not llm_qc_rerun_required
            else "blocked"
            if llm_qc_rerun_required
            else "blocked"
            if attempt_number == max_attempts
            else "retry"
        )
        attempts.append(
            {
                "attempt": attempt_number,
                "input_svg_hashes": before_hashes,
                "repaired_svg_hashes": repaired_hashes,
                "validation_errors_before": validation_before,
                "validation_errors_after": validation_after,
                "rendered_qc_errors_before": rendered_before,
                "rendered_qc_errors_after": rendered_after,
                "llm_qc_errors_before": llm_before,
                "llm_qc_errors_after": None if llm_qc_rerun_required else [],
                "repair_action": repair_action,
                "repair_feedback_path": _relative(repair_feedback_path, workspace),
                "repair_feedback_repair_count": len(repair_feedback.get("repairs", [])),
                "figure_regeneration": regeneration_result,
                "rerendered_after_repair": rerendered,
                "llm_qc_rerun_required": llm_qc_rerun_required,
                "llm_qc_block_reason": (
                    "fresh governed vision-LLM QC required after repair"
                    if llm_qc_rerun_required
                    else None
                ),
                "screenshot_paths": screenshot_paths,
                "status": attempt_status,
            }
        )
        _write_validation_report(validation_path, validation_after)
        if attempt_status == "passed":
            final_status = "passed"
            break
        if llm_qc_rerun_required:
            break

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "presentation_id": presentation_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": final_status,
        "max_attempts": max_attempts,
        "inputs": [
            _input_record(slides_path, workspace, "slides_md"),
            _input_record(html_path, workspace, "html_output"),
            _input_record(mapping_path, workspace, "slide_mapping"),
        ],
        "outputs": [
            _relative(validation_path, workspace),
            _relative(visual_qc_path, workspace),
            _relative(receipt_path, workspace),
        ],
        "attempts": attempts,
    }
    if not dry_run:
        write_json_atomic(receipt_path, receipt)
    return receipt


def _resolve_inside(workspace: Path, candidate: Path, *, label: str) -> Path:
    path = candidate.expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _relative(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _input_record(path: Path, workspace: Path, kind: str) -> dict[str, str | None]:
    return {
        "kind": kind,
        "path": _relative(path, workspace),
        "sha256": _sha256_file(path) if path.is_file() else None,
    }


def _svg_hashes(generated_dir: Path) -> list[dict[str, str]]:
    if not generated_dir.is_dir():
        return []
    return [
        {"path": path.name, "sha256": _sha256_file(path)}
        for path in sorted(generated_dir.glob("*.svg"))
        if path.is_file()
    ]


def _validate_svg_files(generated_dir: Path) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    if not generated_dir.is_dir():
        return [
            {
                "check_id": "generated_dir_exists",
                "severity": "error",
                "message": "generated SVG directory is missing",
            }
        ]
    svg_paths = sorted(path for path in generated_dir.glob("*.svg") if path.is_file())
    if not svg_paths:
        return [
            {
                "check_id": "no_svg_files",
                "severity": "error",
                "message": "no generated SVG files were found",
            }
        ]
    for path in svg_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "<svg" not in text:
            errors.append(_validation_error(path, "svg_parseable", "file does not contain an SVG root"))
        if "viewBox" not in text and "viewbox" not in text:
            errors.append(_validation_error(path, "svg_viewbox", "SVG is missing viewBox"))
        if not re.search(r"<(path|circle|rect|line|polyline|polygon|text|image|ellipse)\b", text):
            errors.append(_validation_error(path, "svg_nonblank", "SVG has no visible child shapes"))
    return errors


def _validation_error(path: Path, check_id: str, message: str) -> dict[str, object]:
    return {
        "check_id": check_id,
        "severity": "error",
        "path": path.name,
        "message": message,
    }


def _load_rendered_qc_errors(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "check_id": "rendered_qc_report_parseable",
                "severity": "error",
                "message": "rendered QC report is not JSON",
            }
        ]
    errors = loaded.get("errors", [])
    return errors if isinstance(errors, list) else []


def _screenshot_paths(path: Path, workspace: Path) -> list[str]:
    if not path.is_dir():
        return []
    return [
        _relative(screenshot, workspace)
        for screenshot in sorted(path.glob("*.png"))
        if screenshot.is_file()
    ]


def _load_llm_qc_errors(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [
            {
                "check_id": "vision_llm_qc_report_parseable",
                "severity": "blocker",
                "description": "visual LLM QC report is not JSON",
                "screenshot_path": _display_path(path),
            }
        ]
    errors = loaded.get("errors", []) if isinstance(loaded, dict) else []
    return [dict(item) for item in errors if isinstance(item, dict)]


def _build_visual_repair_feedback(
    workspace: Path,
    *,
    presentation_id: str,
    llm_qc_path: Path,
    slide_mapping_path: Path,
    asset_inventory_path: Path,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    mapping = _load_slide_mapping(slide_mapping_path)
    figure_slides = _load_inventory_figure_slides(asset_inventory_path)
    grouped: dict[int, list[dict[str, object]]] = {}
    screenshot_numbers: dict[int, set[int]] = {}
    for issue in issues:
        screenshot_number = _int_or_none(issue.get("slide_number"))
        if screenshot_number is None:
            continue
        slide_number = _markdown_slide_for_screenshot(screenshot_number, mapping)
        if slide_number is None:
            continue
        if figure_slides and slide_number not in figure_slides:
            continue
        grouped.setdefault(slide_number, []).append(
            {
                "check_id": str(issue.get("check_id") or "other"),
                "severity": str(issue.get("severity") or "blocker"),
                "description": str(issue.get("description") or "").strip(),
                "evidence": str(issue.get("evidence") or "").strip(),
                "suggested_fix": str(issue.get("suggested_fix") or "").strip(),
                "screenshot_path": str(issue.get("screenshot_path") or "").strip(),
            }
        )
        screenshot_numbers.setdefault(slide_number, set()).add(screenshot_number)

    repairs = []
    prior_repair_profiles = _load_inventory_repair_profiles(asset_inventory_path)
    for slide_number in sorted(grouped):
        slide_issues = grouped[slide_number]
        repairs.append(
            {
                "slide_number": slide_number,
                "source_screenshot_slide_numbers": sorted(screenshot_numbers.get(slide_number, set())),
                "issue_count": len(slide_issues),
                "check_ids": sorted({str(issue["check_id"]) for issue in slide_issues}),
                "directives": _repair_directives(slide_issues),
                "issues": slide_issues,
            }
        )
    for slide_number in sorted(set(prior_repair_profiles) - set(grouped)):
        profile = prior_repair_profiles[slide_number]
        directives = [
            str(item)
            for item in profile.get("directives", [])
            if str(item).strip()
        ] or ["preserve prior projection-safe visual repair"]
        repairs.append(
            {
                "slide_number": slide_number,
                "source_screenshot_slide_numbers": [
                    item
                    for item in profile.get("source_screenshot_slide_numbers", [])
                    if isinstance(item, int)
                ],
                "issue_count": int(profile.get("issue_count") or 0),
                "check_ids": ["prior_visual_repair"],
                "directives": directives,
                "issues": [
                    {
                        "check_id": "prior_visual_repair",
                        "severity": "preserve",
                        "description": "Preserve prior LLM-guided repair profile while regenerating all programmatic figures.",
                        "evidence": "asset_inventory.json repair_profile",
                        "suggested_fix": "keep projection-safe repair renderer active for this figure",
                        "screenshot_path": "",
                    }
                ],
            }
        )
    repairs.sort(key=lambda item: int(item["slide_number"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "presentation_id": presentation_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_llm_qc_report": _relative_or_display(llm_qc_path, workspace),
        "source_slide_mapping": _relative_or_display(slide_mapping_path, workspace),
        "source_asset_inventory": _relative_or_display(asset_inventory_path, workspace),
        "repair_owner": "scripts/presentations/run_presentation_repair_loop.py",
        "svg_creator_owner": "scripts/presentations/generate_programmatic_figures.py",
        "repairs": repairs,
    }


def _repair_directives(issues: list[dict[str, object]]) -> list[str]:
    directives: list[str] = []
    text = " ".join(
        str(issue.get(field) or "")
        for issue in issues
        for field in ("check_id", "description", "evidence", "suggested_fix")
    ).lower()
    if any(term in text for term in ("illegible", "too small", "unreadable", "tiny")):
        directives.append("use projection-safe large labels and remove tiny body copy")
    if any(term in text for term in ("clipped", "overflow", "ellipsis", "truncated", "cut off")):
        directives.append("remove clipped footer/citation text and shorten labels")
    if any(term in text for term in ("overlap", "connector", "line through")):
        directives.append("route connectors outside label centers with clear padding")
    if any(term in text for term in ("placeholder", "generic")):
        directives.append("replace generic visual with slide-specific evidence-linked structure")
    if not directives:
        directives.append("regenerate as a simplified projection-safe editorial SVG")
    return directives


def _load_slide_mapping(path: Path) -> dict[int, int]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    mapping: dict[int, int] = {}
    for key, value in loaded.items():
        try:
            mapping[int(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return mapping


def _markdown_slide_for_screenshot(
    screenshot_slide_number: int,
    mapping: dict[int, int],
) -> int | None:
    if screenshot_slide_number <= 0 or not mapping:
        return None
    section_index = screenshot_slide_number - 1
    ranges = sorted((section, slide) for slide, section in mapping.items())
    selected: int | None = None
    for index, (start_section, slide_number) in enumerate(ranges):
        next_section = ranges[index + 1][0] if index + 1 < len(ranges) else None
        if section_index >= start_section and (next_section is None or section_index < next_section):
            selected = slide_number
            break
    return selected


def _load_inventory_figure_slides(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    figures = loaded.get("figures", []) if isinstance(loaded, dict) else []
    slides: set[int] = set()
    if isinstance(figures, list):
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            slide_number = _int_or_none(figure.get("slide_number"))
            if slide_number is not None:
                slides.add(slide_number)
    return slides


def _load_inventory_repair_profiles(path: Path) -> dict[int, dict[str, object]]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    figures = loaded.get("figures", []) if isinstance(loaded, dict) else []
    profiles: dict[int, dict[str, object]] = {}
    if isinstance(figures, list):
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            slide_number = _int_or_none(figure.get("slide_number"))
            repair_profile = figure.get("repair_profile")
            if slide_number is not None and isinstance(repair_profile, dict):
                profiles[slide_number] = repair_profile
    return profiles


def _figure_regeneration_inputs_ready(*paths: Path) -> bool:
    return all(path.is_file() for path in paths)


def _render_and_inspect(
    *,
    slides_path: Path,
    html_path: Path,
    mapping_path: Path,
    visual_qc_path: Path,
    screenshot_root: Path,
) -> None:
    build_presentation = _load_reveal_builder()
    build_presentation(slides_path, html_path, slide_mapping_path=mapping_path)
    inspect_rendered_slides(
        html_path,
        output_report=visual_qc_path,
        screenshots_dir=screenshot_root,
    )


def _load_reveal_builder() -> Any:
    script_path = REPO_ROOT / "scripts" / "presentations" / "md_to_revealjs.py"
    spec = importlib.util.spec_from_file_location("_conference_pipeline_md_to_revealjs", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Reveal.js builder: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_presentation


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_path(path: Path) -> str:
    return path.as_posix()


def _relative_or_display(path: Path, workspace: Path) -> str:
    try:
        return _relative(path, workspace)
    except ValueError:
        return _display_path(path)


def _apply_conservative_repairs(generated_dir: Path) -> list[dict[str, str]]:
    for path in (sorted(generated_dir.glob("*.svg")) if generated_dir.is_dir() else []):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "<svg" in text and "viewBox" not in text and "viewbox" not in text:
            text = re.sub(r"<svg\\b", '<svg viewBox="0 0 1280 720"', text, count=1)
            path.write_text(text, encoding="utf-8")
    return _svg_hashes(generated_dir)


def _write_validation_report(path: Path, errors: list[dict[str, object]]) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "failed" if errors else "passed",
            "errors": errors,
        },
    )
