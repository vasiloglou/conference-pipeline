"""Programmatic SVG figure generation for presentation decks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import svgwrite

from conference_pipeline.model_governance import default_model, default_provider
from conference_pipeline.provider_receipts import anthropic_messages_create
from conference_pipeline.presentation.renderer_registry import (
    FIGURE_DESIGN_TASK_ID,
    validate_figure_design_brief,
)
from conference_pipeline.workflow_runs import write_json_atomic

SCHEMA_VERSION = 2
SVG_WIDTH = 1280
SVG_HEIGHT = 720
SVG_CODEGEN_TASK_ID = "presentation.svg_figure_generation"
# Registry-backed so the approved SVG codegen selection follows model
# governance (claude-sonnet-5 as of the July 2026 model-upgrade sweep).
APPROVED_SVG_CODEGEN_MODEL = default_model(SVG_CODEGEN_TASK_ID)
APPROVED_SVG_CODEGEN_PROVIDER = default_provider(SVG_CODEGEN_TASK_ID)
FIGURE_NEED_RE = re.compile(r"FIGURE_NEED:\s*(.*?)(?:\n[A-Z_]+:|\Z)", re.DOTALL)
SLIDE_SPLIT_RE = re.compile(r"\n---\n")
SLIDE_TITLE_RE = re.compile(r"^## Slide\s+(\d+):\s*(.+)$", re.MULTILINE)
COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)

PALETTE = {
    "ink": "#16213e",
    "muted": "#607085",
    "blue": "#2f80ed",
    "green": "#2ca58d",
    "red": "#d95550",
    "yellow": "#f2c94c",
    "purple": "#7b61ff",
    "surface": "#f7fafc",
    "line": "#d9e2ec",
    "white": "#ffffff",
}


@dataclass(frozen=True)
class FigureNeed:
    slide_number: int
    slide_title: str
    need: str
    slide_content: str

    def to_dict(self) -> dict[str, object]:
        return {
            "slide_number": self.slide_number,
            "slide_title": self.slide_title,
            "need": self.need,
            "slide_content": self.slide_content,
        }


@dataclass(frozen=True)
class FigureSpec:
    """Source-grounded contract for one programmatic figure."""

    slide_number: int
    slide_title: str
    figure_need: str
    semantic_goal: str
    paper_or_idea_refs: tuple[str, ...]
    depiction_type: str
    entities: tuple[str, ...]
    relationships: tuple[str, ...]
    quantities: tuple[str, ...]
    visual_encoding: tuple[str, ...]
    must_depict: tuple[str, ...]
    must_not_do: tuple[str, ...]
    source_trace: tuple[dict[str, object], ...]
    model_provider: str = APPROVED_SVG_CODEGEN_PROVIDER
    model_id: str = APPROVED_SVG_CODEGEN_MODEL
    model_task_id: str = SVG_CODEGEN_TASK_ID
    model_execution: str = "governed_required_for_live_codegen"

    def to_dict(self) -> dict[str, object]:
        source_ids = _source_ids_from_trace(self.source_trace)
        source_labels = list(self.paper_or_idea_refs) or _source_labels_from_trace(self.source_trace)
        return {
            "slide_number": self.slide_number,
            "slide_title": self.slide_title,
            "figure_need": self.figure_need,
            "semantic_goal": self.semantic_goal,
            "paper_or_idea_refs": list(self.paper_or_idea_refs),
            "depiction_type": self.depiction_type,
            "entities": list(self.entities),
            "relationships": list(self.relationships),
            "required_entities": list(self.entities),
            "required_relationships": list(self.relationships),
            "quantities": list(self.quantities),
            "visual_encoding": list(self.visual_encoding),
            "must_depict": list(self.must_depict),
            "must_not_do": list(self.must_not_do),
            "source_labels": source_labels,
            "source_ids": source_ids,
            "contribution_note": self.semantic_goal,
            "forbidden_patterns": list(self.must_not_do),
            "source_trace": [dict(item) for item in self.source_trace],
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "model_task_id": self.model_task_id,
            "model_execution": self.model_execution,
        }


def parse_figure_needs(slides_md: str) -> list[dict[str, object]]:
    """Parse FIGURE_NEED comments from consolidated slide markdown."""

    needs: list[dict[str, object]] = []
    for block in _slide_blocks(slides_md):
        title_match = SLIDE_TITLE_RE.search(block)
        if not title_match:
            continue
        slide_number = int(title_match.group(1))
        slide_title = title_match.group(2).strip()
        for comment in COMMENT_RE.finditer(block):
            need_match = FIGURE_NEED_RE.search(comment.group(1).strip())
            if not need_match:
                continue
            need = " ".join(need_match.group(1).split())
            if need:
                needs.append(
                    FigureNeed(
                        slide_number,
                        slide_title,
                        need,
                        block.strip(),
                    ).to_dict()
                )
    return needs


def _slide_records(slides_md: str) -> list[FigureNeed]:
    records: list[FigureNeed] = []
    for block in _slide_blocks(slides_md):
        title_match = SLIDE_TITLE_RE.search(block)
        if not title_match:
            continue
        records.append(
            FigureNeed(
                slide_number=int(title_match.group(1)),
                slide_title=title_match.group(2).strip(),
                need="",
                slide_content=block.strip(),
            )
        )
    return records


def _add_density_fallback_needs(
    slides_md: str,
    needs: list[dict[str, object]],
    *,
    target_svgs_per_slide: float,
    allow_density_fallback: bool = False,
) -> list[dict[str, object]]:
    if target_svgs_per_slide <= 0 or not allow_density_fallback:
        return needs

    slide_records = _slide_records(slides_md)
    if not slide_records:
        return needs

    # The title slide counts toward rendered deck density, but the title-slide
    # owner controls its visual treatment, so density fallback SVGs should land
    # on content slides only.
    required = math.ceil(len(slide_records) * target_svgs_per_slide)
    missing = required - len(needs)
    if missing <= 0:
        return needs

    slides_with_figures = {int(need["slide_number"]) for need in needs}
    fallback_needs = list(needs)
    for record in slide_records:
        if record.slide_number == 0 or "SLIDE_TYPE: title" in record.slide_content:
            continue
        if missing <= 0:
            break
        if record.slide_number in slides_with_figures:
            continue
        fallback = FigureNeed(
            slide_number=record.slide_number,
            slide_title=record.slide_title,
            need=_fallback_figure_need(record),
            slide_content=record.slide_content,
        ).to_dict()
        fallback["density_fallback"] = True
        fallback_needs.append(
            fallback
        )
        slides_with_figures.add(record.slide_number)
        missing -= 1
    return fallback_needs


def _fallback_figure_need(record: FigureNeed) -> str:
    bullets = _extract_bullets(record.slide_content)
    if bullets:
        return f"{record.slide_title}: {_truncate(bullets[0], 68)}"
    return f"{record.slide_title}: map cited evidence to an architecture decision"


def generate_programmatic_figures(
    workspace_root: Path,
    presentation_id: str,
    *,
    input_slides: Path,
    source_pack: Path,
    ranked_papers: Path,
    asset_inventory: Path,
    output_slides: Path,
    generated_dir: Path,
    repair_feedback: Path | None = None,
    figure_specs_input: Path | None = None,
    figure_specs_output: Path | None = None,
    target_svgs_per_slide: float = 0.0,
    allow_density_fallback: bool = False,
    live_sonnet_specs: bool = False,
    sonnet_batch_size: int = 8,
    sonnet_timeout_seconds: float = 240.0,
    max_waivers: int = 0,
    dry_run: bool = False,
) -> dict[str, object]:
    """Generate SVGs for FIGURE_NEED markers and write an asset inventory."""

    workspace = workspace_root.expanduser().resolve()
    if not workspace.is_dir():
        raise ValueError(f"workspace must be an existing directory: {workspace}")
    input_path = _resolve_inside(workspace, input_slides, label="input_slides")
    source_pack_path = _resolve_inside(workspace, source_pack, label="source_pack")
    ranked_papers_path = _resolve_inside(workspace, ranked_papers, label="ranked_papers")
    inventory_path = _resolve_inside(workspace, asset_inventory, label="asset_inventory")
    output_path = _resolve_inside(workspace, output_slides, label="output_slides")
    generated_root = _resolve_inside(workspace, generated_dir, label="generated_dir")
    figure_specs_path = _resolve_inside(
        workspace,
        figure_specs_output if figure_specs_output is not None else generated_dir / "figure_specs.json",
        label="figure_specs_output",
    )
    figure_specs_input_path = (
        _resolve_inside(workspace, figure_specs_input, label="figure_specs_input")
        if figure_specs_input is not None
        else None
    )
    repair_feedback_path = (
        _resolve_inside(workspace, repair_feedback, label="repair_feedback")
        if repair_feedback is not None
        else None
    )
    if max_waivers < 0:
        raise ValueError("max_waivers must be non-negative")
    if target_svgs_per_slide < 0:
        raise ValueError("target_svgs_per_slide must be non-negative")
    if sonnet_batch_size <= 0:
        raise ValueError("sonnet_batch_size must be positive")
    if sonnet_timeout_seconds <= 0:
        raise ValueError("sonnet_timeout_seconds must be positive")
    if figure_specs_input_path is not None and live_sonnet_specs:
        raise ValueError("--figure-specs-input and --live-sonnet-specs are mutually exclusive")

    slides_md = input_path.read_text(encoding="utf-8")
    needs = parse_figure_needs(slides_md)
    needs = _add_density_fallback_needs(
        slides_md,
        needs,
        target_svgs_per_slide=target_svgs_per_slide,
        allow_density_fallback=allow_density_fallback,
    )
    input_records = [
        _input_record(input_path, workspace, "slides_markdown"),
        _input_record(source_pack_path, workspace, "source_pack"),
        _input_record(ranked_papers_path, workspace, "ranked_papers"),
    ]
    if figure_specs_input_path is not None:
        input_records.append(_input_record(figure_specs_input_path, workspace, "figure_specs_input"))
    repair_profiles = _load_repair_feedback(repair_feedback_path) if repair_feedback_path else {}
    if repair_feedback_path is not None:
        input_records.append(_input_record(repair_feedback_path, workspace, "repair_feedback"))
    source_context = _load_source_context(
        source_pack_path=source_pack_path,
        ranked_papers_path=ranked_papers_path,
    )
    if figure_specs_input_path is not None:
        initial_specs = _load_figure_specs_input(figure_specs_input_path)
        reused_sonnet_receipt = _load_reused_sonnet_receipt(figure_specs_input_path)
    else:
        reused_sonnet_receipt = None
        initial_specs = [
            build_figure_spec(
                slide_number=int(need["slide_number"]),
                slide_title=str(need["slide_title"]),
                need=str(need["need"]),
                slide_content=str(need.get("slide_content", "")),
                source_context=source_context,
            )
            for need in needs
        ]
    sonnet_receipt: dict[str, object] | None = None
    sonnet_progress_path = (
        generated_root / "sonnet_figure_spec_progress.jsonl"
        if live_sonnet_specs and not dry_run
        else None
    )
    if sonnet_progress_path is not None:
        sonnet_progress_path.parent.mkdir(parents=True, exist_ok=True)
        sonnet_progress_path.write_text("", encoding="utf-8")
    if live_sonnet_specs and not dry_run:
        initial_specs, sonnet_receipt = _refine_specs_with_sonnet(
            initial_specs,
            batch_size=sonnet_batch_size,
            timeout_seconds=sonnet_timeout_seconds,
            progress_path=sonnet_progress_path,
        )
    elif reused_sonnet_receipt is not None:
        sonnet_receipt = reused_sonnet_receipt
    model_governance_status = (
        "live_sonnet_invoked"
        if live_sonnet_specs and sonnet_receipt is not None
        else (
            "live_sonnet_specs_reused_local_render"
            if figure_specs_input_path is not None and sonnet_receipt is not None
            else "deterministic_fallback_no_live_model_call"
        )
    )
    specs_by_slide_need = {
        (spec.slide_number, spec.figure_need): spec for spec in initial_specs
    }
    missing_specs = [
        (int(need["slide_number"]), str(need["need"]))
        for need in needs
        if (int(need["slide_number"]), str(need["need"])) not in specs_by_slide_need
    ]
    if missing_specs:
        sample = ", ".join(f"slide {slide}" for slide, _ in missing_specs[:5])
        raise ValueError(f"figure_specs_input is missing specs for {len(missing_specs)} figures: {sample}")
    if not dry_run:
        generated_root.mkdir(parents=True, exist_ok=True)
        for stale_svg in generated_root.glob("slide_*.svg"):
            stale_svg.unlink()

    figures: list[dict[str, object]] = []
    figure_specs: list[dict[str, object]] = []
    waivers: list[dict[str, object]] = []
    generated_by_slide: dict[int, dict[str, object]] = {}

    for need in needs:
        slide_number = int(need["slide_number"])
        need_text = str(need["need"])
        slide_content = str(need.get("slide_content", ""))
        density_fallback = bool(need.get("density_fallback"))
        repair_profile = repair_profiles.get(slide_number)
        figure_spec = specs_by_slide_need[(slide_number, need_text)]
        figure_design_brief = build_figure_design_brief(
            figure_spec,
            slide_content=slide_content,
        )
        brief_errors = validate_figure_design_brief(figure_design_brief)
        if brief_errors:
            raise ValueError(f"figure_design_brief validation failed: {json.dumps(brief_errors)}")
        figure_spec_payload = figure_spec.to_dict()
        figure_spec_payload["figure_design_brief"] = figure_design_brief
        figure_specs.append(figure_spec_payload)
        figure_type = _figure_type(
            need_text,
            slide_title=str(need["slide_title"]),
            slide_content=slide_content,
        )
        if figure_spec.depiction_type != "landscape":
            figure_type = figure_spec.depiction_type
        if _requires_waiver(need_text):
            waivers.append(
                {
                    "slide_number": slide_number,
                    "slide_title": need["slide_title"],
                    "need": need_text,
                    "reason": "figure need requests unsupported non-programmatic visual material",
                }
            )
            continue

        slug = _slugify(need_text)
        svg_path = generated_root / f"slide_{slide_number:03d}_{slug}.svg"
        svg_rel = _relative(svg_path, workspace)
        if not dry_run:
            generated_root.mkdir(parents=True, exist_ok=True)
            _render_svgwrite(
                svg_path,
                slide_number=slide_number,
                slide_title=str(need["slide_title"]),
                need=need_text,
                slide_content=slide_content,
                figure_type=figure_type,
                density_fallback=density_fallback,
                repair_profile=repair_profile,
                figure_spec=figure_spec,
            )
        digest = _sha256_file(svg_path) if svg_path.exists() else None
        figure = {
            "slide_number": slide_number,
            "slide_title": need["slide_title"],
            "need": need_text,
            "figure_type": figure_type,
            "figure_spec": figure_spec.to_dict(),
            "model_governance": {
                "task_id": SVG_CODEGEN_TASK_ID,
                "provider": APPROVED_SVG_CODEGEN_PROVIDER,
                "model_id": APPROVED_SVG_CODEGEN_MODEL,
                "status": model_governance_status,
            },
            "figure_design_governance": {
                "task_id": FIGURE_DESIGN_TASK_ID,
                "provider": APPROVED_SVG_CODEGEN_PROVIDER,
                "model_id": APPROVED_SVG_CODEGEN_MODEL,
                "status": "deterministic_local_from_blueprint",
            },
            "figure_design_brief": figure_design_brief,
            "renderer": "svgwrite",
            "renderer_family": figure_design_brief["renderer_family"],
            "path": svg_rel,
            "sha256": digest,
            "production_evidence": bool(digest) and not dry_run,
            "semantic_review_ref": _relative(generated_root / "semantic_review.json", workspace),
            "status": "generated" if not dry_run else "planned",
        }
        if density_fallback:
            figure["density_fallback"] = True
        if repair_profile:
            figure["repair_profile"] = {
                "source": "visual_llm_qc_feedback",
                "issue_count": len(repair_profile.get("issues", [])),
                "directives": list(repair_profile.get("directives", [])),
                "source_screenshot_slide_numbers": list(
                    repair_profile.get("source_screenshot_slide_numbers", [])
                ),
            }
        figures.append(figure)
        generated_by_slide[slide_number] = figure

    output_markdown = _inject_generated_figures(
        slides_md,
        output_path=output_path,
        generated_by_slide=generated_by_slide,
        workspace=workspace,
    )
    status = "blocked" if len(waivers) > max_waivers else ("planned" if dry_run else "complete")
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "presentation_id": presentation_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "max_waivers": max_waivers,
        "model_governance": {
            "task_id": SVG_CODEGEN_TASK_ID,
            "provider": APPROVED_SVG_CODEGEN_PROVIDER,
            "model_id": APPROVED_SVG_CODEGEN_MODEL,
            "status": model_governance_status,
        },
        "figure_design_governance": {
            "task_id": FIGURE_DESIGN_TASK_ID,
            "provider": APPROVED_SVG_CODEGEN_PROVIDER,
            "model_id": APPROVED_SVG_CODEGEN_MODEL,
            "status": "deterministic_local_from_blueprint",
        },
        "density_fallback": {
            "requested_target_svgs_per_slide": target_svgs_per_slide,
            "enabled": allow_density_fallback,
        },
        "inputs": input_records,
        "figure_specs": figure_specs,
        "sonnet_figure_spec_receipt": sonnet_receipt,
        "figures": figures,
        "waivers": waivers,
        "outputs": [
            _relative(output_path, workspace),
            _relative(inventory_path, workspace),
            _relative(figure_specs_path, workspace),
            _relative(generated_root, workspace) + "/",
        ],
    }
    if sonnet_receipt is not None:
        inventory["outputs"].append(
            _relative(generated_root / "sonnet_figure_spec_receipt.json", workspace)
        )
        inventory["outputs"].append(
            _relative(generated_root / "sonnet_figure_spec_progress.jsonl", workspace)
        )

    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_markdown, encoding="utf-8")
        figure_specs_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            figure_specs_path,
            {
                "schema_version": SCHEMA_VERSION,
                "presentation_id": presentation_id,
                "generated_at": inventory["generated_at"],
                "model_governance": inventory["model_governance"],
                "sonnet_figure_spec_receipt": sonnet_receipt,
                "figure_specs": figure_specs,
            },
        )
        if sonnet_receipt is not None:
            write_json_atomic(generated_root / "sonnet_figure_spec_receipt.json", sonnet_receipt)
        write_json_atomic(inventory_path, inventory)
    return inventory


def _slide_blocks(slides_md: str) -> list[str]:
    return [block.strip() for block in SLIDE_SPLIT_RE.split(slides_md) if block.strip()]


def _clean_slide_content(block: str) -> str:
    return COMMENT_RE.sub("", block).strip()


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


def _source_ids_from_trace(source_trace: tuple[dict[str, object], ...]) -> list[str]:
    ids: list[str] = []
    for record in source_trace:
        for key in ("source_id", "paper_id", "artifact_id", "url"):
            value = str(record.get(key) or "").strip()
            if value:
                ids.append(value)
                break
    return _unique_preserve_order(ids, limit=8)


def _source_labels_from_trace(source_trace: tuple[dict[str, object], ...]) -> list[str]:
    labels: list[str] = []
    for record in source_trace:
        title = str(record.get("title") or record.get("source") or "").strip()
        if title:
            labels.append(_short_paper_label(title))
    return _unique_preserve_order(labels, limit=8)


def _input_record(path: Path, workspace: Path, kind: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": _relative(path, workspace),
        "sha256": _sha256_file(path),
    }


def _requires_waiver(need: str) -> bool:
    lower = need.lower()
    return "unsupported" in lower or "photorealistic reconstruction" in lower


def _load_repair_feedback(path: Path) -> dict[int, dict[str, object]]:
    """Load normalized visual-QC repair feedback keyed by markdown slide."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    repairs = payload.get("repairs", []) if isinstance(payload, dict) else []
    profiles: dict[int, dict[str, object]] = {}
    if not isinstance(repairs, list):
        return profiles
    for repair in repairs:
        if not isinstance(repair, dict):
            continue
        try:
            slide_number = int(repair.get("slide_number"))
        except (TypeError, ValueError):
            continue
        issues = repair.get("issues")
        directives = repair.get("directives")
        source_slides = repair.get("source_screenshot_slide_numbers")
        profiles[slide_number] = {
            "issues": issues if isinstance(issues, list) else [],
            "directives": directives if isinstance(directives, list) else [],
            "source_screenshot_slide_numbers": source_slides
            if isinstance(source_slides, list)
            else [],
        }
    return profiles


def _load_source_context(*, source_pack_path: Path, ranked_papers_path: Path) -> dict[str, object]:
    ranked_payload = _load_json_object(ranked_papers_path)
    source_pack_payload = _load_json_object(source_pack_path)
    papers = ranked_payload.get("papers") if isinstance(ranked_payload, dict) else None
    normalized_papers: list[dict[str, object]] = []
    if isinstance(papers, list):
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "").strip()
            if not title:
                continue
            normalized_papers.append(
                {
                    "rank": paper.get("rank"),
                    "title": title,
                    "paper_id": paper.get("paper_id"),
                    "url": paper.get("url"),
                    "evidence_snippet": str(paper.get("evidence_snippet") or "").strip(),
                    "source": paper.get("source"),
                }
            )
    source_counts = {}
    if isinstance(source_pack_payload, dict) and isinstance(source_pack_payload.get("counts"), dict):
        source_counts = dict(source_pack_payload["counts"])
    return {
        "papers": normalized_papers,
        "source_counts": source_counts,
        "source_pack_generated_at": source_pack_payload.get("generated_at")
        if isinstance(source_pack_payload, dict)
        else None,
    }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_figure_specs_input(path: Path) -> list[FigureSpec]:
    payload = _load_json_object(path)
    records = payload.get("figure_specs")
    if not isinstance(records, list):
        raise ValueError("figure_specs_input must contain a figure_specs list")
    specs: list[FigureSpec] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"figure_specs_input record {index} must be an object")
        try:
            specs.append(
                FigureSpec(
                    slide_number=int(record["slide_number"]),
                    slide_title=str(record["slide_title"]),
                    figure_need=str(record["figure_need"]),
                    semantic_goal=str(record["semantic_goal"]),
                    paper_or_idea_refs=tuple(_string_sequence(record.get("paper_or_idea_refs"))),
                    depiction_type=str(record["depiction_type"]),
                    entities=tuple(_string_sequence(record.get("entities"))),
                    relationships=tuple(_string_sequence(record.get("relationships"))),
                    quantities=tuple(_string_sequence(record.get("quantities"))),
                    visual_encoding=tuple(_string_sequence(record.get("visual_encoding"))),
                    must_depict=tuple(_string_sequence(record.get("must_depict"))),
                    must_not_do=tuple(_string_sequence(record.get("must_not_do"))),
                    source_trace=tuple(
                        item for item in record.get("source_trace", []) if isinstance(item, dict)
                    ),
                    model_provider=str(
                        record.get("model_provider") or APPROVED_SVG_CODEGEN_PROVIDER
                    ),
                    model_id=str(record.get("model_id") or APPROVED_SVG_CODEGEN_MODEL),
                    model_task_id=str(record.get("model_task_id") or SVG_CODEGEN_TASK_ID),
                    model_execution=str(record.get("model_execution") or "loaded_from_receipt"),
                )
            )
        except KeyError as exc:
            raise ValueError(f"figure_specs_input record {index} is missing {exc.args[0]}") from exc
    return specs


def _load_reused_sonnet_receipt(path: Path) -> dict[str, object] | None:
    payload = _load_json_object(path)
    receipt = payload.get("sonnet_figure_spec_receipt")
    if isinstance(receipt, dict) and receipt:
        return receipt
    sidecar = path.parent / "sonnet_figure_spec_receipt.json"
    if sidecar.exists():
        sidecar_payload = _load_json_object(sidecar)
        if sidecar_payload:
            return sidecar_payload
    return None


def _string_sequence(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _refine_specs_with_sonnet(
    specs: list[FigureSpec],
    *,
    batch_size: int,
    timeout_seconds: float,
    progress_path: Path | None = None,
) -> tuple[list[FigureSpec], dict[str, object]]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for --live-sonnet-specs")
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for --live-sonnet-specs") from exc

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
    refined: list[FigureSpec] = []
    batches: list[dict[str, object]] = []
    batch_total = math.ceil(len(specs) / batch_size)
    for batch_index, start in enumerate(range(0, len(specs), batch_size), start=1):
        batch = specs[start : start + batch_size]
        prompt = _sonnet_spec_prompt(batch)
        request_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        slide_numbers = [spec.slide_number for spec in batch]
        _append_sonnet_progress(
            progress_path,
            {
                "event": "batch_started",
                "batch_index": batch_index,
                "batch_total": batch_total,
                "figure_count": len(batch),
                "slide_numbers": slide_numbers,
                "request_sha256": request_sha256,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        )
        print(
            f"sonnet figure spec batch {batch_index}/{batch_total}: slides {slide_numbers}",
            file=sys.stderr,
            flush=True,
        )
        response = anthropic_messages_create(
            client,
            caller=__file__,
            model=APPROVED_SVG_CODEGEN_MODEL,
            max_tokens=min(16000, max(6000, batch_size * 900)),
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _anthropic_response_text(response)
        parsed = _extract_json_object(text)
        returned = parsed.get("figures") if isinstance(parsed, dict) else None
        by_slide = {
            _int_or_none(item.get("slide_number")): item
            for item in returned
            if isinstance(item, dict)
        } if isinstance(returned, list) else {}
        for base in batch:
            refined.append(_merge_sonnet_spec(base, by_slide.get(base.slide_number)))
        usage = getattr(response, "usage", None)
        batch_record = {
            "batch_index": batch_index,
            "slide_numbers": slide_numbers,
            "request_sha256": request_sha256,
            "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "response_chars": len(text),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "provider_receipt_path": getattr(response.receipt, "receipt_path", None),
        }
        batches.append(batch_record)
        _append_sonnet_progress(
            progress_path,
            {
                "event": "batch_completed",
                "batch_total": batch_total,
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                **batch_record,
            },
        )
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "task_id": SVG_CODEGEN_TASK_ID,
        "provider": APPROVED_SVG_CODEGEN_PROVIDER,
        "model_id": APPROVED_SVG_CODEGEN_MODEL,
        "status": "complete",
        "request_count": len(batches),
        "figure_count": len(specs),
        "batch_size": batch_size,
        "batches": batches,
    }
    return refined, receipt


def _append_sonnet_progress(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _sonnet_spec_prompt(specs: list[FigureSpec]) -> str:
    payload = {
        "instructions": (
            "Revise these programmatic SVG FigureSpec records. Return strict JSON only. "
            "Make each figure concrete and source-grounded. Prefer mechanisms, entities, "
            "axes, molecules, workflows, code panes, 3D scenes, or benchmark encodings over "
            "decorative bullet boxes. Do not invent numeric values. Keep depiction_type one "
            "of: architecture, benchmark_chart, scaling_chart, spatial_task_panel, "
            "molecular_scene, temporal_tracking, code_pipeline, agent_workflow, "
            "retrieval_graph, tradeoff_matrix, timeline, ladder, risk_surface, "
            "evidence_action_map, constraint_triangle, intervention_flow, "
            "watchlist_timeline, source_backbone, tenant_boundary, "
            "alignment_landscape, prompt_flow, technique_overlap, landscape."
        ),
        "output_schema": {
            "figures": [
                {
                    "slide_number": "integer matching input",
                    "semantic_goal": "short depiction goal",
                    "depiction_type": "allowed depiction type",
                    "entities": ["3-5 concrete labels to print in the SVG"],
                    "relationships": ["2-4 source-grounded relationships"],
                    "quantities": ["only quantities already present in input"],
                    "visual_encoding": ["3-5 drawing instructions/primitives"],
                    "must_depict": ["3-6 specific visual requirements"],
                    "must_not_do": ["generic/copy/fake-chart prohibitions"],
                }
            ]
        },
        "figures": [spec.to_dict() for spec in specs],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _anthropic_response_text(response: object) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def _extract_json_object(text: str) -> dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Sonnet FigureSpec response did not contain a JSON object")
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Sonnet FigureSpec response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Sonnet FigureSpec response JSON must be an object")
    return payload


def _merge_sonnet_spec(base: FigureSpec, override: dict[str, object] | None) -> FigureSpec:
    if not isinstance(override, dict):
        return base
    depiction_type = str(override.get("depiction_type") or base.depiction_type)
    if depiction_type not in {
        "architecture",
        "benchmark_chart",
        "scaling_chart",
        "spatial_task_panel",
        "molecular_scene",
        "temporal_tracking",
        "code_pipeline",
        "agent_workflow",
        "retrieval_graph",
        "tradeoff_matrix",
        "timeline",
        "ladder",
        "risk_surface",
        "evidence_action_map",
        "constraint_triangle",
        "intervention_flow",
        "watchlist_timeline",
        "source_backbone",
        "tenant_boundary",
        "alignment_landscape",
        "prompt_flow",
        "technique_overlap",
        "landscape",
    }:
        depiction_type = base.depiction_type
    depiction_type = _normalize_semantic_depiction_type(base, override, depiction_type)
    return FigureSpec(
        slide_number=base.slide_number,
        slide_title=base.slide_title,
        figure_need=base.figure_need,
        semantic_goal=_string_field(override, "semantic_goal", base.semantic_goal),
        paper_or_idea_refs=base.paper_or_idea_refs,
        depiction_type=depiction_type,
        entities=tuple(_string_list_field(override, "entities", list(base.entities), limit=5)),
        relationships=tuple(
            _string_list_field(override, "relationships", list(base.relationships), limit=5)
        ),
        quantities=tuple(_string_list_field(override, "quantities", list(base.quantities), limit=5)),
        visual_encoding=tuple(
            _string_list_field(override, "visual_encoding", list(base.visual_encoding), limit=6)
        ),
        must_depict=tuple(
            _string_list_field(override, "must_depict", list(base.must_depict), limit=6)
        ),
        must_not_do=tuple(
            _string_list_field(override, "must_not_do", list(base.must_not_do), limit=6)
        ),
        source_trace=base.source_trace,
        model_execution="anthropic_live_codegen",
    )


def _string_field(payload: dict[str, object], key: str, fallback: str) -> str:
    value = payload.get(key)
    return str(value).strip() if isinstance(value, str) and value.strip() else fallback


def _string_list_field(
    payload: dict[str, object],
    key: str,
    fallback: list[str],
    *,
    limit: int,
) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return fallback[:limit]
    items = [str(item).strip() for item in value if str(item).strip()]
    return _unique_preserve_order(items or fallback, limit=limit)


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _normalize_semantic_depiction_type(
    base: FigureSpec,
    override: dict[str, object],
    depiction_type: str,
) -> str:
    """Map vague live specs onto concrete renderers when the spec asks for one."""

    combined = " ".join(
        [
            base.figure_need,
            base.slide_title,
            str(override.get("semantic_goal") or ""),
            " ".join(_string_list_field(override, "entities", list(base.entities), limit=8)),
            " ".join(_string_list_field(override, "must_depict", list(base.must_depict), limit=8)),
            " ".join(
                _string_list_field(override, "visual_encoding", list(base.visual_encoding), limit=8)
            ),
        ]
    ).lower()
    if "ladder" in combined or "rung" in combined:
        return "ladder"
    if "triangle" in combined or "feasible output" in combined or "feasible region" in combined:
        return "constraint_triangle"
    if _contains_any(combined, ("probe stage", "intervention stage", "causal circuit")):
        return "intervention_flow"
    if _contains_any(combined, ("watchlist", "trigger condition", "waypoint", "what to watch")):
        return "watchlist_timeline"
    if _contains_any(
        combined,
        ("source backbone", "raw sources", "extraction layer", "verified output"),
    ):
        return "source_backbone"
    if _contains_any(
        combined,
        (
            "tenant",
            "data leakage",
            "leakage path",
            "shared model layer",
            "token edit boundary",
            "cross-tenant",
        ),
    ):
        return "tenant_boundary"
    if _contains_any(
        combined,
        (
            "multimodal alignment",
            "alignment complexity",
            "visual grounding alignment",
            "contextual intent alignment",
            "refusal feature",
            "copy-paste faithfulness",
        ),
    ):
        return "alignment_landscape"
    if _contains_any(
        combined,
        (
            "reference image",
            "text prompt",
            "generated output",
            "few-shot",
            "masked token",
            "remasking",
            "prompt field",
        ),
    ):
        return "prompt_flow"
    if _contains_any(combined, ("venn", "overlap", "diffusion core", "cross-domain")):
        return "technique_overlap"
    if _contains_any(
        combined,
        (
            "risk surface",
            "attack surface",
            "attack-surface",
            "attack vector",
            "system boundary",
            "capability frontier",
            "split landscape",
            "threat node",
            "capability and risk",
        ),
    ):
        return "risk_surface"
    if _contains_any(
        combined,
        (
            "evidence-to-action",
            "evidence to action",
            "decision point",
            "feedback arrow",
            "leadership action",
        ),
    ):
        return "evidence_action_map"
    return depiction_type


def build_figure_spec(
    *,
    slide_number: int,
    slide_title: str,
    need: str,
    slide_content: str,
    source_context: dict[str, object],
) -> FigureSpec:
    """Build an auditable source-grounded specification for one SVG figure."""

    bullets = _extract_bullets(slide_content)
    metrics = tuple(_extract_metrics(f"{slide_title}\n{need}\n{slide_content}"))
    matched_papers = _match_source_papers(
        slide_title=slide_title,
        need=need,
        bullets=bullets,
        source_context=source_context,
    )
    depiction_type = _depiction_type(slide_title=slide_title, need=need, slide_content=slide_content)
    entities = tuple(
        _unique_preserve_order(
            [
                *_extract_entities(f"{slide_title} {need}"),
                *[
                    _short_paper_label(str(paper.get("title") or ""))
                    for paper in matched_papers[:3]
                    if paper.get("title")
                ],
            ],
            limit=4,
        )
    )
    if not entities:
        entities = tuple(_keywords(f"{slide_title} {need}", fallback=["Evidence", "Mechanism"], count=4))
    relationships = tuple(_relationships_for_depiction(depiction_type, entities))
    visual_encoding = tuple(_visual_encoding_for_depiction(depiction_type, metrics))
    paper_refs = tuple(_short_paper_label(str(paper.get("title") or "")) for paper in matched_papers[:3])
    source_trace = tuple(_source_trace_record(paper) for paper in matched_papers[:3])
    must_depict = tuple(
        _unique_preserve_order(
            [
                _depiction_requirement(depiction_type),
                *entities[:4],
                *metrics[:3],
            ],
            limit=8,
        )
    )
    must_not_do = (
        "Do not copy slide bullets into decorative boxes",
        "Do not use generic template phrases as the main content",
        "Do not invent numeric chart values when the slide/source has no metric",
        "Do not render a requested visual example as abstract labels only",
    )
    semantic_goal = _semantic_goal(slide_title=slide_title, need=need, depiction_type=depiction_type)
    return FigureSpec(
        slide_number=slide_number,
        slide_title=slide_title,
        figure_need=need,
        semantic_goal=semantic_goal,
        paper_or_idea_refs=paper_refs,
        depiction_type=depiction_type,
        entities=entities,
        relationships=relationships,
        quantities=metrics,
        visual_encoding=visual_encoding,
        must_depict=must_depict,
        must_not_do=must_not_do,
        source_trace=source_trace,
    )


def build_figure_design_brief(
    figure_spec: FigureSpec,
    *,
    slide_content: str,
) -> dict[str, object]:
    """Create the renderer-selection brief consumed by registry validation."""

    source_labels = _metadata_list(slide_content, "APPROVED_SOURCE_LABELS")
    source_ids = _source_ids_from_trace(figure_spec.source_trace)
    return {
        "blueprint_id": _metadata_value(slide_content, "BLUEPRINT_ID"),
        "slide_id": _metadata_value(slide_content, "BLUEPRINT_SLIDE_ID"),
        "cluster_id": _metadata_value(slide_content, "BLUEPRINT_CLUSTER_ID"),
        "source_labels": source_labels,
        "source_ids": source_ids,
        "claim_to_visualize": figure_spec.semantic_goal,
        "depiction_type": figure_spec.depiction_type,
        "renderer_family": _renderer_family_for_spec(figure_spec),
        "required_labels": list(figure_spec.must_depict),
        "required_entities": list(figure_spec.entities),
        "required_relationships": list(figure_spec.relationships),
        "contribution_note": figure_spec.semantic_goal,
        "forbidden_patterns": list(figure_spec.must_not_do),
        "narration_hook": _metadata_value(slide_content, "NARRATION_BEAT")
        or figure_spec.semantic_goal,
        "quantitative_chart": _is_quantitative_chart(figure_spec),
    }


def _renderer_family_for_spec(figure_spec: FigureSpec) -> str:
    if figure_spec.depiction_type == "retrieval_graph":
        return "direct_svg"
    return "direct_svg"


def _is_quantitative_chart(figure_spec: FigureSpec) -> bool:
    return figure_spec.depiction_type in {"benchmark_chart", "scaling_chart"} and bool(
        figure_spec.quantities
    )


def _metadata_value(slide_content: str, key: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", slide_content, re.MULTILINE)
    if not match:
        return None
    value = " ".join(match.group(1).split())
    return value or None


def _metadata_list(slide_content: str, key: str) -> list[str]:
    value = _metadata_value(slide_content, key)
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _match_source_papers(
    *,
    slide_title: str,
    need: str,
    bullets: list[str],
    source_context: dict[str, object],
) -> list[dict[str, object]]:
    papers = source_context.get("papers")
    if not isinstance(papers, list):
        return []
    query_tokens = _content_tokens(" ".join([slide_title, need, *bullets]))
    scored: list[tuple[int, int, dict[str, object]]] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, dict):
            continue
        text = f"{paper.get('title', '')} {paper.get('evidence_snippet', '')}"
        paper_tokens = _content_tokens(text)
        overlap = len(query_tokens & paper_tokens)
        rank_bonus = max(0, 5 - int(paper.get("rank") or index + 1))
        scored.append((overlap + rank_bonus, -index, paper))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [paper for score, _, paper in scored if score > 0][:4] or [
        paper for _, _, paper in scored[:3]
    ]


def _content_tokens(text: str) -> set[str]:
    ignored = {
        "diagram",
        "chart",
        "graph",
        "showing",
        "visual",
        "figure",
        "slide",
        "overview",
        "summary",
        "current",
        "models",
        "model",
        "reasoning",
        "benchmark",
        "benchmarks",
        "language",
        "paper",
        "papers",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
        if token not in ignored
    }


def _depiction_type(*, slide_title: str, need: str, slide_content: str) -> str:
    text = f"{slide_title} {need} {slide_content}".lower()
    if _contains_any(
        text,
        (
            "attack surface",
            "risk surface",
            "security risk",
            "threat landscape",
            "adversarial",
            "privacy risk",
        ),
    ):
        return "risk_surface"
    if _contains_any(text, ("protein-ligand", "binding pocket", "molecule", "molecular")):
        return "molecular_scene"
    if _contains_any(text, ("code snippet", "recipe card", "python code", "code generation", "pys spatial", "pyspatial")):
        return "code_pipeline"
    if _contains_any(text, ("video frames", "tracking", "temporal", "motion vector")):
        return "temporal_tracking"
    if _contains_any(text, ("3d", "geometry", "geometric", "spatial", "voxel", "point cloud", "gaussian", "scene")):
        return "spatial_task_panel"
    if _contains_any(text, ("agent", "tool", "orchestrator", "workflow", "scaffold")):
        return "agent_workflow"
    if _contains_any(text, ("retrieval", "graph", "knowledge graph", "topology", "decision tree")):
        return "retrieval_graph"
    if _contains_any(text, ("scaling", "gpu", "tokens", "throughput", "latency", "kv-cache", "kv cache")):
        return "scaling_chart"
    if _contains_any(
        text,
        (
            "bar chart",
            "benchmark comparison",
            "performance",
            "accuracy",
            "score",
            "scatter plot",
            "pie chart",
            "table",
            "curve",
            "chart",
            "plot",
        ),
    ):
        return "benchmark_chart"
    if _contains_any(text, ("constraint triangle", "feasible region", "feasible output")):
        return "constraint_triangle"
    if _contains_any(text, ("venn", "matrix", "trade-off", "trade off")):
        return "tradeoff_matrix"
    if _contains_any(text, ("intervention flow", "causal circuit", "probe stage")):
        return "intervention_flow"
    if _contains_any(text, ("watchlist", "what to watch", "trigger condition")):
        return "watchlist_timeline"
    if _contains_any(text, ("source backbone", "raw sources", "verified output")):
        return "source_backbone"
    if _contains_any(text, ("architecture", "pipeline", "system", "framework")):
        return "architecture"
    if _contains_any(text, ("timeline", "roadmap", "evolution", "trajectory")):
        return "timeline"
    if _contains_any(
        text,
        (
            "diagram",
            "schematic",
            "visualization",
            "visualisation",
            "overview",
            "comparison",
            "cyclic",
            "conceptual",
            "mechanism",
            "process",
            "flowchart",
        ),
    ):
        return "architecture"
    return "landscape"


def _extract_entities(text: str) -> list[str]:
    entities: list[str] = []
    patterns = (
        r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)+\b",
        r"\b[A-Z]{2,}(?:-[A-Z0-9]+)?\b",
        r"\b[A-Z][a-z]+(?:Spatial|Bench|Anchor|Thinker|Scene|Scan|VLM|LLM|GPT|OCR)\b",
    )
    for pattern in patterns:
        entities.extend(match.group(0) for match in re.finditer(pattern, text))
    titleish = re.findall(r"\b(?:Geek|OmniSpatial|IV-Bench|PySpatial|VideoAnchor|GPT4Scene|Mordal|ViPER|ScholScan|ProxyThinker)\b", text)
    entities.extend(titleish)
    return _unique_preserve_order([_clean_entity(entity) for entity in entities if entity], limit=10)


def _clean_entity(entity: str) -> str:
    return re.sub(r"\s+", " ", entity.replace("_", " ")).strip(" :-")


def _unique_preserve_order(items: list[str] | tuple[str, ...], *, limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def _short_paper_label(title: str) -> str:
    title = re.split(r":|\(|\[", title, maxsplit=1)[0].strip()
    words = title.split()
    return " ".join(words[:5]) if len(words) > 5 else title


def _source_trace_record(paper: dict[str, object]) -> dict[str, object]:
    return {
        "rank": paper.get("rank"),
        "title": paper.get("title"),
        "paper_id": paper.get("paper_id"),
        "url": paper.get("url"),
        "evidence_snippet": paper.get("evidence_snippet"),
        "source": paper.get("source"),
    }


def _relationships_for_depiction(depiction_type: str, entities: tuple[str, ...]) -> list[str]:
    if depiction_type == "code_pipeline":
        return ["visual/geometric input -> Python program", "program execution -> spatial answer"]
    if depiction_type == "molecular_scene":
        return ["protein pocket constrains ligand pose", "geometric fit drives binding prediction"]
    if depiction_type == "temporal_tracking":
        return ["frame sequence preserves object identity", "motion path reveals spatial continuity"]
    if depiction_type == "spatial_task_panel":
        return ["2D observation -> 3D structure", "geometry task -> reasoning failure/success"]
    if depiction_type == "agent_workflow":
        return ["goal -> tool calls", "observations -> verifier -> next action"]
    if depiction_type == "retrieval_graph":
        return ["query -> topology-aware retrieval", "retrieved nodes -> grounded answer"]
    if depiction_type in {"benchmark_chart", "scaling_chart"}:
        return ["condition changes x-axis", "performance/reliability changes y-axis"]
    if depiction_type == "architecture":
        return ["input modality -> model components", "representation -> output decision"]
    if entities:
        return [f"{entities[0]} anchors the depicted mechanism"]
    return ["evidence maps to mechanism"]


def _visual_encoding_for_depiction(depiction_type: str, metrics: tuple[str, ...]) -> list[str]:
    encodings = {
        "code_pipeline": ["left-to-right transformation", "code pane", "computed spatial answer"],
        "molecular_scene": ["protein contour", "ligand nodes", "binding arrows"],
        "temporal_tracking": ["frame strip", "object trail", "motion arrows"],
        "spatial_task_panel": ["3D axes", "object geometry", "task callouts"],
        "agent_workflow": ["agent/tool nodes", "feedback loop", "verification gate"],
        "retrieval_graph": ["query node", "retrieved graph neighborhood", "answer path"],
        "scaling_chart": ["axis chart", "measured labels", "memory/throughput curve"],
        "benchmark_chart": ["comparison bars", "metric labels", "confidence caveat"],
        "architecture": ["module stack", "data-flow arrows", "interface labels"],
        "tradeoff_matrix": ["2x2 matrix", "explicit trade-off labels"],
        "timeline": ["ordered milestones", "transition markers"],
    }
    base = encodings.get(depiction_type, ["concept map", "source-grounded labels", "mechanism arrows"])
    if metrics:
        return [*base, "slide/source metrics only"]
    return base


def _depiction_requirement(depiction_type: str) -> str:
    requirements = {
        "code_pipeline": "show the visual problem becoming executable code",
        "molecular_scene": "show ligand, pocket, and geometric fit",
        "temporal_tracking": "show frames plus object movement over time",
        "spatial_task_panel": "show a concrete spatial or 3D task",
        "agent_workflow": "show agent/tool/verifier interaction",
        "retrieval_graph": "show query-to-neighborhood retrieval path",
        "scaling_chart": "show axes tied to real slide/source quantities",
        "benchmark_chart": "show benchmark comparison with metric caveat",
        "architecture": "show mechanism, not just outcomes",
        "tradeoff_matrix": "show explicit axes and quadrant trade-offs",
        "timeline": "show ordered progression",
    }
    return requirements.get(depiction_type, "show a source-grounded mechanism")


def _semantic_goal(*, slide_title: str, need: str, depiction_type: str) -> str:
    return (
        f"Depict {slide_title} as a {depiction_type.replace('_', ' ')}: "
        f"{_truncate(need, 150)}"
    )


def _figure_type(need: str, *, slide_title: str = "", slide_content: str = "") -> str:
    need_lower = need.lower()
    title_lower = slide_title.lower()
    context_lower = f"{slide_title} {need} {slide_content}".lower()
    if _contains_any(need_lower, ("overview diagram", "summary graphic", "landscape map")):
        return "diagram"
    if _contains_any(
        context_lower,
        (
            "radar",
            "cybersecurity",
            "security risk",
            "attack surface",
            "risk surface",
            "security surface",
            "threat landscape",
            "adversarial",
            "privacy risk",
        ),
    ):
        return "radar"
    if _contains_any(context_lower, ("attention map", "retrieval head", "factual token")):
        return "attention_map"
    if _contains_any(need_lower, ("timeline", "roadmap", "evolution", "trajectory")):
        return "timeline"
    if _contains_any(need_lower, ("bar chart", "bar graph")):
        return "bar_chart"
    if _contains_any(need_lower, ("graph", "chart", "plot", "curve", "benchmark comparison")):
        return "line_chart"
    if _contains_any(need_lower, ("kv-cache", "kv cache", "memory allocation", "memory footprint", "cache dominance", "token eviction", "memory system")):
        return "memory_breakdown"
    if _contains_any(context_lower, ("trade-off", "trade off", "matrix", "skill-capability")):
        return "matrix"
    if _contains_any(need_lower, ("search", "evolutionary", "selection", "candidate", "decision tree")):
        return "search"
    if _contains_any(need_lower, ("flowchart", "pipeline", "process", "workflow", "flow diagram", "context flow")):
        return "flowchart"
    if _contains_any(
        context_lower,
        (
            "agent",
            "agents",
            "agentic",
            "scaffold",
            "deliberation",
            "rationale",
            "orchestrator",
        ),
    ):
        return "deliberation"
    if _contains_any(context_lower, ("heatmap", "table")):
        return "heatmap"
    if _contains_any(need_lower, ("frontier",)):
        return "line_chart"
    if _contains_any(context_lower, ("equation", "mathematical", "formula")):
        return "equation"
    if _contains_any(
        context_lower,
        (
            "loss landscape",
            "flat minima",
            "sharp minima",
            "optimization landscape",
            "energy landscape",
            "fitness landscape",
        ),
    ):
        return "surface"
    if _contains_any(
        context_lower,
        (
            "3d",
            "spatial",
            "voxel",
            "point cloud",
            "gaussian",
            "scene",
            "geometry",
            "geometric",
            "binding pocket",
            "protein-ligand",
            "motion vector",
            "robot",
            "robotics",
        ),
    ):
        return "spatial"
    return "diagram"


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    for phrase in phrases:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug[:48] or "figure"


def _render_svgwrite(
    path: Path,
    *,
    slide_number: int,
    slide_title: str,
    need: str,
    slide_content: str,
    figure_type: str,
    density_fallback: bool = False,
    repair_profile: dict[str, object] | None = None,
    figure_spec: FigureSpec | None = None,
) -> None:
    drawing = svgwrite.Drawing(
        str(path),
        size=(f"{SVG_WIDTH}px", f"{SVG_HEIGHT}px"),
        viewBox=f"0 0 {SVG_WIDTH} {SVG_HEIGHT}",
        profile="full",
    )
    context = _visual_context(slide_title, need, slide_content, figure_spec=figure_spec)
    entity_trace = "; ".join(str(item) for item in context.get("entities", []) if str(item).strip())
    if entity_trace:
        drawing.set_desc(
            title=slide_title,
            desc=f"Source-grounded figure entities: {entity_trace}",
        )
    _draw_editorial_background(drawing)
    if density_fallback:
        _draw_density_fallback(drawing, context)
        drawing.save(pretty=True)
        return
    if repair_profile:
        _draw_repaired_editorial_figure(
            drawing,
            slide_number=slide_number,
            slide_title=slide_title,
            need=need,
            figure_type=figure_type,
            context=context,
            repair_profile=repair_profile,
        )
        drawing.save(pretty=True)
        return

    _draw_editorial_header(drawing, slide_number, slide_title, need, context)

    if _draw_semantic_figure(drawing, context):
        pass
    elif figure_type == "flowchart":
        _draw_editorial_flowchart(drawing, context)
    elif figure_type == "timeline":
        _draw_editorial_timeline(drawing, context)
    elif figure_type == "matrix":
        _draw_editorial_matrix(drawing, context)
    elif figure_type == "search":
        _draw_editorial_search(drawing, context)
    elif figure_type == "memory_breakdown":
        _draw_editorial_memory(drawing, context)
    elif figure_type == "attention_map":
        _draw_editorial_attention_map(drawing, context)
    elif figure_type == "deliberation":
        _draw_editorial_deliberation(drawing, context)
    elif figure_type == "radar":
        _draw_editorial_radar(drawing, context)
    elif figure_type == "heatmap":
        _draw_editorial_heatmap(drawing, context)
    elif figure_type == "bar_chart":
        _draw_editorial_bar_chart(drawing, context)
    elif figure_type == "line_chart":
        _draw_editorial_line_chart(drawing, context)
    elif figure_type == "equation":
        _draw_editorial_equation(drawing, context)
    elif figure_type == "surface":
        _draw_editorial_surface(drawing, context)
    elif figure_type == "spatial":
        _draw_editorial_spatial(drawing, context)
    else:
        _draw_editorial_landscape(drawing, context)

    _draw_evidence_strip(drawing, context)
    drawing.save(pretty=True)


EDITORIAL = {
    "navy": "#1a1a2e",
    "panel": "#222742",
    "panel_2": "#2f3354",
    "ice": "#a8d8ea",
    "coral": "#ff6b6b",
    "green": "#51cf66",
    "amber": "#ffd166",
    "purple": "#9b8cff",
    "white": "#ffffff",
    "gray": "#d9dce8",
    "muted": "#8e96b8",
}


def _visual_context(
    slide_title: str,
    need: str,
    slide_content: str,
    *,
    figure_spec: FigureSpec | None = None,
) -> dict[str, object]:
    spec_dict = figure_spec.to_dict() if figure_spec is not None else {}
    return {
        "title": slide_title,
        "need": need,
        "content": slide_content,
        "bullets": _extract_bullets(slide_content),
        "source_refs": _extract_source_refs(slide_content),
        "metrics": _extract_metrics(slide_content),
        "figure_spec": spec_dict,
        "entities": list(spec_dict.get("entities", [])) if spec_dict else [],
        "relationships": list(spec_dict.get("relationships", [])) if spec_dict else [],
        "depiction_type": spec_dict.get("depiction_type") if spec_dict else "",
        "paper_or_idea_refs": list(spec_dict.get("paper_or_idea_refs", [])) if spec_dict else [],
        "quantities": list(spec_dict.get("quantities", [])) if spec_dict else [],
        "keywords": _keywords(
            f"{slide_title} {need} {slide_content}",
            fallback=["synthesis", "memory", "scaffold", "evaluation"],
            count=8,
        ),
    }


def _extract_bullets(text: str) -> list[str]:
    text = _clean_slide_content(text)
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullet = stripped[2:].strip()
            if bullet.lower().startswith("references:"):
                continue
            bullets.append(bullet)
    return bullets[:8]


def _extract_source_refs(text: str) -> list[str]:
    refs: list[str] = []
    for match in re.finditer(r"SOURCE_REFS:\s*(.+)", text):
        refs.extend(item.strip() for item in match.group(1).split(",") if item.strip())
    for match in re.finditer(r"SOURCE:\s*(.+)", text):
        refs.append(match.group(1).strip())
    return refs[:3]


def _extract_metrics(text: str) -> list[str]:
    metric_re = re.compile(
        r"\b(?:~|<|>)?\d+(?:\.\d+)?\s?(?:x|%|K|M|B|T|ms|s|tokens?|FLOPs?|params?|layers?)\b",
        re.IGNORECASE,
    )
    metrics: list[str] = []
    for match in metric_re.finditer(text):
        value = match.group(0).strip()
        if value not in metrics:
            metrics.append(value)
    return metrics[:6]


def _draw_editorial_background(drawing: svgwrite.Drawing) -> None:
    drawing.add(drawing.rect(insert=(0, 0), size=(SVG_WIDTH, SVG_HEIGHT), fill=EDITORIAL["navy"]))
    drawing.add(drawing.rect(insert=(36, 34), size=(1208, 652), rx=18, fill=EDITORIAL["panel"], opacity=0.72))
    drawing.add(
        drawing.line(
            start=(52, 120),
            end=(1228, 120),
            stroke=EDITORIAL["ice"],
            stroke_width=1.5,
            opacity=0.28,
        )
    )


def _draw_editorial_header(
    drawing: svgwrite.Drawing,
    slide_number: int,
    slide_title: str,
    need: str,
    context: dict[str, object],
) -> None:
    _draw_badge(drawing, 74, 62, f"SLIDE {slide_number:02d}", EDITORIAL["ice"])
    title_lines = _wrap_text(slide_title, 62)[:2]
    for offset, line in enumerate(title_lines):
        drawing.add(
            drawing.text(
                line,
                insert=(192, 78 + offset * 31),
                fill=EDITORIAL["white"],
                font_size=29,
                font_family="Arial",
                font_weight="bold",
            )
        )

def _draw_badge(
    drawing: svgwrite.Drawing,
    x: int,
    y: int,
    text: str,
    color: str,
    *,
    width: int | None = None,
) -> None:
    box_w = width or max(92, min(210, 18 + len(text) * 8))
    drawing.add(drawing.rect(insert=(x, y - 22), size=(box_w, 32), rx=16, fill=color, opacity=0.18))
    drawing.add(
        drawing.text(
            text,
            insert=(x + box_w / 2, y),
            text_anchor="middle",
            fill=color,
            font_size=13,
            font_family="Arial",
            font_weight="bold",
        )
    )


def _draw_semantic_figure(drawing: svgwrite.Drawing, context: dict[str, object]) -> bool:
    depiction_type = str(context.get("depiction_type") or "")
    if not depiction_type:
        return False
    if depiction_type == "molecular_scene":
        _draw_semantic_molecular_scene(drawing, context)
    elif depiction_type == "temporal_tracking":
        _draw_semantic_temporal_tracking(drawing, context)
    elif depiction_type == "code_pipeline":
        _draw_semantic_code_pipeline(drawing, context)
    elif depiction_type == "spatial_task_panel":
        if _is_radial_signal_context(context):
            _draw_semantic_radial_signals(drawing, context)
        else:
            _draw_semantic_spatial_task_panel(drawing, context)
    elif depiction_type == "agent_workflow":
        _draw_semantic_agent_workflow(drawing, context)
    elif depiction_type == "retrieval_graph":
        _draw_semantic_retrieval_graph(drawing, context)
    elif depiction_type in {"benchmark_chart", "scaling_chart"}:
        _draw_semantic_source_chart(drawing, context, depiction_type)
    elif depiction_type == "architecture":
        _draw_semantic_architecture(drawing, context)
    elif depiction_type == "tradeoff_matrix":
        _draw_semantic_tradeoff_matrix(drawing, context)
    elif depiction_type == "timeline":
        _draw_semantic_timeline(drawing, context)
    elif depiction_type == "ladder":
        _draw_semantic_ladder(drawing, context)
    elif depiction_type == "risk_surface":
        _draw_semantic_risk_surface(drawing, context)
    elif depiction_type == "evidence_action_map":
        _draw_semantic_evidence_action_map(drawing, context)
    elif depiction_type == "constraint_triangle":
        _draw_semantic_constraint_triangle(drawing, context)
    elif depiction_type == "intervention_flow":
        _draw_semantic_intervention_flow(drawing, context)
    elif depiction_type == "watchlist_timeline":
        _draw_semantic_watchlist_timeline(drawing, context)
    elif depiction_type == "source_backbone":
        _draw_semantic_source_backbone(drawing, context)
    elif depiction_type == "tenant_boundary":
        _draw_semantic_tenant_boundary(drawing, context)
    elif depiction_type == "alignment_landscape":
        _draw_semantic_alignment_landscape(drawing, context)
    elif depiction_type == "prompt_flow":
        _draw_semantic_prompt_flow(drawing, context)
    elif depiction_type == "technique_overlap":
        _draw_semantic_technique_overlap(drawing, context)
    elif depiction_type == "landscape":
        if _semantic_landscape_override(drawing, context):
            pass
        else:
            return False
    else:
        return False
    return True


def _semantic_labels(context: dict[str, object], count: int, fallback: list[str]) -> list[str]:
    candidates = [
        *[str(item) for item in context.get("entities", [])],
        *[str(item) for item in context.get("paper_or_idea_refs", [])],
        *[str(item) for item in context.get("keywords", [])],
        *fallback,
    ]
    labels = _unique_preserve_order([_truncate(label, 34) for label in candidates], limit=count)
    while len(labels) < count:
        labels.append(fallback[min(len(labels), len(fallback) - 1)])
    return labels[:count]


def _semantic_metrics(context: dict[str, object], fallback: list[str]) -> list[str]:
    metrics = [str(item) for item in context.get("metrics", []) if str(item).strip()]
    return _unique_preserve_order([*metrics, *fallback], limit=max(4, len(fallback)))


def _source_chart_metrics(context: dict[str, object]) -> list[str]:
    spec = context.get("figure_spec") if isinstance(context.get("figure_spec"), dict) else {}
    candidates = [
        *[str(item) for item in context.get("metrics", []) if str(item).strip()],
        *[str(item) for item in context.get("quantities", []) if str(item).strip()],
        *[str(item) for item in spec.get("quantities", []) if str(item).strip()],
    ]
    placeholders = (
        "source metric",
        "relative",
        "no invented",
        "trend",
        "only quantities",
        "already present",
    )
    metrics: list[str] = []
    for candidate in candidates:
        cleaned = " ".join(candidate.split())
        lower = cleaned.lower()
        if not cleaned or any(placeholder in lower for placeholder in placeholders):
            continue
        if cleaned not in metrics:
            metrics.append(_truncate(cleaned, 18))
        if len(metrics) >= 4:
            break
    return metrics


def _draw_semantic_callout(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    spec = context.get("figure_spec") if isinstance(context.get("figure_spec"), dict) else {}
    goal = str(spec.get("semantic_goal") or context.get("need") or "")
    refs = [str(item) for item in context.get("paper_or_idea_refs", []) if str(item).strip()]
    if refs:
        goal = f"{_truncate(goal, 92)} | anchored by {', '.join(refs[:2])}"
    drawing.add(
        drawing.rect(
            insert=(92, 538),
            size=(1096, 64),
            rx=14,
            fill=EDITORIAL["panel_2"],
            opacity=0.9,
            stroke=EDITORIAL["ice"],
            stroke_width=1.5,
        )
    )
    _draw_wrapped_text(drawing, goal, 640, 564, 16, EDITORIAL["gray"], anchor="middle", width=92)


def _draw_semantic_timeline(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Training cutoff", "New papers", "Live ingestion", "Updated eval", "Decision"],
    )
    need_text = f"{context.get('title', '')} {context.get('need', '')}".lower()
    if _contains_any(need_text, ("funnel", "diversity collaps", "trajectory diversity")):
        x0, y0 = 168, 252
        funnel = [(x0, y0), (1035, y0 + 72), (1035, y0 + 188), (x0, y0 + 270)]
        drawing.add(
            drawing.polygon(
                points=funnel,
                fill=EDITORIAL["ice"],
                opacity=0.12,
                stroke=EDITORIAL["ice"],
                stroke_width=3,
            )
        )
        colors = [EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"], EDITORIAL["purple"]]
        for index, color in enumerate(colors):
            points = [
                (x0 + 28, y0 + 42 + index * 50),
                (420, y0 + 68 + index * 28),
                (710, y0 + 112 + index * 12),
                (1000, y0 + 134 + index * 3),
            ]
            drawing.add(drawing.polyline(points=points, fill="none", stroke=color, stroke_width=5, opacity=0.86))
        ticks = [(x0 + 40, labels[0]), (598, labels[1]), (1016, labels[2])]
        for index, (x, label) in enumerate(ticks):
            drawing.add(drawing.circle(center=(x, 514), r=25, fill=[EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]][index], opacity=0.92))
            drawing.add(drawing.text(str(index + 1), insert=(x - 7, 522), fill=EDITORIAL["navy"], font_size=20, font_family="Arial", font_weight="bold"))
            _draw_wrapped_text(drawing, label, x, 560, 15, EDITORIAL["white"], anchor="middle", width=18)
        intervention = _first_label_containing(labels, ("pruning", "intervention"), "Strategic Pruning Intervention")
        drawing.add(drawing.line(start=(598, 500), end=(598, 380), stroke=EDITORIAL["green"], stroke_width=4, opacity=0.9))
        drawing.add(drawing.polygon(points=[(598, 360), (584, 386), (612, 386)], fill=EDITORIAL["green"], opacity=0.9))
        _draw_wrapped_text(drawing, intervention, 598, 330, 17, EDITORIAL["green"], anchor="middle", width=24)
        diversity = _first_label_containing(labels, ("diversity", "trajectory"), "Trajectory Diversity")
        _draw_wrapped_text(drawing, diversity, 206, 218, 18, EDITORIAL["ice"], anchor="middle", width=24)
        _draw_wrapped_text(drawing, "Collapsed reasoning path", 1006, 220, 15, EDITORIAL["coral"], anchor="middle", width=22)
        return

    x0, y = 150, 340
    drawing.add(drawing.line(start=(x0, y), end=(1130, y), stroke=EDITORIAL["muted"], stroke_width=6, opacity=0.65))
    for index, label in enumerate(labels[:5]):
        x = x0 + index * 245
        color = [EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["purple"]][index]
        drawing.add(drawing.circle(center=(x, y), r=31, fill=color, opacity=0.92))
        drawing.add(drawing.text(str(index + 1), insert=(x - 8, y + 10), fill=EDITORIAL["navy"], font_size=23, font_family="Arial", font_weight="bold"))
        _draw_wrapped_text(drawing, label, x, y + 82, 16, EDITORIAL["white"], anchor="middle", width=17)


def _draw_semantic_force_field_decoder(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Text Embedding", "Force-Field Terms", "Bond Stretching", "Electrostatics", "Molecular Representation"],
    )
    x_positions = [120, 330, 570, 815, 1040]
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        color = [EDITORIAL["ice"], EDITORIAL["purple"], EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["green"]][index]
        drawing.add(drawing.rect(insert=(x - 80, 280), size=(160, 110), rx=16, fill=color, opacity=0.17, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, 333, 17, EDITORIAL["white"], anchor="middle", width=15)
        if index < 4:
            _draw_editorial_arrow(drawing, x + 84, 335, x_positions[index + 1] - 86, 335, EDITORIAL["muted"])
    atoms = [(960, 470), (1016, 432), (1082, 470), (1028, 514)]
    for index, point in enumerate(atoms):
        drawing.add(drawing.circle(center=point, r=20, fill=[EDITORIAL["coral"], EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"]][index], opacity=0.82))
        if index:
            drawing.add(drawing.line(start=atoms[index - 1], end=point, stroke=EDITORIAL["white"], stroke_width=4, opacity=0.65))
    _draw_wrapped_text(drawing, "angle bending + vdW", 610, 468, 16, EDITORIAL["gray"], anchor="middle", width=26)


def _draw_semantic_electron_density(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Atomic Nucleus Grid", "AI-Predicted Electron Density Cloud", "Hamiltonian Operator H", "Wavefunction Output psi", "Error-Detection Gap"],
    )
    for row in range(3):
        for col in range(3):
            cx, cy = 260 + col * 72, 272 + row * 72
            drawing.add(drawing.circle(center=(cx, cy), r=14, fill=EDITORIAL["white"], opacity=0.86))
            drawing.add(drawing.circle(center=(cx, cy), r=42, fill=EDITORIAL["ice"], opacity=0.12 + 0.04 * (row + col)))
    _draw_wrapped_text(drawing, labels[0], 334, 526, 18, EDITORIAL["ice"], anchor="middle", width=26)
    blocks = [(560, labels[2], EDITORIAL["purple"]), (760, "Neural Hamiltonian Predictor", EDITORIAL["green"]), (980, labels[3], EDITORIAL["coral"])]
    for x, label, color in blocks:
        drawing.add(drawing.rect(insert=(x - 88, 288), size=(176, 96), rx=16, fill=color, opacity=0.18, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, 340, 17, EDITORIAL["white"], anchor="middle", width=18)
    _draw_editorial_arrow(drawing, 648, 336, 670, 336, EDITORIAL["muted"])
    _draw_editorial_arrow(drawing, 848, 336, 892, 336, EDITORIAL["muted"])
    _draw_wrapped_text(drawing, labels[1], 334, 210, 19, EDITORIAL["green"], anchor="middle", width=32)
    _draw_wrapped_text(drawing, labels[4], 820, 502, 17, EDITORIAL["amber"], anchor="middle", width=30)


def _draw_semantic_molecular_scene(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    context_text = _semantic_context_text(context)
    if _contains_any(context_text, ("force-field", "force field", "bond stretching", "electrostatics")):
        _draw_semantic_force_field_decoder(drawing, context)
        return
    if _contains_any(context_text, ("electron density", "hamiltonian", "wavefunction")):
        _draw_semantic_electron_density(drawing, context)
        return
    labels = _semantic_labels(
        context,
        5,
        ["Protein Pocket", "Generated Ligand", "Binding Geometry", "Prompt", "Generator"],
    )
    protein_label = _first_label_containing(labels, ("protein", "pocket"), "Protein Pocket")
    ligand_label = _first_label_containing(labels, ("ligand", "molecule"), "Generated Ligand")
    geometry_label = _first_label_containing(labels, ("geometry", "fit", "affinity"), "Binding Geometry")
    prompt_label = _first_label_containing(labels, ("prompt", "toxicity"), "Prompt: low toxicity, high affinity")
    generator_label = _first_label_containing(labels, ("generator", "dit", "graph"), "Graph DiT Generator")
    pocket_points = [
        (305, 272),
        (380, 214),
        (505, 218),
        (580, 292),
        (560, 416),
        (470, 486),
        (342, 458),
        (278, 362),
    ]
    drawing.add(
        drawing.polygon(
            points=pocket_points,
            fill=EDITORIAL["ice"],
            opacity=0.16,
            stroke=EDITORIAL["ice"],
            stroke_width=5,
        )
    )
    ligand = [(752, 320), (830, 282), (918, 330), (890, 430), (778, 442)]
    for index, point in enumerate(ligand):
        color = [EDITORIAL["coral"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["ice"]][index]
        drawing.add(drawing.circle(center=point, r=30, fill=color, opacity=0.88, stroke=EDITORIAL["white"], stroke_width=2))
        if index:
            drawing.add(drawing.line(start=ligand[index - 1], end=point, stroke=EDITORIAL["white"], stroke_width=7, opacity=0.7))
    _draw_editorial_arrow(drawing, 670, 370, 590, 370, EDITORIAL["coral"])
    _draw_wrapped_text(drawing, protein_label, 420, 530, 20, EDITORIAL["ice"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, ligand_label, 842, 508, 20, EDITORIAL["white"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, geometry_label, 640, 316, 18, EDITORIAL["amber"], anchor="middle", width=20)
    drawing.add(drawing.rect(insert=(112, 222), size=(220, 70), rx=14, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["green"], stroke_width=3, opacity=0.92))
    _draw_wrapped_text(drawing, prompt_label, 222, 260, 15, EDITORIAL["green"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, generator_label, 640, 462, 17, EDITORIAL["purple"], anchor="middle", width=24)


def _draw_semantic_temporal_tracking(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Frame T1", "Frame T2", "Frame T3", "Object Trail", "World State"],
    )
    frame_xs = [205, 490, 775]
    object_points: list[tuple[int, int]] = []
    for index, x in enumerate(frame_xs):
        drawing.add(drawing.rect(insert=(x, 250), size=(205, 140), rx=10, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=3))
        drawing.add(drawing.line(start=(x + 26, 350 - index * 12), end=(x + 145, 288 + index * 8), stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.5))
        point = (x + 58 + index * 28, 330 - index * 18)
        object_points.append(point)
        drawing.add(drawing.circle(center=point, r=18, fill=EDITORIAL["coral"], opacity=0.9))
        _draw_wrapped_text(drawing, labels[index], x + 102, 425, 16, EDITORIAL["gray"], anchor="middle", width=16)
    for start, end in zip(object_points, object_points[1:]):
        _draw_editorial_arrow(drawing, start[0] + 24, start[1], end[0] - 24, end[1], EDITORIAL["green"])
    drawing.add(drawing.rect(insert=(486, 514), size=(308, 50), rx=12, fill=EDITORIAL["panel_2"], opacity=0.94))
    _draw_wrapped_text(drawing, labels[3], 640, 544, 18, EDITORIAL["white"], anchor="middle", width=34)
    _draw_wrapped_text(drawing, labels[4], 640, 220, 20, EDITORIAL["amber"], anchor="middle", width=30)


def _draw_semantic_code_pipeline(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 4, ["Visual geometry", "Python program", "3D constraints", "Answer"])
    x_positions = [122, 388, 668, 946]
    colors = [EDITORIAL["ice"], EDITORIAL["purple"], EDITORIAL["amber"], EDITORIAL["green"]]
    for index, (x, label, color) in enumerate(zip(x_positions, labels, colors)):
        drawing.add(drawing.rect(insert=(x, 262), size=(194, 150), rx=16, fill=color, opacity=0.18, stroke=color, stroke_width=4))
        if index == 1:
            drawing.add(drawing.text("Python program", insert=(x + 20, 292), fill=EDITORIAL["ice"], font_size=15, font_family="Arial", font_weight="bold"))
            for line_index, code in enumerate(["for p in points:", "  fit_plane(p)", "return angle"]):
                drawing.add(drawing.text(code, insert=(x + 20, 322 + line_index * 26), fill=EDITORIAL["white"], font_size=17, font_family="Courier New"))
        elif index == 2:
            drawing.add(drawing.line(start=(x + 48, 362), end=(x + 138, 304), stroke=color, stroke_width=5))
            drawing.add(drawing.line(start=(x + 48, 362), end=(x + 145, 376), stroke=color, stroke_width=5))
            drawing.add(drawing.circle(center=(x + 48, 362), r=9, fill=color))
            drawing.add(drawing.circle(center=(x + 138, 304), r=9, fill=color))
            drawing.add(drawing.circle(center=(x + 145, 376), r=9, fill=color))
        else:
            _draw_wrapped_text(drawing, label, x + 97, 336, 21, EDITORIAL["white"], anchor="middle", width=14)
        _draw_wrapped_text(drawing, label, x + 97, 452, 17, EDITORIAL["gray"], anchor="middle", width=16)
        if index < 3:
            _draw_editorial_arrow(drawing, x + 205, 337, x + 254, 337, EDITORIAL["muted"])


def _is_radial_signal_context(context: dict[str, object]) -> bool:
    text = " ".join(
        [
            str(context.get("title", "")),
            str(context.get("need", "")),
            str(context.get("depiction_type", "")),
            " ".join(str(item) for item in context.get("entities", [])),
        ]
    ).lower()
    return "radial" in text and "signal" in text


def _draw_semantic_radial_signals(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _radial_signal_labels(context)
    cx, cy = 640, 330
    radius_x, radius_y = 348, 164
    colors = [
        EDITORIAL["ice"],
        EDITORIAL["green"],
        EDITORIAL["amber"],
        EDITORIAL["purple"],
        EDITORIAL["coral"],
        EDITORIAL["ice"],
        EDITORIAL["green"],
        EDITORIAL["amber"],
    ]
    drawing.add(
        drawing.circle(
            center=(cx, cy),
            r=66,
            fill=EDITORIAL["panel_2"],
            stroke=EDITORIAL["green"],
            stroke_width=4,
        )
    )
    _draw_wrapped_text(drawing, "Story hub", cx, cy + 2, 20, EDITORIAL["white"], anchor="middle", width=12)
    drawing.add(drawing.text("Drive", insert=(cx - 24, cy - 86), fill=EDITORIAL["green"], font_size=17, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.text("Signal 1-8 ring", insert=(92, 154), fill=EDITORIAL["ice"], font_size=18, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.text("Spoke paths point inward", insert=(950, 154), fill=EDITORIAL["gray"], font_size=17, font_family="Arial"))
    for index, label in enumerate(labels):
        angle = -math.pi / 2 + index * 2 * math.pi / len(labels)
        node_x = cx + math.cos(angle) * radius_x
        node_y = cy + math.sin(angle) * radius_y
        color = colors[index % len(colors)]
        start_x = cx + math.cos(angle) * 78
        start_y = cy + math.sin(angle) * 78
        end_x = cx + math.cos(angle) * (radius_x - 88)
        end_y = cy + math.sin(angle) * (radius_y - 34)
        _draw_editorial_arrow(drawing, end_x, end_y, start_x, start_y, color)
        drawing.add(
            drawing.rect(
                insert=(node_x - 82, node_y - 30),
                size=(164, 60),
                rx=14,
                fill=color,
                opacity=0.18,
                stroke=color,
                stroke_width=3,
            )
        )
        _draw_wrapped_text(drawing, label, node_x, node_y + 6, 15, EDITORIAL["white"], anchor="middle", width=15)


def _radial_signal_labels(context: dict[str, object]) -> list[str]:
    labels: list[str] = []
    for entity in context.get("entities", []):
        value = str(entity).strip()
        if value.lower().startswith("signal ") and ":" in value:
            labels.append(_truncate(value, 28))
    for bullet in context.get("bullets", []):
        lowered = str(bullet).lower()
        if "systems" in lowered and "models" in lowered:
            labels.append("Systems unit")
        elif "evaluation" in lowered:
            labels.append("Evaluation infra")
        elif "scaling" in lowered:
            labels.append("Conditional scaling")
        elif "agent" in lowered or "attack surface" in lowered:
            labels.append("Agent attack surface")
        elif "test-time" in lowered or "adaptive inference" in lowered:
            labels.append("Test-time compute")
        elif "multimodal" in lowered or "world state" in lowered:
            labels.append("Spatial world state")
        elif "scientific" in lowered or "biomolecular" in lowered:
            labels.append("Science workflows")
        elif "efficiency" in lowered or "resource governance" in lowered:
            labels.append("Resource governance")
        else:
            labels.append(_safe_label(str(bullet), words=3).title())
    fallback = [
        "Systems unit",
        "Evaluation infra",
        "Conditional scaling",
        "Agent attack surface",
        "Test-time compute",
        "Spatial world state",
        "Science workflows",
        "Resource governance",
    ]
    labels = _unique_preserve_order(labels, limit=8)
    while len(labels) < 8:
        labels.append(fallback[len(labels)])
    return labels[:8]


def _draw_semantic_spatial_task_panel(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 4, ["2D view", "Depth/pose", "3D relation", "Model answer"])
    panel_positions = [(126, 238), (382, 238), (638, 238), (894, 238)]
    for index, ((x, y), label) in enumerate(zip(panel_positions, labels)):
        color = [EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["green"]][index]
        drawing.add(drawing.rect(insert=(x, y), size=(210, 270), rx=14, fill=EDITORIAL["panel_2"], stroke=color, stroke_width=4, opacity=0.96))
        if index == 0:
            drawing.add(drawing.rect(insert=(x + 50, y + 80), size=(78, 60), fill=color, opacity=0.45))
            drawing.add(drawing.circle(center=(x + 140, y + 120), r=34, fill=EDITORIAL["coral"], opacity=0.78))
        elif index == 1:
            _draw_3d_axes(drawing, x + 96, y + 140, color)
            drawing.add(drawing.polygon(points=[(x + 92, y + 88), (x + 152, y + 116), (x + 126, y + 176), (x + 64, y + 150)], fill=color, opacity=0.42, stroke=color, stroke_width=3))
        elif index == 2:
            drawing.add(drawing.line(start=(x + 58, y + 172), end=(x + 160, y + 88), stroke=color, stroke_width=6))
            drawing.add(drawing.line(start=(x + 58, y + 172), end=(x + 160, y + 188), stroke=EDITORIAL["coral"], stroke_width=6))
            drawing.add(drawing.text("angle", insert=(x + 90, y + 126), fill=EDITORIAL["white"], font_size=18, font_family="Arial"))
        else:
            drawing.add(drawing.rect(insert=(x + 50, y + 78), size=(110, 44), rx=8, fill=color, opacity=0.38))
            drawing.add(drawing.text("pass/fail", insert=(x + 105, y + 154), fill=EDITORIAL["white"], font_size=22, font_family="Arial", font_weight="bold", text_anchor="middle"))
        _draw_wrapped_text(drawing, label, x + 105, y + 224, 16, EDITORIAL["gray"], anchor="middle", width=21)
    for x1 in [336, 592, 848]:
        _draw_editorial_arrow(drawing, x1, 373, x1 + 36, 373, EDITORIAL["muted"])


def _draw_3d_axes(drawing: svgwrite.Drawing, x: int, y: int, color: str) -> None:
    drawing.add(drawing.line(start=(x, y), end=(x + 70, y + 28), stroke=color, stroke_width=4))
    drawing.add(drawing.line(start=(x, y), end=(x - 62, y + 34), stroke=color, stroke_width=4))
    drawing.add(drawing.line(start=(x, y), end=(x, y - 78), stroke=color, stroke_width=4))
    drawing.add(drawing.text("3D axes", insert=(x - 42, y - 104), fill=color, font_size=18, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.text("X", insert=(x + 78, y + 40), fill=color, font_size=19, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.text("Y", insert=(x - 90, y + 50), fill=color, font_size=19, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.text("Z", insert=(x - 18, y - 84), fill=color, font_size=19, font_family="Arial", font_weight="bold"))


def _draw_semantic_agent_workflow(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 5, ["Goal", "Planner", "Tool call", "Observation", "Verifier"])
    centers = [(220, 355), (430, 268), (670, 268), (900, 355), (640, 478)]
    colors = [EDITORIAL["ice"], EDITORIAL["purple"], EDITORIAL["amber"], EDITORIAL["green"], EDITORIAL["coral"]]
    for index, center in enumerate(centers[:-1]):
        next_center = centers[index + 1]
        start, end = _node_edge_points(center, next_center, radius=92)
        _draw_editorial_arrow(drawing, start[0], start[1], end[0], end[1], EDITORIAL["muted"])
    _draw_editorial_arrow(drawing, 636, 408, 448, 318, EDITORIAL["coral"])
    for center, label, color in zip(centers, labels, colors):
        drawing.add(drawing.circle(center=center, r=82, fill=EDITORIAL["panel_2"], opacity=1.0, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, _compact_center_label(label), center[0], center[1] - 18, 15, EDITORIAL["white"], anchor="middle", width=18)


def _draw_semantic_retrieval_graph(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 5, ["Query", "Seed node", "Topology", "Evidence path", "Answer"])
    centers = [(200, 360), (410, 300), (610, 410), (820, 294), (1040, 360)]
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["purple"], EDITORIAL["amber"], EDITORIAL["coral"]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3), (0, 2)]
    for start_idx, end_idx in edges:
        drawing.add(drawing.line(start=centers[start_idx], end=centers[end_idx], stroke=EDITORIAL["gray"], stroke_width=2.6, opacity=0.58))
    _draw_editorial_arrow(drawing, 252, 348, 956, 358, EDITORIAL["ice"])
    for center, label, color in zip(centers, labels, colors):
        drawing.add(drawing.circle(center=center, r=72, fill=EDITORIAL["panel_2"], opacity=1.0, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, _compact_center_label(label), center[0], center[1] + 5, 15, EDITORIAL["white"], anchor="middle", width=14)


def _draw_semantic_source_chart(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
    depiction_type: str,
) -> None:
    if _contains_any(_semantic_context_text(context), ("single-cell", "single cell", "umap", "t-sne", "tsne", "cell type")):
        _draw_semantic_umap_cluster(drawing, context)
        return
    labels = _semantic_labels(
        context,
        5,
        ["Task Complexity", "Compute Budget", "Reasoning Steps", "Allocation Loop", "Output Quality"],
    )
    metrics = _source_chart_metrics(context)
    chart_x, chart_y, chart_w, chart_h = 205, 170, 735, 250
    axis_y = chart_y + chart_h
    drawing.add(drawing.line(start=(chart_x, axis_y), end=(chart_x + chart_w, axis_y), stroke=EDITORIAL["gray"], stroke_width=4))
    drawing.add(drawing.line(start=(chart_x, chart_y), end=(chart_x, axis_y), stroke=EDITORIAL["gray"], stroke_width=4))
    _draw_wrapped_text(drawing, _compact_axis_label(labels[1], words=2), chart_x - 82, chart_y + 112, 16, EDITORIAL["gray"], anchor="middle", width=10)
    for tick_label, tick_x in (("Low", chart_x + 36), ("Medium", chart_x + chart_w / 2), ("High", chart_x + chart_w - 48)):
        drawing.add(drawing.text(tick_label, insert=(tick_x, axis_y + 18), fill=EDITORIAL["gray"], font_size=14, font_family="Arial", text_anchor="middle"))
    if depiction_type == "scaling_chart":
        points = [(chart_x + 42, chart_y + 220), (chart_x + 240, chart_y + 176), (chart_x + 458, chart_y + 104), (chart_x + 685, chart_y + 48)]
        drawing.add(drawing.polyline(points=points, fill="none", stroke=EDITORIAL["coral"], stroke_width=8))
        for index, (x, y) in enumerate(points):
            drawing.add(drawing.circle(center=(x, y), r=13, fill=EDITORIAL["navy"], stroke=EDITORIAL["coral"], stroke_width=5))
            if index < len(metrics):
                metric_label = _compact_metric_label(metrics[index])
                if index == 0:
                    drawing.add(drawing.text(metric_label, insert=(x + 22, y - 24), fill=EDITORIAL["white"], font_size=14, font_family="Arial", text_anchor="start"))
                else:
                    drawing.add(drawing.text(metric_label, insert=(x, y - 24), fill=EDITORIAL["white"], font_size=14, font_family="Arial", text_anchor="middle"))
            label = _compact_axis_label(labels[index], words=3)
            card_x = x - 66
            drawing.add(drawing.rect(insert=(card_x, axis_y + 34), size=(132, 66), rx=8, fill=EDITORIAL["panel_2"], opacity=0.92, stroke=EDITORIAL["muted"], stroke_width=1.5))
            _draw_wrapped_text(drawing, label, x, axis_y + 58, 14, EDITORIAL["gray"], anchor="middle", width=13)
        _draw_source_chart_entity_panel(drawing, context)
        drawing.add(drawing.path(d=f"M 640 310 C 720 270, 750 360, 650 395", fill="none", stroke=EDITORIAL["amber"], stroke_width=4, opacity=0.75))
    else:
        heights = [92, 138, 190, 220]
        for index, (label, height) in enumerate(zip(labels, heights)):
            x = chart_x + 44 + index * 174
            color = [EDITORIAL["coral"], EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"]][index]
            drawing.add(drawing.rect(insert=(x, axis_y - height), size=(112, height), rx=10, fill=color, opacity=0.78))
            if index < len(metrics):
                drawing.add(drawing.text(_compact_metric_label(metrics[index]), insert=(x + 56, axis_y - height - 18), fill=EDITORIAL["white"], font_size=14, font_family="Arial", text_anchor="middle"))
            drawing.add(drawing.rect(insert=(x - 16, axis_y + 34), size=(144, 72), rx=8, fill=EDITORIAL["panel_2"], opacity=0.92, stroke=EDITORIAL["muted"], stroke_width=1.5))
            _draw_wrapped_text(drawing, _compact_axis_label(label, words=3), x + 56, axis_y + 60, 14, EDITORIAL["gray"], anchor="middle", width=13)
        _draw_source_chart_entity_panel(drawing, context)


def _draw_source_chart_entity_panel(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    return


def _draw_semantic_umap_cluster(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["T Cell", "B Cell", "Monocyte", "NK Cell", "UMAP Projection"],
    )
    cluster_labels = [
        _replace_generic_cell_label(labels[0], "T Cell"),
        _replace_generic_cell_label(labels[1], "B Cell"),
        _replace_generic_cell_label(labels[2], "Monocyte"),
        _replace_generic_cell_label(labels[3], "NK Cell"),
    ]
    x0, y0, w, h = 210, 170, 700, 390
    drawing.add(drawing.rect(insert=(x0, y0), size=(w, h), rx=16, fill=EDITORIAL["panel_2"], opacity=0.92, stroke=EDITORIAL["muted"], stroke_width=2.5))
    drawing.add(drawing.line(start=(x0 + 58, y0 + h - 46), end=(x0 + w - 42, y0 + h - 46), stroke=EDITORIAL["gray"], stroke_width=3))
    drawing.add(drawing.line(start=(x0 + 58, y0 + h - 46), end=(x0 + 58, y0 + 42), stroke=EDITORIAL["gray"], stroke_width=3))
    drawing.add(drawing.text("UMAP-1", insert=(x0 + w - 102, y0 + h - 14), fill=EDITORIAL["gray"], font_size=16, font_family="Arial"))
    drawing.add(drawing.text("UMAP-2", insert=(x0 + 16, y0 + 74), fill=EDITORIAL["gray"], font_size=16, font_family="Arial", transform=f"rotate(-90 {x0 + 16},{y0 + 74})"))
    drawing.add(drawing.text("Single-cell transcriptomic embedding", insert=(x0 + w / 2, y0 + 34), fill=EDITORIAL["white"], font_size=20, font_family="Arial", font_weight="bold", text_anchor="middle"))
    clusters = [
        ((355, 392), EDITORIAL["ice"], cluster_labels[0], [(-32, -8), (-10, -34), (12, -12), (28, 18), (-20, 24), (8, 36)]),
        ((548, 290), EDITORIAL["green"], cluster_labels[1], [(-26, 18), (-8, -18), (24, -24), (36, 10), (6, 28), (-30, -2)]),
        ((690, 414), EDITORIAL["amber"], cluster_labels[2], [(-36, -10), (-12, -28), (18, -6), (38, 24), (-6, 36), (-28, 18)]),
        ((760, 270), EDITORIAL["coral"], cluster_labels[3], [(-24, 14), (-8, -22), (20, -14), (34, 14), (4, 34), (-32, -4)]),
    ]
    for (cx, cy), color, label, offsets in clusters:
        drawing.add(drawing.ellipse(center=(cx, cy), r=(72, 52), fill=color, opacity=0.12, stroke=color, stroke_width=2.5))
        for dx, dy in offsets:
            drawing.add(drawing.circle(center=(cx + dx, cy + dy), r=9, fill=color, opacity=0.88))
        drawing.add(drawing.rect(insert=(cx - 74, cy + 58), size=(148, 34), rx=8, fill=EDITORIAL["panel_2"], opacity=0.94))
        _draw_wrapped_text(drawing, label, cx, cy + 80, 14, EDITORIAL["white"], anchor="middle", width=15)
    legend_x, legend_y = 955, 250
    drawing.add(drawing.rect(insert=(legend_x, legend_y), size=(170, 170), rx=14, fill=EDITORIAL["panel_2"], opacity=0.92, stroke=EDITORIAL["muted"], stroke_width=2))
    drawing.add(drawing.text("Cell identity", insert=(legend_x + 20, legend_y + 32), fill=EDITORIAL["ice"], font_size=16, font_family="Arial", font_weight="bold"))
    for idx, (label, color) in enumerate(zip(cluster_labels, [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]])):
        y = legend_y + 62 + idx * 28
        drawing.add(drawing.circle(center=(legend_x + 24, y - 5), r=7, fill=color))
        _draw_wrapped_text(drawing, label, legend_x + 42, y, 13, EDITORIAL["gray"], width=16)


def _draw_semantic_architecture(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 5, ["Input", "Encoder", "Representation", "Reasoner", "Output"])
    x, y = 118, 288
    box_w, box_h, step = 170, 132, 216
    for index, label in enumerate(labels):
        color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["purple"], EDITORIAL["amber"], EDITORIAL["coral"]][index]
        node_x = x + index * step
        drawing.add(drawing.rect(insert=(node_x, y), size=(box_w, box_h), rx=16, fill=EDITORIAL["panel_2"], opacity=0.97, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, _compact_center_label(label), node_x + box_w / 2, y + 68, 16, EDITORIAL["white"], anchor="middle", width=17)
        if index < 4:
            _draw_editorial_arrow(drawing, node_x + box_w + 8, y + 66, x + (index + 1) * step - 12, y + 66, EDITORIAL["muted"])
    relationships = [str(item) for item in context.get("relationships", [])]
    if relationships:
        _draw_wrapped_text(drawing, relationships[0], 640, 480, 19, EDITORIAL["ice"], anchor="middle", width=58)


def _draw_semantic_tradeoff_matrix(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(context, 4, ["High reliability", "High cost", "Low cost", "Low reliability"])
    x0, y0, w, h = 300, 220, 580, 340
    drawing.add(drawing.rect(insert=(x0, y0), size=(w, h), rx=14, fill=EDITORIAL["panel_2"], opacity=0.88, stroke=EDITORIAL["muted"], stroke_width=3))
    drawing.add(drawing.line(start=(x0 + w / 2, y0), end=(x0 + w / 2, y0 + h), stroke=EDITORIAL["muted"], stroke_width=2))
    drawing.add(drawing.line(start=(x0, y0 + h / 2), end=(x0 + w, y0 + h / 2), stroke=EDITORIAL["muted"], stroke_width=2))
    cell_w, cell_h = w / 2, h / 2
    positions = [
        (x0 + cell_w / 2, y0 + 88, x0 + 30, y0 + 28),
        (x0 + cell_w + cell_w / 2, y0 + 88, x0 + cell_w + 30, y0 + 28),
        (x0 + cell_w / 2, y0 + cell_h + 94, x0 + 30, y0 + cell_h + 48),
        (x0 + cell_w + cell_w / 2, y0 + cell_h + 94, x0 + cell_w + 30, y0 + cell_h + 48),
    ]
    colors = [EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["ice"], EDITORIAL["coral"]]
    for label, (cx, cy, bx, by), color in zip(labels, positions, colors):
        drawing.add(drawing.circle(center=(cx, cy), r=34, fill=color, opacity=0.22, stroke=color, stroke_width=3))
        drawing.add(
            drawing.rect(
                insert=(bx, by),
                size=(230, 86),
                rx=12,
                fill=EDITORIAL["panel_2"],
                opacity=0.94,
                stroke=color,
                stroke_width=2,
            )
        )
        _draw_wrapped_text(drawing, label, bx + 115, by + 25, 14, EDITORIAL["white"], anchor="middle", width=24)
    drawing.add(drawing.text("Capability", insert=(x0 + w / 2, y0 + h + 45), fill=EDITORIAL["gray"], font_size=18, font_family="Arial", text_anchor="middle"))
    drawing.add(drawing.text("Risk / cost", insert=(x0 - 50, y0 + h / 2), fill=EDITORIAL["gray"], font_size=18, font_family="Arial", transform=f"rotate(-90 {x0 - 50},{y0 + h / 2})"))


def _draw_semantic_ladder(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Unit Test", "Integration Test", "Eval Suite", "Reliability Gate", "Infrastructure Layer"],
    )
    rail_xs = (430, 760)
    y_positions = [488, 418, 348, 278, 208]
    for x in rail_xs:
        drawing.add(drawing.line(start=(x, 500), end=(x, 190), stroke=EDITORIAL["ice"], stroke_width=6, opacity=0.78))
    colors = [EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["green"], EDITORIAL["ice"]]
    for index, (label, y, color) in enumerate(zip(labels, y_positions, colors)):
        drawing.add(drawing.line(start=(rail_xs[0], y), end=(rail_xs[1], y), stroke=color, stroke_width=8, opacity=0.9))
        if index < len(labels) - 1:
            _draw_wrapped_text(drawing, label, 595, y - 12, 18, EDITORIAL["white"], anchor="middle", width=28)
        if index < len(y_positions) - 1:
            _draw_editorial_arrow(drawing, 810, y - 6, 810, y_positions[index + 1] + 18, EDITORIAL["muted"])
    drawing.add(drawing.rect(insert=(330, 160), size=(530, 48), rx=12, fill=EDITORIAL["green"], opacity=0.18, stroke=EDITORIAL["green"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[-1], 595, 190, 19, EDITORIAL["green"], anchor="middle", width=34)


def _draw_semantic_risk_surface(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Agent", "Capability Frontier", "Risk Surface", "Tool Access", "Autonomy Level"],
    )
    full_labels = [str(item).strip() for item in context.get("entities", []) if str(item).strip()]
    while len(full_labels) < len(labels):
        full_labels.append(labels[len(full_labels)])
    center = (640, 316)
    drawing.add(drawing.ellipse(center=center, r=(458, 188), fill="none", stroke=EDITORIAL["coral"], stroke_width=4, stroke_dasharray="12,10", opacity=0.44))
    drawing.add(drawing.circle(center=center, r=72, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["white"], stroke_width=4))
    _draw_wrapped_text(
        drawing,
        _compact_center_label(full_labels[0]),
        center[0],
        center[1] - 8,
        18,
        EDITORIAL["white"],
        anchor="middle",
        width=11,
    )
    left_nodes = [(260, 248, "Web"), (240, 362, labels[3]), (410, 438, "Code")]
    right_nodes = [(920, 248, "Misuse"), (950, 362, labels[2]), (790, 438, labels[4])]
    for x, y, label in left_nodes:
        drawing.add(drawing.rect(insert=(x - 92, y - 44), size=(184, 88), rx=14, fill=EDITORIAL["ice"], opacity=0.18, stroke=EDITORIAL["ice"], stroke_width=3))
        _draw_wrapped_text(drawing, label, x, y - 16, 16, EDITORIAL["white"], anchor="middle", width=20)
        _draw_editorial_arrow(drawing, center[0] - 72, center[1], x + 96, y, EDITORIAL["ice"])
    for x, y, label in right_nodes:
        drawing.add(drawing.rect(insert=(x - 92, y - 44), size=(184, 88), rx=14, fill=EDITORIAL["amber"], opacity=0.18, stroke=EDITORIAL["amber"], stroke_width=3))
        _draw_wrapped_text(drawing, label, x, y - 16, 16, EDITORIAL["white"], anchor="middle", width=20)
        _draw_editorial_arrow(drawing, center[0] + 72, center[1], x - 96, y, EDITORIAL["amber"])
    _draw_wrapped_text(drawing, full_labels[1], 330, 520, 16, EDITORIAL["ice"], anchor="middle", width=54)
    _draw_wrapped_text(drawing, full_labels[2], 910, 520, 16, EDITORIAL["amber"], anchor="middle", width=54)
    _draw_wrapped_text(drawing, "Attack Surface", 640, 168, 20, EDITORIAL["coral"], anchor="middle", width=20)


def _draw_semantic_evidence_action_map(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
) -> None:
    if _contains_any(_semantic_context_text(context), ("four main pillars", "pillar nodes", "key pillars")):
        _draw_semantic_pillar_summary(drawing, context)
        return
    labels = _semantic_labels(
        context,
        5,
        ["Evidence Base", "Decision Point", "Action", "Feedback", "Mechanism"],
    )
    left = (230, 350)
    center = (610, 350)
    actions = [(940, 250), (1010, 350), (940, 450)]
    drawing.add(drawing.ellipse(center=left, r=(118, 64), fill=EDITORIAL["ice"], opacity=0.18, stroke=EDITORIAL["ice"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[0], left[0], left[1] + 6, 19, EDITORIAL["white"], anchor="middle", width=17)
    drawing.add(drawing.polygon(points=[(center[0], center[1] - 88), (center[0] + 112, center[1]), (center[0], center[1] + 88), (center[0] - 112, center[1])], fill=EDITORIAL["purple"], opacity=0.2, stroke=EDITORIAL["purple"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[1], center[0], center[1] + 6, 19, EDITORIAL["white"], anchor="middle", width=16)
    _draw_editorial_arrow(drawing, left[0] + 125, left[1], center[0] - 118, center[1], EDITORIAL["muted"])
    for index, (x, y) in enumerate(actions):
        label = labels[2] if index == 0 else ["Invest", "Govern", "Deploy"][index]
        drawing.add(drawing.rect(insert=(x - 86, y - 38), size=(172, 76), rx=16, fill=EDITORIAL["green"], opacity=0.18, stroke=EDITORIAL["green"], stroke_width=3))
        _draw_wrapped_text(drawing, label, x, y + 5, 18, EDITORIAL["white"], anchor="middle", width=14)
        _draw_editorial_arrow(drawing, center[0] + 118, center[1], x - 90, y, EDITORIAL["green"])
        drawing.add(drawing.path(d=f"M {x} {y + 44} C {x - 160} 620, {left[0] + 40} 610, {left[0]} {left[1] + 72}", fill="none", stroke=EDITORIAL["coral"], stroke_width=3, opacity=0.56))
    _draw_wrapped_text(drawing, labels[3], 575, 590, 18, EDITORIAL["coral"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[4], 610, 242, 18, EDITORIAL["amber"], anchor="middle", width=18)


def _draw_semantic_pillar_summary(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        4,
        ["Efficient Training", "Contractive Stability", "Scalable Data Engine", "Autoregressive Generation"],
    )
    center = (640, 360)
    drawing.add(drawing.ellipse(center=center, r=(150, 58), fill=EDITORIAL["purple"], opacity=0.2, stroke=EDITORIAL["purple"], stroke_width=4))
    _draw_wrapped_text(drawing, "Efficient Generative Modeling", center[0], center[1] + 6, 20, EDITORIAL["white"], anchor="middle", width=28)
    positions = [(250, 255), (960, 255), (250, 500), (960, 500)]
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]]
    subtitles = ["Latent Consistency", "Out-of-Sample Recovery", "NAIDv2, 75M tokens", "LlamaGen"]
    for (x, y), label, color, subtitle in zip(positions, labels, colors, subtitles):
        drawing.add(drawing.rect(insert=(x - 145, y - 58), size=(290, 116), rx=18, fill=color, opacity=0.16, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, y - 10, 17, EDITORIAL["white"], anchor="middle", width=26)
        _draw_wrapped_text(drawing, subtitle, x, y + 38, 15, color, anchor="middle", width=24)
        _draw_editorial_arrow(drawing, center[0], center[1], x, y, EDITORIAL["muted"])
    drawing.add(drawing.path(d="M 330 500 C 250 610, 280 645, 250 255", fill="none", stroke=EDITORIAL["amber"], stroke_width=4, stroke_dasharray="9 7", opacity=0.72))


def _draw_semantic_constraint_triangle(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Fidelity", "Diversity", "Constraint", "Feasible Region", "Boundary"],
    )
    points = [(640, 210), (320, 500), (980, 500)]
    drawing.add(
        drawing.polygon(
            points=points,
            fill=EDITORIAL["purple"],
            opacity=0.18,
            stroke=EDITORIAL["ice"],
            stroke_width=5,
        )
    )
    inner = [(640, 300), (460, 452), (820, 452)]
    drawing.add(
        drawing.polygon(
            points=inner,
            fill=EDITORIAL["green"],
            opacity=0.22,
            stroke=EDITORIAL["green"],
            stroke_width=3,
        )
    )
    vertex_positions = [(640, 176), (270, 516), (1030, 516)]
    for label, pos, color in zip(labels[:3], vertex_positions, [EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["coral"]]):
        drawing.add(drawing.circle(center=pos, r=54, fill=color, opacity=0.22, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, pos[0], pos[1] - 14, 15, EDITORIAL["white"], anchor="middle", width=17)
    _draw_wrapped_text(drawing, labels[3], 640, 430, 20, EDITORIAL["green"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, labels[4], 640, 542, 17, EDITORIAL["gray"], anchor="middle", width=28)


def _draw_semantic_intervention_flow(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Interpretability", "Probe", "Intervention", "Causal Circuit", "Behavior"],
    )
    x_positions = [130, 355, 590, 825, 1040]
    colors = [EDITORIAL["ice"], EDITORIAL["purple"], EDITORIAL["amber"], EDITORIAL["green"], EDITORIAL["coral"]]
    for index, (x, label, color) in enumerate(zip(x_positions, labels, colors)):
        drawing.add(drawing.rect(insert=(x - 78, 304), size=(156, 104), rx=16, fill=color, opacity=0.18, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, 358, 17, EDITORIAL["white"], anchor="middle", width=13)
        if index < len(x_positions) - 1:
            _draw_editorial_arrow(drawing, x + 86, 356, x_positions[index + 1] - 90, 356, EDITORIAL["muted"])
    drawing.add(drawing.rect(insert=(260, 462), size=(620, 72), rx=12, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.88))
    _draw_wrapped_text(drawing, "observe activations -> manipulate circuit -> measure behavior", 570, 492, 15, EDITORIAL["gray"], anchor="middle", width=74)


def _draw_semantic_watchlist_timeline(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
) -> None:
    labels = _semantic_labels(
        context,
        5,
        [
            "Frontier Model Releases",
            "Agent Autonomy Thresholds",
            "Regulatory Milestones",
            "Incident Patterns",
            "Eval Benchmark Saturation",
        ],
    )
    x0, y = 160, 360
    drawing.add(drawing.line(start=(x0, y), end=(1080, y), stroke=EDITORIAL["ice"], stroke_width=5, opacity=0.76))
    for index, label in enumerate(labels):
        x = x0 + index * 230
        color = [EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["coral"], EDITORIAL["green"]][index]
        drawing.add(drawing.circle(center=(x, y), r=32, fill=color, opacity=0.28, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, y - 70 if index % 2 == 0 else y + 92, 16, EDITORIAL["white"], anchor="middle", width=17)
        _draw_wrapped_text(drawing, "trigger", x, y + 52 if index % 2 == 0 else y - 46, 14, EDITORIAL["gray"], anchor="middle", width=10)
    _draw_editorial_arrow(drawing, 1080, y, 1140, y, EDITORIAL["ice"])


def _draw_semantic_source_backbone(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Raw Sources", "Extraction Layer", "Backbone Graph", "Synthesis Layer", "Verified Output"],
    )
    x_positions = [140, 360, 620, 860, 1080]
    colors = [EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["green"], EDITORIAL["coral"]]
    for index, (x, label, color) in enumerate(zip(x_positions, labels, colors)):
        if index == 2:
            drawing.add(drawing.circle(center=(x, 355), r=72, fill=color, opacity=0.18, stroke=color, stroke_width=4))
            for angle in (0, 2.1, 4.2):
                nx = x + int(math.cos(angle) * 42)
                ny = 328 + int(math.sin(angle) * 28)
                drawing.add(drawing.circle(center=(nx, ny), r=10, fill=color, opacity=0.46))
                drawing.add(drawing.line(start=(x, 328), end=(nx, ny), stroke=color, stroke_width=2, opacity=0.32))
        else:
            drawing.add(drawing.rect(insert=(x - 78, 302), size=(156, 110), rx=16, fill=color, opacity=0.18, stroke=color, stroke_width=4))
        if index == 2:
            drawing.add(drawing.rect(insert=(x - 74, 348), size=(148, 44), rx=10, fill=EDITORIAL["panel_2"], opacity=0.94))
        _draw_wrapped_text(drawing, _compact_center_label(label), x, 360, 16, EDITORIAL["white"], anchor="middle", width=15)
        if index < len(x_positions) - 1:
            _draw_editorial_arrow(drawing, x + 86, 356, x_positions[index + 1] - 90, 356, EDITORIAL["muted"])
    _draw_wrapped_text(drawing, "ingest -> extract -> link -> synthesize -> verify", 640, 500, 18, EDITORIAL["gray"], anchor="middle", width=54)


def _semantic_landscape_override(drawing: svgwrite.Drawing, context: dict[str, object]) -> bool:
    text = _semantic_context_text(context)
    if _contains_any(
        text,
        ("tenant", "data leakage", "leakage path", "shared model layer", "cross-tenant"),
    ):
        _draw_semantic_tenant_boundary(drawing, context)
        return True
    if _contains_any(
        text,
        (
            "reference image",
            "text prompt",
            "generated output",
            "few-shot",
            "masked token",
            "remasking",
            "prompt field",
        ),
    ):
        _draw_semantic_prompt_flow(drawing, context)
        return True
    if _contains_any(text, ("venn", "overlap", "diffusion core", "cross-domain")):
        _draw_semantic_technique_overlap(drawing, context)
        return True
    if _contains_any(text, ("funnel", "drop-off", "drop off", "weak binder", "survivors", "lab entry")):
        _draw_semantic_attrition_funnel(drawing, context)
        return True
    if _contains_any(text, ("satellite image", "geochat", "bounding box", "grounded region")):
        _draw_semantic_satellite_query(drawing, context)
        return True
    if _contains_any(text, ("optical layer", "sar layer", "ir layer", "sm3det", "layered map")):
        _draw_semantic_layered_map(drawing, context)
        return True
    if _contains_any(text, ("hierarchical", "flat keyword", "claim-level", "paper-level")):
        _draw_semantic_hierarchical_search(drawing, context)
        return True
    if _contains_any(text, ("clipped sgd", "clipping operator", "high-probability", "heavy-tailed")):
        _draw_semantic_math_chain(drawing, context)
        return True
    if _contains_any(text, ("objective function", "societal complexity", "acadreason", "context overflow")):
        _draw_semantic_objective_web(drawing, context)
        return True
    if _contains_any(text, ("dashboard", "eos detector", "sharpness monitor", "weight decay controller")):
        _draw_semantic_training_dashboard(drawing, context)
        return True
    if _contains_any(text, ("feature dictionary", "search feature", "llm layer index")):
        _draw_semantic_feature_dictionary_ui(drawing, context)
        return True
    if _contains_any(text, ("expert head", "hallucination alert", "mechanistic reliability")):
        _draw_semantic_expert_dashboard(drawing, context)
        return True
    if _contains_any(text, ("unified world models", "td-mpc2", "closed-loop decision", "video generation")):
        _draw_semantic_world_model_landscape(drawing, context)
        return True
    if _contains_any(text, ("interactive simulator", "standardized task suite", "diffusion policy agent")):
        _draw_semantic_simulator_benchmark(drawing, context)
        return True
    if _contains_any(
        text,
        (
            "multimodal alignment",
            "alignment complexity",
            "visual grounding alignment",
            "contextual intent alignment",
            "refusal feature",
            "copy-paste faithfulness",
        ),
    ):
        _draw_semantic_alignment_landscape(drawing, context)
        return True
    return False


def _semantic_context_text(context: dict[str, object]) -> str:
    spec = context.get("figure_spec") if isinstance(context.get("figure_spec"), dict) else {}
    values = [
        context.get("title", ""),
        context.get("need", ""),
        spec.get("semantic_goal", ""),
        " ".join(str(item) for item in context.get("entities", [])),
        " ".join(str(item) for item in context.get("relationships", [])),
        " ".join(str(item) for item in spec.get("must_depict", []) if isinstance(spec.get("must_depict"), list)),
    ]
    return " ".join(str(value) for value in values).lower()


def _draw_semantic_tenant_boundary(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        [
            "Tenant A Context",
            "Tenant B Context",
            "Shared Model Layer",
            "Copy-Paste Leakage Path",
            "HIRE Token Edit Boundary",
        ],
    )
    tenant_a = _first_label_containing(labels, ("tenant a",), "Tenant A Context")
    tenant_b = _first_label_containing(labels, ("tenant b",), "Tenant B Context")
    shared = _first_label_containing(labels, ("shared", "model layer"), "Shared Model Layer")
    leak = _first_label_containing(labels, ("leak", "copy-paste"), "Copy-Paste Leakage Path")
    boundary = _first_label_containing(labels, ("hire", "boundary", "token edit"), "HIRE Token Edit Boundary")

    drawing.add(drawing.rect(insert=(112, 235), size=(250, 220), rx=22, fill=EDITORIAL["ice"], opacity=0.16, stroke=EDITORIAL["ice"], stroke_width=4))
    drawing.add(drawing.rect(insert=(918, 235), size=(250, 220), rx=22, fill=EDITORIAL["green"], opacity=0.16, stroke=EDITORIAL["green"], stroke_width=4))
    drawing.add(drawing.rect(insert=(454, 255), size=(372, 180), rx=20, fill=EDITORIAL["panel_2"], opacity=0.94, stroke=EDITORIAL["purple"], stroke_width=4))
    _draw_wrapped_text(drawing, tenant_a, 237, 326, 21, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, tenant_b, 1043, 326, 21, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, shared, 640, 327, 22, EDITORIAL["white"], anchor="middle", width=24)

    drawing.add(drawing.line(start=(365, 350), end=(918, 350), stroke=EDITORIAL["coral"], stroke_width=5, stroke_dasharray="13 10", opacity=0.88))
    _draw_editorial_arrow(drawing, 365, 350, 918, 350, EDITORIAL["coral"])
    _draw_wrapped_text(drawing, leak, 640, 226, 18, EDITORIAL["coral"], anchor="middle", width=32)
    drawing.add(drawing.rect(insert=(582, 388), size=(236, 64), rx=14, fill=EDITORIAL["amber"], opacity=0.22, stroke=EDITORIAL["amber"], stroke_width=3))
    _draw_wrapped_text(drawing, boundary, 700, 425, 17, EDITORIAL["amber"], anchor="middle", width=24)
    drawing.add(drawing.line(start=(640, 435), end=(640, 492), stroke=EDITORIAL["amber"], stroke_width=4, opacity=0.85))
    _draw_wrapped_text(drawing, "intent-context check", 640, 522, 17, EDITORIAL["gray"], anchor="middle", width=24)


def _draw_semantic_alignment_landscape(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    if _contains_any(_semantic_context_text(context), ("peace objective", "ai for peace", "deception mitigation")):
        _draw_semantic_peace_framework(drawing, context)
        return
    labels = _semantic_labels(
        context,
        5,
        [
            "Text Alignment RLHF",
            "Visual Grounding Alignment",
            "Contextual Intent Alignment",
            "Refusal Feature Adversarial Training",
            "Copy-Paste Faithfulness",
        ],
    )
    text_zone = _first_label_containing(labels, ("text", "rlhf"), "Text Alignment RLHF")
    visual_zone = _first_label_containing(labels, ("visual", "grounding"), "Visual Grounding Alignment")
    intent_zone = _first_label_containing(labels, ("contextual", "intent"), "Contextual Intent Alignment")
    refusal = _first_label_containing(labels, ("refusal", "adversarial"), "Refusal Feature Adversarial Training")
    faithfulness = _first_label_containing(labels, ("copy-paste", "faithfulness"), "Copy-Paste Faithfulness")

    drawing.add(drawing.text("Complexity gradient", insert=(120, 210), fill=EDITORIAL["gray"], font_size=17, font_family="Arial"))
    drawing.add(drawing.rect(insert=(130, 235), size=(980, 250), rx=24, fill=EDITORIAL["panel_2"], opacity=0.88, stroke=EDITORIAL["muted"], stroke_width=2))
    zones = [
        (155, text_zone, EDITORIAL["ice"], "established"),
        (470, visual_zone, EDITORIAL["green"], "emerging"),
        (785, intent_zone, EDITORIAL["coral"], "frontier"),
    ]
    for x, label, color, tag in zones:
        drawing.add(drawing.ellipse(center=(x + 125, 350), r=(142, 104), fill=color, opacity=0.18, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x + 125, 330, 19, EDITORIAL["white"], anchor="middle", width=20)
        _draw_wrapped_text(drawing, tag, x + 125, 405, 16, color, anchor="middle", width=16)
        _draw_editorial_arrow(drawing, x + 260, 350, x + 300, 350, EDITORIAL["muted"])
    drawing.add(drawing.ellipse(center=(482, 500), r=(240, 58), fill=EDITORIAL["purple"], opacity=0.13, stroke=EDITORIAL["purple"], stroke_width=3, stroke_dasharray="10 8"))
    _draw_wrapped_text(drawing, faithfulness, 482, 506, 17, EDITORIAL["purple"], anchor="middle", width=28)
    drawing.add(drawing.rect(insert=(720, 485), size=(304, 74), rx=16, fill=EDITORIAL["amber"], opacity=0.18, stroke=EDITORIAL["amber"], stroke_width=3))
    _draw_wrapped_text(drawing, refusal, 872, 528, 16, EDITORIAL["amber"], anchor="middle", width=28)


def _draw_semantic_peace_framework(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["WebWeaver Research Agent", "ACADREASON Evaluation", "Multi-Agent Debate Consensus", "Peace Objective", "Deception Mitigation"],
    )
    center = (640, 360)
    nodes = [(265, 280, labels[0], EDITORIAL["ice"]), (1000, 280, labels[1], EDITORIAL["amber"]), (640, 555, labels[2], EDITORIAL["purple"])]
    for x, y, label, color in nodes:
        start, end = _node_edge_points((x, y), center, radius=122)
        _draw_editorial_arrow(drawing, start[0], start[1], end[0], end[1], color)
    drawing.add(drawing.polygon(points=[(640, 250), (735, 305), (735, 415), (640, 470), (545, 415), (545, 305)], fill=EDITORIAL["panel_2"], opacity=1.0, stroke=EDITORIAL["green"], stroke_width=4))
    drawing.add(drawing.rect(insert=(548, 335), size=(184, 52), rx=12, fill=EDITORIAL["panel_2"], opacity=1.0, stroke=EDITORIAL["green"], stroke_width=2))
    _draw_wrapped_text(drawing, labels[3], center[0], center[1] + 4, 19, EDITORIAL["white"], anchor="middle", width=20)
    for x, y, label, color in nodes:
        drawing.add(drawing.rect(insert=(x - 145, y - 50), size=(290, 100), rx=18, fill=EDITORIAL["panel_2"], opacity=1.0, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, _compact_center_label(label), x, y + 4, 16, EDITORIAL["white"], anchor="middle", width=27)
    drawing.add(drawing.ellipse(center=center, r=(520, 260), fill="none", stroke=EDITORIAL["coral"], stroke_width=4, stroke_dasharray="12 10", opacity=0.78))
    _draw_wrapped_text(drawing, labels[4], center[0], 168, 20, EDITORIAL["coral"], anchor="middle", width=26)


def _draw_semantic_prompt_flow(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        [
            "Reference Image",
            "Text Prompt",
            "DemoDiff",
            "Generated Output",
            "Remasking Step",
        ],
    )
    left_a = _first_label_containing(labels, ("reference", "few-shot", "example"), labels[0])
    left_b = _first_label_containing(labels, ("prompt", "masked", "token"), labels[1])
    center = _first_label_containing(labels, ("demodiff", "remasking", "generator"), labels[2])
    output = _first_label_containing(labels, ("output", "refined"), labels[3])
    loop = _first_label_containing(labels, ("remask", "settle", "flow", "harpoon"), labels[4])
    panels = [(120, left_a, EDITORIAL["ice"]), (332, left_b, EDITORIAL["green"]), (574, center, EDITORIAL["purple"]), (876, output, EDITORIAL["coral"])]
    for x, label, color in panels:
        drawing.add(drawing.rect(insert=(x, 246), size=(180, 200), rx=18, fill=EDITORIAL["panel_2"], stroke=color, stroke_width=4, opacity=0.94))
        _draw_wrapped_text(drawing, label, x + 90, 286, 16, EDITORIAL["white"], anchor="middle", width=18)
        if _contains_any(label.lower(), ("image", "reference", "example")):
            drawing.add(drawing.circle(center=(x + 72, 366), r=28, fill=color, opacity=0.36))
            drawing.add(drawing.rect(insert=(x + 102, 342), size=(42, 58), rx=8, fill=color, opacity=0.24))
        elif _contains_any(label.lower(), ("prompt", "masked", "token")):
            for idx in range(4):
                drawing.add(drawing.rect(insert=(x + 34 + idx * 34, 352), size=(25, 26), rx=5, fill=color if idx % 2 else EDITORIAL["muted"], opacity=0.72))
            drawing.add(drawing.text("[MASK]", insert=(x + 52, 412), fill=EDITORIAL["gray"], font_size=14, font_family="Arial"))
        elif _contains_any(label.lower(), ("output", "refined")):
            for idx in range(5):
                drawing.add(drawing.rect(insert=(x + 30 + idx * 28, 350), size=(20, 36), rx=5, fill=color, opacity=0.7))
        else:
            for idx in range(3):
                drawing.add(drawing.rect(insert=(x + 44 + idx * 33, 340 + idx * 12), size=(84, 34), rx=8, fill=color, opacity=0.18, stroke=color, stroke_width=2))
    for start, end in ((300, 332), (512, 574), (754, 876)):
        _draw_editorial_arrow(drawing, start, 346, end - 10, 346, EDITORIAL["muted"])
    drawing.add(drawing.path(d="M 690 258 C 760 190, 850 210, 830 302", fill="none", stroke=EDITORIAL["amber"], stroke_width=4, stroke_dasharray="8 7", opacity=0.85))
    _draw_wrapped_text(drawing, loop, 780, 214, 16, EDITORIAL["amber"], anchor="middle", width=26)


def _draw_semantic_technique_overlap(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    context_text = _semantic_context_text(context)
    if _contains_any(context_text, ("scaling mediator", "mediator spoke", "conditional scaling", "wheel")):
        _draw_semantic_mediator_wheel(drawing, context)
        return
    if _contains_any(
        context_text,
        ("decodable", "causally used", "sae instance", "shared features", "feature overlap"),
    ):
        _draw_semantic_feature_venn(drawing, context)
        return
    if _contains_any(context_text, ("text embedding", "image embedding", "truth direction", "3d plot")):
        _draw_semantic_embedding_scatter(drawing, context)
        return
    if _contains_any(context_text, ("scientific validation", "malicious exploitation", "dual-use")):
        _draw_semantic_dual_use_venn(drawing, context)
        return
    if _contains_any(context_text, ("unified world model", "td-mpc2", "robot manipulation", "shared latent state")):
        _draw_semantic_world_model_summary(drawing, context)
        return
    labels = _semantic_labels(
        context,
        5,
        [
            "Image / Video",
            "Text",
            "Tabular / Science",
            "Go-with-the-Flow",
            "Don't Settle Too Early",
        ],
    )
    image = _first_label_containing(labels, ("image", "video"), "Image / Video")
    text = _first_label_containing(labels, ("text", "settle"), "Text")
    science = _first_label_containing(labels, ("tabular", "science", "harpoon"), "Tabular / Science")
    ellipses = [
        ((470, 350), (230, 145), EDITORIAL["ice"], image),
        ((650, 350), (230, 145), EDITORIAL["green"], text),
        ((560, 440), (235, 145), EDITORIAL["amber"], science),
    ]
    for index, (center, radius, color, label) in enumerate(ellipses):
        drawing.add(drawing.ellipse(center=center, r=radius, fill=color, opacity=0.15, stroke=color, stroke_width=4))
        label_y = center[1] + 62 if index == 2 else center[1] - 100
        drawing.add(drawing.rect(insert=(center[0] - 112, label_y - 25), size=(224, 34), rx=8, fill=EDITORIAL["panel_2"], opacity=0.86))
        _draw_wrapped_text(drawing, label, center[0], label_y, 16, EDITORIAL["white"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, labels[3], 430, 410, 16, EDITORIAL["ice"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, labels[4], 720, 410, 16, EDITORIAL["green"], anchor="middle", width=24)
    drawing.add(drawing.rect(insert=(430, 318), size=(250, 46), rx=10, fill=EDITORIAL["panel_2"], opacity=0.94))
    _draw_wrapped_text(drawing, "Shared mechanism overlap", 555, 346, 16, EDITORIAL["white"], anchor="middle", width=28)
    drawing.add(drawing.rect(insert=(468, 522), size=(244, 34), rx=8, fill=EDITORIAL["panel_2"], opacity=0.86))
    _draw_wrapped_text(drawing, "Output convergence", 590, 544, 15, EDITORIAL["gray"], anchor="middle", width=24)


def _draw_semantic_embedding_scatter(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Text Embedding Cluster", "Image Embedding Cluster", "Aligned Region", "Misaligned Region", "Truth Direction Axis"],
    )
    origin = (332, 505)
    drawing.add(drawing.line(start=origin, end=(840, 505), stroke=EDITORIAL["muted"], stroke_width=3))
    drawing.add(drawing.line(start=origin, end=(332, 205), stroke=EDITORIAL["muted"], stroke_width=3))
    drawing.add(drawing.line(start=origin, end=(500, 392), stroke=EDITORIAL["muted"], stroke_width=3))
    drawing.add(drawing.text("PC1: semantic content", insert=(695, 535), fill=EDITORIAL["gray"], font_size=15, font_family="Arial"))
    drawing.add(drawing.text("PC2: modality", insert=(208, 220), fill=EDITORIAL["gray"], font_size=15, font_family="Arial"))
    drawing.add(drawing.text("PC3: truth signal", insert=(492, 388), fill=EDITORIAL["gray"], font_size=15, font_family="Arial"))
    image_points = [(500, 360), (540, 346), (585, 374), (620, 338), (662, 370), (594, 314)]
    text_points = [(535, 385), (575, 366), (615, 394), (704, 278), (748, 252), (792, 292)]
    for x, y in image_points:
        drawing.add(drawing.circle(center=(x, y), r=10, fill=EDITORIAL["ice"], opacity=0.92))
    for x, y in text_points:
        drawing.add(drawing.circle(center=(x, y), r=10, fill=EDITORIAL["amber"], opacity=0.92))
    drawing.add(drawing.ellipse(center=(586, 366), r=(128, 70), fill=EDITORIAL["green"], opacity=0.13, stroke=EDITORIAL["green"], stroke_width=3))
    drawing.add(drawing.ellipse(center=(748, 278), r=(102, 62), fill=EDITORIAL["coral"], opacity=0.13, stroke=EDITORIAL["coral"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[2], 578, 268, 17, EDITORIAL["green"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, labels[3], 848, 286, 17, EDITORIAL["coral"], anchor="middle", width=20)
    _draw_editorial_arrow(drawing, 430, 440, 715, 292, EDITORIAL["purple"])
    _draw_wrapped_text(drawing, labels[4], 472, 430, 16, EDITORIAL["purple"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, labels[1], 214, 345, 17, EDITORIAL["ice"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[0], 1000, 345, 17, EDITORIAL["amber"], anchor="middle", width=18)


def _draw_semantic_dual_use_venn(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["WebWeaver", "Multi-Agent Debate", "ACADREASON", "Scientific Validation", "Malicious Exploitation"],
    )
    drawing.add(drawing.line(start=(244, 510), end=(1036, 510), stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.55))
    drawing.add(drawing.line(start=(244, 510), end=(244, 198), stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.55))
    drawing.add(drawing.text("Openness of Release", insert=(790, 540), fill=EDITORIAL["gray"], font_size=16, font_family="Arial"))
    drawing.add(drawing.text("Capability Level", insert=(155, 295), fill=EDITORIAL["gray"], font_size=16, font_family="Arial", transform="rotate(-90 155,295)"))
    drawing.add(drawing.ellipse(center=(512, 356), r=(245, 160), fill=EDITORIAL["ice"], opacity=0.17, stroke=EDITORIAL["ice"], stroke_width=4))
    drawing.add(drawing.ellipse(center=(748, 356), r=(245, 160), fill=EDITORIAL["coral"], opacity=0.17, stroke=EDITORIAL["coral"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[3], 430, 222, 20, EDITORIAL["ice"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, labels[4], 830, 222, 20, EDITORIAL["coral"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, labels[0], 392, 356, 18, EDITORIAL["white"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, labels[2], 448, 430, 18, EDITORIAL["white"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, labels[1], 640, 348, 19, EDITORIAL["amber"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, "Dual-Use Capabilities", 640, 420, 18, EDITORIAL["purple"], anchor="middle", width=24)
    _draw_wrapped_text(drawing, "autonomous reasoning + evidence synthesis + long context", 640, 472, 15, EDITORIAL["gray"], anchor="middle", width=42)


def _draw_semantic_world_model_summary(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Unified World Model", "Masked World Model", "TD-MPC2", "Robot Manipulation Scene", "Shared Latent State"],
    )
    scene = (620, 352)
    drawing.add(drawing.rect(insert=(scene[0] - 118, scene[1] - 82), size=(236, 164), rx=18, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=4))
    _draw_3d_axes(drawing, scene[0] - 48, scene[1] + 48, EDITORIAL["ice"])
    drawing.add(drawing.path(d=f"M {scene[0]-16} {scene[1]+24} L {scene[0]+38} {scene[1]-26} L {scene[0]+72} {scene[1]+10}", fill="none", stroke=EDITORIAL["amber"], stroke_width=7))
    drawing.add(drawing.rect(insert=(scene[0] + 54, scene[1] + 20), size=(42, 28), rx=6, fill=EDITORIAL["green"], opacity=0.8))
    _draw_wrapped_text(drawing, labels[3], scene[0], scene[1] + 108, 16, EDITORIAL["white"], anchor="middle", width=22)
    panels = [(270, 238, labels[0], EDITORIAL["coral"]), (935, 238, labels[1], EDITORIAL["purple"]), (270, 452, labels[2], EDITORIAL["green"])]
    for x, y, label, color in panels:
        drawing.add(drawing.rect(insert=(x - 112, y - 54), size=(224, 108), rx=16, fill=color, opacity=0.16, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, x, y - 10, 17, EDITORIAL["white"], anchor="middle", width=20)
        if "Unified" in label:
            for idx in range(4):
                drawing.add(drawing.rect(insert=(x - 82 + idx * 40, y + 18), size=(28, 22), rx=4, fill=color, opacity=0.62))
        elif "Masked" in label:
            for row in range(2):
                for col in range(4):
                    drawing.add(drawing.rect(insert=(x - 70 + col * 36, y + 8 + row * 24), size=(26, 18), rx=3, fill=EDITORIAL["muted"] if (row + col) % 2 else color, opacity=0.7))
        else:
            drawing.add(drawing.path(d=f"M {x-72} {y+32} C {x-22} {y-4}, {x+22} {y+52}, {x+76} {y+16}", fill="none", stroke=color, stroke_width=5))
        _draw_editorial_arrow(drawing, scene[0] - 120 if x < scene[0] else scene[0] + 120, scene[1], x + (118 if x < scene[0] else -118), y, color)
    drawing.add(drawing.circle(center=(900, 470), r=54, fill=EDITORIAL["amber"], opacity=0.18, stroke=EDITORIAL["amber"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[4], 900, 476, 17, EDITORIAL["amber"], anchor="middle", width=16)
    _draw_editorial_arrow(drawing, scene[0] + 118, scene[1] + 28, 846, 452, EDITORIAL["amber"])


def _draw_semantic_feature_venn(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Decodable but Not Used", "Causally Used", "Shared Features", "RNN Hidden State", "Transformer KV Cache"],
    )
    left = _first_label_containing(labels, ("decodable", "sae instance a"), labels[0])
    right = _first_label_containing(labels, ("causal", "sae instance b"), labels[1])
    shared = _first_label_containing(labels, ("shared", "truth"), labels[2])
    rnn = _first_label_containing(labels, ("rnn", "hidden", "state"), labels[3])
    transformer = _first_label_containing(labels, ("transformer", "retrieval", "kv"), labels[4])
    drawing.add(drawing.ellipse(center=(500, 360), r=(230, 155), fill=EDITORIAL["ice"], opacity=0.18, stroke=EDITORIAL["ice"], stroke_width=4))
    drawing.add(drawing.ellipse(center=(745, 360), r=(230, 155), fill=EDITORIAL["coral"], opacity=0.18, stroke=EDITORIAL["coral"], stroke_width=4))
    drawing.add(drawing.rect(insert=(515, 326), size=(228, 74), rx=12, fill=EDITORIAL["panel_2"], opacity=0.94, stroke=EDITORIAL["purple"], stroke_width=2))
    _draw_wrapped_text(drawing, left, 350, 292, 17, EDITORIAL["ice"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, right, 902, 292, 17, EDITORIAL["coral"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, shared, 629, 350, 16, EDITORIAL["purple"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, rnn, 370, 452, 15, EDITORIAL["white"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, transformer, 900, 452, 15, EDITORIAL["white"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, "Look Twice retracing -> causal correction", 640, 540, 17, EDITORIAL["green"], anchor="middle", width=42)


def _draw_semantic_mediator_wheel(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Optimizer Choice", "Data Quality", "Architecture", "Causal Priors", "Precision & Evaluation"],
    )
    extra = ["Topology", "Physical Priors", "Evaluation"]
    cx, cy = 640, 360
    spokes = [*labels, *extra]
    outer_nodes: list[tuple[float, float, str, str]] = []
    for index, label in enumerate(spokes[:8]):
        angle = -math.pi / 2 + index * math.tau / 8
        x = cx + math.cos(angle) * 305
        y = cy + math.sin(angle) * 205
        color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"], EDITORIAL["purple"], EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"]][index]
        drawing.add(drawing.line(start=(cx, cy), end=(x, y), stroke=color, stroke_width=3 + index % 3, opacity=0.62))
        outer_nodes.append((x, y, label, color))
    for x, y, label, color in outer_nodes:
        drawing.add(drawing.circle(center=(x, y), r=50, fill=EDITORIAL["panel_2"], opacity=0.97, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, _compact_center_label(label), x, y + 4, 15, EDITORIAL["white"], anchor="middle", width=17)
    drawing.add(drawing.circle(center=(cx, cy), r=76, fill=EDITORIAL["panel_2"], opacity=1.0, stroke=EDITORIAL["purple"], stroke_width=5))
    _draw_wrapped_text(drawing, "Conditional Scaling", cx, cy + 5, 18, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, "Entropy-SGD: wide-valley bias", 236, 550, 16, EDITORIAL["ice"], anchor="middle", width=28)
    _draw_wrapped_text(drawing, "Sharpness vs generalization", 990, 550, 16, EDITORIAL["amber"], anchor="middle", width=28)


def _draw_semantic_attrition_funnel(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["AI-generated papers", "Quality Metrics", "Critical Filter", "Experiment Failure Rate", "Final Yield"],
    )
    funnel = [(260, 225), (1020, 225), (760, 520), (520, 520)]
    drawing.add(drawing.polygon(points=funnel, fill=EDITORIAL["ice"], opacity=0.12, stroke=EDITORIAL["ice"], stroke_width=4))
    bands = [(250, labels[0], EDITORIAL["ice"]), (332, labels[1], EDITORIAL["amber"]), (414, labels[2], EDITORIAL["purple"]), (498, labels[4], EDITORIAL["green"])]
    for y, label, color in bands:
        drawing.add(drawing.line(start=(330, y), end=(950, y), stroke=color, stroke_width=3, opacity=0.55))
    for index in range(24):
        x = 355 + (index % 8) * 78
        y = 238 + (index // 8) * 30
        drawing.add(drawing.circle(center=(x, y), r=7, fill=EDITORIAL["white"], opacity=0.6))
    for index in range(10):
        x = 555 + (index % 5) * 46
        y = 448 + (index // 5) * 26
        hex_points = [
            (x + math.cos(math.pi / 3 * step) * 12, y + math.sin(math.pi / 3 * step) * 12)
            for step in range(6)
        ]
        drawing.add(drawing.polygon(points=hex_points, fill=EDITORIAL["green"], opacity=0.7))
    drawing.add(drawing.path(d="M 640 528 C 606 560, 608 602, 640 615 C 672 602, 674 560, 640 528", fill=EDITORIAL["coral"], opacity=0.88))
    label_positions = [
        (labels[0], 640, 218, EDITORIAL["ice"], 340),
        (labels[1], 640, 306, EDITORIAL["amber"], 300),
        (labels[2], 640, 390, EDITORIAL["purple"], 330),
        (labels[4], 640, 536, EDITORIAL["green"], 330),
    ]
    for label, x, y, color, width in label_positions:
        drawing.add(drawing.rect(insert=(x - width / 2, y - 27), size=(width, 38), rx=9, fill=EDITORIAL["panel_2"], opacity=0.96))
        _draw_wrapped_text(drawing, label, x, y - 2, 17, color, anchor="middle", width=max(22, int(width // 11)))
    _draw_wrapped_text(drawing, labels[3], 985, 392, 17, EDITORIAL["coral"], anchor="middle", width=26)
    _draw_wrapped_text(drawing, "1 Weak Binder", 640, 640, 18, EDITORIAL["coral"], anchor="middle", width=18)


def _draw_semantic_satellite_query(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Satellite Image Input", "GeoChat VLM", "Grounded Region Query", "Natural Language Answer", "Bounding Box Overlay"],
    )
    drawing.add(drawing.rect(insert=(120, 240), size=(330, 260), rx=18, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=4))
    for row in range(5):
        for col in range(6):
            color = [EDITORIAL["green"], EDITORIAL["ice"], EDITORIAL["amber"]][(row + col) % 3]
            drawing.add(drawing.rect(insert=(135 + col * 52, 258 + row * 45), size=(48, 41), fill=color, opacity=0.16 + 0.04 * row))
    drawing.add(drawing.rect(insert=(250, 320), size=(92, 70), rx=4, fill="none", stroke=EDITORIAL["coral"], stroke_width=5))
    _draw_wrapped_text(drawing, labels[0], 285, 532, 18, EDITORIAL["ice"], anchor="middle", width=26)
    _draw_wrapped_text(drawing, labels[4], 296, 304, 16, EDITORIAL["coral"], anchor="middle", width=22)
    drawing.add(drawing.rect(insert=(520, 302), size=(210, 148), rx=18, fill=EDITORIAL["purple"], opacity=0.18, stroke=EDITORIAL["purple"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[1], 625, 360, 20, EDITORIAL["white"], anchor="middle", width=20)
    _draw_wrapped_text(drawing, labels[2], 625, 424, 15, EDITORIAL["gray"], anchor="middle", width=24)
    drawing.add(drawing.rect(insert=(820, 286), size=(252, 178), rx=18, fill=EDITORIAL["green"], opacity=0.16, stroke=EDITORIAL["green"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[3], 946, 362, 20, EDITORIAL["white"], anchor="middle", width=24)
    _draw_editorial_arrow(drawing, 452, 370, 516, 370, EDITORIAL["muted"])
    _draw_editorial_arrow(drawing, 734, 370, 818, 370, EDITORIAL["muted"])
    drawing.add(drawing.path(d="M 946 470 C 780 585, 410 580, 296 390", fill="none", stroke=EDITORIAL["coral"], stroke_width=4, stroke_dasharray="9 8", opacity=0.76))


def _draw_semantic_layered_map(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Optical Layer", "SAR Layer", "IR Layer", "SM3Det Fusion", "Detected Objects"],
    )
    layer_specs = [(210, 220, labels[0], EDITORIAL["ice"]), (270, 286, labels[1], EDITORIAL["amber"]), (330, 352, labels[2], EDITORIAL["coral"])]
    for x, y, label, color in layer_specs:
        pts = [(x, y), (760, y - 38), (900, y + 52), (350, y + 94)]
        drawing.add(drawing.polygon(points=pts, fill=color, opacity=0.16, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x + 40, y + 32, 17, EDITORIAL["white"], anchor="start", width=18)
    drawing.add(drawing.polygon(points=[(448, 470), (890, 438), (1010, 510), (560, 548)], fill=EDITORIAL["green"], opacity=0.2, stroke=EDITORIAL["green"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[3], 720, 488, 20, EDITORIAL["green"], anchor="middle", width=26)
    for x, y in [(660, 500), (760, 482), (842, 512)]:
        drawing.add(drawing.rect(insert=(x, y), size=(46, 30), fill="none", stroke=EDITORIAL["white"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[4], 900, 568, 17, EDITORIAL["white"], anchor="middle", width=22)
    for x in (470, 560, 650):
        _draw_editorial_arrow(drawing, x, 430, x + 82, 468, EDITORIAL["muted"])


def _draw_semantic_hierarchical_search(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Flat Keyword Search", "Hierarchical Register Index", "Paper-Level Node", "Claim-Level Node", "Query"],
    )
    drawing.add(drawing.line(start=(640, 220), end=(640, 545), stroke=EDITORIAL["muted"], stroke_width=2))
    _draw_wrapped_text(drawing, labels[4], 640, 190, 20, EDITORIAL["ice"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[0], 320, 245, 19, EDITORIAL["coral"], anchor="middle", width=24)
    for idx in range(4):
        drawing.add(drawing.rect(insert=(205, 300 + idx * 58), size=(230, 38), rx=8, fill=EDITORIAL["coral"], opacity=0.16, stroke=EDITORIAL["coral"], stroke_width=2))
        _draw_wrapped_text(drawing, f"paper result {idx + 1}", 320, 325 + idx * 58, 14, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[1], 930, 245, 19, EDITORIAL["green"], anchor="middle", width=28)
    drawing.add(drawing.circle(center=(930, 318), r=42, fill=EDITORIAL["green"], opacity=0.2, stroke=EDITORIAL["green"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[2], 930, 322, 15, EDITORIAL["white"], anchor="middle", width=15)
    claim_points = [(812, 438), (930, 478), (1048, 438)]
    for point in claim_points:
        drawing.add(drawing.line(start=(930, 360), end=point, stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.62))
        drawing.add(drawing.circle(center=point, r=44, fill=EDITORIAL["panel_2"], opacity=0.97, stroke=EDITORIAL["ice"], stroke_width=3))
        _draw_wrapped_text(drawing, _compact_center_label(labels[3]), point[0], point[1] + 4, 14, EDITORIAL["white"], anchor="middle", width=14)
    _draw_editorial_arrow(drawing, 640, 205, 320, 292, EDITORIAL["muted"])
    _draw_editorial_arrow(drawing, 640, 205, 930, 274, EDITORIAL["muted"])


def _draw_semantic_math_chain(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Clipped SGD Update", "Clipping Operator clip(g, lambda)", "Heavy-Tailed Noise xi", "High-Probability Bound", "Convergence Rate"],
    )
    drawing.add(drawing.rect(insert=(190, 210), size=(890, 74), rx=14, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=3))
    drawing.add(drawing.text("x[t+1] = x[t] - eta * clip(grad f(x[t]) + xi[t], lambda)", insert=(640, 258), fill=EDITORIAL["white"], font_size=24, font_family="Courier New", text_anchor="middle"))
    nodes = [(210, labels[2], EDITORIAL["coral"]), (430, labels[1], EDITORIAL["purple"]), (670, labels[0], EDITORIAL["ice"]), (930, labels[3], EDITORIAL["green"])]
    for start, end in ((302, 430), (522, 670), (762, 930)):
        _draw_editorial_arrow(drawing, start, 405, end - 96, 405, EDITORIAL["muted"])
    for x, label, color in nodes:
        drawing.add(drawing.ellipse(center=(x, 405), r=(96, 56), fill=EDITORIAL["panel_2"], opacity=0.97, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, _compact_center_label(label), x, 410, 15, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[4], 640, 520, 18, EDITORIAL["green"], anchor="middle", width=30)


def _draw_semantic_objective_web(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Objective Function L(theta)", "WebWeaver Context Overflow", "ACADREASON Reasoning Gaps", "Multi-Agent Debate Disagreement", "Societal Complexity Web"],
    )
    formula_x, formula_y, formula_w, formula_h = 92, 314, 326, 104
    drawing.add(drawing.rect(insert=(formula_x, formula_y), size=(formula_w, formula_h), rx=16, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=4))
    drawing.add(drawing.text("L(theta) = sum loss_i", insert=(formula_x + formula_w / 2, 374), fill=EDITORIAL["white"], font_size=18, font_family="Courier New", text_anchor="middle"))
    _draw_wrapped_text(drawing, labels[0], formula_x + formula_w / 2, 458, 16, EDITORIAL["ice"], anchor="middle", width=28)
    centers = [(560, 260), (610, 410), (820, 330), (980, 240), (1010, 480)]
    colors = [EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["purple"], EDITORIAL["green"], EDITORIAL["ice"]]
    for start in centers:
        for end in centers:
            if start < end:
                drawing.add(drawing.line(start=start, end=end, stroke=EDITORIAL["gray"], stroke_width=2.4, opacity=0.34))
    for center, label, color in zip(centers, labels[1:], colors):
        _draw_editorial_arrow(drawing, formula_x + formula_w, formula_y + formula_h / 2, center[0] - 116, center[1], EDITORIAL["muted"])
    for center, label, color in zip(centers, labels[1:], colors):
        drawing.add(drawing.ellipse(center=center, r=(108, 54), fill=EDITORIAL["panel_2"], opacity=0.97, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, center[0], center[1] + 5, 15, EDITORIAL["white"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, "Power asymmetry | value plurality | temporal drift", 790, 585, 17, EDITORIAL["gray"], anchor="middle", width=48)


def _draw_semantic_training_dashboard(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Loss Landscape", "EoS Detector", "Sharpness Monitor", "Weight Decay Controller", "Convergence Alert"],
    )
    drawing.add(drawing.rect(insert=(98, 220), size=(1084, 342), rx=22, fill=EDITORIAL["panel_2"], opacity=0.92, stroke=EDITORIAL["muted"], stroke_width=3))
    # Landscape contour panel.
    drawing.add(drawing.rect(insert=(130, 250), size=(390, 282), rx=18, fill=EDITORIAL["navy"], stroke=EDITORIAL["ice"], stroke_width=3))
    for radius, color in [(130, EDITORIAL["ice"]), (92, EDITORIAL["green"]), (48, EDITORIAL["coral"])]:
        drawing.add(drawing.ellipse(center=(325, 380), r=(radius, radius * 0.55), fill="none", stroke=color, stroke_width=3, opacity=0.65))
    _draw_wrapped_text(drawing, labels[0], 325, 506, 17, EDITORIAL["ice"], anchor="middle", width=26)
    widgets = [(590, 258, labels[1], EDITORIAL["coral"]), (790, 258, labels[2], EDITORIAL["amber"]), (590, 405, labels[3], EDITORIAL["green"]), (790, 405, labels[4], EDITORIAL["purple"])]
    for x, y, label, color in widgets:
        drawing.add(drawing.rect(insert=(x, y), size=(170, 100), rx=14, fill=color, opacity=0.16, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, x + 85, y + 44, 16, EDITORIAL["white"], anchor="middle", width=16)
        drawing.add(drawing.line(start=(x + 35, y + 74), end=(x + 135, y + 74), stroke=EDITORIAL["muted"], stroke_width=3))
        drawing.add(drawing.circle(center=(x + 115, y + 74), r=8, fill=color))
    _draw_editorial_arrow(drawing, 520, 380, 590, 455, EDITORIAL["green"])
    _draw_wrapped_text(drawing, "high sharpness -> EoS -> weight decay", 970, 362, 17, EDITORIAL["gray"], anchor="middle", width=30)


def _draw_semantic_feature_dictionary_ui(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Feature Dictionary UI", "Search Bar", "LLM Layer Index", "RNN State Feature", "Attention Feature"],
    )
    drawing.add(drawing.rect(insert=(135, 222), size=(1010, 350), rx=20, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["ice"], stroke_width=3, opacity=0.94))
    drawing.add(drawing.rect(insert=(175, 268), size=(790, 48), rx=16, fill=EDITORIAL["navy"], stroke=EDITORIAL["green"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[1], 230, 298, 16, EDITORIAL["green"], anchor="start", width=20)
    drawing.add(drawing.rect(insert=(175, 336), size=(178, 180), rx=14, fill=EDITORIAL["ice"], opacity=0.12, stroke=EDITORIAL["ice"], stroke_width=2))
    _draw_wrapped_text(drawing, labels[0], 264, 250, 19, EDITORIAL["white"], anchor="middle", width=26)
    _draw_wrapped_text(drawing, labels[2], 264, 430, 16, EDITORIAL["ice"], anchor="middle", width=20)
    cards = [(410, labels[3], EDITORIAL["amber"]), (720, labels[4], EDITORIAL["purple"])]
    for x, label, color in cards:
        drawing.add(drawing.rect(insert=(x, 350), size=(260, 116), rx=16, fill=color, opacity=0.16, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, x + 130, 386, 17, EDITORIAL["white"], anchor="middle", width=24)
        for idx in range(5):
            drawing.add(drawing.rect(insert=(x + 42 + idx * 32, 420 - idx * 7), size=(20, 34 + idx * 7), rx=4, fill=color, opacity=0.58))
    _draw_editorial_arrow(drawing, 965, 292, 1040, 380, EDITORIAL["muted"])


def _draw_semantic_expert_dashboard(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Expert Head Activations", "State-Tracking Signal", "Retrieval Bottleneck", "Hallucination Alert", "Monitoring Dashboard"],
    )
    drawing.add(drawing.rect(insert=(110, 225), size=(1060, 350), rx=22, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.94))
    _draw_wrapped_text(drawing, labels[4], 640, 255, 20, EDITORIAL["white"], anchor="middle", width=32)
    _draw_wrapped_text(drawing, labels[0], 640, 300, 17, EDITORIAL["ice"], anchor="middle", width=32)
    panels = [(170, labels[1], EDITORIAL["ice"]), (470, labels[2], EDITORIAL["amber"]), (770, labels[3], EDITORIAL["coral"])]
    for x, label, color in panels:
        drawing.add(drawing.rect(insert=(x, 335), size=(240, 145), rx=16, fill=color, opacity=0.15, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label, x + 120, 366, 17, EDITORIAL["white"], anchor="middle", width=22)
    drawing.add(drawing.polyline(points=[(205, 440), (250, 395), (295, 448), (340, 392), (385, 434)], fill="none", stroke=EDITORIAL["ice"], stroke_width=4))
    for idx, h in enumerate([44, 76, 58]):
        drawing.add(drawing.rect(insert=(535 + idx * 45, 455 - h), size=(30, h), rx=4, fill=EDITORIAL["amber"], opacity=0.72))
    drawing.add(drawing.circle(center=(890, 425), r=34, fill=EDITORIAL["coral"], opacity=0.82))
    drawing.add(drawing.text("!", insert=(883, 438), fill=EDITORIAL["white"], font_size=34, font_family="Arial", font_weight="bold"))


def _draw_semantic_world_model_landscape(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Unified World Models", "Masked World Models", "TD-MPC2", "Video Generation", "Closed-Loop Decision"],
    )
    x0, y0, w, h = 230, 225, 780, 345
    drawing.add(drawing.rect(insert=(x0, y0), size=(w, h), rx=18, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.94))
    drawing.add(drawing.line(start=(x0 + 60, y0 + h - 55), end=(x0 + w - 50, y0 + h - 55), stroke=EDITORIAL["gray"], stroke_width=3))
    drawing.add(drawing.line(start=(x0 + 60, y0 + h - 55), end=(x0 + 60, y0 + 50), stroke=EDITORIAL["gray"], stroke_width=3))
    _draw_wrapped_text(drawing, labels[3], x0 + 90, y0 + h + 28, 16, EDITORIAL["gray"], anchor="middle", width=18)
    _draw_wrapped_text(drawing, labels[4], x0 + w - 70, y0 + h + 28, 16, EDITORIAL["green"], anchor="middle", width=22)
    _draw_wrapped_text(drawing, "Action Integration", x0 - 40, y0 + 90, 16, EDITORIAL["gray"], anchor="middle", width=18)
    nodes = [(x0 + 520, y0 + 165, labels[0], EDITORIAL["green"]), (x0 + 430, y0 + 215, labels[1], EDITORIAL["ice"]), (x0 + 610, y0 + 95, labels[2], EDITORIAL["coral"])]
    for x, y, label, color in nodes:
        drawing.add(drawing.circle(center=(x, y), r=43, fill=color, opacity=0.2, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, y + 5, 15, EDITORIAL["white"], anchor="middle", width=14)
    _draw_editorial_arrow(drawing, x0 + 135, y0 + h - 90, x0 + 690, y0 + 105, EDITORIAL["amber"])
    _draw_wrapped_text(drawing, "Paradigm Shift", x0 + 420, y0 + 102, 18, EDITORIAL["amber"], anchor="middle", width=20)


def _draw_semantic_simulator_benchmark(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _semantic_labels(
        context,
        5,
        ["Unified World Model Simulator", "TD-MPC2 Agent", "Diffusion Policy Agent", "Standardized Task Suite", "Interactive Evaluation Loop"],
    )
    center = (640, 360)
    drawing.add(drawing.circle(center=center, r=82, fill=EDITORIAL["purple"], opacity=0.22, stroke=EDITORIAL["purple"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[3], center[0], center[1] + 5, 18, EDITORIAL["white"], anchor="middle", width=18)
    simulator = (640, 210)
    agents = [(330, 450, labels[1], EDITORIAL["ice"]), (970, 450, labels[2], EDITORIAL["green"])]
    drawing.add(drawing.rect(insert=(simulator[0] - 150, simulator[1] - 52), size=(300, 104), rx=18, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["coral"], stroke_width=4))
    _draw_wrapped_text(drawing, labels[0], simulator[0], simulator[1] + 5, 18, EDITORIAL["white"], anchor="middle", width=26)
    _draw_editorial_arrow(drawing, 640, 262, 640, 278, EDITORIAL["muted"])
    for x, y, label, color in agents:
        drawing.add(drawing.rect(insert=(x - 132, y - 50), size=(264, 100), rx=18, fill=color, opacity=0.16, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x, y + 5, 18, EDITORIAL["white"], anchor="middle", width=22)
        _draw_editorial_arrow(drawing, center[0], center[1], x, y - 55, EDITORIAL["muted"])
        _draw_editorial_arrow(drawing, x, y + 55, center[0], center[1] + 85, color)
    drawing.add(drawing.path(d="M 320 520 C 520 650, 920 650, 990 520", fill="none", stroke=EDITORIAL["amber"], stroke_width=5, stroke_dasharray="10 8", opacity=0.82))
    _draw_wrapped_text(drawing, labels[4], 640, 620, 18, EDITORIAL["amber"], anchor="middle", width=32)


def _draw_repaired_editorial_figure(
    drawing: svgwrite.Drawing,
    *,
    slide_number: int,
    slide_title: str,
    need: str,
    figure_type: str,
    context: dict[str, object],
    repair_profile: dict[str, object],
) -> None:
    """Render a projection-safe replacement guided by vision-LLM QC feedback."""

    del repair_profile
    del slide_number, slide_title, need
    if figure_type in {"flowchart", "search"}:
        _draw_repaired_flow(drawing, context, figure_type)
    elif figure_type == "matrix":
        _draw_repaired_matrix(drawing, context)
    elif figure_type == "radar":
        _draw_repaired_radar(drawing, context)
    elif figure_type in {"bar_chart", "line_chart", "memory_breakdown", "attention_map"}:
        _draw_repaired_chart(drawing, context, figure_type)
    else:
        _draw_repaired_map(drawing, context)


def _draw_repaired_header(
    drawing: svgwrite.Drawing,
    slide_number: int,
    slide_title: str,
    need: str,
) -> None:
    drawing.add(
        drawing.rect(
            insert=(66, 54),
            size=(142, 44),
            rx=22,
            fill=EDITORIAL["ice"],
            opacity=0.2,
        )
    )
    drawing.add(
        drawing.text(
            f"SLIDE {slide_number:02d}",
            insert=(137, 83),
            text_anchor="middle",
            fill=EDITORIAL["ice"],
            font_size=20,
            font_family="Arial",
            font_weight="bold",
        )
    )
    for offset, line in enumerate(_wrap_text(slide_title, 46)[:2]):
        drawing.add(
            drawing.text(
                line,
                insert=(238, 76 + offset * 39),
                fill=EDITORIAL["white"],
                font_size=36,
                font_family="Arial",
                font_weight="bold",
            )
        )


def _draw_repaired_flow(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
    figure_type: str,
) -> None:
    labels = _repair_labels(context, figure_type, 4)
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]]
    y = 354
    for index, label in enumerate(labels):
        x = 86 + index * 294
        drawing.add(
            drawing.rect(
                insert=(x, y - 70),
                size=(232, 140),
                rx=18,
                fill=colors[index],
                opacity=0.22,
                stroke=colors[index],
                stroke_width=4,
            )
        )
        drawing.add(drawing.circle(center=(x + 38, y - 34), r=22, fill=colors[index], opacity=0.95))
        drawing.add(
            drawing.text(
                str(index + 1),
                insert=(x + 30, y - 26),
                fill=EDITORIAL["navy"],
                font_size=24,
                font_family="Arial",
                font_weight="bold",
            )
        )
        _draw_wrapped_text(
            drawing,
            label,
            x + 116,
            y + 10,
            36,
            EDITORIAL["white"],
            anchor="middle",
            width=11,
        )
        if index < 3:
            _draw_editorial_arrow(drawing, x + 242, y, x + 284, y, EDITORIAL["muted"])
    _draw_large_callout(drawing, _repair_callout(context, figure_type))


def _draw_repaired_matrix(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _repair_labels(context, "matrix", 4)
    x0, y0 = 220, 238
    for index, label in enumerate(labels):
        x = x0 + (index % 2) * 430
        y = y0 + (index // 2) * 150
        color = [EDITORIAL["green"], EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["coral"]][index]
        drawing.add(drawing.rect(insert=(x, y), size=(340, 116), rx=18, fill=color, opacity=0.22, stroke=color, stroke_width=4))
        _draw_wrapped_text(drawing, label, x + 170, y + 68, 38, EDITORIAL["white"], anchor="middle", width=12)
    _draw_large_callout(drawing, "Trade-offs are explicit design inputs")


def _draw_repaired_radar(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    axes = _repair_labels(context, "radar", 4)
    cx, cy, radius = 640, 388, 150
    for level in range(1, 4):
        points = _radar_points(cx, cy, radius * level / 3, len(axes), [1.0] * len(axes))
        drawing.add(drawing.polygon(points=points, fill="none", stroke=EDITORIAL["muted"], stroke_width=2, opacity=0.35))
    points = _radar_points(cx, cy, radius, len(axes), [0.82, 0.58, 0.72, 0.64])
    drawing.add(drawing.polygon(points=points, fill=EDITORIAL["coral"], opacity=0.32, stroke=EDITORIAL["coral"], stroke_width=5))
    for index, axis in enumerate(axes):
        angle = -math.pi / 2 + index * 2 * math.pi / len(axes)
        x = cx + math.cos(angle) * (radius + 82)
        y = cy + math.sin(angle) * (radius + 82)
        _draw_wrapped_text(drawing, axis, x, y, 31, EDITORIAL["gray"], anchor="middle", width=11)
    _draw_large_callout(drawing, "Evaluation spans multiple risk surfaces")


def _draw_repaired_chart(
    drawing: svgwrite.Drawing,
    context: dict[str, object],
    figure_type: str,
) -> None:
    labels = _repair_labels(context, figure_type, 3)
    x0, base_y = 220, 548
    chart_text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    if figure_type == "line_chart":
        chart_title = "KV-cache memory growth" if "kv" in chart_text or "context" in chart_text else "Metric trend"
        y_label = "Memory x" if "kv" in chart_text or "context" in chart_text else "Score"
        x_label = "Context"
    else:
        chart_title = "Relative inference throughput" if "hardware" in chart_text or "throughput" in chart_text else "Benchmark score"
        y_label = "Score"
        x_label = "Variant"
    drawing.add(
        drawing.text(
            chart_title,
            insert=(x0 - 40, 122),
            fill=EDITORIAL["white"],
            font_size=50,
            font_family="Arial",
            font_weight="bold",
        )
    )
    drawing.add(
        drawing.text(
            y_label,
            insert=(x0 - 42, 174),
            fill=EDITORIAL["gray"],
            font_size=40,
            font_family="Arial",
            font_weight="bold",
        )
    )
    drawing.add(drawing.line(start=(x0 - 40, base_y), end=(1088, base_y), stroke=EDITORIAL["muted"], stroke_width=5))
    drawing.add(drawing.line(start=(x0 - 40, 198), end=(x0 - 40, base_y), stroke=EDITORIAL["muted"], stroke_width=4, opacity=0.75))
    if figure_type == "line_chart":
        points = [(x0, base_y - 56), (x0 + 270, base_y - 132), (x0 + 560, base_y - 228), (x0 + 868, base_y - 340)]
        drawing.add(drawing.polyline(points=points, fill="none", stroke=EDITORIAL["coral"], stroke_width=12))
        for (x, y), label, value in zip(points, ["4K", "16K", "64K", "128K"], ["1x", "4x", "16x", "32x"]):
            drawing.add(drawing.circle(center=(x, y), r=19, fill=EDITORIAL["navy"], stroke=EDITORIAL["coral"], stroke_width=7))
            drawing.add(
                drawing.text(
                    label,
                    insert=(x - 28, base_y + 58),
                    fill=EDITORIAL["gray"],
                    font_size=36,
                    font_family="Arial",
                    font_weight="bold",
                )
            )
            drawing.add(
                drawing.text(
                    value,
                    insert=(x - 24, y - 42),
                    fill=EDITORIAL["white"],
                    font_size=34,
                    font_family="Arial",
                    font_weight="bold",
                )
            )
    else:
        heights = [178, 278, 360]
        values = ["1.0x", "1.7x", "2.4x"] if "hardware" in chart_text or "throughput" in chart_text else ["Base", "+18%", "+31%"]
        for index, (label, height, value) in enumerate(zip(labels, heights, values)):
            x = x0 + index * 286
            color = [EDITORIAL["coral"], EDITORIAL["ice"], EDITORIAL["green"]][index]
            drawing.add(drawing.rect(insert=(x, base_y - height), size=(188, height), rx=16, fill=color, opacity=0.84))
            drawing.add(
                drawing.text(
                    value,
                    insert=(x + 94, base_y - height + 66),
                    fill=EDITORIAL["navy"],
                    font_size=44,
                    font_family="Arial",
                    font_weight="bold",
                    text_anchor="middle",
                )
            )
            _draw_wrapped_text(drawing, label, x + 94, base_y + 56, 38, EDITORIAL["gray"], anchor="middle", width=10)
    if figure_type == "line_chart":
        drawing.add(
            drawing.text(
                x_label,
                insert=(640, 640),
                fill=EDITORIAL["gray"],
                font_size=40,
                font_family="Arial",
                font_weight="bold",
                text_anchor="middle",
            )
        )
    _draw_large_callout(drawing, _repair_callout(context, figure_type))


def _draw_repaired_map(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _repair_labels(context, "diagram", 4)
    centers = [(280, 340), (520, 470), (760, 340), (1000, 470)]
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"], EDITORIAL["purple"]]
    previous: tuple[int, int] | None = None
    for center, label, color in zip(centers, labels, colors):
        if previous is not None:
            start, end = _node_edge_points(previous, center, radius=132)
            _draw_editorial_arrow(drawing, start[0], start[1], end[0], end[1], EDITORIAL["muted"])
        drawing.add(drawing.circle(center=center, r=116, fill=color, opacity=0.24, stroke=color, stroke_width=5))
        _draw_wrapped_text(drawing, label, center[0], center[1] + 10, 48, EDITORIAL["white"], anchor="middle", width=10)
        previous = center
    _draw_large_callout(drawing, _repair_callout(context, "diagram"))


def _draw_density_fallback(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    """Render density-target figures as simple, readable visual anchors."""

    labels = _repair_labels(context, "flowchart", 4)
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]]
    y = 340
    for index, label in enumerate(labels):
        x = 92 + index * 294
        drawing.add(
            drawing.rect(
                insert=(x, y - 74),
                size=(226, 148),
                rx=20,
                fill=colors[index],
                opacity=0.24,
                stroke=colors[index],
                stroke_width=5,
            )
        )
        _draw_wrapped_text(
            drawing,
            label,
            x + 113,
            y + 8,
            34,
            EDITORIAL["white"],
            anchor="middle",
            width=11,
        )
        if index < 3:
            _draw_editorial_arrow(drawing, x + 238, y, x + 282, y, EDITORIAL["muted"])
    callout = _repair_callout(context, "flowchart")
    drawing.add(
        drawing.rect(
            insert=(148, 520),
            size=(984, 104),
            rx=18,
            fill=EDITORIAL["panel_2"],
            opacity=0.88,
            stroke=EDITORIAL["ice"],
            stroke_width=2,
        )
    )
    _draw_wrapped_text(drawing, callout, 640, 562, 24, EDITORIAL["ice"], anchor="middle", width=58)


def _node_edge_points(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy) or 1.0
    ux = dx / distance
    uy = dy / distance
    return (
        (start[0] + ux * radius, start[1] + uy * radius),
        (end[0] - ux * radius, end[1] - uy * radius),
    )


def _draw_large_callout(drawing: svgwrite.Drawing, text: str) -> None:
    del drawing, text


def _repair_labels(context: dict[str, object], figure_type: str, count: int) -> list[str]:
    """Return content-specific large labels for projection-safe repair figures."""

    text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    candidates: list[str]
    if "architecture synthesis" in text:
        candidates = ["Design goal", "Search space", "Task score", "Tailored model"]
    elif "decoding model mechanics" in text:
        candidates = ["Token trace", "Retrieval head", "Causal test", "Factuality"]
    elif "long-context" in text or "long context" in text or "kv-cache" in text or "kv cache" in text:
        candidates = ["Context", "KV cache", "Eviction", "Recall"]
    elif "summary" in text or "takeaway" in text:
        candidates = ["Synthesis", "Memory", "Scaffolds", "Evaluation"]
    elif "agentic workflow" in text or ("agents" in text and "tools" in text):
        candidates = ["Task goal", "Agent-tool sets", "Evaluate runs", "Best workflow"]
    elif figure_type == "bar_chart" and ("hardware" in text or "throughput" in text):
        candidates = ["Standard", "Search tuned", "Hardware fit", "Deployed"]
    elif "star" in text and ("synthesis" in text or "tailored" in text):
        candidates = ["Task", "Search", "Proxy", "Model"]
    elif "evolutionary" in text or "candidate" in text:
        candidates = ["Candidate ops", "Mutate blocks", "Score fitness", "Keep winners"]
    elif "multi-agent" in text or "orchestrator" in text:
        candidates = ["Control", "Agents", "Tools", "Memory"]
    elif "mutual deliberation" in text or "second opinion" in text:
        candidates = ["Draft", "Critique", "Revise", "Agree"]
    elif "decision tree" in text or "rationale" in text:
        candidates = ["Rationale pool", "Score paths", "Prune flaws", "Select answer"]
    elif "retrieval head" in text or "factual token" in text:
        candidates = ["Query", "Head", "Fact", "Answer"]
    elif "memory system" in text:
        candidates = ["Context stream", "Cache", "Episodic store", "Retriever"]
    elif "kv-cache" in text or "kv cache" in text or "memory allocation" in text:
        candidates = ["Weights", "KV cache", "Activations", "Runtime state"]
    elif "multi-modal" in text or "multimodal" in text:
        candidates = ["Text", "Vision", "Router", "Decode"]
    elif "cybench" in text or "cybersecurity" in text:
        candidates = ["Exploit", "Defend", "Forensics", "Planning"]
    elif "benchmark" in text or "evaluation" in text:
        candidates = ["Baseline", "Long context", "Agentic task", "Risk score"]
    elif "training cutoff" in text or "livexiv" in text or "live ingestion" in text:
        candidates = ["Cutoff", "New papers", "Live ingest", "Refresh eval"]
    elif figure_type == "matrix":
        candidates = ["Capability", "Latency", "Memory", "Cost"]
    elif figure_type == "radar":
        candidates = ["Exploit", "Defend", "Forensics", "Planning"]
    elif figure_type == "line_chart":
        candidates = ["4K", "16K", "64K", "128K"]
    elif figure_type == "bar_chart":
        candidates = ["Baseline", "Search tuned", "Hardware fit", "Deployed"]
    elif figure_type == "flowchart":
        candidates = ["Inputs", "Transform", "Validate", "Output"]
    else:
        candidates = ["Signal", "Memory", "Scaffold", "Measure"]
    while len(candidates) < count:
        candidates.append(candidates[len(candidates) % len(candidates)])
    return candidates[:count]


def _repair_callout(context: dict[str, object], figure_type: str) -> str:
    text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    if "agentic workflow" in text or ("agents" in text and "tools" in text):
        return "Search selects the agent pipeline that solves the task"
    if "star" in text and ("synthesis" in text or "tailored" in text):
        return "Constraints become measured architecture choices"
    if "evolutionary" in text or "candidate" in text:
        return "Evolutionary search prunes the architecture space"
    if "kv-cache" in text or "kv cache" in text:
        return "Long context is limited by memory movement"
    if "retrieval head" in text or "factual token" in text:
        return "Retrieval heads route facts back into generation"
    if "deliberation" in text or "rationale" in text or "second opinion" in text:
        return "Extra inference compute becomes an architecture layer"
    if "cybench" in text or "cybersecurity" in text:
        return "Evaluation spans multiple security risk surfaces"
    if "benchmark" in text or "evaluation" in text:
        return "Modern architectures need live task evaluation"
    if "summary" in text or "takeaway" in text:
        return "Synthesis, memory, and scaffolds converge"
    if figure_type == "line_chart":
        return "Memory grows with context length"
    if figure_type == "bar_chart":
        return "Measured comparisons replace vague claims"
    return "Architecture becomes an adaptive system"


def _surface_callout(context: dict[str, object]) -> str:
    text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    if "sharp" in text or "flat" in text or "minima" in text:
        return "Sharp and flat basins expose different generalization and robustness trade-offs."
    if "optimization" in text or "loss" in text:
        return "The landscape view makes optimization behavior visible instead of merely verbal."
    return _short_bullet(context, 0)


def _spatial_callout(context: dict[str, object]) -> str:
    text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    if "explicit" in text and ("3d" in text or "spatial structure" in text or "2d" in text):
        return "Explicit 3D structure turns ambiguous 2D cues into navigable spatial evidence."
    if "protein" in text or "ligand" in text:
        return "Spatial structure is the evidence object: geometry constrains the model's prediction."
    if "robot" in text or "motion" in text or "action" in text:
        return "Action quality depends on grounding predicted motion in the observed scene."
    if "voxel" in text or "point cloud" in text or "gaussian" in text:
        return "Different 3D representations trade off density, editability, and geometric fidelity."
    return _short_bullet(context, 0)


def _spatial_card_body(context: dict[str, object]) -> str:
    text = f"{context.get('title', '')} {context.get('need', '')} {context.get('content', '')}".lower()
    if "explicit" in text and ("3d" in text or "spatial structure" in text or "2d" in text):
        return "Make geometry explicit before the model reasons over the scene."
    if "protein" in text or "ligand" in text:
        return "Use geometry to constrain plausible molecular interactions."
    if "robot" in text or "motion" in text or "action" in text:
        return "Bind action plans to observed motion and scene state."
    if "voxel" in text or "point cloud" in text or "gaussian" in text:
        return "Choose the representation that preserves the needed geometry."
    return _short_bullet(context, 0)


def _draw_editorial_landscape(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    if _contains_any(str(context.get("need", "")).lower(), ("overview diagram", "summary graphic")):
        labels = ["Static", "Synthesis", "Memory", "Scaffolds"]
        bodies = [
            "manual families and fixed design assumptions",
            "automated search and tailored architectures",
            "KV-cache and recall constraints shape context",
            "deliberation and agents become architecture",
        ]
    else:
        labels = _semantic_labels(context, 4, ["Evidence", "Mechanism", "System", "Outcome"])
        relationships = [str(item) for item in context.get("relationships", []) if str(item).strip()]
        bodies = [
            _truncate(relationships[index], 78)
            if index < len(relationships)
            else _short_bullet(context, index)
            for index in range(4)
        ]
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"], EDITORIAL["purple"]]
    for index, label in enumerate(labels):
        x = 86 + index * 292
        _draw_metric_card(
            drawing,
            x,
            184,
            248,
            230,
            label.title(),
            bodies[index],
            colors[index],
        )
        if index < len(labels) - 1:
            _draw_editorial_arrow(drawing, x + 250, 298, x + 286, 298, EDITORIAL["muted"])
    spec = context.get("figure_spec") if isinstance(context.get("figure_spec"), dict) else {}
    goal = str(spec.get("semantic_goal") or "Source-grounded figure links the entities into a concrete mechanism.")
    _draw_center_callout(drawing, _truncate(goal, 112), y=470, color=EDITORIAL["ice"])


def _draw_editorial_matrix(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    columns = _context_labels(context, ["Small", "Mid", "Large", "Scaffold"], 4)
    rows = ["Capability", "Latency", "Memory", "Cost"]
    x0, y0 = 260, 178
    cell_w, cell_h = 190, 82
    drawing.add(drawing.text("Capability trade-off surface", insert=(82, 194), fill=EDITORIAL["ice"], font_size=22, font_family="Arial", font_weight="bold"))
    for col, label in enumerate(columns):
        _draw_wrapped_text(drawing, label.title(), x0 + col * cell_w + cell_w / 2, y0 - 24, 16, EDITORIAL["gray"], anchor="middle", width=16)
    for row, label in enumerate(rows):
        drawing.add(drawing.text(label, insert=(94, y0 + row * cell_h + 50), fill=EDITORIAL["gray"], font_size=17, font_family="Arial"))
        for col in range(4):
            value = (row * 5 + col * 7) % 10
            color = [EDITORIAL["green"], EDITORIAL["ice"], EDITORIAL["amber"], EDITORIAL["coral"]][(row + col) % 4]
            opacity = 0.16 + value / 45
            drawing.add(drawing.rect(insert=(x0 + col * cell_w, y0 + row * cell_h), size=(cell_w - 12, cell_h - 10), rx=10, fill=color, opacity=opacity))
            label_text = ["fit", "watch", "cost", "risk"][(row + col) % 4]
            drawing.add(drawing.text(label_text, insert=(x0 + col * cell_w + cell_w / 2 - 6, y0 + row * cell_h + 49), text_anchor="middle", fill=EDITORIAL["white"], font_size=15, font_family="Arial", font_weight="bold"))
    _draw_side_notes(drawing, context, x=1048, y=184)


def _draw_editorial_flowchart(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    steps = _context_labels(context, ["Constraints", "Search", "Validate", "Architecture"], 4)
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]]
    y = 292
    for index, label in enumerate(steps):
        x = 86 + index * 294
        drawing.add(drawing.rect(insert=(x, y - 80), size=(232, 154), rx=14, fill=colors[index], opacity=0.16, stroke=colors[index], stroke_width=2))
        drawing.add(drawing.circle(center=(x + 38, y - 42), r=17, fill=colors[index], opacity=0.85))
        drawing.add(drawing.text(str(index + 1), insert=(x + 32, y - 36), fill=EDITORIAL["navy"], font_size=18, font_family="Arial", font_weight="bold"))
        _draw_wrapped_text(drawing, label.title(), x + 116, y - 10, 20, EDITORIAL["white"], anchor="middle", width=17)
        _draw_wrapped_text(drawing, _short_bullet(context, index), x + 116, y + 40, 12, EDITORIAL["gray"], anchor="middle", width=25)
        if index < 3:
            _draw_editorial_arrow(drawing, x + 236, y, x + 282, y, EDITORIAL["muted"])
    _draw_center_callout(drawing, "Production architecture is now a pipeline: constraints in, measured structure out.", y=500, color=EDITORIAL["green"])


def _draw_editorial_search(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    drawing.add(drawing.text("Candidate pool", insert=(96, 194), fill=EDITORIAL["ice"], font_size=21, font_family="Arial", font_weight="bold"))
    values = _deterministic_values(str(context.get("need", "")), 18)
    for index, value in enumerate(values):
        col = index % 6
        row = index // 6
        x = 110 + col * 56
        y = 240 + row * 56
        color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"]][index % 3]
        drawing.add(drawing.circle(center=(x, y), r=13 + int(value * 10), fill=color, opacity=0.38 + value * 0.3))
    _draw_editorial_arrow(drawing, 475, 296, 555, 296, EDITORIAL["muted"])
    _draw_metric_card(drawing, 570, 214, 230, 164, "Proxy fitness", "cheap ranking before full training", EDITORIAL["amber"])
    _draw_editorial_arrow(drawing, 812, 296, 892, 296, EDITORIAL["muted"])
    _draw_metric_card(drawing, 910, 198, 260, 198, "Selected architecture", "hardware-aware structure with measured trade-offs", EDITORIAL["green"])
    _draw_center_callout(drawing, "Search quality depends on candidate coverage, fast proxy signals, and hardware constraints.", y=485, color=EDITORIAL["amber"])


def _draw_editorial_memory(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = ["KV cache", "Activations", "Weights", "Scheduler"]
    widths = [520, 190, 150, 90]
    colors = [EDITORIAL["coral"], EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["purple"]]
    x, y = 120, 260
    drawing.add(drawing.text("Inference memory pressure", insert=(120, 210), fill=EDITORIAL["ice"], font_size=22, font_family="Arial", font_weight="bold"))
    cursor = x
    for label, width, color in zip(labels, widths, colors):
        drawing.add(drawing.rect(insert=(cursor, y), size=(width, 82), rx=8, fill=color, opacity=0.72))
        _draw_wrapped_text(drawing, label, cursor + width / 2, y + 48, 18, EDITORIAL["navy"], anchor="middle", width=14)
        cursor += width + 10
    drawing.add(drawing.line(start=(x, y + 118), end=(x + 990, y + 118), stroke=EDITORIAL["muted"], stroke_width=2))
    for index, tick in enumerate(["4K", "16K", "64K", "128K"]):
        tx = x + index * 310
        drawing.add(drawing.line(start=(tx, y + 108), end=(tx, y + 128), stroke=EDITORIAL["muted"], stroke_width=2))
        drawing.add(drawing.text(tick, insert=(tx - 14, y + 156), fill=EDITORIAL["gray"], font_size=15, font_family="Arial"))
    _draw_metric_card(drawing, 140, 430, 250, 120, "Bottleneck", "memory bandwidth, not just attention math", EDITORIAL["coral"])
    _draw_metric_card(drawing, 515, 430, 250, 120, "Control knobs", "eviction, compression, quantization", EDITORIAL["green"])
    _draw_metric_card(drawing, 890, 430, 250, 120, "Evaluation", "latency + recall + consistency", EDITORIAL["ice"])


def _draw_editorial_attention_map(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    x0, y0, cell = 292, 172, 48
    drawing.add(drawing.text("Retrieval-head activation", insert=(84, 200), fill=EDITORIAL["ice"], font_size=22, font_family="Arial", font_weight="bold"))
    values = _deterministic_values(str(context.get("title", "")), 84)
    for row in range(7):
        for col in range(12):
            idx = row * 12 + col
            hot = row == 3 and col in {7, 8}
            color = EDITORIAL["coral"] if hot else EDITORIAL["ice"]
            opacity = 0.18 + values[idx] * (0.34 if not hot else 0.72)
            drawing.add(drawing.rect(insert=(x0 + col * cell, y0 + row * cell), size=(cell - 5, cell - 5), rx=7, fill=color, opacity=opacity))
    drawing.add(drawing.rect(insert=(x0 + 7 * cell - 5, y0 + 3 * cell - 5), size=(cell * 2, cell), rx=8, fill="none", stroke=EDITORIAL["coral"], stroke_width=4))
    _draw_editorial_arrow(drawing, 175, 470, x0 + 7 * cell, y0 + 3 * cell + 20, EDITORIAL["coral"])
    drawing.add(drawing.text("factual token", insert=(96, 488), fill=EDITORIAL["coral"], font_size=18, font_family="Arial", font_weight="bold"))
    _draw_side_notes(drawing, context, x=940, y=188)


def _draw_editorial_deliberation(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _context_labels(context, ["Generate", "Critique", "Select", "Act"], 4)
    centers = [(250, 280), (520, 208), (790, 280), (520, 430)]
    colors = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"], EDITORIAL["purple"]]
    for center, label, color in zip(centers, labels, colors):
        drawing.add(drawing.circle(center=center, r=82, fill=color, opacity=0.22, stroke=color, stroke_width=3))
        _draw_wrapped_text(drawing, label.title(), center[0], center[1] + 4, 20, EDITORIAL["white"], anchor="middle", width=14)
    for start, end in ((centers[0], centers[1]), (centers[1], centers[2]), (centers[2], centers[3]), (centers[3], centers[0])):
        _draw_editorial_arrow(drawing, start[0], start[1], end[0], end[1], EDITORIAL["muted"])
    _draw_metric_card(drawing, 940, 210, 220, 190, "Scaffold value", "extra inference compute becomes an architectural component", EDITORIAL["green"])


def _draw_editorial_radar(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    if _contains_any(str(context.get("need", "")).lower(), ("cybersecurity", "exploitation", "forensics")):
        axes = ["Exploitation", "Defense", "Forensics", "Planning", "Risk"]
    else:
        axes = _context_labels(context, ["Exploit", "Defend", "Forensics", "Planning", "Safety"], 5)
    cx, cy, radius = 560, 378, 145
    for level in range(1, 5):
        pts = _radar_points(cx, cy, radius * level / 4, len(axes), [1.0] * len(axes))
        drawing.add(drawing.polygon(points=pts, fill="none", stroke=EDITORIAL["muted"], stroke_width=1, opacity=0.35))
    values = [0.72, 0.58, 0.66, 0.48, 0.61]
    pts = _radar_points(cx, cy, radius, len(axes), values)
    drawing.add(drawing.polygon(points=pts, fill=EDITORIAL["coral"], opacity=0.28, stroke=EDITORIAL["coral"], stroke_width=4))
    for index, axis in enumerate(axes):
        angle = -math.pi / 2 + index * 2 * math.pi / len(axes)
        x = cx + math.cos(angle) * (radius + 62)
        y = cy + math.sin(angle) * (radius + 62)
        _draw_wrapped_text(drawing, axis.title(), x, y, 15, EDITORIAL["gray"], anchor="middle", width=13)
    _draw_side_notes(drawing, context, x=890, y=184)


def _draw_editorial_timeline(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    stages = _context_labels(context, ["Training cutoff", "New papers", "Live ingestion", "Updated eval"], 4)
    x0, y = 150, 340
    drawing.add(drawing.line(start=(x0, y), end=(1130, y), stroke=EDITORIAL["muted"], stroke_width=6, opacity=0.65))
    for index, label in enumerate(stages):
        x = x0 + index * 326
        color = [EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["ice"], EDITORIAL["green"]][index]
        drawing.add(drawing.circle(center=(x, y), r=34, fill=color, opacity=0.92))
        drawing.add(drawing.text(str(index + 1), insert=(x - 8, y + 10), fill=EDITORIAL["navy"], font_size=24, font_family="Arial", font_weight="bold"))
        _draw_wrapped_text(drawing, label.title(), x, y + 84, 17, EDITORIAL["white"], anchor="middle", width=16)
    _draw_center_callout(drawing, "Static evaluation ages; live benchmarks keep the target moving.", y=505, color=EDITORIAL["ice"])


def _draw_editorial_heatmap(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    rows = _context_labels(context, ["Data", "Model", "Compute", "Risk"], 4)
    cols = _context_labels(context, ["Low", "Medium", "High", "Stress"], 4)
    values = _deterministic_values(str(context.get("need", "")), 16)
    x0, y0, cell = 356, 202, 86
    for row in range(4):
        drawing.add(drawing.text(rows[row].title(), insert=(182, y0 + row * cell + 52), fill=EDITORIAL["gray"], font_size=17, font_family="Arial"))
        for col in range(4):
            value = values[row * 4 + col]
            color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["amber"], EDITORIAL["coral"]][col]
            drawing.add(drawing.rect(insert=(x0 + col * cell, y0 + row * cell), size=(cell - 7, cell - 7), rx=9, fill=color, opacity=0.18 + value * 0.58))
    for col, label in enumerate(cols):
        _draw_wrapped_text(drawing, label.title(), x0 + col * cell + 38, y0 - 30, 15, EDITORIAL["gray"], anchor="middle", width=12)
    _draw_side_notes(drawing, context, x=840, y=204)


def _draw_editorial_bar_chart(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _context_labels(context, ["Baseline", "Search-tuned", "Tailored"], 3)
    values = [0.48, 0.68, 0.86]
    colors = [EDITORIAL["coral"], EDITORIAL["ice"], EDITORIAL["green"]]
    x0, base_y = 250, 510
    drawing.add(drawing.text("Relative performance index", insert=(250, 202), fill=EDITORIAL["ice"], font_size=21, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.line(start=(x0 - 40, base_y), end=(970, base_y), stroke=EDITORIAL["muted"], stroke_width=2))
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        height = int(255 * value)
        x = x0 + index * 230
        drawing.add(drawing.rect(insert=(x, base_y - height), size=(140, height), rx=10, fill=color, opacity=0.82))
        drawing.add(drawing.text(label.title(), insert=(x + 70, base_y + 38), text_anchor="middle", fill=EDITORIAL["gray"], font_size=16, font_family="Arial"))
        drawing.add(drawing.text(["base", "better", "best"][index], insert=(x + 70, base_y - height - 16), text_anchor="middle", fill=color, font_size=17, font_family="Arial", font_weight="bold"))
    _draw_side_notes(drawing, context, x=978, y=215)


def _draw_editorial_line_chart(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    x0, y0, width, height = 205, 205, 780, 300
    drawing.add(drawing.text("Scaling pressure curve", insert=(205, 188), fill=EDITORIAL["ice"], font_size=21, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.line(start=(x0, y0 + height), end=(x0 + width, y0 + height), stroke=EDITORIAL["muted"], stroke_width=2))
    drawing.add(drawing.line(start=(x0, y0), end=(x0, y0 + height), stroke=EDITORIAL["muted"], stroke_width=2))
    labels = ["4K", "16K", "64K", "128K"]
    points = [(x0, y0 + height - 42), (x0 + 245, y0 + height - 82), (x0 + 505, y0 + height - 174), (x0 + width, y0 + 48)]
    drawing.add(drawing.polyline(points=points, fill="none", stroke=EDITORIAL["coral"], stroke_width=7))
    for (x, y), label in zip(points, labels):
        drawing.add(drawing.circle(center=(x, y), r=11, fill=EDITORIAL["navy"], stroke=EDITORIAL["coral"], stroke_width=5))
        drawing.add(drawing.text(label, insert=(x - 18, y0 + height + 38), fill=EDITORIAL["gray"], font_size=15, font_family="Arial"))
    _draw_metric_card(drawing, 1010, 235, 188, 165, "Design target", "flatten the curve with memory-aware architecture", EDITORIAL["green"])


def _draw_editorial_equation(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    drawing.add(drawing.rect(insert=(170, 238), size=(940, 182), rx=18, fill=EDITORIAL["panel_2"], stroke=EDITORIAL["purple"], stroke_width=3))
    drawing.add(drawing.text("quality = capability / (latency + memory + risk)", insert=(250, 330), fill=EDITORIAL["white"], font_size=34, font_family="Arial", font_weight="bold"))
    _draw_center_callout(drawing, _short_bullet(context, 0), y=480, color=EDITORIAL["purple"])


def _draw_editorial_surface(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    semantic_labels = _semantic_labels(
        context,
        5,
        [
            "Sharp Minimum SGD",
            "Flat Minimum SAM",
            "Post-RL Fine-tuning Shift",
            "Catastrophic Forgetting Zone",
            "Hard Example Boundary",
        ],
    )
    sharp_label = _first_label_containing(semantic_labels, ("sharp", "sgd"), "Sharp Minimum SGD")
    flat_label = _first_label_containing(semantic_labels, ("flat", "sam"), "Flat Minimum SAM")
    shift_label = _first_label_containing(
        semantic_labels,
        ("post-rl", "post rl", "fine", "shift"),
        "Post-RL Fine-tuning Shift",
    )
    forgetting_label = _first_label_containing(
        semantic_labels,
        ("forget", "catastrophic"),
        "Catastrophic Forgetting Zone",
    )
    boundary_label = _first_label_containing(
        semantic_labels,
        ("hard", "boundary"),
        "Hard Example Boundary",
    )
    labels = [sharp_label, flat_label, boundary_label]
    values = _deterministic_values(str(context.get("need", "")), 24)
    x0, y0 = 165, 476
    drawing.add(drawing.text("3D loss surface", insert=(98, 196), fill=EDITORIAL["ice"], font_size=22, font_family="Arial", font_weight="bold"))
    drawing.add(drawing.line(start=(x0, y0), end=(830, y0), stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.7))
    drawing.add(drawing.line(start=(x0, y0), end=(x0, 182), stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.7))
    drawing.add(drawing.text("x-axis: parameter direction", insert=(555, 514), fill=EDITORIAL["gray"], font_size=16, font_family="Arial"))
    drawing.add(drawing.text("Loss", insert=(104, 222), fill=EDITORIAL["gray"], font_size=18, font_family="Arial", font_weight="bold"))
    colors = [EDITORIAL["coral"], EDITORIAL["amber"], EDITORIAL["ice"], EDITORIAL["green"]]
    for band in range(4):
        points: list[tuple[float, float]] = []
        for index in range(9):
            x = x0 + 35 + index * 72
            wave = math.sin(index * 0.88 + band * 0.72) * (22 + band * 3)
            slope = abs(index - 4) * (13 - band)
            jitter = values[(band * 6 + index) % len(values)] * 18
            y = y0 - 60 - band * 54 - wave - slope - jitter
            points.append((x, y))
        drawing.add(
            drawing.polyline(
                points=points,
                fill="none",
                stroke=colors[band],
                stroke_width=5,
                opacity=0.62 + band * 0.08,
            )
        )
    minima = [(386, 366, EDITORIAL["coral"], labels[0]), (650, 388, EDITORIAL["green"], labels[1])]
    for x, y, color, label in minima:
        drawing.add(drawing.circle(center=(x, y), r=18, fill=color, opacity=0.9))
        drawing.add(drawing.circle(center=(x, y), r=44, fill="none", stroke=color, stroke_width=3, opacity=0.55))
        drawing.add(drawing.rect(insert=(x - 96, y - 92), size=(192, 42), rx=10, fill=EDITORIAL["panel_2"], opacity=0.94, stroke=color, stroke_width=2))
        _draw_wrapped_text(drawing, label.title(), x, y - 66, 16, EDITORIAL["white"], anchor="middle", width=20)
    _draw_editorial_arrow(drawing, 406, 346, 512, 250, EDITORIAL["coral"])
    drawing.add(drawing.rect(insert=(450, 198), size=(236, 50), rx=10, fill=EDITORIAL["panel_2"], opacity=0.94, stroke=EDITORIAL["coral"], stroke_width=2))
    _draw_wrapped_text(drawing, forgetting_label, 568, 228, 16, EDITORIAL["coral"], anchor="middle", width=26)
    _draw_editorial_arrow(drawing, 650, 366, 734, 362, EDITORIAL["green"])
    drawing.add(drawing.rect(insert=(735, 372), size=(220, 52), rx=10, fill=EDITORIAL["panel_2"], opacity=0.94, stroke=EDITORIAL["green"], stroke_width=2))
    _draw_wrapped_text(drawing, shift_label, 845, 402, 16, EDITORIAL["green"], anchor="middle", width=25)
    drawing.add(
        drawing.ellipse(
            center=(650, 388),
            r=(138, 58),
            fill="none",
            stroke=EDITORIAL["purple"],
            stroke_width=3,
            stroke_dasharray="10 8",
            opacity=0.88,
        )
    )
    _draw_metric_card(drawing, 900, 220, 230, 170, "Boundary", f"{_compact_center_label(boundary_label)} defines curriculum edge", EDITORIAL["purple"])
    _draw_center_callout(drawing, _surface_callout(context), y=520, color=EDITORIAL["green"])


def _draw_editorial_spatial(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    labels = _context_labels(context, ["Tokens", "Geometry", "Actions", "Evidence"], 4)
    values = _deterministic_values(str(context.get("need", "")), 32)
    drawing.add(drawing.text("Spatial evidence model", insert=(96, 196), fill=EDITORIAL["ice"], font_size=22, font_family="Arial", font_weight="bold"))
    origin = (230, 458)
    drawing.add(drawing.line(start=origin, end=(520, 458), stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.75))
    drawing.add(drawing.line(start=origin, end=(230, 215), stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.75))
    drawing.add(drawing.line(start=origin, end=(360, 330), stroke=EDITORIAL["muted"], stroke_width=3, opacity=0.75))
    for offset, color in ((0, EDITORIAL["ice"]), (38, EDITORIAL["green"]), (76, EDITORIAL["purple"])):
        polygon = [
            (280 + offset, 402 - offset * 0.55),
            (520 + offset, 372 - offset * 0.55),
            (625 + offset, 438 - offset * 0.55),
            (388 + offset, 472 - offset * 0.55),
        ]
        drawing.add(drawing.polygon(points=polygon, fill=color, opacity=0.14, stroke=color, stroke_width=3))
    for index, value in enumerate(values[:24]):
        x = 306 + (index % 8) * 43 + value * 18
        y = 250 + (index // 8) * 70 + values[-index - 1] * 28
        color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"], EDITORIAL["purple"]][index % 4]
        drawing.add(drawing.circle(center=(x, y), r=5 + int(value * 8), fill=color, opacity=0.42 + value * 0.36))
    for index, label in enumerate(labels[:3]):
        x = 705
        y = 222 + index * 94
        color = [EDITORIAL["ice"], EDITORIAL["green"], EDITORIAL["coral"]][index]
        drawing.add(drawing.rect(insert=(x, y), size=(160, 56), rx=12, fill=color, opacity=0.2, stroke=color, stroke_width=2))
        _draw_wrapped_text(drawing, label.title(), x + 80, y + 34, 16, EDITORIAL["white"], anchor="middle", width=14)
        _draw_editorial_arrow(drawing, x - 72, y + 28, x - 8, y + 28, EDITORIAL["muted"])
    _draw_metric_card(drawing, 930, 248, 220, 150, labels[3].title(), _spatial_card_body(context), EDITORIAL["amber"])
    _draw_center_callout(drawing, _spatial_callout(context), y=520, color=EDITORIAL["ice"])


def _draw_metric_card(
    drawing: svgwrite.Drawing,
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    body: str,
    color: str,
) -> None:
    drawing.add(drawing.rect(insert=(x, y), size=(w, h), rx=16, fill=color, opacity=0.13, stroke=color, stroke_width=2))
    drawing.add(drawing.rect(insert=(x, y), size=(w, 44), rx=16, fill=color, opacity=0.28))
    drawing.add(drawing.rect(insert=(x, y + 25), size=(w, 20), fill=color, opacity=0.28))
    _draw_wrapped_text(drawing, title, x + w / 2, y + 29, 17, EDITORIAL["white"], anchor="middle", width=max(12, w // 13))
    _draw_wrapped_text(drawing, body, x + w / 2, y + 92, 14, EDITORIAL["gray"], anchor="middle", width=max(16, w // 11))


def _draw_side_notes(drawing: svgwrite.Drawing, context: dict[str, object], *, x: int, y: int) -> None:
    drawing.add(drawing.rect(insert=(x, y), size=(186, 280), rx=14, fill=EDITORIAL["panel_2"], opacity=0.9))
    drawing.add(drawing.text("Evidence", insert=(x + 18, y + 36), fill=EDITORIAL["ice"], font_size=20, font_family="Arial", font_weight="bold"))
    cues = _evidence_cues(context, count=3)
    for index, cue in enumerate(cues):
        y_pos = y + 82 + index * 72
        drawing.add(drawing.circle(center=(x + 22, y_pos), r=4, fill=[EDITORIAL["green"], EDITORIAL["ice"], EDITORIAL["coral"]][index % 3]))
        _draw_wrapped_text(drawing, cue, x + 38, y_pos + 6, 17, EDITORIAL["gray"], width=14)


def _draw_center_callout(drawing: svgwrite.Drawing, text: str, *, y: int, color: str) -> None:
    drawing.add(drawing.rect(insert=(110, y), size=(1060, 110), rx=14, fill=color, opacity=0.12, stroke=color, stroke_width=2))
    _draw_wrapped_text(drawing, text, 640, y + 34, 16, EDITORIAL["white"], anchor="middle", width=88)


def _draw_evidence_strip(drawing: svgwrite.Drawing, context: dict[str, object]) -> None:
    return


def _draw_editorial_arrow(
    drawing: svgwrite.Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
) -> None:
    drawing.add(drawing.line(start=(x1, y1), end=(x2, y2), stroke=color, stroke_width=3, opacity=0.85))
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 15
    left = (x2 - head_len * math.cos(angle - math.pi / 6), y2 - head_len * math.sin(angle - math.pi / 6))
    right = (x2 - head_len * math.cos(angle + math.pi / 6), y2 - head_len * math.sin(angle + math.pi / 6))
    drawing.add(drawing.polygon(points=[(x2, y2), left, right], fill=color, opacity=0.85))


def _draw_wrapped_text(
    drawing: svgwrite.Drawing,
    text: str,
    x: float,
    y: float,
    size: int,
    color: str,
    *,
    anchor: str = "start",
    width: int = 24,
) -> None:
    for offset, line in enumerate(_wrap_text(text, width)[:4]):
        drawing.add(
            drawing.text(
                line,
                insert=(x, y + offset * (size + 5)),
                text_anchor=anchor,
                fill=color,
                font_size=size,
                font_family="Arial",
                font_weight="bold" if size >= 17 else "normal",
            )
        )


def _context_labels(context: dict[str, object], fallback: list[str], count: int) -> list[str]:
    keywords = [str(item) for item in context.get("keywords", [])]
    labels: list[str] = []
    for item in keywords:
        cleaned = item.replace("_", " ").strip()
        if len(cleaned) < 4:
            continue
        if cleaned.lower() in {label.lower() for label in labels}:
            continue
        labels.append(cleaned)
        if len(labels) == count:
            break
    while len(labels) < count:
        labels.append(fallback[len(labels) % len(fallback)])
    return labels[:count]


def _first_label_containing(labels: list[str], needles: tuple[str, ...], fallback: str) -> str:
    for label in labels:
        lower = label.lower()
        if any(needle in lower for needle in needles):
            return label
    return fallback


def _short_bullet(context: dict[str, object], index: int) -> str:
    bullets = list(context.get("bullets", []))
    if bullets:
        return _truncate(str(bullets[index % len(bullets)]), 72)
    return _truncate(str(context.get("need", "")), 72)


def _truncate(text: str, length: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= length:
        return _remove_dangling_tail(_remove_dangling_parenthetical(cleaned))
    clipped = cleaned[: max(0, length)].rsplit(" ", 1)[0].strip()
    clipped = _remove_dangling_tail(_remove_dangling_parenthetical(clipped))
    return clipped or _remove_dangling_tail(_remove_dangling_parenthetical(cleaned[:length].strip()))


def _compact_center_label(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= 28:
        return cleaned
    without_parenthetical = _remove_dangling_parenthetical(re.sub(r"\s*\([^)]*\)", "", cleaned).strip())
    if without_parenthetical:
        return without_parenthetical
    return _truncate(cleaned, 28)


def _remove_dangling_parenthetical(text: str) -> str:
    if "(" in text and ")" not in text[text.rfind("(") :]:
        return text[: text.rfind("(")].strip()
    return text


def _remove_dangling_tail(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    cleaned = re.sub(r"\s+[+/%-]\s*$", "", cleaned).strip()
    dangling = {"and", "or", "of", "in", "to", "from", "with", "for", "by", "via", "as"}
    while cleaned.split() and cleaned.split()[-1].lower().strip(".,;:") in dangling:
        cleaned = " ".join(cleaned.split()[:-1]).strip()
    return cleaned.rstrip(" ,;:+-/")


def _safe_label(text: str, *, words: int = 3) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+/% -]+", " ", text)
    tokens = [token for token in cleaned.split() if len(token) > 1]
    return " ".join(tokens[:words]).title() or "Signal"


def _compact_axis_label(text: str, *, words: int = 3) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", str(text))
    cleaned = _remove_dangling_parenthetical(cleaned)
    cleaned = re.sub(r"[^A-Za-z0-9+/% -]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token and not re.fullmatch(r"[+/%-]+", token)]
    label = " ".join(tokens[:words]).strip()
    return _truncate(label, 30) or "Signal"


def _compact_metric_label(text: str) -> str:
    cleaned = " ".join(str(text).split())
    token_match = re.search(r"\b(\d{4,})\s*tokens?\b", cleaned, flags=re.IGNORECASE)
    if token_match:
        value = int(token_match.group(1))
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M tokens"
        return f"{value / 1_000:g}k tokens"
    return _truncate(cleaned, 12)


def _replace_generic_cell_label(label: str, fallback: str) -> str:
    cleaned = " ".join(str(label).split()).strip()
    if re.fullmatch(r"Cell Type [A-D]", cleaned, flags=re.IGNORECASE):
        return fallback
    if cleaned.lower() in {"cell type", "cell identity"}:
        return fallback
    return _truncate(cleaned, 24) or fallback


def _entity_legend_label(text: str) -> str:
    cleaned = str(text).replace("²", "")
    cleaned = re.sub(r"[^A-Za-z0-9+/%() -]+", " ", cleaned)
    tokens = [token for token in cleaned.split() if token]
    label = " ".join(tokens[:7]).strip()
    return _truncate(label, 52) or "Entity"


def _safe_sentence(text: str, *, max_words: int = 10) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+/% -]+", " ", text)
    tokens = cleaned.split()
    return " ".join(tokens[:max_words]).strip().capitalize() or "Evidence-linked architecture signal"


def _evidence_cues(context: dict[str, object], *, count: int) -> list[str]:
    refs = [str(item) for item in context.get("source_refs", []) if str(item).strip()]
    keywords = [str(item) for item in context.get("keywords", []) if str(item).strip()]
    bullets = [str(item) for item in context.get("bullets", []) if str(item).strip()]
    candidates = refs + keywords + bullets + [str(context.get("need", ""))]
    cues: list[str] = []
    for candidate in candidates:
        label = _safe_label(candidate, words=2)
        if label.lower() in {"ref paper", "paper abstract", "source refs", "slide specific"}:
            continue
        if label and label.lower() not in {item.lower() for item in cues}:
            cues.append(label)
        if len(cues) >= count:
            break
    while len(cues) < count:
        cues.append(["Claim", "Evidence", "Implication"][len(cues)])
    return cues[:count]


def _radar_points(
    cx: int,
    cy: int,
    radius: float,
    count: int,
    values: list[float],
) -> list[tuple[float, float]]:
    points = []
    for index in range(count):
        angle = -math.pi / 2 + index * 2 * math.pi / count
        r = radius * values[index]
        points.append((cx + math.cos(angle) * r, cy + math.sin(angle) * r))
    return points


def _draw_flowchart(drawing: svgwrite.Drawing, need: str) -> None:
    labels = _keywords(need, fallback=["Source", "Transform", "Validate", "Publish"], count=4)
    x_positions = [120, 390, 660, 930]
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        drawing.add(drawing.rect(insert=(x, 310), size=(190, 96), rx=14, fill=PALETTE["white"], stroke=PALETTE["blue"], stroke_width=4))
        _add_centered_text(drawing, label.title(), x + 95, 365, 20)
        if index < len(x_positions) - 1:
            _draw_arrow(drawing, x + 205, 358, x_positions[index + 1] - 20, 358)


def _draw_timeline(drawing: svgwrite.Drawing, need: str) -> None:
    labels = _keywords(need, fallback=["Baseline", "Iteration", "Validation", "Release"], count=4)
    y = 390
    drawing.add(drawing.line(start=(150, y), end=(1120, y), stroke=PALETTE["line"], stroke_width=8))
    for index, label in enumerate(labels):
        x = 160 + index * 310
        drawing.add(drawing.circle(center=(x, y), r=34, fill=PALETTE["blue" if index % 2 == 0 else "green"]))
        drawing.add(drawing.text(str(index + 1), insert=(x - 8, y + 10), fill=PALETTE["white"], font_size=28, font_family="Inter, Arial, sans-serif", font_weight="bold"))
        _add_centered_text(drawing, label.title(), x, y + 86, 20)


def _draw_heatmap(drawing: svgwrite.Drawing, need: str) -> None:
    labels = _keywords(need, fallback=["Data", "Model", "Compute", "Eval"], count=4)
    start_x, start_y, cell = 360, 260, 82
    values = _deterministic_values(need, 16)
    for row in range(4):
        for col in range(4):
            value = values[row * 4 + col]
            color = _interpolate_color("#e7f0ff", "#2f80ed", value)
            drawing.add(drawing.rect(insert=(start_x + col * cell, start_y + row * cell), size=(cell - 6, cell - 6), rx=10, fill=color))
    for index, label in enumerate(labels):
        drawing.add(drawing.text(label.title(), insert=(170, start_y + index * cell + 48), fill=PALETTE["ink"], font_size=20, font_family="Inter, Arial, sans-serif"))
        drawing.add(drawing.text(label.title(), insert=(start_x + index * cell, start_y - 24), fill=PALETTE["ink"], font_size=20, font_family="Inter, Arial, sans-serif"))


def _draw_bar_chart(drawing: svgwrite.Drawing, need: str) -> None:
    labels = _keywords(need, fallback=["FP32", "FP16", "FP8", "INT4"], count=4)
    values = _deterministic_values(need, 4)
    base_y = 555
    drawing.add(drawing.line(start=(170, base_y), end=(1080, base_y), stroke=PALETTE["ink"], stroke_width=3))
    for index, (label, value) in enumerate(zip(labels, values)):
        height = 120 + int(value * 210)
        x = 250 + index * 185
        drawing.add(drawing.rect(insert=(x, base_y - height), size=(105, height), rx=12, fill=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["yellow"]][index % 4]))
        _add_centered_text(drawing, label.upper(), x + 52, base_y + 36, 18)


def _draw_line_chart(drawing: svgwrite.Drawing, need: str) -> None:
    values = _deterministic_values(need, 8)
    x0, y0, width, height = 180, 250, 900, 300
    drawing.add(drawing.line(start=(x0, y0 + height), end=(x0 + width, y0 + height), stroke=PALETTE["ink"], stroke_width=3))
    drawing.add(drawing.line(start=(x0, y0), end=(x0, y0 + height), stroke=PALETTE["ink"], stroke_width=3))
    points = []
    for index, value in enumerate(values):
        x = x0 + index * (width / (len(values) - 1))
        y = y0 + height - (70 + value * 210)
        points.append((x, y))
    drawing.add(drawing.polyline(points=points, fill="none", stroke=PALETTE["blue"], stroke_width=8))
    for x, y in points:
        drawing.add(drawing.circle(center=(x, y), r=10, fill=PALETTE["white"], stroke=PALETTE["blue"], stroke_width=5))
    drawing.add(drawing.path(d=f"M {x0} {y0 + height - 60} C {x0 + 240} {y0 + height - 180}, {x0 + 560} {y0 + height - 210}, {x0 + width} {y0 + 85}", fill="none", stroke=PALETTE["green"], stroke_width=5, stroke_dasharray="14 12"))
    drawing.add(drawing.text("baseline", insert=(x0 + width - 130, y0 + height - 96), fill=PALETTE["blue"], font_size=20, font_family="Inter, Arial, sans-serif"))
    drawing.add(drawing.text("programmatic trend", insert=(x0 + width - 230, y0 + 80), fill=PALETTE["green"], font_size=20, font_family="Inter, Arial, sans-serif"))


def _draw_equation(drawing: svgwrite.Drawing, need: str) -> None:
    drawing.add(drawing.rect(insert=(210, 290), size=(860, 170), rx=18, fill=PALETTE["white"], stroke=PALETTE["purple"], stroke_width=4))
    drawing.add(drawing.text("objective = signal / (compute + noise)", insert=(300, 370), fill=PALETTE["ink"], font_size=34, font_family="Inter, Arial, sans-serif", font_weight="bold"))
    drawing.add(drawing.text("derived from the slide figure need", insert=(394, 418), fill=PALETTE["muted"], font_size=22, font_family="Inter, Arial, sans-serif"))


def _draw_surface(drawing: svgwrite.Drawing, need: str) -> None:
    values = _deterministic_values(need, 9)
    for row in range(3):
        for col in range(3):
            cx = 400 + col * 150 + row * 54
            cy = 260 + row * 92
            radius = 42 + int(values[row * 3 + col] * 35)
            drawing.add(drawing.ellipse(center=(cx, cy), r=(radius + 35, radius), fill=_interpolate_color("#dff6f0", "#2ca58d", values[row * 3 + col]), stroke=PALETTE["white"], stroke_width=4))
    drawing.add(drawing.path(d="M 315 545 C 470 420, 690 580, 970 390", fill="none", stroke=PALETTE["ink"], stroke_width=5))


def _draw_diagram(drawing: svgwrite.Drawing, need: str) -> None:
    labels = _keywords(need, fallback=["Inputs", "Representation", "Model", "Evaluation"], count=4)
    centers = [(320, 330), (600, 250), (870, 350), (600, 500)]
    for index, center in enumerate(centers):
        drawing.add(drawing.circle(center=center, r=72, fill=[PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["yellow"]][index % 4], opacity=0.92))
        _add_centered_text(drawing, labels[index].title(), center[0], center[1] + 6, 19, fill=PALETTE["white"])
    for start, end in ((centers[0], centers[1]), (centers[1], centers[2]), (centers[2], centers[3]), (centers[3], centers[0])):
        _draw_arrow(drawing, start[0], start[1], end[0], end[1], color=PALETTE["ink"], width=3)


def _draw_arrow(
    drawing: svgwrite.Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = PALETTE["blue"],
    width: int = 4,
) -> None:
    drawing.add(drawing.line(start=(x1, y1), end=(x2, y2), stroke=color, stroke_width=width))
    angle = math.atan2(y2 - y1, x2 - x1)
    head_len = 16
    left = (
        x2 - head_len * math.cos(angle - math.pi / 6),
        y2 - head_len * math.sin(angle - math.pi / 6),
    )
    right = (
        x2 - head_len * math.cos(angle + math.pi / 6),
        y2 - head_len * math.sin(angle + math.pi / 6),
    )
    drawing.add(drawing.polygon(points=[(x2, y2), left, right], fill=color))


def _add_centered_text(
    drawing: svgwrite.Drawing,
    text: str,
    x: float,
    y: float,
    size: int,
    *,
    fill: str = PALETTE["ink"],
) -> None:
    lines = _wrap_text(text, 18)
    for offset, line in enumerate(lines[:2]):
        drawing.add(
            drawing.text(
                line,
                insert=(x, y + offset * (size + 4)),
                fill=fill,
                font_size=size,
                font_family="Inter, Arial, sans-serif",
                font_weight="bold",
                text_anchor="middle",
            )
        )


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if sum(len(item) for item in current) + len(current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _keywords(need: str, *, fallback: list[str], count: int) -> list[str]:
    ignored = {
        "showing",
        "diagram",
        "chart",
        "graph",
        "visual",
        "overview",
        "comparison",
        "illustrating",
        "across",
        "architecture",
        "architectures",
        "dynamic",
        "fixed",
        "from",
        "modern",
        "paradigm",
        "static",
        "traditional",
        "model",
        "models",
        "framework",
        "architecture",
        "architectures",
        "benchmark",
        "benchmarks",
        "claim",
        "cited",
        "editorial",
        "evidence",
        "implication",
        "implications",
        "map",
        "paper",
        "papers",
        "slide",
        "slides",
        "specific",
        "tying",
    }
    words = [
        word
        for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", need)
        if word.lower() not in ignored
    ]
    unique: list[str] = []
    for word in words:
        if word.lower() not in {item.lower() for item in unique}:
            unique.append(word)
    labels = unique[:count]
    while len(labels) < count:
        labels.append(fallback[len(labels) % len(fallback)])
    return labels


def _deterministic_values(seed: str, count: int) -> list[float]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return [0.15 + (digest[index % len(digest)] / 255.0) * 0.75 for index in range(count)]


def _interpolate_color(start: str, end: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    s = tuple(int(start[index : index + 2], 16) for index in (1, 3, 5))
    e = tuple(int(end[index : index + 2], 16) for index in (1, 3, 5))
    values = tuple(round(sv + (ev - sv) * amount) for sv, ev in zip(s, e))
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _inject_generated_figures(
    slides_md: str,
    *,
    output_path: Path,
    generated_by_slide: dict[int, dict[str, object]],
    workspace: Path,
) -> str:
    blocks = _slide_blocks(slides_md)
    updated: list[str] = []
    for block in blocks:
        title_match = SLIDE_TITLE_RE.search(block)
        if not title_match:
            updated.append(block)
            continue
        slide_number = int(title_match.group(1))
        figure = generated_by_slide.get(slide_number)
        if not figure:
            updated.append(block)
            continue
        svg_path = workspace / str(figure["path"])
        image_ref = Path(_relative(svg_path, output_path.parent)).as_posix()
        image_markdown = f"![Generated figure for slide {slide_number}]({image_ref})"
        block = re.sub(
            r"\n+!\[[^\]]*Generated figure[^\]]*\]\(assets/generated/slide_\d{3}_[^)]+\.svg\)",
            "",
            block,
        )
        comment_match = next(
            (match for match in COMMENT_RE.finditer(block) if "FIGURE_NEED:" in match.group(1)),
            None,
        )
        if comment_match:
            block = block[: comment_match.end()] + f"\n\n{image_markdown}" + block[comment_match.end() :]
        else:
            block = block.rstrip() + f"\n\n{image_markdown}"
        updated.append(block)
    return "\n\n---\n\n".join(updated).rstrip() + "\n"
