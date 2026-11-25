#!/usr/bin/env python3
"""Generate programmatic SVG figures for presentation FIGURE_NEED markers."""

from __future__ import annotations

import argparse
import ast
import contextvars
import hashlib
import html
import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from textwrap import shorten
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.svg_geometry import geometry_findings
from conference_pipeline.presentation.programmatic_figures import generate_programmatic_figures
from conference_pipeline.presentation.structured_video import _visual_legibility_findings
from conference_pipeline.presentation.visual_plan import validate_visual_plan_payload
from conference_pipeline.provider_receipts import anthropic_messages_create, provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline.model_governance import model_for_alias

PHASE16_REQUIRED_PROVIDER = "anthropic"
PHASE16_REQUIRED_MODEL = model_for_alias("opus")  # the Phase 16 gate pins the Opus TIER; callers read the same alias
PHASE16_REQUIRED_MODEL_ROUTE = f"{PHASE16_REQUIRED_PROVIDER}:{PHASE16_REQUIRED_MODEL}"
PHASE16_ADVANCED_RENDERERS = frozenset({"drawsvg", "graphviz", "altair-vl-convert", "pycairo"})
PHASE16_DISALLOWED_DEFAULT_RENDERERS = frozenset({"matplotlib", "plotly"})
PHASE16_MAX_VISIBLE_TEXT_LABELS = 26
PHASE16_MAX_ENTITY_NODES = 6
PHASE16_MAX_RELATIONSHIP_EDGES = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--input-slides", type=Path)
    parser.add_argument("--deck-contract", type=Path)
    parser.add_argument("--source-binding-report", type=Path)
    parser.add_argument("--source-pack", type=Path)
    parser.add_argument("--ranked-papers", type=Path)
    parser.add_argument("--asset-inventory", required=True, type=Path)
    parser.add_argument("--visual-plan", type=Path)
    parser.add_argument("--output-slides", type=Path)
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument(
        "--figure-specs-output",
        type=Path,
        help="Optional output JSON for source-grounded FigureSpec records.",
    )
    parser.add_argument(
        "--figure-specs-input",
        type=Path,
        help=(
            "Optional prior FigureSpec JSON receipt to reuse for renderer-only repairs. "
            "Cannot be combined with --live-sonnet-specs."
        ),
    )
    parser.add_argument(
        "--repair-feedback",
        type=Path,
        help="Normalized visual-QC repair feedback JSON used to regenerate failed figures.",
    )
    parser.add_argument(
        "--target-svgs-per-slide",
        type=float,
        default=0.0,
        help="Minimum SVG density to generate per markdown slide.",
    )
    parser.add_argument(
        "--allow-density-fallback",
        action="store_true",
        help=(
            "Permit production density fallback figures. By default, only explicit "
            "FIGURE_NEED slides receive SVGs so bullet-box filler cannot satisfy coverage."
        ),
    )
    parser.add_argument(
        "--live-sonnet-specs",
        action="store_true",
        help=(
            "Call the governed Anthropic Sonnet codegen model (claude-sonnet-5) "
            "to refine source-grounded FigureSpec records before rendering. "
            "Requires ANTHROPIC_API_KEY."
        ),
    )
    parser.add_argument(
        "--sonnet-batch-size",
        type=int,
        default=8,
        help="Number of FigureSpec records to send per Sonnet request.",
    )
    parser.add_argument(
        "--sonnet-timeout-seconds",
        type=float,
        default=240.0,
        help="Per-request timeout for live Sonnet FigureSpec calls.",
    )
    parser.add_argument(
        "--figure-workers",
        type=int,
        default=4,
        help="Concurrent figures generated at once (provider-I/O-bound). "
             "1 disables parallelism.",
    )
    parser.add_argument(
        "--qc-verdicts-json",
        type=Path,
        default=None,
        help="Optional {slide_id: {status, svg_sha256}} pixel-judge verdicts; a "
             "figure whose file matches a 'pass' verdict is reused unconditionally "
             "(the vision judge outranks the deterministic estimators).",
    )
    parser.add_argument(
        "--qc-feedback-json",
        type=Path,
        default=None,
        help="Optional {slide_id: feedback} map from the visual QC stage; listed slides skip reuse and regenerate with the feedback seeded into attempt 1.",
    )
    parser.add_argument(
        "--svg-stack-research",
        type=Path,
        help="Research note used to govern the Phase 16 SVG renderer stack.",
    )
    parser.add_argument("--provider", help="Paid provider for LLM-authored figure code.")
    parser.add_argument("--model", help="Provider model id for LLM-authored figure code.")
    parser.add_argument(
        "--provider-cost-approved",
        action="store_true",
        help="Required acknowledgement before paid Phase 16 provider calls.",
    )
    parser.add_argument(
        "--allowed-renderer",
        action="append",
        default=[],
        help="Allowed advanced renderer family for LLM-authored SVG code.",
    )
    parser.add_argument(
        "--disallowed-default-renderer",
        action="append",
        default=[],
        help="Renderer family that must not be selected as a Phase 16 default.",
    )
    parser.add_argument(
        "--figure-code-dir",
        type=Path,
        help="Directory for accepted generated Python figure code.",
    )
    parser.add_argument(
        "--model-receipt-output",
        type=Path,
        help="Output receipt for Opus 4.8 SVG code generation.",
    )
    parser.add_argument(
        "--code-review-output",
        type=Path,
        help="Output report for generated Python code review.",
    )
    parser.add_argument(
        "--anthropic-timeout-seconds",
        type=float,
        default=240.0,
        help="Per-request timeout for Anthropic figure-code calls.",
    )
    parser.add_argument(
        "--figure-attribution",
        choices=("chips", "in_figure"),
        default=PHASE16_FIGURE_ATTRIBUTION_MODE,
        help="Where scholarship is attributed. 'chips' (default): the slide's source chips "
        "carry work titles, authors, and links, and the figure draws NO citation cards or "
        "author lines (they were the dominant pixel-review defect: cards covering nodes). "
        "'in_figure': the pre-2026-09 contract with 'Work — Authors' cards inside the figure.",
    )
    parser.add_argument(
        "--best-effort-max-weight",
        type=int,
        default=PHASE16_BEST_EFFORT_MAX_WEIGHT,
        help="Residual geometry weight a best-effort figure may carry and still count as "
        f"accepted (default {PHASE16_BEST_EFFORT_MAX_WEIGHT}); the orchestrator raises it on "
        "its final seeded retry so one stubborn figure is recorded for the pixel judge "
        "instead of stalling the deck.",
    )
    parser.add_argument(
        "--codegen-max-attempts",
        type=int,
        default=3,
        help="Maximum Opus repair attempts per generated figure.",
    )
    parser.add_argument("--max-waivers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.workspace = args.workspace.expanduser().resolve()
    for attr in (
        "input_slides",
        "deck_contract",
        "source_binding_report",
        "source_pack",
        "ranked_papers",
        "asset_inventory",
        "visual_plan",
        "output_slides",
        "generated_dir",
        "figure_specs_output",
        "figure_specs_input",
        "repair_feedback",
        "svg_stack_research",
        "figure_code_dir",
        "model_receipt_output",
        "code_review_output",
        "qc_feedback_json",
        "qc_verdicts_json",
    ):
        setattr(args, attr, _resolve_workspace_arg(args.workspace, getattr(args, attr, None)))
    if args.deck_contract is not None:
        missing = [
            flag
            for flag, value in (
                ("--source-binding-report", args.source_binding_report),
                ("--visual-plan", args.visual_plan),
            )
            if value is None
        ]
        if missing:
            parser.error(f"--deck-contract requires {', '.join(missing)}")
        if _phase16_llm_route_requested(args):
            route_missing = [
                flag
                for flag, value in (
                    ("--provider", args.provider),
                    ("--model", args.model),
                    ("--svg-stack-research", args.svg_stack_research),
                    ("--figure-code-dir", args.figure_code_dir),
                    ("--figure-specs-output", args.figure_specs_output),
                    ("--model-receipt-output", args.model_receipt_output),
                    ("--code-review-output", args.code_review_output),
                )
                if value is None or value == ""
            ]
            if route_missing:
                parser.error(
                    "Phase 16 LLM-authored figure generation requires "
                    + ", ".join(route_missing)
                )
            if args.provider != PHASE16_REQUIRED_PROVIDER or args.model != PHASE16_REQUIRED_MODEL:
                parser.error(
                    "Phase 16 LLM-authored figure generation requires "
                    f"--provider {PHASE16_REQUIRED_PROVIDER} --model {PHASE16_REQUIRED_MODEL}"
                )
            if not args.provider_cost_approved:
                parser.error("--provider-cost-approved is required for paid Phase 16 provider calls")
            allowed = {str(renderer).strip() for renderer in args.allowed_renderer if str(renderer).strip()}
            if not allowed:
                parser.error("Phase 16 LLM-authored figure generation requires --allowed-renderer")
            if PHASE16_DISALLOWED_DEFAULT_RENDERERS & allowed:
                parser.error("matplotlib and plotly cannot be Phase 16 allowed default renderers")
            disallowed = {
                str(renderer).strip() for renderer in args.disallowed_default_renderer if str(renderer).strip()
            }
            missing_disallowed = PHASE16_DISALLOWED_DEFAULT_RENDERERS - disallowed
            if missing_disallowed:
                parser.error(
                    "Phase 16 must explicitly disallow default renderers: "
                    + ", ".join(sorted(missing_disallowed))
                )
            if "drawsvg" not in allowed:
                parser.error("Phase 16 currently requires drawsvg in --allowed-renderer")
            unsupported = allowed - PHASE16_ADVANCED_RENDERERS
            if unsupported:
                parser.error(
                    "unsupported Phase 16 renderer families: " + ", ".join(sorted(unsupported))
                )
    else:
        missing = [
            flag
            for flag, value in (
                ("--input-slides", args.input_slides),
                ("--source-pack", args.source_pack),
                ("--ranked-papers", args.ranked_papers),
                ("--output-slides", args.output_slides),
            )
            if value is None
        ]
        if missing:
            parser.error(f"legacy markdown figure generation requires {', '.join(missing)}")
    return args


def _resolve_workspace_arg(workspace: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    if value.exists():
        return value
    return workspace / value


def _phase16_llm_route_requested(args: argparse.Namespace) -> bool:
    return any(
        value
        for value in (
            args.provider,
            args.model,
            args.svg_stack_research,
            args.figure_code_dir,
            args.model_receipt_output,
            args.code_review_output,
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _workspace_relative(path: Path, workspace: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip()).strip("-")
    return normalized or "figure"


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_svg(
    path: Path,
    *,
    title: str,
    source_labels: list[str],
    source_ids: list[str],
    hook: str,
    entities: list[str],
    relationships: list[str],
    contribution_note: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = source_labels[:4] or ["SRC"]
    label_rows = "\n".join(
        f'<text x="84" y="{255 + index * 34}" class="label">{_svg_escape(label)}</text>'
        for index, label in enumerate(labels)
    )
    entity_rows = "\n".join(
        f'<text x="510" y="{255 + index * 34}" class="label">{_svg_escape(entity)}</text>'
        for index, entity in enumerate(entities[:4])
    )
    relationship_rows = "\n".join(
        f'<text x="84" y="{430 + index * 34}" class="label">{_svg_escape(relationship)}</text>'
        for index, relationship in enumerate(relationships[:3])
    )
    source_id_rows = "\n".join(
        f'<text x="840" y="{255 + index * 28}" class="source">{_svg_escape(source_id[:46])}</text>'
        for index, source_id in enumerate(source_ids[:4])
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img" aria-labelledby="title desc">
  <title id="title">{_svg_escape(title)}</title>
  <desc id="desc">{_svg_escape(hook)}</desc>
  <style>
    .bg {{ fill: #f8fafc; }}
    .rail {{ fill: #0f766e; }}
    .card {{ fill: #ffffff; stroke: #1f2937; stroke-width: 3; rx: 12; }}
    .title {{ font: 700 46px Arial, sans-serif; fill: #111827; }}
    .label {{ font: 600 25px Arial, sans-serif; fill: #111827; }}
    .hook {{ font: 400 27px Arial, sans-serif; fill: #374151; }}
    .source {{ font: 500 17px Arial, sans-serif; fill: #4b5563; }}
  </style>
  <rect class="bg" width="1280" height="720"/>
  <rect class="rail" x="0" y="0" width="28" height="720"/>
  <rect class="card" x="72" y="88" width="1136" height="500"/>
  <text x="84" y="164" class="title">{_svg_escape(title[:72])}</text>
  <text x="84" y="222" class="hook">{_svg_escape(hook[:96])}</text>
  {label_rows}
  {entity_rows}
  {source_id_rows}
  {relationship_rows}
  <text x="84" y="560" class="hook">{_svg_escape(contribution_note[:108])}</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_structured_visual_plan(args: argparse.Namespace) -> dict[str, Any]:
    deck_contract = _load_json(args.deck_contract)
    source_binding = _load_json(args.source_binding_report)
    source_by_slide = {
        str(slide.get("slide_id")): slide
        for slide in source_binding.get("slides", [])
        if isinstance(slide, dict)
    }
    generated_dir = args.generated_dir
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    semantic_review_ref = _relative_to_deck(args.generated_dir / "semantic_review.json", args.asset_inventory.parent)
    figure_specs_ref = _relative_to_deck(
        args.figure_specs_output or args.generated_dir / "figure_specs.json",
        args.asset_inventory.parent,
    )
    slides: list[dict[str, Any]] = []
    figure_specs: list[dict[str, Any]] = []
    generated_assets: list[dict[str, Any]] = []
    for index, slide in enumerate(deck_contract.get("slides", []), start=1):
        if not isinstance(slide, dict):
            continue
        slide_id = str(slide.get("slide_id") or f"slide-{index:03d}")
        placeholders = slide.get("visual_placeholders")
        placeholder = placeholders[0] if isinstance(placeholders, list) and placeholders else {}
        if not isinstance(placeholder, dict):
            placeholder = {}
        visual_id = str(placeholder.get("visual_id") or f"{slide_id}-visual")
        visual_role = str(placeholder.get("role") or "Source-grounded mechanism visual")
        source_row = source_by_slide.get(slide_id, {})
        visible_refs = source_row.get("visible_references") if isinstance(source_row, dict) else []
        source_refs = _source_refs(visible_refs)
        source_labels = [
            str(ref.get("label") or ref.get("title") or "SRC").strip() for ref in source_refs
        ]
        if not source_labels:
            source_labels = [str(label) for label in slide.get("source_label_slots", []) or []]
        source_ids = [
            str(ref.get("source_id") or ref.get("url") or ref.get("title") or "").strip()
            for ref in source_refs
            if str(ref.get("source_id") or ref.get("url") or ref.get("title") or "").strip()
        ]
        title = str(slide.get("title") or slide_id)
        entities = _required_entities(title=title, visual_role=visual_role, source_refs=source_refs)
        relationships = _required_relationships(
            title=title,
            visual_role=visual_role,
            source_refs=source_refs,
        )
        contribution_note = _contribution_note(source_refs)
        hook = _narration_hook(
            str((slide.get("narration_hooks") or ["Explain the source-grounded visual."])[0]),
            entities=entities,
            relationships=relationships,
        )
        depiction_type = _depiction_type(title=title, visual_role=visual_role, visual_concept=visual_role)
        visual_concept = _source_grounded_visual_concept(
            title=title,
            visual_role=visual_role,
            depiction_type=depiction_type,
            entities=entities,
            relationships=relationships,
        )
        depiction_type = _depiction_type(title=title, visual_role=visual_role, visual_concept=visual_concept)
        composition_budget = _composition_budget(
            entities=entities,
            relationships=relationships,
            visual_concept=visual_concept,
        )
        asset_path = generated_dir / f"{_slug(slide_id)}-{_slug(visual_id)}.svg"
        relative_asset = _workspace_relative(asset_path, args.workspace)
        figure_design_brief = {
            "visual_role": visual_role,
            "depiction_type": depiction_type,
            "claim_to_visualize": title,
            "visual_concept": visual_concept,
            "composition_budget": composition_budget,
            "required_entities": entities,
            "required_relationships": relationships,
            "source_labels": source_labels,
            "source_ids": source_ids,
            "contribution_note": contribution_note,
            "narration_hook": hook,
            "forbidden_patterns": _forbidden_patterns(),
        }
        figure = {
            "visual_id": visual_id,
            "figure_design_brief": figure_design_brief,
            "depiction_type": depiction_type,
            "visual_concept": visual_concept,
            "composition_budget": composition_budget,
            "renderer_family": "direct_svg",
            "required_entities": entities,
            "required_relationships": relationships,
            "source_labels": source_labels or ["SRC-UNRESOLVED"],
            "source_ids": source_ids,
            "contribution_note": contribution_note,
            "narration_hook": hook,
            "forbidden_patterns": _forbidden_patterns(),
            "asset_paths": [relative_asset],
        }
        slides.append({"slide_id": slide_id, "figures": [figure]})
        spec = {
            "slide_id": slide_id,
            "visual_id": visual_id,
            "title": title,
            "figure_design_brief": figure_design_brief,
            "depiction_type": depiction_type,
            "visual_concept": visual_concept,
            "composition_budget": composition_budget,
            "renderer_family": figure["renderer_family"],
            "required_entities": entities,
            "required_relationships": relationships,
            "source_labels": figure["source_labels"],
            "source_ids": source_ids,
            "asset_path": relative_asset,
            "contribution_note": contribution_note,
            "narration_hook": hook,
            "forbidden_patterns": _forbidden_patterns(),
        }
        figure_specs.append(spec)
        generated_assets.append(
            {
                "path": relative_asset,
                "slide_id": slide_id,
                "visual_id": visual_id,
                "renderer_family": figure["renderer_family"],
                "source_labels": figure["source_labels"],
                "source_ids": source_ids,
                "sha256": None,
                "production_evidence": False,
                "semantic_review_ref": semantic_review_ref,
                "model_governance": {
                    "provider": "local",
                    "model_id": "deterministic-structured-svg",
                    "status": "deterministic_local_no_provider_call",
                },
            }
        )
        if not args.dry_run:
            _write_svg(
                asset_path,
                title=title,
                source_labels=figure["source_labels"],
                source_ids=source_ids,
                hook=hook,
                entities=entities,
                relationships=relationships,
                contribution_note=contribution_note,
            )
            generated_assets[-1]["sha256"] = _sha256_file(asset_path)
            generated_assets[-1]["production_evidence"] = True

    visual_plan = {
        "schema_version": 2,
        "presentation_id": str(deck_contract.get("presentation_id") or args.presentation_id),
        "slides": slides,
    }
    validation = validate_visual_plan_payload(visual_plan)
    if validation.errors:
        raise ValueError(json.dumps(validation.to_dict(), indent=2))
    inventory = {
        "schema_version": 2,
        "presentation_id": args.presentation_id,
        "generated_at": generated_at,
        "status": "dry_run" if args.dry_run else "succeeded",
        "generator": "scripts/presentations/generate_programmatic_figures.py",
        "source_mode": "structured-contract",
        "semantic_review_required": True,
        "semantic_review_ref": semantic_review_ref,
        "figure_specs_ref": figure_specs_ref,
        "model_governance": {
            "provider": "local",
            "model_id": "deterministic-structured-svg",
            "status": "deterministic_local_no_provider_call",
        },
        "figure_specs": figure_specs,
        "generated_assets": generated_assets,
    }
    specs_payload = {
        "schema_version": 2,
        "presentation_id": args.presentation_id,
        "generated_at": generated_at,
        "source_binding_report": _workspace_relative(args.source_binding_report, args.workspace),
        "figure_specs": figure_specs,
    }
    if args.dry_run:
        return {
            "visual_plan": visual_plan,
            "asset_inventory": inventory,
            "figure_specs": specs_payload,
        }
    write_json_atomic(args.visual_plan, visual_plan)
    write_json_atomic(args.asset_inventory, inventory)
    if args.figure_specs_output:
        write_json_atomic(args.figure_specs_output, specs_payload)
    return inventory


def generate_phase16_llm_visual_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Generate Phase 16 SVGs from Opus-authored bounded Python code."""

    try:
        import drawsvg  # noqa: F401
    except ImportError as exc:
        raise ValueError("Phase 16 requires drawsvg>=2.4.0; install project dependencies") from exc
    try:
        import anthropic
    except ImportError as exc:
        raise ValueError("Phase 16 requires the anthropic Python package") from exc

    deck_contract = _load_json(args.deck_contract)
    source_binding = _load_json(args.source_binding_report)
    visual_plan = _load_json(args.visual_plan)
    research_text = args.svg_stack_research.read_text(encoding="utf-8", errors="replace")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    generated_dir = args.generated_dir
    code_dir = args.figure_code_dir
    generated_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    rows = _phase16_figure_rows(
        deck_contract=deck_contract,
        source_binding=source_binding,
        visual_plan=visual_plan,
        args=args,
    )
    if args.dry_run:
        return {
            "schema_version": 2,
            "status": "dry_run",
            "presentation_id": args.presentation_id,
            "figure_count": len(rows),
            "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
            "selected_renderer_family": "drawsvg",
        }

    client = anthropic.Anthropic(timeout=args.anthropic_timeout_seconds)
    generated_assets: list[dict[str, Any]] = []
    figure_specs: list[dict[str, Any]] = []
    code_reviews: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    updated_slides: list[dict[str, Any]] = []

    acceptance_cache = _phase16_load_cache(args)
    qc_verdicts: dict[str, dict[str, str]] = {}
    if getattr(args, "qc_verdicts_json", None) and args.qc_verdicts_json.is_file():
        try:
            payload_v = json.loads(args.qc_verdicts_json.read_text(encoding="utf-8"))
            if isinstance(payload_v, dict):
                qc_verdicts = {
                    str(k): v for k, v in payload_v.items() if isinstance(v, dict)
                }
        except (OSError, json.JSONDecodeError):
            qc_verdicts = {}
    qc_feedback: dict[str, str] = {}
    if getattr(args, "qc_feedback_json", None) and args.qc_feedback_json.is_file():
        try:
            payload_fb = json.loads(args.qc_feedback_json.read_text(encoding="utf-8"))
            if isinstance(payload_fb, dict):
                qc_feedback = {str(k): str(v) for k, v in payload_fb.items()}
        except (OSError, json.JSONDecodeError):
            qc_feedback = {}
    updated_cache: dict[str, Any] = {}
    reused_count = 0
    figure_results: list[dict[str, Any]] = []

    def _process_row(row: dict[str, Any]) -> dict[str, Any]:
        """Reuse-or-generate a single figure. Pure w.r.t. shared aggregation
        state (writes only this figure's own files), so it is safe to run
        concurrently; aggregation happens serially afterwards in input order."""
        row_feedback = qc_feedback.get(str(row["slide_id"]), "")
        accepted = None
        if not row_feedback:
            accepted = _phase16_reuse_candidate(
                row=row, args=args, research_text=research_text,
                cache=acceptance_cache, qc_verdicts=qc_verdicts,
            )
        if accepted is not None:
            print(
                f"[phase16] {row['slide_id']} {row['visual_id']}: reused previously "
                "accepted figure (prompt + files unchanged, all checks pass)",
                file=sys.stderr,
            )
            return accepted
        return _generate_single_phase16_figure(
            row=row,
            args=args,
            client=client,
            research_text=research_text,
            initial_feedback=row_feedback,
        )

    workers = max(1, int(getattr(args, "figure_workers", 1) or 1))
    with provider_receipt_context(
        # D-QUICK-260715-rz1: this previously pinned workspace_root to the
        # repo root (PROJECT_ROOT) instead of the conference workspace,
        # which is exactly why phase16_opus_svg_codegen was one of the
        # confirmed misplaced-receipt segments. Use args.workspace instead.
        workspace_root=args.workspace,
        stage_id="phase16_opus_svg_codegen",
        caller=__file__,
    ):
        if workers == 1 or len(rows) <= 1:
            accepted_rows = [_process_row(row) for row in rows]
        else:
            # Figures are independent and the work is provider-I/O-bound; fan
            # out across threads. copy_context() per task propagates the
            # provider-receipt ContextVar (threads don't inherit it).
            snapshots = [contextvars.copy_context() for _ in rows]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                accepted_rows = list(
                    executor.map(
                        lambda pair: pair[0].run(_process_row, pair[1]),
                        zip(snapshots, rows),
                    )
                )
            print(f"[phase16] generated {len(rows)} figures with {workers} workers", file=sys.stderr)

        for row, accepted in zip(rows, accepted_rows):
            if accepted.get("reused"):
                reused_count += 1
            model_calls.extend(accepted["model_calls"])
            code_reviews.extend(accepted["code_reviews"])
            updated_slides.append(
                {
                    "slide_id": row["slide_id"],
                    "section_id": row.get("section_id"),
                    "figures": [accepted["visual_plan_figure"]],
                }
            )
            figure_specs.append(accepted["figure_spec"])
            generated_assets.append(accepted["asset"])
            fully_passed = accepted["code_reviews"][0].get("status") == "pass"
            figure_results.append(
                {
                    "slide_id": row["slide_id"],
                    "visual_id": row["visual_id"],
                    "status": "reused"
                    if accepted.get("reused")
                    else ("generated" if fully_passed else "best_effort"),
                }
            )
            if fully_passed:
                # Only fully-passing figures are reusable; best-effort accepts
                # must regenerate next run.
                updated_cache[str(row["visual_id"])] = {
                    "prompt_sha256": accepted.get("prompt_sha256"),
                    "response_sha256": accepted.get("response_sha256"),
                    "code_sha256": accepted["figure_spec"]["code_sha256"],
                    "svg_sha256": accepted["figure_spec"]["svg_sha256"],
                }

    write_json_atomic(_phase16_cache_path(args), updated_cache)

    new_visual_plan = {
        "schema_version": visual_plan.get("schema_version", 2),
        "presentation_id": str(visual_plan.get("presentation_id") or args.presentation_id),
        "slides": updated_slides,
    }
    validation = validate_visual_plan_payload(new_visual_plan)
    if validation.errors:
        raise ValueError(json.dumps(validation.to_dict(), indent=2))

    all_errors = [
        finding
        for review in code_reviews
        for finding in review.get("findings", [])
        if isinstance(finding, dict) and finding.get("severity") == "error"
    ]
    code_review_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "presentation_id": args.presentation_id,
        "status": "fail" if all_errors else "pass",
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "renderer_family": "drawsvg",
        "render_spec_generation_allowed": not all_errors,
        "reviews": code_reviews,
    }
    model_receipt_payload = {
        "schema_version": 1,
        "generated_at": generated_at,
        "presentation_id": args.presentation_id,
        "status": "pass",
        "generation_route": "llm_authored_code",
        "provider": PHASE16_REQUIRED_PROVIDER,
        "model": PHASE16_REQUIRED_MODEL,
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "selected_renderer_family": "drawsvg",
        "allowed_renderers": sorted(set(args.allowed_renderer)),
        "disallowed_default_renderers": sorted(set(args.disallowed_default_renderer)),
        "svg_stack_research": _workspace_relative(args.svg_stack_research, PROJECT_ROOT),
        "request_count": len(model_calls),
        "figure_count": len(figure_specs),
        "reused_figure_count": reused_count,
        "figure_results": figure_results,
        "provider_receipt_paths": [
            call["provider_receipt_path"]
            for call in model_calls
            if call.get("provider_receipt_path")
        ],
        "calls": model_calls,
    }
    if code_review_payload["status"] != "pass":
        model_receipt_payload["status"] = "blocked"

    figure_specs_payload = {
        "schema_version": 2,
        "generated_at": generated_at,
        "presentation_id": args.presentation_id,
        "generation_route": "llm_authored_code",
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "figure_specs": figure_specs,
    }
    semantic_review_ref = _relative_to_deck(
        args.asset_inventory.parent / "figure_semantic_review.json",
        args.asset_inventory.parent,
    )
    inventory = {
        "schema_version": 2,
        "presentation_id": args.presentation_id,
        "generated_at": generated_at,
        "status": "succeeded" if code_review_payload["status"] == "pass" else "blocked",
        "generator": "scripts/presentations/generate_programmatic_figures.py",
        "source_mode": "structured-contract",
        "generation_route": "llm_authored_code",
        "semantic_review_required": True,
        "semantic_review_ref": semantic_review_ref,
        "figure_specs_ref": _relative_to_deck(args.figure_specs_output, args.asset_inventory.parent),
        "model_governance": {
            "provider": PHASE16_REQUIRED_PROVIDER,
            "model_id": PHASE16_REQUIRED_MODEL,
            "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
            "status": "live_opus_4_8_generated_code",
        },
        "figure_specs": figure_specs,
        "generated_assets": generated_assets,
    }

    write_json_atomic(args.visual_plan, new_visual_plan)
    write_json_atomic(args.asset_inventory, inventory)
    write_json_atomic(args.figure_specs_output, figure_specs_payload)
    write_json_atomic(args.model_receipt_output, model_receipt_payload)
    write_json_atomic(args.code_review_output, code_review_payload)
    return inventory


def _phase16_figure_rows(
    *,
    deck_contract: dict[str, Any],
    source_binding: dict[str, Any],
    visual_plan: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    deck_by_slide = {
        str(slide.get("slide_id")): slide
        for slide in deck_contract.get("slides", [])
        if isinstance(slide, dict)
    }
    source_by_slide = {
        str(slide.get("slide_id")): slide
        for slide in source_binding.get("slides", [])
        if isinstance(slide, dict)
    }
    rows: list[dict[str, Any]] = []
    for slide in visual_plan.get("slides", []):
        if not isinstance(slide, dict):
            continue
        slide_id = str(slide.get("slide_id") or "").strip()
        if not slide_id:
            continue
        figures = slide.get("figures")
        if not isinstance(figures, list) or not figures:
            continue
        figure = figures[0] if isinstance(figures[0], dict) else {}
        deck_slide = deck_by_slide.get(slide_id, {})
        source_slide = source_by_slide.get(slide_id, {})
        refs = _source_refs(source_slide.get("visible_references") if isinstance(source_slide, dict) else [])
        brief = figure.get("figure_design_brief") if isinstance(figure, dict) else {}
        if not isinstance(brief, dict):
            brief = {}
        visual_id = str(figure.get("visual_id") or f"{slide_id}-figure")
        source_labels = _string_list(figure.get("source_labels")) or [
            str(ref.get("label") or "").strip() for ref in refs if str(ref.get("label") or "").strip()
        ]
        source_ids = _string_list(figure.get("source_ids")) or [
            str(ref.get("source_id") or ref.get("url") or "").strip()
            for ref in refs
            if str(ref.get("source_id") or ref.get("url") or "").strip()
        ]
        required_entities = _phase16_required_entities(figure=figure, brief=brief, refs=refs)
        required_relationships = _phase16_required_relationships(figure=figure)
        title = str(deck_slide.get("title") or brief.get("claim_to_visualize") or slide_id)
        visual_role = str(
            brief.get("visual_role")
            or figure.get("visual_role")
            or figure.get("depiction_type")
            or deck_slide.get("visual_role")
            or ""
        )
        depiction_type = str(
            figure.get("depiction_type")
            or brief.get("depiction_type")
            or _depiction_type(title=title, visual_role=visual_role)
        )
        visual_concept = _phase16_visual_concept(
            figure=figure,
            brief=brief,
            deck_slide=deck_slide,
            title=title,
            visual_role=visual_role,
            depiction_type=depiction_type,
            entities=required_entities,
            relationships=required_relationships,
        )
        # The plan's own depiction vocabulary ("evidence_chart: ...") outranks a
        # keyword-derived "system_workflow" carried over from an earlier pass.
        depiction_type = _depiction_from_concept(visual_concept) or depiction_type
        composition_budget = _composition_budget(
            entities=required_entities,
            relationships=required_relationships,
            visual_concept=visual_concept,
        )
        source_chip_bindings = _source_chip_bindings(refs)
        asset_path = args.generated_dir / f"{_slug(slide_id)}-{_slug(visual_id)}.svg"
        code_path = args.figure_code_dir / f"{_slug(slide_id)}-{_slug(visual_id)}.py"
        people = _unique(
            [
                *(_string_list(brief.get("people"))),
                *[
                    person
                    for ref in refs
                    for person in [*(_string_list(ref.get("speakers"))), *(_string_list(ref.get("authors")))]
                ],
            ],
            limit=10,
        )
        systems = _unique(
            [
                *(_string_list(brief.get("systems"))),
                *[system for ref in refs for system in _string_list(ref.get("systems"))],
            ],
            limit=8,
        )
        datasets = _unique(
            [
                *(_string_list(brief.get("datasets"))),
                *[dataset for ref in refs for dataset in _string_list(ref.get("datasets"))],
            ],
            limit=6,
        )
        rows.append(
            {
                "slide_id": slide_id,
                "section_id": str(slide.get("section_id") or deck_slide.get("section_id") or ""),
                "cluster_ids": _string_list(deck_slide.get("cluster_ids")) if isinstance(deck_slide, dict) else [],
                "visual_id": visual_id,
                "title": title,
                "claim": str(brief.get("claim_to_visualize") or deck_slide.get("viewer_facing_claim") or ""),
                "depiction_type": depiction_type,
                "visual_role": visual_role,
                "visual_concept": visual_concept,
                "composition_budget": composition_budget,
                "proof_object": str(brief.get("proof_object") or deck_slide.get("proof_object") or ""),
                "narration_hook": str(figure.get("narration_hook") or ""),
                "contribution_note": str(figure.get("contribution_note") or ""),
                "source_labels": source_labels,
                "source_ids": source_ids,
                "source_chip_bindings": source_chip_bindings,
                "reference_ids": _string_list(figure.get("reference_ids")) or source_ids,
                "required_entities": required_entities,
                "required_relationships": required_relationships,
                "forbidden_patterns": _string_list(figure.get("forbidden_patterns")) or _forbidden_patterns(),
                "people": people,
                "systems": systems,
                "datasets": datasets,
                "source_refs": refs,
                "asset_path": asset_path,
                "relative_asset": _workspace_relative(asset_path, args.workspace),
                "code_path": code_path,
                "relative_code": _workspace_relative(code_path, args.workspace),
                "original_figure": figure,
            }
        )
    return assign_figure_archetypes(rows)


def _phase16_svg_is_bullet_box(asset_path: Path) -> bool:
    """Mirror the figure semantic-review bullet-box heuristic so the codegen loop
    can self-repair: a rendered figure that is only a grid of short text labels
    with no arrow/flow relationships reads as bullet boxes and is rejected."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    nodes = [
        html.unescape(re.sub(r"<[^>]+>", "", match.group(1)).strip())
        for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL)
    ]
    nodes = [node for node in nodes if node]
    if len(nodes) < 3:
        return False
    short_lines = [node for node in nodes if len(node.split()) <= 10]
    has_arrow = any("->" in node or "=>" in node for node in nodes)
    return len(short_lines) == len(nodes) and not has_arrow


def _phase16_legibility_issues(row: dict[str, Any], *, workspace: Path) -> list[str]:
    """Run the render-time visual legibility gate on a single rendered figure so the
    codegen loop self-repairs against the exact checks the renderer enforces
    (truncated text fragments, too-many-labels, tiny fonts, scaffold/raw-id leakage,
    and source/person attribution mismatches)."""

    asset_path = row.get("asset_path")
    if not isinstance(asset_path, Path) or not asset_path.is_file():
        return []
    source_chips = []
    for ref in row.get("source_refs", []) or []:
        if not isinstance(ref, dict):
            continue
        source_chips.append(
            {
                "label": ref.get("label") or ref.get("source_label") or "",
                "authors": _string_list(ref.get("authors")),
                "speakers": _string_list(ref.get("speakers")),
                "title": str(ref.get("title") or ""),
                "contribution_note": str(ref.get("contribution_note") or ""),
                "systems": _string_list(ref.get("systems")),
                "works": _string_list(ref.get("works")),
            }
        )
    slide = {
        "slide_id": row.get("slide_id"),
        "visual_asset_refs": [asset_path.name],
        "source_chips": source_chips,
    }
    try:
        findings = _visual_legibility_findings([slide], deck_dir=asset_path.parent, workspace=workspace)
    except Exception:  # pragma: no cover - legibility gate must never crash codegen
        return []
    # Strip the slide/file prefix, keep the human-readable issue tail.
    issues = [finding.split(":", 2)[-1] if finding.count(":") >= 2 else finding for finding in findings]
    issues.extend(_phase16_oversized_arrowhead_issues(asset_path))
    return issues


# A best-effort figure whose residual defects weigh no more than this is
# accepted with the defects recorded as residual warnings (the pixel-level
# review stays the authority and regenerates it with judge feedback); heavier
# residue keeps the receipt blocked so the orchestrator retries with the
# measured defects seeded. One line through a label = 2; one dangling arrow = 3.
# Calibrated on the ICML 2026 agentic-AI canary: 24/25 figures converge to
# weight <= 2 within the orchestrator's two seeded runs; the one that does not
# (a dense quadrant chart) sat at 4-5 across sixteen attempts. Five lets a
# single stubborn figure through with its residue recorded for the pixel judge
# instead of stalling the whole deck.
PHASE16_BEST_EFFORT_MAX_WEIGHT = 5
# Attribution lives on the slide's source chips (title, authors, paper + poster
# links). In-figure citation cards and author lines were the single largest
# residual defect class in the ICML 2026 agentic-AI pixel review ("citation
# cards cover the node", "author line overflows its card"), and they duplicate
# what the chips already show. Reviewer approval 2026-09-09.
PHASE16_FIGURE_ATTRIBUTION_MODE = "chips"
_GEOMETRY_ERROR_WEIGHTS = {
    "dangling_connector": 3,
    "arrowhead_over_text": 3,
    "text_outside_canvas": 3,
    "text_overflow": 2,
    "connector_through_text": 2,
    "shape_overlap": 3,  # cards covering a node: the pixel judge rejects it every time
    "directionless_connector": 2,
    "arrowhead_inside_shape": 3,
    "label_on_border": 2,
    "text_overlap": 1,
}


# Residue the best-effort path may never accept. Geometry residue is cosmetic
# and the pixel judge can waive it; a required label missing from the rendered
# figure is a content gap that the semantic-review stage rejects outright, so
# downgrading it only buys a wasted render/PDF/screenshot round-trip before the
# deck fails one stage later (ICML 2026 topic 01, slide 22).
_NON_DOWNGRADABLE_ISSUE_PREFIXES = ("required_entity_coverage",)


def _has_non_downgradable_issue(quality_issues: list[str]) -> bool:
    return any(
        str(issue).strip().startswith(_NON_DOWNGRADABLE_ISSUE_PREFIXES)
        for issue in quality_issues
    )


def _phase16_residual_weight(quality_issues: list[str]) -> int:
    """Severity weight of the blocking issues left on a rendered attempt."""

    weight = 0
    for issue in quality_issues:
        check_id = issue.split(":", 1)[0].strip()
        weight += _GEOMETRY_ERROR_WEIGHTS.get(check_id, 3)  # non-geometry contract issues are heavy
    return weight


_COVERAGE_IGNORED_TOKENS = frozenset({"source", "slide", "figure", "visual", "show", "showing", "depict"})


def _svg_visible_text(svg_text: str) -> str:
    nodes = []
    for match in re.finditer(r"<text\b[^>]*>(.*?)</text>", svg_text, flags=re.DOTALL | re.IGNORECASE):
        value = html.unescape(re.sub(r"<[^>]+>", " ", match.group(1)))
        nodes.append(" ".join(value.split()))
    return " ".join(node for node in nodes if node)


def _coverage_tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower()) if t not in _COVERAGE_IGNORED_TOKENS]


def _term_present_in_svg_text(term: str, visible_text: str) -> bool:
    """The semantic review's rule (review_programmatic_figures._term_present),
    applied at codegen time to the RENDERED text so a figure that drops a
    required label during layout repair is repaired in the loop instead of
    failing the later semantic-review stage."""

    normalized = str(term).strip().lower()
    if not normalized:
        return True
    lower = visible_text.lower()
    if normalized in lower:
        return True
    tokens = set(_coverage_tokens(normalized))
    svg_tokens = set(_coverage_tokens(visible_text))
    if tokens and tokens.issubset(svg_tokens):
        return True
    if len(tokens) >= 3:
        return sum(1 for t in tokens if t in svg_tokens) / len(tokens) >= 0.6
    return False


def _phase16_coverage_issues(row: dict[str, Any]) -> list[str]:
    asset_path = row.get("asset_path")
    if not isinstance(asset_path, Path) or not asset_path.is_file():
        return []
    try:
        visible = _svg_visible_text(asset_path.read_text(encoding="utf-8", errors="replace"))
    except OSError:  # pragma: no cover
        return []
    issues = []
    for term in _string_list(row.get("required_entities")):
        if not _term_present_in_svg_text(term, visible):
            issues.append(
                f"required_entity_coverage: the required label '{term}' is not visible in the rendered "
                "figure; draw it exactly as given (as a node title or annotation)"
            )
    return issues[:4]


def _phase16_geometry_issues(asset_path: Path) -> tuple[list[str], list[str]]:
    """Measured geometry defects in the rendered SVG (see
    ``conference_pipeline.presentation.svg_geometry``): labels that cross their
    box border, arrows that never reach a node, connectors through text, text
    outside the canvas. ``error`` findings block acceptance so the model gets
    exact repair coordinates on the next attempt; ``warn`` findings ride along
    as advisory hints. Against the ICML 2026 agentic-AI canary these errors
    reproduced 12 of the 14 pixel-review failures with 3 extra flags."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return [], []
    try:
        findings = geometry_findings(svg_text)
    except Exception:  # pragma: no cover - geometry must never crash codegen
        return [], []
    errors = [f"{f.check_id}: {f.message}" for f in findings if f.severity == "error"]
    warnings = [f"{f.check_id}: {f.message}" for f in findings if f.severity == "warn"]
    return errors[:6], warnings[:4]


def _phase16_advisory_issues(asset_path: Path) -> list[str]:
    """Width-ESTIMATE geometry checks (flat 0.55em glyphs). Kept as advisory
    repair feedback only; the measured checks in ``_phase16_geometry_issues``
    are the blocking ones and the vision QC stage remains the authoritative
    pixel-level judge."""

    issues: list[str] = []
    issues.extend(_phase16_text_exceeds_shape_issues(asset_path))
    issues.extend(_phase16_text_overlap_issues(asset_path))
    issues.extend(_phase16_line_over_text_issues(asset_path))
    issues.extend(_phase16_text_exceeds_canvas_issues(asset_path))
    return issues


def _phase16_line_over_text_issues(asset_path: Path) -> list[str]:
    """Detect straight connector lines that cross a text label's estimated
    bounding box (reviewers consistently flag line-through-text as a blocker)."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    lines = [
        tuple(float(g) for g in m.groups())
        for m in re.finditer(
            r"<line\b[^>]*x1=[\"']?(-?[\d.]+)[\"']?[^>]*y1=[\"']?(-?[\d.]+)[\"']?"
            r"[^>]*x2=[\"']?(-?[\d.]+)[\"']?[^>]*y2=[\"']?(-?[\d.]+)",
            svg_text,
        )
    ]
    # Simple two-point connector paths ("M x,y L x,y"); multi-segment paths
    # (arrowhead triangles etc.) deliberately do not match.
    lines.extend(
        tuple(float(g) for g in m.groups())
        for m in re.finditer(
            r"<path\b[^>]*d=[\"']\s*M\s*(-?[\d.]+)[, ]\s*(-?[\d.]+)\s*"
            r"L\s*(-?[\d.]+)[, ]\s*(-?[\d.]+)\s*[\"']",
            svg_text,
        )
    )
    if not lines:
        return []
    issues: list[str] = []
    for text in _phase16_positioned_texts(svg_text):
        top = text["y"] - text["fs"]
        bottom = text["y"] + text["fs"] * 0.25
        for x1, y1, x2, y2 in lines:
            if _segment_intersects_rect(x1, y1, x2, y2, text["x0"], top, text["x1"], bottom):
                issues.append(
                    f"line_over_text: a connector line crosses the label "
                    f"'{text['body'][:32]}'; route lines around labels with >=10px clearance"
                )
                break
    return issues[:4]


def _segment_intersects_rect(
    x1: float, y1: float, x2: float, y2: float,
    rx0: float, ry0: float, rx1: float, ry1: float,
) -> bool:
    if max(x1, x2) < rx0 or min(x1, x2) > rx1 or max(y1, y2) < ry0 or min(y1, y2) > ry1:
        return False
    if rx0 <= x1 <= rx1 and ry0 <= y1 <= ry1:
        return True
    if rx0 <= x2 <= rx1 and ry0 <= y2 <= ry1:
        return True

    def _side(px: float, py: float, qx: float, qy: float, ax: float, ay: float) -> float:
        return (qx - px) * (ay - py) - (qy - py) * (ax - px)

    corners = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
    signs = [_side(x1, y1, x2, y2, cx, cy) for cx, cy in corners]
    return min(signs) < 0 < max(signs)


def _phase16_oversized_arrowhead_issues(asset_path: Path) -> list[str]:
    """Standalone polygons in these figures are almost always arrowheads; the
    style contract caps them at ~10px on a side."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    issues: list[str] = []
    shapes: list[list[tuple[str, str]]] = [
        re.findall(r"(-?[\d.]+)[,\s]+(-?[\d.]+)", m.group(1))
        for m in re.finditer(r"<polygon\b[^>]*points=[\"']([^\"']+)", svg_text)
    ]
    # Closed triangular paths ("M.. L.. L.. Z") are arrowheads too.
    shapes.extend(
        re.findall(r"(-?[\d.]+)[,\s]+(-?[\d.]+)", m.group(1))
        for m in re.finditer(
            r"<path\b[^>]*d=[\"']\s*(M[^\"']*Z)\s*[\"']", svg_text
        )
        if m.group(1).count("L") in (2, 3)
    )
    for coords in shapes:
        if len(coords) < 3:
            continue
        xs = [float(x) for x, _ in coords]
        ys = [float(y) for _, y in coords]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if min(width, height) > 16 and max(width, height) > 20:
            issues.append(
                f"oversized_arrowhead: a filled shape spans {width:.0f}x{height:.0f}px; "
                "arrowheads must stay ~10px per side and never overlap boxes or text"
            )
    return issues[:3]


def _phase16_text_exceeds_canvas_issues(asset_path: Path) -> list[str]:
    """Text whose estimated span crosses the 1280x720 canvas edge renders
    visibly clipped mid-word."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    issues: list[str] = []
    for text in _phase16_positioned_texts(svg_text):
        if text["x1"] > 1276 or text["x0"] < 4:
            issues.append(
                f"text_exceeds_canvas: '{text['body'][:40]}' extends past the canvas "
                f"edge (span {text['x0']:.0f}..{text['x1']:.0f}); shorten the label or "
                "move it left"
            )
    return issues[:3]


def _phase16_positioned_texts(svg_text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in re.finditer(r"<text\b([^>]*)>(.*?)</text>", svg_text, flags=re.DOTALL | re.I):
        attrs, raw_body = m.group(1), m.group(2)
        body = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw_body)).split())
        x_m = re.search(r"\bx=[\"']?(-?[\d.]+)", attrs)
        y_m = re.search(r"\by=[\"']?(-?[\d.]+)", attrs)
        if not body or not x_m or not y_m:
            continue
        fs_m = re.search(r"font-size=[\"']?([\d.]+)", attrs)
        anchor_m = re.search(r"text-anchor=[\"']?(\w+)", attrs)
        font_size = float(fs_m.group(1)) if fs_m else 16.0
        anchor = anchor_m.group(1) if anchor_m else "start"
        width = len(body) * font_size * 0.55
        x = float(x_m.group(1))
        if anchor == "middle":
            x0, x1 = x - width / 2, x + width / 2
        elif anchor == "end":
            x0, x1 = x - width, x
        else:
            x0, x1 = x, x + width
        out.append(
            {"body": body, "y": float(y_m.group(1)), "x0": x0, "x1": x1, "fs": font_size}
        )
    return out


def _phase16_text_overlap_issues(asset_path: Path) -> list[str]:
    """Estimate pairwise text-on-text collisions (e.g. SRC chips drawn over a
    node's wrapped title) that attribute-level checks cannot see."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    texts = _phase16_positioned_texts(svg_text)
    issues: list[str] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if abs(a["y"] - b["y"]) >= 0.7 * max(a["fs"], b["fs"]):
                continue
            inter = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
            if inter > 14:
                issues.append(
                    f"text_overlap_estimate: '{a['body'][:28]}' collides with "
                    f"'{b['body'][:28]}' (~{inter:.0f}px); give every label its own "
                    "clear position with >=10px separation"
                )
    return issues[:4]


def _phase16_text_exceeds_shape_issues(asset_path: Path) -> list[str]:
    """Estimate whether centered text overflows its containing circle — overflow
    puts the label under the incoming arrowheads (an audience-visible defect the
    attribute-level checks cannot see)."""

    try:
        svg_text = asset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return []
    circles = [
        (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        for m in re.finditer(
            r"<circle\b[^>]*cx=[\"']?(-?[\d.]+)[\"']?[^>]*cy=[\"']?(-?[\d.]+)[\"']?[^>]*r=[\"']?([\d.]+)",
            svg_text,
        )
    ]
    rects = []
    for m in re.finditer(
        r"<rect\b[^>]*x=[\"']?(-?[\d.]+)[\"']?[^>]*y=[\"']?(-?[\d.]+)[\"']?"
        r"[^>]*width=[\"']?([\d.]+)[\"']?[^>]*height=[\"']?([\d.]+)",
        svg_text,
    ):
        rx, ry, rw, rh = (float(g) for g in m.groups())
        # ignore the full-canvas background rect
        if rw >= 1200 and rh >= 650:
            continue
        rects.append((rx, ry, rw, rh))
    if not circles and not rects:
        return []
    issues: list[str] = []
    for text in _phase16_positioned_texts(svg_text):
        body, y = text["body"], text["y"]
        x_center = (text["x0"] + text["x1"]) / 2
        half_width = (text["x1"] - text["x0"]) / 2
        for cx, cy, r in circles:
            if abs(y - cy) <= r and abs(x_center - cx) <= r and half_width > (r - 10):
                issues.append(
                    f"text_exceeds_shape: '{body[:40]}' is wider than its circle "
                    f"(half-width ~{half_width:.0f}px vs radius {r:.0f}px); enlarge the "
                    "shape, shorten/wrap the label, or use a box sized to the text"
                )
                break
        else:
            for rx, ry, rw, rh in rects:
                inside_vertically = ry <= y <= ry + rh
                center_inside = rx <= x_center <= rx + rw
                if not (inside_vertically and center_inside):
                    continue
                # The 0.55em glyph estimate has slack; only flag text whose
                # estimated span clearly crosses the box border.
                if text["x0"] < rx - 12 or text["x1"] > rx + rw + 12:
                    issues.append(
                        f"text_exceeds_shape: '{body[:40]}' extends past its box edge "
                        f"(box width {rw:.0f}px); widen the box or shorten/wrap the label"
                    )
                    break
    return issues


def _generate_single_phase16_figure(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    client: Any,
    research_text: str,
    initial_feedback: str = "",
) -> dict[str, Any]:
    feedback = initial_feedback
    attempt_records: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []
    defect_history: list[str] = []
    accepted_code = ""
    accepted_prompt = ""
    accepted_response = ""
    accepted_review: dict[str, Any] | None = None
    best_effort: dict[str, Any] | None = None

    for attempt in range(1, max(1, int(args.codegen_max_attempts)) + 1):
        prompt = _phase16_codegen_prompt(row=row, research_text=research_text, feedback=feedback)
        prompt_sha = _sha256_text(prompt)
        result = anthropic_messages_create(
            client,
            caller=__file__,
            model=PHASE16_REQUIRED_MODEL,
            max_tokens=6000,
            system=(
                "You write safe, bounded Python code for editorial SVG figures. "
                "Return code only."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = _anthropic_text(result)
        response_sha = _sha256_text(response_text)
        provider_receipt_path = getattr(result.receipt, "receipt_path", None)
        code = _normalize_phase16_generated_code(_extract_python_code(response_text))
        review = _review_phase16_generated_code(code, row=row)
        execution_status = "not_run"
        execution_error = ""
        if review["status"] == "pass":
            row["code_path"].write_text(code.rstrip() + "\n", encoding="utf-8")
            execution_error = _execute_phase16_generated_code(row["code_path"], row["asset_path"])
            execution_status = "fail" if execution_error else "pass"
            if execution_error:
                review = {
                    **review,
                    "status": "fail",
                    "findings": [
                        *review["findings"],
                        {
                            "check_id": "execution_failed",
                            "severity": "error",
                            "message": execution_error,
                        },
                    ],
                }
            else:
                quality_issues: list[str] = []
                if _phase16_svg_is_bullet_box(row["asset_path"]):
                    quality_issues.append(
                        "bullet-box: it is a grid of short text boxes; connect the named entities "
                        "with explicit '->' arrow relationships or a non-boxy diagram"
                    )
                quality_issues.extend(_phase16_legibility_issues(row, workspace=args.workspace))
                quality_issues.extend(_phase16_coverage_issues(row))
                geometry_errors, geometry_warnings = _phase16_geometry_issues(row["asset_path"])
                quality_issues.extend(geometry_errors)
                advisory = [*geometry_warnings, *_phase16_advisory_issues(row["asset_path"])]
                if advisory:
                    # Estimate-based layout findings inform the NEXT attempt's
                    # feedback when something else fails, but never block an
                    # otherwise-passing figure (the approved canary fails these
                    # estimators on 11/24 of its figures).
                    review = {
                        **review,
                        "findings": [
                            *review["findings"],
                            {
                                "check_id": "layout_estimate_advisory",
                                "severity": "warn",
                                "message": "; ".join(advisory[:4]),
                            },
                        ],
                    }
                if quality_issues:
                    execution_status = "fail"
                    review = {
                        **review,
                        "status": "fail",
                        "findings": [
                            *review["findings"],
                            {
                                "check_id": "visual_legibility",
                                "severity": "error",
                                "message": (
                                    "Rendered figure fails the visual legibility gate: "
                                    + "; ".join(quality_issues[:8])
                                    + ". Redraw it: use at most ~26 concise labels, each a COMPLETE "
                                    "phrase (never end a label on a connector word like 'and', 'to', "
                                    "'of', 'the', 'at'); font-size >= 16; attribute each named person "
                                    "only to their own source label; never show raw source ids or "
                                    "scaffold text; connect entities with '->' arrow relationships."
                                ),
                            },
                        ],
                    }
            if not execution_error:
                # This attempt rendered a valid SVG (even if a soft legibility check
                # failed). Keep the LIGHTEST renderable attempt (fewest/least severe
                # residual defects) so the repair loop's oscillation cannot leave
                # a worse figure than an earlier attempt produced.
                weight = _phase16_residual_weight(quality_issues)
                if best_effort is None or weight <= best_effort["weight"]:
                    best_effort = {
                        "code": code,
                        "prompt": prompt,
                        "response": response_text,
                        "review": review,
                        "weight": weight,
                        "issues": list(quality_issues),
                    }
        attempt_record = {
            "attempt": attempt,
            "prompt_sha256": prompt_sha,
            "response_sha256": response_sha,
            "code_sha256": _sha256_text(code),
            "provider_receipt_path": provider_receipt_path,
            "code_review_status": review["status"],
            "execution_status": execution_status,
            "findings": review["findings"],
        }
        attempt_records.append(attempt_record)
        model_calls.append(
            {
                "slide_id": row["slide_id"],
                "visual_id": row["visual_id"],
                "attempt": attempt,
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "provider_receipt_path": provider_receipt_path,
            }
        )
        if review["status"] == "pass" and execution_status == "pass":
            accepted_code = code
            accepted_prompt = prompt
            accepted_response = response_text
            accepted_review = review
            break
        latest = _phase16_attempt_feedback(review=review, execution_error=execution_error)
        feedback = _phase16_cumulative_feedback(
            latest=latest, history=defect_history, attempt=attempt, initial_feedback=initial_feedback
        )
        if latest:
            defect_history.append(f"attempt {attempt}: {latest}")

    if (not accepted_code or accepted_review is None) and best_effort is not None:
        # No attempt fully passed the soft legibility gate, but at least one rendered
        # a valid SVG. Accept the best-effort render (re-executing to restore its SVG)
        # so the deck completes; the post-hoc semantic review still records the gap.
        accepted_code = best_effort["code"]
        accepted_prompt = best_effort["prompt"]
        accepted_response = best_effort["response"]
        accepted_review = best_effort["review"]
        row["code_path"].write_text(accepted_code.rstrip() + "\n", encoding="utf-8")
        _execute_phase16_generated_code(row["code_path"], row["asset_path"])
        residual_weight = int(best_effort.get("weight") or 0)
        max_weight = int(getattr(args, "best_effort_max_weight", PHASE16_BEST_EFFORT_MAX_WEIGHT))
        blocking_content_gap = _has_non_downgradable_issue(best_effort.get("issues") or [])
        if residual_weight <= max_weight and not blocking_content_gap:
            # Light residue: record it as warnings so the receipt passes and the
            # pixel-level review decides; heavy residue stays a blocking error.
            accepted_review = _phase16_downgrade_residual(accepted_review, best_effort.get("issues") or [])
        print(
            f"[phase16] {row['slide_id']} {row['visual_id']}: accepted best-effort render "
            f"after exhausting repair attempts (residual weight {residual_weight}, "
            f"{'blocking: required label missing' if blocking_content_gap else ('accepted' if residual_weight <= max_weight else 'blocking')})",
            file=sys.stderr,
        )

    if not accepted_code or accepted_review is None:
        raise RuntimeError(
            f"Opus SVG code generation failed for {row['slide_id']} {row['visual_id']}: {feedback}"
        )

    return _phase16_accepted_result(
        row,
        accepted_review=accepted_review,
        attempt_records=attempt_records,
        model_calls=model_calls,
        prompt_sha=_sha256_text(accepted_prompt),
        response_sha=_sha256_text(accepted_response),
        reused=False,
    )


def _phase16_downgrade_residual(review: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    """Turn a best-effort attempt's blocking findings into residual warnings."""

    findings = []
    for finding in review.get("findings", []):
        if isinstance(finding, dict) and finding.get("severity") == "error":
            findings.append({**finding, "severity": "warn", "check_id": f"residual_{finding.get('check_id')}"})
        else:
            findings.append(finding)
    return {
        **review,
        "status": "pass_with_residual",
        "findings": findings,
        "residual_issues": list(issues),
    }


def _phase16_accepted_result(
    row: dict[str, Any],
    *,
    accepted_review: dict[str, Any],
    attempt_records: list[dict[str, Any]],
    model_calls: list[dict[str, Any]],
    prompt_sha: str,
    response_sha: str,
    reused: bool,
) -> dict[str, Any]:
    svg_sha = _sha256_file(row["asset_path"])
    code_sha = _sha256_file(row["code_path"])
    figure_spec = {
        "slide_id": row["slide_id"],
        "section_id": row.get("section_id"),
        "cluster_ids": row.get("cluster_ids", []),
        "visual_id": row["visual_id"],
        "title": row["title"],
        "claim_to_visualize": row["claim"],
        "proof_object": row["proof_object"],
        "depiction_type": row["depiction_type"],
        # The drawing form the figure was required to take, and why it was
        # chosen, so a reviewer can audit the deck's visual variety from the
        # spec instead of eyeballing 25 rendered slides.
        "archetype": row.get("archetype") or "",
        "archetype_reason": row.get("archetype_reason") or "",
        "visual_concept": row["visual_concept"],
        "composition_budget": row["composition_budget"],
        "figure_design_brief": _phase16_figure_design_brief(row),
        "generation_route": "llm_authored_code",
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "provider": PHASE16_REQUIRED_PROVIDER,
        "model": PHASE16_REQUIRED_MODEL,
        "renderer_family": "drawsvg",
        "asset_path": row["relative_asset"],
        "code_path": row["relative_code"],
        "code_sha256": code_sha,
        "svg_sha256": svg_sha,
        "prompt_sha256": prompt_sha,
        "response_sha256": response_sha,
        "required_entities": row["required_entities"],
        "required_relationships": row["required_relationships"],
        "source_labels": row["source_labels"],
        "source_ids": row["source_ids"],
        "reference_ids": row["reference_ids"],
        "people": row["people"],
        "systems": row["systems"],
        "datasets": row["datasets"],
        "contribution_note": row["contribution_note"],
        "narration_hook": row["narration_hook"],
        "production_evidence": True,
    }
    visual_plan_figure = {
        **row["original_figure"],
        "asset_paths": [row["relative_asset"]],
        "generation_route": "llm_authored_code",
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "provider": PHASE16_REQUIRED_PROVIDER,
        "model": PHASE16_REQUIRED_MODEL,
        "renderer_family": "drawsvg",
        "depiction_type": row["depiction_type"],
        "visual_concept": row["visual_concept"],
        "composition_budget": row["composition_budget"],
        "figure_design_brief": _phase16_figure_design_brief(row),
        "required_entities": row["required_entities"],
        "required_relationships": row["required_relationships"],
        "source_labels": row["source_labels"],
        "source_ids": row["source_ids"],
        "reference_ids": row["reference_ids"],
        "code_path": row["relative_code"],
        "code_sha256": code_sha,
        "svg_sha256": svg_sha,
        "prompt_sha256": prompt_sha,
        "response_sha256": response_sha,
        "production_evidence": True,
    }
    asset = {
        "path": row["relative_asset"],
        "slide_id": row["slide_id"],
        "section_id": row.get("section_id"),
        "cluster_ids": row.get("cluster_ids", []),
        "visual_id": row["visual_id"],
        "renderer_family": "drawsvg",
        "depiction_type": row["depiction_type"],
        "visual_concept": row["visual_concept"],
        "composition_budget": row["composition_budget"],
        "generation_route": "llm_authored_code",
        "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
        "source_labels": row["source_labels"],
        "source_ids": row["source_ids"],
        "reference_ids": row["reference_ids"],
        "people": row["people"],
        "systems": row["systems"],
        "datasets": row["datasets"],
        "sha256": svg_sha,
        "code_path": row["relative_code"],
        "code_sha256": code_sha,
        "prompt_sha256": prompt_sha,
        "response_sha256": response_sha,
        "production_evidence": True,
        "semantic_review_ref": "figure_semantic_review.json",
        "model_governance": {
            "provider": PHASE16_REQUIRED_PROVIDER,
            "model_id": PHASE16_REQUIRED_MODEL,
            "model_route": PHASE16_REQUIRED_MODEL_ROUTE,
            "status": "live_opus_4_8_generated_code",
        },
    }
    code_review = {
        "slide_id": row["slide_id"],
        "visual_id": row["visual_id"],
        "status": accepted_review["status"],
        "code_path": row["relative_code"],
        "code_sha256": code_sha,
        "renderer_family": "drawsvg",
        "attempts": attempt_records,
        "findings": accepted_review["findings"],
        "reused": reused,
    }
    if reused:
        asset["reused"] = True
        figure_spec["reused"] = True
    return {
        "figure_spec": figure_spec,
        "visual_plan_figure": visual_plan_figure,
        "asset": asset,
        "code_reviews": [code_review],
        "model_calls": model_calls,
        "reused": reused,
        "prompt_sha256": prompt_sha,
        "response_sha256": response_sha,
    }


def _phase16_cache_path(args: argparse.Namespace) -> Path:
    return args.generated_dir / "figure_acceptance_cache.json"


def _phase16_load_cache(args: argparse.Namespace) -> dict[str, Any]:
    path = _phase16_cache_path(args)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _phase16_reuse_candidate(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    research_text: str,
    cache: dict[str, Any],
    qc_verdicts: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Reuse a previously-accepted figure when its prompt (spec + rules +
    research) is unchanged, the SVG/code on disk match the cached hashes, and
    the figure still passes EVERY current check. A rerun then only spends model
    calls on the figures that actually need repair."""

    entry = cache.get(str(row["visual_id"]))
    prompt = _phase16_codegen_prompt(row=row, research_text=research_text, feedback="")
    prompt_sha = _sha256_text(prompt)
    code_path: Path = row["code_path"]
    asset_path: Path = row["asset_path"]
    if not code_path.is_file() or not asset_path.is_file():
        return None
    # A figure the PIXEL JUDGE passed, byte-identical on disk, is reused
    # unconditionally — the vision verdict outranks the deterministic
    # estimators, which would otherwise churn previously-approved figures.
    verdict = (qc_verdicts or {}).get(str(row["slide_id"])) or {}
    svg_sha_now = _sha256_file(asset_path)
    if (
        verdict.get("status") == "pass"
        and verdict.get("svg_sha256") == svg_sha_now
        and isinstance(entry, dict)
        and entry.get("prompt_sha256") == prompt_sha
    ):
        code = code_path.read_text(encoding="utf-8", errors="replace")
        review = _review_phase16_generated_code(code, row=row)
        if review["status"] != "pass":
            # Requirements changed since the pixel judge approved these bytes
            # (e.g. SRC codes are now forbidden) — the pin does not survive.
            review = None
    else:
        review = None
    if review is not None:
        return _phase16_accepted_result(
            row,
            accepted_review=review,
            attempt_records=[
                {
                    "attempt": 0,
                    "reused": True,
                    "accepted_by": "visual_qc_verdict",
                    "prompt_sha256": prompt_sha,
                    "response_sha256": "",
                    "code_sha256": _sha256_file(code_path),
                    "provider_receipt_path": None,
                    "code_review_status": "pass",
                    "execution_status": "reused",
                    "findings": review["findings"],
                }
            ],
            model_calls=[],
            prompt_sha=prompt_sha,
            response_sha="",
            reused=True,
        )
    if isinstance(entry, dict):
        if entry.get("prompt_sha256") != prompt_sha:
            return None
        if (
            _sha256_file(code_path) != entry.get("code_sha256")
            or svg_sha_now != entry.get("svg_sha256")
        ):
            return None
    # Without a cache entry the figure may still be reusable: its code/SVG on
    # disk came from a receipted generation run, and acceptance is decided by
    # the same full check suite either way.
    code = code_path.read_text(encoding="utf-8", errors="replace")
    review = _review_phase16_generated_code(code, row=row)
    if review["status"] != "pass":
        return None
    if _phase16_svg_is_bullet_box(asset_path):
        return None
    if _phase16_legibility_issues(row, workspace=args.workspace):
        return None
    attempt_records = [
        {
            "attempt": 0,
            "reused": True,
            "prompt_sha256": prompt_sha,
            "response_sha256": str((entry or {}).get("response_sha256") or ""),
            "code_sha256": _sha256_file(code_path),
            "provider_receipt_path": None,
            "code_review_status": review["status"],
            "execution_status": "reused",
            "findings": review["findings"],
        }
    ]
    return _phase16_accepted_result(
        row,
        accepted_review=review,
        attempt_records=attempt_records,
        model_calls=[],
        prompt_sha=prompt_sha,
        response_sha=str((entry or {}).get("response_sha256") or ""),
        reused=True,
    )


def _phase16_codegen_prompt(*, row: dict[str, Any], research_text: str, feedback: str) -> str:
    payload = {
        "slide_id": row["slide_id"],
        "section_id": row.get("section_id"),
        "visual_id": row["visual_id"],
        "title_short": _short_visible_term(row["title"]),
        "claim_short": _short_visible_term(row["claim"]),
        "proof_object_short": _short_visible_term(row["proof_object"]),
        "visual_concept_short": _short_visible_term(row["visual_concept"]),
        "figure_brief": shorten(" ".join(str(row.get("visual_concept") or "").split()), width=460, placeholder=""),
        "depiction_type": row.get("depiction_type") or "",
        "archetype": row.get("archetype") or "",
        "archetype_instructions": row.get("archetype_instructions") or "",
        "composition_budget": row.get("composition_budget") or {},
        "required_entities": row["required_entities"],
        "required_relationships": row["required_relationships"],
        "source_labels": row["source_labels"],
        "source_chip_bindings": row.get("source_chip_bindings") or [],
        "people": row["people"],
        "systems": row["systems"],
        "datasets": row["datasets"],
        "contribution_note": row["contribution_note"],
        "forbidden_patterns": row["forbidden_patterns"],
    }
    research_excerpt = shorten(" ".join(research_text.split()), width=1800, placeholder="")
    repair_note = f"\nPrevious attempt feedback:\n{feedback}\n" if feedback else ""
    if _attribution_mode() == "in_figure":
        attribution_rules = (
            "- NEVER print SRC-style codes (SRC-001 etc.) or raw provenance identifiers (no 'paper:', "
            "'transcript:', 'segment:', 'paper_abstract:', 'speech_transcript:', and no alphanumeric id "
            "strings) anywhere in the figure. Attribute scholarship DIRECTLY: when a box names a work, put "
            "the work's short name as the box title and its lead author(s) beneath it, exactly as given in "
            "source_chip_bindings ('Work — Authors'). Never place one work's authors next to a different work."
        )
    else:
        payload.pop("source_chip_bindings", None)
        payload.pop("people", None)
        attribution_rules = (
            "- ATTRIBUTION IS NOT PART OF THE FIGURE. The slide already shows source chips with each work's "
            "title, authors, and links. Do NOT draw citation cards, reference boxes, author lines, 'Work — "
            "Authors' labels, footers, or any person's name anywhere in the figure. NEVER print SRC-style "
            "codes (SRC-001 etc.) or raw provenance identifiers (no 'paper:', 'transcript:', 'segment:', "
            "'paper_abstract:', 'speech_transcript:', and no alphanumeric id strings). A work's short name "
            "may appear ONLY as the title of a node that represents that work/system. Spend the freed space "
            "on larger nodes, wider spacing, and clearer connectors."
        )
    return f"""Write one safe Python file that generates a 1280x720 editorial SVG proof visual.

Use only:
- import drawsvg as draw
- optional import math

Hard requirements:
- Return Python code only, no markdown fences.
- Keep the file concise: under 140 lines when possible.
- Define exactly: build_svg(output_path: str) -> None
- Use variable name d for the draw.Drawing object.
- The final statement inside build_svg must be exactly: d.save_svg(output_path)
- Use drawsvg primitives and call d.save_svg(output_path).
- Do not read files, write any path except output_path, use network, use shell, inspect environment variables, or import os/sys/subprocess/pathlib/requests/urllib.
- Use a source-grounded mechanism/evidence/map composition, not a decorative image, copied bullet list, source-ID table, or generic card.
- Include visible SVG text for every required_entities item exactly as provided.
- Include visible SVG text for every required_relationships item exactly as provided.
- visual_concept_short and figure_brief are the layout contract, not decoration. Follow them before choosing shapes.
- composition_budget is binding: use one dominant flow direction, no more than max_entity_nodes main nodes, no more than max_relationship_edges connector families, and no more than max_visible_text_labels visible labels. Group related entities into aligned lanes or clusters instead of drawing a spaghetti map.
- Prefer the dominant_flow_direction from composition_budget. For pipelines and mechanism diagrams, default to clean left-to-right flow with orthogonal connector lanes and no crossing connectors. For comparisons, use two or three aligned columns. For evaluation or tradeoff visuals, use a single chart-like focal area plus side annotations.
- Keep the bottom 70px of the canvas (y > 650) COMPLETELY EMPTY — no citation rows, no summary lines, no footer of any kind; all explanatory text sits above y=650 with clear spacing.
{attribution_rules}
- Never place two text elements at the same coordinates; every label gets its own clear position.
- Use only concise string literals from title_short, visual_concept_short, required_entities, required_relationships, systems, datasets, and (when provided) source_labels, source_chip_bindings, and people.
- Do not display internal scaffold labels such as source evidence, visual mechanism, audience claim, evidence maps to deployment action, source contribution, visual proof object, viewer-facing claim, named study, slide claim, mechanism view, control layer, deployment decision, deployment takeaway, or source record. Use the ACTUAL named works and concepts from required_entities/required_relationships as node and edge labels — never a generic three-box 'source record -> mechanism view -> deployment takeaway' template.
- Do not copy long claims, proof paragraphs, or contribution notes into Python string literals.
- Keep every Python string literal under 100 characters; split visible text into list items instead of multi-line quoted strings.
- Use plain ASCII quotes in code. Do not create unterminated or multi-line string literals.
- The rendered slide DOWNSCALES this 1280x720 SVG to about 73% (a 930px panel), so fonts must be LARGE: every text element >= 20px, node/box titles >= 26px, the caption >= 22px, footer SRC labels >= 20px. Never use any font size below 18px anywhere. Avoid black (#000/#000000). Use only these colors: #1a1a2e, #a8d8ea, #ff6b6b, #51cf66, #ffd43b, #9775fa, #4ecdc4, #ffffff, #e0e0e0, #cccccc, #333344, #2a2a4a, #16213e.
- Use at most 26 visible <text> labels total. Substantive, evidence-dense figures are REQUIRED — depict the figure_brief's components, data points, and edges as a real mechanism/comparison/architecture diagram, never a thin 3-box summary. Organize density with disciplined layout (grid alignment, generous box sizing) rather than by dropping content.
- figure_brief describes exactly what to draw (depiction type, components, quantitative data, edges); follow it.
- archetype is the REQUIRED visual form for this figure (archetype_instructions says how to draw it). It was chosen so the deck is not a run of identical box-and-arrow diagrams: a timeline, layered stack, before/after split, evidence chart, tradeoff quadrant, comparison matrix, or ladder must look like that form, not like a flow chart with a different title.
- Every visible text label must be a COMPLETE phrase. Never end a label on a connector word (and, or, the, a, an, to, of, in, on, for, with, at, is, are, that, this, by, as) and never end a label with a hyphen. Do not split one sentence across stacked labels; keep each label short and self-contained.
- Connect the named entities with explicit relationships drawn as arrows. You may render each required_relationships string as ONE caption-style line in clear space (for example directly under the heading or above the footer) instead of writing label text on the arrows themselves — on-edge text that touches a connector line is a defect.
- STACK multiple relationship caption lines vertically (one per row, ~30px apart) as ONE block placed directly under the figure heading at the top, in guaranteed-empty space — never beside or under boxes; two captions side by side on the same row WILL overlap and is a defect.
- Anchor every connector ON the border of its source and target shapes — never at a shape's center, and never start, end, or route a line underneath or through ANY text (including the shape's own label). Compute the box edge coordinates explicitly. Draw arrowheads ONLY with a drawsvg Marker whose TIP is at the marker origin: head = draw.Marker(-1.0, -0.5, 0.1, 0.5, scale=8, orient='auto'); head.append(draw.Lines(-1.0, 0.5, -1.0, -0.5, 0.0, 0.0, close=True, fill=color)); then draw.Line(..., marker_end=head). With the tip at (0,0) the head trails BEHIND the endpoint, so a line that ends on the border keeps its whole head outside the box. End the line 4-6px SHORT of the target border so the arrowhead tip has visible clearance instead of sitting flush against the outline. Never define the triangle with its tip at positive x (that pushes the head into the box), never hand-draw arrowhead polygons, and never place any part of a head inside a box or over a label.
- When a phrase wraps onto two lines, break it so no line ends on a connector word (and, or, of, the, to, for, with, in); prefer breaking after a noun, or shorten the phrase instead of wrapping.
- Attribute each named person only next to their own source label; do not place a person's name beside a different source's label.
- Do NOT repeat the slide title / claim text inside the figure; the slide already shows it as the heading and sidebar. Depict the mechanism, evidence, or relationship instead.
- Draw arrows restrained: thin connectors (stroke-width 2-3). Arrowheads must be TINY — at most ~10px on a side. Never draw large solid/filled triangles, oversized polygon arrow caps, or big chevrons; arrowheads must never touch, overlap, or visually dominate the node boxes or their text.
- Never draw a connector line through or under text. Place each edge label fully ABOVE its connector with at least 8px of clear gap between the text baseline and the line; if there is no room, move the label beside the line instead.
- Connector aesthetics are blocking: arrowheads may not intrude into node text, connectors may not terminate at box centers, and a flow chart with crossing lines or ambiguous direction is unacceptable even if all labels are technically present.
- Size every shape FROM its text with this exact formula: box_width = max(len(label) for label in box_labels) * font_size * 0.62 + 28; box_height = line_count * font_size * 1.5 + 20. Compute these as Python variables BEFORE drawing; never hardcode a width narrower than the formula. Never let text extend past the shape border. Prefer rounded boxes over circles for multi-word labels.
- Draw ALL boxes first, then route connectors between box borders along clear lanes; the arrowhead ends at the target border, never over text. If a label would overlap a line, move the label, not the line.
- Leave at least 16px of clear space between any two node boxes; boxes that touch or sit a few pixels apart read as one fused shape.
- A connector may only touch the two boxes it joins. Never route a line across, over, or along the outline of a third box, and never let two connectors share an endpoint or cross each other - move one into its own lane.
- Leave clear padding around every label; never let text overlap another label, a node box edge, or the bottom source-chip footer row.
- Never show internal pipeline or scaffold labels (for example 'Phase 16', 'phase16', 'render_spec', 'cluster_id', file paths, or stage names) anywhere in the figure, including the footer.
- Keep all text inside the canvas.
- Prefer drawsvg because the Phase 16 stack research selects direct SVG primitives for editorial proof objects.

Phase 16 stack research excerpt:
{research_excerpt}

Figure contract JSON:
{json.dumps(payload, indent=2, sort_keys=True)}
{repair_note}"""


def _review_phase16_generated_code(code: str, *, row: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    if not code.strip():
        findings.append(_code_finding("empty_code", "error", "generated response did not contain code"))
        return {"status": "fail", "findings": findings}
    lower = code.lower()
    blocked_fragments = (
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "http.client",
        "os.",
        "sys.",
        "pathlib",
        "shutil",
        "open(",
        "read_text",
        "write_text",
        ".env",
        "/users/",
        "../",
        "__import__",
        "eval(",
        "exec(",
        "compile(",
        "globals(",
        "locals(",
        "getattr(",
        "setattr(",
        "delattr(",
        "matplotlib",
        "plotly",
        "#000",
    )
    for fragment in blocked_fragments:
        if fragment in lower:
            findings.append(
                _code_finding(
                    "blocked_code_fragment",
                    "error",
                    f"blocked fragment found in generated code: {fragment}",
                )
            )
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        findings.append(_code_finding("syntax_error", "error", f"{exc.msg} at line {exc.lineno}"))
        return {"status": "fail", "findings": findings}
    allowed_imports = {"drawsvg", "math"}
    top_level_allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef)
    for node in tree.body:
        if not isinstance(node, top_level_allowed):
            findings.append(
                _code_finding(
                    "top_level_side_effect",
                    "error",
                    f"top-level {type(node).__name__} is not allowed",
                )
            )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module not in allowed_imports:
                    findings.append(
                        _code_finding("blocked_import", "error", f"import {alias.name} is not allowed")
                    )
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module not in allowed_imports:
                findings.append(
                    _code_finding("blocked_import", "error", f"from {node.module} import ... is not allowed")
                )
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {
                "open",
                "eval",
                "exec",
                "compile",
                "__import__",
                "input",
                "getattr",
                "setattr",
                "delattr",
                "globals",
                "locals",
            }:
                findings.append(_code_finding("blocked_call", "error", f"{name}() is not allowed"))
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "remove", "unlink", "rmdir"}:
            findings.append(_code_finding("blocked_attribute", "error", f".{node.attr} is not allowed"))
    build_svg = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_svg"),
        None,
    )
    if build_svg is None:
        findings.append(_code_finding("missing_build_svg", "error", "build_svg(output_path) is required"))
    if "save_svg" not in code and "return " not in code:
        findings.append(
            _code_finding(
                "missing_svg_output",
                "error",
                "drawsvg output must call save_svg or return a drawsvg Drawing",
            )
        )
    if "drawsvg" not in code and "draw." not in code:
        findings.append(_code_finding("missing_drawsvg", "error", "generated code must use drawsvg"))
    lower_code = code.lower()
    for scaffold in (
        "source evidence",
        "visual mechanism",
        "audience claim",
        "evidence maps to deployment action",
        "source contribution",
        "visual proof object",
        "viewer-facing claim",
        "named study",
        "slide claim",
        "mechanism view",
        "control layer",
        "deployment decision",
        "deployment takeaway",
        "source record",
    ):
        if scaffold in lower_code:
            findings.append(
                _code_finding(
                    "internal_scaffold_visible",
                    "error",
                    f"generated code includes internal scaffold text: {scaffold}",
                )
            )
    for raw_id_marker in (
        "paper_abstract",
        "speech_transcript",
        "transcript:",
        "paper:",
        "segment:",
        "seg0",
        "src-0",
        "SRC-0",
    ):
        if raw_id_marker in lower_code:
            findings.append(
                _code_finding(
                    "raw_source_id_visible",
                    "error",
                    f"generated code includes raw provenance identifier text: {raw_id_marker}",
                )
            )
    for term in [*row["required_entities"], *row["required_relationships"]]:
        if term and not _term_present_in_code(str(term), code):
            findings.append(
                _code_finding("missing_source_bound_term", "error", f"code omits required visible term: {term}")
            )
    return {"status": "fail" if any(f["severity"] == "error" for f in findings) else "pass", "findings": findings}


def _execute_phase16_generated_code(code_path: Path, svg_path: Path) -> str:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    if svg_path.exists():
        svg_path.unlink()
    try:
        module_name = f"phase16_{_slug(code_path.stem)}"
        spec = importlib.util.spec_from_file_location(module_name, code_path)
        if spec is None or spec.loader is None:
            return "could not load generated code module"
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        build_svg = getattr(module, "build_svg", None)
        if not callable(build_svg):
            return "build_svg is missing or not callable"
        result = build_svg(str(svg_path))
        if not svg_path.exists() and hasattr(result, "save_svg"):
            result.save_svg(str(svg_path))
    except Exception as exc:  # noqa: BLE001 - generated-code review must capture exact failure class.
        return f"{type(exc).__name__}: {exc}"
    if not svg_path.exists():
        return "build_svg did not create the expected SVG file"
    if svg_path.stat().st_size < 500:
        return "generated SVG is too small to be a production proof visual"
    return ""


def _anthropic_text(result: Any) -> str:
    response = getattr(result, "response", result)
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return str(response)
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _extract_python_code(response_text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    code = fenced.group(1) if fenced else response_text
    code = code.strip()
    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1].strip()
    return code


def _normalize_phase16_generated_code(code: str) -> str:
    """Apply bounded mechanical repairs that preserve Opus-authored drawing code."""

    if "save_svg" in code or "return " in code:
        return code
    match = re.search(r"(?m)^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*draw\.Drawing\(", code)
    if not match:
        return code
    indent = match.group(1)
    if not indent:
        return code
    variable = match.group(2)
    return code.rstrip() + f"\n{indent}{variable}.save_svg(output_path)\n"


_ATTRIBUTION_MODE: dict[str, str] = {"mode": PHASE16_FIGURE_ATTRIBUTION_MODE}


def _attribution_mode() -> str:
    return _ATTRIBUTION_MODE.get("mode") or PHASE16_FIGURE_ATTRIBUTION_MODE


def set_attribution_mode(mode: str) -> None:
    _ATTRIBUTION_MODE["mode"] = mode if mode in ("chips", "in_figure") else PHASE16_FIGURE_ATTRIBUTION_MODE


def _phase16_required_entities(
    *,
    figure: dict[str, Any],
    brief: dict[str, Any],
    refs: list[dict[str, Any]],
) -> list[str]:
    terms: list[str] = []
    if _attribution_mode() == "in_figure":
        terms.extend(
            person
            for person in _string_list(brief.get("people"))[:3]
            if person.strip().lower() not in _PLACEHOLDER_ENTITY_NAMES
        )
    # Editorial figure briefs put their concrete components into the figure's
    # required_entities at pass-1; when present, they ARE the figure content.
    brief_entities = [
        e
        for e in _string_list(figure.get("required_entities"))
        if e.strip().lower() not in _PLACEHOLDER_ENTITY_NAMES
        and not _is_internal_scaffold_text(e)
    ]
    if len(brief_entities) >= 4:
        return _unique([_short_visible_term(e) for e in brief_entities], limit=8)
    terms.extend(_string_list(brief.get("systems"))[:2])
    terms.extend(_string_list(brief.get("datasets"))[:1])
    if len(terms) < 4:
        for ref in refs[:2]:
            # Short work names ("Hybrid Reinforcement"), not full 60+ char titles
            # that wrap into orphaned text fragments inside the figure.
            work = _concise_phrase(
                str(ref.get("resolved_title") or ref.get("title") or ""), max_words=3
            )
            if work:
                terms.append(work)
    claim_text = str(brief.get("claim_to_visualize") or "").lower()
    if len(terms) < 4:
        terms.extend(
            entity
            for entity in _string_list(figure.get("required_entities"))[:3]
            if entity.strip().lower() not in _PLACEHOLDER_ENTITY_NAMES
            # Single words lifted from the slide title produce word-chain
            # diagrams; only multi-word entities or true proper nouns qualify.
            and not (len(entity.split()) == 1 and entity.strip().lower() in claim_text)
        )
    proof = _concise_phrase(str(brief.get("proof_object") or ""), max_words=6)
    if proof:
        terms.append(proof)
    cleaned = [_short_visible_term(term) for term in terms]
    return _unique(cleaned, limit=6) or ["Evidence", "Mechanism"]


def _phase16_required_relationships(*, figure: dict[str, Any]) -> list[str]:
    relationships = [
        _short_visible_term(item)
        for item in _string_list(figure.get("required_relationships"))
        if item.strip() and not _is_internal_scaffold_text(item)
    ]
    if relationships:
        return _unique(relationships, limit=3)
    # No real relationship survived: build one from the figure's named entities and
    # claim rather than emitting generic scaffold ("named study -> ...").
    concept = _relationship_concept(
        str(figure.get("claim_to_visualize") or figure.get("claim") or figure.get("title") or "")
    )
    entities = [
        e
        for e in _string_list(figure.get("required_entities"))
        if e.strip() and not _is_internal_scaffold_text(e)
    ]
    if entities:
        anchor = _concise_phrase(entities[0], max_words=3)
        if anchor and concept:
            relationships.append(f"{anchor} -> {concept}")
    if not relationships:
        relationships.append(f"named evidence -> {concept}")
    return _unique(relationships, limit=3)


def _phase16_visual_concept(
    *,
    figure: dict[str, Any],
    brief: dict[str, Any],
    deck_slide: dict[str, Any],
    title: str,
    visual_role: str,
    depiction_type: str,
    entities: list[str],
    relationships: list[str],
) -> str:
    for candidate in (
        brief.get("visual_concept"),
        figure.get("visual_concept"),
        visual_role,
        deck_slide.get("visual_concept"),
        deck_slide.get("visual_role"),
    ):
        concept = _clean_visual_concept(candidate)
        if concept:
            return concept
    return _source_grounded_visual_concept(
        title=title,
        visual_role=visual_role,
        depiction_type=depiction_type,
        entities=entities,
        relationships=relationships,
    )


def _source_grounded_visual_concept(
    *,
    title: str,
    visual_role: str,
    depiction_type: str,
    entities: list[str],
    relationships: list[str],
) -> str:
    concept = _clean_visual_concept(visual_role)
    if concept:
        return concept
    anchor = entities[0] if entities else _relationship_concept(title)
    target = entities[1] if len(entities) > 1 else _relationship_concept(title)
    edge = relationships[0] if relationships else f"{anchor} -> {target}"
    if depiction_type == "comparison":
        return _short_visible_term(
            f"Side-by-side comparison showing {anchor} and {target}, with {edge} as the takeaway"
        )
    if depiction_type in {"benchmark", "evidence_map"}:
        return _short_visible_term(
            f"Evidence map with one focal result area and aligned source callouts for {anchor}"
        )
    return _short_visible_term(
        f"Left-to-right mechanism flow from {anchor} to {target}, with {edge} as the main connector"
    )


def _clean_visual_concept(value: object) -> str:
    text = " ".join(str(value or "").replace("\n", " ").split())
    if not text or _is_internal_scaffold_text(text):
        return ""
    if text.strip().lower() in {"none", "story visual", "source-grounded mechanism visual"}:
        return ""
    return shorten(text, width=220, placeholder="")


def _composition_budget(
    *,
    entities: list[str],
    relationships: list[str],
    visual_concept: str,
) -> dict[str, Any]:
    max_nodes = min(PHASE16_MAX_ENTITY_NODES, max(1, len(entities)))
    max_edges = min(PHASE16_MAX_RELATIONSHIP_EDGES, max(1, len(relationships)))
    return {
        # Heading + caption block + (title, subtitle) per node + one caption per
        # edge + two source cards: a 3-node figure does not need 26 labels, and
        # unused label budget is where the model invents crowding.
        "max_visible_text_labels": min(
            PHASE16_MAX_VISIBLE_TEXT_LABELS,
            (8 if _attribution_mode() == "in_figure" else 4) + 3 * max_nodes + 2 * max_edges,
        ),
        "max_entity_nodes": max_nodes,
        "max_relationship_edges": max_edges,
        "dominant_flow_direction": _dominant_flow_direction(visual_concept),
        "connector_policy": "orthogonal edge-to-edge lanes; no connectors through text; tiny arrowheads",
        "density_policy": "group related evidence into aligned lanes rather than adding extra nodes",
    }


def _dominant_flow_direction(visual_concept: str) -> str:
    lower = visual_concept.lower()
    if any(term in lower for term in ("stack", "layer", "timeline", "schedule")):
        return "top_to_bottom"
    if any(term in lower for term in ("side-by-side", "comparison", "matrix", "columns")):
        return "left_to_right_columns"
    return "left_to_right"


def _phase16_figure_design_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "visual_role": row.get("visual_role") or row["visual_concept"],
        "depiction_type": row["depiction_type"],
        "archetype": row.get("archetype") or "",
        "claim_to_visualize": row["title"],
        "visual_concept": row["visual_concept"],
        "composition_budget": row["composition_budget"],
        "proof_object": row["proof_object"],
        "required_entities": row["required_entities"],
        "required_relationships": row["required_relationships"],
        "source_labels": row["source_labels"],
        "source_ids": row["source_ids"],
        "contribution_note": row["contribution_note"],
        "narration_hook": row["narration_hook"],
        "forbidden_patterns": row["forbidden_patterns"],
        "people": row["people"],
        "systems": row["systems"],
        "datasets": row["datasets"],
    }


def _is_internal_scaffold_text(value: object) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    return any(
        scaffold in text
        for scaffold in (
            "source evidence",
            "visual mechanism",
            "audience claim",
            "evidence maps to deployment action",
            "source contribution",
            "visual proof object",
            "viewer-facing claim",
            "named study",
            "slide claim",
            "mechanism view",
            "control layer",
            "deployment decision",
            "deployment takeaway",
            "source record",
        )
    )


def _short_visible_term(value: str) -> str:
    compact = " ".join(str(value).replace("\n", " ").split())
    if len(compact) <= 74:
        return compact
    return shorten(compact, width=74, placeholder="")


def _phase16_cumulative_feedback(
    *, latest: str, history: list[str], attempt: int, initial_feedback: str = ""
) -> str:
    """Repair feedback that remembers every earlier attempt.

    With only the latest defects in the prompt the model fixed one thing per
    attempt and re-introduced another (a label overflow on attempt 1, an arrow
    through text on attempt 2, the overflow again on attempt 3 ...) and three
    ICML 2026 figures never converged in five attempts. Carrying the full
    defect history forward, with the standing constraints stated explicitly,
    turns the loop into a monotone repair."""

    parts: list[str] = []
    if initial_feedback:
        parts.append(f"Standing feedback from the pixel-level review:\n{initial_feedback}")
    if history:
        parts.append(
            "Defects found in EARLIER attempts (each was fixed once and must NOT come back):\n"
            + "\n".join(f"- {item}" for item in history[-4:])
        )
    if latest:
        parts.append(
            f"Defects in the LATEST attempt ({attempt}) that you must fix now, keeping every "
            f"earlier fix in place:\n{latest}"
        )
    parts.append(
        "Before drawing, compute every box width from its longest label with the given "
        "formula and every arrow endpoint from the target box's border; verify each "
        "label fits with >=6px clearance and no connector passes through any label."
    )
    return "\n\n".join(part for part in parts if part).strip()


def _phase16_attempt_feedback(*, review: dict[str, Any], execution_error: str) -> str:
    parts = [
        str(finding.get("message") or finding.get("check_id") or "")
        for finding in review.get("findings", [])
        if isinstance(finding, dict)
    ]
    if execution_error:
        parts.append(execution_error)
    return "\n".join(part for part in parts if part).strip()


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _term_present_in_code(term: str, code: str) -> bool:
    normalized = term.strip().lower()
    lower_code = code.lower()
    if normalized in lower_code:
        return True
    code_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", lower_code))
    term_tokens = {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", normalized)
        if token not in {"source", "visual", "figure"}
    }
    if bool(term_tokens) and term_tokens.issubset(code_tokens):
        return True
    # Long multi-word entities may be legitimately shortened by the model when
    # repairing layout; a clear majority of the distinctive tokens suffices.
    if len(term_tokens) >= 3:
        present = sum(1 for token in term_tokens if token in code_tokens)
        return present / len(term_tokens) >= 0.6
    return False


def _code_finding(check_id: str, severity: str, message: str) -> dict[str, str]:
    return {"check_id": check_id, "severity": severity, "message": message}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _relative_to_deck(path: Path, deck_dir: Path) -> str:
    try:
        return path.resolve().relative_to(deck_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _source_refs(visible_refs: object) -> list[dict[str, Any]]:
    if not isinstance(visible_refs, list):
        return []
    refs: list[dict[str, Any]] = []
    for ref in visible_refs:
        if not isinstance(ref, dict):
            continue
        label = str(ref.get("label") or ref.get("title") or "").strip()
        source_id = str(ref.get("source_id") or ref.get("url") or "").strip()
        contribution = str(ref.get("contribution_note") or ref.get("relevance_rationale") or "").strip()
        if label and source_id and contribution:
            refs.append(ref)
    return refs


def _source_chip_bindings(refs: list[dict[str, Any]]) -> list[str]:
    """SRC-free attribution lines ('Work — Lead Authors') so the figure names
    scholarship directly; viewers never see opaque SRC codes."""

    bindings: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        work = str(ref.get("resolved_title") or ref.get("title") or "").strip()
        if not work or work.lower() in seen:
            continue
        seen.add(work.lower())
        people = [
            str(p).strip()
            for p in (ref.get("speakers") or []) + (ref.get("authors") or [])
            if str(p).strip() and str(p).strip().lower() not in _PLACEHOLDER_ENTITY_NAMES
        ]
        attribution = ", ".join(people[:3]) + (" et al." if len(people) > 3 else "")
        line = _short_visible_term(work)
        if attribution:
            line += f" — {attribution}"
        bindings.append(line)
    return bindings


_PLACEHOLDER_ENTITY_NAMES = frozenset(
    {"unknown", "n/a", "na", "none", "anonymous", "unnamed", "unidentified", ""}
)


def _parse_brief_field(visual_role: str, field: str) -> list[str]:
    """Parse 'components'/'data'/'edges' lists from an editorial visual_role
    string of the form 'depiction: summary | components: a; b | edges: A -> B'."""

    match = re.search(rf"\|\s*{field}:\s*([^|]+)", visual_role)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(";") if item.strip()]


def _required_entities(*, title: str, visual_role: str, source_refs: list[dict[str, Any]]) -> list[str]:
    # Editorial blueprints encode an evidence-grounded figure brief into the
    # visual role; its concrete components are the figure's content.
    components = _parse_brief_field(visual_role, "components")
    if len(components) >= 4:
        return _unique(components, limit=8)
    # Entities must be REAL depictable things — named people and named works —
    # never the slide title split into words (that produces content-free
    # "word-chain" diagrams) and never another topic's hardcoded keyword labels.
    entities: list[str] = list(components)
    for ref in source_refs:
        for person in [*(_string_list(ref.get("speakers"))), *(_string_list(ref.get("authors")))]:
            if person.strip().lower() in _PLACEHOLDER_ENTITY_NAMES:
                continue  # unresolved speaker/author placeholders can't be depicted
            entities.append(person)
    for ref in source_refs[:3]:
        work = _concise_phrase(
            str(ref.get("resolved_title") or ref.get("title") or ""), max_words=3
        )
        if work:
            entities.append(work)
    concept = _relationship_concept(title)
    if concept and concept.lower() not in {e.lower() for e in entities}:
        entities.append(concept)
    return _unique(entities, limit=5) or [_relationship_concept(title), "Evidence"]


_FIGURE_TRAILING_TRIM = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
        "at", "is", "are", "was", "were", "be", "been", "being", "that", "this",
        "by", "as", "from", "into", "over", "than", "but", "nor", "via", "per",
        "when", "while", "where", "which", "who", "its", "their", "our",
        "favored", "favors", "favor", "becomes", "become", "becoming", "remains",
    }
)


def _concise_phrase(text: str, max_words: int = 4) -> str:
    """A short, complete phrase: first ``max_words`` words with trailing
    connector/verb words trimmed so figure node/edge labels never dangle.
    Work titles cut at the colon ("Sculpting Subspaces: constrained full..."
    -> "Sculpting Subspaces") since the head is the natural short name."""
    compact = _short_visible_term(text).replace("Implications:", "")
    head = compact.split(":", 1)[0].strip()
    if len(head.split()) >= 2:
        compact = head
    words = [w for w in compact.split() if w]
    words = words[:max_words]
    while words and words[-1].lower().strip(".,;:'\"()/-") in _FIGURE_TRAILING_TRIM:
        words.pop()
    return " ".join(words).strip(" -:,")


def _relationship_concept(title: str) -> str:
    """A short noun-phrase concept from the slide title for figure edge labels."""
    return _concise_phrase(title, max_words=4) or "the claim"


def _required_relationships(
    *,
    title: str,
    visual_role: str,
    source_refs: list[dict[str, Any]],
) -> list[str]:
    # Editorial briefs carry explicit evidence edges; prefer them.
    edges = [e for e in _parse_brief_field(visual_role, "edges") if "->" in e]
    if len(edges) >= 2:
        return _unique(edges, limit=4)
    # Build REAL relationships that connect the named work(s) to the slide's claim,
    # using actual source names — not generic scaffold ("named study -> slide claim
    # -> mechanism view") and not another topic's hardcoded chains.
    concept = _relationship_concept(title)
    relationships: list[str] = []
    for ref in source_refs[:2]:
        work = str(ref.get("resolved_title") or ref.get("title") or "").strip()
        work = _concise_phrase(work, max_words=3)
        if work and concept and work.lower() != concept.lower():
            relationships.append(f"{work} -> {concept}")
    if not relationships:
        relationships.append(f"named evidence -> {concept}")
    return _unique(relationships, limit=3) or [f"named evidence -> {concept}"]


def _contribution_note(source_refs: list[dict[str, Any]]) -> str:
    notes: list[str] = []
    for ref in source_refs[:3]:
        people = _string_list(ref.get("speakers")) or _string_list(ref.get("authors"))
        speaker_prefix = f"{', '.join(people)}: " if people else ""
        note = str(ref.get("contribution_note") or ref.get("relevance_rationale") or "").strip()
        # Strip a leading raw "SRC-NNN" label from the note: the label belongs in the
        # source chip, not as visible figure text (reads as raw source metadata).
        note = re.sub(r"^SRC-\d+\s*", "", note).strip()
        if note:
            notes.append(f"{speaker_prefix}{note}".strip())
    return " | ".join(notes) or "Visible sources ground the visual claim."


def _narration_hook(base_hook: str, *, entities: list[str], relationships: list[str]) -> str:
    entity = entities[0] if entities else "source entity"
    relationship = relationships[0] if relationships else "source relationship"
    return f"{base_hook} Depict {entity} through {relationship}."


def _depiction_type(*, title: str, visual_role: str, visual_concept: str = "") -> str:
    """Depiction type for a figure.

    The visual plan already commits to a depiction vocabulary in the
    ``visual_concept`` prefix (``evidence_chart:``, ``tradeoff_map:``,
    ``comparison_matrix:``, ``architecture:``, ``mechanism_pipeline:``). Honour
    it first; the title/role keyword fallback otherwise labels almost every
    figure ``system_workflow`` and the deck becomes a run of box-and-arrow
    diagrams (the reviewer's "I wonder if you can show something more than
    block diagrams")."""

    refined = _depiction_from_concept(visual_concept)
    if refined:
        return refined
    lower = f"{title} {visual_role}".lower()
    if "mechanism" in lower or "system-boundary" in lower or "boundary" in lower:
        return "mechanism"
    if "matrix" in lower or "comparison" in lower:
        return "comparison"
    if "benchmark" in lower or "evaluation" in lower:
        return "benchmark"
    if "evidence map" in lower or "source" in lower:
        return "evidence_map"
    return "system_workflow"


_CONCEPT_PREFIX_TO_DEPICTION: dict[str, str] = {
    "evidence_chart": "evidence_chart",
    "chart": "evidence_chart",
    "benchmark": "benchmark",
    "tradeoff_map": "tradeoff_map",
    "quadrant": "tradeoff_map",
    "comparison_matrix": "comparison",
    "comparison": "comparison",
    "matrix": "comparison",
    "architecture": "architecture",
    "layered_stack": "architecture",
    "timeline": "timeline",
    "mechanism_pipeline": "mechanism",
    "mechanism": "mechanism",
    "pipeline": "system_workflow",
    "system_workflow": "system_workflow",
    "failure_mode": "failure_mode",
    "evidence_map": "evidence_map",
}

# Box-and-arrow families: a deck made only of these reads as one diagram
# repeated; the archetype rotation below breaks the run without changing the
# evidence the figure must carry.
BOX_AND_ARROW_DEPICTIONS = frozenset({"mechanism", "system_workflow", "architecture"})
MAX_CONSECUTIVE_BOX_AND_ARROW = 2

ARCHETYPE_INSTRUCTIONS: dict[str, str] = {
    "flow": (
        "FLOW: a left-to-right (or top-to-bottom) sequence of rounded boxes joined by "
        "orthogonal arrows; one lane per actor when there are several actors."
    ),
    "layered_stack": (
        "LAYERED STACK: horizontal bands stacked top-to-bottom (one band per layer/level), "
        "full-width, with the interface between bands labelled in the gap; no arrows between "
        "bands except one vertical control/feedback arrow at the side."
    ),
    "timeline": (
        "TIMELINE: one horizontal axis with ordered milestones/steps as ticks; each step gets a "
        "short label above and (where the brief gives one) an outcome/metric below; a shaded band "
        "marks the phase where the failure or gain appears."
    ),
    "before_after": (
        "BEFORE/AFTER SPLIT: two side-by-side panels with the same skeleton, left = without the "
        "method/failure case, right = with the method/fixed case; mark exactly what differs with "
        "a coloured highlight and a one-line caption under each panel."
    ),
    "evidence_chart": (
        "EVIDENCE CHART: draw real axes (drawsvg lines), bars or plotted traces for the numbers the "
        "brief gives, tick labels, a legend; annotate the single most important gap/delta; never "
        "fake ticks or bars without values."
    ),
    "tradeoff_map": (
        "TRADEOFF MAP: a 2x2 quadrant with named axes; place each named system/method as a dot "
        "with a short label in the quadrant the evidence supports; caption the winning region."
    ),
    "comparison_matrix": (
        "COMPARISON MATRIX: rows = methods/systems, columns = criteria; cells carry a check, cross, "
        "or short value; highlight the row the brief favours."
    ),
    "ladder": (
        "LADDER / STAIRCASE: ordered levels climbing left-to-right as steps, one level per rung, "
        "with the capability gained written on each step and the cited work placed at the step it "
        "reached."
    ),
    "failure_mode": (
        "FAILURE MODE: the normal path drawn faintly, the failing branch drawn boldly in the alert "
        "colour, with the exact point of divergence circled and the consequence stated at the end."
    ),
    "evidence_map": (
        "EVIDENCE MAP: one focal claim in the centre, the supporting works around it as labelled "
        "cards, each linked by a short connector annotated with what that work contributes."
    ),
}

_DEFAULT_ARCHETYPE_BY_DEPICTION: dict[str, str] = {
    "mechanism": "flow",
    "system_workflow": "flow",
    "architecture": "layered_stack",
    "evidence_chart": "evidence_chart",
    "benchmark": "evidence_chart",
    "tradeoff_map": "tradeoff_map",
    "comparison": "comparison_matrix",
    "timeline": "timeline",
    "failure_mode": "failure_mode",
    "evidence_map": "evidence_map",
}


def _depiction_from_concept(visual_concept: str) -> str:
    head = str(visual_concept or "").split(":", 1)[0].strip().lower().replace(" ", "_")
    if not head or len(head) > 24:
        return ""
    return _CONCEPT_PREFIX_TO_DEPICTION.get(head, "")


def _alternate_archetype(*, title: str, visual_concept: str, depiction_type: str) -> str:
    """A non-flow archetype that still fits the brief, chosen from its wording."""

    lower = f"{title} {visual_concept}".lower()
    if any(term in lower for term in ("step", "phase", "over training", "schedule", "episode", "iteration", "horizon")):
        return "timeline"
    if any(term in lower for term in ("fail", "attack", "collapse", "drift", "leak", "poison", "conceal", "deceiv")):
        return "before_after"
    if any(term in lower for term in ("layer", "stack", "level", "tier", "hierarch")):
        return "layered_stack"
    if any(term in lower for term in ("trade", "cost", "versus", " vs ", "budget")):
        return "tradeoff_map"
    if any(term in lower for term in ("scal", "progress", "capabilit", "maturity", "from ", "toward")):
        return "ladder"
    return "before_after" if depiction_type == "mechanism" else "layered_stack"


def assign_figure_archetypes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give every figure an ``archetype`` and cap runs of box-and-arrow diagrams.

    Rows are in deck order. Each row keeps its depiction's default archetype
    unless it would be the third box-and-arrow figure in a row, in which case a
    brief-compatible alternative is chosen and recorded (``archetype_reason``)."""

    streak = 0
    for row in rows:
        depiction = str(row.get("depiction_type") or "system_workflow")
        archetype = _DEFAULT_ARCHETYPE_BY_DEPICTION.get(depiction, "flow")
        reason = "default for depiction_type"
        if depiction in BOX_AND_ARROW_DEPICTIONS:
            if streak >= MAX_CONSECUTIVE_BOX_AND_ARROW:
                archetype = _alternate_archetype(
                    title=str(row.get("title") or ""),
                    visual_concept=str(row.get("visual_concept") or ""),
                    depiction_type=depiction,
                )
                reason = (
                    f"the previous {streak} figures were box-and-arrow diagrams; rotated to keep "
                    "the deck visually varied"
                )
                streak = 0
            else:
                streak += 1
        else:
            streak = 0
        row["archetype"] = archetype
        row["archetype_reason"] = reason
        row["archetype_instructions"] = ARCHETYPE_INSTRUCTIONS.get(archetype, ARCHETYPE_INSTRUCTIONS["flow"])
    return rows


def _forbidden_patterns() -> list[str]:
    return [
        "decorative",
        "bullet-box",
        "fake chart ticks",
        "copied bullet text in boxes",
        "generic diagram template",
    ]


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _unique(values: list[str], *, limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        item = " ".join(str(value).split())
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_attribution_mode(getattr(args, "figure_attribution", PHASE16_FIGURE_ATTRIBUTION_MODE))
    if args.topic != args.presentation_id:
        print(
            "--topic must match --presentation-id for topic-deep-dive figure generation",
            file=sys.stderr,
        )
        return 2

    if args.deck_contract is not None:
        if _phase16_llm_route_requested(args):
            payload = generate_phase16_llm_visual_plan(args)
        else:
            payload = generate_structured_visual_plan(args)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    inventory = generate_programmatic_figures(
        args.workspace,
        args.presentation_id,
        input_slides=args.input_slides,
        source_pack=args.source_pack,
        ranked_papers=args.ranked_papers,
        asset_inventory=args.asset_inventory,
        output_slides=args.output_slides,
        generated_dir=args.generated_dir,
        repair_feedback=args.repair_feedback,
        figure_specs_input=args.figure_specs_input,
        figure_specs_output=args.figure_specs_output,
        target_svgs_per_slide=args.target_svgs_per_slide,
        allow_density_fallback=args.allow_density_fallback,
        live_sonnet_specs=args.live_sonnet_specs,
        sonnet_batch_size=args.sonnet_batch_size,
        sonnet_timeout_seconds=args.sonnet_timeout_seconds,
        max_waivers=args.max_waivers,
        dry_run=args.dry_run,
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))
    return 1 if inventory["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
