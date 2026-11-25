#!/usr/bin/env python3
"""Generate render_spec.json from structured presentation contracts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.render_spec import validate_render_spec_payload
from conference_pipeline.workflow_runs import write_json_atomic
from scripts.presentations.md_to_revealjs import relationalai_theme_contract


PHASE16_SOURCE_CONTRACT_HASH_MAP = {
    "brand_lock_sha256": "01-BRAND-REFERENCE-LOCK.json",
    "source_board_sha256": "03-SOURCE-EDITORIAL-BOARD.json",
    "material_clusters_sha256": "03-MATERIAL-CLUSTERS.json",
    "section_spine_sha256": "03-SECTION-SPINE.json",
    "storyboard_rows_sha256": "03-STORYBOARD-ROWS.json",
    "visual_direction_sha256": "03-VISUAL-DIRECTION.md",
    "storyboard_review_sha256": "03-STORYBOARD-REVIEW.json",
    "cross_ai_synthesis_sha256": "04-CREATIVE-STORYBOARD-CROSS-AI-SYNTHESIS.json",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_render_spec(
    *,
    deck_contract: dict[str, Any],
    source_binding: dict[str, Any],
    visual_plan: dict[str, Any],
    asset_inventory: dict[str, Any],
    figure_specs: dict[str, Any] | None = None,
    delivery_single_speaker: dict[str, Any],
    delivery_notebooklm: dict[str, Any],
    input_hashes: dict[str, str],
    delivery_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    input_hashes = _merge_phase16_source_contract_hashes(input_hashes, source_binding)
    _assert_contract_alignment(
        deck_contract=deck_contract,
        source_binding=source_binding,
        delivery_single_speaker=delivery_single_speaker,
        delivery_notebooklm=delivery_notebooklm,
    )
    source_by_slide = {
        str(slide.get("slide_id")): slide
        for slide in source_binding.get("slides", [])
        if isinstance(slide, dict)
    }
    visual_by_slide = {
        str(slide.get("slide_id")): slide
        for slide in visual_plan.get("slides", [])
        if isinstance(slide, dict)
    }
    assets_by_slide = {
        str(asset.get("slide_id")): asset
        for asset in asset_inventory.get("generated_assets", [])
        if isinstance(asset, dict)
    }
    figure_specs_by_slide = _figure_specs_by_slide(figure_specs or {})
    single_timing_by_slide = _delivery_timing_by_slide(delivery_single_speaker)
    slides: list[dict[str, Any]] = []
    reinjected_bookends = bookend_slide_ids(deck_contract)
    for slide in deck_contract.get("slides", []):
        if not isinstance(slide, dict):
            continue
        if str(slide.get("slide_id") or "") in reinjected_bookends:
            continue  # add_bookend_slides.py re-injects these after this run
        slide_id = str(slide.get("slide_id") or f"slide-{len(slides) + 1:03d}")
        source_row = source_by_slide.get(slide_id, {})
        visual_row = visual_by_slide.get(slide_id, {})
        figures = visual_row.get("figures") if isinstance(visual_row, dict) else []
        visual_refs = []
        figure_spec_row = figure_specs_by_slide.get(slide_id, {})
        if isinstance(figure_spec_row, dict) and figure_spec_row.get("asset_path"):
            visual_refs.append(str(figure_spec_row["asset_path"]))
        if isinstance(figures, list):
            for figure in figures:
                if isinstance(figure, dict):
                    for path in figure.get("asset_paths", []):
                        if path and str(path) not in visual_refs:
                            visual_refs.append(str(path))
        if not visual_refs:
            asset = assets_by_slide.get(slide_id, {})
            if isinstance(asset, dict) and asset.get("path"):
                visual_refs.append(str(asset["path"]))
        if not visual_refs:
            raise ValueError(
                f"{slide_id} is missing visual_asset_refs from visual_plan.json and asset_inventory.json"
            )
        timing_seconds = single_timing_by_slide.get(slide_id)
        if timing_seconds is None:
            timing_seconds = float((slide.get("timing_beats") or [{"seconds": 30}])[0].get("seconds", 30))
        slides.append(
            {
                "slide_id": slide_id,
                "state_id": f"{slide_id}-state",
                "layout_blocks": [
                    {
                        "kind": "heading",
                        "text": _compact_text(
                            slide.get("display_title")
                            or slide.get("title")
                            or _first_string(slide.get("visible_text"))
                            or slide_id,
                            limit=72,
                        ),
                    }
                ],
                "source_chips": source_row.get("visible_references") or [],
                "visual_asset_refs": visual_refs,
                "evidence_caption": _compact_text(
                    slide.get("evidence_caption")
                    or _first_string(slide.get("visible_text"))
                    or slide.get("proof_object")
                    or "Source-grounded visual evidence.",
                    limit=88,
                ),
                "visual_takeaway": _compact_text(
                    slide.get("visual_takeaway")
                    or slide.get("viewer_facing_claim")
                    or slide.get("title")
                    or "The figure anchors the claim in cited evidence.",
                    limit=96,
                ),
                "figure_focus": str(slide.get("figure_focus") or slide.get("visual_role") or ""),
                "split_required": bool(slide.get("split_required")),
                "speaker_notes": " ".join(slide.get("narration_hooks", []))
                or "Discuss the source-grounded visual evidence.",
                "timing_seconds": float(timing_seconds),
            }
        )
    delivery_paths = delivery_paths or {
        "single_speaker": "delivery_single_speaker.json",
        "notebooklm": "delivery_notebooklm.json",
    }
    payload = {
        "schema_version": 1,
        "presentation_id": str(deck_contract.get("presentation_id") or "presentation"),
        "input_hashes": input_hashes,
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": str(deck_contract.get("analysis_report_url") or "https://drive.google.com/"),
        "theme_contract": relationalai_theme_contract(),
        "slides": slides,
        "delivery": {
            "single_speaker": {
                "contract_path": delivery_paths["single_speaker"],
                "total_duration_seconds": delivery_single_speaker.get("total_duration_seconds"),
            },
            "notebooklm": {
                "contract_path": delivery_paths["notebooklm"],
                "total_duration_seconds": delivery_notebooklm.get("total_duration_seconds"),
            },
        },
    }
    result = validate_render_spec_payload(payload)
    if result.errors:
        raise ValueError(json.dumps(result.to_dict(), indent=2))
    return payload


def _assert_contract_alignment(
    *,
    deck_contract: dict[str, Any],
    source_binding: dict[str, Any],
    delivery_single_speaker: dict[str, Any],
    delivery_notebooklm: dict[str, Any],
) -> None:
    bookends = bookend_slide_ids(deck_contract)
    deck_slide_ids = _slide_ids(deck_contract.get("slides"), bookends)
    if not deck_slide_ids:
        raise ValueError("presentation_deck.json must contain slides before render_spec generation")
    approved_expansion = deck_contract.get("approved_expansion")
    if len(deck_slide_ids) > 14:
        if not isinstance(approved_expansion, dict) or approved_expansion.get("approved") is not True:
            raise ValueError("render_spec generation blocked: decks above 14 slides require approved_expansion")
        # add_bookend_slides.py raises target_slide_count by its two structural
        # slides and records the pre-bookend editorial count as
        # base_slide_count; that editorial count is what the excluded-id list
        # above leaves behind, so the two always compare like with like.
        expected_slide_count = (
            approved_expansion.get("base_slide_count")
            if approved_expansion.get("bookend_slides_added")
            else approved_expansion.get("target_slide_count")
        )
        if expected_slide_count != len(deck_slide_ids):
            raise ValueError(
                "render_spec generation blocked: approved_expansion target_slide_count drift "
                f"(approved {expected_slide_count}, editorial slides {len(deck_slide_ids)})"
            )
    source_slide_ids = _slide_ids(source_binding.get("slides"), bookends)
    if source_slide_ids != deck_slide_ids:
        only_deck = [slide_id for slide_id in deck_slide_ids if slide_id not in source_slide_ids]
        only_binding = [slide_id for slide_id in source_slide_ids if slide_id not in deck_slide_ids]
        raise ValueError(
            "render_spec generation blocked: source_binding slide IDs drift from presentation_deck"
            f" (deck only: {only_deck[:4]}; binding only: {only_binding[:4]};"
            f" deck {len(deck_slide_ids)} vs binding {len(source_slide_ids)})"
        )
    for label, delivery in (
        ("delivery_single_speaker.json", delivery_single_speaker),
        ("delivery_notebooklm.json", delivery_notebooklm),
    ):
        bookend_ids = bookends
        segment_slide_ids = [
            str(segment.get("slide_id") or "")
            for segment in delivery.get("segments", [])
            if isinstance(segment, dict) and str(segment.get("slide_id") or "") not in bookend_ids
        ]
        if segment_slide_ids != deck_slide_ids:
            raise ValueError(f"render_spec generation blocked: {label} segment slide IDs drift from deck")


# ``add_bookend_slides.py`` injects exactly these two structural slides after
# render-spec generation and re-derives them on every run, so they are never
# part of a deck/binding/delivery id comparison or of the approved slide count.
# Other decks (the executive-overview family) tag EVERY slide with a
# ``slide_kind`` such as "content" or "title"; those are editorial slides and
# must not be filtered out here.
BOOKEND_SLIDE_KINDS = frozenset({"events", "resources"})


def is_bookend_slide(slide: object) -> bool:
    return (
        isinstance(slide, dict)
        and str(slide.get("slide_kind") or "").strip().lower() in BOOKEND_SLIDE_KINDS
    )


def bookend_slide_ids(deck_contract: dict[str, Any]) -> set[str]:
    """Ids of bookend slides that were injected AFTER render-spec generation.

    Two deck families carry bookends and they need opposite treatment:

    * Topic decks get theirs from ``add_bookend_slides.py``, which runs after
      this generator and re-derives them on every run. It records
      ``approved_expansion.bookend_slides_added``. On a re-run those slides are
      already in the deck contract but absent from every other contract, so
      they must be excluded from the comparison and re-injected afterwards.
    * The executive-overview decks author their bookends up front, as ordinary
      editorial slides present in every contract. Excluding them would drop two
      real slides from the render spec.

    So the marker decides, not the ``slide_kind`` field alone. Ids come from
    the deck because it is the only contract that labels these slides at all.
    """

    approved = deck_contract.get("approved_expansion")
    if not isinstance(approved, dict) or not approved.get("bookend_slides_added"):
        return set()
    slides = deck_contract.get("slides")
    if not isinstance(slides, list):
        return set()
    return {
        str(slide.get("slide_id") or "")
        for slide in slides
        if is_bookend_slide(slide) and str(slide.get("slide_id") or "").strip()
    }


def _slide_ids(value: object, exclude: set[str] | None = None) -> list[str]:
    """Editorial slide ids only: the structural bookend slides are never part
    of the deck/binding/delivery id comparison or the approved slide count."""

    excluded = exclude or set()
    if not isinstance(value, list):
        return []
    return [
        str(item.get("slide_id") or "")
        for item in value
        if isinstance(item, dict)
        and str(item.get("slide_id") or "").strip()
        and str(item.get("slide_id") or "") not in excluded
    ]


def _first_string(value: object) -> str:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    if isinstance(value, str):
        return value.strip()
    return ""


_COMPLETE_SENTENCE_HARD_CEILING = 400


def _compact_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    # A caption that is already one complete, correctly-punctuated sentence
    # (optionally quote-closed) reads far better slightly over the nominal
    # limit than mid-word truncated into a fragment with a fake trailing
    # period. See the matching comment in structured_video._compact_text.
    if len(text) <= _COMPLETE_SENTENCE_HARD_CEILING and re.search(r"[.!?][\"'’”]?$", text):
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() + "."


def _merge_phase16_source_contract_hashes(
    input_hashes: dict[str, str],
    source_binding: dict[str, Any],
) -> dict[str, str]:
    merged = dict(input_hashes)
    contract = source_binding.get("phase16_source_contract")
    if not isinstance(contract, dict):
        return merged
    for source_key, input_name in PHASE16_SOURCE_CONTRACT_HASH_MAP.items():
        value = contract.get(source_key)
        if isinstance(value, str) and len(value) == 64:
            merged[input_name] = value
    return merged


def _delivery_timing_by_slide(delivery_contract: dict[str, Any]) -> dict[str, float]:
    timings: dict[str, float] = {}
    segments = delivery_contract.get("segments")
    if not isinstance(segments, list):
        return timings
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        slide_id = segment.get("slide_id")
        seconds = segment.get("duration_seconds")
        if isinstance(slide_id, str) and isinstance(seconds, int | float) and seconds > 0:
            timings[slide_id] = float(seconds)
    return timings


def _figure_specs_by_slide(figure_specs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = figure_specs.get("figure_specs")
    if not isinstance(rows, list):
        return {}
    by_slide: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slide_id = str(row.get("slide_id") or "").strip()
        asset_path = str(row.get("asset_path") or "").strip()
        if slide_id and asset_path:
            by_slide[slide_id] = row
    return by_slide


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--presentation-id")
    parser.add_argument("--presentation-type")
    parser.add_argument("--topic")
    parser.add_argument("--presentation-blueprint", type=Path, help="Hash-bound presentation_blueprint.json input.")
    parser.add_argument("--deck-contract", type=Path, required=True)
    parser.add_argument("--source-binding-report", type=Path, required=True)
    parser.add_argument("--visual-plan", type=Path, required=True)
    parser.add_argument("--asset-inventory", type=Path, required=True)
    parser.add_argument("--figure-specs", type=Path, help="Hash-bound Phase 16 figure_specs.json input.")
    parser.add_argument(
        "--figure-generation-model-receipt",
        type=Path,
        help="Hash-bound Opus 4.8 model receipt for generated figure code.",
    )
    parser.add_argument(
        "--figure-code-review",
        type=Path,
        help="Hash-bound generated Python figure-code review report.",
    )
    parser.add_argument(
        "--figure-svg-validation",
        type=Path,
        help="Hash-bound structural SVG validation report.",
    )
    parser.add_argument(
        "--figure-semantic-review",
        type=Path,
        help="Hash-bound semantic figure review report.",
    )
    parser.add_argument("--delivery-single-speaker", type=Path, help="Hash-bound delivery_single_speaker.json input.")
    parser.add_argument("--delivery-notebooklm", type=Path, help="Hash-bound delivery_notebooklm.json input.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    figure_inputs = {
        "--figure-specs": args.figure_specs,
        "--figure-generation-model-receipt": args.figure_generation_model_receipt,
        "--figure-code-review": args.figure_code_review,
        "--figure-svg-validation": args.figure_svg_validation,
        "--figure-semantic-review": args.figure_semantic_review,
    }
    if any(figure_inputs.values()):
        missing = [flag for flag, value in figure_inputs.items() if value is None]
        if missing:
            parser.error("Phase 16 figure-backed render spec requires " + ", ".join(missing))
    if args.workspace is not None:
        workspace = args.workspace.expanduser().resolve()
        args.workspace = workspace
        for attr in (
            "presentation_blueprint",
            "deck_contract",
            "source_binding_report",
            "visual_plan",
            "asset_inventory",
            "figure_specs",
            "figure_generation_model_receipt",
            "figure_code_review",
            "figure_svg_validation",
            "figure_semantic_review",
            "delivery_single_speaker",
            "delivery_notebooklm",
            "output",
        ):
            setattr(args, attr, _resolve_workspace_arg(workspace, getattr(args, attr, None)))
    return args


def _resolve_workspace_arg(workspace: Path, value: Path | None) -> Path | None:
    if value is None or value.is_absolute():
        return value
    if value.exists():
        return value
    return workspace / value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.deck_contract.parent
    presentation_blueprint = args.presentation_blueprint or input_dir / "presentation_blueprint.json"
    delivery_single_speaker = args.delivery_single_speaker or input_dir / "delivery_single_speaker.json"
    delivery_notebooklm = args.delivery_notebooklm or input_dir / "delivery_notebooklm.json"
    input_paths = {
        "presentation_blueprint.json": presentation_blueprint,
        "presentation_deck.json": args.deck_contract,
        "source_binding_report.json": args.source_binding_report,
        "visual_plan.json": args.visual_plan,
        "asset_inventory.json": args.asset_inventory,
        "delivery_single_speaker.json": delivery_single_speaker,
        "delivery_notebooklm.json": delivery_notebooklm,
    }
    figure_review_paths = {
        "figure_specs.json": args.figure_specs,
        "figure_generation_model_receipt.json": args.figure_generation_model_receipt,
        "figure_code_review_report.json": args.figure_code_review,
        "figure_svg_validation_report.json": args.figure_svg_validation,
        "figure_semantic_review.json": args.figure_semantic_review,
    }
    if all(figure_review_paths.values()):
        _assert_figure_reviews_pass(
            model_receipt=_load_json(args.figure_generation_model_receipt),
            code_review=_load_json(args.figure_code_review),
            svg_validation=_load_json(args.figure_svg_validation),
            semantic_review=_load_json(args.figure_semantic_review),
        )
        input_paths.update({name: path for name, path in figure_review_paths.items() if path is not None})
    payload = build_render_spec(
        deck_contract=_load_json(args.deck_contract),
        source_binding=_load_json(args.source_binding_report),
        visual_plan=_load_json(args.visual_plan),
        asset_inventory=_load_json(args.asset_inventory),
        figure_specs=_load_json(args.figure_specs) if args.figure_specs is not None else None,
        delivery_single_speaker=_load_json(delivery_single_speaker),
        delivery_notebooklm=_load_json(delivery_notebooklm),
        input_hashes={name: sha256_file(path) for name, path in input_paths.items()},
        delivery_paths={
            "single_speaker": str(delivery_single_speaker.name),
            "notebooklm": str(delivery_notebooklm.name),
        },
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    write_json_atomic(args.output, payload)
    return 0


def _assert_figure_reviews_pass(
    *,
    model_receipt: dict[str, Any],
    code_review: dict[str, Any],
    svg_validation: dict[str, Any],
    semantic_review: dict[str, Any],
) -> None:
    failures: list[str] = []
    if model_receipt.get("status") != "pass":
        failures.append("figure_generation_model_receipt.json status is not pass")
    if code_review.get("status") != "pass":
        failures.append("figure_code_review_report.json status is not pass")
    if code_review.get("render_spec_generation_allowed") is False:
        failures.append("figure_code_review_report.json blocks render spec generation")
    svg_errors = None
    summary = svg_validation.get("summary")
    if isinstance(summary, dict):
        svg_errors = summary.get("error")
    if svg_validation.get("status") not in {None, "pass"}:
        failures.append("figure_svg_validation_report.json status is not pass")
    if svg_errors not in {None, 0}:
        failures.append("figure_svg_validation_report.json has structural SVG errors")
    if semantic_review.get("status") != "pass":
        failures.append("figure_semantic_review.json status is not pass")
    semantic_gate = semantic_review.get("semantic_review")
    if not isinstance(semantic_gate, dict) or semantic_gate.get("render_spec_generation_allowed") is not True:
        failures.append("figure_semantic_review.json does not allow render spec generation")
    if failures:
        raise ValueError("Phase 16 figure review gates failed: " + "; ".join(failures))


if __name__ == "__main__":
    raise SystemExit(main())
