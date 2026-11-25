#!/usr/bin/env python3
"""Build a structured editorial storyboard for a presentation deck.

The storyboard is a pre-TTS planning artifact. It records which slides are
editorially distinct, which slides are obvious merge candidates, how figures
should function in the narrative, and which figures deserve separate zoom/detail
beats. It is intentionally deterministic so it can run before paid generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from presentation_editorial_gate import parse_slides  # noqa: E402


DEFAULT_TARGET_DURATION_SECONDS = 38 * 60
DEFAULT_MAX_DURATION_SECONDS = 40 * 60
DEFAULT_MAX_ZOOM_COUNT = 16

ROLE_KEYWORDS = (
    ("architecture", ("architecture", "system", "module", "pipeline")),
    ("benchmark", ("benchmark", "metric", "evaluation", "score")),
    ("comparison", ("comparison", "contrast", "tradeoff", "matrix")),
    ("evidence_map", ("evidence", "map", "attribution", "trace")),
    ("experiment", ("experiment", "validation", "wet-lab", "ablation")),
    ("failure_mode", ("failure", "attack", "risk", "vulnerability", "misuse")),
    ("mechanism", ("mechanism", "workflow", "process", "flow", "loop")),
    ("molecule_structure", ("molecule", "protein", "ligand", "structure")),
)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _spec_slide_number(row: dict[str, Any]) -> int | None:
    for source in (row.get("figure_spec"), row):
        if not isinstance(source, dict):
            continue
        try:
            return int(source.get("slide_number"))
        except (TypeError, ValueError):
            continue
    return None


def _figure_specs_by_slide(asset_inventory: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(asset_inventory)
    rows: list[Any] = []
    if isinstance(payload, dict):
        rows = payload.get("figure_specs") or payload.get("figures") or []
    elif isinstance(payload, list):
        rows = payload
    specs: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        spec = row.get("figure_spec") if isinstance(row.get("figure_spec"), dict) else row
        if not isinstance(spec, dict):
            continue
        slide_number = _spec_slide_number(row)
        if slide_number is not None:
            specs[slide_number] = spec
    return specs


def _paper_refs(slide: dict[str, Any], spec: dict[str, Any] | None) -> list[str]:
    refs = [_clean(ref) for ref in slide.get("refs", []) if _clean(ref)]
    if refs:
        return refs
    if spec:
        spec_refs = spec.get("paper_or_idea_refs")
        if isinstance(spec_refs, list):
            return [_clean(ref) for ref in spec_refs if _clean(ref)]
        source_trace = spec.get("source_trace")
        if isinstance(source_trace, list):
            titles = []
            for row in source_trace:
                if isinstance(row, dict) and _clean(row.get("title")):
                    titles.append(_clean(row.get("title")))
            if titles:
                return titles[:3]
    return []


def _cluster_key(slide: dict[str, Any], spec: dict[str, Any] | None) -> str:
    refs = _paper_refs(slide, spec)
    if refs:
        return "refs:" + " | ".join(sorted(ref.lower() for ref in refs[:3]))
    title = _clean(slide.get("title", "")).lower()
    return "title:" + " ".join(re.findall(r"[a-z0-9]+", title)[:5])


def _figure_role(slide: dict[str, Any], spec: dict[str, Any] | None) -> str:
    source = " ".join(
        [
            _clean(slide.get("figure_need")),
            _clean(spec.get("depiction_type") if spec else ""),
            _clean(spec.get("semantic_goal") if spec else ""),
        ]
    ).lower()
    for role, terms in ROLE_KEYWORDS:
        if any(term in source for term in terms):
            return role
    return "mechanism" if slide.get("images") else "story_context"


def _narration_beat(slide: dict[str, Any], spec: dict[str, Any] | None, role: str) -> str:
    if spec:
        relationships = spec.get("relationships")
        if isinstance(relationships, list) and relationships:
            return f"Integrate the figure by naming this relationship: {_clean(relationships[0])}"
        semantic_goal = _clean(spec.get("semantic_goal"))
        if semantic_goal:
            return f"Use the figure to make this claim concrete: {semantic_goal}"
    bullets = slide.get("bullets") or []
    if bullets:
        return f"Use the visual as a {role.replace('_', ' ')} for: {_clean(bullets[0])}"
    return f"Use the visual as a {role.replace('_', ' ')} anchor for the slide claim."


def _representative_claim(slides: list[dict[str, Any]]) -> str:
    for slide in slides:
        bullets = slide.get("bullets") or []
        if bullets:
            return _clean(bullets[0])
    return _clean(slides[0].get("title", "Editorial cluster"))


def build_storyboard(
    *,
    deck_dir: Path,
    slides_md: Path | None = None,
    asset_inventory: Path | None = None,
    target_duration_seconds: int = DEFAULT_TARGET_DURATION_SECONDS,
    max_duration_seconds: int = DEFAULT_MAX_DURATION_SECONDS,
    max_zoom_count: int = DEFAULT_MAX_ZOOM_COUNT,
) -> dict[str, Any]:
    deck_dir = deck_dir.resolve()
    slides_md = slides_md or deck_dir / "5_draft_slides_with_images.md"
    asset_inventory = asset_inventory or deck_dir / "asset_inventory.json"
    slides = parse_slides(slides_md)
    specs = _figure_specs_by_slide(asset_inventory)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for slide in slides:
        grouped[_cluster_key(slide, specs.get(int(slide["num"])))].append(slide)

    selected_zoom_slides: list[int] = []
    clusters: list[dict[str, Any]] = []
    for cluster_slides in grouped.values():
        cluster_slides = sorted(cluster_slides, key=lambda slide: int(slide["num"]))
        retained = [int(cluster_slides[0]["num"])]
        merged = [int(slide["num"]) for slide in cluster_slides[1:]]
        slide_nums = retained + merged
        representative_papers: list[str] = []
        figure_roles: dict[str, str] = {}
        narration_beats: dict[str, str] = {}
        for slide in cluster_slides:
            num = int(slide["num"])
            spec = specs.get(num)
            for ref in _paper_refs(slide, spec):
                if ref and ref not in representative_papers:
                    representative_papers.append(ref)
            role = _figure_role(slide, spec)
            figure_roles[str(num)] = role
            narration_beats[str(num)] = _narration_beat(slide, spec, role)
            if (
                slide.get("images")
                and len(selected_zoom_slides) < max_zoom_count
                and role in {"architecture", "benchmark", "evidence_map", "failure_mode", "mechanism"}
            ):
                selected_zoom_slides.append(num)

        if not representative_papers:
            representative_papers = ["source-backed slide evidence"]

        clusters.append(
            {
                "claim": _representative_claim(cluster_slides),
                "representative_papers": representative_papers[:6],
                "retained_slides": retained,
                "merged_slides": merged,
                "dropped_slides": [],
                "retained_rationale": (
                    f"Slide {retained[0]} is the clearest representative for this cluster."
                ),
                "merged_rationale": (
                    "Merge the remaining slides into the retained narration when they repeat the same paper set or claim."
                    if merged
                    else "No obvious duplicate slide in this cluster."
                ),
                "dropped_rationale": "No slides are dropped by the deterministic storyboard; operator or LLM review may drop low-value repeats.",
                "figure_roles": figure_roles,
                "narration_beats": narration_beats,
                "source_slide_count": len(slide_nums),
            }
        )

    return {
        "schema_version": 1,
        "deck_id": deck_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/presentations/build_editorial_storyboard.py",
        "target_duration_seconds": target_duration_seconds,
        "max_duration_seconds": max_duration_seconds,
        "max_zoom_count": max_zoom_count,
        "input_slide_count": len(slides),
        "cluster_count": len(clusters),
        "selected_zoom_slides": selected_zoom_slides,
        "compression_summary": {
            "retained_representative_slide_count": sum(len(c["retained_slides"]) for c in clusters),
            "merge_candidate_slide_count": sum(len(c["merged_slides"]) for c in clusters),
            "policy": "Moderate compression: merge repeated paper/claim clusters, retain materially distinct evidence.",
        },
        "clusters": clusters,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--slides-md", type=Path)
    parser.add_argument("--asset-inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-duration-seconds", type=int, default=DEFAULT_TARGET_DURATION_SECONDS)
    parser.add_argument("--max-duration-seconds", type=int, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--max-zoom-count", type=int, default=DEFAULT_MAX_ZOOM_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output or args.deck_dir / "editorial_storyboard.json"
    storyboard = build_storyboard(
        deck_dir=args.deck_dir,
        slides_md=args.slides_md,
        asset_inventory=args.asset_inventory,
        target_duration_seconds=args.target_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        max_zoom_count=args.max_zoom_count,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(storyboard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "editorial storyboard: "
        f"{storyboard['deck_id']} clusters={storyboard['cluster_count']} "
        f"merge_candidates={storyboard['compression_summary']['merge_candidate_slide_count']} "
        f"zoom_candidates={len(storyboard['selected_zoom_slides'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
