#!/usr/bin/env python3
"""Offline regression replay of the figure trial v2 gate stack (v2.4 Phase 01).

Recomputes EVERY gate for the six labeled trial-v2 figures from artifacts
already on disk (best-attempt renders, chosen crops, final SVGs) plus re-run
structured critiques — the old trial receipt supplies ONLY inputs (paths,
grounding metadata, old CLIP scores as a comparison baseline), never its
verdicts or outcome labels (RESEARCH Pitfall 7). Provider calls: one
structured critique per figure (six), one pooled provenance re-rank for
sl_11, plus at most one retry per critique parse failure — all receipted
under stage id ``figure_trial_v2_replay`` and capped by a $2.00
halt-and-approve budget that is NEVER auto-raised.

Classification per figure (amended Phase 01 done gates): ``fail`` iff the
re-run critique verdict != pass OR coverage is None OR coverage < threshold;
otherwise ``coverage_pass``. DINOv2/CLIP margins under the bound_papers_v1
pool and the four phase16 estimator findings are recorded as Phase 02
calibration EVIDENCE, never as classification inputs.

The replay writes a SEPARATE ``figure_trial_replay_receipt.json`` — the
schema-1 trial receipt on disk is immutable provenance and is never
rewritten.

Usage (live execution is plan 01-05):
  HF_HUB_OFFLINE=1 .venv/bin/python scripts/presentations/replay_figure_trial.py \
    --deck-dir conferences/iclr2026/decks/03_evaluation_benchmarks_data --live
"""

from __future__ import annotations

import os

# Pitfall 6 (HF_HUB_OFFLINE scope): the replay is offline apart from the
# receipted provider calls — set BEFORE any transformers-touching import so
# DINOv2 loads from the warm pinned cache and the receipt records
# hf_hub_offline true. setdefault keeps an explicit operator override.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.figure_grounding import (
    RANKER_MAX_POOLED_CANDIDATES,
    bare_openreview_id,
    caption_below_bbox,
    rank_candidates,
)
from conference_pipeline.presentation.figure_similarity import (
    clip_image_similarity,
    dinov2_image_similarity,
    dinov2_runtime_info,
    margin_sign_test,
    preflight_dinov2,
)
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic


def _load_trial_module() -> Any:
    """Load run_figure_trial.py via importlib to reuse its gate functions.

    Same pattern as the unit tests; the trial module's module-scope PHASE16
    load is cheap and offline — no model, provider, or browser work happens
    at import time. Reused symbols: component_coverage, _run_critique,
    _margin_distractor_pool, BudgetTracker, TrialBudgetExceeded, PHASE16,
    SONNET_MODEL.
    """

    spec = importlib.util.spec_from_file_location(
        "figure_trial_runner_for_replay",
        Path(__file__).resolve().parent / "run_figure_trial.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRIAL = _load_trial_module()

REPLAY_STAGE_ID = "figure_trial_v2_replay"

# Expected per-figure classifications, keyed by the 5-char slide_id prefix.
# sl_11 (wrong-figure redraw) and sl_19 (expert-benchmark data redraw) must
# fail; the other four must pass the coverage gate (amended done gates — no
# margin-sign or estimator expectation participates in classification).
EXPECTED_CLASSIFICATIONS = {
    "sl_05": "coverage_pass",
    "sl_06": "coverage_pass",
    "sl_10": "coverage_pass",
    "sl_11": "fail",
    "sl_19": "fail",
    "sl_23": "coverage_pass",
}

SL11_PREFIX = "sl_11"
# The paper that actually makes sl_11's claim (Proof Flow) is bound only via
# a speech_transcript source id, which bare_openreview_id cannot resolve —
# there is no note id to ground against, so the "grounded_correct" branch of
# the sl_11 gate is structurally unreachable on this data (RESEARCH Pattern
# 4 / Empirical Results). Kept as a named constant so the gate logic stays
# honest about WHY only the skip path can satisfy it.
SL11_CORRECT_NOTE_ID: str | None = None


def classify_replay_figure(
    *,
    critique_verdict: Any,
    coverage: float | None,
    coverage_threshold: float,
) -> str:
    """Pure replay classification (amended done gates; Pitfall 7).

    ``fail`` iff the RE-RUN critique verdict is not "pass" OR coverage is
    None (unlabeled blocking gate) OR coverage is strictly below threshold;
    otherwise ``coverage_pass``. Margins, estimator findings, and
    claim_match are recorded evidence ONLY — they are deliberately not
    parameters of this function so they cannot influence classification.
    """

    if critique_verdict != "pass":
        return "fail"
    if coverage is None:
        return "fail"
    if coverage < coverage_threshold:
        return "fail"
    return "coverage_pass"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck-dir",
        type=Path,
        default=PROJECT_ROOT / "conferences/iclr2026/decks/03_evaluation_benchmarks_data",
    )
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--grounding-root",
        type=Path,
        default=PROJECT_ROOT / "conferences/iclr2026/cache/figure-grounding",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="replay receipt path (default: <deck-dir>/figure_trial_replay_receipt.json)",
    )
    parser.add_argument(
        "--budget-cap-usd",
        type=float,
        default=2.0,
        help="halt-and-approve spend cap for the ~7 receipted calls (never auto-raised)",
    )
    parser.add_argument("--critique-model", default=TRIAL.SONNET_MODEL)
    parser.add_argument("--ranker-model", default=TRIAL.SONNET_MODEL)
    parser.add_argument("--coverage-threshold", type=float, default=0.8)
    parser.add_argument("--claim-match-threshold", type=float, default=0.7)
    parser.add_argument("--live", action="store_true", help="Make live provider calls")
    parser.add_argument("--anthropic-timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def _resolvable_notes(source_ids: list[str]) -> list[str]:
    """Deduped bare OpenReview note ids from a slide's bound source ids."""

    notes: list[str] = []
    for source_id in source_ids:
        note_id = bare_openreview_id(source_id)
        if note_id and note_id not in notes:
            notes.append(note_id)
    return notes


def _collect_figure_inputs(
    grounded_results: list[dict[str, Any]],
    binding_by_slide: dict[str, dict[str, Any]],
    grounding_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve and existence-check every replay input path up front.

    CRITICAL (Pitfall 7): the trial receipt supplies ONLY inputs here —
    paths, grounding metadata, the figure design brief claim, and the old
    CLIP similarity as a comparison baseline. No trial verdict or outcome
    label is carried into classification.
    """

    figures: list[dict[str, Any]] = []
    missing: list[str] = []
    for result in grounded_results:
        slide_id = str(result["slide_id"])
        prefix = slide_id[:5]
        outcome = result["outcome"]
        grounding = outcome["grounding"]
        rendered_png = Path(str(outcome["rendered_png"]))
        crop_path = Path(str(grounding["crop_path"]))
        asset_path = Path(str(outcome["asset_path"]))

        binding_slide = binding_by_slide.get(slide_id)
        if binding_slide is None:
            missing.append(f"{slide_id}: no slide entry in source_binding_report.json")
            source_ids: list[str] = []
        else:
            source_ids = [
                str(ref.get("source_id") or "")
                for ref in binding_slide.get("visible_references", [])
            ]
        notes = _resolvable_notes(source_ids)

        for label, path in (
            ("rendered_png", rendered_png),
            ("grounding.crop_path", crop_path),
            ("asset_path", asset_path),
        ):
            if not path.is_file():
                missing.append(f"{slide_id}: {label} missing: {path}")

        chosen_note = str(grounding.get("openreview_id") or "")
        chosen_crops_dir = grounding_root / "crops" / chosen_note
        if not chosen_crops_dir.is_dir():
            missing.append(
                f"{slide_id}: chosen-crop note crops dir missing: {chosen_crops_dir}"
            )
        for note in notes:
            crops_dir = grounding_root / "crops" / note
            if not crops_dir.is_dir():
                # Bound papers without crops on disk contribute nothing to
                # the bound_papers_v1 pool by definition — warn, don't fail.
                print(
                    f"[replay] {slide_id}: bound note {note} has no crops dir "
                    f"on disk (excluded from margin pool)",
                    file=sys.stderr,
                )

        if prefix == SL11_PREFIX:
            pool_notes = [
                note for note in notes if (grounding_root / "crops" / note).is_dir()
            ]
            for note in pool_notes:
                extraction = grounding_root / "crops" / note / "extraction_results.json"
                paper_pdf = grounding_root / "papers" / f"{note}.pdf"
                if not extraction.is_file():
                    missing.append(f"{slide_id}: {extraction} missing (sl_11 re-rank pool)")
                if not paper_pdf.is_file():
                    missing.append(f"{slide_id}: {paper_pdf} missing (sl_11 re-rank captions)")

        figure_design_brief = result.get("figure_design_brief") or {}
        figures.append(
            {
                "slide_id": slide_id,
                "prefix": prefix,
                "tier": str(result.get("tier") or ""),
                "rendered_png": rendered_png,
                "crop_path": crop_path,
                "asset_path": asset_path,
                "source_ids": source_ids,
                "notes": notes,
                "grounding_openreview_id": chosen_note,
                "grounding_paper_title": str(grounding.get("paper_title") or ""),
                "trial_clip_similarity": outcome.get("clip_similarity"),
                "claim_to_visualize": str(
                    figure_design_brief.get("claim_to_visualize") or ""
                ),
            }
        )
    return figures, missing


def _metric_margin(
    sim_chosen: float | None, distractor_sims: list[float | None]
) -> dict[str, Any]:
    """Margin sign test for one embedding metric (recorded evidence only)."""

    if sim_chosen is None:
        return {"margin": None, "margin_sign": None}
    scored = [sim for sim in distractor_sims if sim is not None]
    result = margin_sign_test(sim_chosen, scored)
    return {"margin": result["margin"], "margin_sign": result["margin_sign"]}


def _replay_figure(
    *,
    figure: dict[str, Any],
    client: Any,
    args: argparse.Namespace,
    budget: Any,
    grounding_root: Path,
) -> dict[str, Any]:
    """Recompute the full gate tuple for one figure from on-disk artifacts.

    The only provider call is the structured critique re-run (the old trial
    critique JSON lacks component lists), retried ONCE within the cap when
    its parse status comes back "parse_failed"; the second result is kept
    either way. Everything else — coverage arithmetic, DINOv2/CLIP sims,
    bound_papers_v1 margins, the four phase16 estimators — is local.
    """

    slide_id = figure["slide_id"]
    rendered_png: Path = figure["rendered_png"]
    crop_path: Path = figure["crop_path"]
    asset_path: Path = figure["asset_path"]

    print(f"[replay] {slide_id}: structured critique re-run", file=sys.stderr)
    critique_args = SimpleNamespace(critique_model=args.critique_model)
    critique = TRIAL._run_critique(
        client=client,
        args=critique_args,
        budget=budget,
        rendered_png=rendered_png,
        source_crop=crop_path,
        tier=figure["tier"],
    )
    retried_after_parse_failure = False
    if critique.get("critique_parse_status") == "parse_failed":
        print(
            f"[replay] {slide_id}: critique parse failed; retrying once",
            file=sys.stderr,
        )
        critique = TRIAL._run_critique(
            client=client,
            args=critique_args,
            budget=budget,
            rendered_png=rendered_png,
            source_crop=crop_path,
            tier=figure["tier"],
        )
        retried_after_parse_failure = True

    coverage = TRIAL.component_coverage(
        critique.get("matched_components") or [],
        critique.get("missing_components") or [],
        critique.get("invented_components") or [],
    )
    classification = classify_replay_figure(
        critique_verdict=critique.get("verdict"),
        coverage=coverage,
        coverage_threshold=args.coverage_threshold,
    )

    dinov2_warnings: list[str] = []
    dinov2_similarity = dinov2_image_similarity(
        rendered_png, crop_path, warnings=dinov2_warnings
    )
    clip_similarity = clip_image_similarity(rendered_png, crop_path)

    # Margin under bound_papers_v1 (recorded as Phase 02 calibration
    # evidence): DINOv2 and CLIP margins are co-recorded over the SAME pool
    # for the DINOv2-vs-CLIP comparison.
    pool = TRIAL._margin_distractor_pool(figure["source_ids"], crop_path, grounding_root)
    dinov2_distractor_sims: list[float | None] = []
    clip_distractor_sims: list[float | None] = []
    for distractor in pool["distractors"]:
        dinov2_distractor_sims.append(
            dinov2_image_similarity(rendered_png, distractor, warnings=dinov2_warnings)
        )
        clip_distractor_sims.append(clip_image_similarity(rendered_png, distractor))
    margin = {
        "dinov2": _metric_margin(dinov2_similarity, dinov2_distractor_sims),
        "clip": _metric_margin(clip_similarity, clip_distractor_sims),
        "distractor_count": len(pool["distractors"]),
        "pool_notes": pool["notes"],
        "protocol": pool["protocol"],
    }

    # Estimator findings on the FINAL SVG (advisory evidence; real findings
    # expected for 4/6 figures, sl_11 and sl_23 legitimately empty).
    estimator_findings = {
        "oversized_arrowhead": TRIAL.PHASE16._phase16_oversized_arrowhead_issues(asset_path),
        "line_over_text": TRIAL.PHASE16._phase16_line_over_text_issues(asset_path),
        "text_overlap": TRIAL.PHASE16._phase16_text_overlap_issues(asset_path),
        "text_exceeds_canvas": TRIAL.PHASE16._phase16_text_exceeds_canvas_issues(asset_path),
    }

    expected = EXPECTED_CLASSIFICATIONS[figure["prefix"]]
    entry: dict[str, Any] = {
        "slide_id": slide_id,
        "tier": figure["tier"],
        "classification": classification,
        "expected_classification": expected,
        "classification_matches": classification == expected,
        "coverage": coverage,
        "coverage_threshold": args.coverage_threshold,
        "critique": {
            "verdict": critique.get("verdict"),
            "matched_components": critique.get("matched_components") or [],
            "missing_components": critique.get("missing_components") or [],
            "invented_components": critique.get("invented_components") or [],
            "structural_issues": critique.get("structural_issues") or [],
            "legibility_issues": critique.get("legibility_issues") or [],
            "feedback": critique.get("feedback"),
            "critique_parse_status": critique.get("critique_parse_status"),
            "retried_after_parse_failure": retried_after_parse_failure,
            "receipt_path": critique.get("receipt_path"),
        },
        "dinov2_similarity": dinov2_similarity,
        "clip_similarity": clip_similarity,
        "trial_clip_similarity": figure["trial_clip_similarity"],
        "margin": margin,
        "estimator_findings": estimator_findings,
        "inputs": {
            "rendered_png": str(rendered_png),
            "crop_path": str(crop_path),
            "asset_path": str(asset_path),
            "svg_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "crop_sha256": hashlib.sha256(crop_path.read_bytes()).hexdigest(),
        },
    }
    if dinov2_warnings:
        entry["dinov2_warnings"] = dinov2_warnings
    return entry


def _build_sl11_candidate_pool(
    figure: dict[str, Any], grounding_root: Path
) -> list[dict[str, Any]]:
    """Candidate pool for the sl_11 pooled re-rank — built WITHOUT network.

    Candidates come from each bound note's on-disk
    ``crops/<note>/extraction_results.json`` — never the grounding module's
    extraction helper, which would re-extract and overwrite the cached
    crops; captions are derived offline via PyMuPDF from the cached PDF.
    """

    pool: list[dict[str, Any]] = []
    for note in figure["notes"]:
        crops_dir = grounding_root / "crops" / note
        extraction_path = crops_dir / "extraction_results.json"
        if not extraction_path.is_file():
            continue
        entries = json.loads(extraction_path.read_text(encoding="utf-8"))
        pdf_path = grounding_root / "papers" / f"{note}.pdf"
        if note == figure["grounding_openreview_id"]:
            paper_title = figure["grounding_paper_title"]
        else:
            paper_title = f"OpenReview {note}"
        for entry in entries:
            bbox = entry.get("bbox") or {}
            bbox_tuple = (
                float(bbox.get("x0", 0.0)),
                float(bbox.get("y0", 0.0)),
                float(bbox.get("x1", 0.0)),
                float(bbox.get("y1", 0.0)),
            )
            image_path = crops_dir / str(entry.get("image_filename") or "")
            if not image_path.is_file():
                image_path = Path(str(entry.get("image_path") or ""))
            if not image_path.is_file():
                continue
            page = int(entry.get("page") or 0)
            pool.append(
                {
                    "page": page,
                    "bbox": list(bbox_tuple),
                    "image_path": str(image_path),
                    "caption": caption_below_bbox(pdf_path, page, bbox_tuple),
                    "paper_title": paper_title,
                    "note_id": note,
                }
            )
    return pool


def _sl11_pooled_rerank(
    *,
    figure: dict[str, Any],
    client: Any,
    args: argparse.Namespace,
    budget: Any,
    grounding_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One receipted pooled provenance re-rank for sl_11 (CALIB-04 evidence).

    Returns ``(pooled_rerank_record, sl11_gate)``. The done gate is
    satisfied when the ranker's confidence falls below threshold or it
    rejects every candidate (would_skip — the correct paper, Proof Flow,
    cannot be in the pool) OR, hypothetically, when the ranker grounds the
    correct paper's figure (unreachable here: SL11_CORRECT_NOTE_ID is None
    because Proof Flow has no resolvable OpenReview id).
    """

    pool = _build_sl11_candidate_pool(figure, grounding_root)
    print(
        f"[replay] {figure['slide_id']}: pooled re-rank over "
        f"{len(pool)} candidates from notes {sorted({c['note_id'] for c in pool})}",
        file=sys.stderr,
    )
    verdict = rank_candidates(
        client=client,
        caller=__file__,
        model=args.ranker_model,
        candidates=pool,
        slide_claim=figure["claim_to_visualize"],
        slide_title=figure["slide_id"],
        on_call=budget.record,
    )
    claim_match = float(verdict.get("claim_match") or 0.0)
    try:
        chosen_index = int(verdict.get("chosen_index", -1))
    except (TypeError, ValueError):
        chosen_index = -1
    ranked_count = min(len(pool), RANKER_MAX_POOLED_CANDIDATES)
    chosen_valid = 0 <= chosen_index < ranked_count
    would_skip = (claim_match < args.claim_match_threshold) or not chosen_valid

    if would_skip:
        satisfied, mode = True, "skip_path"
    elif (
        SL11_CORRECT_NOTE_ID is not None
        and pool[chosen_index]["note_id"] == SL11_CORRECT_NOTE_ID
    ):
        satisfied, mode = True, "grounded_correct"
    else:
        satisfied, mode = False, None

    pooled_rerank = {
        "claim_match": claim_match,
        "claim_match_threshold": args.claim_match_threshold,
        "chosen_index": chosen_index,
        "would_skip": would_skip,
        "figure_kind": str(verdict.get("figure_kind") or "other"),
        "rationale": str(verdict.get("rationale") or ""),
        "receipt_path": verdict.get("receipt_path"),
        "pool": {
            "notes": sorted({candidate["note_id"] for candidate in pool}),
            "candidate_count": len(pool),
        },
    }
    sl11_gate = {"satisfied": satisfied, "mode": mode}
    return pooled_rerank, sl11_gate


def _out_clobbers_trial_receipt(out_path: Path, trial_receipt_path: Path) -> bool:
    """True when ``--out`` resolves to the original trial receipt path (WR-02).

    The replay must never rewrite the schema-1 trial receipt — the immutable
    provenance record it verifies by sha256 (``replay_of.sha256``). Pure
    resolve-and-compare so the guard is unit-testable without ``--live``.
    """

    return out_path.resolve() == trial_receipt_path.resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.live:
        print(
            "refusing to run without --live (this replay makes ~7 paid provider calls)",
            file=sys.stderr,
        )
        return 2

    deck_dir = args.deck_dir.resolve()
    grounding_root = args.grounding_root.resolve()
    out_path = (
        args.out if args.out is not None else deck_dir / "figure_trial_replay_receipt.json"
    )

    # DINOv2 preflight (D-02): hard-fail BEFORE the Anthropic client exists —
    # no provider spend on a replay whose primary metric cannot be computed.
    try:
        dinov2_info = preflight_dinov2()
    except RuntimeError as exc:
        print(f"[replay] {exc}", file=sys.stderr)
        return 4
    print(
        f"[replay] dinov2 preflight ok: device={dinov2_info['device']} "
        f"revision={dinov2_info['revision']} "
        f"hf_hub_offline={dinov2_info['hf_hub_offline']}",
        file=sys.stderr,
    )

    trial_receipt_path = deck_dir / "figure_trial_receipt.json"
    # WR-02: refuse --out paths that resolve to the original trial receipt —
    # this runs before any provider-capable object exists (the Anthropic
    # client and TRIAL.BudgetTracker are constructed further down).
    if _out_clobbers_trial_receipt(out_path, trial_receipt_path):
        print(
            "[replay] refusing to overwrite the original trial receipt",
            file=sys.stderr,
        )
        return 2
    binding_path = deck_dir / "source_binding_report.json"
    missing_contracts = [
        str(path) for path in (trial_receipt_path, binding_path) if not path.is_file()
    ]
    if missing_contracts:
        for path in missing_contracts:
            print(f"[replay] required input missing: {path}", file=sys.stderr)
        return 1

    trial_receipt_bytes = trial_receipt_path.read_bytes()
    trial_receipt = json.loads(trial_receipt_bytes.decode("utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding_by_slide = {
        str(slide.get("slide_id") or ""): slide for slide in binding.get("slides", [])
    }

    grounded_results = [
        result
        for result in trial_receipt.get("results", [])
        if "grounding" in (result.get("outcome") or {})
    ]
    prefixes = {str(result["slide_id"])[:5] for result in grounded_results}
    if prefixes != set(EXPECTED_CLASSIFICATIONS):
        print(
            f"[replay] grounded slide prefixes {sorted(prefixes)} do not match the "
            f"expected replay set {sorted(EXPECTED_CLASSIFICATIONS)}",
            file=sys.stderr,
        )
        return 1

    figures_inputs, missing = _collect_figure_inputs(
        grounded_results, binding_by_slide, grounding_root
    )
    if missing:
        for item in missing:
            print(f"[replay] missing input: {item}", file=sys.stderr)
        return 1

    import anthropic

    client = anthropic.Anthropic(timeout=args.anthropic_timeout_seconds)
    budget = TRIAL.BudgetTracker(args.budget_cap_usd)

    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    figure_entries: list[dict[str, Any]] = []
    sl11_gate: dict[str, Any] | None = None
    halted: str | None = None
    try:
        with provider_receipt_context(
            workspace_root=args.workspace, stage_id=REPLAY_STAGE_ID, caller=__file__,
        ):
            for figure in figures_inputs:
                entry = _replay_figure(
                    figure=figure,
                    client=client,
                    args=args,
                    budget=budget,
                    grounding_root=grounding_root,
                )
                figure_entries.append(entry)
            # sl_11 pooled re-rank: one extra receipted call, same budget.
            sl11_figure = next(
                figure for figure in figures_inputs if figure["prefix"] == SL11_PREFIX
            )
            pooled_rerank, sl11_gate = _sl11_pooled_rerank(
                figure=sl11_figure,
                client=client,
                args=args,
                budget=budget,
                grounding_root=grounding_root,
            )
            for entry in figure_entries:
                if entry["slide_id"] == sl11_figure["slide_id"]:
                    entry["pooled_rerank"] = pooled_rerank
    except TRIAL.TrialBudgetExceeded as exc:
        # Halt-and-approve: record the halt, keep whatever completed, and
        # NEVER raise the cap in code (project rule + user memory).
        halted = str(exc)
        print(f"[replay] HALTED: {exc}", file=sys.stderr)

    observed = {
        entry["slide_id"][:5]: entry["classification"] for entry in figure_entries
    }
    all_match = len(observed) == len(EXPECTED_CLASSIFICATIONS) and all(
        observed.get(prefix) == expected
        for prefix, expected in EXPECTED_CLASSIFICATIONS.items()
    )

    receipt = {
        "schema_version": 1,
        "stage_id": REPLAY_STAGE_ID,
        "replay_of": {
            "path": str(trial_receipt_path),
            "sha256": hashlib.sha256(trial_receipt_bytes).hexdigest(),
        },
        "started_at": started_at,
        "ended_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "models": {
            "critique_model": args.critique_model,
            "ranker_model": args.ranker_model,
        },
        "dinov2": dinov2_runtime_info(),
        "coverage_threshold": args.coverage_threshold,
        "claim_match_threshold": args.claim_match_threshold,
        "margin_pool_protocol": "bound_papers_v1",
        "budget": {
            "cap_usd": args.budget_cap_usd,
            "estimated_spend_usd": round(budget.spent_usd, 4),
            "calls": budget.calls,
            "halt": halted,
        },
        "classification_summary": {
            "expected": EXPECTED_CLASSIFICATIONS,
            "observed": observed,
            "all_match": all_match,
        },
        "sl11_gate": sl11_gate,
        "figures": figure_entries,
    }
    write_json_atomic(out_path, receipt)
    print(
        f"[replay] done: {len(figure_entries)}/{len(figures_inputs)} figures, "
        f"${budget.spent_usd:.2f} of ${args.budget_cap_usd:.2f} spent, "
        f"all_match={all_match}, "
        f"sl11_gate={'unreached' if sl11_gate is None else sl11_gate}, "
        f"receipt: {out_path}",
        file=sys.stderr,
    )

    if halted is not None:
        return 3
    if all_match and sl11_gate is not None and sl11_gate["satisfied"]:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
