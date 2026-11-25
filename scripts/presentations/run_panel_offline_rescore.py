"""Offline panel re-scoring orchestrator for Phase 02 (v2.5).

Handles three data paths:
  Path A — $0 baseline aggregation (11 slides from calibration_review_panel.json + qwen solo profile)
  Path B — Fresh anchor panel calls (6 anchor slides: 3 accept, 3 failure)
  Path C — Fresh perturbation panel calls (3 perturbed renders)

Produces all Phase 02 JSON artifacts: 02-panel-rescore.json, 02-budget-projection.json,
02-voting-rule-selection.json, 02-dinov2-floor.json, 02-adjudication.json,
02-validation-report.md, and 02-GO-NOGO.md if the kill signal fires.

Requirements: PANEL-05, PANEL-08, PANEL-09, PANEL-11, PANEL-12, PANEL-13
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from scripts.presentations.figure_quality_panel import (
    JudgeVerdict,
    PanelVerdict,
    aggregate_panel,
    claude_judge_adapter,
    gemini_judge_adapter,
    qwen_judge_adapter,
)
from scripts.presentations.calibrate_figure_thresholds import select_stack_floor
from conference_pipeline.workflow_runs import write_json_atomic


FRAMING_CAVEAT = (
    "These results measure consistency between the de-codex'd 3-judge vision-LLM "
    "panel and prior AI-panel labels. They do NOT constitute accuracy against human "
    "ground truth. Human ground-truth labeling is deferred to a separate milestone."
)

RULES = ["any_accept", "majority", "unanimous"]

ANCHOR_PREFIXES = {
    "sl_05": "accept",
    "sl_06": "accept",
    "sl_10": "accept",
    "sl_11": "failure",
    "sl_19": "failure",
    "sl_23": "failure",
}

RENDER_MAP = {
    "sl_05": "slide_005.png",
    "sl_06": "slide_006.png",
    "sl_10": "slide_010.png",
    "sl_11": "slide_011.png",
    "sl_19": "slide_019.png",
    "sl_23": "slide_023.png",
}

# Approximate per-call cost in USD for paid judges (gemini + claude); qwen is $0
_COST_PER_PAID_CALL = 0.01


# ---------------------------------------------------------------------------
# Testable helper functions (contract-tested in tests/test_panel_offline_rescore.py)
# ---------------------------------------------------------------------------

def apply_voting_rule(rule: str, votes: dict[str, str]) -> str:
    pass_count = sum(1 for v in votes.values() if v == "pass")
    if rule == "any_accept":
        return "pass" if pass_count >= 1 else "fail"
    if rule == "majority":
        return "pass" if pass_count >= 2 else "fail"
    if rule == "unanimous":
        return "pass" if pass_count == 3 else "fail"
    raise ValueError(f"Unknown rule: {rule}")


def build_qwen_baseline_lookup(per_slide: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for entry in per_slide:
        if entry.get("baseline_present"):
            lookup[entry["slide_id"]] = entry
    return lookup


def build_adjudication_entry(
    slide_id: str,
    panel_verdict: str,
    prior_label: str,
    panel_votes: dict[str, str],
    panel_split: bool,
) -> dict:
    return {
        "slide_id": slide_id,
        "panel_verdict": panel_verdict,
        "prior_label": prior_label,
        "panel_votes": panel_votes,
        "split_votes": panel_split,
        "adjudication": None,
        "notes": None,
    }


def compute_voting_rule_metrics(rule: str, anchor_results: list[dict]) -> dict:
    failure_anchors = [r for r in anchor_results if r["anchor_type"] == "failure"]
    accept_anchors = [r for r in anchor_results if r["anchor_type"] == "accept"]
    perturbations = [r for r in anchor_results if r["anchor_type"] == "perturbation"]

    under_reject = sum(
        1 for r in failure_anchors if apply_voting_rule(rule, r["votes"]) == "pass"
    )
    over_reject = sum(
        1 for r in accept_anchors if apply_voting_rule(rule, r["votes"]) == "fail"
    )
    perturb_det = sum(
        1 for r in perturbations if apply_voting_rule(rule, r["votes"]) == "fail"
    )

    return {
        "rule": rule,
        "under_rejection_count": under_reject,
        "under_rejection_rate": under_reject / len(failure_anchors) if failure_anchors else None,
        "over_rejection_count": over_reject,
        "over_rejection_rate": over_reject / len(accept_anchors) if accept_anchors else None,
        "perturbation_detection_count": perturb_det,
        "perturbation_detection_rate": perturb_det / len(perturbations) if perturbations else None,
    }


def select_voting_rule_from_metrics(rule_metrics: list[dict]) -> tuple[str | None, dict | None, str, str]:
    """Select a rule only if it satisfies the full Phase 02 anchor criterion."""
    rule_priority = {"any_accept": 0, "majority": 1, "unanimous": 2}
    separating = [
        m
        for m in rule_metrics
        if m["under_rejection_count"] == 0
        and m["over_rejection_count"] == 0
        and m.get("perturbation_detection_rate") in (None, 1.0)
    ]
    if separating:
        chosen = min(separating, key=lambda m: rule_priority[m["rule"]])
        return (
            chosen["rule"],
            chosen,
            "selected",
            f"Most permissive rule satisfying full anchor-set separation: {chosen['rule']}",
        )
    least_bad = min(
        rule_metrics,
        key=lambda m: (
            m["under_rejection_count"],
            m["over_rejection_count"],
            rule_priority[m["rule"]],
        ),
    )
    return (
        None,
        None,
        "no_separating_rule",
        "No candidate rule satisfies the full anchor-set criterion: failure anchors rejected and accept anchors accepted. "
        f"Least-bad rule by under-/over-rejection count is {least_bad['rule']}.",
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_judge_verdict(vote: str, tags: list[str]) -> JudgeVerdict:
    v = "pass" if vote == "pass" else "fail"
    return JudgeVerdict(
        verdict=v,
        failure_modes=tags,
        rationale="",
        status="ok",
        receipt_path=None,
        faithfulness_verdict=v,
        faithfulness_tags=[],
        legibility_verdict=v,
        legibility_tags=[],
        style_verdict=v,
        style_tags=[],
    )


def _build_figure_record(
    slide_id: str,
    panel: PanelVerdict,
    panel_path: str,
    prior_label: str,
    prior_source: str,
    estimator_findings: dict | None,
    anchor_type: str | None = None,
    dry_run: bool = False,
) -> dict:
    votes = {k: v for k, v in panel.votes.items()}
    panel_split = not panel.unanimous and len(set(votes.values())) > 1
    # More precise: 2-1 split means exactly 2 unique values with at least one each
    if len(votes) == 3:
        pass_count = sum(1 for v in votes.values() if v == "pass")
        panel_split = pass_count in (1, 2)

    record: dict[str, Any] = {
        "slide_id": slide_id,
        "panel_path": panel_path,
        "votes": votes,
        "majority_verdict": panel.majority_verdict,
        "failure_modes": panel.failure_modes,
        "panel_split": panel_split,
        "prior_label": prior_label,
        "prior_source": prior_source,
        "estimator_findings": estimator_findings,
        "panel_vs_prior_disagree": panel.majority_verdict != prior_label if prior_label not in ("unknown", None) else False,
    }
    if anchor_type is not None:
        record["anchor_type"] = anchor_type
    if dry_run:
        record["dry_run"] = True
    return record


def _slide_prefix(slide_id: str) -> str:
    return "_".join(slide_id.split("_")[:2])


def _format_estimator_advisory(findings: dict | None) -> str:
    if not findings:
        return ""
    lines = [
        "Advisory (do not block your vote on this):",
        "Automated estimator signals for this figure:",
    ]
    for k, v in findings.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

class _BudgetTracker:
    def __init__(self, cap_usd: float) -> None:
        self.cap = cap_usd
        self.spend = 0.0

    def charge(self, n_paid_calls: int) -> None:
        self.spend += n_paid_calls * _COST_PER_PAID_CALL
        if self.spend >= self.cap:
            raise RuntimeError(
                f"Budget cap exceeded: ${self.spend:.4f} >= cap ${self.cap:.4f}. "
                "Halt and seek operator approval before continuing."
            )


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline panel re-scoring orchestrator (Phase 02, v2.5)."
    )
    parser.add_argument("--deck-dir", required=True, help="Deck directory containing rendered_frames/")
    parser.add_argument("--panel-json", required=True, help="Path to calibration_review_panel.json")
    parser.add_argument("--qwen-profile", required=True, help="Path to 01-qwen-solo-profile.json")
    parser.add_argument("--dataset", required=True, help="Path to calibration_dataset.json")
    parser.add_argument("--perturbation-manifest", required=True, help="Path to 02-perturbation-manifest.json")
    parser.add_argument("--phase-dir", required=True, help="Output directory for all phase artifacts")
    parser.add_argument(
        "--calibration-thresholds",
        default=None,
        help="Path to calibration_threshold_analysis.json (for t_cov); defaults to deck-dir lookup",
    )
    parser.add_argument("--budget-cap", type=float, default=0.25, help="Halt-and-approve cap in USD")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls; write stub artifacts")
    args = parser.parse_args(argv)

    phase_dir = Path(args.phase_dir)
    deck_dir = Path(args.deck_dir)
    rendered_frames = deck_dir / "rendered_frames"

    # -----------------------------------------------------------------
    # Step 0: Pre-flight
    # -----------------------------------------------------------------
    phase_dir.mkdir(parents=True, exist_ok=True)

    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not gemini_key:
        print("WARNING: GEMINI_API_KEY / GOOGLE_API_KEY not set — live calls will fail.", file=sys.stderr)
    if not anthropic_key:
        print("WARNING: ANTHROPIC_API_KEY not set — live calls will fail.", file=sys.stderr)

    budget = _BudgetTracker(args.budget_cap)

    # -----------------------------------------------------------------
    # Step 1: Path A — $0 baseline aggregation
    # -----------------------------------------------------------------
    print("Step 1: Path A — $0 baseline aggregation")

    panel_json_path = Path(args.panel_json)
    raw_panel = json.loads(panel_json_path.read_text())
    items = raw_panel.get("items", raw_panel if isinstance(raw_panel, list) else [])

    # Deduplicate: keep max-attempt per slide_id
    by_slide: dict[str, dict] = {}
    for item in items:
        sid = item["slide_id"]
        if sid not in by_slide or item.get("attempt", 0) > by_slide[sid].get("attempt", 0):
            by_slide[sid] = item

    # Drop codex; build gemini+claude baseline lookup
    baseline_gemini_claude: dict[str, dict] = {}
    for sid, item in by_slide.items():
        votes = {k: v for k, v in item.get("votes", {}).items() if k != "codex"}
        tags_by_model = {k: v for k, v in item.get("tags_by_model", {}).items() if k != "codex"}
        baseline_gemini_claude[sid] = {
            "gemini_vote": votes.get("gemini", "fail"),
            "gemini_tags": tags_by_model.get("gemini", []),
            "claude_vote": votes.get("claude", "fail"),
            "claude_tags": tags_by_model.get("claude", []),
        }

    # Load qwen solo profile
    qwen_profile_path = Path(args.qwen_profile)
    qwen_profile = json.loads(qwen_profile_path.read_text())
    qwen_lookup = build_qwen_baseline_lookup(qwen_profile["per_slide"])

    # Load calibration dataset
    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text())
    dataset_rows = dataset.get("rows", dataset if isinstance(dataset, list) else [])

    estimator_by_prefix: dict[str, dict] = {}
    prior_label_by_prefix: dict[str, str] = {}
    dinov2_by_prefix: dict[str, float | None] = {}

    for row in dataset_rows:
        if not row.get("is_final_render"):
            continue
        sid = row["slide_id"]
        prefix = _slide_prefix(sid)
        estimator_by_prefix[prefix] = row.get("estimator_findings") or {}
        prior_label_by_prefix[prefix] = row.get("label", "unknown")
        dinov2_by_prefix[prefix] = row.get("dinov2_similarity")

    # Collect baseline slide IDs (only those present in calibration_review_panel.json)
    baseline_slide_ids = list(baseline_gemini_claude.keys())
    print(f"  Baseline slide IDs from panel JSON: {len(baseline_slide_ids)}")

    baseline_figures: list[dict] = []

    for sid in baseline_slide_ids:
        prefix = _slide_prefix(sid)
        gc = baseline_gemini_claude[sid]

        gemini_v = _make_judge_verdict(gc["gemini_vote"], gc["gemini_tags"])

        if sid in qwen_lookup:
            qe = qwen_lookup[sid]
            qwen_v = _make_judge_verdict(
                qe.get("qwen_verdict", "fail"), qe.get("qwen_failure_modes", [])
            )
            claude_v = _make_judge_verdict(gc["claude_vote"], gc["claude_tags"])
        else:
            # qwen not available for this slide — use fail as conservative default
            print(f"  WARNING: no qwen entry for {sid}; using fail as default.")
            qwen_v = _make_judge_verdict("fail", ["other"])
            claude_v = _make_judge_verdict(gc["claude_vote"], gc["claude_tags"])

        panel = aggregate_panel([gemini_v, claude_v, qwen_v], ["gemini", "claude", "qwen"])

        estimator = estimator_by_prefix.get(prefix)
        prior = prior_label_by_prefix.get(prefix, "unknown")
        prior_source = "calibration_review_panel.json"

        record = _build_figure_record(
            slide_id=sid,
            panel=panel,
            panel_path="baseline_aggregation",
            prior_label=prior,
            prior_source=prior_source,
            estimator_findings=estimator,
        )
        baseline_figures.append(record)

    print(f"  Baseline aggregation complete: {len(baseline_figures)} figures.")

    # -----------------------------------------------------------------
    # Step 2: Budget projection before Path B + C
    # -----------------------------------------------------------------
    print("Step 2: Budget projection")

    n_anchor = len(ANCHOR_PREFIXES)
    n_perturb = 3
    anchor_cost = n_anchor * 2 * _COST_PER_PAID_CALL
    perturb_cost = n_perturb * 2 * _COST_PER_PAID_CALL
    total_cost = anchor_cost + perturb_cost

    print(f"  {n_anchor} anchor slides × 2 paid judges × ~${_COST_PER_PAID_CALL}/call = ~${anchor_cost:.2f}")
    print(f"  {n_perturb} perturbation × 2 paid judges × ~${_COST_PER_PAID_CALL}/call = ~${perturb_cost:.2f}")
    print(f"  Total worst-case: ~${total_cost:.2f} (cap: ${args.budget_cap:.2f})")

    budget_projection = {
        "n_anchor_slides": n_anchor,
        "n_perturbation_cases": n_perturb,
        "n_paid_judges": 2,
        "cost_per_call_usd": _COST_PER_PAID_CALL,
        "anchor_cost_usd": anchor_cost,
        "perturbation_cost_usd": perturb_cost,
        "total_worst_case_usd": total_cost,
        "budget_cap_usd": args.budget_cap,
        "dry_run": args.dry_run,
    }
    write_json_atomic(phase_dir / "02-budget-projection.json", budget_projection)

    # -----------------------------------------------------------------
    # Step 3: Path B — Fresh anchor calls
    # -----------------------------------------------------------------
    print("Step 3: Path B — Fresh anchor calls")

    # Build prefix → full slide_id from dataset
    full_sid_by_prefix: dict[str, str] = {}
    for row in dataset_rows:
        if not row.get("is_final_render"):
            continue
        sid = row["slide_id"]
        prefix = _slide_prefix(sid)
        full_sid_by_prefix[prefix] = sid

    anchor_figures: list[dict] = []

    for prefix, anchor_type in ANCHOR_PREFIXES.items():
        render_filename = RENDER_MAP[prefix]
        render_path = rendered_frames / render_filename
        full_sid = full_sid_by_prefix.get(prefix, prefix)
        estimator = estimator_by_prefix.get(prefix)

        if args.dry_run:
            stub_votes = {"gemini": "fail", "claude": "fail", "qwen": "fail"}
            stub_panel = aggregate_panel(
                [_make_judge_verdict("fail", []), _make_judge_verdict("fail", []), _make_judge_verdict("fail", [])],
                ["gemini", "claude", "qwen"],
            )
            record = _build_figure_record(
                slide_id=full_sid,
                panel=stub_panel,
                panel_path="anchor_fresh",
                prior_label="unknown",
                prior_source="none",
                estimator_findings=estimator,
                anchor_type=anchor_type,
                dry_run=True,
            )
            anchor_figures.append(record)
            print(f"  [dry-run] {prefix} ({anchor_type}) — skipped API calls.")
            continue

        if not render_path.exists():
            print(f"  WARNING: render not found: {render_path}; using fail stub.", file=sys.stderr)
            stub_panel = aggregate_panel(
                [_make_judge_verdict("fail", ["other"]), _make_judge_verdict("fail", ["other"]), _make_judge_verdict("fail", ["other"])],
                ["gemini", "claude", "qwen"],
            )
            record = _build_figure_record(
                slide_id=full_sid,
                panel=stub_panel,
                panel_path="anchor_fresh",
                prior_label="unknown",
                prior_source="none",
                estimator_findings=estimator,
                anchor_type=anchor_type,
            )
            record["error"] = f"render not found: {render_path}"
            anchor_figures.append(record)
            continue

        print(f"  Calling panel for {prefix} ({anchor_type}) ...")
        # Advisory text is informational; we pass it as part of the prompt via estimator findings
        # The judge adapters don't accept an advisory arg, so we log it here only.
        if estimator:
            advisory = _format_estimator_advisory(estimator)
            print(f"    Estimator advisory:\n{advisory}")

        # qwen_judge_adapter raises VisionRuntimeUnavailableError on server
        # unavailability. Run Qwen first so local-runtime failures halt before
        # paid Gemini/Claude calls are made for this figure.
        q = qwen_judge_adapter(render_path, None)
        g = gemini_judge_adapter(render_path, None)
        c = claude_judge_adapter(render_path, None)

        # Charge paid judges (gemini + claude; qwen is $0)
        budget.charge(2)

        panel = aggregate_panel([g, c, q], ["gemini", "claude", "qwen"])
        record = _build_figure_record(
            slide_id=full_sid,
            panel=panel,
            panel_path="anchor_fresh",
            prior_label="unknown",
            prior_source="none",
            estimator_findings=estimator,
            anchor_type=anchor_type,
        )
        anchor_figures.append(record)
        print(f"  {prefix}: gemini={g.verdict} claude={c.verdict} qwen={q.verdict} → {panel.majority_verdict}")

    # -----------------------------------------------------------------
    # Step 4: Path C — Perturbation calls
    # -----------------------------------------------------------------
    print("Step 4: Path C — Perturbation calls")

    perturbation_manifest_path = Path(args.perturbation_manifest)
    perturbation_manifest = json.loads(perturbation_manifest_path.read_text())
    cases = perturbation_manifest if isinstance(perturbation_manifest, list) else (
        perturbation_manifest.get("cases") or perturbation_manifest.get("perturbation_cases", [])
    )

    perturbation_figures: list[dict] = []

    for case in cases:
        case_id = case.get("case_id", "unknown")
        injection_type = case.get("injection_type", "unknown")
        output_path = Path(case["output_path"])
        source_slide = case.get("source_slide", "unknown")

        if args.dry_run:
            stub_panel = aggregate_panel(
                [_make_judge_verdict("fail", []), _make_judge_verdict("fail", []), _make_judge_verdict("fail", [])],
                ["gemini", "claude", "qwen"],
            )
            record = _build_figure_record(
                slide_id=case_id,
                panel=stub_panel,
                panel_path="perturbation",
                prior_label="fail",
                prior_source="perturbation_manifest",
                estimator_findings=None,
                anchor_type="perturbation",
                dry_run=True,
            )
            record["case_id"] = case_id
            record["injection_type"] = injection_type
            record["source_slide"] = source_slide
            perturbation_figures.append(record)
            print(f"  [dry-run] {case_id} ({injection_type}) — skipped API calls.")
            continue

        if not output_path.exists():
            print(f"  WARNING: perturbation PNG not found: {output_path}", file=sys.stderr)
            continue

        print(f"  Calling panel for perturbation {case_id} ({injection_type}) ...")
        q = qwen_judge_adapter(output_path, None)
        g = gemini_judge_adapter(output_path, None)
        c = claude_judge_adapter(output_path, None)

        budget.charge(2)

        panel = aggregate_panel([g, c, q], ["gemini", "claude", "qwen"])
        record = _build_figure_record(
            slide_id=case_id,
            panel=panel,
            panel_path="perturbation",
            prior_label="fail",
            prior_source="perturbation_manifest",
            estimator_findings=None,
            anchor_type="perturbation",
        )
        record["case_id"] = case_id
        record["injection_type"] = injection_type
        record["source_slide"] = source_slide
        perturbation_figures.append(record)
        print(f"  {case_id}: gemini={g.verdict} claude={c.verdict} qwen={q.verdict} → {panel.majority_verdict}")

    # -----------------------------------------------------------------
    # Step 5: Write 02-panel-rescore.json
    # -----------------------------------------------------------------
    print("Step 5: Write 02-panel-rescore.json")

    all_figures = baseline_figures + anchor_figures + perturbation_figures
    rescore_payload: dict[str, Any] = {
        "framing": FRAMING_CAVEAT,
        "budget_projection_usd": total_cost,
        "spend_usd": budget.spend,
        "figures": all_figures,
    }
    rescore_path = phase_dir / "02-panel-rescore.json"
    write_json_atomic(rescore_path, rescore_payload)
    print(f"  Written: {rescore_path}")

    # -----------------------------------------------------------------
    # Step 6: go/no-go kill signal check
    # -----------------------------------------------------------------
    print("Step 6: go/no-go kill signal check")

    sl23_records = [r for r in anchor_figures if _slide_prefix(r["slide_id"]) == "sl_23"]
    if sl23_records and not args.dry_run:
        sl23 = sl23_records[0]
        sl23_votes = sl23["votes"]
        sl23_verdict_per_rule = {rule: apply_voting_rule(rule, sl23_votes) for rule in RULES}
        print(f"  sl_23 votes: {sl23_votes}")
        print(f"  sl_23 verdict per rule: {sl23_verdict_per_rule}")

        if all(v == "pass" for v in sl23_verdict_per_rule.values()):
            perturb_results = [
                f"{r.get('case_id', r['slide_id'])}: {r['majority_verdict']}"
                for r in perturbation_figures
            ]
            gonogo_md = (
                "# go/no-go Kill Signal — Phase 03 BLOCKED\n\n"
                "**Condition:** sl_23 (canonical oversized-arrows failure anchor) was accepted "
                "by the de-codex'd panel under ALL three candidate voting rules.\n\n"
                "## sl_23 Votes\n\n"
                f"```\n{json.dumps(sl23_votes, indent=2)}\n```\n\n"
                "## sl_23 Verdict per Rule\n\n"
                + "\n".join(f"- {rule}: {v}" for rule, v in sl23_verdict_per_rule.items())
                + "\n\n## Perturbation Detection Results\n\n"
                + "\n".join(f"- {r}" for r in perturb_results)
                + "\n\n## Finding\n\n"
                "The panel cannot reliably reject a known-bad figure (sl_23: oversized arrowheads) "
                "under any candidate voting rule. Phase 03 does not proceed until this is resolved.\n\n"
                f"**{FRAMING_CAVEAT}**\n"
            )
            (phase_dir / "02-GO-NOGO.md").write_text(gonogo_md, encoding="utf-8")
            print("  KILL SIGNAL: sl_23 accepted by all rules. Written 02-GO-NOGO.md. Exiting 2.")
            sys.exit(2)
        else:
            print(f"  go/no-go OK: at least one rule rejects sl_23.")
    else:
        print("  go/no-go check skipped (dry-run or sl_23 not in anchor results).")

    # -----------------------------------------------------------------
    # Step 7: Voting rule selection (PANEL-11)
    # -----------------------------------------------------------------
    print("Step 7: Voting rule selection")

    all_anchor_records = [r for r in (anchor_figures + perturbation_figures) if "anchor_type" in r]

    rule_metrics = [compute_voting_rule_metrics(rule, all_anchor_records) for rule in RULES]

    chosen_rule, chosen_rule_metrics, selection_status, selection_rationale = (
        select_voting_rule_from_metrics(rule_metrics)
    )

    # Pairwise judge agreement on baseline figures
    def _agree_pct(figs: list[dict], judge_a: str, judge_b: str) -> float | None:
        pairs = [(r["votes"].get(judge_a), r["votes"].get(judge_b)) for r in figs]
        valid = [(a, b) for a, b in pairs if a is not None and b is not None]
        if not valid:
            return None
        return sum(1 for a, b in valid if a == b) / len(valid)

    gemini_claude_agree = _agree_pct(baseline_figures, "gemini", "claude")
    gemini_qwen_agree = _agree_pct(baseline_figures, "gemini", "qwen")
    claude_qwen_agree = _agree_pct(baseline_figures, "claude", "qwen")

    # Drop-one sensitivity: for each judge removed, recount majority flips
    def _drop_one_sensitivity(figs: list[dict], drop_judge: str) -> int:
        flips = 0
        for r in figs:
            full_votes = r["votes"]
            reduced = {k: v for k, v in full_votes.items() if k != drop_judge}
            reduced_pass = sum(1 for v in reduced.values() if v == "pass")
            reduced_maj = "pass" if reduced_pass >= len(reduced) / 2 else "fail"
            if reduced_maj != r["majority_verdict"]:
                flips += 1
        return flips

    drop_gemini_flips = _drop_one_sensitivity(baseline_figures, "gemini")
    drop_claude_flips = _drop_one_sensitivity(baseline_figures, "claude")
    drop_qwen_flips = _drop_one_sensitivity(baseline_figures, "qwen")

    voting_rule_selection: dict[str, Any] = {
        "framing": FRAMING_CAVEAT,
        "candidate_rules": rule_metrics,
        "selection_status": selection_status,
        "full_anchor_set_criterion_satisfied": selection_status == "selected",
        "chosen_rule": chosen_rule,
        "chosen_rule_metrics": chosen_rule_metrics,
        "selection_rationale": selection_rationale,
        "pairwise_agreement": {
            "gemini_claude_agree_pct": gemini_claude_agree,
            "gemini_qwen_agree_pct": gemini_qwen_agree,
            "claude_qwen_agree_pct": claude_qwen_agree,
        },
        "drop_one_sensitivity": {
            "drop_gemini_verdict_flips": drop_gemini_flips,
            "drop_claude_verdict_flips": drop_claude_flips,
            "drop_qwen_verdict_flips": drop_qwen_flips,
        },
        "n_baseline_figures": len(baseline_figures),
        "n_anchor_figures": len(anchor_figures),
        "n_perturbation_figures": len(perturbation_figures),
    }
    write_json_atomic(phase_dir / "02-voting-rule-selection.json", voting_rule_selection)
    print(f"  Chosen rule: {chosen_rule} ({selection_status})")

    # -----------------------------------------------------------------
    # Step 8: DINOv2 floor re-derivation (PANEL-05)
    # -----------------------------------------------------------------
    print("Step 8: DINOv2 floor re-derivation")

    # Build verdict map from rescore output
    verdict_by_prefix: dict[str, str] = {}
    for r in all_figures:
        prefix = _slide_prefix(r["slide_id"])
        verdict_by_prefix[prefix] = r["majority_verdict"]

    # Build updated rows: join final-render dataset rows with new panel majority_verdict
    updated_rows: list[dict] = []
    for row in dataset_rows:
        if not row.get("is_final_render"):
            continue
        sid = row["slide_id"]
        prefix = _slide_prefix(sid)
        new_label = verdict_by_prefix.get(prefix, row.get("label", "unknown"))
        updated_row = dict(row)
        updated_row["label"] = new_label
        updated_rows.append(updated_row)

    # Get t_cov
    t_cov_source = "default_0.5"
    t_cov = 0.5
    cal_thresholds_path = args.calibration_thresholds
    if cal_thresholds_path is None:
        # Try default location in deck-dir
        default_cal = deck_dir / "calibration_threshold_analysis.json"
        if default_cal.exists():
            cal_thresholds_path = str(default_cal)

    if cal_thresholds_path and Path(cal_thresholds_path).exists():
        cal_data = json.loads(Path(cal_thresholds_path).read_text())
        # Try several key paths
        t_cov_val = (
            cal_data.get("coverage_threshold")
            or (cal_data.get("thresholds", {}).get("coverage", {}) or {}).get("value")
        )
        if t_cov_val is not None:
            t_cov = float(t_cov_val)
            t_cov_source = str(cal_thresholds_path)

    floor_result = select_stack_floor(updated_rows, t_cov, "dinov2_similarity")
    floor_value = floor_result.get("floor")
    floor_status = floor_result.get("status", "degenerate")

    dinov2_floor: dict[str, Any] = {
        "floor_value": floor_value,
        "floor_source": "v2-5/phase02-panel-sweep-on-v2.4-dinov2-scores",
        "floor_status": floor_status,
        "floor_direction": "reject-only — below floor -> auto-reject/redraw; never auto-accept on high DINOv2",
        "prior_v2_4_floor": 0.46,
        "t_cov_used": t_cov,
        "t_cov_source": t_cov_source,
        "note": FRAMING_CAVEAT,
        "select_stack_floor_detail": {k: v for k, v in floor_result.items() if k not in ("per_metric",)},
    }
    write_json_atomic(phase_dir / "02-dinov2-floor.json", dinov2_floor)
    print(f"  DINOv2 floor: {floor_value} ({floor_status}), t_cov={t_cov}")

    # -----------------------------------------------------------------
    # Step 9: Adjudication scaffold (PANEL-12)
    # -----------------------------------------------------------------
    print("Step 9: Adjudication scaffold")

    adjudication_entries: list[dict] = []
    for r in all_figures:
        if r.get("panel_split") or r.get("panel_vs_prior_disagree"):
            entry = build_adjudication_entry(
                slide_id=r["slide_id"],
                panel_verdict=r["majority_verdict"],
                prior_label=r.get("prior_label", "unknown"),
                panel_votes=r["votes"],
                panel_split=bool(r.get("panel_split")),
            )
            adjudication_entries.append(entry)

    write_json_atomic(phase_dir / "02-adjudication.json", {"entries": adjudication_entries})
    print(f"  Adjudication entries scaffolded: {len(adjudication_entries)}")

    # -----------------------------------------------------------------
    # Step 10: Validation report
    # -----------------------------------------------------------------
    print("Step 10: Validation report")

    baseline_pass_ct = sum(1 for r in baseline_figures if r["majority_verdict"] == "pass")
    baseline_fail_ct = len(baseline_figures) - baseline_pass_ct
    baseline_split_ct = sum(1 for r in baseline_figures if r.get("panel_split"))

    def _table_row(r: dict) -> str:
        votes = r.get("votes", {})
        return (
            f"| {r['slide_id']} "
            f"| {r.get('anchor_type', 'baseline')} "
            f"| {votes.get('gemini', '?')} "
            f"| {votes.get('claude', '?')} "
            f"| {votes.get('qwen', '?')} "
            f"| {r['majority_verdict']} "
            f"| {r.get('panel_vs_prior_disagree', False)} |"
        )

    anchor_table_rows = "\n".join(_table_row(r) for r in anchor_figures)
    perturb_table_rows = "\n".join(
        f"| {r.get('case_id', r['slide_id'])} | {r.get('injection_type', '?')} "
        f"| {r.get('votes', {}).get('gemini', '?')} "
        f"| {r.get('votes', {}).get('claude', '?')} "
        f"| {r.get('votes', {}).get('qwen', '?')} "
        f"| {'yes' if r['majority_verdict'] == 'fail' else 'no'} |"
        for r in perturbation_figures
    )

    rule_table_rows = "\n".join(
        f"| {m['rule']} "
        f"| {m['under_rejection_count']} "
        f"| {m['under_rejection_rate']} "
        f"| {m['over_rejection_count']} "
        f"| {m['over_rejection_rate']} "
        f"| {m['perturbation_detection_count']} "
        f"| {m['perturbation_detection_rate']} |"
        for m in rule_metrics
    )

    gonogo_status = "PASS — Phase 03 unblocked" if selection_status == "selected" else "FAIL — see 02-GO-NOGO.md"

    report_md = f"""\
# Phase 02 Validation Report

**{FRAMING_CAVEAT}**

---

## 1. Baseline Aggregation (Path A)

- Slides aggregated: {len(baseline_figures)}
- Pass: {baseline_pass_ct}, Fail: {baseline_fail_ct}
- 2-1 split count: {baseline_split_ct}

---

## 2. Anchor-Set Results (Path B)

| slide_id | anchor_type | gemini | claude | qwen | majority_verdict | vs_prior_disagree |
|----------|-------------|--------|--------|------|-----------------|-------------------|
{anchor_table_rows if anchor_table_rows else "| (dry-run — no live results) | | | | | | |"}

---

## 3. Perturbation Detection (Path C)

| case_id | injection_type | gemini | claude | qwen | detected? |
|---------|---------------|--------|--------|------|-----------|
{perturb_table_rows if perturb_table_rows else "| (dry-run — no live results) | | | | | |"}

---

## 4. Voting Rule Selection

| rule | under_reject_n | under_reject_rate | over_reject_n | over_reject_rate | perturb_detect_n | perturb_detect_rate |
|------|---------------|-------------------|--------------|-----------------|-----------------|---------------------|
{rule_table_rows}

**Chosen rule:** `{chosen_rule}` — {selection_rationale}

---

## 5. DINOv2 Floor

- Floor value: {floor_value}
- Status: {floor_status}
- Direction: reject-only — below floor → auto-reject/redraw; never auto-accept on high DINOv2
- t_cov used: {t_cov} (source: {t_cov_source})
- Prior v2.4 floor: 0.46

---

## 6. Pairwise Judge Diagnostics

| Pair | Agreement % |
|------|-------------|
| gemini ↔ claude | {gemini_claude_agree:.1%} |
| gemini ↔ qwen | {gemini_qwen_agree:.1%} |
| claude ↔ qwen | {claude_qwen_agree:.1%} |

Drop-one sensitivity (verdict flips on {len(baseline_figures)} baseline figures):
- Drop gemini: {drop_gemini_flips} flips
- Drop claude: {drop_claude_flips} flips
- Drop qwen: {drop_qwen_flips} flips

---

## 7. Panel Split-Rate

{baseline_split_ct} of {len(baseline_figures)} baseline figures ({baseline_split_ct / max(len(baseline_figures), 1):.1%}) had a 2-1 split.

---

## 8. Adjudication Surface

{len(adjudication_entries)} disagreements scaffolded in `02-adjudication.json`.

---

## 9. go/no-go Status

**{gonogo_status}**

Budget spent: ${budget.spend:.4f} of ${args.budget_cap:.2f} cap.
"""

    (phase_dir / "02-validation-report.md").write_text(report_md, encoding="utf-8")
    print(f"  Written: {phase_dir / '02-validation-report.md'}")

    if selection_status != "selected":
        gonogo_md = (
            "# go/no-go Kill Signal — Phase 03 BLOCKED\n\n"
            "**Condition:** no candidate voting rule satisfied the full Phase 02 "
            "anchor-set criterion: reject sl_11/sl_19/sl_23 and accept "
            "sl_05/sl_06/sl_10, while detecting perturbations.\n\n"
            "## Candidate Rule Metrics\n\n"
            "```json\n"
            f"{json.dumps(rule_metrics, indent=2)}\n"
            "```\n\n"
            "## Finding\n\n"
            f"{selection_rationale}\n\n"
            "Phase 03 must not proceed until the operator reviews the evidence and "
            "either changes the rule policy, changes the anchor-set requirement, or "
            "orders additional remediation.\n\n"
            f"**{FRAMING_CAVEAT}**\n"
        )
        (phase_dir / "02-GO-NOGO.md").write_text(gonogo_md, encoding="utf-8")
        print("  KILL SIGNAL: no candidate rule separates the full anchor set. Written 02-GO-NOGO.md. Exiting 2.")

    print(f"\nPhase 02 complete. Artifacts written to: {phase_dir}")
    print(f"  Total spend: ${budget.spend:.4f}")
    print(f"  go/no-go: {gonogo_status}")
    return 0 if selection_status == "selected" else 2


if __name__ == "__main__":
    sys.exit(main())
