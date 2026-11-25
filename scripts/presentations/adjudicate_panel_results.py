"""Codex adjudication layer over Phase 02 panel evidence.

This script is intentionally local-only: it reads existing Gemini/Claude/Qwen
panel artifacts and writes richer adjudication decisions. It does not call live
providers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, ".")

from conference_pipeline.workflow_runs import write_json_atomic


FRAMING_CAVEAT = (
    "These results measure consistency between the de-codex'd 3-judge vision-LLM "
    "panel and prior AI-panel labels. They do NOT constitute accuracy against human "
    "ground truth. Human ground-truth labeling is deferred to a separate milestone."
)

POLICY_VERSION = "codex_adjudication_v1"
JUDGES = ("gemini", "claude", "qwen")
TAXONOMY = {
    "accept",
    "accept_with_notes",
    "revise",
    "redraw",
    "needs_human",
    "benchmark_mismatch",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_votes(votes: dict[str, Any]) -> dict[str, str]:
    missing = [judge for judge in JUDGES if judge not in votes]
    if missing:
        raise ValueError(f"Missing required judge vote(s): {', '.join(missing)}")
    normalized: dict[str, str] = {}
    for judge in JUDGES:
        vote = votes[judge]
        if vote not in {"pass", "fail"}:
            raise ValueError(f"Invalid vote for {judge}: {vote!r}")
        normalized[judge] = vote
    return normalized


def expected_label_for_record(record: dict[str, Any]) -> str | None:
    anchor_type = record.get("anchor_type")
    if anchor_type == "accept":
        return "pass"
    if anchor_type in {"failure", "perturbation"}:
        return "fail"
    prior_label = record.get("prior_label")
    if prior_label in {"pass", "fail"}:
        return prior_label
    return None


def _confidence(*, unanimous: bool, conflict: bool, decision: str) -> str:
    if decision in {"needs_human", "benchmark_mismatch"}:
        return "low" if conflict and not unanimous else "medium"
    if unanimous and not conflict:
        return "high"
    return "medium"


def _next_action(decision: str, failure_modes: list[str]) -> str:
    modes = ", ".join(failure_modes) if failure_modes else "the recorded concern"
    if decision == "accept":
        return "Keep the current render; no redraw needed."
    if decision == "accept_with_notes":
        return f"Keep the current render, but carry notes about {modes} into the handoff."
    if decision == "revise":
        return f"Run a targeted revision using feedback about {modes}."
    if decision == "redraw":
        return f"Force redraw/regeneration; target {modes}."
    if decision == "needs_human":
        return "Hold out of automatic success and request human review before production adoption."
    if decision == "benchmark_mismatch":
        return "Hold out of automatic success and review the prior/anchor label or rubric."
    raise ValueError(f"Unknown decision: {decision}")


def _rationale(
    *,
    record: dict[str, Any],
    expected_label: str | None,
    decision: str,
    pass_count: int,
    fail_count: int,
    conflict: bool,
) -> str:
    slide_id = record.get("slide_id", record.get("case_id", "unknown"))
    majority = record.get("majority_verdict")
    panel_path = record.get("panel_path", "unknown")
    if record.get("anchor_type") == "perturbation" or panel_path == "perturbation":
        return (
            f"{slide_id} is a by-construction perturbation; the panel cast "
            f"{fail_count} fail vote(s), so the safe adjudication is redraw."
        )
    if expected_label == "pass" and fail_count == 3:
        return (
            f"{slide_id} was expected to pass, but all three judges failed it. "
            "Treat this as an anchor/benchmark mismatch or stale-label signal, not as a silent rejection."
        )
    if expected_label == "fail" and pass_count >= 2:
        return (
            f"{slide_id} was expected to fail, but the panel majority passed it. "
            "Treat this as a benchmark/rubric mismatch before using it as an automatic gate."
        )
    if majority == "fail":
        return (
            f"{slide_id} received a fail majority ({fail_count}/3 fail). "
            "The adjudication routes it to repair rather than acceptance."
        )
    if conflict:
        return (
            f"{slide_id} conflicts with the expected or prior label. "
            "Do not promote it to automatic acceptance."
        )
    if pass_count == 3:
        return f"{slide_id} received unanimous pass votes with no label conflict."
    return (
        f"{slide_id} received a pass majority with split votes or recorded concerns; "
        "accept it only with notes."
    )


def adjudicate_figure(record: dict[str, Any]) -> dict[str, Any]:
    votes = validate_votes(record.get("votes", {}))
    counts = Counter(votes.values())
    pass_count = counts["pass"]
    fail_count = counts["fail"]
    unanimous = pass_count in {0, 3}
    expected_label = expected_label_for_record(record)
    majority = record.get("majority_verdict")
    if majority not in {"pass", "fail"}:
        raise ValueError(f"Invalid majority_verdict for {record.get('slide_id')}: {majority!r}")

    failure_modes = sorted(set(record.get("failure_modes") or []))
    concrete_visual_modes = {
        "wrong-source",
        "missing-components",
        "invented-components",
        "legibility",
        "style-drift",
    }
    has_concrete_visual_concern = bool(concrete_visual_modes.intersection(failure_modes))
    conflict = bool(expected_label and majority != expected_label)
    panel_path = record.get("panel_path", "unknown")
    anchor_type = record.get("anchor_type")

    if anchor_type == "perturbation" or panel_path == "perturbation":
        decision = "redraw" if fail_count >= 1 else "needs_human"
    elif expected_label == "pass" and fail_count == 3:
        decision = "benchmark_mismatch"
    elif expected_label == "fail" and pass_count >= 2:
        decision = "benchmark_mismatch"
    elif majority == "fail":
        decision = "redraw" if fail_count == 3 or expected_label == "fail" else "revise"
    elif conflict:
        decision = "benchmark_mismatch"
    elif pass_count == 3 and not failure_modes:
        decision = "accept"
    elif has_concrete_visual_concern and pass_count == 2:
        decision = "revise"
    else:
        decision = "accept_with_notes"

    result = {
        "id": record.get("case_id") or record.get("slide_id"),
        "slide_id": record.get("slide_id"),
        "case_id": record.get("case_id"),
        "panel_path": panel_path,
        "anchor_type": anchor_type,
        "prior_label": record.get("prior_label"),
        "expected_label": expected_label,
        "majority_verdict": majority,
        "judge_votes": votes,
        "vote_summary": {"pass": pass_count, "fail": fail_count},
        "panel_split": bool(record.get("panel_split")),
        "panel_vs_prior_disagree": bool(record.get("panel_vs_prior_disagree")),
        "prior_label_conflict": conflict,
        "failure_modes": failure_modes,
        "estimator_findings": record.get("estimator_findings") or {},
        "decision": decision,
        "confidence": _confidence(unanimous=unanimous, conflict=conflict, decision=decision),
        "rationale": _rationale(
            record=record,
            expected_label=expected_label,
            decision=decision,
            pass_count=pass_count,
            fail_count=fail_count,
            conflict=conflict,
        ),
        "next_action": _next_action(decision, failure_modes),
        "framing": FRAMING_CAVEAT,
    }
    validate_decision_record(result)
    return result


def validate_decision_record(record: dict[str, Any]) -> None:
    decision = record.get("decision")
    if decision not in TAXONOMY:
        raise ValueError(f"Decision outside taxonomy: {decision!r}")
    if record.get("prior_label_conflict") and decision in {"accept", "accept_with_notes"}:
        raise ValueError(
            f"Conflicting record {record.get('id')} cannot be promoted to {decision}"
        )


def validate_decisions(decisions: list[dict[str, Any]]) -> None:
    for decision in decisions:
        validate_decision_record(decision)


def _qwen_summary(qwen_profile: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "model",
        "slide_count",
        "baseline_slide_count",
        "anchor_slide_count",
        "qwen_accept_rate",
        "gemini_accept_rate",
        "claude_accept_rate",
        "qwen_agrees_gemini_pct",
        "qwen_agrees_claude_pct",
        "anchor_defect_detected",
        "panel11_precondition_satisfied",
    ]
    return {key: qwen_profile.get(key) for key in keys if key in qwen_profile}


def build_payload(
    *,
    panel_rescore: dict[str, Any],
    voting_selection: dict[str, Any],
    adjudication_scaffold: dict[str, Any],
    qwen_profile: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    decisions = [adjudicate_figure(record) for record in panel_rescore.get("figures", [])]
    validate_decisions(decisions)
    counts = Counter(decision["decision"] for decision in decisions)
    conflict_ids = [
        decision["id"]
        for decision in decisions
        if decision["decision"] in {"needs_human", "benchmark_mismatch"}
    ]
    payload = {
        "policy_version": POLICY_VERSION,
        "generated_at": generated_at or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "framing": FRAMING_CAVEAT,
        "input_summary": {
            "panel_figure_count": len(panel_rescore.get("figures", [])),
            "adjudication_scaffold_entries": len(adjudication_scaffold.get("entries", [])),
            "panel_spend_usd": panel_rescore.get("spend_usd"),
        },
        "voting_rule_selection": {
            "selection_status": voting_selection.get("selection_status"),
            "chosen_rule": voting_selection.get("chosen_rule"),
            "full_anchor_set_criterion_satisfied": voting_selection.get(
                "full_anchor_set_criterion_satisfied"
            ),
            "selection_rationale": voting_selection.get("selection_rationale"),
            "candidate_rules": voting_selection.get("candidate_rules", []),
        },
        "qwen_profile_summary": _qwen_summary(qwen_profile),
        "summary": {
            "decision_counts": {key: counts.get(key, 0) for key in sorted(TAXONOMY)},
            "needs_human_or_benchmark_mismatch": conflict_ids,
        },
        "decisions": decisions,
    }
    if (
        payload["voting_rule_selection"]["selection_status"] == "no_separating_rule"
        and payload["voting_rule_selection"]["chosen_rule"] is not None
    ):
        raise ValueError("no_separating_rule cannot have a chosen_rule")
    return payload


def _md_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(payload: dict[str, Any]) -> str:
    counts = payload["summary"]["decision_counts"]
    lines = [
        "# Phase 03 Codex Adjudication Decisions",
        "",
        f"**Policy version:** `{payload['policy_version']}`",
        f"**Generated:** `{payload['generated_at']}`",
        "",
        f"**Caveat:** {payload['framing']}",
        "",
        "## Voting Rule Context",
        "",
        f"- Selection status: `{payload['voting_rule_selection']['selection_status']}`",
        f"- Chosen rule: `{payload['voting_rule_selection']['chosen_rule']}`",
        "- Forced binary rule: `false`",
        "",
        "## Decision Counts",
        "",
        "| Decision | Count |",
        "|----------|-------|",
    ]
    for decision in sorted(TAXONOMY):
        lines.append(f"| `{decision}` | {counts.get(decision, 0)} |")

    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| ID | Path | Expected | Votes | Decision | Confidence | Next Action |",
            "|----|------|----------|-------|----------|------------|-------------|",
        ]
    )
    for item in payload["decisions"]:
        votes = ", ".join(f"{judge}:{item['judge_votes'][judge]}" for judge in JUDGES)
        lines.append(
            "| "
            f"{_md_escape(item['id'])} | "
            f"{_md_escape(item['panel_path'])} | "
            f"{_md_escape(item['expected_label'])} | "
            f"{_md_escape(votes)} | "
            f"`{_md_escape(item['decision'])}` | "
            f"{_md_escape(item['confidence'])} | "
            f"{_md_escape(item['next_action'])} |"
        )

    holdouts = payload["summary"]["needs_human_or_benchmark_mismatch"]
    lines.extend(["", "## Holdouts", ""])
    if holdouts:
        for item_id in holdouts:
            lines.append(f"- `{item_id}`")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-rescore", type=Path, required=True)
    parser.add_argument("--voting-selection", type=Path, required=True)
    parser.add_argument("--adjudication-scaffold", type=Path, required=True)
    parser.add_argument("--qwen-profile", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_payload(
        panel_rescore=load_json(args.panel_rescore),
        voting_selection=load_json(args.voting_selection),
        adjudication_scaffold=load_json(args.adjudication_scaffold),
        qwen_profile=load_json(args.qwen_profile),
    )
    write_json_atomic(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        "Wrote Codex adjudication decisions: "
        f"{args.output_json} and {args.output_md}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
