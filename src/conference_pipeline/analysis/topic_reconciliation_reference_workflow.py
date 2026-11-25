"""Governed topic reconciliation and reference-validation workflow.

This module packages the production sequence that proved reliable during the
ICLR 2026 analysis run. Local models may write or help repair topic synthesis,
but the deterministic reference validator remains the final authority before
final narrative or slide synthesis can consume the reconciliation outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_INPUT_TOKEN_CEILING,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_QWEN36_MODEL,
    _topic_ids_from_source_packs,
    run_topic_reconciliation,
    strip_foreign_title_aliases,
    strip_unledgered_citations,
)
from conference_pipeline.analysis.topic_reference_candidate_resolution import (
    resolve_topic_reference_candidates,
)
from conference_pipeline.analysis.topic_reference_repair import (
    apply_deterministic_reference_safety_fixes,
    run_topic_reference_repair,
)
from conference_pipeline.analysis.topic_reference_validation import (
    DEFAULT_RUN_ID,
    build_reference_ledger,
    run_topic_reference_validation,
    workspace_relpath,
)
from conference_pipeline.receipt_lineage import stamp_substep_receipt, unbind_substep_receipt
from conference_pipeline.workflow_runs import write_json_atomic

OWNER_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW = (
    "analysis.topic_reconciliation_reference_workflow"
)
RECEIPT_TYPE_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW = (
    "conference_pipeline.topic_reconciliation_reference_workflow"
)
SCHEMA_VERSION = 1
DEFAULT_REPAIR_INPUT_TOKEN_CEILING = 300_000


class TopicReconciliationReferenceWorkflowError(RuntimeError):
    """Raised when the governed reconciliation workflow cannot run."""


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07H-topic-reconciliation-reference-workflow.json"
    )


def _step_summary(name: str, owner_id: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "owner_id": owner_id,
        "status": receipt.get("status"),
        "receipt_path": receipt.get("receipt_path"),
        "topic_count": receipt.get("topic_count"),
        "completed_topic_count": receipt.get("completed_topic_count"),
        "checks": receipt.get("checks", {}),
        "blocker_count": len(receipt.get("blockers") or []),
    }


def _blocked_topics(validation_receipt: Mapping[str, Any]) -> list[str]:
    topics = {
        str(blocker["topic_id"])
        for blocker in validation_receipt.get("blockers", [])
        if isinstance(blocker, Mapping) and blocker.get("topic_id")
    }
    return sorted(topics)


def _final_checks(validation_receipt: Mapping[str, Any]) -> dict[str, Any]:
    checks = validation_receipt.get("checks")
    if isinstance(checks, Mapping):
        return dict(checks)
    return {}


def _zero_reference_blockers(checks: Mapping[str, Any]) -> bool:
    return (
        int(checks.get("invalid_reference_count", 0)) == 0
        and int(checks.get("unknown_reference_count", 0)) == 0
        and int(checks.get("title_mismatch_count", 0)) == 0
        and int(checks.get("bare_reference_warning_count", 0)) == 0
    )


def _workflow_blockers(
    *,
    final_validation_receipt: Mapping[str, Any],
    steps: list[dict[str, Any]],
    mode: str,
    candidate_resolution_enabled: bool,
    broad_repair_enabled: bool,
) -> list[dict[str, Any]]:
    blockers = list(final_validation_receipt.get("blockers") or [])
    checks = _final_checks(final_validation_receipt)
    if final_validation_receipt.get("status") != "succeeded" or not _zero_reference_blockers(checks):
        blockers.append(
            {
                "code": "topic_reconciliation_reference_workflow_blocked",
                "reason": (
                    "Final deterministic 07E validation did not report zero invalid, "
                    "unknown, title-mismatch, and bare-reference counts."
                ),
                "retryable": True,
            }
        )
    if mode != "live" and blockers and (candidate_resolution_enabled or broad_repair_enabled):
        blockers.append(
            {
                "code": "topic_reconciliation_reference_workflow_requires_live_repair",
                "reason": (
                    "Deterministic fixes did not clear all blockers; candidate or broad "
                    "LLM repair requires --mode live."
                ),
                "retryable": True,
            }
        )
    if not steps:
        blockers.append(
            {
                "code": "topic_reconciliation_reference_workflow_empty",
                "reason": "No reconciliation workflow steps ran.",
                "retryable": False,
            }
        )
    return blockers


def _write_interim_reference_workflow_receipt(
    *,
    workspace: Path,
    run_id: str,
    mode: str,
    model: str,
    run_reconciliation: bool,
    run_candidate_resolution: bool,
    run_broad_repair: bool,
    apply_repairs: bool,
    reconciliation_receipt: Mapping[str, Any],
    steps: list[dict[str, Any]],
    generated_at: str,
    started: float,
) -> None:
    """Write an interim 07H receipt reflecting reconciliation-only progress.

    Fix option 5: a mid-run kill after 07D-topic-reconciliation but before the
    final 07E validation chain must not leave a 07H receipt that still shows a
    prior all-blocked snapshot. The final write at the end of
    :func:`run_topic_reconciliation_reference_workflow` remains authoritative.
    """

    receipt_path = _receipt_path(workspace, run_id)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW,
        "owner_id": OWNER_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW,
        "status": "in_progress",
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "topic_count": reconciliation_receipt.get("topic_count"),
        "completed_topic_count": reconciliation_receipt.get("completed_topic_count"),
        "run_reconciliation": run_reconciliation,
        "run_candidate_resolution": run_candidate_resolution,
        "run_broad_repair": run_broad_repair,
        "apply_repairs": apply_repairs,
        "source_packs_enriched": reconciliation_receipt.get("source_packs_enriched"),
        "final_validation_receipt_path": None,
        "steps": steps,
        "checks": {
            "final_validation_status": "in_progress",
            "final_completed_topic_count": reconciliation_receipt.get("completed_topic_count"),
            "final_topic_count": reconciliation_receipt.get("topic_count"),
            "step_count": len(steps),
            "deterministic_safety_fix_step_ran": False,
            "candidate_resolution_step_ran": False,
            "broad_repair_step_ran": False,
        },
        "blockers": list(reconciliation_receipt.get("blockers") or []),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    write_json_atomic(receipt_path, receipt)


def run_topic_reconciliation_reference_workflow(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    model: str = DEFAULT_QWEN36_MODEL,
    topics: list[str] | None = None,
    input_token_ceiling: int = DEFAULT_INPUT_TOKEN_CEILING,
    repair_input_token_ceiling: int = DEFAULT_REPAIR_INPUT_TOKEN_CEILING,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    resume: bool = False,
    run_reconciliation: bool = True,
    run_candidate_resolution: bool = True,
    run_broad_repair: bool = True,
    apply_repairs: bool = True,
    enrich_source_packs: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "topic-reconciliation-reference-workflow")
    generated_at = utc_now()
    steps: list[dict[str, Any]] = []

    if run_reconciliation:
        reconciliation_receipt = run_topic_reconciliation(
            workspace=workspace,
            run_id=run_id,
            mode=mode,
            model=model,
            topics=topics,
            input_token_ceiling=input_token_ceiling,
            max_output_tokens=max_output_tokens,
            resume=resume,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07D-topic-reconciliation",
                "analysis.topic_reconciliation",
                reconciliation_receipt,
            )
        )
        _write_interim_reference_workflow_receipt(
            workspace=workspace,
            run_id=run_id,
            mode=mode,
            model=model,
            run_reconciliation=run_reconciliation,
            run_candidate_resolution=run_candidate_resolution,
            run_broad_repair=run_broad_repair,
            apply_repairs=apply_repairs,
            reconciliation_receipt=reconciliation_receipt,
            steps=steps,
            generated_at=generated_at,
            started=started,
        )

    validation_receipt = run_topic_reference_validation(
        workspace=workspace,
        run_id=run_id,
        topics=topics,
        enrich_source_packs=enrich_source_packs,
    )
    steps.append(
        _step_summary(
            "07E-topic-reference-validation-initial",
            "analysis.topic_reference_validation",
            validation_receipt,
        )
    )

    if validation_receipt.get("status") != "succeeded":
        blocked_topics = _blocked_topics(validation_receipt) or topics
        deterministic_repair_receipt = run_topic_reference_repair(
            workspace=workspace,
            run_id=run_id,
            mode="dry-run",
            model=model,
            topics=blocked_topics,
            input_token_ceiling=repair_input_token_ceiling,
            max_output_tokens=max_output_tokens,
            apply_repaired=apply_repairs,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07F-topic-reference-deterministic-safety-fixes",
                "analysis.topic_reference_repair",
                deterministic_repair_receipt,
            )
        )
        validation_receipt = run_topic_reference_validation(
            workspace=workspace,
            run_id=run_id,
            topics=topics,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07E-topic-reference-validation-after-deterministic-repair",
                "analysis.topic_reference_validation",
                validation_receipt,
            )
        )

    if (
        validation_receipt.get("status") != "succeeded"
        and run_candidate_resolution
        and mode == "live"
    ):
        blocked_topics = _blocked_topics(validation_receipt) or topics
        candidate_receipt = resolve_topic_reference_candidates(
            workspace=workspace,
            run_id=run_id,
            mode=mode,
            model=model,
            topics=blocked_topics,
            dry_run=False,
            apply=apply_repairs,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07G-topic-reference-candidate-resolution",
                "analysis.topic_reference_candidate_resolution",
                candidate_receipt,
            )
        )
        validation_receipt = run_topic_reference_validation(
            workspace=workspace,
            run_id=run_id,
            topics=topics,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07E-topic-reference-validation-after-candidate-resolution",
                "analysis.topic_reference_validation",
                validation_receipt,
            )
        )

    if validation_receipt.get("status") != "succeeded" and run_broad_repair and mode == "live":
        blocked_topics = _blocked_topics(validation_receipt) or topics
        broad_repair_receipt = run_topic_reference_repair(
            workspace=workspace,
            run_id=run_id,
            mode=mode,
            model=model,
            topics=blocked_topics,
            input_token_ceiling=repair_input_token_ceiling,
            max_output_tokens=max_output_tokens,
            apply_repaired=apply_repairs,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07F-topic-reference-broad-llm-repair",
                "analysis.topic_reference_repair",
                broad_repair_receipt,
            )
        )
        validation_receipt = run_topic_reference_validation(
            workspace=workspace,
            run_id=run_id,
            topics=topics,
            enrich_source_packs=enrich_source_packs,
        )
        steps.append(
            _step_summary(
                "07E-topic-reference-validation-final",
                "analysis.topic_reference_validation",
                validation_receipt,
            )
        )

    # Last resort, and only after 07F/07G have had their chance: whatever
    # citation repair could not canonicalize is a fabricated id -- typically a
    # transcript the model quoted out of a deep-dive report, whose ledger id
    # the report does not carry, so the model invented one. Repair CAN fix a
    # merely malformed id, so this must never run before it: doing so deletes
    # citations that would have been recovered.
    unledgered_removed: list[dict[str, Any]] = []
    if validation_receipt.get("status") != "succeeded" and mode == "live":
        blocked_topics = _blocked_topics(validation_receipt) or topics
        for topic_id in blocked_topics:
            markdown_path = (
                workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"
            )
            if not markdown_path.exists():
                continue
            text = markdown_path.read_text(encoding="utf-8", errors="replace")
            ledger = build_reference_ledger(workspace, topic_id)
            cleaned, removed = strip_unledgered_citations(text, ledger=ledger)
            # A wrong label is not a wrong citation: strip the borrowed alias
            # and keep the reference, rather than dropping the whole line.
            cleaned, relabelled = strip_foreign_title_aliases(cleaned, ledger=ledger)
            removed = [*removed, *relabelled]
            # The deterministic safety fixes (backticked shorthand ids such
            # as `thecvf.com_..._EDGE_x` -> `paper_abstract:thecvf.com_..._EDGE_x`)
            # are only kept by 07F when the topic then validates clean, so on
            # a topic that also carried fabricated ids they were discarded.
            # Now that the fabrications are gone, canonicalize once more.
            cleaned, canonicalized = apply_deterministic_reference_safety_fixes(
                text=cleaned, ledger=ledger
            )
            if canonicalized:
                removed.append(
                    {
                        "action": "deterministic_safety_fixes_after_strip",
                        "change_count": int(canonicalized),
                    }
                )
            if not removed:
                continue
            markdown_path.write_text(cleaned, encoding="utf-8")
            unledgered_removed.extend(
                {"topic_id": topic_id, **entry} for entry in removed
            )
        if unledgered_removed:
            validation_receipt = run_topic_reference_validation(
                workspace=workspace,
                run_id=run_id,
                topics=topics,
                enrich_source_packs=enrich_source_packs,
            )
            steps.append(
                _step_summary(
                    "07E-topic-reference-validation-after-unledgered-strip",
                    "analysis.topic_reference_validation",
                    validation_receipt,
                )
            )

    blockers = _workflow_blockers(
        final_validation_receipt=validation_receipt,
        steps=steps,
        mode=mode,
        candidate_resolution_enabled=run_candidate_resolution,
        broad_repair_enabled=run_broad_repair,
    )
    status = "blocked" if blockers else "succeeded"
    receipt_path = _receipt_path(workspace, run_id)
    final_checks = _final_checks(validation_receipt)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW,
        "owner_id": OWNER_TOPIC_RECONCILIATION_REFERENCE_WORKFLOW,
        "status": status,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "topic_count": final_checks.get("topic_count") or len(topics or []),
        "completed_topic_count": final_checks.get("completed_topic_count"),
        "run_reconciliation": run_reconciliation,
        "run_candidate_resolution": run_candidate_resolution,
        "run_broad_repair": run_broad_repair,
        "apply_repairs": apply_repairs,
        "source_packs_enriched": validation_receipt.get("source_packs_enriched"),
        "final_validation_receipt_path": validation_receipt.get("receipt_path"),
        "steps": steps,
        "checks": {
            "final_validation_status": validation_receipt.get("status"),
            "final_completed_topic_count": final_checks.get("completed_topic_count"),
            "final_topic_count": final_checks.get("topic_count"),
            "final_ledger_entry_count": final_checks.get("ledger_entry_count"),
            "final_citation_count": final_checks.get("citation_count"),
            "final_invalid_reference_count": final_checks.get("invalid_reference_count"),
            "final_unknown_reference_count": final_checks.get("unknown_reference_count"),
            "final_title_mismatch_count": final_checks.get("title_mismatch_count"),
            "final_bare_reference_warning_count": final_checks.get(
                "bare_reference_warning_count"
            ),
            "step_count": len(steps),
            "deterministic_safety_fix_step_ran": any(
                step["name"] == "07F-topic-reference-deterministic-safety-fixes"
                for step in steps
            ),
            "candidate_resolution_step_ran": any(
                step["name"] == "07G-topic-reference-candidate-resolution"
                for step in steps
            ),
            "broad_repair_step_ran": any(
                step["name"] == "07F-topic-reference-broad-llm-repair"
                for step in steps
            ),
            # What the last-resort ledger guard removed after repair gave up.
            "unledgered_citation_removal_count": len(unledgered_removed),
        },
        "unledgered_citations_removed": unledgered_removed,
        "blockers": blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    stamp_substep_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        substep_id="topic-reconciliation-reference-workflow",
        complete=not topics or set(topics) >= set(_topic_ids_from_source_packs(workspace)),
    )
    write_json_atomic(receipt_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run governed topic reconciliation, reference repair, and final validation."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--input-token-ceiling", type=int, default=DEFAULT_INPUT_TOKEN_CEILING)
    parser.add_argument(
        "--repair-input-token-ceiling",
        type=int,
        default=DEFAULT_REPAIR_INPUT_TOKEN_CEILING,
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-reconciliation", action="store_true")
    parser.add_argument("--skip-candidate-resolution", action="store_true")
    parser.add_argument("--skip-broad-repair", action="store_true")
    parser.add_argument("--no-apply-repairs", action="store_true")
    parser.add_argument("--no-enrich-source-packs", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        receipt = run_topic_reconciliation_reference_workflow(
            workspace=args.workspace,
            run_id=args.run_id,
            mode=args.mode,
            model=args.model,
            topics=args.topics,
            input_token_ceiling=args.input_token_ceiling,
            repair_input_token_ceiling=args.repair_input_token_ceiling,
            max_output_tokens=args.max_output_tokens,
            resume=args.resume,
            run_reconciliation=not args.skip_reconciliation,
            run_candidate_resolution=not args.skip_candidate_resolution,
            run_broad_repair=not args.skip_broad_repair,
            apply_repairs=not args.no_apply_repairs,
            enrich_source_packs=not args.no_enrich_source_packs,
        )
    except TopicReconciliationReferenceWorkflowError as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2, sort_keys=True))
        return 1
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "topic_count": receipt.get("topic_count"),
                    "completed_topic_count": receipt.get("completed_topic_count"),
                    "checks": receipt.get("checks"),
                    "blockers": receipt.get("blockers"),
                    "receipt_path": receipt.get("receipt_path"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if receipt.get("status") == "succeeded" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_main())
