#!/usr/bin/env python3
"""Run Qwen topic reconciliations as an isolated experiment.

This intentionally does not call the production 07D/07E/07F stage runners:
outputs stay under ``analysis/topic_reconciliation_experiments`` and provider
receipts use an experiment run id. If the lane looks good, production can later
wire the same reusable modules into the ICLR runner.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_INPUT_TOKEN_CEILING,
    DEFAULT_MAX_OUTPUT_TOKENS,
    ReconciliationInput,
    collect_reconciliation_input,
    estimate_tokens,
    sha256_text,
    workspace_relpath,
)
from conference_pipeline.analysis.topic_reference_repair import (
    RepairInput,
    apply_deterministic_reference_safety_fixes,
    build_reference_repair_prompt,
)
from conference_pipeline.analysis.topic_reference_validation import (
    ReferenceValidationResult,
    build_reference_ledger,
    format_reference_ledger_prompt_section,
    render_validation_markdown,
    validate_reconciliation_references,
)
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.research.deep_dive_lanes import DEFAULT_QWEN36_MODEL
from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.workflow_runs import write_json_atomic


SCHEMA_VERSION = 1
DEFAULT_EXPERIMENT_PREFIX = "qwen36-reconciliation-reference-repair"
DEFAULT_REPAIR_LEDGER_ENTRIES = 240
DEFAULT_FULL_REPAIR_TOKEN_CEILING = 90_000


@dataclass(frozen=True)
class TopicExperimentPaths:
    reconciliation_md: Path
    reconciliation_json: Path
    validation_json: Path
    validation_md: Path
    repair_md: Path
    repair_json: Path
    repair_validation_json: Path
    repair_validation_md: Path
    final_md: Path
    final_validation_json: Path
    final_validation_md: Path
    ledger_json: Path


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _topic_ids_from_source_packs(workspace: Path) -> list[str]:
    root = workspace / "analysis" / "narrative_source_packs"
    return [path.stem for path in sorted(root.glob("*.json"))]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip() + "\n"
    return stripped + ("\n" if stripped else "")


def _topic_paths(output_root: Path, topic_id: str) -> TopicExperimentPaths:
    topic_dir = output_root / topic_id
    return TopicExperimentPaths(
        reconciliation_md=topic_dir / "01-qwen-reconciliation.md",
        reconciliation_json=topic_dir / "01-qwen-reconciliation.json",
        validation_json=topic_dir / "02-reference-validation.json",
        validation_md=topic_dir / "02-reference-validation.md",
        repair_md=topic_dir / "03-qwen-reference-repair.md",
        repair_json=topic_dir / "03-qwen-reference-repair.json",
        repair_validation_json=topic_dir / "04-repair-validation.json",
        repair_validation_md=topic_dir / "04-repair-validation.md",
        final_md=topic_dir / "05-final-candidate.md",
        final_validation_json=topic_dir / "05-final-validation.json",
        final_validation_md=topic_dir / "05-final-validation.md",
        ledger_json=topic_dir / "reference-ledger.json",
    )


def _validation_payload(
    *,
    workspace: Path,
    generated_at: str,
    markdown_path: Path,
    markdown_text: str,
    result: ReferenceValidationResult,
) -> dict[str, Any]:
    payload = result.to_dict()
    payload.update(
        {
            "generated_at": generated_at,
            "markdown_path": workspace_relpath(workspace, markdown_path),
            "markdown_sha256": sha256_text(markdown_text),
        }
    )
    return payload


async def _run_researcher(
    *,
    researcher: LocalLLMResearcher,
    prompt: str,
    workspace: Path,
    run_id: str,
    stage_id: str,
) -> str:
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=stage_id,
        caller=__file__,
    ):
        return await researcher.run(prompt)


def _write_summary_markdown(
    *,
    path: Path,
    receipt: dict[str, Any],
    topic_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# Qwen Reconciliation Dry Run",
        "",
        f"Status: `{receipt['status']}`",
        f"Run id: `{receipt['run_id']}`",
        f"Model: `{receipt['model']}`",
        f"Topics: `{receipt['completed_topic_count']} / {receipt['topic_count']}`",
        f"Initial invalid refs: `{receipt['checks']['initial_invalid_reference_count']}`",
        f"Final invalid refs: `{receipt['checks']['final_invalid_reference_count']}`",
        f"Repair attempted: `{receipt['checks']['repair_attempted_topic_count']}` topics",
        f"Elapsed: `{receipt['elapsed_seconds']}` seconds",
        "",
        "## Topic Results",
        "",
        "| Topic | Initial | Final | Initial Invalid | Final Invalid | Repair | Final Candidate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in topic_results:
        lines.append(
            "| "
            f"`{item['topic_id']}` | "
            f"`{item['initial_validation']['status']}` | "
            f"`{item['final_validation']['status']}` | "
            f"`{item['initial_validation']['invalid_reference_count']}` | "
            f"`{item['final_validation']['invalid_reference_count']}` | "
            f"`{item['repair_action']}` | "
            f"`{item['final_markdown_path']}` |"
        )
    if receipt.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in receipt["blockers"]:
            lines.append(
                f"- `{blocker.get('topic_id', 'unknown')}` `{blocker.get('code')}`: "
                f"{blocker.get('reason')}"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _topic_sidecar(
    *,
    workspace: Path,
    generated_at: str,
    run_id: str,
    model: str,
    inputs: ReconciliationInput,
    text: str,
    markdown_path: Path,
    reused: bool,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.experiment.topic_reconciliation",
        "owner_id": "experiment.topic_reconciliation_qwen36",
        "status": "succeeded",
        "generated_at": generated_at,
        "run_id": run_id,
        "model": model,
        "topic_id": inputs.topic_id,
        "title": inputs.title,
        "input_digest_sha256": inputs.input_digest_sha256,
        "input_token_estimate": inputs.input_token_estimate,
        "output_token_estimate": estimate_tokens(text),
        "markdown_path": workspace_relpath(workspace, markdown_path),
        "markdown_sha256": sha256_text(text),
        "source_pack_path": workspace_relpath(workspace, inputs.source_pack_path),
        "paper_summary_path": (
            workspace_relpath(workspace, inputs.paper_summary_path)
            if inputs.paper_summary_path
            else None
        ),
        "transcript_summary_path": (
            workspace_relpath(workspace, inputs.transcript_summary_path)
            if inputs.transcript_summary_path
            else None
        ),
        "deep_dive_paths": [
            workspace_relpath(workspace, path) for path in inputs.deep_dive_paths
        ],
        "deep_dive_report_count": len(inputs.deep_dive_paths),
        "reused": reused,
    }


def _build_compact_reference_repair_prompt(
    *,
    repair_input: RepairInput,
    max_ledger_entries: int,
) -> str:
    return (
        "You are repairing citation/reference bindings in one conference topic "
        "reconciliation brief. The deterministic validator found bad references. "
        "Repair only references and nearby labels; do not add new claims or new "
        "sections. Use canonical ids from the ledger exactly. If no safe id exists, "
        "remove the citation rather than inventing one. Return the full repaired "
        "markdown only, with no code fence and no explanation.\n\n"
        f"Topic id: {repair_input.topic_id}\n"
        f"Topic title: {repair_input.title}\n\n"
        "## Validator Result\n\n"
        f"{json.dumps(repair_input.initial_validation.to_dict(), ensure_ascii=True, indent=2, sort_keys=True)}\n\n"
        f"{format_reference_ledger_prompt_section(repair_input.ledger, max_entries=max_ledger_entries)}\n\n"
        "## Markdown To Repair\n\n"
        f"{repair_input.original_markdown}\n"
    )


def run_experiment(
    *,
    workspace: Path,
    output_root: Path | None,
    run_id: str,
    model: str,
    topics: list[str] | None,
    limit: int | None,
    input_token_ceiling: int,
    max_output_tokens: int,
    repair_ledger_entries: int,
    full_repair_token_ceiling: int,
    resume: bool,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if output_root is None:
        output_root = (
            workspace
            / "analysis"
            / "topic_reconciliation_experiments"
            / f"{DEFAULT_EXPERIMENT_PREFIX}-{run_id}"
        )
    else:
        output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_topics = topics or _topic_ids_from_source_packs(workspace)
    if limit is not None:
        selected_topics = selected_topics[:limit]
    if not selected_topics:
        raise RuntimeError("narrative_source_packs_missing")

    generated_at = utc_now()
    started = time.monotonic()
    researcher = LocalLLMResearcher(
        smart_model=model,
        fast_model=model,
        max_tokens=max_output_tokens,
    )

    topic_results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for index, topic_id in enumerate(selected_topics, start=1):
        topic_started = time.monotonic()
        paths = _topic_paths(output_root, topic_id)
        paths.reconciliation_md.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(selected_topics)}] {topic_id}: collecting input", flush=True)
        try:
            inputs = collect_reconciliation_input(
                workspace,
                topic_id=topic_id,
                input_token_ceiling=input_token_ceiling,
            )
            ledger = build_reference_ledger(workspace, topic_id)
            write_json_atomic(paths.ledger_json, ledger.to_dict())

            existing_sidecar: dict[str, Any] | None = None
            if resume and paths.reconciliation_json.exists():
                try:
                    existing_sidecar = json.loads(
                        paths.reconciliation_json.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    existing_sidecar = None

            if (
                resume
                and existing_sidecar
                and existing_sidecar.get("input_digest_sha256")
                == inputs.input_digest_sha256
                and paths.final_md.exists()
                and paths.final_validation_json.exists()
            ):
                try:
                    final_validation_payload = json.loads(
                        paths.final_validation_json.read_text(encoding="utf-8")
                    )
                    initial_validation_payload = json.loads(
                        paths.validation_json.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    final_validation_payload = {}
                    initial_validation_payload = {}
                final_text = paths.final_md.read_text(
                    encoding="utf-8", errors="replace"
                )
                if (
                    final_validation_payload.get("status") == "succeeded"
                    and final_validation_payload.get("markdown_sha256")
                    == sha256_text(final_text)
                ):
                    result = {
                        "schema_version": SCHEMA_VERSION,
                        "topic_id": topic_id,
                        "title": ledger.title,
                        "status": "succeeded",
                        "elapsed_seconds": round(time.monotonic() - topic_started, 3),
                        "reconciliation_markdown_path": workspace_relpath(
                            workspace, paths.reconciliation_md
                        ),
                        "final_markdown_path": workspace_relpath(
                            workspace, paths.final_md
                        ),
                        "ledger_path": workspace_relpath(workspace, paths.ledger_json),
                        "repair_action": "reused_final_candidate",
                        "deterministic_reference_fix_count": 0,
                        "input_token_estimate": inputs.input_token_estimate,
                        "output_token_estimate": estimate_tokens(final_text),
                        "initial_validation": initial_validation_payload,
                        "final_validation": final_validation_payload,
                        "reused": True,
                    }
                    topic_results.append(result)
                    print(
                        f"[{index}/{len(selected_topics)}] {topic_id}: "
                        "reused final validated candidate",
                        flush=True,
                    )
                    continue

            if (
                resume
                and existing_sidecar
                and existing_sidecar.get("input_digest_sha256")
                == inputs.input_digest_sha256
                and paths.reconciliation_md.exists()
            ):
                text = paths.reconciliation_md.read_text(
                    encoding="utf-8", errors="replace"
                )
                reused = True
                print(f"[{index}/{len(selected_topics)}] {topic_id}: reused reconciliation", flush=True)
            else:
                print(
                    f"[{index}/{len(selected_topics)}] {topic_id}: Qwen reconciliation "
                    f"input ~{inputs.input_token_estimate:,} tokens",
                    flush=True,
                )
                text = asyncio.run(
                    _run_researcher(
                        researcher=researcher,
                        prompt=inputs.prompt,
                        workspace=workspace,
                        run_id=run_id,
                        stage_id="experiment-topic-reconciliation-qwen36",
                    )
                )
                text = _strip_code_fence(text)
                paths.reconciliation_md.write_text(text, encoding="utf-8")
                reused = False

            reconciliation_sidecar = _topic_sidecar(
                workspace=workspace,
                generated_at=generated_at,
                run_id=run_id,
                model=model,
                inputs=inputs,
                text=text,
                markdown_path=paths.reconciliation_md,
                reused=reused,
            )
            write_json_atomic(paths.reconciliation_json, reconciliation_sidecar)

            initial_validation = validate_reconciliation_references(
                text=text,
                ledger=ledger,
                require_citations=True,
            )
            write_json_atomic(
                paths.validation_json,
                _validation_payload(
                    workspace=workspace,
                    generated_at=generated_at,
                    markdown_path=paths.reconciliation_md,
                    markdown_text=text,
                    result=initial_validation,
                ),
            )
            paths.validation_md.write_text(
                render_validation_markdown(initial_validation),
                encoding="utf-8",
            )

            repaired_text = text
            repair_action = "skipped_already_valid"
            repair_prompt_tokens = 0
            deterministic_fix_count = 0
            repair_output_token_estimate = 0
            repair_provider_receipts_before = list(researcher.provider_receipt_paths)
            if initial_validation.status != "succeeded":
                repair_action = "llm_repair"
                repair_input = RepairInput(
                    topic_id=topic_id,
                    title=ledger.title,
                    original_markdown_path=paths.reconciliation_md,
                    original_markdown=text,
                    initial_validation=initial_validation,
                    ledger=ledger,
                    prompt="",
                    input_digest_sha256="",
                    input_token_estimate=0,
                )
                repair_prompt = build_reference_repair_prompt(
                    repair_input=repair_input,
                    original_reconciliation_prompt=inputs.prompt,
                    max_ledger_entries=repair_ledger_entries,
                )
                repair_prompt_mode = "full_evidence"
                repair_prompt_tokens = estimate_tokens(repair_prompt)
                if repair_prompt_tokens > full_repair_token_ceiling:
                    repair_prompt = _build_compact_reference_repair_prompt(
                        repair_input=repair_input,
                        max_ledger_entries=repair_ledger_entries,
                    )
                    repair_prompt_mode = "compact_ledger"
                    repair_prompt_tokens = estimate_tokens(repair_prompt)
                if repair_prompt_tokens > input_token_ceiling:
                    raise RuntimeError(
                        "topic_reference_repair_input_too_large:"
                        f"{topic_id}:{repair_prompt_tokens}>{input_token_ceiling}"
                    )
                print(
                    f"[{index}/{len(selected_topics)}] {topic_id}: repairing "
                    f"{initial_validation.invalid_reference_count} invalid refs "
                    f"with {repair_prompt_mode} prompt "
                    f"(~{repair_prompt_tokens:,} tokens)",
                    flush=True,
                )
                repaired_text = asyncio.run(
                    _run_researcher(
                        researcher=researcher,
                        prompt=repair_prompt,
                        workspace=workspace,
                        run_id=run_id,
                        stage_id="experiment-topic-reference-repair-qwen36",
                    )
                )
                repaired_text = _strip_code_fence(repaired_text)
                repaired_text, deterministic_fix_count = apply_deterministic_reference_safety_fixes(
                    text=repaired_text,
                    ledger=ledger,
                )
                repair_output_token_estimate = estimate_tokens(repaired_text)
                paths.repair_md.write_text(repaired_text, encoding="utf-8")

            repair_validation = validate_reconciliation_references(
                text=repaired_text,
                ledger=ledger,
                require_citations=True,
            )
            if initial_validation.status != "succeeded":
                write_json_atomic(
                    paths.repair_validation_json,
                    _validation_payload(
                        workspace=workspace,
                        generated_at=generated_at,
                        markdown_path=paths.repair_md,
                        markdown_text=repaired_text,
                        result=repair_validation,
                    ),
                )
                paths.repair_validation_md.write_text(
                    render_validation_markdown(repair_validation),
                    encoding="utf-8",
                )
                repair_sidecar = {
                    "schema_version": SCHEMA_VERSION,
                    "receipt_type": "conference_pipeline.experiment.topic_reference_repair",
                    "owner_id": "experiment.topic_reference_repair_qwen36",
                    "status": repair_validation.status,
                    "generated_at": generated_at,
                    "run_id": run_id,
                    "model": model,
                    "topic_id": topic_id,
                    "title": ledger.title,
                    "repair_action": repair_action,
                    "repair_prompt_mode": repair_prompt_mode,
                    "input_token_estimate": repair_prompt_tokens,
                    "output_token_estimate": repair_output_token_estimate,
                    "deterministic_reference_fix_count": deterministic_fix_count,
                    "initial_validation": initial_validation.to_dict(),
                    "repaired_validation": repair_validation.to_dict(),
                    "repair_markdown_path": workspace_relpath(workspace, paths.repair_md),
                    "repair_markdown_sha256": sha256_text(repaired_text),
                    "provider_receipt_paths": list(
                        researcher.provider_receipt_paths[
                            len(repair_provider_receipts_before) :
                        ]
                    ),
                }
                write_json_atomic(paths.repair_json, repair_sidecar)

            paths.final_md.write_text(repaired_text, encoding="utf-8")
            final_validation = repair_validation
            write_json_atomic(
                paths.final_validation_json,
                _validation_payload(
                    workspace=workspace,
                    generated_at=generated_at,
                    markdown_path=paths.final_md,
                    markdown_text=repaired_text,
                    result=final_validation,
                ),
            )
            paths.final_validation_md.write_text(
                render_validation_markdown(final_validation),
                encoding="utf-8",
            )

            result = {
                "schema_version": SCHEMA_VERSION,
                "topic_id": topic_id,
                "title": ledger.title,
                "status": final_validation.status,
                "elapsed_seconds": round(time.monotonic() - topic_started, 3),
                "reconciliation_markdown_path": workspace_relpath(
                    workspace, paths.reconciliation_md
                ),
                "final_markdown_path": workspace_relpath(workspace, paths.final_md),
                "ledger_path": workspace_relpath(workspace, paths.ledger_json),
                "repair_action": repair_action,
                "deterministic_reference_fix_count": deterministic_fix_count,
                "input_token_estimate": inputs.input_token_estimate,
                "output_token_estimate": estimate_tokens(repaired_text),
                "initial_validation": initial_validation.to_dict(),
                "final_validation": final_validation.to_dict(),
                "reused": reused,
            }
            topic_results.append(result)
            if final_validation.status != "succeeded":
                blockers.extend(final_validation.blockers)
            print(
                f"[{index}/{len(selected_topics)}] {topic_id}: final "
                f"{final_validation.status}, invalid refs "
                f"{initial_validation.invalid_reference_count} -> "
                f"{final_validation.invalid_reference_count}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - experiment boundary.
            blocker = {
                "code": str(exc).split(":", 1)[0] or "experiment_topic_failed",
                "topic_id": topic_id,
                "reason": str(exc),
                "retryable": True,
            }
            blockers.append(blocker)
            print(f"[{index}/{len(selected_topics)}] {topic_id}: blocked: {exc}", flush=True)

    checks = {
        "topic_count": len(selected_topics),
        "completed_topic_count": len(topic_results),
        "initial_invalid_reference_count": sum(
            int(item["initial_validation"]["invalid_reference_count"])
            for item in topic_results
        ),
        "final_invalid_reference_count": sum(
            int(item["final_validation"]["invalid_reference_count"])
            for item in topic_results
        ),
        "final_unknown_reference_count": sum(
            int(item["final_validation"]["unknown_reference_count"])
            for item in topic_results
        ),
        "final_title_mismatch_count": sum(
            int(item["final_validation"]["title_mismatch_count"])
            for item in topic_results
        ),
        "final_bare_reference_warning_count": sum(
            int(item["final_validation"]["bare_reference_warning_count"])
            for item in topic_results
        ),
        "repair_attempted_topic_count": sum(
            1 for item in topic_results if item["repair_action"] == "llm_repair"
        ),
        "final_succeeded_topic_count": sum(
            1 for item in topic_results if item["final_validation"]["status"] == "succeeded"
        ),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.experiment.qwen_topic_reconciliation_dryrun",
        "owner_id": "experiment.qwen36_topic_reconciliation_reference_repair",
        "status": "blocked" if blockers else "succeeded",
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "model": model,
        "input_token_ceiling": input_token_ceiling,
        "max_output_tokens": max_output_tokens,
        "repair_ledger_entries": repair_ledger_entries,
        "full_repair_token_ceiling": full_repair_token_ceiling,
        "topic_count": len(selected_topics),
        "completed_topic_count": len(topic_results),
        "output_root": workspace_relpath(workspace, output_root),
        "topic_result_paths": [
            workspace_relpath(workspace, _topic_paths(output_root, item["topic_id"]).final_md)
            for item in topic_results
        ],
        "provider_calls_made": True,
        "live_provider_calls_made": True,
        "paid_provider_calls_made": False,
        "usage": researcher.usage(),
        "checks": checks,
        "blockers": blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    write_json_atomic(output_root / "summary.json", receipt)
    write_json_atomic(output_root / "topic-results.json", {"topics": topic_results})
    _write_summary_markdown(
        path=output_root / "summary.md",
        receipt=receipt,
        topic_results=topic_results,
    )
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all-topic Qwen reconciliation as a non-production experiment."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id", default=f"{DEFAULT_EXPERIMENT_PREFIX}-{safe_run_id()}")
    parser.add_argument("--model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--input-token-ceiling", type=int, default=DEFAULT_INPUT_TOKEN_CEILING)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--repair-ledger-entries", type=int, default=DEFAULT_REPAIR_LEDGER_ENTRIES)
    parser.add_argument(
        "--full-repair-token-ceiling",
        type=int,
        default=DEFAULT_FULL_REPAIR_TOKEN_CEILING,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_experiment(
        workspace=args.workspace,
        output_root=args.output_root,
        run_id=args.run_id,
        model=args.model,
        topics=args.topics,
        limit=args.limit,
        input_token_ceiling=args.input_token_ceiling,
        max_output_tokens=args.max_output_tokens,
        repair_ledger_entries=args.repair_ledger_entries,
        full_repair_token_ceiling=args.full_repair_token_ceiling,
        resume=args.resume,
    )
    if args.summary:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
