#!/usr/bin/env python3
"""Run GPT-5.5 reference-aware topic reconciliations as an experiment lane.

Outputs stay under ``analysis/topic_reconciliation_experiments`` so this does
not overwrite the production Qwen reconciliation files. The runner validates
canonical references after each topic and writes an aggregate receipt for review.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_INPUT_TOKEN_CEILING,
    DEFAULT_MAX_OUTPUT_TOKENS,
    ReconciliationInput,
    TopicReconciliationError,
    collect_reconciliation_input,
    estimate_tokens,
    sha256_text,
    workspace_relpath,
)
from conference_pipeline.analysis.topic_reference_validation import (
    ReferenceValidationResult,
    build_reference_ledger,
    render_validation_markdown,
    validate_reconciliation_references,
)
from conference_pipeline.provider_receipts import (
    openai_chat_completions_create,
    provider_receipt_context,
)
from conference_pipeline.workflow_runs import write_json_atomic


SCHEMA_VERSION = 1
OWNER_ID = "experiment.topic_reconciliation_gpt55_reference_aware"
RECEIPT_TYPE = "conference_pipeline.experiment.topic_reconciliation_gpt55"
DEFAULT_MODEL = "gpt-5.5"  # model-literal-ok: this experiment lane is defined by GPT-5.5 (OWNER_ID, RECEIPT_TYPE, STAGE_ID all name it); swapping the model would make it a different experiment
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_STAGE_ID = "07D-topic-reconciliation-experiment-gpt55-reference-aware"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
        stripped = "\n".join(lines).strip()
    return stripped + ("\n" if stripped else "")


def _output_paths(workspace: Path, topic_id: str) -> tuple[Path, Path, Path, Path]:
    root = workspace / "analysis" / "topic_reconciliation_experiments"
    return (
        root / f"{topic_id}__gpt55_reference_aware.md",
        root / f"{topic_id}__gpt55_reference_aware.json",
        root / f"{topic_id}__gpt55_reference_aware_reference_validation.json",
        root / f"{topic_id}__gpt55_reference_aware_reference_validation.md",
    )


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07D-topic-reconciliation-experiment-gpt55-reference-aware-all.json"
    )


def _summary_path(workspace: Path, run_id: str) -> Path:
    return (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07D-topic-reconciliation-experiment-gpt55-reference-aware-all.md"
    )


def _reference_aware_prompt(inputs: ReconciliationInput) -> str:
    return (
        inputs.prompt
        + "\n\n"
        "## GPT-5.5 Reference-Aware Expansion Requirements\n\n"
        "- Produce a publishable analyst brief, not a terse checkpoint summary.\n"
        "- Preserve concrete paper titles, talk titles, session/workshop labels, "
        "speaker names, timestamps, paths, and short quotes when they appear in "
        "the supplied materials.\n"
        "- Include enough source-detail blocks or citation-rich bullets that a "
        "human can trace the strongest claims back to canonical ids.\n"
        "- Cite only canonical ids from the Reference Ledger, exactly as written.\n"
        "- Do not cite a canonical id beside a title, speaker, or talk label unless "
        "that association is present in the supplied evidence or ledger row.\n"
        "- Prefer 8-14 slide-worthy claims with evidence, tensions, and caveats; "
        "avoid padding with generic conference commentary.\n"
    )


def _usage_payload(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _can_reuse(
    *,
    sidecar_path: Path,
    validation_path: Path,
    markdown_path: Path,
    inputs: ReconciliationInput,
    model: str,
) -> bool:
    sidecar = _load_json(sidecar_path)
    validation = _load_json(validation_path)
    if not sidecar or not validation or not markdown_path.exists():
        return False
    if sidecar.get("status") != "succeeded":
        return False
    if sidecar.get("model") != model:
        return False
    if sidecar.get("input_digest_sha256") != inputs.input_digest_sha256:
        return False
    return validation.get("status") == "succeeded"


def _write_summary_markdown(
    *,
    path: Path,
    receipt: dict[str, Any],
    topic_results: list[dict[str, Any]],
) -> None:
    lines = [
        "# GPT-5.5 Reference-Aware Topic Reconciliation",
        "",
        f"Status: `{receipt['status']}`",
        f"Model: `{receipt['model']}`",
        f"Topics: `{receipt['completed_topic_count']} / {receipt['topic_count']}`",
        f"Reused: `{receipt['reused_topic_count']}`",
        f"Elapsed: `{receipt['elapsed_seconds']}` seconds",
        f"Estimated cost: `${receipt['usage'].get('estimated_cost_usd', 0.0):.6f}`",
        "",
        "## Topic Results",
        "",
        "| Topic | Status | Validation | Input Tokens | Output Tokens | Citations | Invalid Refs | Cost | Output |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in topic_results:
        validation = item.get("reference_validation", {})
        usage = item.get("usage", {})
        cost = item.get("estimated_cost_usd")
        cost_text = "" if cost is None else f"${float(cost):.6f}"
        lines.append(
            "| "
            f"`{item['topic_id']}` | "
            f"`{item['status']}` | "
            f"`{validation.get('status', 'unknown')}` | "
            f"`{item.get('input_token_estimate', 0)}` | "
            f"`{usage.get('completion_tokens') or item.get('output_token_estimate', 0)}` | "
            f"`{validation.get('citation_count', 0)}` | "
            f"`{validation.get('invalid_reference_count', 0)}` | "
            f"`{cost_text}` | "
            f"`{item.get('markdown_path', '')}` |"
        )
    if receipt.get("blockers"):
        lines.extend(["", "## Blockers", ""])
        for blocker in receipt["blockers"]:
            lines.append(
                f"- `{blocker.get('topic_id', 'unknown')}` `{blocker.get('code')}`: "
                f"{blocker.get('reason')}"
            )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_topic(
    *,
    client: Any | None,
    workspace: Path,
    run_id: str,
    model: str,
    inputs: ReconciliationInput,
    max_output_tokens: int,
    resume: bool,
    force: bool,
) -> dict[str, Any]:
    markdown_path, sidecar_path, validation_json_path, validation_md_path = _output_paths(
        workspace, inputs.topic_id
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    prompt = _reference_aware_prompt(inputs)
    prompt_sha256 = sha256_text(prompt)
    reused = False
    elapsed = 0.0
    usage: dict[str, int] = {}
    receipt_path: str | None = None
    estimated_cost: float | None = None

    if (
        resume
        and not force
        and _can_reuse(
            sidecar_path=sidecar_path,
            validation_path=validation_json_path,
            markdown_path=markdown_path,
            inputs=inputs,
            model=model,
        )
    ):
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        reused = True
    else:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - environment guard.
                raise TopicReconciliationError("openai_not_available") from exc
            client = OpenAI()
        started = time.monotonic()
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior technical conference analyst. Write "
                        "reference-faithful synthesis from supplied evidence only. "
                        "Never invent citations, titles, speakers, URLs, or ids."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": int(max_output_tokens),
        }
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=DEFAULT_STAGE_ID,
            caller=__file__,
        ):
            provider_result = openai_chat_completions_create(
                client,
                caller=__file__,
                **request_kwargs,
            )
        elapsed = round(time.monotonic() - started, 3)
        response = provider_result.response
        text = _strip_code_fence(_response_text(response))
        if not text:
            raise TopicReconciliationError(f"empty_model_response:{inputs.topic_id}")
        usage = _usage_payload(response)
        receipt_path = provider_result.receipt.receipt_path
        estimated_cost = provider_result.receipt.estimated_cost_usd
        markdown_path.write_text(text, encoding="utf-8")

    ledger = build_reference_ledger(workspace, inputs.topic_id)
    validation_result = validate_reconciliation_references(text=text, ledger=ledger)
    validation_payload = _validation_payload(
        workspace=workspace,
        generated_at=generated_at,
        markdown_path=markdown_path,
        markdown_text=text,
        result=validation_result,
    )
    write_json_atomic(validation_json_path, validation_payload)
    validation_md_path.write_text(
        render_validation_markdown(validation_result), encoding="utf-8"
    )

    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "owner_id": OWNER_ID,
        "status": validation_result.status,
        "generated_at": generated_at,
        "run_id": run_id,
        "model": model,
        "topic_id": inputs.topic_id,
        "title": inputs.title,
        "input_digest_sha256": inputs.input_digest_sha256,
        "prompt_sha256": prompt_sha256,
        "input_token_estimate": inputs.input_token_estimate,
        "prompt_token_estimate": estimate_tokens(prompt),
        "output_token_estimate": estimate_tokens(text),
        "markdown_path": workspace_relpath(workspace, markdown_path),
        "markdown_sha256": sha256_text(text),
        "reference_validation_path": workspace_relpath(workspace, validation_json_path),
        "reference_validation_status": validation_result.status,
        "reference_validation_counts": {
            "ledger_entry_count": validation_result.ledger_entry_count,
            "citation_count": validation_result.citation_count,
            "invalid_reference_count": validation_result.invalid_reference_count,
            "unknown_reference_count": validation_result.unknown_reference_count,
            "title_mismatch_count": validation_result.title_mismatch_count,
            "bare_reference_warning_count": validation_result.bare_reference_warning_count,
        },
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
        "usage": usage,
        "provider_receipt_path": receipt_path,
        "estimated_cost_usd": estimated_cost,
        "elapsed_seconds": elapsed,
        "reused": reused,
    }
    write_json_atomic(sidecar_path, sidecar)
    return sidecar | {"reference_validation": validation_result.to_dict()}


def run(
    *,
    workspace: Path,
    run_id: str,
    model: str,
    topics: list[str] | None,
    input_token_ceiling: int,
    max_output_tokens: int,
    resume: bool,
    force: bool,
    provider_cost_approved: bool,
) -> dict[str, Any]:
    if not provider_cost_approved and not resume:
        raise TopicReconciliationError("provider_cost_not_approved")
    workspace = workspace.resolve()
    topic_ids = topics or _topic_ids_from_source_packs(workspace)
    if not topic_ids:
        raise TopicReconciliationError("narrative_source_packs_missing")

    started = time.monotonic()
    generated_at = utc_now()
    client: Any | None = None
    results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []

    for index, topic_id in enumerate(topic_ids, start=1):
        print(f"[{index}/{len(topic_ids)}] {topic_id}", flush=True)
        try:
            inputs = collect_reconciliation_input(
                workspace,
                topic_id=topic_id,
                input_token_ceiling=input_token_ceiling,
            )
            result = _run_topic(
                client=client,
                workspace=workspace,
                run_id=run_id,
                model=model,
                inputs=inputs,
                max_output_tokens=max_output_tokens,
                resume=resume,
                force=force,
            )
            results.append(result)
            validation = result.get("reference_validation", {})
            print(
                "  -> "
                f"{result['status']} reused={result['reused']} "
                f"citations={validation.get('citation_count', 0)} "
                f"invalid={validation.get('invalid_reference_count', 0)} "
                f"elapsed={result.get('elapsed_seconds', 0)}s",
                flush=True,
            )
            if result["status"] != "succeeded":
                blockers.append(
                    {
                        "topic_id": topic_id,
                        "code": "reference_validation_blocked",
                        "reason": f"status={result['status']}",
                        "retryable": True,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - aggregate receipt boundary.
            blockers.append(
                {
                    "topic_id": topic_id,
                    "code": str(exc).split(":", 1)[0] or "topic_failed",
                    "reason": str(exc),
                    "retryable": True,
                }
            )
            print(f"  -> blocked: {exc}", flush=True)

    elapsed = round(time.monotonic() - started, 3)
    total_estimated_cost = sum(
        float(result.get("estimated_cost_usd") or 0.0) for result in results
    )
    usage = {
        "prompt_tokens": sum(
            int((result.get("usage") or {}).get("prompt_tokens") or 0)
            for result in results
        ),
        "completion_tokens": sum(
            int((result.get("usage") or {}).get("completion_tokens") or 0)
            for result in results
        ),
        "total_tokens": sum(
            int((result.get("usage") or {}).get("total_tokens") or 0)
            for result in results
        ),
        "estimated_cost_usd": total_estimated_cost,
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "owner_id": OWNER_ID,
        "status": "blocked" if blockers else "succeeded",
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "model": model,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(results),
        "reused_topic_count": sum(1 for result in results if result.get("reused")),
        "elapsed_seconds": elapsed,
        "usage": usage,
        "outputs": results,
        "blockers": blockers,
    }
    receipt_path = _receipt_path(workspace, run_id)
    summary_path = _summary_path(workspace, run_id)
    write_json_atomic(receipt_path, receipt)
    _write_summary_markdown(path=summary_path, receipt=receipt, topic_results=results)
    receipt["receipt_path"] = workspace_relpath(workspace, receipt_path)
    receipt["summary_path"] = workspace_relpath(workspace, summary_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--input-token-ceiling", type=int, default=DEFAULT_INPUT_TOKEN_CEILING)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--provider-cost-approved", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run(
        workspace=args.workspace,
        run_id=args.run_id,
        model=args.model,
        topics=args.topics,
        input_token_ceiling=args.input_token_ceiling,
        max_output_tokens=args.max_output_tokens,
        resume=args.resume,
        force=args.force,
        provider_cost_approved=args.provider_cost_approved,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
