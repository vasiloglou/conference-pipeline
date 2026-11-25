"""Plan, skill, and receipt audits for the unified stage script registry."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.registry_diagnostics import (
    aggregate_agent_diagnostic,
    attach_agent_diagnostic,
)
from conference_pipeline.stage_script_registry import (
    DEFAULT_STAGE_SCRIPT_REGISTRY_PATH,
    REPO_ROOT,
    diagnostic_profile_for_script_id,
    forbidden_script_ids_for_stage,
    load_stage_script_registry,
    refs_for_script_id,
    refs_for_script_ids,
    registry_sha256,
    required_script_ids_for_stage,
    script_ref_lookup,
    usage_doc_refs_for_script_id,
    validate_registered_diagnostic_profiles,
    validate_registered_script_existence,
    validate_registered_usage_doc_refs,
)
from conference_pipeline.stage_contracts import resolve_stage_contract_binding
from conference_pipeline.stage_owner_audit import _json_policy_units, _text_policy_units

PLAN_AUDIT_FILE_SUFFIXES = frozenset({".md", ".json", ".yaml", ".yml"})
SCRIPT_REF_RE = re.compile(r"(?<![\w./-])(scripts/[A-Za-z0-9_./-]+\.py)")
SUCCEEDED_STATUSES = frozenset({"complete", "pass", "passed", "succeeded"})
NEGATIVE_POLICY_TERMS = (
    "reject",
    "rejected",
    "forbidden",
    "banned",
    "must not",
    "do not",
    "cannot",
    "not use",
    "not approved",
    "deprecated",
    "legacy",
)
def audit_stage_script_registry_integrity(
    *,
    registry_path: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Validate registry schema, script ids, and required file existence."""

    registry = load_stage_script_registry(registry_path)
    blockers = validate_registered_script_existence(registry, repo_root=repo_root)
    blockers.extend(validate_registered_diagnostic_profiles(registry))
    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
        "script_registry_sha256": registry_sha256(registry_path),
    }
    return attach_agent_diagnostic(
        report,
        tool="stage_script_registry.integrity",
        checked_inputs={
            "registry_path": report["registry_path"],
            "repo_root": str(repo_root),
        },
        accepted_evidence=[
            "Registry schema loaded.",
            "All existence: required script paths were checked.",
            "All registry script diagnostic profiles were checked.",
        ],
        pass_next_actions=[
            "Proceed to plan, skill, docs, or run evidence verification."
        ],
        blocked_next_actions=[
            "Repair the malformed or missing registry entry before running downstream audits."
        ],
    )


def audit_stage_script_usage_docs(
    docs_root: str | Path,
    *,
    registry_path: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Verify registry-owned usage documentation refs resolve and stay policy-safe."""

    registry = load_stage_script_registry(registry_path)
    blockers = validate_registered_usage_doc_refs(
        registry,
        repo_root=repo_root,
        docs_root=Path(docs_root),
    )
    rejected_refs = refs_for_script_ids(
        registry,
        _deprecated_and_forbidden_script_ids(registry),
    )
    rejected_hits = _rejected_owner_hits(
        Path(docs_root),
        rejected_refs,
        allow_historical_reference_documents=True,
    )
    if rejected_hits:
        blockers.append(
            {
                "code": "stage_script_registry_forbidden_script_used_in_docs",
                "hits": rejected_hits,
                "retryable": False,
            }
        )
    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
        "script_registry_sha256": registry_sha256(registry_path),
        "docs_root": str(docs_root),
    }
    return attach_agent_diagnostic(
        report,
        tool="stage_script_registry.usage_docs",
        checked_inputs={
            "registry_path": report["registry_path"],
            "docs_root": str(docs_root),
        },
        accepted_evidence=[
            "Every registry script entry was checked for usage_doc_refs.",
            "Referenced documentation files and anchors were resolved.",
            "Deprecated and forbidden scripts were checked for negative-policy context.",
        ],
        pass_next_actions=[
            "Proceed to strict skill and generated-plan verification."
        ],
        blocked_next_actions=[
            "Repair the cited usage_doc_refs or documentation anchors, then rerun with --docs-root."
        ],
    )


def audit_plan_stage_script_registry(
    plan_root: str | Path,
    *,
    task_ids: Iterable[str] | None = None,
    registry_path: str | Path | None = None,
    require_registry_digest: bool = False,
    require_verifier_command: bool = False,
    require_usage_doc_refs: bool = False,
) -> dict[str, object]:
    """Audit generated planning docs against the unified stage script registry."""

    root = Path(plan_root)
    registry = load_stage_script_registry(registry_path)
    selected_task_ids = tuple(
        str(task_id)
        for task_id in (
            task_ids
            if task_ids is not None
            else registry.get("planning_required_task_ids") or sorted(registry["task_bindings"])
        )
    )
    text = _collect_text(root)
    blockers: list[dict[str, object]] = []

    for task_id in selected_task_ids:
        binding = (registry.get("task_bindings") or {}).get(task_id)
        if not isinstance(binding, Mapping):
            blockers.append(
                {
                    "code": "stage_script_registry_task_missing",
                    "task_id": task_id,
                    "retryable": False,
                }
            )
            continue
        missing_script_ids = [
            script_id
            for script_id in binding.get("required_script_ids") or ()
            if not _any_ref_in_text(registry, str(script_id), text)
        ]
        if missing_script_ids:
            blockers.append(
                {
                    "code": "stage_script_registry_required_script_missing_from_plan",
                    "task_id": task_id,
                    "missing_script_ids": missing_script_ids,
                    "missing_refs": list(refs_for_script_ids(registry, missing_script_ids)),
                    "retryable": False,
                }
            )
        if require_usage_doc_refs:
            missing_usage_doc_refs = [
                script_id
                for script_id in binding.get("required_script_ids") or ()
                if not _any_usage_doc_ref_in_text(registry, str(script_id), text)
            ]
            if missing_usage_doc_refs:
                blockers.append(
                    {
                        "code": "stage_script_registry_usage_doc_ref_missing_from_plan",
                        "task_id": task_id,
                        "missing_script_ids": missing_usage_doc_refs,
                        "missing_usage_doc_refs": list(
                            _usage_doc_refs_for_script_ids(registry, missing_usage_doc_refs)
                        ),
                        "retryable": False,
                    }
                )
        missing_terms = [
            str(term)
            for term in binding.get("required_command_terms") or ()
            if str(term) not in text
        ]
        if missing_terms:
            blockers.append(
                {
                    "code": "stage_script_registry_required_term_missing_from_plan",
                    "task_id": task_id,
                    "missing_terms": missing_terms,
                    "retryable": False,
                }
            )

    rejected_refs = refs_for_script_ids(
        registry,
        _deprecated_and_forbidden_script_ids(registry),
    )
    rejected_hits = _rejected_owner_hits(root, rejected_refs)
    if rejected_hits:
        blockers.append(
            {
                "code": "stage_script_registry_forbidden_script_used_in_plan",
                "hits": rejected_hits,
                "retryable": False,
            }
        )

    digest = registry_sha256(registry_path)
    if require_registry_digest and digest not in text:
        blockers.append(
            {
                "code": "stage_script_registry_digest_missing_from_plan",
                "expected_sha256": digest,
                "retryable": False,
            }
        )
    if require_verifier_command and "scripts/verify_stage_script_usage.py" not in text:
        blockers.append(
            {
                "code": "stage_script_registry_verifier_command_missing_from_plan",
                "required_ref": "scripts/verify_stage_script_usage.py",
                "retryable": False,
            }
        )

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
        "script_registry_sha256": digest,
        "checked_task_ids": list(selected_task_ids),
    }
    return attach_agent_diagnostic(
        report,
        tool="stage_script_registry.plan",
        checked_inputs={
            "plan_root": str(plan_root),
            "registry_path": report["registry_path"],
            "checked_task_ids": list(selected_task_ids),
            "require_registry_digest": require_registry_digest,
            "require_verifier_command": require_verifier_command,
            "require_usage_doc_refs": require_usage_doc_refs,
        },
        accepted_evidence=[
            "Generated plan text was scanned for required script refs.",
            "Generated plan text was scanned for forbidden completion owners.",
            "Registry digest, verifier command, and usage docs were checked when requested.",
        ],
        pass_next_actions=[
            "The generated plan satisfies the selected registry task bindings."
        ],
        blocked_next_actions=[
            "Update the generated plan owner-command table and proof gates, then rerun the same verifier command."
        ],
    )


def audit_skill_stage_script_contracts(
    skills_root: str | Path,
    *,
    registry_path: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Verify planning-skill script mentions are represented in the registry."""

    registry = load_stage_script_registry(registry_path)
    root = Path(skills_root)
    blockers: list[dict[str, object]] = []
    lookup = script_ref_lookup(registry)

    contracts = registry.get("skill_contracts") or {}
    for skill_name, contract in contracts.items():
        if not isinstance(contract, Mapping):
            blockers.append(
                {
                    "code": "stage_script_registry_malformed_skill_contract",
                    "skill": skill_name,
                    "retryable": False,
                }
            )
            continue
        skill_root = _resolve_skill_contract_root(
            root,
            str(skill_name),
            str(contract.get("skill_root") or ""),
            repo_root=repo_root,
        )
        text = _collect_text(skill_root)
        if not text:
            blockers.append(
                {
                    "code": "stage_script_registry_skill_text_missing",
                    "skill": skill_name,
                    "skill_root": str(skill_root),
                    "retryable": False,
                }
            )
            continue
        for script_id in contract.get("required_script_ids") or ():
            if not _any_ref_in_text(registry, str(script_id), text):
                blockers.append(
                    {
                        "code": "stage_script_registry_required_skill_script_missing",
                        "skill": skill_name,
                        "script_id": script_id,
                        "refs": list(refs_for_script_id(registry, str(script_id))),
                        "retryable": False,
                    }
                )
            if not _any_usage_doc_ref_in_text(registry, str(script_id), text):
                blockers.append(
                    {
                        "code": "stage_script_registry_required_skill_usage_doc_missing",
                        "skill": skill_name,
                        "script_id": script_id,
                        "usage_doc_refs": list(
                            usage_doc_refs_for_script_id(registry, str(script_id))
                        ),
                        "retryable": False,
                    }
                )

        for mention, mention_path in _script_mentions(skill_root, repo_root=repo_root):
            if mention in lookup or mention_path in lookup:
                continue
            blockers.append(
                {
                    "code": "stage_script_registry_skill_script_unregistered",
                    "skill": skill_name,
                    "script_ref": mention,
                    "canonical_path": mention_path,
                    "retryable": False,
                }
            )

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
        "script_registry_sha256": registry_sha256(registry_path),
    }
    return attach_agent_diagnostic(
        report,
        tool="stage_script_registry.skills",
        checked_inputs={
            "skills_root": str(skills_root),
            "registry_path": report["registry_path"],
        },
        accepted_evidence=[
            "Planning-skill required script refs were checked.",
            "Planning-skill usage_doc_refs were checked.",
            "Explicit script mentions were compared against the registry.",
        ],
        pass_next_actions=[
            "Planning skills are synchronized with the stage script registry."
        ],
        blocked_next_actions=[
            "Repair the cited skill instruction or registry entry, then rerun --skills-root."
        ],
    )


def _resolve_skill_contract_root(
    skills_root: Path,
    skill_name: str,
    contract_skill_root: str,
    *,
    repo_root: Path,
) -> Path:
    if skills_root.name == skill_name:
        return skills_root
    candidate = skills_root / skill_name
    if candidate.exists():
        return candidate
    repo_candidate = repo_root / contract_skill_root
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def audit_run_stage_script_usage(
    workspace: str | Path,
    *,
    run_id: str,
    registry_path: str | Path | None = None,
    strict: bool = False,
    execution_evidence: bool = False,
) -> dict[str, object]:
    """Audit workflow run receipts for required script usage evidence."""

    workspace_root = Path(workspace)
    registry = load_stage_script_registry(registry_path)
    # Strict verification is a production proof gate.  It must never fall
    # back to arbitrary receipt prose merely because callers omitted the
    # optional CLI flag.
    effective_execution_evidence = strict or execution_evidence
    run_dir = workspace_root / "storage" / "workflow-runs" / run_id
    digest = registry_sha256(registry_path)
    blockers: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    if not run_dir.is_dir():
        report = {
            "status": "blocked",
            "blockers": [
                {
                    "code": "stage_script_registry_run_dir_missing",
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "retryable": True,
                }
            ],
            "warnings": [],
            "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
            "script_registry_sha256": digest,
            "run_dir": str(run_dir),
        }
        return attach_agent_diagnostic(
            report,
            tool="stage_script_registry.run",
            checked_inputs={
                "workspace": str(workspace),
                "run_id": run_id,
                "run_dir": str(run_dir),
                "strict": strict,
                "execution_evidence": effective_execution_evidence,
            },
            accepted_evidence=[],
            blocked_next_actions=[
                "Use the correct run id or rerun the registry-bound workflow stage."
            ],
        )

    receipt_paths = sorted(path for path in run_dir.glob("*.json") if path.name != "summary.json")
    completed_stage_ids = _completed_registry_stage_ids_from_summary(run_dir, registry)
    checked_stage_ids: list[str] = []
    seen_stage_ids: set[str] = set()
    for receipt_path in receipt_paths:
        stage_id_from_name = _stage_id_from_receipt_name(receipt_path)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issue = {
                "code": "stage_script_registry_receipt_unreadable",
                "stage_id": stage_id_from_name,
                "receipt": _display_path(receipt_path),
                "reason": str(exc),
                "retryable": True,
            }
            if strict and stage_id_from_name in (registry.get("stage_bindings") or {}):
                blockers.append(issue)
            else:
                warnings.append(issue)
            continue
        if not isinstance(receipt, dict):
            if strict and stage_id_from_name in (registry.get("stage_bindings") or {}):
                blockers.append(
                    {
                        "code": "stage_script_registry_receipt_not_object",
                        "stage_id": stage_id_from_name,
                        "receipt": _display_path(receipt_path),
                        "retryable": True,
                    }
                )
            continue
        stage_id = str(receipt.get("stage_id") or stage_id_from_name)
        if stage_id not in (registry.get("stage_bindings") or {}):
            continue
        try:
            resolved_binding = resolve_stage_contract_binding(
                workspace=workspace_root,
                stage_id=stage_id,
                registry_path=registry_path,
            )
        except ValueError as exc:
            if strict:
                blockers.append(
                    {
                        "code": "stage_script_registry_workspace_binding_missing",
                        "stage_id": stage_id,
                        "receipt": _display_path(receipt_path),
                        "reason": str(exc),
                        "retryable": False,
                    }
                )
            continue
        if strict and str(receipt.get("status") or "").lower() not in SUCCEEDED_STATUSES:
            continue
        checked_stage_ids.append(stage_id)
        seen_stage_ids.add(stage_id)
        if effective_execution_evidence:
            evidence_text = "\n".join(
                _execution_evidence_strings(
                    receipt,
                    registry=registry,
                    allowed_owner_ids=resolved_binding.required_script_ids,
                )
            )
        else:
            evidence_text = "\n".join(
                _flatten_strings(receipt) + _presentation_manifest_evidence(workspace_root, stage_id)
            )
        required_ids = tuple(resolved_binding.required_script_ids)
        forbidden_ids = forbidden_script_ids_for_stage(registry, stage_id)
        stage_contract = receipt.get("stage_contract")
        missing_ids = [
            script_id
            for script_id in required_ids
            if not _any_ref_in_text(registry, script_id, evidence_text)
        ]
        if missing_ids:
            blockers.append(
                {
                    "code": "stage_script_registry_required_script_missing_from_receipt",
                    "stage_id": stage_id,
                    "receipt": _display_path(receipt_path),
                    "missing_script_ids": list(missing_ids),
                    "missing_refs": list(refs_for_script_ids(registry, missing_ids)),
                    "retryable": True,
                }
            )
        forbidden_hits = [
            ref
            for ref in refs_for_script_ids(registry, forbidden_ids)
            if ref in evidence_text
        ]
        if forbidden_hits:
            blockers.append(
                {
                    "code": "stage_script_registry_forbidden_script_used_in_receipt",
                    "stage_id": stage_id,
                    "receipt": _display_path(receipt_path),
                    "forbidden_refs": sorted(set(forbidden_hits)),
                    "retryable": False,
                }
            )
        if strict:
            receipt_digest = receipt.get("script_registry_sha256")
            if receipt_digest and receipt_digest != digest:
                blockers.append(
                    {
                        "code": "stage_script_registry_digest_mismatch",
                        "stage_id": stage_id,
                        "receipt": _display_path(receipt_path),
                        "expected_sha256": digest,
                        "observed_sha256": receipt_digest,
                        "retryable": True,
                    }
                )
            elif receipt_digest is None:
                blockers.append(
                    {
                        "code": "stage_script_registry_digest_missing_from_receipt",
                        "stage_id": stage_id,
                        "receipt": _display_path(receipt_path),
                        "expected_sha256": digest,
                        "retryable": True,
                    }
                )
            observed_required = receipt.get("required_script_ids")
            if isinstance(observed_required, list):
                missing_declared = sorted(set(required_ids) - {str(item) for item in observed_required})
                if missing_declared:
                    blockers.append(
                        {
                            "code": "stage_script_registry_required_ids_missing_from_receipt",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "missing_script_ids": missing_declared,
                            "retryable": True,
                        }
                    )
            runtime_contract = resolved_binding.runtime_contract
            if isinstance(runtime_contract, dict):
                if not isinstance(stage_contract, dict):
                    blockers.append(
                        {
                            "code": "stage_contract_missing_from_receipt",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "expected": "stage_contract.status == pass",
                            "observed": "<missing>",
                            "retryable": True,
                        }
                    )
                elif stage_contract.get("status") != "pass":
                    blockers.append(
                        {
                            "code": "stage_contract_blocked_in_receipt",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "expected": "stage_contract.status == pass",
                            "observed": stage_contract.get("status"),
                            "blockers": stage_contract.get("blockers") or [],
                            "retryable": True,
                        }
                    )
            if resolved_binding.registry_stage_id != stage_id:
                if not isinstance(stage_contract, dict):
                    blockers.append(
                        {
                            "code": "stage_script_registry_stage_contract_binding_mismatch",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "expected_registry_stage_id": resolved_binding.registry_stage_id,
                            "observed_registry_stage_id": "<missing>",
                            "retryable": True,
                        }
                    )
                    continue
                if stage_contract.get("registry_stage_id") != resolved_binding.registry_stage_id:
                    blockers.append(
                        {
                            "code": "stage_script_registry_stage_contract_binding_mismatch",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "expected_registry_stage_id": resolved_binding.registry_stage_id,
                            "observed_registry_stage_id": stage_contract.get("registry_stage_id"),
                            "retryable": True,
                        }
                    )
                contract_required_ids = stage_contract.get("required_script_ids")
                if not isinstance(contract_required_ids, list) or {
                    str(item) for item in contract_required_ids
                } != set(required_ids):
                    blockers.append(
                        {
                            "code": "stage_script_registry_stage_contract_required_ids_mismatch",
                            "stage_id": stage_id,
                            "receipt": _display_path(receipt_path),
                            "expected_script_ids": list(required_ids),
                            "observed_script_ids": contract_required_ids,
                            "retryable": True,
                        }
                    )

    if strict:
        for stage_id in sorted(completed_stage_ids - seen_stage_ids):
            blockers.append(
                {
                    "code": "stage_script_registry_completed_stage_receipt_missing",
                    "stage_id": stage_id,
                    "run_id": run_id,
                    "retryable": True,
                }
            )
        if not checked_stage_ids and not completed_stage_ids and not blockers:
            blockers.append(
                {
                    "code": "stage_script_registry_no_completed_stage_receipts",
                    "run_id": run_id,
                    "retryable": True,
                }
            )

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "registry_path": str(registry_path or DEFAULT_STAGE_SCRIPT_REGISTRY_PATH),
        "script_registry_sha256": digest,
        "checked_stage_ids": checked_stage_ids,
        "run_dir": str(run_dir),
    }
    return attach_agent_diagnostic(
        report,
        tool="stage_script_registry.run",
        checked_inputs={
            "workspace": str(workspace),
            "run_id": run_id,
            "run_dir": str(run_dir),
            "strict": strict,
            "execution_evidence": effective_execution_evidence,
            "receipt_count": len(receipt_paths),
            "completed_registry_stage_ids": sorted(completed_stage_ids),
        },
        accepted_evidence=[
            f"Checked registry-bound stage receipts: {', '.join(checked_stage_ids) or '<none>'}.",
            "Receipt commands, command metadata, presentation manifest owner refs, registry digest, and required_script_ids were inspected as applicable.",
        ],
        pass_next_actions=[
            "Registry-bound run evidence is sufficient for the checked stages."
        ],
        blocked_next_actions=[
            "Repair or rerun the cited stage receipts before treating the run as production-complete."
        ],
    )


def _completed_registry_stage_ids_from_summary(
    run_dir: Path,
    registry: Mapping[str, Any],
) -> set[str]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return set()
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    registry_stage_ids = set((registry.get("stage_bindings") or {}).keys())
    completed: set[str] = set()
    for stage in _iter_summary_stage_records(summary):
        stage_id = _stage_id_from_summary_record(stage)
        if stage_id not in registry_stage_ids:
            continue
        status = str(stage.get("status") or stage.get("result") or "").lower()
        if status in SUCCEEDED_STATUSES:
            completed.add(stage_id)
    return completed


def _iter_summary_stage_records(summary: Any) -> list[Mapping[str, Any]]:
    if not isinstance(summary, Mapping):
        return []
    records: list[Mapping[str, Any]] = []
    for key in ("stages", "stage_results", "completed_stages"):
        value = summary.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, Mapping):
                    records.append(item)
                elif key == "completed_stages" and isinstance(item, str):
                    records.append({"stage_id": item, "status": "succeeded"})
        elif isinstance(value, Mapping):
            for stage_id, item in value.items():
                if isinstance(item, Mapping):
                    merged = {"stage_id": stage_id, **item}
                    records.append(merged)
    return records


def _stage_id_from_summary_record(record: Mapping[str, Any]) -> str:
    for key in ("stage_id", "id", "stage", "name"):
        value = record.get(key)
        if value:
            return str(value)
    return ""


def _collect_text(root: Path) -> str:
    if root.is_file():
        return root.read_text(encoding="utf-8")
    if not root.exists():
        return ""
    parts: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in PLAN_AUDIT_FILE_SUFFIXES.union({".py"}):
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _any_ref_in_text(registry: Mapping[str, Any], script_id: str, text: str) -> bool:
    return any(ref in text for ref in refs_for_script_id(registry, script_id))


def _any_usage_doc_ref_in_text(
    registry: Mapping[str, Any],
    script_id: str,
    text: str,
) -> bool:
    return any(ref in text for ref in usage_doc_refs_for_script_id(registry, script_id))


def _usage_doc_refs_for_script_ids(
    registry: Mapping[str, Any],
    script_ids: Iterable[str],
) -> tuple[str, ...]:
    refs: list[str] = []
    for script_id in script_ids:
        for ref in usage_doc_refs_for_script_id(registry, str(script_id)):
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)


def _deprecated_and_forbidden_script_ids(registry: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []
    for script_id, script in (registry.get("scripts") or {}).items():
        if str(script.get("status") or "") in {"deprecated", "forbidden"}:
            ids.append(str(script_id))
    return tuple(ids)


def _script_mentions(skill_root: Path, *, repo_root: Path) -> list[tuple[str, str]]:
    mentions: list[tuple[str, str]] = []
    if not skill_root.exists():
        return mentions
    for path in sorted(skill_root.rglob("*")):
        if not path.is_file() or path.suffix not in PLAN_AUDIT_FILE_SUFFIXES.union({".py"}):
            continue
        text = path.read_text(encoding="utf-8")
        for match in SCRIPT_REF_RE.finditer(text):
            ref = match.group(1)
            skill_local = (skill_root / ref).resolve()
            repo_local = (repo_root / ref).resolve()
            if skill_local.exists():
                try:
                    canonical = skill_local.relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    canonical = ref
            elif repo_local.exists():
                canonical = repo_local.relative_to(repo_root.resolve()).as_posix()
            else:
                canonical = ref
            mentions.append((ref, canonical))
    return mentions


def _rejected_owner_hits(
    root: Path,
    rejected_owner_refs: Iterable[str],
    *,
    allow_historical_reference_documents: bool = False,
) -> list[str]:
    hits: list[str] = []
    refs = tuple(ref for ref in rejected_owner_refs if ref)
    if not refs:
        return hits
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file() or path.suffix not in PLAN_AUDIT_FILE_SUFFIXES:
            continue
        file_text = path.read_text(encoding="utf-8")
        if (
            allow_historical_reference_documents
            and _is_explicit_historical_reference_document(file_text)
        ):
            continue
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        # Units, not lines: a wrapped sentence or a JSON array under a
        # "rejected_..."/"deprecated_..." key is one policy statement, judged
        # once; a heading still cannot bless a bare command beneath it. Shared
        # with stage_owner_audit so the two plan audits cannot drift apart.
        if path.suffix == ".json":
            units = _json_policy_units(file_text, is_metadata_line=_line_is_registry_metadata_line)
        else:
            units = _text_policy_units(file_text, is_metadata_line=_line_is_registry_metadata_line)
        for unit_text, negative in units:
            if not any(ref in unit_text for ref in refs):
                continue
            if negative or _line_is_negative_owner_policy(unit_text.lower()):
                continue
            hits.append(f"{relative}: {' '.join(unit_text.split())[:200]}")
    return hits


def _line_is_negative_owner_policy(lower_line: str) -> bool:
    return any(term in lower_line for term in NEGATIVE_POLICY_TERMS)


def _line_is_registry_metadata_line(lower_line: str) -> bool:
    metadata_keys = (
        '"path"',
        '"refs"',
        '"usage_doc_refs"',
        '"diagnostic_profile"',
        '"replacement_script_ids"',
        '"script_id"',
        '"status"',
        '"scope"',
        '"stage_bindings"',
        '"task_bindings"',
    )
    return any(key in lower_line for key in metadata_keys)


def _is_explicit_historical_reference_document(text: str) -> bool:
    """Allow docs-only legacy references with an explicit non-runbook declaration.

    Production plan scans never opt into this exception.  Documentation may
    preserve a historical reference only when its opening policy block says it
    is not a current operator runbook; generic "historical" or "deprecated"
    mentions are not enough.
    """

    header = text[:1200].lower()
    normalized_header = re.sub(r"\s+", " ", header.replace(">", " "))
    return "historical" in normalized_header and "not a current operator runbook" in normalized_header


def _presentation_manifest_evidence(workspace_root: Path, stage_id: str) -> list[str]:
    manifest_path = workspace_root / "decks" / "presentation_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    stage_specs = manifest.get("stage_specs") if isinstance(manifest, Mapping) else None
    if not isinstance(stage_specs, list):
        return []
    evidence: list[str] = []
    for spec in stage_specs:
        if not isinstance(spec, Mapping):
            continue
        spec_stage_id = str(spec.get("id") or spec.get("stage_id") or spec.get("stage") or "")
        if spec_stage_id == stage_id:
            evidence.extend(_flatten_strings(spec))
    return evidence


def _execution_evidence_strings(
    receipt: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
    allowed_owner_ids: Iterable[str] = (),
) -> list[str]:
    """Build evidence text ONLY from execution-derived receipt fields.

    Unlike ``_flatten_strings(receipt)`` (which accepts any flattened prose,
    including a bare owner-ref string mentioned elsewhere in the receipt),
    this restricts the evidence blob to fields that can only be populated by
    actually running the owner script/command: ``command``, ``commands``,
    ``command_metadata``, ``executed_owner_refs``, and
    ``executed_script_ids``.  A receipt's self-declared stage contract is
    verified separately; it is not execution evidence.  Workspace-specific
    ``owner_execution_evidence`` is accepted only for the resolved owner ids
    and only when its value contains evidence distinct from the owner id.
    """

    strings: list[str] = []
    for key in (
        "command",
        "commands",
        "command_metadata",
        "executed_owner_refs",
        "executed_script_ids",
    ):
        if key in receipt:
            strings.extend(_flatten_strings(receipt[key]))
    owner_execution_evidence = receipt.get("owner_execution_evidence")
    if isinstance(owner_execution_evidence, Mapping) and registry is not None:
        for script_id in allowed_owner_ids:
            evidence = owner_execution_evidence.get(script_id)
            if _has_distinct_owner_execution_evidence(evidence, script_id):
                strings.extend(refs_for_script_id(registry, str(script_id)))
    return strings


def _has_distinct_owner_execution_evidence(value: Any, script_id: str) -> bool:
    evidence_values = [item.strip() for item in _flatten_strings(value) if item.strip()]
    return bool(evidence_values) and any(item != script_id for item in evidence_values)


def _flatten_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            strings.extend(_flatten_strings(item))
    elif isinstance(value, list | tuple):
        for item in value:
            strings.extend(_flatten_strings(item))
    return strings


def _stage_id_from_receipt_name(path: Path) -> str:
    stem = path.stem
    if "-" not in stem:
        return stem
    parts = stem.split("-", 2)
    return parts[-1]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--skills-root", type=Path)
    parser.add_argument("--docs-root", type=Path)
    parser.add_argument("--registry-path", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--execution-evidence", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    aggregate_profile: Mapping[str, Any] | None = None
    try:
        aggregate_profile = diagnostic_profile_for_script_id(
            load_stage_script_registry(args.registry_path),
            "verification.stage_script_usage",
        )
    except Exception:
        aggregate_profile = None
    reports: dict[str, object] = {
        "registry": audit_stage_script_registry_integrity(registry_path=args.registry_path),
    }
    if args.plan_root is not None:
        reports["plan"] = audit_plan_stage_script_registry(
            args.plan_root,
            registry_path=args.registry_path,
            require_registry_digest=args.strict,
            require_verifier_command=args.strict,
            require_usage_doc_refs=args.strict,
        )
    if args.docs_root is not None:
        reports["docs"] = audit_stage_script_usage_docs(
            args.docs_root,
            registry_path=args.registry_path,
        )
    if args.skills_root is not None:
        reports["skills"] = audit_skill_stage_script_contracts(
            args.skills_root,
            registry_path=args.registry_path,
        )
    if args.workspace is not None or args.run_id is not None:
        if args.workspace is None or args.run_id is None:
            raise SystemExit("--workspace and --run-id must be provided together")
        reports["run"] = audit_run_stage_script_usage(
            args.workspace,
            run_id=args.run_id,
            registry_path=args.registry_path,
            strict=args.strict,
            execution_evidence=args.execution_evidence,
        )
    status = (
        "pass"
        if all(
            isinstance(report, Mapping) and report.get("status") == "pass"
            for report in reports.values()
        )
        else "blocked"
    )
    payload = {
        "status": status,
        "diagnostic": aggregate_agent_diagnostic(
            tool="verify_stage_script_usage",
            status=status,
            reports=reports,
            diagnostic_profile=aggregate_profile,
            checked_inputs={
                "plan_root": str(args.plan_root) if args.plan_root is not None else None,
                "workspace": str(args.workspace) if args.workspace is not None else None,
                "run_id": args.run_id,
                "skills_root": str(args.skills_root) if args.skills_root is not None else None,
                "docs_root": str(args.docs_root) if args.docs_root is not None else None,
                "strict": args.strict,
                "execution_evidence": args.execution_evidence,
            },
            pass_next_actions=[
                "Registry verification passed; continue to the next planned stage or final proof gate."
            ],
            blocked_next_actions=[
                "Use the child report next_actions to repair the plan, docs, skills, or receipts, then rerun this command."
            ],
        ),
        "reports": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
