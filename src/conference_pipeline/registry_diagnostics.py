"""Agent-actionable diagnostic helpers for registry-owned tools."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

NEXT_ACTION_BY_CODE = {
    "stage_script_registry_usage_doc_refs_missing": (
        "Add usage_doc_refs for the listed script_id in "
        "src/conference_pipeline/stage_owner_registry.json, then rerun the "
        "registry docs verifier."
    ),
    "stage_script_registry_usage_doc_anchor_missing": (
        "Create the missing anchor in docs/reference/stage-owner-audit.md or "
        "update the registry usage_doc_refs entry to the correct anchor."
    ),
    "stage_script_registry_usage_doc_missing": (
        "Create the referenced usage doc file or point usage_doc_refs at an "
        "existing registry-owned documentation anchor."
    ),
    "stage_script_registry_forbidden_script_used_in_docs": (
        "Edit the cited docs so deprecated or forbidden scripts appear only in "
        "negative-policy or historical context."
    ),
    "stage_script_registry_required_script_missing_from_plan": (
        "Update the generated plan owner-command table to include the missing "
        "registry script refs, then rerun verify_stage_script_usage.py."
    ),
    "stage_script_registry_usage_doc_ref_missing_from_plan": (
        "Add the registry usage_doc_refs for the missing script ids to the "
        "generated plan tool table, then rerun the strict plan verifier."
    ),
    "stage_script_registry_required_term_missing_from_plan": (
        "Update the generated plan command shape so it includes the required "
        "terms for the governed task id."
    ),
    "stage_script_registry_digest_missing_from_plan": (
        "Regenerate or edit the plan so it records the current stage script "
        "registry sha256 digest."
    ),
    "stage_script_registry_verifier_command_missing_from_plan": (
        "Add the final scripts/verify_stage_script_usage.py proof gate to the "
        "generated plan."
    ),
    "stage_script_registry_required_skill_script_missing": (
        "Update the planning skill instructions so the required registered "
        "script is explicitly mentioned."
    ),
    "stage_script_registry_required_skill_usage_doc_missing": (
        "Update the planning skill instructions to cite the script's registry "
        "usage_doc_refs anchor."
    ),
    "stage_script_registry_skill_script_unregistered": (
        "Register the mentioned script in stage_owner_registry.json or remove "
        "the skill mention if it is not an approved tool."
    ),
    "stage_script_registry_diagnostic_profile_missing": (
        "Add diagnostic_profile to the listed script_id in "
        "src/conference_pipeline/stage_owner_registry.json."
    ),
    "stage_script_registry_diagnostic_profile_invalid": (
        "Populate diagnostic_profile success_summary, blocked_summary, "
        "next_actions, and blocked_next_actions for the listed script_id."
    ),
    "stage_script_registry_deprecated_diagnostic_replacement_missing": (
        "Add diagnostic_profile.replacement_script_ids so deprecated or "
        "forbidden tools name their approved replacement path."
    ),
    "stage_script_registry_deprecated_diagnostic_replacement_unknown": (
        "Correct diagnostic_profile.replacement_script_ids so every replacement "
        "is a registered script_id."
    ),
    "stage_script_registry_run_dir_missing": (
        "Check the run id and workspace, or rerun the workflow stage so "
        "storage/workflow-runs/<run-id>/ exists."
    ),
    "stage_script_registry_receipt_unreadable": (
        "Regenerate the stage receipt or repair the malformed JSON before "
        "treating the run as complete."
    ),
    "stage_script_registry_receipt_not_object": (
        "Regenerate the stage receipt as a JSON object with command evidence "
        "and registry metadata."
    ),
    "stage_script_registry_required_script_missing_from_receipt": (
        "Rerun the stage with the required owner command metadata, or record an "
        "explicit registry-approved waiver if the evidence is intentionally absent."
    ),
    "stage_script_registry_forbidden_script_used_in_receipt": (
        "Rerun the stage with an approved owner script; deprecated or forbidden "
        "scripts cannot satisfy production completion."
    ),
    "stage_script_registry_digest_mismatch": (
        "Rerun or refresh the stage receipt against the current registry digest "
        "before using it as proof."
    ),
    "stage_script_registry_digest_missing_from_receipt": (
        "Rerun the stage with script_registry_sha256 recorded in the receipt."
    ),
    "stage_script_registry_required_ids_missing_from_receipt": (
        "Rerun the stage or repair the receipt so required_script_ids declares "
        "all scripts required by the registry binding."
    ),
    "stage_script_registry_completed_stage_receipt_missing": (
        "Regenerate the missing completed-stage receipt or mark the stage "
        "incomplete before final proof."
    ),
    "stage_script_registry_no_completed_stage_receipts": (
        "Run at least one registry-bound stage or provide the correct run id "
        "before strict registry proof."
    ),
    "plan_owner_registry_approved_owner_missing": (
        "Update the generated plan to include the approved owner ref for the "
        "listed task id."
    ),
    "plan_owner_registry_required_term_missing": (
        "Update the generated plan command shape to include the required term."
    ),
    "plan_owner_registry_rejected_owner_used": (
        "Replace the rejected owner with an approved registry owner, or move the "
        "reference into explicit negative-policy context."
    ),
    "workspace_gitignore_missing": (
        "Run scripts/generate_workspace_gitignore.py --workspace <workspace> "
        "--write --check --receipt-path <receipt>."
    ),
    "workspace_gitignore_stale": (
        "Run scripts/generate_workspace_gitignore.py --workspace <workspace> "
        "--write --check --receipt-path <receipt> to refresh the managed block."
    ),
    "drive_or_cache_family_file_tracked_in_git": (
        "Move or untrack the listed payload artifact; Drive/cache-owned "
        "families should not be committed unless declared as exceptions."
    ),
    "workspace_gitignore_expected_ignored_path_not_ignored": (
        "Refresh the workspace .gitignore managed block from the storage "
        "manifest and rerun --check."
    ),
    "workspace_gitignore_expected_trackable_path_ignored": (
        "Refresh the workspace .gitignore or declare a concrete exception in "
        "storage_manifest.json."
    ),
    "storage_manifest_missing": (
        "Create conferences/<slug>/storage/storage_manifest.json before "
        "generating or checking workspace gitignore policy."
    ),
}


def attach_agent_diagnostic(
    report: Mapping[str, Any],
    *,
    tool: str,
    checked_inputs: Mapping[str, Any] | None = None,
    accepted_evidence: list[str] | None = None,
    outputs: list[str] | None = None,
    rerun_command: str | None = None,
    diagnostic_profile: Mapping[str, Any] | None = None,
    pass_next_actions: list[str] | None = None,
    blocked_next_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Return a report enriched with an agent-readable diagnostic envelope."""

    enriched = dict(report)
    blockers = enrich_issues(enriched.get("blockers") or ())
    warnings = enrich_issues(enriched.get("warnings") or ())
    enriched["blockers"] = blockers
    if "warnings" in enriched or warnings:
        enriched["warnings"] = warnings
    status = str(enriched.get("status") or ("blocked" if blockers else "pass"))
    if status == "pass":
        next_actions = _unique(
            _profile_actions(diagnostic_profile, "next_actions")
            + list(pass_next_actions or ())
        )
    else:
        next_actions = _unique(
            [str(item["next_action"]) for item in blockers if item.get("next_action")]
            + _profile_actions(diagnostic_profile, "blocked_next_actions")
            + list(blocked_next_actions or ())
        )
    summary = _summary_for(
        tool=tool,
        status=status,
        blockers=blockers,
        warnings=warnings,
        diagnostic_profile=diagnostic_profile,
    )
    enriched["diagnostic"] = {
        "tool": tool,
        "tool_id": tool,
        "status": status,
        "summary": summary,
        "checked_inputs": dict(checked_inputs or {}),
        "accepted_evidence": list(accepted_evidence or ()),
        "outputs": list(outputs or _profile_actions(diagnostic_profile, "outputs")),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "next_actions": next_actions,
        "rerun_command": rerun_command
        or _profile_text(diagnostic_profile, "rerun_command"),
    }
    return enriched


def attach_registry_tool_diagnostic(
    report: Mapping[str, Any],
    *,
    script_id: str,
    registry_path: str | Path | None = None,
    checked_inputs: Mapping[str, Any] | None = None,
    accepted_evidence: list[str] | None = None,
    outputs: list[str] | None = None,
    rerun_command: str | None = None,
    pass_next_actions: list[str] | None = None,
    blocked_next_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Attach diagnostics using a registered script's diagnostic_profile."""

    profile: Mapping[str, Any] | None = None
    try:
        from conference_pipeline.stage_script_registry import (
            diagnostic_profile_for_script_id,
            load_stage_script_registry,
        )

        profile = diagnostic_profile_for_script_id(
            load_stage_script_registry(registry_path),
            script_id,
        )
    except Exception:
        profile = None
    return attach_agent_diagnostic(
        report,
        tool=script_id,
        checked_inputs=checked_inputs,
        accepted_evidence=accepted_evidence,
        outputs=outputs,
        rerun_command=rerun_command,
        diagnostic_profile=profile,
        pass_next_actions=pass_next_actions,
        blocked_next_actions=blocked_next_actions,
    )


def enrich_issues(issues: Any) -> list[dict[str, Any]]:
    """Add expected/observed/next_action context to blocker or warning dicts."""

    enriched: list[dict[str, Any]] = []
    for raw in issues or ():
        if not isinstance(raw, Mapping):
            enriched.append(
                {
                    "code": "unstructured_issue",
                    "observed": str(raw),
                    "retryable": False,
                    "next_action": "Inspect the unstructured issue and convert it to a structured blocker.",
                }
            )
            continue
        issue = dict(raw)
        code = str(issue.get("code") or "unknown_issue")
        issue.setdefault("expected", _expected_from_issue(issue))
        issue.setdefault("observed", _observed_from_issue(issue))
        issue.setdefault("next_action", NEXT_ACTION_BY_CODE.get(code, _default_next_action(code)))
        enriched.append(issue)
    return enriched


def aggregate_agent_diagnostic(
    *,
    tool: str,
    status: str,
    reports: Mapping[str, Any],
    checked_inputs: Mapping[str, Any] | None = None,
    outputs: list[str] | None = None,
    rerun_command: str | None = None,
    diagnostic_profile: Mapping[str, Any] | None = None,
    pass_next_actions: list[str] | None = None,
    blocked_next_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a top-level diagnostic from child tool reports."""

    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    accepted: list[str] = []
    next_actions: list[str] = []
    for name, report in reports.items():
        if not isinstance(report, Mapping):
            continue
        diagnostic = report.get("diagnostic")
        if isinstance(diagnostic, Mapping):
            if report.get("status") == "pass":
                accepted.append(f"{name}: {diagnostic.get('summary')}")
            next_actions.extend(str(item) for item in diagnostic.get("next_actions") or ())
        blockers.extend(issue for issue in report.get("blockers") or () if isinstance(issue, Mapping))
        warnings.extend(issue for issue in report.get("warnings") or () if isinstance(issue, Mapping))
    if status == "pass":
        next_actions = _unique(
            _profile_actions(diagnostic_profile, "next_actions")
            + list(pass_next_actions or next_actions)
        )
    else:
        next_actions = _unique(
            next_actions
            + _profile_actions(diagnostic_profile, "blocked_next_actions")
            + list(blocked_next_actions or ())
        )
    return {
        "tool": tool,
        "tool_id": tool,
        "status": status,
        "summary": _summary_for(
            tool=tool,
            status=status,
            blockers=blockers,
            warnings=warnings,
            diagnostic_profile=diagnostic_profile,
        ),
        "checked_inputs": dict(checked_inputs or {}),
        "accepted_evidence": accepted,
        "outputs": list(outputs or _profile_actions(diagnostic_profile, "outputs")),
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "next_actions": next_actions,
        "rerun_command": rerun_command
        or _profile_text(diagnostic_profile, "rerun_command"),
    }


def _summary_for(
    *,
    tool: str,
    status: str,
    blockers: list[Mapping[str, Any]],
    warnings: list[Mapping[str, Any]],
    diagnostic_profile: Mapping[str, Any] | None = None,
) -> str:
    if status == "pass":
        profile_summary = _profile_text(diagnostic_profile, "success_summary")
        if profile_summary:
            return profile_summary
        if warnings:
            return f"{tool} passed with {len(warnings)} warning(s)."
        return f"{tool} passed; required evidence was found."
    profile_summary = _profile_text(diagnostic_profile, "blocked_summary")
    if profile_summary:
        return profile_summary
    return f"{tool} blocked with {len(blockers)} blocker(s) and {len(warnings)} warning(s)."


def _expected_from_issue(issue: Mapping[str, Any]) -> Any:
    for key in (
        "expected_sha256",
        "missing_refs",
        "missing_terms",
        "required_ref",
        "usage_doc_refs",
        "missing_usage_doc_refs",
        "missing_owner_refs",
        "missing_script_ids",
        "missing_fields",
        "unknown_replacement_script_ids",
        "allowed_categories",
    ):
        if key in issue:
            return issue[key]
    code = str(issue.get("code") or "")
    if "missing" in code:
        return "Required evidence should be present."
    if "forbidden" in code or "rejected" in code:
        return "Only approved registry owners may appear as completion evidence."
    return "Registry policy should be satisfied."


def _observed_from_issue(issue: Mapping[str, Any]) -> Any:
    for key in (
        "observed_sha256",
        "observed_category",
        "forbidden_refs",
        "hits",
        "reason",
        "path",
        "receipt",
        "run_dir",
        "script_ref",
    ):
        if key in issue:
            return issue[key]
    code = str(issue.get("code") or "")
    if "missing" in code:
        return "Missing from checked inputs."
    if "stale" in code:
        return "Checked artifact is stale."
    return "Observed evidence did not satisfy registry policy."


def _default_next_action(code: str) -> str:
    return (
        f"Inspect blocker `{code}`, repair the cited plan/receipt/manifest/doc, "
        "then rerun the same registry verification command."
    )


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _profile_text(profile: Mapping[str, Any] | None, key: str) -> str | None:
    if not isinstance(profile, Mapping):
        return None
    value = profile.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _profile_actions(profile: Mapping[str, Any] | None, key: str) -> list[str]:
    if not isinstance(profile, Mapping):
        return []
    value = profile.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []
