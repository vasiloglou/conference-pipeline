"""Canonical stage script registry loader and helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_SCRIPT_REGISTRY_PATH = Path(__file__).with_name("stage_owner_registry.json")
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
USAGE_DOC_REF_RE = re.compile(
    r"^(?P<path>(?:docs|\.claude)/[A-Za-z0-9_./-]+\.md)#(?P<anchor>[A-Za-z0-9][A-Za-z0-9_-]*)$"
)
NEGATIVE_USAGE_DOC_TERMS = frozenset(
    {
        "deprecated",
        "forbidden",
        "rejected",
        "must not",
        "do not",
        "legacy",
    }
)


class StageScriptRegistryError(ValueError):
    """Raised when the stage script registry is malformed."""


def registry_sha256(path: str | Path | None = None) -> str:
    """Return the SHA-256 digest of the registry file bytes."""

    registry_path = Path(path) if path is not None else DEFAULT_STAGE_SCRIPT_REGISTRY_PATH
    return hashlib.sha256(registry_path.read_bytes()).hexdigest()


def load_stage_script_registry(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the unified stage script registry."""

    registry_path = Path(path) if path is not None else DEFAULT_STAGE_SCRIPT_REGISTRY_PATH
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise StageScriptRegistryError(
            f"Unsupported stage script registry schema: {registry_path}"
        )
    if schema_version == 1:
        payload = _upgrade_v1_registry(payload)
    _validate_v2_registry(payload, registry_path)
    return payload


def legacy_owner_registry_view(path: str | Path | None = None) -> dict[str, Any]:
    """Return the legacy task-owner view expected by existing callers."""

    registry = load_stage_script_registry(path)
    tasks = dict(registry["tasks"])
    task_bindings = registry.get("task_bindings") or {}
    for task_id in registry.get("planning_required_task_ids") or ():
        task_id = str(task_id)
        if task_id in tasks:
            continue
        binding = task_bindings.get(task_id)
        if not isinstance(binding, Mapping):
            continue
        approved_script_ids = tuple(
            str(script_id)
            for script_id in (
                binding.get("approved_script_ids")
                or binding.get("required_script_ids")
                or ()
            )
        )
        forbidden_script_ids = tuple(
            str(script_id) for script_id in binding.get("forbidden_script_ids") or ()
        )
        tasks[task_id] = {
            "approved_owner_refs": list(refs_for_script_ids(registry, approved_script_ids)),
            "rejected_owner_refs": list(refs_for_script_ids(registry, forbidden_script_ids)),
            "required_command_terms": [
                str(term) for term in binding.get("required_command_terms") or ()
            ],
        }
    return {
        "schema_version": registry["schema_version"],
        "planning_required_task_ids": list(registry.get("planning_required_task_ids") or ()),
        "tasks": tasks,
    }


def validate_registered_script_existence(
    registry: Mapping[str, Any] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, object]]:
    """Return blockers for registered required scripts/modules that are missing."""

    loaded = dict(registry or load_stage_script_registry())
    blockers: list[dict[str, object]] = []
    for script_id, script in _scripts(loaded).items():
        existence = str(script.get("existence") or "required")
        if existence != "required":
            continue
        path_value = str(script.get("path") or "")
        if not path_value:
            blockers.append(
                {
                    "code": "stage_script_registry_path_missing",
                    "script_id": script_id,
                    "retryable": False,
                }
            )
            continue
        path = (repo_root / path_value).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError:
            blockers.append(
                {
                    "code": "stage_script_registry_path_escapes_repo",
                    "script_id": script_id,
                    "path": path_value,
                    "retryable": False,
                }
            )
            continue
        if not path.exists():
            blockers.append(
                {
                    "code": "stage_script_registry_required_script_missing",
                    "script_id": script_id,
                    "path": path_value,
                    "retryable": False,
                }
            )
    return blockers


def refs_for_script_id(registry: Mapping[str, Any], script_id: str) -> tuple[str, ...]:
    """Return all textual refs that identify a registered script."""

    script = _scripts(registry).get(script_id)
    if not isinstance(script, Mapping):
        raise KeyError(script_id)
    refs = [str(ref) for ref in script.get("refs") or () if str(ref)]
    path = str(script.get("path") or "")
    if path and path not in refs:
        refs.append(path)
    return tuple(refs)


def usage_doc_refs_for_script_id(
    registry: Mapping[str, Any],
    script_id: str,
) -> tuple[str, ...]:
    """Return canonical usage documentation refs for a registered script."""

    script = _scripts(registry).get(script_id)
    if not isinstance(script, Mapping):
        raise KeyError(script_id)
    return tuple(str(ref) for ref in script.get("usage_doc_refs") or () if str(ref))


def diagnostic_profile_for_script_id(
    registry: Mapping[str, Any],
    script_id: str,
) -> Mapping[str, Any]:
    """Return the agent-facing diagnostic profile for a registered script."""

    script = _scripts(registry).get(script_id)
    if not isinstance(script, Mapping):
        raise KeyError(script_id)
    profile = script.get("diagnostic_profile")
    if not isinstance(profile, Mapping):
        raise KeyError(script_id)
    return profile


def validate_registered_diagnostic_profiles(
    registry: Mapping[str, Any] | None = None,
) -> list[dict[str, object]]:
    """Return blockers for missing or malformed agent diagnostic profiles."""

    loaded = dict(registry or load_stage_script_registry())
    blockers: list[dict[str, object]] = []
    scripts = _scripts(loaded)
    for script_id, script in scripts.items():
        profile = script.get("diagnostic_profile")
        if not isinstance(profile, Mapping):
            blockers.append(
                {
                    "code": "stage_script_registry_diagnostic_profile_missing",
                    "script_id": script_id,
                    "retryable": False,
                }
            )
            continue
        missing_text = [
            field
            for field in ("success_summary", "blocked_summary")
            if not isinstance(profile.get(field), str) or not str(profile.get(field)).strip()
        ]
        missing_lists = [
            field
            for field in ("next_actions", "blocked_next_actions")
            if not _nonempty_string_list(profile.get(field))
        ]
        if missing_text or missing_lists:
            blockers.append(
                {
                    "code": "stage_script_registry_diagnostic_profile_invalid",
                    "script_id": script_id,
                    "missing_fields": missing_text + missing_lists,
                    "retryable": False,
                }
            )
        if str(script.get("status") or "") in {"deprecated", "forbidden"}:
            replacements = profile.get("replacement_script_ids")
            if not _nonempty_string_list(replacements):
                blockers.append(
                    {
                        "code": "stage_script_registry_deprecated_diagnostic_replacement_missing",
                        "script_id": script_id,
                        "retryable": False,
                    }
                )
                continue
            unknown = sorted(
                str(item)
                for item in replacements
                if str(item) not in scripts
            )
            if unknown:
                blockers.append(
                    {
                        "code": "stage_script_registry_deprecated_diagnostic_replacement_unknown",
                        "script_id": script_id,
                        "unknown_replacement_script_ids": unknown,
                        "retryable": False,
                    }
                )
    return blockers


def validate_registered_usage_doc_refs(
    registry: Mapping[str, Any] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    docs_root: Path | None = None,
) -> list[dict[str, object]]:
    """Return blockers for missing or stale registry-owned usage docs."""

    loaded = dict(registry or load_stage_script_registry())
    docs_root = docs_root or (repo_root / "docs")
    repo_root_resolved = repo_root.resolve()
    docs_root_resolved = docs_root.resolve()
    blockers: list[dict[str, object]] = []
    for script_id, script in _scripts(loaded).items():
        usage_refs = list(usage_doc_refs_for_script_id(loaded, script_id))
        if not usage_refs:
            blockers.append(
                {
                    "code": "stage_script_registry_usage_doc_refs_missing",
                    "script_id": script_id,
                    "retryable": False,
                }
            )
            continue
        negative_context = ""
        for usage_ref in usage_refs:
            match = USAGE_DOC_REF_RE.match(usage_ref)
            if not match:
                blockers.append(
                    {
                        "code": "stage_script_registry_usage_doc_ref_invalid",
                        "script_id": script_id,
                        "usage_doc_ref": usage_ref,
                        "retryable": False,
                    }
                )
                continue
            usage_doc_path = match.group("path")
            doc_path = (repo_root / usage_doc_path).resolve()
            allowed_docs_root = (
                docs_root_resolved
                if usage_doc_path.startswith("docs/")
                else (repo_root / ".claude").resolve()
            )
            try:
                doc_path.relative_to(repo_root_resolved)
                doc_path.relative_to(allowed_docs_root)
            except ValueError:
                blockers.append(
                    {
                        "code": "stage_script_registry_usage_doc_ref_outside_docs_root",
                        "script_id": script_id,
                        "usage_doc_ref": usage_ref,
                        "retryable": False,
                    }
                )
                continue
            if not doc_path.is_file():
                blockers.append(
                    {
                        "code": "stage_script_registry_usage_doc_missing",
                        "script_id": script_id,
                        "usage_doc_ref": usage_ref,
                        "retryable": False,
                    }
                )
                continue
            text = doc_path.read_text(encoding="utf-8")
            anchor = match.group("anchor")
            if not _usage_doc_anchor_exists(text, anchor):
                blockers.append(
                    {
                        "code": "stage_script_registry_usage_doc_anchor_missing",
                        "script_id": script_id,
                        "usage_doc_ref": usage_ref,
                        "retryable": False,
                    }
                )
                continue
            negative_context += "\n" + _usage_doc_anchor_context(text, anchor)

        if str(script.get("status") or "") in {"deprecated", "forbidden"}:
            lower_context = negative_context.lower()
            if not any(term in lower_context for term in NEGATIVE_USAGE_DOC_TERMS):
                blockers.append(
                    {
                        "code": "stage_script_registry_deprecated_usage_doc_not_negative",
                        "script_id": script_id,
                        "usage_doc_refs": usage_refs,
                        "retryable": False,
                    }
                )
    return blockers


def refs_for_script_ids(
    registry: Mapping[str, Any],
    script_ids: Iterable[str],
) -> tuple[str, ...]:
    """Return de-duplicated refs for a list of script ids."""

    refs: list[str] = []
    for script_id in script_ids:
        for ref in refs_for_script_id(registry, str(script_id)):
            if ref not in refs:
                refs.append(ref)
    return tuple(refs)


def script_ids_for_task(registry: Mapping[str, Any], task_id: str) -> tuple[str, ...]:
    """Return required script ids for a governed task id."""

    binding = _task_bindings(registry).get(task_id)
    if not isinstance(binding, Mapping):
        return ()
    return tuple(str(item) for item in binding.get("required_script_ids") or ())


def forbidden_script_ids_for_task(registry: Mapping[str, Any], task_id: str) -> tuple[str, ...]:
    """Return forbidden script ids for a governed task id."""

    binding = _task_bindings(registry).get(task_id)
    if not isinstance(binding, Mapping):
        return ()
    return tuple(str(item) for item in binding.get("forbidden_script_ids") or ())


def required_script_ids_for_stage(registry: Mapping[str, Any], stage_id: str) -> tuple[str, ...]:
    """Return required script ids for a workflow/production/presentation stage."""

    binding = _stage_bindings(registry).get(stage_id)
    if not isinstance(binding, Mapping):
        return ()
    return tuple(str(item) for item in binding.get("required_script_ids") or ())


def forbidden_script_ids_for_stage(registry: Mapping[str, Any], stage_id: str) -> tuple[str, ...]:
    """Return forbidden script ids for a workflow/production/presentation stage."""

    binding = _stage_bindings(registry).get(stage_id)
    if not isinstance(binding, Mapping):
        return ()
    return tuple(str(item) for item in binding.get("forbidden_script_ids") or ())


def script_ref_lookup(registry: Mapping[str, Any]) -> dict[str, set[str]]:
    """Map every textual ref/path to the script ids that declare it."""

    lookup: dict[str, set[str]] = {}
    for script_id in _scripts(registry):
        for ref in refs_for_script_id(registry, script_id):
            lookup.setdefault(ref, set()).add(script_id)
    return lookup


def _upgrade_v1_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    tasks = dict(payload.get("tasks") or {})
    scripts: dict[str, dict[str, object]] = {}
    task_bindings: dict[str, dict[str, object]] = {}
    for task_id, task_value in tasks.items():
        task = task_value if isinstance(task_value, Mapping) else {}
        script_ids: list[str] = []
        for owner_ref in task.get("approved_owner_refs") or ():
            script_id = f"legacy.{task_id}.{len(script_ids) + 1}"
            scripts[script_id] = {
                "refs": [str(owner_ref)],
                "path": str(owner_ref),
                "kind": "script",
                "status": "required",
                "scope": "legacy",
                "existence": "allowed",
                "usage_doc_refs": [
                    "docs/reference/stage-owner-audit.md#verification-stage-script-usage"
                ],
                "diagnostic_profile": {
                    "success_summary": "Legacy upgraded registry owner passed compatibility validation.",
                    "blocked_summary": "Legacy upgraded registry owner is missing required compatibility evidence.",
                    "next_actions": [
                        "Proceed only as a compatibility proof; migrate the owner to a schema v2 script entry."
                    ],
                    "blocked_next_actions": [
                        "Add the missing owner evidence or migrate the task to a schema v2 script binding."
                    ],
                    "outputs": [
                        "compatibility audit report"
                    ],
                },
            }
            script_ids.append(script_id)
        task_bindings[str(task_id)] = {
            "approved_script_ids": script_ids,
            "required_script_ids": script_ids,
            "required_command_terms": list(task.get("required_command_terms") or ()),
            "forbidden_script_ids": [],
        }
    return {
        "schema_version": 2,
        "registry_name": "stage_script_registry",
        "planning_required_task_ids": list(payload.get("planning_required_task_ids") or ()),
        "scripts": scripts,
        "stage_bindings": {},
        "task_bindings": task_bindings,
        "skill_contracts": {},
        "evidence_rules": {},
        "tasks": tasks,
    }


def _validate_v2_registry(payload: Mapping[str, Any], registry_path: Path) -> None:
    for key in (
        "scripts",
        "stage_bindings",
        "task_bindings",
        "skill_contracts",
        "evidence_rules",
        "tasks",
    ):
        if not isinstance(payload.get(key), Mapping):
            raise StageScriptRegistryError(
                f"Stage script registry missing {key} mapping: {registry_path}"
            )

    scripts = _scripts(payload)
    allowed_status = {"allowed", "deprecated", "forbidden", "required"}
    allowed_existence = {"allowed", "planned", "required"}
    for script_id, script in scripts.items():
        if not isinstance(script, Mapping):
            raise StageScriptRegistryError(f"Malformed registry script: {script_id}")
        if str(script.get("status") or "") not in allowed_status:
            raise StageScriptRegistryError(
                f"Registry script {script_id} has invalid status"
            )
        if str(script.get("existence") or "") not in allowed_existence:
            raise StageScriptRegistryError(
                f"Registry script {script_id} has invalid existence"
            )
        refs = script.get("refs")
        if not isinstance(refs, list) or not refs:
            raise StageScriptRegistryError(f"Registry script {script_id} has no refs")
        usage_doc_refs = script.get("usage_doc_refs")
        if not isinstance(usage_doc_refs, list) or not usage_doc_refs:
            raise StageScriptRegistryError(
                f"Registry script {script_id} has no usage_doc_refs"
            )
        if not all(isinstance(ref, str) and ref for ref in usage_doc_refs):
            raise StageScriptRegistryError(
                f"Registry script {script_id} has invalid usage_doc_refs"
            )
        if not str(script.get("path") or ""):
            raise StageScriptRegistryError(f"Registry script {script_id} has no path")
        profile = script.get("diagnostic_profile")
        if not isinstance(profile, Mapping):
            raise StageScriptRegistryError(
                f"Registry script {script_id} has no diagnostic_profile"
            )
        for field in ("success_summary", "blocked_summary"):
            if not isinstance(profile.get(field), str) or not str(profile.get(field)).strip():
                raise StageScriptRegistryError(
                    f"Registry script {script_id} has invalid diagnostic_profile.{field}"
                )
        for field in ("next_actions", "blocked_next_actions"):
            if not _nonempty_string_list(profile.get(field)):
                raise StageScriptRegistryError(
                    f"Registry script {script_id} has invalid diagnostic_profile.{field}"
                )

    for collection_name in ("stage_bindings", "task_bindings"):
        for binding_id, binding in dict(payload[collection_name]).items():
            if not isinstance(binding, Mapping):
                raise StageScriptRegistryError(
                    f"Malformed {collection_name} binding: {binding_id}"
                )
            for field in (
                "approved_script_ids",
                "optional_script_ids",
                "required_script_ids",
                "forbidden_script_ids",
            ):
                values = binding.get(field, [])
                if values is None:
                    values = []
                if not isinstance(values, list):
                    raise StageScriptRegistryError(
                        f"{collection_name}.{binding_id}.{field} must be a list"
                    )
                unknown = sorted(str(value) for value in values if str(value) not in scripts)
                if unknown:
                    raise StageScriptRegistryError(
                        f"{collection_name}.{binding_id}.{field} references unknown scripts: {unknown}"
                    )

    _validate_runtime_contracts(payload)
    _validate_lineage_substeps(payload)

    for skill_name, contract in dict(payload["skill_contracts"]).items():
        if not isinstance(contract, Mapping):
            raise StageScriptRegistryError(f"Malformed skill contract: {skill_name}")
        for field in ("required_script_ids", "allowed_script_ids"):
            values = contract.get(field, [])
            if values is None:
                values = []
            if not isinstance(values, list):
                raise StageScriptRegistryError(
                    f"skill_contracts.{skill_name}.{field} must be a list"
                )
            unknown = sorted(str(value) for value in values if str(value) not in scripts)
            if unknown:
                raise StageScriptRegistryError(
                    f"skill_contracts.{skill_name}.{field} references unknown scripts: {unknown}"
                )

    for script_id, script in scripts.items():
        if str(script.get("status") or "") not in {"deprecated", "forbidden"}:
            continue
        profile = script.get("diagnostic_profile")
        replacements = (
            profile.get("replacement_script_ids")
            if isinstance(profile, Mapping)
            else None
        )
        if not _nonempty_string_list(replacements):
            raise StageScriptRegistryError(
                f"Registry script {script_id} deprecated/forbidden diagnostic_profile has no replacement_script_ids"
            )
        unknown = sorted(
            str(value)
            for value in replacements
            if str(value) not in scripts
        )
        if unknown:
            raise StageScriptRegistryError(
                f"Registry script {script_id} diagnostic_profile references unknown replacements: {unknown}"
            )


_LINEAGE_MODES = frozenset({"output_digest", "content"})
_UPSTREAM_LAYOUTS = frozenset({"production", "generic"})


def _validate_lineage_substeps(payload: Mapping[str, Any]) -> None:
    """Validate ``lineage_substeps``: sub-step receipts that bind like stages.

    Each entry needs a ``receipt`` path template, and may list
    ``required_upstream_stages`` (known stage bindings with a
    ``receipt_filename``), ``required_substeps`` (other declared
    sub-steps), ``outputs`` (file or directory paths) and ``input_artifacts``
    (``{name, path, exclude_keys?}``). Receipt paths are unique and the
    sub-step graph is acyclic.
    """

    substeps = payload.get("lineage_substeps")
    if substeps is None:
        return
    if not isinstance(substeps, Mapping):
        raise StageScriptRegistryError("lineage_substeps must be an object")
    stage_bindings = payload.get("stage_bindings")
    stage_bindings = stage_bindings if isinstance(stage_bindings, Mapping) else {}
    receipts: dict[str, str] = {}
    graph: dict[str, list[str]] = {}
    for substep_id, spec in substeps.items():
        if not isinstance(spec, Mapping):
            raise StageScriptRegistryError(f"lineage_substeps.{substep_id} must be an object")
        receipt = spec.get("receipt")
        if not isinstance(receipt, str) or not receipt:
            raise StageScriptRegistryError(f"lineage_substeps.{substep_id}.receipt must be a path")
        if receipt in receipts:
            raise StageScriptRegistryError(
                f"lineage_substeps.{substep_id}.receipt duplicates {receipts[receipt]}"
            )
        receipts[receipt] = str(substep_id)
        for stage_id in spec.get("required_upstream_stages") or []:
            binding = stage_bindings.get(stage_id)
            if not isinstance(binding, Mapping) or not binding.get("receipt_filename"):
                raise StageScriptRegistryError(
                    f"lineage_substeps.{substep_id}.required_upstream_stages names {stage_id!r}, "
                    "which is not a stage binding with a receipt_filename"
                )
        upstream = [str(item) for item in spec.get("required_substeps") or []]
        unknown = [item for item in upstream if item not in substeps]
        if unknown:
            raise StageScriptRegistryError(
                f"lineage_substeps.{substep_id}.required_substeps names unknown sub-steps {unknown}"
            )
        for artifact in spec.get("input_artifacts") or []:
            if (
                not isinstance(artifact, Mapping)
                or not isinstance(artifact.get("path"), str)
                or not artifact.get("path")
                or not all(isinstance(key, str) for key in artifact.get("exclude_keys") or [])
            ):
                raise StageScriptRegistryError(
                    f"lineage_substeps.{substep_id}.input_artifacts entries need a path "
                    "and optional string exclude_keys"
                )
        outputs = spec.get("outputs")
        if outputs is not None and (
            not isinstance(outputs, list) or not all(isinstance(item, str) and item for item in outputs)
        ):
            raise StageScriptRegistryError(f"lineage_substeps.{substep_id}.outputs must be a list of paths")
        graph[str(substep_id)] = upstream
    state: dict[str, int] = {}

    def _visit(node: str, trail: tuple[str, ...]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            raise StageScriptRegistryError(f"lineage_substeps cycle: {' -> '.join((*trail, node))}")
        state[node] = 1
        for child in graph.get(node, []):
            _visit(child, (*trail, node))
        state[node] = 2

    for node in graph:
        _visit(node, ())


def _validate_runtime_contracts(payload: Mapping[str, Any]) -> None:
    """Validate the lineage vocabulary of every stage binding's runtime contract.

    ``required_receipts[]`` entries must be objects with a string ``path`` and
    may carry ``lineage`` (``output_digest`` or ``content``); the mtime-based
    ``fresh_after`` keys are rejected. ``required_upstream_stages`` must name
    known stage bindings, ``lineage.input_artifacts[]`` entries need a string
    ``path``, and ``receipt_filename`` basenames must be unique across bindings
    so a receipt path resolves to exactly one stage.
    """

    stage_bindings = payload.get("stage_bindings")
    stage_bindings = stage_bindings if isinstance(stage_bindings, Mapping) else {}
    known_stage_ids = {str(key) for key in stage_bindings}
    receipt_filenames: dict[str, str] = {}

    def _contract_views(binding_id: str, binding: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
        views: list[tuple[str, Mapping[str, Any]]] = []
        contract = binding.get("runtime_contract")
        if isinstance(contract, Mapping):
            views.append((binding_id, contract))
        profiles = binding.get("runtime_contract_profiles")
        if isinstance(profiles, Mapping):
            for profile_name, profile_contract in profiles.items():
                if isinstance(profile_contract, Mapping):
                    views.append((f"{binding_id}[{profile_name}]", profile_contract))
        workspace_bindings = binding.get("workspace_bindings")
        if isinstance(workspace_bindings, Mapping):
            for workspace_id, workspace_binding in workspace_bindings.items():
                if isinstance(workspace_binding, Mapping):
                    views.extend(_contract_views(f"{binding_id}@{workspace_id}", workspace_binding))
        return views

    for binding_id, binding in stage_bindings.items():
        if not isinstance(binding, Mapping):
            continue
        filename = binding.get("receipt_filename")
        if filename is not None:
            if not isinstance(filename, str) or not filename or "/" in filename:
                raise StageScriptRegistryError(
                    f"stage_bindings.{binding_id}.receipt_filename must be a bare filename"
                )
            owner = receipt_filenames.get(filename)
            if owner is not None:
                raise StageScriptRegistryError(
                    f"stage_bindings.{binding_id}.receipt_filename {filename!r} already used by {owner}"
                )
            receipt_filenames[filename] = str(binding_id)
            order = binding.get("stage_order")
            if order is not None and (isinstance(order, bool) or not isinstance(order, int) or order < 1):
                raise StageScriptRegistryError(
                    f"stage_bindings.{binding_id}.stage_order must be a positive integer"
                )
        for view_id, contract in _contract_views(str(binding_id), binding):
            for index, spec in enumerate(contract.get("required_receipts") or []):
                label = f"stage_bindings.{view_id}.required_receipts[{index}]"
                if not isinstance(spec, Mapping):
                    raise StageScriptRegistryError(f"{label} must be an object")
                if not isinstance(spec.get("path"), str) or not spec.get("path"):
                    raise StageScriptRegistryError(f"{label} must declare a string path")
                if "fresh_after" in spec or "fresh_after_optional" in spec:
                    raise StageScriptRegistryError(
                        f"{label} uses fresh_after; bind by lineage digest instead"
                    )
                lineage_mode = spec.get("lineage")
                if lineage_mode is not None and lineage_mode not in _LINEAGE_MODES:
                    raise StageScriptRegistryError(
                        f"{label}.lineage must be one of {sorted(_LINEAGE_MODES)}"
                    )
            upstream = contract.get("required_upstream_stages")
            if upstream is not None:
                if not isinstance(upstream, list):
                    raise StageScriptRegistryError(
                        f"stage_bindings.{view_id}.required_upstream_stages must be a list of stage ids"
                    )
                upstream_ids: list[str] = []
                for item in upstream:
                    if isinstance(item, str):
                        upstream_ids.append(item)
                        continue
                    layouts = item.get("layouts") if isinstance(item, Mapping) else None
                    if (
                        not isinstance(item, Mapping)
                        or not isinstance(item.get("stage"), str)
                        or set(item) - {"stage", "layouts"}
                        or not isinstance(layouts, list)
                        or not layouts
                        or not set(layouts) <= _UPSTREAM_LAYOUTS
                    ):
                        raise StageScriptRegistryError(
                            f"stage_bindings.{view_id}.required_upstream_stages entries must be a stage id or "
                            f"{{stage, layouts}} with layouts drawn from {sorted(_UPSTREAM_LAYOUTS)}"
                        )
                    upstream_ids.append(item["stage"])
                unknown = sorted(item for item in upstream_ids if item not in known_stage_ids)
                if unknown:
                    raise StageScriptRegistryError(
                        f"stage_bindings.{view_id}.required_upstream_stages references unknown stages: {unknown}"
                    )
            lineage_spec = contract.get("lineage")
            if lineage_spec is not None:
                if not isinstance(lineage_spec, Mapping):
                    raise StageScriptRegistryError(f"stage_bindings.{view_id}.lineage must be an object")
                for index, spec in enumerate(lineage_spec.get("input_artifacts") or []):
                    label = f"stage_bindings.{view_id}.lineage.input_artifacts[{index}]"
                    if not isinstance(spec, Mapping) or not isinstance(spec.get("path"), str) or not spec.get("path"):
                        raise StageScriptRegistryError(f"{label} must declare a string path")
            for table_key in ("output_artifact_checks", "input_artifact_checks"):
                table = contract.get(table_key)
                if table is None:
                    continue
                if not isinstance(table, list) or not all(
                    isinstance(row, list) and 2 <= len(row) <= 3 and isinstance(row[0], str) and isinstance(row[1], str)
                    for row in table
                ):
                    raise StageScriptRegistryError(
                        f"stage_bindings.{view_id}.{table_key} rows must be [name, path_key, sha_key?]"
                    )


def _scripts(registry: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    scripts = registry.get("scripts")
    if not isinstance(scripts, Mapping):
        raise StageScriptRegistryError("Stage script registry missing scripts mapping")
    return scripts  # type: ignore[return-value]


def _nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and item.strip() for item in value
    )


def _usage_doc_anchor_exists(text: str, anchor: str) -> bool:
    quoted_anchor = re.escape(anchor)
    if re.search(rf"""(?:id|name)=["']{quoted_anchor}["']""", text):
        return True
    return any(_markdown_heading_slug(line) == anchor for line in text.splitlines())


def _usage_doc_anchor_context(text: str, anchor: str) -> str:
    id_match = re.search(
        rf"""(?:id|name)=["']{re.escape(anchor)}["']""",
        text,
    )
    if id_match:
        return text[id_match.start() : id_match.start() + 1600]
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _markdown_heading_slug(line) == anchor:
            return "\n".join(lines[index : index + 16])
    return ""


def _markdown_heading_slug(line: str) -> str:
    if not line.startswith("#"):
        return ""
    heading = line.lstrip("#").strip().lower()
    heading = re.sub(r"`([^`]+)`", r"\1", heading)
    heading = re.sub(r"[^a-z0-9 -]", "", heading)
    heading = re.sub(r"\s+", "-", heading)
    return heading.strip("-")


def _stage_bindings(registry: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    bindings = registry.get("stage_bindings")
    if not isinstance(bindings, Mapping):
        raise StageScriptRegistryError("Stage script registry missing stage_bindings")
    return bindings  # type: ignore[return-value]


def _task_bindings(registry: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    bindings = registry.get("task_bindings")
    if not isinstance(bindings, Mapping):
        raise StageScriptRegistryError("Stage script registry missing task_bindings")
    return bindings  # type: ignore[return-value]
