"""Anti-tautology runtime-contract coverage + receipt-name collision guards.

`stage_contracts.verify_stage_contract` proves a *specific* receipt satisfies a
stage binding's `runtime_contract`. It does not prove the `runtime_contract`
itself is non-vacuous -- a binding can declare `required_script_ids` while its
`runtime_contract` never actually evidences those owners with a distinct
artifact (receipt path, output glob, or check). This module closes that gap
with two deterministic, registry-level (not receipt-level) audits:

- `audit_runtime_contract_coverage`: every `required_script_ids` entry in a
  stage binding must map to at least one `required_receipts` /
  `required_outputs` / `required_checks` entry naming an artifact that only
  the real owner produces, unless governed via `coverage_exempt_script_ids` +
  `coverage_exempt_rationale`.
- `audit_receipt_name_collisions`: within one stage binding, two distinct
  required-receipt paths must not share a stage-number prefix (e.g. `14A`)
  unless the pair is on a governed `receipt_prefix_collision_allow`
  allowlist. This blocks new naming collisions like the one that let a
  legacy heuristic receipt masquerade as the real owner's evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_PREFIX_RE = re.compile(r"^(\d+[A-Z]?)")
_SEPARATOR_RE = re.compile(r"[/._]")


def audit_runtime_contract_coverage(
    registry: Mapping[str, Any],
    *,
    stage_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fail closed on any `required_script_ids` entry lacking real evidence."""

    stage_bindings = registry.get("stage_bindings") or {}
    selected_ids = list(stage_ids) if stage_ids is not None else sorted(stage_bindings)
    blockers: list[dict[str, Any]] = []
    uncovered: list[dict[str, str]] = []

    for stage_id in selected_ids:
        binding = stage_bindings.get(stage_id)
        if not isinstance(binding, Mapping):
            continue
        binding_views: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
        contract = binding.get("runtime_contract")
        if isinstance(contract, Mapping):
            binding_views.append((stage_id, binding, contract))

        workspace_bindings = binding.get("workspace_bindings")
        if isinstance(workspace_bindings, Mapping):
            for workspace, workspace_binding in workspace_bindings.items():
                if not isinstance(workspace_binding, Mapping):
                    continue
                workspace_contract = workspace_binding.get("runtime_contract")
                if isinstance(workspace_contract, Mapping):
                    binding_views.append(
                        (
                            f"{stage_id}@{workspace}",
                            workspace_binding,
                            workspace_contract,
                        )
                    )

        for coverage_stage_id, owner_binding, contract in binding_views:
            if bool(contract.get("non_production")):
                # Mirrors verify_stage_contract's own short-circuit: a stage
                # explicitly marked non-production (legacy alias, validation-only,
                # or not-yet-implemented) has no live runtime evidence to cover.
                continue
            evidence_blob = _evidence_blob(contract)
            exempt_ids = {
                str(item) for item in owner_binding.get("coverage_exempt_script_ids") or []
            }
            exempt_rationale = owner_binding.get("coverage_exempt_rationale")
            exempt_rationale = (
                exempt_rationale if isinstance(exempt_rationale, Mapping) else {}
            )

            for raw_script_id in owner_binding.get("required_script_ids") or []:
                script_id = str(raw_script_id)
                if script_id in exempt_ids:
                    rationale = exempt_rationale.get(script_id)
                    if not (isinstance(rationale, str) and rationale.strip()):
                        blockers.append(
                            _blocker(
                                "stage_contract_coverage_exemption_missing_rationale",
                                stage_id=coverage_stage_id,
                                script_id=script_id,
                                expected="non-empty coverage_exempt_rationale entry",
                                observed=rationale if rationale is not None else "<missing>",
                            )
                        )
                    continue
                candidates = _candidate_tokens(script_id)
                if any(candidate in evidence_blob for candidate in candidates):
                    continue
                uncovered.append({"stage_id": coverage_stage_id, "script_id": script_id})
                blockers.append(
                    _blocker(
                        "stage_contract_owner_id_uncovered",
                        stage_id=coverage_stage_id,
                        script_id=script_id,
                        expected=(
                            "a required_receipts/required_outputs/required_checks entry "
                            "evidencing this owner id, or a governed coverage_exempt_"
                            "script_ids entry with rationale"
                        ),
                        observed="<no matching evidence>",
                    )
                )

    return {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "uncovered": uncovered,
    }


def audit_receipt_name_collisions(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on new stage-number-prefix collisions within one binding."""

    stage_bindings = registry.get("stage_bindings") or {}
    allowlist_pairs = _allowlisted_pairs(registry)
    blockers: list[dict[str, Any]] = []

    for stage_id, binding in stage_bindings.items():
        if not isinstance(binding, Mapping):
            continue
        contract = binding.get("runtime_contract")
        contract = contract if isinstance(contract, Mapping) else {}
        basenames: list[str] = []
        for spec in _as_list(contract.get("required_receipts")):
            path = spec.get("path")
            if isinstance(path, str) and path:
                basenames.append(Path(path).name)

        prefix_map: dict[str, list[str]] = {}
        for name in basenames:
            match = _PREFIX_RE.match(name)
            if not match:
                continue
            prefix_map.setdefault(match.group(1), []).append(name)

        for prefix, names in prefix_map.items():
            distinct_names = sorted(set(names))
            if len(distinct_names) < 2:
                continue
            for i, name_a in enumerate(distinct_names):
                for name_b in distinct_names[i + 1 :]:
                    if frozenset({name_a, name_b}) in allowlist_pairs:
                        continue
                    blockers.append(
                        {
                            "code": "stage_contract_receipt_prefix_collision",
                            "stage_id": stage_id,
                            "prefix": prefix,
                            "receipts": [name_a, name_b],
                            "retryable": False,
                        }
                    )

    return {"status": "pass" if not blockers else "blocked", "blockers": blockers}


def _allowlisted_pairs(registry: Mapping[str, Any]) -> set[frozenset[str]]:
    allowlist_raw = registry.get("receipt_prefix_collision_allow") or []
    pairs: set[frozenset[str]] = set()
    if not isinstance(allowlist_raw, list):
        return pairs
    for entry in allowlist_raw:
        if not isinstance(entry, Mapping):
            continue
        pair = entry.get("pair")
        rationale = entry.get("rationale")
        if (
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(item, str) and item for item in pair)
            and isinstance(rationale, str)
            and rationale.strip()
        ):
            pairs.add(frozenset(str(item) for item in pair))
    return pairs


def _evidence_blob(contract: Mapping[str, Any]) -> str:
    strings: list[str] = []
    for spec in _as_list(contract.get("required_receipts")):
        path = spec.get("path")
        if isinstance(path, str):
            strings.append(path)
        for check in _as_list(spec.get("checks")):
            strings.extend(_check_spec_strings(check))
    for spec in _as_list(contract.get("required_outputs")):
        value = spec.get("glob") or spec.get("path")
        if isinstance(value, str):
            strings.append(value)
    for spec in _as_list(contract.get("required_checks")):
        strings.extend(_check_spec_strings(spec))
    return " ".join(_normalize(value) for value in strings if isinstance(value, str))


def _check_spec_strings(spec: Mapping[str, Any]) -> list[str]:
    strings: list[str] = []
    path = spec.get("path")
    if isinstance(path, str):
        strings.append(path)
    for key in ("equals", "not_equals", "equals_path"):
        value = spec.get(key)
        if isinstance(value, str):
            strings.append(value)
    one_of = spec.get("one_of")
    if isinstance(one_of, list):
        strings.extend(str(item) for item in one_of if isinstance(item, str))
    return strings


def _candidate_tokens(script_id: str) -> list[str]:
    tail = script_id.rsplit(".", 1)[-1] if "." in script_id else script_id
    return sorted({_normalize(script_id), _normalize(tail)})


def _normalize(value: str) -> str:
    return _SEPARATOR_RE.sub("-", value.lower())


def _as_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _blocker(
    code: str,
    *,
    stage_id: str,
    script_id: str,
    expected: Any,
    observed: Any,
) -> dict[str, Any]:
    return {
        "code": code,
        "stage_id": stage_id,
        "script_id": script_id,
        "expected": expected,
        "observed": observed,
        "retryable": False,
    }
