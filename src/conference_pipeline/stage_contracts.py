"""Runtime stage contract verification.

The stage script registry proves which owners a stage must use. This module
checks whether a specific workspace/run receipt actually enforced those owners
before a production runner records success.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.event_media import (
    EVENT_CATEGORY_COUNTS,
    EVENT_MEDIA_CAPTIONS_RELATIVE_PATH,
    EVENT_MEDIA_INVENTORY_RELATIVE_PATH,
    R4_REUSED_POSTER_ROUTE_COUNT,
    R5_EVENT_MEDIA_RUN_ID,
    canonical_event_media_inventory_digest,
    validate_event_media_inventory,
    validate_r4_poster_reuse,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_ARTIFACT_PATHS,
    R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT,
    R6_STAGE_ID,
    R6ArtifactContract,
)
from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
    R6_RUN_ID as R6_EVENT_MEDIA_RUN_ID,
)
from conference_pipeline.acquisition.registration_media import (
    FULL_R4_SCOPE,
    OVERLAY_LEDGER_RELATIVE_PATH,
    PINNED_R3_PREDECESSORS,
    R4_RUN_ID,
    canonical_registration_media_digest,
    load_pinned_r3_predecessors,
    validate_registration_media_overlay,
)
from conference_pipeline.acquisition.cvpr_transcription_ocr import (
    PROCESSOR_IDENTITY,
    PROCESSOR_VERSION,
    R4_OVERLAY_DIGEST,
    R5_EVENT_MEDIA_CANONICAL_DIGEST,
    R5_EVENT_MEDIA_FILE_SHA256,
    R5_EVENT_MEDIA_RECEIPT_SHA256,
    STAGE5_INVENTORY,
)
from conference_pipeline.acquisition.youtube_talks import caption_sidecar_matches_video_id
from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    load_conference_identity,
    parse_virtual_site_boundary,
)
from conference_pipeline import receipt_lineage
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)

SUCCEEDED_STATUSES = frozenset({"succeeded", "success", "complete", "completed", "pass"})
# Whether a supplied receipt must carry a ``lineage`` block. Callers that bind
# lineage pass ``require_lineage=True`` explicitly; the default flips to True
# once every production runner binds lineage (v5.1 plan 01-02), so a runner
# that has not been migrated keeps working until then and fails closed after.
REQUIRE_LINEAGE_DEFAULT = True
NON_LIVE_MODES = frozenset({"mock", "dry-run", "dry_run", "planned"})
MODEL_REGISTRY_PATH = Path(__file__).with_name("model_governance") / "registry.yaml"
TASK_SPECS_PATH = Path(__file__).with_name("costs") / "task_specs.yaml"
PRICING_CATALOG_PATH = Path(__file__).with_name("costs") / "pricing_catalog.json"


@dataclass(frozen=True)
class StageContractResult:
    """Structured result suitable for embedding in stage receipts."""

    status: str
    stage_id: str
    registry_stage_id: str
    script_registry_sha256: str
    required_script_ids: list[str]
    accepted_evidence: list[str]
    blockers: list[dict[str, Any]]
    warnings: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage_id": self.stage_id,
            "registry_stage_id": self.registry_stage_id,
            "script_registry_sha256": self.script_registry_sha256,
            "required_script_ids": list(self.required_script_ids),
            "accepted_evidence": list(self.accepted_evidence),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class StageContractBinding:
    """Resolved registry binding for one workspace and stage."""

    stage_id: str
    registry_stage_id: str
    required_script_ids: list[str]
    runtime_contract: Mapping[str, Any]


@dataclass(frozen=True)
class PreSpendContractResult:
    """Secret-free zero-spend result for a prospective provider dispatch.

    This is deliberately not a :class:`StageContractResult`: a passing
    precondition only proves that an adapter may *consider* dispatch. It never
    proves final stage outputs, records success, or carries a dispatch grant.
    """

    status: str
    stage_id: str
    registry_stage_id: str
    script_registry_sha256: str
    required_script_ids: list[str]
    accepted_evidence: list[str]
    blockers: list[dict[str, Any]]
    provider_dispatch_permitted: bool = False
    provider_calls_made: int = 0
    actual_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "contract_mode": "pre-spend",
            "stage_id": self.stage_id,
            "registry_stage_id": self.registry_stage_id,
            "script_registry_sha256": self.script_registry_sha256,
            "required_script_ids": list(self.required_script_ids),
            "accepted_evidence": list(self.accepted_evidence),
            "blockers": list(self.blockers),
            "provider_dispatch_permitted": self.provider_dispatch_permitted,
            "provider_calls_made": self.provider_calls_made,
            "actual_cost_usd": self.actual_cost_usd,
        }


def resolve_stage_contract_binding(
    *,
    workspace: str | Path,
    stage_id: str,
    registry_path: str | Path | None = None,
    profile: str | None = None,
) -> StageContractBinding:
    """Resolve a stage binding, enforcing any workspace-specific owner policy.

    A registry stage can provide generic defaults plus ``workspace_bindings``.
    Once a stage declares those bindings, a workspace must normalize to one of
    their explicit identities: silently falling back to a generic owner would
    let one conference satisfy another's production contract.
    """

    registry = load_stage_script_registry(registry_path)
    binding = (registry.get("stage_bindings") or {}).get(stage_id)
    if not isinstance(binding, Mapping):
        raise ValueError(f"workspace stage binding missing for {stage_id!r}")

    workspace_bindings = binding.get("workspace_bindings")
    workspace_binding: Mapping[str, Any] = {}
    if isinstance(workspace_bindings, Mapping):
        workspace_identity = _normalized_workspace_identity(workspace)
        candidate = workspace_bindings.get(workspace_identity)
        if not isinstance(candidate, Mapping):
            raise ValueError(
                "workspace stage binding missing for "
                f"{stage_id!r} and {workspace_identity!r}"
            )
        workspace_binding = candidate

    runtime_contract = workspace_binding.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        runtime_contract = binding.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        runtime_contract = {}
    if profile is not None:
        profiles = workspace_binding.get("runtime_contract_profiles")
        profiles = profiles if isinstance(profiles, Mapping) else binding.get("runtime_contract_profiles")
        selected = profiles.get(profile) if isinstance(profiles, Mapping) else None
        if not isinstance(selected, Mapping):
            raise ValueError(
                f"runtime contract profile {profile!r} missing for {stage_id!r}"
            )
        runtime_contract = selected
    registry_stage_id = str(
        workspace_binding.get("registry_stage_id")
        or runtime_contract.get("registry_stage_id")
        or stage_id
    )
    required_script_ids = workspace_binding.get("required_script_ids")
    if not isinstance(required_script_ids, list):
        required_script_ids = list(required_script_ids_for_stage(registry, stage_id))

    return StageContractBinding(
        stage_id=stage_id,
        registry_stage_id=registry_stage_id,
        required_script_ids=[str(item) for item in required_script_ids],
        runtime_contract=runtime_contract,
    )


def verify_pre_spend_contract(
    *,
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    mode: str,
    frontier_authority: Mapping[str, Any] | None,
    base_full_estimate_digest: str,
    candidate_frontier_sha256: str,
    retained_join_quantity: int,
    provider: str,
    provider_stage: str,
    volatile_preflight: Mapping[str, Any] | None,
    registry_path: str | Path | None = None,
) -> PreSpendContractResult:
    """Validate a frontier-bound authorization before a provider adapter exists.

    The result is intentionally zero-spend and side-effect-free. A later owner
    must still perform its own dispatch, provider receipt, and final output
    contract validation; callers must never treat ``precondition_passed`` as a
    successful production stage.
    """

    script_digest = registry_sha256(registry_path)
    blockers: list[dict[str, Any]] = []
    accepted: list[str] = []
    try:
        binding = resolve_stage_contract_binding(
            workspace=workspace,
            stage_id=stage_id,
            registry_path=registry_path,
        )
    except ValueError:
        return _pre_spend_result(
            stage_id=stage_id,
            registry_stage_id=stage_id,
            script_digest=script_digest,
            required_script_ids=[],
            accepted=accepted,
            blockers=[_precondition_blocker(stage_id, "workspace_owner_binding")],
        )

    authority = frontier_authority if isinstance(frontier_authority, Mapping) else {}

    def require(condition: bool, precondition: str) -> None:
        if condition:
            accepted.append(f"{stage_id}: {precondition}")
        else:
            blockers.append(_precondition_blocker(stage_id, precondition))

    require(mode == "live", "live_mode")
    require(
        authority.get("authority_kind") == "source_discovery_frontier_bound",
        "frontier_authority_kind",
    )
    require(authority.get("run_id") == run_id, "run_id")
    require(
        _same_workspace(authority.get("workspace"), workspace),
        "workspace_identity",
    )
    require(
        _digest_matches(authority.get("base_full_estimate_digest"), base_full_estimate_digest),
        "base_estimate_lineage",
    )
    require(
        _digest_matches(authority.get("model_registry_sha256"), _sha256_file(MODEL_REGISTRY_PATH)),
        "model_registry_digest",
    )
    require(
        _digest_matches(authority.get("candidate_frontier_sha256"), candidate_frontier_sha256),
        "candidate_frontier_digest",
    )
    require(
        authority.get("retained_join_quantity") == retained_join_quantity
        and isinstance(retained_join_quantity, int)
        and retained_join_quantity > 0,
        "retained_join_quantity",
    )
    require(
        isinstance(authority.get("authority_digest"), str)
        and bool(str(authority.get("authority_digest")).strip()),
        "authority_digest",
    )
    require(
        authority.get("provider") == provider and authority.get("stage") == provider_stage,
        "provider_stage_scope",
    )

    allocations = authority.get("provider_stage_allocations")
    require(
        isinstance(allocations, list)
        and bool(allocations)
        and all(
            isinstance(allocation, Mapping)
            and isinstance(allocation.get("provider"), str)
            and bool(str(allocation.get("provider")).strip())
            and isinstance(allocation.get("stage"), str)
            and bool(str(allocation.get("stage")).strip())
            and _is_finite_nonnegative_number(allocation.get("ceiling_usd"))
            for allocation in allocations
        ),
        "priced_provider_stage_allocations",
    )
    matching_allocations = (
        [
            allocation
            for allocation in allocations
            if isinstance(allocation, Mapping)
            and allocation.get("provider") == provider
            and allocation.get("stage") == provider_stage
            and _is_finite_nonnegative_number(allocation.get("ceiling_usd"))
        ]
        if isinstance(allocations, list)
        else []
    )
    require(len(matching_allocations) == 1, "provider_stage_allocation")
    spend_authority = authority.get("spend_authority")
    require(
        isinstance(spend_authority, Mapping)
        and spend_authority.get("mode") in {"automatic_under_usd_800", "explicit_immutable"}
        and spend_authority.get("provider_stage_allocations") == allocations,
        "spend_authority_allocation",
    )
    require(authority.get("fresh_volatile_preflight_required") is True, "fresh_preflight_required")
    require(
        isinstance(volatile_preflight, Mapping)
        and volatile_preflight.get("status") == "succeeded"
        and volatile_preflight.get("frontier_authority_digest") == authority.get("authority_digest"),
        "fresh_volatile_preflight_identity",
    )

    return _pre_spend_result(
        stage_id=stage_id,
        registry_stage_id=binding.registry_stage_id,
        script_digest=script_digest,
        required_script_ids=binding.required_script_ids,
        accepted=accepted,
        blockers=blockers,
    )


def verify_tavily_priority_gap_pre_spend_contract(
    *,
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    mode: str,
    frontier_authority: Mapping[str, Any] | None,
    base_full_estimate_digest: str,
    program_spine_sha256: str,
    harvest_inventory_sha256: str,
    query_policy_sha256: str,
    priority_gap_frontier_sha256: str,
    priority_gap_quantity: int,
    provider: str,
    provider_stage: str,
    volatile_preflight: Mapping[str, Any] | None,
    registry_path: str | Path | None = None,
) -> PreSpendContractResult:
    """Validate the non-model, named Tavily priority-gap spend boundary.

    This is deliberately a second *scoped* pre-spend contract rather than a
    broader ``tavily`` switch.  The source-discovery variant above retains its
    own join-specific fields; this one binds the exact, already-public-harvested
    priority frontier.  Both return only a secret-free zero-spend result.
    """

    script_digest = registry_sha256(registry_path)
    accepted: list[str] = []
    blockers: list[dict[str, Any]] = []
    try:
        binding = resolve_stage_contract_binding(
            workspace=workspace,
            stage_id=stage_id,
            registry_path=registry_path,
        )
    except ValueError:
        return _pre_spend_result(
            stage_id=stage_id,
            registry_stage_id=stage_id,
            script_digest=script_digest,
            required_script_ids=[],
            accepted=accepted,
            blockers=[_precondition_blocker(stage_id, "workspace_owner_binding")],
        )

    authority = frontier_authority if isinstance(frontier_authority, Mapping) else {}

    def require(condition: bool, precondition: str) -> None:
        if condition:
            accepted.append(f"{stage_id}: {precondition}")
        else:
            blockers.append(_precondition_blocker(stage_id, precondition))

    priority_ids = authority.get("priority_program_item_ids")
    normalized_ids = (
        [str(item) for item in priority_ids]
        if isinstance(priority_ids, list) and all(isinstance(item, str) and item for item in priority_ids)
        else []
    )
    allocations = authority.get("provider_stage_allocations")
    matching_allocations = (
        [
            allocation
            for allocation in allocations
            if isinstance(allocation, Mapping)
            and allocation.get("provider") == provider
            and allocation.get("stage") == provider_stage
            and _is_finite_nonnegative_number(allocation.get("ceiling_usd"))
        ]
        if isinstance(allocations, list)
        else []
    )
    spend_authority = authority.get("spend_authority")
    provider_scope = volatile_preflight.get("provider_probe_scope") if isinstance(volatile_preflight, Mapping) else None

    require(mode == "live", "live_mode")
    require(
        authority.get("authority_kind") == "cvpr2026_tavily_priority_gap_bound",
        "priority_gap_authority_kind",
    )
    require(authority.get("run_id") == run_id, "run_id")
    require(_same_workspace(authority.get("workspace"), workspace), "workspace_identity")
    require(
        _digest_matches(authority.get("base_full_estimate_digest"), base_full_estimate_digest),
        "base_estimate_lineage",
    )
    require(
        _digest_matches(authority.get("model_registry_sha256"), _sha256_file(MODEL_REGISTRY_PATH)),
        "model_registry_digest",
    )
    require(
        _digest_matches(authority.get("task_specs_sha256"), _sha256_file(TASK_SPECS_PATH)),
        "task_specs_digest",
    )
    require(
        _digest_matches(authority.get("pricing_catalog_sha256"), _sha256_file(PRICING_CATALOG_PATH)),
        "pricing_catalog_digest",
    )
    require(
        _digest_matches(authority.get("program_spine_sha256"), program_spine_sha256),
        "program_spine_digest",
    )
    require(
        _digest_matches(authority.get("harvest_inventory_sha256"), harvest_inventory_sha256),
        "harvest_inventory_digest",
    )
    require(
        _digest_matches(authority.get("query_policy_sha256"), query_policy_sha256),
        "query_policy_digest",
    )
    require(
        _digest_matches(authority.get("priority_gap_frontier_sha256"), priority_gap_frontier_sha256),
        "priority_gap_frontier_digest",
    )
    require(
        isinstance(priority_gap_quantity, int)
        and priority_gap_quantity > 0
        and authority.get("priority_gap_quantity") == priority_gap_quantity
        and len(normalized_ids) == priority_gap_quantity
        and normalized_ids == sorted(normalized_ids)
        and len(set(normalized_ids)) == len(normalized_ids),
        "priority_gap_quantity",
    )
    require(
        authority.get("task_id") == "acquisition.youtube_talk_harvest.tavily_priority_gap_search"
        and authority.get("cost_task_id") == "cvpr2026_tavily_priority_gap_search",
        "governed_task_identity",
    )
    require(
        isinstance(authority.get("authority_digest"), str)
        and bool(str(authority.get("authority_digest")).strip()),
        "authority_digest",
    )
    require(
        authority.get("provider") == provider and authority.get("stage") == provider_stage,
        "provider_stage_scope",
    )
    require(
        isinstance(allocations, list)
        and bool(allocations)
        and all(
            isinstance(allocation, Mapping)
            and isinstance(allocation.get("provider"), str)
            and isinstance(allocation.get("stage"), str)
            and _is_finite_nonnegative_number(allocation.get("ceiling_usd"))
            for allocation in allocations
        ),
        "priced_provider_stage_allocations",
    )
    require(len(matching_allocations) == 1, "provider_stage_allocation")
    require(
        isinstance(spend_authority, Mapping)
        and spend_authority.get("mode") in {"automatic_under_usd_800", "explicit_immutable"}
        and spend_authority.get("provider_stage_allocations") == allocations,
        "spend_authority_allocation",
    )
    require(authority.get("fresh_volatile_preflight_required") is True, "fresh_preflight_required")
    require(
        isinstance(volatile_preflight, Mapping)
        and volatile_preflight.get("status") == "succeeded"
        and volatile_preflight.get("frontier_authority_digest") == authority.get("authority_digest")
        and provider_scope == {"provider": provider, "stage": provider_stage},
        "fresh_tavily_preflight_identity",
    )
    return _pre_spend_result(
        stage_id=stage_id,
        registry_stage_id=binding.registry_stage_id,
        script_digest=script_digest,
        required_script_ids=binding.required_script_ids,
        accepted=accepted,
        blockers=blockers,
    )


def verify_stage_contract(
    *,
    workspace: str | Path,
    run_id: str,
    stage_id: str,
    required_owner_ids: Sequence[str] | None = None,
    receipt: Mapping[str, Any] | None = None,
    registry_path: str | Path | None = None,
    mode: str | None = None,
    strict_execution: bool = False,
    profile: str | None = None,
    require_lineage: bool | None = None,
    layout: str = receipt_lineage.LAYOUT_PRODUCTION,
) -> StageContractResult:
    """Verify registry-owned runtime evidence for a stage.

    The verifier is intentionally data-driven. Registry stage bindings may
    define ``runtime_contract`` with ``required_receipts``,
    ``required_outputs``, ``required_checks``, ``required_upstream_stages``,
    ``lineage.input_artifacts`` and ``forbidden_modes``. A successful live
    receipt must also carry the current registry digest and all required
    owner ids.

    A receipt that carries ``lineage`` is verified against the upstream
    receipts and input artifacts it recorded: any required receipt it never
    bound, any upstream whose ``output_digest`` (or ``content_digest``) moved,
    any upstream whose own lineage no longer verifies, and any input artifact
    whose sha256 moved is a blocker. With ``require_lineage`` (default
    :data:`REQUIRE_LINEAGE_DEFAULT`) a receipt without ``lineage`` is itself a
    blocker.
    """

    workspace_path = Path(workspace)
    registry = load_stage_script_registry(registry_path)
    digest = registry_sha256(registry_path)
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    accepted: list[str] = []

    try:
        resolved_binding = resolve_stage_contract_binding(
            workspace=workspace,
            stage_id=stage_id,
            registry_path=registry_path,
            profile=profile,
        )
    except ValueError as exc:
        blockers.append(
            _blocker(
                "stage_contract_workspace_binding_missing",
                stage_id=stage_id,
                expected="registered workspace stage binding",
                observed=str(exc),
            )
        )
        return _result(
            stage_id=stage_id,
            registry_stage_id=stage_id,
            digest=digest,
            required_script_ids=[],
            accepted=accepted,
            blockers=blockers,
            warnings=warnings,
        )

    contract = resolved_binding.runtime_contract
    if not contract:
        blockers.append(
            _blocker(
                "stage_contract_runtime_contract_missing",
                stage_id=stage_id,
                expected="runtime_contract object",
                observed="<missing>",
            )
        )
        contract = {}

    required_ids = list(required_owner_ids or resolved_binding.required_script_ids)
    registry_stage_id = resolved_binding.registry_stage_id
    artifact_id = _artifact_id_from_receipt(receipt)

    if bool(contract.get("non_production")):
        accepted.append(f"{stage_id}: explicitly marked non-production in registry")
        return _result(
            stage_id=stage_id,
            registry_stage_id=registry_stage_id,
            digest=digest,
            required_script_ids=required_ids,
            accepted=accepted,
            blockers=[],
            warnings=warnings,
        )

    receipt_mode = str(mode or (receipt or {}).get("mode") or "").strip()
    forbidden_modes = {
        str(item).strip()
        for item in contract.get("forbidden_modes", ["mock", "dry-run", "planned"])
        if str(item).strip()
    }
    if receipt_mode in forbidden_modes or receipt_mode.replace("_", "-") in forbidden_modes:
        blockers.append(
            _blocker(
                "stage_contract_non_live_evidence_rejected",
                stage_id=stage_id,
                expected="live receipt evidence",
                observed=receipt_mode,
            )
        )
    elif receipt_mode:
        accepted.append(f"{stage_id}: receipt mode {receipt_mode}")

    if receipt is None:
        warnings.append(
            {
                "code": "stage_contract_receipt_not_supplied",
                "stage_id": stage_id,
                "retryable": True,
            }
        )
    else:
        _check_receipt_registry_fields(
            receipt=receipt,
            stage_id=stage_id,
            digest=digest,
            required_ids=required_ids,
            blockers=blockers,
            accepted=accepted,
            strict_execution=strict_execution,
        )
        must_have_lineage = (
            REQUIRE_LINEAGE_DEFAULT if require_lineage is None else bool(require_lineage)
        )
        if must_have_lineage and not isinstance(receipt.get("lineage"), Mapping):
            blockers.append(
                _blocker(
                    "stage_contract_lineage_missing",
                    stage_id=stage_id,
                    expected="receipt.lineage object binding upstream digests",
                    observed="<missing>",
                )
            )

    verified_lineage_paths: set[str] = set()
    for spec in _as_list(contract.get("required_receipts")):
        _check_required_receipt(
            workspace_path,
            run_id,
            stage_id,
            spec,
            blockers,
            accepted,
            artifact_id=artifact_id,
            receipt=receipt,
            verified_lineage_paths=verified_lineage_paths,
            registry_path=registry_path,
        )

    for spec in _as_list(contract.get("required_outputs")):
        _check_required_output(
            workspace_path,
            run_id,
            stage_id,
            spec,
            blockers,
            accepted,
            artifact_id=artifact_id,
        )

    for spec in _as_list(contract.get("required_checks")):
        _check_required_check(receipt or {}, stage_id, spec, blockers, accepted)

    for validator in _semantic_validator_names(contract.get("semantic_validators")):
        _check_semantic_validator(
            validator=validator,
            workspace=workspace_path,
            run_id=run_id,
            stage_id=stage_id,
            receipt=receipt or {},
            contract=contract,
            blockers=blockers,
            accepted=accepted,
        )

    if receipt is not None and isinstance(receipt.get("lineage"), Mapping):
        _check_lineage_upstream_stages(
            workspace_path,
            run_id,
            stage_id,
            contract,
            receipt["lineage"],
            blockers,
            accepted,
            verified_lineage_paths=verified_lineage_paths,
            layout=layout,
            registry_path=registry_path,
        )
        _check_lineage_input_artifacts(
            workspace_path,
            run_id,
            stage_id,
            contract,
            receipt["lineage"],
            blockers,
            accepted,
            artifact_id=artifact_id,
        )

    return _result(
        stage_id=stage_id,
        registry_stage_id=registry_stage_id,
        digest=digest,
        required_script_ids=required_ids,
        accepted=accepted,
        blockers=blockers,
        warnings=warnings,
    )


def _check_lineage_upstream_stages(
    workspace: Path,
    run_id: str,
    stage_id: str,
    contract: Mapping[str, Any],
    lineage: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
    *,
    verified_lineage_paths: set[str],
    layout: str = receipt_lineage.LAYOUT_PRODUCTION,
    registry_path: str | Path | None = None,
) -> None:
    """Every ``required_upstream_stages`` id scoped to ``layout`` must be bound and still current."""

    upstream = lineage.get("upstream_receipts")
    upstream = upstream if isinstance(upstream, Mapping) else {}
    for upstream_stage_id in receipt_lineage.required_upstream_stage_ids(contract, layout=layout):
        bound = [
            str(rel_path)
            for rel_path, entry in upstream.items()
            if isinstance(entry, Mapping) and entry.get("stage_id") == upstream_stage_id
        ]
        if not bound:
            blockers.append(
                _blocker(
                    "stage_contract_lineage_upstream_unbound",
                    stage_id=stage_id,
                    expected=f"lineage entry for upstream stage {upstream_stage_id}",
                    observed=sorted(str(key) for key in upstream),
                    extra={"upstream_stage_id": upstream_stage_id},
                )
            )
            continue
        for rel_path in bound:
            if rel_path in verified_lineage_paths:
                continue
            verified_lineage_paths.add(rel_path)
            found = receipt_lineage.verify_lineage(
                workspace,
                run_id,
                lineage,
                stage_id=stage_id,
                only_path=rel_path,
                transitive=True,
                registry_path=registry_path,
            )
            if found:
                blockers.extend(found)
            else:
                accepted.append(f"{stage_id}: upstream stage {upstream_stage_id} lineage current ({rel_path})")


def _check_lineage_input_artifacts(
    workspace: Path,
    run_id: str,
    stage_id: str,
    contract: Mapping[str, Any],
    lineage: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
    *,
    artifact_id: str | None,
) -> None:
    """Registry-declared input artifacts must be bound; every bound one must be current."""

    recorded = lineage.get("input_artifacts")
    recorded = recorded if isinstance(recorded, Mapping) else {}
    lineage_spec = contract.get("lineage")
    lineage_spec = lineage_spec if isinstance(lineage_spec, Mapping) else {}
    for spec in _as_list(lineage_spec.get("input_artifacts")):
        name = str(spec.get("name") or Path(str(spec.get("path") or "")).stem)
        if name in recorded:
            continue
        path = _resolve_workspace_template(workspace, run_id, spec.get("path"), artifact_id=artifact_id)
        if bool(spec.get("optional")) and (path is None or not path.exists()):
            continue
        blockers.append(
            _blocker(
                "stage_contract_lineage_upstream_unbound",
                stage_id=stage_id,
                expected=f"lineage.input_artifacts entry {name}",
                observed=sorted(str(key) for key in recorded),
                extra={"input": name},
            )
        )
    found = receipt_lineage.verify_input_artifacts(workspace, lineage, stage_id=stage_id)
    if found:
        blockers.extend(found)
    elif recorded:
        accepted.append(f"{stage_id}: {len(recorded)} input artifact digests current")


def _check_receipt_registry_fields(
    *,
    receipt: Mapping[str, Any],
    stage_id: str,
    digest: str,
    required_ids: Sequence[str],
    blockers: list[dict[str, Any]],
    accepted: list[str],
    strict_execution: bool = False,
) -> None:
    observed_digest = receipt.get("script_registry_sha256")
    if observed_digest != digest:
        blockers.append(
            _blocker(
                "stage_contract_registry_digest_mismatch",
                stage_id=stage_id,
                expected=digest,
                observed=observed_digest if observed_digest is not None else "<missing>",
            )
        )
    else:
        accepted.append(f"{stage_id}: registry digest matched")

    observed_ids = receipt.get("required_script_ids")
    observed_set = {str(item) for item in observed_ids} if isinstance(observed_ids, list) else set()
    missing_ids = sorted(set(required_ids) - observed_set)
    if missing_ids:
        blockers.append(
            _blocker(
                "stage_contract_required_owner_ids_missing",
                stage_id=stage_id,
                expected=sorted(required_ids),
                observed=sorted(observed_set),
                extra={"missing_script_ids": missing_ids},
            )
        )
    else:
        accepted.append(f"{stage_id}: required owner ids recorded")

    if strict_execution:
        _check_owner_execution_evidence(
            receipt=receipt,
            stage_id=stage_id,
            required_ids=required_ids,
            blockers=blockers,
            accepted=accepted,
        )


def _check_owner_execution_evidence(
    *,
    receipt: Mapping[str, Any],
    stage_id: str,
    required_ids: Sequence[str],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    """Require distinct, execution-derived evidence per required owner id.

    A bare echo of ``required_script_ids`` back onto the receipt is NOT
    sufficient in strict mode -- the receipt must carry an independent
    ``owner_execution_evidence`` mapping (owner_id -> evidence token) where
    the token is an executed command string, a produced-receipt/output path,
    or a truthy provider-call marker. A token equal to the bare owner id
    (i.e. a copy of the registry id list) does not count as evidence.
    """

    evidence_map = receipt.get("owner_execution_evidence")
    evidence_map = evidence_map if isinstance(evidence_map, Mapping) else {}
    missing_evidence: list[str] = []
    for owner_id in required_ids:
        token = evidence_map.get(owner_id)
        is_echo = isinstance(token, str) and token.strip() == owner_id
        if not token or is_echo or (isinstance(token, str) and not token.strip()):
            missing_evidence.append(owner_id)
    if missing_evidence:
        blockers.append(
            _blocker(
                "stage_contract_owner_execution_evidence_missing",
                stage_id=stage_id,
                expected="owner_execution_evidence entry per required owner id",
                observed=sorted(missing_evidence),
            )
        )
    else:
        accepted.append(f"{stage_id}: owner execution evidence present")


def _check_required_receipt(
    workspace: Path,
    run_id: str,
    stage_id: str,
    spec: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
    *,
    artifact_id: str | None,
    receipt: Mapping[str, Any] | None = None,
    verified_lineage_paths: set[str] | None = None,
    registry_path: str | Path | None = None,
) -> None:
    if "fresh_after" in spec or "fresh_after_optional" in spec:
        blockers.append(
            _blocker(
                "stage_contract_spec_key_unsupported",
                stage_id=stage_id,
                expected="required_receipts binding by lineage digest (no fresh_after)",
                observed=sorted(key for key in spec if str(key).startswith("fresh_after")),
            )
        )
        return
    lineage_mode = str(spec.get("lineage") or receipt_lineage.LINEAGE_MODE_OUTPUT)
    if lineage_mode not in receipt_lineage.LINEAGE_MODES:
        blockers.append(
            _blocker(
                "stage_contract_spec_key_unsupported",
                stage_id=stage_id,
                expected=sorted(receipt_lineage.LINEAGE_MODES),
                observed={"lineage": lineage_mode},
            )
        )
        return
    path = _resolve_workspace_template(workspace, run_id, spec.get("path"), artifact_id=artifact_id)
    if path is None:
        blockers.append(
            _blocker(
                "stage_contract_required_receipt_path_missing",
                stage_id=stage_id,
                expected="required receipt path",
                observed=spec,
            )
        )
        return
    if not path.is_file():
        if bool(spec.get("optional")):
            accepted.append(f"{stage_id}: optional receipt {_display(path)} absent (accepted)")
            return
        blockers.append(
            _blocker(
                "stage_contract_required_receipt_missing",
                stage_id=stage_id,
                expected=_display(path),
                observed="<missing>",
            )
        )
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        blockers.append(
            _blocker(
                "stage_contract_required_receipt_unreadable",
                stage_id=stage_id,
                expected=_display(path),
                observed=str(exc),
            )
        )
        return
    if not isinstance(payload, Mapping):
        blockers.append(
            _blocker(
                "stage_contract_required_receipt_not_object",
                stage_id=stage_id,
                expected=_display(path),
                observed=type(payload).__name__,
            )
        )
        return
    allowed = {
        str(item)
        for item in spec.get("statuses", ["succeeded"])
        if str(item)
    }
    status = str(payload.get("status") or "")
    if allowed and status not in allowed:
        blockers.append(
            _blocker(
                "stage_contract_required_receipt_status_invalid",
                stage_id=stage_id,
                expected=sorted(allowed),
                observed=status or "<missing>",
                extra={"receipt": _display(path)},
            )
        )
        return
    if receipt is not None and isinstance(receipt.get("lineage"), Mapping):
        lineage = receipt["lineage"]
        rel_path = receipt_lineage.workspace_relpath(workspace, path)
        upstream = lineage.get("upstream_receipts")
        upstream = upstream if isinstance(upstream, Mapping) else {}
        if rel_path not in upstream:
            blockers.append(
                _blocker(
                    "stage_contract_lineage_upstream_unbound",
                    stage_id=stage_id,
                    expected=f"lineage entry for {rel_path}",
                    observed=sorted(str(key) for key in upstream),
                    extra={"receipt": rel_path},
                )
            )
            return
        if verified_lineage_paths is None or rel_path not in verified_lineage_paths:
            if verified_lineage_paths is not None:
                verified_lineage_paths.add(rel_path)
            found = receipt_lineage.verify_lineage(
                workspace,
                run_id,
                lineage,
                stage_id=stage_id,
                only_path=rel_path,
                transitive=lineage_mode != receipt_lineage.LINEAGE_MODE_CONTENT,
                registry_path=registry_path,
            )
            if found:
                blockers.extend(found)
                return
            entry = upstream.get(rel_path)
            digest = str((entry or {}).get("digest") or "")[:12] if isinstance(entry, Mapping) else ""
            accepted.append(f"{stage_id}: lineage bound to {rel_path} ({lineage_mode} {digest})")
    for check in _as_list(spec.get("checks")):
        _check_required_check(payload, stage_id, check, blockers, accepted)
    accepted.append(f"{stage_id}: required receipt {_display(path)}")


def _check_required_output(
    workspace: Path,
    run_id: str,
    stage_id: str,
    spec: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
    *,
    artifact_id: str | None,
) -> None:
    pattern_value = spec.get("glob") or spec.get("path")
    pattern = _template(str(pattern_value or ""), run_id=run_id, artifact_id=artifact_id)
    if not pattern:
        blockers.append(
            _blocker(
                "stage_contract_required_output_path_missing",
                stage_id=stage_id,
                expected="required output path or glob",
                observed=spec,
            )
        )
        return
    min_count = int(spec.get("min_count", 1) or 1)
    if spec.get("glob") or any(char in pattern for char in "*?[]"):
        matches = sorted(workspace.glob(pattern))
        if len(matches) < min_count:
            blockers.append(
                _blocker(
                    "stage_contract_required_output_glob_missing",
                    stage_id=stage_id,
                    expected=f"{pattern} count >= {min_count}",
                    observed=len(matches),
                )
            )
            return
        accepted.append(f"{stage_id}: output glob {pattern} count={len(matches)}")
        return

    path = (workspace / pattern).resolve()
    expected_type = str(spec.get("type") or "file")
    ok = path.is_dir() if expected_type == "directory" else path.is_file()
    if not ok:
        blockers.append(
            _blocker(
                "stage_contract_required_output_missing",
                stage_id=stage_id,
                expected=f"{expected_type} {_display(path)}",
                observed="<missing>",
            )
        )
        return
    accepted.append(f"{stage_id}: required output {_display(path)}")


def _check_required_check(
    receipt: Mapping[str, Any],
    stage_id: str,
    spec: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    path = str(spec.get("path") or "")
    if not path:
        blockers.append(
            _blocker(
                "stage_contract_required_check_path_missing",
                stage_id=stage_id,
                expected="receipt check path",
                observed=spec,
            )
        )
        return
    observed = _get_path(receipt, path)
    if spec.get("truthy"):
        if not bool(observed):
            blockers.append(
                _blocker(
                    "stage_contract_required_check_not_truthy",
                    stage_id=stage_id,
                    expected=f"{path} truthy",
                    observed=observed,
                )
            )
            return
    elif "equals_path" in spec:
        expected = _get_path(receipt, str(spec["equals_path"]))
        if observed != expected:
            blockers.append(
                _blocker(
                    "stage_contract_required_check_mismatch",
                    stage_id=stage_id,
                    expected={str(spec["equals_path"]): expected},
                    observed={path: observed},
                )
            )
            return
    elif "equals" in spec:
        expected = spec["equals"]
        if observed != expected:
            blockers.append(
                _blocker(
                    "stage_contract_required_check_mismatch",
                    stage_id=stage_id,
                    expected={path: expected},
                    observed={path: observed},
                )
            )
            return
    elif "not_equals" in spec:
        forbidden = spec["not_equals"]
        if observed == forbidden:
            blockers.append(
                _blocker(
                    "stage_contract_required_check_forbidden_value",
                    stage_id=stage_id,
                    expected=f"{path} != {forbidden!r}",
                    observed={path: observed},
                )
            )
            return
    elif "one_of" in spec:
        allowed_values = spec["one_of"]
        if observed not in allowed_values:
            blockers.append(
                _blocker(
                    "stage_contract_required_check_not_in_allowed_set",
                    stage_id=stage_id,
                    expected={path: list(allowed_values)},
                    observed={path: observed},
                )
            )
            return
    else:
        blockers.append(
            _blocker(
                "stage_contract_required_check_no_predicate",
                stage_id=stage_id,
                expected="truthy, equals, or equals_path",
                observed=spec,
            )
        )
        return
    accepted.append(f"{stage_id}: receipt check {path}")


def _semantic_validator_names(value: Any) -> list[str]:
    """Return explicit registry-owned semantic validators, rejecting junk names.

    Most stage contracts are expressible as independent receipt predicates.
    Some safety properties necessarily relate multiple files or receipt fields
    (for example, a ledger digest must bind the actual upstream ledger, not a
    human review report).  Those contracts opt into a named validator here;
    this keeps the registry declarative while keeping cross-field checks in
    one reusable receipt verifier rather than duplicating receipt machinery in
    each stage owner.
    """

    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def _check_semantic_validator(
    *,
    validator: str,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    if validator == "cvpr_transcription_ocr":
        _check_cvpr_transcription_ocr_semantics(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
            receipt=receipt,
            contract=contract,
            blockers=blockers,
            accepted=accepted,
        )
        return
    if validator == "cvpr_transcription_ocr_r6":
        _check_cvpr_transcription_ocr_r6_semantics(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
            receipt=receipt,
            contract=contract,
            blockers=blockers,
            accepted=accepted,
        )
        return
    if validator == "cvpr_registration_media_stage":
        _check_cvpr_registration_media_stage_semantics(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
            receipt=receipt,
            contract=contract,
            blockers=blockers,
            accepted=accepted,
        )
        return
    if validator == "cvpr_event_media_stage":
        _check_cvpr_event_media_stage_semantics(
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
            receipt=receipt,
            contract=contract,
            blockers=blockers,
            accepted=accepted,
        )
        return
    blockers.append(
        _blocker(
            "stage_contract_semantic_validator_unknown",
            stage_id=stage_id,
            expected="registered semantic validator",
            observed=validator,
        )
    )


def _registration_media_secret_surface_present(value: Any) -> bool:
    """Reject credential-bearing receipt fields without inspecting any secret."""

    forbidden_keys = {
        "cookie",
        "cookies",
        "cookie_path",
        "session_path",
        "authorization",
        "headers",
        "header",
        "token",
        "password",
        "browser_profile",
        "raw_html",
        "html",
        "body",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in forbidden_keys or "cookie" in key_text or "browser_profile" in key_text:
                return True
            if _registration_media_secret_surface_present(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_registration_media_secret_surface_present(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return (
            "cvpr2026_virtual_cookies_file" in lowered
            or "authorization:" in lowered
            or "bearer " in lowered
            or "api_key=" in lowered
        )
    return False


def _registration_media_requests_complete_with_authenticated_access(
    request_rows: Sequence[Any],
) -> bool:
    """Return whether every r4 request has a final, truthful access result.

    A registration wall or a post-sentinel account denial is an attempt
    outcome, never evidence that an authenticated recovery completed.  Keep
    this small predicate separate from the filesystem-heavy semantic checker
    so its fail-closed status rule stays directly testable.
    """

    completed_statuses = {
        "direct_youtube_recovered",
        "official_video_not_published",
        "unreachable_authenticated_media",
        "no_admissible_official_youtube_link",
        "unsupported_official_embed_host",
        "outside_authenticated_official_event_boundary",
    }
    return all(
        isinstance(row, Mapping)
        and row.get("result_status") in completed_statuses
        and (
            row.get("transport_eligible") is not True
            or row.get("auth_status") == "authenticated"
        )
        for row in request_rows
    )


def _registration_media_execution_evidence_matches(
    *, receipt: Mapping[str, Any], checks: Mapping[str, Any]
) -> bool:
    """Check the sanitized transport count is consistent across a receipt."""

    owner_evidence = receipt.get("owner_execution_evidence")
    owner_evidence = owner_evidence if isinstance(owner_evidence, Mapping) else {}
    download_owner = owner_evidence.get("acquisition.download_corpus.cvpr2026")
    download_owner = download_owner if isinstance(download_owner, Mapping) else {}
    authenticated_request_count = checks.get("registration_media_authenticated_request_count")
    transport_created = checks.get("registration_media_transport_created")
    return (
        isinstance(authenticated_request_count, int)
        and not isinstance(authenticated_request_count, bool)
        and authenticated_request_count >= 1
        and transport_created is True
        and download_owner.get("authenticated_request_count") == authenticated_request_count
        and download_owner.get("transport_created") is True
    )


def _check_cvpr_registration_media_stage_semantics(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    """Enforce r4's pinned-predecessor and complete-overlay truthfulness."""

    checks = receipt.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    expected_run_id = str(contract.get("required_run_id") or "")
    if (
        expected_run_id != R4_RUN_ID
        or run_id != R4_RUN_ID
        or receipt.get("run_id") != R4_RUN_ID
        or receipt.get("stage_id") != "download-corpus"
    ):
        blockers.append(
            _blocker(
                "stage_contract_registration_media_lineage_mismatch",
                stage_id=stage_id,
                expected=R4_RUN_ID,
                observed={
                    "contract_run_id": expected_run_id,
                    "verify_run_id": run_id,
                    "receipt_run_id": receipt.get("run_id"),
                    "receipt_stage_id": receipt.get("stage_id"),
                },
            )
        )
        return
    overlay_path = workspace / OVERLAY_LEDGER_RELATIVE_PATH
    overlay = _read_json_object(overlay_path)
    if overlay is None:
        blockers.append(
            _blocker(
                "stage_contract_overlay_ledger_missing",
                stage_id=stage_id,
                expected=str(OVERLAY_LEDGER_RELATIVE_PATH),
                observed="<missing-or-unreadable>",
            )
        )
        return
    try:
        boundary = parse_virtual_site_boundary(load_conference_identity(workspace).virtual_url)
        validate_registration_media_overlay(overlay, boundary=boundary)
        actual_predecessors = load_pinned_r3_predecessors(workspace)
    except (ConferenceIdentityError, OSError, ValueError):
        blockers.append(
            _blocker(
                "stage_contract_registration_media_predecessor_invalid",
                stage_id=stage_id,
                expected="exact pinned r3 predecessor files and publisher digests",
                observed="invalid",
            )
        )
        return
    lineage = overlay.get("predecessor_lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    if lineage.get("predecessors") != PINNED_R3_PREDECESSORS or actual_predecessors != PINNED_R3_PREDECESSORS:
        blockers.append(
            _blocker(
                "stage_contract_registration_media_predecessor_invalid",
                stage_id=stage_id,
                expected="exact pinned r3 predecessor records",
                observed="overlay or workspace predecessor drift",
            )
        )
        return
    actual_file_digest = _sha256_file(overlay_path)
    canonical_digest = canonical_registration_media_digest(overlay)
    recovery_counts = overlay.get("recovery_counts")
    recovery_counts = recovery_counts if isinstance(recovery_counts, Mapping) else {}
    request_rows = overlay.get("official_event_requests")
    request_rows = request_rows if isinstance(request_rows, list) else []
    media_rows = overlay.get("official_event_media_records")
    media_rows = media_rows if isinstance(media_rows, list) else []
    accounting_rows = overlay.get("program_item_accounting")
    accounting_rows = accounting_rows if isinstance(accounting_rows, list) else []
    complete = (
        overlay.get("status") == "registration_media_complete"
        and len(request_rows) == FULL_R4_SCOPE.policy_request_count
        and _registration_media_requests_complete_with_authenticated_access(request_rows)
        and recovery_counts.get("completed_request_count") == FULL_R4_SCOPE.policy_request_count
        and recovery_counts.get("transport_eligible_request_count") == FULL_R4_SCOPE.transport_eligible_request_count
        and recovery_counts.get("boundary_no_network_request_count") == FULL_R4_SCOPE.boundary_request_count
        and recovery_counts.get("direct_youtube_media_record_count") == len(media_rows)
        and checks.get("direct_youtube_media_record_count") == len(media_rows)
        and all(
            isinstance(row, Mapping)
            and row.get("audio_downloaded") is False
            and row.get("video_downloaded") is False
            for row in media_rows
        )
        and len(accounting_rows) == FULL_R4_SCOPE.total_program_item_count
        and all(
            not isinstance(row, Mapping)
            or row.get("result_origin") != "r4_registration_media"
            or row.get("result_status")
            in {
                "direct_youtube_recovered",
                "official_video_not_published",
                "unreachable_authenticated_media",
                "no_admissible_official_youtube_link",
                "unsupported_official_embed_host",
                "outside_authenticated_official_event_boundary",
            }
            for row in accounting_rows
        )
    )
    capacity = checks.get("disk_capacity")
    capacity = capacity if isinstance(capacity, Mapping) else {}
    preflight = checks.get("registration_media_session_preflight")
    preflight = preflight if isinstance(preflight, Mapping) else {}
    execution_evidence_matches = _registration_media_execution_evidence_matches(
        receipt=receipt,
        checks=checks,
    )
    lineage_digest = lineage.get("candidate_plan_sha256")
    if (
        checks.get("registration_media_overlay_path") != str(OVERLAY_LEDGER_RELATIVE_PATH)
        or checks.get("registration_media_overlay_ledger_sha256") != actual_file_digest
        or checks.get("registration_media_overlay_canonical_sha256") != canonical_digest
        or overlay.get("overlay_sha256") != canonical_digest
        or capacity.get("positive_available_bytes") is not True
        or capacity.get("status") not in {"pass", "pass_pre_request"}
        or capacity.get("candidate_plan_sha256") != lineage_digest
        or not isinstance(capacity.get("manifest_sha256"), str)
        or len(str(capacity.get("manifest_sha256"))) != 64
        or checks.get("no_cap_hits") is not True
        or checks.get("secrets_copied") is not False
        or preflight.get("preflight_status") != "pass"
        or preflight.get("secrets_copied") is not False
        or not execution_evidence_matches
        or _registration_media_secret_surface_present(receipt)
        or not complete
    ):
        blockers.append(
            _blocker(
                "stage_contract_registration_media_overlay_invalid",
                stage_id=stage_id,
                expected="complete digest-bound r4 overlay, capacity, sanitized preflight, and no secret surface",
                observed={
                    "overlay_complete": complete,
                    "overlay_file_digest_matches": checks.get("registration_media_overlay_ledger_sha256") == actual_file_digest,
                    "overlay_canonical_digest_matches": checks.get("registration_media_overlay_canonical_sha256") == canonical_digest,
                    "capacity_status": capacity.get("status"),
                    "preflight_status": preflight.get("preflight_status"),
                    "execution_evidence_matches": execution_evidence_matches,
                },
            )
        )
        return
    accepted.append(f"{stage_id}: exact r4 predecessor, overlay, capacity, and preflight evidence")


def _check_cvpr_event_media_stage_semantics(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    """Verify the full r5 virtual-program census and caption-only boundary."""

    checks = receipt.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    expected_run_id = str(contract.get("required_run_id") or "")
    if (
        expected_run_id != R5_EVENT_MEDIA_RUN_ID
        or run_id != R5_EVENT_MEDIA_RUN_ID
        or receipt.get("run_id") != R5_EVENT_MEDIA_RUN_ID
        or receipt.get("stage_id") != "download-corpus"
    ):
        blockers.append(
            _blocker(
                "stage_contract_event_media_lineage_mismatch",
                stage_id=stage_id,
                expected=R5_EVENT_MEDIA_RUN_ID,
                observed={
                    "contract_run_id": expected_run_id,
                    "verify_run_id": run_id,
                    "receipt_run_id": receipt.get("run_id"),
                    "receipt_stage_id": receipt.get("stage_id"),
                },
            )
        )
        return

    inventory_path = workspace / EVENT_MEDIA_INVENTORY_RELATIVE_PATH
    inventory = _read_json_object(inventory_path)
    if inventory is None:
        blockers.append(
            _blocker(
                "stage_contract_event_media_inventory_missing",
                stage_id=stage_id,
                expected=str(EVENT_MEDIA_INVENTORY_RELATIVE_PATH),
                observed="<missing-or-unreadable>",
            )
        )
        return
    try:
        boundary = parse_virtual_site_boundary(load_conference_identity(workspace).virtual_url)
        validate_event_media_inventory(
            inventory,
            boundary=boundary,
            require_terminal=True,
            require_caption_terminal=True,
        )
    except (ConferenceIdentityError, ValueError):
        blockers.append(
            _blocker(
                "stage_contract_event_media_inventory_invalid",
                stage_id=stage_id,
                expected="terminal full-program inventory with terminal caption outcomes",
                observed="validation failed",
            )
        )
        return

    try:
        r4_path = workspace / OVERLAY_LEDGER_RELATIVE_PATH
        r4_overlay = _read_json_object(r4_path)
        if r4_overlay is None:
            raise ValueError("missing r4 overlay")
        validate_registration_media_overlay(r4_overlay, boundary=boundary)
        r4_requests = r4_overlay.get("official_event_requests")
        r4_media_records = r4_overlay.get("official_event_media_records")
        if not isinstance(r4_requests, list) or not isinstance(r4_media_records, list):
            raise ValueError("r4 overlay lacks reusable poster evidence")
        validate_r4_poster_reuse(
            inventory,
            boundary=boundary,
            r4_requests=r4_requests,
            r4_media_records=r4_media_records,
        )
    except ValueError:
        blockers.append(
            _blocker(
                "stage_contract_event_media_r4_lineage_invalid",
                stage_id=stage_id,
                expected="valid immutable r4 registration-media overlay",
                observed="missing or invalid",
            )
        )
        return

    lineage = inventory.get("source_lineage")
    lineage = lineage if isinstance(lineage, Mapping) else {}
    counts = inventory.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    candidates = inventory.get("caption_candidates")
    candidates = candidates if isinstance(candidates, list) else []
    actual_file_digest = _sha256_file(inventory_path)
    actual_canonical_digest = canonical_event_media_inventory_digest(inventory)
    r4_canonical_digest = canonical_registration_media_digest(r4_overlay)
    captions_root = (workspace / EVENT_MEDIA_CAPTIONS_RELATIVE_PATH).resolve()
    invalid_caption_sidecars: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or candidate.get("caption_status") != "youtube_caption_available":
            continue
        caption = candidate.get("caption")
        if not isinstance(caption, Mapping) or not isinstance(caption.get("local_path"), str):
            invalid_caption_sidecars.append(str(candidate.get("video_id")))
            continue
        try:
            path = (workspace / str(caption["local_path"])).resolve()
            path.relative_to(captions_root)
            sidecar_valid = (
                path.is_file()
                and path.stat().st_size == caption.get("byte_count")
                and _sha256_file(path) == caption.get("sha256")
                and caption_sidecar_matches_video_id(path, str(candidate.get("video_id") or ""))
            )
        except (OSError, ValueError):
            sidecar_valid = False
        if not sidecar_valid:
            invalid_caption_sidecars.append(str(candidate.get("video_id")))

    owner_evidence = receipt.get("owner_execution_evidence")
    owner_evidence = owner_evidence if isinstance(owner_evidence, Mapping) else {}
    owner = owner_evidence.get("acquisition.download_corpus.cvpr2026")
    owner = owner if isinstance(owner, Mapping) else {}
    request_count = checks.get("event_media_authenticated_request_count")
    resumed_request_count = checks.get("event_media_resumed_authenticated_request_count")
    expected_counts = {
        "event_route_count": sum(EVENT_CATEGORY_COUNTS.values()),
        "event_categories": EVENT_CATEGORY_COUNTS,
        "r4_reused_poster_route_count": R4_REUSED_POSTER_ROUTE_COUNT,
    }
    execution_evidence_matches = (
        isinstance(request_count, int)
        and not isinstance(request_count, bool)
        and request_count > 0
        and isinstance(resumed_request_count, int)
        and not isinstance(resumed_request_count, bool)
        and 0 <= resumed_request_count <= request_count
        and owner.get("owner_function") == "run_cvpr_event_media_stage"
        and owner.get("transport_created") is True
        and owner.get("authenticated_request_count") == request_count
        and owner.get("resumed_authenticated_request_count") == resumed_request_count
        and owner.get("caption_only") is True
    )
    preflight = checks.get("event_media_session_preflight")
    preflight = preflight if isinstance(preflight, Mapping) else {}
    resume_mode = checks.get("event_media_resume_mode")
    resume_sentinel = preflight.get("resume_sentinel")
    resume_sentinel = resume_sentinel if isinstance(resume_sentinel, Mapping) else {}
    resume_sentinel_valid = (
        isinstance(resume_mode, bool)
        and (
            not resume_mode
            or (
                resume_sentinel.get("authenticated_request_count") == 1
                and resume_sentinel.get("status") == "authenticated"
                and resume_sentinel.get("auth_status") == "authenticated"
                and isinstance(resume_sentinel.get("http_status"), int)
            )
        )
    )
    if (
        checks.get("event_media_inventory_path") != str(EVENT_MEDIA_INVENTORY_RELATIVE_PATH)
        or checks.get("event_media_inventory_sha256") != actual_file_digest
        or checks.get("event_media_inventory_canonical_sha256") != actual_canonical_digest
        or lineage.get("r4_registration_media_overlay_sha256") != r4_canonical_digest
        or any(counts.get(key) != value for key, value in expected_counts.items())
        or checks.get("event_route_count") != expected_counts["event_route_count"]
        or checks.get("event_category_counts") != EVENT_CATEGORY_COUNTS
        or checks.get("event_route_accounting_complete") is not True
        or checks.get("caption_candidate_accounting_complete") is not True
        or checks.get("event_media_coverage_status") != "pass"
        or preflight.get("preflight_status") != "pass"
        or preflight.get("secrets_copied") is not False
        or not resume_sentinel_valid
        or checks.get("audio_downloaded") is not False
        or checks.get("video_downloaded") is not False
        or checks.get("asr_dispatched") is not False
        or checks.get("ocr_dispatched") is not False
        or checks.get("provider_calls_made") != 0
        or checks.get("secrets_copied") is not False
        or receipt.get("downloads_made") != 0
        or receipt.get("provider_calls_made") != 0
        or receipt.get("secrets_copied") is not False
        or not execution_evidence_matches
        or invalid_caption_sidecars
        or _registration_media_secret_surface_present(receipt)
    ):
        blockers.append(
            _blocker(
                "stage_contract_event_media_inventory_invalid",
                stage_id=stage_id,
                expected="full r5 virtual-program accounting, r4 lineage, verified sidecars, and caption-only execution",
                observed={
                    "file_digest_matches": checks.get("event_media_inventory_sha256") == actual_file_digest,
                    "canonical_digest_matches": checks.get("event_media_inventory_canonical_sha256") == actual_canonical_digest,
                    "r4_lineage_matches": lineage.get("r4_registration_media_overlay_sha256") == r4_canonical_digest,
                    "counts": counts,
                    "execution_evidence_matches": execution_evidence_matches,
                    "resume_sentinel_valid": resume_sentinel_valid,
                    "invalid_caption_sidecars": invalid_caption_sidecars,
                },
            )
        )
        return
    accepted.append(
        f"{stage_id}: complete 5,258-route virtual-program census with r4 reuse and verified caption-only evidence"
    )


def _check_cvpr_transcription_ocr_semantics(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    """Validate the exact r5 caption handoff before Stage 5 can succeed.

    r4 is accepted only as the immutable digest embedded by the r5 publisher;
    it never supplies a queue, an inventory path, or an alternate run id.
    Every check is local and receipt-bound, so this verifier cannot dispatch a
    provider or discover extra files while evaluating a claimed success.
    """

    checks = receipt.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    expected_run_id = R5_EVENT_MEDIA_RUN_ID
    stage4_path = workspace / "storage" / "workflow-runs" / expected_run_id / "stage-04-download-corpus.json"
    output_path = workspace / STAGE5_INVENTORY

    if (
        contract.get("required_run_id") != expected_run_id
        or run_id != expected_run_id
        or receipt.get("run_id") != expected_run_id
        or checks.get("stage_04_run_id") != expected_run_id
        or checks.get("stage_04_receipt_path") != stage4_path.relative_to(workspace).as_posix()
        or checks.get("stage_05_receipt_path")
        != f"storage/workflow-runs/{expected_run_id}/stage-05-transcription-ocr.json"
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_lineage_mismatch",
                stage_id=stage_id,
                expected="the single cvpr2026-live-production-r5-event-media Stage 5 identity",
                observed={
                    "contract_run_id": contract.get("required_run_id"),
                    "verify_run_id": run_id,
                    "receipt_run_id": receipt.get("run_id"),
                    "stage_04_run_id": checks.get("stage_04_run_id"),
                    "stage_04_receipt_path": checks.get("stage_04_receipt_path"),
                    "stage_05_receipt_path": checks.get("stage_05_receipt_path"),
                },
            )
        )
        return

    stage4_receipt = _read_json_object(stage4_path)
    if stage4_receipt is None:
        # Ordinary receipt/output checks already record the missing path.  Do
        # not misreport missing controls as a lineage or count mismatch.
        return

    actual_stage4_receipt_digest = _sha256_file(stage4_path)
    event_media_ref = stage4_receipt.get("event_media_inventory")
    event_media_ref = event_media_ref if isinstance(event_media_ref, Mapping) else {}
    stage4_contract = stage4_receipt.get("stage_contract")
    stage4_contract = stage4_contract if isinstance(stage4_contract, Mapping) else {}
    stage4_checks = stage4_receipt.get("checks")
    stage4_checks = stage4_checks if isinstance(stage4_checks, Mapping) else {}
    expected_event_media_path = EVENT_MEDIA_INVENTORY_RELATIVE_PATH.as_posix()
    event_media_path = workspace / expected_event_media_path
    if (
        stage4_receipt.get("stage_id") != "download-corpus"
        or stage4_receipt.get("stage_order") != 4
        or stage4_receipt.get("status") != "succeeded"
        or stage4_receipt.get("mode") != "live"
        or stage4_receipt.get("run_id") != expected_run_id
        or stage4_contract.get("status") != "pass"
        or stage4_contract.get("stage_id") != "download-corpus"
        or actual_stage4_receipt_digest != R5_EVENT_MEDIA_RECEIPT_SHA256
        or event_media_ref.get("path") != expected_event_media_path
        or event_media_ref.get("inventory_sha256") != R5_EVENT_MEDIA_CANONICAL_DIGEST
        or stage4_checks.get("event_media_inventory_path") != expected_event_media_path
        or stage4_checks.get("event_media_inventory_sha256") != R5_EVENT_MEDIA_FILE_SHA256
        or stage4_checks.get("event_media_inventory_canonical_sha256")
        != R5_EVENT_MEDIA_CANONICAL_DIGEST
        or stage4_checks.get("r4_registration_media_overlay_sha256") != R4_OVERLAY_DIGEST
        or checks.get("stage_04_receipt_sha256") != actual_stage4_receipt_digest
        or checks.get("event_media_inventory_path") != expected_event_media_path
        or checks.get("event_media_inventory_sha256") != R5_EVENT_MEDIA_FILE_SHA256
        or checks.get("event_media_inventory_canonical_sha256") != R5_EVENT_MEDIA_CANONICAL_DIGEST
        or checks.get("r4_registration_media_overlay_sha256") != R4_OVERLAY_DIGEST
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_lineage_mismatch",
                stage_id=stage_id,
                expected="published r5 receipt fields and independently recomputed predecessor digest",
                observed={
                    "stage4_receipt_sha256": actual_stage4_receipt_digest,
                    "stage4_contract": dict(stage4_contract),
                    "stage4_inventory": dict(event_media_ref),
                    "stage4_checks": {
                        key: stage4_checks.get(key)
                        for key in (
                            "event_media_inventory_path",
                            "event_media_inventory_sha256",
                            "event_media_inventory_canonical_sha256",
                            "r4_registration_media_overlay_sha256",
                        )
                    },
                    "receipt_checks": {
                        key: checks.get(key)
                        for key in (
                            "stage_04_receipt_sha256",
                            "event_media_inventory_path",
                            "event_media_inventory_sha256",
                            "event_media_inventory_canonical_sha256",
                            "r4_registration_media_overlay_sha256",
                        )
                    },
                },
            )
        )
        return

    event_media_inventory = _read_json_object(event_media_path)
    if event_media_inventory is None:
        return
    actual_event_media_file_digest = _sha256_file(event_media_path)
    actual_event_media_canonical_digest = canonical_event_media_inventory_digest(event_media_inventory)
    try:
        boundary = parse_virtual_site_boundary(load_conference_identity(workspace).virtual_url)
        validate_event_media_inventory(
            event_media_inventory,
            boundary=boundary,
            require_terminal=True,
            require_caption_terminal=True,
        )
    except (ConferenceIdentityError, ValueError):
        blockers.append(
            _blocker(
                "stage_contract_r5_event_media_inventory_invalid",
                stage_id=stage_id,
                expected="publisher-valid terminal r5 event-media inventory",
                observed="inventory schema or boundary validation failed",
            )
        )
        return

    retry_control_paths = {
        "archive": f"storage/workflow-runs/{expected_run_id}/stage-05-transcription-ocr.pre-repair-blocked.json",
        "consumed_authorization": f"storage/workflow-runs/{expected_run_id}/stage-05-transcription-ocr.retry-authorization.json",
        "attempt": f"storage/workflow-runs/{expected_run_id}/stage-05-transcription-ocr.retry-attempt.json",
    }
    retry_control_digests = checks.get("retry_control_digests")
    retry_controls_valid = isinstance(retry_control_digests, Mapping) and set(retry_control_digests) == set(
        retry_control_paths
    )
    if retry_controls_valid:
        for control_name, control_path_value in retry_control_paths.items():
            control = retry_control_digests.get(control_name)
            if not isinstance(control, Mapping) or set(control) != {"path", "sha256"}:
                retry_controls_valid = False
                break
            control_digest = control.get("sha256")
            if (
                control.get("path") != control_path_value
                or not _is_sha256(control_digest)
                or not (workspace / control_path_value).is_file()
                or _sha256_file(workspace / control_path_value) != control_digest
            ):
                retry_controls_valid = False
                break
    if not retry_controls_valid:
        blockers.append(
            _blocker(
                "stage_contract_r5_retry_control_invalid",
                stage_id=stage_id,
                expected="exact archive, consumed_authorization, and attempt control digests at fixed paths",
                observed="retry control map is missing, malformed, substituted, unreadable, or does not match local bytes",
            )
        )
        return
    accepted.append(f"{stage_id}: exact non-circular retry control digests")

    output_inventory = _read_json_object(output_path)
    if output_inventory is None:
        return

    if (
        actual_event_media_file_digest != R5_EVENT_MEDIA_FILE_SHA256
        or actual_event_media_canonical_digest != R5_EVENT_MEDIA_CANONICAL_DIGEST
        or event_media_inventory.get("source_lineage", {}).get("r4_registration_media_overlay_sha256")
        != R4_OVERLAY_DIGEST
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_event_media_digest_mismatch",
                stage_id=stage_id,
                expected={
                    "file_sha256": R5_EVENT_MEDIA_FILE_SHA256,
                    "canonical_sha256": R5_EVENT_MEDIA_CANONICAL_DIGEST,
                    "r4_overlay_sha256": R4_OVERLAY_DIGEST,
                },
                observed={
                    "file_sha256": actual_event_media_file_digest,
                    "canonical_sha256": actual_event_media_canonical_digest,
                    "r4_overlay_sha256": event_media_inventory.get("source_lineage", {}).get(
                        "r4_registration_media_overlay_sha256"
                    ),
                },
            )
        )
        return

    # The expected accounting is DERIVED from the digest-verified r5
    # event-media inventory bound above, never pinned as literals. A contract
    # that hardcodes one run's counts turns a fact about that run into a rule
    # the codebase enforces forever, which is exactly what pushed later
    # transcript recoveries (r6 ASR, r7 caption recheck, the three keynotes)
    # into side directories that nothing consumed.
    expected_counts = _r5_expected_counts_from_event_media(event_media_inventory)
    output_digest = _sha256_file(output_path)
    actual_output_canonical_digest = _canonical_json_digest(output_inventory, excluded_keys={"inventory_sha256"})
    compatibility = output_inventory.get("compatibility")
    input_digests = output_inventory.get("input_digests")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    input_digests = input_digests if isinstance(input_digests, Mapping) else {}
    expected_inputs = {
        "stage4_receipt_sha256": actual_stage4_receipt_digest,
        "event_media_file_sha256": R5_EVENT_MEDIA_FILE_SHA256,
        "event_media_canonical_sha256": R5_EVENT_MEDIA_CANONICAL_DIGEST,
        "r4_overlay_sha256": R4_OVERLAY_DIGEST,
    }
    owner_evidence = receipt.get("owner_execution_evidence")
    owner_evidence = owner_evidence if isinstance(owner_evidence, Mapping) else {}
    owner = owner_evidence.get("transcription_ocr.cvpr2026")
    owner = owner if isinstance(owner, Mapping) else {}
    if (
        checks.get("transcription_ocr_inventory_path") != STAGE5_INVENTORY.as_posix()
        or checks.get("transcription_ocr_inventory_sha256") != output_digest
        or checks.get("transcription_ocr_inventory_canonical_sha256") != output_inventory.get("inventory_sha256")
        or output_inventory.get("inventory_sha256") != actual_output_canonical_digest
        or output_inventory.get("schema_version") != 1
        or output_inventory.get("receipt_type") != "conference_pipeline.cvpr2026_transcription_ocr_inventory"
        or output_inventory.get("run_id") != expected_run_id
        or output_inventory.get("processor_identity") != PROCESSOR_IDENTITY
        or output_inventory.get("processor_version") != PROCESSOR_VERSION
        or output_inventory.get("semantic_class") != "caption_first_no_asr"
        or any(compatibility.get(key) != value for key, value in expected_inputs.items())
        or any(input_digests.get(key) != value for key, value in expected_inputs.items())
        or compatibility.get("processor_identity") != PROCESSOR_IDENTITY
        or compatibility.get("processor_version") != PROCESSOR_VERSION
        or owner.get("owner_function") != "run_cvpr_transcription_ocr_stage"
        or owner.get("inventory_path") != STAGE5_INVENTORY.as_posix()
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_processor_mismatch",
                stage_id=stage_id,
                expected="r5-bound caption materializer output and CVPR owner evidence",
                observed={
                    "output_digest": output_digest,
                    "receipt_output_digest": checks.get("transcription_ocr_inventory_sha256"),
                    "processor_identity": output_inventory.get("processor_identity"),
                    "processor_version": output_inventory.get("processor_version"),
                    "owner": dict(owner),
                },
            )
        )
        return

    _check_cvpr_r5_class_accounting(
        stage_id=stage_id,
        checks=checks,
        inventory=output_inventory,
        expected_counts=expected_counts,
        blockers=blockers,
        accepted=accepted,
    )
    _check_cvpr_r5_capacity_binding(
        stage_id=stage_id,
        checks=checks,
        inventory=output_inventory,
        blockers=blockers,
        accepted=accepted,
    )
    _check_cvpr_r5_workshop_table(
        stage_id=stage_id,
        workspace=workspace,
        checks=checks,
        inventory=output_inventory,
        blockers=blockers,
        accepted=accepted,
    )
    _check_cvpr_r5_empty_lanes(
        stage_id=stage_id,
        receipt=receipt,
        checks=checks,
        inventory=output_inventory,
        blockers=blockers,
        accepted=accepted,
    )
    _check_cvpr_r5_workshop_boundary(
        stage_id=stage_id,
        inventory=output_inventory,
        blockers=blockers,
        accepted=accepted,
    )


def _check_cvpr_transcription_ocr_r6_semantics(
    *,
    workspace: Path,
    run_id: str,
    stage_id: str,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    """Validate the separate, declared-input r6 overlay without touching r5.

    This validator never dispatches a provider or reads arbitrary workspace
    files.  It only reads the fixed receipt paths declared in the r6 owner
    contract and refuses a claimed success when those receipts disagree.
    """

    checks = receipt.get("checks")
    checks = checks if isinstance(checks, Mapping) else {}
    receipt_paths = {
        "recovery_inventory": R6_ARTIFACT_PATHS["recovery_inventory"],
        "audio_recovery_receipt": R6_ARTIFACT_PATHS["audio_recovery_receipt"],
        "material_discovery_receipt": R6_ARTIFACT_PATHS["material_discovery_receipt"],
        "material_acquisition_receipt": R6_ARTIFACT_PATHS["material_acquisition_receipt"],
        "estimate_receipt": R6_ARTIFACT_PATHS["cost_estimate_receipt"],
        "benchmark_receipt": R6_ARTIFACT_PATHS["benchmark_receipt"],
        "provider_receipts": R6_ARTIFACT_PATHS["provider_receipts"],
        "budget_ledger": R6_ARTIFACT_PATHS["budget_ledger"],
    }
    if (
        contract.get("required_run_id") != R6_EVENT_MEDIA_RUN_ID
        or run_id != R6_EVENT_MEDIA_RUN_ID
        or receipt.get("run_id") != R6_EVENT_MEDIA_RUN_ID
        or receipt.get("stage_id") != R6_STAGE_ID
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_lineage_mismatch",
                stage_id=stage_id,
                expected="the fixed cvpr2026 r6 overlay identity",
                observed={
                    "contract_run_id": contract.get("required_run_id"),
                    "verify_run_id": run_id,
                    "receipt_run_id": receipt.get("run_id"),
                    "receipt_stage_id": receipt.get("stage_id"),
                },
            )
        )
        return

    expected_owner_id = "transcription_ocr.cvpr2026.r6"
    if receipt.get("required_script_ids") != [expected_owner_id]:
        blockers.append(
            _blocker(
                "stage_contract_r6_required_owner_identity_invalid",
                stage_id=stage_id,
                expected=[expected_owner_id],
                observed=receipt.get("required_script_ids"),
            )
        )
        return

    r5_lineage = receipt.get("r5_lineage")
    r5_lineage = r5_lineage if isinstance(r5_lineage, Mapping) else {}
    r5_keys = ("receipt_sha256", "inventory_sha256", "inventory_canonical_sha256")
    if (
        r5_lineage.get("run_id") != R5_EVENT_MEDIA_RUN_ID
        or any(not _is_sha256(r5_lineage.get(key)) for key in r5_keys)
        or any(checks.get(f"r5_{key}") != r5_lineage.get(key) for key in r5_keys)
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_r5_predecessor_invalid",
                stage_id=stage_id,
                expected="immutable r5 receipt, file, and canonical predecessor digests",
                observed={key: r5_lineage.get(key) for key in r5_keys},
            )
        )
        return

    payloads: dict[str, Mapping[str, Any]] = {}
    for name, relative_path in receipt_paths.items():
        path = workspace / relative_path
        payload = _read_json_object(path)
        actual_digest = _sha256_file(path) if path.is_file() else None
        if (
            checks.get(f"{name}_path") != relative_path.as_posix()
            or checks.get(f"{name}_sha256") != actual_digest
        ):
            blockers.append(
                _blocker(
                    "stage_contract_r6_declared_receipt_invalid",
                    stage_id=stage_id,
                    expected=f"current digest-bound {relative_path.as_posix()}",
                    observed={"path": checks.get(f"{name}_path"), "exists": path.is_file()},
                )
            )
            return
        if payload is None or payload.get("run_id") != R6_EVENT_MEDIA_RUN_ID:
            blockers.append(
                _blocker(
                    "stage_contract_r6_declared_receipt_invalid",
                    stage_id=stage_id,
                    expected=f"readable r6 receipt at {relative_path.as_posix()}",
                    observed="missing, unreadable, or wrong run id",
                )
            )
            return
        payloads[name] = payload

    recovery = payloads["recovery_inventory"]
    recovery_counts = recovery.get("counts")
    recovery_lineage = recovery.get("r5_lineage")
    recovery_lineage = recovery_lineage if isinstance(recovery_lineage, Mapping) else {}
    if (
        recovery.get("receipt_type") != "conference_pipeline.cvpr2026_r6_media_recovery_inventory"
        or not isinstance(recovery_counts, Mapping)
        or recovery_counts.get("asr_fallback_required") != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
        or any(recovery_lineage.get(key) != r5_lineage.get(key) for key in ("run_id", *r5_keys))
        or checks.get("r6_recovery_record_count") != R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_recovery_inventory_invalid",
                stage_id=stage_id,
                expected="the exact 2,002-record immutable-r5 recovery inventory",
                observed={"counts": recovery_counts, "receipt_count": checks.get("r6_recovery_record_count")},
            )
        )
        return

    # R6 material success is meaningful only when the official-page scope was
    # recomputed from the immutable r5-derived recovery inventory.  Keep this
    # validation here (rather than trusting receipt echoes) so final strict
    # success rejects dropped, extra, empty, or stale route observations.
    try:
        from conference_pipeline.acquisition.cvpr_r6_material_index import (
            MaterialPolicyError,
            validate_material_discovery_receipt,
        )
        from conference_pipeline.acquisition.cvpr_r6_media_recovery import (
            derive_r6_source_route_scope,
            validate_r6_audio_recovery_receipt,
            validate_r6_source_route_scope,
        )
        from conference_pipeline.acquisition.public_material_acquisition import (
            PublicMaterialAcquisitionError,
            validate_material_acquisition_receipt,
        )

        scope = derive_r6_source_route_scope(recovery)
        validate_r6_source_route_scope(recovery, recovery)
        audio_classification = validate_r6_audio_recovery_receipt(
            recovery_inventory=recovery,
            receipt=payloads["audio_recovery_receipt"],
            contract=R6ArtifactContract.for_test_fixture(workspace),
        )
        discovery = payloads["material_discovery_receipt"]
        acquisition = payloads["material_acquisition_receipt"]
        validate_material_discovery_receipt(
            workspace=workspace,
            receipt=discovery,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
        validate_material_acquisition_receipt(
            workspace=workspace,
            discovery_receipt=discovery,
            receipt=acquisition,
            recovery_inventory=recovery,
            require_source_scope=True,
        )
        if (
            checks.get("source_scope_count") != scope.source_scope_count
            or checks.get("source_scope_sha256") != scope.source_scope_sha256
            or checks.get("route_observation_sha256")
            != discovery.get("route_observation_sha256")
        ):
            raise ValueError("final receipt source-scope binding drifted")
    except (ValueError, MaterialPolicyError, PublicMaterialAcquisitionError) as exc:
        blockers.append(
            _blocker(
                "stage_contract_r6_source_scope_invalid",
                stage_id=stage_id,
                expected="the exact nonzero recovery-derived route scope and one terminal observation per route",
                observed=str(exc),
            )
        )
        return

    policy_digest = checks.get("material_policy_digest")
    if (
        not _is_sha256(policy_digest)
        or any(
            payloads[name].get("policy_digest") != policy_digest
            for name in (
                "material_discovery_receipt",
                "material_acquisition_receipt",
                "estimate_receipt",
                "benchmark_receipt",
                "provider_receipts",
            )
        )
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_policy_digest_mismatch",
                stage_id=stage_id,
                expected="one finite policy digest across discovery, acquisition, estimate, benchmark, and provider receipts",
                observed=policy_digest,
            )
        )
        return

    ledger = payloads["budget_ledger"]
    if (
        ledger.get("ledger_type") != "r6_committed_plus_actual"
        or not isinstance(ledger.get("committed"), list)
        or not isinstance(ledger.get("actuals"), list)
        or not _is_sha256(ledger.get("ledger_sha256"))
        or checks.get("budget_ledger_digest") != ledger.get("ledger_sha256")
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_budget_ledger_invalid",
                stage_id=stage_id,
                expected="committed-plus-actual r6 ledger bound into the final receipt",
                observed={"ledger_type": ledger.get("ledger_type"), "receipt_digest": checks.get("budget_ledger_digest")},
            )
        )
        return
    try:
        from conference_pipeline.costs.budget_enforcement import (
            R6BudgetLedgerError,
            validate_r6_budget_ledger,
        )

        no_paid_ledger = ledger.get("status") == "not_applicable"
        validate_r6_budget_ledger(
            workspace,
            workspace / R6_ARTIFACT_PATHS["budget_ledger"],
            schedule=None if no_paid_ledger else payloads["estimate_receipt"].get("dispatch_schedule"),
            provider_receipts=payloads["provider_receipts"],
            require_complete=not no_paid_ledger,
            no_paid=no_paid_ledger,
        )
    except (R6BudgetLedgerError, TypeError, ValueError) as exc:
        blockers.append(
            _blocker(
                "stage_contract_r6_budget_ledger_invalid",
                stage_id=stage_id,
                expected="a recomputed complete r6 ledger with matching provider receipt bindings",
                observed=str(exc),
            )
        )
        return

    # The generic strict gate proves only that a non-empty evidence token was
    # supplied.  R6 has a materially stronger requirement: the single
    # registered owner must supply a sanitized mapping bound to this exact
    # execution's processing inventory, receipt bytes, dispatch schedule, and
    # terminal accounting.
    evidence_map = receipt.get("owner_execution_evidence")
    evidence_map = evidence_map if isinstance(evidence_map, Mapping) else {}
    evidence = evidence_map.get(expected_owner_id)
    evidence = evidence if isinstance(evidence, Mapping) else {}
    processing_path = workspace / R6_ARTIFACT_PATHS["processing_inventory"]
    processing_digest = _sha256_file(processing_path) if processing_path.is_file() else None
    expected_receipt_digests = {
        name: checks.get(f"{name}_sha256") for name in receipt_paths
    }
    mode = evidence.get("mode")
    terminal_accounting = evidence.get("terminal_accounting")
    owner_evidence_valid = not (
        set(evidence_map) != {expected_owner_id}
        or evidence.get("owner_path") != "scripts/run_cvpr2026_r6_media_recovery.py"
        or mode not in {"full", "no-paid-lane"}
        or evidence.get("processing_inventory_sha256") != processing_digest
        or evidence.get("receipt_digests") != expected_receipt_digests
        or not isinstance(terminal_accounting, Mapping)
        or terminal_accounting.get("processed_item_count")
        != checks.get("processed_item_count")
        or terminal_accounting.get("terminal_blocker_count")
        != checks.get("terminal_blocker_count")
    )
    if not owner_evidence_valid:
        blockers.append(
            _blocker(
                "stage_contract_r6_owner_execution_evidence_invalid",
                stage_id=stage_id,
                expected="one sanitized execution-derived r6 owner evidence mapping",
                observed=evidence,
            )
        )
    schedule_digest = evidence.get("dispatch_schedule_digest")
    if owner_evidence_valid and mode == "no-paid-lane":
        no_paid = all(
            payloads[name].get("status") == "not_applicable"
            and payloads[name].get("kind") == "no_paid_lane"
            for name in ("benchmark_receipt", "provider_receipts")
        )
        if (
            not no_paid
            or schedule_digest is not None
            or ledger.get("status") != "not_applicable"
            or ledger.get("reservations") != []
            or ledger.get("committed") != []
            or ledger.get("actuals") != []
        ):
            blockers.append(
                _blocker(
                    "stage_contract_r6_no_paid_evidence_invalid",
                    stage_id=stage_id,
                    expected="canonical no-paid envelopes and zero-dispatch ledger",
                    observed={"benchmark": payloads["benchmark_receipt"].get("status"), "ledger": ledger.get("status")},
                )
            )
            return
    elif owner_evidence_valid:
        try:
            from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
                validate_r6_dispatch_schedule,
            )

            schedule = payloads["estimate_receipt"].get("dispatch_schedule")
            checked_schedule = validate_r6_dispatch_schedule(schedule)
            if (
                schedule_digest != checked_schedule["dispatch_schedule_digest"]
                or payloads["benchmark_receipt"].get("dispatch_schedule_digest")
                != checked_schedule["dispatch_schedule_digest"]
                or payloads["provider_receipts"].get("dispatch_schedule_digest")
                != checked_schedule["dispatch_schedule_digest"]
                or ledger.get("dispatch_schedule_digest")
                != checked_schedule["dispatch_schedule_digest"]
                or ledger.get("next_dispatch_ordinal") != len(checked_schedule["entries"])
                or not isinstance(ledger.get("reservations"), list)
                or len(ledger["reservations"]) != len(checked_schedule["entries"])
                or any(
                    not isinstance(row, Mapping) or row.get("state") != "settled"
                    for row in ledger["reservations"]
                )
            ):
                raise ValueError("schedule settlement evidence drifted")
        except (ValueError, TypeError) as exc:
            blockers.append(
                _blocker(
                    "stage_contract_r6_schedule_execution_invalid",
                    stage_id=stage_id,
                    expected="complete settled canonical dispatch schedule",
                    observed=str(exc),
                )
            )
            return

    processor_inputs = receipt.get("declared_processor_inputs")
    if checks.get("processor_input_mode") != "declared_local_payloads" or not isinstance(processor_inputs, list):
        blockers.append(
            _blocker(
                "stage_contract_r6_processor_inputs_invalid",
                stage_id=stage_id,
                expected="explicit declared local processor inputs",
                observed=checks.get("processor_input_mode"),
            )
        )
        return
    declared_paths: dict[str, Mapping[str, Any]] = {}
    for payload, field, local_field in (
        (payloads["audio_recovery_receipt"], "records", "local_media_path"),
        (payloads["material_acquisition_receipt"], "payloads", "local_payload_path"),
    ):
        rows = payload.get(field)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    local_path = row.get(local_field)
                    if isinstance(local_path, str):
                        declared_paths[local_path] = row
    allowed_processor_fields = {
        "record_id",
        "semantic_class",
        "local_path",
        "sha256",
        "byte_count",
        "mime_type",
        "duration_seconds",
    }
    for item in processor_inputs:
        if not isinstance(item, Mapping):
            blockers.append(_blocker("stage_contract_r6_processor_inputs_invalid", stage_id=stage_id, expected="mapping", observed=type(item).__name__))
            return
        local_path = item.get("local_path")
        if (
            set(item) - allowed_processor_fields
            or not isinstance(local_path, str)
            or "://" in local_path
            or any(token in local_path for token in ("*", "?", "["))
            or local_path not in declared_paths
        ):
            blockers.append(_blocker("stage_contract_r6_processor_inputs_invalid", stage_id=stage_id, expected="declared local payload only", observed=local_path))
            return
        source = declared_paths[local_path]
        if any(item.get(key) != source.get(key) for key in ("sha256", "byte_count", "mime_type")):
            blockers.append(_blocker("stage_contract_r6_processor_inputs_invalid", stage_id=stage_id, expected="matching acquired payload digest", observed=local_path))
            return
        if item.get("semantic_class") == "speech":
            duration = item.get("duration_seconds")
            if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(float(duration)) or float(duration) <= 0:
                blockers.append(_blocker("stage_contract_r6_processor_inputs_invalid", stage_id=stage_id, expected="finite positive paid speech duration", observed=duration))
                return

    sublanes = receipt.get("sublanes")
    if not isinstance(sublanes, Mapping) or any(status not in {"pass", "not_applicable"} for status in sublanes.values()):
        blockers.append(_blocker("stage_contract_r6_sublane_failed", stage_id=stage_id, expected="all sublanes pass or be explicitly not applicable", observed=sublanes))

    processed = checks.get("processed_item_count")
    blockers_count = checks.get("terminal_blocker_count")
    if (
        not isinstance(processed, int)
        or isinstance(processed, bool)
        or not isinstance(blockers_count, int)
        or isinstance(blockers_count, bool)
        or processed < 0
        or blockers_count < 0
        or processed != audio_classification.acquired_count
        or blockers_count != audio_classification.terminal_count
    ):
        blockers.append(
            _blocker(
                "stage_contract_r6_accounting_invalid",
                stage_id=stage_id,
                expected=f"processed plus terminal blockers equals {R6_EXPECTED_UNAVAILABLE_CAPTION_COUNT}",
                observed={"processed": processed, "terminal_blockers": blockers_count},
            )
        )

    if not blockers:
        accepted.append(f"{stage_id}: exact r6 receipt, local input, policy, ledger, and complete-accounting contract")


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _canonical_json_digest(
    payload: Mapping[str, Any], *, excluded_keys: frozenset[str]
) -> str:
    canonical = {key: value for key, value in payload.items() if key not in excluded_keys}
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _r5_expected_counts_from_event_media(event_media_inventory: Mapping[str, Any]) -> dict[str, int]:
    """Derive the r5 caption-first accounting from the bound event-media inventory.

    Counts are read from the digest-verified upstream, so the contract checks
    internal consistency (inventory counts, transcript list, blocker list,
    and receipt checks all agree with the upstream) rather than equality
    with a number typed into this module. The three zero entries are policy
    for the ``caption_first_no_asr`` semantic class, not run facts.
    """

    counts = event_media_inventory.get("counts")
    counts = counts if isinstance(counts, Mapping) else {}
    candidates = event_media_inventory.get("caption_candidates")
    candidates = candidates if isinstance(candidates, list) else []
    statuses = [
        str(candidate.get("caption_status") or "")
        for candidate in candidates
        if isinstance(candidate, Mapping)
    ]
    terminal = counts.get("terminal_event_route_count")
    return {
        "terminal_route_count": int(terminal) if isinstance(terminal, int) and not isinstance(terminal, bool) else 0,
        "caption_candidate_count": len(candidates),
        "caption_used_no_asr_count": sum(1 for status in statuses if status == "youtube_caption_available"),
        "youtube_caption_unavailable_count": sum(
            1 for status in statuses if status == "youtube_caption_unavailable"
        ),
        "asr_queue_count": 0,
        "ocr_queue_count": 0,
        "provider_call_count": 0,
    }


def _check_cvpr_r5_class_accounting(
    *,
    stage_id: str,
    checks: Mapping[str, Any],
    inventory: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    counts = inventory.get("counts")
    transcripts = inventory.get("caption_transcripts")
    unavailable = inventory.get("blockers")
    counts = counts if isinstance(counts, Mapping) else {}
    transcripts = transcripts if isinstance(transcripts, list) else None
    unavailable = unavailable if isinstance(unavailable, list) else None
    receipt_count_keys = {
        "terminal_route_count": "event_route_count",
        "caption_candidate_count": "caption_candidate_count",
        "caption_used_no_asr_count": "caption_used_no_asr_count",
        "youtube_caption_unavailable_count": "youtube_caption_unavailable_count",
        "asr_queue_count": "asr_queue_count",
        "ocr_queue_count": "ocr_queue_count",
        "provider_call_count": "provider_call_count",
    }
    receipt_count_match = all(
        checks.get(receipt_count_keys[key]) == value for key, value in expected_counts.items()
    )
    transcript_ids = {
        item.get("source_identity")
        for item in transcripts or []
        if isinstance(item, Mapping) and isinstance(item.get("source_identity"), str)
    }
    unavailable_ids = {
        item.get("source_identity")
        for item in unavailable or []
        if isinstance(item, Mapping) and isinstance(item.get("source_identity"), str)
    }
    valid_transcripts = (
        transcripts is not None
        and len(transcripts) == expected_counts["caption_used_no_asr_count"]
        and all(
            isinstance(item, Mapping)
            and item.get("source_class") == "caption_transcript"
            and item.get("outcome") == "caption_used_no_asr"
            and item.get("video_id") == item.get("source_identity")
            and isinstance(item.get("language"), str)
            and bool(item.get("language").strip())
            and isinstance(item.get("segments"), list)
            and bool(item.get("segments"))
            and _is_sha256(item.get("caption_sha256"))
            and _is_sha256(item.get("output_sha256"))
            for item in transcripts
        )
    )
    valid_unavailable = (
        unavailable is not None
        and len(unavailable) == expected_counts["youtube_caption_unavailable_count"]
        and all(
            isinstance(item, Mapping)
            and item.get("code") == "youtube_caption_unavailable"
            and item.get("asr_eligible") is False
            for item in unavailable
        )
    )
    if (
        dict(counts) != dict(expected_counts)
        or not receipt_count_match
        or not valid_transcripts
        or not valid_unavailable
        or len(transcript_ids) != len(transcripts or [])
        or len(unavailable_ids) != len(unavailable or [])
        or transcript_ids.intersection(unavailable_ids)
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_class_accounting_mismatch",
                stage_id=stage_id,
                expected={
                    "derived_from": "digest-verified r5 event-media inventory",
                    **dict(expected_counts),
                },
                observed={
                    "counts": counts,
                    "transcript_count": len(transcripts or []),
                    "unavailable_count": len(unavailable or []),
                    "receipt_counts": {
                        receipt_count_keys[key]: checks.get(receipt_count_keys[key])
                        for key in expected_counts
                    },
                },
            )
        )
        return
    accepted.append(f"{stage_id}: exact r5 class-separated caption accounting")


def _check_cvpr_r5_capacity_binding(
    *,
    stage_id: str,
    checks: Mapping[str, Any],
    inventory: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    capacity = inventory.get("capacity_reservation")
    capacity = capacity if isinstance(capacity, Mapping) else {}
    receipt_capacity = checks.get("disk_capacity")
    receipt_capacity = receipt_capacity if isinstance(receipt_capacity, Mapping) else {}
    accepted_bytes = capacity.get("accepted_caption_input_bytes")
    reservation = capacity.get("reservation_bytes")
    safety_margin = capacity.get("safety_margin_bytes")
    inventory_counts = inventory.get("counts")
    inventory_counts = inventory_counts if isinstance(inventory_counts, Mapping) else {}
    accepted_caption_count = inventory_counts.get("caption_used_no_asr_count")
    if (
        not isinstance(accepted_caption_count, int)
        or isinstance(accepted_caption_count, bool)
        or capacity.get("accepted_caption_count") != accepted_caption_count
        or not isinstance(accepted_bytes, int)
        or isinstance(accepted_bytes, bool)
        or accepted_bytes <= 0
        or not isinstance(reservation, int)
        or isinstance(reservation, bool)
        or reservation < accepted_bytes
        or not isinstance(safety_margin, int)
        or isinstance(safety_margin, bool)
        or safety_margin <= 0
        or receipt_capacity.get("positive_available_bytes") is not True
        or not _is_sha256(receipt_capacity.get("manifest_sha256"))
        or receipt_capacity.get("output_reservation_bytes") != reservation
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_capacity_binding_mismatch",
                stage_id=stage_id,
                expected="positive capacity bound to the r5 manifest and all 350 caption inputs",
                observed={"capacity": dict(capacity), "receipt_capacity": dict(receipt_capacity)},
            )
        )
        return
    accepted.append(f"{stage_id}: positive r5 caption capacity and reservation binding")


def _check_cvpr_r5_workshop_table(
    *,
    stage_id: str,
    workspace: Path,
    checks: Mapping[str, Any],
    inventory: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    table = inventory.get("workshop_pdf_provenance_table")
    if not isinstance(table, list):
        blockers.append(
            _blocker(
                "stage_contract_r5_workshop_provenance_invalid",
                stage_id=stage_id,
                expected="bounded digest-bound workshop PDF provenance table",
                observed=type(table).__name__,
            )
        )
        return
    accepted_count = 0
    blocker_count = 0
    identities: set[str] = set()
    invalid_rows: list[str] = []
    for row in table:
        if not isinstance(row, Mapping) or not isinstance(row.get("source_identity"), str):
            invalid_rows.append("<invalid>")
            continue
        source_identity = row["source_identity"]
        if source_identity in identities:
            invalid_rows.append(source_identity)
            continue
        identities.add(source_identity)
        predecessor = row.get("predecessor_receipt_sha256")
        if not _is_sha256(predecessor):
            expected_outcome = "workshop_pdf_lineage_unproven"
        elif row.get("mime_type") != "application/pdf":
            expected_outcome = "workshop_pdf_mime_invalid"
        else:
            declared_path = row.get("declared_path")
            path = workspace / str(declared_path) if isinstance(declared_path, str) else None
            if path is None or not path.is_file() or workspace not in (path.resolve(), *path.resolve().parents):
                expected_outcome = "workshop_pdf_path_unproven"
            elif path.read_bytes()[:4] == b"PK\x03\x04":
                expected_outcome = "workshop_pdf_zip_content"
            elif (
                not isinstance(row.get("byte_count"), int)
                or not _is_sha256(row.get("sha256"))
                or path.stat().st_size != row.get("byte_count")
                or _sha256_file(path) != row.get("sha256")
            ):
                expected_outcome = "workshop_pdf_digest_mismatch"
            else:
                expected_outcome = "workshop_paper_text_extraction"
        if row.get("outcome_class") != expected_outcome:
            invalid_rows.append(source_identity)
        if expected_outcome == "workshop_paper_text_extraction":
            accepted_count += 1
            if not _is_sha256(row.get("output_digest")):
                invalid_rows.append(source_identity)
        else:
            blocker_count += 1
            if row.get("output_digest") is not None:
                invalid_rows.append(source_identity)
    table_digest = _canonical_json_digest({"table": table}, excluded_keys=frozenset())
    # The receipt binds the table itself, not an unbounded directory or a
    # count copied from a predecessor receipt.
    expected_table_digest = hashlib.sha256(
        json.dumps(table, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        invalid_rows
        or checks.get("workshop_pdf_provenance_table_sha256") != expected_table_digest
        or checks.get("workshop_pdf_provenance_accepted_count") != accepted_count
        or checks.get("workshop_pdf_provenance_blocker_count") != blocker_count
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_workshop_provenance_invalid",
                stage_id=stage_id,
                expected="one class-correct disposition per declared bounded workshop PDF row",
                observed={
                    "table_digest": table_digest,
                    "expected_table_digest": expected_table_digest,
                    "invalid_rows": invalid_rows,
                    "accepted_count": accepted_count,
                    "blocker_count": blocker_count,
                },
            )
        )
        return
    accepted.append(f"{stage_id}: bounded class-correct workshop PDF provenance")


def _check_cvpr_r5_empty_lanes(
    *,
    stage_id: str,
    receipt: Mapping[str, Any],
    checks: Mapping[str, Any],
    inventory: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    zero_lists = ("asr_queue", "ocr_queue", "media_recovery_queue", "provider_calls")
    if (
        any(inventory.get(key) != [] for key in zero_lists)
        or checks.get("asr_eligible_input_count") != 0
        or checks.get("ocr_eligible_input_count") != 0
        or checks.get("provider_calls_made") != 0
        or receipt.get("provider_calls_made", 0) != 0
        or receipt.get("downloads_made", 0) != 0
    ):
        blockers.append(
            _blocker(
                "stage_contract_r5_queue_activity",
                stage_id=stage_id,
                expected="empty media, ASR, OCR, and provider lanes for the r5 caption-only handoff",
                observed={
                    key: inventory.get(key) for key in zero_lists
                }
                | {
                    "asr_eligible_input_count": checks.get("asr_eligible_input_count"),
                    "ocr_eligible_input_count": checks.get("ocr_eligible_input_count"),
                    "provider_calls_made": checks.get("provider_calls_made"),
                },
            )
        )
        return
    accepted.append(f"{stage_id}: zero ASR, OCR, media-recovery, and provider activity")


def _check_cvpr_r5_workshop_boundary(
    *,
    stage_id: str,
    inventory: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    boundary = inventory.get("workshop_talk_boundary")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    if (
        boundary.get("status") != "not_claimed"
        or boundary.get("reason") != "no receipt-backed workshop talk media"
    ):
        blockers.append(
            _blocker(
                "stage_contract_workshop_coverage_contradiction",
                stage_id=stage_id,
                expected="no workshop talk-aware coverage claim without receipt-backed talk media",
                observed=dict(boundary),
            )
        )
        return
    accepted.append(f"{stage_id}: workshop talk-aware coverage truthfully not claimed")


def _check_cvpr_registration_media_overlay(
    *,
    stage_id: str,
    stage_04_checks: Mapping[str, Any],
    overlay_ledger: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> dict[str, list[str]]:
    """Validate complete r4 candidate accounting and derive ASR/OCR queues.

    A registration attempt can legitimately expose no admissible direct YouTube
    link.  Therefore no truthy-media requirement appears here: the complete
    candidate result set and its final authenticated dispositions are the
    evidence.  Only concrete receipt-backed media records may enter ASR/OCR.
    """

    expected_source_ids = {"asr": [], "ocr": []}
    candidate_results = overlay_ledger.get("candidate_results")
    media_records = overlay_ledger.get("media_records")
    candidate_results = candidate_results if isinstance(candidate_results, list) else None
    media_records = media_records if isinstance(media_records, list) else None
    candidate_count = stage_04_checks.get("official_event_candidate_count")
    accounted_count = stage_04_checks.get("official_event_candidate_accounted_count")
    direct_youtube_count = stage_04_checks.get("direct_youtube_media_record_count")
    if (
        not isinstance(candidate_count, int)
        or isinstance(candidate_count, bool)
        or candidate_count < 0
        or not isinstance(accounted_count, int)
        or isinstance(accounted_count, bool)
        or accounted_count < 0
        or candidate_results is None
        or candidate_count != accounted_count
        or candidate_count != len(candidate_results)
        or any(
            not isinstance(result, Mapping)
            or not isinstance(result.get("program_item_id"), str)
            or not str(result.get("program_item_id")).strip()
            or not isinstance(result.get("authenticated_attempt_status"), str)
            or not str(result.get("authenticated_attempt_status")).strip()
            or not isinstance(result.get("disposition"), str)
            or not str(result.get("disposition")).strip()
            for result in candidate_results or []
        )
    ):
        blockers.append(
            _blocker(
                "stage_contract_registration_media_candidate_accounting_invalid",
                stage_id=stage_id,
                expected=(
                    "nonnegative equal candidate/accounted counts with one final "
                    "authenticated result per r4 official-event candidate"
                ),
                observed={
                    "candidate_count": candidate_count,
                    "accounted_count": accounted_count,
                    "result_count": len(candidate_results or []),
                },
            )
        )
        return expected_source_ids

    if (
        not isinstance(direct_youtube_count, int)
        or isinstance(direct_youtube_count, bool)
        or direct_youtube_count < 0
        or media_records is None
    ):
        blockers.append(
            _blocker(
                "stage_contract_registration_media_record_count_invalid",
                stage_id=stage_id,
                expected="nonnegative direct-YouTube count and media_records list",
                observed={
                    "direct_youtube_media_record_count": direct_youtube_count,
                    "media_record_type": type(media_records).__name__,
                },
            )
        )
        return expected_source_ids

    direct_records = 0
    invalid_media_records: list[int] = []
    for index, record in enumerate(media_records):
        if not isinstance(record, Mapping):
            invalid_media_records.append(index)
            continue
        media_id = record.get("media_id")
        asr_fallback_required = record.get("asr_fallback_required")
        ocr_eligible = record.get("ocr_eligible")
        if (
            not isinstance(media_id, str)
            or not media_id.strip()
            or not isinstance(asr_fallback_required, bool)
            or not isinstance(ocr_eligible, bool)
        ):
            invalid_media_records.append(index)
            continue
        if record.get("direct_youtube") is True:
            direct_records += 1
        if asr_fallback_required:
            expected_source_ids["asr"].append(media_id)
        if ocr_eligible:
            expected_source_ids["ocr"].append(media_id)
    if invalid_media_records or direct_youtube_count != direct_records:
        blockers.append(
            _blocker(
                "stage_contract_registration_media_record_count_invalid",
                stage_id=stage_id,
                expected="nonnegative direct-YouTube count matching valid overlay media records",
                observed={
                    "declared_direct_youtube_count": direct_youtube_count,
                    "derived_direct_youtube_count": direct_records,
                    "invalid_media_record_indexes": invalid_media_records,
                },
            )
        )
        return {"asr": [], "ocr": []}

    for lane, ids in expected_source_ids.items():
        if len(ids) != len(set(ids)):
            blockers.append(
                _blocker(
                    "stage_contract_registration_media_duplicate_queue_source",
                    stage_id=stage_id,
                    expected="each eligible r4 media id appears once per queue",
                    observed={lane: ids},
                )
            )
            return {"asr": [], "ocr": []}
    accepted.append(
        f"{stage_id}: complete r4 candidate accounting and derived registration-media queues"
    )
    return {lane: sorted(ids) for lane, ids in expected_source_ids.items()}


def _check_cvpr_processing_queues(
    *,
    stage_id: str,
    queues: Any,
    expected_source_ids: Mapping[str, Sequence[str]],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    queues = queues if isinstance(queues, Mapping) else {}
    invalid_lanes: list[str] = []
    empty_queue_activity: list[str] = []
    for lane in ("asr", "ocr"):
        lane_payload = queues.get(lane)
        if not isinstance(lane_payload, Mapping):
            invalid_lanes.append(lane)
            continue
        queue_count = lane_payload.get("queue_count")
        provider_calls = lane_payload.get("provider_calls_made")
        actual_cost = lane_payload.get("actual_cost_usd")
        source_media_ids = lane_payload.get("source_media_ids")
        normalized_source_ids = (
            sorted(source_media_ids)
            if isinstance(source_media_ids, list)
            and all(isinstance(item, str) and item.strip() for item in source_media_ids)
            else None
        )
        if (
            not isinstance(queue_count, int)
            or isinstance(queue_count, bool)
            or queue_count < 0
            or not isinstance(provider_calls, int)
            or isinstance(provider_calls, bool)
            or provider_calls < 0
            or not _is_finite_nonnegative_number(actual_cost)
            or normalized_source_ids is None
        ):
            invalid_lanes.append(lane)
            continue
        expected_ids = sorted(expected_source_ids.get(lane, []))
        if (
            len(normalized_source_ids) != len(set(normalized_source_ids))
            or normalized_source_ids != expected_ids
            or queue_count != len(expected_ids)
        ):
            blockers.append(
                _blocker(
                    "stage_contract_queue_content_mismatch",
                    stage_id=stage_id,
                    expected={lane: expected_ids},
                    observed={
                        "source_media_ids": source_media_ids,
                        "queue_count": queue_count,
                    },
                )
            )
            continue
        if queue_count == 0 and (provider_calls != 0 or float(actual_cost) != 0.0):
            empty_queue_activity.append(lane)
    if invalid_lanes:
        blockers.append(
            _blocker(
                "stage_contract_queue_accounting_invalid",
                stage_id=stage_id,
                expected="nonnegative ASR/OCR queue, call, and finite cost fields",
                observed=sorted(invalid_lanes),
            )
        )
    if empty_queue_activity:
        blockers.append(
            _blocker(
                "stage_contract_empty_queue_provider_activity",
                stage_id=stage_id,
                expected="zero provider calls and zero cost when a queue is empty",
                observed=sorted(empty_queue_activity),
            )
        )
    if not invalid_lanes and not empty_queue_activity:
        accepted.append(f"{stage_id}: ASR/OCR queue and provider facts are truthful")


def _check_cvpr_source_class_separation(
    *,
    stage_id: str,
    accounting: Any,
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    accounting = accounting if isinstance(accounting, Mapping) else {}
    if accounting.get("paper_text_counted_as_speech") is not False:
        blockers.append(
            _blocker(
                "stage_contract_source_class_contradiction",
                stage_id=stage_id,
                expected="paper_text_counted_as_speech=false",
                observed=accounting.get("paper_text_counted_as_speech"),
            )
        )
        return
    accepted.append(f"{stage_id}: paper text remains separate from speech")


def _check_cvpr_workshop_coverage(
    *,
    stage_id: str,
    checks: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    accepted: list[str],
) -> None:
    claimed = checks.get("workshop_talk_aware_coverage")
    posture = checks.get("workshop_talk_coverage_posture")
    if claimed is False and posture == "not_claimed_missing_media_evidence":
        accepted.append(f"{stage_id}: workshop talk coverage truthfully not claimed")
        return
    media = checks.get("workshop_media_evidence")
    media = media if isinstance(media, Mapping) else {}
    has_media = bool(media.get("accepted_media_records"))
    has_talk_map = checks.get("vetted_talk_map") is True
    has_governed_discovery = bool(checks.get("governed_boundary_discovery_receipt"))
    if claimed is True and has_media and (has_talk_map or has_governed_discovery):
        accepted.append(f"{stage_id}: workshop talk coverage has media and mapping evidence")
        return
    blockers.append(
        _blocker(
            "stage_contract_workshop_coverage_contradiction",
            stage_id=stage_id,
            expected=(
                "not_claimed_missing_media_evidence, or a coverage claim with "
                "accepted media plus vetted talk map/governed discovery"
            ),
            observed={
                "claimed": claimed,
                "posture": posture,
                "accepted_media_records": media.get("accepted_media_records"),
                "vetted_talk_map": checks.get("vetted_talk_map"),
                "governed_boundary_discovery_receipt": checks.get(
                    "governed_boundary_discovery_receipt"
                ),
            },
        )
    )


def _result(
    *,
    stage_id: str,
    registry_stage_id: str,
    digest: str,
    required_script_ids: Sequence[str],
    accepted: list[str],
    blockers: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> StageContractResult:
    return StageContractResult(
        status="pass" if not blockers else "blocked",
        stage_id=stage_id,
        registry_stage_id=registry_stage_id,
        script_registry_sha256=digest,
        required_script_ids=list(required_script_ids),
        accepted_evidence=accepted,
        blockers=blockers,
        warnings=warnings,
    )


def _pre_spend_result(
    *,
    stage_id: str,
    registry_stage_id: str,
    script_digest: str,
    required_script_ids: Sequence[str],
    accepted: list[str],
    blockers: list[dict[str, Any]],
) -> PreSpendContractResult:
    return PreSpendContractResult(
        status="precondition_passed" if not blockers else "blocked",
        stage_id=stage_id,
        registry_stage_id=registry_stage_id,
        script_registry_sha256=script_digest,
        required_script_ids=list(required_script_ids),
        accepted_evidence=accepted,
        blockers=blockers,
    )


def _precondition_blocker(stage_id: str, precondition: str) -> dict[str, Any]:
    """Return a non-sensitive blocker for a failed prospective-call guard."""

    return {
        "code": "precondition_contract_unmet",
        "stage_id": stage_id,
        "precondition": precondition,
        "retryable": True,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_matches(observed: Any, expected: str) -> bool:
    return isinstance(observed, str) and bool(observed) and observed == expected


def _same_workspace(observed: Any, workspace: str | Path) -> bool:
    if not isinstance(observed, str) or not observed.strip():
        return False
    try:
        return Path(observed).resolve() == Path(workspace).resolve()
    except OSError:
        return False


def _is_finite_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _blocker(
    code: str,
    *,
    stage_id: str,
    expected: Any,
    observed: Any,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "stage_id": stage_id,
        "expected": expected,
        "observed": observed,
        "retryable": True,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _as_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _normalized_workspace_identity(workspace: str | Path) -> str:
    """Return the only accepted conference workspace identity for a binding."""

    raw_path = Path(workspace)
    parts = raw_path.parts
    if ".." in parts:
        raise ValueError("workspace stage binding rejects traversal-like identity")

    if raw_path.is_absolute():
        conference_indexes = [
            index for index, part in enumerate(parts) if part == "conferences"
        ]
        if len(conference_indexes) == 1:
            start = conference_indexes[0]
            candidate = parts[start:]
            if len(candidate) == 2 and candidate[1]:
                return "/".join(candidate)
    elif len(parts) == 2 and parts[0] == "conferences" and parts[1]:
        return "/".join(parts)

    raise ValueError("workspace stage binding requires conferences/<slug>")


def _resolve_workspace_template(
    workspace: Path,
    run_id: str,
    value: Any,
    *,
    artifact_id: str | None,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return (workspace / _template(value, run_id=run_id, artifact_id=artifact_id)).resolve()


def _template(value: str, *, run_id: str, artifact_id: str | None = None) -> str:
    result = value.replace("<run-id>", run_id).replace("{run_id}", run_id)
    if artifact_id:
        result = result.replace("<id>", artifact_id).replace("{id}", artifact_id)
    return result


def _artifact_id_from_receipt(receipt: Mapping[str, Any] | None) -> str | None:
    if not isinstance(receipt, Mapping):
        return None
    selected_scope = receipt.get("selected_scope")
    if isinstance(selected_scope, Mapping):
        for key in ("presentation_id", "deck_id", "id", "topic"):
            value = selected_scope.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("presentation_id", "deck_id", "topic"):
        value = receipt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    command_metadata = receipt.get("command_metadata")
    if isinstance(command_metadata, list):
        for item in command_metadata:
            if not isinstance(item, Mapping):
                continue
            item_scope = item.get("selected_scope")
            if not isinstance(item_scope, Mapping):
                continue
            value = item_scope.get("presentation_id") or item_scope.get("topic")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _get_path(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _display(path: Path) -> str:
    return path.as_posix()
