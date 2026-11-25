"""Pure readiness policy for the shared production preflight stage.

This module deliberately has no environment, network, client-construction, disk,
or receipt-writing side effects.  The runner injects current observations and is
responsible for invoking the canonical receipt and budget primitives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conference_pipeline.costs.budget_enforcement import check_postcall, check_preflight
from conference_pipeline.costs.estimator import collect_workspace_workload, estimate_workspace
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.workflow_runs import validate_run_id


SAFE_PROBE_CLASSIFICATIONS = frozenset({"metadata", "capability", "local"})
# Above this projection the runner may not authorize itself: the operator must
# supply an explicit approval file.  Lowered 800 -> 400 on 2026-09-20 at the
# operator's instruction, after a conference corpus expansion put a full-run
# projection ($411.33) within self-authorizing range.  The receipt/contract
# mode string ``automatic_under_usd_800`` is a stable identifier for "the
# self-authorizing mode", not a restatement of this number; receipts written
# before this change carry the threshold that applied to them.
AUTOMATIC_AUTHORIZATION_THRESHOLD_USD = 400.0


@dataclass(frozen=True)
class ProviderProbeSpec:
    """An explicitly approved, non-mutating provider readiness probe."""

    provider: str
    stage: str
    credential_names: tuple[str, ...]
    operation: str
    classification: str
    outcome_schema: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderProbeResult:
    """Sanitized result of an injected provider probe execution."""

    provider: str
    stage: str
    status: str
    error_type: str | None = None
    retryable: bool = False
    attempt_count: int = 1

    @classmethod
    def succeeded(cls, spec: ProviderProbeSpec) -> "ProviderProbeResult":
        return cls(provider=spec.provider, stage=spec.stage, status="succeeded")

    @classmethod
    def failed(
        cls,
        spec: ProviderProbeSpec,
        *,
        error_type: str,
        retryable: bool = False,
    ) -> "ProviderProbeResult":
        return cls(
            provider=spec.provider,
            stage=spec.stage,
            status="failed",
            error_type=error_type,
            retryable=retryable,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderModelChoice:
    """A provider/model pair approved for a named production run."""

    provider: str
    model: str


@dataclass(frozen=True)
class ProviderStageAllocation:
    """A non-transferable hard ceiling for one provider and one stage."""

    provider: str
    stage: str
    ceiling_usd: float

    @property
    def key(self) -> str:
        return provider_stage_key(self.provider, self.stage)


@dataclass(frozen=True)
class CompleteRunEstimateSnapshot:
    """Canonical non-secret full-run estimate evidence used for approval."""

    scope: str
    corpus_scope: str
    workload: dict[str, Any]
    pricing_task_spec: dict[str, Any]
    paid_stages: tuple[str, ...]
    provider_models: tuple[ProviderModelChoice, ...]
    provider_stage_costs: tuple[tuple[str, float], ...]
    approved_headroom_usd: float
    digest: str
    generated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "corpus_scope": self.corpus_scope,
            "workload": self.workload,
            "pricing_task_spec": self.pricing_task_spec,
            "paid_stages": list(self.paid_stages),
            "provider_models": [asdict(item) for item in self.provider_models],
            "provider_stage_costs": dict(self.provider_stage_costs),
            "approved_headroom_usd": self.approved_headroom_usd,
            "digest": self.digest,
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class ApprovalDeclaration:
    """Immutable operator input; policy code may validate but never rewrite it."""

    run_id: str
    corpus_scope: str
    paid_stages: tuple[str, ...]
    provider_models: tuple[ProviderModelChoice, ...]
    estimate_digest: str
    scope_fingerprint: str
    total_ceiling_usd: float
    provider_stage_allocations: tuple[ProviderStageAllocation, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "corpus_scope": self.corpus_scope,
            "paid_stages": list(self.paid_stages),
            "provider_models": [asdict(item) for item in self.provider_models],
            "estimate_digest": self.estimate_digest,
            "scope_fingerprint": self.scope_fingerprint,
            "total_ceiling_usd": self.total_ceiling_usd,
            "provider_stage_allocations": [asdict(item) for item in self.provider_stage_allocations],
        }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _json_safe(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _without_generated_at(value: Any) -> Any:
    """Remove volatile timestamp metadata from canonical estimate digest inputs."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_generated_at(item)
            for key, item in value.items()
            if key != "generated_at"
        }
    if isinstance(value, list):
        return [_without_generated_at(item) for item in value]
    return value


def canonical_json_digest(value: Any) -> str:
    """Digest canonical, non-secret input data without time-dependent metadata."""

    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def provider_stage_key(provider: str, stage: str) -> str:
    return f"{provider}:{stage}"


def build_complete_run_estimate_snapshot(
    *,
    workload: Any,
    estimate: Any,
    corpus_scope: str,
    paid_stages: Sequence[str],
    provider_stage_costs: Mapping[str, float],
    provider_models: Sequence[ProviderModelChoice] = (),
    approved_headroom_usd: float = 0.0,
    generated_at: str | None = None,
) -> CompleteRunEstimateSnapshot:
    """Normalize full-scope workload/cost inputs; generated-at never affects its digest."""

    workload_payload = _without_generated_at(_json_safe(workload))
    estimate_payload = _without_generated_at(_json_safe(estimate))
    normalized_stages = tuple(sorted(set(paid_stages)))
    normalized_models = tuple(sorted(provider_models, key=lambda item: (item.provider, item.model)))
    normalized_costs = tuple(sorted((str(key), float(value)) for key, value in provider_stage_costs.items()))
    canonical = {
        "scope": "full",
        "corpus_scope": corpus_scope,
        "workload": workload_payload,
        "pricing_task_spec": estimate_payload,
        "paid_stages": normalized_stages,
        "provider_models": [asdict(item) for item in normalized_models],
        "provider_stage_costs": normalized_costs,
        "approved_headroom_usd": approved_headroom_usd,
    }
    return CompleteRunEstimateSnapshot(
        scope="full",
        corpus_scope=corpus_scope,
        workload=workload_payload,
        pricing_task_spec=estimate_payload,
        paid_stages=normalized_stages,
        provider_models=normalized_models,
        provider_stage_costs=normalized_costs,
        approved_headroom_usd=approved_headroom_usd,
        digest=canonical_json_digest(canonical),
        generated_at=generated_at,
    )


def prepare_preflight_estimate(
    workspace: Path,
    *,
    corpus_scope: str,
    paid_stages: Sequence[str],
    provider_stage_costs: Mapping[str, float],
    provider_models: Sequence[ProviderModelChoice] = (),
    approved_headroom_usd: float = 0.0,
    workload_collector: Callable[[Path], Any] = collect_workspace_workload,
    estimator: Callable[..., Any] = estimate_workspace,
    generated_at: str | None = None,
) -> CompleteRunEstimateSnapshot:
    """Call the canonical local estimators in full scope and normalize the result.

    The Plan 01-02 runner command owns atomic persistence of this returned
    snapshot.  Keeping this helper side-effect free makes the scope/cost binding
    independently testable and prevents a preflight policy call from writing a
    receipt or creating a production run artifact.
    """

    workload = workload_collector(workspace)
    estimate = estimator(workspace, scope="full", mode="estimate")
    return build_complete_run_estimate_snapshot(
        workload=workload,
        estimate=estimate,
        corpus_scope=corpus_scope,
        paid_stages=paid_stages,
        provider_stage_costs=provider_stage_costs,
        provider_models=provider_models,
        approved_headroom_usd=approved_headroom_usd,
        generated_at=generated_at,
    )


def build_approval_scope_fingerprint(
    *,
    run_id: str,
    corpus_scope: str,
    paid_stages: Sequence[str],
    provider_models: Sequence[ProviderModelChoice],
    estimate_digest: str,
    total_ceiling_usd: float,
    provider_stage_allocations: Sequence[ProviderStageAllocation] = (),
) -> str:
    """Canonical fingerprint for each approval-scope invalidator."""

    return canonical_json_digest(
        {
            "run_id": run_id,
            "corpus_scope": corpus_scope,
            "paid_stages": sorted(set(paid_stages)),
            "provider_models": [
                asdict(item)
                for item in sorted(provider_models, key=lambda item: (item.provider, item.model))
            ],
            "estimate_digest": estimate_digest,
            "total_ceiling_usd": total_ceiling_usd,
            "provider_stage_allocations": [
                asdict(item)
                for item in sorted(provider_stage_allocations, key=lambda item: item.key)
            ],
        }
    )


def _snapshot_is_complete(snapshot: CompleteRunEstimateSnapshot) -> bool:
    if snapshot.scope != "full" or not snapshot.workload or not snapshot.pricing_task_spec:
        return False
    if (
        not snapshot.corpus_scope
        or not snapshot.paid_stages
        or not snapshot.provider_models
        or not snapshot.provider_stage_costs
        or not isinstance(snapshot.approved_headroom_usd, (int, float))
        or not math.isfinite(snapshot.approved_headroom_usd)
        or snapshot.approved_headroom_usd < 0
    ):
        return False
    costs = dict(snapshot.provider_stage_costs)
    if len(costs) != len(snapshot.provider_stage_costs):
        return False
    if any(
        not key
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for key, value in costs.items()
    ):
        return False
    return all(
        any(key.endswith(f":{stage}") for key in costs)
        for stage in snapshot.paid_stages
    )


def derive_automatic_approval(
    snapshot: CompleteRunEstimateSnapshot | None,
    *,
    run_id: str,
    threshold_usd: float = AUTOMATIC_AUTHORIZATION_THRESHOLD_USD,
) -> ApprovalDeclaration | None:
    """Derive the only file-free authority permitted by the operator policy.

    This returns a declaration only from the current full-scope snapshot.  It
    never accepts a hand-authored marker, transfers spare capacity between
    stages, or turns an unpriced/non-finite projection into an authorization.
    """

    if (
        snapshot is None
        or not _snapshot_is_complete(snapshot)
        or not isinstance(threshold_usd, (int, float))
        or not math.isfinite(threshold_usd)
        or threshold_usd < 0
    ):
        return None
    try:
        safe_run_id = validate_run_id(run_id)
    except ValueError:
        return None

    projected_total = sum(cost for _, cost in snapshot.provider_stage_costs)
    total_ceiling_usd = round(projected_total + snapshot.approved_headroom_usd, 6)
    if not math.isfinite(total_ceiling_usd) or total_ceiling_usd < 0 or total_ceiling_usd > threshold_usd:
        return None

    allocations = tuple(
        ProviderStageAllocation(
            provider=key.split(":", 1)[0],
            stage=key.split(":", 1)[1],
            ceiling_usd=cost,
        )
        for key, cost in snapshot.provider_stage_costs
        if ":" in key
    )
    if len(allocations) != len(snapshot.provider_stage_costs):
        return None
    return ApprovalDeclaration(
        run_id=safe_run_id,
        corpus_scope=snapshot.corpus_scope,
        paid_stages=snapshot.paid_stages,
        provider_models=snapshot.provider_models,
        estimate_digest=snapshot.digest,
        scope_fingerprint=build_approval_scope_fingerprint(
            run_id=safe_run_id,
            corpus_scope=snapshot.corpus_scope,
            paid_stages=snapshot.paid_stages,
            provider_models=snapshot.provider_models,
            estimate_digest=snapshot.digest,
            total_ceiling_usd=total_ceiling_usd,
            provider_stage_allocations=allocations,
        ),
        total_ceiling_usd=total_ceiling_usd,
        provider_stage_allocations=allocations,
    )


def validate_approval_scope(
    approval: ApprovalDeclaration | None,
    snapshot: CompleteRunEstimateSnapshot | None,
) -> list[dict[str, Any]]:
    """Fail closed unless immutable approval exactly covers the current full-run snapshot."""

    if snapshot is None or not _snapshot_is_complete(snapshot):
        return [_blocker("canonical_estimate_required", affected_stage="runner-preflight")]
    if approval is None:
        return [_blocker("approval_required", affected_stage="runner-preflight")]

    blockers: list[dict[str, Any]] = []
    if approval.estimate_digest != snapshot.digest:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    expected_fingerprint = build_approval_scope_fingerprint(
        run_id=approval.run_id,
        corpus_scope=approval.corpus_scope,
        paid_stages=approval.paid_stages,
        provider_models=approval.provider_models,
        estimate_digest=approval.estimate_digest,
        total_ceiling_usd=approval.total_ceiling_usd,
        provider_stage_allocations=approval.provider_stage_allocations,
    )
    if approval.scope_fingerprint != expected_fingerprint:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    try:
        validate_run_id(approval.run_id)
    except ValueError:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    if not approval.corpus_scope or approval.corpus_scope != _snapshot_corpus_scope(snapshot):
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    if tuple(sorted(set(approval.paid_stages))) != snapshot.paid_stages:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    if not approval.provider_models or approval.total_ceiling_usd <= 0:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))
    if tuple(sorted(approval.provider_models, key=lambda item: (item.provider, item.model))) != snapshot.provider_models:
        blockers.append(_blocker("approval_scope_mismatch", affected_stage="runner-preflight"))

    allocations = {allocation.key: allocation for allocation in approval.provider_stage_allocations}
    for key, projected_cost in snapshot.provider_stage_costs:
        allocation = allocations.get(key)
        if allocation is None:
            blockers.append(_blocker("missing_provider_stage_allocation", affected_stage=key))
        elif not math.isfinite(allocation.ceiling_usd) or allocation.ceiling_usd < projected_cost or allocation.ceiling_usd < 0:
            blockers.append(_blocker("provider_stage_allocation_exceeded", affected_stage=key))
    required_total = round(
        sum(cost for _, cost in snapshot.provider_stage_costs) + snapshot.approved_headroom_usd,
        6,
    )
    if (
        not math.isfinite(approval.total_ceiling_usd)
        or approval.total_ceiling_usd < required_total
    ):
        blockers.append(_blocker("run_ceiling_exceeded", affected_stage="runner-preflight"))
    return normalize_root_causes(blockers)


def _snapshot_corpus_scope(snapshot: CompleteRunEstimateSnapshot) -> str | None:
    return snapshot.corpus_scope


def authorize_prospective_call(
    approval: ApprovalDeclaration | None,
    snapshot: CompleteRunEstimateSnapshot | None,
    *,
    provider: str,
    stage: str,
    projected_cost_usd: float | None,
    spent_total_usd: float,
    spent_by_provider_stage: Mapping[str, float],
) -> dict[str, Any]:
    """Apply named-run total and exact provider-stage caps before a paid call.

    Runner integration must additionally invoke ``check_preflight`` immediately
    before dispatch and ``check_postcall`` after completion.  This policy only
    owns the named approval ledger; it never transfers unused allocations.
    """

    blockers = validate_approval_scope(approval, snapshot)
    if (
        projected_cost_usd is None
        or not isinstance(projected_cost_usd, (int, float))
        or not math.isfinite(projected_cost_usd)
        or projected_cost_usd < 0
    ):
        blockers.append(_blocker("unpriced_call", affected_provider=provider, affected_stage=stage))
    if (
        approval is not None
        and isinstance(projected_cost_usd, (int, float))
        and math.isfinite(projected_cost_usd)
        and projected_cost_usd >= 0
    ):
        allocation_key = provider_stage_key(provider, stage)
        allocations = {allocation.key: allocation.ceiling_usd for allocation in approval.provider_stage_allocations}
        ceiling = allocations.get(allocation_key)
        if ceiling is None:
            blockers.append(_blocker("missing_provider_stage_allocation", affected_provider=provider, affected_stage=stage))
        elif spent_by_provider_stage.get(allocation_key, 0.0) + projected_cost_usd > ceiling:
            blockers.append(_blocker("provider_stage_allocation_exceeded", affected_provider=provider, affected_stage=stage))
        if spent_total_usd + projected_cost_usd > approval.total_ceiling_usd:
            blockers.append(_blocker("run_ceiling_exceeded", affected_stage="runner-preflight"))
    return {"authorized": not blockers, "blockers": normalize_root_causes(blockers)}


def enforce_paid_call_preflight(
    workspace: Path,
    task_id: str,
    approval: ApprovalDeclaration | None,
    snapshot: CompleteRunEstimateSnapshot | None,
    *,
    provider: str,
    stage: str,
    projected_cost_usd: float | None,
    spent_total_usd: float,
    spent_by_provider_stage: Mapping[str, float],
    registry: Any = None,
    shared_preflight: Callable[..., Any] = check_preflight,
) -> dict[str, Any]:
    """Combine named allocation authorization with the shared global/task cap."""

    result = authorize_prospective_call(
        approval,
        snapshot,
        provider=provider,
        stage=stage,
        projected_cost_usd=projected_cost_usd,
        spent_total_usd=spent_total_usd,
        spent_by_provider_stage=spent_by_provider_stage,
    )
    if result["blockers"]:
        return result
    shared = shared_preflight(workspace, task_id, projected_cost_usd, registry=registry)
    if getattr(shared, "breach", False):
        result["authorized"] = False
        result["blockers"] = [_blocker("shared_preflight_cap_exceeded", affected_stage=stage)]
    return result


def enforce_paid_call_postcall(
    workspace: Path,
    run_id: str,
    approval: ApprovalDeclaration | None,
    snapshot: CompleteRunEstimateSnapshot | None,
    *,
    provider: str,
    stage: str,
    actual_cost_usd: float,
    spent_by_provider_stage: Mapping[str, float],
    registry: Any = None,
    shared_postcall: Callable[..., Any] = check_postcall,
) -> dict[str, Any]:
    """Check an actual call against both the shared cap and immutable allocation."""

    blockers = validate_approval_scope(approval, snapshot)
    if not isinstance(actual_cost_usd, (int, float)) or not math.isfinite(actual_cost_usd) or actual_cost_usd < 0:
        blockers.append(_blocker("invalid_actual_cost", affected_provider=provider, affected_stage=stage))
    if approval is not None and isinstance(actual_cost_usd, (int, float)) and math.isfinite(actual_cost_usd) and actual_cost_usd >= 0:
        key = provider_stage_key(provider, stage)
        allocations = {allocation.key: allocation.ceiling_usd for allocation in approval.provider_stage_allocations}
        ceiling = allocations.get(key)
        if ceiling is None or spent_by_provider_stage.get(key, 0.0) + actual_cost_usd > ceiling:
            blockers.append(_blocker("provider_stage_allocation_exceeded", affected_provider=provider, affected_stage=stage))
    shared = shared_postcall(workspace, run_id, actual_cost_usd, registry=registry)
    if getattr(shared, "breach", False):
        blockers.append(_blocker("shared_postcall_cap_exceeded", affected_stage=stage))
    return {"authorized": not blockers, "blockers": normalize_root_causes(blockers)}


VOLATILE_RECHECKS = ("auth", "rate_limit", "approval_headroom", "storage_capacity")


def build_static_readiness_fingerprint(
    *,
    runner_implementation: str,
    workspace_manifest: str,
    corpus_scope: str,
    provider_model_scope: Mapping[str, str],
    approval_scope: str,
    disk_estimate_input: str,
    workspace_identity: str,
    command: str,
) -> dict[str, Any]:
    """Hash every stable, non-secret input required for resume compatibility.

    Credential values and all credential-derived markers are intentionally absent:
    credential health is a volatile recheck required before every dispatch.
    """

    normalized_provider_models = dict(sorted(provider_model_scope.items()))
    forbidden_markers = ("api_key", "authorization", "cookie", "credential", "password", "secret", "token")
    if any(
        not isinstance(provider, str)
        or not isinstance(model, str)
        or any(marker in provider.lower() or marker in model.lower() for marker in forbidden_markers)
        for provider, model in normalized_provider_models.items()
    ):
        raise ValueError("credential-derived values are forbidden in static readiness fingerprints")
    inputs = {
        "runner_implementation": runner_implementation,
        "workspace_manifest": workspace_manifest,
        "corpus_scope": corpus_scope,
        "provider_model_scope": normalized_provider_models,
        "approval_scope": approval_scope,
        "disk_estimate_input": disk_estimate_input,
        "workspace_identity": workspace_identity,
        "command": command,
    }
    if any(not value for key, value in inputs.items() if key != "provider_model_scope") or not inputs["provider_model_scope"]:
        raise ValueError("static readiness fingerprint requires every non-secret compatibility input")
    return {
        "digest": canonical_json_digest(inputs),
        "inputs": inputs,
        "credential_value_marker_persisted": False,
    }


def evaluate_resume_compatibility(
    prior_receipt: Mapping[str, Any] | None,
    current_fingerprint: Mapping[str, Any],
    *,
    stage_id: str,
    new_attempt_id: str | None = None,
) -> dict[str, Any]:
    """Return an immutable resume/supersession decision with mandatory rechecks."""

    base = {"required_volatile_rechecks": list(VOLATILE_RECHECKS)}
    if stage_id == "runner-preflight":
        return {
            **base,
            "eligible_for_skip": False,
            "reason": "runner_preflight_requires_fresh_full_check",
        }
    if not prior_receipt or prior_receipt.get("status") != "succeeded":
        reason = "prior_receipt_not_succeeded"
    elif prior_receipt.get("static_fingerprint") != dict(current_fingerprint):
        reason = "static_fingerprint_mismatch"
    else:
        return {**base, "eligible_for_skip": True, "reason": "exact_succeeded_receipt_match"}

    receipt_id = prior_receipt.get("receipt_id") if prior_receipt else None
    return {
        **base,
        "eligible_for_skip": False,
        "reason": reason,
        "new_attempt": {
            "attempt_id": new_attempt_id,
            "supersedes_receipt_id": receipt_id,
            "reason": reason,
        },
    }


def validate_acknowledgement_clearance(
    acknowledgement: Mapping[str, Any] | None,
    *,
    fresh_recheck: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep acknowledgements as audit history; only a new pass clears a blocker."""

    acknowledgement_id = None
    if acknowledgement and isinstance(acknowledgement.get("acknowledgement_id"), str):
        acknowledgement_id = acknowledgement["acknowledgement_id"]
    return {
        "cleared": bool(fresh_recheck and fresh_recheck.get("status") == "succeeded"),
        "acknowledgement_id": acknowledgement_id,
        "clearance_requires": "fresh_passing_recheck",
    }


# This is the complete planned pipeline contract, not a selection for the stage
# currently being dispatched.  OpenRouter is explicitly retained as a paid-stage
# failover whose readiness must be known before any live work begins.
PLANNED_PROVIDER_PROBES: tuple[ProviderProbeSpec, ...] = (
    ProviderProbeSpec(
        provider="openai",
        stage="analysis-and-presentation",
        credential_names=("OPENAI_API_KEY",),
        operation="GET /v1/models",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
    ProviderProbeSpec(
        provider="anthropic",
        stage="presentation-reflow",
        credential_names=("ANTHROPIC_API_KEY",),
        operation="GET /v1/models",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
    ProviderProbeSpec(
        provider="google",
        stage="transcription-and-podcast",
        credential_names=("GOOGLE_API_KEY",),
        operation="GET /v1beta/models",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
    ProviderProbeSpec(
        provider="tavily",
        stage="source-discovery",
        credential_names=("TAVILY_API_KEY",),
        operation="GET /credits",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
    ProviderProbeSpec(
        provider="openrouter",
        stage="remote-extraction-failover",
        credential_names=("OPENROUTER_API_KEY",),
        operation="GET /api/v1/models",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
    ProviderProbeSpec(
        provider="openreview",
        stage="source-discovery",
        credential_names=("OPENREVIEW_USERNAME", "OPENREVIEW_PASSWORD"),
        operation="GET /notes?limit=1",
        classification="metadata",
        outcome_schema=("status", "provider", "stage"),
    ),
)


def _blocker(
    code: str,
    *,
    affected_provider: str | None = None,
    affected_stage: str | None = None,
    credential_names: Sequence[str] = (),
    error_type: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe blocker without accepting error-message material."""

    result: dict[str, Any] = {
        "code": code,
        "affected_provider": affected_provider,
        "affected_stage": affected_stage,
        "credential_names": list(credential_names),
        "error_type": error_type,
    }
    return {key: value for key, value in result.items() if value not in (None, [], "")}


def validate_credential_inventory(
    planned_probes: Sequence[ProviderProbeSpec],
    credential_values: Mapping[str, object],
) -> list[dict[str, Any]]:
    """Return one blocker per provider/stage with missing planned credentials.

    Values are read only to determine empty versus present and are never returned,
    hashed, or passed to receipt-oriented helpers.
    """

    blockers: list[dict[str, Any]] = []
    for spec in planned_probes:
        missing = [name for name in spec.credential_names if not credential_values.get(name)]
        if missing:
            blockers.append(
                _blocker(
                    "missing_credential",
                    affected_provider=spec.provider,
                    affected_stage=spec.stage,
                    credential_names=missing,
                )
            )
    return blockers


def validate_provider_probe_specs(
    planned_probes: Sequence[ProviderProbeSpec],
    *,
    sdk_available: Mapping[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Ensure every planned provider has an approved, safe probe declaration."""

    blockers: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for spec in planned_probes:
        key = (spec.provider, spec.stage)
        if key in seen:
            blockers.append(
                _blocker("duplicate_probe_declaration", affected_provider=spec.provider, affected_stage=spec.stage)
            )
            continue
        seen.add(key)
        if not spec.credential_names:
            blockers.append(_blocker("missing_probe_credentials", affected_provider=spec.provider, affected_stage=spec.stage))
        if not spec.operation.startswith("GET ") or spec.classification not in SAFE_PROBE_CLASSIFICATIONS:
            blockers.append(
                _blocker("invalid_probe_classification", affected_provider=spec.provider, affected_stage=spec.stage)
            )
        if not spec.outcome_schema or "status" not in spec.outcome_schema:
            blockers.append(_blocker("invalid_probe_outcome_schema", affected_provider=spec.provider, affected_stage=spec.stage))
        if sdk_available is not None and not sdk_available.get(spec.provider, False):
            blockers.append(_blocker("provider_sdk_unavailable", affected_provider=spec.provider, affected_stage=spec.stage))
    return blockers


def execute_probe_with_single_recovery(
    spec: ProviderProbeSpec,
    probe: Callable[[ProviderProbeSpec], ProviderProbeResult],
    *,
    reload_environment: Callable[[], None],
    rebuild_client: Callable[[], None],
) -> ProviderProbeResult:
    """Perform at most one injected reload/rebuild recovery for the same probe."""

    first = probe(spec)
    if first.status == "succeeded" or not first.retryable:
        return replace(first, attempt_count=1)

    reload_environment()
    rebuild_client()
    second = probe(spec)
    return replace(second, attempt_count=2)


def evaluate_probe_result(result: ProviderProbeResult) -> list[dict[str, Any]]:
    """Normalize a failed probe to a root blocker without provider error text."""

    if result.status == "succeeded":
        return []
    return [
        _blocker(
            "probe_failed",
            affected_provider=result.provider,
            affected_stage=result.stage,
            error_type=result.error_type or "unknown",
        )
    ]


def validate_storage_estimate(
    estimate_path: Path,
    *,
    available_bytes: int,
) -> dict[str, Any]:
    """Fail closed unless a complete scoped estimate plus margin fits available bytes."""

    try:
        payload = json.loads(estimate_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "required_bytes": None, "blockers": [_blocker("invalid_storage_estimate")]}

    required_fields = {
        "corpus_scope",
        "inventory_digest",
        "manifest_digest",
        "remaining_families",
        "projected_remaining_bytes",
        "estimator_version",
        "safety_margin_bytes",
    }
    if not isinstance(payload, dict) or not required_fields.issubset(payload):
        return {"ready": False, "required_bytes": None, "blockers": [_blocker("incomplete_storage_estimate")]}

    families = payload["remaining_families"]
    projected = payload["projected_remaining_bytes"]
    margin = payload["safety_margin_bytes"]
    if (
        not isinstance(families, dict)
        or not families
        or any(not isinstance(value, int) or value <= 0 for value in families.values())
        or not isinstance(projected, int)
        or projected <= 0
        or not isinstance(margin, int)
        or margin <= 0
        or not isinstance(available_bytes, int)
        or available_bytes < 0
        or sum(families.values()) != projected
        or not all(isinstance(payload[field], str) and payload[field] for field in ("corpus_scope", "inventory_digest", "manifest_digest", "estimator_version"))
    ):
        return {"ready": False, "required_bytes": None, "blockers": [_blocker("invalid_storage_estimate")]}

    required_bytes = projected + margin
    if available_bytes < required_bytes:
        return {
            "ready": False,
            "required_bytes": required_bytes,
            "blockers": [_blocker("insufficient_storage", affected_stage="download-corpus")],
        }
    return {"ready": True, "required_bytes": required_bytes, "blockers": []}


def normalize_root_causes(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate independent root causes and suppress explicitly dependent symptoms."""

    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, str | None]] = set()
    for raw in blockers:
        if raw.get("depends_on"):
            continue
        code = raw.get("code")
        if not isinstance(code, str) or not code:
            continue
        provider = raw.get("affected_provider")
        stage = raw.get("affected_stage") or raw.get("stage")
        key = (code, provider if isinstance(provider, str) else None, stage if isinstance(stage, str) else None)
        if key in seen:
            continue
        seen.add(key)
        credential_names = raw.get("credential_names", ())
        normalized.append(
            _blocker(
                code,
                affected_provider=provider if isinstance(provider, str) else None,
                affected_stage=stage if isinstance(stage, str) else None,
                credential_names=credential_names if isinstance(credential_names, (list, tuple)) else (),
                error_type=raw.get("error_type") if isinstance(raw.get("error_type"), str) else None,
            )
        )
    return normalized


_REMEDIATION = {
    "missing_credential": ("credential_environment", "Set the named environment variable in the canonical repository environment file."),
    "provider_sdk_unavailable": ("provider_sdk", "Install or repair the approved provider SDK, then rerun the same declared probe."),
    "invalid_probe_classification": ("probe_declaration", "Declare an approved non-mutating metadata, capability, or local probe."),
    "invalid_probe_outcome_schema": ("probe_declaration", "Declare the safe status-oriented probe outcome schema."),
    "probe_failed": ("provider_probe", "Resolve the reported provider access condition and rerun the declared probe."),
    "approval_required": ("immutable_approval", "Obtain a new immutable approval for the current full-run scope."),
    "insufficient_storage": ("storage_capacity", "Provide sufficient available storage for the validated estimate plus its margin."),
}


def build_remediation_report(blockers: Sequence[Mapping[str, Any]], *, label: str = "") -> dict[str, Any]:
    """Render secret-free grouped remediation as JSON, Markdown, and console text.

    ``label`` names the conference whose runner blocked (its orchestrator's
    display label); the report never derives one itself.
    """

    title = f"{label} production preflight" if label else "Production preflight"
    entries: list[dict[str, Any]] = []
    markdown_lines = [f"# {title} remediation", ""]
    console_lines = [f"{title} blocked:"]
    for blocker in normalize_root_causes(blockers):
        code = str(blocker["code"])
        proof, remedy = _REMEDIATION.get(code, ("fresh_passing_recheck", "Correct the root cause and rerun the failed check."))
        entry = {
            **sanitize_for_receipt(blocker),
            "remediation": remedy,
            "clearance_evidence": proof,
        }
        entries.append(entry)
        target = entry.get("affected_provider") or entry.get("affected_stage") or "preflight"
        markdown_lines.append(f"- `{code}` ({target}): {remedy} Clearance: `{proof}`.")
        console_lines.append(f"- {code} ({target}): {remedy}")
    return {"json": entries, "markdown": "\n".join(markdown_lines) + "\n", "console": "\n".join(console_lines)}
