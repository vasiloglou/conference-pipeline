"""Stage 1: runner preflight, shared by every conference orchestrator.

The base is the approval / estimate / capacity preflight, the stricter of the
runner implementations it replaces. Before any environment reload, provider
client or probe, it requires three immutable inputs inside the named run
directory:

1. the canonical no-provider full-run estimate
   (:func:`prepare_preflight_estimate`, ``preflight-full-run-estimate.json``);
2. a spend authority: an explicit immutable approval
   (``provider-approval.json``) whose scope fingerprint matches the estimate,
   or the narrow automatic authority a complete, finite estimate under the
   self-authorizing threshold derives;
3. a capacity input (:func:`prepare_preflight_storage_estimate`,
   ``preflight-storage-estimate.json``) bound to the estimate's provenance and
   the workspace's reviewed storage policy, checked against the bytes free now.

It then resolves the orchestrator's owner in the stage owner registry, runs
the conference's extra checks, reloads the environment once, inventories the
declared credentials (presence only), runs each declared non-mutating
provider probe with a single recovery, verifies the strict stage contract and
writes the receipt, a remediation report and (when blocked) a blocker receipt.
A preflight never resumes: every run re-checks (``rerun_required``).

Paid calls later in the run go through :func:`dispatch_paid_call`, which
re-validates the same estimate and authority before the owner runs and
enforces the per-provider/stage ceilings after it.

Everything a conference decides comes in through the
:class:`~conference_pipeline.production.context.ProductionRunnerContext`
(workspace, run, stage table, receipt type, code identity, repo root) and
:class:`PreflightSettings` (display label, corpus scope, owner identity,
failover providers). Provider probes and extra readiness checks are the
``provider_probes`` / ``extra_preflight_checks`` hooks of
:class:`~conference_pipeline.production.hooks.ConferenceHooks`. The module
carries no conference literal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.costs.estimator import collect_workspace_workload, estimate_workspace
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks
from conference_pipeline.production.receipts import DEFAULT_STAGE_IO, StageIO
from conference_pipeline.production_preflight import (
    AUTOMATIC_AUTHORIZATION_THRESHOLD_USD,
    PLANNED_PROVIDER_PROBES,
    ApprovalDeclaration,
    CompleteRunEstimateSnapshot,
    ProviderModelChoice,
    ProviderProbeResult,
    ProviderProbeSpec,
    ProviderStageAllocation,
    build_complete_run_estimate_snapshot,
    build_remediation_report,
    canonical_json_digest,
    derive_automatic_approval,
    enforce_paid_call_postcall,
    enforce_paid_call_preflight,
    evaluate_probe_result,
    execute_probe_with_single_recovery,
    validate_approval_scope,
    validate_credential_inventory,
    validate_provider_probe_specs,
    validate_storage_estimate,
)
from conference_pipeline.production_preflight import (
    prepare_preflight_estimate as build_preflight_estimate,
)
from conference_pipeline.provider_receipts import (
    BLOCKER_RECEIPT_MARKER,
    BLOCKER_RECEIPT_SCHEMA_VERSION,
    BlockerClass,
    BlockerReceipt,
    sanitize_for_receipt,
    write_blocker_receipt,
)
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from conference_pipeline.workflow_runs import validate_run_id

STAGE_ID = "runner-preflight"

PREPARED_ESTIMATE_FILENAME = "preflight-full-run-estimate.json"
APPROVAL_FILENAME = "provider-approval.json"
CAPACITY_ESTIMATE_FILENAME = "preflight-storage-estimate.json"
CAPACITY_POLICY_FILENAME = "preflight-storage-policy.json"
REMEDIATION_JSON_FILENAME = "runner-preflight-remediation.json"
REMEDIATION_MARKDOWN_FILENAME = "runner-preflight-remediation.md"
VOLATILE_PREFLIGHT_DIR = "volatile-preflight"

AUTOMATIC_AUTHORITY_MODE = "automatic_under_usd_800"
EXPLICIT_AUTHORITY_MODE = "explicit_immutable"
# A complete full-run projection must route at least one paid stage through a
# failover provider, so a primary-provider outage cannot strand a paid run.
DEFAULT_FAILOVER_PROVIDERS: tuple[str, ...] = ("openrouter",)
PROVIDER_DISPLAY_NAMES: Mapping[str, str] = {"openrouter": "OpenRouter"}
# The registry fields an owner registration is compared on.
OWNER_REGISTRATION_FIELDS: tuple[str, ...] = ("path", "scope", "existence", "status")
SAFE_PROBE_ERRORS = frozenset({"timeout", "rate_limited", "unauthorized", "forbidden", "client_unavailable", "unknown"})

ProbeFn = Callable[[ProviderProbeSpec], ProviderProbeResult]


def _label_from_slug(slug: str) -> str:
    return re.sub(r"\d+$", "", slug).upper() or slug.upper()


@dataclass(frozen=True)
class PreflightSettings:
    """What a conference decides about its preflight.

    ``label`` names the conference in remediation text and blocker codes
    (``<label>_runner_owner_binding_invalid``); ``corpus_scope`` is the scope
    every estimate, approval and capacity input binds to. ``owner_id`` is the
    stage-owner registry id recorded as the preflight's owner (``None``: the
    registry's required id for the stage); ``owner_registration`` is the exact
    registration that owner must have (``None``: the owner must exist, and its
    registration is recorded as found). ``canonical_env_file`` is the file a
    probe recovery reloads (``None``: ``<repo_root>/.env``).
    """

    label: str
    corpus_scope: str
    owner_id: str | None = None
    owner_registration: Mapping[str, str] | None = None
    canonical_env_file: Path | None = None
    failover_providers: tuple[str, ...] = DEFAULT_FAILOVER_PROVIDERS
    openreview_auth_label: str | None = None

    @property
    def owner_binding_code(self) -> str:
        return f"{self.label.lower()}_runner_owner_binding_invalid"


def default_settings(ctx: ProductionRunnerContext, **overrides: Any) -> PreflightSettings:
    """Settings derived from the workspace slug; ``overrides`` replace any field."""

    values: dict[str, Any] = {
        "label": _label_from_slug(ctx.slug),
        "corpus_scope": f"all-discovered-{ctx.slug}",
        "openreview_auth_label": f"{ctx.slug}-production-preflight",
    }
    values.update(overrides)
    return PreflightSettings(**values)


# --------------------------------------------------------------------------
# Named-run inputs
# --------------------------------------------------------------------------


def named_run_dir(workspace: Path, run_id: str) -> Path:
    """A workspace-contained run directory, built only after canonical validation."""

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    target = (resolved_workspace / "storage" / "workflow-runs" / safe_run_id).resolve()
    try:
        target.relative_to(resolved_workspace)
    except ValueError as exc:  # pragma: no cover - validate_run_id guards normal inputs
        raise ValueError("workflow run directory must remain inside the workspace") from exc
    return target


def require_named_run_input(path: Path, *, workspace: Path, run_id: str) -> Path:
    """Reject authority inputs that are not immutable files beneath this run."""

    target = named_run_dir(workspace, run_id)
    resolved = path.resolve()
    try:
        resolved.relative_to(target)
    except ValueError as exc:
        raise ValueError("preflight authority inputs must stay inside the named run directory") from exc
    return resolved


def as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise ValueError("canonical estimator must return an object-shaped report")
    return value


def authority_paths(workspace: Path, run_id: str) -> tuple[Path, Path, Path]:
    """The default prepared-estimate, approval and capacity paths of a run."""

    target = named_run_dir(workspace, run_id)
    return target / PREPARED_ESTIMATE_FILENAME, target / APPROVAL_FILENAME, target / CAPACITY_ESTIMATE_FILENAME


# --------------------------------------------------------------------------
# The canonical full-run estimate
# --------------------------------------------------------------------------


def provider_stage_projection(
    estimate: dict[str, Any], *, failover_providers: Sequence[str] = DEFAULT_FAILOVER_PROVIDERS
) -> tuple[dict[str, float], tuple[ProviderModelChoice, ...]]:
    """Normalize every paid provider/stage pair from the full estimator report.

    A paid task without a finite, non-negative price fails here: the
    estimator prices an unmeasured paid task from its declared fallback
    budget or refuses it, and a refusal must never become a silent $0.
    """

    projection: dict[str, float] = {}
    models: dict[tuple[str, str], ProviderModelChoice] = {}
    tasks = estimate.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("canonical estimate is missing task projections")
    for task in tasks:
        if not isinstance(task, dict) or not task.get("enabled", False):
            continue
        provider = task.get("provider")
        stage = task.get("stage")
        model = task.get("model")
        cost = task.get("estimated_cost_usd")
        if not all(isinstance(value, str) and value for value in (provider, stage, model)):
            continue
        if provider == "local":
            continue
        if not isinstance(cost, (int, float)) or cost < 0:
            raise ValueError(f"unpriced provider-stage projection: {provider}:{stage}")
        key = f"{provider}:{stage}"
        projection[key] = round(projection.get(key, 0.0) + float(cost), 6)
        models[(provider, model)] = ProviderModelChoice(provider=provider, model=model)
    missing = [name for name in failover_providers if not any(key.startswith(f"{name}:") for key in projection)]
    if not projection or missing:
        name = (missing or list(failover_providers) or ["a paid"])[0]
        raise ValueError(f"complete full-run projection must include {PROVIDER_DISPLAY_NAMES.get(name, name)} failover")
    return dict(sorted(projection.items())), tuple(sorted(models.values(), key=lambda item: (item.provider, item.model)))


def storage_estimate_inputs(workspace: Path, workload: dict[str, Any], *, corpus_scope: str) -> dict[str, Any]:
    """Record provenance required to supply the later current-capacity estimate.

    This intentionally does not invent a byte projection. A live preflight
    needs a current, separately measured capacity input and blocks if it is
    not present; preparation records the immutable workload/manifest
    provenance it must bind to.
    """

    manifest_path = workspace / "storage" / "storage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    families = manifest.get("artifact_families") if isinstance(manifest, dict) else None
    family_ids = sorted(
        str(item.get("id"))
        for item in families
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ) if isinstance(families, list) else []
    inventory_path = workspace / "analysis" / "analysis_manifest.json"
    inventory_digest = (
        canonical_json_digest(json.loads(inventory_path.read_text(encoding="utf-8")))
        if inventory_path.exists()
        else canonical_json_digest({"workspace": str(workspace.resolve()), "inventory": "not-yet-produced"})
    )
    return {
        "corpus_scope": corpus_scope,
        "inventory_digest": inventory_digest,
        "manifest_digest": canonical_json_digest(manifest),
        "workload_digest": canonical_json_digest(workload),
        "remaining_artifact_families": family_ids,
        "capacity_input_status": "required_before_live_preflight",
    }


def prepare_preflight_estimate(
    workspace: Path,
    run_id: str,
    *,
    settings: PreflightSettings,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> dict[str, Any]:
    """Build and atomically persist the canonical no-provider full-run estimate."""

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    workload = as_dict(collect_workspace_workload(resolved_workspace))
    estimate = as_dict(estimate_workspace(resolved_workspace, scope="full", mode="estimate"))
    projection, provider_models = provider_stage_projection(estimate, failover_providers=settings.failover_providers)
    snapshot = build_preflight_estimate(
        resolved_workspace,
        corpus_scope=settings.corpus_scope,
        paid_stages=tuple(sorted({key.split(":", 1)[1] for key in projection})),
        provider_stage_costs=projection,
        provider_models=provider_models,
        workload_collector=lambda _workspace: workload,
        estimator=lambda _workspace, *, scope, mode: estimate,
    )
    storage_inputs = storage_estimate_inputs(resolved_workspace, workload, corpus_scope=settings.corpus_scope)
    payload = {
        "schema_version": 1,
        "workspace": str(resolved_workspace),
        "run_id": safe_run_id,
        "scope": "full",
        "corpus_scope": settings.corpus_scope,
        "workload": snapshot.workload,
        "workload_digest": canonical_json_digest(snapshot.workload),
        "pricing_task_spec": snapshot.pricing_task_spec,
        "pricing_catalog_digest": canonical_json_digest(
            {"pricing_refreshed_at": snapshot.pricing_task_spec.get("pricing_refreshed_at")}
        ),
        "task_spec_digest": canonical_json_digest(snapshot.pricing_task_spec.get("tasks", [])),
        "paid_stages": list(snapshot.paid_stages),
        "provider_models": [asdict(model) for model in snapshot.provider_models],
        "provider_stage_projection": dict(snapshot.provider_stage_costs),
        "approved_headroom_usd": snapshot.approved_headroom_usd,
        "total_projected_cost_usd": round(sum(dict(snapshot.provider_stage_costs).values()), 6),
        "total_with_required_headroom_usd": round(
            sum(dict(snapshot.provider_stage_costs).values()) + snapshot.approved_headroom_usd,
            6,
        ),
        "storage_estimate_inputs": storage_inputs,
        "estimate_digest": snapshot.digest,
    }
    write_json(named_run_dir(resolved_workspace, safe_run_id) / PREPARED_ESTIMATE_FILENAME, payload)
    return payload


def _estimate_required() -> list[dict[str, Any]]:
    return [{"code": "canonical_estimate_required", "affected_stage": STAGE_ID}]


def load_prepared_estimate(
    path: Path,
    *,
    workspace: Path,
    run_id: str,
) -> tuple[CompleteRunEstimateSnapshot | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """Load only the atomic artifact produced by the no-provider command."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, _estimate_required()
    if not isinstance(payload, dict):
        return None, None, _estimate_required()
    required = {
        "workspace", "run_id", "scope", "corpus_scope", "workload", "pricing_task_spec",
        "paid_stages", "provider_models", "provider_stage_projection", "estimate_digest",
        "approved_headroom_usd",
        "storage_estimate_inputs",
    }
    if not required.issubset(payload) or payload.get("workspace") != str(workspace.resolve()) or payload.get("run_id") != run_id:
        return None, None, _estimate_required()
    try:
        provider_models = tuple(
            ProviderModelChoice(provider=str(item["provider"]), model=str(item["model"]))
            for item in payload["provider_models"]
            if isinstance(item, dict)
        )
        projection = payload["provider_stage_projection"]
        if not isinstance(projection, dict) or not all(isinstance(key, str) for key in projection):
            raise ValueError("invalid provider-stage projection")
        snapshot = build_complete_run_estimate_snapshot(
            workload=payload["workload"],
            estimate=payload["pricing_task_spec"],
            corpus_scope=str(payload["corpus_scope"]),
            paid_stages=tuple(str(stage) for stage in payload["paid_stages"]),
            provider_stage_costs={str(key): float(value) for key, value in projection.items()},
            provider_models=provider_models,
            approved_headroom_usd=float(payload["approved_headroom_usd"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None, _estimate_required()
    if payload.get("scope") != "full" or payload.get("estimate_digest") != snapshot.digest:
        return None, None, _estimate_required()
    return snapshot, payload, []


# --------------------------------------------------------------------------
# Capacity input
# --------------------------------------------------------------------------


def prepare_preflight_storage_estimate(
    workspace: Path,
    run_id: str,
    *,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> dict[str, Any]:
    """Build the named run's no-provider, policy-bound capacity input.

    The policy (``storage/preflight-storage-policy.json``) is a
    conference-owned, reviewed reservation for the remaining full-production
    artifact families. This binds that reservation to the prepared
    estimate's immutable corpus and manifest provenance; the live preflight
    still measures available bytes immediately before provider work.
    """

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    target = named_run_dir(resolved_workspace, safe_run_id)
    prepared_path = target / PREPARED_ESTIMATE_FILENAME
    snapshot, prepared, blockers = load_prepared_estimate(
        prepared_path,
        workspace=resolved_workspace,
        run_id=safe_run_id,
    )
    if blockers or snapshot is None or prepared is None:
        raise ValueError("a current canonical preflight estimate is required before preparing capacity")
    inputs = prepared.get("storage_estimate_inputs")
    if not isinstance(inputs, dict):
        raise ValueError("prepared estimate is missing storage-estimate provenance")

    policy_path = resolved_workspace / "storage" / CAPACITY_POLICY_FILENAME
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("storage capacity policy must be an object")
    families = policy.get("remaining_families")
    margin = policy.get("safety_margin_bytes")
    estimator_version = policy.get("estimator_version")
    if (
        policy.get("corpus_scope") != inputs.get("corpus_scope")
        or not isinstance(families, dict)
        or not families
        or any(not isinstance(name, str) or not name or not isinstance(value, int) or value <= 0 for name, value in families.items())
        or not isinstance(margin, int)
        or margin <= 0
        or not isinstance(estimator_version, str)
        or not estimator_version
    ):
        raise ValueError("storage capacity policy is incomplete or incompatible with the prepared estimate")

    payload = {
        "schema_version": 1,
        "run_id": safe_run_id,
        "workspace": str(resolved_workspace),
        "corpus_scope": inputs["corpus_scope"],
        "inventory_digest": inputs["inventory_digest"],
        "manifest_digest": inputs["manifest_digest"],
        "prepared_estimate_digest": snapshot.digest,
        "capacity_policy_digest": canonical_json_digest(policy),
        "remaining_families": dict(sorted(families.items())),
        "projected_remaining_bytes": sum(families.values()),
        "estimator_version": estimator_version,
        "safety_margin_bytes": margin,
    }
    write_json(target / CAPACITY_ESTIMATE_FILENAME, payload)
    return payload


def validate_capacity_input(
    path: Path,
    *,
    prepared: Mapping[str, Any],
    available_bytes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind current capacity to the immutable preparation provenance."""

    try:
        capacity_payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ready": False, "required_bytes": None}, [{"code": "storage_estimate_inputs_unavailable"}]
    inputs = prepared.get("storage_estimate_inputs")
    if not isinstance(capacity_payload, dict) or not isinstance(inputs, dict):
        return {"ready": False, "required_bytes": None}, [{"code": "storage_estimate_inputs_unavailable"}]
    if any(capacity_payload.get(field) != inputs.get(field) for field in ("corpus_scope", "inventory_digest", "manifest_digest")):
        return {"ready": False, "required_bytes": None}, [{"code": "storage_estimate_inputs_unavailable"}]
    if capacity_payload.get("prepared_estimate_digest") != prepared.get("estimate_digest"):
        return {"ready": False, "required_bytes": None}, [{"code": "storage_estimate_inputs_unavailable"}]
    result = validate_storage_estimate(path, available_bytes=available_bytes)
    return result, list(result.get("blockers", []))


# --------------------------------------------------------------------------
# Spend authority
# --------------------------------------------------------------------------


def load_approval(path: Path) -> tuple[ApprovalDeclaration | None, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("approval must be an object")
        models = tuple(
            ProviderModelChoice(provider=str(item["provider"]), model=str(item["model"]))
            for item in payload["provider_models"]
            if isinstance(item, dict)
        )
        allocations = tuple(
            ProviderStageAllocation(
                provider=str(item["provider"]),
                stage=str(item["stage"]),
                ceiling_usd=float(item["ceiling_usd"]),
            )
            for item in payload["provider_stage_allocations"]
            if isinstance(item, dict)
        )
        return ApprovalDeclaration(
            run_id=str(payload["run_id"]),
            corpus_scope=str(payload["corpus_scope"]),
            paid_stages=tuple(str(stage) for stage in payload["paid_stages"]),
            provider_models=models,
            estimate_digest=str(payload["estimate_digest"]),
            scope_fingerprint=str(payload["scope_fingerprint"]),
            total_ceiling_usd=float(payload["total_ceiling_usd"]),
            provider_stage_allocations=allocations,
        ), []
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, [{"code": "approval_required", "affected_stage": STAGE_ID}]


def authorization_metadata(approval: ApprovalDeclaration, *, mode: str) -> dict[str, Any]:
    """Return receipt-safe authority provenance without retaining approval input."""

    return {
        "mode": mode,
        "estimate_digest": approval.estimate_digest,
        "threshold_usd": (
            AUTOMATIC_AUTHORIZATION_THRESHOLD_USD if mode == AUTOMATIC_AUTHORITY_MODE else None
        ),
        "total_ceiling_usd": approval.total_ceiling_usd,
        "provider_stage_allocations": {
            allocation.key: allocation.ceiling_usd
            for allocation in sorted(approval.provider_stage_allocations, key=lambda item: item.key)
        },
    }


def resolve_authorization(
    *,
    approval_path: Path,
    snapshot: CompleteRunEstimateSnapshot | None,
    run_id: str,
) -> tuple[ApprovalDeclaration | None, dict[str, Any], list[dict[str, Any]]]:
    """Prefer a valid explicit declaration, otherwise derive the narrow auto authority."""

    approval, approval_blockers = load_approval(approval_path)
    blockers: list[dict[str, Any]] = []
    if approval is not None:
        if approval.run_id != run_id:
            blockers.append({"code": "approval_scope_mismatch", "affected_stage": STAGE_ID})
        if snapshot is not None:
            blockers.extend(validate_approval_scope(approval, snapshot))
        if not blockers:
            return approval, authorization_metadata(approval, mode=EXPLICIT_AUTHORITY_MODE), []

    automatic = derive_automatic_approval(snapshot, run_id=run_id)
    if automatic is not None:
        return automatic, authorization_metadata(automatic, mode=AUTOMATIC_AUTHORITY_MODE), []

    if approval is not None:
        return approval, authorization_metadata(approval, mode=EXPLICIT_AUTHORITY_MODE), blockers
    return None, {"mode": "approval_required"}, approval_blockers


# --------------------------------------------------------------------------
# Probes and environment
# --------------------------------------------------------------------------


def safe_probe_error(error_type: str | None) -> str | None:
    """Collapse runtime failures to a fixed receipt-safe error class."""

    return error_type if error_type in SAFE_PROBE_ERRORS else ("unknown" if error_type else None)


def safe_probe_result(result: ProviderProbeResult) -> ProviderProbeResult:
    return ProviderProbeResult(
        provider=result.provider,
        stage=result.stage,
        status=result.status,
        error_type=safe_probe_error(result.error_type),
        retryable=result.retryable,
        attempt_count=result.attempt_count,
    )


def canonical_env_file(ctx: ProductionRunnerContext, settings: PreflightSettings) -> Path:
    return settings.canonical_env_file or ctx.repo_root / ".env"


def reload_environment_from(env_file: Path) -> None:
    """Reload only the operator-designated canonical environment file, without logging values."""

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            os.environ[key] = value.strip().strip('"').strip("'")


def default_probe(spec: ProviderProbeSpec, *, openreview_auth_label: str) -> ProviderProbeResult:
    """Use only the declared non-mutating operation when a later live run is authorized."""

    try:
        if spec.provider == "openai":
            from openai import OpenAI

            OpenAI().models.list()
        elif spec.provider == "anthropic":
            from anthropic import Anthropic

            Anthropic().models.list()
        elif spec.provider == "google":
            from google import genai

            # Keep the client alive while the SDK constructs and dispatches its
            # pager-backed models request.  Calling through a temporary client
            # can close it before the pager sends the request.
            client = genai.Client()
            client.models.list()
        elif spec.provider == "tavily":
            from urllib.request import Request, urlopen

            request = Request("https://api.tavily.com/usage", headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}"})
            with urlopen(request, timeout=10) as response:  # nosec B310 - approved fixed provider endpoint
                response.read(1)
        elif spec.provider == "openrouter":
            from urllib.request import Request, urlopen

            request = Request("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            with urlopen(request, timeout=10) as response:  # nosec B310 - approved fixed provider endpoint
                response.read(1)
        elif spec.provider == "openreview":
            from conference_pipeline.openreview_auth import load_openreview_auth

            load_openreview_auth(openreview_auth_label, mode="required")
        else:  # pragma: no cover - all specs are declared above
            return ProviderProbeResult.failed(spec, error_type="client_unavailable")
    except ImportError:
        return ProviderProbeResult.failed(spec, error_type="client_unavailable")
    except TimeoutError:
        return ProviderProbeResult.failed(spec, error_type="timeout", retryable=True)
    except Exception:
        return ProviderProbeResult.failed(spec, error_type="unknown", retryable=True)
    return ProviderProbeResult.succeeded(spec)


def declared_probe_specs(ctx: ProductionRunnerContext, hooks: ConferenceHooks = NO_HOOKS) -> tuple[ProviderProbeSpec, ...]:
    """The ``provider_probes`` hook's specs, else every planned provider probe."""

    if hooks.provider_probes is None:
        return PLANNED_PROVIDER_PROBES
    return tuple(hooks.provider_probes(ctx))


def scoped_probe_specs(
    specs: Sequence[ProviderProbeSpec], provider_probe_scope: tuple[str, str] | None
) -> tuple[tuple[ProviderProbeSpec, ...], list[dict[str, Any]]]:
    """One provider's probe re-scoped to a paid stage, or every declared probe."""

    if provider_probe_scope is None:
        return tuple(specs), []
    provider, scoped_stage = provider_probe_scope
    matches = [spec for spec in specs if spec.provider == provider]
    if not isinstance(provider, str) or not provider or not isinstance(scoped_stage, str) or not scoped_stage or len(matches) != 1:
        return (), [{"code": "provider_probe_scope_incompatible", "affected_stage": STAGE_ID}]
    selected = matches[0]
    return (
        ProviderProbeSpec(
            provider=selected.provider,
            stage=scoped_stage,
            credential_names=selected.credential_names,
            operation=selected.operation,
            classification=selected.classification,
            outcome_schema=selected.outcome_schema,
        ),
    ), []


# --------------------------------------------------------------------------
# Owner binding
# --------------------------------------------------------------------------


def owner_binding_blocker(settings: PreflightSettings, owner_id: str, *, invalid_fields: list[str]) -> dict[str, Any]:
    """Return the receipt-safe root cause for runner ownership failures."""

    return {
        "code": settings.owner_binding_code,
        "affected_stage": STAGE_ID,
        "expected_owner_id": owner_id,
        "invalid_fields": invalid_fields,
    }


def preflight_owner_id(settings: PreflightSettings, registry_path: Path | None = None) -> str:
    """The configured owner id, else the registry's required id for the stage."""

    if settings.owner_id:
        return settings.owner_id
    try:
        required = required_script_ids_for_stage(load_stage_script_registry(registry_path), STAGE_ID)
    except (OSError, ValueError, json.JSONDecodeError):
        required = ()
    return str(required[0]) if required else STAGE_ID


def resolve_owner(
    settings: PreflightSettings, registry_path: Path | None
) -> tuple[str, dict[str, str], str | None, list[dict[str, Any]]]:
    """Resolve the preflight owner before receipt evidence is trusted.

    Returns ``(owner_id, registration, registry_sha256, blockers)``.
    """

    owner_id = preflight_owner_id(settings, registry_path)
    try:
        registry = load_stage_script_registry(registry_path)
        digest = registry_sha256(registry_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return owner_id, {}, None, [owner_binding_blocker(settings, owner_id, invalid_fields=["registry"])]

    scripts = registry.get("scripts")
    owner = scripts.get(owner_id) if isinstance(scripts, Mapping) else None
    if not isinstance(owner, Mapping):
        return owner_id, {}, digest, [owner_binding_blocker(settings, owner_id, invalid_fields=["owner"])]

    expected = settings.owner_registration
    fields = tuple(expected) if expected is not None else OWNER_REGISTRATION_FIELDS
    registration = {name: str(owner.get(name) or "") for name in fields}
    if expected is None:
        return owner_id, registration, digest, []
    invalid_fields = [name for name, value in expected.items() if registration[name] != value]
    if invalid_fields:
        return owner_id, registration, digest, [owner_binding_blocker(settings, owner_id, invalid_fields=invalid_fields)]
    return owner_id, registration, digest, []


def owner_execution_evidence(ctx: ProductionRunnerContext, owner_id: str) -> dict[str, object]:
    """Record execution-derived owner evidence without retaining secret inputs."""

    return {
        owner_id: {
            "runner_path": ctx.runner_path,
            "runner_function": "run_runner_preflight",
            "stage_receipt_path": str(ctx.receipt_path(STAGE_ID)),
        }
    }


# --------------------------------------------------------------------------
# Evidence and the stage
# --------------------------------------------------------------------------


def write_preflight_evidence(
    ctx: ProductionRunnerContext,
    *,
    settings: PreflightSettings,
    owner_id: str,
    status: str,
    checks: dict[str, Any],
    blockers: list[dict[str, Any]],
    script_registry_sha256: str | None,
    owner_registration: Mapping[str, str],
    owner_evidence: Mapping[str, object],
    stage_contract: Mapping[str, Any] | None,
    acknowledgement_ids: list[str] | None = None,
    provider_probe_scope: tuple[str, str] | None = None,
    lineage: Mapping[str, Any] | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    """Persist sanitized stage, summary, blocker, and remediation evidence atomically.

    The stage receipt is finalized before it is written: ``lineage`` is the
    one the stage contract was verified with (bound afresh when absent). A
    provider-scoped recheck writes its evidence under ``volatile-preflight/``
    and leaves the stage receipt and run summary alone.
    """

    stage = ctx.stage(STAGE_ID)
    target = ctx.run_dir()
    label = settings.label
    remediation = build_remediation_report(blockers, label=label)
    binding_code = settings.owner_binding_code
    if any(blocker.get("code") == binding_code for blocker in blockers):
        owner_remediation = f"repair the {label} runner registration/evidence and rerun preflight"
        clearance = f"current_{label}_registration_and_strict_execution_receipt"
        for entry in remediation["json"]:
            if entry.get("code") == binding_code:
                entry["remediation"] = owner_remediation
                entry["clearance_evidence"] = clearance
        remediation["markdown"] += (
            f"- `{binding_code}` ({STAGE_ID}): "
            f"{owner_remediation}. Clearance: `{clearance}`.\n"
        )
        remediation["console"] += f"\n- {binding_code} ({STAGE_ID}): {owner_remediation}"
    report = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "stage_id": stage.id,
        "status": status,
        "blockers": remediation["json"],
        "clearance_requires": "fresh_passing_recheck",
        "acknowledgement_ids": list(acknowledgement_ids or []),
    }
    if provider_probe_scope is None:
        remediation_json_path = target / REMEDIATION_JSON_FILENAME
        remediation_markdown_path = target / REMEDIATION_MARKDOWN_FILENAME
        receipt_path = target / stage.receipt_filename
    else:
        provider, scoped_stage = provider_probe_scope
        scope_label = f"{scoped_stage}-{provider}".replace("/", "-")
        evidence_dir = target / VOLATILE_PREFLIGHT_DIR
        remediation_json_path = evidence_dir / f"{scope_label}.remediation.json"
        remediation_markdown_path = evidence_dir / f"{scope_label}.remediation.md"
        receipt_path = evidence_dir / f"{scope_label}.{io.now().replace(':', '').replace('-', '')}.json"
    io.write_json(remediation_json_path, sanitize_for_receipt(report))
    remediation_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_markdown = remediation_markdown_path.with_name(f".{remediation_markdown_path.name}.{os.getpid()}.tmp")
    # Keep generated evidence clean under the repository's whitespace policy
    # while preserving one terminal newline for Markdown readers.
    temporary_markdown.write_text(str(remediation["markdown"]).rstrip() + "\n", encoding="utf-8")
    os.replace(temporary_markdown, remediation_markdown_path)

    blocker_path: Path | None = None
    if status == "blocked":
        blocker_path = write_blocker_receipt(
            BlockerReceipt(
                schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
                receipt_type=BLOCKER_RECEIPT_MARKER,
                blocker_class=BlockerClass.STAGE_BLOCKED,
                stage_id=stage.id,
                reason=f"{label} runner preflight has unresolved secret-free readiness blockers.",
                remediation_path="Reprepare/reapprove as reported, then run a fresh live preflight.",
                automated_unblock_available=False,
                endpoint_label=None,
                timestamp=io.now(),
                run_id=ctx.run_id,
            ),
            ctx.workspace,
        )
    receipt = {
        "schema_version": 1,
        "receipt_type": ctx.receipt_type,
        "receipt_id": f"{ctx.run_id}:{stage.id}:{io.now()}",
        "run_id": ctx.run_id,
        "workspace": str(ctx.workspace),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "mode": "live",
        "status": status,
        "resumable": status == "succeeded",
        "preflight_resume_policy": "rerun_required",
        "provider_probe_scope": (
            {"provider": provider_probe_scope[0], "stage": provider_probe_scope[1]}
            if provider_probe_scope is not None
            else None
        ),
        "script_registry_sha256": script_registry_sha256,
        "required_script_ids": [owner_id],
        "owner_registration": dict(owner_registration),
        "owner_execution_evidence": sanitize_for_receipt(owner_evidence),
        "stage_contract": sanitize_for_receipt(stage_contract) if stage_contract else None,
        "credential_value_marker_persisted": False,
        "checks": sanitize_for_receipt(checks),
        "blocker_receipt_path": str(blocker_path) if blocker_path else None,
        "remediation_report_path": str(remediation_json_path),
        "secrets_copied": False,
    }
    receipt = ctx.finalize(receipt, stage_id=stage.id, lineage=lineage, bind=lineage is None)
    io.write_json(receipt_path, receipt)
    summary = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "workspace": str(ctx.workspace),
        "mode": "live",
        "status": status,
        "halt_status": "halted" if status == "blocked" else None,
        "halt_stage_id": stage.id if status == "blocked" else None,
        "blocker_receipt_path": str(blocker_path) if blocker_path else None,
        "unblock_acknowledgements": list(acknowledgement_ids or []),
        "stages": [receipt],
        "summary_path": str(target / "summary.json"),
    }
    if provider_probe_scope is None:
        io.write_json(target / "summary.json", summary)
    return {
        "status": status,
        "receipt": receipt,
        "summary": summary,
        "console": str(remediation["console"]),
        "remediation_report_path": str(remediation_json_path),
    }


@dataclass(frozen=True)
class PreflightOwners:
    """Callables the stage runs, bound by the runner at call time.

    ``owner_evidence(owner_id)`` returns the execution evidence map (default
    :func:`owner_execution_evidence`); ``probe`` runs one declared probe
    (default :func:`default_probe`); ``reload_environment`` reloads the
    canonical environment file once before the credential inventory and
    again on a probe recovery (default: the settings' canonical env file).
    """

    owner_evidence: Callable[[str], Mapping[str, object]] | None = None
    probe: ProbeFn | None = None
    reload_environment: Callable[[], None] | None = None
    io: StageIO = field(default_factory=lambda: DEFAULT_STAGE_IO)


def run_runner_preflight(
    ctx: ProductionRunnerContext,
    *,
    settings: PreflightSettings,
    prepared_estimate_path: Path,
    approval_path: Path,
    capacity_estimate_path: Path,
    credential_values: Mapping[str, object] | None = None,
    probe: ProbeFn | None = None,
    reload_environment: Callable[[], None] | None = None,
    rebuild_client: Callable[[ProviderProbeSpec], None] | None = None,
    available_bytes: int | None = None,
    acknowledgement_ids: list[str] | None = None,
    stage_registry_path: Path | None = None,
    provider_probe_scope: tuple[str, str] | None = None,
    hooks: ConferenceHooks = NO_HOOKS,
    owners: PreflightOwners | None = None,
) -> dict[str, Any]:
    """Evaluate all safe readiness faults after immutable authority checks pass.

    Returns ``{status, receipt, summary, console, remediation_report_path}``.
    """

    owners = owners or PreflightOwners()
    io = owners.io
    workspace = ctx.workspace
    run_id = ctx.run_id
    prepared_estimate_path = require_named_run_input(prepared_estimate_path, workspace=workspace, run_id=run_id)
    approval_path = require_named_run_input(approval_path, workspace=workspace, run_id=run_id)
    capacity_estimate_path = require_named_run_input(capacity_estimate_path, workspace=workspace, run_id=run_id)
    snapshot, prepared, blockers = load_prepared_estimate(prepared_estimate_path, workspace=workspace, run_id=run_id)
    approval, authorization, authorization_blockers = resolve_authorization(
        approval_path=approval_path,
        snapshot=snapshot,
        run_id=run_id,
    )
    blockers.extend(authorization_blockers)
    if available_bytes is None:
        try:
            available_bytes = shutil.disk_usage(workspace).free
        except OSError:
            blockers.append({"code": "storage_estimate_inputs_unavailable", "affected_stage": STAGE_ID})
    if prepared is not None and available_bytes is not None:
        capacity, capacity_blockers = validate_capacity_input(
            capacity_estimate_path, prepared=prepared, available_bytes=available_bytes
        )
        blockers.extend(capacity_blockers)
    else:
        capacity = {"ready": False, "required_bytes": None}
        blockers.append({"code": "storage_estimate_inputs_unavailable", "affected_stage": STAGE_ID})

    owner_id, owner_registration, registry_digest, owner_blockers = resolve_owner(settings, stage_registry_path)
    evidence_fn = owners.owner_evidence or (lambda owner: owner_execution_evidence(ctx, owner))
    owner_evidence = evidence_fn(owner_id)
    blockers.extend(owner_blockers)
    for extra_check in hooks.extra_preflight_checks:
        blockers.extend(dict(blocker) for blocker in extra_check(ctx))
    probe_specs, scope_blockers = scoped_probe_specs(declared_probe_specs(ctx, hooks), provider_probe_scope)
    blockers.extend(scope_blockers)

    checks: dict[str, Any] = {
        "prepared_estimate": {
            "path": str(prepared_estimate_path),
            "status": "valid" if snapshot is not None else "invalid",
            "estimate_digest": prepared.get("estimate_digest") if prepared else None,
            "provider_stage_projection": prepared.get("provider_stage_projection") if prepared else {},
        },
        "approval": {
            "path": str(approval_path),
            "status": "not_required" if authorization["mode"] == AUTOMATIC_AUTHORITY_MODE else (
                "valid" if approval and not authorization_blockers else "invalid"
            ),
        },
        "spend_authority": authorization,
        "storage_capacity": capacity,
        "owner_binding": {
            "owner_id": owner_id,
            "registration": owner_registration,
            "registry_sha256": registry_digest,
            "status": "valid" if not owner_blockers else "invalid",
        },
        "provider_probes": [],
        "provider_probe_scope": (
            {"provider": provider_probe_scope[0], "stage": provider_probe_scope[1]}
            if provider_probe_scope is not None
            else "all_declared_providers"
        ),
        "credential_value_marker_persisted": False,
        "secrets_copied": False,
    }
    evidence_kwargs: dict[str, Any] = {
        "settings": settings,
        "owner_id": owner_id,
        "script_registry_sha256": registry_digest,
        "owner_registration": owner_registration,
        "owner_evidence": owner_evidence,
        "acknowledgement_ids": acknowledgement_ids,
        "provider_probe_scope": provider_probe_scope,
        "io": io,
    }
    if blockers:
        outcome = write_preflight_evidence(
            ctx, status="blocked", checks=checks, blockers=blockers, stage_contract=None, **evidence_kwargs
        )
        outcome["console"] += "\n- reprepare/reapprove before any environment reload or provider probe."
        return outcome

    reload = (
        reload_environment
        or owners.reload_environment
        or (lambda: reload_environment_from(canonical_env_file(ctx, settings)))
    )
    if credential_values is None:
        reload()
        credential_values = dict(os.environ)
    inventory_blockers = validate_credential_inventory(probe_specs, credential_values)
    spec_blockers = validate_provider_probe_specs(probe_specs)
    blockers.extend(inventory_blockers)
    blockers.extend(spec_blockers)
    checks["credential_inventory"] = [
        {"name": name, "present": bool(credential_values.get(name)), "value_recorded": False}
        for spec in probe_specs
        for name in spec.credential_names
    ]
    auth_label = settings.openreview_auth_label or f"{ctx.slug}-production-preflight"
    probe_fn = probe or owners.probe or (lambda spec: default_probe(spec, openreview_auth_label=auth_label))
    rebuild = rebuild_client or (lambda _spec: None)
    missing_by_provider = {str(blocker.get("affected_provider")) for blocker in inventory_blockers}
    for spec in probe_specs:
        if spec.provider in missing_by_provider:
            continue
        result = execute_probe_with_single_recovery(
            spec,
            probe_fn,
            reload_environment=reload,
            rebuild_client=lambda spec=spec: rebuild(spec),
        )
        safe_result = safe_probe_result(result)
        checks["provider_probes"].append(
            {
                "provider": safe_result.provider,
                "stage": safe_result.stage,
                "operation": spec.operation,
                "classification": spec.classification,
                "status": safe_result.status,
                "attempt_count": safe_result.attempt_count,
                "rate_guidance": "honor_provider_retry_guidance",
                "error_type": safe_result.error_type,
            }
        )
        blockers.extend(evaluate_probe_result(safe_result))
    stage_contract: dict[str, Any] | None = None
    preflight_lineage: Mapping[str, Any] | None = None
    if not blockers:
        provisional_receipt: dict[str, Any] = {
            "mode": "live",
            "script_registry_sha256": registry_digest,
            "required_script_ids": [owner_id],
            "owner_execution_evidence": owner_evidence,
        }
        # Bind once and persist the same lineage the contract was verified
        # with: the written receipt must carry it, not only the verified copy.
        preflight_lineage = ctx.finalize(dict(provisional_receipt), stage_id=STAGE_ID, bind=True).get("lineage")
        contract = ctx.verify_contract(
            bind=False,
            lineage=preflight_lineage,
            stage_id=STAGE_ID,
            receipt=provisional_receipt,
            registry_path=stage_registry_path,
            required_owner_ids=(owner_id,),
            mode="live",
            strict_execution=True,
        )
        stage_contract = contract.to_dict()
        checks["stage_contract"] = stage_contract
        if contract.status != "pass":
            blockers.append(owner_binding_blocker(settings, owner_id, invalid_fields=["strict_execution_evidence"]))
    status = "blocked" if blockers else "succeeded"
    return write_preflight_evidence(
        ctx,
        status=status,
        checks=checks,
        blockers=blockers,
        stage_contract=stage_contract,
        lineage=preflight_lineage,
        **evidence_kwargs,
    )


def stage_run(outcome: Mapping[str, Any]) -> Any:
    """The preflight outcome for the shared orchestrator: the remediation console on a block, silence on success."""

    from conference_pipeline.production.orchestrator import StageRun

    return StageRun.from_outcome(
        outcome,
        success_message="",
        failure_message=str(outcome.get("console") or "") if outcome["status"] == "blocked" else "",
    )


# --------------------------------------------------------------------------
# Paid-call dispatch
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskAuthority:
    """A task-specific spend boundary for :func:`dispatch_paid_call`.

    ``refusal`` (when set) is returned as the dispatch result before any
    budget check; otherwise ``snapshot`` and ``approval`` replace the run's
    generic estimate and authority.
    """

    snapshot: CompleteRunEstimateSnapshot | None
    approval: ApprovalDeclaration | None
    refusal: dict[str, Any] | None = None


def dispatch_paid_call(
    *,
    workspace: Path,
    run_id: str,
    prepared_estimate_path: Path,
    approval_path: Path,
    provider: str,
    stage: str,
    projected_cost_usd: float | None,
    spent_total_usd: float,
    spent_by_provider_stage: Mapping[str, float],
    owner: Callable[[], float],
    task_id: str = "production.dispatch",
    task_authority: Callable[[list[dict[str, Any]]], TaskAuthority] | None = None,
) -> dict[str, Any]:
    """Run both generic and named approval guards before a paid owner executes.

    ``task_authority(estimate_blockers)`` supplies a task-specific boundary
    (a source-discovery join frontier, ...) in place of the run's approval.
    """

    safe_run_id = validate_run_id(run_id)
    resolved_workspace = workspace.resolve()
    prepared_estimate_path = require_named_run_input(
        prepared_estimate_path, workspace=resolved_workspace, run_id=safe_run_id
    )
    approval_path = require_named_run_input(approval_path, workspace=resolved_workspace, run_id=safe_run_id)
    snapshot, _prepared, blockers = load_prepared_estimate(
        prepared_estimate_path, workspace=resolved_workspace, run_id=safe_run_id
    )
    approval: ApprovalDeclaration | None
    if task_authority is not None:
        authority = task_authority(blockers)
        if authority.refusal is not None:
            return authority.refusal
        snapshot, approval = authority.snapshot, authority.approval
    else:
        approval, _authorization, authorization_blockers = resolve_authorization(
            approval_path=approval_path,
            snapshot=snapshot,
            run_id=safe_run_id,
        )
        blockers.extend(authorization_blockers)
    if blockers or snapshot is None:
        return {"authorized": False, "blockers": blockers or [{"code": "canonical_estimate_required"}]}
    preflight = enforce_paid_call_preflight(
        resolved_workspace,
        task_id,
        approval,
        snapshot,
        provider=provider,
        stage=stage,
        projected_cost_usd=projected_cost_usd,
        spent_total_usd=spent_total_usd,
        spent_by_provider_stage=spent_by_provider_stage,
    )
    if not preflight["authorized"]:
        return preflight
    actual_cost = owner()
    postcall = enforce_paid_call_postcall(
        resolved_workspace,
        safe_run_id,
        approval,
        snapshot,
        provider=provider,
        stage=stage,
        actual_cost_usd=actual_cost,
        spent_by_provider_stage=spent_by_provider_stage,
    )
    return {"authorized": bool(postcall["authorized"]), "blockers": list(postcall["blockers"]), "actual_cost_usd": actual_cost}



# --------------------------------------------------------------------------
# Command line: the no-provider preparation commands and authority paths
# --------------------------------------------------------------------------

PREPARE_COMMAND_FORBIDDEN_FLAGS: tuple[str, ...] = (
    "--stage",
    "--mode",
    "--resume",
    "--summary",
    "--approval-file",
    "--prepared-estimate-path",
    "--capacity-estimate-path",
)


def add_preflight_cli_args(parser: argparse.ArgumentParser) -> None:
    """The preparation commands and the three authority-path overrides."""

    parser.add_argument(
        "--prepare-preflight-estimate",
        action="store_true",
        help="Write the canonical no-provider full-run estimate for --run-id, then exit.",
    )
    parser.add_argument(
        "--prepare-storage-estimate",
        action="store_true",
        help="Bind storage/preflight-storage-policy.json to the prepared estimate for --run-id, then exit.",
    )
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--prepared-estimate-path", type=Path)
    parser.add_argument("--capacity-estimate-path", type=Path)


def validate_prepare_command(parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: Sequence[str]) -> None:
    present = sorted(set(PREPARE_COMMAND_FORBIDDEN_FLAGS).intersection(raw_argv))
    command = "--prepare-preflight-estimate" if args.prepare_preflight_estimate else "--prepare-storage-estimate"
    if present:
        parser.error(command + " cannot be combined with " + ", ".join(present))
    if args.prepare_preflight_estimate and args.prepare_storage_estimate:
        parser.error("preparation commands are mutually exclusive")


def preflight_authority_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """The prepared-estimate, approval and capacity paths: overrides, else the run's defaults."""

    prepared, approval, capacity = authority_paths(args.workspace, args.run_id)
    return (
        getattr(args, "prepared_estimate_path", None) or prepared,
        getattr(args, "approval_file", None) or approval,
        getattr(args, "capacity_estimate_path", None) or capacity,
    )


def preflight_extra_commands(
    *,
    prepare_estimate: Callable[[Path, str], Any],
    prepare_storage_estimate: Callable[[Path, str], Any],
    label: str,
) -> dict[str, Any]:
    """``ConferenceHooks.extra_commands`` entries for the two preparation commands."""

    from conference_pipeline.production.hooks import ExtraCommand

    return {
        "prepare_preflight_estimate": ExtraCommand(
            run=lambda args: prepare_estimate(args.workspace, args.run_id),
            failure_label=f"cannot prepare {label} preflight estimate",
            validate=validate_prepare_command,
        ),
        "prepare_storage_estimate": ExtraCommand(
            run=lambda args: prepare_storage_estimate(args.workspace, args.run_id),
            failure_label=f"cannot prepare {label} storage estimate",
            validate=validate_prepare_command,
        ),
    }
