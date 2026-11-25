"""Typed access to the model-governance task registry."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


class ModelGovernanceError(ValueError):
    """Raised when a provider/model selection violates registry policy."""


@dataclass(frozen=True)
class ModelSelection:
    """A concrete provider/model pair allowed for a governed task."""

    provider: str
    model: str
    role: str = "candidate"


@dataclass(frozen=True)
class LocalCandidateRecord:
    """A local/open-weight model candidate for a governed task."""

    model_id: str | None
    source_url: str | None
    checked_date: str
    source_class: str | None
    runtime_support: str
    license: str | None
    expected_memory_gb: float | None
    benchmark_receipt_ref: str | None
    status: str
    rejection_reason: str | None = None
    runtime_kind: str | None = None
    parameter_count: str | None = None
    quantization: str | None = None
    multimodal: bool | None = None
    recommended_um_floor_gb: float | None = None


@dataclass(frozen=True)
class TaskPolicy:
    """Registry policy for one LLM or model-backed task family."""

    task_id: str
    task_family: str
    stage: str
    provider: str
    model: str
    fallback_provider: str | None
    fallback_model: str | None
    budget_cap_usd: float | None
    current_research: dict[str, Any]
    benchmark: dict[str, Any]
    quality_threshold: str
    schema_validity_required: bool
    provenance_required: bool
    confidence_threshold: float | None
    local_runtime_policy: str
    local_runtime_profile_id: str | None
    selection_status: str
    allowed_selections: tuple[ModelSelection, ...]
    local_candidates: tuple[LocalCandidateRecord, ...]
    local_proof: dict = field(default_factory=dict)
    # D-08: per-task override for the global local-first switch.
    # When absent in the registry (None), the global switch governs.
    # When explicitly False, this task always routes to paid — even if
    # an approved local selection exists (quality-critical tasks).
    local_first_enabled: bool = True
    # Optional reasoning-effort contract for the task's default model
    # (e.g. gpt-5.6-sol tasks must run with reasoning_effort: high).
    # None keeps current provider behavior.
    reasoning_effort: str | None = None

    @property
    def default_selection(self) -> ModelSelection:
        """Return the registered default provider/model pair."""
        return ModelSelection(provider=self.provider, model=self.model, role="default")

    @property
    def fallback_selection(self) -> ModelSelection | None:
        """Return the registered fallback provider/model pair, when present."""
        if not self.fallback_provider or not self.fallback_model:
            return None
        return ModelSelection(
            provider=self.fallback_provider,
            model=self.fallback_model,
            role="fallback",
        )


class ModelRegistry:
    """Validated task/model registry loaded from YAML."""

    def __init__(
        self,
        tasks: dict[str, TaskPolicy],
        *,
        schema_version: int,
        source_path: Path | None = None,
    ):
        self.tasks = tasks
        self.schema_version = schema_version
        self.source_path = source_path

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        source_path: Path | None = None,
    ) -> "ModelRegistry":
        """Build a registry from raw YAML data."""
        schema_version = int(payload.get("schema_version", 0))
        if schema_version < 1:
            raise ModelGovernanceError("model registry schema_version must be >= 1")

        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, dict) or not raw_tasks:
            raise ModelGovernanceError("model registry must define tasks")

        tasks: dict[str, TaskPolicy] = {}
        for task_id, raw_policy in raw_tasks.items():
            if not isinstance(raw_policy, dict):
                raise ModelGovernanceError(f"{task_id}: policy must be a mapping")
            tasks[task_id] = _parse_task_policy(task_id, raw_policy, source_path=source_path)
        return cls(tasks, schema_version=schema_version, source_path=source_path)

    def get(self, task_id: str) -> TaskPolicy:
        """Return a task policy by id."""
        try:
            return self.tasks[task_id]
        except KeyError as exc:
            raise ModelGovernanceError(f"unknown model-governance task id: {task_id}") from exc

    def candidates_for_task(self, task_id: str) -> tuple[LocalCandidateRecord, ...]:
        """Return local candidates registered for a task."""
        return self.get(task_id).local_candidates

    def validate(
        self,
        task_id: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> ModelSelection:
        """Validate or return the selected provider/model pair for a task."""
        policy = self.get(task_id)
        selected = ModelSelection(
            provider=provider or policy.provider,
            model=model or policy.model,
            role="selected",
        )
        allowed = {(item.provider, item.model) for item in policy.allowed_selections}
        allowed.add((policy.provider, policy.model))
        if policy.fallback_provider and policy.fallback_model:
            allowed.add((policy.fallback_provider, policy.fallback_model))
        if (selected.provider, selected.model) not in allowed:
            allowed_text = ", ".join(f"{item[0]}/{item[1]}" for item in sorted(allowed))
            raise ModelGovernanceError(
                f"{task_id} selection {selected.provider}/{selected.model} is not registered; "
                f"allowed selections: {allowed_text}"
            )
        return selected


def _parse_task_policy(
    task_id: str,
    raw: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> TaskPolicy:
    required = [
        "task_family",
        "stage",
        "current_research",
        "benchmark",
        "quality_threshold",
        "schema_validity_required",
        "provenance_required",
        "local_runtime_policy",
        "selection_status",
        "allowed_selections",
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ModelGovernanceError(f"{task_id}: missing required policy fields: {missing}")

    provider, model = _resolve_task_model(task_id, raw, source_path=source_path)
    fallback_provider, fallback_model = _resolve_task_fallback(task_id, raw, source_path=source_path)

    allowed_raw = raw["allowed_selections"]
    if not isinstance(allowed_raw, list) or not allowed_raw:
        raise ModelGovernanceError(f"{task_id}: allowed_selections must be a non-empty list")
    allowed = tuple(
        ModelSelection(
            provider=str(item["provider"]),
            model=str(item["model"]),
            role=str(item.get("role", "candidate")),
        )
        for item in allowed_raw
    )
    _validate_selections_against_catalog(task_id, "allowed_selections", allowed, source_path=source_path)

    benchmark = dict(raw["benchmark"])
    _validate_benchmark_policy(task_id, raw, provider, model, fallback_provider, fallback_model, allowed, benchmark)

    local_runtime_profile_id = _optional_str(raw.get("local_runtime_profile_id"))
    _validate_local_runtime_profile_id(task_id, local_runtime_profile_id)

    # D-08: per-task local-first override; absent/None → default True (global governs)
    raw_lfe = raw.get("local_first_enabled")
    local_first_enabled: bool = bool(raw_lfe) if raw_lfe is not None else True

    return TaskPolicy(
        task_id=task_id,
        task_family=str(raw["task_family"]),
        stage=str(raw["stage"]),
        provider=provider,
        model=model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        budget_cap_usd=_optional_float(raw.get("budget_cap_usd")),
        current_research=dict(raw["current_research"]),
        benchmark=benchmark,
        quality_threshold=str(raw["quality_threshold"]),
        schema_validity_required=bool(raw["schema_validity_required"]),
        provenance_required=bool(raw["provenance_required"]),
        confidence_threshold=_optional_float(raw.get("confidence_threshold")),
        local_runtime_policy=str(raw["local_runtime_policy"]),
        local_runtime_profile_id=local_runtime_profile_id,
        selection_status=str(raw["selection_status"]),
        allowed_selections=allowed,
        local_candidates=_parse_local_candidates(task_id, raw.get("local_candidates") or []),
        local_proof=dict(raw.get("local_proof") or {}),
        local_first_enabled=local_first_enabled,
        reasoning_effort=_optional_str(raw.get("reasoning_effort")),
    )


def default_catalog_path() -> Path:
    """Return the installed canonical model-catalog path (models.yaml)."""
    return Path(str(resources.files(__package__).joinpath("models.yaml")))


def _resolve_catalog_path(source_path: Path | None) -> Path:
    """Resolve which models.yaml governs a given registry.yaml.

    A registry loaded from a custom path (e.g. a test-authored tmp_path
    registry) may sit next to its own sibling ``models.yaml`` defining
    test-only aliases — that sibling wins when present, so ad-hoc test
    fixtures stay hermetic instead of polluting or depending on the real
    production catalog. Falls back to the canonical installed catalog
    otherwise (the normal production case).
    """
    if source_path is not None:
        sibling = Path(source_path).parent / "models.yaml"
        if sibling.exists():
            return sibling
    return default_catalog_path()


@lru_cache(maxsize=8)
def load_model_catalog(catalog_path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load a model catalog from ``catalog_path`` (default: the canonical one)."""
    path = Path(catalog_path) if catalog_path else default_catalog_path()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), dict):
        raise ModelGovernanceError(f"model catalog is not a mapping: {path}")
    return payload["models"]


def _catalog_entry(task_id: str, alias: str, *, field: str, source_path: Path | None) -> dict[str, Any]:
    catalog_path = _resolve_catalog_path(source_path)
    entry = load_model_catalog(catalog_path).get(alias)
    if entry is None:
        raise ModelGovernanceError(
            f"{task_id}: {field} references unknown model-catalog alias {alias!r}; "
            f"add it to {catalog_path} or fix the typo"
        )
    if entry.get("status") == "retired":
        raise ModelGovernanceError(
            f"{task_id}: {field} references retired model-catalog alias {alias!r} "
            f"({entry.get('provider')}/{entry.get('model')}); a retired model must not "
            "be (re)assigned to a task"
        )
    return entry


def _resolve_task_model(
    task_id: str, raw: dict[str, Any], *, source_path: Path | None
) -> tuple[str, str]:
    alias = raw.get("model_ref")
    if not alias:
        raise ModelGovernanceError(f"{task_id}: missing required policy field: 'model_ref'")
    entry = _catalog_entry(task_id, str(alias), field="model_ref", source_path=source_path)
    return str(entry["provider"]), str(entry["model"])


def _resolve_task_fallback(
    task_id: str, raw: dict[str, Any], *, source_path: Path | None
) -> tuple[str | None, str | None]:
    alias = _optional_str(raw.get("fallback_model_ref"))
    if alias is None:
        return None, None
    entry = _catalog_entry(task_id, alias, field="fallback_model_ref", source_path=source_path)
    return str(entry["provider"]), str(entry["model"])


def _validate_selections_against_catalog(
    task_id: str,
    field: str,
    selections: tuple[ModelSelection, ...],
    *,
    source_path: Path | None,
) -> None:
    catalog = load_model_catalog(_resolve_catalog_path(source_path))
    catalog_pairs = {(str(v["provider"]), str(v["model"])) for v in catalog.values()}
    retired_pairs = {
        (str(v["provider"]), str(v["model"])) for v in catalog.values() if v.get("status") == "retired"
    }
    for selection in selections:
        pair = (selection.provider, selection.model)
        if pair in retired_pairs:
            raise ModelGovernanceError(
                f"{task_id}: {field} references retired model {pair[0]}/{pair[1]}"
            )
        if pair not in catalog_pairs:
            raise ModelGovernanceError(
                f"{task_id}: {field} references a provider/model pair not in the model "
                f"catalog: {pair[0]}/{pair[1]}; add it to model_governance/models.yaml"
            )


_VALID_LOCAL_CANDIDATE_STATUSES = {"benchmark_pending", "rejected", "no_local_candidate", "approved"}
_VALID_RUNTIME_KINDS = {"mlx_vlm", "vllm_mlx", "mlx_native_fastvlm", "mlx_lm", "ollama"}


def _parse_local_candidates(task_id: str, raw_list: list) -> tuple[LocalCandidateRecord, ...]:
    records = []
    for n, item in enumerate(raw_list):
        if not isinstance(item, dict):
            raise ModelGovernanceError(f"{task_id}: local_candidates[{n}] must be a mapping")
        for req_field in ("checked_date", "runtime_support", "status"):
            if req_field not in item:
                raise ModelGovernanceError(
                    f"{task_id}: local_candidates[{n}] missing required field: {req_field!r}"
                )
        status = str(item["status"])
        if status not in _VALID_LOCAL_CANDIDATE_STATUSES:
            raise ModelGovernanceError(
                f"{task_id}: local_candidates[{n}].status must be one of "
                f"{sorted(_VALID_LOCAL_CANDIDATE_STATUSES)}, got {status!r}"
            )
        if status == "approved":
            receipt_ref = item.get("benchmark_receipt_ref")
            if not receipt_ref:
                raise ModelGovernanceError(
                    f"{task_id}: local candidate with status=approved must include benchmark_receipt_ref"
                )
        runtime_kind = _optional_str(item.get("runtime_kind"))
        if runtime_kind is not None and runtime_kind not in _VALID_RUNTIME_KINDS:
            raise ModelGovernanceError(
                f"{task_id}: local_candidates[{n}].runtime_kind must be one of "
                f"{sorted(_VALID_RUNTIME_KINDS)}, got {runtime_kind!r}"
            )
        multimodal = item.get("multimodal")
        if multimodal is not None and not isinstance(multimodal, bool):
            raise ModelGovernanceError(
                f"{task_id}: local_candidates[{n}].multimodal must be bool, got {type(multimodal).__name__}"
            )
        records.append(
            LocalCandidateRecord(
                model_id=_optional_str(item.get("model_id")),
                source_url=_optional_str(item.get("source_url")),
                checked_date=str(item["checked_date"]),
                source_class=_optional_str(item.get("source_class")),
                runtime_support=str(item["runtime_support"]),
                license=_optional_str(item.get("license")),
                expected_memory_gb=_optional_float(item.get("expected_memory_gb")),
                benchmark_receipt_ref=_optional_str(item.get("benchmark_receipt_ref")),
                status=status,
                rejection_reason=_optional_str(item.get("rejection_reason")),
                runtime_kind=runtime_kind,
                parameter_count=_optional_str(item.get("parameter_count")),
                quantization=_optional_str(item.get("quantization")),
                multimodal=multimodal,
                recommended_um_floor_gb=_optional_float(item.get("recommended_um_floor_gb")),
            )
        )
    return tuple(records)


def _validate_benchmark_policy(
    task_id: str,
    raw: dict[str, Any],
    provider: str,
    model: str,
    fallback_provider: str | None,
    fallback_model: str | None,
    allowed: tuple[ModelSelection, ...],
    benchmark: dict[str, Any],
) -> None:
    allowed_pairs = {(selection.provider, selection.model) for selection in allowed}
    allowed_pairs.add((provider, model))
    if fallback_provider and fallback_model:
        allowed_pairs.add((fallback_provider, fallback_model))
    allowed_models = {model for _, model in allowed_pairs}

    for field in ("candidates",):
        values = benchmark.get(field, [])
        if not isinstance(values, list):
            raise ModelGovernanceError(f"{task_id}: benchmark.{field} must be a list")
        unknown = [str(model) for model in values if str(model) not in allowed_models]
        if unknown:
            raise ModelGovernanceError(
                f"{task_id}: benchmark.{field} contains unregistered models: {unknown}"
            )

    for field in ("authority_selections", "candidate_selections"):
        selections = benchmark.get(field)
        if selections is None:
            continue
        if not isinstance(selections, list) or not selections:
            raise ModelGovernanceError(f"{task_id}: benchmark.{field} must be a non-empty list")
        for index, item in enumerate(selections):
            if not isinstance(item, dict) or "provider" not in item or "model" not in item:
                raise ModelGovernanceError(
                    f"{task_id}: benchmark.{field}[{index}] must define provider and model"
                )
            pair = (str(item["provider"]), str(item["model"]))
            if pair not in allowed_pairs:
                raise ModelGovernanceError(
                    f"{task_id}: benchmark.{field}[{index}] is not registered: "
                    f"{pair[0]}/{pair[1]}"
                )


def _optional_str(value: object) -> str | None:
    if value in (None, "", "none"):
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _validate_local_runtime_profile_id(task_id: str, profile_id: str | None) -> None:
    if profile_id is None:
        return
    from conference_pipeline.model_governance.local_runtime import load_profiles

    profiles = load_profiles()
    if profile_id not in profiles:
        raise ModelGovernanceError(
            f"{task_id}: local_runtime_profile_id {profile_id!r} is not defined "
            "in local_runtime_profiles.yaml"
        )


_PROOF_FIELDS = (
    "research_ref",
    "hardware_fit_ref",
    "benchmark_receipt_ref",
    "operator_approval_ref",
)


def _local_proof_complete(proof: dict) -> bool:
    return all(proof.get(f) for f in _PROOF_FIELDS)


def _resolve_reference_path(ref: str, source_path: Path | None) -> Path:
    path = Path(ref)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path]
    if source_path is not None:
        candidates.extend(parent / path for parent in source_path.resolve().parents)
        candidates.append(source_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_receipt(path: Path, task_id: str):
    if not path.exists():
        raise ModelGovernanceError(
            f"{task_id}: benchmark_receipt_ref does not exist: {path}"
        )
    try:
        from conference_pipeline.benchmarks.receipt_schema import receipt_from_dict

        payload = json.loads(path.read_text(encoding="utf-8"))
        return receipt_from_dict(payload)
    except Exception as exc:
        raise ModelGovernanceError(
            f"{task_id}: benchmark_receipt_ref is not a valid BenchmarkReceipt: {path}"
        ) from exc


def _validate_approved_local_receipt(
    policy: TaskPolicy,
    selection: ModelSelection,
    source_path: Path | None,
) -> None:
    ref = _optional_str(policy.local_proof.get("benchmark_receipt_ref"))
    if ref is None:
        raise ModelGovernanceError(
            f"{policy.task_id}: approved_local_selection requires benchmark_receipt_ref"
        )
    receipt_path = _resolve_reference_path(ref, source_path)
    receipt = _load_receipt(receipt_path, policy.task_id)
    from conference_pipeline.benchmarks.receipt_schema import validate_receipt

    validate_receipt(receipt, task_policy=policy)

    expected = {
        "task_id": policy.task_id,
        "model_id": selection.model,
        "provider": "local",
        "runtime_id": "mlx",
        "execution_mode": "live",
        "selection_decision": "passed",
    }
    actual = {
        "task_id": receipt.task_id,
        "model_id": receipt.model_id,
        "provider": receipt.provider,
        "runtime_id": receipt.runtime_id,
        "execution_mode": receipt.execution_mode,
        "selection_decision": receipt.selection_decision,
    }
    mismatches = [
        f"{key} expected {value!r} got {actual[key]!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ModelGovernanceError(
            f"{policy.task_id}: benchmark_receipt_ref is not valid approved local proof: "
            + "; ".join(mismatches)
        )


def approved_local_selection(
    task_id: str,
    *,
    registry_path: str | Path | None = None,
) -> ModelSelection:
    """Return the approved local ModelSelection for a task.

    Raises ModelGovernanceError naming the first missing proof class when any
    of (research_ref, hardware_fit_ref, benchmark_receipt_ref, operator_approval_ref)
    is absent or empty in the task's local_proof block.
    """
    registry = load_registry(registry_path)
    policy = registry.get(task_id)
    proof = policy.local_proof
    for field_name in _PROOF_FIELDS:
        if not proof.get(field_name):
            raise ModelGovernanceError(
                f"{task_id}: approved_local_selection requires {field_name}; "
                "local production is blocked until all four proof classes are present"
            )
    for sel in policy.allowed_selections:
        if sel.role in ("approved_local", "local"):
            _validate_approved_local_receipt(policy, sel, registry.source_path)
            return sel
    raise ModelGovernanceError(
        f"{task_id}: no registered local selection in allowed_selections; "
        "add a ModelSelection with role='approved_local' to use local execution"
    )


def default_registry_path() -> Path:
    """Return the installed registry YAML path."""
    return Path(str(resources.files(__package__).joinpath("registry.yaml")))


@lru_cache(maxsize=4)
def load_registry(path: str | Path | None = None) -> ModelRegistry:
    """Load the model registry from YAML."""
    registry_path = Path(path) if path else default_registry_path()
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModelGovernanceError(f"registry is not a mapping: {registry_path}")
    return ModelRegistry.from_mapping(payload, source_path=registry_path)


def validate_selection(
    task_id: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    registry_path: str | Path | None = None,
) -> ModelSelection:
    """Validate a provider/model selection against the registry."""
    return load_registry(registry_path).validate(task_id, provider=provider, model=model)


def default_provider(task_id: str) -> str:
    """Return the registered default provider for a task."""
    return load_registry().get(task_id).provider


def default_model(task_id: str) -> str:
    """Return the registered default model for a task."""
    return load_registry().get(task_id).model


def model_for_alias(alias: str) -> str:
    """The active catalog model behind an alias (e.g. ``opus``).

    For call sites that deliberately want a model TIER rather than a governed
    task's default -- an ensemble reviewer, an operator script's "use Opus"
    default. Reading the alias keeps them on the current generation; a
    hardcoded model string silently stayed behind at every upgrade. Raises
    for an unknown or retired alias.
    """

    entry = load_model_catalog().get(alias)
    if entry is None:
        raise ModelGovernanceError(f"unknown model-catalog alias {alias!r}")
    if entry.get("status") != "active":
        raise ModelGovernanceError(f"model-catalog alias {alias!r} is {entry.get('status')}")
    return str(entry["model"])


def default_model_for_provider(task_id: str, provider: str) -> str:
    """Return the registered default model for a task/provider pair."""
    policy = load_registry().get(task_id)
    if provider == policy.provider:
        return policy.model
    for selection in policy.allowed_selections:
        if selection.provider == provider:
            return selection.model
    raise ModelGovernanceError(f"{task_id} has no registered provider: {provider}")


def default_reasoning_effort(task_id: str) -> str | None:
    """Return the registered reasoning effort for a task, when configured."""
    return load_registry().get(task_id).reasoning_effort


def default_fallback_provider(task_id: str) -> str:
    """Return the registered fallback provider for a task, or ``none``."""
    return load_registry().get(task_id).fallback_provider or "none"


def default_fallback_model(task_id: str) -> str:
    """Return the registered fallback model for a task, or an empty string."""
    return load_registry().get(task_id).fallback_model or ""
