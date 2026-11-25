"""BenchmarkReceipt schema and validation (D-05 contract)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass


_VALID_DECISIONS = {"passed", "failed", "skipped"}


@dataclass(frozen=True)
class BenchmarkReceipt:
    # Planning scope
    workspace: str
    task_id: str
    task_family: str
    stage: str
    run_id: str          # uuid4 str
    timestamp: str       # ISO-8601
    execution_mode: str  # "dry-run" | "live"
    # Model run descriptors
    provider: str
    model_id: str
    runtime_id: str      # "openai" | "anthropic" | "mlx" | "mock"
    hardware_profile: str
    prompt_digest: str   # sha256 hex of prompt
    config_digest: str   # sha256 hex of serialized config
    # Input / output
    input_sample_ref: str
    output_ref: str | None
    raw_output_snippet: str
    # Scoring
    rubric_scores: dict
    schema_ok: bool
    provenance_ok: bool
    quality_delta: float | None
    latency_ms: float | None
    cost_usd: float | None
    budget_violations: list
    # Decision
    selection_decision: str  # "passed" | "failed" | "skipped"
    failure_reason: str | None


def validate_receipt(receipt: BenchmarkReceipt, *, task_policy=None) -> None:
    """Raise ValueError for an invalid receipt."""
    if receipt.selection_decision not in _VALID_DECISIONS:
        raise ValueError(
            f"invalid selection_decision {receipt.selection_decision!r}; "
            f"must be one of {sorted(_VALID_DECISIONS)}"
        )
    if (
        receipt.execution_mode == "live"
        and receipt.provider == "local"
        and receipt.runtime_id == "mlx"
        and receipt.selection_decision == "passed"
    ):
        quality = receipt.rubric_scores.get("quality")
        missing = []
        if not receipt.hardware_profile or receipt.hardware_profile == "mock":
            missing.append("non-mock hardware_profile")
        if not receipt.output_ref:
            missing.append("output_ref")
        if not receipt.raw_output_snippet.strip():
            missing.append("raw_output_snippet")
        if receipt.schema_ok is not True:
            missing.append("schema_ok")
        if receipt.provenance_ok is not True:
            missing.append("provenance_ok")
        if not isinstance(quality, (int, float)) or isinstance(quality, bool) or quality <= 0:
            missing.append("positive quality score")
        if receipt.quality_delta is None:
            missing.append("quality_delta")
        if receipt.failure_reason is not None:
            missing.append("empty failure_reason")
        if missing:
            raise ValueError(
                "live local passed receipt requires " + ", ".join(missing)
            )
    if task_policy is not None and task_policy.provenance_required:
        if receipt.schema_ok and not receipt.provenance_ok:
            raise ValueError(
                "receipt has schema_ok=True but provenance_ok=False "
                "for a task with provenance_required=True"
            )


def receipt_to_dict(receipt: BenchmarkReceipt) -> dict:
    return dataclasses.asdict(receipt)


def receipt_from_dict(d: dict) -> BenchmarkReceipt:
    return BenchmarkReceipt(**d)
