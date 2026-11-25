"""Fixture builders for cost-rollup tests.

Provides helpers to write constructed receipt JSON into the on-disk layout
that the real ``scan_*`` functions expect. Also provides a tiny
``EstimateReport`` builder for rollup-join tests.

All paths follow the conventions in ``provider_receipts.py``:
  - Provider-call receipts: ``<workspace>/storage/workflow-runs/<run_id>/provider-calls/<stage>/<file>.json``
  - Local-LLM receipts:     ``<workspace>/storage/workflow-runs/<run_id>/local-llm-receipts/<stage>/<file>.json``
  - Local-embedding:        ``<workspace>/storage/workflow-runs/<run_id>/local-embedding-receipts/<stage>/<file>.json``
  - Local-vision:           ``<workspace>/storage/workflow-runs/<run_id>/local-vision-receipts/<stage>/<file>.json``
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from conference_pipeline.provider_receipts import (
    RECEIPT_MARKER,
    LOCAL_LLM_RECEIPT_MARKER,
    LOCAL_EMBEDDING_RECEIPT_MARKER,
    LOCAL_VISION_RECEIPT_MARKER,
)
from conference_pipeline.costs.estimator import CostEstimate, EstimateReport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COUNTER = 0


def _next_ts() -> str:
    """Return a deterministic, unique-ish ISO timestamp for test filenames."""
    global _COUNTER  # noqa: PLW0603
    _COUNTER += 1
    return f"20260101T000000{_COUNTER:03d}Z"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)
    return path


# ---------------------------------------------------------------------------
# Provider-call receipt writer
# ---------------------------------------------------------------------------


def write_provider_receipt(
    workspace: Path,
    run_id: str,
    stage_id: str,
    *,
    task_id: str | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    operation: str = "chat.completions",
    estimated_cost_usd: float | None = 0.01,
    cost_status: str = "estimated",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a minimal provider-call receipt JSON that ``scan_provider_call_receipts`` will find.

    The receipt is written to:
      ``<workspace>/storage/workflow-runs/<run_id>/provider-calls/<stage_id>/<ts>-receipt.json``
    """
    ts = _next_ts()
    path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "provider-calls"
        / stage_id
        / f"{ts}-receipt.json"
    )
    payload: dict[str, Any] = {
        "receipt_type": RECEIPT_MARKER,
        "schema_version": 1,
        "provider": provider,
        "model": model,
        "operation": operation,
        "stage_id": stage_id,
        "run_id": run_id,
        "cost_status": cost_status,
        "status": "succeeded",
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if estimated_cost_usd is not None:
        payload["estimated_cost_usd"] = estimated_cost_usd
    if extra:
        payload.update(extra)
    return _write_json(path, payload)


# ---------------------------------------------------------------------------
# Local-LLM receipt writer
# ---------------------------------------------------------------------------


def write_local_llm_receipt(
    workspace: Path,
    run_id: str,
    stage_id: str,
    *,
    model: str = "mlx-community/Mistral-7B-v0.1-4bit",
    request_count: int = 3,
    paid_cloud_cost: float = 0.0,
) -> Path:
    """Write a minimal LocalLLMReceipt JSON that ``scan_local_llm_receipts`` will find."""
    ts = _next_ts()
    path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "local-llm-receipts"
        / stage_id
        / f"{ts}-llm.json"
    )
    payload: dict[str, Any] = {
        "receipt_type": LOCAL_LLM_RECEIPT_MARKER,
        "schema_version": 1,
        "endpoint_label": "mlx://local",
        "model": model,
        "request_count": request_count,
        "token_count": None,
        "confidence_path": "local_llm",
        "skipped_record_count": 0,
        "paid_cloud_cost": paid_cloud_cost,
        "timestamp": ts,
        "run_id": run_id,
        "stage_id": stage_id,
        "preflight_mode": False,
        "quality_sample_mode": False,
    }
    return _write_json(path, payload)


# ---------------------------------------------------------------------------
# Local-embedding receipt writer
# ---------------------------------------------------------------------------


def write_local_embedding_receipt(
    workspace: Path,
    run_id: str,
    stage_id: str,
    *,
    model: str = "nomic-embed-text",
    request_count: int = 2,
    cost_usd: float = 0.0,
    # v3.0 03-04: cost-avoided additive fields (LOCAL-04 / D-13..D-16)
    cost_avoided_usd: float | None = None,
    cost_avoided_status: str | None = None,
    benchmark_receipt_ref: str | None = None,
    needs_pricing_backfill: bool | None = None,
    paid_counterfactual_provider: str | None = None,
    paid_counterfactual_model: str | None = None,
) -> Path:
    """Write a minimal LocalEmbeddingReceipt JSON that ``scan_local_embedding_receipts`` will find."""
    ts = _next_ts()
    path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "local-embedding-receipts"
        / stage_id
        / f"{ts}-embed.json"
    )
    payload: dict[str, Any] = {
        "receipt_type": LOCAL_EMBEDDING_RECEIPT_MARKER,
        "schema_version": 1,
        "provider": "mlx",
        "model": model,
        "embedding_dim": 768,
        "endpoint_label": "mlx://nomic-embed-text",
        "input_count": 10,
        "total_input_chars": 5000,
        "request_count": request_count,
        "cost_usd": cost_usd,
        "timestamp": ts,
        "run_id": run_id,
        "stage_id": stage_id,
    }
    # Cost-avoided fields are additive — only include when provided.
    if cost_avoided_usd is not None:
        payload["cost_avoided_usd"] = cost_avoided_usd
    if cost_avoided_status is not None:
        payload["cost_avoided_status"] = cost_avoided_status
    if benchmark_receipt_ref is not None:
        payload["benchmark_receipt_ref"] = benchmark_receipt_ref
    if needs_pricing_backfill is not None:
        payload["needs_pricing_backfill"] = needs_pricing_backfill
    if paid_counterfactual_provider is not None:
        payload["paid_counterfactual_provider"] = paid_counterfactual_provider
    if paid_counterfactual_model is not None:
        payload["paid_counterfactual_model"] = paid_counterfactual_model
    # D-16: if status is unknown, ensure cost_avoided_usd is not written as $0.
    if payload.get("cost_avoided_status") == "unknown":
        payload["cost_avoided_usd"] = None
    return _write_json(path, payload)


# ---------------------------------------------------------------------------
# Local-vision receipt writer
# ---------------------------------------------------------------------------


def write_local_vision_receipt(
    workspace: Path,
    run_id: str,
    stage_id: str,
    *,
    model: str = "mlx-community/llava-1.5-7b",
    image_count: int = 2,
    cost_usd: float = 0.0,
    # v3.0 03-04: cost-avoided additive fields (LOCAL-04 / D-13..D-16)
    cost_avoided_usd: float | None = None,
    cost_avoided_status: str | None = None,
    benchmark_receipt_ref: str | None = None,
    needs_pricing_backfill: bool | None = None,
    paid_counterfactual_provider: str | None = None,
    paid_counterfactual_model: str | None = None,
) -> Path:
    """Write a minimal LocalVisionCallReceipt JSON that ``scan_local_vision_receipts`` will find."""
    ts = _next_ts()
    path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "local-vision-receipts"
        / stage_id
        / f"{ts}-vision.json"
    )
    payload: dict[str, Any] = {
        "receipt_type": LOCAL_VISION_RECEIPT_MARKER,
        "schema_version": 1,
        "provider": "local_vlm",
        "runtime_kind": "mlx_vlm",
        "model": model,
        "endpoint_label": f"mlx-vision://{model}",
        "vision_task": "slide_analysis",
        "image_count": image_count,
        "image_dimensions": [[1920, 1080]] * image_count,
        "image_bytes_size": 100000 * image_count,
        "prompt_token_estimate": None,
        "completion_token_estimate": None,
        "latency_ms": 250.0,
        "cost_usd": cost_usd,
        "estimated_paid_equivalent_usd": None,
        "timestamp": ts,
        "run_id": run_id,
        "stage_id": stage_id,
    }
    # Cost-avoided fields are additive — only include when provided.
    if cost_avoided_usd is not None:
        payload["cost_avoided_usd"] = cost_avoided_usd
    if cost_avoided_status is not None:
        payload["cost_avoided_status"] = cost_avoided_status
    if benchmark_receipt_ref is not None:
        payload["benchmark_receipt_ref"] = benchmark_receipt_ref
    if needs_pricing_backfill is not None:
        payload["needs_pricing_backfill"] = needs_pricing_backfill
    if paid_counterfactual_provider is not None:
        payload["paid_counterfactual_provider"] = paid_counterfactual_provider
    if paid_counterfactual_model is not None:
        payload["paid_counterfactual_model"] = paid_counterfactual_model
    # D-16: if status is unknown, ensure cost_avoided_usd is not written as $0.
    if payload.get("cost_avoided_status") == "unknown":
        payload["cost_avoided_usd"] = None
    return _write_json(path, payload)


# ---------------------------------------------------------------------------
# EstimateReport builder
# ---------------------------------------------------------------------------


def make_estimate_report(
    tasks: list[dict[str, Any]] | None = None,
) -> EstimateReport:
    """Return a tiny EstimateReport with one or two CostEstimate entries.

    Parameters
    ----------
    tasks:
        Optional list of dicts with keys matching ``CostEstimate`` fields.
        Defaults to a single task with ``task_id="task-01"``.
    """
    if tasks is None:
        tasks = [
            {
                "task_id": "task-01",
                "family": "research",
                "stage": "research-stage",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "pricing_unit": "tokens",
                "enabled": True,
                "quantity": 10000.0,
                "estimated_cost_usd": 0.01,
            }
        ]

    cost_estimates = []
    for t in tasks:
        cost_estimates.append(
            CostEstimate(
                task_id=t["task_id"],
                family=t.get("family", "research"),
                stage=t.get("stage", "test-stage"),
                provider=t.get("provider", "openai"),
                model=t.get("model", "gpt-4o-mini"),
                pricing_unit=t.get("pricing_unit", "tokens"),
                enabled=t.get("enabled", True),
                quantity=t.get("quantity", 10000.0),
                estimated_cost_usd=t.get("estimated_cost_usd", 0.01),
            )
        )

    from conference_pipeline.costs.estimator import Workload
    workload = Workload(workspace="test-workspace")

    return EstimateReport(
        mode="dry_run",
        scope="test",
        scenario="test",
        workspace="test-workspace",
        generated_at="2026-01-01T00:00:00Z",
        pricing_refreshed_at="2026-01-01T00:00:00Z",
        workload=workload,
        tasks=cost_estimates,
    )
