"""v1.61 Phase 70 — operator-facing vision-lane status block.

Produces a JSON-serializable dictionary summarizing the vision lane:

* ``endpoint`` — redacted endpoint label (``mlx-vision://<model>`` or
  ``mlx-fastvlm://<model>``). NEVER a host or port number.
* ``model`` — active vision model id from the ``--local-vision-model`` /
  ``GPTR_LOCAL_VISION_MODEL`` env var.
* ``runtime_kind`` — ``vllm_mlx`` or ``mlx_native_fastvlm``.
* ``registry_status`` — ``benchmark_pending`` vs ``approved``, looked up
  for each candidate-bearing task in the registry.
* ``session_receipt_count`` — number of LocalVisionCallReceipt entries
  written under ``<workspace>/storage/`` in the current session.

The function never leaks image bytes, full URLs, or model paths
containing secrets. It returns ``None`` for any field it cannot resolve.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from conference_pipeline.model_governance.registry import load_registry
from conference_pipeline.provider_receipts import scan_local_vision_receipts


_VISION_TASK_IDS = (
    "vision.slide_analysis",
    "vision.figure_caption",
    "vision.chart_summary",
    "presentation.visual_qc",
    "ocr.slide_image_parse",
)


def _redact_endpoint(model_id: str | None, runtime_kind: str | None) -> str | None:
    if not model_id:
        return None
    if runtime_kind == "fastvlm":
        return f"mlx-fastvlm://{model_id}"
    # Default to the vllm-mlx label.
    return f"mlx-vision://{model_id}"


def _registry_status_per_task(model_id: str | None) -> dict[str, str]:
    if not model_id:
        return {tid: "benchmark_pending" for tid in _VISION_TASK_IDS}
    try:
        registry = load_registry()
    except Exception:
        return {tid: "benchmark_pending" for tid in _VISION_TASK_IDS}
    out: dict[str, str] = {}
    for task_id in _VISION_TASK_IDS:
        try:
            policy = registry.get(task_id)
        except Exception:
            out[task_id] = "benchmark_pending"
            continue
        candidate = next(
            (c for c in policy.local_candidates if c.model_id == model_id),
            None,
        )
        out[task_id] = candidate.status if candidate else "benchmark_pending"
    return out


def vision_status_block(*, workspace: Path | None = None) -> dict[str, Any]:
    """Compose the operator-facing vision-lane status block.

    Reads the active selection from environment variables (set by
    ``conference-smoketest --local --local-vision-model ...``):
    ``GPTR_LOCAL_VISION_MODEL``, ``GPTR_LOCAL_VISION_RUNTIME_KIND``.
    Counts session receipts by scanning ``<workspace>/storage/`` for
    :class:`LocalVisionCallReceipt` writes (returns 0 if no workspace).
    """
    model_id = (os.environ.get("GPTR_LOCAL_VISION_MODEL") or "").strip() or None
    runtime_kind = (os.environ.get("GPTR_LOCAL_VISION_RUNTIME_KIND") or "").strip() or None

    receipt_count = 0
    if workspace is not None:
        try:
            receipt_count = len(scan_local_vision_receipts(workspace))
        except Exception:
            receipt_count = 0

    return {
        "lane": "vision",
        "active": bool(model_id),
        "model": model_id,
        "runtime_kind": runtime_kind,
        "endpoint": _redact_endpoint(model_id, runtime_kind),
        "registry_status_by_task": _registry_status_per_task(model_id),
        "session_receipt_count": receipt_count,
    }
