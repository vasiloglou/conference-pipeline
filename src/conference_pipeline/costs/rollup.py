"""Cost rollup engine for the conference-pipeline cost-tracking layer.

``build_cost_rollup`` scans every receipt family for a run, joins actuals
against an ``EstimateReport`` by ``task_id``, and emits a standalone
``storage/workflow-runs/<run_id>/cost_rollup.json`` with:

  - provider × model × stage_id × task_id lines (D-06)
  - each line: estimated_usd, actual_usd, delta_usd, cost_status (D-07)
  - drift flags when |delta| > max(rel_threshold × estimated, abs_floor) (D-10)
  - an ``unattributed`` bucket for receipts with no task_id (D-06)
  - needs_pricing_backfill flag for unpriced receipts (D-09) — never raises
  - distinct $0 local lines per family with call_count (D-13, D-14)
  - run-level totals block

Companion helpers ``build_run_cost_summary`` and ``build_stage_cost_summary``
return compact blocks that Plan 04 mirrors into ``summary.json`` and the
control plane.

Security: all payloads are run through ``sanitize_for_receipt`` before
``_atomic_write_json`` (T-02-03). The write path is resolved under
``<workspace>/storage/workflow-runs/<run_id>/`` (T-02-04).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from conference_pipeline.costs.estimator import EstimateReport
from conference_pipeline.provider_receipts import (
    _atomic_write_json,
    sanitize_for_receipt,
    scan_local_embedding_receipts,
    scan_local_llm_receipts,
    scan_local_vision_receipts,
    scan_provider_call_receipts,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D-07 cost_status mapping helpers
# ---------------------------------------------------------------------------

_PRICED_RECEIPT_STATUSES = {"estimated", "local_no_paid_provider"}


def _map_cost_status(receipt_status: str | None, cost_value: Any) -> str:
    """Map a receipt's cost_status to the rollup tri-state (priced/unknown/missing).

    D-07:
      receipt "estimated" | "local_no_paid_provider" → "priced"
      receipt "unknown"                              → "unknown"
      missing cost field                             → "missing"
    """
    if receipt_status in _PRICED_RECEIPT_STATUSES and isinstance(cost_value, (int, float)):
        return "priced"
    if receipt_status == "unknown":
        return "unknown"
    return "missing"


# ---------------------------------------------------------------------------
# Grouping key helpers
# ---------------------------------------------------------------------------

_GroupKey = tuple[str, str, str, str]  # (provider, model, stage_id, task_id)


def _receipt_group_key(receipt: dict[str, Any]) -> _GroupKey:
    provider = str(receipt.get("provider") or "unknown")
    model = str(receipt.get("model") or "unknown")
    stage_id = str(receipt.get("stage_id") or "unknown")
    task_id = str(receipt.get("task_id") or "unattributed")
    return (provider, model, stage_id, task_id)


# ---------------------------------------------------------------------------
# Per-group accumulator
# ---------------------------------------------------------------------------


def _new_group() -> dict[str, Any]:
    return {
        "actual_usd": 0.0,
        "call_count": 0,
        "cost_status": "priced",  # downgraded if any receipt is unpriced
        "has_missing": False,
        "has_unknown": False,
    }


def _merge_receipt_into_group(group: dict[str, Any], receipt: dict[str, Any]) -> None:
    """Accumulate a single provider-call receipt into a group."""
    cost = receipt.get("estimated_cost_usd")
    raw_status = receipt.get("cost_status", "")
    rollup_status = _map_cost_status(raw_status, cost)

    if rollup_status == "priced" and isinstance(cost, (int, float)):
        group["actual_usd"] += float(cost)
    elif rollup_status == "unknown":
        group["has_unknown"] = True
    else:  # missing
        group["has_missing"] = True

    group["call_count"] += 1

    # Downgrade group cost_status (priced > unknown > missing)
    if rollup_status != "priced":
        if group["cost_status"] == "priced":
            group["cost_status"] = rollup_status
        elif rollup_status == "missing":
            group["cost_status"] = "missing"


# ---------------------------------------------------------------------------
# Drift evaluation (D-10)
# ---------------------------------------------------------------------------


def _is_drifted(
    actual_usd: float,
    estimated_usd: float | None,
    rel_threshold: float,
    abs_floor: float,
) -> bool:
    """Return True iff abs(delta) > max(rel_threshold*estimated, abs_floor)."""
    if estimated_usd is None or estimated_usd == 0.0:
        return False
    delta = abs(actual_usd - estimated_usd)
    threshold = max(rel_threshold * estimated_usd, abs_floor)
    return delta > threshold


# ---------------------------------------------------------------------------
# Local-family line builders
# ---------------------------------------------------------------------------


def _build_local_lines(
    family_label: str,
    receipts: list[dict[str, Any]],
    cost_field: str,
    count_field: str,
) -> list[dict[str, Any]]:
    """Build $0 rollup lines for a local receipt family.

    Groups by stage_id and produces one line per stage.
    Local work is always cost_status="priced" at $0 (D-07 local_no_paid_provider).

    v3.0 03-04 (LOCAL-04 / D-15): each line also carries aggregated cost-avoided
    dimensions: ``cost_avoided_usd`` (sum of priced savings or None when any
    receipt is unknown), ``cost_avoided_status`` (``"priced"`` | ``"unknown"``),
    ``needs_pricing_backfill`` (True when any receipt has unknown avoided cost),
    and ``benchmark_receipt_refs`` (deduplicated list of justifying benchmark refs).

    D-16: unknown savings are never coerced to $0 — when any contributing receipt
    has ``cost_avoided_status="unknown"`` the line's ``cost_avoided_usd`` may be
    the sum of *only* priced contributions (if any), but the line is still marked
    ``"unknown"`` and ``needs_pricing_backfill=True``.
    """
    groups: dict[str, dict[str, Any]] = {}  # stage_id -> accumulator
    for r in receipts:
        stage = str(r.get("stage_id") or "unknown")
        if stage not in groups:
            groups[stage] = {
                "call_count": 0,
                "actual_usd": 0.0,
                # cost-avoided accumulators
                "ca_priced_sum": 0.0,
                "ca_has_priced": False,
                "ca_has_unknown": False,
                "benchmark_refs": [],
            }
        g = groups[stage]
        # count_field can be an int or default to 1 if absent
        count = r.get(count_field)
        if isinstance(count, int):
            g["call_count"] += count
        else:
            g["call_count"] += 1
        # cost field is always 0.0 for local receipts
        cost = r.get(cost_field, 0.0)
        if isinstance(cost, (int, float)):
            g["actual_usd"] += float(cost)

        # --- cost-avoided aggregation (D-15 / D-16) ---
        ca_status = r.get("cost_avoided_status")
        ca_usd = r.get("cost_avoided_usd")
        if ca_status == "priced" and isinstance(ca_usd, (int, float)):
            g["ca_priced_sum"] += float(ca_usd)
            g["ca_has_priced"] = True
        elif ca_status == "unknown":
            g["ca_has_unknown"] = True

        bench_ref = r.get("benchmark_receipt_ref")
        if bench_ref and isinstance(bench_ref, str):
            if bench_ref not in g["benchmark_refs"]:
                g["benchmark_refs"].append(bench_ref)

    lines = []
    for stage_id, acc in sorted(groups.items()):
        # Resolve cost-avoided status for this aggregated line.
        if acc["ca_has_unknown"]:
            # D-16: any unknown receipt makes the whole line unknown.
            # We still report the sum of priced receipts when available, but
            # the status is "unknown" because not all savings are confirmed.
            ca_status_out: str = "unknown"
            ca_usd_out: float | None = acc["ca_priced_sum"] if acc["ca_has_priced"] else None
            ca_needs_backfill: bool = True
        elif acc["ca_has_priced"]:
            ca_status_out = "priced"
            ca_usd_out = round(acc["ca_priced_sum"], 8)
            ca_needs_backfill = False
        else:
            # No cost-avoided metadata on any receipt in this group.
            ca_status_out = "unknown"
            ca_usd_out = None
            ca_needs_backfill = False  # not yet requested; not all receipts carry CA fields

        lines.append(
            {
                "provider": family_label,
                "model": "local",
                "stage_id": stage_id,
                "task_id": "local",
                "estimated_usd": 0.0,
                "actual_usd": round(acc["actual_usd"], 8),
                "delta_usd": 0.0,
                "cost_status": "priced",
                # needs_pricing_backfill is True when cost-avoided savings are
                # unknown, even though the local spend itself is always $0/priced.
                "needs_pricing_backfill": ca_needs_backfill,
                "drifted": False,
                "call_count": acc["call_count"],
                # v3.0 03-04: cost-avoided dimensions (D-15)
                "cost_avoided_usd": ca_usd_out,
                "cost_avoided_status": ca_status_out,
                "benchmark_receipt_refs": acc["benchmark_refs"],
            }
        )
    return lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cost_rollup(
    workspace: Path,
    run_id: str,
    *,
    estimate_report: EstimateReport | None = None,
    drift_rel_threshold: float = 0.20,
    drift_abs_floor_usd: float = 0.01,
    unknown_cost_ceiling_usd: float = 1.0,
) -> dict[str, Any]:
    """Scan all receipt families for ``run_id`` and emit ``cost_rollup.json``.

    Parameters
    ----------
    workspace:
        Conference workspace root (Path).
    run_id:
        Workflow run ID; the rollup is written to
        ``<workspace>/storage/workflow-runs/<run_id>/cost_rollup.json``.
    estimate_report:
        Pre-built ``EstimateReport`` to join against. When ``None``, lines
        have ``estimated_usd=None`` and ``delta_usd=None``.
    drift_rel_threshold:
        Relative drift threshold (default 0.20 = 20 %).
    drift_abs_floor_usd:
        Absolute floor for drift detection (default $0.01).
    unknown_cost_ceiling_usd:
        Conservative ceiling charged for unknown-cost receipts (informational
        only — used in future budget-enforcement integration).

    Returns
    -------
    dict
        The rollup payload (same as what was written to disk).
    """
    workspace = Path(workspace)

    # ------------------------------------------------------------------
    # 1. Scan all five receipt families (Pitfall 5 — separate calls)
    # ------------------------------------------------------------------
    provider_receipts = scan_provider_call_receipts(workspace)
    local_llm_receipts = scan_local_llm_receipts(workspace)
    local_embedding_receipts = scan_local_embedding_receipts(workspace)
    local_vision_receipts = scan_local_vision_receipts(workspace)

    # Filter to this run
    provider_receipts = [r for r in provider_receipts if r.get("run_id") == run_id]
    local_llm_receipts = [r for r in local_llm_receipts if r.get("run_id") == run_id]
    local_embedding_receipts = [r for r in local_embedding_receipts if r.get("run_id") == run_id]
    local_vision_receipts = [r for r in local_vision_receipts if r.get("run_id") == run_id]

    # ------------------------------------------------------------------
    # 2. Build estimate lookup by task_id (D-07 join)
    # ------------------------------------------------------------------
    estimate_by_task: dict[str, float] = {}
    if estimate_report is not None:
        for task in estimate_report.tasks:
            estimate_by_task[task.task_id] = task.estimated_cost_usd

    # ------------------------------------------------------------------
    # 3. Aggregate provider-call receipts into groups
    #    Key: (provider, model, stage_id, task_id)
    # ------------------------------------------------------------------
    groups: dict[_GroupKey, dict[str, Any]] = {}
    for receipt in provider_receipts:
        key = _receipt_group_key(receipt)
        if key not in groups:
            groups[key] = _new_group()
        _merge_receipt_into_group(groups[key], receipt)

    # ------------------------------------------------------------------
    # 4. Convert groups to rollup lines
    # ------------------------------------------------------------------
    lines: list[dict[str, Any]] = []
    for (provider, model, stage_id, task_id), acc in sorted(groups.items()):
        actual_usd = round(acc["actual_usd"], 8)
        cost_status = acc["cost_status"]

        # Join against EstimateReport (D-08)
        estimated_usd: float | None = estimate_by_task.get(task_id)
        if estimated_usd is not None:
            estimated_usd = round(estimated_usd, 8)
        delta_usd: float | None = (
            round(actual_usd - estimated_usd, 8)
            if estimated_usd is not None
            else None
        )

        # Drift flag (D-10)
        drifted = (
            _is_drifted(actual_usd, estimated_usd, drift_rel_threshold, drift_abs_floor_usd)
            if estimated_usd is not None
            else False
        )

        # Pricing backfill flag (D-09 — never raise)
        needs_pricing_backfill = cost_status != "priced"

        lines.append(
            {
                "provider": provider,
                "model": model,
                "stage_id": stage_id,
                "task_id": task_id,
                "estimated_usd": estimated_usd,
                "actual_usd": actual_usd,
                "delta_usd": delta_usd,
                "cost_status": cost_status,
                "needs_pricing_backfill": needs_pricing_backfill,
                "drifted": drifted,
                "call_count": acc["call_count"],
            }
        )
        if needs_pricing_backfill:
            logger.warning(
                "Cost rollup: receipt with cost_status=%r for %s/%s/%s/%s — "
                "needs pricing backfill",
                cost_status, provider, model, stage_id, task_id,
            )

    # ------------------------------------------------------------------
    # 5. Local-family $0 lines (D-13, D-14)
    # ------------------------------------------------------------------
    # LocalLLMReceipt: cost field = "paid_cloud_cost", count = "request_count"
    local_lines = _build_local_lines(
        "local_llm", local_llm_receipts, "paid_cloud_cost", "request_count"
    )
    # LocalEmbeddingReceipt: cost field = "cost_usd", count = "request_count"
    local_lines += _build_local_lines(
        "local_embedding", local_embedding_receipts, "cost_usd", "request_count"
    )
    # LocalVisionCallReceipt: cost field = "cost_usd", count = "image_count"
    local_lines += _build_local_lines(
        "local_vision", local_vision_receipts, "cost_usd", "image_count"
    )

    all_lines = lines + local_lines

    # ------------------------------------------------------------------
    # 6. Run-level totals
    # ------------------------------------------------------------------
    total_actual_usd = round(
        sum(l["actual_usd"] for l in all_lines if isinstance(l.get("actual_usd"), float)), 8
    )
    total_estimated_usd_parts = [
        l["estimated_usd"]
        for l in all_lines
        if isinstance(l.get("estimated_usd"), float)
    ]
    total_estimated_usd = round(sum(total_estimated_usd_parts), 8) if total_estimated_usd_parts else None
    total_delta_usd = (
        round(total_actual_usd - total_estimated_usd, 8)
        if total_estimated_usd is not None
        else None
    )
    has_unknown_cost = any(l.get("cost_status") == "unknown" for l in all_lines)
    has_missing_cost = any(l.get("cost_status") == "missing" for l in all_lines)
    total_local_call_count = sum(
        l.get("call_count", 0)
        for l in local_lines
    )
    drifted_lines = [l for l in all_lines if l.get("drifted")]

    # v3.0 03-04: cost-avoided run-level totals (LOCAL-04 / D-15)
    total_cost_avoided_usd: float = 0.0
    has_unknown_cost_avoided: bool = False
    cost_avoided_backfill_count: int = 0
    for line in local_lines:
        ca_usd = line.get("cost_avoided_usd")
        ca_status = line.get("cost_avoided_status")
        ca_backfill = line.get("needs_pricing_backfill", False)
        # WR-02: a mixed priced+unknown line keeps its confirmed priced sum in
        # cost_avoided_usd while its status is "unknown" (D-16). Accumulate the
        # numeric cost_avoided_usd regardless of status so the run total never
        # silently drops priced savings that are already reported on the line.
        # cost_avoided_usd only ever holds priced contributions (None when no
        # priced savings exist), so this cannot count a guessed/unknown number.
        if isinstance(ca_usd, (int, float)):
            total_cost_avoided_usd += float(ca_usd)
        # WR-01: only a line that actually requests a pricing backfill is a
        # genuine unknown-savings signal. A plain local line that simply carries
        # no cost-avoided metadata (status "unknown", needs_pricing_backfill
        # False) must NOT flip the run-level flag — that conflated "no
        # substitution claimed" with "substitution savings genuinely unknown"
        # and produced spurious pricing-backfill alarms (D-16).
        if ca_backfill:
            has_unknown_cost_avoided = True
            cost_avoided_backfill_count += 1

    run_totals: dict[str, Any] = {
        "actual_usd": total_actual_usd,
        "estimated_usd": total_estimated_usd,
        "delta_usd": total_delta_usd,
        "has_unknown_cost": has_unknown_cost,
        "has_missing_cost": has_missing_cost,
        "local_call_count": total_local_call_count,
        "drifted_line_count": len(drifted_lines),
        "receipt_line_count": len(all_lines),
        # v3.0 03-04: cost-avoided totals (D-15)
        "total_cost_avoided_usd": round(total_cost_avoided_usd, 8),
        "has_unknown_cost_avoided": has_unknown_cost_avoided,
        "cost_avoided_backfill_count": cost_avoided_backfill_count,
    }

    # ------------------------------------------------------------------
    # 7. Build payload and write atomically (T-02-03, T-02-04)
    # ------------------------------------------------------------------
    payload: dict[str, Any] = {
        "run_id": run_id,
        "lines": all_lines,
        "run_totals": run_totals,
    }

    sanitized = sanitize_for_receipt(payload)

    run_dir = workspace / "storage" / "workflow-runs" / run_id
    rollup_path = run_dir / "cost_rollup.json"
    _atomic_write_json(rollup_path, sanitized)

    return sanitized


# ---------------------------------------------------------------------------
# Compact summary builders (for Plan 04 — summary.json mirror)
# ---------------------------------------------------------------------------


def build_run_cost_summary(rollup: dict[str, Any]) -> dict[str, Any]:
    """Return a compact run-level cost summary block from a rollup dict.

    Keys: ``estimated_usd``, ``actual_usd``, ``local_call_count``, ``has_unknown_cost``.
    """
    totals = rollup.get("run_totals") or {}
    return {
        "estimated_usd": totals.get("estimated_usd"),
        "actual_usd": totals.get("actual_usd", 0.0),
        "local_call_count": totals.get("local_call_count", 0),
        "has_unknown_cost": totals.get("has_unknown_cost", False),
    }


def build_stage_cost_summary(rollup: dict[str, Any], stage_id: str) -> dict[str, Any]:
    """Return a compact stage-level cost summary block from a rollup dict.

    Keys: ``estimated_usd``, ``actual_usd``, ``local_call_count``, ``has_unknown_cost``.
    """
    lines = rollup.get("lines") or []
    stage_lines = [l for l in lines if l.get("stage_id") == stage_id]

    actual_usd = round(
        sum(l["actual_usd"] for l in stage_lines if isinstance(l.get("actual_usd"), float)), 8
    )
    estimated_parts = [
        l["estimated_usd"]
        for l in stage_lines
        if isinstance(l.get("estimated_usd"), float)
    ]
    estimated_usd = round(sum(estimated_parts), 8) if estimated_parts else None
    local_call_count = sum(
        l.get("call_count", 0)
        for l in stage_lines
        if l.get("provider") in {"local_llm", "local_embedding", "local_vision"}
    )
    has_unknown_cost = any(l.get("cost_status") == "unknown" for l in stage_lines)

    return {
        "estimated_usd": estimated_usd,
        "actual_usd": actual_usd,
        "local_call_count": local_call_count,
        "has_unknown_cost": has_unknown_cost,
    }
