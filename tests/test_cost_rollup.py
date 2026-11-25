"""Tests for the cost-rollup engine (costs/rollup.py).

All tests are offline, narrow-deterministic, and use ``tmp_path`` with
constructed fixtures. No live providers, no network, no Drive.

Test names are derived from the VALIDATION map:
  test_dry_run_emits_rollup         — COST-01 / D-05, D-06
  test_rollup_reuses_estimate       — COST-01 / D-07 (join to EstimateReport)
  test_drift_threshold              — COST-02 / D-10
  test_unknown_cost_flagged_not_fatal — COST-02 / D-09
  test_unattributed_bucket          — COST-02 / D-06
  test_local_zero_dollar_lines      — COST-05 / D-13, D-14
  test_summary_cost_mirror          — COST-05 (Plan 04 full integration)
  test_import_build_cost_rollup     — smoke import
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.costs.rollup import build_cost_rollup, build_run_cost_summary, build_stage_cost_summary
from conference_pipeline.workflow_runs import StageReceipt, WorkflowRunSummary
from tests.fixtures.cost.receipts import (
    make_estimate_report,
    write_local_embedding_receipt,
    write_local_llm_receipt,
    write_local_vision_receipt,
    write_provider_receipt,
)


# ---------------------------------------------------------------------------
# Smoke import
# ---------------------------------------------------------------------------


def test_import_build_cost_rollup() -> None:
    """Smoke: build_cost_rollup is importable from the expected module path."""
    from conference_pipeline.costs.rollup import build_cost_rollup as _fn  # noqa: F401
    assert callable(_fn)


# ---------------------------------------------------------------------------
# COST-01 / D-05 — rollup file emitted
# ---------------------------------------------------------------------------


def test_dry_run_emits_rollup(tmp_path: Path) -> None:
    """build_cost_rollup writes cost_rollup.json under the run dir with required keys."""
    run_id = "test-run-01"
    write_provider_receipt(
        tmp_path,
        run_id,
        "research-stage",
        task_id="task-01",
        provider="openai",
        model="gpt-4o-mini",
        estimated_cost_usd=0.01,
        cost_status="estimated",
    )
    report = make_estimate_report()
    result = build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    assert rollup_path.exists(), "cost_rollup.json must be written under the run dir"

    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    assert "lines" in payload, "rollup JSON must have a 'lines' key"
    assert "run_totals" in payload, "rollup JSON must have a 'run_totals' key"

    assert len(payload["lines"]) >= 1, "at least one line expected"
    line = payload["lines"][0]
    required_keys = {"provider", "model", "stage_id", "task_id", "estimated_usd",
                     "actual_usd", "delta_usd", "cost_status"}
    assert required_keys.issubset(line.keys()), (
        f"line missing keys: {required_keys - line.keys()}"
    )

    # Threat T-02-03: no secret values in the rollup
    rollup_text = rollup_path.read_text(encoding="utf-8")
    import re
    assert not re.search(r'"api_key"', rollup_text, re.IGNORECASE)
    assert not re.search(r'bearer', rollup_text, re.IGNORECASE)
    assert not re.search(r'"authorization"', rollup_text, re.IGNORECASE)

    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# COST-01 / D-07 — rollup reuses EstimateReport, not re-derived
# ---------------------------------------------------------------------------


def test_rollup_reuses_estimate(tmp_path: Path) -> None:
    """Each rollup line's estimated_usd comes from the joined EstimateReport."""
    run_id = "test-run-02"
    write_provider_receipt(
        tmp_path,
        run_id,
        "research-stage",
        task_id="task-01",
        provider="openai",
        model="gpt-4o-mini",
        estimated_cost_usd=0.05,
        cost_status="estimated",
    )
    # EstimateReport says this task costs $0.02 (not re-derived from receipt)
    report = make_estimate_report(
        tasks=[
            {
                "task_id": "task-01",
                "family": "research",
                "stage": "research-stage",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "pricing_unit": "tokens",
                "enabled": True,
                "quantity": 20000.0,
                "estimated_cost_usd": 0.02,
            }
        ]
    )
    build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]
    matched = [l for l in lines if l.get("task_id") == "task-01" and l.get("provider") == "openai"]
    assert matched, "expected a line for task-01/openai"
    # The estimate comes from EstimateReport (0.02), not the receipt's estimated_cost_usd (0.05)
    assert matched[0]["estimated_usd"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# COST-02 / D-10 — drift flagged only when abs(delta) > max(0.20*est, 0.01)
# ---------------------------------------------------------------------------


def test_drift_threshold(tmp_path: Path) -> None:
    """5% / sub-cent delta is NOT drifted; 30% above $0.01 IS drifted."""
    run_id = "test-run-03"

    # Receipt 1: actual=0.0105 vs estimated=0.01 → 5% delta, $0.0005 — NOT drifted
    write_provider_receipt(
        tmp_path,
        run_id,
        "stage-a",
        task_id="task-small",
        provider="openai",
        model="gpt-4o-mini",
        estimated_cost_usd=0.0105,
        cost_status="estimated",
    )
    # Receipt 2: actual=0.065 vs estimated=0.05 → 30% delta, $0.015 — DRIFTED
    write_provider_receipt(
        tmp_path,
        run_id,
        "stage-b",
        task_id="task-large",
        provider="anthropic",
        model="claude-sonnet-4-6",
        estimated_cost_usd=0.065,
        cost_status="estimated",
    )
    report = make_estimate_report(
        tasks=[
            {
                "task_id": "task-small",
                "family": "research",
                "stage": "stage-a",
                "provider": "openai",
                "model": "gpt-4o-mini",
                "pricing_unit": "tokens",
                "enabled": True,
                "quantity": 10000.0,
                "estimated_cost_usd": 0.01,
            },
            {
                "task_id": "task-large",
                "family": "research",
                "stage": "stage-b",
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "pricing_unit": "tokens",
                "enabled": True,
                "quantity": 10000.0,
                "estimated_cost_usd": 0.05,
            },
        ]
    )
    build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]

    small = next((l for l in lines if l.get("task_id") == "task-small"), None)
    large = next((l for l in lines if l.get("task_id") == "task-large"), None)

    assert small is not None, "task-small line missing"
    assert large is not None, "task-large line missing"
    assert small.get("drifted") is False, "5%/$0.0005 delta must NOT be flagged drifted"
    assert large.get("drifted") is True, "30%/$0.015 delta must be flagged drifted"


# ---------------------------------------------------------------------------
# COST-02 / D-09 — unknown cost_status flagged, does not raise
# ---------------------------------------------------------------------------


def test_unknown_cost_flagged_not_fatal(tmp_path: Path) -> None:
    """A receipt with cost_status='unknown' → line with cost_status='unknown' and
    needs_pricing_backfill=True. build_cost_rollup must not raise."""
    run_id = "test-run-04"
    write_provider_receipt(
        tmp_path,
        run_id,
        "stage-unknown",
        task_id="task-unknown",
        provider="openai",
        model="not-in-catalog",
        estimated_cost_usd=None,
        cost_status="unknown",
    )
    report = make_estimate_report()

    # Must not raise even with unknown cost
    result = build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]

    unknown_lines = [l for l in lines if l.get("cost_status") == "unknown"]
    assert unknown_lines, "expected at least one line with cost_status='unknown'"
    assert unknown_lines[0].get("needs_pricing_backfill") is True


# ---------------------------------------------------------------------------
# COST-02 / D-06 — unattributed bucket (no task_id)
# ---------------------------------------------------------------------------


def test_unattributed_bucket(tmp_path: Path) -> None:
    """A receipt with no task_id → line with task_id='unattributed', never dropped."""
    run_id = "test-run-05"
    # Write receipt with NO task_id field
    write_provider_receipt(
        tmp_path,
        run_id,
        "stage-x",
        # task_id intentionally omitted
        provider="openai",
        model="gpt-4o-mini",
        estimated_cost_usd=0.005,
        cost_status="estimated",
    )
    report = make_estimate_report()
    build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]

    unattributed = [l for l in lines if l.get("task_id") == "unattributed"]
    assert unattributed, "receipt with no task_id must appear in 'unattributed' bucket"


# ---------------------------------------------------------------------------
# COST-05 / D-13, D-14 — $0 local work lines
# ---------------------------------------------------------------------------


def test_local_zero_dollar_lines(tmp_path: Path) -> None:
    """Local-llm, local-embedding, local-vision produce distinct $0 lines with call counts."""
    run_id = "test-run-06"
    write_local_llm_receipt(tmp_path, run_id, "stage-llm", request_count=5)
    write_local_embedding_receipt(tmp_path, run_id, "stage-embed", request_count=3)
    write_local_vision_receipt(tmp_path, run_id, "stage-vision", image_count=2)
    report = make_estimate_report()
    build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]

    # All three local families must produce $0 lines
    llm_lines = [l for l in lines if l.get("provider") == "local_llm"]
    embed_lines = [l for l in lines if l.get("provider") == "local_embedding"]
    vision_lines = [l for l in lines if l.get("provider") == "local_vision"]

    assert llm_lines, "expected at least one local_llm line"
    assert embed_lines, "expected at least one local_embedding line"
    assert vision_lines, "expected at least one local_vision line"

    for l in llm_lines + embed_lines + vision_lines:
        assert l.get("actual_usd") == pytest.approx(0.0), f"local line must have actual_usd=0.0: {l}"
        assert l.get("cost_status") == "priced", f"local line must have cost_status='priced': {l}"
        assert l.get("call_count", 0) >= 1, f"local line must have call_count >= 1: {l}"


# ---------------------------------------------------------------------------
# COST-05 — summary.json cost mirror integration (Plan 04)
# ---------------------------------------------------------------------------

_REQUIRED_COST_SUMMARY_KEYS = {"estimated_usd", "actual_usd", "local_call_count", "has_unknown_cost"}

_STAGE_RECEIPT_DEFAULTS: dict = {
    "schema_version": 1,
    "run_id": "test-run",
    "workspace": "/tmp/ws",
    "stage_id": "stage-a",
    "stage_order": 1,
    "mode": "dry-run",
    "topic": None,
    "selected_scope": None,
    "presentation_manifest": None,
    "implementation_ref": "test",
    "status": "succeeded",
    "started_at": "2026-06-25T00:00:00Z",
    "ended_at": "2026-06-25T00:01:00Z",
    "resumable": False,
    "skip_reason": None,
    "resume_compatibility_key": "k",
    "config_digest": "d",
    "command_digest": "cd",
    "script_registry_sha256": "s",
    "required_script_ids": [],
    "commands": [],
    "command_metadata": [],
    "stage_contract": {},
    "artifacts": [],
    "provider_receipts": [],
    "provider_usage": {},
    "error": None,
    "next_resume_stage": None,
    "receipt_path": Path("/tmp/ws/storage/workflow-runs/test-run/stage-01-stage-a.json"),
}


def test_summary_cost_mirror(tmp_path: Path) -> None:
    """COST-05 full integration: StageReceipt.cost_summary + WorkflowRunSummary.run_cost_summary
    + cost_rollup_path are present in summary.json (via to_dict).

    Validates:
    - StageReceipt accepts optional cost_summary field (sibling to provider_usage)
    - StageReceipt.to_dict() includes "cost_summary" with the four required keys
    - WorkflowRunSummary accepts optional run_cost_summary and cost_rollup_path fields
    - WorkflowRunSummary.to_dict() includes both fields
    - provider_usage is unchanged (additive, not mutated)
    - The local $0 line is present (local_call_count > 0 in stage_summary)
    - The local $0 aggregate is present at the run level (run_cost_summary.local_call_count > 0)
    """
    run_id = "test-run-07"
    write_provider_receipt(
        tmp_path,
        run_id,
        "stage-a",
        task_id="task-01",
        provider="openai",
        model="gpt-4o-mini",
        estimated_cost_usd=0.01,
        cost_status="estimated",
    )
    write_local_llm_receipt(tmp_path, run_id, "stage-a", request_count=2)
    report = make_estimate_report()
    rollup = build_cost_rollup(tmp_path, run_id, estimate_report=report)

    run_cost_summary = build_run_cost_summary(rollup)
    stage_cost_summary = build_stage_cost_summary(rollup, "stage-a")

    # Helper contract: compact blocks have required keys
    assert _REQUIRED_COST_SUMMARY_KEYS.issubset(run_cost_summary.keys())
    assert _REQUIRED_COST_SUMMARY_KEYS.issubset(stage_cost_summary.keys())

    # D-13: the local $0 line contributes local_call_count to the stage and run summaries
    assert stage_cost_summary["local_call_count"] >= 1, "stage summary must count local llm calls"
    assert run_cost_summary["local_call_count"] >= 1, "run summary must count local llm calls"

    # --- Full StageReceipt integration ---
    # StageReceipt must accept an optional cost_summary field (sibling to provider_usage)
    stage_receipt = StageReceipt(
        **{**_STAGE_RECEIPT_DEFAULTS, "cost_summary": stage_cost_summary},
    )
    receipt_dict = stage_receipt.to_dict()
    assert "cost_summary" in receipt_dict, "StageReceipt.to_dict() must include 'cost_summary'"
    cs = receipt_dict["cost_summary"]
    assert _REQUIRED_COST_SUMMARY_KEYS.issubset(cs.keys()), (
        f"cost_summary missing keys: {_REQUIRED_COST_SUMMARY_KEYS - cs.keys()}"
    )
    # provider_usage must be unchanged (additive — not mutated)
    assert "cost_summary" not in receipt_dict["provider_usage"], (
        "cost_summary must be a sibling of provider_usage, not nested inside it"
    )

    # --- Full WorkflowRunSummary integration ---
    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    assert rollup_path.exists(), "cost_rollup.json must exist before building WorkflowRunSummary"

    summary_path = tmp_path / "storage" / "workflow-runs" / run_id / "summary.json"
    run_summary = WorkflowRunSummary(
        schema_version=1,
        run_id=run_id,
        workspace=str(tmp_path),
        mode="dry-run",
        topic=None,
        presentation_type=None,
        started_at="2026-06-25T00:00:00Z",
        ended_at="2026-06-25T00:01:00Z",
        summary_path=summary_path,
        next_resume_stage=None,
        stages=[stage_receipt],
        run_cost_summary=run_cost_summary,
        cost_rollup_path=rollup_path,
    )
    summary_dict = run_summary.to_dict()
    assert "run_cost_summary" in summary_dict, (
        "WorkflowRunSummary.to_dict() must include 'run_cost_summary'"
    )
    assert _REQUIRED_COST_SUMMARY_KEYS.issubset(summary_dict["run_cost_summary"].keys()), (
        "run_cost_summary must have estimated_usd, actual_usd, local_call_count, has_unknown_cost"
    )
    assert "cost_rollup_path" in summary_dict, (
        "WorkflowRunSummary.to_dict() must include 'cost_rollup_path'"
    )
    assert summary_dict["cost_rollup_path"] is not None, (
        "cost_rollup_path must reference the run's cost_rollup.json"
    )
    # The cost_rollup_path must reference a cost_rollup.json
    assert str(summary_dict["cost_rollup_path"]).endswith("cost_rollup.json"), (
        f"cost_rollup_path must point at cost_rollup.json, got: {summary_dict['cost_rollup_path']}"
    )


# ---------------------------------------------------------------------------
# v3.0 03-04: cost-avoided aggregation in rollup (LOCAL-04 / D-13..D-16)
# ---------------------------------------------------------------------------


def test_local_lines_include_cost_avoided(tmp_path: Path) -> None:
    """D-15: local-family rollup lines include cost_avoided_usd, cost_avoided_status,
    needs_pricing_backfill, and benchmark_receipt_refs.  Run totals include
    total_cost_avoided_usd for priced savings."""
    run_id = "test-run-ca-01"

    # Write local embedding receipt with priced cost-avoided.
    write_local_embedding_receipt(
        tmp_path,
        run_id,
        "embed-stage",
        model="mlx-community/bge-m3-mlx-8bit",
        request_count=2,
        cost_usd=0.0,
        cost_avoided_usd=0.0024,
        cost_avoided_status="priced",
        benchmark_receipt_ref="storage/benchmarks/embed-bge-m3-20260625.json",
        needs_pricing_backfill=False,
    )
    # Write local vision receipt with priced cost-avoided.
    write_local_vision_receipt(
        tmp_path,
        run_id,
        "vision-stage",
        model="Qwen3-VL-8B-Instruct-4bit",
        image_count=1,
        cost_usd=0.0,
        cost_avoided_usd=0.05,
        cost_avoided_status="priced",
        benchmark_receipt_ref="storage/benchmarks/vlm-qwen3-ocr-20260625.json",
        needs_pricing_backfill=False,
    )

    report = make_estimate_report()
    result = build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]
    run_totals = payload["run_totals"]

    # Local embedding line must include cost-avoided fields.
    embed_lines = [l for l in lines if l.get("provider") == "local_embedding"]
    assert embed_lines, "expected a local_embedding line"
    embed_line = embed_lines[0]
    assert "cost_avoided_usd" in embed_line, "local_embedding line must have cost_avoided_usd"
    assert embed_line["cost_avoided_usd"] == pytest.approx(0.0024)
    assert embed_line.get("cost_avoided_status") == "priced"
    assert embed_line.get("needs_pricing_backfill") is False
    assert "benchmark_receipt_refs" in embed_line
    assert len(embed_line["benchmark_receipt_refs"]) >= 1

    # Local vision line must include cost-avoided fields.
    vision_lines = [l for l in lines if l.get("provider") == "local_vision"]
    assert vision_lines, "expected a local_vision line"
    vision_line = vision_lines[0]
    assert vision_line.get("cost_avoided_usd") == pytest.approx(0.05)
    assert vision_line.get("cost_avoided_status") == "priced"

    # Run totals must include total_cost_avoided_usd and has_unknown_cost_avoided.
    assert "total_cost_avoided_usd" in run_totals, "run_totals must include total_cost_avoided_usd"
    assert run_totals["total_cost_avoided_usd"] == pytest.approx(0.0024 + 0.05)
    assert "has_unknown_cost_avoided" in run_totals
    assert run_totals["has_unknown_cost_avoided"] is False
    assert "cost_avoided_backfill_count" in run_totals
    assert run_totals["cost_avoided_backfill_count"] == 0


def test_unknown_cost_avoided_flags_backfill_without_fake_savings(tmp_path: Path) -> None:
    """D-16: when any contributing local receipt has unknown cost_avoided, the rollup
    line must have cost_avoided_status='unknown' and needs_pricing_backfill=True.
    Unknown savings must never be coerced to $0 or a guessed number."""
    run_id = "test-run-ca-02"

    # One priced + one unknown contribution in the same stage.
    write_local_embedding_receipt(
        tmp_path,
        run_id,
        "mixed-stage",
        model="mlx-community/bge-m3-mlx-8bit",
        request_count=1,
        cost_avoided_usd=0.005,
        cost_avoided_status="priced",
        benchmark_receipt_ref="storage/benchmarks/bge-m3.json",
        needs_pricing_backfill=False,
    )
    write_local_embedding_receipt(
        tmp_path,
        run_id,
        "mixed-stage",
        model="mlx-community/bge-m3-mlx-8bit",
        request_count=1,
        cost_avoided_usd=None,  # D-16: unknown must stay None
        cost_avoided_status="unknown",
        needs_pricing_backfill=True,
    )

    report = make_estimate_report()
    build_cost_rollup(tmp_path, run_id, estimate_report=report)

    rollup_path = tmp_path / "storage" / "workflow-runs" / run_id / "cost_rollup.json"
    payload = json.loads(rollup_path.read_text(encoding="utf-8"))
    lines = payload["lines"]
    run_totals = payload["run_totals"]

    embed_lines = [l for l in lines if l.get("provider") == "local_embedding"]
    assert embed_lines, "expected local_embedding lines"

    # The line(s) from the mixed stage should flag unknown.
    mixed_lines = [l for l in embed_lines if l.get("stage_id") == "mixed-stage"]
    assert mixed_lines, "expected a line for mixed-stage"
    mixed_line = mixed_lines[0]

    # D-16: when any receipt is unknown, the aggregate status is unknown.
    assert mixed_line.get("cost_avoided_status") == "unknown"
    assert mixed_line.get("needs_pricing_backfill") is True
    # The unknown receipt's savings must NOT be coerced to $0; the aggregate
    # cost_avoided_usd should only count the priced portion or be None.
    # (Implementation may choose to count priced contributions and mark unknown;
    # the critical invariant is that it is not $0 total when unknown receipts exist.)
    avoided_usd = mixed_line.get("cost_avoided_usd")
    assert avoided_usd != 0.0, (
        "cost_avoided_usd must not be coerced to $0 when an unknown receipt contributes"
    )

    # Run totals: has_unknown_cost_avoided must be True; backfill_count >= 1.
    assert run_totals.get("has_unknown_cost_avoided") is True
    assert run_totals.get("cost_avoided_backfill_count", 0) >= 1
