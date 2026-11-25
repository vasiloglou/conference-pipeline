"""COST-04 budget-enforcement test suite.

Five tests pinned by name per the VALIDATION contract:
  test_preflight_halt          — D-02 preflight: estimate pushes cumulative spend over cap
  test_postcall_halt           — D-02 post-call: recorded actual crosses cap → halt
  test_unpriced_fail_closed    — D-04: unpriced call charged at estimate/ceiling; halts on breach
  test_halt_never_raises_cap   — D-03 / T-02-06: breach+halt never mutates cap constants
  test_unblock_resumes         — D-03 / T-02-08: UnblockReceipt(manual_ack) clears halt

All tests are OFFLINE (tmp_path, no live providers, no network).
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from conference_pipeline.costs.budget_enforcement import (  # noqa: F401
    BudgetCapBreached,
    CapResult,
    check_postcall,
    check_preflight,
)


# ---------------------------------------------------------------------------
# COST-04 / D-02: preflight halt
# ---------------------------------------------------------------------------

def test_preflight_halt(tmp_path: Path) -> None:
    """A task whose estimate would push cumulative spend over the cap halts BEFORE it runs.

    This test verifies:
    - check_preflight returns CapResult(breach=True) when existing + estimate > global cap
    - The CapResult carries meaningful reason, existing_usd, charged_usd, cap_usd fields
    """
    from conference_pipeline.costs.budget_enforcement import (
        UNKNOWN_COST_CEILING_USD,
        check_preflight,
    )
    from conference_pipeline.presentation.blueprint_generation import (
        GLOBAL_PROVIDER_BUDGET_CAP_USD,
    )
    from tests.fixtures.cost.receipts import write_provider_receipt

    run_id = "test-run-preflight"
    # Write a receipt that puts existing spend just below the cap
    near_cap = GLOBAL_PROVIDER_BUDGET_CAP_USD - 1.0
    write_provider_receipt(
        tmp_path,
        run_id,
        stage_id="stage-a",
        task_id="task-a",
        provider="openai",
        model="gpt-4o",
        estimated_cost_usd=near_cap,
        cost_status="estimated",
    )

    # A new task estimate of 5.0 would push total over the 800 cap
    result = check_preflight(tmp_path, task_id="task-b", estimated_cost_usd=5.0)

    assert isinstance(result, CapResult), "check_preflight must return a CapResult"
    assert result.breach is True, "should detect breach when existing + estimate > cap"
    assert result.cap_usd == GLOBAL_PROVIDER_BUDGET_CAP_USD
    assert result.existing_usd >= near_cap - 0.001
    assert "cap" in result.reason.lower() or "budget" in result.reason.lower()


def test_r6_ledger_reserves_settles_and_prevents_duplicate_or_unsettled_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda workspace: 10.0)
    registry = type("Registry", (), {"get": lambda self, task_id: type("Policy", (), {"budget_cap_usd": 200.0})()})()
    ledger = tmp_path / "r6-ledger.json"

    reserved = budget_enforcement.reserve_r6_budget(
        tmp_path,
        ledger,
        run_id="r6",
        task_id="transcription.session_speech",
        item_id="item-1",
        benchmark_id="benchmark-1",
        estimate_id="estimate-1",
        maximum_charge_usd=5.0,
        registry=registry,
    )
    assert reserved["state"] == "reserved"
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="unsettled|duplicate"):
        budget_enforcement.reserve_r6_budget(
            tmp_path, ledger, run_id="r6", task_id="transcription.session_speech",
            item_id="item-2", benchmark_id="benchmark-1", estimate_id="estimate-1",
            maximum_charge_usd=5.0, registry=registry,
        )
    settled = budget_enforcement.settle_r6_budget(
        tmp_path,
        ledger,
        reservation_id=reserved["reservation_id"],
        sanitized_provider_receipt={
            "reservation_id": reserved["reservation_id"],
            "actual_cost_usd": 3.0,
            "sanitized": True,
        },
        registry=registry,
    )
    assert settled["state"] == "settled"
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="duplicate|already"):
        budget_enforcement.reserve_r6_budget(
            tmp_path, ledger, run_id="r6", task_id="transcription.session_speech",
            item_id="item-1", benchmark_id="benchmark-1", estimate_id="estimate-1",
            maximum_charge_usd=5.0, registry=registry,
        )


@pytest.mark.parametrize(
    ("task_id", "cap"),
    [("transcription.session_speech", 200.0), ("ocr.slide_image_parse", 30.0)],
)
def test_r6_ledger_enforces_global_and_task_cumulative_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    cap: float,
) -> None:
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda workspace: 799.0)
    registry = type("Registry", (), {"get": lambda self, wanted: type("Policy", (), {"budget_cap_usd": cap})()})()
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="global|task"):
        budget_enforcement.reserve_r6_budget(
            tmp_path,
            tmp_path / "r6-ledger.json",
            run_id="r6",
            task_id=task_id,
            item_id="item-1",
            benchmark_id="benchmark-1",
            estimate_id="estimate-1",
            maximum_charge_usd=2.0,
            registry=registry,
        )


def test_r6_ledger_rejects_task_overage_missing_benchmark_bad_receipt_and_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda workspace: 0.0)
    registry = type("Registry", (), {"get": lambda self, task_id: type("Policy", (), {"budget_cap_usd": 30.0})()})()
    ledger = tmp_path / "r6-ledger.json"

    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="task"):
        budget_enforcement.reserve_r6_budget(
            tmp_path, ledger, run_id="r6", task_id="ocr.slide_image_parse", item_id="item-1",
            benchmark_id="benchmark-1", estimate_id="estimate-1", maximum_charge_usd=31.0,
            registry=registry,
        )
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="benchmark"):
        budget_enforcement.reserve_r6_budget(
            tmp_path, ledger, run_id="r6", task_id="ocr.slide_image_parse", item_id="item-1",
            benchmark_id="", estimate_id="estimate-1", maximum_charge_usd=1.0, registry=registry,
        )

    reservation = budget_enforcement.reserve_r6_budget(
        tmp_path, ledger, run_id="r6", task_id="ocr.slide_image_parse", item_id="item-1",
        benchmark_id="benchmark-1", estimate_id="estimate-1", maximum_charge_usd=5.0, registry=registry,
    )
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="sanitization"):
        budget_enforcement.settle_r6_budget(
            tmp_path, ledger, reservation_id=reservation["reservation_id"],
            sanitized_provider_receipt={"reservation_id": reservation["reservation_id"], "actual_cost_usd": 1.0},
            registry=registry,
        )
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="finite"):
        budget_enforcement.settle_r6_budget(
            tmp_path, ledger, reservation_id=reservation["reservation_id"],
            sanitized_provider_receipt={"reservation_id": reservation["reservation_id"], "actual_cost_usd": float("nan"), "sanitized": True},
            registry=registry,
        )
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="exceeds"):
        budget_enforcement.settle_r6_budget(
            tmp_path, ledger, reservation_id=reservation["reservation_id"],
            sanitized_provider_receipt={"reservation_id": reservation["reservation_id"], "actual_cost_usd": 6.0, "sanitized": True},
            registry=registry,
        )

    payload = json.loads(ledger.read_text())
    payload["reservations"][0]["maximum_charge_usd"] = 1.0
    ledger.write_text(json.dumps(payload))
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="digest drift"):
        budget_enforcement.settle_r6_budget(
            tmp_path, ledger, reservation_id=reservation["reservation_id"],
            sanitized_provider_receipt={"reservation_id": reservation["reservation_id"], "actual_cost_usd": 1.0, "sanitized": True},
            registry=registry,
        )


def test_r6_ledger_settlement_tolerates_spend_growth_but_rejects_apparent_erasure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """canonical_prior_spend_usd is a write-once snapshot, not a live equality
    gate against ``_existing_spend``. A real provider call — success or
    failure — lands in the workspace-wide receipt log that ``_existing_spend``
    scans *before* ``settle_r6_budget`` re-derives it, so spend legitimately
    grows between ``reserve_r6_budget`` and ``settle_r6_budget``. Only a
    later read where existing spend has apparently *dropped* below the
    recorded baseline is tamper evidence (spend cannot be un-spent).
    """
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    spend = {"value": 10.0}
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda workspace: spend["value"])
    registry = type("Registry", (), {"get": lambda self, task_id: type("Policy", (), {"budget_cap_usd": 200.0})()})()
    ledger = tmp_path / "r6-ledger.json"

    reserved = budget_enforcement.reserve_r6_budget(
        tmp_path, ledger, run_id="r6", task_id="transcription.session_speech",
        item_id="item-1", benchmark_id="benchmark-1", estimate_id="estimate-1",
        maximum_charge_usd=5.0, registry=registry,
    )

    spend["value"] = 10.03  # the just-made call's own cost already landed in the workspace log
    settled = budget_enforcement.settle_r6_budget(
        tmp_path, ledger, reservation_id=reserved["reservation_id"],
        sanitized_provider_receipt={
            "reservation_id": reserved["reservation_id"], "actual_cost_usd": 3.0, "sanitized": True,
        },
        registry=registry,
    )
    assert settled["state"] == "settled"

    spend["value"] = 1.0  # apparent decrease below the recorded baseline: fail closed
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="drift"):
        budget_enforcement.reserve_r6_budget(
            tmp_path, ledger, run_id="r6", task_id="transcription.session_speech",
            item_id="item-2", benchmark_id="benchmark-1", estimate_id="estimate-2",
            maximum_charge_usd=5.0, registry=registry,
        )


def test_r6_ledger_conservatively_settles_missing_actual_at_reserved_charge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda workspace: 0.0)
    registry = type("Registry", (), {"get": lambda self, task_id: type("Policy", (), {"budget_cap_usd": 30.0})()})()
    reservation = budget_enforcement.reserve_r6_budget(
        tmp_path,
        tmp_path / "r6-ledger.json",
        run_id="r6",
        task_id="ocr.slide_image_parse",
        item_id="item-1",
        benchmark_id="benchmark-1",
        estimate_id="estimate-1",
        maximum_charge_usd=5.0,
        registry=registry,
    )

    settled = budget_enforcement.settle_r6_budget(
        tmp_path,
        tmp_path / "r6-ledger.json",
        reservation_id=reservation["reservation_id"],
        sanitized_provider_receipt={
            "reservation_id": reservation["reservation_id"],
            "sanitized": True,
            "status": "failed",
        },
        registry=registry,
    )

    assert settled["state"] == "settled"
    assert settled["actual_cost_usd"] == 5.0
    assert settled["actual_cost_source"] == "conservative_reservation"


def test_r6_schedule_bound_reservations_reject_reorder_replay_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite schedule, not a task-wide amount, is the paid-call authority."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        build_r6_dispatch_schedule,
        canonical_r6_queue,
        r6_digest,
        validate_r6_dispatch_schedule,
    )
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda _: 0.0)
    registry = type(
        "Registry",
        (),
        {
            "get": lambda self, task_id: type(
                "Policy", (), {"budget_cap_usd": 200.0 if task_id.startswith("transcription") else 30.0}
            )()
        },
    )()
    queue, queue_digest = canonical_r6_queue(
        speech_ids=["talk-1"], ocr_ids=["slide-1"]
    )
    schedule = build_r6_dispatch_schedule(
        run_id="cvpr2026-live-production-r6-expanded-media",
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=2.0,
        ocr_total_usd=1.0,
    )
    entries = schedule["entries"]
    assert isinstance(entries, list) and len(entries) == 4
    ledger = tmp_path / "r6-ledger.json"

    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="missing, reordered, or changed"):
        budget_enforcement.reserve_r6_scheduled_dispatch(
            tmp_path, ledger, schedule=schedule, entry=entries[1], registry=registry
        )
    first = budget_enforcement.reserve_r6_scheduled_dispatch(
        tmp_path, ledger, schedule=schedule, entry=entries[0], registry=registry
    )
    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="unsettled"):
        budget_enforcement.reserve_r6_scheduled_dispatch(
            tmp_path, ledger, schedule=schedule, entry=entries[1], registry=registry
        )
    budget_enforcement.settle_r6_budget(
        tmp_path,
        ledger,
        reservation_id=first["reservation_id"],
        sanitized_provider_receipt={
            "reservation_id": first["reservation_id"],
            "actual_cost_usd": first["maximum_charge_usd"],
            "sanitized": True,
        },
        registry=registry,
    )
    second = budget_enforcement.reserve_r6_scheduled_dispatch(
        tmp_path, ledger, schedule=schedule, entry=entries[1], registry=registry
    )
    assert second["ordinal"] == 1

    tampered = {**schedule, "entries": [dict(entry) for entry in entries]}
    tampered["entries"][2]["maximum_charge_usd"] = 99.0
    with pytest.raises(Exception, match="digest|fingerprint|totals"):
        validate_r6_dispatch_schedule(tampered)

    omitted = {**schedule, "entries": [dict(entry) for entry in entries[:-1]]}
    omitted["totals"] = {
        "all_in_usd": sum(float(entry["maximum_charge_usd"]) for entry in omitted["entries"]),
        "speech_usd": sum(
            float(entry["maximum_charge_usd"])
            for entry in omitted["entries"]
            if entry["task_id"] == "transcription.session_speech"
        ),
        "ocr_usd": sum(
            float(entry["maximum_charge_usd"])
            for entry in omitted["entries"]
            if entry["task_id"] == "ocr.slide_image_parse"
        ),
    }
    omitted["dispatch_schedule_digest"] = r6_digest(
        {key: value for key, value in omitted.items() if key != "dispatch_schedule_digest"}
    )
    with pytest.raises(Exception, match="omitted|queue/order"):
        validate_r6_dispatch_schedule(omitted)


def test_r6_ledger_validator_rejects_refreshed_digest_with_changed_actual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A digest refresh cannot conceal an inconsistent settled actual row."""

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (
        build_r6_dispatch_schedule,
        canonical_r6_queue,
    )
    from conference_pipeline.costs import budget_enforcement

    monkeypatch.setattr(budget_enforcement, "_global_cap", lambda: 800.0)
    monkeypatch.setattr(budget_enforcement, "_existing_spend", lambda _: 0.0)
    registry = type(
        "Registry",
        (),
        {"get": lambda self, _task: type("Policy", (), {"budget_cap_usd": 200.0})()},
    )()
    queue, queue_digest = canonical_r6_queue(speech_ids=["talk-1"], ocr_ids=[])
    schedule = build_r6_dispatch_schedule(
        run_id="cvpr2026-live-production-r6-expanded-media",
        queue=queue,
        queue_digest=queue_digest,
        speech_total_usd=2.0,
        ocr_total_usd=0.0,
    )
    ledger = tmp_path / "r6-ledger.json"
    reservation = budget_enforcement.reserve_r6_scheduled_dispatch(
        tmp_path, ledger, schedule=schedule, entry=schedule["entries"][0], registry=registry
    )
    budget_enforcement.settle_r6_budget(
        tmp_path,
        ledger,
        reservation_id=reservation["reservation_id"],
        sanitized_provider_receipt={
            "reservation_id": reservation["reservation_id"],
            "actual_cost_usd": reservation["maximum_charge_usd"],
            "sanitized": True,
        },
        registry=registry,
    )
    payload = json.loads(ledger.read_text())
    payload["actuals"][0]["actual_cost_usd"] = 0.0
    payload["ledger_sha256"] = budget_enforcement._r6_digest(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )
    ledger.write_text(json.dumps(payload))

    with pytest.raises(budget_enforcement.R6BudgetLedgerError, match="actual"):
        budget_enforcement.validate_r6_budget_ledger(
            tmp_path, ledger, schedule=schedule, registry=registry
        )
# ---------------------------------------------------------------------------
# COST-04 / D-02: post-call halt
# ---------------------------------------------------------------------------

def test_postcall_halt(tmp_path: Path) -> None:
    """A recorded actual that crosses the cap triggers a post-call breach.

    This test verifies:
    - check_postcall returns CapResult(breach=True) when recorded actuals exceed cap
    """
    from conference_pipeline.costs.budget_enforcement import check_postcall
    from conference_pipeline.presentation.blueprint_generation import (
        GLOBAL_PROVIDER_BUDGET_CAP_USD,
    )
    from tests.fixtures.cost.receipts import write_provider_receipt

    run_id = "test-run-postcall"
    # Existing spend just below cap
    near_cap = GLOBAL_PROVIDER_BUDGET_CAP_USD - 0.5
    write_provider_receipt(
        tmp_path,
        run_id,
        stage_id="stage-b",
        task_id="task-b",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        estimated_cost_usd=near_cap,
        cost_status="estimated",
    )

    # Post-call: we record an actual of 2.0 (total > cap)
    result = check_postcall(tmp_path, run_id=run_id, actual_cost_usd=2.0)

    assert isinstance(result, CapResult), "check_postcall must return a CapResult"
    assert result.breach is True, "should detect breach when actual pushes total over cap"
    assert result.cap_usd == GLOBAL_PROVIDER_BUDGET_CAP_USD


# ---------------------------------------------------------------------------
# COST-04 / D-04: unpriced fail-closed
# ---------------------------------------------------------------------------

def test_unpriced_fail_closed(tmp_path: Path) -> None:
    """An unpriced call (cost is None) is charged at max(task_estimate, UNKNOWN_COST_CEILING_USD).

    This test verifies:
    - check_preflight with estimated_cost_usd=None charges at UNKNOWN_COST_CEILING_USD
    - If that charge breaches the cap, CapResult(breach=True) is returned
    - An unpriced call is NEVER silently skipped (D-04)
    """
    from conference_pipeline.costs.budget_enforcement import (
        UNKNOWN_COST_CEILING_USD,
        check_preflight,
    )
    from conference_pipeline.presentation.blueprint_generation import (
        GLOBAL_PROVIDER_BUDGET_CAP_USD,
    )
    from tests.fixtures.cost.receipts import write_provider_receipt

    run_id = "test-run-unpriced"
    # Existing spend near cap so that ceiling charge would push it over
    near_cap = GLOBAL_PROVIDER_BUDGET_CAP_USD - (UNKNOWN_COST_CEILING_USD / 2.0)
    write_provider_receipt(
        tmp_path,
        run_id,
        stage_id="stage-c",
        task_id="task-c",
        provider="openai",
        model="gpt-4o",
        estimated_cost_usd=near_cap,
        cost_status="estimated",
    )

    # estimated_cost_usd=None simulates an unpriced call
    result = check_preflight(tmp_path, task_id="task-d", estimated_cost_usd=None)

    assert isinstance(result, CapResult), "check_preflight must return a CapResult"
    # The unpriced call must be charged at UNKNOWN_COST_CEILING_USD (fail-closed)
    assert result.charged_usd >= UNKNOWN_COST_CEILING_USD, (
        f"unpriced call must be charged at least UNKNOWN_COST_CEILING_USD "
        f"({UNKNOWN_COST_CEILING_USD}), got {result.charged_usd}"
    )
    assert result.breach is True, (
        "unpriced call charged at ceiling should breach cap when existing spend is near cap"
    )


# ---------------------------------------------------------------------------
# COST-04 / D-03 / T-02-06: halt never raises cap
# ---------------------------------------------------------------------------

def test_halt_never_raises_cap(tmp_path: Path) -> None:
    """After a breach+halt, GLOBAL_PROVIDER_BUDGET_CAP_USD and TaskPolicy.budget_cap_usd are unchanged.

    This test is the locked never-auto-raise invariant (T-02-06, D-03).
    It asserts that:
    - check_preflight returning breach=True does NOT modify the cap constants
    - The module-level GLOBAL_PROVIDER_BUDGET_CAP_USD is byte-for-byte unchanged
    - The grep gate (no assignment to cap constants in budget_enforcement.py) holds
    """
    import re

    from conference_pipeline.costs.budget_enforcement import check_preflight
    from conference_pipeline.presentation import blueprint_generation
    from conference_pipeline.presentation.blueprint_generation import (
        GLOBAL_PROVIDER_BUDGET_CAP_USD,
    )
    from tests.fixtures.cost.receipts import write_provider_receipt

    # Record the cap value before any breach
    cap_before = GLOBAL_PROVIDER_BUDGET_CAP_USD
    module_cap_before = blueprint_generation.GLOBAL_PROVIDER_BUDGET_CAP_USD

    run_id = "test-run-never-raise"
    # Drive existing spend above cap so breach fires
    write_provider_receipt(
        tmp_path,
        run_id,
        stage_id="stage-d",
        task_id="task-e",
        provider="openai",
        model="gpt-4o",
        estimated_cost_usd=GLOBAL_PROVIDER_BUDGET_CAP_USD + 100.0,
        cost_status="estimated",
    )
    result = check_preflight(tmp_path, task_id="task-f", estimated_cost_usd=1.0)
    assert result.breach is True, "test setup: breach must fire"

    # Cap must be byte-for-byte unchanged after breach
    assert blueprint_generation.GLOBAL_PROVIDER_BUDGET_CAP_USD == cap_before, (
        "GLOBAL_PROVIDER_BUDGET_CAP_USD was mutated after breach — LOCKED invariant violated"
    )
    assert blueprint_generation.GLOBAL_PROVIDER_BUDGET_CAP_USD == module_cap_before

    # Grep gate: budget_enforcement.py must have 0 lines that ASSIGN to cap constants
    enforcement_src = Path(__file__).parent.parent / "src" / "conference_pipeline" / "costs" / "budget_enforcement.py"
    non_comment_lines = [
        line for line in enforcement_src.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    assignment_hits = [
        line for line in non_comment_lines
        if re.search(r"GLOBAL_PROVIDER_BUDGET_CAP_USD\s*=|budget_cap_usd\s*=", line)
    ]
    assert assignment_hits == [], (
        f"budget_enforcement.py must NOT assign to cap constants; found: {assignment_hits}"
    )


# ---------------------------------------------------------------------------
# COST-04 / D-03 / T-02-08: unblock resumes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WR-02: per-task cap fires at workflow gate / registry-load fail-closed
# ---------------------------------------------------------------------------


def test_per_task_cap_fires_for_mapped_stage(tmp_path: Path) -> None:
    """WR-02 Test 1: per-task budget cap fires for a real mapped stage.

    The 'podcast' workflow stage maps to registry task 'podcast.tts' (cap=$60).
    With the flat stage id 'podcast', _per_task_cap returns None (no registry match)
    and the per-task cap never fires — this is the current bug.

    After the fix, workflow_runs.py maps 'podcast' -> 'podcast.tts' before calling
    check_preflight. We verify: (a) flat 'podcast' id returns None from _per_task_cap
    (confirming the bug still exists at _per_task_cap level — mapping is at workflow gate),
    and (b) the dotted 'podcast.tts' id DOES fire the per-task cap.

    MUST fail on current code for assertion (b): the workflow gate still passes flat id
    so per-task cap never fires in practice.
    """
    from conference_pipeline.costs.budget_enforcement import _per_task_cap, check_preflight
    from conference_pipeline.model_governance.registry import load_registry

    registry = load_registry()

    # (a) Flat stage id returns None — per-task cap unreachable today at gateway
    flat_cap = _per_task_cap("podcast", registry)
    assert flat_cap is None, (
        "flat stage id 'podcast' must be unknown to registry (returns None) — "
        "confirming per-task cap is unreachable before the workflow gate mapping fix"
    )

    # (b) After fix: workflow gate maps 'podcast' -> 'podcast.tts'; verify dotted id fires cap.
    # podcast.tts cap is $60; global cap is $800; charging $61 must breach per-task cap.
    result = check_preflight(
        tmp_path,
        task_id="podcast.tts",  # the mapped dotted registry id the workflow gate will pass after fix
        estimated_cost_usd=61.0,
        registry=registry,
    )

    assert result.breach is True, (
        "charging $61 > podcast.tts cap ($60) must breach the per-task cap"
    )
    assert "task" in result.reason.lower() or "cap" in result.reason.lower(), (
        f"reason must mention the per-task cap; got: {result.reason!r}"
    )


def test_registry_load_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """WR-02 Test 2: a registry LOAD failure must raise (fail closed), not return None.

    Monkeypatches load_registry to raise RuntimeError.
    MUST fail on current code: the blanket except swallows load errors and returns None.
    """
    import pytest

    from conference_pipeline.costs.budget_enforcement import _per_task_cap

    def _bad_load(*args: object, **kwargs: object) -> None:
        raise RuntimeError("corrupt registry — test-injected failure")

    monkeypatch.setattr(
        "conference_pipeline.model_governance.registry.load_registry",
        _bad_load,
    )
    monkeypatch.setattr(
        "conference_pipeline.costs.budget_enforcement.load_registry",
        _bad_load,
        raising=False,
    )

    with pytest.raises((RuntimeError, Exception)) as exc_info:
        _per_task_cap("podcast.tts", None)  # registry=None triggers load

    assert "corrupt registry" in str(exc_info.value) or isinstance(exc_info.value, RuntimeError), (
        "registry load failure must propagate as an exception, not return None silently"
    )


def test_unknown_task_stays_uncapped(tmp_path: Path) -> None:
    """WR-02 Test 3: an unknown/not-found task id remains legitimately uncapped (returns None).

    Uses a real registry; 'does.not.exist' is not in registry.yaml.
    Must preserve the existing uncapped behavior for unknown task ids.
    """
    from conference_pipeline.costs.budget_enforcement import _per_task_cap
    from conference_pipeline.model_governance.registry import load_registry

    registry = load_registry()
    result = _per_task_cap("does.not.exist", registry)

    assert result is None, (
        "unknown task id must return None (legitimately uncapped), not raise"
    )


def test_unblock_resumes(tmp_path: Path) -> None:
    """UnblockReceipt(manual_ack, operator_id) clears active_blockers and flips halt_status to 'resumed'.

    This test verifies the operator-ack path (T-02-08):
    - After a budget-cap halt, active_blockers is non-empty
    - Writing an UnblockReceipt(unblock_method="manual_ack", operator_id="op-test") clears them
    - persist_summary then flips halt_status to "resumed"
    """
    import json

    from conference_pipeline.provider_receipts import (
        BLOCKER_RECEIPT_MARKER,
        BLOCKER_RECEIPT_SCHEMA_VERSION,
        UNBLOCK_RECEIPT_MARKER,
        UNBLOCK_RECEIPT_SCHEMA_VERSION,
        BlockerClass,
        BlockerReceipt,
        UnblockReceipt,
        active_blockers,
        write_blocker_receipt,
        write_unblock_receipt,
    )

    run_id = "test-run-unblock"

    # Simulate a budget-cap halt by writing a BlockerReceipt
    blocker = BlockerReceipt(
        schema_version=BLOCKER_RECEIPT_SCHEMA_VERSION,
        receipt_type=BLOCKER_RECEIPT_MARKER,
        blocker_class=BlockerClass.BUDGET_CAP_EXCEEDED,
        stage_id="stage-e",
        reason="simulated budget cap breach for test",
        remediation_path="config/provider_cost.budget_cap_usd",
        automated_unblock_available=False,
        endpoint_label=None,
        timestamp="2026-06-25T00:00:00Z",
        run_id=run_id,
    )
    blocker_path = write_blocker_receipt(blocker, tmp_path)
    assert blocker_path.exists(), "blocker receipt must be written"

    # Before unblock: active_blockers should return the blocker
    blockers_before = active_blockers(tmp_path, run_id=run_id)
    assert len(blockers_before) >= 1, "active_blockers must report the budget cap blocker"

    # Write the unblock receipt (manual_ack by operator)
    unblock = UnblockReceipt(
        schema_version=UNBLOCK_RECEIPT_SCHEMA_VERSION,
        receipt_type=UNBLOCK_RECEIPT_MARKER,
        blocker_receipt_path=str(blocker_path),
        unblock_method="manual_ack",
        operator_id="op-test",
        note="operator reviewed budget cap breach and approved resumption",
        timestamp="2026-06-25T00:01:00Z",
        run_id=run_id,
    )
    write_unblock_receipt(unblock, tmp_path)

    # After unblock: active_blockers should be empty for this run
    blockers_after = active_blockers(tmp_path, run_id=run_id)
    assert blockers_after == [], (
        f"active_blockers must be empty after UnblockReceipt(manual_ack); got {blockers_after}"
    )
