"""Budget enforcement core for the conference pipeline.

Provides a shared ledger + two-point cap check:
  - check_preflight: called BEFORE a stage/task runs (D-01, D-02)
  - check_postcall:  called AFTER actuals are recorded (D-02)

Unpriced calls (estimated_cost_usd is None) are charged at
max(task_estimate, UNKNOWN_COST_CEILING_USD) — fail-closed per D-04
(generalizes BudgetTracker.record from run_figure_trial.py).

On breach the caller invokes the existing v1.9 halt_run path with
BlockerClass.BUDGET_CAP_EXCEEDED — this module NEVER builds a new halt
mechanism and NEVER mutates cap constants (D-03, T-02-06, locked invariant).

Cap convergence (D-01): GLOBAL_PROVIDER_BUDGET_CAP_USD is imported from
blueprint_generation — it is NOT redefined here so there is exactly one source
of truth.  Per-task caps come from TaskPolicy.budget_cap_usd via load_registry.

COST-04 deliverable.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# D-01: single source of cap truth — import, never redefine.
# Import is deferred to avoid a hard dependency cycle at module import time;
# use _global_cap() accessor below.
_GLOBAL_CAP_CACHE: float | None = None

# Conservative ceiling for unpriced calls (D-04 / A2).
# Charged when estimated_cost_usd is None — prevents the cap from being a no-op
# on calls that lack cost metadata.  Generous enough to be conservative.
UNKNOWN_COST_CEILING_USD: float = 50.0


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class BudgetCapBreached(Exception):
    """Raised when a cap check determines a breach.

    Carries a CapResult on the ``result`` attribute so callers can inspect
    the details without parsing the message string.
    """

    def __init__(self, message: str, result: "CapResult | None" = None) -> None:
        super().__init__(message)
        self.result = result


class R6BudgetLedgerError(RuntimeError):
    """Raised when a receipt-bound r6 reservation cannot safely proceed."""


@dataclass
class CapResult:
    """Result of a preflight or post-call cap check."""

    breach: bool
    reason: str
    existing_usd: float
    charged_usd: float
    cap_usd: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _global_cap() -> float:
    """Return GLOBAL_PROVIDER_BUDGET_CAP_USD from its single source of truth.

    The deferred import breaks a potential import cycle between costs/ and
    presentation/ while still guaranteeing a single definition site (D-01).
    """
    global _GLOBAL_CAP_CACHE  # noqa: PLW0603
    if _GLOBAL_CAP_CACHE is None:
        from conference_pipeline.presentation.blueprint_generation import (
            GLOBAL_PROVIDER_BUDGET_CAP_USD,
        )

        _GLOBAL_CAP_CACHE = GLOBAL_PROVIDER_BUDGET_CAP_USD
    return _GLOBAL_CAP_CACHE


def _existing_spend(workspace: Path) -> float:
    """Return the canonical total estimated cost recorded across all provider receipts.

    Uses ``provider_receipt_summary`` — the same function used by _guardrail_budget_cap
    in blueprint_generation.py — as the single existing-spend source.
    """
    from conference_pipeline.provider_receipts import provider_receipt_summary

    summary = provider_receipt_summary(workspace)
    return float(summary.get("canonical_total_estimated_cost_usd") or 0.0)


def _per_task_cap(task_id: str, registry: Any | None) -> float | None:
    """Return TaskPolicy.budget_cap_usd for task_id, or None if uncapped / not found.

    Registry load/parse failures PROPAGATE (fail closed — they are infrastructure
    failures, not "task not found"). Unknown task ids return None (legitimately uncapped).
    """
    # Step 1: resolve registry — let load/parse failures propagate (fail closed).
    if registry is None:
        from conference_pipeline.model_governance.registry import load_registry  # noqa: PLC0415

        registry = load_registry()  # raises on failure — do NOT catch here

    # Step 2: look up task — catch only "unknown task" (ModelGovernanceError).
    try:
        from conference_pipeline.model_governance.registry import ModelGovernanceError  # noqa: PLC0415

        policy = registry.get(task_id)
        return policy.budget_cap_usd
    except ModelGovernanceError:
        # Task not in registry — legitimately uncapped, not an error.
        return None


def _charged_amount(
    estimated_cost_usd: float | None,
    task_estimate_for_ceiling: float | None = None,
) -> float:
    """Compute the amount to charge for this call.

    For priced calls this is ``estimated_cost_usd``.
    For unpriced calls (None) this is ``max(task_estimate, UNKNOWN_COST_CEILING_USD)``
    (fail-closed per D-04 / BudgetTracker.record precedent).

    Parameters
    ----------
    estimated_cost_usd:
        The cost estimate for this call.  None means unpriced.
    task_estimate_for_ceiling:
        Optional task-level estimate to use as the lower bound of the ceiling
        when the call is unpriced.  If None, UNKNOWN_COST_CEILING_USD is used.
    """
    if estimated_cost_usd is not None:
        return float(estimated_cost_usd)
    # Unpriced call: fail-closed at max(task_estimate, UNKNOWN_COST_CEILING_USD)
    if task_estimate_for_ceiling is not None:
        return max(float(task_estimate_for_ceiling), UNKNOWN_COST_CEILING_USD)
    return UNKNOWN_COST_CEILING_USD


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_preflight(
    workspace: Path,
    task_id: str,
    estimated_cost_usd: float | None,
    *,
    registry: Any = None,
) -> CapResult:
    """Check whether running task_id would push cumulative spend over the cap.

    Called BEFORE a stage/task runs (D-02 preflight gate).

    Parameters
    ----------
    workspace:
        Conference workspace root (used to scan existing provider receipts).
    task_id:
        The task identifier for the task about to run.  Used to look up a
        per-task cap from the model-governance registry when ``registry`` is
        provided.
    estimated_cost_usd:
        Estimated cost for this task in USD.  ``None`` means the cost is
        unknown; the call is then charged at ``UNKNOWN_COST_CEILING_USD``
        (fail-closed per D-04).
    registry:
        Optional pre-loaded registry (``load_registry()`` result).  When
        ``None`` the global registry is loaded automatically.

    Returns
    -------
    CapResult
        ``breach=True`` when existing + charged > global cap or > per-task cap.
    """
    global_cap = _global_cap()
    existing = _existing_spend(workspace)
    charged = _charged_amount(estimated_cost_usd)
    total_after = existing + charged

    failures: list[str] = []

    # Per-task cap check
    per_task = _per_task_cap(task_id, registry)
    if per_task is not None and charged > per_task:
        failures.append(
            f"task budget cap exceeded: charged ${charged:.4f} > task cap ${per_task:.4f}"
        )

    # Global cap check (matches _guardrail_budget_cap shape)
    if total_after > global_cap:
        failures.append(
            f"global budget cap exceeded: projected ${total_after:.4f} > cap ${global_cap:.2f}"
        )

    if failures:
        reason = "; ".join(failures)
        return CapResult(
            breach=True,
            reason=reason,
            existing_usd=existing,
            charged_usd=charged,
            cap_usd=global_cap,
        )

    return CapResult(
        breach=False,
        reason="within budget",
        existing_usd=existing,
        charged_usd=charged,
        cap_usd=global_cap,
    )


def check_postcall(
    workspace: Path,
    run_id: str,
    actual_cost_usd: float,
    *,
    registry: Any = None,
) -> CapResult:
    """Check whether the recorded actual cost crosses the cap after a call.

    Called AFTER stage receipts are written (D-02 post-call gate).

    Parameters
    ----------
    workspace:
        Conference workspace root.
    run_id:
        The run whose receipts form the existing-spend baseline.
        The run_id is accepted for interface symmetry with the preflight;
        the existing-spend query scans all receipts in the workspace
        (consistent with provider_receipt_summary without run_id filter
        to capture cross-run spend correctly).
    actual_cost_usd:
        The actual cost recorded for the just-completed call, in USD.
    registry:
        Optional pre-loaded registry.  When ``None`` the global registry is
        loaded automatically.

    Returns
    -------
    CapResult
        ``breach=True`` when existing + actual > global cap.
    """
    global_cap = _global_cap()
    existing = _existing_spend(workspace)
    # actual_cost_usd is a delta on top of what is already recorded;
    # check whether the new total (existing + actual) breaches the cap.
    total_after = existing + float(actual_cost_usd)

    if total_after > global_cap:
        reason = (
            f"global budget cap exceeded post-call: "
            f"projected ${total_after:.4f} > cap ${global_cap:.2f}"
        )
        return CapResult(
            breach=True,
            reason=reason,
            existing_usd=existing,
            charged_usd=float(actual_cost_usd),
            cap_usd=global_cap,
        )

    return CapResult(
        breach=False,
        reason="within budget",
        existing_usd=existing,
        charged_usd=float(actual_cost_usd),
        cap_usd=global_cap,
    )


# ---------------------------------------------------------------------------
# Receipt-bound r6 reservation ledger
# ---------------------------------------------------------------------------

_R6_LEDGER_VERSION = 1
_SECRETISH_RECEIPT_KEYS = frozenset(
    {"cookie", "password", "secret", "token", "authorization", "api_key"}
)


def _r6_canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _r6_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_r6_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_finite_nonnegative(value: object, field: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise R6BudgetLedgerError(f"{field} must be finite") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise R6BudgetLedgerError(f"{field} must be finite and nonnegative")
    return numeric


def _require_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(char in value for char in "\\/\x00"):
        raise R6BudgetLedgerError(f"{field} must be a non-empty safe identifier")
    return value


def _ledger_without_digest(ledger: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in ledger.items() if key != "ledger_sha256"}


def _new_r6_ledger(existing_spend_usd: float) -> dict[str, Any]:
    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import R6_RUN_ID  # noqa: PLC0415

    return {
        "schema_version": _R6_LEDGER_VERSION,
        "run_id": R6_RUN_ID,
        "ledger_type": "r6_committed_plus_actual",
        "canonical_prior_spend_usd": existing_spend_usd,
        "reservations": [],
        # These secret-safe projections make the ledger independently useful
        # to the strict final receipt without changing reservation semantics.
        "committed": [],
        "actuals": [],
    }


def _load_r6_ledger(ledger_path: Path, existing_spend_usd: float) -> dict[str, Any]:
    if not ledger_path.exists():
        return _new_r6_ledger(existing_spend_usd)
    try:
        loaded = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R6BudgetLedgerError("r6 ledger is unreadable") from exc
    if not isinstance(loaded, dict) or loaded.get("schema_version") != _R6_LEDGER_VERSION:
        raise R6BudgetLedgerError("r6 ledger has an incompatible schema")
    expected = loaded.get("ledger_sha256")
    if not isinstance(expected, str) or expected != _r6_digest(_ledger_without_digest(loaded)):
        raise R6BudgetLedgerError("r6 ledger digest drift detected")
    baseline = _require_finite_nonnegative(loaded.get("canonical_prior_spend_usd"), "canonical spend")
    # canonical_prior_spend_usd is a write-once snapshot from ledger creation;
    # every cap check in this module (reserve_r6_budget, reserve_r6_scheduled_
    # dispatch) already uses the freshly-derived existing_spend_usd argument
    # for its arithmetic, never this stored field. So this is not a live cap
    # check — it is tamper evidence: honest spend only accumulates, so a
    # currently-observed total *below* the recorded baseline means history
    # was rewritten/erased. A total *above* baseline is the ordinary, expected
    # outcome of this same ledger's own reservations settling (each real
    # provider call — success or failure — lands in the workspace-wide
    # receipt log that existing_spend_usd is derived from), so it must not
    # reject an in-progress or later dispatch through an otherwise-valid
    # ledger.
    if existing_spend_usd < baseline - 0.000001:
        raise R6BudgetLedgerError("canonical existing spend drift detected")
    if not isinstance(loaded.get("reservations"), list):
        raise R6BudgetLedgerError("r6 ledger reservations are malformed")
    return loaded


def _write_r6_ledger(ledger_path: Path, ledger: dict[str, Any]) -> None:
    payload = _ledger_without_digest(ledger)
    payload["ledger_sha256"] = _r6_digest(payload)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_name(f".{ledger_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(_r6_canonical_json(payload) + "\n")
        os.replace(temporary, ledger_path)
    except FileExistsError as exc:
        raise R6BudgetLedgerError("r6 ledger temporary collision") from exc
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


@contextmanager
def _r6_locked_ledger(ledger_path: Path) -> Iterator[None]:
    """Hold an advisory cross-process lock while a reservation is read/written."""

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _reservation_total(reservations: list[Mapping[str, Any]], task_id: str | None = None) -> float:
    total = 0.0
    for reservation in reservations:
        if task_id is not None and reservation.get("task_id") != task_id:
            continue
        state = reservation.get("state")
        if state == "settled":
            total += _require_finite_nonnegative(reservation.get("actual_cost_usd"), "settled actual")
        elif state == "reserved":
            total += _require_finite_nonnegative(reservation.get("maximum_charge_usd"), "reserved charge")
        else:
            raise R6BudgetLedgerError("r6 ledger reservation has an unknown state")
    return total


def _contains_secretish_key(payload: object) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            normalized = str(key).lower()
            # This boolean is positive sanitization evidence, not a secret.
            if normalized != "secrets_copied" and any(
                marker in normalized for marker in _SECRETISH_RECEIPT_KEYS
            ):
                return True
            if _contains_secretish_key(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_secretish_key(value) for value in payload)
    return False


def validate_r6_budget_ledger(
    workspace: Path,
    ledger_path: Path,
    *,
    schedule: Mapping[str, Any] | None,
    registry: Any = None,
    provider_receipts: Mapping[str, Any] | None = None,
    require_complete: bool = False,
    no_paid: bool = False,
) -> dict[str, Any]:
    """Recompute the complete r6 cost proof without changing ledger bytes.

    ``ledger_sha256`` protects transport, not semantics.  This public
    read-only gate therefore replays the fixed dispatch schedule and verifies
    every reservation, committed projection, settled actual, provider receipt
    binding, ordinal, and cap before a caller may construct a provider or
    publish strict final success.
    """

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (  # noqa: PLC0415
        R6ArtifactContractError,
        R6_ALL_IN_CAP_USD,
        R6_OCR_CAP_USD,
        R6_RUN_ID,
        R6_SPEECH_CAP_USD,
        canonical_r6_queue,
        validate_r6_dispatch_schedule,
    )

    existing = _existing_spend(workspace)
    ledger = _load_r6_ledger(ledger_path, existing)
    if ledger.get("run_id") != R6_RUN_ID or ledger.get("ledger_type") != "r6_committed_plus_actual":
        raise R6BudgetLedgerError("r6 ledger run or schema identity is invalid")
    if _contains_secretish_key(ledger):
        raise R6BudgetLedgerError("r6 ledger contains secret-shaped data")
    reservations = ledger.get("reservations")
    committed = ledger.get("committed")
    actuals = ledger.get("actuals")
    if not isinstance(reservations, list) or not isinstance(committed, list) or not isinstance(actuals, list):
        raise R6BudgetLedgerError("r6 ledger rows are malformed")

    def provider_rows() -> list[Mapping[str, Any]]:
        if provider_receipts is None:
            return []
        rows = provider_receipts.get("receipts")
        if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
            raise R6BudgetLedgerError("r6 provider receipt rows are malformed")
        if _contains_secretish_key(provider_receipts):
            raise R6BudgetLedgerError("r6 provider receipt contains secret-shaped data")
        return list(rows)

    receipts = provider_rows()
    if no_paid:
        empty_queue, empty_digest = canonical_r6_queue(speech_ids=[], ocr_ids=[])
        del empty_queue
        if (
            schedule is not None
            or ledger.get("status") != "not_applicable"
            or reservations
            or committed
            or actuals
            or receipts
            or ledger.get("dispatch_schedule_digest") is not None
            or ledger.get("queue_digest") != empty_digest
        ):
            raise R6BudgetLedgerError("r6 no-paid ledger is not canonical zero-dispatch evidence")
        return dict(ledger)

    if schedule is None:
        raise R6BudgetLedgerError("paid r6 ledger requires a canonical dispatch schedule")
    try:
        checked = validate_r6_dispatch_schedule(schedule)
    except R6ArtifactContractError as exc:
        raise R6BudgetLedgerError("r6 ledger dispatch schedule is invalid") from exc
    entries = checked["entries"]
    assert isinstance(entries, list)
    schedule_digest = checked["dispatch_schedule_digest"]
    queue_digest = checked["queue_digest"]
    if (
        ledger.get("dispatch_schedule_digest") != schedule_digest
        or ledger.get("queue_digest") != queue_digest
        or ledger.get("dispatch_schedule") != checked
    ):
        raise R6BudgetLedgerError("r6 ledger schedule or queue identity drifted")
    next_ordinal = ledger.get("next_dispatch_ordinal")
    if not isinstance(next_ordinal, int) or next_ordinal != len(reservations) or not 0 <= next_ordinal <= len(entries):
        raise R6BudgetLedgerError("r6 ledger next dispatch ordinal is inconsistent")
    if len(committed) != len(reservations):
        raise R6BudgetLedgerError("r6 ledger committed projection is incomplete")

    receipts_by_reservation: dict[str, Mapping[str, Any]] = {}
    for receipt in receipts:
        reservation_id = receipt.get("reservation_id")
        if not isinstance(reservation_id, str) or not reservation_id or reservation_id in receipts_by_reservation:
            raise R6BudgetLedgerError("r6 provider receipt reservation binding is invalid")
        if receipt.get("sanitized") is not True:
            raise R6BudgetLedgerError("r6 provider receipt lacks sanitization proof")
        receipts_by_reservation[reservation_id] = receipt

    actual_rows: dict[str, Mapping[str, Any]] = {}
    for row in actuals:
        if not isinstance(row, Mapping):
            raise R6BudgetLedgerError("r6 actual projection row is malformed")
        reservation_id = row.get("reservation_id")
        if not isinstance(reservation_id, str) or reservation_id in actual_rows:
            raise R6BudgetLedgerError("r6 actual projection reservation is invalid")
        actual_rows[reservation_id] = row

    settled_total = 0.0
    settled_by_task = {_task: 0.0 for _task in ("transcription.session_speech", "ocr.slide_image_parse")}
    for ordinal, raw in enumerate(reservations):
        if not isinstance(raw, Mapping):
            raise R6BudgetLedgerError("r6 reservation row is malformed")
        expected = entries[ordinal]
        assert isinstance(expected, Mapping)
        identity = {
            "run_id": R6_RUN_ID,
            "task_id": expected["task_id"],
            "ordinal": ordinal,
            "entry_fingerprint": expected["entry_fingerprint"],
            "dispatch_schedule_digest": schedule_digest,
        }
        expected_reservation_id = _r6_digest(identity)
        if any(raw.get(key) != value for key, value in identity.items()) or raw.get("reservation_id") != expected_reservation_id:
            raise R6BudgetLedgerError("r6 reservation identity, ordinal, or fingerprint drifted")
        if (
            raw.get("item_id") != expected["identity"]
            or raw.get("benchmark_id") != expected["identity"]
            or raw.get("estimate_id") != schedule_digest
            or not math.isclose(
                _require_finite_nonnegative(raw.get("maximum_charge_usd"), "reservation charge"),
                _require_finite_nonnegative(expected.get("maximum_charge_usd"), "schedule charge"),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise R6BudgetLedgerError("r6 reservation projection drifted")
        committed_row = committed[ordinal]
        if not isinstance(committed_row, Mapping) or any(
            committed_row.get(key) != raw.get(key)
            for key in ("reservation_id", "run_id", "task_id", "ordinal", "entry_fingerprint", "dispatch_schedule_digest", "item_id", "benchmark_id", "estimate_id", "maximum_charge_usd")
        ):
            raise R6BudgetLedgerError("r6 committed reservation projection drifted")
        state = raw.get("state")
        if state not in {"reserved", "settled"}:
            raise R6BudgetLedgerError("r6 reservation state is invalid")
        if state == "reserved":
            if ordinal != len(reservations) - 1 or actual_rows.get(expected_reservation_id) is not None:
                raise R6BudgetLedgerError("r6 unsettled reservation sequence is invalid")
            settled_total += _require_finite_nonnegative(raw.get("maximum_charge_usd"), "reserved charge")
            settled_by_task[str(expected["task_id"])] += _require_finite_nonnegative(raw.get("maximum_charge_usd"), "reserved charge")
            continue
        actual = _require_finite_nonnegative(raw.get("actual_cost_usd"), "settled actual")
        maximum = _require_finite_nonnegative(raw.get("maximum_charge_usd"), "reservation charge")
        if actual > maximum:
            raise R6BudgetLedgerError("r6 settled actual exceeds its reservation")
        actual_row = actual_rows.get(expected_reservation_id)
        if (
            not isinstance(actual_row, Mapping)
            or actual_row.get("task_id") != expected["task_id"]
            or not math.isclose(_require_finite_nonnegative(actual_row.get("actual_cost_usd"), "actual projection"), actual, rel_tol=0.0, abs_tol=1e-9)
            or actual_row.get("provider_receipt_sha256") != raw.get("provider_receipt_sha256")
            or not isinstance(raw.get("provider_receipt_sha256"), str)
        ):
            raise R6BudgetLedgerError("r6 settled actual projection drifted")
        if provider_receipts is not None:
            receipt = receipts_by_reservation.get(expected_reservation_id)
            if receipt is None or raw.get("provider_receipt_sha256") != _r6_digest(dict(receipt)):
                raise R6BudgetLedgerError("r6 provider receipt reservation binding drifted")
        settled_total += actual
        settled_by_task[str(expected["task_id"])] += actual

    if set(actual_rows) != {
        str(row.get("reservation_id")) for row in reservations if isinstance(row, Mapping) and row.get("state") == "settled"
    }:
        raise R6BudgetLedgerError("r6 actual projection is missing or extra")
    if provider_receipts is not None and set(receipts_by_reservation) != set(actual_rows):
        raise R6BudgetLedgerError("r6 provider receipt coverage is missing or extra")
    if require_complete and (
        len(reservations) != len(entries)
        or any(not isinstance(row, Mapping) or row.get("state") != "settled" for row in reservations)
    ):
        raise R6BudgetLedgerError("r6 dispatch schedule has unspent or unsettled entries")
    if (
        existing + settled_total > R6_ALL_IN_CAP_USD
        or settled_by_task["transcription.session_speech"] > R6_SPEECH_CAP_USD
        or settled_by_task["ocr.slide_image_parse"] > R6_OCR_CAP_USD
    ):
        raise R6BudgetLedgerError("r6 ledger actual exposure breaches a cap")
    for task_id, cap in (("transcription.session_speech", R6_SPEECH_CAP_USD), ("ocr.slide_image_parse", R6_OCR_CAP_USD)):
        registry_cap = _per_task_cap(task_id, registry)
        if registry_cap is None or registry_cap < cap or settled_by_task[task_id] > registry_cap:
            raise R6BudgetLedgerError("r6 ledger governed task cap is invalid")
    return dict(ledger)


def reserve_r6_budget(
    workspace: Path,
    ledger_path: Path,
    *,
    run_id: str,
    task_id: str,
    item_id: str,
    benchmark_id: str,
    estimate_id: str,
    maximum_charge_usd: float,
    registry: Any = None,
) -> dict[str, Any]:
    """Atomically reserve one finite r6 paid call before dispatch.

    A ledger permits no second dispatch while the first reservation is unsettled.
    This deliberate serial rule makes a caller settle a sanitized provider
    receipt before another call and removes both concurrency and per-call cap
    loopholes.
    """

    run_id = _require_identifier(run_id, "run_id")
    task_id = _require_identifier(task_id, "task_id")
    item_id = _require_identifier(item_id, "item_id")
    benchmark_id = _require_identifier(benchmark_id, "benchmark_id")
    estimate_id = _require_identifier(estimate_id, "estimate_id")
    charge = _require_finite_nonnegative(maximum_charge_usd, "maximum charge")
    if charge <= 0:
        raise R6BudgetLedgerError("maximum charge must be greater than zero")
    existing = _existing_spend(workspace)
    task_cap = _per_task_cap(task_id, registry)
    if task_cap is None:
        raise R6BudgetLedgerError("r6 reservation requires a governed task cap")
    identity = {"run_id": run_id, "task_id": task_id, "item_id": item_id, "benchmark_id": benchmark_id, "estimate_id": estimate_id}
    reservation_id = _r6_digest(identity)
    with _r6_locked_ledger(ledger_path):
        ledger = _load_r6_ledger(ledger_path, existing)
        reservations = ledger["reservations"]
        if any(row.get("reservation_id") == reservation_id for row in reservations):
            raise R6BudgetLedgerError("duplicate r6 reservation identity")
        if any(row.get("state") == "reserved" for row in reservations):
            raise R6BudgetLedgerError("unsettled r6 reservation blocks another provider dispatch")
        global_after = existing + _reservation_total(reservations) + charge
        task_after = _reservation_total(reservations, task_id) + charge
        if global_after > _global_cap():
            raise R6BudgetLedgerError("r6 global all-in budget cap exceeded")
        if task_after > task_cap:
            raise R6BudgetLedgerError("r6 task budget cap exceeded")
        reservation = {
            "reservation_id": reservation_id,
            **identity,
            "maximum_charge_usd": charge,
            "state": "reserved",
        }
        reservations.append(reservation)
        committed = ledger.setdefault("committed", [])
        if isinstance(committed, list):
            committed.append(dict(reservation))
        _write_r6_ledger(ledger_path, ledger)
        return reservation


def reserve_r6_scheduled_dispatch(
    workspace: Path,
    ledger_path: Path,
    *,
    schedule: Mapping[str, Any],
    entry: Mapping[str, Any],
    registry: Any = None,
) -> dict[str, Any]:
    """Reserve exactly the next receipt-bound r6 dispatch-schedule entry.

    Unlike ``reserve_r6_budget``, this surface does not accept a caller chosen
    task/amount/item tuple.  The whole schedule is validated first, persisted
    in the ledger on first use, and consumed strictly in ordinal order.  A
    replay, omission, reordering, changed amount, or schedule substitution
    fails before a provider call can be constructed.
    """

    from conference_pipeline.acquisition.cvpr_r6_artifact_contract import (  # noqa: PLC0415
        R6ArtifactContractError,
        R6_RUN_ID,
        validate_r6_dispatch_schedule,
    )

    try:
        checked = validate_r6_dispatch_schedule(schedule)
    except R6ArtifactContractError as exc:
        raise R6BudgetLedgerError("r6 dispatch schedule is invalid") from exc
    entries = checked["entries"]
    assert isinstance(entries, list)  # Checked by validate_r6_dispatch_schedule.
    if not isinstance(entry, Mapping):
        raise R6BudgetLedgerError("r6 scheduled dispatch entry is required")
    existing = _existing_spend(workspace)
    schedule_digest = str(checked["dispatch_schedule_digest"])
    queue_digest = str(checked["queue_digest"])
    totals = checked["totals"]
    assert isinstance(totals, Mapping)
    if existing + float(totals["all_in_usd"]) > _global_cap():
        raise R6BudgetLedgerError("r6 complete dispatch schedule breaches the all-in cap")
    with _r6_locked_ledger(ledger_path):
        ledger = _load_r6_ledger(ledger_path, existing)
        reservations = ledger["reservations"]
        stored_digest = ledger.get("dispatch_schedule_digest")
        if stored_digest is None:
            if reservations:
                raise R6BudgetLedgerError("existing r6 ledger lacks its dispatch schedule")
            ledger["dispatch_schedule_digest"] = schedule_digest
            ledger["queue_digest"] = queue_digest
            ledger["dispatch_schedule"] = checked
            ledger["next_dispatch_ordinal"] = 0
        elif (
            stored_digest != schedule_digest
            or ledger.get("queue_digest") != queue_digest
            or ledger.get("dispatch_schedule") != checked
        ):
            raise R6BudgetLedgerError("r6 ledger dispatch schedule drift detected")
        next_ordinal = ledger.get("next_dispatch_ordinal")
        if not isinstance(next_ordinal, int) or next_ordinal < 0 or next_ordinal >= len(entries):
            raise R6BudgetLedgerError("r6 dispatch schedule is exhausted or malformed")
        expected = entries[next_ordinal]
        if dict(entry) != dict(expected):
            raise R6BudgetLedgerError("r6 dispatch schedule entry is missing, reordered, or changed")
        if any(row.get("state") == "reserved" for row in reservations):
            raise R6BudgetLedgerError("unsettled r6 reservation blocks another provider dispatch")
        task_id = _require_identifier(str(expected["task_id"]), "task_id")
        task_cap = _per_task_cap(task_id, registry)
        if task_cap is None:
            raise R6BudgetLedgerError("r6 reservation requires a governed task cap")
        schedule_task_total = sum(
            _require_finite_nonnegative(candidate.get("maximum_charge_usd"), "schedule charge")
            for candidate in entries
            if isinstance(candidate, Mapping) and candidate.get("task_id") == task_id
        )
        if schedule_task_total > task_cap:
            raise R6BudgetLedgerError("r6 complete dispatch schedule breaches a task cap")
        charge = _require_finite_nonnegative(expected.get("maximum_charge_usd"), "schedule charge")
        global_after = existing + _reservation_total(reservations) + charge
        task_after = _reservation_total(reservations, task_id) + charge
        if global_after > _global_cap():
            raise R6BudgetLedgerError("r6 global all-in budget cap exceeded")
        if task_after > task_cap:
            raise R6BudgetLedgerError("r6 task budget cap exceeded")
        identity = {
            "run_id": R6_RUN_ID,
            "task_id": task_id,
            "ordinal": next_ordinal,
            "entry_fingerprint": expected["entry_fingerprint"],
            "dispatch_schedule_digest": schedule_digest,
        }
        reservation = {
            "reservation_id": _r6_digest(identity),
            **identity,
            "item_id": str(expected["identity"]),
            "benchmark_id": str(expected["identity"]),
            "estimate_id": schedule_digest,
            "maximum_charge_usd": charge,
            "state": "reserved",
        }
        if any(row.get("reservation_id") == reservation["reservation_id"] for row in reservations):
            raise R6BudgetLedgerError("duplicate r6 scheduled dispatch")
        reservations.append(reservation)
        committed = ledger.setdefault("committed", [])
        if isinstance(committed, list):
            committed.append(dict(reservation))
        ledger["next_dispatch_ordinal"] = next_ordinal + 1
        _write_r6_ledger(ledger_path, ledger)
        return dict(reservation)


def settle_r6_budget(
    workspace: Path,
    ledger_path: Path,
    *,
    reservation_id: str,
    sanitized_provider_receipt: Mapping[str, Any],
    registry: Any = None,
) -> dict[str, Any]:
    """Settle a reservation after the owner has run the post-call gate.

    The r6 owner invokes ``check_postcall`` immediately after receipt capture
    and immediately before this durable settlement.  Reservation already
    covers the maximum possible charge, and settlement rejects any actual
    above that reservation, so the ledger cannot grow beyond its committed
    preflight exposure between those two operations.
    """

    reservation_id = _require_identifier(reservation_id, "reservation_id")
    if not isinstance(sanitized_provider_receipt, Mapping):
        raise R6BudgetLedgerError("provider receipt must be a sanitized mapping")
    if sanitized_provider_receipt.get("sanitized") is not True or _contains_secretish_key(sanitized_provider_receipt):
        raise R6BudgetLedgerError("provider receipt is missing sanitization proof")
    if sanitized_provider_receipt.get("reservation_id") != reservation_id:
        raise R6BudgetLedgerError("provider receipt does not bind the reservation")
    existing = _existing_spend(workspace)
    with _r6_locked_ledger(ledger_path):
        ledger = _load_r6_ledger(ledger_path, existing)
        reservation = next(
            (row for row in ledger["reservations"] if row.get("reservation_id") == reservation_id),
            None,
        )
        if reservation is None:
            raise R6BudgetLedgerError("unknown r6 reservation")
        if reservation.get("state") != "reserved":
            raise R6BudgetLedgerError("r6 reservation is already settled or malformed")
        maximum = _require_finite_nonnegative(reservation.get("maximum_charge_usd"), "maximum charge")
        raw_actual = sanitized_provider_receipt.get("actual_cost_usd")
        if raw_actual is None:
            actual = maximum
            actual_source = "conservative_reservation"
        else:
            actual = _require_finite_nonnegative(raw_actual, "actual cost")
            actual_source = "provider_receipt"
        if actual > maximum:
            raise R6BudgetLedgerError("actual provider charge exceeds its committed reservation")
        reservation["actual_cost_usd"] = actual
        reservation["actual_cost_source"] = actual_source
        reservation["provider_receipt_sha256"] = _r6_digest(dict(sanitized_provider_receipt))
        reservation["state"] = "settled"
        actuals = ledger.setdefault("actuals", [])
        if isinstance(actuals, list):
            actuals.append(
                {
                    "reservation_id": reservation_id,
                    "task_id": reservation.get("task_id"),
                    "actual_cost_usd": actual,
                    "provider_receipt_sha256": reservation["provider_receipt_sha256"],
                }
            )
        _write_r6_ledger(ledger_path, ledger)
        return dict(reservation)
