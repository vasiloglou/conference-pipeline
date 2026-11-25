"""Local-first routing resolver for the model-governance layer.

Implements D-05..D-08 from the Phase 03 context:

- D-05: local-first is a config-gated preference layered over registry defaults.
- D-06: fallback triggers: no approved local proof, local-first disabled,
  or failed readiness preflight.  Readiness is binary available-or-fallback;
  memory-aware selection is LOCAL-FUTURE-01.
- D-07: fallback is automatic and receipted (receipt writing is optional
  when run context is omitted, so the resolver stays pure in tests).
- D-08: one global local-first switch (default ON once approved), with
  a per-task ``local_first_enabled`` registry field able to force paid.

PUBLIC API:
    LocalFirstRoutingDecision  -- frozen dataclass carrying the routing result
    resolve_local_first_selection -- the single call-site entry point
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from conference_pipeline.model_governance.registry import (
    ModelGovernanceError,
    ModelSelection,
    TaskPolicy,
    approved_local_selection,
    load_registry,
)

# Fallback-reason vocabulary (D-06)
_FR_LOCAL_FIRST_DISABLED = "local_first_disabled"
_FR_TASK_FORCED_PAID = "task_forced_paid"
_FR_APPROVED_LOCAL_UNAVAILABLE = "approved_local_unavailable"
_FR_LOCAL_READINESS_FAILED = "local_readiness_failed"


@dataclass(frozen=True)
class LocalFirstRoutingDecision:
    """Routing decision produced by ``resolve_local_first_selection``.

    Carries both the concrete provider/model/role to use for the call
    and the audit trail fields that explain why local was chosen or
    bypassed.
    """

    task_id: str
    provider: str
    model: str
    role: str
    used_local: bool
    fallback_used: bool
    fallback_reason: str | None
    benchmark_receipt_ref: str | None
    paid_fallback_provider: str | None
    paid_fallback_model: str | None


def _paid_selection_from_policy(policy: TaskPolicy) -> ModelSelection:
    """Return the best NON-local paid selection for a local-first fallback.

    Invariant: the local-first paid fallback must never be a local provider. A
    registry ``fallback_provider: local`` (e.g. transcription.session_speech's
    deferred ``whisper-1`` candidate) is a *deferred local candidate*, selectable
    ONLY through ``approved_local_selection()`` after proof + readiness — never
    through a fallback reason. Choosing it here would route a paid fallback to
    local and bypass the proof gate (Phase 10 contract, D-05..D-08).

    Preference order (fallback-first preserved for genuinely-paid fallbacks):
    the registry fallback, then the registry default, returning the first that is
    non-``None`` and non-local; then any non-local entry in ``allowed_selections``;
    only for a genuinely local-only task (no non-local selection anywhere, e.g.
    ``analysis.llm_full_read_extraction`` — out of Phase 10 scope) does it return
    the default as a last resort.
    """
    for sel in (policy.fallback_selection, policy.default_selection):
        if sel is not None and sel.provider != "local":
            return sel
    for sel in policy.allowed_selections:
        if sel.provider != "local":
            return sel
    return policy.default_selection


def _local_first_task_enabled(policy: TaskPolicy) -> bool:
    """Return the per-task local-first toggle (default: True when field absent)."""
    return policy.local_first_enabled


def resolve_local_first_selection(
    task_id: str,
    *,
    local_first_enabled: bool = True,
    registry_path: str | Path | None = None,
    readiness_check: Callable[[str], bool] | None = None,
    # Optional run context — when supplied the resolver will write a
    # LocalFallbackReceipt; omit for pure/test usage.
    workspace: Path | None = None,
    run_id: str | None = None,
    stage_id: str | None = None,
) -> LocalFirstRoutingDecision:
    """Return the best ModelSelection for *task_id* under local-first policy.

    Decision logic (D-05..D-08):

    1. If the global ``local_first_enabled=False``, fall back immediately
       with ``fallback_reason='local_first_disabled'``.
    2. Load the registry; read the per-task ``local_first_enabled`` field
       (defaults ``True``). If ``False``, fall back with
       ``fallback_reason='task_forced_paid'``.
    3. Call ``approved_local_selection(task_id)``; if it raises
       ``ModelGovernanceError`` (any missing proof), fall back with
       ``fallback_reason='approved_local_unavailable'``.
    4. If ``readiness_check`` is supplied and raises or returns ``False``,
       fall back with ``fallback_reason='local_readiness_failed'``.
    5. Return the approved local selection with ``used_local=True``.

    When ``workspace``/``run_id``/``stage_id`` are supplied **and** a
    fallback occurs, the resolver optionally calls ``record_local_fallback``
    from ``provider_receipts`` (Task 2 wires this; Task 1 leaves the hook
    as a no-op import-guard so the resolver stays pure when context is
    absent).
    """
    registry = load_registry(registry_path)
    policy = registry.get(task_id)
    paid = _paid_selection_from_policy(policy)
    benchmark_ref: str | None = policy.local_proof.get("benchmark_receipt_ref")  # type: ignore[attr-defined]

    def _fallback(reason: str) -> LocalFirstRoutingDecision:
        _maybe_write_fallback_receipt(
            task_id=task_id,
            fallback_reason=reason,
            requested_provider="local",
            requested_model=None,
            paid=paid,
            benchmark_receipt_ref=benchmark_ref,
            workspace=workspace,
            run_id=run_id,
            stage_id=stage_id,
        )
        return LocalFirstRoutingDecision(
            task_id=task_id,
            provider=paid.provider,
            model=paid.model,
            role=paid.role,
            used_local=False,
            fallback_used=True,
            fallback_reason=reason,
            benchmark_receipt_ref=benchmark_ref,
            paid_fallback_provider=paid.provider,
            paid_fallback_model=paid.model,
        )

    # Step 1 — global switch
    if not local_first_enabled:
        return _fallback(_FR_LOCAL_FIRST_DISABLED)

    # Step 2 — per-task registry override
    if not _local_first_task_enabled(policy):
        return _fallback(_FR_TASK_FORCED_PAID)

    # Step 3 — approved local selection gate
    try:
        local_sel: ModelSelection = approved_local_selection(
            task_id, registry_path=registry_path
        )
    except ModelGovernanceError:
        return _fallback(_FR_APPROVED_LOCAL_UNAVAILABLE)

    # Step 4 — optional readiness check
    if readiness_check is not None:
        try:
            ok = readiness_check(task_id)
        except Exception:  # noqa: BLE001 — any readiness exception → fallback
            ok = False
        if not ok:
            return _fallback(_FR_LOCAL_READINESS_FAILED)

    # Step 5 — use local
    return LocalFirstRoutingDecision(
        task_id=task_id,
        provider=local_sel.provider,
        model=local_sel.model,
        role=local_sel.role,
        used_local=True,
        fallback_used=False,
        fallback_reason=None,
        benchmark_receipt_ref=benchmark_ref,
        paid_fallback_provider=paid.provider,
        paid_fallback_model=paid.model,
    )


def _maybe_write_fallback_receipt(
    *,
    task_id: str,
    fallback_reason: str,
    requested_provider: str | None,
    requested_model: str | None,
    paid: ModelSelection,
    benchmark_receipt_ref: str | None,
    workspace: Path | None,
    run_id: str | None,
    stage_id: str | None,
) -> None:
    """Write a LocalFallbackReceipt when run context is present; no-op otherwise.

    T-03-02: fallback is always receipted when workspace is supplied.
    T-03-05: sanitize_for_receipt() is called inside record_local_fallback.
    Keeping the guard on ``workspace`` ensures the resolver stays pure in
    tests and dry-run contexts where no receipt directory should be created.
    """
    if workspace is None:
        return
    from conference_pipeline.provider_receipts import record_local_fallback
    record_local_fallback(
        task_id=task_id,
        stage_id=stage_id,
        run_id=run_id,
        requested_provider=requested_provider or "local",
        requested_model=requested_model or "",
        fallback_provider=paid.provider,
        fallback_model=paid.model,
        fallback_reason=fallback_reason,
        benchmark_receipt_ref=benchmark_receipt_ref,
        workspace=workspace,
    )
