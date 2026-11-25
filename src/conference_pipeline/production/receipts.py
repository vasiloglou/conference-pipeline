"""One stage-receipt writer for every shared production stage.

The runners used to carry four near-identical receipt writers (podcast,
presentation, release, proof), two of which recorded a different
``receipt_type`` from the other two. This module writes one shape, stamped
with the runner's own receipt type and code identity from the context, and
returns the ``StageOutcome`` mapping the runners' ``main`` already consumes:
``{"status", "receipt", "receipt_path", "summary"}``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.provider_receipts import utc_now
from conference_pipeline.receipt_lineage import preserve_prior_receipt
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)
from conference_pipeline.workflow_runs import write_json_atomic

StageOutcome = dict[str, Any]


def stage_blocker(stage_id: str, code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "affected_stage": stage_id, **details}


def load_stage_receipt(ctx: ProductionRunnerContext, stage_id: str) -> dict[str, Any] | None:
    """The persisted receipt for ``stage_id`` in this run, or ``None``."""

    import json

    path = ctx.receipt_path(stage_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def resumed_outcome(ctx: ProductionRunnerContext, stage_id: str, *, strict_execution: bool = True) -> StageOutcome | None:
    """A prior succeeded receipt that still verifies, as a resume outcome; else ``None``.

    Verification runs against the persisted receipt with ``bind=False`` so a
    receipt whose upstream moved fails here and the stage re-runs.
    """

    prior = load_stage_receipt(ctx, stage_id)
    if not prior or prior.get("status") != "succeeded":
        return None
    contract = ctx.verify_contract(
        stage_id=stage_id, receipt=prior, bind=False, strict_execution=strict_execution
    )
    if contract.status != "pass":
        return None
    return {
        "status": "succeeded",
        "receipt": prior,
        "receipt_path": str(ctx.receipt_path(stage_id)),
        "summary": {"stage": stage_id, "status": "succeeded", "resumed": True},
    }


def write_stage_receipt(
    ctx: ProductionRunnerContext,
    *,
    stage_id: str,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    stage_contract: Mapping[str, Any] | None,
    started_at: str | None = None,
    mode: str = "live",
    lineage: Mapping[str, Any] | None = None,
    error: str | None = None,
    summary_extra: Mapping[str, Any] | None = None,
    receipt_extra: Mapping[str, Any] | None = None,
) -> StageOutcome:
    """Persist ``stage-NN-<id>.json`` for the run and return the stage outcome.

    The receipt is finalized through the context (lineage, output digest,
    code identity) and the prior receipt at that path is preserved as a
    superseded or attempt copy before the new one lands. ``receipt_extra``
    adds stage-specific top-level fields (an owner-evidence map, a
    provider-calls flag). When the context carries a ``summary_writer`` the
    runner's own run summary replaces the compact per-stage summary.
    """

    stage = ctx.stage(stage_id)
    registry = load_stage_script_registry()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_id": f"{ctx.run_id}:{stage.id}",
        "receipt_type": ctx.receipt_type,
        "run_id": ctx.run_id,
        "workspace": str(ctx.workspace),
        "stage_id": stage.id,
        "stage_order": stage.order,
        "generated_at": utc_now(),
        "started_at": started_at or utc_now(),
        "ended_at": utc_now(),
        "mode": mode,
        "status": status,
        "resumable": status == "succeeded",
        "blockers": list(blockers),
        "checks": dict(checks),
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": list(required_script_ids_for_stage(registry, stage.id)),
        "stage_contract": dict(stage_contract) if stage_contract is not None else None,
        "provider_values_recorded": False,
        "secrets_copied": False,
        "error": error,
        "next_action": stage.next_action,
        "receipt_path": str(ctx.receipt_path(stage.id)),
    }
    if receipt_extra:
        receipt.update(dict(receipt_extra))
    receipt = ctx.finalize(receipt, stage_id=stage.id, lineage=lineage, profile=None)
    path = ctx.receipt_path(stage.id)
    preserve_prior_receipt(path, new_status=str(receipt.get("status") or ""))
    write_json_atomic(path, receipt)
    summary: dict[str, Any] = {
        "stage": stage.id,
        "status": status,
        "blockers": [item.get("code") for item in blockers],
    }
    if summary_extra:
        summary.update(dict(summary_extra))
    if ctx.summary_writer is not None:
        summary = ctx.summary_writer(receipt)
    return {"status": status, "receipt": receipt, "receipt_path": str(path), "summary": summary}


# --------------------------------------------------------------------------
# Receipt conventions for the gated paid stages (Stage 8 / Stage 9)
# --------------------------------------------------------------------------


def _default_load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    from conference_pipeline.production.presentation_steps import load_json

    return load_json(path)


def _default_write_json(path: Path, payload: dict[str, Any]) -> None:
    from conference_pipeline.indexing.corpus_vector_store import atomic_write_json

    atomic_write_json(path, payload)


def _default_resolve_member(workspace: Path, path_value: object) -> tuple[Path | None, str | None]:
    from conference_pipeline.indexing.corpus_vector_store import resolve_workspace_member

    return resolve_workspace_member(workspace, path_value)


@dataclass(frozen=True)
class StageIO:
    """The runner's clock and file conventions a shared stage reads and writes with.

    A runner passes its own functions (bound at call time, so a test that
    replaces the runner's ``utc_now`` still freezes the shared stage's clock);
    a new conference keeps the defaults.
    """

    now: Callable[[], str] = utc_now
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json
    write_json: Callable[[Path, dict[str, Any]], None] = _default_write_json
    resolve_member: Callable[[Path, object], tuple[Path | None, str | None]] = _default_resolve_member


DEFAULT_STAGE_IO = StageIO()


@runtime_checkable
class StageReceiptConvention(Protocol):
    """How a gated shared stage halts, resumes, blocks and succeeds.

    The shared stage owns the gates and the checks it records; the convention
    owns the receipt and run-summary shape (and the return value) of the
    runner that invoked it. Every method returns that runner's stage result,
    or ``None`` where the stage should continue.
    """

    def start(self) -> Any | None: ...

    def resumed(self) -> Any | None: ...

    def block(self, code: str, *, reason: str, remediation: str, checks: Mapping[str, Any]) -> Any: ...

    def lineage_gate(self, checks: Mapping[str, Any], *, input_digests: Mapping[str, Any] | None = None) -> Any | None: ...

    def succeed(self, checks: dict[str, Any], *, owner_evidence: Mapping[str, str]) -> Any: ...


def merge_run_summary(ctx: ProductionRunnerContext, receipt: dict[str, Any], *, io: StageIO = DEFAULT_STAGE_IO) -> dict[str, Any]:
    """``summary.json`` carrying every stage's latest receipt, replaced by stage id.

    The newest receipt's status is the run status (``halted`` when blocked).
    """

    import json

    from conference_pipeline.provider_receipts import sanitize_for_receipt

    stage_id = str(receipt.get("stage_id") or "")
    status = str(receipt.get("status") or "")
    summary_path = ctx.run_dir() / "summary.json"
    try:
        prior_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prior_summary = {}
    raw_stages = prior_summary.get("stages") if isinstance(prior_summary, Mapping) else []
    stages = list(raw_stages) if isinstance(raw_stages, list) else []
    stages = [item for item in stages if not isinstance(item, Mapping) or item.get("stage_id") != stage_id]
    stages.append(receipt)
    summary = {
        "schema_version": 1,
        "run_id": ctx.run_id,
        "workspace": str(ctx.workspace),
        "mode": "live",
        "status": status,
        "halt_status": "halted" if status == "blocked" else None,
        "halt_stage_id": stage_id if status == "blocked" else None,
        "stages": stages,
        "summary_path": str(summary_path),
    }
    io.write_json(summary_path, sanitize_for_receipt(summary))
    return summary


@dataclass
class StrictStageConvention:
    """The default convention: a ``{status, receipt, receipt_path, summary}`` outcome.

    Resume verifies the persisted receipt as-is with ``strict_execution``;
    a blocked receipt records one ``{code, affected_stage}`` blocker and
    zero spend; the pre-spend lineage gate blocks with
    ``upstream_lineage_not_fresh``; success binds the stage's execution
    evidence per required owner id and verifies the contract with
    ``strict_execution=True``. The run summary is the context's
    ``summary_writer`` (default :func:`merge_run_summary`).
    """

    ctx: ProductionRunnerContext
    stage_id: str
    io: StageIO = DEFAULT_STAGE_IO

    def start(self) -> StageOutcome | None:
        return None

    def resumed(self) -> StageOutcome | None:
        import json

        path = self.ctx.receipt_path(self.stage_id)
        if not path.exists():
            return None
        existing, _error = self.io.load_json(path)
        if not isinstance(existing, dict) or existing.get("status") != "succeeded":
            return None
        contract = self.ctx.verify_contract(
            stage_id=self.stage_id, receipt=existing, bind=False, mode="live", strict_execution=True
        )
        if contract.status != "pass":
            return None
        try:
            existing_summary = json.loads((self.ctx.run_dir() / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_summary = {}
        return {"status": "succeeded", "receipt_path": str(path), "receipt": existing, "summary": existing_summary}

    def block(self, code: str, *, reason: str, remediation: str, checks: Mapping[str, Any]) -> StageOutcome:
        del reason, remediation
        return self._write(
            status="blocked",
            blockers=[stage_blocker(self.stage_id, code)],
            checks={**checks, "blocker_code": code, "provider_calls_made": False, "live_provider_calls_made": False},
            stage_contract=None,
        )

    def lineage_gate(
        self, checks: Mapping[str, Any], *, input_digests: Mapping[str, Any] | None = None
    ) -> StageOutcome | None:
        from conference_pipeline.production.context import UPSTREAM_LINEAGE_NOT_FRESH

        del input_digests
        lineage_checks = self.ctx.upstream_lineage_checks(self.stage_id)
        if lineage_checks is None:
            return None
        return self.block(UPSTREAM_LINEAGE_NOT_FRESH, reason="", remediation="", checks={**checks, **lineage_checks})

    def succeed(self, checks: dict[str, Any], *, owner_evidence: Mapping[str, str]) -> StageOutcome:
        required_script_ids = list(required_script_ids_for_stage(load_stage_script_registry(), self.stage_id))
        evidence = {
            owner_id: str(owner_evidence.get(owner_id) or f"{owner_id}:evidence_unavailable")
            for owner_id in required_script_ids
        }
        checks["owner_execution_evidence"] = evidence
        provisional = {
            "mode": "live",
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": required_script_ids,
            "owner_execution_evidence": evidence,
            "checks": checks,
        }
        stage_contract = self.ctx.verify_contract(
            stage_id=self.stage_id, receipt=provisional, bind=True, mode="live", strict_execution=True
        ).to_dict()
        blockers = [] if stage_contract["status"] == "pass" else [stage_blocker(self.stage_id, "final_stage_contract_failed")]
        return self._write(
            status="succeeded" if not blockers else "blocked",
            blockers=blockers,
            checks=checks,
            stage_contract=stage_contract,
        )

    def _write(
        self,
        *,
        status: str,
        blockers: list[dict[str, Any]],
        checks: Mapping[str, Any],
        stage_contract: Mapping[str, Any] | None,
        lineage: Mapping[str, Any] | None = None,
        receipt_fields: Mapping[str, Any] | None = None,
    ) -> StageOutcome:
        """Persist the receipt and run summary.

        ``lineage`` is a resumed receipt's own recorded lineage (``None``
        binds afresh); ``receipt_fields`` replaces top-level fields a stage
        records in its own shape (an owner-evidence map, an integer call
        count) before the receipt is finalized.
        """

        from conference_pipeline.provider_receipts import sanitize_for_receipt

        ctx = self.ctx
        stage = ctx.stage(self.stage_id)
        path = ctx.receipt_path(stage.id)
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "receipt_type": ctx.receipt_type,
            "receipt_id": f"{ctx.run_id}:{stage.id}:{self.io.now()}",
            "run_id": ctx.run_id,
            "workspace": str(ctx.workspace),
            "stage_id": stage.id,
            "stage_order": stage.order,
            "mode": "live",
            "status": status,
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": list(required_script_ids_for_stage(load_stage_script_registry(), stage.id)),
            "owner_execution_evidence": dict(checks.get("owner_execution_evidence") or {}),
            "provider_calls_made": bool(checks.get("provider_calls_made")),
            "downloads_made": 0,
            "checks": sanitize_for_receipt(dict(checks)),
            "stage_contract": sanitize_for_receipt(stage_contract) if stage_contract else None,
            "blockers": sanitize_for_receipt(blockers),
            "secrets_copied": False,
        }
        if receipt_fields:
            receipt.update(dict(receipt_fields))
        receipt = ctx.finalize(receipt, stage_id=stage.id, lineage=lineage, profile=None)
        if stage.order >= 7:
            preserve_prior_receipt(path, new_status=str(receipt.get("status") or ""))
        self.io.write_json(path, sanitize_for_receipt(receipt))
        if ctx.summary_writer is not None:
            summary = ctx.summary_writer(receipt)
        else:
            summary = merge_run_summary(ctx, receipt, io=self.io)
        return {"status": status, "receipt_path": str(path), "receipt": receipt, "summary": summary}
