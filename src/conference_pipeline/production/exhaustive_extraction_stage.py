"""Stage 7A: exhaustive topic extraction, shared by every conference runner.

Gates, does not orchestrate: (1) the receipt convention's start halt;
(2) fail closed, with zero work, unless the Stage 6
(corpus-inventory-index) receipt is readable and succeeded; (3) on
``--resume``, the convention's short-circuit over an already succeeded
receipt that still verifies with its own recorded lineage; (4) the
pre-work lineage gate; (5) the shared heuristic extraction owner
(``conference_pipeline.iclr2025_live.run_exhaustive_topic_extraction``,
injected so a runner can bind it at call time), rebuilding from scratch
unless resuming; (6) the convention's final receipt, verified against the
stage contract (the by-topic / by-artifact / coverage output families).

The owner is local and makes no provider call; this stage constructs no
client and performs no network access itself.

Everything a conference decides comes in through the
:class:`~conference_pipeline.production.context.ProductionRunnerContext`
(workspace, run, stage table, identity, env loader) and the
:class:`ExhaustiveExtractionConvention` (receipt and run-summary shape and
the return value). Two conventions ship here: :class:`StrictExhaustiveConvention`
(a ``{status, receipt, receipt_path, summary}`` outcome with per-owner
execution evidence) and :class:`RunnerSummaryExhaustiveConvention` (an
``(exit_code, run_summary)`` result built from the runner's own receipt
writers, with the owner's receipt as the stage receipt's body). The module
carries no conference literal.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from conference_pipeline.iclr2025_live import (
    run_exhaustive_topic_extraction as shared_extraction_owner,
)
from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.production.context import (
    UPSTREAM_LINEAGE_NOT_FRESH,
    ProductionRunnerContext,
    StageSpec,
)
from conference_pipeline.production.receipts import (
    DEFAULT_STAGE_IO,
    StageIO,
    StrictStageConvention,
    stage_blocker,
)
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.receipt_lineage import sha256_file
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)

STAGE_ID = "exhaustive-topic-extraction"
STAGE6_ID = "corpus-inventory-index"

STAGE6_REQUIRED = "exhaustive_extraction_stage6_required"
OWNER_RAISED = "exhaustive_extraction_owner_raised"
OWNER_NOT_SUCCEEDED = "exhaustive_extraction_owner_not_succeeded"
FINAL_STAGE_CONTRACT_FAILED = "final_stage_contract_failed"

STAGE6_REQUIRED_ERROR = "exhaustive-topic-extraction requires a succeeded Stage 6 receipt"
CONTRACT_FAILED_ERROR = "runtime stage contract verification failed"

# The owner identity recorded as execution evidence: the shared owner's own
# dotted name, never a conference literal.
OWNER_FUNCTION = "run_exhaustive_topic_extraction_stage"
OWNER_MODULE = f"{shared_extraction_owner.__module__}.{shared_extraction_owner.__name__}"

ZERO_SPEND_FLAGS: Mapping[str, bool] = {
    "provider_calls_made": False,
    "live_provider_calls_made": False,
    "secrets_copied": False,
}

ExtractionOwner = Callable[..., Mapping[str, Any]]


def extraction_root(ctx: ProductionRunnerContext) -> Path:
    return ctx.workspace / "storage" / "extractions" / ctx.run_id


def exhaustive_output_counts(workspace: Path, run_id: str) -> dict[str, Any]:
    """Where the owner's output families live and how many files each holds."""

    root = workspace / "storage" / "extractions" / run_id
    by_topic_dir = root / "by-topic"
    by_artifact_dir = root / "by-artifact"
    coverage_path = root / "v15_extraction_coverage.json"
    return {
        "by_topic_dir": workspace_relpath(workspace, by_topic_dir),
        "by_artifact_dir": workspace_relpath(workspace, by_artifact_dir),
        "coverage_path": workspace_relpath(workspace, coverage_path),
        "by_topic_count": sum(1 for _ in by_topic_dir.glob("*.json")) if by_topic_dir.exists() else 0,
        "by_artifact_count": sum(1 for _ in by_artifact_dir.glob("*.json")) if by_artifact_dir.exists() else 0,
        "coverage_present": coverage_path.exists(),
        "coverage_sha256": sha256_file(coverage_path) if coverage_path.exists() else None,
    }


@dataclass(frozen=True)
class Stage6Gate:
    """What the stage read of the Stage 6 receipt it gates on."""

    path: Path
    present: bool
    error: str | None
    status: Any
    readable: bool

    @property
    def passed(self) -> bool:
        return self.readable and self.error is None and self.status == "succeeded"


@runtime_checkable
class ExhaustiveExtractionConvention(Protocol):
    """How Stage 7A halts, blocks, resumes and records the owner's result.

    Every method returns the invoking runner's stage result, or ``None``
    where the stage should continue.
    """

    def start(self) -> Any | None: ...

    def stage6_blocked(self, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> Any: ...

    def resumed(self, *, env_file: Mapping[str, bool]) -> Any | None: ...

    def lineage_gate(self, *, env_file: Mapping[str, bool]) -> Any | None: ...

    def owner_raised(self, error: str, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> Any: ...

    def finish(self, owner_receipt: Any, *, env_file: Mapping[str, bool]) -> Any: ...


def run_exhaustive_topic_extraction_stage(
    ctx: ProductionRunnerContext,
    *,
    convention: ExhaustiveExtractionConvention | None = None,
    resume: bool,
    env_file: Path | None,
    owner: ExtractionOwner | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
) -> Any:
    """Run Stage 7A under ``convention`` (default: :class:`StrictExhaustiveConvention`)."""

    convention = convention or StrictExhaustiveConvention(ctx, STAGE_ID, io=io)
    owner = owner or shared_extraction_owner
    ctx.run_dir().mkdir(parents=True, exist_ok=True)
    env_status = ctx.load_env(env_file)

    halted = convention.start()
    if halted is not None:
        return halted

    stage6_path = ctx.receipt_path(STAGE6_ID)
    stage6, stage6_error = io.load_json(stage6_path)
    gate = Stage6Gate(
        path=stage6_path,
        present=stage6_path.exists(),
        error=stage6_error,
        status=stage6.get("status") if isinstance(stage6, dict) else None,
        readable=isinstance(stage6, dict),
    )
    if not gate.passed:
        return convention.stage6_blocked(gate, env_file=env_status)

    if resume:
        resumed = convention.resumed(env_file=env_status)
        if resumed is not None:
            return resumed

    gated = convention.lineage_gate(env_file=env_status)
    if gated is not None:
        return gated

    try:
        owner_receipt = owner(ctx.workspace, run_id=ctx.run_id, force_rebuild=not resume)
    except Exception as exc:  # noqa: BLE001 - receipt boundary
        return convention.owner_raised(sanitize_for_receipt(str(exc)), gate, env_file=env_status)
    return convention.finish(owner_receipt, env_file=env_status)


# --------------------------------------------------------------------------
# Strict convention: {status, receipt, receipt_path, summary} outcome
# --------------------------------------------------------------------------


@dataclass
class StrictExhaustiveConvention(StrictStageConvention):
    """Blocked receipts list ``{code, affected_stage, ...}`` blockers and record
    zero spend; success verifies the contract with ``strict_execution`` over
    per-owner execution evidence naming the shared owner; resume re-verifies
    the persisted receipt with its own lineage and rewrites it unchanged."""

    def _required_ids(self) -> list[str]:
        return list(required_script_ids_for_stage(load_stage_script_registry(), self.stage_id))

    def _owner_evidence(self, *, with_module: bool) -> dict[str, dict[str, Any]]:
        evidence: dict[str, Any] = {"owner_function": OWNER_FUNCTION}
        if with_module:
            evidence["owner_module"] = OWNER_MODULE
        evidence.update({"provider_calls_made": 0, "downloads_made": 0})
        return {script_id: dict(evidence) for script_id in self._required_ids()}

    def _record(
        self,
        *,
        status: str,
        blockers: list[dict[str, Any]],
        checks: Mapping[str, Any],
        stage_contract: Mapping[str, Any] | None,
        lineage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._write(
            status=status,
            blockers=blockers,
            checks=checks,
            stage_contract=stage_contract,
            lineage=lineage,
            receipt_fields={
                "owner_execution_evidence": self._owner_evidence(with_module=True),
                "provider_calls_made": 0,
            },
        )

    def _verify(
        self, checks: Mapping[str, Any], *, lineage: Mapping[str, Any] | None = None, bind: bool = True
    ) -> dict[str, Any]:
        provisional = {
            "mode": "live",
            "script_registry_sha256": registry_sha256(),
            "required_script_ids": self._required_ids(),
            "owner_execution_evidence": self._owner_evidence(with_module=False),
            "checks": dict(checks),
        }
        return self.ctx.verify_contract(
            stage_id=self.stage_id,
            receipt=provisional,
            bind=bind,
            lineage=lineage,
            mode="live",
            strict_execution=True,
        ).to_dict()

    def stage6_blocked(self, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> dict[str, Any]:
        if not gate.readable:
            blockers = [stage_blocker(self.stage_id, STAGE6_REQUIRED, path=str(gate.path))]
        else:
            blockers = [
                stage_blocker(self.stage_id, STAGE6_REQUIRED, path=str(gate.path), observed_status=gate.status)
            ]
        return self._record(
            status="blocked",
            blockers=blockers,
            checks={
                "env_file": dict(env_file),
                "stage6_receipt_path": str(gate.path),
                "stage6_status": gate.status,
                **ZERO_SPEND_FLAGS,
            },
            stage_contract=None,
        )

    def resumed(self, *, env_file: Mapping[str, bool]) -> dict[str, Any] | None:  # type: ignore[override]
        path = self.ctx.receipt_path(self.stage_id)
        if not path.exists():
            return None
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(existing, dict) or existing.get("status") != "succeeded":
            return None
        existing_checks = existing.get("checks")
        checks = {
            **(dict(existing_checks) if isinstance(existing_checks, Mapping) else {}),
            "env_file": dict(env_file),
        }
        existing_lineage = existing.get("lineage") if isinstance(existing.get("lineage"), Mapping) else None
        stage_contract = self._verify(checks, lineage=existing_lineage, bind=False)
        if stage_contract["status"] != "pass":
            return None
        return self._record(
            status="succeeded", blockers=[], checks=checks, stage_contract=stage_contract, lineage=existing_lineage
        )

    def lineage_gate(self, *, env_file: Mapping[str, bool]) -> dict[str, Any] | None:  # type: ignore[override]
        lineage_checks = self.ctx.upstream_lineage_checks(self.stage_id)
        if lineage_checks is None:
            return None
        return self._record(
            status="blocked",
            blockers=[stage_blocker(self.stage_id, UPSTREAM_LINEAGE_NOT_FRESH)],
            checks={"env_file": dict(env_file), **lineage_checks, **ZERO_SPEND_FLAGS},
            stage_contract=None,
        )

    def owner_raised(self, error: str, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> dict[str, Any]:
        del gate
        return self._record(
            status="blocked",
            blockers=[stage_blocker(self.stage_id, OWNER_RAISED, error=error)],
            checks={"env_file": dict(env_file), **ZERO_SPEND_FLAGS},
            stage_contract=None,
        )

    def finish(self, owner_receipt: Any, *, env_file: Mapping[str, bool]) -> dict[str, Any]:
        ctx = self.ctx
        extraction_status = owner_receipt.get("status") if isinstance(owner_receipt, Mapping) else None
        blockers: list[dict[str, Any]] = []
        if extraction_status != "succeeded":
            blockers.append(stage_blocker(self.stage_id, OWNER_NOT_SUCCEEDED, observed_status=extraction_status))
        root = extraction_root(ctx)
        coverage_file = root / "v15_extraction_coverage.json"
        checks: dict[str, Any] = {
            "env_file": dict(env_file),
            "extraction_receipt_status": extraction_status,
            "coverage_path": workspace_relpath(ctx.workspace, coverage_file),
            "coverage_sha256": sha256_file(coverage_file) if coverage_file.is_file() else None,
            "by_topic_dir": workspace_relpath(ctx.workspace, root / "by-topic"),
            "by_artifact_dir": workspace_relpath(ctx.workspace, root / "by-artifact"),
            "extraction_receipt_blockers": sanitize_for_receipt(
                list(owner_receipt.get("blockers") or []) if isinstance(owner_receipt, Mapping) else []
            ),
            **ZERO_SPEND_FLAGS,
        }
        stage_contract: dict[str, Any] | None = None
        if not blockers:
            stage_contract = self._verify(checks)
            if stage_contract["status"] != "pass":
                blockers.append(stage_blocker(self.stage_id, FINAL_STAGE_CONTRACT_FAILED))
        return self._record(
            status="succeeded" if not blockers else "blocked",
            blockers=blockers,
            checks=checks,
            stage_contract=stage_contract,
        )


# --------------------------------------------------------------------------
# Runner-summary convention: (exit_code, run_summary) from the runner's writers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunnerSummaryExhaustiveConvention:
    """Stage 7A under an orchestrator's own receipt writers.

    ``conventions`` are the orchestrator's
    :class:`~conference_pipeline.production.orchestrator.RunConventions`;
    ``upstream_gate`` its pre-work lineage gate; ``write_blocker(reason,
    remediation)`` writes its blocker receipt and returns the path;
    ``receipt_fields`` are the top-level fields every receipt of the runner
    carries (its environment-variable inventory). The owner's own receipt is
    the body of the stage receipt; a contract failure writes a blocker
    receipt and blocks.
    """

    ctx: ProductionRunnerContext
    conventions: Any
    stage: StageSpec
    mode: str
    started_at: str
    upstream_gate: Callable[..., Any]
    write_blocker: Callable[[str, str], Path]
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]]
    write_json: Callable[[Path, dict[str, Any]], None]
    receipt_fields: Mapping[str, Any] = field(default_factory=dict)

    @property
    def _workspace(self) -> Path:
        return self.ctx.workspace

    def _summary(self, receipt: dict[str, Any], *, status: str, halted: bool, blocker_receipt_path: str | None = None) -> Any:
        return self.conventions.write_summary(
            workspace=self._workspace,
            run_id=self.ctx.run_id,
            mode=self.mode,
            status=status,
            stage_receipt=receipt,
            halt_status="halted" if halted else None,
            halt_stage_id=self.stage.id if halted else None,
            blocker_receipt_path=blocker_receipt_path,
        )

    def with_contract_evidence(
        self, receipt: dict[str, Any], *, env_file: Mapping[str, bool], bind: bool = False
    ) -> dict[str, Any]:
        """The owner's receipt, completed as this runner's stage receipt and verified."""

        ctx = self.ctx
        registry = load_stage_script_registry()
        checks = receipt.get("checks") if isinstance(receipt.get("checks"), dict) else {}
        checks = {
            **checks,
            "env_file": dict(env_file),
            **exhaustive_output_counts(ctx.workspace, ctx.run_id),
            **ZERO_SPEND_FLAGS,
        }
        details = receipt.get("details")
        if isinstance(details, dict):
            token_usage = details.get("token_usage")
            if isinstance(token_usage, dict):
                checks["token_usage"] = token_usage
            coverage = details.get("coverage")
            if isinstance(coverage, dict):
                checks["coverage_classes"] = coverage.get("classes")
        receipt.update(
            {
                "receipt_type": ctx.receipt_type,
                "workspace": str(ctx.workspace),
                "mode": self.mode,
                "stage_order": self.stage.order,
                "started_at": receipt.get("started_at") or self.started_at,
                "ended_at": receipt.get("ended_at") or self.conventions.now(),
                "resumable": receipt.get("status") == "succeeded",
                "receipt_path": str(ctx.receipt_path(self.stage.id)),
                "checks": checks,
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": list(required_script_ids_for_stage(registry, self.stage.id)),
                "stage_contract": receipt.get("stage_contract") if isinstance(receipt.get("stage_contract"), dict) else {},
                **dict(self.receipt_fields),
                "provider_values_recorded": False,
                "secrets_copied": False,
                "next_action": self.stage.next_action,
            }
        )
        # A resume path (``bind=False``) verifies the receipt's own recorded
        # lineage and never mints new lineage for it; a fresh run binds.
        receipt = ctx.finalize(receipt, stage_id=self.stage.id, bind=bind)
        if receipt.get("status") == "succeeded":
            contract = ctx.verify_contract(stage_id=self.stage.id, receipt=receipt, bind=False, mode=self.mode)
            receipt["stage_contract"] = contract.to_dict()
            if contract.status != "pass":
                blocker_codes = [
                    str(blocker.get("code") or "stage_contract_blocked")
                    for blocker in contract.blockers
                    if isinstance(blocker, dict)
                ]
                reason_suffix = ", ".join(blocker_codes[:5]) or "stage_contract_blocked"
                blocker_path = self.write_blocker(
                    f"{CONTRACT_FAILED_ERROR}: {reason_suffix}", self.stage.next_action
                )
                receipt["status"] = "blocked"
                receipt["resumable"] = False
                receipt["error"] = CONTRACT_FAILED_ERROR
                receipt["blocker_receipt_path"] = str(blocker_path)
        return receipt

    def start(self) -> Any | None:
        from conference_pipeline.production.orchestrator import halt_for_active_blocker

        if not self.conventions.active_blockers(self._workspace, run_id=self.ctx.run_id):
            return None
        return halt_for_active_blocker(
            self.conventions,
            workspace=self._workspace,
            run_id=self.ctx.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
        )

    def stage6_blocked(self, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> Any:
        receipt = self.conventions.build_receipt(
            workspace=self._workspace,
            run_id=self.ctx.run_id,
            stage=self.stage,
            mode=self.mode,
            status="blocked",
            started_at=self.started_at,
            checks={
                "env_file": dict(env_file),
                "stage6_receipt_path": workspace_relpath(self._workspace, gate.path),
                "stage6_receipt_present": gate.present,
                "stage6_receipt_error": gate.error,
                "stage6_status": gate.status,
                **ZERO_SPEND_FLAGS,
            },
            error=STAGE6_REQUIRED_ERROR,
        )
        self.conventions.write_receipt(self._workspace, self.ctx.run_id, self.stage, receipt)
        return 1, self._summary(receipt, status="blocked", halted=True)

    def resumed(self, *, env_file: Mapping[str, bool]) -> Any | None:
        path = self.ctx.receipt_path(self.stage.id)
        if not path.exists():
            return None
        existing, existing_error = self.load_json(path)
        if existing_error is not None or not isinstance(existing, dict) or existing.get("status") != "succeeded":
            return None
        existing = self.with_contract_evidence(existing, env_file=env_file)
        if existing.get("status") != "succeeded":
            return None
        self.write_json(path, existing)
        return 0, self._summary(existing, status="succeeded", halted=False)

    def lineage_gate(self, *, env_file: Mapping[str, bool]) -> Any | None:
        return self.upstream_gate(
            workspace=self._workspace,
            run_id=self.ctx.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
            checks={"env_file": dict(env_file)},
        )

    def owner_raised(self, error: str, gate: Stage6Gate, *, env_file: Mapping[str, bool]) -> Any:
        receipt = self.conventions.build_receipt(
            workspace=self._workspace,
            run_id=self.ctx.run_id,
            stage=self.stage,
            mode=self.mode,
            status="blocked",
            started_at=self.started_at,
            checks={
                "env_file": dict(env_file),
                "stage6_receipt_path": workspace_relpath(self._workspace, gate.path),
                "stage6_status": gate.status,
                **ZERO_SPEND_FLAGS,
            },
            error=error,
        )
        return self.finish(receipt, env_file=env_file)

    def finish(self, owner_receipt: Any, *, env_file: Mapping[str, bool]) -> Any:
        receipt = self.with_contract_evidence(owner_receipt, env_file=env_file, bind=True)
        self.conventions.write_receipt(self._workspace, self.ctx.run_id, self.stage, receipt)
        succeeded = receipt.get("status") == "succeeded"
        summary = self._summary(
            receipt,
            status="succeeded" if succeeded else "blocked",
            halted=not succeeded,
            blocker_receipt_path=str(receipt.get("blocker_receipt_path") or "") or None,
        )
        return (0 if succeeded else 1), summary


__all__ = [
    "ExhaustiveExtractionConvention",
    "RunnerSummaryExhaustiveConvention",
    "Stage6Gate",
    "StrictExhaustiveConvention",
    "exhaustive_output_counts",
    "run_exhaustive_topic_extraction_stage",
]
