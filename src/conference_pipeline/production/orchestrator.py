"""Shared CLI, stage dispatch and run-summary conventions for conference orchestrators.

A per-conference orchestrator (``scripts/run_<slug>_production.py``) declares
an :class:`OrchestratorSpec`: its display name, default workspace and run id,
its stage table (:mod:`conference_pipeline.production.stage_table`), one entry
function per stage, and its quirks (:class:`ConferenceHooks`: extra CLI flags,
extra argument checks, non-stage commands). :func:`run` then does what every
orchestrator's ``main()`` used to do by hand:

1. parse the shared core flags (``--workspace``, ``--run-id``, ``--stage``,
   ``--mode``, ``--resume``, ``--full-rebuild``) plus the conference's own
   flags, and run the shared checks (live-only mode, ``--stage`` present);
2. run a selected non-stage command (``--prepare-*`` / ``--authorize-*``) and
   print its JSON payload;
3. dispatch ``--stage`` to its entry function;
4. report the outcome on stdout/stderr and turn it into an exit code.

The module also owns the run-summary conventions that the stage bodies share:
:func:`write_run_summary` (``summary.json``), :func:`halt_for_active_blocker`
(an unacknowledged blocker receipt stops every stage before it starts) and
:func:`run_shared_stage` (a ``conference_pipeline.production`` stage run under
those conventions). Every function takes the orchestrator's own writers and
receipt builders, so receipt bytes, identity and lineage stay the
orchestrator's.

Nothing here names a conference: display names, defaults and help texts come
from the spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.production.context import ProductionRunnerContext, StageSpec
from conference_pipeline.production.hooks import NO_HOOKS, ConferenceHooks, ExtraCommand

FULL_REBUILD_HELP = (
    "corpus-inventory-index only: ignore the chunk manifest, every stored vector, "
    "and the persisted Chroma collection, and rebuild the index from scratch."
)
STAGE_REQUIRED_MESSAGE = "--stage is required for live production dispatch"
EXTRA_COMMAND_ERRORS = (OSError, ValueError, json.JSONDecodeError)

_INVOCATION_ARGV: ContextVar[tuple[str, ...]] = ContextVar("orchestrator_invocation_argv", default=())
_MISSING: Any = object()


def default_json_default(value: Any) -> str:
    return str(value)


@dataclass(frozen=True)
class StageRun:
    """A stage entry function's outcome, normalized for reporting.

    ``exit_code`` None means "0 when status is succeeded, else 1". A message
    override of ``""`` prints nothing; ``None`` prints the reporter's default.
    """

    status: str
    summary: Any = _MISSING
    exit_code: int | None = None
    success_message: str | None = None
    failure_message: str | None = None

    @classmethod
    def from_exit(cls, exit_code: int, summary: Mapping[str, Any]) -> StageRun:
        """An ``(exit_code, run_summary)`` stage entry result."""

        return cls(status=str(summary.get("status") or ""), summary=summary, exit_code=int(exit_code))

    @classmethod
    def from_outcome(cls, outcome: Mapping[str, Any], **messages: str | None) -> StageRun:
        """A ``{"status", "summary", ...}`` stage outcome mapping."""

        return cls(status=str(outcome["status"]), summary=outcome.get("summary", _MISSING), **messages)

    def summary_payload(self) -> Any:
        if self.summary is _MISSING:
            raise KeyError("summary")
        return self.summary


StageHandler = Callable[[argparse.Namespace, StageSpec], Any]
Reporter = Callable[[argparse.Namespace, StageSpec, StageRun, "OrchestratorSpec"], int]


def report_halt_with_summary_path(
    args: argparse.Namespace, stage: StageSpec, run: StageRun, spec: OrchestratorSpec
) -> int:
    """Success on stdout, a halt (with the run summary path) on stderr.

    A zero exit code whose run summary is not ``succeeded`` becomes 1.
    """

    summary = run.summary_payload()
    exit_code = run.exit_code if run.exit_code is not None else (0 if run.status == "succeeded" else 1)
    if exit_code == 0 and summary.get("status") != "succeeded":
        exit_code = 1
    if args.summary:
        print(json.dumps(summary, indent=2, sort_keys=True, default=spec.json_default))
    elif exit_code:
        print(
            f"{spec.display_name} production halted at {args.stage}; "
            f"summary: {summary.get('summary_path')}",
            file=sys.stderr,
        )
    else:
        print(f"{spec.display_name} production stage {args.stage} succeeded")
    return exit_code


def report_with_next_action(
    args: argparse.Namespace, stage: StageSpec, run: StageRun, spec: OrchestratorSpec
) -> int:
    """Success or the stage's ``next_action`` on stderr; exit 0 only on success."""

    succeeded = run.status == "succeeded"
    if args.summary:
        print(json.dumps(run.summary_payload(), indent=2, sort_keys=True, default=spec.json_default))
    elif succeeded:
        message = run.success_message
        if message is None:
            message = f"{spec.display_name} production stage {stage.id} succeeded."
        if message:
            print(message, file=sys.stderr)
    else:
        message = run.failure_message
        if message is None:
            message = f"{spec.display_name} production stage {stage.id} blocked: {stage.next_action}"
        if message:
            print(message, file=sys.stderr)
    if run.exit_code is not None:
        return run.exit_code
    return 0 if succeeded else 1


@dataclass(frozen=True)
class OrchestratorSpec:
    """Everything the shared orchestrator needs from a conference orchestrator."""

    display_name: str
    description: str
    stages: Mapping[str, StageSpec]
    handlers: Mapping[str, StageHandler]
    default_workspace: Path | None = None
    default_run_id: str | None = None
    hooks: ConferenceHooks = NO_HOOKS
    # ``--stage`` is an argparse-required flag unless the orchestrator has
    # non-stage commands; then its absence is reported after those are ruled out.
    stage_required: bool = True
    live_only: bool = True
    # Reject an invalid --run-id with exit 2 before any command or stage runs.
    precheck_run_id: bool = False
    report: Reporter = report_halt_with_summary_path
    json_default: Callable[[Any], Any] = default_json_default
    missing_handler_message: str = "implemented stage has no dispatcher: {stage_id}"


def spec_problems(spec: OrchestratorSpec) -> list[str]:
    """Stage-table/handler mismatches an orchestrator must not ship with."""

    problems: list[str] = []
    for stage_id, stage in spec.stages.items():
        if stage.implemented and stage_id not in spec.handlers:
            problems.append(f"implemented stage {stage_id!r} has no entry function")
    for stage_id in spec.handlers:
        if stage_id not in spec.stages:
            problems.append(f"entry function registered for undeclared stage {stage_id!r}")
    for dest in spec.hooks.extra_commands:
        if dest in {"workspace", "run_id", "stage", "mode", "resume", "full_rebuild", "summary"}:
            problems.append(f"extra command {dest!r} shadows a core flag")
    return problems


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser(spec: OrchestratorSpec) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--workspace", type=Path, default=spec.default_workspace)
    parser.add_argument("--run-id", default=spec.default_run_id)
    if spec.stage_required:
        parser.add_argument("--stage", required=True, choices=sorted(spec.stages))
    else:
        parser.add_argument("--stage", choices=sorted(spec.stages))
    parser.add_argument("--mode", default="live")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-rebuild", action="store_true", help=FULL_REBUILD_HELP)
    if spec.hooks.extra_cli_args is not None:
        spec.hooks.extra_cli_args(parser)
    if not any("--summary" in action.option_strings for action in parser._actions):
        parser.add_argument("--summary", action="store_true")
    return parser


def selected_extra_command(spec: OrchestratorSpec, args: argparse.Namespace) -> tuple[str, ExtraCommand] | None:
    for dest, command in spec.hooks.extra_commands.items():
        if getattr(args, dest, False):
            return dest, command
    return None


def parse_args(argv: Sequence[str] | None, spec: OrchestratorSpec) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(spec)
    args = parser.parse_args(raw_argv)
    selected = selected_extra_command(spec, args)
    if selected is not None:
        _dest, command = selected
        if command.validate is not None:
            command.validate(parser, args, raw_argv)
        return args
    if spec.live_only and args.mode != "live":
        parser.error(f"{spec.display_name} production runner is live-only; non-live evidence is rejected")
    if not args.stage:
        parser.error(STAGE_REQUIRED_MESSAGE)
    if spec.hooks.validate_args is not None:
        spec.hooks.validate_args(parser, args, raw_argv)
    return args


def invocation_argv() -> tuple[str, ...]:
    """The raw command-line arguments of the stage being dispatched (for receipts)."""

    return _INVOCATION_ARGV.get()


def run_extra_command(command: ExtraCommand, args: argparse.Namespace, spec: OrchestratorSpec) -> int:
    try:
        payload = command.run(args)
    except EXTRA_COMMAND_ERRORS as exc:
        print(f"{command.failure_label}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True, default=spec.json_default))
    return 0


def normalize_stage_result(result: Any) -> StageRun:
    if isinstance(result, StageRun):
        return result
    if isinstance(result, tuple) and len(result) == 2:
        exit_code, summary = result
        return StageRun.from_exit(exit_code, summary)
    if isinstance(result, Mapping):
        return StageRun.from_outcome(result)
    raise TypeError(f"stage entry returned {type(result).__name__}; expected StageRun, (exit_code, summary) or outcome")


def dispatch(args: argparse.Namespace, spec: OrchestratorSpec, argv: Sequence[str] | None = None) -> int:
    """Run the selected extra command or ``--stage`` entry function; return the exit code."""

    selected = selected_extra_command(spec, args)
    if selected is not None:
        return run_extra_command(selected[1], args, spec)
    stage = spec.stages[args.stage]
    handler = spec.handlers.get(stage.id)
    if handler is None:
        raise AssertionError(spec.missing_handler_message.format(stage_id=stage.id))
    token = _INVOCATION_ARGV.set(tuple(sys.argv[1:] if argv is None else argv))
    try:
        result = handler(args, stage)
    finally:
        _INVOCATION_ARGV.reset(token)
    return spec.report(args, stage, normalize_stage_result(result), spec)


def run(argv: Sequence[str] | None, spec: OrchestratorSpec, *, args: argparse.Namespace | None = None) -> int:
    """An orchestrator's whole ``main()``: parse, pre-check, dispatch, report.

    ``args`` is the already-parsed namespace when the orchestrator parses
    through its own ``parse_args`` entry point.
    """

    if args is None:
        args = parse_args(argv, spec)
    if spec.precheck_run_id:
        from conference_pipeline.workflow_runs import validate_run_id

        try:
            validate_run_id(args.run_id)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    return dispatch(args, spec, argv)


# --------------------------------------------------------------------------
# Run summary and active-blocker conventions
# --------------------------------------------------------------------------


def write_run_summary(
    *,
    workspace: Path,
    run_id: str,
    mode: str,
    status: str,
    stage_receipt: Mapping[str, Any],
    summary_path: Path,
    generated_at: str,
    unblock_receipts: Sequence[Mapping[str, Any]],
    write_json: Callable[[Path, dict[str, Any]], None],
    halt_status: str | None = None,
    halt_stage_id: str | None = None,
    blocker_receipt_path: str | None = None,
) -> dict[str, Any]:
    """Write ``summary.json`` for one stage invocation and return it.

    A ``succeeded`` request over a stage receipt that is not succeeded is
    reported as that receipt's status and halted at its stage, so the summary
    never claims more than the receipt does.
    """

    unblock_acknowledgements = [
        str(receipt.get("receipt_path"))
        for receipt in unblock_receipts
        if receipt.get("run_id") == run_id and receipt.get("receipt_path")
    ]
    stage_receipt_status = str(stage_receipt.get("status") or "")
    effective_status = status
    effective_halt_status = halt_status
    effective_halt_stage_id = halt_stage_id
    effective_blocker_receipt_path = blocker_receipt_path
    if status == "succeeded" and stage_receipt_status != "succeeded":
        effective_status = stage_receipt_status or "blocked"
        effective_halt_status = effective_halt_status or "halted"
        effective_halt_stage_id = effective_halt_stage_id or str(stage_receipt.get("stage_id") or "")
        effective_blocker_receipt_path = effective_blocker_receipt_path or str(
            stage_receipt.get("blocker_receipt_path") or ""
        )
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "workspace": str(workspace),
        "mode": mode,
        "status": effective_status,
        "generated_at": generated_at,
        "summary_path": str(summary_path),
        "halt_status": effective_halt_status,
        "halt_stage_id": effective_halt_stage_id,
        "blocker_receipt_path": effective_blocker_receipt_path or None,
        "unblock_acknowledgements": unblock_acknowledgements,
        "stages": [stage_receipt],
    }
    write_json(summary_path, payload)
    return payload


@dataclass(frozen=True)
class RunConventions:
    """The orchestrator's own receipt and summary writers, bound for one call.

    ``build_receipt(**kwargs)`` builds a stage receipt (the orchestrator's
    receipt type and code identity), ``write_receipt(workspace, run_id, stage,
    receipt)`` persists it (preserving the prior one), ``write_summary(**kwargs)``
    writes ``summary.json``, ``active_blockers(workspace, run_id=...)`` lists
    unacknowledged blocker receipts, ``context(workspace, run_id)`` builds the
    shared-stage context, ``run_dir(workspace, run_id)`` is the run directory,
    ``now()`` the timestamp source and ``validate_run_id`` the run-id guard.
    ``write_blocker(**kwargs)`` writes a blocker receipt (``workspace``,
    ``run_id``, ``stage_id``, ``blocker_class``, ``reason``,
    ``remediation_path``) and returns its path; it is required only by
    stages recorded through :class:`StageRecorder`.
    """

    build_receipt: Callable[..., dict[str, Any]]
    write_receipt: Callable[[Path, str, StageSpec, dict[str, Any]], None]
    write_summary: Callable[..., dict[str, Any]]
    active_blockers: Callable[..., list[dict[str, Any]]]
    context: Callable[[Path, str], ProductionRunnerContext]
    run_dir: Callable[[Path, str], Path]
    now: Callable[[], str]
    validate_run_id: Callable[[str], str]
    write_blocker: Callable[..., Path] | None = None


def halt_for_active_blocker(
    conventions: RunConventions,
    *,
    workspace: Path,
    run_id: str,
    stage: StageSpec,
    mode: str,
    started_at: str,
) -> tuple[int, dict[str, Any]]:
    """Write a blocked stage receipt and run summary for the first active blocker."""

    blockers = conventions.active_blockers(workspace, run_id=run_id)
    first = blockers[0]
    blocker_path = str(first.get("receipt_path") or "")
    receipt = conventions.build_receipt(
        workspace=workspace,
        run_id=run_id,
        stage=stage,
        mode=mode,
        status="blocked",
        started_at=started_at,
        checks={"active_blocker": first},
        error="active blocker receipt must be acknowledged before resume",
        blocker_receipt_path=blocker_path,
    )
    conventions.write_receipt(workspace, run_id, stage, receipt)
    summary = conventions.write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=mode,
        status="blocked",
        stage_receipt=receipt,
        halt_status="halted",
        halt_stage_id=str(first.get("stage_id") or stage.id),
        blocker_receipt_path=blocker_path,
    )
    return 1, summary


@dataclass(frozen=True)
class StageRecorder:
    """One stage invocation's receipt outcomes under the run conventions.

    ``blocked`` writes a blocker receipt, the blocked stage receipt (carrying
    the reason as its error and the blocker receipt path) and a halted run
    summary, and returns ``(1, summary)``; ``succeeded`` writes the stage
    receipt and run summary and returns ``(0, summary)`` (the summary reports
    a receipt the stage contract downgraded as blocked);
    ``halt_for_active_blocker`` halts on the first unacknowledged blocker.
    """

    conventions: RunConventions
    workspace: Path
    run_id: str
    stage: StageSpec
    mode: str
    started_at: str

    def halt_for_active_blocker(self) -> tuple[int, dict[str, Any]]:
        return halt_for_active_blocker(
            self.conventions,
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
        )

    def blocked(
        self,
        *,
        checks: Mapping[str, Any],
        reason: str,
        remediation: str,
        blocker_class: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        from conference_pipeline.provider_receipts import BlockerClass

        if self.conventions.write_blocker is None:
            raise ValueError("run conventions declare no blocker-receipt writer")
        blocker_path = self.conventions.write_blocker(
            workspace=self.workspace,
            run_id=self.run_id,
            stage_id=self.stage.id,
            blocker_class=blocker_class or BlockerClass.STAGE_BLOCKED,
            reason=reason,
            remediation_path=remediation,
        )
        receipt = self.conventions.build_receipt(
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            status="blocked",
            started_at=self.started_at,
            checks=dict(checks),
            error=reason,
            blocker_receipt_path=str(blocker_path),
        )
        self.conventions.write_receipt(self.workspace, self.run_id, self.stage, receipt)
        summary = self.conventions.write_summary(
            workspace=self.workspace,
            run_id=self.run_id,
            mode=self.mode,
            status="blocked",
            stage_receipt=receipt,
            halt_status="halted",
            halt_stage_id=self.stage.id,
            blocker_receipt_path=str(blocker_path),
        )
        return 1, summary

    def succeeded(self, *, checks: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        receipt = self.conventions.build_receipt(
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            status="succeeded",
            started_at=self.started_at,
            checks=dict(checks),
        )
        self.conventions.write_receipt(self.workspace, self.run_id, self.stage, receipt)
        summary = self.conventions.write_summary(
            workspace=self.workspace,
            run_id=self.run_id,
            mode=self.mode,
            status="succeeded",
            stage_receipt=receipt,
        )
        return 0, summary


def run_shared_stage(
    conventions: RunConventions,
    args: argparse.Namespace,
    stage: StageSpec,
    run_stage: Callable[[ProductionRunnerContext], Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Run a ``conference_pipeline.production`` stage under the run conventions.

    The shared stage writes its own receipt (finalized with the orchestrator's
    identity through the context); this only halts on an acknowledged-blocker
    debt first and records the run summary after.
    """

    workspace = args.workspace.resolve()
    run_id = conventions.validate_run_id(args.run_id)
    started_at = conventions.now()
    conventions.run_dir(workspace, run_id).mkdir(parents=True, exist_ok=True)
    if conventions.active_blockers(workspace, run_id=run_id):
        return halt_for_active_blocker(
            conventions, workspace=workspace, run_id=run_id, stage=stage, mode=args.mode, started_at=started_at
        )
    outcome = run_stage(conventions.context(workspace, run_id))
    receipt = outcome["receipt"]
    status = str(outcome["status"])
    summary = conventions.write_summary(
        workspace=workspace,
        run_id=run_id,
        mode=args.mode,
        status=status,
        stage_receipt=receipt,
        halt_status=None if status == "succeeded" else "halted",
        halt_stage_id=None if status == "succeeded" else stage.id,
    )
    summary["stage_summary"] = outcome.get("summary")
    return (0 if status == "succeeded" else 1), summary


@dataclass(frozen=True)
class RunnerSummaryConvention:
    """A gated shared stage under the orchestrator's own receipt writers.

    The :class:`~conference_pipeline.production.receipts.StageReceiptConvention`
    for orchestrators whose stages return ``(exit_code, run_summary)``: an
    unacknowledged blocker receipt halts first; resume accepts a succeeded
    receipt whose contract still verifies (``require_lineage``) and rewrites
    the run summary; ``block_stage`` and ``upstream_gate`` are the
    orchestrator's own blocked-receipt writers (a blocker receipt with reason
    and remediation, and the pre-spend lineage gate); success builds, writes
    and summarizes a receipt through ``conventions``. Execution evidence is
    not recorded under this convention.
    """

    conventions: RunConventions
    stage: StageSpec
    workspace: Path
    run_id: str
    mode: str
    started_at: str
    block_stage: Callable[..., tuple[int, dict[str, Any]]]
    upstream_gate: Callable[..., tuple[int, dict[str, Any]] | None]
    verify_contract: Callable[..., Any]
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]]

    def start(self) -> tuple[int, dict[str, Any]] | None:
        if not self.conventions.active_blockers(self.workspace, run_id=self.run_id):
            return None
        return halt_for_active_blocker(
            self.conventions,
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
        )

    def resumed(self) -> tuple[int, dict[str, Any]] | None:
        path = self.conventions.run_dir(self.workspace, self.run_id) / self.stage.receipt_filename
        if not path.exists():
            return None
        existing, _error = self.load_json(path)
        if not isinstance(existing, dict) or existing.get("status") != "succeeded":
            return None
        contract = self.verify_contract(
            workspace=self.workspace,
            run_id=self.run_id,
            stage_id=self.stage.id,
            receipt=existing,
            mode=self.mode,
            require_lineage=True,
        )
        if contract.status != "pass":
            return None
        summary = self.conventions.write_summary(
            workspace=self.workspace, run_id=self.run_id, mode=self.mode, status="succeeded", stage_receipt=existing
        )
        return 0, summary

    def block(
        self, code: str, *, reason: str, remediation: str, checks: Mapping[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        return self.block_stage(
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
            blocker_code=code,
            reason=reason,
            remediation_path=remediation,
            checks=dict(checks),
        )

    def lineage_gate(
        self, checks: Mapping[str, Any], *, input_digests: Mapping[str, Any] | None = None
    ) -> tuple[int, dict[str, Any]] | None:
        return self.upstream_gate(
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            started_at=self.started_at,
            checks={**checks, **dict(input_digests or {})},
        )

    def succeed(self, checks: dict[str, Any], *, owner_evidence: Mapping[str, str]) -> tuple[int, dict[str, Any]]:
        del owner_evidence
        receipt = self.conventions.build_receipt(
            workspace=self.workspace,
            run_id=self.run_id,
            stage=self.stage,
            mode=self.mode,
            status="succeeded",
            started_at=self.started_at,
            checks=checks,
        )
        self.conventions.write_receipt(self.workspace, self.run_id, self.stage, receipt)
        succeeded = receipt.get("status") == "succeeded"
        summary = self.conventions.write_summary(
            workspace=self.workspace,
            run_id=self.run_id,
            mode=self.mode,
            status="succeeded" if succeeded else "blocked",
            stage_receipt=receipt,
            halt_status=None if succeeded else "halted",
            halt_stage_id=None if succeeded else self.stage.id,
            blocker_receipt_path=str(receipt.get("blocker_receipt_path") or "") or None,
        )
        return (0 if succeeded else 1), summary
