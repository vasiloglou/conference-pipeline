"""Step primitives the shared presentation deck chain is built from.

Every owner-script step in the chain goes through :func:`owner_step` (or
:func:`narration_content_step` for the lint/review scripts whose report, not
exit code, is the verdict). Each step derives its inputs from argv plus
declared extras, and on resume is skipped only while the deck's step ledger
(``decks/<id>/step_lineage.json``, ``presentation.step_lineage``) says every
input it read and every output it wrote still carry the digests it recorded.
A step reached with no ledger bound always runs: existence of an output is
not evidence that it is current.

A failed step returns a :class:`StepFailure`-shaped outcome (blocker code +
sanitized checks); the stage shell decides how that is recorded. Nothing here
writes a stage receipt.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import contextvars
import json
import os
import time
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from glob import glob
from hashlib import sha256
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.step_lineage import (
    DEFAULT_REJECT_STATUSES,
    DeckStepLineage,
    SkipDecision,
    StepInput,
    StepSpec,
)
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.provider_receipts import provider_receipt_context, sanitize_for_receipt, utc_now
from conference_pipeline.workflow_runs import write_json_atomic

STAGE_ID = "presentation"

CommandRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]

LINEAGE: contextvars.ContextVar[DeckStepLineage | None] = contextvars.ContextVar(
    "production_presentation_lineage", default=None
)
ADOPT_EXISTING: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "production_presentation_adopt_existing", default=False
)
# Step ids the operator asked to regenerate even when their ledger entry is
# current (e.g. after a generator fix that changes no recorded input). The
# step's new outputs then re-run everything downstream through normal lineage.
RERUN_STEPS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "production_presentation_rerun_steps", default=frozenset()
)
# Slide ids (or unique prefixes) whose figure specs the operator asked to
# re-author; the other slides' specs are left alone and the receipt merged.
FIGURE_SLIDE_IDS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "production_presentation_figure_slide_ids", default=()
)

# Flags whose values are files the step WRITES and that never name an input
# of any presentation step; paths behind them are excluded from the derived
# input set (a receipt written by a previous run must not read as an input
# that appeared). Flags that are outputs for one step and inputs for another
# (--slide-mapping, --delivery-*) are handled by each step's declared outputs.
STEP_OUTPUT_FLAGS = frozenset(
    {
        "--output",
        "--output-json",
        "--output-md",
        "--output-html",
        "--output-pdf",
        "--markdown-output",
        "--markdown-out",
        "--receipt",
        "--receipt-path",
        "--receipt-output",
        "--report-output",
        "--report-out",
        "--review-out",
        "--json-out",
        "--prompt-output",
        "--frame-manifest",
        "--frames-dir",
        "--contact-sheet",
        "--single-speaker-report",
        "--notebooklm-report",
        "--model-receipt-output",
        "--code-review-output",
        "--figure-specs-output",
        "--figure-code-dir",
        "--debug-script-md",
        "--debug-notebooklm-script-md",
    }
)


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Load a JSON object, returning ``(payload, error)`` -- never raises."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"expected JSON object, found {type(payload).__name__}"
    return payload, None


def default_command_runner(repo_root: Path) -> CommandRunner:
    """A list-based subprocess runner rooted at the repository -- never a shell."""

    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=repo_root, check=False, text=True, capture_output=True)

    return _run


@dataclass
class PresentationChainContext:
    """What every step of one presentation run shares.

    ``block`` turns a failure into the outcome dict the chain returns; the
    stage shell supplies it so per-deck failures are recorded under the
    runner's conventions. ``checks`` are the stage-level checks a blocked
    outcome carries alongside the step's own.
    """

    ctx: ProductionRunnerContext
    command_runner: CommandRunner
    stage_id: str = STAGE_ID
    automated_approver_id: str = ""
    checks: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.automated_approver_id:
            self.automated_approver_id = f"conference_pipeline.{self.ctx.runner_id}_production_runner"

    def block(self, *, blocker_code: str, checks: Mapping[str, Any]) -> dict[str, Any]:
        """A zero-receipt blocked outcome for one step; the shell records it."""

        payload = sanitize_for_receipt({**dict(checks), "blocker_code": blocker_code})
        blocker = {"code": blocker_code, "affected_stage": self.stage_id}
        return {
            "status": "blocked",
            "blocker_code": blocker_code,
            "checks": payload,
            "receipt": {"blockers": [blocker], "checks": payload},
        }


@contextlib.contextmanager
def bind_deck_ledger(pc: PresentationChainContext, deck_dir: Path, *, adopt_existing: bool) -> Iterator[DeckStepLineage]:
    """Bind ``deck_dir``'s step ledger for the duration of a chain."""

    deck_dir.mkdir(parents=True, exist_ok=True)
    ledger = DeckStepLineage.for_deck_dir(deck_dir, workspace=pc.ctx.workspace, repo_root=pc.ctx.repo_root)
    lineage_token = LINEAGE.set(ledger)
    adopt_token = ADOPT_EXISTING.set(bool(adopt_existing))
    try:
        yield ledger
    finally:
        ADOPT_EXISTING.reset(adopt_token)
        LINEAGE.reset(lineage_token)


def step_spec(
    pc: PresentationChainContext,
    *,
    command: Sequence[str],
    outputs: Sequence[Path],
    step_id: str | None = None,
    extra_inputs: Sequence[StepInput | Path] = (),
    tracked_outputs: Sequence[StepInput | Path] = (),
    editable_outputs: Sequence[Path] = (),
    in_place: bool = False,
    values: Mapping[str, Any] | None = None,
    accept_statuses: frozenset[str] | None = None,
    reject_statuses: frozenset[str] = DEFAULT_REJECT_STATUSES,
    directory_patterns: Mapping[Path, str] | None = None,
    ignored_inputs: Sequence[Path] = (),
) -> StepSpec:
    return StepSpec.from_command(
        list(command),
        workspace=pc.ctx.workspace,
        outputs=tuple(outputs),
        step_id=step_id,
        output_flags=STEP_OUTPUT_FLAGS,
        extra_inputs=tuple(extra_inputs),
        tracked_outputs=tuple(tracked_outputs),
        editable_outputs=tuple(editable_outputs),
        in_place=in_place,
        values=dict(values or {}),
        accept_statuses=accept_statuses,
        reject_statuses=reject_statuses,
        directory_patterns=directory_patterns,
        ignored_paths=tuple(ignored_inputs),
        repo_root=pc.ctx.repo_root,
    )


def step_gate(spec: StepSpec, *, resume: bool) -> tuple[bool, dict[str, Any] | None]:
    """``(skip, snapshot)`` for one step against the ledger bound to this chain.

    With ``--adopt-existing-outputs`` a step that has no ledger entry but
    whose outputs already exist is adopted from current disk state first, so
    a deck built before the ledger existed does not re-run its paid chain
    once; without it, that first resume re-runs the step.
    """

    ledger = LINEAGE.get()
    if ledger is None:
        return False, None
    if spec.step_id in RERUN_STEPS.get():
        decision = SkipDecision(False, "operator_rerun_requested")
        if resume:
            ledger.note_rerun(spec, decision)
        return False, ledger.snapshot_inputs(spec)
    if resume and ADOPT_EXISTING.get():
        ledger.adopt(spec)
    decision = ledger.should_skip(spec, resume=resume)
    if decision.skip:
        ledger.note_skip(spec, decision)
        return True, None
    if resume:
        ledger.note_rerun(spec, decision)
    return False, ledger.snapshot_inputs(spec)


def step_record(spec: StepSpec, snapshot: dict[str, Any] | None) -> None:
    ledger = LINEAGE.get()
    if ledger is not None and snapshot is not None:
        ledger.record(spec, snapshot)


def owner_step(
    pc: PresentationChainContext,
    *,
    resume: bool,
    command: list[str],
    paid: bool,
    outputs: tuple[Path, ...],
    failed_blocker_code: str,
    missing_blocker_code: str,
    extra_checks: Mapping[str, Any],
    step_id: str | None = None,
    extra_inputs: Sequence[StepInput | Path] = (),
    tracked_outputs: Sequence[StepInput | Path] = (),
    editable_outputs: Sequence[Path] = (),
    in_place: bool = False,
    values: Mapping[str, Any] | None = None,
    accept_statuses: frozenset[str] | None = None,
    reject_statuses: frozenset[str] = DEFAULT_REJECT_STATUSES,
    directory_patterns: Mapping[Path, str] | None = None,
    ignored_inputs: Sequence[Path] = (),
) -> dict[str, Any] | None:
    """Run (or skip, when its ledger entry is current) one owner-script subprocess.

    Returns a blocked outcome on failure, or ``None`` on success (including a
    resume-safe skip); the caller records the owner-evidence token only on
    ``None``. ``command`` is always a ``list[str]``; never ``shell=True``.
    ``in_place`` marks a step that rewrites files earlier steps produced,
    ``editable_outputs`` marks operator-edit surfaces that are adopted rather
    than regenerated, ``accept_statuses`` marks a verdict output that may only
    be reused when it passed.
    """

    spec = step_spec(
        pc,
        command=command,
        outputs=outputs,
        step_id=step_id,
        extra_inputs=extra_inputs,
        tracked_outputs=tracked_outputs,
        editable_outputs=editable_outputs,
        in_place=in_place,
        values=values,
        accept_statuses=accept_statuses,
        reject_statuses=reject_statuses,
        directory_patterns=directory_patterns,
        ignored_inputs=ignored_inputs,
    )
    skip, snapshot = step_gate(spec, resume=resume)
    if skip:
        return None
    result = run_command(pc, command, paid=paid)
    if result.returncode != 0:
        return pc.block(
            blocker_code=failed_blocker_code,
            checks={
                **extra_checks,
                "command": command,
                "stdout": sanitize_for_receipt(result.stdout[-2000:]),
                "stderr": sanitize_for_receipt(result.stderr[-2000:]),
            },
        )
    missing = [str(path) for path in outputs if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        return pc.block(
            blocker_code=missing_blocker_code,
            checks={**extra_checks, "command": command, "missing_outputs": missing},
        )
    step_record(spec, snapshot)
    return None


def run_command(pc: PresentationChainContext, command: list[str], *, paid: bool) -> subprocess.CompletedProcess[str]:
    """Run one subprocess, inside a provider-receipt context when it is paid."""

    if paid:
        with provider_receipt_context(
            workspace_root=pc.ctx.workspace, run_id=pc.ctx.run_id, stage_id=pc.stage_id, caller=pc.ctx.receipt_caller
        ):
            return pc.command_runner(command)
    return pc.command_runner(command)


def narration_content_step(
    pc: PresentationChainContext,
    *,
    resume: bool,
    command: list[str],
    paid: bool,
    report_path: Path,
    failed_blocker_code: str,
    extra_checks: Mapping[str, Any],
    values: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run one narration lint/review owner script whose report content, not
    exit code, is the verdict.

    The script intentionally exits non-zero when its report records a CONTENT
    finding; a genuine execution failure (the report is never written) still
    blocks, a content finding does not -- the chain always proceeds to the
    paired fix step. The report's own ``status`` is a content verdict, so no
    status rejects reuse here.
    """

    spec = step_spec(pc, command=command, outputs=(report_path,), values=values, reject_statuses=frozenset())
    skip, snapshot = step_gate(spec, resume=resume)
    if skip:
        return None
    result = run_command(pc, command, paid=paid)
    if not report_path.is_file() or report_path.stat().st_size <= 0:
        return pc.block(
            blocker_code=failed_blocker_code,
            checks={
                **extra_checks,
                "command": command,
                "stdout": sanitize_for_receipt(result.stdout[-2000:]),
                "stderr": sanitize_for_receipt(result.stderr[-2000:]),
            },
        )
    step_record(spec, snapshot)
    return None


def ledger_drift(ctx: ProductionRunnerContext, receipt: Mapping[str, Any]) -> dict[str, list[str]]:
    """Per deck, the recorded paths whose disk state moved since the receipt.

    A succeeded presentation receipt is reusable only while every deck ledger
    it recorded still exists and reports no drift; otherwise the per-deck
    chain must be walked so the ledger can decide which steps re-run. A
    missing ledger counts as drift.
    """

    checks = receipt.get("checks") if isinstance(receipt.get("checks"), Mapping) else {}
    recorded = checks.get("step_lineage") if isinstance(checks.get("step_lineage"), Mapping) else None
    if recorded is None:
        return {"<stage>": ["step_lineage not recorded on receipt"]}
    drift: dict[str, list[str]] = {}
    for deck_id in recorded:
        deck_dir = ctx.deck_dir(str(deck_id))
        if not (deck_dir / "step_lineage.json").is_file():
            drift[str(deck_id)] = ["<ledger missing>"]
            continue
        moved = DeckStepLineage.for_deck_dir(deck_dir, workspace=ctx.workspace, repo_root=ctx.repo_root).drift()
        if moved:
            drift[str(deck_id)] = moved
    return drift


def corpus_inventory_source_inputs(ctx: ProductionRunnerContext, deck_id: str, kind: str) -> list[str]:
    """Candidate source files for a deck's source-corpus inventory.

    Topic decks glob the analysis-foundation artifacts of their topic. The
    executive decks have no numbered ``analysis/<id>.md`` of their own, so
    they fall back to the two conference-level analysis artifacts PLUS every
    topic's reference ledger: an executive summary IS the union of the
    topics, and the ledgers are the one shape the ledger harvester reads
    (markdown alone contributes only contribution notes bound to source ids
    that already exist, and left an executive deck with zero sources).
    """

    workspace = ctx.workspace
    if kind == "executive":
        ledgers = sorted(
            str(path.relative_to(workspace)) for path in (workspace / "analysis" / "topic_reference_ledgers").glob("*.json")
        )
        return [
            *(
                str(rel)
                for rel in (str(ctx.analysis_executive_summary), str(ctx.analysis_trend_report))
                if (workspace / rel).is_file()
            ),
            *ledgers,
        ]
    patterns = [
        f"analysis/{deck_id}.md",
        f"analysis/deep_dive_topics/{deck_id}__dd0*__global.md",
        f"analysis/deep_dive_topics/{deck_id}__dd0*__heuristic.md",
        f"analysis/exhaustive_summaries/{deck_id}__papers.json",
        f"analysis/exhaustive_summaries/{deck_id}__papers.md",
        f"analysis/exhaustive_summaries/{deck_id}__transcripts.json",
        f"analysis/exhaustive_summaries/{deck_id}__transcripts.md",
        f"analysis/narrative_source_packs/{deck_id}.json",
        f"analysis/provenance/{deck_id}.json",
        f"analysis/provenance/{deck_id}.md",
        f"analysis/provenance/query_plans/{deck_id}.json",
        f"analysis/topic_reference_candidate_resolution/{deck_id}*.*",
        f"analysis/topic_reference_ledgers/{deck_id}*.json",
        f"analysis/topic_reference_repairs/{deck_id}*.*",
        f"analysis/topic_reference_validation/{deck_id}*.*",
        f"{ctx.decks_dir}/{deck_id}/extracted_content.json",
        f"{ctx.decks_dir}/{deck_id}/ranked_papers.json",
        f"{ctx.decks_dir}/{deck_id}/source_attributions.json",
        f"{ctx.decks_dir}/{deck_id}/source_pack_ranked_papers.json",
        f"{ctx.decks_dir}/{deck_id}/source_pack_receipt.json",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        for match in sorted(glob(str(workspace / pattern))):
            candidates.append(str(Path(match).relative_to(workspace)))
    return [item for item in candidates if (workspace / item).is_file()]


def accepted_review(report_path: Path, *, deck_id: str, accepted_deck_ids: frozenset[str]) -> dict[str, Any] | None:
    """The operator-acceptance record for a deck whose review failed, or None.

    None unless the deck was named with ``--accept-presentation-review-findings``
    and its current review report exists with a non-passing status; a passing
    review takes the normal path.
    """

    if deck_id not in accepted_deck_ids or not report_path.is_file():
        return None
    payload, error = load_json(report_path)
    if error or not isinstance(payload, dict) or payload.get("status") in ("passed", "warnings"):
        return None
    feedback = payload.get("aggregated_feedback") if isinstance(payload.get("aggregated_feedback"), dict) else {}
    return {
        "accepted_by": "operator:--accept-presentation-review-findings",
        "report": str(report_path),
        "report_sha256": file_sha256(report_path),
        "status": payload.get("status"),
        "accepted_blockers": list(feedback.get("blockers") or []),
    }


def supersede_failed_review(report_path: Path) -> Path | None:
    """Move a non-passing presentation review aside so a resumed run re-reviews."""

    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    status = payload.get("status") if isinstance(payload, dict) else None
    if status in ("passed", "warnings"):
        return None
    stamp = utc_now().replace(":", "").replace("-", "")
    attempts = report_path.parent / "gate_a_attempts"
    attempts.mkdir(exist_ok=True)
    superseded = attempts / f"{report_path.stem}.superseded-{stamp}-{time.time_ns()}{report_path.suffix}"
    os.replace(report_path, superseded)
    return superseded


def bootstrap_delivery_stub(*, deck_dir: Path, deck_contract: Mapping[str, Any]) -> None:
    """Write minimal stub delivery contracts if they do not already exist.

    ``generate_render_spec.py`` hard-requires both delivery contracts to exist
    with a ``segments`` list whose ``slide_id`` sequence matches the deck
    contract; the real presenter-script step overwrites these stubs.
    """

    slides = deck_contract.get("slides")
    slides = slides if isinstance(slides, list) else []
    for name, style in (("delivery_single_speaker.json", "single_speaker"), ("delivery_notebooklm.json", "notebooklm")):
        path = deck_dir / name
        if path.is_file() and path.stat().st_size > 0:
            # A delivery contract from an earlier deck build (a regenerated
            # blueprint renames slides) blocks generate_render_spec with
            # "segment slide IDs drift from deck" -- ICML 2026, 2026-10-01.
            # It is stale either way; keep it beside the stub, never delete it.
            existing, _error = load_json(path)
            existing_ids = [
                seg.get("slide_id") for seg in (existing or {}).get("segments") or [] if isinstance(seg, dict)
            ]
            contract_ids = [slide.get("slide_id") for slide in slides if isinstance(slide, dict)]
            if existing_ids == contract_ids:
                continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path.rename(path.with_name(f"{path.stem}.stale-{stamp}{path.suffix}"))
        segments: list[dict[str, Any]] = []
        clock = 0.0
        for index, slide in enumerate(slides):
            if not isinstance(slide, dict):
                continue
            segments.append(
                {
                    "segment_id": f"seg-{index + 1:02d}",
                    "slide_id": slide.get("slide_id"),
                    "slide_number": index + 1,
                    "speaker": "stub",
                    "start_seconds": clock,
                    "end_seconds": clock + 12.0,
                    "duration_seconds": 12.0,
                    "narration": "(bootstrap stub -- regenerated at presenter-script stage)",
                    "source_chips": [],
                    "source_ids": [],
                    "source_labels": [],
                    "visual_asset_refs": [],
                    "slide_kind": slide.get("slide_kind", "content"),
                }
            )
            clock += 12.0
        write_json_atomic(
            path,
            {
                "schema_version": 1,
                "presentation_id": deck_contract.get("presentation_id"),
                "style": style,
                "approve_for_tts": False,
                "segments": segments,
                "total_duration_seconds": clock,
            },
        )


__all__ = [
    "ADOPT_EXISTING",
    "RERUN_STEPS",
    "FIGURE_SLIDE_IDS",
    "LINEAGE",
    "STAGE_ID",
    "STEP_OUTPUT_FLAGS",
    "CommandRunner",
    "PresentationChainContext",
    "accepted_review",
    "bind_deck_ledger",
    "bootstrap_delivery_stub",
    "corpus_inventory_source_inputs",
    "default_command_runner",
    "file_sha256",
    "ledger_drift",
    "load_json",
    "narration_content_step",
    "owner_step",
    "run_command",
    "step_gate",
    "step_record",
    "step_spec",
    "supersede_failed_review",
]
