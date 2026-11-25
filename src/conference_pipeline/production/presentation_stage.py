"""Stage 10: the presentation stage shell, shared by every conference runner.

Gates, the per-deck loop over :func:`run_deck_chain`, the cross-deck link
audit, the owner-evidence map and the final strict stage contract. The runner
supplies its identity and conventions through the context; the deck chain
lives in :mod:`conference_pipeline.production.presentation_chain`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.model_governance import default_model
from conference_pipeline.presentation.legacy_markdown_gate import (
    LEGACY_MARKDOWN_STAGE_USED as PRESENTATION_LEGACY_MARKDOWN_STAGE_USED,
)
from conference_pipeline.presentation.legacy_markdown_gate import collect_legacy_markdown_paths
from conference_pipeline.presentation.step_lineage import DeckStepLineage, presentation_output_digest
from conference_pipeline.production.context import UPSTREAM_LINEAGE_NOT_FRESH, ProductionRunnerContext
from conference_pipeline.production.decks import presentation_deck_scope
from conference_pipeline.production.presentation_chain import (
    FIGURE_ROUTE_DRAWSVG,
    PRESENTATION_BLUEPRINT_GENERATION_TASK_ID,
    PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID,
    PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID,
    PRESENTATION_PRESENTER_TTS_TASK_ID,
    PRESENTATION_REPAIR_TASK_ID,
    PRESENTATION_REVIEW_TASK_ID,
    PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID,
    PRESENTATION_VISUAL_QC_TASK_ID,
    run_deck_chain,
)
from conference_pipeline.production.presentation_steps import (
    ADOPT_EXISTING,
    FIGURE_SLIDE_IDS,
    LINEAGE,
    RERUN_STEPS,
    STAGE_ID,
    CommandRunner,
    PresentationChainContext,
    default_command_runner,
    file_sha256,
    load_json,
    owner_step,
)
from conference_pipeline.production.presentation_steps import ledger_drift as ledger_drift_report
from conference_pipeline.production.receipts import StageOutcome, stage_blocker, write_stage_receipt
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.stage_script_registry import (
    load_stage_script_registry,
    registry_sha256,
    required_script_ids_for_stage,
)


def phase13_coverage_exempt_evidence(runner_id: str) -> dict[str, str]:
    """Honest static evidence tokens for the five ``presentation.phase13_*``
    umbrella aliases no automated runner invokes (a separate canary-recovery
    workflow owns them). Each token references the aggregate artifact its own
    ``coverage_exempt_rationale`` names -- never a bare id echo."""

    who = f"the automated {runner_id} presentation stage"
    return {
        "presentation.phase13_bulk_readiness": (
            "umbrella_alias_out_of_scope:13-BULK-READINESS.json,13-BULK-READINESS.md "
            f"(dedicated phase13_bulk_readiness stage binding; canary-recovery workflow only -- not invoked by {who})"
        ),
        "presentation.phase13_canary_review_packet": (
            "umbrella_alias_out_of_scope:decks/<id>/13-CANARY-REVIEW-PACKET.json,decks/<id>/13-CANARY-REVIEW-PACKET.md "
            f"(dedicated phase13_canary_review_packet stage binding; canary-recovery workflow only -- not invoked by {who})"
        ),
        "presentation.phase13_canary_objective_evidence": (
            "umbrella_alias_out_of_scope:decks/<id>/13-CANARY-REVIEW-PACKET.json,decks/<id>/13-CANARY-REVIEW-PACKET.md "
            f"(folded into the phase13_canary_review_packet stage binding; canary-recovery workflow only -- not invoked by {who})"
        ),
        "presentation.phase13_completion": (
            "umbrella_alias_out_of_scope:decks/presentation_manifest.json "
            f"(aggregate artifact per the phase13_completion coverage_exempt_rationale -- not produced by {who})"
        ),
        "presentation.phase13_rollout": (
            "umbrella_alias_out_of_scope:decks/presentation_manifest.json "
            f"(folded into the phase13_completion aggregate per coverage_exempt_rationale -- not produced by {who})"
        ),
    }


def _stage_receipt(
    ctx: ProductionRunnerContext,
    *,
    status: str,
    blockers: list[dict[str, Any]],
    checks: Mapping[str, Any],
    stage_contract: Mapping[str, Any] | None,
) -> StageOutcome:
    """Persist the presentation receipt in the shared shape."""

    owner_evidence = dict(checks.get("owner_execution_evidence") or {}) if isinstance(checks, Mapping) else {}
    return write_stage_receipt(
        ctx,
        stage_id=STAGE_ID,
        status=status,
        blockers=sanitize_for_receipt(blockers),
        checks=sanitize_for_receipt(dict(checks)),
        stage_contract=sanitize_for_receipt(stage_contract) if stage_contract else None,
        receipt_extra={
            "owner_execution_evidence": owner_evidence,
            "provider_calls_made": bool(checks.get("provider_calls_made")) if isinstance(checks, Mapping) else False,
            "downloads_made": 0,
        },
    )


def _blocked_receipt(ctx: ProductionRunnerContext, *, blocker_code: str, checks: dict[str, Any]) -> StageOutcome:
    """A zero-spend blocked presentation receipt."""

    return _stage_receipt(
        ctx,
        status="blocked",
        blockers=[stage_blocker(STAGE_ID, blocker_code)],
        checks={**checks, "blocker_code": blocker_code, "provider_calls_made": False, "live_provider_calls_made": False},
        stage_contract=None,
    )


def required_input_paths(ctx: ProductionRunnerContext) -> dict[str, Path]:
    """Return the workspace-resolved paths Phase 10's presentation owner requires.

    Extends Phase 9's ``podcast_required_input_paths`` shape with two
    additional gates (10-RESEARCH.md architecture diagram): the Stage 9
    (podcast) receipt itself, and the Phase 9 BK-09 backup receipt
    (``backup/<podcast-stage-id>-backup-write-receipt.json``) as a pure
    SEQUENCING gate -- its content is never consumed, only its presence and
    ``status == "succeeded"`` are checked, matching the diagram's explicit
    "content NOT consumed" annotation. Bound to ``STAGES["qa-gap-fill-
    synthesis"]``/``STAGES["podcast"]`` -- CVPR2026 has no generic
    ``receipt_path(workspace, run_id, stage)`` helper; ``run_dir(...) /
    stage.receipt_filename`` is this codebase's own established equivalent.
    """

    stage9 = ctx.stage("podcast")
    target = ctx.run_dir()
    workspace = ctx.workspace
    return {
        "stage8_receipt": ctx.receipt_path("qa-gap-fill-synthesis"),
        "analysis_manifest": workspace / ctx.analysis_manifest,
        "executive_summary": workspace / ctx.analysis_executive_summary,
        "trend_report": workspace / ctx.analysis_trend_report,
        "stage9_receipt": target / stage9.receipt_filename,
        "stage9_backup_receipt": target / "backup" / f"{stage9.id}-backup-write-receipt.json",
    }



def extract_deck_blocker(
    deck: Mapping[str, str], blocked: Mapping[str, Any]
) -> dict[str, Any]:
    """Reduce one deck's ``block_presentation_stage(...)``-shaped outcome
    into a compact, secret-free per-deck blocker record.

    Continue-past-failures policy (operator decision, Plan 10-06): a
    per-deck gate failure (paper matching, any of the 4 early
    Critical-Finding-2 scripts, blueprint generation, figures/QC, etc.) no
    longer halts the entire multi-deck run. The per-deck loop in
    ``run_presentation_stage`` calls this to record what happened for the
    failing deck before continuing to the next one -- never silently
    swallowed; every entry surfaces in the final receipt's
    ``checks["deck_blockers"]``.

    ``blocked`` is already a full stage-receipt-shaped outcome (its own
    ``_presentation_stage_receipt`` call already wrote a transient receipt
    to disk as a side effect, later overwritten by the loop's own final
    receipt) -- only the single blocker code and the check payload that
    produced it are extracted here; both were already sanitized by
    ``_presentation_stage_receipt``.
    """

    receipt = blocked.get("receipt") if isinstance(blocked, Mapping) else None
    blockers = receipt.get("blockers") if isinstance(receipt, Mapping) else None
    deck_checks = receipt.get("checks") if isinstance(receipt, Mapping) else None
    blocker_code: str | None = None
    if isinstance(blockers, list) and blockers:
        first = blockers[0]
        if isinstance(first, Mapping):
            code = first.get("code")
            blocker_code = str(code) if code is not None else None
    if not blocker_code and isinstance(deck_checks, Mapping):
        code = deck_checks.get("blocker_code")
        blocker_code = str(code) if code is not None else None
    return {
        "deck_id": str(deck.get("id") or ""),
        "deck_kind": str(deck.get("kind") or ""),
        "blocker_code": blocker_code,
        "detail": dict(deck_checks) if isinstance(deck_checks, Mapping) else None,
    }



def run_presentation_stage(
    ctx: ProductionRunnerContext,
    *,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    presentation_substage: str = "all",
    deck_ids: Sequence[str] | None = None,
    blueprint_approved_by: str | None = None,
    visual_qc_max_rounds: int | None = None,
    accept_visual_qc_findings: bool = False,
    figure_route: str = FIGURE_ROUTE_DRAWSVG,
    allow_figure_variety_findings: bool = False,
    accepted_review_deck_ids: frozenset[str] = frozenset(),
    adopt_existing_outputs: bool = False,
    rerun_steps: Sequence[str] = (),
    figure_slide_ids: Sequence[str] = (),
    command_runner: CommandRunner | None = None,
) -> StageOutcome:
    """Run the presentation stage; ``rerun_steps`` names deck steps to regenerate.

    ``figure_slide_ids`` re-authors only those slides' figure specs (and
    implies re-running the figure-spec step); the rest of the deck's figures
    are kept.

    A named step re-runs even when its ledger entry is current (a generator
    fix changes no recorded input); everything downstream then follows
    through ordinary lineage. See ``_run_presentation_stage`` for the stage.
    """

    wanted = {str(step).strip() for step in rerun_steps if str(step).strip()}
    slide_ids = tuple(str(sid).strip() for sid in figure_slide_ids if str(sid).strip())
    if slide_ids:
        wanted.add("generate_figure_specs")
    token = RERUN_STEPS.set(frozenset(wanted))
    slides_token = FIGURE_SLIDE_IDS.set(slide_ids)
    try:
        return _run_presentation_stage(
        ctx,
        resume=resume,
        provider_cost_approved=provider_cost_approved,
        env_file=env_file,
        presentation_substage=presentation_substage,
        deck_ids=deck_ids,
        blueprint_approved_by=blueprint_approved_by,
        visual_qc_max_rounds=visual_qc_max_rounds,
        accept_visual_qc_findings=accept_visual_qc_findings,
        figure_route=figure_route,
        allow_figure_variety_findings=allow_figure_variety_findings,
        accepted_review_deck_ids=accepted_review_deck_ids,
        adopt_existing_outputs=adopt_existing_outputs,
        command_runner=command_runner,
        )
    finally:
        FIGURE_SLIDE_IDS.reset(slides_token)
        RERUN_STEPS.reset(token)


def _run_presentation_stage(
    ctx: ProductionRunnerContext,
    *,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    presentation_substage: str = "all",
    deck_ids: Sequence[str] | None = None,
    blueprint_approved_by: str | None = None,
    visual_qc_max_rounds: int | None = None,
    accept_visual_qc_findings: bool = False,
    figure_route: str = FIGURE_ROUTE_DRAWSVG,
    allow_figure_variety_findings: bool = False,
    accepted_review_deck_ids: frozenset[str] = frozenset(),
    adopt_existing_outputs: bool = False,
    command_runner: CommandRunner | None = None,
) -> StageOutcome:
    """Run the presentation stage for every in-scope deck and write its receipt.

    Gates first, at zero spend: a resume short-circuit over a succeeded,
    still-passing receipt whose deck ledgers report no drift; the required
    inputs (Stage 8 receipt, analysis manifest, executive summary, trend
    report, Stage 9 receipt and its backup receipt); the pre-spend lineage
    gate; the legacy-Markdown rejection; provider-cost approval. Then the
    per-deck chain (``run_deck_chain``) under the continue-past-failures
    policy, the cross-deck link audit, and the final strict stage contract.
    ``deck_ids`` restricts the loop to a canary subset that can never
    succeed; ``adopt_existing_outputs`` bootstraps ledgers for decks built
    before the ledger existed.
    """

    stage = ctx.stage(STAGE_ID)
    target = ctx.run_dir()
    target.mkdir(parents=True, exist_ok=True)

    env_file_status = ctx.load_env(env_file)
    approval = ctx.provider_cost_approval(provider_cost_approved)

    # RESUME SHORT-CIRCUIT (10-RESEARCH.md Pattern 1): a repeat ``--resume``
    # invocation over an already-succeeded, still-passing Stage 10 receipt
    # must never re-run any per-deck subprocess or spend again -- unless a
    # file some deck step wrote or read has moved since (an edited
    # blueprint, a patched narration source): then the per-deck chain is
    # walked and each deck's ledger decides which steps actually re-run.
    canonical_receipt_path = target / stage.receipt_filename
    ledger_drift: dict[str, list[str]] = {}
    if resume and canonical_receipt_path.exists():
        existing, _existing_error = load_json(canonical_receipt_path)
        if isinstance(existing, dict) and existing.get("status") == "succeeded":
            resume_contract = ctx.verify_contract(stage_id=STAGE_ID, receipt=existing, bind=False, mode="live", strict_execution=True)
            ledger_drift = ledger_drift_report(ctx, existing)
            if resume_contract.status == "pass" and not ledger_drift:
                try:
                    existing_summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing_summary = {}
                return {
                    "status": "succeeded",
                    "receipt_path": str(canonical_receipt_path),
                    "receipt": existing,
                    "summary": existing_summary,
                }

    checks: dict[str, Any] = {
        "env_file": env_file_status,
        "provider_cost_approval": approval,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "presentation_substage": presentation_substage,
        "figure_route": figure_route,
        "resume_ledger_drift": ledger_drift,
        "secrets_copied": False,
    }

    # Gate: the six required inputs, fail closed with zero spend (PROD-12,
    # done gate 1).
    input_paths = required_input_paths(ctx)
    stage8_receipt, stage8_error = load_json(input_paths["stage8_receipt"])
    analysis_manifest, analysis_manifest_error = load_json(input_paths["analysis_manifest"])
    stage9_receipt, stage9_error = load_json(input_paths["stage9_receipt"])
    stage9_backup_receipt, stage9_backup_error = load_json(input_paths["stage9_backup_receipt"])

    input_blockers: list[dict[str, Any]] = []
    if stage8_error or not isinstance(stage8_receipt, dict):
        input_blockers.append(
            {
                "code": "presentation_stage8_receipt_missing_or_malformed",
                "path": workspace_relpath(
                    ctx.workspace, input_paths["stage8_receipt"]
                ),
                "error": stage8_error,
            }
        )
    elif stage8_receipt.get("status") != "succeeded":
        input_blockers.append(
            {
                "code": "presentation_stage8_receipt_not_succeeded",
                "observed_status": stage8_receipt.get("status"),
            }
        )

    if analysis_manifest_error or not isinstance(analysis_manifest, dict):
        input_blockers.append(
            {
                "code": "presentation_analysis_manifest_missing_or_malformed",
                "path": workspace_relpath(
                    ctx.workspace, input_paths["analysis_manifest"]
                ),
                "error": analysis_manifest_error,
            }
        )
    for label in ("executive_summary", "trend_report"):
        if not input_paths[label].is_file():
            input_blockers.append(
                {
                    "code": f"presentation_{label}_missing",
                    "path": workspace_relpath(ctx.workspace, input_paths[label]),
                }
            )

    if stage9_error or not isinstance(stage9_receipt, dict):
        input_blockers.append(
            {
                "code": "presentation_stage9_receipt_missing_or_malformed",
                "path": workspace_relpath(
                    ctx.workspace, input_paths["stage9_receipt"]
                ),
                "error": stage9_error,
            }
        )
    elif stage9_receipt.get("status") != "succeeded":
        input_blockers.append(
            {
                "code": "presentation_stage9_receipt_not_succeeded",
                "observed_status": stage9_receipt.get("status"),
            }
        )

    # SEQUENCING gate only (10-RESEARCH.md architecture diagram): the
    # backup receipt's own content is never consumed downstream, only its
    # presence and ``status == "succeeded"`` are checked.
    if stage9_backup_error or not isinstance(stage9_backup_receipt, dict):
        input_blockers.append(
            {
                "code": "presentation_backup_receipt_missing",
                "path": workspace_relpath(
                    ctx.workspace, input_paths["stage9_backup_receipt"]
                ),
                "error": stage9_backup_error,
            }
        )
    elif stage9_backup_receipt.get("status") != "succeeded":
        input_blockers.append(
            {
                "code": "presentation_backup_receipt_not_succeeded",
                "observed_status": stage9_backup_receipt.get("status"),
            }
        )

    checks.update(
        {
            "stage8_receipt_path": workspace_relpath(
                ctx.workspace, input_paths["stage8_receipt"]
            ),
            "analysis_manifest_path": str(ctx.analysis_manifest),
            "executive_summary_path": str(ctx.analysis_executive_summary),
            "trend_report_path": str(ctx.analysis_trend_report),
            "stage9_receipt_path": workspace_relpath(
                ctx.workspace, input_paths["stage9_receipt"]
            ),
            "stage9_backup_receipt_path": workspace_relpath(
                ctx.workspace, input_paths["stage9_backup_receipt"]
            ),
        }
    )
    if input_blockers:
        return _blocked_receipt(
            ctx,
            blocker_code="presentation_required_inputs_blocked",
            checks={**checks, "input_blockers": input_blockers},
        )
    assert stage8_receipt is not None
    assert analysis_manifest is not None
    assert stage9_receipt is not None
    assert stage9_backup_receipt is not None
    lineage_checks = ctx.upstream_lineage_checks(STAGE_ID)
    if lineage_checks is not None:
        return _blocked_receipt(
            ctx,
            blocker_code=UPSTREAM_LINEAGE_NOT_FRESH,
            checks={**checks, **lineage_checks},
        )

    # Gate: legacy-Markdown rejection (ROADMAP done gate 1) -- the structured
    # artifact pipeline is this stage's only production evidence path.
    legacy_markdown_findings = collect_legacy_markdown_paths(ctx.workspace / ctx.decks_dir)
    if legacy_markdown_findings:
        return _blocked_receipt(
            ctx,
            blocker_code=PRESENTATION_LEGACY_MARKDOWN_STAGE_USED,
            checks={
                **checks,
                "legacy_markdown_paths": [
                    workspace_relpath(ctx.workspace, path)
                    for path in legacy_markdown_findings
                ],
            },
        )

    # Gate: provider-cost approval -- halts (never auto-raises the cap)
    # before any subprocess is ever constructed.
    if not bool(approval["approved"]):
        return _blocked_receipt(
            ctx,
            blocker_code="presentation_provider_cost_not_approved",
            checks=checks,
        )

    # Deck scope: 12 topic decks (from analysis_manifest.json) + 2
    # executive-overview decks -- LOCKED full-coverage decision, no canary
    # subset (10-CONTEXT.md / 10-RESEARCH.md Open Question 1).
    try:
        deck_scope = presentation_deck_scope(analysis_manifest, ctx.executive_decks)
    except RuntimeError as exc:
        return _blocked_receipt(
            ctx,
            blocker_code="presentation_deck_scope_failed",
            checks={**checks, "error": sanitize_for_receipt(str(exc))},
        )
    deck_scope_filter = [str(deck_id) for deck_id in (deck_ids or []) if str(deck_id).strip()]
    if deck_scope_filter:
        known_ids = {deck["id"] for deck in deck_scope}
        unknown_ids = sorted(set(deck_scope_filter) - known_ids)
        if unknown_ids:
            return _blocked_receipt(
                ctx,
                blocker_code="presentation_deck_scope_filter_unknown_deck",
                checks={
                    **checks,
                    "deck_scope_filter": deck_scope_filter,
                    "unknown_deck_ids": unknown_ids,
                    "known_deck_ids": sorted(known_ids),
                },
            )
        checks["deck_scope_full_count"] = len(deck_scope)
        checks["deck_scope_filter"] = deck_scope_filter
        deck_scope = [deck for deck in deck_scope if deck["id"] in set(deck_scope_filter)]
    checks["deck_scope"] = [dict(deck) for deck in deck_scope]
    checks["deck_count"] = len(deck_scope)

    runner = command_runner or default_command_runner(ctx.repo_root)
    pc = PresentationChainContext(ctx=ctx, command_runner=runner, checks=checks)
    owner_evidence_candidates: dict[str, str] = {}
    deck_records: list[dict[str, Any]] = []

    # Sub-stage A (deck/figures) part 1+2 (Plans 10-02/10-03): per-deck paper
    # matching + the 4 early Critical-Finding-2 scripts + blueprint/review +
    # deck contract + source binding + figures/render/export/screenshots/
    # visual QC/bounded repair/artifact-truth (the 5th, LATE, CF2 script).
    # Sub-stage B (delivery/video, Plan 10-04) -- delivery-contract
    # generation + both-mode narration lint/review/fix (pre-TTS) + both
    # videos + video-sync validation + the 3-model presentation review --
    # is appended to the SAME per-deck chain function, gated internally by
    # ``run_delivery_video``. Gated to {deck_figures, delivery_video, all}:
    # for "deck_figures", ``run_delivery_video=False`` so sub-stage B is
    # never even attempted; for "delivery_video", sub-stage A's own steps
    # still execute but the idempotent per-step skip (their outputs already
    # exist from a prior "deck_figures" run) makes them no-ops, and only
    # sub-stage B genuinely runs.
    deck_blockers: list[dict[str, Any]] = []
    step_lineage: dict[str, dict[str, Any]] = {}
    if presentation_substage in ("deck_figures", "delivery_video", "all"):
        run_delivery_video = presentation_substage in ("delivery_video", "all")
        for deck in deck_scope:
            blocked = run_deck_chain(
                pc,
                deck=deck,
                resume=resume,
                checks=checks,
                owner_evidence_candidates=owner_evidence_candidates,
                deck_records=deck_records,
                run_delivery_video=run_delivery_video,
                figure_route=figure_route,
                allow_figure_variety_findings=allow_figure_variety_findings,
                accepted_review_deck_ids=accepted_review_deck_ids,
                blueprint_approved_by=blueprint_approved_by,
                visual_qc_max_rounds=visual_qc_max_rounds,
                accept_visual_qc_findings=accept_visual_qc_findings,
                adopt_existing_outputs=adopt_existing_outputs,
            )
            step_lineage[str(deck["id"])] = DeckStepLineage.for_deck_dir(
                ctx.deck_dir(str(deck["id"])),
                workspace=ctx.workspace,
                repo_root=ctx.repo_root,
            ).report_entry()
            if blocked is not None:
                # Continue-past-failures policy (operator decision, Plan
                # 10-06): a per-deck gate failure -- content-sufficiency
                # (e.g. blueprint generation's MIN_SOURCES), paper matching,
                # figures/QC, or any other step in the chain -- is recorded
                # and the loop proceeds to the remaining decks, rather than
                # halting the entire multi-deck run on the first blocker.
                # Never silently swallowed: every entry is visible in the
                # final receipt's checks["deck_blockers"]. Decks that
                # already succeeded (this one or any prior) keep their real,
                # un-invalidated artifacts on disk -- a subsequent
                # ``--resume`` run's idempotent per-step skip only
                # re-attempts a deck recorded here.
                deck_blockers.append(extract_deck_blocker(deck, blocked))
                continue

    checks["deck_blockers"] = deck_blockers
    # Per-deck step ledgers: what each deck's steps were built from, and one
    # digest per deck over the current heads of every file a step wrote.
    # The stage receipt's own lineage lifts ``step_lineage_output_digest``
    # into its output digest, so a re-run of any per-deck step changes the
    # stage's output and invalidates Stages 11-12.
    checks["step_lineage"] = step_lineage
    checks["step_lineage_output_digest"] = presentation_output_digest(
        {
            deck_id: DeckStepLineage.for_deck_dir(
                ctx.deck_dir(deck_id),
                workspace=ctx.workspace,
                repo_root=ctx.repo_root,
            )
            for deck_id in step_lineage
        }
    )

    # Cross-deck link audit (Plan 10-03 Task 3): runs ONCE, after the full
    # per-deck deck/figures pass across ALL in-scope decks -- never per-deck
    # (unlike every step above). $0, deterministic; not wrapped in
    # provider_receipt_context. Its ledger is the decks directory's own; its
    # inputs are every deck's rendered html.
    link_audit_receipt_output = ctx.workspace / ctx.decks_dir / "presentation_link_audit_receipt.json"
    if presentation_substage in ("deck_figures", "all"):
        link_audit_command = [
            sys.executable,
            "scripts/presentations/audit_presentation_links.py",
            "--workspace",
            str(ctx.workspace),
            "--decks-dir",
            str(ctx.decks_dir),
            "--receipt-output",
            str(link_audit_receipt_output),
        ]
        decks_dir = ctx.workspace / ctx.decks_dir
        link_audit_token = LINEAGE.set(
            DeckStepLineage.for_deck_dir(decks_dir, workspace=ctx.workspace, repo_root=ctx.repo_root)
        )
        link_audit_adopt_token = ADOPT_EXISTING.set(bool(adopt_existing_outputs))
        try:
            blocked = owner_step(
                pc,
                resume=resume,
                command=link_audit_command,
                paid=False,
                outputs=(link_audit_receipt_output,),
                failed_blocker_code="presentation_link_audit_failed",
                missing_blocker_code="presentation_link_audit_outputs_missing",
                extra_checks=checks,
                directory_patterns={decks_dir: "*/presentation.html"},
            )
        finally:
            ADOPT_EXISTING.reset(link_audit_adopt_token)
            LINEAGE.reset(link_audit_token)
        if blocked is not None:
            return blocked
        owner_evidence_candidates["presentation.link_audit"] = (
            f"{link_audit_receipt_output}:{file_sha256(link_audit_receipt_output)}"
        )

    # Plan 10-03 fully wires sub-stage A (deck/figures): figures/render/
    # export/screenshots/visual QC/bounded repair/artifact-truth/link-audit.
    # ``presentation_substage == "deck_figures"`` halts CLEANLY here with an
    # honest interim receipt -- genuinely reflecting deck/figures completion
    # (real provider spend occurred: blueprint, figures, vision QC) -- and
    # never invokes any delivery/video step (Plan 10-04's own substage).
    # ``verify_stage_contract``'s own required-outputs check (delivery
    # contracts, video-sync reports, presenter script, produced review --
    # none of which this substage produces) still fails a later "all"-scope
    # resume attempt from short-circuiting off of this interim receipt. This
    # early-return branch is evaluated BEFORE the final owner-evidence map is
    # assembled below -- deck/figures-only ``model_governance_task_ids``
    # entries are still recorded on this interim receipt.
    if deck_scope_filter:
        checks["deck_records"] = deck_records
        checks["owner_execution_evidence"] = dict(owner_evidence_candidates)
        checks["deck_blocker_count"] = len(deck_blockers)
        checks["deck_success_count"] = len(deck_scope) - len(deck_blockers)
        return _stage_receipt(
            ctx,
            status="blocked",
            blockers=[
                stage_blocker(STAGE_ID, 
                    "presentation_deck_scope_filtered",
                    deck_ids=deck_scope_filter,
                    deck_blocker_count=len(deck_blockers),
                    blocked_deck_ids=[item["deck_id"] for item in deck_blockers],
                    full_scope_deck_count=checks["deck_scope_full_count"],
                )
            ],
            checks={**checks, "provider_calls_made": bool(owner_evidence_candidates)},
            stage_contract=None,
        )

    if presentation_substage == "deck_figures":
        deck_figures_model_governance_task_ids: list[str] = []
        if "presentation.generate_editorial_blueprint" in owner_evidence_candidates:
            deck_figures_model_governance_task_ids.append(PRESENTATION_BLUEPRINT_GENERATION_TASK_ID)
        if "presentation.generate_programmatic_figures" in owner_evidence_candidates:
            assert default_model(PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID)
            deck_figures_model_governance_task_ids.append(PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID)
        if "presentation.inspect_rendered_slides_llm" in owner_evidence_candidates:
            assert default_model(PRESENTATION_VISUAL_QC_TASK_ID)
            deck_figures_model_governance_task_ids.append(PRESENTATION_VISUAL_QC_TASK_ID)
        if "presentation.repair_loop" in owner_evidence_candidates and not str(
            owner_evidence_candidates["presentation.repair_loop"]
        ).startswith("repair_not_required:"):
            assert default_model(PRESENTATION_REPAIR_TASK_ID)
            deck_figures_model_governance_task_ids.append(PRESENTATION_REPAIR_TASK_ID)
        checks["deck_records"] = deck_records
        checks["owner_execution_evidence"] = dict(owner_evidence_candidates)
        checks["deck_blocker_count"] = len(deck_blockers)
        checks["deck_success_count"] = len(deck_scope) - len(deck_blockers)
        if deck_figures_model_governance_task_ids:
            checks["model_governance_task_ids"] = deck_figures_model_governance_task_ids
        if deck_blockers:
            # Honest partial-completion outcome (continue-past-failures
            # policy, operator decision, Plan 10-06): one or more decks hit
            # a per-deck blocker. Never fabricate "succeeded" for a scope
            # that isn't fully real -- but the decks that DID succeed have
            # genuine, un-invalidated artifacts already on disk (real
            # provider spend occurred for them), so ``provider_calls_made``
            # reflects that real work, unlike a zero-spend
            # ``block_presentation_stage`` gate block. A subsequent
            # ``--resume`` run's idempotent per-step skip re-attempts only
            # the deck(s) recorded in ``checks["deck_blockers"]``.
            return _stage_receipt(
                ctx,
                status="blocked",
                blockers=[
                    stage_blocker(STAGE_ID, 
                        "presentation_partial_deck_scope_blocked",
                        deck_blocker_count=len(deck_blockers),
                        deck_ids=[item["deck_id"] for item in deck_blockers],
                    )
                ],
                checks={
                    **checks,
                    "presentation_deck_figures_substage_complete": False,
                    "provider_calls_made": bool(owner_evidence_candidates),
                },
                stage_contract=None,
            )
        return _stage_receipt(
            ctx,
            status="succeeded",
            blockers=[],
            checks={
                **checks,
                "presentation_deck_figures_substage_complete": True,
                "provider_calls_made": bool(owner_evidence_candidates),
            },
            stage_contract=None,
        )

    # ------------------------------------------------------------------
    # Plan 10-04 Task 3: assemble the complete 35-entry owner-evidence map
    # (reached only for presentation_substage in {"delivery_video", "all"} --
    # "deck_figures" already returned its own honest interim receipt above)
    # and run the final verify_stage_contract(strict_execution=True) check.
    # ------------------------------------------------------------------

    # The 2 executive-overview decks are driven through the SAME per-deck
    # chain as the 12 topic decks (matching the ICML analog, per this plan's
    # own "either X or Y" option) rather than the separate, legacy, fully
    # self-contained generate_executive_overview_decks.py generator -- that
    # script duplicates blueprint/figure/deck-contract generation with its
    # own (pre-render_spec-era) machinery and would race/overwrite this
    # stage's already-established structured per-deck artifacts for the same
    # 2 deck ids. ``presentation.executive_overview_decks`` (a required,
    # coverage-exempt id) is instead evidenced by referencing the real,
    # just-produced ``presentation_deck.json`` for each executive deck --
    # genuine artifacts, not a fabricated placeholder.
    executive_deck_records = [
        record for record in deck_records if record.get("deck_kind") == "executive"
    ]
    if executive_deck_records:
        owner_evidence_candidates["presentation.executive_overview_decks"] = "; ".join(
            f"{record['presentation_deck_path']}:{file_sha256(Path(record['presentation_deck_path']))}"
            for record in executive_deck_records
        )
    # The 5 ``presentation.phase13_*`` umbrella-alias ids are never invoked
    # by the automated CVPR2026 runner (a separate ICLR2026 canary-recovery
    # workflow owns them) -- each gets a static, honest, non-fabricated
    # evidence token referencing its named aggregate artifact (see the
    # constant's own docstring above).
    owner_evidence_candidates.update(phase13_coverage_exempt_evidence(ctx.runner_id))

    # A per-deck blocker in the FULL substage must block the stage exactly as
    # it does in the deck_figures substage. Until 2026-09-16 only that branch
    # checked; a full run with two decks failing Gate A returned "succeeded"
    # with the failures listed under checks["deck_blockers"], which would have
    # let Stage 11 inventory decks the reviewers had refused.
    if deck_blockers:
        checks["deck_records"] = deck_records
        checks["owner_execution_evidence"] = dict(owner_evidence_candidates)
        checks["deck_blocker_count"] = len(deck_blockers)
        checks["deck_success_count"] = len(deck_scope) - len(deck_blockers)
        return _stage_receipt(
            ctx,
            status="blocked",
            blockers=[
                stage_blocker(STAGE_ID, 
                    "presentation_partial_deck_scope_blocked",
                    deck_blocker_count=len(deck_blockers),
                    deck_ids=[item["deck_id"] for item in deck_blockers],
                )
            ],
            checks={**checks, "provider_calls_made": bool(owner_evidence_candidates)},
            stage_contract=None,
        )

    checks["deck_records"] = deck_records
    checks["owner_execution_evidence"] = dict(owner_evidence_candidates)
    # Only the resolving ids this plan actually invokes are recorded --
    # never the non-resolving workflow-stage ids ICML2026's own receipt
    # mislabels as model-governance task ids (RESEARCH Pitfall 9), and only
    # once each step has genuinely run.
    model_governance_task_ids: list[str] = []
    if "presentation.generate_editorial_blueprint" in owner_evidence_candidates:
        model_governance_task_ids.append(PRESENTATION_BLUEPRINT_GENERATION_TASK_ID)
    if "presentation.generate_programmatic_figures" in owner_evidence_candidates:
        assert default_model(PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID)
    if "presentation.inspect_rendered_slides_llm" in owner_evidence_candidates:
        assert default_model(PRESENTATION_VISUAL_QC_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_VISUAL_QC_TASK_ID)
    if "presentation.repair_loop" in owner_evidence_candidates and not str(
        owner_evidence_candidates["presentation.repair_loop"]
    ).startswith("repair_not_required:"):
        assert default_model(PRESENTATION_REPAIR_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_REPAIR_TASK_ID)
    if "presentation.presenter_script" in owner_evidence_candidates:
        assert default_model(PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID)
    if (
        "presentation.build_presenter_video" in owner_evidence_candidates
        or "presentation.build_notebooklm_video" in owner_evidence_candidates
    ):
        assert default_model(PRESENTATION_PRESENTER_TTS_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_PRESENTER_TTS_TASK_ID)
    if "presentation.build_notebooklm_video" in owner_evidence_candidates:
        assert default_model(PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID)
    if "presentation.presentation_review" in owner_evidence_candidates:
        assert default_model(PRESENTATION_REVIEW_TASK_ID)
        model_governance_task_ids.append(PRESENTATION_REVIEW_TASK_ID)
    if model_governance_task_ids:
        checks["model_governance_task_ids"] = model_governance_task_ids

    required_script_ids = list(required_script_ids_for_stage(load_stage_script_registry(), STAGE_ID))
    provisional_receipt = {
        "mode": "live",
        "script_registry_sha256": registry_sha256(),
        "required_script_ids": required_script_ids,
        "owner_execution_evidence": dict(owner_evidence_candidates),
        "checks": checks,
    }
    stage_contract = ctx.verify_contract(stage_id=STAGE_ID, receipt=provisional_receipt, bind=True, mode="live", strict_execution=True).to_dict()

    if stage_contract["status"] != "pass":
        return _blocked_receipt(
            ctx,
            blocker_code="presentation_final_stage_contract_failed",
            checks={**checks, "stage_contract": stage_contract},
        )

    return _stage_receipt(
        ctx,
        status="succeeded",
        blockers=[],
        checks={**checks, "provider_calls_made": bool(owner_evidence_candidates)},
        stage_contract=stage_contract,
    )


