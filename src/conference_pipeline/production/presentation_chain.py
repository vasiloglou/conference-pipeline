"""The per-deck presentation chain, shared by every conference runner.

One deck goes: paper matching (paid, in-process) -> source corpus inventory ->
speaker/author/work ledger -> source repair -> research cluster plan ->
editorial blueprint (paid) + review -> deck contract -> source pack + source
binding -> figures (compiler or drawsvg route) -> SVG validation -> render
spec -> bookend slides -> HTML / PDF / PPTX -> screenshot inspection ->
vision judge + bounded repair -> artifact truth; then, for the delivery
substage: presenter script -> engaging narration -> narration lint/review/fix
(three lanes) -> Gate A whole-presentation review -> both videos -> video
sync validation. Executive decks are verified against the digest chain's
artifacts rather than rebuilt. Every step goes through the ledger-aware
primitives in :mod:`conference_pipeline.production.presentation_steps`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import conference_pipeline.paper_matching as presentation_paper_matching
import conference_pipeline.presentation.commands as presentation_commands
from conference_pipeline.model_governance import model_for_alias
from conference_pipeline.presentation.step_lineage import StepInput, StepSpec
from conference_pipeline.production.presentation_steps import (
    FIGURE_SLIDE_IDS,
    LINEAGE,
    RERUN_STEPS,
    PresentationChainContext,
    accepted_review,
    bind_deck_ledger,
    bootstrap_delivery_stub,
    corpus_inventory_source_inputs,
    file_sha256,
    load_json,
    narration_content_step,
    owner_step,
    step_gate,
    step_record,
    step_spec,
    supersede_failed_review,
)
from conference_pipeline.presentation.advisory_qc import (
    ADVISORY_REPORT,
    ADVISORY_SUMMARY,
    advisory_command,
    write_advisory_summary,
)
from conference_pipeline.presentation.figure_findings import undecided_figure_findings
from conference_pipeline.presentation.review_settings import advisory_visual_qc_selection
from conference_pipeline.provider_receipts import provider_receipt_context, sanitize_for_receipt, utc_now
from conference_pipeline.workflow_runs import write_json_atomic

PAPER_MATCHING_DIR = Path("storage") / "paper-matching"
TOPIC_PRESENTATION_TYPE = "topic-deep-dive"
EXECUTIVE_PRESENTATION_TYPE = "executive-summary"

# Model-governance task ids the chain records (only the resolving ids it
# actually invokes; workflow-stage ids are never model-governance task ids).
PRESENTATION_DECK_GENERATION_TASK_ID = "presentation.deck_generation"
PRESENTATION_REPAIR_TASK_ID = "presentation.repair"
PRESENTATION_SVG_FIGURE_GENERATION_TASK_ID = "presentation.svg_figure_generation"
PRESENTATION_VISUAL_QC_TASK_ID = "presentation.visual_qc"
PRESENTATION_FIGURE_CAPTION_TASK_ID = "vision.figure_caption"
PRESENTATION_PAPER_MATCHING_TASK_ID = "research.paper_matching_keypoints"
PRESENTATION_BLUEPRINT_GENERATION_TASK_ID = "presentation.blueprint_generation"
PRESENTATION_PRESENTER_SCRIPT_GENERATION_TASK_ID = "presentation.presenter_script_generation"
PRESENTATION_REVIEW_TASK_ID = "presentation.presentation_review"
PRESENTATION_PRESENTER_TTS_TASK_ID = "presentation.presenter_tts"
PRESENTATION_PRESENTER_DIALOGUE_GENERATION_TASK_ID = "presentation.presenter_dialogue_generation"

# Two ways to get a figure onto a slide. ``drawsvg``: a model writes drawing
# code and a bounded redraw loop chases the pixel judge's findings.
# ``compiler``: the model authors a declarative spec and
# conference_pipeline.figures lays it out from measured text, so the judge
# becomes a recorded regression check rather than a gate a retry can chase.
FIGURE_ROUTE_DRAWSVG = "drawsvg"
FIGURE_ROUTE_COMPILER = "compiler"
FIGURE_ROUTES = (FIGURE_ROUTE_DRAWSVG, FIGURE_ROUTE_COMPILER)

# ``generate_programmatic_figures.py``'s Phase 16 codegen route pins its own
# required model (the opus alias); the registry task id is still recorded.
PHASE16_PROVIDER = "anthropic"
PHASE16_MODEL = model_for_alias("opus")
PHASE16_ALLOWED_RENDERER = "drawsvg"
PHASE16_DISALLOWED_RENDERERS: tuple[str, ...] = ("matplotlib", "plotly")
SVG_STACK_RESEARCH = (
    ".planning/workstreams/v2-1/phases/16-brand-locked-editorial-topic-4-canary-rebuild/"
    "16-SVG-STACK-RESEARCH.md"
)
REPAIR_MAX_ATTEMPTS = 2
FIGURE_CODEGEN_MAX_ATTEMPTS = 5
FIGURE_PASS_RETRIES = 2


def run_advisory_visual_qc(
    pc: Any,
    *,
    workspace: Path,
    run_id: str | None,
    deck_id: str,
    presentation_type: str,
    deck_dir: Path,
    screenshots_dir: Path,
    primary_report: Path,
) -> dict[str, Any] | None:
    """A second vision model's opinion on the rendered slides. Advisory: it never blocks.

    Configured per conference (``presentation_review.advisory_visual_qc``); with nothing
    configured this returns ``None`` and runs nothing. The second model reports more than
    the primary judge and over-states severity, so what it produces is a summary handed to
    the Gate A reviewers as hints -- not evidence for the stage, not a gate. A misconfigured
    section, a missing key or a failed call is recorded in the returned summary and the
    chain carries on. Slides whose screenshot is unchanged are not re-judged (the vision
    pass reuses a prior verdict for the same provider, model and image), so a resume is free.
    """

    try:
        selection = advisory_visual_qc_selection(workspace)
    except Exception as exc:  # noqa: BLE001 - advice must never stop the chain
        return {"status": "misconfigured", "error": str(exc)[:300], "advisory": True}
    if selection is None:
        return None
    provider, model = selection
    report_path = deck_dir / ADVISORY_REPORT
    try:
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=pc.stage_id,
            caller=pc.ctx.receipt_caller,
        ):
            pc.command_runner(
                advisory_command(
                    workspace=workspace,
                    deck_id=deck_id,
                    presentation_type=presentation_type,
                    screenshots_dir=screenshots_dir,
                    report_output=report_path,
                    provider=provider,
                    model=model,
                )
            )
    except Exception as exc:  # noqa: BLE001 - a failed advisory call is recorded, not raised
        summary = write_advisory_summary(report_path, deck_dir / ADVISORY_SUMMARY, primary_report_path=primary_report)
        return {**summary, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return write_advisory_summary(report_path, deck_dir / ADVISORY_SUMMARY, primary_report_path=primary_report)


def write_figure_regression(
    *,
    visual_qc_report: Mapping[str, Any],
    render_spec_path: Path,
    deck_dir: Path,
    output_path: Path,
    sample_size: int | None = None,
) -> int:
    """Record the pixel judge's findings against the ARCHETYPE that drew them.

    This is the compiler route's answer to the redraw loop. A compiled figure
    cannot be improved by looking at it again -- the spec determines the pixels
    -- so the judge's value is longitudinal: which archetypes keep drawing
    findings, across decks and over time. That is a compiler backlog, not a
    per-figure retry, and nothing here re-renders or re-authors anything.

    Returns the number of findings recorded.
    """

    from conference_pipeline.figures.regression import archetype_regression, load_manifests

    render_spec, _error = load_json(render_spec_path)
    manifests = load_manifests(deck_dir / "assets" / "compiled")
    if not manifests:
        return 0
    payload = archetype_regression(
        visual_qc_report=visual_qc_report,
        render_spec=render_spec if isinstance(render_spec, dict) else {},
        manifests=manifests,
        sample_size=sample_size,
        seed=str(deck_dir.name),
    )
    write_json_atomic(output_path, payload)
    return int(payload.get("finding_total") or 0)


def write_qc_figure_feedback(
    *,
    visual_qc_report: Mapping[str, Any],
    render_spec_path: Path,
    output_path: Path,
) -> int:
    """Turn vision-QC blocker findings into per-slide redraw feedback for
    ``generate_programmatic_figures.py --qc-feedback-json`` (the pixel judge
    closes the loop the geometric SVG estimators cannot). Mirrors
    ``run_topic_presentation_build.py``'s ``write_qc_figure_feedback``.
    Bookend/template slides and slide-title defects (deck-contract problems,
    not figure problems) are excluded. Returns the number of slides flagged."""

    render_spec, _render_spec_error = load_json(render_spec_path)
    slides = (
        render_spec.get("slides")
        if isinstance(render_spec, dict) and isinstance(render_spec.get("slides"), list)
        else []
    )
    feedback: dict[str, str] = {}
    for entry in visual_qc_report.get("slides") or []:
        if not isinstance(entry, dict) or entry.get("status") != "fail":
            continue
        try:
            number = int(entry.get("slide_number") or 0)
        except (TypeError, ValueError):
            continue
        if not (1 <= number <= len(slides)):
            continue
        slide = slides[number - 1]
        slide_id = str(slide.get("slide_id") or "") if isinstance(slide, dict) else ""
        if not slide_id.startswith("sl_"):
            continue
        issues = "; ".join(
            f"{issue.get('check_id')}: {issue.get('description')}"
            for issue in entry.get("issues") or []
            if isinstance(issue, dict)
            and issue.get("severity") in ("blocker", "warn")
            and "slide title" not in str(issue.get("description") or "").lower()
        )
        if issues:
            feedback[slide_id] = (
                "A pixel-level visual review of the RENDERED slide found these defects "
                f"in your previous figure: {issues[:700]}. Redraw with these fixed -- "
                "more spacing, no text under arrows/lines, every label inside its shape. "
                "Layout rules for the redraw: keep every diagram node, connector and "
                "annotation inside the upper 72% of the canvas; put the work/attribution "
                "boxes in ONE row along the bottom with a clear gap above them so they "
                "never touch or cover a diagram node; route connectors around labels, "
                "never through them; wrap long labels onto two lines or widen the box so "
                "no text crosses a border; and drop any callout that cannot fit without "
                "overlapping."
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(feedback)


def resolve_report_url(
    *,
    workspace: Path,
    deck_contract_output: Path,
    render_spec_output: Path,
) -> dict[str, Any]:
    """Replace the bare ``https://drive.google.com/`` placeholder that the
    blueprint generator falls back to (no report URL in config.yaml at
    blueprint time) with the workspace's canonical ``urls.final_report_url``
    in the deck contract and render_spec, so the bookend "resources" slide
    links the real report. A no-op when the contract already carries a real
    link or the workspace has none; every rewrite is reported back."""

    canonical = presentation_commands.presentation_final_report_url(workspace)
    resolution: dict[str, Any] = {"canonical": canonical, "rewritten": []}
    if canonical.rstrip("/") == "https://drive.google.com":
        return resolution
    for path in (deck_contract_output, render_spec_output):
        payload, error = load_json(path)
        if error or not isinstance(payload, dict):
            continue
        current = str(payload.get("analysis_report_url") or "").strip()
        if current and current.rstrip("/") != "https://drive.google.com":
            continue
        payload["analysis_report_url"] = canonical
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resolution["rewritten"].append(str(path))
    return resolution


def merge_semantic_figure_feedback(
    *,
    semantic_review: Mapping[str, Any],
    output_path: Path,
) -> int:
    """Append the deterministic semantic reviewer's per-figure findings (for
    example a required entity label missing from a redrawn SVG) to the
    redraw feedback file so the next figure pass restores them. Returns the
    number of slides whose feedback was extended."""

    existing, _error = load_json(output_path)
    feedback: dict[str, str] = (
        {str(key): str(value) for key, value in existing.items()} if isinstance(existing, dict) else {}
    )
    touched: set[str] = set()
    for finding in semantic_review.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        slide_id = str(finding.get("slide_id") or "")
        message = str(finding.get("message") or "").strip()
        if not slide_id.startswith("sl_") or not message:
            continue
        note = (
            f" The deterministic semantic review of your redraw found: {message}. "
            "Every required entity label must appear VERBATIM as visible text in the figure."
        )
        feedback[slide_id] = feedback.get(slide_id, "") + note
        touched.add(slide_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(touched)


# Executive decks are authored by the digest chain
# (``scripts/presentations/build_executive_summary.py`` ->
# ``build_executive_videos.py``), not by the per-deck chain below. These are
# the artifacts that chain produces, and what this stage verifies for an
# executive deck instead of rebuilding it.
EXECUTIVE_DIGEST_SUBSTAGE_A_ARTIFACTS: tuple[str, ...] = (
    "presentation_blueprint.json",
    "blueprint_review.json",
    "presentation_deck.json",
    "source_binding_report.json",
    "visual_plan.json",
    "render_spec.json",
    "presentation_slide_mapping.json",
    "presentation.html",
    "presentation.pdf",
    "figure_svg_validation_report.json",
    "slide_frame_manifest.json",
)
EXECUTIVE_DIGEST_SUBSTAGE_B_ARTIFACTS: tuple[str, ...] = (
    "delivery_single_speaker.json",
    "delivery_notebooklm.json",
    "presentation_video.mp4",
    "presentation_video_notebooklm.mp4",
    "video_sync_validation_single_speaker.json",
    "video_sync_validation_notebooklm.json",
)


def executive_digest_chain(
    pc: PresentationChainContext,
    *,
    deck_id: str,
    kind: str,
    deck_dir: Path,
    deck_records: list[dict[str, Any]],
    owner_evidence_candidates: dict[str, str],
    run_delivery_video: bool,
    common_checks: Mapping[str, Any],
    presentation_type: str,
    resume: bool,
    accept_visual_qc_findings: bool = False,
) -> dict[str, Any] | None:
    """Verify and record an executive deck built by the digest chain.

    The 2 executive decks are NOT produced by the per-deck chain. That
    routing was tried and cannot work: this stage feeds an executive deck
    ``analysis/executive_summary.md`` and ``analysis/trend_report.md``, while
    ``build_speaker_author_work_ledger`` harvests entries only from a
    ``topic_reference_ledgers`` payload or a ``__packets.json``. The ledger
    came back empty, the research-cluster source pool draws solely from
    ledger entries, and the chain raised "research_cluster_plan requires at
    least one eligible source candidate" five steps before the blueprint.
    Feeding it the twelve topic reference ledgers gets a blueprint from 905
    entries, but every paper entry is then ``missing_contribution`` and so
    chip-ineligible, the mention plan is all transcripts, ``source_paper_plan``
    is empty and the deterministic source-coverage gate fails. The input
    contract cannot fill the ledger it exists to fill.

    The digest chain is the executive path, for every conference: each topic's
    presenter script and blueprint go through Opus topic digests
    (``generate_executive_topic_digests.py``), then
    ``generate_executive_overview_decks.py`` builds the deck and its diagrams,
    then ``build_executive_videos.py`` renders both videos. It produced ICLR
    2026's and ICML 2026's executive decks, and CVPR 2026's.

    So this stage verifies those artifacts rather than rebuilding them --
    fail-closed on a missing one, naming the command that produces it. Never a
    placeholder: every recorded path is a real file, hashed like any other.

    Verification includes the SAME render gates a topic deck gets -- the
    deterministic screenshot inspection and the vision-LLM judge. Until
    2026-09-15 it did not, and "the files exist" was the entire executive
    acceptance test: CVPR 2026 shipped both executive decks with a slide whose
    text ran 956px past the canvas and whose footer was painted over the topic
    list, while the same judge was flagging that exact defect class on seven of
    the twelve topic decks. A deck nobody looked at is not a deck that passed.
    """

    workspace = pc.ctx.workspace
    run_id = pc.ctx.run_id

    missing = [
        name
        for name in EXECUTIVE_DIGEST_SUBSTAGE_A_ARTIFACTS
        if not (deck_dir / name).is_file()
    ]
    build_command = (
        ".venv/bin/python scripts/presentations/build_executive_summary.py "
        f"--workspace {workspace} --scope both --regenerate-digests"
    )
    if missing:
        return pc.block(
            blocker_code="presentation_executive_digest_artifacts_missing",
            checks={
                **common_checks,
                "missing_artifacts": missing,
                "produced_by": build_command,
            },
        )
    if run_delivery_video:
        missing_b = [
            name
            for name in EXECUTIVE_DIGEST_SUBSTAGE_B_ARTIFACTS
            if not (deck_dir / name).is_file()
        ]
        if missing_b:
            return pc.block(
                blocker_code="presentation_executive_digest_video_artifacts_missing",
                checks={
                    **common_checks,
                    "missing_artifacts": missing_b,
                    "produced_by": (
                        ".venv/bin/python scripts/presentations/build_executive_videos.py "
                        f"--workspace {workspace} --deck-id {deck_id}"
                    ),
                },
            )
        # Path-based delivery checks ($0) on the two videos build_executive_videos.py
        # rendered: duration budget, stream alignment, build-receipt agreement,
        # black tail. Same checks a topic deck gets after its video sync.
        video_delivery_output = deck_dir / "video_delivery_validation.json"
        blocked = owner_step(
            pc,
            resume=resume,
            command=[
                sys.executable,
                "scripts/presentations/validate_video_delivery.py",
                "--deck-dir",
                str(deck_dir),
                "--presentation-id",
                deck_id,
                "--video",
                str(deck_dir / "presentation_video.mp4"),
                "--video",
                str(deck_dir / "presentation_video_notebooklm.mp4"),
                "--report-out",
                str(video_delivery_output),
            ],
            paid=False,
            outputs=(video_delivery_output,),
            failed_blocker_code="presentation_video_delivery_failed",
            missing_blocker_code="presentation_video_delivery_outputs_missing",
            extra_checks=common_checks,
        )
        if blocked is not None:
            return blocked
        owner_evidence_candidates["presentation.video_delivery_validation"] = (
            f"{video_delivery_output}:{file_sha256(video_delivery_output)}"
        )

    # The deck's slide count has to agree across the contract, the render
    # spec and the frames the video is built from: a deck rebuilt without
    # re-exporting its frames would otherwise pass every existence check and
    # desynchronise the video.
    counts: dict[str, int] = {}
    try:
        deck_payload = json.loads((deck_dir / "presentation_deck.json").read_text(encoding="utf-8"))
        counts["presentation_deck"] = len(deck_payload.get("slides") or [])
        spec_payload = json.loads((deck_dir / "render_spec.json").read_text(encoding="utf-8"))
        counts["render_spec"] = len(spec_payload.get("slides") or [])
        frames_payload = json.loads((deck_dir / "slide_frame_manifest.json").read_text(encoding="utf-8"))
        counts["slide_frame_manifest"] = len(frames_payload.get("frames") or frames_payload.get("slides") or [])
    except (OSError, ValueError) as error:
        return pc.block(
            blocker_code="presentation_executive_digest_artifacts_unreadable",
            checks={**common_checks, "error": str(error)[:400]},
        )
    if len(set(counts.values())) != 1:
        return pc.block(
            blocker_code="presentation_executive_digest_slide_count_mismatch",
            checks={**common_checks, "slide_counts": counts, "produced_by": build_command},
        )

    # Render gates. Deterministic geometry first ($0): it catches cropped and
    # overlapping text without spending a provider call, so a deck broken that
    # way never reaches the paid judge.
    screenshots_dir = deck_dir / "assets" / "generated" / "qc" / "screenshots"
    inspect_report_output = deck_dir / "assets" / "generated" / "qc" / "inspect_report.json"
    blocked = owner_step(
        pc,
        resume=resume,
        command=[
            sys.executable,
            "scripts/presentations/inspect_rendered_slides.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            deck_id,
            "--presentation-type",
            presentation_type,
            "--topic",
            deck_id,
            "--input-html",
            str(deck_dir / "presentation.html"),
            "--report-output",
            str(inspect_report_output),
            "--screenshots-dir",
            str(screenshots_dir),
        ],
        paid=False,
        outputs=(inspect_report_output,),
        failed_blocker_code="presentation_screenshots_failed",
        missing_blocker_code="presentation_screenshots_outputs_missing",
        extra_checks=common_checks,
        tracked_outputs=(StepInput(screenshots_dir, pattern="*.png"),),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.inspect_rendered_slides"] = (
        f"{inspect_report_output}:{file_sha256(inspect_report_output)}"
    )

    # Vision-LLM judge (PAID). Same special-case as the topic path: the script
    # exits non-zero when it records findings, so the report content is the
    # verdict and a missing report is the execution failure.
    visual_llm_qc_output = deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
    visual_qc_spec = step_spec(
        pc,
        command=[
            sys.executable,
            "scripts/presentations/inspect_rendered_slides_llm.py",
            "--presentation-id",
            deck_id,
            "--presentation-type",
            presentation_type,
        ],
        outputs=(visual_llm_qc_output,),
        step_id="inspect_rendered_slides_llm",
        extra_inputs=(StepInput(screenshots_dir, pattern="*.png", optional=False),),
        reject_statuses=frozenset(),
    )
    visual_qc_skip, visual_qc_snapshot = step_gate(visual_qc_spec, resume=resume)
    if not visual_qc_skip:
        visual_qc_command = [
            sys.executable,
            "scripts/presentations/inspect_rendered_slides_llm.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            deck_id,
            "--presentation-type",
            presentation_type,
            "--topic",
            deck_id,
            "--screenshots-dir",
            str(screenshots_dir),
            "--report-output",
            str(visual_llm_qc_output),
        ]
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=pc.stage_id,
            caller=pc.ctx.receipt_caller,
        ):
            visual_qc_result = pc.command_runner(visual_qc_command)
        if not visual_llm_qc_output.is_file() or visual_llm_qc_output.stat().st_size <= 0:
            return pc.block(
                blocker_code="presentation_visual_qc_outputs_missing",
                checks={
                    **common_checks,
                    "command": visual_qc_command,
                    "stdout": sanitize_for_receipt(visual_qc_result.stdout[-2000:]),
                    "stderr": sanitize_for_receipt(visual_qc_result.stderr[-2000:]),
                },
            )
        step_record(visual_qc_spec, visual_qc_snapshot)
    qc_report, qc_error = load_json(visual_llm_qc_output)
    if qc_error or not isinstance(qc_report, dict):
        return pc.block(
            blocker_code="presentation_visual_qc_failed",
            checks={**common_checks, "error": qc_error or "visual QC report is not an object"},
        )
    qc_slides = qc_report.get("slides") if isinstance(qc_report.get("slides"), list) else []
    qc_failing = [
        slide for slide in qc_slides if isinstance(slide, Mapping) and slide.get("status") == "fail"
    ]
    # There is no repair loop here: an executive deck's figures are compiled
    # from its digest plan, so the fix is a plan/spec change, not a redraw.
    # Findings therefore either block or are accepted on the record -- never
    # laundered into a pass.
    if qc_failing and not accept_visual_qc_findings:
        return pc.block(
            blocker_code="presentation_visual_qc_findings_unresolved",
            checks={
                **common_checks,
                "visual_qc_report_path": str(visual_llm_qc_output),
                "failing_slides": [
                    {
                        "slide_number": slide.get("slide_number"),
                        "findings": slide.get("findings"),
                    }
                    for slide in qc_failing[:12]
                ],
                "failing_slide_count": len(qc_failing),
                "produced_by": build_command,
            },
        )

    # Advisory second-opinion visual QC (only when the conference configures one), the same
    # step a topic deck gets. An executive deck has no Gate A review to hand it to, so the
    # summary is recorded on the deck and left beside the QC report for the operator. It is
    # not owner evidence and never blocks.
    advisory_summary = run_advisory_visual_qc(
        pc,
        workspace=workspace,
        run_id=run_id,
        deck_id=deck_id,
        presentation_type=presentation_type,
        deck_dir=deck_dir,
        screenshots_dir=screenshots_dir,
        primary_report=visual_llm_qc_output,
    )

    record: dict[str, Any] = {
        "deck_id": deck_id,
        "deck_kind": kind,
        "authored_by": "digest_chain",
        "slide_count": counts["render_spec"],
        "inspect_report_path": str(inspect_report_output),
        "visual_llm_qc_report_path": str(visual_llm_qc_output),
        "visual_qc_failing_slide_count": len(qc_failing),
        "visual_qc_findings_accepted": bool(qc_failing and accept_visual_qc_findings),
    }
    if advisory_summary is not None:
        record["visual_qc_advisory"] = {
            key: advisory_summary.get(key)
            for key in ("status", "provider", "model", "flagged_slide_count", "flagged_only_by_advisory", "reason", "error")
            if advisory_summary.get(key) is not None
        }
        record["visual_qc_advisory_summary_path"] = str(deck_dir / ADVISORY_SUMMARY)
    # Same record keys a topic deck uses, so a consumer of ``deck_records``
    # reads one shape for all 14 decks (and ``presentation_deck_path`` is what
    # the stage's ``presentation.executive_overview_decks`` evidence reads).
    names = {
        "presentation_blueprint_path": "presentation_blueprint.json",
        "blueprint_review_path": "blueprint_review.json",
        "presentation_deck_path": "presentation_deck.json",
        "source_binding_report_path": "source_binding_report.json",
        "visual_plan_path": "visual_plan.json",
        "render_spec_path": "render_spec.json",
        "presentation_slide_mapping_path": "presentation_slide_mapping.json",
        "presentation_html_path": "presentation.html",
        "presentation_pdf_path": "presentation.pdf",
        "figure_svg_validation_path": "figure_svg_validation_report.json",
        "slide_frame_manifest_path": "slide_frame_manifest.json",
    }
    if run_delivery_video:
        names.update(
            {
                "delivery_single_speaker_path": "delivery_single_speaker.json",
                "delivery_notebooklm_path": "delivery_notebooklm.json",
                "presentation_video_path": "presentation_video.mp4",
                "presentation_video_notebooklm_path": "presentation_video_notebooklm.mp4",
                "video_sync_validation_single_speaker_path": "video_sync_validation_single_speaker.json",
                "video_sync_validation_notebooklm_path": "video_sync_validation_notebooklm.json",
                "video_delivery_validation_path": "video_delivery_validation.json",
            }
        )
    for key, name in names.items():
        record[key] = str(deck_dir / name)
    deck_records.append(record)

    owner_evidence_candidates["presentation.generate_editorial_blueprint"] = (
        f"{deck_dir / 'presentation_blueprint.json'}:"
        f"{file_sha256(deck_dir / 'presentation_blueprint.json')}"
    )
    owner_evidence_candidates["presentation.blueprint_review"] = (
        f"{deck_dir / 'blueprint_review.json'}:{file_sha256(deck_dir / 'blueprint_review.json')}"
    )
    if run_delivery_video:
        for name, evidence_id in (
            ("presentation_video.mp4", "presentation.build_presenter_video"),
            ("presentation_video_notebooklm.mp4", "presentation.build_notebooklm_video"),
        ):
            owner_evidence_candidates[evidence_id] = (
                f"{deck_dir / name}:{file_sha256(deck_dir / name)}"
            )
    return None


def run_deck_chain(
    pc: PresentationChainContext,
    *,
    deck: Mapping[str, str],
    resume: bool,
    checks: Mapping[str, Any],
    owner_evidence_candidates: dict[str, str],
    deck_records: list[dict[str, Any]],
    run_delivery_video: bool = False,
    blueprint_approved_by: str | None = None,
    visual_qc_max_rounds: int | None = None,
    accept_visual_qc_findings: bool = False,
    figure_route: str = FIGURE_ROUTE_DRAWSVG,
    allow_figure_variety_findings: bool = False,
    accepted_review_deck_ids: frozenset[str] = frozenset(),
    adopt_existing_outputs: bool = False,
) -> dict[str, Any] | None:
    """Bind this deck's step ledger for the chain's duration, then run it.

    See ``_presentation_run_deck_chain_steps`` for the chain itself. The
    ledger lives at ``decks/<deck_id>/step_lineage.json``; every owner step
    in the chain consults it through the context variable instead of an
    existence check.
    """

    deck_dir = pc.ctx.deck_dir(str(deck["id"]))
    with bind_deck_ledger(pc, deck_dir, adopt_existing=adopt_existing_outputs):
        return _deck_chain_steps(
            pc,
            deck=deck,
            resume=resume,
            checks=checks,
            owner_evidence_candidates=owner_evidence_candidates,
            deck_records=deck_records,
            run_delivery_video=run_delivery_video,
            blueprint_approved_by=blueprint_approved_by,
            visual_qc_max_rounds=visual_qc_max_rounds,
            accept_visual_qc_findings=accept_visual_qc_findings,
            figure_route=figure_route,
            allow_figure_variety_findings=allow_figure_variety_findings,
            accepted_review_deck_ids=accepted_review_deck_ids,
        )



def _deck_chain_steps(
    pc: PresentationChainContext,
    *,
    deck: Mapping[str, str],
    resume: bool,
    checks: Mapping[str, Any],
    owner_evidence_candidates: dict[str, str],
    deck_records: list[dict[str, Any]],
    run_delivery_video: bool = False,
    blueprint_approved_by: str | None = None,
    visual_qc_max_rounds: int | None = None,
    accept_visual_qc_findings: bool = False,
    figure_route: str = FIGURE_ROUTE_DRAWSVG,
    allow_figure_variety_findings: bool = False,
    accepted_review_deck_ids: frozenset[str] = frozenset(),
) -> dict[str, Any] | None:
    """Advance one deck through the full sub-stage A+B chain
    (10-RESEARCH.md architecture diagram): paper matching (PAID, in-process)
    -> the 4 early Critical-Finding-2 deterministic scripts (source corpus
    inventory -> speaker/author work ledger -> source repair -> research
    cluster plan, in ``run_topic_presentation_build.py``'s exact order) ->
    blueprint generation (PAID, Opus) + review ($0) -> deck-contract
    generation ($0) -> source binding ($0) -> figures/render/export/
    screenshots/visual-QC/bounded-repair/artifact-truth (Plan 10-03) -- all
    always run. When ``run_delivery_video`` is ``True`` (``presentation_substage``
    in ``{"delivery_video", "all"}``), the chain continues into Plan 10-04's
    sub-stage B: delivery-contract generation + both-mode narration
    lint/review/fix (pre-TTS) -> both videos -> video-sync validation -> the
    3-model presentation review. When ``False`` (``presentation_substage ==
    "deck_figures"``), the function returns immediately after artifact-truth
    validation -- delivery/video steps are never even attempted (not merely
    skipped via the idempotent per-step mechanism).

    Returns a ``block_presentation_stage(...)``-shaped outcome dict on any
    per-deck failure, or ``None`` if this deck advanced cleanly through
    every step wired for the requested scope. Per the continue-past-failures
    policy (operator decision, Plan 10-06), the caller's per-deck loop
    records a non-``None`` return via ``_presentation_extract_deck_blocker``
    and continues to the next deck -- it no longer returns immediately and
    halts the whole stage on one deck's blocker. Mutates
    ``owner_evidence_candidates`` in place, keyed by
    ``stage_owner_registry.json`` script-registry ids (never model-governance
    task ids or workflow-stage ids -- RESEARCH Pitfall 10); paper matching
    itself has no entry in ``required_script_ids_for_stage``, so no evidence
    token is recorded for this step. Single-speaker narration lint/review/fix
    likewise have no dedicated required_script_ids entries of their own (only
    the NotebookLM-mode narration_script_review/fix and
    narration_series_review/fix ids exist in the registry) -- they still run
    (real work, real reports) but record no owner-evidence key.
    """

    workspace = pc.ctx.workspace
    run_id = pc.ctx.run_id

    deck_id = str(deck["id"])
    deck_title = str(deck["title"])
    kind = str(deck["kind"])
    presentation_type = (
        EXECUTIVE_PRESENTATION_TYPE
        if kind == "executive"
        else TOPIC_PRESENTATION_TYPE
    )
    deck_dir = pc.ctx.deck_dir(deck_id)
    deck_dir.mkdir(parents=True, exist_ok=True)
    common_checks = {**checks, "deck_id": deck_id, "deck_kind": kind}

    if kind == "executive":
        # Verified, not rebuilt -- see the helper's docstring for why the
        # per-deck chain below cannot produce an executive deck.
        return executive_digest_chain(
            pc,
            deck_id=deck_id,
            kind=kind,
            deck_dir=deck_dir,
            deck_records=deck_records,
            owner_evidence_candidates=owner_evidence_candidates,
            run_delivery_video=run_delivery_video,
            common_checks=common_checks,
            presentation_type=presentation_type,
            resume=resume,
            accept_visual_qc_findings=accept_visual_qc_findings,
        )

    # Step 1: paper matching -- an IN-PROCESS call to
    # ``conference_pipeline.paper_matching.run_paper_matching_stage``
    # (registry task ``research.paper_matching_keypoints``, PAID), never a
    # subprocess (RESEARCH Critical Finding 3; mocked via monkeypatch in
    # tests, not ``command_runner``). ``ranked_papers.json`` is
    # compatibility-only (done gate 3) -- ``matched_papers.json`` is the
    # real evidence this step verifies.
    matched_papers_path = (
        workspace / PAPER_MATCHING_DIR / run_id / deck_id / "matched_papers.json"
    )
    paper_matching_spec = StepSpec(
        step_id="paper_matching",
        inputs=(
            StepInput(workspace / "analysis" / f"{deck_id}.md"),
            StepInput(workspace / "storage" / "extractions" / run_id / "by-topic" / f"{deck_id}.json"),
        ),
        outputs=(matched_papers_path,),
        values={"run_id": run_id, "topic_id": deck_id},
    )
    paper_matching_skip, paper_matching_snapshot = step_gate(paper_matching_spec, resume=resume)
    if not paper_matching_skip:
        try:
            with provider_receipt_context(
                workspace_root=workspace,
                run_id=run_id,
                stage_id=pc.stage_id,
                caller=pc.ctx.receipt_caller,
            ):
                presentation_paper_matching.run_paper_matching_stage(
                    workspace_root=workspace,
                    run_id=run_id,
                    topic_id=deck_id,
                )
        except Exception as exc:  # noqa: BLE001 - presentation per-deck boundary
            return pc.block(
                blocker_code="presentation_paper_matching_failed",
                checks={**common_checks, "error": sanitize_for_receipt(str(exc))},
            )
    if not matched_papers_path.is_file() or matched_papers_path.stat().st_size <= 0:
        return pc.block(
            blocker_code="presentation_paper_matching_outputs_missing",
            checks={**common_checks, "matched_papers_path": str(matched_papers_path)},
        )
    if not paper_matching_skip:
        step_record(paper_matching_spec, paper_matching_snapshot)

    # Steps 2-5: the 4 early Critical-Finding-2 deterministic ($0) scripts,
    # in ``run_topic_presentation_build.py``'s exact order: corpus inventory
    # -> work ledger -> source repair -> cluster plan.
    corpus_inventory_output = deck_dir / "source_corpus_inventory.json"
    source_inputs = corpus_inventory_source_inputs(pc.ctx, deck_id, kind)
    if not source_inputs:
        return pc.block(
            blocker_code="presentation_source_corpus_inventory_no_inputs_found",
            checks=common_checks,
        )
    corpus_inventory_command = [
        sys.executable,
        "scripts/presentations/build_presentation_source_corpus_inventory.py",
        "--workspace",
        str(workspace),
        "--topic-id",
        deck_id,
        "--deck-dir",
        str(deck_dir),
        *[
            arg
            for source_input in source_inputs
            for arg in ("--source-input", str(workspace / source_input))
        ],
        "--output-json",
        str(corpus_inventory_output),
        "--receipt-path",
        str(deck_dir / "source_corpus_inventory_receipt.json"),
        "--summary",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=corpus_inventory_command,
        paid=False,
        outputs=(corpus_inventory_output,),
        failed_blocker_code="presentation_source_corpus_inventory_failed",
        missing_blocker_code="presentation_source_corpus_inventory_outputs_missing",
        extra_checks=common_checks,
        # Deck-local source inputs (ranked papers, source pack receipt, ...)
        # are products of later steps in this same chain; binding them here
        # would make every second run re-derive the inventory. Analysis-side
        # inputs stay bound.
        ignored_inputs=tuple(
            workspace / source_input
            for source_input in source_inputs
            if source_input.startswith(f"{pc.ctx.decks_dir}/")
        ),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.source_corpus_inventory"] = (
        f"{corpus_inventory_output}:{file_sha256(corpus_inventory_output)}"
    )

    work_ledger_output = deck_dir / "speaker_author_work_ledger.json"
    work_ledger_command = [
        sys.executable,
        "scripts/presentations/build_speaker_author_work_ledger.py",
        "--workspace",
        str(workspace),
        "--inventory-json",
        str(corpus_inventory_output),
        "--output-json",
        str(work_ledger_output),
        "--receipt-path",
        str(deck_dir / "speaker_author_work_ledger_receipt.json"),
        "--summary",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=work_ledger_command,
        paid=False,
        outputs=(work_ledger_output,),
        failed_blocker_code="presentation_speaker_author_work_ledger_failed",
        missing_blocker_code="presentation_speaker_author_work_ledger_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.speaker_author_work_ledger"] = (
        f"{work_ledger_output}:{file_sha256(work_ledger_output)}"
    )

    source_repair_output = deck_dir / "source_repair_findings.json"
    source_repair_command = [
        sys.executable,
        "scripts/presentations/repair_presentation_sources.py",
        "--workspace",
        str(workspace),
        "--inventory-json",
        str(corpus_inventory_output),
        "--ledger-json",
        str(work_ledger_output),
        "--output-json",
        str(source_repair_output),
        "--receipt-path",
        str(deck_dir / "source_repair_receipt.json"),
        "--summary",
    ]
    # Reviewed repairs recorded for this deck must survive every rebuild.
    reviewed_repairs = deck_dir / "source_repair_reviewed.json"
    if reviewed_repairs.is_file():
        source_repair_command += ["--reviewed-repairs-json", str(reviewed_repairs)]
    blocked = owner_step(
        pc,
        resume=resume,
        command=source_repair_command,
        paid=False,
        outputs=(source_repair_output,),
        failed_blocker_code="presentation_repair_presentation_sources_failed",
        missing_blocker_code="presentation_repair_presentation_sources_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.repair_presentation_sources"] = (
        f"{source_repair_output}:{file_sha256(source_repair_output)}"
    )

    cluster_plan_output = deck_dir / "research_cluster_plan.json"
    cluster_plan_command = [
        sys.executable,
        "scripts/presentations/build_research_cluster_plan.py",
        "--workspace",
        str(workspace),
        "--inventory-json",
        str(corpus_inventory_output),
        "--ledger-json",
        str(work_ledger_output),
        "--repair-findings-json",
        str(source_repair_output),
        "--output-json",
        str(cluster_plan_output),
        "--markdown-output",
        str(deck_dir / "research_cluster_plan.md"),
        "--receipt-path",
        str(deck_dir / "research_cluster_plan_receipt.json"),
        "--summary",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=cluster_plan_command,
        paid=False,
        outputs=(cluster_plan_output,),
        failed_blocker_code="presentation_research_cluster_plan_failed",
        missing_blocker_code="presentation_research_cluster_plan_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.research_cluster_plan"] = (
        f"{cluster_plan_output}:{file_sha256(cluster_plan_output)}"
    )

    # Step 6: blueprint generation (PAID, Opus) + review ($0).
    blueprint_output = deck_dir / "presentation_blueprint.json"
    blueprint_md_output = deck_dir / "presentation_blueprint.md"
    blueprint_receipt_output = deck_dir / "presentation_blueprint_receipt.json"
    analysis_md_path = workspace / "analysis" / f"{deck_id}.md"
    blueprint_command = [
        sys.executable,
        "scripts/presentations/generate_editorial_blueprint.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--topic-title",
        deck_title,
    ]
    if analysis_md_path.is_file():
        blueprint_command.extend(["--analysis-md", str(analysis_md_path)])
    blueprint_command.extend(
        [
            "--speaker-author-work-ledger",
            str(work_ledger_output),
            "--source-corpus-inventory",
            str(corpus_inventory_output),
            "--source-repair-findings",
            str(source_repair_output),
            "--research-cluster-plan",
            str(cluster_plan_output),
            "--final-report-url",
            presentation_commands.presentation_final_report_url(workspace),
            "--output-json",
            str(blueprint_output),
            "--output-md",
            str(blueprint_md_output),
            "--receipt-path",
            str(blueprint_receipt_output),
            "--provider-cost-approved",
        ]
    )
    blocked = owner_step(
        pc,
        resume=resume,
        command=blueprint_command,
        paid=True,
        outputs=(blueprint_output, blueprint_md_output, blueprint_receipt_output),
        failed_blocker_code="presentation_blueprint_generation_failed",
        missing_blocker_code="presentation_blueprint_generation_outputs_missing",
        extra_checks=common_checks,
        # A versioned repair (repair_presentation_blueprint.py) rewrites all
        # three; the generator is not re-run for it, its consumers are.
        editable_outputs=(blueprint_output, blueprint_md_output, blueprint_receipt_output),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.generate_editorial_blueprint"] = (
        f"{blueprint_output}:{file_sha256(blueprint_output)}"
    )

    blueprint_review_output = deck_dir / "blueprint_review.json"
    # The review is an approval token, not commentary: its LLM-judge gate is
    # only recorded when the judge artifact (written by the
    # conference-presentation-blueprint-review skill) exists on disk, and a
    # review that did not reach ``pass`` is never resume-skipped.
    blueprint_llm_judge_artifact = deck_dir / "blueprint_llm_judge.md"
    existing_review, _existing_review_error = load_json(blueprint_review_output)
    if blueprint_review_output.is_file() and not (
        isinstance(existing_review, dict) and existing_review.get("status") == "pass"
    ):
        blueprint_review_output.unlink()
    blueprint_review_command = [
        sys.executable,
        "scripts/presentations/review_presentation_blueprint.py",
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--presentation-blueprint",
        str(blueprint_output),
        "--blueprint-receipt",
        str(blueprint_receipt_output),
        "--output-json",
        str(blueprint_review_output),
        "--output-md",
        str(deck_dir / "blueprint_review.md"),
        "--approve-for-draft-generation",
        "--approve-for-deck-contract-generation",
        "--approved-by",
        blueprint_approved_by or pc.automated_approver_id,
    ]
    if blueprint_llm_judge_artifact.is_file() and blueprint_llm_judge_artifact.stat().st_size > 0:
        blueprint_review_command.extend(
            ["--llm-judge-approved", "--llm-judge-artifact", str(blueprint_llm_judge_artifact)]
        )
    blocked = owner_step(
        pc,
        resume=resume,
        command=blueprint_review_command,
        paid=False,
        outputs=(blueprint_review_output,),
        failed_blocker_code="presentation_blueprint_review_failed",
        missing_blocker_code="presentation_blueprint_review_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(StepInput(blueprint_llm_judge_artifact, optional=True),),
        accept_statuses=frozenset({"pass"}),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.review_presentation_blueprint"] = (
        f"{blueprint_review_output}:{file_sha256(blueprint_review_output)}"
    )

    # Step 7: deck-contract generation ($0 -- no model_governance import).
    deck_contract_output = deck_dir / "presentation_deck.json"
    deck_contract_command = [
        sys.executable,
        "scripts/presentations/generate_deck_contract.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--presentation-blueprint",
        str(blueprint_output),
        "--blueprint-receipt",
        str(blueprint_receipt_output),
        "--blueprint-review",
        str(blueprint_review_output),
        "--research-cluster-plan",
        str(cluster_plan_output),
        "--speaker-author-work-ledger",
        str(work_ledger_output),
        "--output",
        str(deck_contract_output),
        "--receipt-output",
        str(deck_dir / "presentation_deck_receipt.json"),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=deck_contract_command,
        paid=False,
        outputs=(deck_contract_output,),
        failed_blocker_code="presentation_deck_contract_failed",
        missing_blocker_code="presentation_deck_contract_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.generate_deck_contract"] = (
        f"{deck_contract_output}:{file_sha256(deck_contract_output)}"
    )

    # Step 8: source binding -- build_deck_source_pack.py ($0) followed by
    # bind_presentation_sources.py ($0); both share the single
    # "presentation_source_binding_failed" blocker code (one named "source
    # binding" step, two owner scripts, per the artifacts-produced list).
    #
    # build_deck_source_pack.py requires
    # storage/extractions/<run-id>/by-topic/<presentation_id>.json
    # (conference_pipeline.presentation.source_pack.build_deck_source_pack),
    # which the extraction stage only ever produces for the 12 numbered
    # topics -- there is no by-topic record for the synthetic
    # executive_summary_short/executive_summary_long ids (they synthesize
    # ACROSS all 12 topics from Phase 8's executive_summary.md/
    # trend_report.md, never from Stage 4/5's raw per-topic extraction
    # records). This mirrors the executive-deck fallback already
    # established for the 4 Critical-Finding-2 scripts (Plan 10-02).
    # bind_presentation_sources.py's own --source-pack argument is optional
    # and is not even passed below, so skipping this step for executive
    # decks drops no information bind_presentation_sources.py would
    # otherwise consume. owner_evidence_candidates records one
    # non-per-deck-required token for the whole stage (see the
    # initialize-once-overwrite-per-deck note above), so skipping this step
    # for the 2 executive decks does not starve the
    # presentation.source_pack evidence entry -- it is still recorded from
    # whichever of the 12 topic decks runs it.
    source_pack_receipt_output = deck_dir / "source_pack_receipt.json"
    if kind != "executive":
        source_pack_command = [
            sys.executable,
            "scripts/presentations/build_deck_source_pack.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            deck_id,
            "--presentation-type",
            presentation_type,
            "--topic",
            deck_id,
            "--extraction-run-id",
            run_id,
        ]
        blocked = owner_step(
            pc,
            resume=resume,
            command=source_pack_command,
            paid=False,
            outputs=(source_pack_receipt_output,),
            failed_blocker_code="presentation_source_binding_failed",
            missing_blocker_code="presentation_source_binding_outputs_missing",
            extra_checks=common_checks,
        )
        if blocked is not None:
            return blocked
        owner_evidence_candidates["presentation.source_pack"] = (
            f"{source_pack_receipt_output}:{file_sha256(source_pack_receipt_output)}"
        )

    source_binding_output = deck_dir / "source_binding_report.json"
    source_binding_command = [
        sys.executable,
        "scripts/presentations/bind_presentation_sources.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--deck-contract",
        str(deck_contract_output),
        "--speaker-author-work-ledger",
        str(work_ledger_output),
        "--research-cluster-plan",
        str(cluster_plan_output),
        "--matched-papers",
        str(matched_papers_path),
        "--output",
        str(source_binding_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=source_binding_command,
        paid=False,
        outputs=(source_binding_output,),
        failed_blocker_code="presentation_source_binding_failed",
        missing_blocker_code="presentation_source_binding_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.bind_presentation_sources"] = (
        f"{source_binding_output}:{file_sha256(source_binding_output)}"
    )

    # Step 9: programmatic figure generation (PAID -- drawsvg LLM codegen)
    # + structural review/validate + render-spec generation (Plan 10-03).
    #
    # ``generate_programmatic_figures.py``'s own ``--deck-contract`` branch
    # is a two-pass design (mirrors ``run_topic_presentation_build.py``'s
    # ``stage_figures_pass1``/``stage_figures_pass2``): pass 1 is the
    # deterministic ($0) structured-visual-plan writer (creates
    # ``visual_plan.json``/``asset_inventory.json`` with placeholder SVGs);
    # pass 2 is the PAID Phase 16 LLM-authored-drawsvg-code route, which
    # *reads* the pass-1 ``visual_plan.json`` and overwrites it with
    # real generated SVGs -- the script's own ``generate_phase16_llm_visual_plan``
    # requires ``--visual-plan`` to already exist. Both share one blocker-code
    # family (mirrors Step 8's two-scripts-one-blocker-code precedent above).
    deck_contract_payload, deck_contract_error = load_json(deck_contract_output)
    if deck_contract_error or not isinstance(deck_contract_payload, dict):
        return pc.block(
            blocker_code="presentation_deck_contract_unreadable",
            checks={**common_checks, "error": deck_contract_error},
        )
    bootstrap_delivery_stub(deck_dir=deck_dir, deck_contract=deck_contract_payload)

    visual_plan_output = deck_dir / "visual_plan.json"
    asset_inventory_output = deck_dir / "asset_inventory.json"
    generated_dir = deck_dir / "assets" / "generated"
    # compile_figures.py writes SVGs to assets/compiled/, not assets/generated/
    # (that is the drawsvg route's own output directory). Any lineage tracking
    # or validation of "the figures this deck just produced" must follow the
    # route, or it watches/scans an empty directory on the compiler route and
    # a stale one left over from a prior drawsvg attempt on the same deck.
    figure_output_dir = (
        deck_dir / "assets" / "compiled"
        if figure_route == FIGURE_ROUTE_COMPILER
        else generated_dir
    )
    def _claim_fidelity_step(phase: str):
        report = deck_dir / f"claim_fidelity_review_{phase}.json"
        result = owner_step(
            pc, resume=resume,
            command=[sys.executable, "scripts/presentations/review_claim_fidelity.py",
                     "--workspace", str(workspace), "--deck-dir", str(deck_dir),
                     "--phase", phase, "--report-output", str(report)],
            step_id=f"claim_fidelity_review_{phase}", paid=True, outputs=(report,),
            failed_blocker_code="presentation_claim_fidelity_failed",
            missing_blocker_code="presentation_claim_fidelity_outputs_missing",
            extra_checks=common_checks, extra_inputs=(source_binding_output, deck_contract_output,
                deck_dir / "figure_specs", deck_dir / "figure_spec_receipt.json", *(
                (deck_dir / "6_presenter_script.ENGAGING.md", deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md")
                if phase == "narration" else ())),
            directory_patterns={deck_dir / "figure_specs": "*.json"},
            accept_statuses=frozenset({"passed"}),
        )
        if result is None:
            owner_evidence_candidates["presentation.claim_fidelity_review"] = f"{report}:{file_sha256(report)}"
        return result

    figure_specs_output = deck_dir / "figure_specs.json"
    if figure_route == FIGURE_ROUTE_COMPILER:
        # Route B: author declarative specs (PAID, one cheap call per slide),
        # then compile them deterministically ($0). The compiler writes
        # visual_plan.json and asset_inventory.json itself, so the rest of the
        # chain -- render spec, export, screenshots -- is unchanged.
        figure_spec_receipt_output = deck_dir / "figure_spec_receipt.json"
        numeric_ledger_output = deck_dir / "numeric_claims_ledger.json"
        spec_authoring_command = [
            sys.executable,
            "scripts/presentations/generate_figure_specs.py",
            "--deck-dir",
            str(deck_dir),
            "--workspace",
            str(workspace),
            "--receipt",
            str(figure_spec_receipt_output),
            # The spec critic reads each spec back before it is compiled and
            # buys one re-author for a structure that misstates the claim. It
            # ran only when the script was invoked by hand (decks 01-03); the
            # runner never asked for it, so a deck built end to end through
            # the runner shipped its first draft of every spec (deck 04).
            "--critic",
        ]
        figure_slide_ids = FIGURE_SLIDE_IDS.get()
        if figure_slide_ids:
            # Re-author only these slides' figures; the generator merges the
            # receipt so the untouched slides stay active for the compiler.
            spec_authoring_command.extend(["--slide-ids", ",".join(figure_slide_ids)])
        if resume and "generate_figure_specs" not in RERUN_STEPS.get() and not figure_slide_ids:
            # A compiler or authoring-code revision invalidates the step
            # ledger, but already passing specs need no new paid authoring.
            # The generator recompiles each candidate before it reuses it.
            # An operator --rerun-steps generate_figure_specs means re-author.
            spec_authoring_command.append("--skip-existing")
        blocked = owner_step(
            pc,
            resume=resume,
            command=spec_authoring_command,
            paid=True,
            outputs=(figure_spec_receipt_output, numeric_ledger_output),
            failed_blocker_code="presentation_figure_spec_authoring_failed",
            missing_blocker_code="presentation_figure_spec_authoring_outputs_missing",
            extra_checks=common_checks,
            extra_inputs=(deck_contract_output, source_binding_output, blueprint_output),
            tracked_outputs=(figure_specs_output,),
        )
        if blocked is not None:
            return blocked

        blocked = _claim_fidelity_step("figures")
        if blocked is not None:
            return blocked

        figure_compile_receipt_output = deck_dir / "figure_compile_receipt.json"
        compile_command = [
            sys.executable,
            "scripts/presentations/compile_figures.py",
            "--deck-dir",
            str(deck_dir),
            "--write-render-contracts",
            *(["--vision-deck"] if pc.ctx.figure_vision_deck else []),
            "--receipt",
            str(figure_compile_receipt_output),
        ]
        if allow_figure_variety_findings:
            compile_command.append("--allow-variety-findings")
        blocked = owner_step(
            pc,
            resume=resume,
            command=compile_command,
            paid=False,
            outputs=(figure_compile_receipt_output, visual_plan_output, asset_inventory_output),
            failed_blocker_code="presentation_figure_compile_failed",
            missing_blocker_code="presentation_figure_compile_outputs_missing",
            extra_checks=common_checks,
            # The compiler reads figure_specs/*-figure.json, and
            # patch_figure_spec.py edits those files by hand. figure_specs.json is
            # this step's own output, so it cannot signal a patch: the ICML 2026
            # deck 01 slide 7/15/17 patches were skipped as "current" and the
            # videos kept the old label until compile_figures was forced.
            extra_inputs=(
                StepInput(deck_dir / "figure_specs", pattern="*-figure.json"),
                figure_specs_output,
                numeric_ledger_output,
                figure_spec_receipt_output,
                deck_contract_output,
            ),
            tracked_outputs=(StepInput(figure_output_dir, pattern="*.svg"),),
        )
        if blocked is not None:
            return blocked
        owner_evidence_candidates["presentation.generate_programmatic_figures"] = (
            f"{visual_plan_output}:{file_sha256(visual_plan_output)}"
        )

    # Route A steps below run only for the drawsvg route; the compiler route has
    # already produced visual_plan.json and asset_inventory.json above.
    drawsvg_route = figure_route == FIGURE_ROUTE_DRAWSVG

    figures_pass1_command = [
        sys.executable,
        "scripts/presentations/generate_programmatic_figures.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--deck-contract",
        str(deck_contract_output),
        "--source-binding-report",
        str(source_binding_output),
        "--visual-plan",
        str(visual_plan_output),
        "--asset-inventory",
        str(asset_inventory_output),
        "--generated-dir",
        str(generated_dir),
        "--figure-specs-output",
        str(figure_specs_output),
    ]
    blocked = (
        owner_step(
            pc,
            resume=resume,
            command=figures_pass1_command,
            paid=False,
            outputs=(visual_plan_output, asset_inventory_output),
            failed_blocker_code="presentation_figure_generation_failed",
            missing_blocker_code="presentation_figure_generation_outputs_missing",
            extra_checks=common_checks,
            step_id="generate_programmatic_figures_pass1",
            tracked_outputs=(StepInput(generated_dir, pattern="*.svg"), figure_specs_output),
        )
        if drawsvg_route
        else None
    )
    if blocked is not None:
        return blocked

    figure_code_dir = deck_dir / "assets" / "generated" / "code"
    model_receipt_output = deck_dir / "figure_generation_model_receipt.json"
    code_review_output = deck_dir / "figure_code_review_report.json"
    figures_pass2_command = [
        sys.executable,
        "scripts/presentations/generate_programmatic_figures.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--deck-contract",
        str(deck_contract_output),
        "--source-binding-report",
        str(source_binding_output),
        "--visual-plan",
        str(visual_plan_output),
        "--asset-inventory",
        str(asset_inventory_output),
        "--generated-dir",
        str(generated_dir),
        "--figure-specs-output",
        str(figure_specs_output),
        "--svg-stack-research",
        str(pc.ctx.repo_root / SVG_STACK_RESEARCH),
        "--provider",
        PHASE16_PROVIDER,
        "--model",
        PHASE16_MODEL,
        "--provider-cost-approved",
        "--allowed-renderer",
        PHASE16_ALLOWED_RENDERER,
        *[
            arg
            for renderer in PHASE16_DISALLOWED_RENDERERS
            for arg in ("--disallowed-default-renderer", renderer)
        ],
        "--figure-code-dir",
        str(figure_code_dir),
        "--model-receipt-output",
        str(model_receipt_output),
        "--code-review-output",
        str(code_review_output),
        "--codegen-max-attempts",
        str(FIGURE_CODEGEN_MAX_ATTEMPTS),
    ]
    blocked = (
        owner_step(
            pc,
            resume=resume,
            command=figures_pass2_command,
            paid=True,
            outputs=(visual_plan_output, asset_inventory_output, model_receipt_output, code_review_output),
            failed_blocker_code="presentation_figure_generation_failed",
            missing_blocker_code="presentation_figure_generation_outputs_missing",
            extra_checks=common_checks,
            step_id="generate_programmatic_figures_pass2",
            # Pass 2 reads pass 1's plan/inventory and rewrites them: an in-place
            # editor of its own two required outputs.
            in_place=True,
            tracked_outputs=(StepInput(generated_dir, pattern="*.svg"), figure_specs_output),
        )
        if drawsvg_route
        else None
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.generate_programmatic_figures"] = (
        f"{visual_plan_output}:{file_sha256(visual_plan_output)}"
    )

    figure_semantic_review_output = deck_dir / "figure_semantic_review.json"
    figure_review_command = [
        sys.executable,
        "scripts/presentations/review_programmatic_figures.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--dir",
        str(generated_dir),
        "--asset-inventory",
        str(asset_inventory_output),
        "--report-output",
        str(figure_semantic_review_output),
        "--markdown-output",
        str(deck_dir / "figure_semantic_review.md"),
    ]
    blocked = (
        owner_step(
            pc,
            resume=resume,
            command=figure_review_command,
            paid=False,
            outputs=(figure_semantic_review_output,),
            failed_blocker_code="presentation_figure_review_failed",
            missing_blocker_code="presentation_figure_review_outputs_missing",
            extra_checks=common_checks,
            directory_patterns={generated_dir: "*.svg"},
        )
        if drawsvg_route
        else None
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.review_programmatic_figures"] = (
        f"{figure_semantic_review_output}:{file_sha256(figure_semantic_review_output)}"
    )

    svg_validation_output = deck_dir / "figure_svg_validation_report.json"
    svg_validation_command = [
        sys.executable,
        "scripts/presentations/validate_svgs.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--dir",
        str(figure_output_dir),
        "--json",
        "--report-output",
        str(svg_validation_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=svg_validation_command,
        paid=False,
        outputs=(svg_validation_output,),
        failed_blocker_code="presentation_svg_validation_failed",
        missing_blocker_code="presentation_svg_validation_outputs_missing",
        extra_checks=common_checks,
        directory_patterns={figure_output_dir: "*.svg"},
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.validate_svgs"] = (
        f"{svg_validation_output}:{file_sha256(svg_validation_output)}"
    )

    render_spec_output = deck_dir / "render_spec.json"
    render_spec_command = [
        sys.executable,
        "scripts/presentations/generate_render_spec.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--presentation-blueprint",
        str(blueprint_output),
        "--deck-contract",
        str(deck_contract_output),
        "--source-binding-report",
        str(source_binding_output),
        "--visual-plan",
        str(visual_plan_output),
        "--asset-inventory",
        str(asset_inventory_output),
        "--figure-specs",
        str(figure_specs_output),
        "--figure-generation-model-receipt",
        str(model_receipt_output),
        "--figure-code-review",
        str(code_review_output),
        "--figure-svg-validation",
        str(svg_validation_output),
        "--figure-semantic-review",
        str(figure_semantic_review_output),
        "--output",
        str(render_spec_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=render_spec_command,
        paid=False,
        outputs=(render_spec_output,),
        failed_blocker_code="presentation_render_spec_failed",
        missing_blocker_code="presentation_render_spec_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.generate_render_spec"] = (
        f"{render_spec_output}:{file_sha256(render_spec_output)}"
    )

    # Step 9b: opening "events" + closing "sources & further reading"
    # bookend slides (``presentation.add_bookend_slides``, $0, deterministic,
    # idempotent -- it strips and re-derives its own two slides from the
    # current render_spec source chips). This is where the canonical
    # analysis-report link becomes visible deck text, which the late
    # artifact-truth validator requires (FINAL_REPORT_LINK_MISSING).
    # It rewrites the deck contract and render_spec in place (an ``in_place``
    # step in the deck ledger): it is skipped only while its own write is
    # still the head of both files' history, and when it does change the
    # render_spec every downstream render/export/screenshot/QC step below
    # re-runs because the write they recorded is gone.
    bookend_command = [
        sys.executable,
        "scripts/presentations/add_bookend_slides.py",
        "--deck-dir",
        str(deck_dir),
    ]
    report_url_resolution = resolve_report_url(
        workspace=workspace,
        deck_contract_output=deck_contract_output,
        render_spec_output=render_spec_output,
    )
    blocked = owner_step(
        pc,
        resume=resume,
        command=bookend_command,
        paid=False,
        # The binding report is restamped in place across the bookend rewrite
        # (its deck digest), so it is an in-place output of this step too.
        outputs=(render_spec_output, deck_contract_output, source_binding_output),
        failed_blocker_code="presentation_bookend_slides_failed",
        missing_blocker_code="presentation_bookend_slides_outputs_missing",
        extra_checks={**common_checks, "analysis_report_url": report_url_resolution},
        in_place=True,
        extra_inputs=(render_spec_output, deck_contract_output, work_ledger_output, cluster_plan_output),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.add_bookend_slides"] = (
        f"{render_spec_output}:{file_sha256(render_spec_output)}"
    )

    # Step 10 (Plan 10-03 Task 2): render (HTML) -> export (PDF) -> pptx
    # export -> screenshots -> visual QC (screenshots $0 + vision-LLM PAID)
    # -> bounded repair loop (PAID, max 2 attempts, on QC-blocker findings).
    html_output = deck_dir / "presentation.html"
    slide_mapping_output = deck_dir / "presentation_slide_mapping.json"
    render_html_command = [
        sys.executable,
        "scripts/presentations/md_to_revealjs.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--render-spec",
        str(render_spec_output),
        "--output-html",
        str(html_output),
        "--slide-mapping",
        str(slide_mapping_output),
        "--require-brand-assets",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=render_html_command,
        paid=False,
        outputs=(html_output, slide_mapping_output),
        failed_blocker_code="presentation_render_html_failed",
        missing_blocker_code="presentation_render_html_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(StepInput(figure_output_dir, pattern="*.svg"),),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.md_to_revealjs"] = (
        f"{html_output}:{file_sha256(html_output)}"
    )

    pdf_output = deck_dir / "presentation.pdf"
    render_export_receipt_output = deck_dir / "render_export_receipt.json"
    export_pdf_command = [
        sys.executable,
        "scripts/presentations/export_revealjs_pdf.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--render-spec",
        str(render_spec_output),
        "--input-html",
        str(html_output),
        "--output-pdf",
        str(pdf_output),
        "--slide-mapping",
        str(slide_mapping_output),
        "--frame-manifest",
        str(deck_dir / "slide_frame_manifest.json"),
        "--frames-dir",
        str(deck_dir / "rendered_frames"),
        "--contact-sheet",
        str(deck_dir / "render_contact_sheet.png"),
        "--receipt-output",
        str(render_export_receipt_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=export_pdf_command,
        paid=False,
        outputs=(pdf_output, render_export_receipt_output),
        failed_blocker_code="presentation_pdf_export_failed",
        missing_blocker_code="presentation_pdf_export_outputs_missing",
        extra_checks=common_checks,
        tracked_outputs=(
            deck_dir / "slide_frame_manifest.json",
            StepInput(deck_dir / "rendered_frames", pattern="*.png"),
            deck_dir / "render_contact_sheet.png",
        ),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.export_pdf"] = (
        f"{pdf_output}:{file_sha256(pdf_output)}"
    )

    pptx_export_output = deck_dir / "presentation_pptx_export.pdf"
    pptx_command = [
        sys.executable,
        "scripts/presentations/md_to_pptx.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--render-spec",
        str(render_spec_output),
        "--output-pdf",
        str(pptx_export_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=pptx_command,
        paid=False,
        outputs=(pptx_export_output,),
        failed_blocker_code="presentation_pptx_export_failed",
        missing_blocker_code="presentation_pptx_export_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.md_to_pptx"] = (
        f"{pptx_export_output}:{file_sha256(pptx_export_output)}"
    )

    screenshots_dir = deck_dir / "assets" / "generated" / "qc" / "screenshots"
    inspect_report_output = deck_dir / "assets" / "generated" / "qc" / "inspect_report.json"
    screenshots_command = [
        sys.executable,
        "scripts/presentations/inspect_rendered_slides.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--input-html",
        str(html_output),
        "--report-output",
        str(inspect_report_output),
        "--screenshots-dir",
        str(screenshots_dir),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=screenshots_command,
        paid=False,
        outputs=(inspect_report_output,),
        failed_blocker_code="presentation_screenshots_failed",
        missing_blocker_code="presentation_screenshots_outputs_missing",
        extra_checks=common_checks,
        # presentation.html references the figures by path, so a re-authored
        # figure leaves it byte-identical and this step read as current: on
        # 2026-10-01 both ICML canary decks re-drew their figures and the
        # screenshots and vision judge were skipped. The drawn SVGs are inputs.
        extra_inputs=(StepInput(figure_output_dir, pattern="*.svg", optional=True),),
        tracked_outputs=(StepInput(screenshots_dir, pattern="*.png"),),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.inspect_rendered_slides"] = (
        f"{inspect_report_output}:{file_sha256(inspect_report_output)}"
    )

    # Visual QC (PAID vision-LLM pass): special-cased (not via
    # ``_presentation_owner_step``) because ``inspect_rendered_slides_llm.py``
    # intentionally exits non-zero when it records QC-blocker findings (the
    # report content, not the exit code, is the verdict -- mirrors
    # ``run_topic_presentation_build.py``'s own ``stage_visual_qc`` try/except
    # convention). A genuine execution failure (report never written) still
    # blocks; a QC-content failure instead routes to the bounded repair loop.
    visual_llm_qc_output = deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
    visual_qc_command = [
        sys.executable,
        "scripts/presentations/inspect_rendered_slides_llm.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--screenshots-dir",
        str(screenshots_dir),
        "--report-output",
        str(visual_llm_qc_output),
    ]

    def _run_visual_qc() -> dict[str, Any] | None:
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=pc.stage_id,
            caller=pc.ctx.receipt_caller,
        ):
            visual_qc_result = pc.command_runner(visual_qc_command)
        if not visual_llm_qc_output.is_file() or visual_llm_qc_output.stat().st_size <= 0:
            return pc.block(
                blocker_code="presentation_visual_qc_outputs_missing",
                checks={
                    **common_checks,
                    "command": visual_qc_command,
                    "stdout": sanitize_for_receipt(visual_qc_result.stdout[-2000:]),
                    "stderr": sanitize_for_receipt(visual_qc_result.stderr[-2000:]),
                },
            )
        return None

    def _load_visual_qc() -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool, str | None]:
        report, error = load_json(visual_llm_qc_output)
        if error or not isinstance(report, dict):
            return None, [], False, error
        slides = report.get("slides") if isinstance(report.get("slides"), list) else []
        failing = [slide for slide in slides if isinstance(slide, dict) and slide.get("status") == "fail"]
        passed = report.get("status") in ("passed", "warnings") and not failing
        return report, failing, passed, None

    visual_qc_spec = step_spec(
        pc,
        command=visual_qc_command,
        outputs=(visual_llm_qc_output,),
        directory_patterns={screenshots_dir: "*.png"},
        reject_statuses=frozenset(),
    )
    visual_qc_skip, visual_qc_snapshot = step_gate(visual_qc_spec, resume=resume)
    if not visual_qc_skip:
        blocked = _run_visual_qc()
        if blocked is not None:
            return blocked
        step_record(visual_qc_spec, visual_qc_snapshot)
    visual_qc_report, qc_failing_slides, qc_passed, visual_qc_report_error = _load_visual_qc()
    if visual_qc_report is None:
        return pc.block(
            blocker_code="presentation_visual_qc_failed",
            checks={**common_checks, "error": visual_qc_report_error},
        )

    # Bounded pixel-judge repair loop (max REPAIR_MAX_ATTEMPTS
    # rounds, PAID). In the structured deck-contract pipeline the repair-loop
    # owner script can only apply conservative SVG repairs and then asks for
    # a fresh governed vision QC (``llm_qc_rerun_required``) -- it cannot
    # re-render or re-judge on its own -- so each round here: writes the
    # pixel-judge findings as per-slide redraw feedback, runs the owner
    # script (receipt = owner evidence), regenerates only the flagged
    # figures (accepted figures are reused from the per-figure cache),
    # re-renders/exports/screenshots, and re-runs the vision QC. A still-
    # failing QC after the bound routes to block_presentation_stage --
    # never an unbounded loop, never an auto-raised cost cap (T-10-04).
    # Mirrors ``run_topic_presentation_build.py``'s QC_RETRY_STAGES loop.
    qc_feedback_output = deck_dir / "qc_figure_feedback.json"
    repair_receipt_output = deck_dir / "repair_loop_receipt.json"
    qc_max_rounds = (
        REPAIR_MAX_ATTEMPTS
        if visual_qc_max_rounds is None
        else max(0, min(int(visual_qc_max_rounds), REPAIR_MAX_ATTEMPTS))
    )
    figure_regression_output = deck_dir / "figure_archetype_regression.json"
    if not drawsvg_route:
        # A compiled figure is a pure function of its spec: re-running the same
        # spec produces byte-identical SVG, so a redraw round could only burn
        # money to reproduce the same pixels. The pixel judge still RUNS -- its
        # findings are recorded as a regression signal against the archetype --
        # but it drives the compiler's backlog, not a retry.
        qc_max_rounds = 0
        write_figure_regression(
            visual_qc_report=visual_qc_report,
            render_spec_path=render_spec_output,
            deck_dir=deck_dir,
            output_path=figure_regression_output,
        )
    # The operator's review page: every rendered slide beside its vision
    # findings, the critic findings still open after three rounds, and any
    # outcome already recorded. Written here, at the canary stop, for every
    # conference; $0, deterministic.
    figure_review_page_output = deck_dir / "figure_review_page.html"
    blocked = owner_step(
        pc,
        resume=resume,
        command=[
            sys.executable,
            "scripts/presentations/build_figure_review_page.py",
            "--deck-dir",
            str(deck_dir),
            "--output",
            str(figure_review_page_output),
        ],
        paid=False,
        outputs=(figure_review_page_output,),
        failed_blocker_code="presentation_figure_review_page_failed",
        missing_blocker_code="presentation_figure_review_page_missing",
        extra_checks=common_checks,
        extra_inputs=(
            StepInput(screenshots_dir, pattern="*.png", optional=True),
            StepInput(visual_llm_qc_output, optional=True),
            # Named directly: the spec receipt variable exists on the compiler route only.
            StepInput(deck_dir / "figure_spec_receipt.json", optional=True),
            StepInput(deck_dir / "figure_specs", pattern="*.critic.json", optional=True),
        ),
    )
    if blocked is not None:
        return blocked

    qc_rounds = 0
    while not qc_passed and qc_rounds < qc_max_rounds:
        qc_rounds += 1
        feedback_count = write_qc_figure_feedback(
            visual_qc_report=visual_qc_report,
            render_spec_path=render_spec_output,
            output_path=qc_feedback_output,
        )
        repair_command = [
            sys.executable,
            "scripts/presentations/run_presentation_repair_loop.py",
            "--workspace",
            str(workspace),
            "--presentation-id",
            deck_id,
            "--presentation-type",
            presentation_type,
            "--topic",
            deck_id,
            # No ``slides.md`` exists in the structured deck-contract
            # pipeline (legacy Markdown production evidence is rejected by
            # this stage's own gate); the path is a required CLI argument
            # but is never written to disk, so the owner script's own
            # ``_figure_regeneration_inputs_ready`` check safely falls back
            # to its conservative SVG-repair path instead of the legacy
            # markdown-driven figure-regeneration route.
            "--slides-md",
            str(deck_dir / "slides.md"),
            "--generated-dir",
            str(generated_dir),
            "--html-output",
            str(html_output),
            "--slide-mapping",
            str(slide_mapping_output),
            "--validation-report",
            str(deck_dir / "figure_svg_validation_report.json"),
            "--visual-qc-report",
            str(inspect_report_output),
            "--screenshots-dir",
            str(screenshots_dir),
            "--receipt-output",
            str(repair_receipt_output),
            "--asset-inventory",
            str(asset_inventory_output),
            "--llm-qc-report",
            str(visual_llm_qc_output),
            "--max-attempts",
            str(REPAIR_MAX_ATTEMPTS),
        ]
        # The owner script exits non-zero when it records a blocked receipt
        # (structured pipeline: conservative repairs + QC rerun required);
        # the receipt, not the exit code, is the evidence -- only a missing
        # receipt is an execution failure here.
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=pc.stage_id,
            caller=pc.ctx.receipt_caller,
        ):
            repair_result = pc.command_runner(repair_command)
        if not repair_receipt_output.is_file() or repair_receipt_output.stat().st_size <= 0:
            return pc.block(
                blocker_code="presentation_repair_loop_outputs_missing",
                checks={
                    **common_checks,
                    "command": repair_command,
                    "stdout": sanitize_for_receipt(repair_result.stdout[-2000:]),
                    "stderr": sanitize_for_receipt(repair_result.stderr[-2000:]),
                },
            )
        retry_figures_command = list(figures_pass2_command)
        if feedback_count:
            retry_figures_command.extend(["--qc-feedback-json", str(qc_feedback_output)])
        # Redraw, then semantic-review the redraw. A redrawn figure that
        # dropped a required entity label fails the deterministic reviewer;
        # that finding is fed back and the flagged figures are redrawn once
        # more before the review is treated as a real blocker.
        for semantic_pass in (1, 2):
            blocked = None
            for figure_pass_attempt in range(1, FIGURE_PASS_RETRIES + 1):
                blocked = owner_step(
                    pc,
                    resume=False,
                    command=retry_figures_command,
                    paid=True,
                    outputs=(visual_plan_output, asset_inventory_output, model_receipt_output, code_review_output),
                    failed_blocker_code="presentation_figure_generation_failed",
                    missing_blocker_code="presentation_figure_generation_outputs_missing",
                    extra_checks={
                        **common_checks,
                        "qc_round": qc_rounds,
                        "semantic_pass": semantic_pass,
                        "figure_pass_attempt": figure_pass_attempt,
                    },
                )
                if blocked is None:
                    break
            if blocked is not None:
                return blocked
            figure_review_result = pc.command_runner(figure_review_command)
            semantic_review, semantic_review_error = load_json(figure_semantic_review_output)
            if semantic_review_error or not isinstance(semantic_review, dict):
                return pc.block(
                    blocker_code="presentation_figure_review_outputs_missing",
                    checks={
                        **common_checks,
                        "qc_round": qc_rounds,
                        "command": figure_review_command,
                        "stdout": sanitize_for_receipt(figure_review_result.stdout[-2000:]),
                        "stderr": sanitize_for_receipt(figure_review_result.stderr[-2000:]),
                        "error": semantic_review_error,
                    },
                )
            if semantic_review.get("status") == "pass":
                break
            if semantic_pass == 2:
                return pc.block(
                    blocker_code="presentation_figure_review_failed",
                    checks={
                        **common_checks,
                        "qc_round": qc_rounds,
                        "semantic_review_status": semantic_review.get("status"),
                        "semantic_findings": sanitize_for_receipt(
                            json.dumps(semantic_review.get("findings") or [])[:2000]
                        ),
                    },
                )
            merge_semantic_figure_feedback(
                semantic_review=semantic_review, output_path=qc_feedback_output
            )
            if "--qc-feedback-json" not in retry_figures_command:
                retry_figures_command.extend(["--qc-feedback-json", str(qc_feedback_output)])
        for retry_command, retry_paid, retry_outputs, retry_failed_code, retry_missing_code in (
            (
                svg_validation_command,
                False,
                (svg_validation_output,),
                "presentation_svg_validation_failed",
                "presentation_svg_validation_outputs_missing",
            ),
            (
                render_spec_command,
                False,
                (render_spec_output,),
                "presentation_render_spec_failed",
                "presentation_render_spec_outputs_missing",
            ),
            (
                bookend_command,
                False,
                (render_spec_output, deck_contract_output),
                "presentation_bookend_slides_failed",
                "presentation_bookend_slides_outputs_missing",
            ),
            (
                render_html_command,
                False,
                (html_output, slide_mapping_output),
                "presentation_render_html_failed",
                "presentation_render_html_outputs_missing",
            ),
            (
                export_pdf_command,
                False,
                (pdf_output,),
                "presentation_pdf_export_failed",
                "presentation_pdf_export_outputs_missing",
            ),
            (
                pptx_command,
                False,
                (pptx_export_output,),
                "presentation_pptx_export_failed",
                "presentation_pptx_export_outputs_missing",
            ),
            (
                screenshots_command,
                False,
                (inspect_report_output,),
                "presentation_screenshots_failed",
                "presentation_screenshots_outputs_missing",
            ),
        ):
            blocked = owner_step(
                pc,
                resume=False,
                command=retry_command,
                paid=retry_paid,
                outputs=retry_outputs,
                failed_blocker_code=retry_failed_code,
                missing_blocker_code=retry_missing_code,
                extra_checks={**common_checks, "qc_round": qc_rounds},
            )
            if blocked is not None:
                return blocked
        visual_qc_snapshot = (
            LINEAGE.get().snapshot_inputs(visual_qc_spec)
            if LINEAGE.get() is not None
            else None
        )
        blocked = _run_visual_qc()
        if blocked is not None:
            return blocked
        step_record(visual_qc_spec, visual_qc_snapshot)
        visual_qc_report, qc_failing_slides, qc_passed, visual_qc_report_error = _load_visual_qc()
        if visual_qc_report is None:
            return pc.block(
                blocker_code="presentation_visual_qc_failed",
                checks={**common_checks, "qc_round": qc_rounds, "error": visual_qc_report_error},
            )

    visual_qc_acceptance: dict[str, Any] | None = None
    visual_qc_acceptance_path: Path | None = None
    if not qc_passed:
        qc_outcome = {
            "qc_rounds": qc_rounds,
            "qc_max_rounds": qc_max_rounds,
            "visual_qc_status": visual_qc_report.get("status"),
            "qc_failing_slide_numbers": [slide.get("slide_number") for slide in qc_failing_slides],
        }
        if not accept_visual_qc_findings:
            return pc.block(
                blocker_code="presentation_repair_loop_failed",
                checks={**common_checks, **qc_outcome},
            )
        # Explicit OPERATOR acceptance (review canary): the deck proceeds to
        # delivery/video with the still-failing vision-QC findings recorded
        # verbatim on the deck record and in the owner evidence -- never
        # laundered into a pass. The gate itself is unchanged for any run
        # without this flag.
        visual_qc_acceptance = {
            **qc_outcome,
            "accepted_by": "operator:--accept-visual-qc-findings",
            "visual_qc_report": str(visual_llm_qc_output),
            "visual_qc_report_sha256": file_sha256(visual_llm_qc_output),
        }
        common_checks = {**common_checks, "visual_qc_accepted_with_findings": visual_qc_acceptance}
        owner_evidence_candidates["presentation.repair_loop"] = (
            f"visual_qc_accepted_with_findings:{visual_llm_qc_output}:{file_sha256(visual_llm_qc_output)}"
        )
        # The acceptance must reach the Gate A reviewers, not just the receipt.
        # Without it, both reviewers read the failed vision-QC report and
        # refused every deck on that basis alone (2026-09-15), which made the
        # operator's decision unrecordable anywhere that mattered. The record
        # carries the findings verbatim so a reviewer sees exactly what was
        # accepted -- and can still fail the deck for anything else.
        visual_qc_acceptance_path = deck_dir / "assets" / "generated" / "visual_qc_acceptance.json"
        acceptance_record = {
            "schema_version": 1,
            "run_id": run_id,
            "deck_id": deck_id,
            "accepted_at": utc_now(),
            **visual_qc_acceptance,
            "accepted_findings": [
                {
                    "slide_number": slide.get("slide_number"),
                    "findings": slide.get("findings") or slide.get("issues") or [],
                }
                for slide in qc_failing_slides
            ],
        }
        # The record is an input of the deck review: rewriting it with only a
        # new timestamp would make every resume re-run that paid review.
        previous_record, _previous_error = load_json(visual_qc_acceptance_path)
        unchanged = isinstance(previous_record, dict) and {
            key: value for key, value in previous_record.items() if key != "accepted_at"
        } == {key: value for key, value in acceptance_record.items() if key != "accepted_at"}
        if not unchanged:
            write_json_atomic(visual_qc_acceptance_path, acceptance_record)

    if qc_rounds:
        # The re-rendered artifacts replaced the first-pass ones; refresh
        # every evidence token so each references the bytes now on disk.
        owner_evidence_candidates["presentation.generate_programmatic_figures"] = (
            f"{visual_plan_output}:{file_sha256(visual_plan_output)}"
        )
        owner_evidence_candidates["presentation.review_programmatic_figures"] = (
            f"{figure_semantic_review_output}:{file_sha256(figure_semantic_review_output)}"
        )
        owner_evidence_candidates["presentation.validate_svgs"] = (
            f"{svg_validation_output}:{file_sha256(svg_validation_output)}"
        )
        owner_evidence_candidates["presentation.generate_render_spec"] = (
            f"{render_spec_output}:{file_sha256(render_spec_output)}"
        )
        owner_evidence_candidates["presentation.md_to_revealjs"] = (
            f"{html_output}:{file_sha256(html_output)}"
        )
        owner_evidence_candidates["presentation.export_pdf"] = (
            f"{pdf_output}:{file_sha256(pdf_output)}"
        )
        owner_evidence_candidates["presentation.md_to_pptx"] = (
            f"{pptx_export_output}:{file_sha256(pptx_export_output)}"
        )
        owner_evidence_candidates["presentation.inspect_rendered_slides"] = (
            f"{inspect_report_output}:{file_sha256(inspect_report_output)}"
        )
        owner_evidence_candidates["presentation.repair_loop"] = (
            f"{repair_receipt_output}:{file_sha256(repair_receipt_output)}"
        )
    elif visual_qc_acceptance is None:
        # ``presentation.repair_loop`` is a required (though coverage-exempt)
        # script id, but the repair loop is genuinely CONDITIONAL -- it only
        # runs when visual QC finds a blocker. A clean-QC deck must still
        # record a real, non-fabricated evidence token (Rule 1: this required
        # id would otherwise silently fail strict_execution for any run where
        # every deck passes QC on the first try) -- reference the real,
        # just-verified visual-QC report that made repair unnecessary, never
        # a bare id echo or an empty string.
        owner_evidence_candidates["presentation.repair_loop"] = (
            f"repair_not_required:{visual_llm_qc_output}:{file_sha256(visual_llm_qc_output)}"
        )

    owner_evidence_candidates["presentation.inspect_rendered_slides_llm"] = (
        f"{visual_llm_qc_output}:{file_sha256(visual_llm_qc_output)}"
    )

    # Advisory second-opinion visual QC (only when the conference configures one). It is
    # NOT owner evidence and never blocks; its summary reaches Gate A as hints.
    advisory_summary = run_advisory_visual_qc(
        pc,
        workspace=workspace,
        run_id=run_id,
        deck_id=deck_id,
        presentation_type=presentation_type,
        deck_dir=deck_dir,
        screenshots_dir=screenshots_dir,
        primary_report=visual_llm_qc_output,
    )
    if advisory_summary is not None:
        common_checks = {**common_checks, "visual_qc_advisory": {
            key: advisory_summary.get(key)
            for key in ("status", "provider", "model", "flagged_slide_count", "flagged_only_by_advisory", "error")
            if advisory_summary.get(key) is not None
        }}

    # Step 11 (Plan 10-03 Task 3): artifact-truth validation -- the 5th and
    # FINAL Critical-Finding-2 script, running LATE (after rendering/QC/
    # repair), never to be confused with the 4 EARLY CF2 scripts from Plan
    # 10-02 (source_corpus_inventory/speaker_author_work_ledger/
    # repair_presentation_sources/research_cluster_plan). $0, deterministic
    # -- no model_governance/anthropic/openai import; not wrapped in
    # provider_receipt_context. Checks are restricted to the deck/figures
    # sub-stage's own evidence (html/pdf/frames/source/receipts) --
    # audio/mp4 checks are excluded because delivery/video (Plan 10-04)
    # has not run yet for this substage.
    artifact_truth_output = deck_dir / "artifact_truth_report.json"
    # A report that did not reach ``pass`` is never resume-skipped; a
    # re-render above (bookends, redraw rounds) invalidates the prior report
    # through the ledger, since the html/pdf/frames it judged are its inputs.
    existing_artifact_truth, _existing_artifact_truth_error = load_json(artifact_truth_output)
    if artifact_truth_output.is_file() and (
        not isinstance(existing_artifact_truth, dict)
        or existing_artifact_truth.get("status") != "pass"
    ):
        artifact_truth_output.unlink()
    artifact_truth_command = [
        sys.executable,
        "scripts/presentations/validate_artifact_truth.py",
        "--deck-dir",
        str(deck_dir),
        "--report-out",
        str(artifact_truth_output),
        "--checks",
        "html,pdf,frames,source,receipts",
        "--summary",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=artifact_truth_command,
        paid=False,
        outputs=(artifact_truth_output,),
        failed_blocker_code="presentation_artifact_truth_failed",
        missing_blocker_code="presentation_artifact_truth_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(
            html_output,
            pdf_output,
            deck_dir / "slide_frame_manifest.json",
            StepInput(deck_dir / "rendered_frames", pattern="*.png"),
            render_spec_output,
            deck_contract_output,
            source_binding_output,
            render_export_receipt_output,
        ),
        accept_statuses=frozenset({"pass"}),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.artifact_truth_validation"] = (
        f"{artifact_truth_output}:{file_sha256(artifact_truth_output)}"
    )

    if not run_delivery_video:
        deck_figures_only_record = (
            deck_figures_record(
                deck_id=deck_id,
                kind=kind,
                matched_papers_path=matched_papers_path,
                corpus_inventory_output=corpus_inventory_output,
                work_ledger_output=work_ledger_output,
                source_repair_output=source_repair_output,
                cluster_plan_output=cluster_plan_output,
                blueprint_output=blueprint_output,
                blueprint_review_output=blueprint_review_output,
                deck_contract_output=deck_contract_output,
                source_binding_output=source_binding_output,
                visual_plan_output=visual_plan_output,
                asset_inventory_output=asset_inventory_output,
                figure_semantic_review_output=figure_semantic_review_output,
                svg_validation_output=svg_validation_output,
                render_spec_output=render_spec_output,
                html_output=html_output,
                pdf_output=pdf_output,
                pptx_export_output=pptx_export_output,
                artifact_truth_output=artifact_truth_output,
                inspect_report_output=inspect_report_output,
                visual_llm_qc_output=visual_llm_qc_output,
            )
        )
        if visual_qc_acceptance is not None:
            deck_figures_only_record["visual_qc_accepted_with_findings"] = dict(visual_qc_acceptance)
        deck_records.append(deck_figures_only_record)
        return None

    # ------------------------------------------------------------------
    # Plan 10-04 Task 1: delivery-contract generation ($0) + both-mode
    # narration lint/review/fix, running BEFORE any TTS/video step
    # (must_haves.truths -- the locked slide-sync policy).
    # ------------------------------------------------------------------
    narration_deck_kind = "topic" if kind == "topic" else "executive"

    delivery_single_speaker_output = deck_dir / "delivery_single_speaker.json"
    delivery_notebooklm_output = deck_dir / "delivery_notebooklm.json"
    presenter_script_md_output = deck_dir / "6_presenter_script.md"
    presenter_script_notebooklm_md_output = deck_dir / "6_presenter_script_notebooklm_2host.md"
    presenter_script_command = [
        sys.executable,
        "scripts/presentations/generate_presenter_script.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--render-spec",
        str(render_spec_output),
        "--slide-mapping",
        str(slide_mapping_output),
        "--delivery-single-speaker",
        str(delivery_single_speaker_output),
        "--delivery-notebooklm",
        str(delivery_notebooklm_output),
        "--debug-script-md",
        str(presenter_script_md_output),
        "--debug-notebooklm-script-md",
        str(presenter_script_notebooklm_md_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=presenter_script_command,
        paid=False,
        outputs=(
            delivery_single_speaker_output,
            delivery_notebooklm_output,
            presenter_script_md_output,
            presenter_script_notebooklm_md_output,
        ),
        failed_blocker_code="presentation_delivery_generation_failed",
        missing_blocker_code="presentation_delivery_generation_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.presenter_script"] = (
        f"{presenter_script_md_output}:{file_sha256(presenter_script_md_output)}"
    )

    # ------------------------------------------------------------------
    # Deck acceptance -- the multi-model produced-deck review, run BEFORE
    # any TTS or video rendering.
    #
    # This gate used to sit at the very END of the delivery/video substage,
    # after both videos were rendered. On the first full CVPR 2026 run that
    # cost 24 videos and two and a half hours: the review then failed all
    # twelve decks on figure defects its reviewers had confirmed in the
    # SCREENSHOTS, and the figure fix that followed invalidated every video.
    # It reads the blueprint, the deck contract, the render spec, the source
    # binding, the visual-QC report, the screenshots and the narration
    # script -- all of which exist here -- so nothing expensive needs to
    # have been built for it to refuse a deck.
    # ------------------------------------------------------------------

    # Engaging narration (PAID, Claude-authored two-host + single-speaker
    # drafts replacing the mechanical templated script) -> rebuild both
    # delivery contracts to carry it ($0) -> re-stamp cross-referenced
    # contract hashes ($0). Mirrors ``run_topic_presentation_build.py``'s
    # engaging_narration / rebuild_delivery / reconcile_hashes stages; the
    # two $0 steps rewrite files in place (``in_place`` ledger steps).
    # ``engaging_narration_raw.json`` is the narration source of truth the
    # delivery rebuild reads; it and the two markdown drafts are operator-
    # edit surfaces: a patched file is adopted, never regenerated, and the
    # rebuild/reconcile/video steps that consumed the old text re-run.
    engaging_single_md_output = deck_dir / "6_presenter_script.ENGAGING.md"
    engaging_notebooklm_md_output = deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md"
    engaging_raw_output = deck_dir / "engaging_narration_raw.json"
    engaging_narration_command = [
        sys.executable,
        "scripts/presentations/generate_engaging_narration.py",
        "--deck-dir",
        str(deck_dir),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=engaging_narration_command,
        paid=True,
        outputs=(engaging_single_md_output, engaging_notebooklm_md_output, engaging_raw_output),
        failed_blocker_code="presentation_engaging_narration_failed",
        missing_blocker_code="presentation_engaging_narration_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(
            presenter_script_md_output,
            presenter_script_notebooklm_md_output,
            render_spec_output,
            blueprint_output,
        ),
        editable_outputs=(engaging_single_md_output, engaging_notebooklm_md_output, engaging_raw_output),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.engaging_narration"] = (
        f"{engaging_notebooklm_md_output}:{file_sha256(engaging_notebooklm_md_output)}"
    )

    rebuild_delivery_command = [
        sys.executable,
        "scripts/presentations/rebuild_delivery_engaging.py",
        "--deck-dir",
        str(deck_dir),
    ]
    reconcile_hashes_command = [
        sys.executable,
        "scripts/presentations/reconcile_contract_hashes.py",
        "--deck-dir",
        str(deck_dir),
    ]

    def _rebuild_delivery_and_reconcile(pass_label: str) -> dict[str, Any] | None:
        # Both steps rewrite files in place; the ledger records each pass
        # under its own step id so the pre- and post-narration-gate rebuilds
        # keep separate histories.
        rebuild_steps = []
        if pass_label == "pre-gates":
            # The ENGAGING markdown the fixers edit must start as a rendering
            # of the raw JSON; otherwise the post-gates sync would copy stale
            # markdown over an operator's direct edit to the raw narration.
            rebuild_steps.append(
                (
                    [
                        sys.executable,
                        "scripts/presentations/generate_engaging_narration.py",
                        "--deck-dir",
                        str(deck_dir),
                        "--render-from-raw",
                    ],
                    (engaging_notebooklm_md_output, engaging_single_md_output),
                    (engaging_raw_output,),
                    "presentation_narration_md_render_failed",
                    "presentation_narration_md_render_outputs_missing",
                )
            )
        if pass_label == "post-gates":
            # The narration fixers edit the ENGAGING markdown in place, but the
            # delivery (and TTS) is rebuilt from engaging_narration_raw.json:
            # carry their text fixes into the raw JSON first, or they are
            # reviewed and then discarded ($0, refuses structural edits).
            rebuild_steps.append(
                (
                    [
                        sys.executable,
                        "scripts/presentations/generate_engaging_narration.py",
                        "--deck-dir",
                        str(deck_dir),
                        "--sync-raw-from-md",
                    ],
                    (engaging_raw_output,),
                    (engaging_notebooklm_md_output, engaging_single_md_output),
                    "presentation_narration_raw_sync_failed",
                    "presentation_narration_raw_sync_outputs_missing",
                )
            )
        for rebuild_command, rebuild_outputs, rebuild_inputs, rebuild_failed_code, rebuild_missing_code in (
            *rebuild_steps,
            (
                rebuild_delivery_command,
                (delivery_single_speaker_output, delivery_notebooklm_output),
                (engaging_raw_output, render_spec_output, engaging_notebooklm_md_output, engaging_single_md_output),
                "presentation_delivery_rebuild_failed",
                "presentation_delivery_rebuild_outputs_missing",
            ),
            (
                reconcile_hashes_command,
                (render_spec_output, delivery_single_speaker_output, delivery_notebooklm_output),
                (deck_contract_output, source_binding_output, visual_plan_output),
                "presentation_contract_hash_reconcile_failed",
                "presentation_contract_hash_reconcile_outputs_missing",
            ),
        ):
            rebuild_blocked = owner_step(
                pc,
                resume=resume,
                command=rebuild_command,
                paid=False,
                outputs=rebuild_outputs,
                failed_blocker_code=rebuild_failed_code,
                missing_blocker_code=rebuild_missing_code,
                extra_checks=common_checks,
                step_id=f"{Path(rebuild_command[1]).stem}#{pass_label}",
                in_place=True,
                extra_inputs=tuple(rebuild_outputs) + tuple(rebuild_inputs),
            )
            if rebuild_blocked is not None:
                return rebuild_blocked
        owner_evidence_candidates["presentation.rebuild_delivery_engaging"] = (
            f"{delivery_notebooklm_output}:{file_sha256(delivery_notebooklm_output)}"
        )
        owner_evidence_candidates["presentation.reconcile_contract_hashes"] = (
            f"{render_spec_output}:{file_sha256(render_spec_output)}"
        )
        return None

    blocked = _rebuild_delivery_and_reconcile("pre-gates")
    if blocked is not None:
        return blocked

    # Single-speaker narration: lint ($0) -> review (PAID) -> fix (PAID).
    # ``review``/``fix`` are content-gated (a non-zero exit means a review
    # finding, not a crash) -- special-cased via
    # ``_presentation_narration_content_step`` mirroring Plan 10-03's
    # ``inspect_rendered_slides_llm.py`` precedent. Neither
    # ``lint_presenter_narration``/``review_presenter_narration``/
    # ``fix_presenter_narration`` has its own required_script_ids entry
    # (single-speaker narration has no standalone registry id), so no
    # owner-evidence key is recorded for these three calls.
    presenter_lint_output = deck_dir / "presenter_narration_lint_report.json"
    presenter_lint_command = [
        sys.executable,
        "scripts/presentations/lint_presenter_narration.py",
        "--script-md",
        str(presenter_script_md_output),
        "--deck-kind",
        narration_deck_kind,
        "--topic-id",
        deck_id,
        "--report-out",
        str(presenter_lint_output),
    ]
    blocked = narration_content_step(
        pc,
        resume=resume,
        command=presenter_lint_command,
        paid=False,
        report_path=presenter_lint_output,
        failed_blocker_code="presentation_narration_lint_failed",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked

    presenter_review_json_output = deck_dir / "presenter_narration_review.json"
    presenter_review_md_output = deck_dir / "presenter_narration_review.md"
    presenter_review_command = [
        sys.executable,
        "scripts/presentations/review_presenter_narration.py",
        "--script-md",
        str(presenter_script_md_output),
        "--lint-report",
        str(presenter_lint_output),
        "--deck-kind",
        narration_deck_kind,
        "--topic-id",
        deck_id,
        "--review-out",
        str(presenter_review_md_output),
        "--json-out",
        str(presenter_review_json_output),
    ]
    blocked = narration_content_step(
        pc,
        resume=resume,
        command=presenter_review_command,
        paid=True,
        report_path=presenter_review_json_output,
        failed_blocker_code="presentation_narration_review_failed",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked

    presenter_fix_output = deck_dir / "presenter_narration_fix_report.json"
    presenter_fix_command = [
        sys.executable,
        "scripts/presentations/fix_presenter_narration.py",
        "--script-md",
        str(presenter_script_md_output),
        "--report-out",
        str(presenter_fix_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=presenter_fix_command,
        paid=True,
        outputs=(presenter_fix_output, presenter_script_md_output),
        failed_blocker_code="presentation_narration_fix_failed",
        missing_blocker_code="presentation_narration_fix_outputs_missing",
        extra_checks=common_checks,
        in_place=True,
        extra_inputs=(presenter_review_json_output,),
    )
    if blocked is not None:
        return blocked

    # NotebookLM narration: lint ($0) -> review (PAID) -> fix (PAID) ->
    # series-review (PAID) -> series-fix (PAID). ``review_narration_series.py``/
    # ``fix_narration_series.py`` support either a whole-``--decks-dir`` pass
    # or repeated ``--script-md`` entries; this plan follows the literal
    # per-deck ordering (a single ``--script-md`` entry for this deck), not a
    # separate once-only cross-deck pass -- see SUMMARY.md for the documented
    # forward-risk this implies for ``approve_for_tts`` in a live run.
    narration_lint_output = deck_dir / "narration_lint_report.json"
    narration_lint_command = [
        sys.executable,
        "scripts/presentations/lint_narration_script.py",
        "--script-md",
        str(engaging_notebooklm_md_output),
        "--deck-kind",
        narration_deck_kind,
        "--topic-id",
        deck_id,
        "--report-out",
        str(narration_lint_output),
    ]
    blocked = narration_content_step(
        pc,
        resume=resume,
        command=narration_lint_command,
        paid=False,
        report_path=narration_lint_output,
        failed_blocker_code="presentation_narration_lint_failed",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked

    narration_review_json_output = deck_dir / "narration_review.json"
    narration_review_md_output = deck_dir / "narration_review.md"
    narration_review_command = [
        sys.executable,
        "scripts/presentations/review_narration_script.py",
        "--script-md",
        str(engaging_notebooklm_md_output),
        "--lint-report",
        str(narration_lint_output),
        "--deck-kind",
        narration_deck_kind,
        "--topic-id",
        deck_id,
        "--review-out",
        str(narration_review_md_output),
        "--json-out",
        str(narration_review_json_output),
    ]
    blocked = narration_content_step(
        pc,
        resume=resume,
        command=narration_review_command,
        paid=True,
        report_path=narration_review_json_output,
        failed_blocker_code="presentation_narration_review_failed",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.narration_script_review"] = (
        f"{narration_review_json_output}:{file_sha256(narration_review_json_output)}"
    )

    narration_script_fix_output = deck_dir / "narration_script_fix_report.json"
    narration_script_fix_command = [
        sys.executable,
        "scripts/presentations/fix_narration_script.py",
        "--script-md",
        str(engaging_notebooklm_md_output),
        "--deck-kind",
        narration_deck_kind,
        "--report-out",
        str(narration_script_fix_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=narration_script_fix_command,
        paid=True,
        outputs=(narration_script_fix_output, engaging_notebooklm_md_output),
        failed_blocker_code="presentation_narration_fix_failed",
        missing_blocker_code="presentation_narration_fix_outputs_missing",
        extra_checks=common_checks,
        in_place=True,
        extra_inputs=(narration_review_json_output,),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.narration_script_fix"] = (
        f"{narration_script_fix_output}:{file_sha256(narration_script_fix_output)}"
    )

    narration_series_review_output = deck_dir / "narration_series_review.json"
    narration_series_review_md_output = deck_dir / "narration_series_review.md"
    narration_series_review_command = [
        sys.executable,
        "scripts/presentations/review_narration_series.py",
        "--script-md",
        str(engaging_notebooklm_md_output),
        "--report-out",
        str(narration_series_review_output),
        "--markdown-out",
        str(narration_series_review_md_output),
    ]
    blocked = narration_content_step(
        pc,
        resume=resume,
        command=narration_series_review_command,
        paid=True,
        report_path=narration_series_review_output,
        failed_blocker_code="presentation_narration_series_review_failed",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.narration_series_review"] = (
        f"{narration_series_review_output}:{file_sha256(narration_series_review_output)}"
    )

    narration_series_fix_output = deck_dir / "narration_series_fix_report.json"
    narration_series_fix_command = [
        sys.executable,
        "scripts/presentations/fix_narration_series.py",
        "--script-md",
        str(engaging_notebooklm_md_output),
        "--report-out",
        str(narration_series_fix_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=narration_series_fix_command,
        paid=True,
        outputs=(narration_series_fix_output, engaging_notebooklm_md_output),
        failed_blocker_code="presentation_narration_series_fix_failed",
        missing_blocker_code="presentation_narration_series_fix_outputs_missing",
        extra_checks=common_checks,
        in_place=True,
        extra_inputs=(narration_series_review_output,),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.narration_series_fix"] = (
        f"{narration_series_fix_output}:{file_sha256(narration_series_fix_output)}"
    )

    # ------------------------------------------------------------------
    # Plan 10-04 Task 2: both videos (PAID) -> video-sync validation ($0)
    # -> the 3-model whole-presentation review (PAID).
    # ------------------------------------------------------------------
    # The narration fixers above edit the engaging script in place; the
    # post-gates pass syncs those edits into engaging_narration_raw.json, then
    # rebuilds the delivery contracts (and re-stamps hashes) so TTS
    # synthesizes the fixed narration.
    blocked = _rebuild_delivery_and_reconcile("post-gates")
    if blocked is not None:
        return blocked

    if figure_route == FIGURE_ROUTE_COMPILER:
        blocked = _claim_fidelity_step("narration")
        if blocked is not None:
            return blocked

    # ------------------------------------------------------------------
    # Gate A: the whole-presentation review (PAID, Claude + Gemini; Codex
    # when available). It runs HERE -- after the engaging rewrite, the
    # narration lints and fixes, and the delivery rebuild, and before any
    # TTS or video -- so the reviewers judge the narration that will be
    # spoken. Placed before the engaging rewrite (2026-09-12..15) it handed
    # them the mechanical draft beside a delivery contract carrying the
    # engaging text, and a reviewer failed a deck on the contradiction.
    # ------------------------------------------------------------------
    review_narration_script = (
        engaging_single_md_output
        if engaging_single_md_output.is_file()
        else presenter_script_md_output
    )
    presentation_review_output = deck_dir / "presentation_review.json"
    presentation_review_md_output = deck_dir / "presentation_review.md"
    presentation_review_prompt_output = deck_dir / "presentation_review.prompt.md"
    contact_sheet_path = deck_dir / "render_contact_sheet.png"
    presentation_review_command = [
        sys.executable,
        "scripts/presentations/review_presentation.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--requirements",
        str(blueprint_output),
        "--narration-script",
        str(review_narration_script),
        "--screenshots-dir",
        str(screenshots_dir),
        "--blueprint",
        str(blueprint_output),
        "--blueprint-review",
        str(blueprint_review_output),
        "--deck-contract",
        str(deck_contract_output),
        "--render-spec",
        str(render_spec_output),
        "--source-binding",
        str(source_binding_output),
        "--delivery-contract",
        str(delivery_notebooklm_output),
        "--visual-qc-report",
        str(visual_llm_qc_output),
        "--report-output",
        str(presentation_review_output),
        "--markdown-output",
        str(presentation_review_md_output),
        "--prompt-output",
        str(presentation_review_prompt_output),
    ]
    if contact_sheet_path.is_file():
        presentation_review_command.extend(["--contact-sheet", str(contact_sheet_path)])
    advisory_summary_path = deck_dir / ADVISORY_SUMMARY
    if advisory_summary_path.is_file():
        presentation_review_command.extend(["--visual-qc-advisory", str(advisory_summary_path)])
    claim_fidelity_report_paths = tuple(
        deck_dir / f"claim_fidelity_review_{phase}.json" for phase in ("figures", "narration")
    )
    for report_path in claim_fidelity_report_paths:
        if report_path.is_file():
            presentation_review_command.extend(["--claim-fidelity-report", str(report_path)])
    if visual_qc_acceptance_path is not None and visual_qc_acceptance_path.is_file():
        presentation_review_command.extend(
            ["--visual-qc-acceptance", str(visual_qc_acceptance_path)]
        )
    # Explicit OPERATOR acceptance of a failed deck review: the deck proceeds
    # to video without another review call, with the review and its blocking
    # findings recorded verbatim -- never relabelled as a pass. Only for the
    # decks named on the command line.
    if deck_id in accepted_review_deck_ids:
        undecided = undecided_figure_findings(deck_dir)
        if undecided:
            # Figure fidelity policy, item 9: operator acceptance needs every
            # critic finding decided first (patch_figure_spec.py).
            return pc.block(
                blocker_code="presentation_figure_findings_undecided",
                checks={**common_checks, "undecided_figure_findings": undecided},
            )
    review_acceptance = accepted_review(
        presentation_review_output, deck_id=deck_id, accepted_deck_ids=accepted_review_deck_ids
    )
    if review_acceptance is not None:
        common_checks = {**common_checks, "presentation_review_accepted_with_findings": review_acceptance}
        owner_evidence_candidates["presentation.presentation_review"] = (
            f"accepted_with_findings:{presentation_review_output}:{review_acceptance['report_sha256']}"
        )
    else:
        # A review that FAILED is a verdict to be re-evaluated, not an output
        # to be reused. Set it aside (kept, never deleted) so the step runs
        # again against the current deck; the ledger additionally reuses a
        # passed verdict only while every artifact it judged still carries
        # the digest it recorded.
        supersede_failed_review(presentation_review_output)
    blocked = None if review_acceptance is not None else owner_step(
        pc,
        resume=resume,
        command=presentation_review_command,
        paid=True,
        outputs=(presentation_review_output,),
        failed_blocker_code="presentation_review_failed",
        missing_blocker_code="presentation_review_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(
            StepInput(contact_sheet_path, optional=True),
            StepInput(deck_dir / "assets" / "generated" / "visual_qc_acceptance.json", optional=True),
            StepInput(advisory_summary_path, optional=True),
            *(StepInput(path, optional=True) for path in claim_fidelity_report_paths),
        ),
        directory_patterns={screenshots_dir: "*.png"},
        accept_statuses=frozenset({"passed", "warnings"}),
    )
    if blocked is not None:
        return blocked
    if review_acceptance is None:
        owner_evidence_candidates["presentation.presentation_review"] = (
            f"{presentation_review_output}:{file_sha256(presentation_review_output)}"
        )

    # Advisory editorial gate ($0) before any TTS. Narration-only mode: the
    # compiled deck has no legacy slide markdown or storyboard, so only the
    # narration rules apply (duration estimate against the 40-minute budget,
    # zoom-beat density, figure entities named in narration). The report is
    # recorded on the deck record and never blocks: the budget call stays with
    # the operator, who reads it before paying for audio.
    editorial_gate_output = deck_dir / "presentation_editorial_gate_report.json"
    editorial_notebooklm_script = (
        engaging_notebooklm_md_output
        if engaging_notebooklm_md_output.is_file()
        else presenter_script_notebooklm_md_output
    )
    editorial_gate_command = [
        sys.executable,
        "scripts/presentations/presentation_editorial_gate.py",
        "--deck-dir",
        str(deck_dir),
        "--narration-only",
        "--advisory",
        "--deck-kind",
        narration_deck_kind,
        "--render-spec",
        str(render_spec_output),
        "--presenter-script-md",
        str(review_narration_script),
        "--notebooklm-script-md",
        str(editorial_notebooklm_script),
        "--asset-inventory",
        str(asset_inventory_output),
        "--report-out",
        str(editorial_gate_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=editorial_gate_command,
        paid=False,
        outputs=(editorial_gate_output,),
        failed_blocker_code="presentation_editorial_gate_failed",
        missing_blocker_code="presentation_editorial_gate_outputs_missing",
        extra_checks=common_checks,
        reject_statuses=frozenset(),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.editorial_gate"] = (
        f"{editorial_gate_output}:{file_sha256(editorial_gate_output)}"
    )

    video_build_dir = deck_dir / "video_build"
    presentation_video_output = deck_dir / "presentation_video.mp4"
    build_video_command = [
        sys.executable,
        "scripts/presentations/build_presenter_video.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--slides-html",
        str(html_output),
        "--slide-mapping",
        str(slide_mapping_output),
        "--delivery-contract",
        str(delivery_single_speaker_output),
        "--blueprint",
        str(blueprint_output),
        "--blueprint-review",
        str(blueprint_review_output),
        "--build-dir",
        str(video_build_dir),
        "--output",
        str(presentation_video_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=build_video_command,
        paid=True,
        outputs=(presentation_video_output,),
        failed_blocker_code="presentation_video_generation_failed",
        missing_blocker_code="presentation_video_generation_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.build_presenter_video"] = (
        f"{presentation_video_output}:{file_sha256(presentation_video_output)}"
    )

    video_build_notebooklm_dir = deck_dir / "video_build_notebooklm"
    presentation_video_notebooklm_output = deck_dir / "presentation_video_notebooklm.mp4"
    build_notebooklm_video_command = [
        sys.executable,
        "scripts/presentations/build_notebooklm_style_presenter_video.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--slides-html",
        str(html_output),
        "--slide-mapping",
        str(slide_mapping_output),
        "--delivery-contract",
        str(delivery_notebooklm_output),
        "--blueprint",
        str(blueprint_output),
        "--blueprint-review",
        str(blueprint_review_output),
        "--build-dir",
        str(video_build_notebooklm_dir),
        "--output",
        str(presentation_video_notebooklm_output),
        "--no-dialogue-fallback",
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=build_notebooklm_video_command,
        paid=True,
        outputs=(presentation_video_notebooklm_output,),
        failed_blocker_code="presentation_notebooklm_video_failed",
        missing_blocker_code="presentation_notebooklm_video_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.build_notebooklm_video"] = (
        f"{presentation_video_notebooklm_output}:{file_sha256(presentation_video_notebooklm_output)}"
    )

    video_sync_single_output = deck_dir / "video_sync_validation_single_speaker.json"
    video_sync_notebooklm_output = deck_dir / "video_sync_validation_notebooklm.json"
    video_sync_command = [
        sys.executable,
        "scripts/presentations/validate_video_sync.py",
        "--workspace",
        str(workspace),
        "--presentation-id",
        deck_id,
        "--presentation-type",
        presentation_type,
        "--topic",
        deck_id,
        "--render-spec",
        str(render_spec_output),
        "--delivery-single-speaker",
        str(delivery_single_speaker_output),
        "--delivery-notebooklm",
        str(delivery_notebooklm_output),
        "--single-speaker-video",
        str(presentation_video_output),
        "--notebooklm-video",
        str(presentation_video_notebooklm_output),
        "--single-speaker-report",
        str(video_sync_single_output),
        "--notebooklm-report",
        str(video_sync_notebooklm_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=video_sync_command,
        paid=False,
        outputs=(video_sync_single_output, video_sync_notebooklm_output),
        failed_blocker_code="presentation_video_sync_failed",
        missing_blocker_code="presentation_video_sync_outputs_missing",
        extra_checks=common_checks,
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.video_sync_validation"] = (
        f"{video_sync_single_output}:{file_sha256(video_sync_single_output)}"
    )

    # Path-based delivery checks ($0) on both final MP4s: duration budget,
    # audio/video stream alignment, agreement with the build receipts, and no
    # black tail. These were ICML's validation-stage checks; the shared chain
    # runs them here so every conference's videos get them without a
    # presentation manifest.
    video_delivery_output = deck_dir / "video_delivery_validation.json"
    video_delivery_command = [
        sys.executable,
        "scripts/presentations/validate_video_delivery.py",
        "--deck-dir",
        str(deck_dir),
        "--presentation-id",
        deck_id,
        "--video",
        str(presentation_video_output),
        "--video",
        str(presentation_video_notebooklm_output),
        "--report-out",
        str(video_delivery_output),
    ]
    blocked = owner_step(
        pc,
        resume=resume,
        command=video_delivery_command,
        paid=False,
        outputs=(video_delivery_output,),
        failed_blocker_code="presentation_video_delivery_failed",
        missing_blocker_code="presentation_video_delivery_outputs_missing",
        extra_checks=common_checks,
        extra_inputs=(
            StepInput(video_build_dir / "narration_track.m4a", optional=True),
            StepInput(video_build_notebooklm_dir / "narration_track.m4a", optional=True),
        ),
    )
    if blocked is not None:
        return blocked
    owner_evidence_candidates["presentation.video_delivery_validation"] = (
        f"{video_delivery_output}:{file_sha256(video_delivery_output)}"
    )


    deck_video_record = deck_figures_record(
        deck_id=deck_id,
        kind=kind,
        matched_papers_path=matched_papers_path,
        corpus_inventory_output=corpus_inventory_output,
        work_ledger_output=work_ledger_output,
        source_repair_output=source_repair_output,
        cluster_plan_output=cluster_plan_output,
        blueprint_output=blueprint_output,
        blueprint_review_output=blueprint_review_output,
        deck_contract_output=deck_contract_output,
        source_binding_output=source_binding_output,
        visual_plan_output=visual_plan_output,
        asset_inventory_output=asset_inventory_output,
        figure_semantic_review_output=figure_semantic_review_output,
        svg_validation_output=svg_validation_output,
        render_spec_output=render_spec_output,
        html_output=html_output,
        pdf_output=pdf_output,
        pptx_export_output=pptx_export_output,
        artifact_truth_output=artifact_truth_output,
        inspect_report_output=inspect_report_output,
        visual_llm_qc_output=visual_llm_qc_output,
        delivery_single_speaker_output=delivery_single_speaker_output,
        delivery_notebooklm_output=delivery_notebooklm_output,
        presenter_script_md_output=presenter_script_md_output,
        presentation_video_output=presentation_video_output,
        presentation_video_notebooklm_output=presentation_video_notebooklm_output,
        video_sync_single_output=video_sync_single_output,
        video_sync_notebooklm_output=video_sync_notebooklm_output,
        video_delivery_output=video_delivery_output,
        editorial_gate_output=editorial_gate_output,
        presentation_review_output=presentation_review_output,
    )
    if visual_qc_acceptance is not None:
        deck_video_record["visual_qc_accepted_with_findings"] = dict(visual_qc_acceptance)
    deck_records.append(deck_video_record)
    return None



def deck_figures_record(
    *,
    deck_id: str,
    kind: str,
    matched_papers_path: Path,
    corpus_inventory_output: Path,
    work_ledger_output: Path,
    source_repair_output: Path,
    cluster_plan_output: Path,
    blueprint_output: Path,
    blueprint_review_output: Path,
    deck_contract_output: Path,
    source_binding_output: Path,
    visual_plan_output: Path,
    asset_inventory_output: Path,
    figure_semantic_review_output: Path,
    svg_validation_output: Path,
    render_spec_output: Path,
    html_output: Path,
    pdf_output: Path,
    pptx_export_output: Path,
    artifact_truth_output: Path,
    inspect_report_output: Path,
    visual_llm_qc_output: Path,
    delivery_single_speaker_output: Path | None = None,
    delivery_notebooklm_output: Path | None = None,
    presenter_script_md_output: Path | None = None,
    presentation_video_output: Path | None = None,
    presentation_video_notebooklm_output: Path | None = None,
    video_sync_single_output: Path | None = None,
    video_sync_notebooklm_output: Path | None = None,
    video_delivery_output: Path | None = None,
    editorial_gate_output: Path | None = None,
    presentation_review_output: Path | None = None,
) -> dict[str, Any]:
    """Build one ``deck_records`` entry -- the deck/figures (sub-stage A)
    path set is always present; the delivery/video (sub-stage B, Plan 10-04)
    path set is only present when ``run_delivery_video`` was ``True`` for
    this deck (``None`` fields are omitted rather than recorded as literal
    ``"None"`` strings)."""

    record: dict[str, Any] = {
        "deck_id": deck_id,
        "deck_kind": kind,
        "matched_papers_path": str(matched_papers_path),
        "source_corpus_inventory_path": str(corpus_inventory_output),
        "speaker_author_work_ledger_path": str(work_ledger_output),
        "source_repair_findings_path": str(source_repair_output),
        "research_cluster_plan_path": str(cluster_plan_output),
        "presentation_blueprint_path": str(blueprint_output),
        "blueprint_review_path": str(blueprint_review_output),
        "presentation_deck_path": str(deck_contract_output),
        "source_binding_report_path": str(source_binding_output),
        "visual_plan_path": str(visual_plan_output),
        "asset_inventory_path": str(asset_inventory_output),
        "figure_semantic_review_path": str(figure_semantic_review_output),
        "figure_svg_validation_path": str(svg_validation_output),
        "render_spec_path": str(render_spec_output),
        "presentation_html_path": str(html_output),
        "presentation_pdf_path": str(pdf_output),
        "presentation_pptx_export_path": str(pptx_export_output),
        "artifact_truth_report_path": str(artifact_truth_output),
        "inspect_report_path": str(inspect_report_output),
        "visual_llm_qc_report_path": str(visual_llm_qc_output),
    }
    optional_paths = {
        "delivery_single_speaker_path": delivery_single_speaker_output,
        "delivery_notebooklm_path": delivery_notebooklm_output,
        "presenter_script_md_path": presenter_script_md_output,
        "presentation_video_path": presentation_video_output,
        "presentation_video_notebooklm_path": presentation_video_notebooklm_output,
        "video_sync_validation_single_speaker_path": video_sync_single_output,
        "video_sync_validation_notebooklm_path": video_sync_notebooklm_output,
        "video_delivery_validation_path": video_delivery_output,
        "editorial_gate_report_path": editorial_gate_output,
        "presentation_review_path": presentation_review_output,
    }
    for key, value in optional_paths.items():
        if value is not None:
            record[key] = str(value)
    return record


