"""Command-line interface for the conference pipeline."""

import asyncio
import os
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Optional

import typer

from conference_pipeline.analysis.errors import AnalysisContractError
from conference_pipeline.config import PipelineSettings, load_settings
from conference_pipeline.smoke_test import (
    DEFAULT_CONFERENCE_SMOKE_TEST_STAGES,
    DEFAULT_SMOKE_TEST_MODE,
    DEFAULT_SMOKE_TEST_TOPIC,
    DEFAULT_SMOKE_TEST_WORKSPACE,
    run_conference_smoketest,
    run_small_neurips_smoketest,
)
from conference_pipeline.presentation_proof import (
    DEFAULT_PRESENTATION_PROOF_MODE,
    DEFAULT_PRESENTATION_PROOF_PRESENTATION_TYPE,
    DEFAULT_PRESENTATION_PROOF_STAGES,
    DEFAULT_PRESENTATION_PROOF_TOPIC,
    DEFAULT_PRESENTATION_PROOF_WORKSPACE,
    run_presentation_proof,
)
from conference_pipeline.workflow_unblock import acknowledge_workflow_blocker
from conference_pipeline.workflow_runs import (
    WorkflowExecutionMode,
    generate_run_id,
    load_workflow_summary,
)
from conference_pipeline.workflow import (
    format_workflow_stage,
    get_default_workspace_path,
    ordered_workflow_stages,
    parse_stage_selection,
    resolve_workspace,
    run_selected_stages,
)
from utils.logging import setup_logging

app = typer.Typer(
    name="conference-pipeline",
    help="Conference video, slides, analysis, and workflow pipeline",
)
workflow_app = typer.Typer(help="Canonical workflow stages for conference production.")
app.add_typer(workflow_app, name="workflow")

# v1.9 Phase 45: vector-store maintenance subcommands.
vector_store_app = typer.Typer(
    help="Vector-store maintenance: rebuild Chroma collections, inspect dim guards.",
)
app.add_typer(vector_store_app, name="vector-store")

# v6.0 Phase 3: the public-release commands (init, plan, build, check, publish).
from conference_pipeline.publishing.release_cli import release_app  # noqa: E402

app.add_typer(release_app, name="release")

# v6.0 Phase 5: admit late paper sets into a workspace corpus by stable identity.
from conference_pipeline.production.paper_admission_cli import papers_app  # noqa: E402

app.add_typer(papers_app, name="papers")
# v6.0 Phase 5.1: bind existing talk transcripts into evidence lanes.
from conference_pipeline.production.talk_admission_cli import talks_app  # noqa: E402

app.add_typer(talks_app, name="talks")


def get_settings(
    storage_root: Optional[Path] = None,
    env_file: Optional[Path] = None,
) -> PipelineSettings:
    """Load settings with optional overrides."""
    settings = load_settings(env_file)
    if storage_root:
        settings.storage_root = storage_root
    return settings


def _get_pipeline_class():
    from conference_pipeline.pipeline import NeuripsPipeline

    return NeuripsPipeline


def _get_models():
    from conference_pipeline.models import PresentationStatus, TARGET_SESSION_TYPES

    return PresentationStatus, TARGET_SESSION_TYPES


def _get_auth_classes():
    from conference_pipeline.auth import NeuripsBrowserSession, NeuripsSession

    return NeuripsBrowserSession, NeuripsSession


def _get_session_scraper():
    from conference_pipeline.session_scraper import SessionScraper

    return SessionScraper


def _get_storage_backend():
    from conference_pipeline.storage import StorageBackend

    return StorageBackend


def _coerce_workflow_mode(mode: str) -> str:
    try:
        return WorkflowExecutionMode(mode).value
    except ValueError as exc:
        raise typer.BadParameter(
            "Invalid --mode: must be one of live, dry-run, mock"
        ) from exc


def _echo_summary_json(summary_payload: object) -> None:
    try:
        typer.echo(json.dumps(summary_payload, indent=2, sort_keys=True))
    except BlockingIOError:
        summary_path = None
        if isinstance(summary_payload, Mapping):
            raw_summary_path = summary_payload.get("summary_path")
            if raw_summary_path:
                summary_path = str(raw_summary_path)
        message = "Workflow summary JSON was too large for stdout."
        if summary_path:
            message = f"{message} Summary written to: {summary_path}"
        try:
            typer.echo(message, err=True)
        except BlockingIOError:
            pass


def _is_presentation_manifest_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "PresentationManifestValidationError"


def _is_presentation_validation_error(exc: Exception) -> bool:
    return exc.__class__.__name__ == "PresentationValidationError"


def _parse_smoke_stages(stages: str) -> tuple[str, ...]:
    try:
        return tuple(stage.id for stage in parse_stage_selection(stages))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_workflow_with_summary_output(
    resolved_workspace,
    selected_stages,
    *,
    topic: Optional[str],
    presentation_type: Optional[str],
    mode: str,
    run_id: str,
    resume: bool,
    summary: bool,
    strict_required_artifacts: bool = False,
) -> None:
    run_kwargs = {
        "topic": topic,
        "mode": mode,
        "run_id": run_id,
        "resume": resume,
        "write_receipts": True,
        "strict_required_artifacts": strict_required_artifacts,
    }
    if presentation_type is not None:
        run_kwargs["presentation_type"] = presentation_type
    try:
        run_summary = run_selected_stages(
            resolved_workspace,
            selected_stages,
            **run_kwargs,
        )
    except AnalysisContractError as exc:
        if summary:
            try:
                _echo_summary_json(load_workflow_summary(resolved_workspace, run_id))
            except ValueError:
                pass
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        if summary:
            try:
                _echo_summary_json(load_workflow_summary(resolved_workspace, run_id))
            except ValueError:
                pass
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        if summary:
            try:
                _echo_summary_json(load_workflow_summary(resolved_workspace, run_id))
            except ValueError:
                pass
        if _is_presentation_manifest_error(exc) or _is_presentation_validation_error(exc):
            raise typer.BadParameter(str(exc)) from exc
        raise

    if run_summary is None:
        return
    if getattr(run_summary, "halt_status", None) == "halted":
        if summary:
            _echo_summary_json(run_summary.to_dict())
        else:
            typer.echo(f"Workflow halted at stage: {run_summary.halt_stage_id}", err=True)
            if run_summary.blocker_receipt_path:
                typer.echo(
                    f"Blocker receipt: {run_summary.blocker_receipt_path}",
                    err=True,
                )
            typer.echo(f"Workflow summary: {run_summary.summary_path}", err=True)
        raise typer.Exit(code=1)
    if summary:
        _echo_summary_json(run_summary.to_dict())
    else:
        typer.echo(f"Workflow summary: {run_summary.summary_path}")


@workflow_app.command("list")
def workflow_list(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/neurips2025)",
    ),
):
    """List canonical workflow stages for a conference workspace."""

    resolve_workspace(workspace)
    for stage in ordered_workflow_stages():
        typer.echo(format_workflow_stage(stage))


@workflow_app.command("run")
def workflow_run(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/neurips2025)",
    ),
    stages: str = typer.Option(
        ...,
        "--stages",
        help="Comma-separated stage ids such as discover,download,transcribe",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        help="Optional topic selector for topic-aware output stages",
    ),
    presentation_type: Optional[str] = typer.Option(
        None,
        "--presentation-type",
        help="Presentation type for first-class presentation stages.",
    ),
    mode: str = typer.Option(
        "live",
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Workflow run id for receipts and resume.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume by skipping compatible succeeded receipts in --run-id.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print machine-readable summary JSON after the run.",
    ),
    strict_required_artifacts: bool = typer.Option(
        False,
        "--strict-required-artifacts",
        help="Fail validation when required in-scope presentation artifacts are missing.",
    ),
):
    """Run selected canonical workflow stages against a conference workspace.

    Stage 'analyze' emits conferences/<slug>/analysis/analysis_manifest.json
    which the 'podcast' and 'deck' stages consume as their source of truth for
    topic selection and paths. If the manifest is missing, run 'workflow run
    --stages analyze' to emit it, or 'python3 scripts/backfill_analysis_manifest.py
    --workspace <path>' for a legacy workspace.
    """

    try:
        resolved_workspace = resolve_workspace(workspace)
        selected_stages = parse_stage_selection(stages)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    coerced_mode = _coerce_workflow_mode(mode)
    if resume and run_id is None:
        raise typer.BadParameter("--resume requires --run-id")

    # --summary controls display only; workflow run always writes receipts in Phase 11.
    effective_run_id = run_id or generate_run_id()
    try:
        _run_workflow_with_summary_output(
            resolved_workspace,
            selected_stages,
            topic=topic,
            presentation_type=presentation_type,
            mode=coerced_mode,
            run_id=effective_run_id,
            resume=resume,
            summary=summary,
            strict_required_artifacts=strict_required_artifacts,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command("presentation-validate")
def presentation_validate(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace.",
    ),
    presentation_type: str = typer.Option(
        ...,
        "--presentation-type",
        help="Presentation type to validate.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        help="Topic id for topic-deep-dive presentations.",
    ),
    mode: str = typer.Option(
        "dry-run",
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Workflow run id for receipts and the validation report.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume by skipping compatible succeeded validation receipts in --run-id.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print machine-readable summary JSON after the run.",
    ),
    strict_required_artifacts: bool = typer.Option(
        False,
        "--strict-required-artifacts",
        help="Fail validation when required in-scope presentation artifacts are missing.",
    ),
):
    """Validate structural readiness of manifest-registered presentation artifacts."""

    if resume and run_id is None:
        raise typer.BadParameter("--resume requires --run-id")
    if presentation_type == "topic-deep-dive" and not topic:
        raise typer.BadParameter("--topic is required when --presentation-type topic-deep-dive")

    try:
        resolved_workspace = resolve_workspace(workspace)
        from conference_pipeline.presentation.validation import VALIDATION_STAGE_IDS

        selected_stages = parse_stage_selection(",".join(VALIDATION_STAGE_IDS))
        _run_workflow_with_summary_output(
            resolved_workspace,
            selected_stages,
            topic=topic,
            presentation_type=presentation_type,
            mode=_coerce_workflow_mode(mode),
            run_id=run_id or generate_run_id(),
            resume=resume,
            summary=summary,
            strict_required_artifacts=strict_required_artifacts,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@workflow_app.command("resume")
def workflow_resume(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/neurips2025)",
    ),
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Workflow run id to resume from receipts.",
    ),
    stages: str = typer.Option(
        ...,
        "--stages",
        help="Comma-separated stage ids such as discover,download,transcribe",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        help="Optional topic selector for topic-aware output stages",
    ),
    presentation_type: Optional[str] = typer.Option(
        None,
        "--presentation-type",
        help="Presentation type for first-class presentation stages.",
    ),
    mode: str = typer.Option(
        "live",
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print machine-readable summary JSON after the run.",
    ),
):
    """Resume selected canonical workflow stages from compatible receipts."""

    try:
        resolved_workspace = resolve_workspace(workspace)
        selected_stages = parse_stage_selection(stages)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        _run_workflow_with_summary_output(
            resolved_workspace,
            selected_stages,
            topic=topic,
            presentation_type=presentation_type,
            mode=_coerce_workflow_mode(mode),
            run_id=run_id,
            resume=True,
            summary=summary,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc


@workflow_app.command("summary")
def workflow_summary(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/neurips2025)",
    ),
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Workflow run id to inspect.",
    ),
):
    """Print the machine-readable summary JSON for a workflow run."""

    try:
        resolved_workspace = resolve_workspace(workspace)
        summary_payload = load_workflow_summary(resolved_workspace, run_id)
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _echo_summary_json(summary_payload)


@workflow_app.command("stage-status")
def workflow_stage_status(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/cvpr2026)",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Run id whose receipts to classify."),
    layout: str = typer.Option(
        "auto",
        "--layout",
        help="production (registry receipt_filename bindings), generic (workflow stages), or auto.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
):
    """Read-only: FRESH / STALE (why) / MISSING / BLOCKED per stage, and the next stage to run."""

    from conference_pipeline.receipt_lineage import stage_lineage_report

    if layout not in ("auto", "production", "generic"):
        raise typer.BadParameter("--layout must be auto, production or generic")
    report = stage_lineage_report(workspace, run_id, layout=layout)
    if json_output:
        _echo_summary_json(report)
        return
    typer.echo(f"run {report['run_id']} ({report['layout']} layout)")
    for stage_id, entry in report["stages"].items():
        typer.echo(_stage_status_line(stage_id, entry))
    typer.echo(f"next stage to run: {report['next_stage_to_run'] or '<none>'}")
    if report.get("substeps"):
        typer.echo("sub-steps (receipts consumed by stages, written by their own commands):")
        for substep_id, entry in report["substeps"].items():
            typer.echo(_stage_status_line(substep_id, entry))
        typer.echo(f"next sub-step to run: {report.get('next_substep_to_run') or '<none>'}")


def _stage_status_line(stage_id: str, entry: dict) -> str:
    state = str(entry["state"]).upper()
    line = f"  {stage_id:40s} {state}"
    if entry.get("state") == "stale":
        reasons = entry.get("reasons") or []
        first = reasons[0] if reasons else {}
        detail = str(first.get("code") or "").replace("stage_contract_", "")
        if first.get("upstream_stage_id"):
            detail += f" {first['upstream_stage_id']}"
        elif first.get("input"):
            detail += f" {first['input']}"
        if first.get("paths"):
            detail += f" [{', '.join(first['paths'])}]"
        if first.get("chain"):
            detail += f" via {' -> '.join(Path(item).name for item in first['chain'])}"
        line += f" ({detail}{'; +%d more' % (len(reasons) - 1) if len(reasons) > 1 else ''})"
    elif entry.get("status") and entry.get("state") not in ("fresh",):
        line += f" ({entry['status']})"
    return line


@workflow_app.command("bind-lineage")
def workflow_bind_lineage(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/cvpr2026)",
    ),
    run_id: str = typer.Option(..., "--run-id", help="Production run id whose receipts to stamp."),
    backfill: bool = typer.Option(
        False,
        "--backfill",
        help="Stamp output_digest and lineage onto succeeded receipts written before lineage existed.",
    ),
    stages: str | None = typer.Option(None, "--stages", help="Comma-separated stage ids (default: every registered stage)."),
    apply: bool = typer.Option(False, "--apply", help="Write the stamped receipts; without it nothing is modified."),
    assume_current: str | None = typer.Option(
        None,
        "--assume-current",
        help="Comma-separated stage ids the operator attests are consistent with their current upstreams.",
    ),
    operator: str | None = typer.Option(None, "--operator", help="Who attests --assume-current (recorded on the receipt)."),
    force: bool = typer.Option(False, "--force", help="Re-stamp receipts that already carry lineage."),
    json_output: bool = typer.Option(False, "--json", help="Print the full report as JSON."),
):
    """Stamp receipt lineage onto an existing run (read-only unless --apply)."""

    from conference_pipeline.receipt_lineage import backfill_run_lineage

    if not backfill:
        raise typer.BadParameter("pass --backfill; binding on fresh runs happens in the runners themselves")
    if assume_current and not operator:
        raise typer.BadParameter("--assume-current requires --operator")
    report = backfill_run_lineage(
        workspace,
        run_id,
        stages=[item.strip() for item in stages.split(",") if item.strip()] if stages else None,
        apply=apply,
        assume_current=[item.strip() for item in assume_current.split(",") if item.strip()] if assume_current else (),
        operator=operator,
        force=force,
    )
    if json_output:
        _echo_summary_json(report)
        return
    typer.echo(f"run {report['run_id']} ({'applied' if report['applied'] else 'dry run'})")
    for stage_id, entry in report["stages"].items():
        line = f"  {stage_id:32s} {entry.get('action')}"
        if entry.get("unresolved"):
            line += f"  unresolved={len(entry['unresolved'])}"
        if entry.get("verdict"):
            line += f"  verdict={entry['verdict']}"
        typer.echo(line)


@workflow_app.command("unblock")
def workflow_unblock(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace (for example conferences/neurips2025)",
    ),
    run_id: str = typer.Option(
        ...,
        "--run-id",
        help="Workflow run id whose active blocker is being acknowledged.",
    ),
    blocker_path: str = typer.Option(
        ...,
        "--blocker-path",
        help="Absolute or workspace-relative path to the blocker receipt JSON.",
    ),
    operator: str = typer.Option(
        ...,
        "--operator",
        help="Operator id recorded on the manual acknowledgement receipt.",
    ),
    note: Optional[str] = typer.Option(
        None,
        "--note",
        help="Optional operator note for the acknowledgement receipt.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print the refreshed workflow summary JSON after acknowledgement.",
    ),
):
    """Write a manual unblock receipt for an active workflow blocker."""

    try:
        result = acknowledge_workflow_blocker(
            workspace=workspace,
            run_id=run_id,
            blocker_path=blocker_path,
            operator_id=operator,
            note=note,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if summary:
        _echo_summary_json(result["summary"])
    else:
        typer.echo(f"Unblock receipt: {result['unblock_receipt_path']}")
        typer.echo(f"Workflow summary: {result['summary_path']}")


@app.command("control-plane")
def control_plane(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
):
    """Start the local operator UI control plane."""

    import uvicorn

    uvicorn.run(
        "conference_pipeline.control_plane.app:create_app",
        factory=True,
        host=host,
        port=port,
        log_level="info",
    )


@app.command("conference-smoketest")
def conference_smoketest(
    workspace: Path = typer.Option(
        DEFAULT_SMOKE_TEST_WORKSPACE,
        "--workspace",
        help="Path to the conference workspace.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Workflow run id for receipts and the smoke-test report.",
    ),
    mode: str = typer.Option(
        DEFAULT_SMOKE_TEST_MODE,
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    topic: Optional[str] = typer.Option(
        None,
        "--topic",
        help="Optional topic selector for topic-aware output stages.",
    ),
    stages: str = typer.Option(
        ",".join(DEFAULT_CONFERENCE_SMOKE_TEST_STAGES),
        "--stages",
        help="Comma-separated workflow stages for the smoke-test proof.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from compatible receipts in --run-id.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print the machine-readable smoke-test report JSON.",
    ),
    # v1.9 Phase 47: opt-in local-LLM lane.
    local: bool = typer.Option(
        False,
        "--local",
        help=(
            "Run with local LLM/embedding providers only (no OpenAI). "
            "Refuses to run if OPENAI_API_KEY is set in the environment."
        ),
    ),
    local_provider: str = typer.Option(
        "mlx",
        "--local-provider",
        help="Local provider for the embedding lane: 'mlx' (default) or 'ollama'.",
    ),
    # v1.61 Phase 70: opt-in local vision lane (separate from --local-provider
    # which controls the embedding/text lane). Reuses --local's loopback /
    # OPENAI_API_KEY-refused enforcement; has no effect when --local is absent.
    local_vision_model: Optional[str] = typer.Option(
        None,
        "--local-vision-model",
        help=(
            "Local vision model id (e.g. mlx-community/Qwen3-VL-8B-Instruct-4bit "
            "or apple/FastVLM-7B-MLX). Requires --local."
        ),
    ),
    local_vision_provider: str = typer.Option(
        "vllm_mlx",
        "--local-vision-provider",
        help=(
            "Local vision runtime: 'vllm_mlx' (mlx-vlm.server / vllm-mlx, "
            "default) or 'fastvlm' (in-process Apple FastVLM-MLX). Requires "
            "--local."
        ),
    ),
    local_vision_base_url: Optional[str] = typer.Option(
        None,
        "--local-vision-base-url",
        help=(
            "Loopback URL of the running vision server (vllm_mlx). "
            "Falls back to GPTR_LOCAL_VISION_BASE_URL or http://127.0.0.1:8090/v1."
        ),
    ),
):
    """Run a generic conference smoke-test proof and write a report."""

    if resume and run_id is None:
        raise typer.BadParameter("--resume requires --run-id")

    # v1.9 Phase 47: --local pre-checks. Refuse to run if OPENAI_API_KEY is
    # set — the operator must explicitly unset it to prove the local-only
    # path. This is a deliberate paranoid check; auto-unsetting hides
    # configuration drift.
    #
    # v1.9 review CRITICAL-2: load .env FIRST so a key in .env cannot bypass
    # the precondition. Downstream stages call ``get_settings(...)`` which
    # itself calls ``load_dotenv()``; without this pre-load, an operator
    # with ``OPENAI_API_KEY`` only in ``.env`` would pass the precondition
    # and then have the cloud key injected into ``os.environ`` later by
    # ``get_settings``, defeating the local-only intent.
    if local:
        try:
            from dotenv import load_dotenv
        except ImportError:  # pragma: no cover - dotenv is a hard dep elsewhere
            load_dotenv = None  # type: ignore[assignment]
        if load_dotenv is not None:
            load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            typer.echo(
                "--local requires OPENAI_API_KEY to be unset to prove the "
                "local-only path. Run: unset OPENAI_API_KEY (also remove "
                "it from .env)",
                err=True,
            )
            raise typer.Exit(code=2)
        if local_provider not in ("mlx", "ollama"):
            typer.echo(
                f"--local-provider must be 'mlx' or 'ollama'; got "
                f"{local_provider!r}",
                err=True,
            )
            raise typer.Exit(code=2)
        # Set provider env vars consumed by build_embedding_client and
        # LocalLLMResearcher (via GPTR_LOCAL_LLM_RUNTIME_KIND for any
        # downstream reader that wants the runtime hint).
        os.environ["EMBEDDING_PROVIDER"] = local_provider
        os.environ["GPTR_LOCAL_LLM_RUNTIME_KIND"] = local_provider
        # v1.61 Phase 70: optional vision-lane env vars consumed by
        # SlideAnalysisEngine (provider='local_vlm') and
        # LLMOcrEngine(provider='local').
        if local_vision_model:
            if local_vision_provider not in ("vllm_mlx", "fastvlm"):
                typer.echo(
                    f"--local-vision-provider must be 'vllm_mlx' or 'fastvlm'; "
                    f"got {local_vision_provider!r}",
                    err=True,
                )
                raise typer.Exit(code=2)
            os.environ["GPTR_LOCAL_VISION_MODEL"] = local_vision_model
            os.environ["GPTR_LOCAL_VISION_RUNTIME_KIND"] = local_vision_provider
            if local_vision_provider == "vllm_mlx":
                base_url = (
                    local_vision_base_url
                    or os.environ.get("GPTR_LOCAL_VISION_BASE_URL")
                    or "http://127.0.0.1:8090/v1"
                )
                os.environ["GPTR_LOCAL_VISION_BASE_URL"] = base_url
    elif local_vision_model:
        typer.echo(
            "--local-vision-model requires --local. The vision lane reuses "
            "--local's loopback / OPENAI_API_KEY-refused enforcement.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        report = run_conference_smoketest(
            workspace=workspace,
            run_id=run_id,
            mode=_coerce_workflow_mode(mode),
            topic=topic,
            stages=_parse_smoke_stages(stages),
            resume=resume,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if summary:
        _echo_summary_json(report.to_dict())
    else:
        typer.echo(f"Smoke-test report: {report.report_path}")
        # v1.61 Phase 70: surface vision-lane status when configured so the
        # operator sees the redacted endpoint + per-task registry status +
        # session receipt count without needing to grep the JSON report.
        if report.local_vision_status is not None:
            block = report.local_vision_status
            typer.echo(
                f"Vision lane: {block['endpoint']} "
                f"(runtime={block['runtime_kind']}, "
                f"session_receipts={block['session_receipt_count']})"
            )
            for task_id, status in block["registry_status_by_task"].items():
                typer.echo(f"  {task_id}: {status}")


@app.command("extraction-status")
def extraction_status_cmd(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Path to the conference workspace root.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Print machine-readable JSON instead of human-readable text.",
    ),
):
    """Show extraction posture from on-disk receipts — no live endpoint calls."""
    from conference_pipeline.extractions.classifier import get_extraction_status

    resolved_workspace = resolve_workspace(workspace)
    result = get_extraction_status(resolved_workspace)

    if json_output:
        typer.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        typer.echo(f"extraction_mode:        {result['extraction_mode']}")
        typer.echo(
            f"local_llm_readiness:    {'present' if result['local_llm_readiness'] else 'none'}"
        )
        typer.echo(f"cloud_escalation_count: {result['cloud_escalation_count']}")
        typer.echo(f"ocr_posture:            {result['ocr_posture']}")
        typer.echo(f"blockers:               {result['blockers']}")
        if result["receipt_links"]:
            typer.echo("receipt_links:")
            for link in result["receipt_links"]:
                typer.echo(f"  {link}")


@app.command("small-neurips-smoketest")
def small_neurips_smoketest(
    workspace: Path = typer.Option(
        DEFAULT_SMOKE_TEST_WORKSPACE,
        "--workspace",
        help="Path to the conference workspace.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Workflow run id for receipts and the smoke-test report.",
    ),
    mode: str = typer.Option(
        DEFAULT_SMOKE_TEST_MODE,
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    topic: str = typer.Option(
        DEFAULT_SMOKE_TEST_TOPIC,
        "--topic",
        help="Topic selector for the smoke-test output stages.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from compatible receipts in --run-id.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print the machine-readable smoke-test report JSON.",
    ),
):
    """Run the small NeurIPS smoke-test proof and write a report."""

    if resume and run_id is None:
        raise typer.BadParameter("--resume requires --run-id")

    try:
        report = run_small_neurips_smoketest(
            workspace=workspace,
            run_id=run_id,
            mode=_coerce_workflow_mode(mode),
            topic=topic,
            resume=resume,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if summary:
        _echo_summary_json(report.to_dict())
    else:
        typer.echo(f"Smoke-test report: {report.report_path}")


@app.command("presentation-proof")
def presentation_proof(
    workspace: Path = typer.Option(
        DEFAULT_PRESENTATION_PROOF_WORKSPACE,
        "--workspace",
        help="Path to the conference workspace.",
    ),
    presentation_id: Optional[str] = typer.Option(
        None,
        "--presentation-id",
        help="Optional presentation id to record in the proof report.",
    ),
    presentation_type: str = typer.Option(
        DEFAULT_PRESENTATION_PROOF_PRESENTATION_TYPE,
        "--presentation-type",
        help="Presentation type to prove.",
    ),
    topic: Optional[str] = typer.Option(
        DEFAULT_PRESENTATION_PROOF_TOPIC,
        "--topic",
        help="Topic id for topic-deep-dive presentations.",
    ),
    mode: str = typer.Option(
        DEFAULT_PRESENTATION_PROOF_MODE,
        "--mode",
        help="Execution mode: live, dry-run, or mock.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Workflow run id for receipts and the presentation-proof report.",
    ),
    stages: str = typer.Option(
        ",".join(DEFAULT_PRESENTATION_PROOF_STAGES),
        "--stages",
        help="Comma-separated presentation stages for the proof.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from compatible receipts in --run-id.",
    ),
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Print the machine-readable presentation-proof report JSON.",
    ),
    validation_report_path: Optional[Path] = typer.Option(
        None,
        "--validation-report-path",
        help="Optional validation report path to link from the proof report.",
    ),
):
    """Run a receipt-backed presentation pipeline proof."""

    if resume and run_id is None:
        raise typer.BadParameter("--resume requires --run-id")

    try:
        report = run_presentation_proof(
            workspace=workspace,
            presentation_id=presentation_id,
            presentation_type=presentation_type,
            topic=topic,
            mode=_coerce_workflow_mode(mode),
            run_id=run_id,
            stages=_parse_smoke_stages(stages),
            resume=resume,
            validation_report_path=validation_report_path,
        )
    except AnalysisContractError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except Exception as exc:
        if _is_presentation_manifest_error(exc) or _is_presentation_validation_error(exc):
            raise typer.BadParameter(str(exc)) from exc
        raise

    if summary:
        _echo_summary_json(report.to_dict())
    else:
        typer.echo(f"Presentation proof report: {report.report_path}")


@app.command()
def discover(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Scrape NeurIPS 2025 schedule and build presentation list."""
    NeuripsPipeline = _get_pipeline_class()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()
        try:
            await pipeline.login()
            presentations = await pipeline.discover_presentations()
            typer.echo(f"Discovered {len(presentations)} presentations")
        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command("discover-sessions")
def discover_sessions_cmd(
    output: Path = typer.Option(
        Path("sessions.json"),
        "--output",
        "-o",
        help="Output JSON file for discovered sessions",
    ),
    year: Optional[int] = typer.Option(
        None,
        "--year",
        "-y",
        help="Conference year (defaults to settings.year)",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not use Playwright (faster but may miss JS content)",
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
):
    """Discover sessions by type and write sessions.json for manual filtering."""
    NeuripsBrowserSession, NeuripsSession = _get_auth_classes()
    _, TARGET_SESSION_TYPES = _get_models()
    SessionScraper = _get_session_scraper()
    settings = load_settings(env_file)
    if year:
        from urllib.parse import urlsplit

        settings.year = year
        conf_host = (urlsplit(settings.neurips_base_url).hostname or "neurips.cc")  # conference-literal-ok: guarded fallback host derived from configured base URL
        settings.virtual_base_url = f"https://{conf_host}/virtual/{year}"
        settings.storage_root = Path(f"./data/{conf_host.split('.')[0]}_{year}")

    setup_logging("INFO")

    async def run():
        # Prepare optional browser session
        browser_session = None
        if not no_browser:
            browser_session = NeuripsBrowserSession(settings)
            await browser_session.initialize()
            if settings.neurips_email and settings.neurips_password:
                await browser_session.login()

        http_session = await NeuripsSession.create(settings)

        try:
            scraper = SessionScraper(http_session, settings, browser_session)
            sessions = await scraper.discover_sessions(TARGET_SESSION_TYPES)
            sessions.sort(key=lambda s: (s.session_type.value, s.title))
            await scraper.save_sessions_json(sessions, output)
            typer.echo(f"Discovered {len(sessions)} sessions -> {output}")
        finally:
            await http_session.close()
            if browser_session:
                await browser_session.close()

    asyncio.run(run())


@app.command()
def download_assets(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Download audio/video/slides for presentations."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.download_assets_for_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                presentations = pipeline.storage.load_presentations_by_status(
                    PresentationStatus.DISCOVERED
                )
                for pres in presentations:
                    await pipeline.download_assets_for_presentation(pres)
                    typer.echo(f"Downloaded assets for: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def transcribe(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Transcribe audio for presentations."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.transcribe_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                presentations = pipeline.storage.load_presentations_by_status(
                    PresentationStatus.ASSETS_DOWNLOADED
                )
                for pres in presentations:
                    await pipeline.transcribe_presentation(pres)
                    typer.echo(f"Transcribed: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def extract_slides(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Extract slide images from PDFs."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.process_slides_for_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                # Process presentations with assets or transcripts
                for status in [
                    PresentationStatus.TRANSCRIBED,
                    PresentationStatus.ASSETS_DOWNLOADED,
                ]:
                    presentations = pipeline.storage.load_presentations_by_status(status)
                    for pres in presentations:
                        await pipeline.process_slides_for_presentation(pres)
                        typer.echo(f"Processed slides for: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def ocr_slides(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM-based OCR"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run OCR on slide images."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    settings.use_llm_ocr = use_llm
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.process_slides_for_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                presentations = pipeline.storage.load_presentations_by_status(
                    PresentationStatus.SLIDES_EXTRACTED
                )
                for pres in presentations:
                    await pipeline.process_slides_for_presentation(pres)
                    typer.echo(f"OCR complete for: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def extract_refs(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Extract paper references from slides and transcripts."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.extract_references_for_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                presentations = pipeline.storage.load_presentations_by_status(
                    PresentationStatus.OCR_DONE
                )
                for pres in presentations:
                    await pipeline.extract_references_for_presentation(pres)
                    typer.echo(f"Extracted references for: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def download_papers(
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID (all if not specified)"
    ),
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Resolve references and download open-access papers."""
    NeuripsPipeline = _get_pipeline_class()
    PresentationStatus, _ = _get_models()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.initialize()

        try:
            if pres_id:
                pres = pipeline.storage.load_presentation(pres_id)
                if pres:
                    await pipeline.download_papers_for_presentation(pres)
                else:
                    typer.echo(f"Presentation not found: {pres_id}", err=True)
            else:
                presentations = pipeline.storage.load_presentations_by_status(
                    PresentationStatus.REFERENCES_EXTRACTED
                )
                for pres in presentations:
                    await pipeline.download_papers_for_presentation(pres)
                    typer.echo(f"Downloaded papers for: {pres.id}")

        finally:
            await pipeline.cleanup()

    asyncio.run(run())


@app.command()
def run_all(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    presentation_ids: Optional[str] = typer.Option(
        None, "--presentations", "-p", help="Comma-separated presentation IDs"
    ),
    retry_errors: bool = typer.Option(
        False, "--retry-errors", "-r",
        help="Retry presentations that previously failed with errors"
    ),
    reprocess_completed: bool = typer.Option(
        False, "--reprocess-completed",
        help="Re-process presentations that already completed successfully"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run acquisition by default, with legacy targeted full-pipeline options.

    With no processing flags, this is a compatibility alias for the canonical
    acquisition workflow: discover, download, and transcribe. Use the workflow
    command for analysis, podcast, and deck stages.

    When --presentations, --retry-errors, or --reprocess-completed is provided,
    the command preserves the older full-pipeline runner behavior.

    Examples:

        # Acquisition alias
        neurips-pipeline run-all

        # Retry presentations through the legacy full-pipeline runner
        neurips-pipeline run-all --retry-errors

        # Process specific presentations through the legacy full-pipeline runner
        neurips-pipeline run-all -p "pres-123,pres-456"
    """
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    if presentation_ids or retry_errors or reprocess_completed:
        NeuripsPipeline = _get_pipeline_class()
        pres_list = None
        if presentation_ids:
            pres_list = [p.strip() for p in presentation_ids.split(",")]

        async def run():
            pipeline = NeuripsPipeline(settings)
            await pipeline.run_full_pipeline(
                pres_list,
                retry_errors=retry_errors,
                skip_completed=not reprocess_completed,
            )

        asyncio.run(run())
        return

    run_selected_stages(
        resolve_workspace(get_default_workspace_path()),
        parse_stage_selection("discover,download,transcribe"),
        env_file=env_file,
        storage_root=storage_root,
    )


@app.command()
def resume(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    retry_errors: bool = typer.Option(
        False, "--retry-errors", "-r",
        help="Also retry presentations that previously failed"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Resume the pipeline from where it left off.

    This is equivalent to 'run-all' but makes the intent clearer.
    The pipeline automatically continues processing from each presentation's
    last saved status.

    Use --retry-errors to also retry presentations that previously failed.
    """
    NeuripsPipeline = _get_pipeline_class()
    settings = get_settings(storage_root, env_file)
    setup_logging("DEBUG" if verbose else "INFO")

    async def run():
        pipeline = NeuripsPipeline(settings)
        await pipeline.run_full_pipeline(
            retry_errors=retry_errors,
            skip_completed=True,
        )

    asyncio.run(run())


@app.command()
def reset_errors(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    pres_id: Optional[str] = typer.Argument(
        None, help="Specific presentation ID to reset (all errors if not specified)"
    ),
):
    """Reset errored presentations to DISCOVERED status for retry.

    This allows you to review which presentations failed and selectively
    reset them for reprocessing.
    """
    PresentationStatus, _ = _get_models()
    StorageBackend = _get_storage_backend()
    settings = get_settings(storage_root, env_file)
    storage = StorageBackend(settings)

    if pres_id:
        pres = storage.load_presentation(pres_id)
        if pres is None:
            typer.echo(f"Presentation not found: {pres_id}", err=True)
            raise typer.Exit(1)
        if pres.status != PresentationStatus.ERROR:
            typer.echo(f"Presentation {pres_id} is not in ERROR status (current: {pres.status.value})")
            raise typer.Exit(1)

        typer.echo(f"Resetting {pres_id}: {pres.error_message}")
        pres.status = PresentationStatus.DISCOVERED
        pres.error_message = None
        storage.save_presentation(pres)
        typer.echo(f"✅ Reset {pres_id} to DISCOVERED status")
    else:
        errored = storage.load_presentations_by_status(PresentationStatus.ERROR)
        if not errored:
            typer.echo("No presentations in ERROR status")
            return

        typer.echo(f"Found {len(errored)} errored presentations:")
        for pres in errored:
            error_msg = pres.error_message[:60] + "..." if pres.error_message and len(pres.error_message) > 60 else pres.error_message
            typer.echo(f"  - {pres.id}: {error_msg}")

        if typer.confirm(f"\nReset all {len(errored)} presentations to DISCOVERED?"):
            for pres in errored:
                pres.status = PresentationStatus.DISCOVERED
                pres.error_message = None
                storage.save_presentation(pres)
            typer.echo(f"✅ Reset {len(errored)} presentations")
        else:
            typer.echo("Cancelled")


@app.command()
def stats(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
):
    """Show pipeline statistics."""
    StorageBackend = _get_storage_backend()
    settings = get_settings(storage_root, env_file)
    storage = StorageBackend(settings)

    stats = storage.get_stats()

    typer.echo("\n📊 Pipeline Statistics")
    typer.echo("=" * 40)
    typer.echo(f"Total presentations: {stats.get('total_presentations', 0)}")
    typer.echo(f"With transcripts: {stats.get('presentations_with_transcripts', 0)}")
    typer.echo(f"With OCR: {stats.get('presentations_with_ocr', 0)}")
    typer.echo(f"Total papers: {stats.get('total_papers', 0)}")
    typer.echo(f"Papers with PDFs: {stats.get('papers_with_pdfs', 0)}")
    typer.echo(f"Total references: {stats.get('total_references', 0)}")

    typer.echo("\n📋 Status breakdown:")
    for status, count in stats.get("presentations_by_status", {}).items():
        typer.echo(f"  {status}: {count}")


@app.command()
def list_presentations(
    storage_root: Optional[Path] = typer.Option(
        None, "--storage", "-s", help="Storage root directory"
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env", "-e", help="Path to .env file"
    ),
    status: Optional[str] = typer.Option(
        None, "--status", help="Filter by status"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Max items to show"),
):
    """List presentations in the database."""
    PresentationStatus, _ = _get_models()
    StorageBackend = _get_storage_backend()
    settings = get_settings(storage_root, env_file)
    storage = StorageBackend(settings)

    if status:
        try:
            pres_status = PresentationStatus(status)
            presentations = storage.load_presentations_by_status(pres_status)
        except ValueError:
            typer.echo(f"Invalid status: {status}", err=True)
            raise typer.Exit(1)
    else:
        presentations = storage.load_all_presentations()

    typer.echo(f"\n📋 Presentations ({len(presentations)} total, showing up to {limit})")
    typer.echo("=" * 60)

    for pres in presentations[:limit]:
        status_emoji = {
            PresentationStatus.DISCOVERED: "🔍",
            PresentationStatus.ASSETS_DOWNLOADED: "📥",
            PresentationStatus.TRANSCRIBED: "🎙️",
            PresentationStatus.SLIDES_EXTRACTED: "🖼️",
            PresentationStatus.OCR_DONE: "📝",
            PresentationStatus.REFERENCES_EXTRACTED: "📚",
            PresentationStatus.PAPERS_DOWNLOADED: "✅",
            PresentationStatus.ERROR: "❌",
        }.get(pres.status, "❓")

        title = pres.title[:50] + "..." if len(pres.title) > 50 else pres.title
        typer.echo(f"{status_emoji} [{pres.id}] {title}")
        typer.echo(f"   Status: {pres.status.value}")
        if pres.slideslive_id:
            typer.echo(f"   SlidesLive: {pres.slideslive_id}")
        typer.echo()


# ---------------------------------------------------------------------------
# v1.9 Phase 45 — `vector-store` subcommands
# ---------------------------------------------------------------------------


@vector_store_app.command("rebuild")
def vector_store_rebuild(
    workspace: Path = typer.Option(
        ...,
        "--workspace",
        help="Conference workspace path (contains storage/storage_manifest.json).",
    ),
    provider: str = typer.Option(
        ...,
        "--provider",
        help="Embedding provider to (re)build with: openai | mlx | ollama | fake.",
    ),
    collection_base: str = typer.Option(
        "neurips_presentations",
        "--collection-base",
        help="Base name for the Chroma collection; suffixed with provider+model+dim.",
    ),
    source: Optional[Path] = typer.Option(
        None,
        "--source",
        help="Path to the storage manifest used as the source-of-truth for "
        "idempotency (defaults to <workspace>/storage/storage_manifest.json).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip the source-manifest-hash idempotency check; always rebuild.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print intended actions; touch nothing.",
    ),
    operator: Optional[str] = typer.Option(
        None,
        "--operator",
        help="Operator id recorded in the rebuild receipt for audit.",
    ),
):
    """Drop and rebuild a Chroma collection idempotently for the given provider.

    Reads a manifest hash from --source (default
    <workspace>/storage/storage_manifest.json). If the hash matches the
    persisted Chroma collection's metadata['source_manifest_hash'], the
    rebuild is a no-op (exit 0). --force overrides the idempotency check.
    --dry-run prints the intended actions without modifying state.

    Writes a VectorStoreRebuildReceipt under
    <workspace>/storage/vector-store-receipts/rebuilds/.
    """
    import hashlib
    import time
    from datetime import datetime, timezone

    from conference_pipeline.indexing.embeddings import build_embedding_client
    from conference_pipeline.indexing.vector_store_naming import (
        derive_collection_name,
    )
    from conference_pipeline.provider_receipts import (
        VECTOR_STORE_REBUILD_RECEIPT_MARKER,
        VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION,
        VectorStoreRebuildReceipt,
        write_vector_store_rebuild_receipt,
    )

    started = time.monotonic()
    workspace = workspace.resolve()
    source_path = (
        source.resolve()
        if source is not None
        else workspace / "storage" / "storage_manifest.json"
    )
    if not source_path.exists():
        typer.echo(
            f"vector-store rebuild: source manifest not found at {source_path}",
            err=True,
        )
        raise typer.Exit(code=2)

    manifest_bytes = source_path.read_bytes()
    source_manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    # Build the embedding client to learn its model + dim. We pass an
    # injection-friendly _loader for mlx so the rebuild does not require
    # the optional package on the operator's machine for the dry-run path.
    config = {"provider": provider}
    try:
        client = build_embedding_client(config)
    except (ValueError, NotImplementedError) as exc:
        typer.echo(f"vector-store rebuild: {exc}", err=True)
        raise typer.Exit(code=2)

    model = getattr(client, "model", "unknown")
    embedding_dim = int(getattr(client, "embedding_dim", 0))
    if embedding_dim == 0:
        typer.echo(
            f"vector-store rebuild: client {client!r} reports embedding_dim=0; "
            "rebuild requires a known dim.",
            err=True,
        )
        raise typer.Exit(code=2)

    collection_name = derive_collection_name(
        collection_base, provider, model, embedding_dim
    )

    if dry_run:
        typer.echo(
            f"vector-store rebuild [dry-run]: would rebuild collection "
            f"{collection_name!r} from {source_path}"
        )
        receipt = VectorStoreRebuildReceipt(
            schema_version=VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION,
            receipt_type=VECTOR_STORE_REBUILD_RECEIPT_MARKER,
            provider=provider,
            model=model,
            embedding_dim=embedding_dim,
            collection_name=collection_name,
            source_manifest_hash=source_manifest_hash,
            source_manifest_path=str(source_path),
            chunk_count=0,
            duration_s=round(time.monotonic() - started, 6),
            outcome="dry-run",
            timestamp=datetime.now(timezone.utc).isoformat(),
            operator_id=operator,
            workspace_path=str(workspace),
        )
        write_vector_store_rebuild_receipt(receipt, workspace)
        return

    # Real rebuild path: open the persistent Chroma client; check idempotency;
    # delete + recreate the collection. Embedding the source chunks is
    # delegated to the existing analysis-step; this CLI only manages the
    # collection lifecycle. (A future enhancement can re-run the full
    # embed-and-upsert loop here.)
    try:
        import chromadb  # type: ignore[import-not-found]
    except ImportError:
        typer.echo("vector-store rebuild: chromadb not installed", err=True)
        raise typer.Exit(code=2)

    chroma_path = workspace / "storage" / "vector-store" / "chroma_db"
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    # Idempotency check: if the existing collection's metadata records the
    # same source_manifest_hash and --force is not set, no-op.
    existing_collections = chroma_client.list_collections()
    existing = next(
        (c for c in existing_collections if getattr(c, "name", None) == collection_name),
        None,
    )
    if existing is not None and not force:
        existing_hash = (existing.metadata or {}).get("source_manifest_hash")
        if existing_hash == source_manifest_hash:
            typer.echo(
                f"vector-store rebuild: collection {collection_name!r} is "
                f"already up-to-date with source hash; no-op."
            )
            receipt = VectorStoreRebuildReceipt(
                schema_version=VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION,
                receipt_type=VECTOR_STORE_REBUILD_RECEIPT_MARKER,
                provider=provider,
                model=model,
                embedding_dim=embedding_dim,
                collection_name=collection_name,
                source_manifest_hash=source_manifest_hash,
                source_manifest_path=str(source_path),
                chunk_count=existing.count() if hasattr(existing, "count") else 0,
                duration_s=round(time.monotonic() - started, 6),
                outcome="no-op-idempotent",
                timestamp=datetime.now(timezone.utc).isoformat(),
                operator_id=operator,
                workspace_path=str(workspace),
            )
            write_vector_store_rebuild_receipt(receipt, workspace)
            return

    # Drop + recreate. Operator must re-run the analysis pipeline to
    # re-populate the collection; this CLI does not auto-embed (would
    # duplicate the existing analysis-step).
    #
    # v1.9 review HIGH-1: the delete → create sequence is non-atomic. If
    # ``create_collection`` raises (transient Chroma fault, schema
    # validation error, disk-full) AFTER the existing collection has
    # already been deleted, the operator is left with no collection AND
    # no receipt. Wrap the destructive section so a failure (a) is
    # surfaced via a ``VectorStoreRebuildReceipt`` with ``outcome="failed"``
    # and (b) exits with code 3 carrying the sanitized error.
    if existing is not None:
        chroma_client.delete_collection(collection_name)

    from datetime import datetime as _dt
    new_metadata = {
        "dim": embedding_dim,
        "provider": provider,
        "model": model,
        "source_manifest_hash": source_manifest_hash,
        "created_at_iso": _dt.now(timezone.utc).isoformat(),
    }
    try:
        chroma_client.create_collection(collection_name, metadata=new_metadata)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        from conference_pipeline.iclr2025_live import sanitize_for_receipt

        sanitized = sanitize_for_receipt(str(exc))
        failed_receipt = VectorStoreRebuildReceipt(
            schema_version=VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION,
            receipt_type=VECTOR_STORE_REBUILD_RECEIPT_MARKER,
            provider=provider,
            model=model,
            embedding_dim=embedding_dim,
            collection_name=collection_name,
            source_manifest_hash=source_manifest_hash,
            source_manifest_path=str(source_path),
            chunk_count=0,
            duration_s=round(time.monotonic() - started, 6),
            outcome="failed",
            timestamp=datetime.now(timezone.utc).isoformat(),
            operator_id=operator,
            workspace_path=str(workspace),
        )
        write_vector_store_rebuild_receipt(failed_receipt, workspace)
        typer.echo(
            f"vector-store rebuild: create_collection failed for "
            f"{collection_name!r} after delete; collection is now MISSING. "
            f"Re-run the rebuild after addressing: {sanitized}",
            err=True,
        )
        raise typer.Exit(code=3)

    typer.echo(
        f"vector-store rebuild: dropped + recreated {collection_name!r}; "
        "re-run the analysis pipeline to populate it."
    )
    receipt = VectorStoreRebuildReceipt(
        schema_version=VECTOR_STORE_REBUILD_RECEIPT_SCHEMA_VERSION,
        receipt_type=VECTOR_STORE_REBUILD_RECEIPT_MARKER,
        provider=provider,
        model=model,
        embedding_dim=embedding_dim,
        collection_name=collection_name,
        source_manifest_hash=source_manifest_hash,
        source_manifest_path=str(source_path),
        chunk_count=0,
        duration_s=round(time.monotonic() - started, 6),
        outcome="rebuilt",
        timestamp=datetime.now(timezone.utc).isoformat(),
        operator_id=operator,
        workspace_path=str(workspace),
    )
    write_vector_store_rebuild_receipt(receipt, workspace)


@app.command("promote-vision-candidate")
def promote_vision_candidate_cmd(
    task_id: str = typer.Option(
        ..., "--task", "-t", help="Vision task id (e.g. vision.slide_analysis)"
    ),
    model_id: str = typer.Option(
        ..., "--model", "-m", help="Local candidate model id to promote to approved"
    ),
    receipt: Optional[Path] = typer.Option(
        None,
        "--receipt",
        "-r",
        help=(
            "Path to the benchmark receipt JSON. Required — manual promotion "
            "without a receipt fails closed."
        ),
    ),
    registry_path: Optional[Path] = typer.Option(
        None,
        "--registry",
        help="Override the registry.yaml path (defaults to the installed registry).",
    ),
    thresholds_path: Optional[Path] = typer.Option(
        None,
        "--thresholds",
        help="Optional JSON file overriding per-metric promotion thresholds.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute the diff without writing the registry."
    ),
):
    """v1.61: Flip a local vision candidate from benchmark_pending to approved.

    Promotion is only applied when the supplied benchmark receipt clears all
    configured per-metric thresholds. No receipt → fail-closed. Receipt below
    thresholds → fail-closed.
    """
    from conference_pipeline.benchmarks.promote_vision import (
        PromotionError,
        load_thresholds,
        promote_vision_candidate,
    )

    try:
        thresholds = load_thresholds(thresholds_path) if thresholds_path else None
        diff = promote_vision_candidate(
            task_id=task_id,
            model_id=model_id,
            receipt_path=receipt,
            registry_path=registry_path,
            thresholds=thresholds,
            dry_run=dry_run,
        )
    except PromotionError as exc:
        typer.echo(f"promote-vision-candidate failed: {exc}", err=True)
        raise typer.Exit(code=2)

    typer.echo(json.dumps(diff.to_dict(), indent=2, sort_keys=True))


@app.command("promote-local-candidate")
def promote_local_candidate_cmd(
    task_id: str = typer.Option(
        ..., "--task", "-t", help="Registry task id (e.g. transcription.session_speech)"
    ),
    model_id: str = typer.Option(
        ..., "--model", "-m", help="Local candidate model id to promote to approved"
    ),
    receipt: Optional[Path] = typer.Option(
        ...,
        "--receipt",
        "-r",
        help=(
            "Path to the benchmark receipt JSON. Required — promotion "
            "without a receipt fails closed."
        ),
    ),
    research_ref: str = typer.Option(
        ...,
        "--research-ref",
        help="Proof path for source-research document (required).",
    ),
    hardware_fit_ref: str = typer.Option(
        ...,
        "--hardware-fit-ref",
        help="Proof path for hardware fitness assessment document (required).",
    ),
    operator_approval_ref: str = typer.Option(
        ...,
        "--operator-approval-ref",
        help="Proof path for operator sign-off document (required).",
    ),
    registry_path: Optional[Path] = typer.Option(
        None,
        "--registry",
        help="Override the registry.yaml path (defaults to the installed registry).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute the diff without writing the registry."
    ),
):
    """v3.0 Phase 03-01: Generic operator-gated local model promotion.

    Flip a local candidate from benchmark_pending to approved for any task family
    (transcription, vision, OCR, etc.).  All four proof refs are required:
    --receipt, --research-ref, --hardware-fit-ref, and --operator-approval-ref.

    Missing any proof ref or a receipt below thresholds causes a non-zero exit (2).
    """
    from conference_pipeline.benchmarks.promotion import (
        promote_local_candidate,
        PromotionRequiresOperatorApprovalError,
    )
    from conference_pipeline.benchmarks.promote_vision import PromotionError

    try:
        diff = promote_local_candidate(
            task_id=task_id,
            model_id=model_id,
            receipt_path=receipt,
            research_ref=research_ref,
            hardware_fit_ref=hardware_fit_ref,
            operator_approval_ref=operator_approval_ref,
            registry_path=registry_path,
            dry_run=dry_run,
        )
    except (PromotionError, PromotionRequiresOperatorApprovalError) as exc:
        typer.echo(f"promote-local-candidate failed: {exc}", err=True)
        raise typer.Exit(code=2)

    typer.echo(json.dumps(diff.to_dict(), indent=2, sort_keys=True))


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
