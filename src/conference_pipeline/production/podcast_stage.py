"""Stage 9: the focused overview podcast (script + rendered audio), shared by every runner.

Gates, does not orchestrate: (1) the receipt convention's start halt and the
resume short-circuit over an already succeeded, still-verifying receipt;
(2) fail closed, with zero spend, unless the Stage 8 receipt succeeded with a
passing contract (and, by default, passing executive-summary and
trend-report checks) and the analysis manifest, executive summary and trend
report are present; (3) the pre-spend lineage gate; (4) provider-cost
approval before either subprocess is constructed; (5) the $0 podcast source
bundle and fact ranks; (6) ``scripts/generate_focused_podcast.py`` (inside a
provider-receipt context) then ``scripts/render_podcast_audio.py`` through
an injectable list-based command runner, blocking on any failed command or
missing output -- a script-only outcome can never succeed; (7) the
audio-ready package, the provider-receipt cost summary and the convention's
success receipt, binding ``script.json`` and ``podcast.mp3`` as the owners'
execution evidence.

The conference supplies its layout and on-air title through
:class:`PodcastLayout` (default: derived from the workspace's conference
identity) and its receipt shape through the
:class:`~conference_pipeline.production.receipts.StageReceiptConvention`;
the module carries no conference literal.
"""

from __future__ import annotations

import functools
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.model_governance import default_model
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.receipts import (
    DEFAULT_STAGE_IO,
    StageIO,
    StageReceiptConvention,
    StrictStageConvention,
)
from conference_pipeline.provider_receipts import (
    provider_receipt_context,
    provider_receipt_summary,
    sanitize_for_receipt,
)
from conference_pipeline.receipt_lineage import sha256_file

STAGE_ID = "podcast"
STAGE8_ID = "qa-gap-fill-synthesis"
PODCAST_SCRIPT_MODEL_TASK_ID = "podcast.focused_script_generation"
PODCAST_TTS_TASK_ID = "podcast.tts"

SCRIPT_GENERATOR = "scripts/generate_focused_podcast.py"
AUDIO_RENDERER = "scripts/render_podcast_audio.py"
HOST_A = "Maya"
HOST_B = "Leo"
TARGET_MINUTES = "12"
TOP_N_FACTS = "50"
MAX_FACT_TIER = "3"
MIN_SCRIPT_WORDS = 100
FACT_LIMIT = 80

# Required owner ids -> the artifact whose path:digest is their execution evidence.
OWNER_FOCUSED_SCRIPT = "podcast.focused_script"
OWNER_RENDER_AUDIO = "podcast.render_audio"

REPO_ROOT = Path(__file__).resolve().parents[3]

CommandRunner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class PodcastLayout:
    """Where the overview podcast lives (workspace-relative) and what it is called."""

    source: Path
    ranks: Path
    output_dir: Path
    title: str
    source_heading: str
    audio_package_receipt_type: str
    # Also gate on the Stage 8 receipt's executive-summary / trend-report checks.
    require_stage8_output_status: bool = True

    @property
    def audio_manifest(self) -> Path:
        return self.output_dir / "audio_manifest.json"

    @property
    def audio_package(self) -> Path:
        return self.output_dir / "audio_ready_package.json"

    @classmethod
    def for_context(cls, ctx: ProductionRunnerContext) -> PodcastLayout:
        """The default layout: ``<podcast_dir>/<slug>_overview`` and the conference's display name."""

        name = ctx.conference_display_name(default=ctx.slug)
        overview = ctx.podcast_dir / f"{ctx.slug}_overview"
        return cls(
            source=ctx.analysis_manifest.parent / "podcast_source.md",
            ranks=overview / "ranks.json",
            output_dir=overview / "focused_podcast",
            title=f"{name}: What Changed in AI Research",
            source_heading=f"# {name} Podcast Source",
            audio_package_receipt_type=f"conference_pipeline.{ctx.slug}_podcast_audio_ready_package",
        )


def run_podcast_command(command: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Run a Stage 9 owner script -- ``command`` is always a list, never a shell string."""

    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def required_input_paths(ctx: ProductionRunnerContext) -> dict[str, Path]:
    """The Stage 8 receipt and the Stage 7/8 outputs the podcast gates on."""

    return {
        "stage8_receipt": ctx.receipt_path(STAGE8_ID),
        "analysis_manifest": ctx.workspace / ctx.analysis_manifest,
        "executive_summary": ctx.workspace / ctx.analysis_executive_summary,
        "trend_report": ctx.workspace / ctx.analysis_trend_report,
    }


def normalize_podcast_fact_text(text: str, *, max_chars: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip("-* ")
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "."


def podcast_paragraph_candidates(markdown: str) -> list[str]:
    candidates: list[str] = []
    for block in re.split(r"\n\s*\n", markdown):
        stripped = block.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "Source Anchors" in stripped or "sha256" in stripped:
            continue
        stripped = re.sub(r"^\s*[-*]\s+", "", stripped)
        if len(stripped.split()) < 12:
            continue
        candidates.append(normalize_podcast_fact_text(stripped))
    return candidates


def build_podcast_source_and_ranks(
    ctx: ProductionRunnerContext,
    *,
    layout: PodcastLayout,
    analysis_manifest: Mapping[str, Any],
    stage8_receipt: Mapping[str, Any],
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    """Write the narration source bundle and the ranked fact list ($0)."""

    workspace = ctx.workspace
    inputs = required_input_paths(ctx)
    executive_text = inputs["executive_summary"].read_text(encoding="utf-8")
    trend_text = inputs["trend_report"].read_text(encoding="utf-8")
    topics = analysis_manifest.get("topics")
    topic_lines: list[str] = []
    if isinstance(topics, list):
        for item in topics:
            if not isinstance(item, dict):
                continue
            topic_lines.append(
                "- {number}. {title} ({path})".format(
                    number=item.get("number") or item.get("id") or "?",
                    title=item.get("title") or item.get("id") or "Untitled topic",
                    path=item.get("markdown_path") or item.get("deep_analysis_path") or "",
                )
            )

    source_text = "\n\n".join(
        [
            layout.source_heading,
            (
                "This source bundle is generated from the passed Phase 8 executive "
                "summary and trend report. It is the narration source for the "
                "rendered overview podcast."
            ),
            "## Stage 8 Binding",
            f"- run_id: {ctx.run_id}",
            f"- stage_08_receipt_sha256: {sha256_file(inputs['stage8_receipt'])}",
            f"- analysis_manifest_content_sha256: {analysis_manifest.get('content_sha256')}",
            f"- executive_summary_sha256: {sha256_file(inputs['executive_summary'])}",
            f"- trend_report_sha256: {sha256_file(inputs['trend_report'])}",
            "## Topic Report Index",
            "\n".join(topic_lines),
            "## Executive Summary",
            executive_text,
            "## Trend Report",
            trend_text,
        ]
    )
    source_path = workspace / layout.source
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(source_text.strip() + "\n", encoding="utf-8")

    facts: list[dict[str, Any]] = []
    for tier, source_name, markdown, base_score in (
        (1, "executive_summary", executive_text, 100.0),
        (2, "trend_report", trend_text, 80.0),
    ):
        for index, text in enumerate(podcast_paragraph_candidates(markdown), start=1):
            facts.append(
                {
                    "id": f"{source_name}_{index:03d}",
                    "tier": tier,
                    "score": round(base_score - (index * 0.1), 3),
                    "text": text,
                    "source": source_name,
                }
            )
    for index, line in enumerate(topic_lines, start=1):
        facts.append(
            {
                "id": f"topic_index_{index:03d}",
                "tier": 3,
                "score": round(60.0 - (index * 0.1), 3),
                "text": normalize_podcast_fact_text(line),
                "source": "analysis_manifest.topic_index",
            }
        )
    facts.sort(key=lambda item: (int(item.get("tier") or 4), -float(item.get("score") or 0.0)))
    stage8_checks = stage8_receipt.get("checks")
    ranks_payload = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.podcast_fact_ranks",
        "generated_at": io.now(),
        "run_id": ctx.run_id,
        "source_path": str(layout.source),
        "stage8_receipt_sha256": sha256_file(inputs["stage8_receipt"]),
        "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
        "stage8_checks": {
            "executive_summary_status": (
                stage8_checks.get("executive_summary_status") if isinstance(stage8_checks, dict) else None
            ),
            "trend_report_status": stage8_checks.get("trend_report_status") if isinstance(stage8_checks, dict) else None,
        },
        "facts": facts[:FACT_LIMIT],
        "secrets_copied": False,
    }
    ranks_path = workspace / layout.ranks
    io.write_json(ranks_path, ranks_payload)
    return {
        "source_path": workspace_relpath(workspace, source_path),
        "source_sha256": sha256_file(source_path),
        "fact_ranks_path": workspace_relpath(workspace, ranks_path),
        "fact_ranks_sha256": sha256_file(ranks_path),
        "fact_count": len(ranks_payload["facts"]),
    }


def podcast_script_stats(script_path: Path) -> dict[str, Any]:
    payload = json.loads(script_path.read_text(encoding="utf-8"))
    segments = payload.get("segments") if isinstance(payload, dict) else None
    segment_count = len(segments) if isinstance(segments, list) else 0
    word_count = 0
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                word_count += len(str(segment.get("text") or "").split())
    return {
        "title": payload.get("title") if isinstance(payload, dict) else None,
        "segment_count": segment_count,
        "word_count": word_count,
    }


def write_audio_ready_package(
    ctx: ProductionRunnerContext,
    *,
    layout: PodcastLayout,
    source_binding: Mapping[str, Any],
    script_stats: Mapping[str, Any],
    audio_manifest: Mapping[str, Any],
    provider_summary: Mapping[str, Any],
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    workspace = ctx.workspace
    output_dir = layout.output_dir
    package = {
        "schema_version": 1,
        "receipt_type": layout.audio_package_receipt_type,
        "generated_at": io.now(),
        "run_id": ctx.run_id,
        "source_binding": dict(source_binding),
        "script_json_path": str(output_dir / "script.json"),
        "script_json_sha256": sha256_file(workspace / output_dir / "script.json"),
        "script_markdown_path": str(output_dir / "script.md"),
        "script_markdown_sha256": sha256_file(workspace / output_dir / "script.md"),
        "show_notes_path": str(output_dir / "show_notes.md"),
        "show_notes_sha256": sha256_file(workspace / output_dir / "show_notes.md"),
        "audio_manifest_path": str(layout.audio_manifest),
        "audio_manifest_sha256": sha256_file(workspace / layout.audio_manifest),
        "podcast_mp3_path": str(output_dir / "podcast.mp3"),
        "podcast_mp3_sha256": sha256_file(workspace / output_dir / "podcast.mp3"),
        "script_stats": dict(script_stats),
        "audio_stats": {
            "segment_count": audio_manifest.get("segment_count"),
            "audio_file_count": audio_manifest.get("audio_file_count"),
            "podcast_byte_count": audio_manifest.get("podcast_byte_count"),
        },
        "provider_summary": dict(provider_summary),
        "secrets_copied": False,
    }
    io.write_json(workspace / layout.audio_package, package)
    return package


def _input_blockers(
    ctx: ProductionRunnerContext,
    inputs: Mapping[str, Path],
    *,
    layout: PodcastLayout,
    stage8_receipt: Mapping[str, Any] | None,
    stage8_error: str | None,
    analysis_manifest: Mapping[str, Any] | None,
    analysis_manifest_error: str | None,
) -> list[dict[str, Any]]:
    workspace = ctx.workspace
    blockers: list[dict[str, Any]] = []
    if stage8_error or not isinstance(stage8_receipt, dict):
        blockers.append(
            {
                "code": "podcast_stage8_receipt_missing_or_malformed",
                "path": workspace_relpath(workspace, inputs["stage8_receipt"]),
                "error": stage8_error,
            }
        )
    elif stage8_receipt.get("status") != "succeeded":
        blockers.append({"code": "podcast_stage8_receipt_not_succeeded", "observed_status": stage8_receipt.get("status")})
    else:
        contract = stage8_receipt.get("stage_contract")
        if not isinstance(contract, dict) or contract.get("status") != "pass":
            blockers.append(
                {
                    "code": "podcast_stage8_contract_not_pass",
                    "observed_status": contract.get("status") if isinstance(contract, dict) else None,
                }
            )
        elif layout.require_stage8_output_status:
            stage8_checks = stage8_receipt.get("checks")
            stage8_checks = stage8_checks if isinstance(stage8_checks, dict) else {}
            for label in ("executive_summary_status", "trend_report_status"):
                if stage8_checks.get(label) != "pass":
                    blockers.append(
                        {"code": f"podcast_stage8_{label}_not_pass", "observed_status": stage8_checks.get(label)}
                    )
    if analysis_manifest_error or not isinstance(analysis_manifest, dict):
        blockers.append(
            {
                "code": "podcast_analysis_manifest_missing_or_malformed",
                "path": workspace_relpath(workspace, inputs["analysis_manifest"]),
                "error": analysis_manifest_error,
            }
        )
    for label in ("executive_summary", "trend_report"):
        if not inputs[label].is_file():
            blockers.append({"code": f"podcast_{label}_missing", "path": workspace_relpath(workspace, inputs[label])})
    return blockers


def podcast_commands(
    ctx: ProductionRunnerContext, *, layout: PodcastLayout, script_model: str, tts_model: str
) -> tuple[list[str], list[str]]:
    """The script-generation and audio-render owner commands (lists, never shell strings)."""

    output_dir = ctx.workspace / layout.output_dir
    script_command = [
        sys.executable,
        SCRIPT_GENERATOR,
        str(ctx.workspace / layout.source),
        "--fact-ranks",
        str(ctx.workspace / layout.ranks),
        "--output-dir",
        str(output_dir),
        "--target-minutes",
        TARGET_MINUTES,
        "--top-n-facts",
        TOP_N_FACTS,
        "--max-tier",
        MAX_FACT_TIER,
        "--model",
        script_model,
        "--two-hosts",
        "--host-a",
        HOST_A,
        "--host-b",
        HOST_B,
        "--title",
        layout.title,
    ]
    audio_command = [
        sys.executable,
        AUDIO_RENDERER,
        "--script-json",
        str(output_dir / "script.json"),
        "--output-dir",
        str(output_dir),
        "--model",
        tts_model,
        "--host-a",
        HOST_A,
        "--host-b",
        HOST_B,
        "--workspace",
        str(ctx.workspace),
        "--run-id",
        ctx.run_id,
        "--stage-id",
        STAGE_ID,
        "--summary",
    ]
    return script_command, audio_command


def run_podcast_stage(
    ctx: ProductionRunnerContext,
    *,
    convention: StageReceiptConvention | None = None,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    command_runner: CommandRunner | None = None,
    receipt_summary: Callable[..., dict[str, Any]] | None = None,
    layout: PodcastLayout | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
) -> Any:
    """Run Stage 9 under ``convention`` and return that convention's stage result.

    ``command_runner`` runs both owner commands (default: a list-based
    subprocess at the repository root); ``receipt_summary(workspace, run_id=,
    stage_id=)`` aggregates the canonical provider receipts those commands
    wrote (default: :func:`provider_receipt_summary`).
    """

    workspace = ctx.workspace
    run_id = ctx.run_id
    layout = layout or PodcastLayout.for_context(ctx)
    convention = convention or StrictStageConvention(ctx, STAGE_ID, io=io)
    ctx.run_dir().mkdir(parents=True, exist_ok=True)
    env_file_status = ctx.load_env(env_file)
    approval = ctx.provider_cost_approval(provider_cost_approved)

    halted = convention.start()
    if halted is not None:
        return halted
    if resume:
        resumed = convention.resumed()
        if resumed is not None:
            return resumed

    inputs = required_input_paths(ctx)
    stage8_receipt, stage8_error = io.load_json(inputs["stage8_receipt"])
    analysis_manifest, analysis_manifest_error = io.load_json(inputs["analysis_manifest"])
    input_blockers = _input_blockers(
        ctx,
        inputs,
        layout=layout,
        stage8_receipt=stage8_receipt,
        stage8_error=stage8_error,
        analysis_manifest=analysis_manifest,
        analysis_manifest_error=analysis_manifest_error,
    )
    base_checks: dict[str, Any] = {
        "env_file": env_file_status,
        "provider_cost_approval": approval,
        "stage8_receipt_path": workspace_relpath(workspace, inputs["stage8_receipt"]),
        "analysis_manifest_path": str(ctx.analysis_manifest),
        "executive_summary_path": str(ctx.analysis_executive_summary),
        "trend_report_path": str(ctx.analysis_trend_report),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if input_blockers:
        return convention.block(
            "podcast_required_inputs_blocked",
            reason="Phase 9 podcast required inputs are missing, stale, or failed",
            remediation="rerun Phase 8 QA/gap-fill/synthesis before podcast production",
            checks={**base_checks, "input_blockers": input_blockers},
        )
    assert isinstance(stage8_receipt, dict)
    assert isinstance(analysis_manifest, dict)
    gated = convention.lineage_gate(base_checks)
    if gated is not None:
        return gated
    if not bool(approval["approved"]):
        return convention.block(
            "podcast_provider_cost_not_approved",
            reason="Phase 9 podcast script and TTS require paid provider approval",
            remediation="rerun podcast with --provider-cost-approved after confirming budget",
            checks=base_checks,
        )

    source_binding = build_podcast_source_and_ranks(
        ctx, layout=layout, analysis_manifest=analysis_manifest, stage8_receipt=stage8_receipt, io=io
    )
    runner = command_runner or functools.partial(run_podcast_command, cwd=ctx.repo_root)
    script_model = default_model(PODCAST_SCRIPT_MODEL_TASK_ID)
    tts_model = default_model(PODCAST_TTS_TASK_ID)
    output_dir = workspace / layout.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    script_command, audio_command = podcast_commands(ctx, layout=layout, script_model=script_model, tts_model=tts_model)

    # Only the script command runs inside a provider-receipt context: the audio
    # renderer opens its own from --workspace/--run-id/--stage-id, and both
    # children inherit the receipt environment either way.
    with provider_receipt_context(workspace_root=workspace, run_id=run_id, stage_id=STAGE_ID, caller=ctx.receipt_caller):
        script_result = runner(script_command)
    if script_result.returncode != 0:
        return convention.block(
            "podcast_script_generation_failed",
            reason="Phase 9 focused podcast script generation failed",
            remediation="inspect script provider receipt and rerun podcast",
            checks={
                **base_checks,
                **source_binding,
                "script_command": script_command,
                "script_stdout": sanitize_for_receipt(script_result.stdout[-2000:]),
                "script_stderr": sanitize_for_receipt(script_result.stderr[-2000:]),
            },
        )
    script_path = output_dir / "script.json"
    script_markdown_path = output_dir / "script.md"
    show_notes_path = output_dir / "show_notes.md"
    if not script_path.is_file() or not script_markdown_path.is_file() or not show_notes_path.is_file():
        return convention.block(
            "podcast_script_outputs_missing",
            reason="Phase 9 script generation did not produce the required script outputs",
            remediation="rerun podcast script generation",
            checks={**base_checks, **source_binding},
        )
    script_stats = podcast_script_stats(script_path)
    if int(script_stats["segment_count"]) <= 0 or int(script_stats["word_count"]) < MIN_SCRIPT_WORDS:
        return convention.block(
            "podcast_script_too_short",
            reason="Phase 9 script generation returned too little narration",
            remediation="rerun podcast script generation with repaired source/ranks",
            checks={**base_checks, **source_binding, "script_stats": script_stats},
        )

    audio_result = runner(audio_command)
    if audio_result.returncode != 0:
        return convention.block(
            "podcast_audio_render_failed",
            reason="Phase 9 podcast audio render failed",
            remediation="inspect TTS provider receipts and rerun podcast",
            checks={
                **base_checks,
                **source_binding,
                "script_stats": script_stats,
                "audio_command": audio_command,
                "audio_stdout": sanitize_for_receipt(audio_result.stdout[-2000:]),
                "audio_stderr": sanitize_for_receipt(audio_result.stderr[-2000:]),
            },
        )
    audio_manifest_path = workspace / layout.audio_manifest
    audio_manifest, audio_manifest_error = io.load_json(audio_manifest_path)
    podcast_mp3_path = output_dir / "podcast.mp3"
    if (
        audio_manifest_error
        or not isinstance(audio_manifest, dict)
        or not podcast_mp3_path.is_file()
        or podcast_mp3_path.stat().st_size <= 0
    ):
        return convention.block(
            "podcast_audio_outputs_missing",
            reason="Phase 9 audio render did not produce the required MP3 and manifest",
            remediation="rerun podcast audio render",
            checks={
                **base_checks,
                **source_binding,
                "script_stats": script_stats,
                "audio_manifest_error": audio_manifest_error,
            },
        )
    provider_summary = (receipt_summary or provider_receipt_summary)(workspace, run_id=run_id, stage_id=STAGE_ID)
    audio_package = write_audio_ready_package(
        ctx,
        layout=layout,
        source_binding=source_binding,
        script_stats=script_stats,
        audio_manifest=audio_manifest,
        provider_summary=provider_summary,
        io=io,
    )
    calls_made = provider_summary.get("canonical_receipt_files_scanned", 0) > 0
    checks = {
        **base_checks,
        **source_binding,
        "stage8_receipt_sha256": sha256_file(inputs["stage8_receipt"]),
        "analysis_manifest_sha256": sha256_file(inputs["analysis_manifest"]),
        "analysis_manifest_content_sha256": analysis_manifest.get("content_sha256"),
        "executive_summary_sha256": sha256_file(inputs["executive_summary"]),
        "trend_report_sha256": sha256_file(inputs["trend_report"]),
        "podcast_script_status": "pass",
        "podcast_audio_status": "pass",
        "podcast_output_dir": str(layout.output_dir),
        "script_json_path": str(layout.output_dir / "script.json"),
        "script_json_sha256": sha256_file(script_path),
        "script_markdown_path": str(layout.output_dir / "script.md"),
        "script_markdown_sha256": sha256_file(script_markdown_path),
        "show_notes_path": str(layout.output_dir / "show_notes.md"),
        "show_notes_sha256": sha256_file(show_notes_path),
        "audio_manifest_path": str(layout.audio_manifest),
        "audio_manifest_sha256": sha256_file(audio_manifest_path),
        "audio_ready_package_path": str(layout.audio_package),
        "audio_ready_package_sha256": sha256_file(workspace / layout.audio_package),
        "podcast_mp3_path": str(layout.output_dir / "podcast.mp3"),
        "podcast_audio_sha256": sha256_file(podcast_mp3_path),
        "podcast_audio_byte_count": podcast_mp3_path.stat().st_size,
        "script_stats": script_stats,
        "audio_package": audio_package,
        "script_generation_model": script_model,
        "tts_model": tts_model,
        "model_governance_task_ids": [PODCAST_SCRIPT_MODEL_TASK_ID, PODCAST_TTS_TASK_ID],
        "script_command": script_command,
        "audio_command": audio_command,
        "provider_summary": provider_summary,
        "provider_calls_made": calls_made,
        "live_provider_calls_made": calls_made,
        "paid_provider_calls_made": calls_made,
        "estimated_cost_usd": provider_summary.get("canonical_total_estimated_cost_usd"),
        "secrets_copied": False,
    }
    owner_evidence = {
        OWNER_FOCUSED_SCRIPT: f"{checks['script_json_path']}:{checks['script_json_sha256']}",
        OWNER_RENDER_AUDIO: f"{checks['podcast_mp3_path']}:{checks['podcast_audio_sha256']}",
    }
    return convention.succeed(checks, owner_evidence=owner_evidence)
