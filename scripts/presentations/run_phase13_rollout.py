#!/usr/bin/env python3
"""Prepare Phase 13 deck rollout commands with approval guardrails.

By default this script writes an auditable plan and does not edit files. Pass
``--execute`` to run deterministic pre-TTS steps. Live NotebookLM dialogue
transformation is still gated behind ``--enable-live-notebooklm-transform``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
EXECUTIVE_DECK_KINDS = {
    "executive_summary": "executive",
    "executive_summary_short": "executive_short",
    "executive_summary_long": "executive_long",
}
APPROVED_HUMAN_REVIEW_STATUSES = frozenset({"accepted", "approved"})
APPROVED_HUMAN_REVIEW_DECISIONS = frozenset({"approved_for_bulk_rollout"})
APPROVED_CANARY_PACKET_STATUSES = frozenset({"approved_for_bulk_rollout"})
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
PHASE15_CANARY_REPAIR_DECK_ID = "04_safety_alignment_privacy"
DEFAULT_PHASE15_PHASE_DIR = (
    REPO_ROOT
    / ".planning"
    / "workstreams"
    / "v2-1"
    / "phases"
    / "15-presentation-artifact-truth-and-source-grounded-canary-repair"
)
DEFAULT_STAGE_OWNER_REGISTRY = REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json"
PHASE15_UPSTREAM_SUMMARIES = tuple(f"15-{index:02d}-SUMMARY.md" for index in range(1, 12))
PHASE15_REQUIRED_OWNER_COMMANDS = (
    "presentation.source_corpus_inventory",
    "presentation.speaker_author_work_ledger",
    "presentation.repair_presentation_sources",
    "presentation.research_cluster_plan",
    "presentation.artifact_truth_validation",
    "presentation.video_sync_validation",
)
PHASE15_NARRATION_REVIEW_GATES = (
    "conference-narration-review",
    "conference-narration-fix",
    "hash-matched review waiver",
)
PHASE15_AUDIO_RECEIPT_REQUIRED_FIELDS = (
    "script_json_sha256",
    "delivery_contract_sha256",
    "audio_segment_refs",
    "byte_count",
    "duration_seconds",
    "silence_voice_gap_summary",
    "provider_receipt_refs",
    "model_receipt_refs",
    "cost_receipt_refs",
    "production_evidence",
)
PROVIDER_ENV_KEYS = (
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "PRESENTATION_GEMINI_TIMEOUT_MS",
    "PRESENTATION_GEMINI_DIALOGUE_WORKERS",
    "PRESENTATION_GEMINI_DIALOGUE_TIMEOUT_SECONDS",
    "PRESENTATION_GEMINI_MAX_RETRIES",
)


@dataclass(frozen=True)
class DeckTarget:
    deck_id: str
    deck_dir: Path
    deck_kind: str

    @property
    def presentation_type(self) -> str:
        return "topic-deep-dive" if self.deck_kind == "topic" else "executive-summary"


@dataclass(frozen=True)
class RolloutStep:
    name: str
    argv: tuple[str, ...]
    required_inputs: tuple[Path, ...] = ()
    live_provider: bool = False
    optional: bool = False


def _isoformat_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _script(name: str) -> str:
    return str(SCRIPT_DIR / name)


def _is_topic_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) >= 3 and path.name[:2].isdigit() and path.name[2] == "_"


def discover_decks(decks_dir: Path) -> list[DeckTarget]:
    targets: list[DeckTarget] = []
    for deck_dir in sorted(path for path in decks_dir.iterdir() if path.is_dir()):
        if _is_topic_dir(deck_dir):
            targets.append(DeckTarget(deck_id=deck_dir.name, deck_dir=deck_dir, deck_kind="topic"))
            continue
        deck_kind = EXECUTIVE_DECK_KINDS.get(deck_dir.name)
        if deck_kind:
            targets.append(DeckTarget(deck_id=deck_dir.name, deck_dir=deck_dir, deck_kind=deck_kind))
    return targets


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _phase15_summary_state(phase_dir: Path) -> dict[str, Any]:
    missing = [
        summary_name
        for summary_name in PHASE15_UPSTREAM_SUMMARIES
        if not (phase_dir / summary_name).is_file()
    ]
    return {
        "phase_dir": str(phase_dir),
        "required_summaries": list(PHASE15_UPSTREAM_SUMMARIES),
        "missing_summaries": missing,
        "passed": not missing,
    }


def _phase15_owner_registry_state(registry_path: Path) -> dict[str, Any]:
    registry = _load_json_object(registry_path)
    if registry is None:
        return {
            "registry_path": str(registry_path),
            "required_owner_commands": list(PHASE15_REQUIRED_OWNER_COMMANDS),
            "missing_owner_commands": list(PHASE15_REQUIRED_OWNER_COMMANDS),
            "passed": False,
        }
    sections = {
        name: registry.get(name) if isinstance(registry.get(name), dict) else {}
        for name in ("scripts", "task_bindings")
    }
    missing = [
        task_id
        for task_id in PHASE15_REQUIRED_OWNER_COMMANDS
        if any(task_id not in section for section in sections.values())
    ]
    return {
        "registry_path": str(registry_path),
        "required_owner_commands": list(PHASE15_REQUIRED_OWNER_COMMANDS),
        "missing_owner_commands": missing,
        "passed": not missing,
    }


def _phase15_pre_repair_state(report_path: Path, *, deck_dir: Path) -> dict[str, Any]:
    report = _load_json_object(report_path)
    if report is None:
        return {
            "path": str(report_path),
            "present": report_path.exists(),
            "passed": False,
            "pre_repair_status": "missing_or_invalid",
            "blocker_count": 0,
            "topic4_path_bound": False,
        }
    status = str(report.get("status") or "").strip().lower()
    blockers = report.get("blockers") if isinstance(report.get("blockers"), list) else []
    deck_id = deck_dir.name
    topic4_path_bound = any(
        isinstance(blocker, dict) and deck_id in str(blocker.get("path") or "")
        for blocker in blockers
    )
    return {
        "path": str(report_path),
        "present": True,
        "passed": status == "fail" and bool(blockers) and topic4_path_bound,
        "pre_repair_status": status,
        "blocker_count": len(blockers),
        "topic4_path_bound": topic4_path_bound,
    }


def phase15_canary_repair_gate(
    *,
    targets: list[DeckTarget],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.phase15_canary_repair:
        return None
    selected_deck_ids = [target.deck_id for target in targets]
    gate: dict[str, Any] = {
        "mode": "phase15_canary_repair",
        "allowed_deck_id": PHASE15_CANARY_REPAIR_DECK_ID,
        "selected_deck_ids": selected_deck_ids,
        "narration_review_gates": list(PHASE15_NARRATION_REVIEW_GATES),
        "audio_receipt_required_fields": list(PHASE15_AUDIO_RECEIPT_REQUIRED_FIELDS),
    }
    if args.canary_deck_id != PHASE15_CANARY_REPAIR_DECK_ID or selected_deck_ids != [
        PHASE15_CANARY_REPAIR_DECK_ID
    ]:
        return {**gate, "status": "blocked_phase15_canary_scope"}

    target = targets[0]
    summary_state = _phase15_summary_state(args.phase15_phase_dir.resolve())
    gate["summaries"] = summary_state
    if not summary_state["passed"]:
        return {**gate, "status": "blocked_missing_phase15_summaries"}

    owner_state = _phase15_owner_registry_state(args.stage_owner_registry.resolve())
    gate.update(owner_state)
    if not owner_state["passed"]:
        return {**gate, "status": "blocked_missing_phase15_owner_commands"}

    pre_report_path = (
        args.pre_repair_report.resolve()
        if args.pre_repair_report
        else target.deck_dir / "artifact_truth_pre_repair_report.json"
    )
    pre_state = _phase15_pre_repair_state(pre_report_path, deck_dir=target.deck_dir)
    gate.update(pre_state)
    if not pre_state["passed"]:
        return {**gate, "status": "blocked_pre_repair_artifact_truth_not_failed"}

    return {**gate, "status": "passed"}


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key not in PROVIDER_ENV_KEYS:
            continue
        values[key] = value.strip().strip('"').strip("'")
    return values


def provider_env_status(env_file: Path) -> dict[str, Any]:
    loaded = load_env_file(env_file)
    merged_present = {
        key: bool(os.environ.get(key) or loaded.get(key))
        for key in ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    }
    return {
        "env_file": str(env_file),
        "env_file_exists": env_file.exists(),
        "loaded_key_count": len(loaded),
        "has_google_or_gemini_key": any(merged_present.values()),
        "required_key_presence": merged_present,
        "values_recorded": False,
    }


def _phase15_blocked_report(
    *,
    workspace: Path,
    args: argparse.Namespace,
    gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/run_phase13_rollout.py",
        "status": gate["status"],
        "workspace": str(workspace),
        "execute": args.execute,
        "enable_live_notebooklm_transform": args.enable_live_notebooklm_transform,
        "provider_environment": provider_env_status(args.env_file),
        "canary_deck_id": args.canary_deck_id,
        "phase15_canary_repair_gate": gate,
        "decks": [],
    }


def receipt_is_approved(receipt_path: Path) -> bool:
    receipt = _load_json_object(receipt_path)
    if receipt is None:
        return False
    status = str(receipt.get("status", "")).strip().lower()
    decision = str(receipt.get("decision", "")).strip().lower()
    return (
        status in APPROVED_HUMAN_REVIEW_STATUSES
        or decision in APPROVED_HUMAN_REVIEW_DECISIONS
    )


def canary_packet_state(packet_path: Path | None, *, canary_deck_id: str) -> dict[str, Any]:
    if packet_path is None:
        return {
            "path": None,
            "present": False,
            "approved": False,
            "status": "not_provided",
            "objective_evidence_status": None,
            "contact_sheet_present": False,
            "deck_id": None,
        }
    packet = _load_json_object(packet_path)
    if packet is None:
        return {
            "path": str(packet_path),
            "present": packet_path.exists(),
            "approved": False,
            "status": "missing_or_invalid",
            "objective_evidence_status": None,
            "contact_sheet_present": False,
            "deck_id": None,
        }
    objective_evidence = packet.get("objective_evidence_report")
    if not isinstance(objective_evidence, dict):
        objective_evidence = {}
    status = str(packet.get("status", "unknown"))
    deck_id = str(packet.get("deck_id", ""))
    objective_status = str(objective_evidence.get("status", "unknown"))
    contact_sheet_present = bool(objective_evidence.get("contact_sheet_present"))
    deck_matches = deck_id == canary_deck_id
    approved = (
        status in APPROVED_CANARY_PACKET_STATUSES
        and deck_matches
        and objective_status == "passed"
        and contact_sheet_present
        and not packet.get("blockers")
    )
    return {
        "path": str(packet_path),
        "present": True,
        "approved": approved,
        "status": status,
        "objective_evidence_status": objective_status,
        "contact_sheet_present": contact_sheet_present,
        "deck_id": deck_id,
        "deck_matches": deck_matches,
        "blockers": packet.get("blockers", []),
        "human_review_approved": packet.get("human_review", {}).get("approved")
        if isinstance(packet.get("human_review"), dict)
        else None,
    }


def _workspace_rel(deck_id: str, *parts: str) -> str:
    return (Path("decks") / deck_id / Path(*parts)).as_posix()


def build_deck_steps(
    target: DeckTarget,
    *,
    workspace: Path,
    conference_label: str,
    enable_live_notebooklm_transform: bool,
    dialogue_workers: int,
) -> list[RolloutStep]:
    deck_id = target.deck_id
    deck_dir = target.deck_dir
    slides_md = deck_dir / "5_draft_slides_with_images.md"
    presenter_md = deck_dir / "6_presenter_script.md"
    asset_inventory = deck_dir / "asset_inventory.json"
    storyboard = deck_dir / "editorial_storyboard.json"
    steps: list[RolloutStep] = []

    if target.deck_kind == "topic":
        steps.append(
            RolloutStep(
                name="build_editorial_storyboard",
                argv=(
                    sys.executable,
                    _script("build_editorial_storyboard.py"),
                    "--deck-dir",
                    str(deck_dir),
                    "--output",
                    str(storyboard),
                ),
                required_inputs=(slides_md,),
            )
        )
    steps.append(
        RolloutStep(
            name="repair_figure_metadata",
            argv=(
                sys.executable,
                _script("repair_slide_figure_metadata.py"),
                "--slides-md",
                str(slides_md),
                "--asset-inventory",
                str(asset_inventory),
                "--report-out",
                str(deck_dir / "figure_metadata_repair_report.json"),
            ),
            required_inputs=(slides_md, asset_inventory),
        )
    )
    if target.deck_kind == "topic":
        steps.append(
            RolloutStep(
                name="apply_storyboard_zoom_policy",
                argv=(
                    sys.executable,
                    _script("apply_editorial_storyboard_to_script.py"),
                    "--script-md",
                    str(presenter_md),
                    "--storyboard",
                    str(storyboard),
                    "--report-out",
                    str(deck_dir / "editorial_script_policy_report.json"),
                ),
                required_inputs=(presenter_md, storyboard),
            )
        )
    steps.extend(
        [
            RolloutStep(
                name="add_title_slide",
                argv=(
                    sys.executable,
                    _script("add_title_slide_to_slides_md.py"),
                    "--workspace",
                    str(workspace),
                    "--presentation-id",
                    deck_id,
                    "--presentation-type",
                    target.presentation_type,
                    "--slides-md",
                    _workspace_rel(deck_id, "5_draft_slides_with_images.md"),
                    "--conference-label",
                    conference_label,
                ),
                required_inputs=(slides_md,),
            ),
            RolloutStep(
                name="render_revealjs",
                argv=(
                    sys.executable,
                    _script("md_to_revealjs.py"),
                    "--workspace",
                    str(workspace),
                    "--presentation-id",
                    deck_id,
                    "--presentation-type",
                    target.presentation_type,
                    "--slides-md",
                    _workspace_rel(deck_id, "5_draft_slides_with_images.md"),
                    "--output-html",
                    _workspace_rel(deck_id, "presentation.html"),
                    "--slide-mapping",
                    _workspace_rel(deck_id, "presentation_slide_mapping.json"),
                    "--require-brand-assets",
                ),
                required_inputs=(slides_md,),
            ),
            RolloutStep(
                name="build_presenter_script_json",
                argv=(
                    sys.executable,
                    _script("build_presenter_video.py"),
                    "--workspace",
                    str(workspace),
                    "--presentation-id",
                    deck_id,
                    "--presentation-type",
                    target.presentation_type,
                    "--script-md",
                    _workspace_rel(deck_id, "6_presenter_script.md"),
                    "--slides-html",
                    _workspace_rel(deck_id, "presentation.html"),
                    "--slide-mapping",
                    _workspace_rel(deck_id, "presentation_slide_mapping.json"),
                    "--build-dir",
                    _workspace_rel(deck_id, "video_build"),
                    "--output",
                    _workspace_rel(deck_id, "presentation_video.mp4"),
                    "--stop-after-script-json",
                ),
                required_inputs=(presenter_md, deck_dir / "presentation.html"),
            ),
            RolloutStep(
                name="transform_notebooklm_script_json",
                argv=(
                    sys.executable,
                    _script("build_notebooklm_style_presenter_video.py"),
                    "--workspace",
                    str(workspace),
                    "--presentation-id",
                    deck_id,
                    "--presentation-type",
                    target.presentation_type,
                    "--script-md",
                    _workspace_rel(deck_id, "6_presenter_script.md"),
                    "--slides-html",
                    _workspace_rel(deck_id, "presentation.html"),
                    "--slide-mapping",
                    _workspace_rel(deck_id, "presentation_slide_mapping.json"),
                    "--build-dir",
                    _workspace_rel(deck_id, "video_build_notebooklm"),
                    "--output",
                    _workspace_rel(deck_id, "presentation_video_notebooklm.mp4"),
                    "--podcast-script-out",
                    _workspace_rel(deck_id, "6_presenter_script_notebooklm_2host.md"),
                    "--duration-reference-script-json",
                    _workspace_rel(deck_id, "video_build", "script.json"),
                    "--conference-label",
                    conference_label,
                    "--turns-per-slide",
                    "2",
                    "--max-turn-chars",
                    "220",
                    "--char-budget-scale",
                    "0.55",
                    "--force-transform",
                    "--stop-after-transform",
                    "--dialogue-workers",
                    str(dialogue_workers),
                ),
                required_inputs=(
                    presenter_md,
                    deck_dir / "presentation.html",
                    deck_dir / "video_build" / "script.json",
                ),
                live_provider=True,
                optional=not enable_live_notebooklm_transform,
            ),
            RolloutStep(
                name="lint_presenter_narration",
                argv=(
                    sys.executable,
                    _script("lint_presenter_narration.py"),
                    "--script-md",
                    str(presenter_md),
                    "--script-json",
                    str(deck_dir / "video_build" / "script.json"),
                    "--deck-kind",
                    target.deck_kind,
                    "--topic-id",
                    deck_id,
                    "--conference-label",
                    conference_label,
                    "--report-out",
                    str(deck_dir / "presenter_narration_lint_report.json"),
                ),
                required_inputs=(presenter_md, deck_dir / "video_build" / "script.json"),
            ),
            RolloutStep(
                name="review_presenter_narration",
                argv=(
                    sys.executable,
                    _script("review_presenter_narration.py"),
                    "--script-md",
                    str(presenter_md),
                    "--script-json",
                    str(deck_dir / "video_build" / "script.json"),
                    "--lint-report",
                    str(deck_dir / "presenter_narration_lint_report.json"),
                    "--deck-kind",
                    target.deck_kind,
                    "--topic-id",
                    deck_id,
                    "--conference-label",
                    conference_label,
                    "--review-out",
                    str(deck_dir / "presenter_narration_review.md"),
                    "--json-out",
                    str(deck_dir / "presenter_narration_review.json"),
                ),
                required_inputs=(
                    presenter_md,
                    deck_dir / "video_build" / "script.json",
                    deck_dir / "presenter_narration_lint_report.json",
                ),
            ),
            RolloutStep(
                name="lint_notebooklm_narration",
                argv=(
                    sys.executable,
                    _script("lint_narration_script.py"),
                    "--script-md",
                    str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
                    "--source-script-md",
                    str(presenter_md),
                    "--deck-kind",
                    target.deck_kind,
                    "--topic-id",
                    deck_id,
                    "--conference-label",
                    conference_label,
                    "--report-out",
                    str(deck_dir / "narration_lint_report.json"),
                ),
                required_inputs=(deck_dir / "6_presenter_script_notebooklm_2host.md", presenter_md),
            ),
            RolloutStep(
                name="review_notebooklm_narration",
                argv=(
                    sys.executable,
                    _script("review_narration_script.py"),
                    "--script-md",
                    str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
                    "--source-script-md",
                    str(presenter_md),
                    "--lint-report",
                    str(deck_dir / "narration_lint_report.json"),
                    "--deck-kind",
                    target.deck_kind,
                    "--topic-id",
                    deck_id,
                    "--conference-label",
                    conference_label,
                    "--review-out",
                    str(deck_dir / "narration_review.md"),
                    "--json-out",
                    str(deck_dir / "narration_review.json"),
                ),
                required_inputs=(
                    deck_dir / "6_presenter_script_notebooklm_2host.md",
                    presenter_md,
                    deck_dir / "narration_lint_report.json",
                ),
            ),
            RolloutStep(
                name="run_editorial_gate",
                argv=(
                    sys.executable,
                    _script("presentation_editorial_gate.py"),
                    "--deck-dir",
                    str(deck_dir),
                    "--deck-kind",
                    target.deck_kind,
                    "--report-out",
                    str(deck_dir / "presentation_editorial_gate_report.json"),
                ),
                required_inputs=(slides_md, presenter_md),
            ),
        ]
    )
    return steps


def build_series_review_steps(
    targets: list[DeckTarget],
    *,
    workspace: Path,
    conference_label: str,
) -> list[RolloutStep]:
    decks_dir = workspace / "decks"
    presenter_command: list[str] = [
        sys.executable,
        _script("review_narration_series.py"),
        "--script-name",
        "6_presenter_script.md",
        "--conference-label",
        conference_label,
        "--report-out",
        str(decks_dir / "presenter_narration_series_review.json"),
        "--markdown-out",
        str(decks_dir / "presenter_narration_series_review.md"),
    ]
    notebook_command: list[str] = [
        sys.executable,
        _script("review_narration_series.py"),
        "--script-name",
        "6_presenter_script_notebooklm_2host.md",
        "--conference-label",
        conference_label,
        "--report-out",
        str(decks_dir / "narration_series_review.json"),
        "--markdown-out",
        str(decks_dir / "narration_series_review.md"),
    ]
    presenter_inputs: list[Path] = []
    notebook_inputs: list[Path] = []
    for target in targets:
        presenter_path = target.deck_dir / "6_presenter_script.md"
        notebook_path = target.deck_dir / "6_presenter_script_notebooklm_2host.md"
        presenter_command.extend(["--script-md", str(presenter_path)])
        notebook_command.extend(["--script-md", str(notebook_path)])
        presenter_inputs.append(presenter_path)
        notebook_inputs.append(notebook_path)
    return [
        RolloutStep(
            name="review_presenter_narration_series",
            argv=tuple(presenter_command),
            required_inputs=tuple(presenter_inputs),
        ),
        RolloutStep(
            name="review_notebooklm_narration_series",
            argv=tuple(notebook_command),
            required_inputs=tuple(notebook_inputs),
        ),
    ]


def _step_to_report(step: RolloutStep, *, status: str, exit_code: int | None = None, stderr: str = "") -> dict[str, Any]:
    missing_inputs = [str(path) for path in step.required_inputs if not path.exists()]
    return {
        "name": step.name,
        "status": status,
        "exit_code": exit_code,
        "command": list(step.argv),
        "missing_inputs": missing_inputs,
        "live_provider": step.live_provider,
        "optional": step.optional,
        "stderr_tail": stderr[-2000:] if stderr else "",
    }


def run_step(
    step: RolloutStep,
    *,
    execute: bool,
    allow_live_provider: bool,
    provider_env: dict[str, str],
) -> dict[str, Any]:
    if not execute:
        return _step_to_report(step, status="planned")
    missing_inputs = [path for path in step.required_inputs if not path.exists()]
    if missing_inputs:
        return _step_to_report(
            step,
            status="skipped_missing_inputs" if step.optional else "failed_missing_inputs",
        )
    if step.live_provider and not allow_live_provider:
        return _step_to_report(step, status="skipped_live_provider")
    command_env = os.environ.copy()
    command_env.update(provider_env)
    completed = subprocess.run(
        step.argv,
        cwd=Path(__file__).resolve().parent.parent.parent,
        check=False,
        capture_output=True,
        text=True,
        env=command_env,
    )
    status = "passed" if completed.returncode == 0 else "failed"
    return _step_to_report(
        step,
        status=status,
        exit_code=completed.returncode,
        stderr=completed.stderr,
    )


def _select_targets(
    *,
    decks_dir: Path,
    deck_ids: list[str],
    all_decks: bool,
) -> list[DeckTarget]:
    targets = discover_decks(decks_dir)
    if all_decks:
        return targets
    by_id = {target.deck_id: target for target in targets}
    selected: list[DeckTarget] = []
    missing: list[str] = []
    for deck_id in deck_ids:
        target = by_id.get(deck_id)
        if target is None:
            missing.append(deck_id)
        else:
            selected.append(target)
    if missing:
        raise ValueError(f"unknown deck id(s): {', '.join(missing)}")
    return selected


def build_report(
    *,
    workspace: Path,
    targets: list[DeckTarget],
    args: argparse.Namespace,
) -> dict[str, Any]:
    canary_receipt = workspace / "decks" / args.canary_deck_id / "human_review_receipt.json"
    canary_approved = receipt_is_approved(canary_receipt)
    canary_packet = canary_packet_state(
        args.canary_review_packet,
        canary_deck_id=args.canary_deck_id,
    )
    provider_status = provider_env_status(args.env_file)
    provider_env = load_env_file(args.env_file)
    phase15_gate = phase15_canary_repair_gate(targets=targets, args=args)
    if phase15_gate and phase15_gate["status"] != "passed":
        return _phase15_blocked_report(workspace=workspace, args=args, gate=phase15_gate)
    if args.execute and args.enable_live_notebooklm_transform and not provider_status["has_google_or_gemini_key"]:
        return {
            "schema_version": 1,
            "generated_at": _isoformat_z(),
            "generator": "scripts/presentations/run_phase13_rollout.py",
            "status": "blocked_missing_provider_env",
            "workspace": str(workspace),
            "execute": args.execute,
            "enable_live_notebooklm_transform": args.enable_live_notebooklm_transform,
            "provider_environment": provider_status,
            "canary_deck_id": args.canary_deck_id,
            "canary_receipt": str(canary_receipt),
            "canary_approved": canary_approved,
            "canary_review_packet": canary_packet,
            "decks": [],
        }
    non_canary_targets = [target.deck_id for target in targets if target.deck_id != args.canary_deck_id]
    if args.require_approved_canary and non_canary_targets and not canary_approved:
        return {
            "schema_version": 1,
            "generated_at": _isoformat_z(),
            "generator": "scripts/presentations/run_phase13_rollout.py",
            "status": "blocked_pending_canary_human_review",
            "workspace": str(workspace),
            "execute": args.execute,
            "enable_live_notebooklm_transform": args.enable_live_notebooklm_transform,
            "provider_environment": provider_status,
            "canary_deck_id": args.canary_deck_id,
            "canary_receipt": str(canary_receipt),
            "canary_approved": False,
            "canary_review_packet": canary_packet,
            "block_reason": "canary_human_review_not_approved",
            "blocked_decks": non_canary_targets,
            "decks": [],
        }
    if args.require_approved_canary and non_canary_targets and not canary_packet["approved"]:
        return {
            "schema_version": 1,
            "generated_at": _isoformat_z(),
            "generator": "scripts/presentations/run_phase13_rollout.py",
            "status": "blocked_pending_canary_review_packet",
            "workspace": str(workspace),
            "execute": args.execute,
            "enable_live_notebooklm_transform": args.enable_live_notebooklm_transform,
            "provider_environment": provider_status,
            "canary_deck_id": args.canary_deck_id,
            "canary_receipt": str(canary_receipt),
            "canary_approved": canary_approved,
            "canary_review_packet": canary_packet,
            "block_reason": "canary_review_packet_not_approved",
            "blocked_decks": non_canary_targets,
            "decks": [],
        }

    deck_reports: list[dict[str, Any]] = []
    any_failed = False
    for target in targets:
        steps = build_deck_steps(
            target,
            workspace=workspace,
            conference_label=args.conference_label,
            enable_live_notebooklm_transform=args.enable_live_notebooklm_transform,
            dialogue_workers=args.dialogue_workers,
        )
        step_reports = [
            run_step(
                step,
                execute=args.execute,
                allow_live_provider=args.enable_live_notebooklm_transform,
                provider_env=provider_env,
            )
            for step in steps
        ]
        if any(step["status"].startswith("failed") for step in step_reports):
            any_failed = True
        any_skipped_live = any(step["status"] == "skipped_live_provider" for step in step_reports)
        deck_reports.append(
            {
                "deck_id": target.deck_id,
                "deck_kind": target.deck_kind,
                "deck_dir": str(target.deck_dir),
                "status": (
                    "failed"
                    if any(step["status"].startswith("failed") for step in step_reports)
                    else "planned"
                    if not args.execute
                    else "partial"
                    if any_skipped_live
                    else "passed"
                ),
                "steps": step_reports,
            }
        )

    series_reviews = [
        run_step(
            step,
            execute=args.execute,
            allow_live_provider=False,
            provider_env=provider_env,
        )
        for step in build_series_review_steps(
            targets,
            workspace=workspace,
            conference_label=args.conference_label,
        )
    ]
    if any(step["status"].startswith("failed") for step in series_reviews):
        any_failed = True

    if any_failed:
        status = "failed"
    elif not args.execute:
        status = "planned"
    elif any(deck["status"] == "partial" for deck in deck_reports):
        status = "partial"
    else:
        status = "passed"
    return {
        "schema_version": 1,
        "generated_at": _isoformat_z(),
        "generator": "scripts/presentations/run_phase13_rollout.py",
        "status": status,
        "workspace": str(workspace),
        "execute": args.execute,
        "enable_live_notebooklm_transform": args.enable_live_notebooklm_transform,
        "provider_environment": provider_status,
        "canary_deck_id": args.canary_deck_id,
        "canary_receipt": str(canary_receipt),
        "canary_approved": canary_approved,
        "canary_review_packet": canary_packet,
        "phase15_canary_repair_gate": phase15_gate,
        "decks": deck_reports,
        "series_reviews": series_reviews,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 13 Rollout Plan",
        "",
        f"Generated: `{report.get('generated_at', '')}`",
        f"Status: `{report.get('status', 'unknown')}`",
        f"Workspace: `{report.get('workspace', '')}`",
        f"Execute: `{report.get('execute', False)}`",
        f"Live NotebookLM transform: `{report.get('enable_live_notebooklm_transform', False)}`",
        f"Canary approved: `{report.get('canary_approved', False)}`",
    ]
    canary_packet = report.get("canary_review_packet")
    if isinstance(canary_packet, dict):
        lines.extend(
            [
                f"Canary packet approved: `{canary_packet.get('approved', False)}`",
                f"Canary packet status: `{canary_packet.get('status', 'unknown')}`",
                f"Canary objective evidence: `{canary_packet.get('objective_evidence_status', 'unknown')}`",
            ]
        )
    provider_environment = report.get("provider_environment")
    if isinstance(provider_environment, dict):
        lines.extend(
            [
                "",
                "## Provider Environment",
                "",
                f"Env file exists: `{provider_environment.get('env_file_exists', False)}`",
                f"Has Google/Gemini key: `{provider_environment.get('has_google_or_gemini_key', False)}`",
                "Secret values recorded: `False`",
            ]
        )
    blocked = report.get("blocked_decks")
    if isinstance(blocked, list) and blocked:
        lines.extend(["", "## Blocked Decks", ""])
        for deck_id in blocked:
            lines.append(f"- `{deck_id}`")
    decks = report.get("decks", [])
    if isinstance(decks, list) and decks:
        lines.extend(
            [
                "",
                "## Decks",
                "",
                "| Deck | Kind | Status | Planned | Passed | Skipped | Failed |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for deck in decks:
            steps = deck.get("steps", []) if isinstance(deck, dict) else []
            statuses = [step.get("status", "") for step in steps if isinstance(step, dict)]
            lines.append(
                "| {deck_id} | {kind} | {status} | {planned} | {passed} | {skipped} | {failed} |".format(
                    deck_id=deck.get("deck_id", ""),
                    kind=deck.get("deck_kind", ""),
                    status=deck.get("status", ""),
                    planned=sum(status == "planned" for status in statuses),
                    passed=sum(status == "passed" for status in statuses),
                    skipped=sum(status.startswith("skipped") for status in statuses),
                    failed=sum(status.startswith("failed") for status in statuses),
                )
            )
    series_reviews = report.get("series_reviews", [])
    if isinstance(series_reviews, list) and series_reviews:
        lines.extend(
            [
                "",
                "## Series Reviews",
                "",
                "| Review | Status | Planned | Passed | Skipped | Failed |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for review in series_reviews:
            if not isinstance(review, dict):
                continue
            status = str(review.get("status", ""))
            lines.append(
                "| {name} | {status} | {planned} | {passed} | {skipped} | {failed} |".format(
                    name=review.get("name", ""),
                    status=status,
                    planned=1 if status == "planned" else 0,
                    passed=1 if status == "passed" else 0,
                    skipped=1 if status.startswith("skipped") else 0,
                    failed=1 if status.startswith("failed") else 0,
                )
            )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True, help="Conference workspace root.")
    parser.add_argument("--deck-id", action="append", default=[], help="Deck id to process; repeatable.")
    parser.add_argument("--all", action="store_true", help="Process every topic and executive deck.")
    parser.add_argument(
        "--conference-label",
        default=None,
        help=(
            "Conference label for title slides and NotebookLM transformation. "
            "Defaults to --workspace's own config.yaml conference.name."
        ),
    )
    parser.add_argument("--canary-deck-id", default="04_safety_alignment_privacy")
    parser.add_argument(
        "--canary-review-packet",
        type=Path,
        help=(
            "Approved Phase 13 canary packet JSON. Required for non-canary "
            "decks when --require-approved-canary is set."
        ),
    )
    parser.add_argument(
        "--require-approved-canary",
        action="store_true",
        help="Block non-canary decks unless canary human_review_receipt.json records approval.",
    )
    parser.add_argument("--execute", action="store_true", help="Run deterministic pre-TTS steps.")
    parser.add_argument(
        "--enable-live-notebooklm-transform",
        action="store_true",
        help="Run NotebookLM dialogue transformation. This can invoke a live provider.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Canonical provider env file loaded before live NotebookLM transform.",
    )
    parser.add_argument("--dialogue-workers", type=int, default=4)
    parser.add_argument(
        "--phase15-canary-repair",
        action="store_true",
        help=(
            "Fail closed unless the selected deck is 04_safety_alignment_privacy, "
            "the Phase 15 upstream summaries exist, Phase 15 owner commands are "
            "registered, and artifact_truth_pre_repair_report.json has status fail."
        ),
    )
    parser.add_argument(
        "--phase15-phase-dir",
        type=Path,
        default=DEFAULT_PHASE15_PHASE_DIR,
        help="Phase 15 directory used to verify upstream summaries before canary repair.",
    )
    parser.add_argument(
        "--stage-owner-registry",
        type=Path,
        default=DEFAULT_STAGE_OWNER_REGISTRY,
        help="Stage owner registry used to verify Phase 15 owner commands.",
    )
    parser.add_argument(
        "--pre-repair-report",
        type=Path,
        help="Pre-repair artifact truth report; defaults to the selected canary deck report.",
    )
    parser.add_argument("--report-out", type=Path, help="JSON report path.")
    parser.add_argument("--markdown-out", type=Path, help="Markdown dashboard output path.")
    args = parser.parse_args(argv)
    if not args.all and not args.deck_id:
        parser.error("provide --all or at least one --deck-id")
    if args.all and args.deck_id:
        parser.error("use --all or --deck-id, not both")
    if args.conference_label is None:
        # Never fall back to a specific conference's hardcoded label -- read
        # it from the workspace that was actually passed.
        args.conference_label = conference_name(args.workspace, default="the conference")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    decks_dir = workspace / "decks"
    if not decks_dir.is_dir():
        print(f"decks directory not found: {decks_dir}", file=sys.stderr)
        return 2
    try:
        targets = _select_targets(decks_dir=decks_dir, deck_ids=args.deck_id, all_decks=args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = build_report(workspace=workspace, targets=targets, args=args)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    if report["status"] == "blocked_missing_provider_env":
        print("phase13 rollout blocked: missing provider env for live transform", file=sys.stderr)
        return 2
    if report["status"] == "blocked_pending_canary_human_review":
        print("phase13 rollout blocked: canary human review is not approved", file=sys.stderr)
        return 2
    if report["status"] == "blocked_pending_canary_review_packet":
        print("phase13 rollout blocked: canary review packet is not approved", file=sys.stderr)
        return 2
    if str(report["status"]).startswith("blocked_phase15") or report["status"] in {
        "blocked_missing_phase15_summaries",
        "blocked_missing_phase15_owner_commands",
        "blocked_pre_repair_artifact_truth_not_failed",
    }:
        print(f"phase15 canary repair blocked: {report['status']}", file=sys.stderr)
        return 2
    if report["status"] == "failed":
        print("phase13 rollout failed", file=sys.stderr)
        return 1
    print(f"phase13 rollout: {report['status']} ({len(report['decks'])} deck(s))", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
