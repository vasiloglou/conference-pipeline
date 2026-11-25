"""Command planning for manifest-backed presentation workflow stages."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from conference_pipeline.presentation.blueprint import require_approved_blueprint
from conference_pipeline.presentation.legacy_markdown_gate import (
    reject_legacy_markdown_in_production_mode,
)
from conference_pipeline.presentation.manifest import MANIFEST_RELATIVE_PATH


_PATH_ESCAPE_ERROR = "presentation artifact path must stay inside the conference workspace"
_MISSING_INPUT_ERROR = "Required presentation input artifact is missing:"
_MATCHED_PAPERS_ARTIFACT_REF = "storage/paper-matching/<run-id>/<topic>/matched_papers.json"
_EXTRACTION_TOPIC_ARTIFACT_REF = "storage/extractions/<run-id>/by-topic/<topic>.json"
_UNSUPPORTED_REASONS = {
    "human_review": "human_review is operator-gated and not a script command",
}
_UNSUPPORTED_OWNER_REFS: dict[str, str] = {}
_OWNER_OUTPUT_ARTIFACT_REFS: dict[str, dict[str, tuple[str, ...]]] = {
    "presentation_blueprint": {
        "scripts/presentations/generate_editorial_blueprint.py": (
            "decks/<presentation>/presentation_blueprint.json",
            "decks/<presentation>/presentation_blueprint.md",
            "decks/<presentation>/presentation_blueprint_receipt.json",
        ),
        "scripts/presentations/review_presentation_blueprint.py": (
            "decks/<presentation>/blueprint_review.json",
            "decks/<presentation>/blueprint_review.md",
        ),
    },
    "deck_contract_generation": {
        "scripts/presentations/generate_deck_contract.py": (
            "decks/<presentation>/presentation_deck.json",
            "decks/<presentation>/presentation_deck_receipt.json",
        ),
    },
    "source_binding": {
        "scripts/presentations/build_deck_source_pack.py": (
            "decks/<presentation>/ranked_papers.json",
            "decks/<presentation>/source_pack_receipt.json",
        ),
        "scripts/presentations/bind_presentation_sources.py": (
            "decks/<presentation>/source_binding_report.json",
        ),
    },
    "visual_plan_generation": {
        "scripts/presentations/generate_programmatic_figures.py": (
            "decks/<presentation>/visual_plan.json",
            "decks/<presentation>/asset_inventory.json",
            "decks/<presentation>/assets/generated/figure_specs.json",
            "decks/<presentation>/assets/generated/semantic_review.json",
            "decks/<presentation>/assets/generated/semantic_review.md",
            "decks/<presentation>/assets/generated/",
        ),
    },
    "render_spec_generation": {
        "scripts/presentations/generate_render_spec.py": (
            "decks/<presentation>/render_spec.json",
        ),
    },
    "render_export": {
        "scripts/presentations/md_to_revealjs.py": (
            "decks/<presentation>/presentation.html",
            "decks/<presentation>/presentation_slide_mapping.json",
        ),
        "scripts/presentations/md_to_pptx.py": (
            "decks/<presentation>/presentation.pdf",
        ),
    },
    "delivery_generation": {
        "scripts/presentations/generate_presenter_script.py": (
            "decks/<presentation>/delivery_single_speaker.json",
            "decks/<presentation>/delivery_notebooklm.json",
            "decks/<presentation>/6_presenter_script.md",
        ),
    },
    "presentation_review": {
        "scripts/presentations/review_presentation.py": (
            "decks/<presentation>/presentation_review.json",
            "decks/<presentation>/presentation_review.md",
            "decks/<presentation>/presentation_review.prompt.md",
        ),
    },
    "video_generation": {
        "scripts/presentations/build_presenter_video.py": (
            "decks/<presentation>/video_build/",
            "decks/<presentation>/presentation_video.mp4",
        ),
        "scripts/presentations/build_notebooklm_style_presenter_video.py": (
            "decks/<presentation>/video_build_notebooklm/",
            "decks/<presentation>/presentation_video_notebooklm.mp4",
        ),
    },
    "video_sync_validation": {
        "scripts/presentations/validate_video_sync.py": (
            "decks/<presentation>/video_sync_validation_single_speaker.json",
            "decks/<presentation>/video_sync_validation_notebooklm.json",
        ),
        "scripts/presentations/validate_video_delivery.py": (
            "decks/<presentation>/video_delivery_validation.json",
        ),
    },
}


@dataclass(frozen=True)
class PresentationCommandPlan:
    stage_id: str
    owner_ref: str
    argv: list[str]
    selected_scope: dict[str, object]
    input_artifacts: list[dict[str, object]]
    output_artifacts: list[dict[str, object]]
    support_level: str
    provider_requirements: list[str]
    mode_readiness: dict[str, str]
    unsupported_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "owner_ref": self.owner_ref,
            "argv": list(self.argv),
            "selected_scope": dict(self.selected_scope),
            "input_artifacts": [dict(artifact) for artifact in self.input_artifacts],
            "output_artifacts": [dict(artifact) for artifact in self.output_artifacts],
            "support_level": self.support_level,
            "provider_requirements": list(self.provider_requirements),
            "mode_readiness": dict(self.mode_readiness),
            "unsupported_reason": self.unsupported_reason,
        }


def build_presentation_command_plans(
    workspace_root: Path,
    selection: object,
    *,
    python_executable: str,
    mode: str = "dry-run",
    run_id: str | None = None,
) -> list[PresentationCommandPlan]:
    """Build argv and receipt-facing metadata for one selected presentation stage."""

    root = workspace_root.resolve()
    _require_conference_workspace(root)
    _require_manifest_file(root, selection)

    stage_spec = selection.stage_spec
    reject_legacy_markdown_in_production_mode(
        [*stage_spec.input_artifacts, *stage_spec.output_artifacts],
        context=f"command_plan.{stage_spec.id}",
        support_level=stage_spec.support_level,
        production_evidence=stage_spec.support_level not in {"legacy-mock-only", "debug-export-only"},
    )
    selected_scope = selection.selected_scope.to_dict()
    presentation_id = str(selected_scope["presentation_id"])
    presentation_type = str(selected_scope["presentation_type"])
    display_title = str(selection.presentation.title)
    topic = selected_scope.get("topic")
    paper_topic_id = _paper_matching_topic_id(topic, presentation_id)
    deck_dir = _artifact_path(root, f"decks/{presentation_id}", presentation_id)
    common_args = _common_scope_args(root, presentation_id, presentation_type, topic)
    input_artifact_refs = _input_artifact_refs_for_mode(stage_spec.id, stage_spec.input_artifacts, mode)
    if stage_spec.id in {"deck_contract_generation", "source_binding", "render_spec_generation"}:
        selected_scope = {**selected_scope, "artifact_mode": "structured-json"}
    input_artifacts = _artifact_metadata(
        root,
        input_artifact_refs,
        presentation_id,
        run_id=run_id,
        topic_id=paper_topic_id,
        kind="input",
    )
    output_artifacts = _artifact_metadata(
        root,
        stage_spec.output_artifacts,
        presentation_id,
        run_id=run_id,
        topic_id=paper_topic_id,
        kind="output",
    )

    unsupported_reason = _UNSUPPORTED_REASONS.get(stage_spec.id)
    if unsupported_reason is not None:
        owner_ref = _UNSUPPORTED_OWNER_REFS.get(stage_spec.id, stage_spec.owner_refs[0])
        return [
            _plan(
                stage_spec,
                owner_ref=owner_ref,
                argv=[
                    "presentation-stage-unsupported",
                    stage_spec.id,
                    *common_args,
                ],
                selected_scope=selected_scope,
                input_artifacts=input_artifacts,
                output_artifacts=[],
                unsupported_reason=unsupported_reason,
            )
        ]

    builders = {
        "presentation_blueprint": _build_presentation_blueprint,
        "deck_contract_generation": _build_deck_contract_generation,
        "source_binding": _build_source_binding,
        "visual_plan_generation": _build_visual_plan_generation,
        "render_spec_generation": _build_render_spec_generation,
        "render_export": _build_render_export,
        "visual_qc": _build_visual_qc,
        "delivery_generation": _build_delivery_generation,
        "presentation_review": _build_presentation_review,
        "video_generation": _build_video_generation,
        "video_sync_validation": _build_video_sync_validation,
    }
    try:
        builder = builders[stage_spec.id]
    except KeyError as exc:
        raise ValueError(f"Unsupported presentation command stage: {stage_spec.id}") from exc

    _require_input_artifacts(
        root,
        input_artifact_refs,
        presentation_id,
        run_id=run_id,
        topic_id=paper_topic_id,
    )
    builder_kwargs = {
        "root": root,
        "deck_dir": deck_dir,
        "presentation_id": presentation_id,
        "presentation_type": presentation_type,
        "topic": topic,
        "paper_topic_id": paper_topic_id,
        "run_id": run_id,
        "python_executable": python_executable,
        "common_args": common_args,
        "mode": mode,
    }
    if stage_spec.id == "visual_plan_generation":
        builder_kwargs["target_svgs_per_slide"] = selection.presentation.target_svgs_per_slide
    commands = builder(**builder_kwargs)
    return [
        _plan(
            stage_spec,
            owner_ref=owner_ref,
            argv=argv,
            selected_scope=selected_scope,
            input_artifacts=input_artifacts,
            output_artifacts=_owner_output_artifacts(
                root,
                stage_spec.id,
                owner_ref,
                presentation_id,
                run_id=run_id,
                topic_id=paper_topic_id,
                stage_output_artifacts=output_artifacts,
            ),
            unsupported_reason=None,
        )
        for owner_ref, argv in commands
    ]


def _plan(
    stage_spec: Any,
    *,
    owner_ref: str,
    argv: list[str],
    selected_scope: dict[str, object],
    input_artifacts: list[dict[str, object]],
    output_artifacts: list[dict[str, object]],
    unsupported_reason: str | None,
) -> PresentationCommandPlan:
    return PresentationCommandPlan(
        stage_id=stage_spec.id,
        owner_ref=owner_ref,
        argv=argv,
        selected_scope=selected_scope,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        support_level=stage_spec.support_level,
        provider_requirements=list(stage_spec.provider_requirements),
        mode_readiness=stage_spec.mode_readiness.to_dict(),
        unsupported_reason=unsupported_reason,
    )


def _require_conference_workspace(root: Path) -> None:
    """Refuse a root that is not a conference workspace.

    Every deck path is built as ``root / "decks/<id>/..."``, and the writers
    create what is missing.  A root that is merely *plausible* therefore gets
    a deck tree rather than an error: running with ``--workspace
    conferences/<slug>`` from inside ``conferences/<slug>`` resolved to
    ``conferences/<slug>/conferences/<slug>`` and quietly grew a second deck
    tree there, stranding a deck's ``presentation_review.json`` where no gate
    would ever read it (found 2026-09-20, written 2026-09-12).  A workspace
    always carries ``config.yaml``; nothing else here requires it, so this is
    the one place that can tell a workspace from a look-alike directory.
    """

    if not (root / "config.yaml").exists():
        raise ValueError(
            f"Not a conference workspace (no config.yaml): {root}. "
            "Pass --workspace as a path relative to the repository root, and "
            "run from the repository root rather than from inside the workspace."
        )


def _require_manifest_file(root: Path, selection: object) -> None:
    manifest_path = getattr(selection.loaded_manifest, "manifest_path", None)
    expected = root / MANIFEST_RELATIVE_PATH
    if manifest_path is None or not Path(manifest_path).exists():
        raise ValueError(f"Missing {MANIFEST_RELATIVE_PATH}: {expected}")


def _common_scope_args(
    root: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
) -> list[str]:
    args = [
        "--workspace",
        str(root),
        "--presentation-id",
        presentation_id,
        "--presentation-type",
        presentation_type,
    ]
    if topic is not None:
        args.extend(["--topic", str(topic)])
    return args


def _artifact_path(root: Path, artifact_ref: str, presentation_id: str) -> Path:
    relative_path = _artifact_relative_path(artifact_ref, presentation_id)
    return _resolve_workspace_relative_path(root, relative_path)


def _artifact_relative_path(
    artifact_ref: str,
    presentation_id: str,
    *,
    run_id: str | None = None,
    topic_id: str | None = None,
) -> str:
    relative_path = artifact_ref.replace("<presentation>", presentation_id)
    if "<run-id>" in relative_path:
        if run_id is None:
            raise ValueError("Presentation paper matching input requires workflow run_id")
        relative_path = relative_path.replace("<run-id>", run_id)
    if "<topic>" in relative_path:
        if topic_id is None:
            raise ValueError("Presentation paper matching input requires topic id")
        relative_path = relative_path.replace("<topic>", topic_id)
    if relative_path == "analysis/NN_slug.md":
        return f"analysis/{presentation_id}.md"
    if relative_path.startswith("presentations/"):
        raise ValueError(f"legacy presentations/ artifact refs are unsupported: {relative_path}")
    if "*" in relative_path:
        return relative_path
    _validate_relative_path(relative_path)
    return relative_path


def _validate_relative_path(relative_path: str) -> None:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(_PATH_ESCAPE_ERROR)


def _resolve_workspace_relative_path(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(_PATH_ESCAPE_ERROR) from exc
    return resolved


def _workspace_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _paper_matching_topic_id(topic: object, presentation_id: str) -> str:
    if topic is not None and str(topic).strip():
        return str(topic).strip()
    return presentation_id


def _artifact_metadata(
    root: Path,
    artifact_refs: tuple[str, ...],
    presentation_id: str,
    *,
    run_id: str | None = None,
    topic_id: str | None = None,
    kind: str,
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for artifact_ref in artifact_refs:
        relative_path = _artifact_relative_path(
            artifact_ref,
            presentation_id,
            run_id=run_id,
            topic_id=topic_id,
        )
        if artifact_ref == _MATCHED_PAPERS_ARTIFACT_REF and run_id and topic_id:
            relative_path = _resolve_matched_papers_relative_path(root, run_id, topic_id)
        elif artifact_ref == _EXTRACTION_TOPIC_ARTIFACT_REF and run_id and topic_id:
            relative_path = _resolve_extraction_topic_relative_path(root, run_id, topic_id)
        artifacts.append(
            {
                "kind": kind,
                "path": relative_path,
                "absolute_path": str(_metadata_path(root, relative_path)),
            }
        )
    return artifacts


def _owner_output_artifacts(
    root: Path,
    stage_id: str,
    owner_ref: str,
    presentation_id: str,
    *,
    run_id: str | None = None,
    topic_id: str | None = None,
    stage_output_artifacts: list[dict[str, object]],
) -> list[dict[str, object]]:
    owner_refs = _OWNER_OUTPUT_ARTIFACT_REFS.get(stage_id)
    if owner_refs is None:
        return [dict(artifact) for artifact in stage_output_artifacts]
    artifact_refs = owner_refs.get(owner_ref)
    if artifact_refs is None:
        return []
    return _artifact_metadata(
        root,
        artifact_refs,
        presentation_id,
        run_id=run_id,
        topic_id=topic_id,
        kind="output",
    )


def _metadata_path(root: Path, relative_path: str) -> Path:
    if "*" not in relative_path:
        return _resolve_workspace_relative_path(root, relative_path)
    prefix = relative_path.split("*", 1)[0].rstrip("/")
    return _resolve_workspace_relative_path(root, prefix)


def _require_input_artifacts(
    root: Path,
    artifact_refs: tuple[str, ...],
    presentation_id: str,
    *,
    run_id: str | None = None,
    topic_id: str | None = None,
) -> None:
    for artifact_ref in artifact_refs:
        relative_path = _artifact_relative_path(
            artifact_ref,
            presentation_id,
            run_id=run_id,
            topic_id=topic_id,
        )
        if artifact_ref == _MATCHED_PAPERS_ARTIFACT_REF and run_id and topic_id:
            relative_path = _resolve_matched_papers_relative_path(root, run_id, topic_id)
        elif artifact_ref == _EXTRACTION_TOPIC_ARTIFACT_REF and run_id and topic_id:
            relative_path = _resolve_extraction_topic_relative_path(root, run_id, topic_id)
        if "*" in relative_path:
            parent_prefix = relative_path.split("*", 1)[0].rstrip("/")
            parent = _resolve_workspace_relative_path(root, parent_prefix)
            if not parent.exists() or not any(parent.glob(Path(relative_path).name)):
                raise ValueError(f"{_MISSING_INPUT_ERROR} {relative_path}")
            continue
        path = _resolve_workspace_relative_path(root, relative_path)
        if not path.exists():
            raise ValueError(f"{_MISSING_INPUT_ERROR} {relative_path}")


def _require_file(root: Path, relative_path: str) -> Path:
    path = _resolve_workspace_relative_path(root, relative_path)
    if not path.exists():
        raise ValueError(f"{_MISSING_INPUT_ERROR} {relative_path}")
    return path


def _matched_papers_relative_path(run_id: str, topic_id: str) -> str:
    return f"storage/paper-matching/{run_id}/{topic_id}/matched_papers.json"


def _extraction_topic_relative_path(run_id: str, topic_id: str) -> str:
    return f"storage/extractions/{run_id}/by-topic/{topic_id}.json"


def _analysis_provenance_run_id(root: Path, topic_id: str) -> str | None:
    provenance_path = _resolve_workspace_relative_path(
        root,
        f"analysis/provenance/{topic_id}.json",
    )
    if not provenance_path.exists():
        return None
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    run_id = payload.get("run_id") if isinstance(payload, dict) else None
    if isinstance(run_id, str) and run_id.strip():
        return run_id.strip()
    return None


def _resolve_matched_papers_relative_path(root: Path, run_id: str, topic_id: str) -> str:
    candidates = [_matched_papers_relative_path(run_id, topic_id)]
    provenance_run_id = _analysis_provenance_run_id(root, topic_id)
    if provenance_run_id and provenance_run_id != run_id:
        candidates.append(_matched_papers_relative_path(provenance_run_id, topic_id))
    for relative_path in candidates:
        if _resolve_workspace_relative_path(root, relative_path).exists():
            return relative_path
    return candidates[0]


def _resolve_extraction_topic_relative_path(root: Path, run_id: str, topic_id: str) -> str:
    candidates = [_extraction_topic_relative_path(run_id, topic_id)]
    provenance_run_id = _analysis_provenance_run_id(root, topic_id)
    if provenance_run_id and provenance_run_id != run_id:
        candidates.append(_extraction_topic_relative_path(provenance_run_id, topic_id))
    for relative_path in candidates:
        if _resolve_workspace_relative_path(root, relative_path).exists():
            return relative_path
    return candidates[0]


def _deck_analysis_file(root: Path, presentation_id: str) -> Path:
    slide_brief_rel = f"analysis/slide_briefs/{presentation_id}.md"
    slide_brief = _resolve_workspace_relative_path(root, slide_brief_rel)
    if slide_brief.exists():
        return slide_brief
    return _require_file(root, f"analysis/{presentation_id}.md")


def _script_argv(
    python_executable: str,
    owner_ref: str,
    common_args: list[str],
    *args: str,
) -> list[str]:
    return [python_executable, owner_ref, *common_args, *args]


def _dry_run_args(mode: str) -> list[str]:
    return [] if mode == "live" else ["--dry-run"]


def _input_artifact_refs_for_mode(
    stage_id: str,
    input_artifacts: tuple[str, ...],
    mode: str,
) -> tuple[str, ...]:
    if stage_id == "draft_generation" and mode != "live":
        return ("analysis/NN_slug.md",)
    return input_artifacts


def _live_sonnet_figure_args(mode: str) -> list[str]:
    return ["--live-sonnet-specs"] if mode == "live" else []


def _conference_label(root: Path) -> str:
    config_path = root / "config.yaml"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            payload = {}
        conference = payload.get("conference") if isinstance(payload, dict) else None
        name = conference.get("name") if isinstance(conference, dict) else None
        if isinstance(name, str) and name.strip():
            return name.strip()
    return root.name.replace("_", " ").replace("-", " ").title()


def _presentation_final_report_url(root: Path) -> str:
    config_path = root / "config.yaml"
    if config_path.exists():
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            payload = {}
        urls = payload.get("urls") if isinstance(payload, dict) else None
        if isinstance(urls, dict):
            for key in ("final_report_url", "final_report", "analysis_report_url", "analysis_report"):
                value = urls.get(key)
                if isinstance(value, str) and value.startswith(("https://", "http://")):
                    return value
    return "https://drive.google.com/"


# Public names for the shared production stages (the underscored forms are
# kept for this module's own callers).
conference_label = _conference_label
presentation_final_report_url = _presentation_final_report_url


def _relationalai_template_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for env_var in ("RAI_PRESENTATION_TEMPLATE", "RELATIONALAI_PRESENTATION_TEMPLATE"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            root / "decks" / "templates" / "RelationalAI.pptx",
            root / "decks" / "templates" / "RelationalAI-template.pptx",
            root / "templates" / "RelationalAI.pptx",
            root / "templates" / "RelationalAI-template.pptx",
            Path.home() / "Downloads" / "RelationalAI.pptx",
            Path.home() / "Downloads" / "RelationalAI-template.pptx",
        ]
    )
    cloud_root = Path.home() / "Library" / "CloudStorage"
    if cloud_root.exists():
        candidates.extend(cloud_root.glob("GoogleDrive-*/My Drive/RelationalAI.pptx"))
        candidates.extend(cloud_root.glob("GoogleDrive-*/My Drive/RelationalAI-template.pptx"))
    return candidates


def _relationalai_template_arg(root: Path) -> list[str]:
    for candidate in _relationalai_template_candidates(root):
        if candidate.exists():
            return ["--template", str(candidate.resolve())]
    return []


def _build_presentation_blueprint(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    analysis_md = _deck_analysis_file(root, presentation_id)
    ledger_path = _require_file(root, f"decks/{presentation_id}/speaker_author_work_ledger.json")
    exhaustive_papers = _require_file(
        root,
        f"analysis/exhaustive_summaries/{presentation_id}__papers.json",
    )
    presentation_frame = _require_file(root, f"decks/{presentation_id}/presentation_frame.json")
    generator_owner_ref = "scripts/presentations/generate_editorial_blueprint.py"
    review_owner_ref = "scripts/presentations/review_presentation_blueprint.py"
    return [
        (
            generator_owner_ref,
            _script_argv(
                python_executable,
                generator_owner_ref,
                common_args,
                "--analysis-md",
                str(analysis_md),
                "--speaker-author-work-ledger",
                str(ledger_path),
                "--exhaustive-papers",
                str(exhaustive_papers),
                "--presentation-frame",
                str(presentation_frame),
                "--final-report-url",
                _presentation_final_report_url(root),
                "--output-json",
                str(deck_dir / "presentation_blueprint.json"),
                "--output-md",
                str(deck_dir / "presentation_blueprint.md"),
                "--receipt-path",
                str(deck_dir / "presentation_blueprint_receipt.json"),
                "--provider-cost-approved",
            ),
        ),
        (
            review_owner_ref,
            _script_argv(
                python_executable,
                review_owner_ref,
                common_args,
                "--presentation-blueprint",
                str(deck_dir / "presentation_blueprint.json"),
                "--blueprint-receipt",
                str(deck_dir / "presentation_blueprint_receipt.json"),
                "--blueprint-review",
                str(deck_dir / "blueprint_review.json"),
                "--output-dir",
                str(deck_dir),
                *_dry_run_args(mode),
            ),
        ),
    ]


def _build_deck_contract_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_deck_contract.py"
    blueprint_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint.json")
    receipt_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint_receipt.json")
    review_path = _require_file(root, f"decks/{presentation_id}/blueprint_review.json")
    if mode == "live":
        require_approved_blueprint(
            blueprint_path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=True,
        )
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--presentation-blueprint",
                str(blueprint_path),
                "--blueprint-receipt",
                str(receipt_path),
                "--blueprint-review",
                str(review_path),
                "--output",
                str(deck_dir / "presentation_deck.json"),
                "--receipt-output",
                str(deck_dir / "presentation_deck_receipt.json"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_source_binding(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic
    if run_id is None:
        raise ValueError("source_binding requires workflow run_id for matched_papers.json")
    deck_contract = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_deck.json")
    matched_papers = _require_file(
        root,
        _resolve_matched_papers_relative_path(root, run_id, paper_topic_id),
    )
    extraction_topic = _require_file(
        root,
        _resolve_extraction_topic_relative_path(root, run_id, paper_topic_id),
    )
    extraction_run_id = Path(extraction_topic).parents[1].name
    source_pack_owner = "scripts/presentations/build_deck_source_pack.py"
    binding_owner = "scripts/presentations/bind_presentation_sources.py"
    return [
        (
            source_pack_owner,
            _script_argv(
                python_executable,
                source_pack_owner,
                common_args,
                "--extraction-run-id",
                extraction_run_id,
                "--output-dir",
                str(deck_dir),
                *_dry_run_args(mode),
            ),
        ),
        (
            binding_owner,
            _script_argv(
                python_executable,
                binding_owner,
                common_args,
                "--deck-contract",
                str(deck_contract),
                "--matched-papers",
                str(matched_papers),
                "--source-metadata",
                str(extraction_topic),
                "--source-pack",
                str(deck_dir / "source_pack_receipt.json"),
                "--output",
                str(deck_dir / "source_binding_report.json"),
                *_dry_run_args(mode),
            ),
        ),
    ]


def _build_visual_plan_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
    target_svgs_per_slide: float,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_programmatic_figures.py"
    deck_contract = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_deck.json")
    source_binding = _require_file(root, f"{_workspace_relative(root, deck_dir)}/source_binding_report.json")
    source_pack = _require_file(root, f"{_workspace_relative(root, deck_dir)}/source_pack_receipt.json")
    ranked_papers = _require_file(root, f"{_workspace_relative(root, deck_dir)}/ranked_papers.json")
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--deck-contract",
                str(deck_contract),
                "--source-binding-report",
                str(source_binding),
                "--source-pack",
                str(source_pack),
                "--ranked-papers",
                str(ranked_papers),
                "--visual-plan",
                str(deck_dir / "visual_plan.json"),
                "--asset-inventory",
                str(deck_dir / "asset_inventory.json"),
                "--generated-dir",
                str(deck_dir / "assets" / "generated"),
                "--figure-specs-output",
                str(deck_dir / "assets" / "generated" / "figure_specs.json"),
                "--target-svgs-per-slide",
                f"{target_svgs_per_slide:.2f}",
                *_live_sonnet_figure_args(mode),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_render_spec_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_render_spec.py"
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--deck-contract",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_deck.json")),
                "--source-binding-report",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/source_binding_report.json")),
                "--visual-plan",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/visual_plan.json")),
                "--asset-inventory",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/asset_inventory.json")),
                "--output",
                str(deck_dir / "render_spec.json"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_render_export(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    render_spec = _require_file(root, f"{_workspace_relative(root, deck_dir)}/render_spec.json")
    reveal_owner = "scripts/presentations/md_to_revealjs.py"
    pdf_owner = "scripts/presentations/md_to_pptx.py"
    return [
        (
            reveal_owner,
            _script_argv(
                python_executable,
                reveal_owner,
                common_args,
                "--render-spec",
                str(render_spec),
                "--output-html",
                str(deck_dir / "presentation.html"),
                "--slide-mapping",
                str(deck_dir / "presentation_slide_mapping.json"),
                "--require-brand-assets",
                *_relationalai_template_arg(root),
                *_dry_run_args(mode),
            ),
        ),
        (
            pdf_owner,
            _script_argv(
                python_executable,
                pdf_owner,
                common_args,
                "--render-spec",
                str(render_spec),
                "--output-pdf",
                str(deck_dir / "presentation.pdf"),
                *_dry_run_args(mode),
            ),
        ),
    ]


def _build_delivery_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_presenter_script.py"
    render_spec = _require_file(root, f"{_workspace_relative(root, deck_dir)}/render_spec.json")
    slide_mapping = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_slide_mapping.json")
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--render-spec",
                str(render_spec),
                "--slide-mapping",
                str(slide_mapping),
                "--conference-label",
                _conference_label(root),
                "--delivery-single-speaker",
                str(deck_dir / "delivery_single_speaker.json"),
                "--delivery-notebooklm",
                str(deck_dir / "delivery_notebooklm.json"),
                "--debug-script-md",
                str(deck_dir / "6_presenter_script.md"),
                "--debug-notebooklm-script-md",
                str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_presentation_review(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/review_presentation.py"
    deck_relative = _workspace_relative(root, deck_dir)
    requirements = _require_file(root, f"decks/{presentation_id}/presentation_blueprint.json")
    blueprint_review = _require_file(root, f"{deck_relative}/blueprint_review.json")
    deck_contract = _require_file(root, f"{deck_relative}/presentation_deck.json")
    render_spec = _require_file(root, f"{deck_relative}/render_spec.json")
    source_binding = _require_file(root, f"{deck_relative}/source_binding_report.json")
    delivery_contract = _require_file(root, f"{deck_relative}/delivery_notebooklm.json")
    video_receipt = _require_file(root, f"{deck_relative}/video_sync_validation_notebooklm.json")
    visual_qc_report = _require_file(
        root,
        f"{deck_relative}/assets/generated/visual_llm_qc_report.json",
    )
    contact_sheet = _require_file(root, f"{deck_relative}/render_contact_sheet.png")
    narration_script = _require_file(root, f"{deck_relative}/6_presenter_script.md")
    screenshots_dir = _require_file(
        root,
        f"{deck_relative}/assets/generated/qc/screenshots",
    )
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--requirements",
                str(requirements),
                "--narration-script",
                str(narration_script),
                "--screenshots-dir",
                str(screenshots_dir),
                "--blueprint",
                str(requirements),
                "--blueprint-review",
                str(blueprint_review),
                "--deck-contract",
                str(deck_contract),
                "--render-spec",
                str(render_spec),
                "--source-binding",
                str(source_binding),
                "--delivery-contract",
                str(delivery_contract),
                "--video-receipt",
                str(video_receipt),
                "--visual-qc-report",
                str(visual_qc_report),
                "--contact-sheet",
                str(contact_sheet),
                "--report-output",
                str(deck_dir / "presentation_review.json"),
                "--markdown-output",
                str(deck_dir / "presentation_review.md"),
                "--prompt-output",
                str(deck_dir / "presentation_review.prompt.md"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_video_sync_validation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/validate_video_sync.py"
    commands: list[tuple[str, list[str]]] = [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--render-spec",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/render_spec.json")),
                "--delivery-single-speaker",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/delivery_single_speaker.json")),
                "--delivery-notebooklm",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/delivery_notebooklm.json")),
                "--single-speaker-video",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_video.mp4")),
                "--notebooklm-video",
                str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_video_notebooklm.mp4")),
                "--single-speaker-report",
                str(deck_dir / "video_sync_validation_single_speaker.json"),
                "--notebooklm-report",
                str(deck_dir / "video_sync_validation_notebooklm.json"),
                *_dry_run_args(mode),
            ),
        )
    ]
    if mode == "live":
        # Path-based delivery checks ($0) on both final MP4s, the same step the
        # shared production chain runs after video sync. The presentation
        # stage's runtime contract requires decks/*/video_delivery_validation.json
        # (registry, 2026-09-27), so a command plan without it can never pass
        # the stage contract. The owner takes no scope args and has no
        # --dry-run mode, so it is only planned for live runs.
        delivery_ref = "scripts/presentations/validate_video_delivery.py"
        commands.append(
            (
                delivery_ref,
                [
                    python_executable,
                    delivery_ref,
                    "--deck-dir",
                    str(deck_dir),
                    "--presentation-id",
                    presentation_id,
                    "--video",
                    str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_video.mp4")),
                    "--video",
                    str(_require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_video_notebooklm.mp4")),
                    "--report-out",
                    str(deck_dir / "video_delivery_validation.json"),
                ],
            )
        )
    return commands


def _build_draft_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    display_title: str,
    target_slide_count: int,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_slides_agent.py"
    output_slides = deck_dir / "1_draft_slides.md"
    if mode == "live":
        blueprint_path = _require_file(
            root,
            f"decks/{presentation_id}/presentation_blueprint.json",
        )
        receipt_path = _require_file(
            root,
            f"decks/{presentation_id}/presentation_blueprint_receipt.json",
        )
        review_path = _require_file(
            root,
            f"decks/{presentation_id}/blueprint_review.json",
        )
        require_approved_blueprint(
            blueprint_path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=True,
        )
        return [
            (
                owner_ref,
                _script_argv(
                    python_executable,
                    owner_ref,
                    common_args,
                    "--display-title",
                    display_title,
                    "--from-blueprint",
                    "--presentation-blueprint",
                    str(blueprint_path),
                    "--blueprint-receipt",
                    str(receipt_path),
                    "--blueprint-review",
                    str(review_path),
                    "--output-dir",
                    str(deck_dir),
                    "--output-slides-md",
                    str(output_slides),
                    "--slides",
                    str(target_slide_count),
                ),
            )
        ]
    analysis_md = _deck_analysis_file(root, presentation_id)
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--display-title",
                display_title,
                "--analysis-md",
                str(analysis_md),
                "--output-dir",
                str(deck_dir),
                "--output-slides-md",
                str(output_slides),
                "--slides",
                str(target_slide_count),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_research_enrichment(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
    target_hyperlinks_per_slide: float,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic
    draft = deck_dir / "1_draft_slides.md"
    if run_id is None:
        raise ValueError("research_enrichment requires workflow run_id for matched_papers.json")
    matched_papers = _require_file(
        root,
        _resolve_matched_papers_relative_path(root, run_id, paper_topic_id),
    )
    extraction_topic = _require_file(
        root,
        _resolve_extraction_topic_relative_path(root, run_id, paper_topic_id),
    )
    extraction_run_id = Path(extraction_topic).parents[1].name
    source_attributions = deck_dir / "source_attributions.json"
    blueprint_args: list[str] = []
    if mode == "live":
        blueprint_path = _require_file(
            root,
            f"decks/{presentation_id}/presentation_blueprint.json",
        )
        receipt_path = _require_file(
            root,
            f"decks/{presentation_id}/presentation_blueprint_receipt.json",
        )
        review_path = _require_file(
            root,
            f"decks/{presentation_id}/blueprint_review.json",
        )
        require_approved_blueprint(
            blueprint_path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=True,
        )
        blueprint_args = [
            "--presentation-blueprint",
            str(blueprint_path),
            "--blueprint-review",
            str(review_path),
        ]
    steps = [
        (
            "scripts/presentations/build_deck_source_pack.py",
            [
                "--extraction-run-id",
                extraction_run_id,
                "--output-dir",
                str(deck_dir),
                *_dry_run_args(mode),
            ],
        ),
        (
            "scripts/presentations/enrich_slides_with_papers.py",
            [
                "--draft-slides",
                str(draft),
                "--matched-papers",
                str(matched_papers),
                *blueprint_args,
                "--output-slides",
                str(deck_dir / "2_draft_slides_enriched_with_papers.md"),
                *_dry_run_args(mode),
            ],
        ),
        (
            "scripts/presentations/reflow_slides.py",
            [
                "--input-slides",
                str(deck_dir / "2_draft_slides_enriched_with_papers.md"),
                "--output-slides",
                str(deck_dir / "3_draft_slides_reflowed.md"),
                *_dry_run_args(mode),
            ],
        ),
        (
            "scripts/presentations/sanity_check_slides.py",
            [
                "--input-slides",
                str(deck_dir / "3_draft_slides_reflowed.md"),
                "--output-slides",
                str(deck_dir / "4_draft_slides_sanity_check.md"),
                "--report-json",
                str(deck_dir / "sanity_check_report.json"),
                *_dry_run_args(mode),
            ],
        ),
        (
            "scripts/presentations/enrich_hyperlinks_hybrid.py",
            [
                "--input-slides",
                str(deck_dir / "4_draft_slides_sanity_check.md"),
                "--source-attributions",
                str(source_attributions),
                "--matched-papers",
                str(matched_papers),
                *blueprint_args,
                "--target-links-per-slide",
                f"{target_hyperlinks_per_slide:.2f}",
                "--output-slides",
                str(deck_dir / "4.5_draft_slides_with_hyperlinks.md"),
                *_dry_run_args(mode),
            ],
        ),
    ]
    return [
        (owner_ref, _script_argv(python_executable, owner_ref, common_args, *args))
        for owner_ref, args in steps
    ]


def _build_slide_consolidation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/consolidate_slides.py"
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--input-slides",
                str(deck_dir / "4.5_draft_slides_with_hyperlinks.md"),
                "--output-slides",
                str(deck_dir / "3.5_draft_slides_consolidated.md"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_figure_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
    target_svgs_per_slide: float,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_programmatic_figures.py"
    input_slides = deck_dir / "3.5_draft_slides_consolidated.md"
    deck_relative = _workspace_relative(root, deck_dir)
    source_pack = _require_file(root, f"{deck_relative}/source_pack_receipt.json")
    ranked_papers = _require_file(root, f"{deck_relative}/ranked_papers.json")
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--input-slides",
                str(input_slides),
                "--source-pack",
                str(source_pack),
                "--ranked-papers",
                str(ranked_papers),
                "--asset-inventory",
                str(deck_dir / "asset_inventory.json"),
                "--output-slides",
                str(deck_dir / "5_draft_slides_with_images.md"),
                "--generated-dir",
                str(deck_dir / "assets" / "generated"),
                "--figure-specs-output",
                str(deck_dir / "assets" / "generated" / "figure_specs.json"),
                "--target-svgs-per-slide",
                f"{target_svgs_per_slide:.2f}",
                *_live_sonnet_figure_args(mode),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_figure_validation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del root, presentation_id, presentation_type, topic, paper_topic_id, run_id, mode
    structural_owner_ref = "scripts/presentations/validate_svgs.py"
    semantic_owner_ref = "scripts/presentations/review_programmatic_figures.py"
    generated_dir = deck_dir / "assets" / "generated"
    return [
        (
            structural_owner_ref,
            _script_argv(
                python_executable,
                structural_owner_ref,
                common_args,
                "--dir",
                str(generated_dir),
                "--fix",
                "--no-backup",
                "--json",
                "--report-output",
                str(generated_dir / "validation_report.md"),
            ),
        ),
        (
            semantic_owner_ref,
            _script_argv(
                python_executable,
                semantic_owner_ref,
                common_args,
                "--dir",
                str(generated_dir),
                "--asset-inventory",
                str(deck_dir / "asset_inventory.json"),
                "--report-output",
                str(generated_dir / "semantic_review.json"),
                "--markdown-output",
                str(generated_dir / "semantic_review.md"),
            ),
        ),
    ]


def _build_revealjs_render(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    display_title: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id, mode
    title_owner_ref = "scripts/presentations/add_title_slide_to_slides_md.py"
    render_owner_ref = "scripts/presentations/md_to_revealjs.py"
    slides_md = deck_dir / "5_draft_slides_with_images.md"
    return [
        (
            title_owner_ref,
            _script_argv(
                python_executable,
                title_owner_ref,
                common_args,
                "--slides-md",
                str(slides_md),
                "--display-title",
                display_title,
                "--conference-label",
                _conference_label(root),
            ),
        ),
        (
            render_owner_ref,
            _script_argv(
                python_executable,
                render_owner_ref,
                common_args,
                "--slides-md",
                str(slides_md),
                "--output-html",
                str(deck_dir / "presentation.html"),
                "--slide-mapping",
                str(deck_dir / "presentation_slide_mapping.json"),
                "--require-brand-assets",
                *_relationalai_template_arg(root),
            ),
        )
    ]


def _build_visual_qc(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del root, presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/inspect_rendered_slides.py"
    llm_owner_ref = "scripts/presentations/inspect_rendered_slides_llm.py"
    screenshots_dir = deck_dir / "assets" / "generated" / "qc" / "screenshots"
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--input-html",
                str(deck_dir / "presentation.html"),
                "--report-output",
                str(deck_dir / "assets" / "generated" / "visual_qc_report.md"),
                "--screenshots-dir",
                str(screenshots_dir),
                *_dry_run_args(mode),
            ),
        ),
        (
            llm_owner_ref,
            _script_argv(
                python_executable,
                llm_owner_ref,
                common_args,
                "--screenshots-dir",
                str(screenshots_dir),
                "--report-output",
                str(deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_presenter_script(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/generate_presenter_script.py"
    slides_md = _require_file(root, f"{_workspace_relative(root, deck_dir)}/5_draft_slides_with_images.md")
    slide_mapping = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_slide_mapping.json")
    blueprint_args: list[str] = []
    if mode == "live":
        blueprint_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint.json")
        receipt_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint_receipt.json")
        review_path = _require_file(root, f"decks/{presentation_id}/blueprint_review.json")
        require_approved_blueprint(
            blueprint_path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=True,
        )
        blueprint_args = ["--blueprint", str(blueprint_path), "--blueprint-review", str(review_path)]
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--slides-md",
                str(slides_md),
                "--slide-mapping",
                str(slide_mapping),
                *blueprint_args,
                "--conference-label",
                _conference_label(root),
                "--output-script-md",
                str(deck_dir / "6_presenter_script.md"),
                "--output-notebooklm-script-md",
                str(deck_dir / "6_presenter_script_notebooklm_2host.md"),
                *_dry_run_args(mode),
            ),
        )
    ]


def _build_video_generation(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_type, topic, paper_topic_id, run_id
    _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation_slide_mapping.json")
    slides_html = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation.html")
    delivery_single = _require_file(
        root,
        f"{_workspace_relative(root, deck_dir)}/delivery_single_speaker.json",
    )
    delivery_notebooklm = _require_file(
        root,
        f"{_workspace_relative(root, deck_dir)}/delivery_notebooklm.json",
    )
    blueprint_args: list[str] = []
    if mode == "live":
        blueprint_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint.json")
        receipt_path = _require_file(root, f"decks/{presentation_id}/presentation_blueprint_receipt.json")
        review_path = _require_file(root, f"decks/{presentation_id}/blueprint_review.json")
        require_approved_blueprint(
            blueprint_path,
            review_path,
            receipt_path=receipt_path,
            require_live_receipt=True,
        )
        blueprint_args = ["--blueprint", str(blueprint_path), "--blueprint-review", str(review_path)]
    steps = [
        (
            "scripts/presentations/build_presenter_video.py",
            [
                "--delivery-contract",
                str(delivery_single),
                "--slides-html",
                str(slides_html),
                "--slide-mapping",
                str(deck_dir / "presentation_slide_mapping.json"),
                *blueprint_args,
                "--build-dir",
                str(deck_dir / "video_build"),
                "--output",
                str(deck_dir / "presentation_video.mp4"),
                *_dry_run_args(mode),
            ],
        ),
        (
            "scripts/presentations/build_notebooklm_style_presenter_video.py",
            [
                "--delivery-contract",
                str(delivery_notebooklm),
                "--slides-html",
                str(slides_html),
                "--slide-mapping",
                str(deck_dir / "presentation_slide_mapping.json"),
                *blueprint_args,
                "--build-dir",
                str(deck_dir / "video_build_notebooklm"),
                "--output",
                str(deck_dir / "presentation_video_notebooklm.mp4"),
                "--duration-reference-script-json",
                str(deck_dir / "video_build" / "script.json"),
                "--conference-label",
                _conference_label(root),
                *_dry_run_args(mode),
            ],
        ),
    ]
    return [
        (owner_ref, _script_argv(python_executable, owner_ref, common_args, *args))
        for owner_ref, args in steps
    ]


def _build_pdf_export(
    *,
    root: Path,
    deck_dir: Path,
    presentation_id: str,
    presentation_type: str,
    topic: object,
    paper_topic_id: str,
    run_id: str | None,
    python_executable: str,
    common_args: list[str],
    mode: str,
) -> list[tuple[str, list[str]]]:
    del presentation_id, presentation_type, topic, paper_topic_id, run_id
    owner_ref = "scripts/presentations/export_revealjs_pdf.py"
    html_path = _require_file(root, f"{_workspace_relative(root, deck_dir)}/presentation.html")
    return [
        (
            owner_ref,
            _script_argv(
                python_executable,
                owner_ref,
                common_args,
                "--input-html",
                str(html_path),
                "--output-pdf",
                str(deck_dir / "presentation.pdf"),
                *_dry_run_args(mode),
            ),
        )
    ]
