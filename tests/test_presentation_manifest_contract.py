"""Contract tests for presentation_manifest.json."""

from __future__ import annotations

import importlib.util
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.presentation.errors import (
    PresentationContractError,
    PresentationManifestValidationError,
)
from conference_pipeline.model_governance import model_for_alias
from conference_pipeline.presentation.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    MANIFEST_RELATIVE_PATH,
    PRESENTATION_STAGE_IDS,
    DEFAULT_STAGE_SPECS,
    LoadedPresentationManifest,
    PresentationArtifactRef,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    build_backfilled_presentations,
    _compute_content_sha256,
    emit_presentation_manifest,
    load_and_validate_manifest,
)



def _opus_model() -> str:
    """The catalog's current Opus model (these tests pin the tier, not a generation)."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def _load_backfill_module():
    script_path = Path("scripts/backfill_presentation_manifest.py").resolve()
    spec = importlib.util.spec_from_file_location("scripts_backfill_presentation", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scripts_backfill = _load_backfill_module()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
    workspace_path = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace_path / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    write_text(
        workspace_path / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(
        workspace_path / "analysis" / "01_demo_topic.md",
        "# Demo Topic\n\nStable topic body.\n",
    )
    write_text(
        workspace_path / "analysis" / "02_other_topic.md",
        "# Other Topic\n\nStable second topic body.\n",
    )
    topic_dir = workspace_path / "decks" / "01_demo_topic"
    write_text(topic_dir / "1_draft_slides.md", "# Draft\n")
    write_text(
        topic_dir / "2_draft_slides_enriched_with_papers.md",
        "# Enriched Draft\n",
    )
    write_text(topic_dir / "presentation_deck.json", '{"schema_version":1,"slides":[]}\n')
    write_text(topic_dir / "presentation_deck_receipt.json", '{"schema_version":1}\n')
    write_text(topic_dir / "ranked_papers.json", '{"papers":[]}\n')
    write_text(topic_dir / "source_pack_receipt.json", '{"schema_version":1}\n')
    write_text(topic_dir / "source_binding_report.json", '{"schema_version":1,"slides":[]}\n')
    write_text(topic_dir / "visual_plan.json", '{"schema_version":1,"slides":[]}\n')
    write_text(topic_dir / "asset_inventory.json", '{"schema_version":1,"assets":[]}\n')
    write_text(topic_dir / "assets" / "generated" / "figure_specs.json", '{"figures":[]}\n')
    write_text(topic_dir / "render_spec.json", '{"schema_version":1,"slides":[]}\n')
    write_text(topic_dir / "presentation.html", "<html></html>\n")
    write_text(topic_dir / "presentation_slide_mapping.json", '{"slides": []}\n')
    write_text(topic_dir / "presentation.pdf", "fake-pdf\n")
    write_text(topic_dir / "assets" / "generated" / "visual_qc_report.md", "ok\n")
    write_text(topic_dir / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png", "png\n")
    write_text(
        topic_dir / "assets" / "generated" / "visual_llm_qc_report.json",
        '{"schema_version":1,"status":"passed","errors":[],"warnings":[],"slides":[]}\n',
    )
    write_text(topic_dir / "assets" / "generated" / "validation_report.md", "ok\n")
    write_text(topic_dir / "assets" / "generated" / "slide_01.svg", "<svg />\n")
    write_text(topic_dir / "6_presenter_script.md", "# Script\n")
    write_text(topic_dir / "delivery_single_speaker.json", '{"schema_version":1}\n')
    write_text(topic_dir / "delivery_notebooklm.json", '{"schema_version":1}\n')
    write_text(topic_dir / "presentation_review.json", '{"schema_version":1,"status":"passed"}\n')
    write_text(topic_dir / "presentation_review.md", "# Presentation Review\n")
    write_text(topic_dir / "presentation_review.prompt.md", "# Prompt\n")
    write_text(topic_dir / "presentation_video.mp4", "fake-video\n")
    write_text(topic_dir / "presentation_video_notebooklm.mp4", "fake-video\n")
    write_text(topic_dir / "video_sync_validation_single_speaker.json", '{"schema_version":1}\n')
    write_text(topic_dir / "video_sync_validation_notebooklm.json", '{"schema_version":1}\n')
    write_text(topic_dir / "video_build_notebooklm" / "dialogue_cache" / "slide_001.json", "{}\n")
    (workspace_path / "decks" / "master_presentation").mkdir(parents=True, exist_ok=True)
    (workspace_path / "decks" / "executive_summary").mkdir(parents=True, exist_ok=True)
    (workspace_path / "decks" / "executive_summary_short").mkdir(parents=True, exist_ok=True)
    (workspace_path / "decks" / "executive_summary_long").mkdir(parents=True, exist_ok=True)
    write_text(
        workspace_path / "decks" / "special-topics" / "README.md",
        "# Special Topics\n",
    )
    return workflow.resolve_workspace(workspace_path)


def make_entry(workspace: workflow.WorkflowWorkspace) -> PresentationEntry:
    workspace_root = workspace.root
    topic_dir = workspace_root / "decks" / "01_demo_topic"
    analysis_path = workspace_root / "analysis" / "01_demo_topic.md"
    stage_states = {
        stage_id: PresentationStageState(status="missing", artifacts=(), notes=())
        for stage_id in PRESENTATION_STAGE_IDS
    }
    stage_states["deck_contract_generation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_deck.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_deck_receipt.json",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["source_binding"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "source_binding_report.json",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["visual_plan_generation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "visual_plan.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "asset_inventory.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "assets" / "generated",
                workspace_root=workspace_root,
                kind="directory",
            ),
        ),
        notes=(),
    )
    stage_states["render_spec_generation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "render_spec.json",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["render_export"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "presentation.html",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_slide_mapping.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "presentation.pdf",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["visual_qc"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "assets" / "generated" / "visual_qc_report.md",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "assets" / "generated" / "visual_llm_qc_report.json",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["delivery_generation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "delivery_single_speaker.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "delivery_notebooklm.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "6_presenter_script.md",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["presentation_review"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_review.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_review.md",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "presentation_review.prompt.md",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    stage_states["video_generation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "video_build_notebooklm",
                workspace_root=workspace_root,
                kind="directory",
            ),
        ),
        notes=(),
    )
    stage_states["video_sync_validation"] = PresentationStageState(
        status="complete",
        artifacts=(
            PresentationArtifactRef.from_path(
                topic_dir / "video_sync_validation_single_speaker.json",
                workspace_root=workspace_root,
                kind="file",
            ),
            PresentationArtifactRef.from_path(
                topic_dir / "video_sync_validation_notebooklm.json",
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        notes=(),
    )
    return PresentationEntry(
        id="01_demo_topic",
        presentation_type="topic-deep-dive",
        title="Demo Topic",
        topic_scope=PresentationTopicScope(
            mode="single-topic",
            topics=("01_demo_topic",),
        ),
        target_duration_minutes=60,
        target_slide_count=60,
        source_analysis_artifacts=(
            PresentationArtifactRef.from_path(
                analysis_path,
                workspace_root=workspace_root,
                kind="file",
            ),
        ),
        stage_states=stage_states,
    )


def emit_fixture_manifest(workspace: workflow.WorkflowWorkspace) -> LoadedPresentationManifest:
    emit_presentation_manifest(workspace, [make_entry(workspace)], generator="test.fixture@dev")
    return load_and_validate_manifest(workspace)


def test_presentation_manifest_schema_version_is_exactly_1(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    path, _, _ = emit_presentation_manifest(
        workspace,
        [make_entry(workspace)],
        generator="test.fixture@dev",
    )
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == MANIFEST_SCHEMA_VERSION == 1


def test_stage_catalog_declares_required_checkpoint_ids():
    assert tuple(spec.id for spec in DEFAULT_STAGE_SPECS) == PRESENTATION_STAGE_IDS


def test_default_stage_specs_match_checkpoint_ids():
    assert tuple(spec.id for spec in DEFAULT_STAGE_SPECS) == PRESENTATION_STAGE_IDS


def test_stage_specs_declare_inputs_outputs_owners_and_mode_readiness():
    for spec in DEFAULT_STAGE_SPECS:
        assert spec.input_artifacts
        assert spec.output_artifacts
        assert spec.owner_refs
        assert spec.provider_requirements
        assert spec.cache_behavior
        assert spec.support_level == "workspace-local"
        assert spec.mode_readiness.live in {"ready", "planned", "unsupported"}
        assert spec.mode_readiness.dry_run in {"ready", "planned", "unsupported"}
        assert spec.mode_readiness.mock in {"ready", "planned", "unsupported"}


def test_source_binding_owner_ref_uses_structured_binding_script():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "source_binding")

    assert "scripts/presentations/bind_presentation_sources.py" in spec.owner_refs
    assert "scripts/presentations/enrich_hyperlinks_hybrid.py" not in spec.owner_refs
    assert "decks/<presentation>/source_binding_report.json" in spec.output_artifacts


def test_presentation_blueprint_stage_requires_frame_owner_and_artifact():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "presentation_blueprint")

    assert "decks/<presentation>/presentation_frame.json" in spec.input_artifacts
    assert "decks/<presentation>/presentation_frame.json" in spec.output_artifacts
    assert "scripts/presentations/generate_editorial_blueprint.py" in spec.owner_refs
    assert "scripts/presentations/generate_presentation_blueprint.py" not in spec.owner_refs
    assert "task:presentation.blueprint_generation" in spec.provider_requirements


def test_visual_qc_stage_requires_llm_owner_and_report():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "visual_qc")

    assert "scripts/presentations/inspect_rendered_slides.py" in spec.owner_refs
    assert "scripts/presentations/inspect_rendered_slides_llm.py" in spec.owner_refs
    assert "decks/<presentation>/assets/generated/visual_llm_qc_report.json" in spec.output_artifacts
    assert "env:OPENAI_API_KEY" in spec.provider_requirements


def test_presentation_review_stage_requires_three_reviewers_and_reports():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "presentation_review")

    assert "scripts/presentations/review_presentation.py" in spec.owner_refs
    assert "decks/<presentation>/presentation_blueprint.json" in spec.input_artifacts
    assert "decks/<presentation>/6_presenter_script.md" in spec.input_artifacts
    assert "decks/<presentation>/assets/generated/qc/screenshots/" in spec.input_artifacts
    assert "decks/<presentation>/presentation_review.json" in spec.output_artifacts
    assert "decks/<presentation>/presentation_review.md" in spec.output_artifacts
    assert f"model:anthropic:{_opus_model()}" in spec.provider_requirements
    assert f"model:openai:{model_for_alias('sol')}" in spec.provider_requirements
    assert "tool:gemini" in spec.provider_requirements


def test_delivery_generation_declares_narration_script_for_review_gate():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "delivery_generation")

    assert "decks/<presentation>/delivery_single_speaker.json" in spec.output_artifacts
    assert "decks/<presentation>/delivery_notebooklm.json" in spec.output_artifacts
    assert "decks/<presentation>/6_presenter_script.md" in spec.output_artifacts
    assert "debug export only" not in spec.cache_behavior


def test_render_export_stage_requires_render_spec_and_relationalai_template():
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "render_export")

    assert "scripts/presentations/md_to_revealjs.py" in spec.owner_refs
    assert "decks/<presentation>/render_spec.json" in spec.input_artifacts
    assert "decks/<presentation>/presentation_slide_mapping.json" in spec.output_artifacts
    assert "brand:RelationalAI NeurIPS 2025 reference" in spec.provider_requirements
    assert "render_spec.json" in spec.cache_behavior


def test_backfilled_topic_title_strips_slide_brief_suffix(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    write_text(
        workspace.root / "analysis" / "01_demo_topic.md",
        "# Demo Topic Slide Brief\n\nStable topic body.\n",
    )

    presentations = build_backfilled_presentations(workspace)
    entry = next(item for item in presentations if item.id == "01_demo_topic")

    assert entry.title == "Demo Topic"


def test_load_rejects_missing_manifest(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    with pytest.raises(PresentationManifestValidationError, match="Missing presentation_manifest.json"):
        load_and_validate_manifest(workspace)


def test_load_rejects_malformed_json(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    manifest_path = workspace.root / MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("not json", encoding="utf-8")

    with pytest.raises(PresentationManifestValidationError, match="Malformed presentation manifest"):
        load_and_validate_manifest(workspace)


def test_load_rejects_unknown_stage_id(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    path, _, _ = emit_presentation_manifest(
        workspace,
        [make_entry(workspace)],
        generator="test.fixture@dev",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stage_specs"][0]["id"] = "unknown_stage"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(PresentationManifestValidationError, match="unknown stage id"):
        load_and_validate_manifest(workspace)


def test_load_rejects_path_escape(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    path, _, _ = emit_presentation_manifest(
        workspace,
        [make_entry(workspace)],
        generator="test.fixture@dev",
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["presentations"][0]["source_analysis_artifacts"][0]["path"] = "../escape.md"
    data["content_sha256"] = _compute_content_sha256(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        PresentationManifestValidationError,
        match="must stay inside the conference workspace",
    ):
        load_and_validate_manifest(workspace)


def test_load_rejects_missing_declared_artifact(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    emit_fixture_manifest(workspace)
    (workspace.root / "analysis" / "01_demo_topic.md").unlink()

    with pytest.raises(PresentationManifestValidationError, match="does not exist"):
        load_and_validate_manifest(workspace, require_path_existence=True)


def test_content_sha256_stable_across_regeneration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace = make_workspace(tmp_path)

    class FakeDatetime:
        calls = [
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 6, 1, tzinfo=timezone.utc),
        ]

        @classmethod
        def now(cls, tz=None):
            value = cls.calls.pop(0)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr("conference_pipeline.presentation.manifest.datetime", FakeDatetime)

    _, _, sha1 = emit_presentation_manifest(workspace, [make_entry(workspace)], generator="test@dev")
    _, _, sha2 = emit_presentation_manifest(workspace, [make_entry(workspace)], generator="test@dev")

    assert sha1 == sha2


def test_loaded_manifest_carries_both_sha_fields(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    emit_presentation_manifest(workspace, [make_entry(workspace)], generator="test@dev")
    loaded = load_and_validate_manifest(workspace)

    assert loaded.manifest_sha256 != loaded.content_sha256


def test_directory_artifact_digest_changes_when_child_file_changes(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    emit_fixture_manifest(workspace)
    target = workspace.root / "decks" / "01_demo_topic" / "video_build_notebooklm" / "dialogue_cache" / "slide_001.json"
    target.write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(PresentationManifestValidationError, match="sha256 mismatch"):
        load_and_validate_manifest(workspace)


def test_presentation_contract_error_is_not_value_error():
    assert not issubclass(PresentationContractError, ValueError)
    assert not issubclass(PresentationManifestValidationError, ValueError)


def test_backfill_writes_workspace_level_manifest_under_decks(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    assert scripts_backfill.main(["--workspace", str(workspace.root)]) == 0

    manifest_path = workspace.root / "decks" / MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path.name == "presentation_manifest.json"
    assert data["generator"].startswith("scripts.backfill_presentation_manifest@")
    assert {entry["id"] for entry in data["presentations"]} >= {
        "01_demo_topic",
        "master_presentation",
        "executive_summary",
        "executive_summary_short",
        "executive_summary_long",
    }


def test_backfill_skips_noncanonical_deck_directories(tmp_path: Path, caplog):
    workspace = make_workspace(tmp_path)

    with caplog.at_level(logging.WARNING, logger="backfill_presentation_manifest"):
        assert scripts_backfill.main(["--workspace", str(workspace.root)]) == 0

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "special-topics" in messages

    manifest = load_and_validate_manifest(workspace)
    assert "special-topics" not in {entry.id for entry in manifest.manifest.presentations}


def test_backfill_marks_partial_stage_completion_truthfully(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    presentations = build_backfilled_presentations(workspace)
    topic = next(entry for entry in presentations if entry.id == "01_demo_topic")

    assert topic.stage_states["source_binding"].status == "complete"
    assert topic.stage_states["visual_plan_generation"].status == "complete"


def test_backfill_records_truthful_neurips_like_stage_mix_for_01_tools_and_open_source():
    manifest_path = Path("conferences/neurips2025/decks/presentation_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Optional NeurIPS 2025 presentation manifest fixture is absent")

    data = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    topic = next(
        entry for entry in data["presentations"] if entry["id"] == "01_tools_and_open_source"
    )
    states = topic["stage_states"]
    assert states["render_spec_generation"]["status"] == "missing"
    assert states["visual_plan_generation"]["status"] == "partial"
    assert states["render_export"]["status"] == "complete"
    assert states["human_review"]["status"] == "missing"
    assert states["video_generation"]["status"] == "partial"


def test_backfill_marks_human_review_missing_without_marker(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    presentations = build_backfilled_presentations(workspace)
    topic = next(entry for entry in presentations if entry.id == "01_demo_topic")

    assert topic.stage_states["human_review"].status == "missing"
    assert "Explicit review evidence is absent" in topic.stage_states["human_review"].notes[0]


def test_backfill_marks_human_review_complete_with_receipt(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    receipt_path = workspace.root / "decks" / "01_demo_topic" / "human_review_receipt.json"
    receipt_path.write_text('{"status": "accepted"}\n', encoding="utf-8")

    presentations = build_backfilled_presentations(workspace)
    topic = next(entry for entry in presentations if entry.id == "01_demo_topic")

    state = topic.stage_states["human_review"]
    assert state.status == "complete"
    assert tuple(artifact.path for artifact in state.artifacts) == (
        "decks/01_demo_topic/human_review_receipt.json",
    )


def test_backfill_marks_human_review_partial_with_unapproved_receipt(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    receipt_path = workspace.root / "decks" / "01_demo_topic" / "human_review_receipt.json"
    receipt_path.write_text(
        '{"status": "needs_revision", "decision": "needs_revision_before_bulk_rollout"}\n',
        encoding="utf-8",
    )

    presentations = build_backfilled_presentations(workspace)
    topic = next(entry for entry in presentations if entry.id == "01_demo_topic")

    state = topic.stage_states["human_review"]
    assert state.status == "partial"
    assert tuple(artifact.path for artifact in state.artifacts) == (
        "decks/01_demo_topic/human_review_receipt.json",
    )
    assert "does not record an approved decision" in state.notes[0]


def test_backfill_uses_topic_master_and_executive_targets(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    presentations = {entry.id: entry for entry in build_backfilled_presentations(workspace)}

    assert (
        presentations["01_demo_topic"].target_duration_minutes,
        presentations["01_demo_topic"].target_slide_count,
    ) == (60, 60)
    assert (
        presentations["master_presentation"].target_duration_minutes,
        presentations["master_presentation"].target_slide_count,
    ) == (90, 90)
    assert (
        presentations["executive_summary"].target_duration_minutes,
        presentations["executive_summary"].target_slide_count,
    ) == (30, 30)
    assert (
        presentations["executive_summary_short"].target_duration_minutes,
        presentations["executive_summary_short"].target_slide_count,
    ) == (12, 12)
    assert (
        presentations["executive_summary_long"].target_duration_minutes,
        presentations["executive_summary_long"].target_slide_count,
    ) == (30, 30)


def test_backfill_uses_nonempty_slide_mapping_count_for_topic_target(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    mapping_path = workspace.root / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
    mapping_path.write_text(
        json.dumps({"slides": [{"slide_number": 1}, {"slide_number": 2}]}) + "\n",
        encoding="utf-8",
    )

    presentations = {entry.id: entry for entry in build_backfilled_presentations(workspace)}

    assert presentations["01_demo_topic"].target_slide_count == 2


def test_backfill_uses_configured_topic_target_over_existing_mapping(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    config_path = workspace.root / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "workflow:\n"
        + "  deck:\n"
        + "    target_slides: 30\n",
        encoding="utf-8",
    )
    mapping_path = workspace.root / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
    mapping_path.write_text(
        json.dumps({"slides": [{"slide_number": index} for index in range(1, 57)]}) + "\n",
        encoding="utf-8",
    )

    resolved = workflow.resolve_workspace(workspace.root)
    presentations = {entry.id: entry for entry in build_backfilled_presentations(resolved)}

    assert presentations["01_demo_topic"].target_slide_count == 30


def test_backfill_records_configured_density_targets_for_topic_decks(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    config_path = workspace.root / "config.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + "workflow:\n"
        + "  deck:\n"
        + "    target_hyperlinks_per_slide: 2.5\n"
        + "    target_svgs_per_slide: 1.25\n",
        encoding="utf-8",
    )

    resolved = workflow.resolve_workspace(workspace.root)
    presentations = {entry.id: entry for entry in build_backfilled_presentations(resolved)}

    assert presentations["01_demo_topic"].target_hyperlinks_per_slide == 2.5
    assert presentations["01_demo_topic"].target_svgs_per_slide == 1.25
    assert presentations["master_presentation"].target_hyperlinks_per_slide == 0.0
    assert presentations["executive_summary"].target_svgs_per_slide == 0.0
    assert presentations["executive_summary_short"].target_svgs_per_slide == 0.0
    assert presentations["executive_summary_long"].target_svgs_per_slide == 0.0
