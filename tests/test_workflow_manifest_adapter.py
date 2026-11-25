"""Adapter tests for Phase 12: podcast + deck commands from manifest."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.analysis.manifest import emit_analysis_manifest
from conference_pipeline.presentation.manifest import (
    DEFAULT_STAGE_SPECS,
    PRESENTATION_STAGE_IDS,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    emit_presentation_manifest,
)


_CONFIG_YAML = """\
conference:
  name: Demo 2026
storage:
  manifest_path: "storage/storage_manifest.json"
workflow:
  podcast:
    target_minutes: 30
    max_tier: 3
    top_n_facts: 100
    host_a: "Alex"
    host_b: "Sam"
    model: "gemini-2.0-flash"
  deck:
    target_slides: 60
    reflow_model: "gemini"
    max_papers: 2
    required_env: []
"""
TEST_RUN_ID = "test-run"
BLUEPRINT_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace_with_manifest(
    tmp_path: Path,
    *,
    topic_slugs: tuple[str, ...] = ("demo_topic",),
    seed_audio: bool = True,
    seed_presentations: bool = True,
) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    write_text(workspace / "config.yaml", _CONFIG_YAML)
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    for index, slug in enumerate(topic_slugs, start=1):
        topic_id = f"{index:02d}_{slug}"
        write_text(
            workspace / "analysis" / f"{topic_id}.md",
            f"# {slug.replace('_', ' ').title()}\n\nDemo body\n",
        )
        if seed_audio:
            write_text(workspace / "audio" / topic_id / "ranks.json", '{"ranks": []}\n')
        if seed_presentations:
            (workspace / "presentations" / topic_id).mkdir(parents=True, exist_ok=True)

    resolved = workflow.resolve_workspace(workspace)
    emit_analysis_manifest(resolved, generator="test.make_workspace@dev")
    return workspace


def allow_workspace_as_default(monkeypatch: pytest.MonkeyPatch, resolved: workflow.WorkflowWorkspace) -> None:
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)


def emit_test_presentation_manifest(
    resolved: workflow.WorkflowWorkspace,
    *,
    stage_specs=None,
) -> None:
    stage_states = {
        stage_id: PresentationStageState(status="missing", artifacts=(), notes=())
        for stage_id in PRESENTATION_STAGE_IDS
    }
    emit_presentation_manifest(
        resolved,
        (
            PresentationEntry(
                id="01_demo_topic",
                presentation_type="topic-deep-dive",
                title="Demo Topic",
                topic_scope=PresentationTopicScope(
                    mode="single-topic",
                    topics=("01_demo_topic",),
                ),
                target_duration_minutes=60,
                target_slide_count=60,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
            PresentationEntry(
                id="master_presentation",
                presentation_type="master",
                title="Master Presentation",
                topic_scope=PresentationTopicScope(mode="all-topics", topics=()),
                target_duration_minutes=90,
                target_slide_count=90,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
            PresentationEntry(
                id="executive_summary",
                presentation_type="executive-summary",
                title="Executive Summary",
                topic_scope=PresentationTopicScope(mode="all-topics", topics=()),
                target_duration_minutes=30,
                target_slide_count=24,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
        ),
        generator="test.presentation@dev",
        stage_specs=stage_specs,
    )


def seed_presentation_command_inputs(
    workspace: Path,
    *,
    presentation_id: str = "01_demo_topic",
    include_reveal_slides: bool = True,
    include_slide_mapping: bool = True,
    include_video_inputs: bool = True,
) -> None:
    deck_dir = workspace / "decks" / presentation_id
    blueprint_path = deck_dir / "presentation_blueprint.json"
    blueprint_payload = json.loads(
        (BLUEPRINT_FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8")
    )
    slide_count = len(blueprint_payload.get("slides", []))
    if slide_count > 14:
        blueprint_payload["approved_expansion"] = {
            "approved": True,
            "target_slide_count": slide_count,
            "max_runtime_minutes": 40,
            "blueprint_sha256": "0" * 64,
            "storyboard_sha256": "1" * 64,
            "source_plan_sha256": "2" * 64,
            "review_receipt_path": f"decks/{presentation_id}/blueprint_review.json",
            "rationale": "Fixture approves the expanded deck for command-planning tests.",
        }
    for slide in blueprint_payload.get("slides", []):
        if not isinstance(slide, dict):
            continue
        source_plan = slide.get("source_plan")
        labels = [
            str(item.get("label") if isinstance(item, dict) else item).strip()
            for item in source_plan
            if str(item.get("label") if isinstance(item, dict) else item).strip()
        ] if isinstance(source_plan, list) else []
        slide["speaker_author_work_mentions"] = [
            {
                "label": label,
                "source_id": label,
                "title": f"{label} Fixture Work",
                "authors": ["Fixture Author"],
                "works": [f"{label} Fixture Work"],
                "contribution_note": "Fixture contribution for deck-contract dry-run.",
            }
            for label in labels
        ]
    write_text(
        blueprint_path,
        json.dumps(blueprint_payload, indent=2, sort_keys=True) + "\n",
    )
    write_text(
        deck_dir / "presentation_blueprint.md",
        "# Presentation Blueprint\n",
    )
    blueprint_review = json.loads(
        (BLUEPRINT_FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8")
    )
    blueprint_review["blueprint_sha256"] = sha256(blueprint_path.read_bytes()).hexdigest()
    write_text(
        deck_dir / "blueprint_review.json",
        json.dumps(blueprint_review, indent=2, sort_keys=True) + "\n",
    )
    write_text(deck_dir / "blueprint_review.md", "# Blueprint Review\n")
    write_text(
        deck_dir / "presentation_blueprint_receipt.json",
        json.dumps(
            {
                "blueprint_sha256": sha256(blueprint_path.read_bytes()).hexdigest(),
                "mode": "live",
                "production_evidence": True,
                "provider_receipt_paths": ["storage/provider-calls/blueprint-generation/demo.json"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(deck_dir / "1_draft_slides.md", "# Draft\n")
    write_text(
        deck_dir / "presentation_deck.json",
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": presentation_id,
                "title": "Demo Topic",
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "title": "Demo",
                        "purpose": "Test slide",
                        "narrative": "A test slide.",
                        "source_label": "Demo source",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(
        deck_dir / "presentation_deck_receipt.json",
        '{"schema_version": 1, "production_evidence": true}\n',
    )
    write_text(deck_dir / "ranked_papers.json", '{"papers": []}\n')
    write_text(
        workspace
        / "storage"
        / "paper-matching"
        / TEST_RUN_ID
        / presentation_id
        / "matched_papers.json",
        '{"papers": []}\n',
    )
    write_text(
        workspace
        / "storage"
        / "extractions"
        / TEST_RUN_ID
        / "by-topic"
        / f"{presentation_id}.json",
        '{"items": []}\n',
    )
    write_text(deck_dir / "source_attributions.json", "{}\n")
    write_text(deck_dir / "source_pack_receipt.json", '{"schema_version": 1}\n')
    write_text(
        deck_dir / "source_binding_report.json",
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": presentation_id,
                "bindings": [
                    {
                        "slide_id": "slide-001",
                        "source_label": "Demo source",
                        "url": "https://example.com/demo",
                        "rationale": "Fixture source.",
                    }
                ],
                "unresolved_sources": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(deck_dir / "4.5_draft_slides_with_hyperlinks.md", "# Hyperlinks\n")
    write_text(deck_dir / "3.5_draft_slides_consolidated.md", "# Consolidated\n")
    write_text(deck_dir / "assets" / "generated" / "demo.svg", "<svg></svg>\n")
    write_text(
        deck_dir / "visual_plan.json",
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": presentation_id,
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "visual_brief": "Render a simple test visual.",
                        "renderer": "svgwrite",
                        "asset_path": f"decks/{presentation_id}/assets/generated/demo.svg",
                        "source_labels": ["Demo source"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(
        deck_dir / "asset_inventory.json",
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "slide_id": "slide-001",
                        "path": f"decks/{presentation_id}/assets/generated/demo.svg",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    write_text(
        deck_dir / "assets" / "generated" / "figure_specs.json",
        '{"schema_version": 1, "figures": []}\n',
    )
    write_text(
        deck_dir / "render_spec.json",
        json.dumps(
            {
                "schema_version": 1,
                "presentation_id": presentation_id,
                "slides": [
                    {
                        "slide_id": "slide-001",
                        "title": "Demo",
                        "blocks": [],
                        "speaker_notes": "A test slide.",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    if include_reveal_slides:
        write_text(deck_dir / "5_draft_slides_with_images.md", "# Images\n")
    write_text(deck_dir / "presentation.html", "<html></html>\n")
    write_text(deck_dir / "render_contact_sheet.png", "contact-sheet\n")
    if include_slide_mapping:
        write_text(deck_dir / "presentation_slide_mapping.json", '{"1": 0}\n')
    write_text(
        deck_dir / "assets" / "generated" / "visual_llm_qc_report.json",
        '{"schema_version": 1, "status": "passed", "findings": []}\n',
    )
    write_text(deck_dir / "assets" / "generated" / "qc" / "screenshots" / "slide_001.png", "png\n")
    if include_video_inputs:
        write_text(deck_dir / "6_presenter_script.md", "# Presenter Script\n")
        write_text(
            deck_dir / "6_presenter_script_notebooklm_2host.md",
            "# NotebookLM Presenter Script\n",
        )
        write_text(
            deck_dir / "delivery_single_speaker.json",
            '{"schema_version": 1, "delivery_variant": "single_speaker"}\n',
        )
        write_text(
            deck_dir / "delivery_notebooklm.json",
            '{"schema_version": 1, "delivery_variant": "notebooklm"}\n',
        )
        write_text(deck_dir / "presentation_video.mp4", "fake-video\n")
        write_text(deck_dir / "presentation_video_notebooklm.mp4", "fake-video\n")
        write_text(
            deck_dir / "video_sync_validation_single_speaker.json",
            '{"schema_version": 1, "status": "pass", "errors": []}\n',
        )
        write_text(
            deck_dir / "video_sync_validation_notebooklm.json",
            '{"schema_version": 1, "status": "pass", "errors": []}\n',
        )


def seed_declared_stage_inputs(
    workspace: Path,
    spec,
    *,
    presentation_id: str = "01_demo_topic",
) -> None:
    for artifact_ref in spec.input_artifacts:
        relative_path = artifact_ref.replace("<presentation>", presentation_id)
        relative_path = relative_path.replace("<run-id>", TEST_RUN_ID)
        relative_path = relative_path.replace("<topic>", presentation_id)
        if relative_path == "analysis/NN_slug.md":
            relative_path = f"analysis/{presentation_id}.md"
        if relative_path.endswith("/"):
            (workspace / relative_path).mkdir(parents=True, exist_ok=True)
            write_text(workspace / relative_path / "slide_001.png", "png\n")
            continue
        if relative_path.endswith("*.svg"):
            relative_path = relative_path.removesuffix("*.svg") + "declared.svg"
        content = "{}\n" if relative_path.endswith(".json") else "# Declared input\n"
        write_text(workspace / relative_path, content)


def _pre_phase12_deck_commands(
    resolved: workflow.WorkflowWorkspace,
    *,
    topic: str | None,
) -> list[list[str]]:
    deck_config = resolved.workflow.deck
    topic_args = ["--all"] if topic is None else ["--topic", topic[:2]]
    return [
        [
            sys.executable,
            "scripts/presentations/generate_slides_agent.py",
            *topic_args,
            "--slides",
            str(deck_config.target_slides),
            "-v",
        ],
        [
            sys.executable,
            "scripts/presentations/enrich_slides_with_papers.py",
            *topic_args,
            "--slides",
            str(deck_config.target_slides),
            "-v",
        ],
        [
            sys.executable,
            "scripts/presentations/reflow_slides.py",
            *topic_args,
            "--model",
            deck_config.reflow_model,
            "-v",
        ],
        [sys.executable, "scripts/presentations/sanity_check_slides.py", *topic_args, "-v"],
        [sys.executable, "scripts/presentations/enrich_hyperlinks_hybrid.py", *topic_args, "-v"],
        [sys.executable, "scripts/presentations/generate_presenter_script.py", *topic_args, "-v"],
    ]


def test_podcast_commands_sourced_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    loader_called = {"count": 0}

    import conference_pipeline.analysis.manifest as manifest_mod

    original = manifest_mod.load_and_validate_manifest

    def counting_loader(*args, **kwargs):
        loader_called["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(manifest_mod, "load_and_validate_manifest", counting_loader)
    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("podcast"),
        topic="01",
    )

    assert loader_called["count"] >= 1
    assert len(commands) == 1


def test_podcast_single_topic_from_manifest(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(
        tmp_path,
        topic_slugs=("demo_topic", "second_topic"),
    )
    resolved = workflow.resolve_workspace(workspace_path)
    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("podcast"),
        topic="01",
    )
    expected = [
        sys.executable,
        "scripts/generate_focused_podcast.py",
        str(resolved.root / "analysis" / "01_demo_topic.md"),
        "--fact-ranks",
        str(resolved.root / "audio" / "01_demo_topic" / "ranks.json"),
        "--output-dir",
        str(resolved.root / "audio" / "01_demo_topic" / "focused_podcast"),
        "--target-minutes",
        str(resolved.workflow.podcast.target_minutes),
        "--max-tier",
        str(resolved.workflow.podcast.max_tier),
        "--top-n-facts",
        str(resolved.workflow.podcast.top_n_facts),
        "--two-hosts",
        "--host-a",
        resolved.workflow.podcast.host_a,
        "--host-b",
        resolved.workflow.podcast.host_b,
        "--model",
        resolved.workflow.podcast.model,
    ]

    assert len(expected) == 20
    assert commands == [expected]


def test_podcast_all_topics_from_manifest(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(
        tmp_path,
        topic_slugs=("demo_topic", "second_topic"),
    )
    resolved = workflow.resolve_workspace(workspace_path)
    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("podcast"),
        topic=None,
    )

    assert len(commands) == 2
    assert "01_demo_topic.md" in commands[0][2]
    assert "02_second_topic.md" in commands[1][2]


def test_podcast_fails_closed_without_manifest(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    (workspace_path / "analysis" / "analysis_manifest.json").unlink()
    resolved = workflow.resolve_workspace(workspace_path)

    with pytest.raises(ManifestValidationError, match="Missing analysis_manifest.json"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("podcast"),
            topic="01",
        )


def test_podcast_fails_closed_on_schema_version_mismatch(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    manifest_path = workspace_path / "analysis" / "analysis_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["schema_version"] = 0
    manifest_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ManifestValidationError, match="schema_version 0 is not supported"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("podcast"),
            topic="01",
        )


def test_podcast_fails_closed_when_markdown_missing(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    (workspace_path / "analysis" / "01_demo_topic.md").unlink()

    with pytest.raises(ManifestValidationError, match="markdown_path"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("podcast"),
            topic="01",
        )


def test_deck_commands_sourced_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic="01",
    )

    assert len(commands) == 6


@pytest.mark.parametrize("topic", [None, "01"])
def test_deck_command_argv_byte_equivalent_for_supported_legacy_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    topic: str | None,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic=topic,
    )
    expected = _pre_phase12_deck_commands(resolved, topic=topic)
    script_names = [
        "generate_slides_agent.py",
        "enrich_slides_with_papers.py",
        "reflow_slides.py",
        "sanity_check_slides.py",
        "enrich_hyperlinks_hybrid.py",
        "generate_presenter_script.py",
    ]

    assert len(commands) == len(expected) == 6
    for index, script in enumerate(script_names):
        assert commands[index] == expected[index], (
            f"deck command {index} ({script}) argv differs from pre-Phase-12 shape:\n"
            f"  got:      {commands[index]}\n"
            f"  expected: {expected[index]}"
        )


def test_presentation_stage_selects_topic_deep_dive_from_manifest(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert commands
    assert commands[0][:2] == [sys.executable, "scripts/presentations/generate_deck_contract.py"]
    assert "--presentation-id" in commands[0]
    assert commands[0][commands[0].index("--presentation-id") + 1] == "01_demo_topic"
    assert commands[0][commands[0].index("--presentation-type") + 1] == "topic-deep-dive"
    assert commands[0][commands[0].index("--topic") + 1] == "01_demo_topic"
    assert "--presentation-blueprint" in commands[0]
    assert "--output" in commands[0]
    assert commands[0][commands[0].index("--output") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "presentation_deck.json"
    )
    assert "--slides" not in commands[0]


def test_presentation_stage_selects_master_from_manifest_without_topic(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path, presentation_id="master_presentation")

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("render_export"),
        presentation_type="master",
    )

    assert commands
    assert commands[0][commands[0].index("--presentation-id") + 1] == "master_presentation"
    assert commands[0][commands[0].index("--presentation-type") + 1] == "master"
    assert commands[0][commands[0].index("--render-spec") + 1] == str(
        resolved.root / "decks" / "master_presentation" / "render_spec.json"
    )
    assert "--topic" not in commands[0]


@pytest.mark.parametrize(
    ("presentation_type", "presentation_id", "display_title"),
    [
        ("master", "master_presentation", "Master Presentation"),
        ("executive-summary", "executive_summary", "Executive Summary"),
    ],
)
def test_non_topic_deck_contract_commands_execute_dry_run_without_topic(
    tmp_path: Path,
    presentation_type: str,
    presentation_id: str,
    display_title: str,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    write_text(workspace_path / "analysis" / f"{presentation_id}.md", "# Analysis\n")
    seed_presentation_command_inputs(workspace_path, presentation_id=presentation_id)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type=presentation_type,
    )

    assert len(commands) == 1
    assert "--topic" not in commands[0]
    result = subprocess.run(commands[0], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["receipt"]["presentation_id"] == presentation_id
    assert payload["receipt"]["status"] == "dry_run"
    assert payload["presentation_deck"]["slides"]


def test_presentation_stage_commands_use_decks_paths(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    command_text = json.dumps(commands)

    assert "presentation-stage-preview" not in command_text
    assert "conferences/neurips2025/presentations/" not in command_text
    assert "decks/01_demo_topic/presentation_deck.json" in command_text
    assert "decks/01_demo_topic/1_draft_slides.md" not in command_text
    assert str(resolved.root / "decks" / "01_demo_topic") in command_text


def test_presentation_stage_commands_reject_legacy_presentations_paths(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    legacy_stage_specs = tuple(
        replace(
            spec,
            input_artifacts=("presentations/<presentation>/5_draft_slides_with_images.md",),
            output_artifacts=(
                "presentations/<presentation>/presentation.html",
                "presentations/<presentation>/presentation_slide_mapping.json",
            ),
        )
        if spec.id == "render_export"
        else spec
        for spec in DEFAULT_STAGE_SPECS
    )
    emit_test_presentation_manifest(resolved, stage_specs=legacy_stage_specs)

    with pytest.raises(Exception, match="PRESENTATION_LEGACY_MARKDOWN_STAGE_USED"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("render_export"),
            presentation_type="topic-deep-dive",
            topic="01",
        )


def test_render_export_emits_real_manifest_argv(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("render_export"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert len(commands) == 2
    command, pdf_command = commands
    assert "presentation-stage-preview" not in command
    assert "presentation-stage-preview" not in pdf_command
    assert command[:2] == [sys.executable, "scripts/presentations/md_to_revealjs.py"]
    assert command[command.index("--render-spec") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "render_spec.json"
    )
    assert command[command.index("--output-html") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "presentation.html"
    )
    assert command[command.index("--slide-mapping") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
    )
    assert "--require-brand-assets" in command
    assert "--slides-md" not in command
    assert pdf_command[:2] == [sys.executable, "scripts/presentations/md_to_pptx.py"]
    assert pdf_command[pdf_command.index("--render-spec") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "render_spec.json"
    )
    assert pdf_command[pdf_command.index("--output-pdf") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "presentation.pdf"
    )


def test_video_generation_emits_safe_dry_run_commands(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("video_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    scripts = {command[1] for command in commands}

    assert scripts == {
        "scripts/presentations/build_presenter_video.py",
        "scripts/presentations/build_notebooklm_style_presenter_video.py",
    }
    for command in commands:
        assert "presentation-stage-preview" not in command
        for flag in (
            "--dry-run",
            "--delivery-contract",
            "--slides-html",
            "--build-dir",
            "--output",
            "--slide-mapping",
        ):
            assert flag in command
        assert command[command.index("--slides-html") + 1] == str(
            resolved.root / "decks" / "01_demo_topic" / "presentation.html"
        )
    script_by_owner = {command[1]: command for command in commands}
    assert script_by_owner["scripts/presentations/build_presenter_video.py"][
        script_by_owner["scripts/presentations/build_presenter_video.py"].index("--delivery-contract") + 1
    ] == str(resolved.root / "decks" / "01_demo_topic" / "delivery_single_speaker.json")
    assert script_by_owner["scripts/presentations/build_notebooklm_style_presenter_video.py"][
        script_by_owner[
            "scripts/presentations/build_notebooklm_style_presenter_video.py"
        ].index("--delivery-contract")
        + 1
    ] == str(resolved.root / "decks" / "01_demo_topic" / "delivery_notebooklm.json")
    assert script_by_owner["scripts/presentations/build_notebooklm_style_presenter_video.py"][
        script_by_owner[
            "scripts/presentations/build_notebooklm_style_presenter_video.py"
        ].index("--duration-reference-script-json")
        + 1
    ] == str(resolved.root / "decks" / "01_demo_topic" / "video_build" / "script.json")
    assert script_by_owner["scripts/presentations/build_notebooklm_style_presenter_video.py"][
        script_by_owner[
            "scripts/presentations/build_notebooklm_style_presenter_video.py"
        ].index("--conference-label")
        + 1
    ] == "Demo 2026"


def test_delivery_generation_command_carries_conference_label(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("delivery_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("delivery_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[1] == "scripts/presentations/generate_presenter_script.py"
    assert command[command.index("--conference-label") + 1] == "Demo 2026"
    assert command[command.index("--render-spec") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "render_spec.json"
    )
    assert command[command.index("--delivery-single-speaker") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "delivery_single_speaker.json"
    )
    assert command[command.index("--debug-script-md") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "6_presenter_script.md"
    )
    assert _metadata_output_paths_by_owner(metadata) == {
        "scripts/presentations/generate_presenter_script.py": {
            "decks/01_demo_topic/delivery_single_speaker.json",
            "decks/01_demo_topic/delivery_notebooklm.json",
            "decks/01_demo_topic/6_presenter_script.md",
        },
    }


def test_presentation_review_command_uses_requirements_script_and_screenshots(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("presentation_review"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("presentation_review"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert len(commands) == 1
    command = commands[0]
    deck_dir = resolved.root / "decks" / "01_demo_topic"
    assert command[1] == "scripts/presentations/review_presentation.py"
    assert command[command.index("--requirements") + 1] == str(
        deck_dir / "presentation_blueprint.json"
    )
    assert command[command.index("--narration-script") + 1] == str(
        deck_dir / "6_presenter_script.md"
    )
    assert command[command.index("--screenshots-dir") + 1] == str(
        deck_dir / "assets" / "generated" / "qc" / "screenshots"
    )
    assert command[command.index("--blueprint") + 1] == str(
        deck_dir / "presentation_blueprint.json"
    )
    assert command[command.index("--blueprint-review") + 1] == str(
        deck_dir / "blueprint_review.json"
    )
    assert command[command.index("--deck-contract") + 1] == str(
        deck_dir / "presentation_deck.json"
    )
    assert command[command.index("--render-spec") + 1] == str(deck_dir / "render_spec.json")
    assert command[command.index("--source-binding") + 1] == str(
        deck_dir / "source_binding_report.json"
    )
    assert command[command.index("--delivery-contract") + 1] == str(
        deck_dir / "delivery_notebooklm.json"
    )
    assert command[command.index("--video-receipt") + 1] == str(
        deck_dir / "video_sync_validation_notebooklm.json"
    )
    assert command[command.index("--visual-qc-report") + 1] == str(
        deck_dir / "assets" / "generated" / "visual_llm_qc_report.json"
    )
    assert command[command.index("--contact-sheet") + 1] == str(
        deck_dir / "render_contact_sheet.png"
    )
    assert "--dry-run" in command
    assert _metadata_output_paths_by_owner(metadata) == {
        "scripts/presentations/review_presentation.py": {
            "decks/01_demo_topic/presentation_review.json",
            "decks/01_demo_topic/presentation_review.md",
            "decks/01_demo_topic/presentation_review.prompt.md",
        },
    }


def test_video_generation_live_commands_do_not_force_dry_run(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("video_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
        mode="live",
    )

    assert commands
    assert all("--dry-run" not in command for command in commands)
    for command in commands:
        assert "--delivery-contract" in command
        assert command[command.index("--slides-html") + 1] == str(
            resolved.root / "decks" / "01_demo_topic" / "presentation.html"
        )


def test_video_generation_requires_slide_mapping(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("video_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    for command in commands:
        assert "--slide-mapping" in command
        assert command[command.index("--slide-mapping") + 1] == str(
            resolved.root / "decks" / "01_demo_topic" / "presentation_slide_mapping.json"
        )


def test_supported_presentation_stage_commands_include_workspace_scope_flags(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)
    supported_stage_ids = [
        "deck_contract_generation",
        "source_binding",
        "visual_plan_generation",
        "render_spec_generation",
        "render_export",
        "visual_qc",
        "delivery_generation",
        "presentation_review",
        "video_generation",
        "video_sync_validation",
    ]

    for stage_id in supported_stage_ids:
        commands = workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage(stage_id),
            presentation_type="topic-deep-dive",
            topic="01",
            run_id=TEST_RUN_ID,
        )
        assert commands, stage_id
        for command in commands:
            assert "--workspace" in command, stage_id
            assert "--presentation-id" in command, stage_id
            assert "--presentation-type" in command, stage_id


def test_supported_presentation_stage_declared_inputs_are_sufficient_for_command_planning(
    tmp_path: Path,
):
    supported_stage_ids = [
        "deck_contract_generation",
        "source_binding",
        "visual_plan_generation",
        "render_spec_generation",
        "render_export",
        "visual_qc",
        "delivery_generation",
        "presentation_review",
        "video_generation",
        "video_sync_validation",
    ]
    specs_by_id = {spec.id: spec for spec in DEFAULT_STAGE_SPECS}

    for stage_id in supported_stage_ids:
        workspace_path = make_workspace_with_manifest(
            tmp_path / stage_id,
            topic_slugs=("demo_topic",),
        )
        resolved = workflow.resolve_workspace(workspace_path)
        emit_test_presentation_manifest(resolved)
        seed_declared_stage_inputs(workspace_path, specs_by_id[stage_id])

        commands = workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage(stage_id),
            presentation_type="topic-deep-dive",
            topic="01",
            run_id=TEST_RUN_ID,
        )

        assert commands, stage_id


def test_presentation_stage_command_planning_fails_closed_when_manifest_missing(
    tmp_path: Path,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)

    with pytest.raises(ValueError, match="decks/presentation_manifest.json"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("deck_contract_generation"),
            presentation_type="topic-deep-dive",
            topic="01",
        )


def test_render_export_fails_closed_when_required_render_spec_missing(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)

    with pytest.raises(
        ValueError,
        match="Required presentation input artifact is missing: decks/01_demo_topic/render_spec.json",
    ):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("render_export"),
            presentation_type="topic-deep-dive",
            topic="01",
        )


def test_video_generation_fails_closed_when_slide_mapping_missing(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path, include_slide_mapping=False)

    with pytest.raises(
        ValueError,
        match="Required presentation input artifact is missing: decks/01_demo_topic/presentation_slide_mapping.json",
    ):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("video_generation"),
            presentation_type="topic-deep-dive",
            topic="01",
        )


def _metadata_output_paths_by_owner(metadata: list[dict[str, object]]) -> dict[str, set[str]]:
    return {
        str(item["owner_ref"]): {
            str(artifact["path"])
            for artifact in item["output_artifacts"]
            if isinstance(artifact, dict)
        }
        for item in metadata
    }


def test_source_binding_command_metadata_uses_owner_specific_outputs(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("source_binding"),
        presentation_type="topic-deep-dive",
        topic="01",
        run_id=TEST_RUN_ID,
    )

    assert _metadata_output_paths_by_owner(metadata) == {
        "scripts/presentations/build_deck_source_pack.py": {
            "decks/01_demo_topic/ranked_papers.json",
            "decks/01_demo_topic/source_pack_receipt.json",
        },
        "scripts/presentations/bind_presentation_sources.py": {
            "decks/01_demo_topic/source_binding_report.json",
        },
    }


def test_source_binding_uses_analysis_provenance_matched_papers_when_run_ids_differ(
    tmp_path: Path,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)
    write_text(
        workspace_path / "analysis" / "provenance" / "01_demo_topic.json",
        json.dumps({"run_id": "analysis-run"}, indent=2) + "\n",
    )
    write_text(
        workspace_path
        / "storage"
        / "paper-matching"
        / "analysis-run"
        / "01_demo_topic"
        / "matched_papers.json",
        '{"papers": []}\n',
    )
    write_text(
        workspace_path
        / "storage"
        / "extractions"
        / "analysis-run"
        / "by-topic"
        / "01_demo_topic.json",
        '{"items": []}\n',
    )

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("source_binding"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        run_id="presentation-run",
    )
    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("source_binding"),
        presentation_type="topic-deep-dive",
        topic="01_demo_topic",
        run_id="presentation-run",
    )

    matched_path = (
        workspace_path
        / "storage"
        / "paper-matching"
        / "analysis-run"
        / "01_demo_topic"
        / "matched_papers.json"
    )
    assert any(str(matched_path) in command for command in commands)
    binding_command = next(
        command
        for command in commands
        if "scripts/presentations/bind_presentation_sources.py" in command
    )
    assert "--matched-papers" in binding_command
    assert str(matched_path) in binding_command
    input_paths = {
        artifact["path"]
        for artifact in metadata[0]["input_artifacts"]
        if isinstance(artifact, dict)
    }
    assert (
        "storage/paper-matching/analysis-run/01_demo_topic/matched_papers.json"
        in input_paths
    )
    assert "storage/extractions/analysis-run/by-topic/01_demo_topic.json" in input_paths


def test_source_binding_live_commands_use_structured_contracts(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("source_binding"),
        presentation_type="topic-deep-dive",
        topic="01",
        run_id=TEST_RUN_ID,
        mode="live",
    )

    binding_command = next(
        command for command in commands if command[1] == "scripts/presentations/bind_presentation_sources.py"
    )
    assert "--deck-contract" in binding_command
    assert binding_command[binding_command.index("--deck-contract") + 1] == str(
        resolved.root / "decks" / "01_demo_topic" / "presentation_deck.json"
    )
    assert "--dry-run" not in binding_command


def test_visual_plan_generation_emits_programmatic_owner_command(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("visual_plan_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
        run_id=TEST_RUN_ID,
    )

    assert [item["owner_ref"] for item in metadata] == [
        "scripts/presentations/generate_programmatic_figures.py"
    ]
    assert metadata[0]["unsupported_reason"] is None
    output_paths = {artifact["path"] for artifact in metadata[0]["output_artifacts"]}
    assert output_paths == {
        "decks/01_demo_topic/visual_plan.json",
        "decks/01_demo_topic/asset_inventory.json",
        "decks/01_demo_topic/assets/generated/figure_specs.json",
        "decks/01_demo_topic/assets/generated/semantic_review.json",
        "decks/01_demo_topic/assets/generated/semantic_review.md",
        "decks/01_demo_topic/assets/generated/",
    }
    assert "--asset-inventory" in metadata[0]["argv"]
    assert "--generated-dir" in metadata[0]["argv"]


def test_visual_qc_runs_llm_owner_after_rendered_qc_failure(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)
    commands_run: list[str] = []

    def command_runner(command, cwd):  # noqa: ARG001
        owner_ref = command[1]
        commands_run.append(owner_ref)
        if owner_ref == "scripts/presentations/inspect_rendered_slides.py":
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(RuntimeError, match="visual_qc failed after running all"):
        workflow.run_stage(
            resolved,
            workflow.get_workflow_stage("visual_qc"),
            topic="01",
            presentation_type="topic-deep-dive",
            run_id=TEST_RUN_ID,
            command_runner=command_runner,
        )

    assert commands_run == [
        "scripts/presentations/inspect_rendered_slides.py",
        "scripts/presentations/inspect_rendered_slides_llm.py",
    ]


def test_video_generation_command_metadata_uses_owner_specific_outputs(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("video_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert _metadata_output_paths_by_owner(metadata) == {
        "scripts/presentations/build_presenter_video.py": {
            "decks/01_demo_topic/video_build/",
            "decks/01_demo_topic/presentation_video.mp4",
        },
        "scripts/presentations/build_notebooklm_style_presenter_video.py": {
            "decks/01_demo_topic/video_build_notebooklm/",
            "decks/01_demo_topic/presentation_video_notebooklm.mp4",
        },
    }


def test_render_export_emits_revealjs_pdf_owner_command(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)
    seed_presentation_command_inputs(workspace_path)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("render_export"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("render_export"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    pdf_command = commands[1]
    assert pdf_command[:2] == [sys.executable, "scripts/presentations/md_to_pptx.py"]
    assert "--workspace" in pdf_command
    assert "--presentation-id" in pdf_command
    assert "--presentation-type" in pdf_command
    assert "--render-spec" in pdf_command
    assert "--output-pdf" in pdf_command
    assert metadata[1]["unsupported_reason"] is None
    assert _metadata_output_paths_by_owner(metadata) == {
        "scripts/presentations/md_to_revealjs.py": {
            "decks/01_demo_topic/presentation.html",
            "decks/01_demo_topic/presentation_slide_mapping.json",
        },
        "scripts/presentations/md_to_pptx.py": {
            "decks/01_demo_topic/presentation.pdf",
        }
    }


def test_human_review_command_plan_reports_unsupported_reason(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("human_review"),
        presentation_type="topic-deep-dive",
        topic="01",
    )
    metadata = workflow.build_stage_command_metadata(
        resolved,
        workflow.get_workflow_stage("human_review"),
        presentation_type="topic-deep-dive",
        topic="01",
    )

    assert commands[0][:2] == ["presentation-stage-unsupported", "human_review"]
    assert "--workspace" in commands[0]
    assert "--presentation-id" in commands[0]
    assert "--presentation-type" in commands[0]
    assert metadata[0]["unsupported_reason"] == (
        "human_review is operator-gated and not a script command"
    )


def test_presentation_stage_requires_topic_for_topic_deep_dive(tmp_path: Path):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    emit_test_presentation_manifest(resolved)

    with pytest.raises(ValueError, match="--topic"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("deck_contract_generation"),
            presentation_type="topic-deep-dive",
        )


def test_deck_stage_omits_deprecated_paper_figure_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic="01",
    )
    command_text = "\n".join(" ".join(command) for command in commands)
    assert "extract_topic_figures.py" not in command_text
    assert "add_images_to_md.py" not in command_text


def test_deck_reflow_uses_reflow_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic="01",
    )
    reflow_cmd = next(command for command in commands if command[1].endswith("reflow_slides.py"))

    assert "--model" in reflow_cmd
    assert resolved.workflow.deck.reflow_model in reflow_cmd
    assert reflow_cmd[-1] == "-v"


def test_deck_slides_scripts_carry_target_slides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic="01",
    )
    for script_name in ("generate_slides_agent.py", "enrich_slides_with_papers.py"):
        command = next(item for item in commands if item[1].endswith(script_name))
        slides_index = command.index("--slides")
        assert command[slides_index + 1] == str(resolved.workflow.deck.target_slides)
        assert command[-1] == "-v"


def test_deck_all_topics_passes_all_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic=None,
    )

    assert len(commands) == 6
    assert all("--all" in command for command in commands)


def test_deck_single_topic_maps_to_two_digit_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_path = make_workspace_with_manifest(
        tmp_path,
        topic_slugs=("demo_topic", "second_topic", "third_topic"),
    )
    resolved = workflow.resolve_workspace(workspace_path)
    allow_workspace_as_default(monkeypatch, resolved)

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("deck"),
        topic="2",
    )

    assert len(commands) == 6
    for command in commands:
        topic_index = command.index("--topic")
        assert command[topic_index + 1] == "02"


@pytest.mark.parametrize("mode", ["live", "dry-run", "mock"])
def test_build_stage_commands_fail_closed_in_all_modes(tmp_path: Path, mode: str):
    from conference_pipeline.workflow_runs import (
        WorkflowExecutionMode,
        WorkflowRunOptions,
        run_workflow_stages,
    )

    workspace_path = make_workspace_with_manifest(tmp_path, topic_slugs=("demo_topic",))
    (workspace_path / "analysis" / "analysis_manifest.json").unlink()
    resolved = workflow.resolve_workspace(workspace_path)
    run_id = f"fail-closed-{mode.replace('-', '_')}"
    options = WorkflowRunOptions(run_id=run_id, mode=WorkflowExecutionMode(mode))
    podcast_stage = workflow.get_workflow_stage("podcast")

    with pytest.raises(ManifestValidationError):
        run_workflow_stages(resolved, [podcast_stage], options)

    receipt_path = (
        resolved.root
        / "storage"
        / "workflow-runs"
        / run_id
        / f"stage-{podcast_stage.order:02d}-podcast.json"
    )
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert "preflight failed" in data["error"]
