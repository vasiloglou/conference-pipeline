from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.blueprint import write_blueprint_receipt
from conference_pipeline.presentation.manifest import (
    DEFAULT_STAGE_SPECS,
    PRESENTATION_STAGE_IDS,
    PresentationEntry,
    PresentationStageState,
    PresentationTopicScope,
    build_backfilled_presentations,
    emit_presentation_manifest,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _frame_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "presentation_id": "01_demo_topic",
        "topic_id": "01",
        "retrieval_title": "Demo Topic",
        "audience_title": "Demo Topic for the Field",
        "field_thesis": "The demo topic is framed around field-level questions before mechanisms.",
        "general_questions": [
            "What changed in the area?",
            "Where do systems still fail?",
            "Which evidence should guide adoption?",
        ],
        "opening_anchor_policy": {
            "rules": ["Open with field-level evidence before specialist mechanisms."]
        },
        "source_analysis_path": "analysis/01_demo_topic.md",
    }


def _write_blueprint_inputs(workspace: workflow.WorkflowWorkspace) -> Path:
    deck_dir = workspace.root / "decks" / "01_demo_topic"
    write_text(
        deck_dir / "presentation_frame.json",
        json.dumps(_frame_payload(), indent=2, sort_keys=True) + "\n",
    )
    write_text(deck_dir / "speaker_author_work_ledger.json", '{"entries":[]}\n')
    write_text(
        workspace.root / "analysis" / "exhaustive_summaries" / "01_demo_topic__papers.json",
        '{"analysis_feed":{}}\n',
    )
    return deck_dir


def _make_workspace(tmp_path: Path) -> workflow.WorkflowWorkspace:
    workspace = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
            "workflow:\n"
            "  deck:\n"
            "    required_env: []\n"
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(workspace / "analysis" / "01_demo_topic.md", "# Demo Topic\n")
    return workflow.resolve_workspace(workspace)


def _write_blueprint_files(workspace: workflow.WorkflowWorkspace) -> Path:
    deck_dir = _write_blueprint_inputs(workspace)
    blueprint_path = deck_dir / "presentation_blueprint.json"
    shutil.copyfile(FIXTURE_ROOT / "valid_blueprint.json", blueprint_path)
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    blueprint["presentation_frame_validation"] = {
        "schema_version": 1,
        "status": "pass",
        "errors": [],
        "warnings": [],
    }
    blueprint["input_hashes"] = {
        **(blueprint.get("input_hashes") if isinstance(blueprint.get("input_hashes"), dict) else {}),
        "presentation_frame.json": sha256_file(deck_dir / "presentation_frame.json"),
    }
    blueprint_path.write_text(json.dumps(blueprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    shutil.copyfile(FIXTURE_ROOT / "approved_blueprint_review.json", deck_dir / "blueprint_review.json")
    review_path = deck_dir / "blueprint_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["blueprint_sha256"] = sha256_file(blueprint_path)
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_text(deck_dir / "presentation_blueprint.md", "# Presentation Blueprint\n")
    write_text(deck_dir / "blueprint_review.md", "# Blueprint Review\n")
    write_blueprint_receipt(
        workspace.root,
        blueprint_path,
        receipt_path=deck_dir / "presentation_blueprint_receipt.json",
        presentation_id="01_demo_topic",
        review_path=review_path,
    )
    receipt_path = deck_dir / "presentation_blueprint_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.update(
        {
            "mode": "live",
            "production_evidence": True,
            "provider_receipt_paths": ["storage/provider-calls/blueprint-generation/demo.json"],
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return deck_dir


def _emit_manifest(workspace: workflow.WorkflowWorkspace) -> None:
    stage_states = {
        stage_id: PresentationStageState(status="missing", artifacts=(), notes=())
        for stage_id in PRESENTATION_STAGE_IDS
    }
    emit_presentation_manifest(
        workspace,
        (
            PresentationEntry(
                id="01_demo_topic",
                presentation_type="topic-deep-dive",
                title="Demo Topic",
                topic_scope=PresentationTopicScope(mode="single-topic", topics=("01_demo_topic",)),
                target_duration_minutes=38,
                target_slide_count=2,
                source_analysis_artifacts=(),
                stage_states=stage_states,
            ),
        ),
        generator="test.blueprint-gate@dev",
    )


def test_presentation_blueprint_stage_immediately_precedes_deck_contract_generation() -> None:
    assert PRESENTATION_STAGE_IDS.index("presentation_blueprint") + 1 == PRESENTATION_STAGE_IDS.index("deck_contract_generation")
    assert tuple(spec.id for spec in DEFAULT_STAGE_SPECS) == PRESENTATION_STAGE_IDS
    spec = next(item for item in DEFAULT_STAGE_SPECS if item.id == "presentation_blueprint")

    assert spec.output_artifacts == (
        "decks/<presentation>/presentation_frame.json",
        "decks/<presentation>/presentation_blueprint.json",
        "decks/<presentation>/presentation_blueprint.md",
        "decks/<presentation>/presentation_blueprint_receipt.json",
        "decks/<presentation>/blueprint_review.json",
        "decks/<presentation>/blueprint_review.md",
    )
    assert "scripts/presentations/generate_editorial_blueprint.py" in spec.owner_refs
    assert "scripts/presentations/generate_presentation_blueprint.py" not in spec.owner_refs
    assert "scripts/presentations/review_presentation_blueprint.py" in spec.owner_refs


def test_live_presentation_blueprint_generation_command_uses_live_mode(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    deck_dir = _write_blueprint_inputs(workspace)
    _emit_manifest(workspace)

    commands = workflow.build_stage_commands(
        workspace,
        workflow.get_workflow_stage("presentation_blueprint"),
        presentation_type="topic-deep-dive",
        topic="01",
        mode="live",
    )

    generator_command = next(
        command
        for command in commands
        if command[1] == "scripts/presentations/generate_editorial_blueprint.py"
    )
    assert "--presentation-frame" in generator_command
    assert str(deck_dir / "presentation_frame.json") in generator_command
    assert "--speaker-author-work-ledger" in generator_command
    assert "--exhaustive-papers" in generator_command
    assert "--final-report-url" in generator_command
    assert "--provider-cost-approved" in generator_command
    assert "--mode" not in generator_command
    assert "--dry-run" not in generator_command


def test_blueprint_state_complete_only_for_hash_bound_approved_receipt(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    _write_blueprint_files(workspace)

    entry = next(item for item in build_backfilled_presentations(workspace))
    state = entry.stage_states["presentation_blueprint"]

    assert state.status == "complete"
    assert {
        artifact.path for artifact in state.artifacts
    } == {
        "decks/01_demo_topic/presentation_frame.json",
        "decks/01_demo_topic/presentation_blueprint.json",
        "decks/01_demo_topic/presentation_blueprint.md",
        "decks/01_demo_topic/presentation_blueprint_receipt.json",
        "decks/01_demo_topic/blueprint_review.json",
        "decks/01_demo_topic/blueprint_review.md",
    }

    review_path = workspace.root / "decks" / "01_demo_topic" / "blueprint_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["blueprint_sha256"] = "0" * 64
    review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    repaired = next(item for item in build_backfilled_presentations(workspace))
    assert repaired.stage_states["presentation_blueprint"].status == "partial"


def test_blueprint_state_requires_current_presentation_frame_hash(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    deck_dir = _write_blueprint_files(workspace)

    frame = _frame_payload()
    frame["audience_title"] = "Mutated Frame"
    write_text(
        deck_dir / "presentation_frame.json",
        json.dumps(frame, indent=2, sort_keys=True) + "\n",
    )

    entry = next(item for item in build_backfilled_presentations(workspace))

    assert entry.stage_states["presentation_blueprint"].status == "partial"
    assert entry.stage_states["presentation_blueprint"].notes == (
        "Presentation blueprint is not hash-bound to the current presentation_frame.json.",
    )


def test_backfilled_legacy_markdown_is_debug_only_without_blueprint_gate(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    deck_dir = workspace.root / "decks" / "01_demo_topic"
    write_text(deck_dir / "1_draft_slides.md", "# Draft\n")

    entry = next(item for item in build_backfilled_presentations(workspace))

    assert entry.stage_states["presentation_blueprint"].status == "missing"
    assert entry.stage_states["deck_contract_generation"].status == "missing"
    assert entry.legacy_markdown is not None
    assert entry.legacy_markdown.production_evidence is False
    assert "1_draft_slides.md" in entry.legacy_markdown.artifacts[0].path


def test_live_deck_contract_generation_consumes_approved_blueprint_token(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    deck_dir = _write_blueprint_files(workspace)
    _emit_manifest(workspace)

    commands = workflow.build_stage_commands(
        workspace,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
        mode="live",
    )
    metadata = workflow.build_stage_command_metadata(
        workspace,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
        mode="live",
    )

    assert "--presentation-blueprint" in commands[0]
    assert str(deck_dir / "presentation_blueprint.json") in commands[0]
    assert "--blueprint-receipt" in commands[0]
    assert str(deck_dir / "presentation_blueprint_receipt.json") in commands[0]
    assert "--blueprint-review" in commands[0]
    assert str(deck_dir / "blueprint_review.json") in commands[0]
    assert commands[0][:2] == [sys.executable, "scripts/presentations/generate_deck_contract.py"]
    assert "--output" in commands[0]
    assert str(deck_dir / "presentation_deck.json") in commands[0]
    assert "--receipt-output" in commands[0]
    assert str(deck_dir / "presentation_deck_receipt.json") in commands[0]
    assert "--analysis-md" not in commands[0]
    assert metadata[0]["argv"] == commands[0]
    assert "decks/01_demo_topic/blueprint_review.json" in json.dumps(metadata)


def test_dry_run_direct_analysis_metadata_is_legacy_mock_only(tmp_path: Path) -> None:
    workspace = _make_workspace(tmp_path)
    _write_blueprint_files(workspace)
    _emit_manifest(workspace)

    metadata = workflow.build_stage_command_metadata(
        workspace,
        workflow.get_workflow_stage("deck_contract_generation"),
        presentation_type="topic-deep-dive",
        topic="01",
        mode="dry-run",
    )

    assert metadata[0]["selected_scope"]["artifact_mode"] == "structured-json"
    assert "--dry-run" in metadata[0]["argv"]
    assert "--analysis-md" not in metadata[0]["argv"]
