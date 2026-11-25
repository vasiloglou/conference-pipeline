from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.presentation.manifest import PRESENTATION_STAGE_IDS
from conference_pipeline.presentation.validation import VALIDATION_STAGE_IDS


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "conferences" / "demo2026"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
            "workflow:\n"
            "  podcast:\n"
            "    target_minutes: 15\n"
            "    max_tier: 2\n"
            "    top_n_facts: 25\n"
            '    host_a: "Taylor"\n'
            '    host_b: "Jordan"\n'
            '    model: "demo-model"\n'
            "  deck:\n"
            "    target_slides: 24\n"
            '    reflow_model: "gemini"\n'
            "    max_papers: 4\n"
            "    required_env:\n"
            '      - "OPENAI_API_KEY"\n'
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    write_text(workspace / "analysis" / "01_demo_topic.md", "# demo\n")
    write_text(workspace / "analysis" / "02_other_topic.md", "# demo\n")
    write_text(workspace / "audio" / "01_demo_topic" / "ranks.json", '{"ranks": []}\n')
    write_text(workspace / "audio" / "02_other_topic" / "ranks.json", '{"ranks": []}\n')
    resolved = workflow.resolve_workspace(workspace)
    from conference_pipeline.analysis.manifest import emit_analysis_manifest

    emit_analysis_manifest(resolved, generator="test.fixture@dev")
    return workspace


@dataclass(frozen=True)
class DummyOutput:
    digest: str


@dataclass(frozen=True)
class DummyVectorOutput:
    chroma_db_path: Path
    digest: str


@dataclass(frozen=True)
class DummyCoverageOutput:
    mapping_path: Path
    digest: str


@dataclass(frozen=True)
class DummyTraceabilityOutput:
    digest: str


def patch_lightweight_analyze_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    def vector_store_stub(inputs, *, embedding_client):
        chroma_db_path = inputs.output_dir / "chroma_db"
        chroma_db_path.mkdir(parents=True, exist_ok=True)
        return DummyVectorOutput(chroma_db_path=chroma_db_path, digest="v")

    def gpt_stub(inputs, *, researcher):
        for topic in inputs.topics:
            write_text(inputs.output_dir / f"{topic}.md", f"# {topic}\n")
        return DummyOutput(digest="g")

    def coverage_stub(inputs):
        mapping_path = inputs.analysis_dir / "presentation_topic_mapping.json"
        write_text(mapping_path, json.dumps({"mappings": []}) + "\n")
        return DummyCoverageOutput(mapping_path=mapping_path, digest="c")

    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_vector_store_setup",
        vector_store_stub,
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_gpt_researcher_hybrid",
        gpt_stub,
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_coverage",
        coverage_stub,
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_gap_fill",
        lambda inputs: DummyOutput(digest="gap"),
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_qa",
        lambda inputs: DummyOutput(digest="q"),
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_traceability",
        lambda inputs, *, researcher, vector_store: DummyTraceabilityOutput(digest="t"),
    )


def write_valid_narrative_source_pack(workspace: Path) -> None:
    source_pack_dir = workspace / "analysis" / "narrative_source_packs"
    write_text(
        source_pack_dir / "01_demo_topic.json",
        json.dumps(
            {
                "topic_id": "01_demo_topic",
                "title": "Demo Topic",
                "research_id": "demo#research",
                "research_path": "storage/research/demo.md",
                "finding_text": "Demo findings connect source reading to slide narrative.",
                "chunks": [
                    {
                        "chunk_id": f"chunk-{index}",
                        "extraction_record_id": f"extract-{index}",
                        "source_artifact_path": f"storage/source/{index}.json",
                        "quote": f"Source quote {index}.",
                    }
                    for index in range(4)
                ],
                "matched_papers": [
                    {
                        "paper_id": f"paper-{index}",
                        "title": f"Demo Paper {index}",
                        "evidence_snippet": "Evidence.",
                    }
                    for index in range(4)
                ],
                "extraction_record_ids": [f"extract-{index}" for index in range(4)],
            },
            indent=2,
        )
        + "\n",
    )


def test_workflow_registry_has_canonical_stage_order_and_schema_fields():
    stages = workflow.ordered_workflow_stages()
    legacy_stage_ids = [
        "discover",
        "download",
        "transcribe",
        "analyze",
        "podcast",
        "deck",
    ]
    extraction_stage_ids = [stage.id for stage in workflow.EXTRACTION_WORKFLOW_STAGES]
    indexing_stage_ids = [stage.id for stage in workflow.INDEXING_WORKFLOW_STAGES]
    research_stage_ids = [stage.id for stage in workflow.RESEARCH_WORKFLOW_STAGES]
    paper_matching_stage_ids = [stage.id for stage in workflow.PAPER_MATCHING_WORKFLOW_STAGES]
    coverage_stage_ids = [stage.id for stage in workflow.COVERAGE_WORKFLOW_STAGES]
    synthesis_stage_ids = [stage.id for stage in workflow.SYNTHESIS_WORKFLOW_STAGES]
    notebooklm_podcast_stage_ids = [stage.id for stage in workflow.NOTEBOOKLM_PODCAST_WORKFLOW_STAGES]

    assert [stage.id for stage in stages] == (
        legacy_stage_ids
        + list(PRESENTATION_STAGE_IDS)
        + list(VALIDATION_STAGE_IDS)
        + extraction_stage_ids
        + indexing_stage_ids
        + research_stage_ids
        + paper_matching_stage_ids
        + coverage_stage_ids
        + synthesis_stage_ids
        + notebooklm_podcast_stage_ids
    )
    assert notebooklm_podcast_stage_ids == ["notebooklm-podcast"]
    # Legacy + presentation + validation orders are strictly monotonic through
    # the presentation contract stages; later production groups append after.
    legacy_count = (
        len(legacy_stage_ids)
        + len(PRESENTATION_STAGE_IDS)
        + len(VALIDATION_STAGE_IDS)
    )
    expected_orders = (
        list(range(1, legacy_count + 1))
        + [stage.order for stage in workflow.EXTRACTION_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.INDEXING_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.RESEARCH_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.PAPER_MATCHING_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.COVERAGE_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.SYNTHESIS_WORKFLOW_STAGES]
        + [stage.order for stage in workflow.NOTEBOOKLM_PODCAST_WORKFLOW_STAGES]
    )
    assert [stage.order for stage in stages] == expected_orders

    for stage in stages:
        assert stage.dispatch_kind in {"python_stage", "subprocess_stage"}
        assert stage.implementation_ref
        assert stage.requires_workspace_files[:2] == ("config.yaml", "storage/storage_manifest.json")
    analyze = workflow.get_workflow_stage("analyze")
    assert analyze.dispatch_kind == "python_stage"
    assert analyze.implementation_ref == "conference_pipeline.analysis.run_analyze_stage@v1"
    deck = workflow.get_workflow_stage("deck")
    assert deck.order == 6
    assert PRESENTATION_STAGE_IDS[0] == "presentation_blueprint"
    assert workflow.get_workflow_stage("presentation_blueprint").order == 7
    assert workflow.get_workflow_stage("deck_contract_generation").order == 8
    assert workflow.get_workflow_stage(VALIDATION_STAGE_IDS[0]).order == len(legacy_stage_ids) + len(PRESENTATION_STAGE_IDS) + 1


def test_presentation_blueprint_workflow_stage_precedes_deck_contract_generation():
    stage_ids = [stage.id for stage in workflow.PRESENTATION_WORKFLOW_STAGES]

    assert stage_ids.index("presentation_blueprint") + 1 == stage_ids.index("deck_contract_generation")
    stage = workflow.get_workflow_stage("presentation_blueprint")
    assert stage.implementation_ref == "presentation_manifest.stage_specs.presentation_blueprint"


def test_workflow_registry_can_hide_presentation_stages_for_scope_unaware_callers():
    legacy_stage_ids = [
        "discover",
        "download",
        "transcribe",
        "analyze",
        "podcast",
        "deck",
    ]

    stages = workflow.ordered_workflow_stages(include_presentation=False)

    assert [stage.id for stage in stages] == legacy_stage_ids
    assert [
        stage.id
        for stage in workflow.parse_stage_selection(None, include_presentation=False)
    ] == legacy_stage_ids
    with pytest.raises(ValueError, match="Unknown workflow stage"):
        workflow.parse_stage_selection(PRESENTATION_STAGE_IDS[0], include_presentation=False)


def test_parse_stage_selection_keeps_registry_order_and_deduplicates():
    selected = workflow.parse_stage_selection("deck_contract_generation,deck,discover,deck,podcast")

    assert [stage.id for stage in selected] == [
        "discover",
        "podcast",
        "deck",
        "deck_contract_generation",
    ]


def test_parse_stage_selection_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown workflow stage"):
        workflow.parse_stage_selection("discover,unknown")


def test_resolve_workspace_loads_defaults_and_topics(tmp_path: Path):
    workspace_root = make_workspace(tmp_path)

    resolved = workflow.resolve_workspace(workspace_root)

    assert resolved.root == workspace_root.resolve()
    assert resolved.config_path == (workspace_root / "config.yaml").resolve()
    assert resolved.manifest_path == (workspace_root / "storage" / "storage_manifest.json").resolve()
    assert resolved.workflow.podcast.target_minutes == 15
    assert resolved.workflow.podcast.host_a == "Taylor"
    assert resolved.workflow.deck.target_slides == 24
    assert resolved.workflow.deck.required_env == ("OPENAI_API_KEY",)
    assert workflow.list_workflow_topics(resolved) == ["01_demo_topic", "02_other_topic"]


def test_resolve_workspace_rejects_missing_config(tmp_path: Path):
    workspace_root = tmp_path / "conferences" / "demo2026"
    workspace_root.mkdir(parents=True)
    write_text(
        workspace_root / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )

    with pytest.raises(ValueError, match="missing config.yaml"):
        workflow.resolve_workspace(workspace_root)


def test_resolve_workspace_rejects_missing_manifest(tmp_path: Path):
    workspace_root = tmp_path / "conferences" / "demo2026"
    workspace_root.mkdir(parents=True)
    write_text(
        workspace_root / "config.yaml",
        'storage:\n  manifest_path: "storage/storage_manifest.json"\n',
    )

    with pytest.raises(ValueError, match="missing storage manifest"):
        workflow.resolve_workspace(workspace_root)


def test_nondefault_workspace_rejects_unscoped_stages(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))

    with pytest.raises(ValueError, match="discover only supports the canonical NeurIPS workspace"):
        workflow.run_stage(resolved, workflow.get_workflow_stage("discover"))

    with pytest.raises(ValueError, match="deck only supports the canonical NeurIPS workspace"):
        workflow.build_stage_commands(
            resolved,
            workflow.get_workflow_stage("deck"),
            topic="01",
        )


def test_nondefault_analyze_without_narrative_source_packs_still_rejects(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))

    with pytest.raises(ValueError, match="analyze only supports the canonical NeurIPS workspace"):
        workflow.run_stage(resolved, workflow.get_workflow_stage("analyze"))


def test_nondefault_analyze_with_narrative_source_packs_emits_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.manifest import load_and_validate_manifest

    workspace_root = make_workspace(tmp_path)
    write_valid_narrative_source_pack(workspace_root)
    patch_lightweight_analyze_steps(monkeypatch)
    resolved = workflow.resolve_workspace(workspace_root)

    result = workflow.run_stage(
        resolved,
        workflow.get_workflow_stage("analyze"),
        run_id="generic-workflow-run",
    )
    loaded = load_and_validate_manifest(resolved)
    topic = next(
        topic
        for topic in loaded.manifest.topics
        if topic.markdown_path == "analysis/01_demo_topic.md"
    )

    assert not isinstance(result, dict)
    assert "narrative_provenance" in result.sub_step_digests
    assert "## Executive Summary" in (
        workspace_root / "analysis" / "01_demo_topic.md"
    ).read_text(encoding="utf-8")
    assert (workspace_root / "analysis" / "analysis_citation_manifest.json").exists()
    assert (workspace_root / "analysis" / "provenance" / "01_demo_topic.md").exists()
    assert (workspace_root / "analysis" / "provenance" / "01_demo_topic.json").exists()
    assert topic.provenance_path == "analysis/provenance/01_demo_topic.md"
    assert topic.provenance_json_path == "analysis/provenance/01_demo_topic.json"


def test_podcast_commands_remain_workspace_scoped_for_nondefault_workspace(tmp_path: Path):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))

    commands = workflow.build_stage_commands(
        resolved,
        workflow.get_workflow_stage("podcast"),
        topic="01",
    )

    assert commands == [
        [
            sys.executable,
            "scripts/generate_focused_podcast.py",
            str(resolved.root / "analysis" / "01_demo_topic.md"),
            "--fact-ranks",
            str(resolved.root / "audio" / "01_demo_topic" / "ranks.json"),
            "--output-dir",
            str(resolved.root / "audio" / "01_demo_topic" / "focused_podcast"),
            "--target-minutes",
            "15",
            "--max-tier",
            "2",
            "--top-n-facts",
            "25",
            "--two-hosts",
            "--host-a",
            "Taylor",
            "--host-b",
            "Jordan",
            "--model",
            "demo-model",
        ]
    ]


def test_subprocess_stage_commands_use_active_python_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)

    commands = []
    for stage_id in ("podcast", "deck"):
        commands.extend(
            workflow.build_stage_commands(
                resolved,
                workflow.get_workflow_stage(stage_id),
                topic="01",
            )
        )

    assert commands
    assert all(command[0] == sys.executable for command in commands)


def test_extraction_stage_allow_list_rejects_unsupported_workspace_name(tmp_path: Path):
    """Phase 14A allow-list (WR-04 regression).

    The exhaustive-topic-extraction stage allow-list keys on workspace.root.name
    today. This test pins the supported set so adding a new conference is a
    deliberate, reviewed change rather than a quiet allow.
    """
    assert workflow.EXTRACTION_SUPPORTED_WORKSPACE_NAMES == frozenset(
        {"iclr2025", "neurips2025"}
    )

    # An unsupported workspace name (mlsys2025) must raise from
    # _validate_workspace_support, even when otherwise correctly shaped.
    workspace = tmp_path / "conferences" / "mlsys2025"
    write_text(
        workspace / "config.yaml",
        (
            "conference:\n"
            "  name: MLSys 2025\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    write_text(
        workspace / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    resolved = workflow.resolve_workspace(workspace)
    extraction_stage = workflow.EXTRACTION_WORKFLOW_STAGES[0]
    with pytest.raises(ValueError, match="only routed against the ICLR 2025"):
        workflow._validate_workspace_support(resolved, extraction_stage)


def test_deck_stage_requires_configured_env_before_running_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = workflow.resolve_workspace(make_workspace(tmp_path))
    commands_run: list[list[str]] = []

    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: resolved.root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Missing required env for deck stage: OPENAI_API_KEY"):
        workflow.run_stage(
            resolved,
            workflow.get_workflow_stage("deck"),
            topic="01",
            command_runner=lambda command, cwd: commands_run.append(list(command)),
        )

    assert commands_run == []
