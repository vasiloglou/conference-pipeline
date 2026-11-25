"""Tests for the Phase 12 analysis orchestrator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.workflow_runs import (
    WorkflowExecutionMode,
    WorkflowRunOptions,
    run_workflow_stages,
)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_workspace(tmp_path: Path, *, seed_manifest: bool = False) -> Path:
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
    write_text(workspace / "papers.json", json.dumps({"papers": []}) + "\n")
    if seed_manifest:
        write_text(
            workspace / "analysis" / "analysis_manifest.json",
            json.dumps({"schema_version": 1, "topics": []}, indent=2) + "\n",
        )
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


def patch_lightweight_steps(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr("conference_pipeline.analysis.orchestrator.step_coverage", coverage_stub)
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


def test_run_analyze_stage_invokes_six_steps_in_order_with_injected_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.orchestrator import MockStepRunner, run_analyze_stage

    workspace = workflow.resolve_workspace(make_workspace(tmp_path))
    call_order: list[str] = []

    def vector_store_stub(inputs, *, embedding_client):
        call_order.append("vector_store_setup")
        chroma_db_path = inputs.output_dir / "chroma_db"
        chroma_db_path.mkdir(parents=True, exist_ok=True)
        return DummyVectorOutput(chroma_db_path=chroma_db_path, digest="v")

    def gpt_stub(inputs, *, researcher):
        call_order.append("gpt_researcher_hybrid")
        assert inputs.overwrite_existing is False
        return DummyOutput(digest="g")

    def coverage_stub(inputs):
        call_order.append("coverage")
        assert inputs.overwrite_existing is False
        mapping_path = inputs.analysis_dir / "presentation_topic_mapping.json"
        write_text(mapping_path, json.dumps({"mappings": []}) + "\n")
        return DummyCoverageOutput(mapping_path=mapping_path, digest="c")

    def gap_stub(inputs):
        call_order.append("gap_fill")
        assert inputs.overwrite_existing is False
        return DummyOutput(digest="gap")

    def qa_stub(inputs):
        call_order.append("qa")
        assert inputs.overwrite_existing is False
        return DummyOutput(digest="q")

    def traceability_stub(inputs, *, researcher, vector_store):
        call_order.append("traceability")
        assert inputs.overwrite_existing is False
        return DummyTraceabilityOutput(digest="t")

    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_vector_store_setup",
        vector_store_stub,
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_gpt_researcher_hybrid",
        gpt_stub,
    )
    monkeypatch.setattr("conference_pipeline.analysis.orchestrator.step_coverage", coverage_stub)
    monkeypatch.setattr("conference_pipeline.analysis.orchestrator.step_gap_fill", gap_stub)
    monkeypatch.setattr("conference_pipeline.analysis.orchestrator.step_qa", qa_stub)
    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.step_traceability",
        traceability_stub,
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.manifest.emit_analysis_manifest",
        lambda ws, *, generator=None: (
            ws.root / "analysis" / "analysis_manifest.json",
            "0" * 64,
            "1" * 64,
        ),
    )

    runner = MockStepRunner.from_fixture(Path("tests/fixtures/analysis"))
    result = run_analyze_stage(workspace, step_runner=runner)

    assert call_order == [
        "vector_store_setup",
        "gpt_researcher_hybrid",
        "coverage",
        "gap_fill",
        "qa",
        "traceability",
    ]
    assert set(result.sub_step_digests) == set(call_order)
    assert result.manifest_content_sha256 == "1" * 64


def test_run_analyze_stage_emits_valid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.manifest import load_and_validate_manifest
    from conference_pipeline.analysis.orchestrator import MockStepRunner, run_analyze_stage

    patch_lightweight_steps(monkeypatch)
    workspace = workflow.resolve_workspace(make_workspace(tmp_path))
    runner = MockStepRunner.from_fixture(Path("tests/fixtures/analysis"))

    result = run_analyze_stage(workspace, step_runner=runner)
    loaded = load_and_validate_manifest(workspace, require_path_existence=False)

    assert result.manifest_path.exists()
    assert loaded.content_sha256 == result.manifest_content_sha256
    assert loaded.manifest.schema_version == 1


def test_analyze_stage_produces_receipt_with_manifest_artifact_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.orchestrator import AnalyzeStageResult

    workspace = workflow.resolve_workspace(make_workspace(tmp_path))
    manifest_path = workspace.root / "analysis" / "analysis_manifest.json"
    write_text(manifest_path, "{}\n")
    synthesized = AnalyzeStageResult(
        manifest_path=manifest_path,
        manifest_sha256="a" * 64,
        manifest_content_sha256="b" * 64,
        sub_step_digests={
            key: "c" * 64
            for key in (
                "vector_store_setup",
                "gpt_researcher_hybrid",
                "coverage",
                "gap_fill",
                "qa",
                "traceability",
            )
        },
    )
    monkeypatch.setattr(
        "conference_pipeline.analysis.run_analyze_stage",
        lambda *args, **kwargs: synthesized,
    )
    monkeypatch.setattr(workflow, "get_default_workspace_path", lambda: workspace.root)

    summary = run_workflow_stages(
        workspace,
        [workflow.get_workflow_stage("analyze")],
        WorkflowRunOptions(
            run_id="analyze-live",
            mode=WorkflowExecutionMode.LIVE,
            topic=None,
        ),
    )

    receipt = summary.stages[0]
    assert receipt.status == "succeeded"
    assert len(receipt.artifacts) == 1
    entry = receipt.artifacts[0]
    assert entry["kind"] == "analysis_manifest"
    assert entry["path"] == "analysis/analysis_manifest.json"
    assert entry["manifest_sha256"] == "a" * 64
    assert entry["manifest_content_sha256"] == "b" * 64
    assert set(entry["sub_step_digests"]) == {
        "vector_store_setup",
        "gpt_researcher_hybrid",
        "coverage",
        "gap_fill",
        "qa",
        "traceability",
    }


def test_analyze_stage_not_called_in_mock_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_live_call(*args, **kwargs):
        raise AssertionError("live stage called in mock mode")

    monkeypatch.setattr(
        "conference_pipeline.analysis.orchestrator.run_analyze_stage",
        fail_live_call,
    )
    workspace = workflow.resolve_workspace(make_workspace(tmp_path))

    summary = run_workflow_stages(
        workspace,
        [workflow.get_workflow_stage("analyze")],
        WorkflowRunOptions(
            run_id="analyze-mock",
            mode=WorkflowExecutionMode.MOCK,
            topic=None,
        ),
    )

    assert summary.stages[0].stage_id == "analyze"
    assert summary.stages[0].status == "succeeded"


def test_run_analyze_stage_consumes_narrative_source_packs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.manifest import load_and_validate_manifest
    from conference_pipeline.analysis.orchestrator import MockStepRunner, run_analyze_stage

    patch_lightweight_steps(monkeypatch)
    workspace = workflow.resolve_workspace(make_workspace(tmp_path))
    source_pack_dir = workspace.root / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True)
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
    runner = MockStepRunner.from_fixture(Path("tests/fixtures/analysis"))

    result = run_analyze_stage(workspace, step_runner=runner, run_id="generic-run")
    loaded = load_and_validate_manifest(workspace)
    topic = loaded.manifest.topics[0]

    assert not isinstance(result, dict)
    assert "narrative_provenance" in result.sub_step_digests
    assert "## Executive Summary" in (workspace.root / "analysis" / "01_demo_topic.md").read_text(
        encoding="utf-8"
    )
    assert topic.provenance_path == "analysis/provenance/01_demo_topic.md"
    assert topic.provenance_json_path == "analysis/provenance/01_demo_topic.json"
    assert (workspace.root / "analysis" / "analysis_citation_manifest.json").exists()


def test_run_analyze_stage_blocks_when_narrative_source_pack_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from conference_pipeline.analysis.orchestrator import MockStepRunner, run_analyze_stage

    patch_lightweight_steps(monkeypatch)
    workspace = workflow.resolve_workspace(make_workspace(tmp_path))
    source_pack_dir = workspace.root / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True)
    write_text(
        source_pack_dir / "01_demo_topic.json",
        json.dumps(
            {
                "topic_id": "01_demo_topic",
                "title": "Demo Topic",
                "finding_text": "",
                "chunks": [],
                "research_id": "",
            },
            indent=2,
        )
        + "\n",
    )
    runner = MockStepRunner.from_fixture(Path("tests/fixtures/analysis"))

    result = run_analyze_stage(workspace, step_runner=runner, run_id="generic-run")

    assert isinstance(result, dict)
    assert result["status"] == "blocked"
    assert result["blockers"][0]["code"] == "narrative_source_pack_insufficient"
    assert "Status: blocked" in (workspace.root / "analysis" / "01_demo_topic.md").read_text(
        encoding="utf-8"
    )
