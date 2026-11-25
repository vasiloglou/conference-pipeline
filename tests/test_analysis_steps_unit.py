"""Unit tests for composable analysis step functions."""

from __future__ import annotations

import ast
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def seed_presentation(
    presentations_dir: Path,
    folder: str,
    *,
    title: str,
    duration_seconds: int = 3600,
    transcript: str = "This presentation covers deterministic fixture content.",
) -> None:
    pres_dir = presentations_dir / folder
    write_text(pres_dir / "meta.json", json.dumps({"title": title}, indent=2) + "\n")
    write_text(
        pres_dir / "transcripts" / "segments.json",
        json.dumps([{"start": 0, "end": duration_seconds, "text": transcript}], indent=2) + "\n",
    )
    write_text(pres_dir / "transcripts" / "full_transcript.txt", transcript + "\n")


def test_step_coverage_is_pure(tmp_path: Path):
    from conference_pipeline.analysis.steps.coverage import CoverageInputs, step_coverage

    analysis_dir = tmp_path / "analysis"
    presentations_dir = tmp_path / "presentations"
    seed_presentation(presentations_dir, "referenced-prez", title="Referenced Presentation")
    seed_presentation(presentations_dir, "unreferenced-prez", title="Unreferenced Presentation")
    seed_presentation(presentations_dir, "source-title-prez", title="Source Title Presentation")
    write_text(
        analysis_dir / "01_topic.md",
        "# Topic\n\nReference to `referenced-prez`.\n\n**Source:** Source Title Presentation\n",
    )
    write_text(analysis_dir / "02_topic.md", "# Other Topic\n\nNo local references.\n")

    first = step_coverage(CoverageInputs(analysis_dir=analysis_dir, presentations_dir=presentations_dir))
    first_text = first.report_path.read_text(encoding="utf-8")
    second = step_coverage(CoverageInputs(analysis_dir=analysis_dir, presentations_dir=presentations_dir))
    second_text = second.report_path.read_text(encoding="utf-8")

    assert first_text == second_text
    assert first.digest == second.digest
    assert "# Coverage Report" in first_text
    assert "## Summary" in first_text
    assert first.mapping_path.exists()


def test_step_gap_fill_is_pure_with_injected_corrections(tmp_path: Path):
    from conference_pipeline.analysis.steps.gap_fill import GapFillInputs, step_gap_fill

    analysis_dir = tmp_path / "analysis"
    presentations_dir = tmp_path / "presentations"
    mapping_path = analysis_dir / "presentation_topic_mapping.json"
    write_text(analysis_dir / "01_topic.md", "# Topic\n\nExisting content.\n")
    seed_presentation(
        presentations_dir,
        "fake-prez",
        title="Fake Presentation",
        duration_seconds=5400,
        transcript="Intro\n\nThis is a substantive deterministic transcript snippet for gap fill.",
    )
    write_text(
        mapping_path,
        json.dumps(
            {
                "mappings": [
                    {
                        "folder": "fake-prez",
                        "topic": "99_wrong_topic",
                        "title": "Fake Presentation",
                        "duration_minutes": 90,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
    )

    empty = step_gap_fill(
        GapFillInputs(
            mapping_path=mapping_path,
            analysis_dir=analysis_dir,
            presentations_dir=presentations_dir,
            topic_corrections={},
        )
    )
    assert empty.docs_modified == ()
    assert "## Additional Presentations" not in (analysis_dir / "01_topic.md").read_text(
        encoding="utf-8"
    )

    filled = step_gap_fill(
        GapFillInputs(
            mapping_path=mapping_path,
            analysis_dir=analysis_dir,
            presentations_dir=presentations_dir,
            topic_corrections={"fake-prez": "01_topic"},
        )
    )
    content = (analysis_dir / "01_topic.md").read_text(encoding="utf-8")
    assert filled.docs_modified == (analysis_dir / "01_topic.md",)
    assert content.count("## Additional Presentations") == 1

    repeated = step_gap_fill(
        GapFillInputs(
            mapping_path=mapping_path,
            analysis_dir=analysis_dir,
            presentations_dir=presentations_dir,
            topic_corrections={"fake-prez": "01_topic"},
        )
    )
    assert repeated.docs_modified == ()
    assert (analysis_dir / "01_topic.md").read_text(encoding="utf-8").count(
        "## Additional Presentations"
    ) == 1


def test_step_qa_is_pure(tmp_path: Path):
    from conference_pipeline.analysis.steps.qa import QAInputs, step_qa

    analysis_dir = tmp_path / "analysis"
    write_text(
        analysis_dir / "01_topic.md",
        (
            "# Summary\n\n"
            "## Key Findings\n\n"
            "**Source:** Demo Presentation\n"
            "**Timestamp:** 00:01:00\n"
            "https://github.com/example/project\n"
            "[Other](02_topic.md)\n\n"
            "## References\n\n"
            "## Cross-References\n"
        ),
    )
    write_text(analysis_dir / "02_topic.md", "# Sparse\n\nNo details.\n")

    first = step_qa(QAInputs(analysis_dir=analysis_dir))
    second = step_qa(QAInputs(analysis_dir=analysis_dir))

    assert first.digest == second.digest
    assert first.report_path.read_text(encoding="utf-8") == second.report_path.read_text(
        encoding="utf-8"
    )
    assert set(first.per_doc_scores) == {"01_topic.md", "02_topic.md"}


def test_topic_corrections_matches_script_source():
    from conference_pipeline.analysis.steps.gap_fill import TOPIC_CORRECTIONS

    script_text = Path("scripts/fill_coverage_gaps.py").read_text(encoding="utf-8")
    match = re.search(r"TOPIC_CORRECTIONS\s*=\s*(\{.*?\n\})", script_text, re.DOTALL)
    assert match, "Could not extract TOPIC_CORRECTIONS literal from scripts/fill_coverage_gaps.py"
    script_corrections = ast.literal_eval(match.group(1))

    assert script_corrections == TOPIC_CORRECTIONS


def test_step_vector_store_setup_invokes_injected_embedding_client(tmp_path: Path, monkeypatch):
    from conference_pipeline.analysis.orchestrator import FakeEmbeddingClient
    from conference_pipeline.analysis.steps.vector_store_setup import (
        VectorStoreInputs,
        step_vector_store_setup,
    )

    class VectorSetupEmbeddingClient(FakeEmbeddingClient):
        provider = "fake"
        model = "fake-embedding"
        embedding_dim = 8

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return self.embed(texts)

    calls: list[tuple[str, object]] = []

    class FakeCollection:
        def add(self, **kwargs):
            calls.append(("add", kwargs))

    class FakeClient:
        def __init__(self, path: str):
            calls.append(("client", path))

        def get_or_create_collection(self, name: str, **kwargs):
            calls.append(("collection", {"name": name, "kwargs": kwargs}))
            return FakeCollection()

    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(PersistentClient=FakeClient),
    )
    presentations_dir = tmp_path / "presentations"
    seed_presentation(presentations_dir, "demo-prez", title="Demo Presentation")
    papers_file = tmp_path / "papers.json"
    write_text(
        papers_file,
        json.dumps({"papers": [{"id": "p1", "title": "PyTorch Demo", "abstract": "reasoning"}]})
        + "\n",
    )

    output = step_vector_store_setup(
        VectorStoreInputs(
            workspace_root=tmp_path,
            papers_file=papers_file,
            presentations_dir=presentations_dir,
            output_dir=tmp_path / "vector_store",
        ),
        embedding_client=VectorSetupEmbeddingClient(),
    )

    assert output.total_chunks >= 1
    assert output.metadata_path.exists()
    assert output.entity_index_path.exists()
    collection_calls = [payload for kind, payload in calls if kind == "collection"]
    assert collection_calls
    assert collection_calls[0]["name"] == "neurips_presentations__fake_fake-embedding_8"
    metadata = collection_calls[0]["kwargs"]["metadata"]
    assert {"dim", "provider", "model"}.issubset(metadata)


def test_step_gpt_researcher_hybrid_uses_canned_researcher(tmp_path: Path):
    from conference_pipeline.analysis.orchestrator import CannedResearcher
    from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
        GPTResearcherInputs,
        step_gpt_researcher_hybrid,
    )

    inputs = GPTResearcherInputs(
        workspace_root=tmp_path,
        vector_store_dir=tmp_path / "vector_store",
        output_dir=tmp_path / "analysis",
        topics=("01_tools_and_open_source", "02_genai_architectures"),
    )
    researcher = CannedResearcher(by_query={"default": "# Topic\n\n[SYN] content"})

    first = step_gpt_researcher_hybrid(inputs, researcher=researcher)
    second = step_gpt_researcher_hybrid(inputs, researcher=researcher)

    assert first.digest == second.digest
    assert [path.name for path in first.report_paths] == [
        "01_tools_and_open_source.md",
        "02_genai_architectures.md",
    ]
    assert all("[SYN] content" in path.read_text(encoding="utf-8") for path in first.report_paths)


def test_step_gpt_researcher_hybrid_preserves_existing_reports_when_requested(tmp_path: Path):
    from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
        GPTResearcherInputs,
        step_gpt_researcher_hybrid,
    )

    class SpyResearcher:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, query: str, *, context_path: Path | None = None) -> str:
            _ = query
            _ = context_path
            self.calls += 1
            return "# Replacement\n\nThis would be destructive."

    analysis_dir = tmp_path / "analysis"
    report_path = analysis_dir / "01_tools_and_open_source.md"
    original = "# Original Deep Analysis\n\n**Source:** trusted-presentation\n"
    write_text(report_path, original)

    researcher = SpyResearcher()
    output = step_gpt_researcher_hybrid(
        GPTResearcherInputs(
            workspace_root=tmp_path,
            vector_store_dir=tmp_path / "vector_store",
            output_dir=analysis_dir,
            topics=("01_tools_and_open_source",),
            overwrite_existing=False,
        ),
        researcher=researcher,
    )

    assert output.report_paths == (report_path,)
    assert researcher.calls == 0
    assert report_path.read_text(encoding="utf-8") == original


def test_step_coverage_preserves_existing_outputs_when_requested(tmp_path: Path):
    from conference_pipeline.analysis.steps.coverage import CoverageInputs, step_coverage

    analysis_dir = tmp_path / "analysis"
    coverage_path = analysis_dir / "coverage_report.md"
    mapping_path = analysis_dir / "presentation_topic_mapping.json"
    original_coverage = "# Existing Coverage\n\n- Total presentations: 79\n"
    original_mapping = json.dumps({"total_presentations": 79, "mappings": [{"folder": "p1"}]}) + "\n"
    write_text(coverage_path, original_coverage)
    write_text(mapping_path, original_mapping)

    output = step_coverage(
        CoverageInputs(
            analysis_dir=analysis_dir,
            presentations_dir=tmp_path / "missing-presentations",
            overwrite_existing=False,
        )
    )

    assert output.report_path == coverage_path
    assert output.mapping_path == mapping_path
    assert coverage_path.read_text(encoding="utf-8") == original_coverage
    assert mapping_path.read_text(encoding="utf-8") == original_mapping


def test_step_gap_fill_preserves_existing_corrected_mapping_when_requested(tmp_path: Path):
    from conference_pipeline.analysis.steps.gap_fill import GapFillInputs, step_gap_fill

    analysis_dir = tmp_path / "analysis"
    mapping_path = analysis_dir / "presentation_topic_mapping.json"
    corrected_path = analysis_dir / "presentation_topic_mapping_corrected.json"
    topic_path = analysis_dir / "01_topic.md"
    write_text(
        mapping_path,
        json.dumps({"mappings": [{"folder": "fake-prez", "topic": "01_topic"}]}) + "\n",
    )
    original_corrected = json.dumps({"preserved": True}) + "\n"
    original_topic = "# Topic\n\nNo generated additions.\n"
    write_text(corrected_path, original_corrected)
    write_text(topic_path, original_topic)

    output = step_gap_fill(
        GapFillInputs(
            mapping_path=mapping_path,
            analysis_dir=analysis_dir,
            presentations_dir=tmp_path / "presentations",
            overwrite_existing=False,
        )
    )

    assert output.docs_modified == ()
    assert output.corrected_mapping_path == corrected_path
    assert corrected_path.read_text(encoding="utf-8") == original_corrected
    assert topic_path.read_text(encoding="utf-8") == original_topic


def test_step_qa_preserves_existing_report_when_requested(tmp_path: Path):
    from conference_pipeline.analysis.steps.qa import QAInputs, step_qa

    analysis_dir = tmp_path / "analysis"
    qa_path = analysis_dir / "qa_report.md"
    original = "# Existing QA\n\n- Average Score: 93.6/100\n"
    write_text(analysis_dir / "01_topic.md", "# Replacement target\n")
    write_text(qa_path, original)

    output = step_qa(QAInputs(analysis_dir=analysis_dir, overwrite_existing=False))

    assert output.report_path == qa_path
    assert output.per_doc_scores == {}
    assert qa_path.read_text(encoding="utf-8") == original


def test_step_traceability_preserves_existing_addendum_when_requested(tmp_path: Path):
    from conference_pipeline.analysis.orchestrator import CannedVectorStore
    from conference_pipeline.analysis.steps.traceability import (
        TraceabilityInputs,
        step_traceability,
    )

    analysis_dir = tmp_path / "analysis"
    output_dir = analysis_dir / "traceability_addenda"
    write_text(analysis_dir / "01_topic.md", "# Topic\n\n> trusted quote\n")
    addendum_path = output_dir / "01_topic_traceability_addendum.md"
    original = "# Existing Addendum\n"
    write_text(addendum_path, original)

    output = step_traceability(
        TraceabilityInputs(
            analysis_dir=analysis_dir,
            presentations_root=tmp_path / "presentations",
            output_dir=output_dir,
            vector_store_dir=tmp_path / "vector_store",
            overwrite_existing=False,
        ),
        vector_store=CannedVectorStore(),
    )

    assert output.addenda_written == (addendum_path,)
    assert addendum_path.read_text(encoding="utf-8") == original


def test_step_traceability_uses_injected_vector_store(tmp_path: Path):
    from conference_pipeline.analysis.orchestrator import CannedResearcher, CannedVectorStore
    from conference_pipeline.analysis.steps.traceability import (
        TraceabilityInputs,
        step_traceability,
    )

    analysis_dir = tmp_path / "analysis"
    write_text(
        analysis_dir / "01_topic.md",
        (
            "# Topic\n\n"
            "**Source:** Demo Presentation\n"
            "> quote text\n"
        ),
    )
    presentations_root = tmp_path / "presentations"
    write_text(
        presentations_root / "prez-01" / "meta.json",
        json.dumps({"title": "Demo Presentation"}) + "\n",
    )

    output = step_traceability(
        TraceabilityInputs(
            analysis_dir=analysis_dir,
            presentations_root=presentations_root,
            output_dir=analysis_dir / "traceability_addenda",
            vector_store_dir=tmp_path / "vector_store",
        ),
        researcher=CannedResearcher(by_query={}),
        vector_store=CannedVectorStore(
            hits_by_query={
                "quote text": [
                    {
                        "metadata": {
                            "presentation_id": "prez-01",
                            "title": "Demo Presentation",
                            "timestamp_range": "00:00:01 - 00:00:03",
                        },
                        "score": 0.9,
                    }
                ]
            }
        ),
    )

    assert output.addenda_written == (
        analysis_dir / "traceability_addenda" / "01_topic_traceability_addendum.md",
    )
    assert "prez-01" in output.addenda_written[0].read_text(encoding="utf-8")
