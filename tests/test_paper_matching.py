from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class RecordingExtractor:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:
        self.calls.append(f"keypoints:{topic_id}:{'artifacts discuss' in artifact_findings}")
        return ["artifacts discuss retrieval augmentation", "talks emphasize evaluation"]


class RecordingResearcher:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.last_context_path: Path | None = None
        self.last_query = ""

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        self.calls.append("paper_research")
        self.last_query = query
        self.last_context_path = context_path
        return json.dumps(
            {
                "papers": [
                    {
                        "title": "Retrieval Augmentation for Oral Presentations",
                        "paper_id": "paper-1",
                        "url": "https://example.test/paper-1",
                        "evidence_snippet": "Matches the artifact keypoints.",
                    }
                ]
            }
        )


class FailingAsyncResearcher:
    async def run(self, query: str, *, context_path: Path | None = None) -> str:  # noqa: ARG002
        raise RuntimeError("local_synthesis_failed")


class RecordingRunner:
    def __init__(self, name: str, calls: list[str], output: str) -> None:
        self.name = name
        self.calls = calls
        self.output = output
        self.queries: list[str] = []

    def research(self, *, topic_id: str, query: str, output_path: Path) -> str:
        self.calls.append(f"{self.name}:{topic_id}")
        self.queries.append(query)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.output, encoding="utf-8")
        return self.output


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChatResponse:
    def __init__(
        self,
        *,
        content: str,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeChatCompletions:
    def __init__(self, response: _FakeChatResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeChatResponse:
        self.calls.append(kwargs)
        return self._response


class _FakeOpenAIClient:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def test_extract_papers_from_research_parses_json_and_markdown():
    from conference_pipeline.paper_matching import extract_papers_from_research

    json_papers = extract_papers_from_research(
        '{"papers": [{"title": "A JSON Paper", "paper_id": "p1"}]}'
    )
    markdown_papers = extract_papers_from_research(
        "1. Title: A Markdown Paper - because it matches\n2. Paper: Another Paper"
    )

    assert [paper.title for paper in json_papers] == ["A JSON Paper"]
    assert [paper.title for paper in markdown_papers] == [
        "A Markdown Paper",
        "Another Paper",
    ]


def test_extract_papers_drops_mismatched_openreview_model_url():
    from conference_pipeline.paper_matching import extract_papers_from_research

    papers = extract_papers_from_research(
        json.dumps(
            {
                "papers": [
                    {
                        "title": "A Real Paper",
                        "paper_id": "abc123XYZ",
                        "url": "https://openreview.net/pdf/unrelated987.pdf",
                    }
                ]
            }
        )
    )

    assert papers[0].url == ""


def test_extract_papers_derives_short_openreview_url_from_paper_id():
    from conference_pipeline.paper_matching import extract_papers_from_research

    papers = extract_papers_from_research(
        json.dumps({"papers": [{"title": "A Real Paper", "paper_id": "abc123XYZ"}]})
    )

    assert papers[0].url == "https://openreview.net/forum?id=abc123XYZ"


def test_run_paper_matching_uses_artifact_keypoints_before_paper_research(tmp_path):
    from conference_pipeline.paper_matching import run_paper_matching

    workspace = tmp_path / "conf"
    artifact_findings = workspace / "storage" / "research" / "run-1" / "findings" / "topic-a.md"
    artifact_findings.parent.mkdir(parents=True)
    artifact_findings.write_text(
        "# Topic A\n\nartifacts discuss retrieval augmentation in oral sessions.\n",
        encoding="utf-8",
    )
    calls: list[str] = []
    extractor = RecordingExtractor(calls)
    researcher = RecordingResearcher(calls)

    result = run_paper_matching(
        workspace_root=workspace,
        run_id="run-1",
        topic_id="topic-a",
        artifact_findings_path=artifact_findings,
        keypoint_extractor=extractor,
        paper_researcher=researcher,
        output_top_n=10,
    )

    assert calls == ["keypoints:topic-a:True", "paper_research"]
    assert researcher.last_context_path == (
        workspace / "storage" / "vector-store" / "run-1" / "stores" / "papers"
    )
    assert "artifacts discuss retrieval augmentation" in researcher.last_query
    matched = json.loads(result.matched_papers_path.read_text())
    ranked = json.loads(result.ranked_papers_path.read_text())  # type: ignore[union-attr]
    assert matched["matching_basis"] == "artifact_keypoints_to_paper_store_gpt_researcher"
    assert matched["papers"][0]["title"] == "Retrieval Augmentation for Oral Presentations"
    assert ranked["papers"][0]["title"] == "Retrieval Augmentation for Oral Presentations"


def test_run_paper_matching_preserves_async_researcher_runtime_error(tmp_path):
    from conference_pipeline.paper_matching import run_paper_matching

    workspace = tmp_path / "conf"
    artifact_findings = workspace / "storage" / "research" / "run-1" / "findings" / "topic-a.md"
    artifact_findings.parent.mkdir(parents=True)
    artifact_findings.write_text(
        "# Topic A\n\nartifacts discuss retrieval augmentation in oral sessions.\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="local_synthesis_failed"):
        run_paper_matching(
            workspace_root=workspace,
            run_id="run-1",
            topic_id="topic-a",
            artifact_findings_path=artifact_findings,
            keypoint_extractor=RecordingExtractor([]),
            paper_researcher=FailingAsyncResearcher(),
            output_top_n=10,
        )


def test_run_paper_matching_researches_artifacts_before_paper_store(tmp_path):
    from conference_pipeline.paper_matching import (
        PaperMatchingConfig,
        run_paper_matching,
    )

    workspace = tmp_path / "conf"
    calls: list[str] = []
    artifact_runner = RecordingRunner(
        "artifact_research",
        calls,
        "# Findings\n\nartifacts discuss retrieval augmentation and evaluation.",
    )
    extractor = RecordingExtractor(calls)
    paper_runner = RecordingRunner(
        "paper_research",
        calls,
        json.dumps({"papers": [{"title": "Artifact Grounded Retrieval"}]}),
    )

    result = run_paper_matching(
        workspace_root=workspace,
        run_id="run-1",
        topic_id="topic-a",
        config=PaperMatchingConfig(
            keypoint_provider="fake",
            keypoint_model="frontier-test",
            research_provider="fake",
            research_model="research-test",
        ),
        artifact_research_runner=artifact_runner,
        keypoint_extractor=extractor,
        paper_research_runner=paper_runner,
    )

    assert calls == [
        "artifact_research:topic-a",
        "keypoints:topic-a:True",
        "paper_research:topic-a",
    ]
    assert "artifacts discuss retrieval augmentation" in paper_runner.queries[0]
    matched = json.loads(result.matched_papers_path.read_text())
    keypoints = json.loads(result.keypoints_path.read_text())
    assert keypoints["keypoint_model"] == "frontier-test"
    assert matched["artifact_research_store"] == "artifacts"
    assert matched["paper_research_store"] == "papers"
    assert matched["papers"][0]["keypoint_trace"]
    assert matched["papers"][0]["title"] == "Artifact Grounded Retrieval"


def test_run_paper_matching_uses_topic_fallback_when_artifact_report_has_no_evidence(
    tmp_path,
):
    from conference_pipeline.paper_matching import (
        PaperMatchingConfig,
        PaperMatchingError,
        run_paper_matching,
    )

    class EmptyExtractor:
        def extract_keypoints(self, *, topic_id: str, artifact_findings: str) -> list[str]:  # noqa: ARG002
            raise PaperMatchingError("empty_keypoints")

    workspace = tmp_path / "conf"
    artifact_findings = workspace / "storage" / "research" / "run-1" / "findings" / "topic.md"
    artifact_findings.parent.mkdir(parents=True)
    artifact_findings.write_text(
        "There are no presentations or transcripts that discuss this topic. "
        "The topic is not present in the provided artifacts.",
        encoding="utf-8",
    )
    calls: list[str] = []
    paper_runner = RecordingRunner(
        "paper_research",
        calls,
        json.dumps({"papers": [{"title": "Spatial VLM Paper"}]}),
    )

    result = run_paper_matching(
        workspace_root=workspace,
        run_id="run-1",
        topic_id="06_multimodal_spatial_vlm",
        config=PaperMatchingConfig(
            keypoint_provider="openai",
            keypoint_model="gpt-5.5",
            research_provider="fake",
            research_model="fixture",
        ),
        artifact_findings_path=artifact_findings,
        keypoint_extractor=EmptyExtractor(),
        paper_research_runner=paper_runner,
    )

    keypoints = json.loads(result.keypoints_path.read_text())
    matched = json.loads(result.matched_papers_path.read_text())
    assert keypoints["keypoint_provider"] == "openai+topic_fallback"
    assert keypoints["keypoint_fallback_reason"] == "artifact_store_reported_no_topic_evidence"
    assert "multimodal spatial vlm methods" in paper_runner.queries[0]
    assert matched["papers"][0]["keypoint_trace"]


def test_store_backed_runner_uses_retriever_and_fails_closed_on_empty(tmp_path):
    from conference_pipeline.paper_matching import (
        PaperMatchingError,
        StoreBackedResearchRunner,
    )

    class Retriever:
        def __init__(self, chunks):
            self.chunks = chunks
            self.calls = []

        def retrieve(self, *, topic_id: str, query: str, k: int):
            self.calls.append((topic_id, query, k))
            return self.chunks

    class Researcher:
        def run(self, query: str, *, context_path: Path | None = None) -> str:
            assert "Use only the papers vector store evidence below" in query
            assert "chunk_id=chunk-1" in query
            assert context_path == tmp_path
            return '{"papers": [{"title": "Store Backed Paper"}]}'

    retriever = Retriever(
        [
            SimpleNamespace(
                chunk_id="chunk-1",
                source_artifact_path="storage/papers/paper.pdf",
                text="Paper Title: Store Backed Paper\nSection: Findings",
            )
        ]
    )
    runner = StoreBackedResearchRunner(
        workspace_root=tmp_path,
        run_id="run-1",
        store_kind="papers",
        retriever=retriever,
        researcher=Researcher(),
        k=5,
    )

    output = runner.research(
        topic_id="topic-a",
        query="artifact-derived keypoints",
        output_path=tmp_path / "paper_gpt_researcher_findings.md",
    )

    assert retriever.calls == [("topic-a", "artifact-derived keypoints", 5)]
    assert "Store Backed Paper" in output
    assert (tmp_path / "paper_gpt_researcher_findings.md").exists()

    empty = StoreBackedResearchRunner(
        workspace_root=tmp_path,
        run_id="run-1",
        store_kind="papers",
        retriever=Retriever([]),
        researcher=Researcher(),
    )
    with pytest.raises(PaperMatchingError, match="papers_store_retrieval_empty"):
        empty.research(
            topic_id="topic-a",
            query="artifact-derived keypoints",
            output_path=tmp_path / "empty.md",
        )

    empty_output = StoreBackedResearchRunner(
        workspace_root=tmp_path,
        run_id="run-1",
        store_kind="papers",
        retriever=retriever,
        researcher=SimpleNamespace(run=lambda *_args, **_kwargs: "\n\n"),
    )
    with pytest.raises(PaperMatchingError, match="papers_store_research_empty"):
        empty_output.research(
            topic_id="topic-a",
            query="artifact-derived keypoints",
            output_path=tmp_path / "empty-output.md",
        )


def test_build_store_research_runner_uses_whole_store_semantic_retrieval(
    tmp_path, monkeypatch
):
    import conference_pipeline.research.researcher as researcher_module
    from conference_pipeline.paper_matching import (
        PaperMatchingConfig,
        build_store_research_runner,
    )

    calls: list[tuple[str, str, int]] = []
    prompts: list[str] = []

    class Retriever:
        def __init__(self, *_args, **_kwargs):
            pass

        def retrieve(self, *, topic_id: str, query: str, k: int):
            calls.append((topic_id, query, k))
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    source_artifact_path="storage/artifacts/session.json",
                    text="Artifact evidence about latent reasoning tradeoffs.",
                )
            ]

    class Researcher:
        def run(self, query: str, *, context_path: Path | None = None) -> str:  # noqa: ARG002
            prompts.append(query)
            return "Research report"

    monkeypatch.setattr(researcher_module, "ChromaVectorStoreRetriever", Retriever)
    monkeypatch.setattr(researcher_module, "build_researcher", lambda _config: Researcher())

    runner = build_store_research_runner(
        workspace_root=tmp_path,
        run_id="run-1",
        store_kind="artifacts",
        config=PaperMatchingConfig(research_provider="fake", research_model="fixture"),
    )

    output = runner.research(
        topic_id="01_reasoning_post_training",
        query="latent reasoning",
        output_path=tmp_path / "artifact_gpt_researcher_findings.md",
    )

    assert output == "Research report"
    assert calls == [("", "latent reasoning", 12)]
    assert "Topic: 01_reasoning_post_training" in prompts[0]


def test_build_store_research_runner_can_opt_into_topic_filtered_retrieval(
    tmp_path, monkeypatch
):
    import conference_pipeline.research.researcher as researcher_module
    from conference_pipeline.paper_matching import (
        PaperMatchingConfig,
        build_store_research_runner,
    )

    calls: list[tuple[str, str, int]] = []

    class Retriever:
        def __init__(self, *_args, **_kwargs):
            pass

        def retrieve(self, *, topic_id: str, query: str, k: int):
            calls.append((topic_id, query, k))
            return [
                SimpleNamespace(
                    chunk_id="chunk-1",
                    source_artifact_path="storage/papers/paper.pdf",
                    text="Paper evidence.",
                )
            ]

    class Researcher:
        def run(self, query: str, *, context_path: Path | None = None) -> str:  # noqa: ARG002
            return "Research report"

    monkeypatch.setattr(researcher_module, "ChromaVectorStoreRetriever", Retriever)
    monkeypatch.setattr(researcher_module, "build_researcher", lambda _config: Researcher())

    runner = build_store_research_runner(
        workspace_root=tmp_path,
        run_id="run-1",
        store_kind="papers",
        config=PaperMatchingConfig(research_provider="fake", research_model="fixture"),
        filter_by_topic=True,
    )

    runner.research(
        topic_id="topic-a",
        query="artifact-derived keypoints",
        output_path=tmp_path / "paper_gpt_researcher_findings.md",
    )

    assert calls == [("topic-a", "artifact-derived keypoints", 12)]


def test_load_paper_matching_config_reads_workflow_defaults(tmp_path):
    from conference_pipeline.paper_matching import load_paper_matching_config

    workspace = tmp_path / "conf"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workflow:
  paper_matching:
    keypoint_provider: openai
    keypoint_model: gpt-5.5
    artifact_research_store: artifacts
    paper_research_store: papers
    research_provider: fake
    research_model: fixture-model
    output_top_n: 7
""",
        encoding="utf-8",
    )

    config = load_paper_matching_config(
        workspace,
        overrides={"output_top_n": 3, "keypoint_model": "override-frontier"},
    )

    assert config.keypoint_provider == "openai"
    assert config.keypoint_model == "override-frontier"
    assert config.research_provider == "fake"
    assert config.research_model == "fixture-model"
    assert config.output_top_n == 3


def test_load_paper_matching_config_pairs_registry_default_provider_and_model(tmp_path):
    from conference_pipeline.model_governance import default_model, default_provider
    from conference_pipeline.paper_matching import load_paper_matching_config

    workspace = tmp_path / "conf"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workflow:
  paper_matching:
    enabled: true
""",
        encoding="utf-8",
    )

    config = load_paper_matching_config(workspace)

    assert config.keypoint_provider == default_provider("research.paper_matching_keypoints")
    assert config.keypoint_model == default_model("research.paper_matching_keypoints")
    assert config.research_provider == default_provider("research.gpt_researcher_smart")
    assert config.research_model == default_model("research.gpt_researcher_smart")


def test_load_paper_matching_config_reads_local_env_overrides(tmp_path, monkeypatch):
    from conference_pipeline.paper_matching import load_paper_matching_config

    workspace = tmp_path / "conf"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workflow:
  paper_matching:
    keypoint_provider: openai
    keypoint_model: gpt-5.5
    research_provider: openai
    research_model: gpt-5-mini
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAPER_MATCHING_KEYPOINT_PROVIDER", "local")
    monkeypatch.setenv("PAPER_MATCHING_KEYPOINT_MODEL", "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit")
    monkeypatch.setenv("PAPER_MATCHING_RESEARCH_PROVIDER", "local")
    monkeypatch.setenv("PAPER_MATCHING_RESEARCH_MODEL", "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit")

    config = load_paper_matching_config(workspace)

    assert config.keypoint_provider == "local"
    assert config.keypoint_model == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
    assert config.research_provider == "local"
    assert config.research_model == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"


def test_local_keypoint_extractor_uses_openai_compatible_client(tmp_path):
    from conference_pipeline.paper_matching import LocalKeypointExtractor
    from conference_pipeline.provider_receipts import provider_receipt_context

    completions = _FakeChatCompletions(
        _FakeChatResponse(
            content='{"keypoints": ["efficient attention", "state space models"]}'
        )
    )
    client = _FakeOpenAIClient(completions)
    extractor = LocalKeypointExtractor(
        model="mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
        base_url="http://127.0.0.1:8080/v1",
        max_keypoints=2,
        _client=client,
    )

    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-local",
        stage_id="paper-matching",
    ):
        keypoints = extractor.extract_keypoints(
            topic_id="02_model_architectures",
            artifact_findings="Artifact findings emphasize efficient attention.",
        )

    assert keypoints == ["efficient attention", "state space models"]
    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
    assert completions.calls[0]["messages"][0]["role"] == "system"


def test_verify_paper_matching_caps_uses_reusable_workflow_contract(tmp_path):
    from conference_pipeline.paper_matching import verify_paper_matching_caps

    workspace = tmp_path / "conf"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workflow:
  paper_matching:
    output_top_n: 30
  deck:
    max_papers: 30
""",
        encoding="utf-8",
    )

    verification = verify_paper_matching_caps(workspace)

    assert verification["status"] == "pass"
    assert verification["observed_caps"] == {
        "workflow.paper_matching.output_top_n": 30,
        "paper_matching.output_top_n": 30,
        "analysis.matched_paper_top_k": 30,
        "workflow.deck.max_papers": 30,
    }


def test_verify_paper_matching_caps_blocks_hidden_downstream_caps(tmp_path):
    from conference_pipeline.paper_matching import verify_paper_matching_caps

    workspace = tmp_path / "conf"
    workspace.mkdir()
    (workspace / "config.yaml").write_text(
        """
workflow:
  paper_matching:
    output_top_n: 20
  deck:
    max_papers: 20
""",
        encoding="utf-8",
    )

    verification = verify_paper_matching_caps(
        workspace,
        paper_matching_output_top_n=8,
        analysis_matched_paper_top_k=5,
        deck_max_papers=10,
    )

    assert verification["status"] == "blocked"
    assert {
        blocker["cap_name"] for blocker in verification["blockers"]
    } == {
        "paper_matching.output_top_n",
        "analysis.matched_paper_top_k",
        "workflow.deck.max_papers",
    }


def test_paper_matching_workflow_stage_dispatches_with_run_id(tmp_path, monkeypatch):
    from conference_pipeline import workflow
    from conference_pipeline.workflow_runs import stage_receipt_path, workflow_run_dir
    import conference_pipeline.paper_matching as paper_matching

    workspace = tmp_path / "iclr2025"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'storage:\n  manifest_path: "storage/storage_manifest.json"\n',
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        '{"artifact_families": []}\n',
        encoding="utf-8",
    )
    resolved = workflow.resolve_workspace(workspace)
    stage = workflow.get_workflow_stage("paper-matching")
    calls: list[tuple[Path, str, str | None]] = []

    def fake_stage(*, workspace_root: Path, run_id: str, topic_id: str | None = None):
        calls.append((workspace_root, run_id, topic_id))
        return {"status": "succeeded", "artifacts": [{"kind": "matched_papers"}]}

    monkeypatch.setattr(paper_matching, "run_paper_matching_stage", fake_stage)

    assert workflow.is_paper_matching_stage_id("paper-matching")
    assert stage.dispatch_kind == "python_stage"
    assert stage_receipt_path(workflow_run_dir(resolved, "run-1"), stage).name == (
        "paper-matching-receipt.json"
    )
    result = workflow.run_stage(
        resolved,
        stage,
        topic="01_demo_topic",
        run_id="run-1",
    )

    assert result == {"status": "succeeded", "artifacts": [{"kind": "matched_papers"}]}
    assert calls == [(resolved.root, "run-1", "01_demo_topic")]


def test_paper_matching_workflow_dry_run_receipt_plans_canonical_outputs(tmp_path):
    from conference_pipeline import workflow
    from conference_pipeline.workflow_runs import (
        WorkflowExecutionMode,
        WorkflowRunOptions,
        run_workflow_stages,
        workflow_run_dir,
    )

    workspace = tmp_path / "iclr2025"
    (workspace / "storage").mkdir(parents=True)
    (workspace / "config.yaml").write_text(
        'storage:\n  manifest_path: "storage/storage_manifest.json"\n',
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        '{"artifact_families": []}\n',
        encoding="utf-8",
    )
    resolved = workflow.resolve_workspace(workspace)
    stage = workflow.get_workflow_stage("paper-matching")

    run_workflow_stages(
        resolved,
        [stage],
        WorkflowRunOptions(
            run_id="run-1",
            mode=WorkflowExecutionMode.DRY_RUN,
            topic="01_demo_topic",
        ),
    )

    receipt_path = workflow_run_dir(resolved, "run-1") / "paper-matching-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    planned_paths = {artifact["path"] for artifact in receipt["artifacts"]}
    assert receipt["status"] == "planned"
    assert "storage/paper-matching/run-1/01_demo_topic/matched_papers.json" in planned_paths
    assert "storage/paper-matching/run-1/01_demo_topic/keypoints.json" in planned_paths


def test_rank_papers_legacy_warning(capsys):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "rank_papers.py"
    spec = importlib.util.spec_from_file_location("legacy_rank_papers", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with pytest.warns(DeprecationWarning):
        module.warn_deprecated()
    captured = capsys.readouterr()
    assert "Deprecated:" in captured.err
    assert "match_papers_from_artifact_keypoints.py" in captured.err


class _FakeKeypointOpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kwargs: Any) -> Any:  # noqa: ARG004
        return SimpleNamespace(
            id="resp-1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"keypoints": ["artifacts discuss retrieval", "evaluation"]}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=80, total_tokens=1280),
        )


def _clear_receipt_env(monkeypatch) -> None:
    for name in (
        "PROVIDER_RECEIPT_WORKSPACE",
        "CONFERENCE_WORKSPACE",
        "WORKSPACE_ROOT",
        "PROVIDER_RECEIPT_RUN_ID",
        "RUN_ID",
        "PROVIDER_RECEIPT_STAGE_ID",
        "PROVIDER_RECEIPT_CALLER",
        "PROVIDER_RECEIPT_TASK_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def test_keypoint_provider_receipts_land_in_the_conference_workspace(tmp_path, monkeypatch):
    import openai

    from conference_pipeline.model_governance import default_model
    from conference_pipeline.paper_matching import run_paper_matching
    from conference_pipeline.paper_matching.matcher import (
        PAPER_MATCHING_KEYPOINT_TASK_ID,
        PAPER_MATCHING_STAGE_ID,
        OpenAIKeypointExtractor,
    )

    _clear_receipt_env(monkeypatch)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.chdir(repo_root)  # the old fallback root: Path.cwd()
    monkeypatch.setattr(openai, "OpenAI", _FakeKeypointOpenAI)
    workspace = repo_root / "conferences" / "conf"
    findings = workspace / "storage" / "research" / "run-1" / "findings" / "topic-a.md"
    findings.parent.mkdir(parents=True)
    findings.write_text("# Topic A\n\nartifacts discuss retrieval.\n", encoding="utf-8")

    run_paper_matching(
        workspace_root=workspace,
        run_id="run-1",
        topic_id="topic-a",
        artifact_findings_path=findings,
        keypoint_extractor=OpenAIKeypointExtractor(
            model=default_model("research.paper_matching_keypoints"), api_key="test-key"
        ),
        paper_researcher=RecordingResearcher([]),
        output_top_n=10,
    )

    assert not (repo_root / "storage").exists()
    receipts = sorted(workspace.glob("storage/**/provider-calls/**/*.json"))
    assert len(receipts) == 1
    assert receipts[0].is_relative_to(
        workspace / "storage" / "workflow-runs" / "run-1" / "provider-calls" / PAPER_MATCHING_STAGE_ID
    )
    receipt = json.loads(receipts[0].read_text())
    assert receipt["task_id"] == PAPER_MATCHING_KEYPOINT_TASK_ID
    assert receipt["stage_id"] == PAPER_MATCHING_STAGE_ID
    assert receipt["run_id"] == "run-1"
    assert Path(receipt["workspace"]) == workspace.resolve()
    # The scope is popped afterwards: nothing leaks into later calls.
    import os

    assert "PROVIDER_RECEIPT_WORKSPACE" not in os.environ


def test_paper_matching_receipt_scope_keeps_an_enclosing_runner_identity(tmp_path, monkeypatch):
    from conference_pipeline.paper_matching.matcher import paper_matching_receipt_scope
    from conference_pipeline.provider_receipts import (
        current_provider_receipt_context,
        provider_receipt_context,
    )

    _clear_receipt_env(monkeypatch)
    workspace = tmp_path / "conf"
    with provider_receipt_context(run_id="runner-run", stage_id="runner-stage", caller="runner"):
        with paper_matching_receipt_scope(workspace, "matching-run", task_id="task-x"):
            context = current_provider_receipt_context()
    assert context.workspace_root == workspace.resolve()
    assert context.run_id == "runner-run"
    assert context.stage_id == "runner-stage"
    assert context.caller == "runner"
    assert context.task_id == "task-x"
