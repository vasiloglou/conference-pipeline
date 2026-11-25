from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from conference_pipeline.research.deep_dive_lanes import (
    DEFAULT_QWEN36_MODEL,
    DEFAULT_SELECTION_MODEL,
    DEFAULT_SELECTION_PROVIDER,
    run_deep_dive_question_selection,
    run_dual_deep_dive_research_lanes,
)
from conference_pipeline.research.researcher import RetrievedChunk



def _alias_model(alias: str) -> str:
    """The catalog's current model for an alias, so model upgrades move these tests with them."""
    from conference_pipeline.model_governance.registry import load_model_catalog

    return load_model_catalog()[alias]["model"]

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _summary_payload(topic_id: str, category: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "owner_id": "analysis.exhaustive_topic_summarization",
        "run_id": "test-run",
        "topic_id": topic_id,
        "title": "Reasoning And Agents",
        "category": category,
        "summary": "Reasoning systems are shifting toward compact latent search.",
        "key_points": [
            "Latent reasoning reduces visible chain-of-thought tokens.",
            "Agentic systems need evidence-grounded planning loops.",
        ],
        "tensions_or_contrasts": [
            "Discrete CoT is interpretable but token hungry; latent reasoning is efficient but opaque.",
        ],
        "open_questions": ["How should evaluation measure hidden reasoning?"],
        "notable_evidence": [
            {"claim": "continuous reasoning appears in several methods", "unit_ids": ["u1"]}
        ],
        "analysis_feed": {"what_to_emphasize": ["efficiency vs inspectability"]},
        "input_tokens": 1000,
        "unit_count": 5,
    }


def _workspace_with_summaries(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    summaries = []
    for category in ("transcripts", "papers"):
        rel = f"analysis/exhaustive_summaries/01_reasoning__{category}.json"
        _write_json(workspace / rel, _summary_payload("01_reasoning", category))
        summaries.append(rel)
    _write_json(
        workspace / "analysis/exhaustive_summaries/index.json",
        {
            "schema_version": 1,
            "owner_id": "analysis.exhaustive_topic_summarization",
            "run_id": "test-run",
            "provider": "openai",
            "model": "gpt-5.5",
            "summary_count": 2,
            "summaries": summaries,
        },
    )
    return workspace


class _FakeOpenAICompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        payload = {
            "deep_dives": [
                {
                    "question": "What is the shift from discrete CoT to latent reasoning buying?",
                    "rationale": "It connects efficiency, interpretability, and evaluation.",
                    "keywords": ["latent reasoning", "continuous reasoning", "chain-of-thought"],
                    "source_categories": ["transcripts", "papers"],
                    "priority_score": 0.96,
                    "slide_usefulness": "Core trend slide.",
                    "evidence_handles": ["u1"],
                },
                {
                    "question": "How should agent loops expose evidence without wasting tokens?",
                    "rationale": "It links agent design to compression.",
                    "keywords": ["agentic planning", "evidence", "token efficiency"],
                    "source_categories": ["transcripts", "papers"],
                    "priority_score": 0.82,
                },
            ]
        }
        return SimpleNamespace(
            id="chatcmpl-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload, sort_keys=True))
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.completions = _FakeOpenAICompletions()
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        payload = {
            "deep_dives": [
                {
                    "question": "What does latent reasoning buy in practice?",
                    "rationale": "It tests the claimed efficiency tradeoff.",
                    "keywords": ["latent reasoning", "efficiency"],
                    "source_categories": ["transcripts", "papers"],
                    "priority_score": 0.9,
                }
            ]
        }
        return SimpleNamespace(
            id="msg-test",
            content=[SimpleNamespace(text=json.dumps(payload, sort_keys=True))],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def test_live_deep_dive_question_selection_supports_openai_candidate(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_summaries(tmp_path)
    client = _FakeOpenAIClient()

    receipt = run_deep_dive_question_selection(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        provider="openai",
        model=_alias_model("luna"),
        min_questions_per_topic=1,
        max_questions_per_topic=2,
        provider_cost_approved=True,
        client=client,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["lineage"]["binding_source"]["substep_id"] == "deep-dive-question-selection"
    assert receipt["provider"] == "openai"
    assert receipt["model"] == _alias_model("luna")
    assert receipt["question_count"] == 2
    selection_path = workspace / receipt["selection_path"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert selection["questions"][0]["deep_dive_id"].startswith("01_reasoning__dd01")
    assert selection["questions"][0]["deep_dive_id"] != "REDACTED"
    assert len(selection["questions"][0]["deep_dive_id"]) <= 62
    assert client.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert client.completions.calls[0]["max_completion_tokens"] == 8000


def test_deep_dive_question_selection_defaults_to_opus() -> None:
    assert DEFAULT_SELECTION_PROVIDER == "anthropic"
    assert DEFAULT_SELECTION_MODEL == _alias_model("opus")


def test_live_deep_dive_question_selection_calls_anthropic_opus(
    tmp_path: Path,
) -> None:
    workspace = _workspace_with_summaries(tmp_path)
    client = _FakeAnthropicClient()

    receipt = run_deep_dive_question_selection(
        workspace=workspace,
        run_id="test-run",
        mode="live",
        min_questions_per_topic=1,
        max_questions_per_topic=1,
        provider_cost_approved=True,
        client=client,
    )

    assert receipt["status"] == "succeeded"
    assert receipt["provider"] == "anthropic"
    assert receipt["model"] == _alias_model("opus")
    call = client.messages.calls[0]
    assert call["model"] == _alias_model("opus")
    assert call["max_tokens"] == 8000
    assert "temperature" not in call


class _FakeRetriever:
    def __init__(self, store: str) -> None:
        self.store = store
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, *, topic_id: str, query: str, k: int) -> list[RetrievedChunk]:
        self.calls.append((topic_id, k))
        return [
            RetrievedChunk(
                chunk_id=f"{self.store}-chunk",
                extraction_record_id="record",
                source_artifact_id="artifact",
                source_artifact_path=f"storage/{self.store}/source.txt",
                text=f"{query} latent reasoning chain-of-thought evidence",
                score=0.9,
                topic_ids=(),
                chunk_strategy="test",
            )
        ]


class _FakeResearcher:
    provider = "local"
    model = DEFAULT_QWEN36_MODEL

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.provider_receipt_paths: list[str] = []

    async def run(self, query: str, *, context_path: Path | None = None) -> str:
        self.calls.append(query)
        self.provider_receipt_paths.append(f"receipt-{len(self.calls)}.json")
        return "# Report\n\n## Answer\n\nLatent reasoning matters.\n"

    def usage(self) -> dict[str, object]:
        return {
            "provider_receipt_paths": list(self.provider_receipt_paths),
            "estimated_cost_usd": 0.0,
        }


def test_dual_deep_dive_lanes_run_global_and_heuristic_without_topic_filter(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_id = "test-run"
    selection_path = workspace / "storage/analysis/deep-dive-topics/test-run/selected_deep_dives.json"
    _write_json(
        selection_path,
        {
            "schema_version": 1,
            "owner_id": "analysis.deep_dive_question_selection",
            "run_id": run_id,
            "questions": [
                {
                    "deep_dive_id": "01_reasoning__dd01_latent_reasoning",
                    "topic_id": "01_reasoning",
                    "topic_title": "Reasoning",
                    "question": "Why does latent reasoning improve token efficiency?",
                    "rationale": "Core trend.",
                    "keywords": ["latent reasoning", "chain-of-thought", "token efficiency"],
                }
            ],
        },
    )
    chunk = {
        "chunk_id": "chunk-1",
        "artifact_id": "artifact-1",
        "source_path": "storage/source.txt",
        "text": "latent reasoning improves token efficiency over chain-of-thought",
        "source_context": {"title": "Talk", "source_url": "https://example.test"},
        "locator": {},
        "content_kind": "transcript",
        "conference_scope": "main_conference",
    }
    for store in ("papers", "artifacts"):
        _write_jsonl(
            workspace / f"storage/vector-store/{run_id}/stores/{store}/chunks.jsonl",
            [chunk | {"chunk_id": f"{store}-chunk"}],
        )

    papers = _FakeRetriever("papers")
    artifacts = _FakeRetriever("artifacts")
    researcher = _FakeResearcher()

    receipt = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=researcher,
        engine="inhouse",
    )

    assert receipt["status"] == "succeeded"
    assert receipt["lineage"]["binding_source"]["substep_id"] == "dual-deep-dive-research"
    assert {entry["stage_id"] for entry in receipt["lineage"]["upstream_receipts"].values()} == {
        "corpus-inventory-index",
        "deep-dive-question-selection",
    }
    assert receipt["local_researcher_model"] == DEFAULT_QWEN36_MODEL
    assert receipt["lane_count"] == 2
    assert papers.calls[0][0] == ""
    assert artifacts.calls[0][0] == ""
    assert len(researcher.calls) == 2
    outputs = {item["lane"]: item for item in receipt["outputs"]}
    assert outputs["global"]["evidence_count"] == 2
    assert outputs["heuristic"]["evidence_count"] == 2
    assert (workspace / outputs["global"]["output_markdown"]).exists()
    assert (workspace / outputs["heuristic"]["evidence_json"]).exists()

    resumed = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=researcher,
        engine="inhouse",
        resume=True,
    )

    assert resumed["status"] == "succeeded"
    assert resumed["lane_count"] == 2
    assert len(researcher.calls) == 2

    # The corpus index is rebuilt (Stage 6 output_digest moves): cached lanes
    # were computed against the old index, so resume re-runs both lanes.
    _write_json(
        workspace / f"storage/workflow-runs/{run_id}/stage-06-corpus-inventory-index.json",
        {"stage_id": "corpus-inventory-index", "stage_order": 6, "status": "succeeded", "output_digest": "9" * 64},
    )
    rebuilt = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=researcher,
        engine="inhouse",
        resume=True,
    )
    assert rebuilt["status"] == "succeeded"
    assert len(researcher.calls) == 4, "lanes over a rebuilt corpus index must re-run"

    # A rebuild interrupted mid-lane (the researcher fails) must not leave the
    # old report looking current: the next resume re-runs the lanes.
    _write_json(
        workspace / f"storage/workflow-runs/{run_id}/stage-06-corpus-inventory-index.json",
        {"stage_id": "corpus-inventory-index", "stage_order": 6, "status": "succeeded", "output_digest": "8" * 64},
    )

    class _FailingResearcher(_FakeResearcher):
        async def run(self, query: str, *, context_path: Path | None = None) -> str:
            raise RuntimeError("interrupted")

    try:
        run_dual_deep_dive_research_lanes(
            workspace=workspace,
            run_id=run_id,
            mode="live",
            selection_path=selection_path,
            retrievers={"papers": papers, "artifacts": artifacts},
            researcher=_FailingResearcher(),
            engine="inhouse",
            resume=True,
        )
    except Exception:  # noqa: BLE001 - the interruption itself is not under test
        pass
    # The crashed re-run unbound the previous receipt before rewriting outputs,
    # so it no longer certifies them.
    receipt_file = workspace / f"storage/workflow-runs/{run_id}/07C-dual-deep-dive-research-lanes.json"
    crashed = json.loads(receipt_file.read_text(encoding="utf-8"))
    assert "substep_outputs" not in crashed and crashed["lineage_unbound_reason"] == "in_progress"
    after_interrupt = _FakeResearcher()
    recovered = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=after_interrupt,
        resume=True,
        engine="inhouse",
    )
    assert recovered["status"] == "succeeded"
    assert len(after_interrupt.calls) == 2, "an interrupted rebuild must not be reused"

    # A question-limited run (max_questions below the selection) is not bound.
    limited = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=_FakeResearcher(),
        engine="inhouse",
        resume=True,
        max_questions=0,
    )
    assert "lineage" not in limited and limited["lineage_unbound_reason"] == "partial_scope"

    # The canonical 07B selection path is what the lineage binds; that is the
    # path this test used. A run over an alternate selection file is partial.
    alternate = workspace / "alternate_selection.json"
    alternate.write_text(selection_path.read_text(encoding="utf-8"), encoding="utf-8")
    other = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=alternate,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=_FakeResearcher(),
        engine="inhouse",
        resume=True,
    )
    assert other["lineage_unbound_reason"] == "partial_scope"
    canonical = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=_FakeResearcher(),
        engine="inhouse",
        resume=True,
    )
    assert canonical["lineage"]["binding_source"]["substep_id"] == "dual-deep-dive-research"


    # 07B dropped a question: its reports and evidence would still be read by
    # 07D's glob, so a complete run moves them out before stamping.
    stale_report = workspace / "analysis" / "deep_dive_topics" / "01_reasoning__dd09_dropped__global.md"
    stale_report.write_text("# old corpus report\n", encoding="utf-8")
    stale_evidence = workspace / f"storage/analysis/deep-dive-topics/{run_id}/01_reasoning__dd09_dropped"
    stale_evidence.mkdir(parents=True)
    (stale_evidence / "global_evidence.json").write_text("{}", encoding="utf-8")
    pruned = run_dual_deep_dive_research_lanes(
        workspace=workspace,
        run_id=run_id,
        mode="live",
        selection_path=selection_path,
        retrievers={"papers": papers, "artifacts": artifacts},
        researcher=_FakeResearcher(),
        engine="inhouse",
        resume=True,
    )
    assert not stale_report.exists() and not stale_evidence.exists()
    assert sorted(pruned["pruned_outputs"]) == [
        "analysis/deep_dive_topics/01_reasoning__dd09_dropped__global.md",
        f"storage/analysis/deep-dive-topics/{run_id}/01_reasoning__dd09_dropped",
    ]
    assert list((workspace / "storage" / "lineage-pruned").rglob("01_reasoning__dd09_dropped__global.md"))
    assert (workspace / "analysis" / "deep_dive_topics" / "01_reasoning__dd01_latent_reasoning__global.md").exists()


def test_dual_lanes_run_on_the_gpt_researcher_engine(tmp_path: Path) -> None:
    """Default engine: GPT-Researcher researches the global lane over the index
    and the heuristic lane within its prefiltered pack; evidence comes from its
    retriever's log and both lanes keep their files."""

    import asyncio as _asyncio

    from conference_pipeline.research.gptr_engine import GPTResearcherEngine
    from conference_pipeline.research.gptr_retriever import LocalHybridRetriever

    class _PackAwareGPTR:
        def __init__(self, *, query, config_path, settings):  # noqa: ANN001, ARG002
            self.query = query
            self.retrievers = [LocalHybridRetriever]
            self.urls: list[str] = []

        async def conduct_research(self):
            results = await _asyncio.to_thread(LocalHybridRetriever(self.query).search, max_results=2)
            self.urls = [r["href"] for r in results]

        async def write_report(self, custom_prompt=""):  # noqa: ANN001
            return "## Answer\n\nSupported by " + ", ".join(self.urls) + "\n\n" + custom_prompt.splitlines()[0]

        def get_source_urls(self):
            return self.urls

    workspace = tmp_path / "workspace"
    run_id = "test-run"
    selection_path = workspace / "storage/analysis/deep-dive-topics/test-run/selected_deep_dives.json"
    _write_json(selection_path, {
        "schema_version": 1, "owner_id": "analysis.deep_dive_question_selection", "run_id": run_id,
        "questions": [{
            "deep_dive_id": "01_reasoning__dd01_latent_reasoning", "topic_id": "01_reasoning",
            "topic_title": "Reasoning", "question": "Why does latent reasoning improve token efficiency?",
            "rationale": "Core trend.", "keywords": ["latent reasoning", "token efficiency"],
        }],
    })
    chunk = {
        "chunk_id": "chunk-1", "artifact_id": "artifact-1", "source_path": "storage/source.txt",
        "text": "latent reasoning improves token efficiency over chain-of-thought",
        "source_context": {"title": "Talk"}, "locator": {}, "content_kind": "transcript",
        "conference_scope": "main_conference",
    }
    for store in ("papers", "artifacts"):
        _write_jsonl(workspace / f"storage/vector-store/{run_id}/stores/{store}/chunks.jsonl", [chunk | {"chunk_id": f"{store}-chunk"}])
    retrievers = {"papers": _FakeRetriever("papers"), "artifacts": _FakeRetriever("artifacts")}
    engine = GPTResearcherEngine(
        workspace, run_id=run_id, retriever_factory=lambda store: retrievers[store],
        researcher_factory=lambda **kw: _PackAwareGPTR(**kw),
    )
    # The engine comes from the conference's recorded choice, not a default.
    (workspace / "config.yaml").write_text(
        "conference:\n  name: Test\nworkflow:\n  research_engine: gpt_researcher\n"
        "  research_engine_decision: operator test\n",
        encoding="utf-8",
    )
    receipt = run_dual_deep_dive_research_lanes(
        workspace=workspace, run_id=run_id, mode="live", selection_path=selection_path,
        retrievers=retrievers, researcher=engine,
    )
    assert receipt["status"] == "succeeded"
    assert receipt["research_engine"] == "gpt_researcher"
    assert receipt["research_engine_choice"] == {
        "engine": "gpt_researcher", "source": "config", "decision": "operator test",
    }
    assert receipt["egress"] == {"runs": 2, "blocked_destinations": []}
    root = workspace / "storage/analysis/deep-dive-topics" / run_id
    global_evidence = json.loads(next(root.rglob("global_evidence.json")).read_text(encoding="utf-8"))
    assert global_evidence["retrieval_engine"] == "gpt_researcher"
    assert {item["store"] for item in global_evidence["evidence"]} == {"papers", "artifacts"}
    heuristic = json.loads(next(root.rglob("heuristic_prefilter_evidence.json")).read_text(encoding="utf-8"))
    assert heuristic["retrieval_engine"] == "gpt_researcher"
    report = (workspace / "analysis/deep_dive_topics/01_reasoning__dd01_latent_reasoning__global.md").read_text(encoding="utf-8")
    assert "chunk://" in report and "## Answer" in report


def test_engine_research_surfaces_the_research_error() -> None:
    """A failing research run reports its own error, not a re-await failure."""
    import pytest

    from conference_pipeline.research import deep_dive_lanes

    class _Failing:
        async def run(self, query: str, custom_prompt: str = "") -> str:
            raise RuntimeError("retriever returned nothing")

    with pytest.raises(RuntimeError, match="retriever returned nothing"):
        deep_dive_lanes._engine_research(_Failing(), {"question": "q"}, instructions="i")


def test_evidence_numbers_resolve_to_passage_links() -> None:
    from conference_pipeline.research.deep_dive_lanes import resolve_evidence_numbers

    evidence = [
        {"store": "papers", "chunk_id": "paper_text:aaa:0"},
        {"store": "artifacts", "chunk_id": "speech_transcript:bbb:3"},
    ]
    report = (
        "One [EVIDENCE #1]. Two (Evidence #1, #2). Three [Evidence #2 and #1]. "
        "Bad [EVIDENCE #7]. Case evidence #2."
    )
    resolved, stats = resolve_evidence_numbers(report, evidence)
    assert "One (chunk://papers/paper_text:aaa:0)." in resolved
    assert "Two ((chunk://papers/paper_text:aaa:0; chunk://artifacts/speech_transcript:bbb:3))." in resolved
    assert "Three (chunk://artifacts/speech_transcript:bbb:3; chunk://papers/paper_text:aaa:0)." in resolved
    assert "Bad ([unresolved citation: Evidence #7])." in resolved
    assert "Case (chunk://artifacts/speech_transcript:bbb:3)." in resolved
    assert stats == {"resolved": 6, "unresolved": ["Evidence #7"]}
    assert "EVIDENCE #" not in resolved.upper().replace("UNRESOLVED CITATION: EVIDENCE #", "")


def test_inhouse_lane_reports_carry_passage_links_not_evidence_numbers(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "test-run"
    selection_path = workspace / "storage/analysis/deep-dive-topics/test-run/selected_deep_dives.json"
    _write_json(selection_path, {
        "schema_version": 1, "owner_id": "analysis.deep_dive_question_selection", "run_id": run_id,
        "questions": [{
            "deep_dive_id": "01_reasoning__dd01_latent_reasoning", "topic_id": "01_reasoning",
            "topic_title": "Reasoning", "question": "Why does latent reasoning improve token efficiency?",
            "rationale": "Core trend.", "keywords": ["latent reasoning", "token efficiency"],
        }],
    })
    chunk = {"chunk_id": "c", "artifact_id": "a", "source_path": "storage/source.txt",
             "text": "latent reasoning improves token efficiency", "source_context": {}, "locator": {},
             "content_kind": "transcript", "conference_scope": "main_conference"}
    for store in ("papers", "artifacts"):
        _write_jsonl(workspace / f"storage/vector-store/{run_id}/stores/{store}/chunks.jsonl",
                     [chunk | {"chunk_id": f"{store}-chunk"}])

    class _CitingResearcher(_FakeResearcher):
        async def run(self, query: str, *, context_path: Path | None = None) -> str:
            assert "[EVIDENCE #n]" in query, "the prompt asks for numbered citations"
            await super().run(query, context_path=context_path)
            return "## Answer\n\nLatent reasoning saves tokens [EVIDENCE #1, #2].\n"

    receipt = run_dual_deep_dive_research_lanes(
        workspace=workspace, run_id=run_id, mode="live", selection_path=selection_path,
        retrievers={"papers": _FakeRetriever("papers"), "artifacts": _FakeRetriever("artifacts")},
        researcher=_CitingResearcher(), engine="inhouse",
    )
    outputs = {item["lane"]: item for item in receipt["outputs"]}
    global_md = (workspace / outputs["global"]["output_markdown"]).read_text(encoding="utf-8")
    assert "(chunk://papers/papers-chunk; chunk://artifacts/artifacts-chunk)" in global_md
    assert "EVIDENCE #" not in global_md
    evidence = json.loads((workspace / outputs["global"]["evidence_json"]).read_text(encoding="utf-8"))
    assert evidence["citations"] == {"resolved": 2, "unresolved": []}
    assert receipt["citations"]["resolved"] == 4 and receipt["citations"]["unresolved"] == []


def test_no_engine_chosen_stops_and_asks_the_operator(tmp_path: Path) -> None:
    import pytest

    from conference_pipeline.research import deep_dive_lanes
    from conference_pipeline.research.engine_choice import ResearchEngineNotChosenError, resolve_research_engine

    workspace = tmp_path / "newconf2027"
    workspace.mkdir()
    (workspace / "config.yaml").write_text("conference:\n  name: New Conf\n", encoding="utf-8")
    with pytest.raises(ResearchEngineNotChosenError, match="Ask the operator"):
        resolve_research_engine(workspace)
    with pytest.raises(ResearchEngineNotChosenError):
        deep_dive_lanes.run_dual_deep_dive_research_lanes(workspace=workspace, run_id="r", mode="dry-run")
    # The command line refuses cleanly with the question, before any work.
    assert deep_dive_lanes.run_main(["--workspace", str(workspace), "--run-id", "r"]) == 2
    # Either engine can be chosen explicitly, per run or in the config.
    assert resolve_research_engine(workspace, "inhouse").source == "cli"
    (workspace / "config.yaml").write_text(
        "conference:\n  name: New Conf\nworkflow:\n  research_engine: inhouse\n", encoding="utf-8"
    )
    choice = resolve_research_engine(workspace)
    assert (choice.engine, choice.source) == ("inhouse", "config")
    (workspace / "config.yaml").write_text(
        "conference:\n  name: New Conf\nworkflow:\n  research_engine: web_search\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be one of"):
        resolve_research_engine(workspace)
