"""Reusable narrative analysis/provenance contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _source_pack() -> dict[str, object]:
    return {
        "topic_id": "02_model_architectures",
        "title": "Model Architectures",
        "research_id": "02_model_architectures#abcdef",
        "research_path": "storage/research/run/findings/02_model_architectures.md",
        "finding_text": "Architecture search, efficient attention, and evaluation pressure define the story.",
        "chunks": [
            {
                "chunk_id": f"chunk-{index}",
                "extraction_record_id": f"extract-{index}",
                "source_artifact_path": f"storage/transcription-ocr/media/{index}.json",
                "quote": f"Architecture quote {index}.",
            }
            for index in range(4)
        ],
        "matched_papers": [
            {
                "paper_id": f"paper-{index}",
                "title": f"Architecture Paper {index}",
                "url": f"https://openreview.net/forum?id=paper{index}",
                "evidence_snippet": "Aligned evidence.",
            }
            for index in range(4)
        ],
        "extraction_record_ids": [f"extract-{index}" for index in range(4)],
        "paper_matching_texts": {},
    }


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
        prompt_tokens: int = 17,
        completion_tokens: int = 23,
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


class _SequenceNarrativeClient:
    model = "test-narrative-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.request_count = 0
        self.user_prompts: list[str] = []

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SimpleNamespace:
        del kind, system_prompt
        self.user_prompts.append(user_prompt)
        self.request_count += 1
        if not self._responses:
            raise AssertionError("unexpected narrative synthesis call")
        return SimpleNamespace(text=self._responses.pop(0))


def test_render_narrative_topic_report_is_deep_analysis_not_slide_brief() -> None:
    from conference_pipeline.analysis.narrative import (
        render_narrative_topic_report,
        validate_narrative_topic_report,
    )

    report = render_narrative_topic_report(_source_pack())
    validation = validate_narrative_topic_report(report)

    assert validation["status"] == "pass"
    assert "## Executive Summary" in report
    assert "## Key Findings" in report
    assert "## Detailed Analysis" in report
    assert validation["detailed_subsection_count"] >= 3
    assert "## Slide Brief" in report
    assert "## References And Sources" in report
    assert "## Cross-References" in report
    assert "## Recommendations" in report
    assert "## Questions For Further Investigation" in report
    assert "## Source Basis" not in report
    assert "Verbatim transcript/OCR evidence" not in report


def test_validate_narrative_topic_report_rejects_old_slide_ready_shape() -> None:
    from conference_pipeline.analysis.narrative import (
        validate_narrative_topic_report,
    )

    report = "\n".join(
        [
            "# Model Architectures",
            "",
            "## Executive Summary",
            "",
            "A short summary.",
            "",
            "## Major Theme 1: Architecture As Design Space",
            "",
            "For slides, this gives the audience a concrete technical lever.",
            "",
            "### Why This Matters",
            "",
            "Brief.",
            "",
            "### Slide Seeds",
            "",
            "- One seed.",
            "",
            "## Major Theme 2: Efficiency",
            "",
            "Brief.",
            "",
            "## Major Theme 3: Evaluation",
            "",
            "Brief.",
            "",
            "## Concise References",
            "",
            "- `paper-1`.",
        ]
    )

    validation = validate_narrative_topic_report(report)

    assert validation["status"] == "fail"
    assert validation["has_detailed_analysis"] is False
    assert validation["has_references_and_sources"] is False


def test_render_narrative_topic_report_blocks_live_without_llm_client() -> None:
    from conference_pipeline.analysis.narrative import (
        render_narrative_topic_report,
        validate_narrative_topic_report,
    )

    report = render_narrative_topic_report(
        _source_pack(),
        allow_deterministic_fallback=False,
    )

    assert "Status: blocked" in report
    assert "missing_llm_client" in report
    assert validate_narrative_topic_report(report)["status"] == "fail"


def test_render_narrative_topic_report_retries_failed_live_validation() -> None:
    from conference_pipeline.analysis.narrative import (
        deterministic_narrative_topic_report,
        render_narrative_topic_report,
        validate_narrative_topic_report,
    )

    client = _SequenceNarrativeClient(
        [
            "# Model Architectures\n\nToo short.\n",
            deterministic_narrative_topic_report(_source_pack()),
        ]
    )

    report = render_narrative_topic_report(
        _source_pack(),
        llm_client=client,
        allow_deterministic_fallback=False,
        max_llm_attempts=2,
    )

    assert validate_narrative_topic_report(report)["status"] == "pass"
    assert client.request_count == 2
    assert "Validation repair instructions" in client.user_prompts[1]


def test_openai_narrative_client_from_env_honors_timeout(monkeypatch) -> None:
    from conference_pipeline.analysis.narrative import OpenAIDeepNarrativeClient

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("NARRATIVE_ANALYSIS_MODEL", "gpt-5.5")
    monkeypatch.setenv("NARRATIVE_ANALYSIS_TIMEOUT_SECONDS", "42")

    client = OpenAIDeepNarrativeClient.from_env()

    assert client is not None
    assert client.model == "gpt-5.5"
    assert client.timeout_seconds == 42.0


def test_local_narrative_synthesis_client_uses_qwen_model(tmp_path) -> None:
    from conference_pipeline.iclr2025_live import LocalOpenAICompatibleSynthesisClient
    from conference_pipeline.provider_receipts import provider_receipt_context

    completions = _FakeChatCompletions(
        _FakeChatResponse(content="# Model Architectures\n\nLocal synthesis.")
    )
    client = LocalOpenAICompatibleSynthesisClient(
        model="mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
        base_url="http://127.0.0.1:8080/v1",
        _client=_FakeOpenAIClient(completions),
    )

    with provider_receipt_context(
        workspace_root=tmp_path,
        run_id="run-local",
        stage_id="analysis",
    ):
        result = client.synthesize(
            kind="deep_narrative",
            system_prompt="system",
            user_prompt="user",
        )

    assert result.text == "# Model Architectures\n\nLocal synthesis."
    assert result.model == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
    assert result.input_tokens == 17
    assert result.output_tokens == 23
    assert result.request_count == 1
    assert completions.calls[0]["model"] == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"


def test_narrative_client_from_env_selects_local_provider(monkeypatch) -> None:
    from conference_pipeline.iclr2025_live import (
        LocalOpenAICompatibleSynthesisClient,
        _narrative_client_from_env,
    )

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("NARRATIVE_ANALYSIS_PROVIDER", "local")
    monkeypatch.setenv("NARRATIVE_ANALYSIS_MODEL", "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit")
    monkeypatch.setenv("GPTR_LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")

    client = _narrative_client_from_env()

    assert isinstance(client, LocalOpenAICompatibleSynthesisClient)
    assert client.model == "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"


def test_render_narrative_topic_report_blocks_insufficient_source_pack() -> None:
    from conference_pipeline.analysis.narrative import (
        render_narrative_topic_report,
        validate_narrative_topic_report,
    )

    report = render_narrative_topic_report(
        {
            "topic_id": "02_model_architectures",
            "title": "Model Architectures",
            "finding_text": "",
            "chunks": [],
            "research_id": "",
        }
    )

    assert "Status: blocked" in report
    assert "## Blocked Receipt" in report
    assert validate_narrative_topic_report(report)["status"] == "fail"


def test_render_narrative_topic_report_blocks_index_only_paper_chunk_pairing() -> None:
    from conference_pipeline.analysis.narrative import render_narrative_topic_report

    pack = _source_pack()
    matched = list(pack["matched_papers"])  # type: ignore[index]
    matched[0] = {
        "paper_id": "paper-0",
        "title": "Architecture Paper 0",
        "paired_chunk_index": 0,
    }
    pack["matched_papers"] = matched

    report = render_narrative_topic_report(pack)

    assert "Status: blocked" in report
    assert "invalid_evidence_pairing" in report


def test_provenance_markdown_and_json_carry_old_marker_channels() -> None:
    from conference_pipeline.analysis.narrative import (
        render_provenance_markdown,
        render_topic_provenance_json,
    )

    pack = _source_pack()
    pack["matched_papers"] = [
        {
            "paper_id": f"paper-{index}",
            "title": f"Architecture Paper {index}",
            "url": f"https://openreview.net/forum?id=paper{index}",
            "evidence_snippet": "Aligned evidence.",
        }
        for index in range(8)
    ]
    provenance = render_provenance_markdown(
        pack,
        run_id="run-1",
        markdown_path="analysis/02_model_architectures.md",
        provenance_json_path="analysis/provenance/02_model_architectures.json",
    )
    structured = render_topic_provenance_json(
        topic_id="02_model_architectures",
        markdown_path="analysis/02_model_architectures.md",
        provenance_path="analysis/provenance/02_model_architectures.md",
        run_id="run-1",
        extraction_record_ids=list(pack["extraction_record_ids"]),
        vector_chunks=list(pack["chunks"]),
        research_id=str(pack["research_id"]),
        matched_papers=list(pack["matched_papers"]),
        slide_brief_path="analysis/slide_briefs/02_model_architectures.md",
        analysis_profile="deep-narrative-v1",
        narrative_model="gpt-5.5",
    )

    assert "## Source Basis" in provenance
    assert "<!-- extraction:extract-0 -->" in provenance
    assert "<!-- chunk:chunk-0 -->" in provenance
    assert "<!-- research:02_model_architectures#abcdef -->" in provenance
    assert "<!-- matched-paper:paper-0 -->" in provenance
    assert "<!-- matched-paper:paper-7 -->" in provenance
    assert structured["provenance_path"] == "analysis/provenance/02_model_architectures.md"
    assert structured["deep_analysis_path"] == "analysis/02_model_architectures.md"
    assert structured["slide_brief_path"] == "analysis/slide_briefs/02_model_architectures.md"
    assert structured["analysis_profile"] == "deep-narrative-v1"
    assert structured["narrative_model"] == "gpt-5.5"
    assert structured["vector_store_chunk_ids"] == ["chunk-0", "chunk-1", "chunk-2", "chunk-3"]


def test_provenance_json_includes_chunk_extraction_ids_without_top_level_ids() -> None:
    from conference_pipeline.analysis.narrative import render_topic_provenance_json

    pack = _source_pack()
    structured = render_topic_provenance_json(
        topic_id="02_model_architectures",
        markdown_path="analysis/02_model_architectures.md",
        provenance_path="analysis/provenance/02_model_architectures.md",
        run_id="run-1",
        extraction_record_ids=[],
        vector_chunks=list(pack["chunks"]),
        research_id=str(pack["research_id"]),
        matched_papers=list(pack["matched_papers"]),
    )

    assert structured["extraction_record_ids"] == [
        "extract-0",
        "extract-1",
        "extract-2",
        "extract-3",
    ]


def test_citation_manifest_collectors_use_sidecar_when_public_doc_has_no_markers(
    tmp_path: Path,
) -> None:
    from conference_pipeline.analysis.narrative import (
        collect_extraction_citations,
        collect_query_research_citations,
        collect_vector_store_citations,
        emit_analysis_citation_manifest,
    )

    workspace = tmp_path
    analysis_dir = workspace / "analysis"
    analysis_dir.mkdir()
    (analysis_dir / "02_model_architectures.md").write_text(
        "# Model Architectures\n\n## Executive Summary\n\nNarrative only.\n",
        encoding="utf-8",
    )
    emit_analysis_citation_manifest(
        workspace,
        "run-1",
        [
            {
                "topic_id": "02_model_architectures",
                "markdown_path": "analysis/02_model_architectures.md",
                "deep_analysis_path": "analysis/02_model_architectures.md",
                "slide_brief_path": "analysis/slide_briefs/02_model_architectures.md",
                "analysis_profile": "deep-narrative-v1",
                "narrative_model": "gpt-5.5",
                "deep_validation": {"status": "pass"},
                "provenance_path": "analysis/provenance/02_model_architectures.md",
                "provenance_json_path": "analysis/provenance/02_model_architectures.json",
                "extraction_record_ids": ["extract-1"],
                "vector_store_chunk_ids": ["chunk-1"],
                "research_ids": ["research-1"],
                "matched_paper_ids": ["paper-1"],
            }
        ],
    )

    report_paths = ["analysis/02_model_architectures.md"]
    assert collect_extraction_citations(workspace, report_paths) == {"extract-1"}
    assert collect_vector_store_citations(workspace, report_paths) == {"chunk-1"}
    assert collect_query_research_citations(workspace, report_paths) == {"research-1"}


def test_render_narrative_outputs_from_source_packs_writes_all_artifacts(tmp_path: Path) -> None:
    from conference_pipeline.analysis.narrative import render_narrative_outputs_from_source_packs

    source_pack_dir = tmp_path / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True)
    (source_pack_dir / "02_model_architectures.json").write_text(
        json.dumps(_source_pack(), indent=2),
        encoding="utf-8",
    )

    result = render_narrative_outputs_from_source_packs(tmp_path, run_id="run-1")

    assert result.status == "succeeded"
    assert (tmp_path / "analysis" / "02_model_architectures.md").exists()
    assert (tmp_path / "analysis" / "slide_briefs" / "02_model_architectures.md").exists()
    assert (tmp_path / "analysis" / "provenance" / "02_model_architectures.md").exists()
    assert (tmp_path / "analysis" / "provenance" / "02_model_architectures.json").exists()
    assert (tmp_path / "analysis" / "analysis_citation_manifest.json").exists()


def test_render_narrative_outputs_reuses_current_passing_reports(tmp_path: Path) -> None:
    from conference_pipeline.analysis.narrative import (
        deterministic_narrative_topic_report,
        render_narrative_outputs_from_source_packs,
    )

    pack = _source_pack()
    source_pack_dir = tmp_path / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True)
    (source_pack_dir / "02_model_architectures.json").write_text(
        json.dumps(pack, indent=2),
        encoding="utf-8",
    )
    report_path = tmp_path / "analysis" / "02_model_architectures.md"
    report_path.write_text(deterministic_narrative_topic_report(pack), encoding="utf-8")
    provenance_path = tmp_path / "analysis" / "provenance" / "02_model_architectures.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "topic_id": pack["topic_id"],
                "research_ids": [pack["research_id"]],
                "vector_store_chunk_ids": [
                    chunk["chunk_id"] for chunk in pack["chunks"]  # type: ignore[index]
                ],
                "deep_validation": {"status": "pass"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    client = _SequenceNarrativeClient([])

    result = render_narrative_outputs_from_source_packs(
        tmp_path,
        run_id="run-1",
        llm_client=client,
        allow_deterministic_fallback=False,
    )

    assert result.status == "succeeded"
    assert client.request_count == 0


def test_render_narrative_outputs_from_source_packs_rejects_path_escape_topic_id(
    tmp_path: Path,
) -> None:
    from conference_pipeline.analysis.narrative import render_narrative_outputs_from_source_packs

    source_pack_dir = tmp_path / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True)
    pack = _source_pack()
    pack["topic_id"] = "../../outside"
    (source_pack_dir / "bad.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid narrative topic_id"):
        render_narrative_outputs_from_source_packs(tmp_path, run_id="run-1")

    assert not (tmp_path.parent / "outside.md").exists()


def test_iclr_write_analysis_reports_rejects_path_escape_topic_id(tmp_path: Path) -> None:
    from conference_pipeline.iclr2025_live import write_analysis_reports

    selection = {
        "topics": [
            {
                "topic_id": "../../outside",
                "title": "Invalid Topic",
                "papers": [],
            }
        ]
    }

    with pytest.raises(ValueError, match="invalid narrative topic_id"):
        write_analysis_reports(tmp_path, selection, run_id="run-1")

    assert not (tmp_path.parent / "outside.md").exists()


# --- compact_chunk_evidence: structured source metadata preservation (Defect 1) ---


def test_compact_chunk_evidence_preserves_title_and_authors() -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": "A short quote.",
        "paper_title": "InsCal: Calibrated Multi-Source Prompt Tuning",
        "authors": ["Ada Lovelace", "Alan Turing"],
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert compacted["chunk_id"] == "chunk-1"
    assert compacted["extraction_record_id"] == "extract-1"
    assert compacted["source_artifact_path"] == "storage/corpus/paper-text/1.txt"
    assert compacted["quote"] == "A short quote."
    assert compacted["paper_title"] == "InsCal: Calibrated Multi-Source Prompt Tuning"
    assert compacted["authors"] == ["Ada Lovelace", "Alan Turing"]


def test_compact_chunk_evidence_falls_back_to_source_context_authors() -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": "A short quote.",
        "source_context": {"authors": ["Grace Hopper"]},
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert compacted["authors"] == ["Grace Hopper"]


def test_compact_chunk_evidence_metadata_free_chunk_has_only_original_four_keys() -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": "A short quote.",
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert set(compacted.keys()) == {
        "chunk_id",
        "extraction_record_id",
        "source_artifact_path",
        "quote",
    }


def test_compact_chunk_evidence_quote_still_truncated_to_600_chars() -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    long_quote = "word " * 300  # 1,500 raw chars, well over 600 once collapsed
    assert len(long_quote) > 1200
    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": long_quote,
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert len(compacted["quote"]) == 600


def test_compact_chunk_evidence_caps_author_roster_at_max_compact_authors() -> None:
    from conference_pipeline.analysis.narrative import (
        MAX_COMPACT_AUTHORS,
        compact_chunk_evidence,
    )

    roster = [f"Author {index}" for index in range(40)]
    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": "A short quote.",
        "authors": roster,
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert len(compacted["authors"]) == MAX_COMPACT_AUTHORS
    assert compacted["authors"] == roster[:MAX_COMPACT_AUTHORS]


def test_compact_chunk_evidence_truncates_title_at_max_compact_title_chars() -> None:
    from conference_pipeline.analysis.narrative import (
        MAX_COMPACT_TITLE_CHARS,
        compact_chunk_evidence,
    )

    long_title = "T" * (MAX_COMPACT_TITLE_CHARS + 100)
    chunk = {
        "chunk_id": "chunk-1",
        "extraction_record_id": "extract-1",
        "source_artifact_path": "storage/corpus/paper-text/1.txt",
        "quote": "A short quote.",
        "paper_title": long_title,
    }

    [compacted] = compact_chunk_evidence([chunk], limit=8)

    assert compacted["paper_title"] == long_title[:MAX_COMPACT_TITLE_CHARS]
    assert len(compacted["paper_title"]) == MAX_COMPACT_TITLE_CHARS


def test_compact_chunk_evidence_limit_caps_count_and_zero_returns_empty() -> None:
    from conference_pipeline.analysis.narrative import compact_chunk_evidence

    chunks = [
        {
            "chunk_id": f"chunk-{index}",
            "extraction_record_id": f"extract-{index}",
            "source_artifact_path": "storage/corpus/paper-text/1.txt",
            "quote": "A short quote.",
        }
        for index in range(5)
    ]

    assert len(compact_chunk_evidence(chunks, limit=2)) == 2
    assert compact_chunk_evidence(chunks, limit=0) == []


def _write_pack_and_current_report(tmp_path: Path, pack: dict[str, object]) -> tuple[Path, Path]:
    from conference_pipeline.analysis.narrative import deterministic_narrative_topic_report

    source_pack_dir = tmp_path / "analysis" / "narrative_source_packs"
    source_pack_dir.mkdir(parents=True, exist_ok=True)
    (source_pack_dir / f"{pack['topic_id']}.json").write_text(json.dumps(pack, indent=2), encoding="utf-8")
    report_path = tmp_path / "analysis" / f"{pack['topic_id']}.md"
    report_path.write_text(deterministic_narrative_topic_report(pack), encoding="utf-8")
    provenance_path = tmp_path / "analysis" / "provenance" / f"{pack['topic_id']}.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "topic_id": pack["topic_id"],
                "research_ids": [pack["research_id"]],
                "vector_store_chunk_ids": [chunk["chunk_id"] for chunk in pack["chunks"]],  # type: ignore[index]
            }
        ),
        encoding="utf-8",
    )
    return report_path, provenance_path


def test_carried_forward_topics_are_left_untouched_and_never_sent(tmp_path: Path) -> None:
    from conference_pipeline.analysis.narrative import render_narrative_outputs_from_source_packs

    pack = _source_pack()
    report_path, provenance_path = _write_pack_and_current_report(tmp_path, pack)
    before = (report_path.read_bytes(), provenance_path.read_bytes())
    client = _SequenceNarrativeClient([])

    result = render_narrative_outputs_from_source_packs(
        tmp_path,
        run_id="run-1",
        llm_client=client,
        allow_deterministic_fallback=False,
        reuse_existing_pass_reports=False,
        carry_forward_topic_ids=frozenset({str(pack["topic_id"])}),
    )

    assert result.status == "succeeded"
    assert client.request_count == 0
    assert (report_path.read_bytes(), provenance_path.read_bytes()) == before
    assert [summary["topic_id"] for summary in result.citation_summaries] == [pack["topic_id"]]


def test_a_carried_topic_whose_pack_changed_blocks_instead_of_rerendering(tmp_path: Path) -> None:
    from conference_pipeline.analysis.narrative import render_narrative_outputs_from_source_packs

    pack = _source_pack()
    report_path, provenance_path = _write_pack_and_current_report(tmp_path, pack)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["vector_store_chunk_ids"] = ["some-other-chunk"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    client = _SequenceNarrativeClient([])

    result = render_narrative_outputs_from_source_packs(
        tmp_path,
        run_id="run-1",
        llm_client=client,
        allow_deterministic_fallback=False,
        reuse_existing_pass_reports=False,
        carry_forward_topic_ids=frozenset({str(pack["topic_id"])}),
    )

    assert result.status == "blocked"
    assert [blocker["code"] for blocker in result.blockers] == ["narrative_carry_forward_not_current"]
    assert client.request_count == 0


def test_a_recorded_pack_digest_must_match_for_reuse(tmp_path: Path) -> None:
    from conference_pipeline.analysis.narrative import (
        _reuse_existing_narrative_if_current,
        source_pack_content_sha256,
    )

    pack = _source_pack()
    report_path, provenance_path = _write_pack_and_current_report(tmp_path, pack)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    provenance["source_pack_sha256"] = source_pack_content_sha256({**pack, "generated_at": "later"})
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    kwargs = dict(report_path=report_path, provenance_json_path=provenance_path, source_pack=pack, run_id="run-1")
    assert _reuse_existing_narrative_if_current(**kwargs) is not None

    provenance["source_pack_sha256"] = source_pack_content_sha256({**pack, "topic_reconciliation": {"x": 1}})
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    assert _reuse_existing_narrative_if_current(**kwargs) is None
