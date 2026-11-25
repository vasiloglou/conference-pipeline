from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.research.heuristic_prefilter import (
    HeuristicPrefilterError,
    build_prefiltered_evidence_pack,
    keyword_digest,
    normalize_keyword,
    select_prefiltered_evidence,
)


def _write_chunks(workspace: Path, run_id: str, store: str, rows: list[dict[str, object]]) -> None:
    chunks_path = workspace / "storage" / "vector-store" / run_id / "stores" / store / "chunks.jsonl"
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_select_prefiltered_evidence_uses_text_and_stable_metadata(tmp_path: Path) -> None:
    run_id = "run-1"
    _write_chunks(
        tmp_path,
        run_id,
        "papers",
        [
            {
                "chunk_id": "paper:1:0",
                "artifact_id": "paper:1",
                "source_family_id": "paper_text",
                "source_path": "paper.pdf",
                "content_kind": "paper",
                "conference_scope": "main_conference",
                "text": "Continuous latent reasoning reduces token budgets.",
                "source_context": {
                    "paper_title": "Latent Reasoning",
                    "source_url": "https://example.test/paper",
                },
                "locator": {"word_start": 0, "word_end": 8},
            }
        ],
    )
    _write_chunks(
        tmp_path,
        run_id,
        "artifacts",
        [
            {
                "chunk_id": "talk:1:0",
                "artifact_id": "talk:1",
                "source_family_id": "speech_transcript",
                "source_path": "talk.mp3",
                "content_kind": "transcript",
                "conference_scope": "workshop",
                "text": "The talk title matches even when the body is generic.",
                "source_context": {
                    "talk_title": "Latent CoT compression",
                    "workshop_title": "Reasoning Workshop",
                    "source_url": "https://example.test/talk",
                },
                "locator": {"start_seconds": 0, "end_seconds": 180},
            },
            {
                "chunk_id": "talk:2:0",
                "artifact_id": "talk:2",
                "source_family_id": "speech_transcript",
                "source_path": "talk2.mp3",
                "content_kind": "transcript",
                "conference_scope": "main_conference",
                "text": "Unrelated content.",
                "source_context": {"talk_title": "Other"},
            },
        ],
    )

    evidence = select_prefiltered_evidence(
        workspace=tmp_path,
        run_id=run_id,
        keywords=["latent reasoning", "Latent CoT"],
        stores=("papers", "artifacts"),
        limit=8,
    )

    assert [item.chunk_id for item in evidence] == ["paper:1:0", "talk:1:0"]
    assert evidence[0].matched_terms == ("latent reasoning",)
    assert evidence[1].matched_terms == ("latent cot",)
    assert evidence[1].source_context["talk_title"] == "Latent CoT compression"


def test_build_prefiltered_pack_records_contract_fields(tmp_path: Path) -> None:
    run_id = "run-1"
    _write_chunks(
        tmp_path,
        run_id,
        "papers",
        [
            {
                "chunk_id": "paper:1:0",
                "artifact_id": "paper:1",
                "source_family_id": "paper_text",
                "source_path": "paper.pdf",
                "content_kind": "paper",
                "conference_scope": "main_conference",
                "text": "Latent continuous reasoning and token efficiency.",
                "source_context": {"source_url": "https://example.test/paper"},
            }
        ],
    )

    pack = build_prefiltered_evidence_pack(
        workspace=tmp_path,
        run_id=run_id,
        deep_dive_id="latent-continuous-reasoning",
        question="How is latent reasoning replacing discrete CoT?",
        keywords=["latent continuous reasoning", "token efficiency"],
        keyword_seed_source="analysis/exhaustive_summaries/01.md",
        stores=("papers",),
    )

    assert pack["workflow"] == "workflow.research.heuristic_prefiltered_gpt_researcher"
    assert pack["retrieval_scope"] == "heuristic_keyword_prefilter"
    assert pack["prefilter_terms_sha256"] == keyword_digest(
        ["latent continuous reasoning", "token efficiency"]
    )
    assert pack["candidate_count"] == 1
    assert pack["selected_evidence_ids"] == ["paper:1:0"]
    assert pack["invalid_citation_ids"] == []
    assert pack["source_family_scope"] == ["paper_text"]


def test_select_prefiltered_evidence_caps_per_source_artifact(tmp_path: Path) -> None:
    run_id = "run-1"
    _write_chunks(
        tmp_path,
        run_id,
        "papers",
        [
            {
                "chunk_id": f"paper:1:{index}",
                "artifact_id": "paper:1",
                "source_family_id": "paper_text",
                "text": "latent reasoning token efficiency",
            }
            for index in range(3)
        ],
    )

    evidence = select_prefiltered_evidence(
        workspace=tmp_path,
        run_id=run_id,
        keywords=["latent reasoning"],
        stores=("papers",),
        limit=10,
        max_per_source_artifact=2,
    )

    assert len(evidence) == 2
    assert {item.source_artifact_id for item in evidence} == {"paper:1"}


def test_empty_keyword_list_blocks() -> None:
    assert normalize_keyword("  Latent   CoT  ") == "latent cot"
    with pytest.raises(HeuristicPrefilterError, match="empty_keyword_list"):
        keyword_digest([" ", ""])
