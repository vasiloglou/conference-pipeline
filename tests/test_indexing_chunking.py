from __future__ import annotations

import json
from pathlib import Path


def test_transcript_segments_create_120_second_chunks_with_timestamps():
    from conference_pipeline.indexing.chunking import transcript_chunks_from_segments

    segments = [
        {"start": 0, "end": 45, "text": "intro"},
        {"start": 45, "end": 95, "text": "method"},
        {"start": 95, "end": 145, "text": "results"},
    ]

    chunks = transcript_chunks_from_segments(segments, title="Demo Talk")

    assert len(chunks) == 2
    assert chunks[0].strategy == "transcript_120s_segments"
    assert chunks[0].locator["timestamp_start"] == 0
    assert chunks[0].locator["timestamp_end"] == 95
    assert "Presentation: Demo Talk" in chunks[0].text
    assert "Timestamp:" in chunks[0].text


def test_talk_aware_transcript_chunks_include_workshop_context():
    from conference_pipeline.indexing.chunking import chunks_for_record

    record = {
        "artifact_id": "workshop-transcript",
        "source_artifact_id": "workshop-transcript",
        "source_context": {
            "source_title": "Workshop Recording",
            "workshop_id": "workshop_demo",
            "workshop_title": "Workshop Demo",
        },
        "quality_metadata": {"input_kind": "transcript"},
    }
    source_text = json.dumps(
        {
            "segments": [
                {
                    "start": 0.0,
                    "end": 30.0,
                    "text": "This talk introduces the benchmark.",
                    "workshop_id": "workshop_demo",
                    "workshop_title": "Workshop Demo",
                    "talk_id": "talk_001",
                    "talk_title": "Talk Demo",
                    "talk_speaker": "Dr Demo",
                }
            ]
        }
    )

    chunks = chunks_for_record(record, source_text=source_text, source_family="speech_transcript")

    assert len(chunks) == 1
    assert "Workshop: Workshop Demo" in chunks[0].text
    assert "Talk: Talk Demo" in chunks[0].text
    assert "Speaker: Dr Demo" in chunks[0].text
    assert "Presentation: Workshop Recording" in chunks[0].text
    assert chunks[0].locator["workshop_id"] == "workshop_demo"
    assert chunks[0].locator["talk_id"] == "talk_001"
    assert chunks[0].locator["talk_title"] == "Talk Demo"


def test_slide_ocr_json_creates_one_locator_rich_slide_chunk():
    from conference_pipeline.indexing.chunking import chunks_for_record

    record = {
        "artifact_id": "slide-1",
        "source_artifact_id": "slide-1",
        "quality_metadata": {"input_kind": "page_image"},
    }
    source_text = json.dumps(
        {
            "slide_index": 7,
            "slide_text": {
                "title": "Routing",
                "bullet_points": ["Sparse experts", "Efficient dispatch"],
                "other_text": "Latency improves.",
            },
        }
    )

    chunks = chunks_for_record(record, source_text=source_text, source_family="artifact")

    assert len(chunks) == 1
    assert chunks[0].strategy == "slide_ocr_page"
    assert chunks[0].locator["slide_index"] == 7
    assert "Sparse experts" in chunks[0].text


def test_paper_metadata_chunk_uses_title_abstract_authors_keywords():
    from conference_pipeline.indexing.chunking import chunks_for_record

    record = {
        "artifact_id": "paper_metadata_fallback:abc",
        "quality_metadata": {
            "source_family": "paper_metadata_fallback",
            "paper_title": "A Better Benchmark",
            "abstract": "We evaluate multimodal agents carefully.",
            "authors": ["A. Researcher"],
            "keywords": ["benchmark", "agents"],
        },
    }

    chunks = chunks_for_record(record, source_text=None, source_family="paper_metadata_fallback")

    assert len(chunks) == 1
    assert chunks[0].strategy == "paper_metadata_title_abstract"
    assert "Paper Title: A Better Benchmark" in chunks[0].text
    assert "A. Researcher" in chunks[0].text
    assert "benchmark, agents" in chunks[0].text


def test_long_text_creates_overlapping_chunks_not_240_char_prefixes():
    from conference_pipeline.indexing.chunking import chunks_for_record

    words = [f"token{i}" for i in range(2200)]
    source_text = " ".join(words)
    record = {
        "artifact_id": "long-paper",
        "source_artifact_id": "long-paper",
        "quality_metadata": {"input_kind": "pdf_text"},
    }

    chunks = chunks_for_record(record, source_text=source_text, source_family="paper_text")

    assert len(chunks) >= 3
    assert all(len(chunk.text) > 240 for chunk in chunks)
    assert "token0" in chunks[0].text
    assert "token900" in chunks[1].text


def test_token_dense_text_respects_embedding_budget(monkeypatch):
    import conference_pipeline.indexing.chunking as chunking

    monkeypatch.setattr(chunking, "_token_count", lambda text: len(text.split()) * 10)

    words = [f"dense{i}" for i in range(1800)]
    record = {
        "artifact_id": "dense-source",
        "source_artifact_id": "dense-source",
    }

    chunks = chunking.chunks_for_record(
        record,
        source_text=" ".join(words),
        source_family="artifact",
    )

    assert len(chunks) > 3
    assert all(
        len(chunk.text.split()) * 10 <= chunking.DEFAULT_CHUNK_TOKENS
        for chunk in chunks
    )
    assert chunks[1].locator["word_start"] < chunks[0].locator["word_end"]
    assert chunks[1].locator["word_end"] > chunks[0].locator["word_end"]


def test_paper_prefix_counts_against_final_token_budget(monkeypatch):
    import conference_pipeline.indexing.chunking as chunking

    monkeypatch.setattr(chunking, "_token_count", lambda text: len(text.split()) * 10)

    record = {
        "artifact_id": "dense-paper",
        "source_artifact_id": "dense-paper",
        "paper_title": "A Long Metadata Header",
        "quality_metadata": {"input_kind": "pdf_text"},
    }
    source_text = " ".join(f"dense{i}" for i in range(500))

    chunks = chunking.chunks_for_record(
        record,
        source_text=source_text,
        source_family="paper_text",
    )

    assert chunks
    for chunk in chunks:
        assert "Paper Title: A Long Metadata Header" in chunk.text
        assert chunking._token_count(chunk.text) <= chunking.DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS


def test_chunk_tokens_are_token_budget_not_word_budget(monkeypatch):
    import conference_pipeline.indexing.chunking as chunking

    monkeypatch.setattr(chunking, "_token_count", lambda text: len(text.split()) * 2)

    chunks = chunking.chunks_for_record(
        {"artifact_id": "token-budget-source"},
        source_text=" ".join(f"term{i}" for i in range(800)),
        source_family="artifact",
    )

    assert len(chunks) > 1
    assert all(chunking._token_count(chunk.text) <= chunking.DEFAULT_CHUNK_TOKENS for chunk in chunks)


def test_over_budget_transcript_window_splits_with_overlap_and_timestamps(monkeypatch):
    import conference_pipeline.indexing.chunking as chunking

    monkeypatch.setattr(chunking, "_token_count", lambda text: len(text.split()) * 10)

    dense_text = " ".join(f"spoken{i}" for i in range(500))
    chunks = chunking.transcript_chunks_from_segments(
        [{"start": 0.0, "end": 119.0, "text": dense_text}],
        title="Dense Talk",
    )

    assert len(chunks) > 1
    assert all(chunk.strategy == "transcript_120s_token_window" for chunk in chunks)
    assert all("timestamp_start" in chunk.locator for chunk in chunks)
    assert all("timestamp_end" in chunk.locator for chunk in chunks)
    first_words = chunks[0].text.split()
    second_words = set(chunks[1].text.split())
    assert any(word in second_words for word in first_words[-chunking.DEFAULT_CHUNK_OVERLAP :])


def test_large_text_token_count_uses_fast_approximation(monkeypatch):
    import builtins
    import conference_pipeline.indexing.chunking as chunking

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise AssertionError("large text should not import tokenizer")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert chunking._token_count("word " * 12000) == 15000


def test_empty_inputs_do_not_create_fake_chunks():
    from conference_pipeline.indexing.chunking import chunks_for_record

    chunks = chunks_for_record(
        {"artifact_id": "empty", "evidence": {"snippet": ""}},
        source_text="",
        source_family="artifact",
    )

    assert chunks == []


def test_empty_paper_metadata_does_not_create_fake_chunk():
    from conference_pipeline.indexing.chunking import chunks_for_record

    chunks = chunks_for_record(
        {
            "artifact_id": "paper_metadata_fallback:empty",
            "quality_metadata": {"source_family": "paper_metadata_fallback"},
        },
        source_text=None,
        source_family="paper_metadata_fallback",
    )

    assert chunks == []


def test_chunker_source_does_not_reintroduce_240_char_vector_text():
    chunker_source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "conference_pipeline"
        / "indexing"
        / "chunking.py"
    ).read_text(encoding="utf-8")

    assert "[:240]" not in chunker_source
    assert "text[:240]" not in chunker_source
