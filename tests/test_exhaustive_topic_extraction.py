"""Phase 14A foundation tests: walker, classifier, records, coverage, runner.

These tests use a self-contained tmp workspace built from the same canonical
shape as `conferences/iclr2025/storage/...` but with a small fixture slice.
No real provider calls. No live network. Each test is bounded ≤ 30s.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest


# ---------------------------------------------------------------------------
# Tmp-workspace fixture
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_TAXONOMY_FIXTURE: dict[str, Any] = {
    "schema_version": 1,
    "conference": "iclr2025",
    "milestone": "v1.5",
    "status": "production-final-v1.5",
    "topic_families": [
        {
            "id": "02_model_architectures",
            "title": "Model Architectures",
            "source_families": ["accepted-papers", "orals"],
            "evidence_status": "live",
        },
        {
            "id": "06_multimodal_systems",
            "title": "Multimodal Systems",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
        {
            "id": "11_datasets_tooling",
            "title": "Datasets and Tooling",
            "source_families": ["accepted-papers"],
            "evidence_status": "live",
        },
    ],
}


def _build_workspace(workspace: Path, *, transcript_text: str | None = None) -> Path:
    """Build a 5-row corpus inventory + matching transcription receipts and
    a single paper download, with on-disk text files for resolution tests.
    """
    workspace.mkdir(parents=True, exist_ok=True)
    # Transcript file
    transcript_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "media"
        / "oral-1"
        / "transcript.txt"
    )
    _write_text(
        transcript_path,
        transcript_text
        or (
            "We propose a novel transformer architecture for multimodal "
            "vision-language reasoning. Our network combines attention "
            "mechanisms with modality-specific encoders to fuse image and "
            "text representations." * 4
        ),
    )
    # Local PDF text
    pdf_text_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "media"
        / "poster-1"
        / "text.txt"
    )
    _write_text(
        pdf_text_path,
        "We introduce a novel dataset and toolkit for benchmarking model "
        "architectures across diverse data sources and evaluation pipelines."
        * 6,
    )
    # OCR JSON
    ocr_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "media"
        / "poster-2"
        / "ocr.json"
    )
    _write_json(
        ocr_path,
        {
            "text": (
                "Multimodal vision-language transformer model architecture "
                "for image-and-text alignment." * 8
            )
        },
    )
    # Page image artifact (no readable text — should be empty_or_too_short)
    image_path = workspace / "storage" / "transcription-ocr" / "media" / "poster-3" / "page.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"png-bytes")
    # Paper PDF text (text-extracted)
    paper_text_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "text-extraction"
        / "openreview"
        / "ABC123"
        / "abstract.txt"
    )
    paper_text_short_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "text-extraction"
        / "openreview"
        / "ABC123.txt"
    )
    _write_text(
        paper_text_short_path,
        "Datasets and tooling for benchmarking transformer architectures and "
        "multimodal evaluation suites." * 8,
    )

    corpus_artifacts = [
        {
            "artifact_id": "media:oral-1:transcript",
            "source_family": "iclr-virtual-media",
            "processing_status": "transcribed",
            "transcript_path": "storage/transcription-ocr/media/oral-1/transcript.txt",
            "ocr_path": None,
            "local_path": "cache/media/oral-1/audio.m4a",
            "topic_id": None,
            "provenance_digest": "digest-oral-1",
        },
        {
            "artifact_id": "media:poster-1:pdf",
            "source_family": "iclr-virtual-media",
            "processing_status": "local_text_extracted",
            "transcript_path": None,
            "ocr_path": None,
            "local_path": "cache/media/poster-1/poster.pdf",
            "topic_id": None,
            "provenance_digest": "digest-poster-1",
            "text_extraction_sha256": "digest-poster-1-text",
        },
        {
            "artifact_id": "media:poster-2:ocr",
            "source_family": "iclr-virtual-media",
            "processing_status": "ocr_done",
            "transcript_path": None,
            "ocr_path": "storage/transcription-ocr/media/poster-2/ocr.json",
            "local_path": "cache/media/poster-2/page.png",
            "topic_id": None,
            "provenance_digest": "digest-poster-2",
        },
        {
            "artifact_id": "media:poster-3:image",
            "source_family": "iclr-virtual-media",
            "processing_status": "image_artifact_indexed",
            "transcript_path": None,
            "ocr_path": None,
            "local_path": "cache/media/poster-3/page.png",
            "topic_id": None,
            "provenance_digest": "digest-poster-3",
        },
        {
            "artifact_id": "openreview:ABC123",
            "source_family": "openreview-venue",
            "processing_status": "text_extracted",
            "transcript_path": None,
            "ocr_path": None,
            "local_path": "cache/openreview/ABC123-paper_pdf.pdf",
            "topic_id": "02_model_architectures",
            "text_extraction_sha256": "digest-paper-abc",
        },
    ]
    _write_json(
        workspace / "storage" / "corpus" / "v15_corpus_inventory.json",
        {
            "schema_version": 1,
            "generated_at": "2026-04-27T00:00:00Z",
            "artifact_count": len(corpus_artifacts),
            "media_artifact_count": 4,
            "selection_source": "test-fixture",
            "artifacts": corpus_artifacts,
            "source_gap_records": [],
        },
    )
    _write_json(
        workspace
        / "storage"
        / "transcription-ocr"
        / "v15_transcription_ocr_receipts.json",
        {
            "schema_version": 1,
            "generated_at": "2026-04-27T00:00:00Z",
            "artifacts": [
                {
                    "artifact_id": "media:oral-1:transcript",
                    "status": "transcribed",
                    "source_path": "cache/media/oral-1/audio.m4a",
                    "text_path": "storage/transcription-ocr/media/oral-1/transcript.txt",
                    "provider_calls": 1,
                    "error": None,
                },
                {
                    "artifact_id": "media:poster-1:pdf",
                    "status": "local_text_extracted",
                    "source_path": "cache/media/poster-1/poster.pdf",
                    "text_path": "storage/transcription-ocr/media/poster-1/text.txt",
                    "provider_calls": 0,
                    "error": None,
                },
                {
                    "artifact_id": "media:poster-2:ocr",
                    "status": "ocr_done",
                    "source_path": "cache/media/poster-2/page.png",
                    "text_path": "storage/transcription-ocr/media/poster-2/ocr.json",
                    "provider_calls": 0,
                    "error": None,
                },
                {
                    "artifact_id": "media:poster-3:image",
                    "status": "image_artifact_indexed",
                    "source_path": "cache/media/poster-3/page.png",
                    "text_path": None,
                    "provider_calls": 0,
                    "error": None,
                },
                {
                    "artifact_id": "openreview:ABC123",
                    "status": "text_extracted",
                    "source_path": "cache/openreview/ABC123-paper_pdf.pdf",
                    "text_path": "storage/transcription-ocr/text-extraction/openreview/ABC123.txt",
                    "provider_calls": 0,
                    "error": None,
                },
            ],
        },
    )
    _write_json(
        workspace / "storage" / "acquisition" / "v15_download_receipts.json",
        {
            "schema_version": 1,
            "generated_at": "2026-04-27T00:00:00Z",
            "local_artifacts": [
                {
                    "paper_id": "ABC123",
                    "local_path": "cache/openreview/ABC123-paper_pdf.pdf",
                    "sha256": "digest-paper-abc",
                    "size_bytes": 100,
                    "source_kind": "paper_pdf",
                    "source_url": "https://openreview.net/pdf/ABC123.pdf",
                    "status": "succeeded",
                    "topic_ids": ["02_model_architectures"],
                    "content_type": "application/pdf",
                    "retention_class": "local-cache",
                    "storage_lane": "cache/acquisition-downloads",
                }
            ],
        },
    )
    _write_json(
        workspace / "storage" / "acquisition" / "v15_openreview_papers.json",
        {
            "schema_version": 1,
            "generated_at": "2026-04-27T00:00:00Z",
            "papers": [
                {
                    "id": "ABC123",
                    "forum": "ABC123",
                    "title": "Selected Transformer Architectures",
                    "abstract": (
                        "Datasets and tooling for benchmarking transformer "
                        "architectures and multimodal evaluation suites."
                    ),
                    "authors": ["Ada Example"],
                    "keywords": ["transformers", "evaluation"],
                    "primary_area": "deep learning",
                    "venue": "ICLR 2025 Poster",
                    "forum_url": "https://openreview.net/forum?id=ABC123",
                    "pdf_url": "https://openreview.net/pdf/ABC123.pdf",
                },
                {
                    "id": "XYZ789",
                    "forum": "XYZ789",
                    "title": "Unselected Metadata-Only Multimodal Benchmark",
                    "abstract": (
                        "This paper studies multimodal systems, datasets, "
                        "tooling, benchmarks, and model architectures using "
                        "OpenReview metadata when no local PDF text was "
                        "selected by the legacy corpus."
                    ),
                    "authors": ["Grace Example"],
                    "keywords": ["multimodal", "benchmarks", "datasets"],
                    "primary_area": "multimodal learning",
                    "venue": "ICLR 2025 Poster",
                    "forum_url": "https://openreview.net/forum?id=XYZ789",
                    "pdf_url": "https://openreview.net/pdf/XYZ789.pdf",
                },
            ],
        },
    )

    # Taxonomy
    _write_json(workspace / "analysis" / "iclr2025_taxonomy.json", _TAXONOMY_FIXTURE)

    # Stage receipts: write a succeeded transcription-ocr receipt so the
    # exhaustive-topic-extraction gate passes.
    _write_json(
        workspace
        / "storage"
        / "workflow-runs"
        / "testrun"
        / "stage-13-transcription-ocr.json",
        {
            "schema_version": 1,
            "generated_at": "2026-04-27T00:00:00Z",
            "run_id": "testrun",
            "stage_id": "transcription-ocr",
            "phase": 13,
            "status": "succeeded",
            "command": "test-fixture",
            "artifacts": [],
            "blockers": [],
            "details": {},
            "secrets_copied": False,
        },
    )

    return workspace


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    return _build_workspace(tmp_path)


# ---------------------------------------------------------------------------
# 1. Stage registration
# ---------------------------------------------------------------------------


def test_stage_registered():
    from conference_pipeline.iclr2025_live import STAGE_FILES
    from conference_pipeline.workflow import (
        EXTRACTION_WORKFLOW_STAGES,
        LEGACY_WORKFLOW_STAGES,
        WORKFLOW_STAGE_BY_ID,
    )

    assert "exhaustive-topic-extraction" in WORKFLOW_STAGE_BY_ID
    assert STAGE_FILES["exhaustive-topic-extraction"] == (
        14,
        "14A-exhaustive-topic-extraction-receipt.json",
    )
    assert any(
        stage.id == "exhaustive-topic-extraction"
        for stage in EXTRACTION_WORKFLOW_STAGES
    )
    # The stage is appended at the end of the registry, after analyze, so
    # its registry index is strictly greater than analyze's. Order numbers
    # remain stable for legacy stages.
    legacy_ids = [stage.id for stage in LEGACY_WORKFLOW_STAGES]
    assert "analyze" in legacy_ids
    assert "exhaustive-topic-extraction" not in legacy_ids


# ---------------------------------------------------------------------------
# 2. Walker rejects paths outside workspace
# ---------------------------------------------------------------------------


def test_walker_rejects_paths_outside_workspace(tmp_workspace: Path):
    from conference_pipeline.extractions.walker import iterate_extraction_inputs

    # Mutate the receipts file to add a row with a path-traversal text_path.
    receipts_path = (
        tmp_workspace
        / "storage"
        / "transcription-ocr"
        / "v15_transcription_ocr_receipts.json"
    )
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    # Also add an entry to the corpus inventory so the walker sees this row.
    inv_path = tmp_workspace / "storage" / "corpus" / "v15_corpus_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    inv["artifacts"].append(
        {
            "artifact_id": "media:malicious",
            "source_family": "iclr-virtual-media",
            "processing_status": "local_text_extracted",
            "transcript_path": "../../etc/passwd",
            "ocr_path": None,
            "local_path": "cache/etc/passwd",
            "topic_id": None,
            "provenance_digest": "digest-malicious",
        }
    )
    receipts["artifacts"].append(
        {
            "artifact_id": "media:malicious",
            "status": "local_text_extracted",
            "source_path": "../../etc/passwd",
            "text_path": "../../etc/passwd",
            "provider_calls": 0,
            "error": None,
        }
    )
    receipts_path.write_text(json.dumps(receipts), encoding="utf-8")
    inv_path.write_text(json.dumps(inv), encoding="utf-8")

    inputs = list(iterate_extraction_inputs(tmp_workspace))
    malicious = [i for i in inputs if i.artifact_id == "media:malicious"]
    assert malicious, "walker should still surface the row"
    # The path must NOT have been resolved into a usable text_path or text_content.
    assert malicious[0].text_path is None
    assert malicious[0].text_content is None


# ---------------------------------------------------------------------------
# 3. Walker covers all completed artifacts in the fixture slice
# ---------------------------------------------------------------------------


def test_walker_covers_all_completed_artifacts(tmp_workspace: Path):
    from conference_pipeline.extractions.walker import iterate_extraction_inputs

    inputs = list(iterate_extraction_inputs(tmp_workspace))
    # 5 corpus rows total; the selected paper is already covered by the corpus
    # row "openreview:ABC123". The unselected fetched OpenReview paper is added
    # as metadata fallback so the paper vector store is not limited to the old
    # 4-papers-per-topic corpus.
    artifact_ids = sorted(i.artifact_id for i in inputs)
    assert artifact_ids == [
        "media:oral-1:transcript",
        "media:poster-1:pdf",
        "media:poster-2:ocr",
        "media:poster-3:image",
        "openreview:ABC123",
        "paper_metadata_fallback:XYZ789",
    ]


def test_walker_accepts_modern_corpus_and_topic_strategy(tmp_path: Path):
    from conference_pipeline.extractions.walker import iterate_extraction_inputs
    from conference_pipeline.iclr2025_live import load_topic_families

    workspace = tmp_path / "modern"
    annotated_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "workshop-talks"
        / "demo"
        / "annotated-transcript.json"
    )
    _write_json(
        annotated_path,
        {
            "segments": [
                {
                    "talk_id": "talk_001",
                    "talk_title": "Reliable Financial Agents",
                    "talk_speaker": "Ada Example",
                    "description": "A workshop keynote about responsible agentic systems.",
                    "text": "Agents use tools, evaluate risk, and reason over multimodal evidence.",
                }
            ]
        },
    )
    paper_text_path = workspace / "storage" / "transcription-ocr" / "text-extraction" / "paper.txt"
    _write_text(
        paper_text_path,
        "A paper on reasoning, post-training, tool use, and reliable agent systems." * 8,
    )
    _write_json(
        workspace / "analysis" / "topic_strategy.json",
        {
            "final_topics": [
                {
                    "topic_id": "01_reasoning_agents",
                    "title": "Reasoning agents and tool use",
                    "source_families": ["speech_transcript", "paper_text"],
                }
            ]
        },
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"lanes": {"session_speech_transcription": {"records": []}}},
    )
    _write_json(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_records": [
                {
                    "artifact_id": "speech_transcript:demo",
                    "indexable_text": True,
                    "indexed_text_kind": "annotated_transcript_json",
                    "indexed_text_path": str(annotated_path.relative_to(workspace)),
                    "indexed_text_sha256": "annotated-digest",
                    "metadata": {
                        "workshop_id": "ICLR.cc/2026/Workshop/DEMO",
                        "talk_id": "talk_001",
                        "talk_title": "Reliable Financial Agents",
                    },
                    "source_family_id": "speech_transcript",
                    "source_path": "storage/workshops/talk-media/demo/raw.mp3",
                    "source_sha256": "audio-digest",
                    "storage_lane": "session_speech_transcription",
                },
                {
                    "artifact_id": "paper_text:demo",
                    "indexable_text": True,
                    "source_family_id": "paper_text",
                    "source_path": "cache/paper.pdf",
                    "source_sha256": "paper-digest",
                    "text_path": str(paper_text_path.relative_to(workspace)),
                    "text_sha256": "paper-text-digest",
                },
                {
                    "artifact_id": "media_asset:audit-only",
                    "indexable_text": False,
                    "source_family_id": "media_asset",
                    "source_path": "cache/poster.png",
                },
            ]
        },
    )

    inputs = list(iterate_extraction_inputs(workspace))
    assert [item.artifact_id for item in inputs] == [
        "speech_transcript:demo",
        "paper_text:demo",
    ]
    assert inputs[0].kind == "transcript"
    assert inputs[0].status == "transcribed"
    assert "Reliable Financial Agents" in (inputs[0].text_content or "")
    assert inputs[0].metadata["workshop_id"] == "ICLR.cc/2026/Workshop/DEMO"
    assert inputs[1].kind == "paper_pdf"
    assert load_topic_families(workspace)[0].id == "01_reasoning_agents"


def test_walker_accepts_corpus_inventory_with_no_transcription_receipts_file(
    tmp_path: Path,
):
    """A workspace whose corpus_inventory.json already embeds transcript/OCR
    paths directly (e.g. a conference's own corpus-inventory-index output)
    may never produce a v15_transcription_ocr_receipts.json or
    transcription_ocr_inventory.json sidecar at all. The walker must still
    extract every corpus row rather than hard-failing on the missing sidecar.
    """
    from conference_pipeline.extractions.walker import iterate_extraction_inputs

    workspace = tmp_path / "no-receipts-sidecar"
    paper_text_path = workspace / "storage" / "corpus" / "paper-text" / "paper.txt"
    _write_text(
        paper_text_path,
        "A paper on reasoning, post-training, tool use, and reliable agent systems." * 8,
    )
    transcript_path = workspace / "storage" / "download-corpus" / "event-media" / "captions" / "demo.vtt"
    _write_text(transcript_path, "WEBVTT\n\nAgents use tools and reason over evidence." * 4)

    # Deliberately no storage/transcription-ocr/* receipts file of any kind.
    assert not (workspace / "storage" / "transcription-ocr").exists()

    _write_json(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_records": [
                {
                    "artifact_id": "event_media_caption:demo",
                    "indexable_text": True,
                    "source_family_id": "speech_transcript",
                    "source_path": str(transcript_path.relative_to(workspace)),
                    "source_sha256": "caption-digest",
                    "transcript_path": str(transcript_path.relative_to(workspace)),
                    "storage_lane": "storage/download-corpus/event-media/captions",
                },
                {
                    "artifact_id": "paper_text:demo",
                    "indexable_text": True,
                    "source_family_id": "paper_text",
                    "source_path": "cache/paper.pdf",
                    "source_sha256": "paper-digest",
                    "text_path": str(paper_text_path.relative_to(workspace)),
                    "text_sha256": "paper-text-digest",
                },
            ]
        },
    )

    inputs = list(iterate_extraction_inputs(workspace))
    assert {item.artifact_id for item in inputs} == {
        "event_media_caption:demo",
        "paper_text:demo",
    }
    by_id = {item.artifact_id: item for item in inputs}
    assert "Agents use tools" in (by_id["event_media_caption:demo"].text_content or "")
    assert by_id["paper_text:demo"].kind == "paper_pdf"


def test_walker_prefers_transcript_text_over_json_sidecar(tmp_path: Path):
    from conference_pipeline.extractions.walker import iterate_extraction_inputs

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    transcript_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "speech-transcripts"
        / "speech-raw-demo.txt"
    )
    sidecar_path = transcript_path.with_suffix(".json")
    transcript_text = (
        "This poster audio presents agentic reasoning benchmarks and "
        "evaluation methods for reliable tool-using systems. "
    ) * 6
    _write_text(transcript_path, transcript_text)
    _write_json(
        sidecar_path,
        {
            "receipt_type": "conference_pipeline.speech_transcript",
            "transcript_path": str(transcript_path.relative_to(workspace)),
            "text_character_count": len(transcript_text),
        },
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"lanes": {"session_speech_transcription": {"records": []}}},
    )
    _write_json(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_records": [
                {
                    "artifact_id": "speech_transcript:main-demo",
                    "indexable_text": True,
                    "indexed_text_kind": "transcript_json",
                    "indexed_text_path": str(sidecar_path.relative_to(workspace)),
                    "indexed_text_sha256": "sidecar-digest",
                    "source_family_id": "speech_transcript",
                    "source_path": "cache/acquisition-downloads/media_asset/poster-1/slideslive_audio/raw.mp3",
                    "source_sha256": "audio-digest",
                    "storage_lane": "session_speech_transcription",
                    "text_byte_count": len(transcript_text.encode("utf-8")),
                    "text_sha256": "transcript-text-digest",
                    "transcript_json_path": str(sidecar_path.relative_to(workspace)),
                    "transcript_path": str(transcript_path.relative_to(workspace)),
                }
            ]
        },
    )

    [input_] = list(iterate_extraction_inputs(workspace))
    assert input_.kind == "transcript"
    assert input_.text_path == str(transcript_path.relative_to(workspace))
    assert input_.text_content == transcript_text
    assert input_.metadata["indexed_text_path"] == str(sidecar_path.relative_to(workspace))
    assert input_.metadata["transcript_path"] == str(transcript_path.relative_to(workspace))


def test_walker_dereferences_transcript_json_sidecar(tmp_path: Path):
    from conference_pipeline.extractions.walker import iterate_extraction_inputs

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    transcript_path = (
        workspace
        / "storage"
        / "transcription-ocr"
        / "speech-transcripts"
        / "speech-raw-sidecar-only.txt"
    )
    sidecar_path = transcript_path.with_suffix(".json")
    transcript_text = (
        "This main conference talk discusses multimodal systems, evaluation "
        "benchmarks, and robust agentic workflows. "
    ) * 6
    _write_text(transcript_path, transcript_text)
    _write_json(
        sidecar_path,
        {
            "receipt_type": "conference_pipeline.speech_transcript",
            "transcript_path": str(transcript_path.relative_to(workspace)),
            "text_character_count": len(transcript_text),
        },
    )
    _write_json(
        workspace / "storage" / "transcription-ocr" / "transcription_ocr_inventory.json",
        {"lanes": {"session_speech_transcription": {"records": []}}},
    )
    _write_json(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_records": [
                {
                    "artifact_id": "speech_transcript:sidecar-only",
                    "indexable_text": True,
                    "indexed_text_kind": "transcript_json",
                    "indexed_text_path": str(sidecar_path.relative_to(workspace)),
                    "indexed_text_sha256": "sidecar-digest",
                    "source_family_id": "speech_transcript",
                    "source_path": "cache/acquisition-downloads/media_asset/poster-2/slideslive_audio/raw.mp3",
                    "source_sha256": "audio-digest",
                    "storage_lane": "session_speech_transcription",
                }
            ]
        },
    )

    [input_] = list(iterate_extraction_inputs(workspace))
    assert input_.kind == "transcript"
    assert input_.text_path == str(transcript_path.relative_to(workspace))
    assert input_.text_content == transcript_text


def test_llm_full_read_limited_receipt_is_partial(tmp_workspace: Path):
    from conference_pipeline.extractions.llm_full_read import run_llm_full_read

    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "ABC123",
                    "forum": "ABC123",
                    "title": "Long Abstract For Full Read",
                    "abstract": (
                        "This abstract discusses multimodal systems, reasoning agents, "
                        "evaluation methods, benchmarks, and datasets in enough detail "
                        "to exceed the full-read minimum character threshold. "
                    )
                    * 3,
                    "forum_url": "https://openreview.net/forum?id=ABC123",
                    "pdf_url": "https://openreview.net/pdf?id=ABC123",
                }
            ]
        },
    )

    receipt = run_llm_full_read(
        workspace_root=tmp_workspace,
        run_id="testrun",
        scope="all",
        limit=0,
        resume=False,
    )

    assert receipt["status"] == "partial"
    # v5.1 sub-step lineage: a partial full read is never bound as current.
    assert "lineage" not in receipt and "substep_outputs" not in receipt
    assert receipt["lineage_unbound_reason"] == "status:partial"
    checks = receipt["checks"]
    assert checks["completion_state"] == "partial"
    assert checks["expected_unit_count"] >= 2
    assert checks["completed_unit_count"] == 0
    assert {
        blocker["code"] for blocker in checks["completion_blockers"]
    } >= {"llm_full_read_limit_set", "llm_full_read_incomplete_units"}


def test_llm_full_read_priority_ordering_preserves_all_units(tmp_workspace: Path):
    from conference_pipeline.extractions.llm_full_read import (
        _prioritize_units,
        iter_units_for_scope,
    )

    _write_json(
        tmp_workspace / "analysis" / "topic_strategy.json",
        {
            "final_topics": [
                {
                    "topic_id": "01_ai_for_science_physics_discovery",
                    "title": "AI for science, physics, and autonomous discovery",
                    "starter_queries": [
                        "physics scientific machine learning differential equations",
                    ],
                }
            ]
        },
    )
    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "NONPRIORITY",
                    "forum": "NONPRIORITY",
                    "title": "Generic Multimodal Benchmark",
                    "abstract": (
                        "This abstract discusses multimodal evaluation, benchmark "
                        "construction, and dataset tooling in sufficient detail. "
                    )
                    * 4,
                },
                {
                    "id": "PHYSICS",
                    "forum": "PHYSICS",
                    "title": "Physics-Informed Discovery of Dynamical Systems",
                    "abstract": (
                        "This abstract studies physics-informed scientific machine "
                        "learning for differential equations and autonomous discovery. "
                    )
                    * 4,
                },
            ]
        },
    )

    units = list(
        iter_units_for_scope(
            tmp_workspace,
            scope="all",
            segment_seconds=180,
            overlap_seconds=60,
            min_chars=200,
        )
    )
    prioritized = _prioritize_units(
        units,
        workspace_root=tmp_workspace,
        priority_topic_ids=["01_ai_for_science_physics_discovery"],
    )

    assert [unit.unit_id for unit in sorted(prioritized, key=lambda unit: unit.unit_id)] == [
        unit.unit_id for unit in sorted(units, key=lambda unit: unit.unit_id)
    ]
    assert prioritized[0].unit_id == "paper_abstract:PHYSICS"
    assert "paper_abstract:NONPRIORITY" in [unit.unit_id for unit in prioritized]


def test_llm_full_read_prompt_uses_workspace_conference_name(tmp_workspace: Path):
    from conference_pipeline.extractions.llm_full_read import (
        FullReadInput,
        _build_messages,
    )
    from conference_pipeline.iclr2025_live import TopicFamily

    _write_text(
        tmp_workspace / "config.yaml",
        'conference:\n  name: "ICML 2026"\n',
    )

    messages = _build_messages(
        workspace_root=tmp_workspace,
        unit=FullReadInput(
            unit_id="paper_abstract:PHYSICS",
            source_artifact_id="paper_abstract:PHYSICS",
            input_kind="paper_abstract",
            text="Physics-informed scientific machine learning for differential equations.",
            source_artifact_path="https://openreview.net/forum?id=PHYSICS",
            source_artifact_digest="digest",
            metadata={"title": "Physics-Informed Scientific ML"},
        ),
        topics=[
            TopicFamily(
                id="01_ai_for_science_physics_discovery",
                title="AI for science, physics, and autonomous discovery",
                source_families=(),
            )
        ],
        topic_block="- 01_ai_for_science_physics_discovery: AI for science.",
    )

    assert "ICML 2026 conference evidence" in messages[0]["content"]
    assert "ICLR 2026" not in messages[0]["content"]
    # Keyword overlap is not topic membership: an all-day wireless workshop
    # recording reached an agentic-AI deck because segments saying "the 6G
    # stack will be written by agents" classified as agentic AI.
    assert "PRIMARY research contribution" in messages[0]["content"]
    assert "outside_taxonomy" in messages[0]["content"]


def test_llm_full_read_resume_metric_accumulator_preserves_record_metadata():
    from conference_pipeline.extractions.llm_full_read import (
        FullReadResult,
        _accumulate_record_metrics,
    )

    result = FullReadResult()
    _accumulate_record_metrics(
        result,
        {
            "text_length": "1234",
            "important": True,
            "topic_ids": ["01_ai_for_science_physics_discovery"],
            "provider_metadata": {
                "provider_receipt_path": "storage/workflow-runs/run/provider-calls/one.json",
                "input_tokens": "101",
                "output_tokens": 17,
            },
        },
    )
    _accumulate_record_metrics(
        result,
        {
            "text_length": 200,
            "skipped_reason": "llm_failed",
        },
    )

    assert result.input_chars == 1434
    assert result.important_count == 1
    assert result.failed_count == 1
    assert result.input_tokens == 101
    assert result.output_tokens == 17
    assert result.provider_receipt_paths == [
        "storage/workflow-runs/run/provider-calls/one.json"
    ]


def test_openreview_metadata_fallback_covers_unselected_papers(tmp_workspace: Path):
    from conference_pipeline.extractions.classifier import classify_artifact
    from conference_pipeline.extractions.walker import iterate_extraction_inputs
    from conference_pipeline.iclr2025_live import load_topic_families

    taxonomy = load_topic_families(tmp_workspace)
    inputs = list(iterate_extraction_inputs(tmp_workspace))
    fallback = next(i for i in inputs if i.artifact_id == "paper_metadata_fallback:XYZ789")

    assert fallback.kind == "paper_metadata_fallback"
    assert fallback.metadata["paper_title"] == "Unselected Metadata-Only Multimodal Benchmark"
    assert fallback.metadata["source_family"] == "paper_metadata_fallback"

    record = classify_artifact(fallback, taxonomy, run_id="testrun")
    assert record.skipped_reason is None
    assert set(record.topic_ids) == {family.id for family in taxonomy}
    assert record.quality_metadata["paper_id"] == "XYZ789"


# ---------------------------------------------------------------------------
# 4. Owner command writes receipt
# ---------------------------------------------------------------------------


def test_owner_command_writes_receipt(tmp_workspace: Path):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    receipt = run_exhaustive_topic_extraction(
        tmp_workspace, run_id="testrun", force_rebuild=False
    )

    assert receipt["status"] == "succeeded", receipt
    assert receipt["stage_id"] == "exhaustive-topic-extraction"
    details = receipt["details"]
    assert details["media_prep_status"] == "waived"
    assert (
        details["paper_reference_scope"]["reference_extraction_status"]
        == "deferred"
    )
    coverage = details["coverage"]
    assert coverage["classes"]["not_attempted"] == 0

    receipt_path = (
        tmp_workspace
        / "storage"
        / "workflow-runs"
        / "testrun"
        / "14A-exhaustive-topic-extraction-receipt.json"
    )
    assert receipt_path.exists()


# ---------------------------------------------------------------------------
# 5. Artifact record schema
# ---------------------------------------------------------------------------


def test_artifact_record_schema():
    from conference_pipeline.extractions.records import ExtractionRecord

    record = ExtractionRecord(
        artifact_id="x",
        topic_ids=("02_model_architectures",),
        source_artifact_id="x",
        source_artifact_path="storage/some/path.txt",
        source_artifact_digest="d",
        evidence={"snippet": "hello", "span": {"start": 0, "end": 5}},
        extraction_method="heuristic",
        confidence=0.95,
        quality_metadata={"input_kind": "transcript", "text_length": 100},
        skipped_reason=None,
        provider_metadata=None,
        run_id="testrun",
        generated_at="2026-04-27T00:00:00Z",
    )
    payload = record.as_payload()
    required = {
        "schema_version",
        "artifact_id",
        "topic_ids",
        "source_artifact_id",
        "source_artifact_path",
        "source_artifact_digest",
        "evidence",
        "extraction_method",
        "confidence",
        "quality_metadata",
        "skipped_reason",
        "provider_metadata",
        "run_id",
        "generated_at",
    }
    assert required.issubset(set(payload.keys())), required - set(payload.keys())
    # Round-trip through json to confirm JSON-serializability.
    json.dumps(payload)


# ---------------------------------------------------------------------------
# 6. By-topic index aggregator
# ---------------------------------------------------------------------------


def test_by_topic_index_aggregates(tmp_path: Path):
    from conference_pipeline.extractions.records import (
        BY_TOPIC_SCHEMA_VERSION,
        aggregate_by_topic,
    )

    by_artifact = tmp_path / "by-artifact"
    by_topic = tmp_path / "by-topic"
    by_artifact.mkdir(parents=True)

    def _write(aid: str, topic_ids: list[str]) -> None:
        _write_json(
            by_artifact / f"{aid}.json",
            {
                "schema_version": 1,
                "artifact_id": aid,
                "topic_ids": topic_ids,
                "source_artifact_id": aid,
                "source_artifact_path": f"storage/{aid}.txt",
                "source_artifact_digest": f"digest-{aid}",
                "evidence": {"snippet": "x", "span": {"start": 0, "end": 1}},
                "extraction_method": "heuristic",
                "confidence": 0.9,
                "quality_metadata": {},
                "skipped_reason": None,
                "provider_metadata": None,
                "run_id": "testrun",
                "generated_at": "2026-04-27T00:00:00Z",
            },
        )

    _write("a", ["02_model_architectures", "06_multimodal_systems"])
    _write("b", ["02_model_architectures"])
    _write("c", ["06_multimodal_systems"])

    counts = aggregate_by_topic(by_artifact, by_topic)
    assert counts["02_model_architectures"] == 2
    assert counts["06_multimodal_systems"] == 2

    arch_payload = json.loads(
        (by_topic / "02_model_architectures.json").read_text(encoding="utf-8")
    )
    assert arch_payload["schema_version"] == BY_TOPIC_SCHEMA_VERSION
    arch_aids = {item["artifact_id"] for item in arch_payload["items"]}
    assert arch_aids == {"a", "b"}


# ---------------------------------------------------------------------------
# 7. Skipped-unusable records have a reason
# ---------------------------------------------------------------------------


def test_skipped_unusable_records_have_reason(tmp_workspace: Path):
    from conference_pipeline.extractions.classifier import classify_artifact
    from conference_pipeline.extractions.walker import ExtractionInput
    from conference_pipeline.iclr2025_live import load_topic_families

    taxonomy = load_topic_families(tmp_workspace)
    empty_input = ExtractionInput(
        artifact_id="media:empty",
        kind="transcript",
        text_path=None,
        digest="d-empty",
        status="transcribed",
        source_path="cache/media/empty.m4a",
        topic_hint=None,
        text_content="",
    )
    record = classify_artifact(empty_input, taxonomy, run_id="testrun")
    assert record.topic_ids == ()
    assert record.skipped_reason == "empty_or_too_short"


# ---------------------------------------------------------------------------
# 8. Coverage report classes
# ---------------------------------------------------------------------------


def test_coverage_report_classes(tmp_workspace: Path):
    from conference_pipeline.extractions.coverage import build_coverage
    from conference_pipeline.extractions.walker import WalkerResult, walk_and_extract

    result = walk_and_extract(tmp_workspace, run_id="testrun", force_rebuild=False)
    coverage = build_coverage(tmp_workspace, "testrun", result)
    classes = coverage["classes"]
    # The fixture deliberately stages 1 page-image record (no text) → skipped_unusable.
    assert classes["covered"] >= 1
    assert classes["not_attempted"] == 0
    assert (
        classes["covered"]
        + classes["skipped_unusable"]
        + classes["unclassifiable"]
        + classes["blocked"]
        == result.expected_count
    )


# ---------------------------------------------------------------------------
# 9. Receipt records media_prep_status (MEDIA-PREP-01)
# ---------------------------------------------------------------------------


def test_receipt_records_media_prep_status(tmp_workspace: Path):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    details = receipt["details"]
    assert details["media_prep_status"] == "waived"
    assert "media_prep_rationale" in details
    assert (
        "transcription-ocr" in details["media_prep_evidence_path"]
        or "stage-13" in details["media_prep_evidence_path"]
    )
    # Same-run receipt was used directly; no cross-run resolution happened.
    assert details["media_prep_evidence_resolved_run_id"] is None


# ---------------------------------------------------------------------------
# 9A. Cross-run transcription-ocr admission (sibling amendment run-id)
# ---------------------------------------------------------------------------


def _remove_same_run_transcription_receipt(workspace: Path, run_id: str) -> None:
    for filename in ("stage-13-transcription-ocr.json", "stage-05-transcription-ocr.json"):
        path = workspace / "storage" / "workflow-runs" / run_id / filename
        if path.exists():
            path.unlink()


def test_gate_falls_back_to_cross_run_succeeded_transcription_ocr_receipt(
    tmp_workspace: Path,
):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    # No transcription-ocr receipt under the invoked run_id itself...
    _remove_same_run_transcription_receipt(tmp_workspace, "testrun")
    # ...but a genuinely succeeded one exists under a sibling amendment run,
    # using the CVPR production-runner's stage-05 filename/schema convention.
    _write_json(
        tmp_workspace
        / "storage"
        / "workflow-runs"
        / "testrun-r5-event-media"
        / "stage-05-transcription-ocr.json",
        {
            "schema_version": 1,
            "run_id": "testrun-r5-event-media",
            "stage_id": "transcription-ocr",
            "stage_order": 5,
            "mode": "live",
            "status": "succeeded",
            "blockers": [],
        },
    )

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    assert receipt["status"] == "succeeded"
    details = receipt["details"]
    assert details["media_prep_evidence_resolved_run_id"] == "testrun-r5-event-media"
    assert "testrun-r5-event-media" in details["media_prep_evidence_path"]


def test_gate_ignores_blocked_sibling_receipt_and_still_fails_closed(
    tmp_workspace: Path,
):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    _remove_same_run_transcription_receipt(tmp_workspace, "testrun")
    # A sibling run's canonical receipt is blocked (not succeeded) -- this
    # must never be admitted, cross-run or otherwise.
    _write_json(
        tmp_workspace
        / "storage"
        / "workflow-runs"
        / "testrun-r5-event-media"
        / "stage-05-transcription-ocr.json",
        {
            "schema_version": 1,
            "run_id": "testrun-r5-event-media",
            "stage_id": "transcription-ocr",
            "stage_order": 5,
            "mode": "live",
            "status": "blocked",
            "blockers": [{"code": "transcription_ocr_adapter_result_invalid"}],
        },
    )

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    assert receipt["status"] == "blocked"
    assert any(
        blocker.get("code") == "transcription_ocr_not_succeeded"
        for blocker in receipt["blockers"]
    )


def test_gate_fails_closed_on_ambiguous_cross_run_candidates(tmp_workspace: Path):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    _remove_same_run_transcription_receipt(tmp_workspace, "testrun")
    succeeded_payload = {
        "schema_version": 1,
        "stage_id": "transcription-ocr",
        "stage_order": 5,
        "mode": "live",
        "status": "succeeded",
        "blockers": [],
    }
    for sibling_run_id in ("testrun-r5-event-media", "testrun-r6-expanded-media"):
        _write_json(
            tmp_workspace
            / "storage"
            / "workflow-runs"
            / sibling_run_id
            / "stage-05-transcription-ocr.json",
            {**succeeded_payload, "run_id": sibling_run_id},
        )

    # Two equally-valid succeeded candidates anywhere in the workspace must
    # never auto-resolve -- fail closed with the exact-run-id blocker.
    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    assert receipt["status"] == "blocked"
    assert any(
        blocker.get("code") == "transcription_ocr_not_succeeded"
        for blocker in receipt["blockers"]
    )


# ---------------------------------------------------------------------------
# 10. Receipt records paper_reference_scope (PAPER-REF-01)
# ---------------------------------------------------------------------------


def test_receipt_records_paper_ref_scope(tmp_workspace: Path):
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    paper_scope = receipt["details"]["paper_reference_scope"]
    assert paper_scope["reference_extraction_status"] == "deferred"
    assert "deferred_to" in paper_scope
    assert isinstance(paper_scope.get("selected_paper_ids"), list)
    assert "ABC123" in paper_scope["selected_paper_ids"]


# ---------------------------------------------------------------------------
# 11. Classifier deterministic for known text
# ---------------------------------------------------------------------------


def test_classifier_deterministic_for_known_text(tmp_workspace: Path):
    from conference_pipeline.extractions.classifier import classify_text_span
    from conference_pipeline.iclr2025_live import load_topic_families

    taxonomy = load_topic_families(tmp_workspace)
    text = (
        "We propose a transformer architecture for multimodal vision and "
        "language reasoning. Attention layers and modality-specific encoders "
        "fuse image and text features." * 6
    )
    matches = classify_text_span(text, taxonomy)
    topic_ids = {m.topic_id for m in matches}
    assert "02_model_architectures" in topic_ids


# ---------------------------------------------------------------------------
# 12. LLM-assist disabled by default
# ---------------------------------------------------------------------------


def test_llm_assist_disabled_by_default(monkeypatch):
    from conference_pipeline.extractions.classifier import classify_with_llm_assist
    from conference_pipeline.iclr2025_live import load_topic_families

    monkeypatch.delenv("EXTRACTION_LLM_ASSIST", raising=False)
    workspace = Path("conferences/iclr2025")
    if workspace.exists():
        taxonomy = load_topic_families(workspace)
    else:
        taxonomy = []
    matches, meta = classify_with_llm_assist("any text", taxonomy)
    assert matches == []
    assert meta == {"skipped": "llm_assist_disabled"}


def test_llm_assist_requires_local_endpoint(monkeypatch):
    from conference_pipeline.extractions.classifier import classify_with_llm_assist

    monkeypatch.setenv("EXTRACTION_LLM_ASSIST", "1")
    monkeypatch.delenv("EXTRACTION_LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACTION_LLM_BASE_URL", raising=False)

    matches, meta = classify_with_llm_assist("ambiguous text", [])

    assert matches == []
    assert meta["skipped"] == "local_endpoint_missing"
    assert meta["provider"] == "local-openai-compatible"
    assert meta["estimated_paid_cost_usd"] == 0.0
    assert meta["paid_cost_cap_usd"] == 0.0
    assert meta["confidence_path"] == "heuristic_none_then_local_llm_assist"


def test_llm_assist_rejects_non_local_endpoint(monkeypatch):
    from conference_pipeline.extractions.classifier import classify_with_llm_assist

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "https://api.openai.com/v1?api_key=secret")

    matches, meta = classify_with_llm_assist("ambiguous text", [])

    assert matches == []
    assert meta["skipped"] == "non_local_endpoint_rejected"
    assert meta["redacted_base_url_identity"] == "https://api.openai.com/v1"
    assert meta["provider"] == "local-openai-compatible"


def test_llm_assist_rejects_malformed_loopback_port(monkeypatch):
    from conference_pipeline.extractions.classifier import classify_with_llm_assist

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "http://localhost:bad/v1")

    matches, meta = classify_with_llm_assist("ambiguous text", [])

    assert matches == []
    assert meta["skipped"] == "non_local_endpoint_rejected"
    assert meta["redacted_base_url_identity"] == "invalid://unknown"


def test_llm_assist_failed_provider_call_keeps_receipt_metadata(monkeypatch):
    from types import SimpleNamespace

    from conference_pipeline.extractions import backends
    from conference_pipeline.extractions.backends import BackendKind
    from conference_pipeline.extractions.classifier import (
        classify_with_llm_assist,
        reset_llm_assist_counter,
    )

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    reset_llm_assist_counter()

    provider_receipt = SimpleNamespace(
        receipt_path="storage/provider-receipts/failed.json",
        input_tokens=11,
        output_tokens=3,
        response_id="local-failed-response",
    )

    class FailingBackend:
        kind = BackendKind.HTTP
        endpoint_label = "ollama-local"
        model = "qwen3-8b"
        base_url = "http://localhost:11434/v1"

        def chat_completion(self, **kwargs):
            exc = RuntimeError("model load failed")
            exc.provider_receipt = provider_receipt
            raise exc

    monkeypatch.setattr(
        backends,
        "select_local_llm_backend",
        lambda model=None: FailingBackend(),
    )

    matches, meta = classify_with_llm_assist("ambiguous text", [])

    assert matches == []
    assert meta["error"] == "llm_assist_failed"
    assert meta["provider_receipt_path"] == "storage/provider-receipts/failed.json"
    assert meta["input_tokens"] == 11
    assert meta["output_tokens"] == 3
    assert meta["request_id"] == "local-failed-response"


def test_llm_assist_parses_wrapped_qwen_json(monkeypatch, tmp_workspace: Path):
    from types import SimpleNamespace

    from conference_pipeline.extractions import backends
    from conference_pipeline.extractions.backends import BackendKind
    from conference_pipeline.extractions.classifier import (
        classify_with_llm_assist,
        reset_llm_assist_counter,
    )
    from conference_pipeline.iclr2025_live import load_topic_families

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    reset_llm_assist_counter()

    provider_receipt = SimpleNamespace(
        receipt_path="storage/provider-receipts/qwen.json",
        input_tokens=21,
        output_tokens=9,
        response_id="local-qwen-response",
    )

    class WrappedJSONBackend:
        kind = BackendKind.MLX
        endpoint_label = "mlx://qwen"
        model = "qwen"

        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                id="local-qwen-response",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "<think>classified locally</think>\n"
                                "```json\n"
                                '{"topic_ids":["06_multimodal_systems"]}\n'
                                "```"
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=21, completion_tokens=9),
                receipt=provider_receipt,
            )

    monkeypatch.setattr(
        backends,
        "select_local_llm_backend",
        lambda model=None: WrappedJSONBackend(),
    )

    matches, meta = classify_with_llm_assist(
        "ambiguous local-model text",
        load_topic_families(tmp_workspace),
    )

    assert [match.topic_id for match in matches] == ["06_multimodal_systems"]
    assert meta["status"] == "succeeded"
    assert meta["provider_receipt_path"] == "storage/provider-receipts/qwen.json"


# ---------------------------------------------------------------------------
# Plan 14A-02: walker resume + rebuild + digest-change semantics
# ---------------------------------------------------------------------------


def test_resume_skip_when_digest_match(tmp_workspace: Path):
    """Re-running without --rebuild does not rewrite per-artifact files."""
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    by_artifact = (
        tmp_workspace / "storage" / "extractions" / "testrun" / "by-artifact"
    )
    mtimes = {p.name: p.stat().st_mtime for p in by_artifact.glob("*.json")}
    assert mtimes, "first run must produce records"

    # Sleep a tick so a write would change mtime if it happened.
    import time

    time.sleep(0.05)

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    new_mtimes = {p.name: p.stat().st_mtime for p in by_artifact.glob("*.json")}
    # Every record file should have an unchanged mtime.
    for name, mtime in mtimes.items():
        assert new_mtimes.get(name) == mtime, f"{name} was rewritten on resume"
    resume_stats = receipt["details"]["resume_stats"]
    assert resume_stats["resumed"] >= len(mtimes)
    assert resume_stats["extracted"] == 0


def test_rebuild_forces_reextraction(tmp_workspace: Path):
    """force_rebuild=True rewrites every per-artifact file."""
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    by_artifact = (
        tmp_workspace / "storage" / "extractions" / "testrun" / "by-artifact"
    )
    initial_count = len(list(by_artifact.glob("*.json")))
    assert initial_count > 0
    receipt = run_exhaustive_topic_extraction(
        tmp_workspace, run_id="testrun", force_rebuild=True
    )
    resume_stats = receipt["details"]["resume_stats"]
    assert resume_stats["extracted"] == initial_count
    assert resume_stats["resumed"] == 0


def test_digest_change_rewrites_record(tmp_workspace: Path):
    """Mutating an upstream digest forces the record to be rewritten."""
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    inv_path = tmp_workspace / "storage" / "corpus" / "v15_corpus_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    target = next(
        a for a in inv["artifacts"] if a["artifact_id"] == "media:oral-1:transcript"
    )
    target["provenance_digest"] = "digest-oral-1-mutated"
    inv_path.write_text(json.dumps(inv), encoding="utf-8")

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    resume_stats = receipt["details"]["resume_stats"]
    # The digest changed only for one row.
    assert resume_stats["rewritten_due_to_digest_change"] >= 1


def test_resume_does_not_inflate_rewrite_counter_for_empty_digest_rows(
    tmp_workspace: Path,
):
    """Idempotent resume must not increment rewritten_due_to_digest_change
    for rows that have no upstream digest available.

    Regression test for WR-01: previously, ``existing_digest_matches`` short-
    circuited to ``False`` for any empty input digest, which caused the
    walker to unlink-and-rewrite the record on every resume run.
    """
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    # Mutate one corpus row to drop both digest fields so the walker yields
    # an empty input digest for that row.
    inv_path = tmp_workspace / "storage" / "corpus" / "v15_corpus_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    target = next(
        a for a in inv["artifacts"] if a["artifact_id"] == "media:poster-1:pdf"
    )
    target.pop("provenance_digest", None)
    target.pop("text_extraction_sha256", None)
    inv_path.write_text(json.dumps(inv), encoding="utf-8")

    run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    resume_stats = receipt["details"]["resume_stats"]
    # Empty-digest rows must not be counted as rewritten on a no-op resume.
    assert resume_stats["rewritten_due_to_digest_change"] == 0, resume_stats


def test_walker_handles_blocked_upstream_artifact(tmp_workspace: Path):
    """A row whose upstream status is not in the completed set is recorded
    with skipped_reason=upstream_blocked and counted as `blocked`."""
    inv_path = tmp_workspace / "storage" / "corpus" / "v15_corpus_inventory.json"
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    inv["artifacts"].append(
        {
            "artifact_id": "media:blocked-1:audio",
            "source_family": "iclr-virtual-media",
            "processing_status": "failed",
            "transcript_path": None,
            "ocr_path": None,
            "local_path": "cache/media/blocked-1/audio.m4a",
            "topic_id": None,
            "provenance_digest": "digest-blocked-1",
        }
    )
    inv_path.write_text(json.dumps(inv), encoding="utf-8")

    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    receipt = run_exhaustive_topic_extraction(tmp_workspace, run_id="testrun")
    coverage = receipt["details"]["coverage"]
    assert coverage["classes"]["blocked"] >= 1, coverage
    record_path = (
        tmp_workspace
        / "storage"
        / "extractions"
        / "testrun"
        / "by-artifact"
        / "media_blocked-1_audio.json"
    )
    assert record_path.exists()
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["skipped_reason"] == "upstream_blocked"


def test_exhaustive_topic_extraction_blocks_on_failed_local_qwen_attempt(
    tmp_path: Path,
    monkeypatch,
):
    from types import SimpleNamespace

    from conference_pipeline.extractions import backends
    from conference_pipeline.extractions.backends import BackendKind
    from conference_pipeline.extractions.classifier import reset_llm_assist_counter
    from conference_pipeline.iclr2025_live import run_exhaustive_topic_extraction

    workspace = _build_workspace(tmp_path, transcript_text=("ambient phrasing " * 40))
    _write_text(
        workspace / "config.yaml",
        "\n".join(
            [
                "workflow:",
                "  extraction:",
                "    exhaustive_topic_extraction:",
                "      provider: local",
                "      provider_mode: mlx",
                "      backend: mlx",
                "      model: mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit",
                "      assist_enabled: true",
                "",
            ]
        ),
    )
    reset_llm_assist_counter()

    provider_receipt = SimpleNamespace(
        receipt_path="storage/provider-receipts/qwen-failed.json",
        input_tokens=17,
        output_tokens=2,
        response_id="local-qwen-failed",
    )

    class BadJSONBackend:
        kind = BackendKind.MLX
        endpoint_label = "mlx://qwen"
        model = "qwen"

        def chat_completion(self, **kwargs):
            return SimpleNamespace(
                id="local-qwen-failed",
                choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
                usage=SimpleNamespace(prompt_tokens=17, completion_tokens=2),
                receipt=provider_receipt,
            )

    monkeypatch.setattr(
        backends,
        "select_local_llm_backend",
        lambda model=None: BadJSONBackend(),
    )

    receipt = run_exhaustive_topic_extraction(
        workspace,
        run_id="testrun",
        force_rebuild=True,
    )

    assert receipt["status"] == "blocked"
    assert any(b["code"] == "local_qwen_assist_failed" for b in receipt["blockers"])
    token_usage = receipt["details"]["token_usage"]
    assert token_usage["local_llm_attempt_count"] == 1
    assert token_usage["local_llm_failed_count"] == 1
    assert token_usage["local_llm_provider_receipts"] == [
        "storage/provider-receipts/qwen-failed.json"
    ]


# ---------------------------------------------------------------------------
# 13. Readiness preflight — LOCAL-LLM-04, EXTRACT-POLICY-02
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared helpers for readiness and quality sample tests
# ---------------------------------------------------------------------------


def _set_loopback_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from conference_pipeline.extractions import backends

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_MODEL", "qwen3-8b")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL", "ollama-local")
    monkeypatch.setattr(backends, "_try_import_mlx", lambda: False)


class _MockModelsOK:
    status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"models": [{"id": "qwen3-8b"}]}


class _FakeCompletion:
    pass


class _FakeCompletions:
    def create(self, **kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion()


class _FakeChat:
    completions = _FakeCompletions()


class _FakeOpenAIClient:
    chat = _FakeChat()


class _FailCompletions:
    def create(self, **kwargs: Any) -> None:
        raise RuntimeError("model load failed")


class _FailOpenAIClient:
    chat = type("C", (), {"completions": _FailCompletions()})()


# ---------------------------------------------------------------------------
# Readiness failure tests
# ---------------------------------------------------------------------------


def test_readiness_fails_when_endpoint_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No EXTRACTION_LOCAL_LLM_BASE_URL → raises with check_failed='reachability'."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.delenv("EXTRACTION_LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACTION_LLM_BASE_URL", raising=False)

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "reachability"
    assert isinstance(exc_info.value.endpoint_label, str)
    assert len(exc_info.value.endpoint_label) > 0


def test_readiness_fails_when_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-loopback URL → raises with check_failed='reachability' before any HTTP call."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL", "ollama-local")

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "reachability"
    assert exc_info.value.endpoint_label == "ollama-local"


@pytest.mark.legacy_http_mock
def test_readiness_fails_when_health_endpoint_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx.get raises ConnectError → raises LocalLLMNotReadyError(check_failed='reachability')."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    _set_loopback_env(monkeypatch)

    def _raise_connect(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _raise_connect)

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "reachability"
    assert exc_info.value.endpoint_label == "ollama-local"


@pytest.mark.legacy_http_mock
def test_readiness_fails_when_model_not_listed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Server returns 200 but model not in list → raises with check_failed='model_not_listed'."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    _set_loopback_env(monkeypatch)

    class _MockModelsOtherModel:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"models": [{"id": "other-model"}]}

    monkeypatch.setattr(httpx, "get", lambda url, **kw: _MockModelsOtherModel())

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "model_not_listed"
    assert exc_info.value.endpoint_label == "ollama-local"
    assert len(exc_info.value.fallback_options) > 0


@pytest.mark.legacy_http_mock
def test_readiness_fails_when_sample_inference_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx.get succeeds but openai client raises → check_failed='inference'."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    _set_loopback_env(monkeypatch)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _MockModelsOK())
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FailOpenAIClient())

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "inference"
    assert exc_info.value.endpoint_label == "ollama-local"


@pytest.mark.legacy_http_mock
def test_readiness_success_writes_preflight_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """All checks pass → writes LocalLLMReceipt(preflight_mode=True) to disk."""
    from conference_pipeline.extractions.classifier import (
        check_local_llm_readiness,
    )
    from conference_pipeline.provider_receipts import (
        LOCAL_LLM_RECEIPT_MARKER,
        scan_local_llm_receipts,
    )

    _set_loopback_env(monkeypatch)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _MockModelsOK())
    monkeypatch.setattr("openai.OpenAI", lambda **kw: _FakeOpenAIClient())

    receipt = check_local_llm_readiness(workspace=tmp_path, run_id="test-run", stage_id="preflight")

    assert receipt.preflight_mode is True
    assert receipt.paid_cloud_cost == 0.0
    assert receipt.endpoint_label == "ollama-local"
    assert receipt.receipt_type == LOCAL_LLM_RECEIPT_MARKER

    # Verify written to disk
    found = scan_local_llm_receipts(tmp_path)
    assert len(found) == 1
    payload = found[0]
    assert payload["preflight_mode"] is True
    assert "localhost" not in str(payload)  # URL never stored


def test_readiness_error_propagates_to_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """LocalLLMNotReadyError from check_local_llm_readiness() reaches the caller.

    Validates EXTRACT-POLICY-02: the runtime gate fails clearly without being
    swallowed inside the classification layer.
    """
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        check_local_llm_readiness,
    )

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.delenv("EXTRACTION_LOCAL_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("EXTRACTION_LLM_BASE_URL", raising=False)

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        check_local_llm_readiness()
    assert exc_info.value.check_failed == "reachability"
    assert isinstance(exc_info.value.endpoint_label, str)
    assert len(exc_info.value.endpoint_label) > 0


# ---------------------------------------------------------------------------
# 14. Quality sample — LOCAL-LLM-05
# ---------------------------------------------------------------------------


def test_quality_sample_raises_on_non_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-loopback URL → run_quality_sample raises LocalLLMNotReadyError immediately."""
    from conference_pipeline.extractions.classifier import (
        LocalLLMNotReadyError,
        run_quality_sample,
    )

    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ASSIST", "1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL", "cloud-endpoint")

    with pytest.raises(LocalLLMNotReadyError) as exc_info:
        run_quality_sample([], [], workspace=None)
    assert exc_info.value.check_failed == "reachability"


@pytest.mark.legacy_http_mock
def test_quality_sample_writes_receipt_and_returns_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_workspace: Path
) -> None:
    """Loopback URL, sample records → quality_sample_mode=True dict + receipt on disk."""
    from conference_pipeline.extractions.classifier import run_quality_sample
    from conference_pipeline.extractions.walker import iterate_extraction_inputs
    from conference_pipeline.iclr2025_live import load_topic_families
    from conference_pipeline.provider_receipts import scan_local_llm_receipts

    _set_loopback_env(monkeypatch)

    inputs = list(iterate_extraction_inputs(tmp_workspace))
    taxonomy = load_topic_families(tmp_workspace)

    result = run_quality_sample(
        inputs,
        taxonomy,
        workspace=tmp_workspace,
        run_id="test-run",
        stage_id="quality-sample",
        sample_size=3,
    )

    assert result["quality_sample_mode"] is True
    assert "confidence_mean" in result
    assert "confidence_p50" in result
    assert "confidence_p90" in result
    assert "failure_classes" in result
    assert result["sample_size"] <= 3
    assert "effective_thresholds" in result
    assert "recommendation" in result

    found = scan_local_llm_receipts(tmp_workspace)
    quality_receipts = [r for r in found if r.get("quality_sample_mode") is True]
    assert len(quality_receipts) >= 1
    payload = quality_receipts[0]
    assert payload["paid_cloud_cost"] == 0.0
    assert payload["effective_thresholds"]["min_mean_confidence"] == pytest.approx(0.65)
    assert "localhost" not in str(payload)


def test_quality_sample_counts_only_provider_backed_local_llm_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> None:
    from conference_pipeline.extractions import classifier
    from conference_pipeline.extractions.records import ExtractionRecord
    from conference_pipeline.provider_receipts import scan_local_llm_receipts

    _set_loopback_env(monkeypatch)

    def make_record(
        artifact_id: str,
        confidence: float,
        provider_metadata: dict[str, object] | None,
    ) -> ExtractionRecord:
        return ExtractionRecord(
            artifact_id=artifact_id,
            topic_ids=("02_model_architectures",),
            source_artifact_id=artifact_id,
            source_artifact_path=f"storage/{artifact_id}.txt",
            source_artifact_digest=f"digest-{artifact_id}",
            evidence={"snippet": "example", "span": {"start": 0, "end": 7}},
            extraction_method="llm_assisted" if provider_metadata else "heuristic",
            confidence=confidence,
            quality_metadata={"input_kind": "transcript", "text_length": 500},
            skipped_reason=None,
            provider_metadata=provider_metadata,
            run_id="quality-accounting",
            generated_at="2026-04-30T00:00:00Z",
        )

    returned_records = iter(
        [
            make_record("heuristic-only", 0.95, None),
            make_record(
                "provider-backed",
                0.9,
                {
                    "provider_receipt_path": "storage/provider-receipts/ok.json",
                    "input_tokens": 8,
                    "output_tokens": 5,
                },
            ),
            make_record(
                "metadata-without-provider-call",
                0.85,
                {"input_tokens": 99, "output_tokens": 1},
            ),
        ]
    )
    monkeypatch.setattr(
        classifier,
        "classify_artifact",
        lambda *args, **kwargs: next(returned_records),
    )

    result = classifier.run_quality_sample(
        [object(), object(), object()],
        [],
        workspace=tmp_workspace,
        run_id="quality-accounting",
        stage_id="quality-sample",
        sample_size=3,
    )

    assert result["request_count"] == 1
    assert result["token_count"] == 13

    quality_receipts = [
        receipt
        for receipt in scan_local_llm_receipts(tmp_workspace)
        if receipt.get("run_id") == "quality-accounting"
    ]
    assert len(quality_receipts) == 1
    assert quality_receipts[0]["request_count"] == 1
    assert quality_receipts[0]["token_count"] == 13


def test_quality_sample_escalates_to_cloud_and_writes_escalation_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> None:
    from conference_pipeline.extractions import classifier
    from conference_pipeline.extractions.records import ExtractionRecord
    from conference_pipeline.provider_receipts import (
        record_provider_call,
        scan_escalation_receipts,
    )

    _set_loopback_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_MODEL", "gpt-test-cloud")
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_RETRY_DELAY_SECONDS", "0")

    def fake_classify(
        record: object,
        *_args: object,
        **_kwargs: object,
    ) -> ExtractionRecord:
        artifact_id = getattr(record, "artifact_id", "sample")
        return ExtractionRecord(
            artifact_id=str(artifact_id),
            topic_ids=(),
            source_artifact_id=str(artifact_id),
            source_artifact_path=f"storage/{artifact_id}.txt",
            source_artifact_digest=f"digest-{artifact_id}",
            evidence={"snippet": "low confidence", "span": {"start": 0, "end": 14}},
            extraction_method="heuristic",
            confidence=0.2,
            quality_metadata={"input_kind": "transcript", "text_length": 500},
            skipped_reason="unclassifiable",
            provider_metadata=None,
            run_id="cloud-escalation-ok",
            generated_at="2026-04-30T00:00:00Z",
        )

    def fake_cloud_call(*, model: str, prompt: str) -> object:
        assert model == "gpt-test-cloud"
        assert "paper-low-1" in prompt

        def invoke() -> object:
            return SimpleNamespace(
                id="cloud-response-ok",
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
                choices=[],
            )

        return record_provider_call(
            invoke,
            provider="openai",
            model=model,
            operation="chat.completions",
            request_kwargs={"model": model, "messages": "redacted-test-prompt"},
        )

    monkeypatch.setattr(classifier, "classify_artifact", fake_classify)
    monkeypatch.setattr(classifier, "_invoke_openai_cloud_escalation", fake_cloud_call)

    result = classifier.run_quality_sample(
        [
            SimpleNamespace(artifact_id="paper-low-1"),
            SimpleNamespace(artifact_id="paper-low-2"),
        ],
        [],
        workspace=tmp_workspace,
        run_id="cloud-escalation-ok",
        stage_id="quality-sample",
        sample_size=2,
        threshold_overrides={
            "cloud_escalation_trigger": 0.9,
            "min_mean_confidence": 0.1,
        },
    )

    assert result["recommendation"]["should_escalate"] is True
    assert result["recommendation"]["cloud_escalation_status"] == "succeeded"
    assert result["cloud_escalation"]["provider"] == "openai"
    assert result["cloud_escalation"]["model"] == "gpt-test-cloud"

    receipts = scan_escalation_receipts(tmp_workspace)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["wrapped_receipt"]["provider"] == "openai"
    assert receipt["wrapped_receipt"]["status"] == "succeeded"
    assert receipt["fail_closed_state"] is False
    assert receipt["affected_record_ids"] == ["paper-low-1", "paper-low-2"]


def test_quality_sample_does_not_escalate_when_confidence_is_adequate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> None:
    from conference_pipeline.extractions import classifier
    from conference_pipeline.extractions.records import ExtractionRecord
    from conference_pipeline.provider_receipts import scan_escalation_receipts

    _set_loopback_env(monkeypatch)

    def fake_classify(record: object, *_args: object, **_kwargs: object) -> ExtractionRecord:
        artifact_id = getattr(record, "artifact_id", "sample")
        return ExtractionRecord(
            artifact_id=str(artifact_id),
            topic_ids=("02_model_architectures",),
            source_artifact_id=str(artifact_id),
            source_artifact_path=f"storage/{artifact_id}.txt",
            source_artifact_digest=f"digest-{artifact_id}",
            evidence={
                "snippet": "clear topic evidence",
                "span": {"start": 0, "end": 20},
            },
            extraction_method="heuristic",
            confidence=0.95,
            quality_metadata={"input_kind": "transcript", "text_length": 500},
            skipped_reason=None,
            provider_metadata=None,
            run_id="cloud-escalation-not-needed",
            generated_at="2026-04-30T00:00:00Z",
        )

    def fail_cloud_call(**_kwargs: object) -> object:
        raise AssertionError("cloud escalation should not run for adequate confidence")

    monkeypatch.setattr(classifier, "classify_artifact", fake_classify)
    monkeypatch.setattr(classifier, "_invoke_openai_cloud_escalation", fail_cloud_call)

    result = classifier.run_quality_sample(
        [
            SimpleNamespace(artifact_id="paper-clear-1"),
            SimpleNamespace(artifact_id="paper-clear-2"),
        ],
        [],
        workspace=tmp_workspace,
        run_id="cloud-escalation-not-needed",
        stage_id="quality-sample",
        sample_size=2,
        threshold_overrides={
            "cloud_escalation_trigger": 0.5,
            "min_mean_confidence": 0.9,
        },
    )

    assert result["recommendation"]["should_proceed"] is True
    assert result["recommendation"]["should_escalate"] is False
    assert "cloud_escalation" not in result
    assert (
        result["cloud_escalation_threshold_recommendation"]
        == "current_threshold_adequate"
    )
    assert scan_escalation_receipts(tmp_workspace) == []


def test_quality_sample_threshold_failure_raises_after_cloud_escalation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> None:
    from conference_pipeline.extractions import (
        QualitySampleThresholdError,
        classifier,
    )
    from conference_pipeline.extractions.records import ExtractionRecord
    from conference_pipeline.provider_receipts import (
        record_provider_call,
        scan_escalation_receipts,
    )

    _set_loopback_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_MODEL", "gpt-test-cloud")
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_RETRY_DELAY_SECONDS", "0")

    def fake_classify(record: object, *_args: object, **_kwargs: object) -> ExtractionRecord:
        artifact_id = getattr(record, "artifact_id", "sample")
        return ExtractionRecord(
            artifact_id=str(artifact_id),
            topic_ids=(),
            source_artifact_id=str(artifact_id),
            source_artifact_path=f"storage/{artifact_id}.txt",
            source_artifact_digest=f"digest-{artifact_id}",
            evidence={"snippet": "low confidence", "span": {"start": 0, "end": 14}},
            extraction_method="heuristic",
            confidence=0.2,
            quality_metadata={"input_kind": "transcript", "text_length": 500},
            skipped_reason="unclassifiable",
            provider_metadata=None,
            run_id="quality-threshold-fails",
            generated_at="2026-04-30T00:00:00Z",
        )

    def fake_cloud_call(*, model: str, prompt: str) -> object:
        del prompt

        def invoke() -> object:
            return SimpleNamespace(
                id="cloud-response-ok",
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=4),
                choices=[],
            )

        return record_provider_call(
            invoke,
            provider="openai",
            model=model,
            operation="chat.completions",
            request_kwargs={"model": model},
        )

    monkeypatch.setattr(classifier, "classify_artifact", fake_classify)
    monkeypatch.setattr(classifier, "_invoke_openai_cloud_escalation", fake_cloud_call)

    with pytest.raises(QualitySampleThresholdError) as exc_info:
        classifier.run_quality_sample(
            [SimpleNamespace(artifact_id="paper-low-1")],
            [],
            workspace=tmp_workspace,
            run_id="quality-threshold-fails",
            stage_id="quality-sample",
            sample_size=1,
            threshold_overrides={
                "cloud_escalation_trigger": 0.9,
                "min_mean_confidence": 0.65,
            },
        )

    assert exc_info.value.recommendation["should_proceed"] is False
    assert exc_info.value.recommendation["cloud_escalation_status"] == "succeeded"
    receipts = scan_escalation_receipts(tmp_workspace)
    assert len(receipts) == 1
    assert receipts[0]["fail_closed_state"] is True


def test_quality_sample_cloud_escalation_failure_retries_then_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_workspace: Path,
) -> None:
    from conference_pipeline.extractions import classifier
    from conference_pipeline.extractions.records import ExtractionRecord
    from conference_pipeline.provider_receipts import (
        record_provider_call,
        scan_escalation_receipts,
    )

    _set_loopback_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_MODEL", "gpt-test-cloud")
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_RETRIES", "2")
    monkeypatch.setenv("EXTRACTION_CLOUD_ESCALATION_RETRY_DELAY_SECONDS", "0")

    def fake_classify(record: object, *_args: object, **_kwargs: object) -> ExtractionRecord:
        artifact_id = getattr(record, "artifact_id", "sample")
        return ExtractionRecord(
            artifact_id=str(artifact_id),
            topic_ids=(),
            source_artifact_id=str(artifact_id),
            source_artifact_path=f"storage/{artifact_id}.txt",
            source_artifact_digest=f"digest-{artifact_id}",
            evidence={"snippet": "low confidence", "span": {"start": 0, "end": 14}},
            extraction_method="heuristic",
            confidence=0.1,
            quality_metadata={"input_kind": "transcript", "text_length": 500},
            skipped_reason="unclassifiable",
            provider_metadata=None,
            run_id="cloud-escalation-fails",
            generated_at="2026-04-30T00:00:00Z",
        )

    attempts = 0

    def fake_cloud_call(*, model: str, prompt: str) -> object:
        del prompt
        nonlocal attempts
        attempts += 1

        def invoke() -> object:
            raise RuntimeError("cloud unavailable")

        return record_provider_call(
            invoke,
            provider="openai",
            model=model,
            operation="chat.completions",
            request_kwargs={"model": model},
        )

    monkeypatch.setattr(classifier, "classify_artifact", fake_classify)
    monkeypatch.setattr(classifier, "_invoke_openai_cloud_escalation", fake_cloud_call)

    with pytest.raises(classifier.CloudEscalationError, match="failed after 2"):
        classifier.run_quality_sample(
            [SimpleNamespace(artifact_id="paper-low-1")],
            [],
            workspace=tmp_workspace,
            run_id="cloud-escalation-fails",
            stage_id="quality-sample",
            sample_size=1,
            threshold_overrides={"cloud_escalation_trigger": 0.9},
        )

    assert attempts == 2
    assert scan_escalation_receipts(tmp_workspace) == []
    failed_receipts = list(
        (
            tmp_workspace
            / "storage"
            / "workflow-runs"
            / "cloud-escalation-fails"
            / "provider-calls"
            / "quality-sample"
        ).glob("*.json")
    )
    assert len(failed_receipts) == 2


# ---------------------------------------------------------------------------
# get_extraction_status() tests (EXTRACT-POLICY-05)
# ---------------------------------------------------------------------------


def test_get_extraction_status_no_receipts(tmp_path: Path) -> None:
    """get_extraction_status() with no receipts on disk returns heuristic mode + blocker."""
    from conference_pipeline.extractions.classifier import get_extraction_status

    result = get_extraction_status(tmp_path)

    assert result["extraction_mode"] == "heuristic", (
        "No receipts → extraction_mode must be 'heuristic'"
    )
    assert result["cloud_escalation_count"] == 0
    assert result["ocr_posture"] == "unknown"
    assert result["receipt_links"] == []
    assert result["local_llm_readiness"] is None
    assert any("EXTRACTION_LOCAL_LLM_BASE_URL" in b for b in result["blockers"]), (
        "Blockers must mention EXTRACTION_LOCAL_LLM_BASE_URL when no receipts found"
    )
    # No URL strings in output (security: endpoint_label only)
    assert "http://" not in str(result)
    assert "https://" not in str(result)


def test_get_extraction_status_with_preflight_receipt(tmp_path: Path) -> None:
    """get_extraction_status() with a preflight receipt returns local_llm mode, no blockers."""
    from conference_pipeline.extractions.classifier import get_extraction_status
    from conference_pipeline.provider_receipts import (
        LOCAL_LLM_RECEIPT_MARKER,
        LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
        LocalLLMReceipt,
        write_local_llm_receipt,
    )

    receipt = LocalLLMReceipt(
        schema_version=LOCAL_LLM_RECEIPT_SCHEMA_VERSION,
        receipt_type=LOCAL_LLM_RECEIPT_MARKER,
        endpoint_label="test-local",
        model="test-model",
        request_count=5,
        token_count=None,
        confidence_path="local_llm",
        skipped_record_count=0,
        paid_cloud_cost=0.0,
        timestamp="2026-04-30T00:00:00Z",
        run_id="test-run",
        stage_id="preflight",
        preflight_mode=True,
        quality_sample_mode=False,
    )
    write_local_llm_receipt(receipt, tmp_path)

    result = get_extraction_status(tmp_path)

    assert result["extraction_mode"] == "local_llm", (
        "Local LLM receipt present → extraction_mode must be 'local_llm'"
    )
    assert result["local_llm_readiness"] is not None, (
        "Preflight receipt found → local_llm_readiness must not be None"
    )
    assert result["cloud_escalation_count"] == 0
    assert result["ocr_posture"] == "unknown"
    assert result["blockers"] == [], (
        "Receipt found → no blockers expected"
    )
    assert len(result["receipt_links"]) >= 1, (
        "Local LLM receipt present → receipt_links must be non-empty"
    )
    # No URL strings in output (security: endpoint_label only)
    assert "http://" not in str(result)
    assert "https://" not in str(result)


def test_llm_full_read_resume_retries_previously_failed_unit(monkeypatch, tmp_workspace: Path):
    """resume=True must not treat a skipped_reason=llm_failed record as done.

    Live regression: 1,614 units failed with a transient backend-selection
    bug on 2026-07-21 (unrelated to these specific units' content), got
    written as skipped_reason=llm_failed records, and every subsequent
    --resume run silently kept treating them as "already processed" because
    only file existence was checked -- the backend fix alone never got a
    chance to actually reprocess them.
    """
    from types import SimpleNamespace

    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.llm_full_read import (
        DEFAULT_MIN_CHARS,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_SEGMENT_SECONDS,
        _unit_output_path,
        iter_units_for_scope,
        run_llm_full_read,
        write_json_atomic,
    )

    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "RETRYME",
                    "forum": "RETRYME",
                    "title": "Retry After Transient Backend Failure",
                    "abstract": (
                        "This abstract discusses retrieval, evaluation, and benchmark "
                        "construction in enough detail to exceed the minimum length. "
                    )
                    * 3,
                    "forum_url": "https://openreview.net/forum?id=RETRYME",
                    "pdf_url": "https://openreview.net/pdf?id=RETRYME",
                }
            ]
        },
    )

    [unit] = list(
        iter_units_for_scope(
            tmp_workspace,
            scope="papers",
            segment_seconds=DEFAULT_SEGMENT_SECONDS,
            overlap_seconds=DEFAULT_OVERLAP_SECONDS,
            min_chars=DEFAULT_MIN_CHARS,
        )
    )
    failed_path = _unit_output_path(tmp_workspace, "testrun", unit.unit_id)
    write_json_atomic(
        failed_path,
        {
            "schema_version": 1,
            "lane": "llm_full_read",
            "unit_id": unit.unit_id,
            "source_artifact_id": unit.source_artifact_id,
            "input_kind": unit.input_kind,
            "topic_ids": [],
            "important": False,
            "confidence": 0.0,
            "skipped_reason": "llm_failed",
            "error": "Connection error.",
            "metadata": unit.metadata,
            "run_id": "testrun",
            "generated_at": "2026-07-21T15:33:01Z",
        },
    )

    class WorkingBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            return SimpleNamespace(
                id="local-retry-response",
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"topic_ids": [], "important": false, "confidence": 0.1}'
                        )
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
            )

    monkeypatch.setattr(
        llm_full_read,
        "select_local_llm_backend",
        lambda model=None: WorkingBackend(),  # noqa: ARG005
    )

    receipt = run_llm_full_read(
        workspace_root=tmp_workspace,
        run_id="testrun",
        scope="papers",
        resume=True,
    )

    checks = receipt["checks"]
    assert checks["skipped_existing_count"] == 0
    assert checks["failed_count"] == 0

    retried_record = json.loads(failed_path.read_text(encoding="utf-8"))
    assert retried_record["skipped_reason"] != "llm_failed"
    assert retried_record["generated_at"] != "2026-07-21T15:33:01Z"


def test_llm_full_read_resume_still_skips_genuinely_completed_unit(
    monkeypatch, tmp_workspace: Path
):
    """A record without skipped_reason=llm_failed is a real result -- resume

    must still skip it and must not call the backend again.
    """
    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.llm_full_read import (
        DEFAULT_MIN_CHARS,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_SEGMENT_SECONDS,
        _unit_output_path,
        iter_units_for_scope,
        run_llm_full_read,
        write_json_atomic,
    )

    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "ALREADYDONE",
                    "forum": "ALREADYDONE",
                    "title": "Already Completed Unit",
                    "abstract": (
                        "This abstract discusses retrieval, evaluation, and benchmark "
                        "construction in enough detail to exceed the minimum length. "
                    )
                    * 3,
                    "forum_url": "https://openreview.net/forum?id=ALREADYDONE",
                    "pdf_url": "https://openreview.net/pdf?id=ALREADYDONE",
                }
            ]
        },
    )

    [unit] = list(
        iter_units_for_scope(
            tmp_workspace,
            scope="papers",
            segment_seconds=DEFAULT_SEGMENT_SECONDS,
            overlap_seconds=DEFAULT_OVERLAP_SECONDS,
            min_chars=DEFAULT_MIN_CHARS,
        )
    )
    done_path = _unit_output_path(tmp_workspace, "testrun", unit.unit_id)
    write_json_atomic(
        done_path,
        {
            "schema_version": 1,
            "lane": "llm_full_read",
            "unit_id": unit.unit_id,
            "source_artifact_id": unit.source_artifact_id,
            "input_kind": unit.input_kind,
            "text_sha256": llm_full_read.sha256_text(unit.text),
            "topic_ids": [],
            "important": False,
            "confidence": 0.2,
            "skipped_reason": "not_important",
            "metadata": unit.metadata,
            "run_id": "testrun",
            "generated_at": "2026-07-20T00:00:00Z",
        },
    )

    class UnreachableBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            raise AssertionError(
                "chat_completion must not be called for an already-completed unit"
            )

    monkeypatch.setattr(
        llm_full_read,
        "select_local_llm_backend",
        lambda model=None: UnreachableBackend(),  # noqa: ARG005
    )

    receipt = run_llm_full_read(
        workspace_root=tmp_workspace,
        run_id="testrun",
        scope="papers",
        resume=True,
    )

    checks = receipt["checks"]
    assert checks["skipped_existing_count"] == 1
    unchanged_record = json.loads(done_path.read_text(encoding="utf-8"))
    assert unchanged_record["generated_at"] == "2026-07-20T00:00:00Z"


def test_llm_full_read_resume_rereads_a_unit_whose_text_changed(monkeypatch, tmp_workspace: Path):
    """v5.1 sub-step lineage stamps the receipt as current against today's
    corpus, so a cached record for the same unit id but different text must be
    re-read, never kept (it would be re-stamped as current)."""

    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.llm_full_read import (
        DEFAULT_MIN_CHARS,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_SEGMENT_SECONDS,
        _unit_output_path,
        iter_units_for_scope,
        run_llm_full_read,
        write_json_atomic,
    )

    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "CHANGED",
                    "forum": "CHANGED",
                    "title": "Changed Unit",
                    "abstract": ("This abstract was revised after the full read ran over it. " * 4),
                    "forum_url": "https://openreview.net/forum?id=CHANGED",
                    "pdf_url": "https://openreview.net/pdf?id=CHANGED",
                }
            ]
        },
    )
    [unit] = list(
        iter_units_for_scope(
            tmp_workspace,
            scope="papers",
            segment_seconds=DEFAULT_SEGMENT_SECONDS,
            overlap_seconds=DEFAULT_OVERLAP_SECONDS,
            min_chars=DEFAULT_MIN_CHARS,
        )
    )
    write_json_atomic(
        _unit_output_path(tmp_workspace, "testrun", unit.unit_id),
        {
            "schema_version": 1,
            "lane": "llm_full_read",
            "unit_id": unit.unit_id,
            "text_sha256": llm_full_read.sha256_text("the text as it was before the corpus changed"),
            "topic_ids": [],
            "important": False,
            "skipped_reason": "not_important",
            "run_id": "testrun",
        },
    )
    calls: list[int] = []

    class CountingBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            calls.append(1)
            raise RuntimeError("fixture backend")

    monkeypatch.setattr(llm_full_read, "select_local_llm_backend", lambda model=None: CountingBackend())  # noqa: ARG005

    receipt = run_llm_full_read(workspace_root=tmp_workspace, run_id="testrun", scope="papers", resume=True)

    assert calls, "a unit whose text changed must be re-read on resume"
    assert receipt["checks"]["skipped_existing_count"] == 0


def test_llm_full_read_resume_rereads_a_unit_whose_metadata_changed(monkeypatch, tmp_workspace: Path):
    """Same text, corrected metadata (title/speaker/timestamps reach the prompt):
    the cached record is re-read, not kept and re-stamped as current."""

    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.llm_full_read import (
        DEFAULT_MIN_CHARS,
        DEFAULT_OVERLAP_SECONDS,
        DEFAULT_SEGMENT_SECONDS,
        _unit_output_path,
        iter_units_for_scope,
        run_llm_full_read,
        write_json_atomic,
    )

    _write_json(
        tmp_workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {
            "papers": [
                {
                    "id": "RETITLED",
                    "forum": "RETITLED",
                    "title": "Corrected Title",
                    "abstract": ("This abstract text is unchanged by the metadata correction. " * 4),
                    "forum_url": "https://openreview.net/forum?id=RETITLED",
                    "pdf_url": "https://openreview.net/pdf?id=RETITLED",
                }
            ]
        },
    )
    [unit] = list(
        iter_units_for_scope(
            tmp_workspace,
            scope="papers",
            segment_seconds=DEFAULT_SEGMENT_SECONDS,
            overlap_seconds=DEFAULT_OVERLAP_SECONDS,
            min_chars=DEFAULT_MIN_CHARS,
        )
    )
    stale_metadata = {**unit.metadata, "title": "Old Wrong Title"}
    assert stale_metadata != unit.metadata
    write_json_atomic(
        _unit_output_path(tmp_workspace, "testrun", unit.unit_id),
        {
            "schema_version": 1,
            "lane": "llm_full_read",
            "unit_id": unit.unit_id,
            "input_kind": unit.input_kind,
            "text_sha256": llm_full_read.sha256_text(unit.text),
            "metadata": stale_metadata,
            "topic_ids": [],
            "important": False,
            "skipped_reason": "not_important",
            "run_id": "testrun",
        },
    )
    calls: list[int] = []

    class CountingBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            calls.append(1)
            raise RuntimeError("fixture backend")

    monkeypatch.setattr(llm_full_read, "select_local_llm_backend", lambda model=None: CountingBackend())  # noqa: ARG005

    # Default: the reading of unchanged text is carried forward, but the stale
    # labels are replaced and the refresh is recorded -- never kept silently.
    receipt = run_llm_full_read(workspace_root=tmp_workspace, run_id="testrun", scope="papers", resume=True)
    record = json.loads(_unit_output_path(tmp_workspace, "testrun", unit.unit_id).read_text(encoding="utf-8"))
    assert not calls
    assert record["metadata"] == unit.metadata and record["metadata"]["title"] != "Old Wrong Title"
    assert record["metadata_refreshed"]["reason"] == "text_unchanged_labels_changed"
    assert receipt["checks"]["metadata_refreshed_count"] == 1

    # Opt-in: a label change re-reads the unit.
    write_json_atomic(
        _unit_output_path(tmp_workspace, "testrun", unit.unit_id),
        {**record, "metadata": stale_metadata},
    )
    receipt = run_llm_full_read(
        workspace_root=tmp_workspace, run_id="testrun", scope="papers", resume=True,
        carry_forward_metadata_only_changes=False,
    )
    assert calls, "with carry-forward off, a unit whose metadata changed is re-read"
    assert receipt["checks"]["skipped_existing_count"] == 0


def test_llm_full_read_complete_run_binds_lineage_and_output_digest(monkeypatch, tmp_workspace: Path):
    """A complete full read stamps sub-step lineage (bound to the corpus
    index) and a digest of its outputs, which consumers bind."""

    from types import SimpleNamespace

    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.llm_full_read import run_llm_full_read

    class QuietBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"important": false, "topic_ids": []}'))]
            )

    monkeypatch.setattr(llm_full_read, "select_local_llm_backend", lambda model=None: QuietBackend())  # noqa: ARG005

    receipt = run_llm_full_read(workspace_root=tmp_workspace, run_id="testrun", scope="all", resume=False)

    assert receipt["status"] == "succeeded", receipt["checks"].get("completion_blockers")
    assert receipt["lineage"]["binding_source"]["substep_id"] == "llm-full-read"
    assert {entry["stage_id"] for entry in receipt["lineage"]["upstream_receipts"].values()} == {
        "corpus-inventory-index"
    }
    [outputs] = receipt["substep_outputs"]["artifacts"]
    assert outputs["path"] == "storage/extractions/testrun/llm-full-read"
    assert outputs["file_count"] >= receipt["checks"]["expected_unit_count"]

    # A corpus rebuild dropped a unit: its record leaves the outputs (moved,
    # not deleted) so consumers no longer see evidence the corpus lacks.
    by_unit = tmp_workspace / "storage" / "extractions" / "testrun" / "llm-full-read" / "by-unit"
    (by_unit / "removed_unit.json").write_text(
        '{"unit_id": "removed_unit", "topic_ids": ["t1"], "important": true}', encoding="utf-8"
    )
    resumed = run_llm_full_read(workspace_root=tmp_workspace, run_id="testrun", scope="all", resume=True)
    assert resumed["checks"]["pruned_obsolete_unit_count"] == 1
    assert not (by_unit / "removed_unit.json").exists()
    assert list((tmp_workspace / "storage" / "extractions" / "testrun" / "llm-full-read.pruned").rglob("removed_unit.json"))
    assert resumed["substep_outputs"]["digest"] == receipt["substep_outputs"]["digest"]


def test_extract_paper_abstract_cuts_heading_to_first_section_and_falls_back() -> None:
    from conference_pipeline.extractions.llm_full_read import extract_paper_abstract

    text = (
        "A Paper Title\nAlice Author\nSome University\nAbstract\n"
        "We study continu-\nous fields and show large gains.\n"
        "1. Introduction\nBody text that is not abstract.\n"
    )
    abstract, source = extract_paper_abstract(text)
    assert source == "abstract_heading"
    assert abstract == "We study continuous fields and show large gains."

    run_in, source = extract_paper_abstract("Title\nAbstract—We propose X.\nIndex Terms—vision\n")
    assert (run_in, source) == ("We propose X.", "abstract_heading")

    fallback, source = extract_paper_abstract("Title line\nNo heading here at all.\n")
    assert source == "leading_text_fallback" and fallback == "Title line No heading here at all."


def test_extract_paper_abstract_handles_real_cvpr_layouts() -> None:
    """Layouts seen in the CVPR 2026 corpus (see the dry count in the commit)."""

    from conference_pipeline.extractions.llm_full_read import extract_paper_abstract

    body = "We present a method that works. " * 8

    # Heading mid-line after the author block.
    mid = f"TinyNet\nA. Author uark.edu  Abstract {body}\n1. Introduction\nx\n"
    assert extract_paper_abstract(mid) == (body.strip(), "abstract_heading")

    # Review copy: numbered heading with a glued line number.
    review = f"CONFIDENTIAL REVIEW COPY\n1. Abstract001 {body}\n2. Introduction028 x\n"
    abstract, source = extract_paper_abstract(review)
    assert source == "abstract_heading" and abstract.startswith("We present")

    # A title starting with the word "Abstract" is not the heading.
    title = "Abstract 3D Perception for Spatial Intelligence"
    titled = f"{title}\nY. Liu\nliu@uni.edu\nAbstract\n{body}\n1. Introduction\nx\n"
    assert extract_paper_abstract(titled, title=title) == (body.strip(), "abstract_heading")

    # "1 predictions" from a hyphenated "top-1" is not a section heading;
    # a lowercase small-caps "1. introduction" is.
    hyphen = "T\nAbstract\nWe rely on top-\n1 predictions, yet more exists.\n" + body + "\n1. introduction\nx\n"
    abstract, _ = extract_paper_abstract(hyphen)
    assert abstract.startswith("We rely on top- 1 predictions") and abstract.endswith("works.")

    # No heading: start after the author block's last email/affiliation.
    no_heading = f"RAGEN-2\nZ. Wang\nNorthwestern University\nz@nw.edu\n{body}\n1. Introduction\nx\n"
    assert extract_paper_abstract(no_heading) == (body.strip(), "after_author_block")


def test_paper_units_come_from_the_corpus_when_there_is_no_openreview_export(monkeypatch, tmp_path: Path):
    """CVPR has no OpenReview export: paper-abstract units are cut from the
    corpus inventory's paper texts, with ids that canonicalize like the
    reference validator's (``paper_abstract:<body>``)."""

    from conference_pipeline.extractions import llm_full_read
    from conference_pipeline.extractions.walker import ExtractionInput

    workspace = tmp_path / "ws"
    text_path = workspace / "storage" / "corpus" / "paper-text" / "p1.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(
        "Great Paper\nAbstract\n" + ("We show a strong result on benchmarks. " * 8) + "\n1. Introduction\nx\n",
        encoding="utf-8",
    )
    _write_json(
        workspace / "storage" / "corpus" / "corpus_inventory.json",
        {
            "artifact_records": [
                {
                    "artifact_id": "workshop_paper:thecvf.com/CVPR/2026/Workshop/W1:abc",
                    "paper_title": "Great Paper",
                    "authors": ["A. Author"],
                    "source_family_id": "workshop_paper_text",
                }
            ]
        },
    )
    inputs = [
        ExtractionInput(
            artifact_id="workshop_paper:thecvf.com/CVPR/2026/Workshop/W1:abc",
            kind="paper_pdf",
            text_path="storage/corpus/paper-text/p1.txt",
            digest="d" * 64,
            status="text_extracted",
            source_path="storage/workshops/papers/p1.pdf",
            topic_hint=None,
            text_content=None,
        ),
        ExtractionInput(
            artifact_id="speech_transcript:t1",
            kind="transcript",
            text_path=None,
            digest="e" * 64,
            status="transcribed",
            source_path="",
            topic_hint=None,
            text_content=None,
        ),
    ]
    monkeypatch.setattr(llm_full_read, "iterate_extraction_inputs", lambda workspace_root: iter(inputs))

    [unit] = list(llm_full_read.iter_paper_abstract_units(workspace))
    assert unit.unit_id == "paper_abstract:thecvf.com_CVPR_2026_Workshop_W1_abc"
    assert unit.source_artifact_id == "workshop_paper:thecvf.com/CVPR/2026/Workshop/W1:abc"
    assert unit.text.startswith("Title: Great Paper\n\nAbstract:\nWe show a strong result")
    assert "Introduction" not in unit.text
    assert unit.metadata["abstract_source"] == "abstract_heading"
    assert unit.metadata["authors"] == ["A. Author"]
    assert llm_full_read.count_indexable_corpus_papers(workspace) == 1

    # With an OpenReview export (ICLR/ICML), the export is the only paper source.
    _write_json(
        workspace / "storage" / "download-corpus" / "openreview_accepted_papers.json",
        {"papers": [{"id": "OR1", "title": "OpenReview Paper", "abstract": "An abstract. " * 30}]},
    )
    assert [u.unit_id for u in llm_full_read.iter_paper_abstract_units(workspace)] == ["paper_abstract:OR1"]


def test_llm_full_read_blocks_when_corpus_papers_produce_no_paper_units(monkeypatch, tmp_workspace: Path):
    """The July CVPR full read reported complete with 0 of 6,594 papers read."""

    from types import SimpleNamespace

    from conference_pipeline.extractions import llm_full_read

    class QuietBackend:
        def chat_completion(self, **kwargs):  # noqa: ARG002
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"important": false, "topic_ids": []}'))]
            )

    monkeypatch.setattr(llm_full_read, "select_local_llm_backend", lambda model=None: QuietBackend())  # noqa: ARG005
    monkeypatch.setattr(llm_full_read, "iter_paper_abstract_units", lambda workspace_root, min_chars=0: iter(()))
    monkeypatch.setattr(llm_full_read, "count_indexable_corpus_papers", lambda workspace_root: 42)

    receipt = llm_full_read.run_llm_full_read(workspace_root=tmp_workspace, run_id="testrun", scope="all", resume=False)

    assert receipt["status"] == "partial"
    codes = {blocker["code"] for blocker in receipt["checks"]["completion_blockers"]}
    assert "llm_full_read_paper_source_missing" in codes
    assert "lineage" not in receipt


def test_walker_moves_aside_records_and_topics_that_no_longer_exist(tmp_workspace: Path):
    """Records for renamed/removed artifacts must not linger in the per-topic index."""
    import json

    from conference_pipeline.extractions.walker import walk_and_extract

    root = tmp_workspace / "storage" / "extractions" / "testrun"
    stale = root / "by-artifact" / "media_gone_old_artifact.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({
        "artifact_id": "media:gone:old", "topic_ids": ["99_retired_topic"], "confidence": 0.9,
    }), encoding="utf-8")
    old_topic = root / "by-topic" / "99_retired_topic.json"
    old_topic.parent.mkdir(parents=True, exist_ok=True)
    old_topic.write_text("{}", encoding="utf-8")

    result = walk_and_extract(tmp_workspace, run_id="testrun", force_rebuild=True)

    assert not stale.exists() and not old_topic.exists()
    assert result.pruned_artifact_count == 1 and result.pruned_topic_count == 1
    pruned = Path(result.pruned_dir)
    assert (pruned / "by-artifact" / stale.name).exists()
    assert (pruned / "by-topic" / old_topic.name).exists()
    topic_ids = {json.loads(p.read_text())["topic_id"] for p in (root / "by-topic").glob("*.json")}
    assert "99_retired_topic" not in topic_ids
    assert len(list((root / "by-artifact").glob("*.json"))) == result.expected_count
