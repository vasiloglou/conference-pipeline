"""Phase 15 contract tests: analysis_manifest.json citation_counts enrichment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline import workflow
from conference_pipeline.analysis.errors import ManifestValidationError
from conference_pipeline.analysis.manifest import (
    AnalysisManifest,
    TopicEntry,
    emit_analysis_manifest,
    load_and_validate_manifest,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_workspace(tmp_path: Path):
    workspace_path = tmp_path / "conferences" / "demo2026"
    _write_text(
        workspace_path / "config.yaml",
        (
            "conference:\n"
            "  name: Demo 2026\n"
            "storage:\n"
            '  manifest_path: "storage/storage_manifest.json"\n'
        ),
    )
    _write_text(
        workspace_path / "storage" / "storage_manifest.json",
        json.dumps({"artifact_families": []}, indent=2) + "\n",
    )
    _write_text(
        workspace_path / "analysis" / "01_demo_topic.md",
        "# Demo Topic\n\nBody.\n",
    )
    _write_text(
        workspace_path / "analysis" / "02_other_topic.md",
        "# Other Topic\n\nBody two.\n",
    )
    return workflow.resolve_workspace(workspace_path)


def test_topic_entry_round_trip_preserves_citation_counts():
    counts = {
        "extraction_record_ids": 12,
        "vector_store_chunk_ids": 8,
        "research_ids": 1,
    }
    entry = TopicEntry(
        id="02_other_topic",
        number=2,
        slug="other_topic",
        title="Other Topic",
        markdown_path="analysis/02_other_topic.md",
        markdown_sha256="0" * 64,
        fact_ranks_path=None,
        fact_ranks_sha256=None,
        figures_dir=None,
        audio_dir=None,
        decks_dir=None,
        citation_counts=counts,
    )
    payload = entry.to_dict()
    assert payload["citation_counts"] == counts

    parsed = TopicEntry.from_json(payload, topic_index=0)
    assert parsed.citation_counts == counts


def test_topic_entry_without_citation_counts_omits_field():
    entry = TopicEntry(
        id="01_demo_topic",
        number=1,
        slug="demo_topic",
        title="Demo Topic",
        markdown_path="analysis/01_demo_topic.md",
        markdown_sha256="0" * 64,
        fact_ranks_path=None,
        fact_ranks_sha256=None,
        figures_dir=None,
        audio_dir=None,
        decks_dir=None,
        citation_counts=None,
    )
    payload = entry.to_dict()
    assert "citation_counts" not in payload


def test_topic_entry_round_trip_preserves_provenance_paths():
    entry = TopicEntry(
        id="02_other_topic",
        number=2,
        slug="other_topic",
        title="Other Topic",
        markdown_path="analysis/02_other_topic.md",
        markdown_sha256="0" * 64,
        fact_ranks_path=None,
        fact_ranks_sha256=None,
        figures_dir=None,
        audio_dir=None,
        decks_dir=None,
        provenance_path="analysis/provenance/02_other_topic.md",
        provenance_json_path="analysis/provenance/02_other_topic.json",
        citation_counts=None,
    )
    payload = entry.to_dict()
    parsed = TopicEntry.from_json(payload, topic_index=0)

    assert payload["provenance_path"] == "analysis/provenance/02_other_topic.md"
    assert payload["provenance_json_path"] == "analysis/provenance/02_other_topic.json"
    assert parsed.provenance_path == "analysis/provenance/02_other_topic.md"
    assert parsed.provenance_json_path == "analysis/provenance/02_other_topic.json"


def test_topic_entry_rejects_bad_citation_counts_shape():
    with pytest.raises(ManifestValidationError):
        TopicEntry.from_json(
            {
                "id": "02_x",
                "number": 2,
                "slug": "x",
                "title": "X",
                "markdown_path": "analysis/02_x.md",
                "markdown_sha256": "0" * 64,
                "citation_counts": {"extraction_record_ids": -1, "vector_store_chunk_ids": 0, "research_ids": 0},
            },
            topic_index=0,
        )


def test_topic_entry_rejects_unknown_citation_count_field():
    with pytest.raises(ManifestValidationError):
        TopicEntry.from_json(
            {
                "id": "02_x",
                "number": 2,
                "slug": "x",
                "title": "X",
                "markdown_path": "analysis/02_x.md",
                "markdown_sha256": "0" * 64,
                "citation_counts": {
                    "extraction_record_ids": 1,
                    "vector_store_chunk_ids": 1,
                    "research_ids": 1,
                    "bogus": 1,
                },
            },
            topic_index=0,
        )


def test_emit_analysis_manifest_persists_citation_counts(tmp_path):
    resolved = _make_workspace(tmp_path)
    citations_by_topic = {
        "01_demo_topic": {
            "extraction_record_ids": 5,
            "vector_store_chunk_ids": 3,
            "research_ids": 1,
        },
        "02_other_topic": {
            "extraction_record_ids": 9,
            "vector_store_chunk_ids": 7,
            "research_ids": 2,
        },
    }
    manifest_path, _, _ = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.tests@phase15",
        citations_by_topic=citations_by_topic,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {topic["id"]: topic for topic in payload["topics"]}
    assert by_id["01_demo_topic"]["citation_counts"] == citations_by_topic["01_demo_topic"]
    assert by_id["02_other_topic"]["citation_counts"] == citations_by_topic["02_other_topic"]

    loaded = load_and_validate_manifest(resolved)
    by_id_loaded = {t.id: t for t in loaded.manifest.topics}
    assert by_id_loaded["01_demo_topic"].citation_counts == citations_by_topic["01_demo_topic"]
    assert by_id_loaded["02_other_topic"].citation_counts == citations_by_topic["02_other_topic"]


def test_emit_analysis_manifest_persists_provenance_paths(tmp_path):
    resolved = _make_workspace(tmp_path)
    _write_text(
        resolved.root / "analysis" / "provenance" / "01_demo_topic.md",
        "# Demo Provenance\n",
    )
    _write_text(
        resolved.root / "analysis" / "provenance" / "01_demo_topic.json",
        "{}\n",
    )

    manifest_path, _, _ = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.tests@phase15",
        provenance_by_topic={
            "01_demo_topic": {
                "provenance_path": "analysis/provenance/01_demo_topic.md",
                "provenance_json_path": "analysis/provenance/01_demo_topic.json",
            }
        },
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {topic["id"]: topic for topic in payload["topics"]}
    assert by_id["01_demo_topic"]["provenance_path"] == "analysis/provenance/01_demo_topic.md"
    assert (
        by_id["01_demo_topic"]["provenance_json_path"]
        == "analysis/provenance/01_demo_topic.json"
    )

    loaded = load_and_validate_manifest(resolved)
    topic = {item.id: item for item in loaded.manifest.topics}["01_demo_topic"]
    assert topic.provenance_path == "analysis/provenance/01_demo_topic.md"
    assert topic.provenance_json_path == "analysis/provenance/01_demo_topic.json"


def test_emit_analysis_manifest_without_citations_omits_field(tmp_path):
    resolved = _make_workspace(tmp_path)
    manifest_path, _, _ = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.tests@phase15",
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for topic in payload["topics"]:
        assert "citation_counts" not in topic


def test_emit_analysis_manifest_partial_citation_counts_default_zero(tmp_path):
    resolved = _make_workspace(tmp_path)
    citations_by_topic = {
        "01_demo_topic": {"extraction_record_ids": 4},
    }
    manifest_path, _, _ = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.tests@phase15",
        citations_by_topic=citations_by_topic,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {topic["id"]: topic for topic in payload["topics"]}
    assert by_id["01_demo_topic"]["citation_counts"] == {
        "extraction_record_ids": 4,
        "vector_store_chunk_ids": 0,
        "research_ids": 0,
    }
    # 02_other_topic was not in the citations dict, so no enrichment.
    assert "citation_counts" not in by_id["02_other_topic"]


def test_emit_analysis_manifest_rejects_unknown_citation_field(tmp_path):
    resolved = _make_workspace(tmp_path)
    citations_by_topic = {
        "01_demo_topic": {"bogus_field": 1},
    }
    with pytest.raises(ManifestValidationError):
        emit_analysis_manifest(
            resolved,
            generator="conference_pipeline.tests@phase15",
            citations_by_topic=citations_by_topic,
        )


def test_load_validate_rejects_negative_citation_count(tmp_path):
    resolved = _make_workspace(tmp_path)
    manifest_path, _, _ = emit_analysis_manifest(
        resolved,
        generator="conference_pipeline.tests@phase15",
        citations_by_topic={
            "01_demo_topic": {
                "extraction_record_ids": 1,
                "vector_store_chunk_ids": 1,
                "research_ids": 1,
            },
        },
    )
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["topics"][0]["citation_counts"]["research_ids"] = -1
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ManifestValidationError):
        load_and_validate_manifest(resolved)
