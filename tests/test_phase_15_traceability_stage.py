"""Phase 15 contract tests: citation sidecar loader + manifest-traceability."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.iclr2025_live import load_citation_sidecar


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data), encoding="utf-8")
    else:
        path.write_text(str(data), encoding="utf-8")


def test_load_citation_sidecar_missing_returns_absent(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "analysis").mkdir(parents=True)
    out = load_citation_sidecar(workspace)
    assert out["present"] is False
    assert out["topics"] == {}
    assert out["path"] == "analysis/analysis_citation_manifest.json"


def test_load_citation_sidecar_corrupt_treated_as_missing(tmp_path):
    workspace = tmp_path / "ws"
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "analysis" / "analysis_citation_manifest.json").write_text(
        "this is not json", encoding="utf-8"
    )
    out = load_citation_sidecar(workspace)
    assert out["present"] is False
    assert out["topics"] == {}


def test_load_citation_sidecar_counts_per_topic(tmp_path):
    workspace = tmp_path / "ws"
    sidecar = workspace / "analysis" / "analysis_citation_manifest.json"
    _write(
        sidecar,
        {
            "schema_version": 1,
            "topics": [
                {
                    "topic_id": "02_x",
                    "extraction_record_ids": ["a", "b", "c"],
                    "vector_store_chunk_ids": ["a#1", "b#1"],
                    "research_ids": ["02_x#abc"],
                },
                {
                    "topic_id": "03_y",
                    "extraction_record_ids": [],
                    "vector_store_chunk_ids": ["c#1"],
                    "research_ids": ["03_y#def"],
                },
            ],
        },
    )
    out = load_citation_sidecar(workspace)
    assert out["present"] is True
    assert out["sha256"]
    assert out["topics"] == {
        "02_x": {
            "extraction_record_ids": 3,
            "vector_store_chunk_ids": 2,
            "research_ids": 1,
        },
        "03_y": {
            "extraction_record_ids": 0,
            "vector_store_chunk_ids": 1,
            "research_ids": 1,
        },
    }


def test_load_citation_sidecar_skips_non_dict_entries(tmp_path):
    workspace = tmp_path / "ws"
    sidecar = workspace / "analysis" / "analysis_citation_manifest.json"
    _write(
        sidecar,
        {
            "topics": [
                "not a dict",
                {"missing": "topic_id"},
                {
                    "topic_id": 42,
                    "extraction_record_ids": ["a"],
                },
                {
                    "topic_id": "04_z",
                    "extraction_record_ids": ["a"],
                    "vector_store_chunk_ids": [],
                    "research_ids": [],
                },
            ],
        },
    )
    out = load_citation_sidecar(workspace)
    assert out["present"] is True
    assert list(out["topics"].keys()) == ["04_z"]
    assert out["topics"]["04_z"]["extraction_record_ids"] == 1
