"""inventory_index_backfill: each kept inventory record gets the text Stage 6 should chunk."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conference_pipeline.indexing import inventory_index_backfill as backfill


def _workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    papers = ws / "storage" / "papers"
    (papers / "late" / "text").mkdir(parents=True)
    (papers / "late" / "units").mkdir(parents=True)
    (ws / "storage" / "acquisition").mkdir(parents=True)
    (papers / "late" / "text" / "p1.txt").write_text("Full text of paper one.", encoding="utf-8")
    (papers / "late" / "units" / "own.jsonl").write_text(
        json.dumps({"unit_id": "own1", "path": "storage/papers/late/text/p1.txt", "title": "One"}) + "\n",
        encoding="utf-8",
    )
    (papers / "late" / "units" / "shared.jsonl").write_text(
        "\n".join(
            json.dumps({"unit_id": u, "text": f"Abstract of {u}.", "title": u}) for u in ("s1", "s2")
        )
        + "\n",
        encoding="utf-8",
    )
    (ws / "storage" / "acquisition" / "existing.json").write_text(
        json.dumps({"papers": [{"forum": "ex1", "title": "Existing", "abstract": "Existing abstract."}]}),
        encoding="utf-8",
    )
    (papers / "late_paper_lanes.json").write_text(
        json.dumps(
            {
                "existing_corpus": {"adapter": "openreview_acquisition_json", "path": "storage/acquisition/existing.json"},
                "lanes": [
                    {"lane_id": "own", "unit_file": "storage/papers/late/units/own.jsonl", "track": "main", "id_kind": "site"},
                    {"lane_id": "shared", "unit_file": "storage/papers/late/units/shared.jsonl", "track": "workshop", "id_kind": "site", "content": "abstract"},
                ],
            }
        ),
        encoding="utf-8",
    )

    def rec(aid: str, path: str, unit: str, **extra: object) -> dict:
        return {"artifact_id": aid, "source_family_id": "paper_text", "indexable_text": True, "source_path": path, "metadata": {"unit_id": unit}, **extra}

    records = [
        rec("a:own", "storage/papers/late/text/p1.txt", "own1"),
        rec("a:s1", "storage/papers/late/units/shared.jsonl", "s1"),
        rec("a:s2", "storage/papers/late/units/shared.jsonl", "s2"),
        rec("a:ex", "storage/acquisition/existing.json", "ex1"),
        rec("a:dup", "storage/papers/late/units/shared.jsonl", "s1", index_exclusion="duplicate_copy"),
        rec("a:none", "storage/papers/late/units/shared.jsonl", "nope"),
        {"artifact_id": "a:off", "indexable_text": False, "source_path": "x", "metadata": {}},
    ]
    (ws / "storage" / "corpus").mkdir(parents=True)
    (ws / "storage" / "corpus" / "corpus_inventory.json").write_text(
        json.dumps({"artifact_records": records, "corpus_records_sha256": "abc"}), encoding="utf-8"
    )
    return ws


def test_resolve_routes_every_kept_record(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    receipt = backfill.run_backfill(ws, "r1", resolve_only=True)
    assert receipt["status"] == "resolved_only"
    assert receipt["records_kept"] == 5  # the excluded copy and the non-indexable record are dropped
    assert receipt["text_routes"] == {"own_text_file": 1, "unit_text_written": 3, "unresolved": 1}
    assert receipt["unresolved_artifact_ids"] == ["a:none"]
    assert (ws / "storage/vector-store/r1/inventory_index_backfill_receipt.json").is_file()


def test_unit_text_is_written_once_and_handed_to_the_build(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    seen: dict = {}

    def fake_build(**kwargs: object) -> SimpleNamespace:
        seen.update(kwargs)
        return SimpleNamespace(
            embedding_readiness={"status": "available"},
            vector_materialization={"status": "succeeded"},
            materialization_error="",
            chunk_counts={"papers": 4},
            embedded_chunk_counts={"papers": 4},
            index_digest_path=ws / "storage/vector-store/r2/index_digest.json",
            index_digest={"stores": {"papers": {"skipped_unreadable": [{"artifact_id": "a:none"}]}}, "embedding_dim": 1024},
        )

    receipt = backfill.run_backfill(ws, "r2", build_fn=fake_build)
    assert receipt["status"] == "succeeded"
    assert receipt["skipped_unreadable"] == {"papers": 1}
    by_id = {r["artifact_id"]: r for r in seen["artifact_records"]}
    assert by_id["a:own"]["text_path"] == "storage/papers/late/text/p1.txt"
    s1 = ws / by_id["a:s1"]["text_path"]
    assert s1.read_text(encoding="utf-8") == "Abstract of s1."
    assert (ws / by_id["a:ex"]["text_path"]).read_text(encoding="utf-8") == "Existing abstract."
    assert "text_path" not in by_id["a:none"]
    mtime = s1.stat().st_mtime_ns
    backfill.run_backfill(ws, "r2", build_fn=fake_build)  # a rerun leaves an unchanged unit text alone
    assert s1.stat().st_mtime_ns == mtime


def test_failed_materialization_fails_the_receipt(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)

    def failing_build(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            embedding_readiness={"status": "available"},
            vector_materialization={"status": "failed"},
            materialization_error="boom",
            chunk_counts={},
            embedded_chunk_counts={},
            index_digest_path=ws / "x.json",
            index_digest={},
        )

    assert backfill.run_backfill(ws, "r3", build_fn=failing_build)["status"] == "failed"


def test_missing_inventory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(backfill.InventoryIndexBackfillError):
        backfill.run_backfill(tmp_path, "r4", resolve_only=True)


def test_a_talk_record_uses_its_own_transcript(tmp_path: Path) -> None:
    """v6.0 Phase 5.1 (spec section 6.7): a lane talk record is indexed from its transcript file."""
    ws = _workspace(tmp_path)
    transcript = ws / "storage" / "transcription-ocr" / "evidence-lanes" / "talks" / "transcripts" / "t1.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("A talk transcript.", encoding="utf-8")
    inventory_path = ws / "storage" / "corpus" / "corpus_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["artifact_records"] += [
        {"artifact_id": "t:1", "source_family_id": "speech_transcript", "indexable_text": True,
         "transcript_path": "storage/transcription-ocr/evidence-lanes/talks/transcripts/t1.txt"},
        {"artifact_id": "t:outside", "source_family_id": "speech_transcript", "indexable_text": True,
         "transcript_path": "../outside.txt"},
    ]
    (tmp_path / "outside.txt").write_text("outside the workspace", encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    receipt = backfill.run_backfill(ws, "r1", resolve_only=True)
    assert receipt["text_routes"] == {"own_text_file": 1, "unit_text_written": 3, "unresolved": 2, "transcript": 1}
    assert receipt["unresolved_artifact_ids"] == ["a:none", "t:outside"]
