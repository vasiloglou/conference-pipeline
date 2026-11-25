"""Talk-split links survive a transcription-ocr rerun through their own ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from conference_pipeline.indexing import workshop_talk_annotation_bindings as bindings
from conference_pipeline.production import corpus_index_stage


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _workspace(tmp_path: Path) -> tuple[Path, dict]:
    ws = tmp_path / "conf"
    annotated = ws / "storage/transcription-ocr/workshop-talks/W1-annotated.json"
    annotated.parent.mkdir(parents=True)
    annotated.write_text('{"segments": []}', encoding="utf-8")
    (ws / "storage/transcription-ocr/workshop-talks/W1-talks.json").write_text("{}", encoding="utf-8")
    lane_record = {"source_sha256": "src1", "transcript_sha256": _sha("hello"), "source_path": "cache/w1.mp3"}
    return ws, lane_record


def test_binding_round_trip_restores_fields(tmp_path: Path) -> None:
    ws, lane_record = _workspace(tmp_path)
    binding = bindings.make_binding(
        ws,
        source_sha256="src1",
        transcript_sha256=lane_record["transcript_sha256"],
        annotated_transcript_json_path=ws / "storage/transcription-ocr/workshop-talks/W1-annotated.json",
        fields={
            "workshop_talks_json_path": str(ws / "storage/transcription-ocr/workshop-talks/W1-talks.json"),
            "workshop_talk_count": "4",
            "unrelated": "dropped",
        },
        bound_by="test",
    )
    bindings.upsert_bindings(ws, [binding], writer="test")
    fields, code = bindings.binding_for_record(ws, bindings.load_bindings(ws), lane_record)
    assert code is None
    assert fields == {
        "annotated_transcript_json_path": "storage/transcription-ocr/workshop-talks/W1-annotated.json",
        "workshop_talks_json_path": "storage/transcription-ocr/workshop-talks/W1-talks.json",
        "workshop_talk_count": 4,
    }
    # A recording the ledger does not know is not a blocker.
    assert bindings.binding_for_record(ws, bindings.load_bindings(ws), {"source_sha256": "other"}) == (None, None)


def test_binding_refuses_a_broken_chain(tmp_path: Path) -> None:
    ws, lane_record = _workspace(tmp_path)
    annotated = ws / "storage/transcription-ocr/workshop-talks/W1-annotated.json"
    bindings.upsert_bindings(
        ws,
        [bindings.make_binding(ws, source_sha256="src1", transcript_sha256=lane_record["transcript_sha256"],
                               annotated_transcript_json_path=annotated, bound_by="test")],
        writer="test",
    )
    ledger = bindings.load_bindings(ws)
    retranscribed = {**lane_record, "transcript_sha256": _sha("different words")}
    assert bindings.binding_for_record(ws, ledger, retranscribed) == (None, bindings.BLOCKER_TRANSCRIPT_CHANGED)
    annotated.write_text('{"segments": [1]}', encoding="utf-8")
    assert bindings.binding_for_record(ws, ledger, lane_record) == (None, bindings.BLOCKER_ANNOTATED_DIGEST_MISMATCH)
    annotated.unlink()
    assert bindings.binding_for_record(ws, ledger, lane_record) == (None, bindings.BLOCKER_ANNOTATED_MISSING)


def test_malformed_ledger_fails_closed(tmp_path: Path) -> None:
    ws, _ = _workspace(tmp_path)
    bindings.ledger_path(ws).write_text('{"receipt_type": "something_else"}', encoding="utf-8")
    with pytest.raises(bindings.AnnotationBindingError):
        bindings.load_bindings(ws)


def test_backfill_recovers_only_unchanged_chains(tmp_path: Path) -> None:
    ws, lane_record = _workspace(tmp_path)
    annotated_rel = "storage/transcription-ocr/workshop-talks/W1-annotated.json"
    annotated_sha = hashlib.sha256((ws / annotated_rel).read_bytes()).hexdigest()
    corpus = {"artifact_records": [
        {"indexed_text_kind": "annotated_transcript_json", "indexed_text_path": annotated_rel,
         "indexed_text_sha256": annotated_sha, "source_sha256": "src1", "text_sha256": lane_record["transcript_sha256"],
         "metadata": {"workshop_talks_json_path": str(ws / "storage/transcription-ocr/workshop-talks/W1-talks.json")}},
        {"indexed_text_kind": "annotated_transcript_json", "indexed_text_path": annotated_rel,
         "indexed_text_sha256": annotated_sha, "source_sha256": "src2", "text_sha256": "old"},
        # Program talk maps are bound by their own module, never by this ledger.
        {"indexed_text_kind": "annotated_transcript_json", "indexed_text_path": "storage/transcription-ocr/talk-maps/base/v_annotated.json",
         "indexed_text_sha256": "x", "source_sha256": "src3", "text_sha256": "y"},
    ]}
    transcription = {"lanes": {"session_speech_transcription": {"records": [
        lane_record, {"source_sha256": "src2", "transcript_sha256": "new"},
    ]}}}
    recovered, skipped = bindings.bindings_from_corpus_inventory(ws, corpus, transcription)
    assert [b["source_sha256"] for b in recovered] == ["src1"]
    assert recovered[0]["workshop_talks_json_path"] == "storage/transcription-ocr/workshop-talks/W1-talks.json"
    assert skipped == [{"source_sha256": "src2", "annotated_transcript_json_path": annotated_rel,
                        "reason": "transcript changed since the index was built"}]


def test_ledger_is_a_registered_corpus_index_input() -> None:
    registry = json.loads(
        (Path(__file__).resolve().parents[1] / "src/conference_pipeline/stage_owner_registry.json").read_text()
    )
    text = json.dumps(registry)
    assert bindings.LEDGER_RELATIVE_PATH.as_posix() in text


def _load_icml_runner():
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_icml2026_production.py"
    spec = importlib.util.spec_from_file_location("icml_runner_for_annotation_bindings", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_icml_indexer_restores_a_split_dropped_by_a_transcription_rerun(tmp_path: Path) -> None:
    runner = _load_icml_runner()
    ws, _ = _workspace(tmp_path)
    audio = ws / "cache/w1.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    transcript = ws / "storage/transcription-ocr/speech-transcripts/speech-w1.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("hello", encoding="utf-8")
    # The lane record as a transcription-ocr rerun leaves it: no split fields.
    lane_record = {
        "source_path": "cache/w1.mp3", "status": "succeeded", "workshop_id": "Z/Workshop/W1",
        "transcript_path": "storage/transcription-ocr/speech-transcripts/speech-w1.txt",
        "source_sha256": "src1", "transcript_sha256": _sha("hello"),
    }
    inventory = {"lanes": {"session_speech_transcription": {"records": [lane_record]}}}
    common = dict(
        workspace=ws, run_id="r", download_stage_receipt_sha256="a", download_corpus_inventory_sha256="b",
        transcription_stage_receipt_sha256="c", transcription_inventory_sha256="d", download_corpus={},
        transcription_inventory=inventory,
    )

    _, blockers = corpus_index_stage.build_lane_inventory_records(
        **common, builders=runner.corpus_index_record_builders()
    )
    assert [b["code"] for b in blockers] == ["workshop_talk_segmentation_required"]

    bindings.upsert_bindings(ws, [bindings.make_binding(
        ws, source_sha256="src1", transcript_sha256=_sha("hello"),
        annotated_transcript_json_path="storage/transcription-ocr/workshop-talks/W1-annotated.json",
        fields={"workshop_talks_json_path": "storage/transcription-ocr/workshop-talks/W1-talks.json"},
        bound_by="test",
    )], writer="test")
    records, blockers = corpus_index_stage.build_lane_inventory_records(
        **common, builders=runner.corpus_index_record_builders()
    )
    assert blockers == []
    assert records[0]["indexed_text_kind"] == "annotated_transcript_json"
    assert records[0]["indexed_text_path"] == "storage/transcription-ocr/workshop-talks/W1-annotated.json"
    assert records[0]["metadata"]["workshop_talks_json_path"] == "storage/transcription-ocr/workshop-talks/W1-talks.json"
