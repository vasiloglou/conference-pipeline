"""Index coverage verification names every listed talk and paper that is not indexed."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.indexing import index_coverage


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")


def test_coverage_report_counts_and_names_gaps(tmp_path: Path) -> None:
    ws = tmp_path / "conf"
    pages = {
        "tut-1": ("ZETA Tutorial Proofs", "v-tut"),
        "tut-2": ("ZETA Tutorial Silent Poster Tutorial", "v-silent"),
        "tut-3": ("ZETA Tutorial No Video Tutorial", None),
        "invited-talk-9": ("ZETA Invited Talk Next Decade", "v-key"),
        "break-1": ("ZETA Break Coffee Break", None),
    }
    assets = []
    for item_id, (title, video) in pages.items():
        _write(ws / f"cache/{item_id}.html", f"<title>{title}</title>")
        assets.append({"item_id": item_id, "kind": None, "slideslive_id": video,
                       "page_url": f"https://zeta/{item_id}", "page_artifact_path": f"cache/{item_id}.html"})
    _write(ws / "storage/download-corpus/slideslive_media_recovery.json", {"page_assets": assets})
    records = [
        {"artifact_id": "s-tut", "source_family_id": "speech_transcript", "indexable_text": True,
         "metadata": {"slideslive_id": "v-tut"}},
        {"artifact_id": "s-silent", "source_family_id": "speech_transcript", "indexable_text": False,
         "index_exclusion": {"reason": "speechless_recording"}, "metadata": {"slideslive_id": "v-silent"}},
        {"artifact_id": "s-key", "source_family_id": "speech_transcript", "indexable_text": True,
         "metadata": {"slideslive_id": "v-key"}},
        {"artifact_id": "p1", "source_family_id": "paper_text", "indexable_text": True, "openreview_id": "abc"},
        {"artifact_id": "p2", "source_family_id": "paper_text", "indexable_text": False, "openreview_id": "dup",
         "index_exclusion": {"reason": "duplicate_title"}},
        {"artifact_id": "w1", "source_family_id": "workshop_paper_text", "indexable_text": True, "openreview_id": "n1"},
    ]
    _write(ws / "storage/corpus/corpus_inventory.json", {"artifact_records": records})
    rows = [
        {"artifact_id": "s-tut", "chunk_id": "s-tut:0", "source_family_id": "speech_transcript", "text": "Presentation: Tutorial: Proofs\n\nhello"},
        {"artifact_id": "p1", "chunk_id": "p1:0", "source_family_id": "paper_text", "text": "Paper Title: A\n\nbody"},
        {"artifact_id": "w1", "chunk_id": "w1:0", "source_family_id": "paper_text", "text": "bare body without header"},
    ]
    _write(ws / "storage/vector-store/run/stores/papers/chunks.jsonl", "".join(json.dumps(r) + "\n" for r in rows[1:]))
    _write(ws / "storage/vector-store/run/stores/artifacts/chunks.jsonl", json.dumps(rows[0]) + "\n")
    _write(ws / "storage/download-corpus/download_corpus.json", {"openreview_import": {"local_path": "storage/download-corpus/sub.json"}})
    _write(ws / "storage/download-corpus/sub.json", {"papers": [
        {"id": "abc", "title": "A"}, {"id": "dup", "title": "Dup"}, {"id": "gone", "title": "Never Downloaded"},
    ]})
    _write(ws / "storage/workshops/papers/zeta_ws/receipts/n1.json", {"note_id": "n1", "venueid": "Z/Workshop/WS", "status": "succeeded", "title": "W"})
    _write(ws / "storage/workshops/papers/zeta_ws/receipts/n2.json", {"note_id": "n2", "venueid": "Z/Workshop/WS", "status": "unreachable", "title": "Lost"})

    json_path, md_path, report = index_coverage.write_index_coverage_report(ws, "run")
    tutorials = report["talks"]["tutorial"]
    assert (tutorials["pages"], tutorials["with_recording"], tutorials["indexed"]) == (3, 2, 1)
    assert tutorials["excluded"] == [{"title": "Tutorial: Silent Poster Tutorial", "url": "https://zeta/tut-2", "reason": "speechless_recording"}]
    assert tutorials["not_indexed"] == [{"title": "Tutorial: No Video Tutorial", "url": "https://zeta/tut-3", "reason": "no recording on the event page"}]
    # The keynote has a record but no passages: named, not silently counted.
    assert report["talks"]["invited-talk"]["not_indexed"][0]["reason"] == "indexable record has no passages"
    assert report["events_without_talks"] == {"Break": 1}
    main = report["papers"]["main_track"]
    assert (main["listed"], main["indexed"]) == (3, 1)
    assert main["excluded_by_reason"] == {"duplicate_title": 1, "no PDF downloaded": 1}
    assert main["missing"] == [{"title": "Never Downloaded", "id": "gone", "reason": "no PDF downloaded"}]
    assert report["papers"]["workshops"]["missing"] == [{"title": "Lost", "workshop": "WS", "reason": "pdf unreachable"}]
    headers = report["passage_headers"]
    assert headers["papers"] == {
        "passages": 2, "with_title_header": 1, "without_header_by_family": {"paper_text": 1},
        "without_header_sample": [{"chunk_id": "w1:0", "family": "paper_text", "starts": "bare body without header"}],
    }
    assert headers["artifacts"]["with_title_header"] == 1
    assert json_path.is_file() and "| tutorial | 3 | 2 |" in md_path.read_text(encoding="utf-8")


def test_a_workshop_with_no_downloaded_papers_is_still_listed(tmp_path: Path) -> None:
    ws = tmp_path / "conf"
    _write(ws / "storage/corpus/corpus_inventory.json", {"artifact_records": []})
    _write(ws / "storage/workshops/inventory.json", {"workshops": [
        {"venueid": "Z/Workshop/Meta", "paper_count": 60, "pdf_availability_class": "openreview_metadata_only"},
    ]})
    _write(ws / "storage/workshops/papers/z_ext/receipts/e1.json",
           {"venueid": "Z/Workshop/Ext", "status": "unreachable", "title": "Ext", "receipt_type": "x.external_workshop_paper_receipt"})
    report = index_coverage.verify_index_coverage(ws, "run")
    workshops = report["papers"]["workshops"]
    assert workshops["listed"] == 61
    assert workshops["excluded_by_reason"] == {"no PDF downloaded (openreview_metadata_only)": 60, "pdf unreachable": 1}
    assert {"title": "60 papers", "workshop": "Meta", "reason": "no PDF downloaded (openreview_metadata_only)"} in workshops["missing"]
