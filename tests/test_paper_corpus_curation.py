"""Paper-corpus curation: image-only PDFs and duplicate copies become audit-only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from conference_pipeline.indexing import paper_corpus_curation as curation
from conference_pipeline.indexing.corpus_vector_store import chunk_record_for_artifact

REAL_TEXT = "word " * 2000


def _record(workspace: Path, artifact_id: str, *, family: str, title: str, text: str = REAL_TEXT,
            sha: str | None = None, **extra: Any) -> dict[str, Any]:
    path = workspace / "text" / f"{artifact_id.replace(':', '_').replace('/', '_')}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "artifact_id": artifact_id,
        "source_family_id": family,
        "paper_title": title,
        "source_sha256": sha or artifact_id,
        "indexable_text": True,
        "text_path": path.relative_to(workspace).as_posix(),
        **extra,
    }


def test_main_track_copy_is_kept_over_findings_and_workshop_copies(tmp_path: Path) -> None:
    main = _record(tmp_path, "paper:a", family="paper_text", title="Dexterous World Models", authors=["A. B"])
    findings = _record(tmp_path, "proceedings_paper:f:1", family="paper_text", title="Dexterous  world-models!",
                       venue_track="findings", authors=["A. B"])
    workshop = _record(tmp_path, "workshop_paper:w:1", family="workshop_paper_text", title="Dexterous World Models",
                       workshop_id="w", workshop_title="Hands Workshop", text=REAL_TEXT * 3)
    before = chunk_record_for_artifact(main)
    report = curation.curate_paper_records(tmp_path, [workshop, findings, main])

    assert main["indexable_text"] and "index_exclusion" not in main
    assert not findings["indexable_text"] and not workshop["indexable_text"]
    assert workshop["index_exclusion"] == {
        "reason": "duplicate_title", "policy": curation.POLICY_VERSION, "kept_artifact_id": "paper:a"
    }
    assert [ref["artifact_id"] for ref in main["also_presented_at"]] == ["proceedings_paper:f:1", "workshop_paper:w:1"]
    assert main["also_presented_at"][1]["workshop_title"] == "Hands Workshop"
    assert chunk_record_for_artifact(main) == before, "kept copy's chunk payload must not change"
    assert report["duplicate_groups"] == 1 and report["excluded_counts"] == {"duplicate_title": 2}


def test_identical_pdf_across_workshops_keeps_one_with_authors(tmp_path: Path) -> None:
    first = _record(tmp_path, "workshop_paper:a:1", family="workshop_paper_text", title="Title One", sha="same")
    second = _record(tmp_path, "workshop_paper:b:1", family="workshop_paper_text", title="Different Listing Title",
                     sha="same", authors=["C. D"])
    report = curation.curate_paper_records(tmp_path, [first, second])
    assert second["indexable_text"] and not first["indexable_text"]
    assert first["index_exclusion"]["reason"] == "duplicate_pdf"
    assert report["excluded_counts"] == {"duplicate_pdf": 1}


def test_image_only_text_is_excluded_before_dedupe(tmp_path: Path) -> None:
    watermark = "This CVPR paper is the Open Access version, provided by the Computer Vision Foundation."
    scanned_main = _record(tmp_path, "paper:s", family="paper_text", title="Scanned Paper", text=watermark)
    workshop_copy = _record(tmp_path, "workshop_paper:w:2", family="workshop_paper_text", title="Scanned Paper")
    unrelated = _record(tmp_path, "paper:u", family="paper_text", title="Unrelated")
    transcript = {"artifact_id": "t", "source_family_id": "speech_transcript", "indexable_text": True}
    report = curation.curate_paper_records(tmp_path, [scanned_main, workshop_copy, unrelated, transcript])

    assert scanned_main["index_exclusion"]["reason"] == "image_only_text_layer"
    assert workshop_copy["indexable_text"], "the text-bearing copy survives"
    assert unrelated["indexable_text"] and transcript["indexable_text"] and "index_exclusion" not in transcript
    assert report["candidates"] == 3 and report["kept"] == 2
    assert report["excluded"][0]["text_bytes"] == len(watermark)


def test_non_indexable_records_are_ignored(tmp_path: Path) -> None:
    record = {"artifact_id": "paper:x", "source_family_id": "paper_text", "indexable_text": False, "paper_title": "X"}
    other = _record(tmp_path, "paper:y", family="paper_text", title="X")
    report = curation.curate_paper_records(tmp_path, [record, other])
    assert report["excluded"] == [] and other["indexable_text"] and "index_exclusion" not in record
