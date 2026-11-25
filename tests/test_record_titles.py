"""Record titling for any conference: papers, slides, workshop papers, recordings.

The fixture is a made-up conference ("ZETA"), so nothing here depends on one
conference's names or site layout beyond the platform's own storage layout.
"""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.indexing import corpus_vector_store as cvs
from conference_pipeline.indexing import paper_corpus_curation as curation
from conference_pipeline.indexing import record_titles as rt
from conference_pipeline.indexing.chunking import chunks_for_record


def _page(title: str, body: str = "") -> str:
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


def _workspace(tmp_path: Path) -> tuple[Path, dict]:
    ws = tmp_path / "conf"
    probes = ws / "cache" / "probes"
    probes.mkdir(parents=True)
    pages = {
        "poster-501": ("poster", "ZETA Poster Sparse Routers Are Enough", None),
        "virtual-2031-801": (None, "ZETA Tutorial Learning With Proofs", "vid-tut"),
        "invited-talk-9": ("invited-talk", "ZETA Invited Talk The Next Decade", "vid-key"),
        "workshop-77": (
            "workshop",
            "Structured Probabilistic Inference &amp; Generative Modeling",
            "vid-ws",
        ),
        "poster-502": ("poster", "ZETA Poster Quiet Poster", "vid-silent"),
    }
    assets = []
    for item_id, (kind, title, video) in pages.items():
        body = '<a href="https://openreview.net/forum?id=note1">p</a>' if item_id == "workshop-77" else ""
        (probes / f"{item_id}.html").write_text(_page(title, body), encoding="utf-8")
        assets.append({
            "item_id": item_id, "kind": kind, "slideslive_id": video,
            "page_url": f"https://zeta.example/virtual/2031/{item_id}",
            "page_artifact_path": f"cache/probes/{item_id}.html",
        })
    recovery = ws / rt.RECOVERY_RELATIVE_PATH
    recovery.parent.mkdir(parents=True)
    recovery.write_text(json.dumps({"page_assets": assets}), encoding="utf-8")
    submissions = ws / "storage/download-corpus/submissions.json"
    submissions.write_text(json.dumps({"papers": [{
        "id": "abc123", "title": "Sparse Routers Are Enough", "authors": ["Ada Lovelace", "Alan Turing"],
        "keywords": ["mixture of experts", "routing"], "primary_area": "deep learning", "venue": "ZETA 2031",
    }]}), encoding="utf-8")
    receipts = ws / "storage/workshops/papers/zeta_spigm/receipts"
    receipts.mkdir(parents=True)
    (receipts / "note1.json").write_text(json.dumps({
        "note_id": "note1", "venueid": "ZETA/2031/Workshop/SPIGM", "title": "Flows for Graphs",
        "authors": "['Grace Hopper']",
    }), encoding="utf-8")
    (ws / "storage/workshops/inventory.json").write_text(
        json.dumps({"workshops": [{"venueid": "ZETA/2031/Workshop/SPIGM"}]}), encoding="utf-8"
    )
    download_corpus = {"openreview_import": {"local_path": "storage/download-corpus/submissions.json"}}
    return ws, download_corpus


def test_site_prefix_is_learned_not_named() -> None:
    titles = ["ZETA Poster A", "ZETA Tutorial B", "Structured Things Workshop", "ZETA Oral C"]
    assert rt.site_prefix(titles) == "ZETA"
    assert rt.split_event_title("ZETA Tutorial Learning With Proofs", kind=None, prefix="ZETA") == (
        "Tutorial: Learning With Proofs", "Learning With Proofs",
    )
    # A page with no site prefix keeps its first word.
    assert rt.split_event_title("Structured Things Workshop", kind="workshop", prefix="ZETA") == (
        "Workshop: Structured Things Workshop", "Structured Things Workshop",
    )
    # "Workshop on X" names the workshop; it is not a type label.
    assert rt.split_event_title("Workshop on Proofs", kind="workshop", prefix="ZETA")[1] == "Workshop on Proofs"
    assert rt.split_event_title("Track: Oral 1A Agents", kind="oral")[0] == "Oral Session: Oral 1A Agents"


def test_every_family_gets_a_title_with_provenance(tmp_path: Path) -> None:
    ws, download_corpus = _workspace(tmp_path)
    (ws / "text").mkdir()
    (ws / "text/header.txt").write_text("A Paper Found Only By Its Header\nJane Doe1, John Roe2\nAbstract\n...", encoding="utf-8")
    records = [
        {"artifact_id": "p1", "source_family_id": "paper_text", "source_path": "a.pdf", "metadata": {},
         "provenance": {"download_artifact": {"source_url": "https://api.example/attachment?id=abc123&name=pdf"}}},
        {"artifact_id": "p2", "source_family_id": "paper_text", "source_path": "b.pdf", "metadata": {},
         "provenance": {"download_artifact": {"source_url": "https://zeta.example/media/Slides/501.pdf"}}},
        {"artifact_id": "p3", "source_family_id": "paper_text", "source_path": "c.pdf", "metadata": {},
         "text_path": "text/header.txt", "provenance": {"download_artifact": {"source_url": "https://elsewhere/x.pdf"}}},
        {"artifact_id": "w1", "source_family_id": "workshop_paper_text",
         "source_path": "storage/workshops/papers/zeta_spigm/note1.pdf", "metadata": {}},
        {"artifact_id": "s1", "source_family_id": "speech_transcript", "metadata": {"slideslive_id": "vid-tut"}},
        {"artifact_id": "s2", "source_family_id": "speech_transcript", "metadata": {"slideslive_id": "vid-key"}},
        {"artifact_id": "s3", "source_family_id": "speech_transcript", "metadata": {"slideslive_id": "vid-ws"}},
    ]
    summary = rt.title_corpus_records(ws, records, download_corpus=download_corpus)
    by_id = {r["artifact_id"]: r for r in records}
    assert by_id["p1"]["paper_title"] == "Sparse Routers Are Enough"
    assert by_id["p1"]["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert by_id["p1"]["keywords"] == ["mixture of experts", "routing"]
    assert by_id["p2"]["paper_title"] == "Sparse Routers Are Enough" and by_id["p2"]["material_type"] == "slides"
    assert by_id["p3"]["paper_title"] == "A Paper Found Only By Its Header"
    assert by_id["p3"]["source_metadata_provenance"]["title"] == rt.PROVENANCE_PDF_HEADER
    assert "authors" not in by_id["p3"]  # header author lines are not trusted
    # Workshop paper: exact join to the workshop page by submission id.
    assert by_id["w1"]["paper_title"] == "Flows for Graphs"
    assert by_id["w1"]["authors"] == ["Grace Hopper"]
    assert by_id["w1"]["workshop_title"] == "Structured Probabilistic Inference & Generative Modeling"
    assert by_id["s1"]["title"] == "Tutorial: Learning With Proofs"
    assert by_id["s1"]["metadata"]["item_kind"] == "tutorial"
    assert by_id["s2"]["title"] == "Invited Talk: The Next Decade"
    assert by_id["s3"]["title"].startswith("Workshop: Structured Probabilistic")
    assert summary["speech_transcript"] == {
        "titled": 3, "untitled": 0, "untitled_sample": [], "by_provenance": {rt.PROVENANCE_EVENT_PAGE: 3},
    }
    assert summary["paper_text"]["titled"] == 3


def test_workshop_names_fall_back_to_acronyms(tmp_path: Path) -> None:
    ws, download_corpus = _workspace(tmp_path)
    (ws / "storage/workshops/papers/zeta_spigm/receipts/note1.json").unlink()  # no exact join left
    sources = rt.load_title_sources(ws, download_corpus)
    assert sources.workshop_names["ZETA/2031/Workshop/SPIGM"] == "Structured Probabilistic Inference & Generative Modeling"


def test_reviewed_speechless_transcripts_become_audit_only() -> None:
    records = [
        {"artifact_id": "s1", "source_family_id": "speech_transcript", "indexable_text": True,
         "metadata": {"slideslive_id": "vid-silent"}},
        {"artifact_id": "s2", "source_family_id": "speech_transcript", "indexable_text": True,
         "metadata": {"slideslive_id": "vid-key"}},
    ]
    excluded = rt.apply_transcript_exclusions(
        records, {"vid-silent": {"reason": "speechless_recording", "evidence": "'Hmm.'"}}
    )
    assert excluded == [{"artifact_id": "s1", "video_id": "vid-silent", "reason": "speechless_recording"}]
    assert records[0]["indexable_text"] is False and records[0]["index_exclusion"]["reason"] == "speechless_recording"
    assert records[1]["indexable_text"] is True


def test_curation_links_slides_instead_of_dropping_them(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "t/paper.txt").write_text("x" * 9000, encoding="utf-8")
    (tmp_path / "t/slides.txt").write_text("y" * 900, encoding="utf-8")
    (tmp_path / "t/ws.txt").write_text("z" * 9000, encoding="utf-8")
    records = [
        {"artifact_id": "paper", "source_family_id": "paper_text", "indexable_text": True, "text_path": "t/paper.txt",
         "paper_title": "Sparse Routers", "source_sha256": "a", "material_type": "paper"},
        {"artifact_id": "slides", "source_family_id": "paper_text", "indexable_text": True, "text_path": "t/slides.txt",
         "paper_title": "Sparse Routers", "source_sha256": "b", "material_type": "slides"},
        {"artifact_id": "ws-copy", "source_family_id": "workshop_paper_text", "indexable_text": True,
         "text_path": "t/ws.txt", "paper_title": "Sparse Routers", "source_sha256": "c"},
    ]
    report = curation.curate_paper_records(tmp_path, records)
    by_id = {r["artifact_id"]: r for r in records}
    assert by_id["slides"]["indexable_text"] is True  # short, but slides; not a title duplicate
    assert by_id["paper"]["indexable_text"] is True
    assert by_id["ws-copy"]["indexable_text"] is False  # workshop copy of a main paper
    assert report["excluded_counts"] == {"duplicate_title": 1}


def test_passage_headers_carry_the_titles(tmp_path: Path) -> None:
    record = {
        "artifact_id": "p1", "source_family_id": "paper_text", "paper_title": "Sparse Routers Are Enough",
        "authors": [f"Author {i}" for i in range(12)], "keywords": ["routing", "moe"], "material_type": "slides",
        "workshop_title": "Structured Probabilistic Inference", "metadata": {},
    }
    payload = cvs.chunk_record_for_artifact(record)
    text = chunks_for_record(payload, source_text="Body text about routers. " * 40, source_family="paper_text")[0].text
    assert text.startswith("Paper Title: Sparse Routers Are Enough\n")
    assert "Authors: Author 0, Author 1" in text and "Author 9, et al." in text and "Author 10" not in text
    assert "Keywords: routing, moe" in text
    assert "Workshop: Structured Probabilistic Inference" in text
    assert "Material: slides" in text
