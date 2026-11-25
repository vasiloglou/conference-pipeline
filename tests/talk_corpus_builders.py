"""Workspace builders for the talk-admission corpus (v6.0-Public-Release Phase 5.1).

Shared by ``tests/test_talk_admission.py`` and ``tests/test_talk_coverage_guard.py``. Each
builder writes the shapes ``tests/fixtures/talk_admission/README.md`` describes, under a
``tmp_path`` workspace. Nothing here reads a real workspace or the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PRESENTATIONS = Path("exports") / "deep-dive" / "openai_claude" / "presentations"
MANIFEST = Path("exports") / "deep-dive" / "manifest.json"
V15_RECEIPT = Path("storage") / "transcription-ocr" / "v15_transcription_ocr_receipts.json"
V15_EVENTS = Path("storage") / "acquisition" / "v15_source_inventory.json"
TALK_SOURCES = Path("storage") / "transcription-ocr" / "talk_sources.json"
TALK_EXCLUSIONS = Path("storage") / "transcription-ocr" / "talk_exclusions.json"
INVENTORY = Path("storage") / "corpus" / "corpus_inventory.json"
HALLUCINATION = "Remember to Like, Comment and Subscribe! https://www.youtube.com " * 40


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def url(kind: str, event_id: str) -> str:
    return f"https://example.org/virtual/2025/{kind}/{event_id.split('-')[-1]}"


def transcript_text(event_id: str) -> str:
    return f"Hello everyone, this is the talk for event {event_id}. " + "We study a method. " * 30


def presentation_file(event_id: str) -> str:
    return (PRESENTATIONS / f"talk-{event_id}.txt").as_posix()


def add_presentation(
    root: Path,
    event_id: str,
    *,
    kind: str,
    title: str | None = None,
    slideslive: str | None = None,
    transcript: str | None = "default",
) -> str:
    """Write one deep-dive presentation file and list it in the manifest; return its path."""

    meta = {
        "id": event_id,
        "title": title or f"Talk {event_id}",
        "authors": [],
        "track": kind,
        "neurips_url": url(kind, event_id),
        "slideslive_id": slideslive,
        "has_video": slideslive is not None,
        "status": "assets_downloaded",
    }
    lines = [f"# {meta['title']}", f"Presentation ID: talk-{event_id}", "", "## Metadata", "", "```json",
             json.dumps(meta, indent=2), "```", ""]
    if transcript is not None:
        text = transcript_text(event_id) if transcript == "default" else transcript
        lines += ["## Audio Transcript", "", text, ""]
    lines += ["## Transcript Segments (Raw JSON)", "", "```json", "[]", "```", "", "## Slide OCR", "", "(none)", ""]
    rel = presentation_file(event_id)
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text("\n".join(lines), encoding="utf-8")
    manifest = read_json(root / MANIFEST) if (root / MANIFEST).is_file() else {"presentations": []}
    manifest["presentations"].append(
        {"id": f"talk-{event_id}", "title": meta["title"], "file": f"openai_claude/presentations/talk-{event_id}.txt"}
    )
    write_json(root / MANIFEST, manifest)
    return rel


def add_manifest_entry_without_file(root: Path, event_id: str) -> None:
    manifest = read_json(root / MANIFEST) if (root / MANIFEST).is_file() else {"presentations": []}
    manifest["presentations"].append(
        {"id": f"talk-{event_id}", "title": f"Talk {event_id}", "file": f"openai_claude/presentations/talk-{event_id}.txt"}
    )
    write_json(root / MANIFEST, manifest)


def v15_transcript_path(item_id: str) -> str:
    return f"storage/transcription-ocr/media/{item_id}/transcript.txt"


def add_v15(root: Path, item_id: str, *, kind: str, text: str | None = "default", write_file: bool = True) -> str:
    """Add one transcribed event to the v1.5 receipt and the event inventory; return its transcript path."""

    receipt = read_json(root / V15_RECEIPT) if (root / V15_RECEIPT).is_file() else {
        "schema_version": 1, "run_id": "fixture", "media_artifacts": []}
    rel = v15_transcript_path(item_id)
    receipt["media_artifacts"].append(
        {
            "artifact_class": "speech_transcript",
            "artifact_id": f"media:{item_id}:raw.m4a",
            "provider": "openai",
            "provider_model": "asr-model-x",
            "source_path": f"cache/acquisition-downloads/media/{item_id}/audio/raw.m4a",
            "status": "transcribed",
            "transcript_path": rel,
        }
    )
    write_json(root / V15_RECEIPT, receipt)
    events = read_json(root / V15_EVENTS) if (root / V15_EVENTS).is_file() else {"items": []}
    events["items"].append({"item_id": item_id, "kind": kind, "source_url": url(kind, item_id), "included": True})
    write_json(root / V15_EVENTS, events)
    if write_file:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(transcript_text(item_id) if text == "default" else (text or ""), encoding="utf-8")
    return rel


def add_v15_non_talk(root: Path, item_id: str) -> None:
    receipt = read_json(root / V15_RECEIPT)
    receipt["media_artifacts"].append(
        {"artifact_class": "poster_asset", "artifact_id": f"media:{item_id}:poster.png", "provider": None,
         "provider_model": None, "source_path": f"cache/acquisition-downloads/media/{item_id}/poster.png",
         "status": "image_artifact_indexed", "transcript_path": None}
    )
    write_json(root / V15_RECEIPT, receipt)


def declare_sources(root: Path, *, deep_dive: bool = False, v15: bool = False) -> None:
    sources = []
    if deep_dive:
        sources.append({"lane_id": "talks-a", "format": "deep_dive_presentations", "path": MANIFEST.as_posix(),
                        "transcript_source": "slideslive_asr_export"})
    if v15:
        sources.append({"lane_id": "talks-b", "format": "transcription_receipt_v15", "path": V15_RECEIPT.as_posix(),
                        "event_inventory": V15_EVENTS.as_posix(), "transcript_source": "session_asr"})
    write_json(root / TALK_SOURCES, {"schema_version": 1, "sources": sources})


def exclude(root: Path, event_id: str, reason: str) -> None:
    current = read_json(root / TALK_EXCLUSIONS) if (root / TALK_EXCLUSIONS).is_file() else {"schema_version": 1, "exclusions": []}
    current["exclusions"].append({"event_id": event_id, "reason": reason, "ruling": "fixture"})
    write_json(root / TALK_EXCLUSIONS, current)


def paper_base(root: Path, *, papers: int = 2) -> None:
    rows = [
        {"id": f"AAAAAAAAA{i}", "forum": f"AAAAAAAAA{i}", "title": f"Main Paper Number {i} About Something",
         "authors": [f"Author {i}"], "abstract": f"Abstract {i}.", "track": "main"}
        for i in range(1, papers + 1)
    ]
    write_json(root / "storage" / "acquisition" / "papers.json", {"papers": rows})
    write_json(
        root / "storage" / "papers" / "late_paper_lanes.json",
        {"schema_version": 1,
         "existing_corpus": {"adapter": "openreview_acquisition_json", "path": "storage/acquisition/papers.json"},
         "lanes": []},
    )


def speech_record(*, transcript_path: str | None = None, source_path: str | None = None, provenance: list[str] | None = None,
                  index_exclusion: str | None = None, artifact_id: str = "speech:1") -> dict[str, Any]:
    record: dict[str, Any] = {"artifact_id": artifact_id, "source_family_id": "speech_transcript",
                              "indexable_text": index_exclusion is None, "provenance": provenance or []}
    if transcript_path is not None:
        record["transcript_path"] = transcript_path
    if source_path is not None:
        record["source_path"] = source_path
    if index_exclusion is not None:
        record["index_exclusion"] = index_exclusion
    return record


def write_inventory(root: Path, records: list[dict[str, Any]]) -> None:
    write_json(root / INVENTORY, {"schema_version": 1, "artifact_records": records})


def touch(root: Path, rel: str, text: str = "some words of a transcript\n") -> str:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text, encoding="utf-8")
    return rel
