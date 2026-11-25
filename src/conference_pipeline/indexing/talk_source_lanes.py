"""Bind a workspace's existing talk transcripts into transcript evidence lanes.

v6.0-Public-Release Phase 5.1 (spec ``notes/phase-5.1-spec.md`` section 6.3). Some
workspaces hold talk transcripts that no corpus-inventory stage ever admitted: a
deep-dive export of SlidesLive talks, or a transcription receipt written by an older
driver. This module reads each source a workspace declares in
``storage/transcription-ocr/talk_sources.json``, applies the named exclusions in
``storage/transcription-ocr/talk_exclusions.json``, keeps one record per recording, and
writes one evidence lane per source through ``indexing.transcript_evidence_lanes``. The
corpus inventory then admits the lane records like any other lane.

Readers are named by the source's format, never by a conference:

- ``deep_dive_presentations``: a manifest listing presentation files. Each file has a
  fenced JSON metadata block and an ``## Audio Transcript`` section;
- ``transcription_receipt_v15``: a receipt whose ``media_artifacts`` name
  ``speech_transcript`` files, plus an event inventory giving each item's kind and URL.

Order (its one home is the spec's section 6.3), per source:

1. read every event; a declared file that is missing refuses the bind;
2. exclude: a named exclusion, then an absent or blank transcript (``empty_transcript``),
   then all but one event per recording (``duplicate_recording``); the kept event has the
   smallest event id, compared as a string;
3. materialize the kept transcripts (deep-dive text is written into the lane; a receipt's
   existing file is used as it is);
4. write the lane receipt and register it. A rebind whose receipt differs only in its
   timestamp keeps the old bytes and leaves the registry alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.indexing import transcript_evidence_lanes as lanes
from conference_pipeline.indexing.paper_lanes import PaperLaneError

SOURCES_PATH = Path("storage") / "transcription-ocr" / "talk_sources.json"
EXCLUSIONS_PATH = Path("storage") / "transcription-ocr" / "talk_exclusions.json"
EMPTY_TRANSCRIPT = "empty_transcript"
DUPLICATE_RECORDING = "duplicate_recording"
PRODUCER = "conference_pipeline.indexing.talk_source_lanes"

_METADATA_BLOCK = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)
_TRANSCRIPT_HEADING = "## Audio Transcript"
_RECORDING_URL_KEY = "slideslive_url"
# A lane id or recording id becomes a path component; anything else is refused before any
# write (review phase-5.1-r1 F1).
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}")

__all__ = [
    "BindResult",
    "DUPLICATE_RECORDING",
    "EMPTY_TRANSCRIPT",
    "PaperLaneError",
    "READERS",
    "TalkEvent",
    "bind_talk_sources",
    "load_talk_exclusions",
    "load_talk_sources",
]


@dataclass(frozen=True)
class TalkEvent:
    event_id: str
    recording_id: str
    item_kind: str | None
    title: str | None
    event_url: str | None
    source_path: str
    text: str | None
    transcript_path: str | None
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class BindResult:
    lanes: Mapping[str, Mapping[str, int]]
    written: bool
    changed_lanes: tuple[str, ...] = field(default_factory=tuple)


def _read_json(path: Path, what: str) -> Any:
    if not path.is_file():
        raise PaperLaneError(f"{what} missing: {path.as_posix()}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PaperLaneError(f"{what} is not valid JSON: {path.as_posix()} ({exc})") from exc


def _lexical(workspace: Path, path: Path | str) -> tuple[Path, bool]:
    """``(normalized absolute path, inside)`` without resolving links: a workspace may link a
    folder into another store (a Drive folder), and that folder is still the workspace's
    (finding F14). Only ``..`` can leave the workspace."""
    root = os.path.normpath(os.path.abspath(workspace))
    full = os.path.normpath(os.path.join(root, os.fspath(path)))
    return Path(full), full == root or full.startswith(root + os.sep)


def _inside(workspace: Path, rel: str, what: str) -> Path:
    path, inside = _lexical(workspace, rel)
    if not inside:
        raise PaperLaneError(f"{what} escapes the workspace: {rel}")
    return path


def _rel(workspace: Path, path: Path) -> str:
    full, inside = _lexical(workspace, path)
    if not inside:
        raise PaperLaneError(f"path escapes the workspace: {path}")
    return Path(os.path.relpath(full, os.path.normpath(os.path.abspath(workspace)))).as_posix()


def load_talk_sources(workspace: Path) -> tuple[Mapping[str, Any], ...]:
    payload = _read_json(workspace / SOURCES_PATH, "talk sources")
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or not sources:
        raise PaperLaneError(f"{SOURCES_PATH.as_posix()} declares no sources")
    seen: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or not source.get("lane_id") or not source.get("path"):
            raise PaperLaneError("a talk source needs lane_id and path")
        if source.get("format") not in READERS:
            raise PaperLaneError(f"unknown talk source format: {source.get('format')!r}")
        if not _safe(str(source["lane_id"])):
            raise PaperLaneError(f"unsafe talk lane id: {source['lane_id']!r}")
        if source["lane_id"] in seen:
            raise PaperLaneError(f"duplicate talk lane id: {source['lane_id']}")
        seen.add(source["lane_id"])
    return tuple(sources)


def _safe(name: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(name)) and ".." not in name


def load_talk_exclusions(workspace: Path) -> dict[str, str]:
    path = workspace / EXCLUSIONS_PATH
    if not path.is_file():
        return {}
    rows = _read_json(path, "talk exclusions").get("exclusions") or []
    out: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("event_id") or not row.get("reason"):
            raise PaperLaneError("a talk exclusion needs event_id and reason")
        out[str(row["event_id"])] = str(row["reason"])
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _event_url(meta: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    """The source's own URL field when declared, else its first ``*_url`` field that is not the recording's."""
    declared = source.get("event_url_field")
    if declared:
        value = meta.get(declared)
        return str(value) if value else None
    for key, value in meta.items():
        if key.endswith("_url") and key != _RECORDING_URL_KEY and value:
            return str(value)
    return None


def _transcript_section(text: str) -> str | None:
    if _TRANSCRIPT_HEADING not in text:
        return None
    body = text.split(_TRANSCRIPT_HEADING, 1)[1]
    return body.split("\n## ", 1)[0].strip()


def _read_deep_dive(workspace: Path, source: Mapping[str, Any]) -> list[TalkEvent]:
    manifest_path = _inside(workspace, str(source["path"]), "talk manifest")
    manifest = _read_json(manifest_path, "talk manifest")
    entries = manifest.get("presentations") if isinstance(manifest, dict) else None
    if not isinstance(entries, list):
        raise PaperLaneError(f"talk manifest has no presentations list: {source['path']}")
    events: list[TalkEvent] = []
    for entry in entries:
        file_path, inside = _lexical(workspace, manifest_path.parent / str(entry.get("file") or ""))
        if not inside:
            raise PaperLaneError(f"presentation escapes the workspace: {entry.get('file')}")
        if not file_path.is_file():
            raise PaperLaneError(f"presentation file missing: {_rel(workspace, file_path)}")
        text = file_path.read_text(encoding="utf-8")
        block = _METADATA_BLOCK.search(text)
        if block is None:
            raise PaperLaneError(f"presentation has no metadata block: {_rel(workspace, file_path)}")
        meta = json.loads(block.group(1))
        event_id = str(meta.get("id") or "").strip()
        if not event_id:
            raise PaperLaneError(f"presentation has no id: {_rel(workspace, file_path)}")
        recording = str(meta.get("slideslive_id") or "").strip()
        events.append(
            TalkEvent(
                event_id=event_id,
                recording_id=recording or f"event:{event_id}",
                item_kind=str(meta["track"]) if meta.get("track") else None,
                title=str(meta["title"]) if meta.get("title") else None,
                event_url=_event_url(meta, source),
                source_path=_rel(workspace, file_path),
                text=_transcript_section(text),
                transcript_path=None,
                provenance=(f"slideslive:{recording}",) if recording else (),
            )
        )
    return events


def _item_id(artifact: Mapping[str, Any]) -> str:
    parts = str(artifact.get("artifact_id") or "").split(":")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return Path(str(artifact.get("transcript_path") or "")).parent.name


def _read_transcription_receipt(workspace: Path, source: Mapping[str, Any]) -> list[TalkEvent]:
    receipt = _read_json(_inside(workspace, str(source["path"]), "transcription receipt"), "transcription receipt")
    artifacts = receipt.get("media_artifacts") if isinstance(receipt, dict) else None
    if not isinstance(artifacts, list):
        raise PaperLaneError(f"transcription receipt has no media_artifacts: {source['path']}")
    items: dict[str, Mapping[str, Any]] = {}
    if source.get("event_inventory"):
        inventory = _read_json(_inside(workspace, str(source["event_inventory"]), "event inventory"), "event inventory")
        for item in inventory.get("items") or []:
            if isinstance(item, dict) and item.get("item_id"):
                items[str(item["item_id"])] = item
    events: list[TalkEvent] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("artifact_class") != lanes.SPEECH_SOURCE_FAMILY:
            continue
        if artifact.get("status") != "transcribed" or not artifact.get("transcript_path"):
            continue
        rel = str(artifact["transcript_path"])
        path = _inside(workspace, rel, "transcript")
        if not path.is_file():
            raise PaperLaneError(f"transcript missing: {rel}")
        item_id = _item_id(artifact)
        item = items.get(item_id, {})
        model = artifact.get("provider_model")
        events.append(
            TalkEvent(
                event_id=item_id,
                recording_id=f"event:{item_id}",
                item_kind=str(item["kind"]) if item.get("kind") else None,
                title=str(item["title"]) if item.get("title") else None,
                event_url=str(item["source_url"]) if item.get("source_url") else None,
                source_path=str(source["path"]),
                text=path.read_text(encoding="utf-8", errors="replace"),
                transcript_path=_rel(workspace, path),
                provenance=(f"asr_model:{model}",) if model else (),
            )
        )
    return events


READERS: Mapping[str, Callable[[Path, Mapping[str, Any]], list[TalkEvent]]] = {
    "deep_dive_presentations": _read_deep_dive,
    "transcription_receipt_v15": _read_transcription_receipt,
}


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    kept: list[tuple[TalkEvent, list[TalkEvent]]]
    gaps: list[tuple[TalkEvent, str]]


def _plan(events: list[TalkEvent], exclusions: Mapping[str, str]) -> _Plan:
    gaps: list[tuple[TalkEvent, str]] = []
    remaining: list[TalkEvent] = []
    for event in events:
        if event.event_id in exclusions:
            gaps.append((event, exclusions[event.event_id]))
        elif not (event.text or "").strip():
            gaps.append((event, EMPTY_TRANSCRIPT))
        else:
            remaining.append(event)
    groups: dict[str, list[TalkEvent]] = {}
    for event in remaining:
        groups.setdefault(event.recording_id, []).append(event)
    kept: list[tuple[TalkEvent, list[TalkEvent]]] = []
    for recording_id, group in groups.items():
        if not _safe(recording_id):
            raise PaperLaneError(f"unsafe recording id: {recording_id!r}")
        group = sorted(group, key=lambda e: e.event_id)
        kept.append((group[0], group[1:]))
        gaps.extend((other, DUPLICATE_RECORDING) for other in group[1:])
    return _Plan(kept=kept, gaps=gaps)


def _lane_record(workspace: Path, lane_id: str, kept: TalkEvent, others: list[TalkEvent], *, write: bool) -> dict[str, Any]:
    if kept.transcript_path is not None:
        transcript_path = kept.transcript_path
    else:
        target = lanes.lane_dir(workspace, lane_id) / "transcripts" / f"{kept.recording_id}.txt"
        text = (kept.text or "").strip()
        if write and (not target.is_file() or target.read_text(encoding="utf-8") != text):
            lanes.write_plain_transcript(workspace, lane_id, kept.recording_id, text)
        transcript_path = _rel(workspace, target)
    covered = [kept, *others]
    return {
        "video_id": kept.recording_id,
        "transcript_path": transcript_path,
        "item_kind": kept.item_kind,
        "title": kept.title,
        "source_event_urls": [e.event_url for e in covered if e.event_url],
        "provenance": [f"source:{kept.source_path}", *(f"event:{e.event_id}" for e in covered), *kept.provenance],
    }


def _same_but_time(old: Mapping[str, Any], new: Mapping[str, Any]) -> bool:
    strip = lambda payload: {k: v for k, v in payload.items() if k != "generated_at"}  # noqa: E731
    return strip(old) == strip(new)


def _registry_pins(workspace: Path, lane_id: str, receipt_bytes: bytes) -> bool:
    path = lanes.registry_path(workspace)
    if not path.is_file():
        return False
    registry = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(receipt_bytes).hexdigest()
    return any(
        isinstance(lane, dict) and lane.get("lane_id") == lane_id and lane.get("receipt_sha256") == digest
        for lane in registry.get("lanes", [])
    )


def bind_talk_sources(workspace: Path, *, dry_run: bool = False) -> BindResult:
    """Bind every declared talk source into its evidence lane; write nothing when ``dry_run``."""
    workspace = Path(os.path.abspath(workspace))
    sources = load_talk_sources(workspace)
    exclusions = load_talk_exclusions(workspace)
    plans = [(source, _plan(READERS[source["format"]](workspace, source), exclusions)) for source in sources]
    counts = {
        str(source["lane_id"]): {"records": len(plan.kept), "coverage_gaps": len(plan.gaps)}
        for source, plan in plans
    }
    if dry_run:
        return BindResult(lanes=counts, written=False)

    changed: list[str] = []
    for source, plan in plans:
        lane_id = str(source["lane_id"])
        records = [_lane_record(workspace, lane_id, kept, others, write=True) for kept, others in plan.kept]
        gap_rows = [{"video_id": f"event:{event.event_id}", "reason": reason} for event, reason in plan.gaps]
        extra = {
            # The file that identifies the event: its own transcript when the source points at
            # one (a receipt), else the file the transcript came from (finding F9).
            "gap_sources": {f"event:{event.event_id}": event.transcript_path or event.source_path for event, _ in plan.gaps},
            "gap_kinds": {f"event:{event.event_id}": event.item_kind for event, _ in plan.gaps if event.item_kind},
            "talk_source": {k: source[k] for k in sorted(source) if k != "lane_id"},
        }
        receipt_file = lanes.lane_receipt_path(workspace, lane_id)
        old_bytes = receipt_file.read_bytes() if receipt_file.is_file() else None
        new_receipt, blockers = lanes.build_lane_receipt(
            workspace=workspace,
            lane_id=lane_id,
            conference=workspace.name,
            producer=PRODUCER,
            transcript_source=str(source.get("transcript_source") or "unspecified"),
            records=records,
            coverage_gaps=gap_rows,
            extra=extra,
        )
        if blockers:
            if old_bytes is not None:
                receipt_file.write_bytes(old_bytes)
            raise PaperLaneError(f"lane {lane_id} refused: {blockers}")
        if old_bytes is not None and _same_but_time(json.loads(old_bytes), new_receipt):
            receipt_file.write_bytes(old_bytes)
        else:
            changed.append(lane_id)
        if not _registry_pins(workspace, lane_id, receipt_file.read_bytes()):
            lanes.register_lane(workspace=workspace, lane_id=lane_id)
    return BindResult(lanes=counts, written=True, changed_lanes=tuple(changed))
