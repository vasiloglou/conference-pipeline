"""Split multi-talk event recordings into named talks, as verifiable talk maps.

A conference's event pages (oral sessions, expo workshops, affinity events,
workshops) are often recorded as ONE long video holding several talks. Until
a recording is split, indexing sees one long document and analysis cannot name
the talks in it. This module turns such recordings into base-scope talk maps
(``indexing.talk_maps``) that Stage 6 verifies and attaches:

* **Which recordings.** Every event page with an embedded recording, from the
  SlidesLive recovery inventory (``storage/download-corpus/
  slideslive_media_recovery.json``), selected by event kind (read from the
  page asset, else from the cached page ``<title>``) and/or explicit item ids.
* **The program.** When an event page links to child item pages (an oral
  session links its orals, in program order), the child pages' titles become
  the scheduled talk list handed to the model, which then names each talk
  with its scheduled title verbatim.
* **The transcript.** Found through the transcription inventory's session
  speech records (``slideslive_id`` -> ``transcript_json_path``), so a
  recording whose audio was deduplicated still resolves.
* **The output.** ``storage/transcription-ocr/talk-maps/base/<video_id>_*``
  with ``video_id`` and ``source_transcript_sha256`` on the talks file and the
  talks digest on the annotated transcript, so a map made for a different or
  changed transcript never attaches.

Conference-neutral: nothing here names a conference; titles are stripped of
their site prefix generically.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.indexing import talk_maps
from conference_pipeline.program_coverage import classify_item_page_title_kind, item_page_title

RECOVERY_RELATIVE_PATH = Path("storage/download-corpus/slideslive_media_recovery.json")
TRANSCRIPTION_INVENTORY_RELATIVE_PATH = Path("storage/transcription-ocr/transcription_ocr_inventory.json")
LEGACY_WORKSHOP_SPLITS_RELATIVE_DIR = Path("storage/transcription-ocr/workshop-talks")
SPEECH_LANE = "session_speech_transcription"
DISCOVERY_METHOD_WITH_PROGRAM = "model_with_program_schedule"
DISCOVERY_METHOD_MODEL = "model_transcript_only"
# Untimed transcripts get evenly timed segments; a talk boundary can only fall
# between segments. One minute keeps a 15-minute oral's boundaries tight
# (the workshop default of five minutes could hand a third of one oral to the
# next) at the same prompt text.
EVENT_SYNTHETIC_SEGMENT_SECONDS = 60.0
# A split whose talks end well before the recording does is incomplete: the
# model sometimes returns only the first talk of a session (ICML 2026 rebuild,
# 2026-09-28: two oral sessions came back as one talk covering 1% and 22%).
MIN_TALK_COVERAGE = 0.8
INCOMPLETE_SUFFIX = ".incomplete"
CHILD_ITEM_KINDS = ("oral", "spotlight", "poster", "talk", "invited-talk", "paper")
_CHILD_LINK_RE = re.compile(r'href="(?:https?://[^"/]+)?(/virtual/(\d{4})/(%s)/(\d+))"' % "|".join(CHILD_ITEM_KINDS))


@dataclass(frozen=True)
class EventRecording:
    """One event page with a recording and a transcript."""

    item_id: str
    kind: str | None
    page_url: str
    page_path: Path
    title: str
    video_id: str
    transcript_json_path: Path
    transcript_path: Path
    transcript_sha256: str
    scheduled_talks: tuple[dict[str, str], ...] = field(default_factory=tuple)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(workspace: Path, value: object) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else workspace / path


def clean_item_title(raw_title: str | None, *, kind: str | None = None, strip_kind: bool = True) -> str:
    """A cached item page's ``<title>`` without the site prefix.

    Item pages title themselves ``<Site> <Kind> <Title>`` (for example
    ``<Site> Oral Do We Need Adam?``); session pages ``Track: <Title>``.
    A scheduled talk drops the kind too (``Do We Need Adam?``); an event keeps
    its type phrase (``Expo Workshop From Digital Agents ...``) because that
    phrase is part of how the event is named.
    """

    title = html.unescape(str(raw_title or "")).strip()
    title = re.sub(r"\s+", " ", title)
    if title.lower().startswith("track:"):
        return title.split(":", 1)[1].strip()
    if kind and strip_kind:
        words = kind.replace("-", " ")
        match = re.match(r"^\S+(?:\s+\d{4})?\s+%s\s+(.+)$" % re.escape(words), title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    if kind and not strip_kind:
        first_word = kind.split("-", 1)[0]
        match = re.match(r"^\S+(?:\s+\d{4})?\s+(%s\b.+)$" % re.escape(first_word), title, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return title


def _page_index(assets: Iterable[Mapping[str, Any]], workspace: Path) -> dict[str, Path]:
    """URL path (``/virtual/<year>/<kind>/<id>``) -> cached page file."""

    index: dict[str, Path] = {}
    for asset in assets:
        url = str(asset.get("page_url") or "")
        path = _resolve(workspace, asset.get("page_artifact_path"))
        if not url or path is None:
            continue
        index[re.sub(r"^https?://[^/]+", "", url).rstrip("/")] = path
    return index


def scheduled_talks_from_page(page_html: str, *, page_index: Mapping[str, Path], own_path: str) -> list[dict[str, str]]:
    """Titles of the child items an event page links to, in page order."""

    talks: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in _CHILD_LINK_RE.finditer(page_html):
        path, _year, kind, _item = match.groups()
        if path == own_path or path in seen:
            continue
        seen.add(path)
        child = page_index.get(path)
        if child is None or not child.is_file():
            continue
        title = clean_item_title(item_page_title(child.read_text(encoding="utf-8", errors="ignore")), kind=kind)
        if title:
            talks.append({"title": title, "item_path": path, "item_kind": kind})
    return talks


def _speech_records_by_video(workspace: Path) -> dict[str, Mapping[str, Any]]:
    inventory = _read_json(workspace / TRANSCRIPTION_INVENTORY_RELATIVE_PATH) or {}
    lane = ((inventory.get("lanes") or {}).get(SPEECH_LANE) or {}) if isinstance(inventory, Mapping) else {}
    records: dict[str, Mapping[str, Any]] = {}
    for record in lane.get("records") or []:
        if isinstance(record, Mapping) and record.get("slideslive_id") and record.get("status") == "succeeded":
            records.setdefault(str(record["slideslive_id"]), record)
    return records


def load_event_recordings(
    workspace: Path,
    *,
    kinds: Sequence[str] = (),
    item_ids: Sequence[str] = (),
) -> tuple[list[EventRecording], list[dict[str, Any]]]:
    """Event recordings selected by kind and/or item id; returns ``(recordings, skipped)``."""

    workspace = workspace.resolve()
    recovery = _read_json(workspace / RECOVERY_RELATIVE_PATH) or {}
    assets = [a for a in recovery.get("page_assets") or [] if isinstance(a, Mapping)]
    page_index = _page_index(assets, workspace)
    speech = _speech_records_by_video(workspace)
    wanted_kinds = {kind for kind in kinds if kind}
    wanted_ids = {item for item in item_ids if item}
    found_ids: set[str] = set()
    recordings: list[EventRecording] = []
    skipped: list[dict[str, Any]] = []
    for asset in assets:
        video_id = str(asset.get("slideslive_id") or "")
        item_id = str(asset.get("item_id") or "")
        if not video_id:
            continue
        page_path = _resolve(workspace, asset.get("page_artifact_path"))
        page_html = page_path.read_text(encoding="utf-8", errors="ignore") if page_path and page_path.is_file() else ""
        kind = asset.get("kind") or classify_item_page_title_kind(page_html)
        if item_id not in wanted_ids and (kind not in wanted_kinds):
            continue
        found_ids.add(item_id)
        record = speech.get(video_id)
        if record is None:
            skipped.append({"item_id": item_id, "video_id": video_id, "reason": "no_succeeded_transcript"})
            continue
        transcript_json = _resolve(workspace, record.get("transcript_json_path"))
        transcript = _resolve(workspace, record.get("transcript_path"))
        if transcript_json is None or transcript is None or not transcript_json.is_file() or not transcript.is_file():
            skipped.append({"item_id": item_id, "video_id": video_id, "reason": "transcript_file_missing"})
            continue
        url = str(asset.get("page_url") or "")
        own_path = re.sub(r"^https?://[^/]+", "", url).rstrip("/")
        recordings.append(
            EventRecording(
                item_id=item_id,
                kind=str(kind) if kind else None,
                page_url=url,
                page_path=page_path or Path(),
                title=clean_item_title(item_page_title(page_html), kind=str(kind) if kind else None, strip_kind=False)
                or item_id,
                video_id=video_id,
                transcript_json_path=transcript_json,
                transcript_path=transcript,
                transcript_sha256=sha256_file(transcript),
                scheduled_talks=tuple(
                    scheduled_talks_from_page(page_html, page_index=page_index, own_path=own_path)
                ),
            )
        )
    for item_id in sorted(wanted_ids - found_ids):
        skipped.append({"item_id": item_id, "reason": "item_not_found_or_no_recording"})
    recordings.sort(key=lambda item: item.item_id)
    return recordings, skipped


def legacy_workshop_identity(workspace: Path, item_id: str) -> dict[str, str]:
    """The id/title an earlier workshop split used, so a re-split keeps them."""

    number = item_id.rsplit("-", 1)[-1]
    directory = workspace / LEGACY_WORKSHOP_SPLITS_RELATIVE_DIR
    for path in sorted(directory.glob("*-talks.json")) if directory.is_dir() else []:
        payload = _read_json(path) or {}
        workshop_id = str(payload.get("workshop_id") or "")
        if workshop_id and re.search(r"(?:^|\D)%s$" % re.escape(number), workshop_id):
            return {"workshop_id": workshop_id, "workshop_title": str(payload.get("workshop_title") or "")}
    return {}


def event_identity(workspace: Path, recording: EventRecording) -> dict[str, str]:
    if recording.kind == "workshop":
        legacy = legacy_workshop_identity(workspace, recording.item_id)
        if legacy:
            return legacy
    return {"workshop_id": recording.item_id, "workshop_title": recording.title}


def talk_map_is_current(workspace: Path, recording: EventRecording) -> bool:
    files = talk_maps.talk_map_files(workspace, talk_maps.BASE_SCOPE, recording.video_id)
    _binding, code = talk_maps.verify_talk_map_chain(
        workspace,
        talks_path=files["talks"],
        annotated_path=files["annotated"],
        video_id=recording.video_id,
        transcript_sha256=recording.transcript_sha256,
    )
    return code is None


def estimate_prompt_tokens(recording: EventRecording) -> int:
    """Rough prompt size: transcript characters / 4, the program, the
    instructions, and ~12 tokens of timestamp label per one-minute segment."""

    characters = recording.transcript_path.stat().st_size
    program = sum(len(talk["title"]) for talk in recording.scheduled_talks)
    minutes = characters / 900  # ~150 spoken words a minute, ~6 characters a word
    return int((characters + program) / 4 + 12 * minutes) + 1500


def split_recording(
    workspace: Path,
    recording: EventRecording,
    *,
    provider_cost_approved: bool,
    run_id: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Discover talks for one recording and write its base-scope talk map."""

    from conference_pipeline.workshop_talk_segmentation import run_workshop_talk_boundary_discovery

    workspace = workspace.resolve()
    files = talk_maps.talk_map_files(workspace, talk_maps.BASE_SCOPE, recording.video_id)
    identity = event_identity(workspace, recording)
    scheduled = [{"title": talk["title"]} for talk in recording.scheduled_talks]
    program_hints: dict[str, Any] = {
        "event_item_id": recording.item_id,
        "event_kind": recording.kind,
        "event_page_url": recording.page_url,
        "event_title": recording.title,
        "scheduled_talks": list(recording.scheduled_talks),
    }
    if recording.kind == "workshop":
        program_hints.update(identity)
    talks_extra = {
        "video_id": recording.video_id,
        "source_transcript_sha256": recording.transcript_sha256,
        "source_transcript_path": str(recording.transcript_path.relative_to(workspace)),
        "discovery_method": DISCOVERY_METHOD_WITH_PROGRAM if scheduled else DISCOVERY_METHOD_MODEL,
        "workshop_association": "event_url",
        "program_hints": program_hints,
    }
    annotated_extra = {
        "video_id": recording.video_id,
        "source_transcript_sha256": recording.transcript_sha256,
    }
    receipt = run_workshop_talk_boundary_discovery(
        workspace=workspace,
        input_transcript_json=recording.transcript_json_path,
        output_talks_json=files["talks"],
        output_batches_json=files["batches"],
        annotated_transcript_json=files["annotated"],
        receipt_path=files["receipt"],
        workshop_id=identity["workshop_id"],
        workshop_title=identity["workshop_title"],
        provider=provider,
        model=model,
        provider_cost_approved=provider_cost_approved,
        run_id=run_id,
        client=client,
        scheduled_talks=scheduled,
        talks_extra=talks_extra,
        annotated_extra=annotated_extra,
        synthetic_segment_seconds=EVENT_SYNTHETIC_SEGMENT_SECONDS,
    )
    summary = {
        "item_id": recording.item_id,
        "video_id": recording.video_id,
        "kind": recording.kind,
        "title": recording.title,
        "status": receipt.get("status"),
        "talk_count": receipt.get("talk_count"),
        "scheduled_talk_count": len(scheduled),
        "estimated_cost_usd": receipt.get("estimated_cost_usd"),
        "blocker_code": receipt.get("blocker_code"),
    }
    if receipt.get("status") != "succeeded":
        return summary
    talks_payload = _read_json(files["talks"]) or {}
    coverage = talk_coverage(files["talks"], files["annotated"])
    summary["coverage"] = coverage
    if scheduled:
        found = {str(t.get("talk_title") or "").strip().lower() for t in talks_payload.get("talks") or []}
        summary["scheduled_titles_matched"] = sum(1 for t in scheduled if t["title"].strip().lower() in found)
    if coverage < MIN_TALK_COVERAGE:
        # Set the map aside (renamed, never deleted) so Stage 6 cannot attach
        # it and --resume retries the recording.
        for key in ("talks", "annotated"):
            if files[key].is_file():
                files[key].rename(files[key].with_name(files[key].name + INCOMPLETE_SUFFIX))
        summary["status"] = "incomplete"
        summary["blocker_code"] = "talk_coverage_below_threshold"
    return summary


def talk_coverage(talks_path: Path, annotated_path: Path) -> float:
    """Share of the recording (by its last segment end) that the talks reach."""

    talks = (_read_json(talks_path) or {}).get("talks") or []
    segments = (_read_json(annotated_path) or {}).get("segments") or []
    if not talks or not segments:
        return 0.0
    total = max(float(segment.get("end") or segment.get("timestamp_end") or 0) for segment in segments)
    reached = max(float(talk.get("timestamp_end") or 0) for talk in talks)
    return round(min(reached / total, 1.0), 3) if total > 0 else 0.0


SUBSTEP_ID = "event-talk-splits"
SUBSTEP_RECEIPT_NAME = "event-talk-splits-batch-receipt.json"


def write_split_batch_receipt(workspace: Path, receipt: dict[str, Any], *, run_id: str) -> Path:
    """Persist a split batch as the run's ``event-talk-splits`` lineage sub-step.

    The receipt binds the talk maps (``outputs`` in the registry) to the
    transcription they were split from; corpus-inventory-index reads the maps
    as a registered input, so a re-split marks the index stale.
    """

    from conference_pipeline.receipt_lineage import sanitize_receipt_for_write, stamp_substep_receipt
    from conference_pipeline.workflow_runs import write_json_atomic

    stamp_substep_receipt(receipt, workspace=workspace, run_id=run_id, substep_id="event-talk-splits")
    path = Path(workspace) / "storage" / "workflow-runs" / run_id / SUBSTEP_RECEIPT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, sanitize_receipt_for_write(receipt))
    return path


__all__ = [
    "SUBSTEP_ID",
    "SUBSTEP_RECEIPT_NAME",
    "write_split_batch_receipt",
    "DISCOVERY_METHOD_MODEL",
    "DISCOVERY_METHOD_WITH_PROGRAM",
    "EventRecording",
    "clean_item_title",
    "estimate_prompt_tokens",
    "event_identity",
    "legacy_workshop_identity",
    "load_event_recordings",
    "scheduled_talks_from_page",
    "split_recording",
    "talk_coverage",
    "talk_map_is_current",
]
