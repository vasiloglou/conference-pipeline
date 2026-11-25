"""Talk maps for long multi-talk recordings: locations and chain verification.

A split recording stays one artifact. Its talk map (``<video_id>_talks.json``)
names the transcript digest it was discovered from; its annotated transcript
(``<video_id>_annotated.json``) stamps talk identity onto every segment and
names the talk map's digest. The index uses the annotated transcript only when
the whole chain holds:

    transcript bytes --sha--> talks.source_transcript_sha256
    talks bytes      --sha--> annotated.talks_sha256
    video_id         ==       talks.video_id == annotated.video_id

Talk maps live next to the transcripts they split: in a lane's ``talks/``
directory for evidence-lane recordings, and under
``storage/transcription-ocr/talk-maps/base/`` for the base caption inventory.
``attach_talk_maps`` binds maps to corpus records at admission time, so any
record family (base captions, any lane) can be split without re-registering
its lane. Conference-neutral.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.indexing.corpus_vector_store import resolve_workspace_member, workspace_relpath

LANES_RELATIVE_DIR = Path("storage/transcription-ocr/evidence-lanes")
TALK_MAPS_RELATIVE_DIR = Path("storage/transcription-ocr/talk-maps")
TALKS_DIR_NAME = "talks"
BASE_SCOPE = "base"
FALLBACK_DISCOVERY_METHOD = "deterministic_whole_recording"
TALK_SUMMARY_KEYS = ("talk_id", "talk_title", "talk_speaker", "talk_kind", "timestamp_start", "timestamp_end")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def talk_map_scope(record: Mapping[str, Any]) -> str:
    return str(record.get("evidence_lane_id") or BASE_SCOPE)


def talk_map_dir(workspace: Path, scope: str) -> Path:
    if scope == BASE_SCOPE:
        return workspace / TALK_MAPS_RELATIVE_DIR / BASE_SCOPE
    return workspace / LANES_RELATIVE_DIR / scope / TALKS_DIR_NAME


def talk_map_files(workspace: Path, scope: str, video_id: str) -> dict[str, Path]:
    directory = talk_map_dir(workspace, scope)
    return {
        "talks": directory / f"{video_id}_talks.json",
        "annotated": directory / f"{video_id}_annotated.json",
        "batches": directory / f"{video_id}_batches.json",
        "receipt": directory / f"{video_id}_discovery_receipt.json",
    }


def authoritative_workshop(talks_payload: Mapping[str, Any]) -> dict[str, str]:
    """Workshop identity only when the map says it came from the event page."""

    if talks_payload.get("workshop_association") != "event_url":
        return {}
    hints = talks_payload.get("program_hints") if isinstance(talks_payload.get("program_hints"), Mapping) else {}
    out: dict[str, str] = {}
    for key in ("workshop_id", "workshop_title"):
        if hints.get(key):
            out[key] = str(hints[key])
    return out


def talk_summary(talks_payload: Mapping[str, Any]) -> dict[str, Any]:
    talks = [talk for talk in talks_payload.get("talks") or [] if isinstance(talk, Mapping)]
    return {
        "talk_count": len(talks),
        "talk_discovery_method": talks_payload.get("discovery_method"),
        "talks": [
            {key: talk.get(key) for key in TALK_SUMMARY_KEYS if talk.get(key) is not None}
            for talk in talks
        ],
        **authoritative_workshop(talks_payload),
    }


def verify_talk_map_chain(
    workspace: Path,
    *,
    talks_path: Path,
    annotated_path: Path,
    video_id: str,
    transcript_sha256: str,
) -> tuple[dict[str, Any], str | None]:
    """Return ``(binding, blocker_code)``; the binding is empty on any failure."""

    if not talks_path.is_file():
        return {}, "talks_json_missing"
    if not annotated_path.is_file():
        return {}, "annotated_transcript_missing"
    talks_payload = _read_json(talks_path)
    if talks_payload is None:
        return {}, "talks_json_unreadable"
    if talks_payload.get("video_id") not in (None, video_id):
        return {}, "talk_map_video_id_mismatch"
    if talks_payload.get("source_transcript_sha256") != transcript_sha256:
        return {}, "talk_map_stale_for_transcript"
    annotated_payload = _read_json(annotated_path)
    if annotated_payload is None or not isinstance(annotated_payload.get("segments"), list):
        return {}, "annotated_transcript_not_json_segments"
    if annotated_payload.get("video_id") not in (None, video_id):
        return {}, "annotated_transcript_video_id_mismatch"
    talks_sha256 = _sha256_file(talks_path)
    if annotated_payload.get("talks_sha256") != talks_sha256:
        return {}, "annotated_transcript_not_bound_to_talk_map"
    annotated_source = annotated_payload.get("source_transcript_sha256")
    if annotated_source is not None and annotated_source != transcript_sha256:
        return {}, "annotated_transcript_stale_for_transcript"
    return {
        "annotated_transcript_json_path": workspace_relpath(workspace, annotated_path),
        "annotated_sha256": _sha256_file(annotated_path),
        "talks_json_path": workspace_relpath(workspace, talks_path),
        "talks_sha256": talks_sha256,
        "discovery_method": talks_payload.get("discovery_method"),
        "discovery_algorithm_version": talks_payload.get("discovery_algorithm_version"),
        "_talks_payload": talks_payload,
    }, None


def attach_talk_maps(
    workspace: Path,
    records: Iterable[Mapping[str, Any]],
    *,
    speech_source_family: str = "speech_transcript",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Point speech records at verified talk maps; returns ``(records, bindings, blockers)``.

    Records that already carry an annotated transcript (bound by their lane
    receipt) are left alone. A whole-recording fallback map adds no talk
    identity and is not attached. A map whose chain does not verify is a
    blocker, never silently ignored.
    """

    out: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        out.append(item)
        if item.get("source_family_id") != speech_source_family or item.get("annotated_transcript_json_path"):
            continue
        video_id = str(item.get("video_id") or "")
        if not video_id:
            continue
        scope = talk_map_scope(item)
        files = talk_map_files(workspace, scope, video_id)
        if not files["talks"].is_file():
            continue
        talks_payload = _read_json(files["talks"]) or {}
        if talks_payload.get("discovery_method") == FALLBACK_DISCOVERY_METHOD:
            continue
        transcript, error = resolve_workspace_member(workspace, item.get("transcript_path"))
        if error or transcript is None or not transcript.is_file():
            continue
        binding, code = verify_talk_map_chain(
            workspace,
            talks_path=files["talks"],
            annotated_path=files["annotated"],
            video_id=video_id,
            transcript_sha256=_sha256_file(transcript),
        )
        if code is not None:
            blockers.append(
                {"code": code, "artifact_id": item.get("artifact_id"), "video_id": video_id, "scope": scope}
            )
            continue
        payload = binding.pop("_talks_payload")
        item["annotated_transcript_json_path"] = binding["annotated_transcript_json_path"]
        item["talks_json_path"] = binding["talks_json_path"]
        item.update(talk_summary(payload))
        bindings.append({"artifact_id": item.get("artifact_id"), "video_id": video_id, "scope": scope, **binding})
    return out, bindings, blockers


__all__ = [
    "BASE_SCOPE",
    "FALLBACK_DISCOVERY_METHOD",
    "LANES_RELATIVE_DIR",
    "TALK_MAPS_RELATIVE_DIR",
    "TALKS_DIR_NAME",
    "attach_talk_maps",
    "authoritative_workshop",
    "talk_map_dir",
    "talk_map_files",
    "talk_map_scope",
    "talk_summary",
    "verify_talk_map_chain",
]
