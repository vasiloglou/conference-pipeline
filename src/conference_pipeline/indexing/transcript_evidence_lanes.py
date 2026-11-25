"""Appendable, digest-bound transcript evidence lanes (conference-neutral).

Why this exists
---------------
A corpus-inventory stage that admits speech transcripts only from one
immutable upstream inventory cannot absorb evidence that arrives later:
captions that YouTube publishes a day after the harvest, an ASR fallback
run, a hand-recovered keynote. On CVPR 2026 every one of those recovery
runs landed in a parallel directory that nothing consumed, so three
plenary keynotes and ~460 paid ASR transcripts never reached the index.

An *evidence lane* is the fix: a small, digest-bound receipt that lists
plain-text transcripts recovered by some later run. Any number of lanes can
be registered for a workspace. The corpus-inventory stage of any conference
runner asks this module for the admitted lane records, dedups them against
what the base inventory already admitted (first lane in precedence order
wins), and appends them as ordinary ``speech_transcript`` artifact records.
The base inventory is never edited and its contract never changes.

Layout (all paths workspace-relative)::

    storage/transcription-ocr/evidence_lanes.json          # registry
    storage/transcription-ocr/evidence-lanes/<lane_id>/lane_receipt.json

Registry::

    {"schema_version": 1,
     "receipt_type": "conference_pipeline.transcript_evidence_lane_registry",
     "lanes": [{"lane_id": "...", "receipt_path": "...", "receipt_sha256": "..."}]}

Lanes are listed in precedence order (a caption lane before an ASR lane,
for example). The registry pins each lane receipt's sha256, and each lane
receipt pins each transcript file's sha256, so a tampered or drifted file
is rejected instead of indexed.

Lane receipt::

    {"schema_version": 1,
     "receipt_type": "conference_pipeline.transcript_evidence_lane",
     "lane_id": "...", "conference": "...", "generated_at": "...",
     "producer": "...", "transcript_source": "youtube_caption_auto|asr_fallback|...",
     "records": [{"video_id": "...", "transcript_path": "...", "sha256": "...",
                  "transcript_source": "...", "language": "en",
                  "word_count": 123, "source_event_urls": [...],
                  "provenance": [...], "title": "..."}],
     "coverage_gaps": [{"video_id": "...", "reason": "silent_audio"}],
     "counts": {"records": N, "coverage_gaps": M}}

Everything here is local and fail-closed: no network, no provider, and any
inconsistency between the registry, a receipt, and the bytes on disk is a
blocker rather than a silent skip.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from conference_pipeline.indexing import talk_maps
from conference_pipeline.indexing.corpus_vector_store import (
    SPEECH_SOURCE_FAMILY,
    resolve_workspace_member,
    workspace_relpath,
)

REGISTRY_RELATIVE_PATH = Path("storage/transcription-ocr/evidence_lanes.json")
LANES_RELATIVE_DIR = Path("storage/transcription-ocr/evidence-lanes")
REGISTRY_RECEIPT_TYPE = "conference_pipeline.transcript_evidence_lane_registry"
LANE_RECEIPT_TYPE = "conference_pipeline.transcript_evidence_lane"
SCHEMA_VERSION = 1

_SHA256_HEX_LENGTH = 64

# Optional per-record metadata a producer may attach; carried through verbatim.
EXTENDED_RECORD_FIELDS = (
    "item_kind",
    "duration_seconds",
    "timing_kind",
    "normalization",
    "channel",
    "channel_category",
    "talks",
    "talk_count",
    "talk_discovery_method",
    "workshop_id",
    "workshop_title",
)
TALKS_DIR_NAME = "talks"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_HEX_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _blocker(code: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "affected_stage": "transcript-evidence-lanes", **details}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Producing a lane
# ---------------------------------------------------------------------------


def lane_dir(workspace: Path, lane_id: str) -> Path:
    return workspace / LANES_RELATIVE_DIR / lane_id


def lane_receipt_path(workspace: Path, lane_id: str) -> Path:
    return lane_dir(workspace, lane_id) / "lane_receipt.json"


def registry_path(workspace: Path) -> Path:
    return workspace / REGISTRY_RELATIVE_PATH


def write_plain_transcript(workspace: Path, lane_id: str, video_id: str, text: str) -> Path:
    """Materialize a plain-text transcript inside the lane directory.

    The indexer chunks whatever bytes ``transcript_path`` points at, so a
    lane whose upstream sidecars are JSON (an ASR receipt with a ``text``
    field, say) must materialize plain text here rather than pointing at
    the JSON. Returns the written path.
    """

    target = lane_dir(workspace, lane_id) / "transcripts" / f"{video_id}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def write_transcript_json(workspace: Path, lane_id: str, video_id: str, payload: Mapping[str, Any]) -> Path:
    """Materialize a timestamped transcript JSON (``{"segments": [...]}``) in the lane.

    The platform chunker turns segment JSON into time-ranged chunks with
    timestamp locators, which plain text would lose. The payload must not
    carry a speech-transcript receipt type (the indexer rejects receipt JSON).
    """

    if str(payload.get("receipt_type") or "").endswith("speech_transcript"):
        raise ValueError("lane transcript JSON must not be a speech-transcript receipt")
    target = lane_dir(workspace, lane_id) / "transcripts" / f"{video_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return target


def talk_map_paths(workspace: Path, lane_id: str, video_id: str) -> tuple[Path, Path]:
    """``(talks_json, annotated_transcript_json)`` for one recording of a lane."""

    files = talk_maps.talk_map_files(workspace, lane_id, video_id)
    return files["talks"], files["annotated"]


def _bind_talk_map(
    workspace: Path,
    lane_id: str,
    video_id: str,
    record: Mapping[str, Any],
    transcript_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Digest-bind a talk map and its annotated transcript to one lane record."""

    annotated_value = record.get("annotated_transcript_json_path")
    if not annotated_value:
        return {}, []
    annotated, error = resolve_workspace_member(workspace, annotated_value)
    if error or annotated is None or not annotated.is_file():
        return {}, [_blocker("lane_annotated_transcript_missing", lane_id=lane_id, video_id=video_id)]
    talks, error = resolve_workspace_member(workspace, record.get("talks_json_path"))
    if error or talks is None or not talks.is_file():
        return {}, [_blocker("lane_talks_json_missing", lane_id=lane_id, video_id=video_id)]
    # The whole chain must hold: transcript -> talk map -> annotation, all for
    # this video. A map for one video with another video's annotation used to
    # bind without complaint.
    binding, code = talk_maps.verify_talk_map_chain(
        workspace,
        talks_path=talks,
        annotated_path=annotated,
        video_id=video_id,
        transcript_sha256=transcript_sha256,
    )
    if code is not None:
        return {}, [_blocker(f"lane_{code}", lane_id=lane_id, video_id=video_id)]
    return {
        "annotated_transcript_json_path": binding["annotated_transcript_json_path"],
        "annotated_sha256": binding["annotated_sha256"],
        "talks_json_path": binding["talks_json_path"],
        "talks_sha256": binding["talks_sha256"],
    }, []


def build_lane_receipt(
    *,
    workspace: Path,
    lane_id: str,
    conference: str,
    producer: str,
    transcript_source: str,
    records: Iterable[Mapping[str, Any]],
    coverage_gaps: Iterable[Mapping[str, Any]] = (),
    notes: str | None = None,
    bind_indexed_text_sha256: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Digest every transcript and write ``lane_receipt.json``.

    Each input record needs ``video_id`` and ``transcript_path`` (workspace
    relative or absolute inside the workspace); ``sha256`` is computed here
    from the bytes on disk, never trusted from the caller. Optional fields
    (``language``, ``word_count``, ``source_event_urls``, ``provenance``,
    ``title``, ``transcript_source``, and ``EXTENDED_RECORD_FIELDS``) are
    carried through. A record may name ``annotated_transcript_json_path`` and
    ``talks_json_path`` (a split multi-talk recording); both are digested and
    the talk map must name this transcript's digest. Returns
    ``(receipt, blockers)``; on any blocker nothing is written.

    ``bind_indexed_text_sha256`` makes admitted records carry the indexed
    text's digest into chunk rows. It is opt-in so existing lanes keep their
    row identity (and their vectors).
    """

    workspace = workspace.resolve()
    blockers: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        video_id = str(record.get("video_id") or "").strip()
        if not video_id:
            blockers.append(_blocker("lane_record_missing_video_id", lane_id=lane_id))
            continue
        if video_id in seen:
            blockers.append(_blocker("lane_record_duplicate_video_id", lane_id=lane_id, video_id=video_id))
            continue
        seen.add(video_id)
        resolved, error = resolve_workspace_member(workspace, record.get("transcript_path"))
        if error or resolved is None or not resolved.is_file():
            blockers.append(
                _blocker(
                    "lane_transcript_missing",
                    lane_id=lane_id,
                    video_id=video_id,
                    path=str(record.get("transcript_path")),
                    reason=error or "not_a_file",
                )
            )
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            blockers.append(_blocker("lane_transcript_empty", lane_id=lane_id, video_id=video_id))
            continue
        transcript_sha256 = _sha256_file(resolved)
        talk_binding, talk_blockers = _bind_talk_map(workspace, lane_id, video_id, record, transcript_sha256)
        if talk_blockers:
            blockers.extend(talk_blockers)
            continue
        extended = {key: record.get(key) for key in EXTENDED_RECORD_FIELDS if record.get(key) is not None}
        admitted.append(
            {
                **extended,
                **talk_binding,
                "video_id": video_id,
                "transcript_path": workspace_relpath(workspace, resolved),
                "sha256": transcript_sha256,
                "byte_count": resolved.stat().st_size,
                "transcript_source": str(record.get("transcript_source") or transcript_source),
                "language": record.get("language"),
                "word_count": record.get("word_count")
                if isinstance(record.get("word_count"), int)
                else len(text.split()),
                "source_event_urls": [str(url) for url in (record.get("source_event_urls") or [])],
                "provenance": [str(item) for item in (record.get("provenance") or [])],
                "title": record.get("title"),
            }
        )
    if blockers:
        return {}, blockers
    gaps = [
        {"video_id": str(gap.get("video_id") or ""), "reason": str(gap.get("reason") or "unspecified")}
        for gap in coverage_gaps
    ]
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": LANE_RECEIPT_TYPE,
        "lane_id": lane_id,
        "conference": conference,
        "generated_at": _utc_now(),
        "producer": producer,
        "transcript_source": transcript_source,
        "records": sorted(admitted, key=lambda item: item["video_id"]),
        "coverage_gaps": sorted(gaps, key=lambda item: item["video_id"]),
        "counts": {"records": len(admitted), "coverage_gaps": len(gaps)},
        "talk_mapped_records": sum(1 for item in admitted if item.get("annotated_transcript_json_path")),
        "bind_indexed_text_sha256": bool(bind_indexed_text_sha256),
        "boundaries": {
            "network_access": False,
            "provider_calls_made": False,
            "transcripts_fabricated": False,
            "secrets_copied": False,
        },
        "notes": notes,
        **dict(extra or {}),
    }
    _write_json(lane_receipt_path(workspace, lane_id), receipt)
    return receipt, []


def register_lane(*, workspace: Path, lane_id: str, position: int | None = None) -> dict[str, Any]:
    """Add (or re-pin) a lane in the registry, in precedence order.

    ``position`` inserts at that index (0 = highest precedence); ``None``
    appends. Re-registering an existing lane id re-pins its receipt digest
    in place. Returns the registry payload.
    """

    workspace = workspace.resolve()
    receipt_file = lane_receipt_path(workspace, lane_id)
    if not receipt_file.is_file():
        raise FileNotFoundError(f"lane receipt missing: {receipt_file}")
    entry = {
        "lane_id": lane_id,
        "receipt_path": workspace_relpath(workspace, receipt_file),
        "receipt_sha256": _sha256_file(receipt_file),
    }
    registry = _read_json(registry_path(workspace)) or {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": REGISTRY_RECEIPT_TYPE,
        "lanes": [],
    }
    lanes = [lane for lane in registry.get("lanes", []) if isinstance(lane, dict)]
    existing = next((index for index, lane in enumerate(lanes) if lane.get("lane_id") == lane_id), None)
    if existing is not None:
        lanes[existing] = entry
    elif position is None:
        lanes.append(entry)
    else:
        lanes.insert(position, entry)
    registry = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": REGISTRY_RECEIPT_TYPE,
        "generated_at": _utc_now(),
        "lanes": lanes,
    }
    _write_json(registry_path(workspace), registry)
    return registry


# ---------------------------------------------------------------------------
# Consuming lanes (what a corpus-inventory stage calls)
# ---------------------------------------------------------------------------


def load_admitted_lane_records(
    *,
    workspace: Path,
    already_admitted_video_ids: Iterable[str] = (),
    artifact_id_prefix: str = "evidence_lane_transcript",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(artifact_records, lane_bindings, coverage_gaps, blockers)``.

    * No registry file: nothing admitted, no blockers (lanes are optional).
    * Registry present: every listed lane must exist, match its pinned
      digest, and every record's transcript must exist, match its digest,
      and be non-empty. Any failure is a blocker (fail closed).
    * Precedence: the base inventory's ``already_admitted_video_ids`` win
      over every lane; then lanes in registry order; a video id that
      appears in a later lane is skipped and reported in that lane's
      binding as ``superseded_count``.

    ``artifact_records`` are in the corpus-inventory ``speech_transcript``
    shape the platform indexer consumes.
    """

    workspace = workspace.resolve()
    registry_file = registry_path(workspace)
    if not registry_file.is_file():
        return [], [], [], []
    registry = _read_json(registry_file)
    if registry is None or registry.get("receipt_type") != REGISTRY_RECEIPT_TYPE:
        return [], [], [], [_blocker("evidence_lane_registry_invalid", path=str(REGISTRY_RELATIVE_PATH))]
    lanes = registry.get("lanes")
    if not isinstance(lanes, list):
        return [], [], [], [_blocker("evidence_lane_registry_invalid", path=str(REGISTRY_RELATIVE_PATH))]

    records: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    claimed: set[str] = {str(video_id) for video_id in already_admitted_video_ids}
    seen_lane_ids: set[str] = set()

    for entry in lanes:
        if not isinstance(entry, Mapping):
            blockers.append(_blocker("evidence_lane_registry_entry_invalid"))
            continue
        lane_id = str(entry.get("lane_id") or "")
        if not lane_id or lane_id in seen_lane_ids:
            blockers.append(_blocker("evidence_lane_id_invalid_or_duplicate", lane_id=lane_id))
            continue
        seen_lane_ids.add(lane_id)
        receipt_file, error = resolve_workspace_member(workspace, entry.get("receipt_path"))
        if error or receipt_file is None or not receipt_file.is_file():
            blockers.append(_blocker("evidence_lane_receipt_missing", lane_id=lane_id, reason=error or "missing"))
            continue
        actual_digest = _sha256_file(receipt_file)
        if not _is_sha256(entry.get("receipt_sha256")) or actual_digest != entry.get("receipt_sha256"):
            blockers.append(_blocker("evidence_lane_receipt_digest_mismatch", lane_id=lane_id))
            continue
        receipt = _read_json(receipt_file)
        if (
            receipt is None
            or receipt.get("receipt_type") != LANE_RECEIPT_TYPE
            or receipt.get("lane_id") != lane_id
            or not isinstance(receipt.get("records"), list)
        ):
            blockers.append(_blocker("evidence_lane_receipt_invalid", lane_id=lane_id))
            continue

        admitted_here = 0
        superseded_here = 0
        lane_blockers_before = len(blockers)
        for record in receipt["records"]:
            if not isinstance(record, Mapping):
                blockers.append(_blocker("evidence_lane_record_invalid", lane_id=lane_id))
                continue
            video_id = str(record.get("video_id") or "")
            if not video_id:
                blockers.append(_blocker("evidence_lane_record_invalid", lane_id=lane_id))
                continue
            transcript_file, error = resolve_workspace_member(workspace, record.get("transcript_path"))
            if error or transcript_file is None or not transcript_file.is_file():
                blockers.append(
                    _blocker("evidence_lane_transcript_missing", lane_id=lane_id, video_id=video_id, reason=error or "missing")
                )
                continue
            if not _is_sha256(record.get("sha256")) or _sha256_file(transcript_file) != record.get("sha256"):
                blockers.append(_blocker("evidence_lane_transcript_digest_mismatch", lane_id=lane_id, video_id=video_id))
                continue
            if transcript_file.stat().st_size == 0:
                blockers.append(_blocker("evidence_lane_transcript_empty", lane_id=lane_id, video_id=video_id))
                continue
            annotated_fields: dict[str, Any] = {}
            if record.get("annotated_transcript_json_path"):
                annotated_file, error = resolve_workspace_member(workspace, record.get("annotated_transcript_json_path"))
                talks_file, talks_error = resolve_workspace_member(workspace, record.get("talks_json_path"))
                if (
                    error
                    or talks_error
                    or annotated_file is None
                    or talks_file is None
                    or not annotated_file.is_file()
                    or not talks_file.is_file()
                    or _sha256_file(annotated_file) != record.get("annotated_sha256")
                    or _sha256_file(talks_file) != record.get("talks_sha256")
                ):
                    blockers.append(
                        _blocker("evidence_lane_annotated_digest_mismatch", lane_id=lane_id, video_id=video_id)
                    )
                    continue
                annotated_fields = {
                    "annotated_transcript_json_path": workspace_relpath(workspace, annotated_file),
                    "indexed_text_kind": "annotated_transcript_json",
                    "talks_json_path": workspace_relpath(workspace, talks_file),
                }
                if receipt.get("bind_indexed_text_sha256") is True:
                    annotated_fields["indexed_text_sha256"] = str(record.get("annotated_sha256"))
            elif receipt.get("bind_indexed_text_sha256") is True:
                annotated_fields = {"indexed_text_sha256": str(record.get("sha256"))}
            if video_id in claimed:
                superseded_here += 1
                continue
            claimed.add(video_id)
            admitted_here += 1
            relative_path = workspace_relpath(workspace, transcript_file)
            records.append(
                {
                    "artifact_id": f"{artifact_id_prefix}:{lane_id}:{video_id}",
                    "source_family_id": SPEECH_SOURCE_FAMILY,
                    "source_path": relative_path,
                    "source_sha256": str(record.get("sha256")),
                    "indexable_text": True,
                    "transcript_path": relative_path,
                    "storage_lane": workspace_relpath(workspace, lane_dir(workspace, lane_id)),
                    "retention_class": SPEECH_SOURCE_FAMILY,
                    "video_id": video_id,
                    "source_event_urls": [str(url) for url in (record.get("source_event_urls") or [])],
                    "provenance": [str(item) for item in (record.get("provenance") or [])]
                    + [f"evidence_lane:{lane_id}"],
                    "transcript_source": record.get("transcript_source") or receipt.get("transcript_source"),
                    "evidence_lane_id": lane_id,
                    "title": record.get("title"),
                    **{
                        key: record.get(key)
                        for key in EXTENDED_RECORD_FIELDS
                        if record.get(key) is not None
                    },
                    **annotated_fields,
                }
            )
        for gap in receipt.get("coverage_gaps") or []:
            if isinstance(gap, Mapping) and gap.get("video_id"):
                gaps.append(
                    {
                        "video_id": str(gap.get("video_id")),
                        "reason": str(gap.get("reason") or "unspecified"),
                        "evidence_lane_id": lane_id,
                    }
                )
        bindings.append(
            {
                "lane_id": lane_id,
                "receipt_path": workspace_relpath(workspace, receipt_file),
                "receipt_sha256": actual_digest,
                "transcript_source": receipt.get("transcript_source"),
                "declared_count": len(receipt["records"]),
                "admitted_count": admitted_here,
                "superseded_count": superseded_here,
                "talk_mapped_count": sum(
                    1
                    for item in receipt["records"]
                    if isinstance(item, Mapping) and item.get("annotated_transcript_json_path")
                ),
                "blocker_count": len(blockers) - lane_blockers_before,
            }
        )
    return records, bindings, gaps, blockers


def summarize_registry(workspace: Path) -> dict[str, Any]:
    """Small read-only summary for reports and receipts (no validation)."""

    registry = _read_json(registry_path(workspace.resolve()))
    if registry is None:
        return {"registered": False, "lanes": []}
    return {
        "registered": True,
        "lanes": [
            {"lane_id": lane.get("lane_id"), "receipt_path": lane.get("receipt_path")}
            for lane in registry.get("lanes", [])
            if isinstance(lane, Mapping)
        ],
    }


__all__ = [
    "EXTENDED_RECORD_FIELDS",
    "LANE_RECEIPT_TYPE",
    "LANES_RELATIVE_DIR",
    "REGISTRY_RECEIPT_TYPE",
    "REGISTRY_RELATIVE_PATH",
    "build_lane_receipt",
    "lane_dir",
    "lane_receipt_path",
    "load_admitted_lane_records",
    "register_lane",
    "registry_path",
    "summarize_registry",
    "talk_map_paths",
    "write_plain_transcript",
    "write_transcript_json",
]
