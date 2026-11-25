"""Bind YouTube talk-harvest directories into transcript evidence lanes.

``acquisition.youtube_transcripts.transcribe_video_list`` writes one harvest
directory per run: ``video_transcript_inventory.json`` (one row per video with
its transcript sidecar path and digest), ``video_transcript_receipt.json``, and
``transcripts/<video_id>.json``. Nothing consumed those directories, so their
transcripts never reached the corpus index.

This module turns any such directory into digest-bound evidence lanes
(``indexing.transcript_evidence_lanes``), one per transcript tier:

* ``<label>-caption``: YouTube caption tracks. Auto captions are normalized
  with ``collapse_rolling_caption_segments`` (rolling cues otherwise repeat
  every phrase two or three times).
* ``<label>-gemini``: model transcriptions of the video URL, with window-level
  timing. Windows flagged as no-speech are dropped.

Each admitted transcript is materialized as timestamped segment JSON inside the
lane, so the chunker produces time-ranged chunks with timestamp locators. Raw
harvest sidecars are never modified; each lane record binds the raw digest.
Videos without usable text become typed coverage gaps, never records.

Conference-neutral: no conference names, hosts, or channel conventions here.
The event category is read from the event-page URL path when present.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.youtube_talks import collapse_rolling_caption_segments
from conference_pipeline.acquisition.youtube_transcripts import strip_no_speech_markers
from conference_pipeline.indexing import talk_maps
from conference_pipeline.indexing import transcript_evidence_lanes as lanes
from conference_pipeline.indexing.corpus_vector_store import workspace_relpath

HARVEST_INVENTORY_NAME = "video_transcript_inventory.json"
HARVEST_RECEIPT_NAME = "video_transcript_receipt.json"
CAPTION_SOURCES = frozenset({"youtube_caption_auto", "youtube_caption_manual"})
AUTO_CAPTION_SOURCE = "youtube_caption_auto"
TIER_CAPTION = "caption"
TIER_MODEL = "gemini"
# Precedence among lane tiers when registering: captions, then everything the
# registry already holds (for example governed ASR), then URL transcriptions.
TIER_ORDER = (TIER_CAPTION, TIER_MODEL)
FALLBACK_DISCOVERY_METHOD = "deterministic_whole_recording"
EVENT_CATEGORY_RE = re.compile(
    r"/(workshop|oral|tutorial|poster|session|meeting|invited|keynote|spotlight|event)s?/[A-Za-z0-9_-]+",
    re.IGNORECASE,
)


@dataclass
class HarvestLanePlan:
    harvest_dir: str
    label: str
    run_id: str | None
    receipt_status: str | None
    lanes: dict[str, dict[str, Any]] = field(default_factory=dict)
    coverage_gaps: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    normalization: dict[str, int] = field(default_factory=dict)
    suspect_transcripts: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "harvest_dir": self.harvest_dir,
            "label": self.label,
            "harvest_run_id": self.run_id,
            "harvest_receipt_status": self.receipt_status,
            "lanes": {
                tier: {"lane_id": lane["lane_id"], "records": len(lane["records"])}
                for tier, lane in self.lanes.items()
            },
            "coverage_gaps": len(self.coverage_gaps),
            "coverage_gap_reasons": _count_by(self.coverage_gaps, "reason"),
            "normalization": dict(self.normalization),
            "suspect_model_transcripts": self.suspect_transcripts,
            "blockers": self.blockers,
        }


def _count_by(items: list[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def default_label(harvest_dir: Path) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", harvest_dir.name).strip("-").lower()
    return f"yt-{slug}"


def normalize_inventory_path(workspace: Path, value: object) -> Path | None:
    """Resolve a harvest-recorded path that may be workspace- or repo-relative.

    Harvest inventories record paths relative to the process working
    directory (usually the repository root). Accept workspace-relative,
    ancestor-relative, or absolute paths, but only if they land inside the
    workspace.
    """

    text = str(value or "").strip()
    if not text:
        return None
    workspace = workspace.resolve()
    raw = Path(text)
    candidates = [raw] if raw.is_absolute() else [workspace / raw, *(ancestor / raw for ancestor in workspace.parents)]
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


def event_category(source_url: object, channel: object) -> str | None:
    match = EVENT_CATEGORY_RE.search(str(source_url or ""))
    if match:
        return match.group(1).lower()
    text = str(channel or "")
    if ":" in text:
        return text.split(":", 1)[1].strip().lower() or None
    return None


def transcript_tier(transcript_source: str) -> str:
    if transcript_source in CAPTION_SOURCES:
        return TIER_CAPTION
    if "gemini" in transcript_source or "url_transcription" in transcript_source:
        return TIER_MODEL
    return re.sub(r"[^a-z0-9]+", "-", transcript_source.lower()).strip("-") or "other"


def normalize_transcript_sidecar(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the lane's timestamped transcript JSON for one harvest sidecar."""

    source = str(payload.get("transcript_source") or "")
    raw_segments = [segment for segment in payload.get("segments") or [] if isinstance(segment, Mapping)]
    raw_text = str(payload.get("text") or "")
    if source == AUTO_CAPTION_SOURCE:
        segments = collapse_rolling_caption_segments([dict(segment) for segment in raw_segments])
        method = "collapse_rolling_captions"
        timing_kind = "caption_cue"
    elif source in CAPTION_SOURCES:
        segments = [dict(segment) for segment in raw_segments]
        method = "none"
        timing_kind = "caption_cue"
    else:
        segments = [
            {**segment, "text": strip_no_speech_markers(str(segment.get("text") or ""))}
            for segment in raw_segments
            if str(segment.get("text") or "").strip() and not segment.get("no_speech")
        ]
        method = "drop_no_speech_windows"
        timing_kind = "model_window"
    kept: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end_value = segment.get("end")
        end = float(end_value) if end_value is not None else start
        item: dict[str, Any] = {"start": round(start, 3), "end": round(max(end, start), 3), "text": text}
        for flag in ("truncated", "sparse"):
            if segment.get(flag):
                item[flag] = True
        kept.append(item)
    text = " ".join(segment["text"] for segment in kept)
    last_end = max((segment["end"] for segment in kept), default=0.0)
    duration = payload.get("duration_seconds")
    return {
        "schema_version": 1,
        "video_id": payload.get("video_id"),
        "video_url": payload.get("video_url"),
        "transcript_source": source,
        "language": payload.get("language"),
        "timing_kind": timing_kind,
        "duration_seconds_effective": round(max(float(duration or 0.0), last_end), 3),
        "segment_count": len(kept),
        "word_count": len(text.split()),
        "char_count": len(text),
        "normalization": {
            "method": method,
            "raw_segment_count": len(raw_segments),
            "raw_word_count": len(raw_text.split()),
            "raw_text_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        },
        "segments": kept,
        "text": text,
    }


def _gap(video: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {"video_id": str(video.get("video_id") or ""), "reason": reason}


def _provenance(video: Mapping[str, Any], sidecar: Mapping[str, Any], *, label: str, run_id: str | None) -> list[str]:
    items = [f"youtube_talk_harvest:{label}"]
    if run_id:
        items.append(f"harvest_run:{run_id}")
    for key, prefix in (("channel", "channel"), ("tier", "tier")):
        if video.get(key):
            items.append(f"{prefix}:{video[key]}")
    for key, value in sorted(video.items()):
        # Harvests name their match-evidence field per conference (``<slug>_evidence``).
        if key.endswith("_evidence") and value:
            items.append(f"match_evidence:{value}")
    source = str(sidecar.get("transcript_source") or "")
    if source:
        items.append(f"transcript_source:{source}")
    for key, prefix in (("provider", "provider"), ("model", "model"), ("task_id", "transcription_task")):
        if sidecar.get(key):
            items.append(f"{prefix}:{sidecar[key]}")
    return items


TRANSCRIPT_EXCLUSIONS_RELATIVE = Path("storage/transcription-ocr/evidence-lanes/transcript_exclusions.json")
SUSPECT_MAX_CHARS = 300
_TITLE_WORD_RE = re.compile(r"[a-z]{4,}")


def load_transcript_exclusions(workspace: Path) -> dict[str, dict[str, Any]]:
    """Reviewed transcripts that must not be indexed, keyed by video id.

    ``{"exclusions": [{"video_id", "reason", "evidence", "reviewed_at"}]}``.
    A model transcript can be fluent and still invented (a silent poster clip
    came back as "I'm going to show you how to make a delicious breakfast");
    such transcripts become coverage gaps, never evidence.
    """

    payload = _read_json(workspace / TRANSCRIPT_EXCLUSIONS_RELATIVE) or {}
    return {
        str(item["video_id"]): dict(item)
        for item in payload.get("exclusions") or []
        if isinstance(item, Mapping) and item.get("video_id") and item.get("reason")
    }


def suspect_model_transcript(text: str, title: str | None) -> bool:
    """Short model transcript sharing no word with its title: flag for review."""

    if len(text) > SUSPECT_MAX_CHARS:
        return False
    title_words = set(_TITLE_WORD_RE.findall((title or "").lower()))
    return not (title_words & set(_TITLE_WORD_RE.findall(text.lower())))


def plan_harvest_lanes(
    *,
    workspace: Path,
    harvest_dir: Path,
    label: str | None = None,
    write: bool = False,
) -> HarvestLanePlan:
    """Classify one harvest directory's videos into lane records and gaps."""

    workspace = workspace.resolve()
    harvest_dir = harvest_dir if harvest_dir.is_absolute() else workspace / harvest_dir
    label = label or default_label(harvest_dir)
    inventory = _read_json(harvest_dir / HARVEST_INVENTORY_NAME)
    receipt = _read_json(harvest_dir / HARVEST_RECEIPT_NAME)
    plan = HarvestLanePlan(
        harvest_dir=workspace_relpath(workspace, harvest_dir),
        label=label,
        run_id=(inventory or {}).get("run_id"),
        receipt_status=(receipt or {}).get("status"),
    )
    if inventory is None or not isinstance(inventory.get("videos"), list):
        plan.blockers.append({"code": "harvest_inventory_missing_or_invalid", "harvest_dir": plan.harvest_dir})
        return plan
    normalization = {"caption_raw_words": 0, "caption_normalized_words": 0, "model_windows_dropped": 0}
    exclusions = load_transcript_exclusions(workspace)
    seen: set[str] = set()
    for video in inventory["videos"]:
        if not isinstance(video, Mapping):
            plan.blockers.append({"code": "harvest_video_row_invalid", "harvest_dir": plan.harvest_dir})
            continue
        video_id = str(video.get("video_id") or "")
        if not video_id or video_id in seen:
            plan.blockers.append({"code": "harvest_video_id_missing_or_duplicate", "video_id": video_id})
            continue
        seen.add(video_id)
        status = str(video.get("status") or "")
        if status != "transcribed":
            reason = str(video.get("playability_status") or video.get("youtube_outcome") or status or "not_transcribed")
            plan.coverage_gaps.append(_gap(video, f"{status or 'not_transcribed'}:{reason}".lower()))
            continue
        transcript = video.get("transcript") if isinstance(video.get("transcript"), Mapping) else {}
        sidecar_path = normalize_inventory_path(workspace, transcript.get("transcript_path"))
        if sidecar_path is None:
            plan.blockers.append({"code": "harvest_transcript_sidecar_missing", "video_id": video_id})
            continue
        sidecar = _read_json(sidecar_path)
        if sidecar is None:
            plan.blockers.append({"code": "harvest_transcript_sidecar_invalid", "video_id": video_id})
            continue
        expected_sha = transcript.get("text_sha256")
        raw_text = str(sidecar.get("text") or "")
        if expected_sha and hashlib.sha256(raw_text.encode("utf-8")).hexdigest() != expected_sha:
            plan.blockers.append({"code": "harvest_transcript_digest_mismatch", "video_id": video_id})
            continue
        normalized = normalize_transcript_sidecar(sidecar)
        if not normalized["text"].strip():
            plan.coverage_gaps.append(_gap(video, "transcribed_without_speech"))
            continue
        if video_id in exclusions:
            plan.coverage_gaps.append(_gap(video, f"excluded:{exclusions[video_id]['reason']}"))
            continue
        source = str(normalized["transcript_source"])
        tier = transcript_tier(source)
        if tier != TIER_CAPTION and suspect_model_transcript(normalized["text"], video.get("title")):
            plan.suspect_transcripts.append(
                {"video_id": video_id, "title": video.get("title"), "text": normalized["text"][:SUSPECT_MAX_CHARS]}
            )
        if tier == TIER_CAPTION:
            normalization["caption_raw_words"] += int(normalized["normalization"]["raw_word_count"])
            normalization["caption_normalized_words"] += int(normalized["word_count"])
        else:
            normalization["model_windows_dropped"] += int(normalized["normalization"]["raw_segment_count"]) - int(
                normalized["segment_count"]
            )
        lane_id = f"{label}-{tier}"
        lane = plan.lanes.setdefault(tier, {"lane_id": lane_id, "transcript_source": source, "records": []})
        target = lanes.lane_dir(workspace, lane_id) / "transcripts" / f"{video_id}.json"
        if write:
            target = lanes.write_transcript_json(workspace, lane_id, video_id, normalized)
        talks_json, annotated_json = lanes.talk_map_paths(workspace, lane_id, video_id)
        record = {
            "video_id": video_id,
            "transcript_path": workspace_relpath(workspace, target),
            "transcript_source": source,
            "language": normalized.get("language"),
            "word_count": normalized["word_count"],
            "source_event_urls": [str(video["source_url"])] if video.get("source_url") else [],
            "provenance": _provenance(video, sidecar, label=label, run_id=plan.run_id),
            "title": video.get("title"),
            "duration_seconds": normalized["duration_seconds_effective"],
            "timing_kind": normalized["timing_kind"],
            "normalization": normalized["normalization"],
            "channel": video.get("channel"),
            "channel_category": event_category(video.get("source_url"), video.get("channel")),
            "_talk_map_paths": (talks_json, annotated_json),
        }
        lane["records"].append(record)
    plan.normalization = normalization
    return plan


def attach_existing_talk_maps(workspace: Path, plan: HarvestLanePlan) -> int:
    """Point records at talk maps that exist on disk; returns how many."""

    attached = 0
    for lane in plan.lanes.values():
        for record in lane["records"]:
            talks_json, annotated_json = record["_talk_map_paths"]
            if talks_json.is_file() and annotated_json.is_file():
                talks = _read_json(talks_json) or {}
                if talks.get("discovery_method") == FALLBACK_DISCOVERY_METHOD:
                    # A whole-recording fallback adds no talk identity; index the
                    # recording as a single talk rather than stamping its title twice.
                    continue
                record["talks_json_path"] = workspace_relpath(workspace, talks_json)
                record["annotated_transcript_json_path"] = workspace_relpath(workspace, annotated_json)
                # Workshop identity only from an event-page association; a
                # fuzzy title match never becomes record metadata.
                record.update(talk_maps.talk_summary(talks))
                attached += 1
    return attached


def _registry_tier_positions(workspace: Path) -> list[str]:
    """Tier of each registered lane, in registry order."""

    registry = _read_json(lanes.registry_path(workspace)) or {}
    tiers: list[str] = []
    for entry in registry.get("lanes") or []:
        receipt = _read_json(workspace / str(entry.get("receipt_path") or "")) or {}
        tiers.append(transcript_tier(str(receipt.get("transcript_source") or "")))
    return tiers


def bind_harvest_lanes(
    *,
    workspace: Path,
    plans: list[HarvestLanePlan],
    conference: str,
    producer: str,
    supersede_existing: bool = False,
) -> dict[str, Any]:
    """Write lane receipts and register them: captions first, model transcripts last.

    ``supersede_existing`` registers the new lanes at the head of the registry
    instead, so their transcripts win for any video an earlier lane also holds
    (for example English re-transcriptions of talks given in another language).
    Base inventory records still take precedence over every lane.
    """

    workspace = workspace.resolve()
    results: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for plan in plans:
        if plan.blockers:
            blockers.extend(plan.blockers)
            continue
        tiers = [tier for tier in TIER_ORDER if tier in plan.lanes] + [
            tier for tier in plan.lanes if tier not in TIER_ORDER
        ]
        if not tiers and plan.coverage_gaps:
            # A harvest with only gaps still records them, on an empty caption lane.
            plan.lanes[TIER_CAPTION] = {
                "lane_id": f"{plan.label}-{TIER_CAPTION}",
                "transcript_source": AUTO_CAPTION_SOURCE,
                "records": [],
            }
            tiers = [TIER_CAPTION]
        for index, tier in enumerate(tiers):
            lane = plan.lanes[tier]
            records = [
                {key: value for key, value in record.items() if not key.startswith("_")}
                for record in lane["records"]
            ]
            _receipt, lane_blockers = lanes.build_lane_receipt(
                workspace=workspace,
                lane_id=lane["lane_id"],
                conference=conference,
                producer=producer,
                transcript_source=str(lane["transcript_source"]),
                records=records,
                coverage_gaps=plan.coverage_gaps if index == 0 else (),
                notes=(
                    f"YouTube talk harvest {plan.harvest_dir} (run {plan.run_id}, receipt status "
                    f"{plan.receipt_status}); {tier} tier. Transcripts are normalized segment JSON "
                    "materialized in this lane; raw harvest sidecars are unchanged."
                ),
                bind_indexed_text_sha256=True,
                extra={
                    "harvest_dir": plan.harvest_dir,
                    "harvest_run_id": plan.run_id,
                    "harvest_receipt_status": plan.receipt_status,
                    "lane_tier": tier,
                },
            )
            if lane_blockers:
                blockers.extend(lane_blockers)
                continue
            registered_tiers = _registry_tier_positions(workspace)
            registry = _read_json(lanes.registry_path(workspace)) or {}
            existing_ids = [str(entry.get("lane_id")) for entry in registry.get("lanes") or []]
            position: int | None = None
            if supersede_existing:
                position = 0
            elif lane["lane_id"] not in existing_ids and tier == TIER_CAPTION:
                caption_positions = [i for i, value in enumerate(registered_tiers) if value == TIER_CAPTION]
                position = (caption_positions[-1] + 1) if caption_positions else 0
            lanes.register_lane(workspace=workspace, lane_id=lane["lane_id"], position=position)
            results.append({"lane_id": lane["lane_id"], "tier": tier, "records": len(records)})
    return {
        "status": "blocked" if blockers else "succeeded",
        "lanes": results,
        "blockers": blockers,
        "registry": lanes.summarize_registry(workspace),
    }


__all__ = [
    "HARVEST_INVENTORY_NAME",
    "HarvestLanePlan",
    "attach_existing_talk_maps",
    "bind_harvest_lanes",
    "default_label",
    "event_category",
    "normalize_inventory_path",
    "normalize_transcript_sidecar",
    "plan_harvest_lanes",
    "transcript_tier",
]
