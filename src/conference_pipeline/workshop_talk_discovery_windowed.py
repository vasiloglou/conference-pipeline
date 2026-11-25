"""Windowed talk-boundary discovery for long multi-talk recordings.

``workshop_talk_segmentation.discover_talks_with_gemini`` sends a whole
transcript in one prompt. A 7-hour workshop stream is far past any sensible
context budget, so this module:

1. turns the transcript into compact 30-second lines (``[1234s] text``);
   recordings with only coarse model-window timing are first spread into
   30-second pseudo-cues by word position (flagged ``synthetic_timestamp``);
2. tiles the recording into overlapping windows that fit a token budget;
3. asks the governed boundary-discovery model for the talks in each window,
   with absolute timestamps and ``continues_before/after_window`` flags;
4. merges window candidates into one monotonic, gap-free talk map;
5. validates the map and falls back to one honest whole-recording talk when
   anything is off (no provider approval, budget halt, provider failure, or a
   failed quality check). Talk titles are never invented by the fallback.

Outputs per recording: ``<video_id>_talks.json`` (bound to the transcript
digest), ``<video_id>_annotated.json`` (segments stamped with talk identity),
``<video_id>_batches.json``, and ``<video_id>_discovery_receipt.json``.

Model and budget come from the model-governance registry by task id.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.youtube_talks import title_similarity
from conference_pipeline.caption_text import caption_segments_payload, looks_like_caption_file
from conference_pipeline.indexing.token_policy import token_count
from conference_pipeline.workshop_talk_segmentation import (
    WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
    WORKSHOP_TALK_DISCOVERY_DEFAULT_BATCH_SECONDS,
    WORKSHOP_TALK_DISCOVERY_DEFAULT_MAX_CONTEXT_TOKENS,
    WORKSHOP_TALK_DISCOVERY_OWNER_ID,
    WORKSHOP_TALK_DISCOVERY_TASK_ID,
    WorkshopTalk,
    _extract_json_payload,
    _iter_raw_talks,
    _provider_config,
    annotate_segments_with_talks,
    build_talk_context_batches,
    invalid_talk_boundary_reasons,
    safe_receipt_error,
    sha256_file,
    utc_now_iso,
    workspace_relpath,
)

WINDOWED_DISCOVERY_RECEIPT_TYPE = "conference_pipeline.windowed_talk_boundary_discovery"
SPLIT_MIN_SECONDS = 2700.0
PROMPT_LINE_SECONDS = 30.0
SYNTHETIC_CUE_SECONDS = 30.0
COARSE_SEGMENT_SECONDS = 60.0
WINDOW_SECONDS = 5400.0
WINDOW_OVERLAP_SECONDS = 900.0
MIN_WINDOW_SECONDS = 1800.0
PROMPT_RESERVED_TOKENS = 6000
ASSUMED_OUTPUT_TOKENS_PER_WINDOW = 1500
WINDOW_MAX_OUTPUT_TOKENS = 32768
MERGE_OVERLAP_RATIO = 0.5
MERGE_START_DELTA_SECONDS = 120.0
MERGE_TITLE_SIMILARITY = 0.6
SNAP_OVERLAP_SECONDS = 180.0
# A gap longer than this becomes an unattributed segment instead of silently
# extending the previous talk (a 2-5 minute gap can be a whole lightning talk).
GAP_EXTEND_SECONDS = 60.0
MIN_TALK_SECONDS = 120.0
# A distinct named talk (lightning paper, short keynote intro) is kept down to
# this length; only unnamed fragments shorter than MIN_TALK_SECONDS fold into
# their neighbour.
MIN_NAMED_TALK_SECONDS = 30.0
MAX_TALKS = 80
# Bumped whenever merge/validation/timing rules change; resume only reuses a
# map produced by the same version.
DISCOVERY_ALGORITHM_VERSION = 2
SYNTHETIC_TIMING_CHAR_WEIGHT = "model_window_char_weight"
SYNTHETIC_TIMING_WORD_POSITION = "model_window_word_position"
_SYNTHETIC_UNIT_CHARS = 16
COVERAGE_START_SLACK_SECONDS = 60.0
COVERAGE_END_SLACK_SECONDS = 120.0
WORDS_PER_MINUTE_ESTIMATE = 150.0
TALK_KINDS = ("talk", "keynote", "panel", "break", "qa", "opening", "closing", "unattributed")
NAMED_TALK_KINDS = ("talk", "keynote")
# Segments with their own identity that must never be absorbed into another.
NAMED_SEGMENT_KINDS = ("talk", "keynote", "panel")
DISCOVERY_METHOD_WINDOWED = "model_windowed"
DISCOVERY_METHOD_FALLBACK = "deterministic_whole_recording"


# ---------------------------------------------------------------------------
# Transcript preparation
# ---------------------------------------------------------------------------


def effective_duration(segments: list[Mapping[str, Any]], declared: float | None = None) -> float:
    last_end = max((float(segment.get("end") or segment.get("start") or 0.0) for segment in segments), default=0.0)
    words = sum(len(str(segment.get("text") or "").split()) for segment in segments)
    estimate = words / WORDS_PER_MINUTE_ESTIMATE * 60.0 if last_end <= 0 else 0.0
    return max(float(declared or 0.0), last_end, estimate)


def qualifies_for_talk_splitting(duration_seconds: float, *, min_seconds: float = SPLIT_MIN_SECONDS) -> bool:
    return duration_seconds >= min_seconds


_SYNTHETIC_UNIT_RE = re.compile(r"\S{1,16}")


def _char_weighted_cues(text: str, start: float, span: float, cue_seconds: float) -> list[dict[str, Any]]:
    """Cut ``text`` into ~``cue_seconds`` cues, allocating time by character count.

    Units are whitespace words, with unspaced runs (Japanese, Chinese) cut into
    16-character pieces; cue text is sliced from the original, so nothing is
    lost or re-spaced.
    """

    units = [(match.start(), match.end()) for match in _SYNTHETIC_UNIT_RE.finditer(text)]
    if not units:
        return []
    total = sum(end - begin for begin, end in units)
    pieces = max(1, int(math.ceil(span / cue_seconds)))
    target = total / pieces
    cues: list[dict[str, Any]] = []
    consumed = 0
    piece_start = 0
    piece_chars = 0
    for index, (begin, end) in enumerate(units):
        piece_chars += end - begin
        if piece_chars >= target or index == len(units) - 1:
            first = consumed / total
            consumed += piece_chars
            cues.append(
                {
                    "start": round(start + span * first, 3),
                    "end": round(start + span * consumed / total, 3),
                    "text": text[units[piece_start][0] : end].strip(),
                    "synthetic_timestamp": True,
                    "synthetic_timestamp_basis": SYNTHETIC_TIMING_CHAR_WEIGHT,
                }
            )
            piece_start = index + 1
            piece_chars = 0
    return cues


def expand_coarse_segments(
    segments: list[Mapping[str, Any]],
    *,
    cue_seconds: float = SYNTHETIC_CUE_SECONDS,
    basis: str = SYNTHETIC_TIMING_CHAR_WEIGHT,
) -> list[dict[str, Any]]:
    """Spread segments longer than ``COARSE_SEGMENT_SECONDS`` into ~``cue_seconds`` cues.

    ``char_weight`` (current) allocates time by character count, so unspaced
    scripts get sensible cue lengths; ``word_position`` reproduces maps made
    before algorithm version 2.
    """

    cues: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        span = end - start
        if span <= COARSE_SEGMENT_SECONDS:
            cues.append({"start": start, "end": max(end, start), "text": text})
            continue
        if basis == SYNTHETIC_TIMING_CHAR_WEIGHT:
            cues.extend(_char_weighted_cues(text, start, span, cue_seconds))
            continue
        words = text.split()
        pieces = max(1, int(math.ceil(span / cue_seconds)))
        per_piece = max(1, int(math.ceil(len(words) / pieces)))
        for index in range(0, len(words), per_piece):
            first = index / len(words)
            last = min(len(words), index + per_piece) / len(words)
            cues.append(
                {
                    "start": round(start + span * first, 3),
                    "end": round(start + span * last, 3),
                    "text": " ".join(words[index : index + per_piece]),
                    "synthetic_timestamp": True,
                    "synthetic_timestamp_basis": SYNTHETIC_TIMING_WORD_POSITION,
                }
            )
    return cues


def prepare_discovery_lines(
    cues: list[Mapping[str, Any]], *, line_seconds: float = PROMPT_LINE_SECONDS
) -> list[dict[str, Any]]:
    """Coalesce cues into ``line_seconds`` lines for a compact prompt."""

    lines: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for cue in cues:
        start = float(cue.get("start") or 0.0)
        end = float(cue.get("end") or start)
        text = str(cue.get("text") or "").strip()
        if not text:
            continue
        if current is None or start >= current["start"] + line_seconds:
            current = {"start": start, "end": end, "text": text}
            lines.append(current)
        else:
            current["end"] = max(current["end"], end)
            current["text"] = f"{current['text']} {text}"
    return lines


def format_lines(lines: list[Mapping[str, Any]]) -> str:
    return "\n".join(f"[{int(line['start'])}s] {line['text']}" for line in lines)


def tile_windows(
    lines: list[Mapping[str, Any]],
    duration: float,
    *,
    window_seconds: float = WINDOW_SECONDS,
    overlap_seconds: float = WINDOW_OVERLAP_SECONDS,
    min_window_seconds: float = MIN_WINDOW_SECONDS,
    max_context_tokens: int = WORKSHOP_TALK_DISCOVERY_DEFAULT_MAX_CONTEXT_TOKENS,
    reserved_tokens: int = PROMPT_RESERVED_TOKENS,
    count_tokens: Callable[[str], int] = token_count,
) -> list[dict[str, Any]]:
    """Overlapping windows whose transcript lines fit the context budget."""

    budget = max(1, max_context_tokens - reserved_tokens)

    def _window_tokens(start: float, end: float) -> int:
        return count_tokens(format_lines([line for line in lines if start <= float(line["start"]) < end]))

    windows: list[dict[str, Any]] = []
    start = 0.0
    duration = max(duration, 1.0)
    while start < duration:
        width = window_seconds
        while width > min_window_seconds and _window_tokens(start, start + width) > budget:
            width = max(min_window_seconds, width / 2)
        end = min(duration, start + width)
        if 0 < duration - end < overlap_seconds and _window_tokens(start, duration + 1) <= budget:
            end = duration
        windows.append({"index": len(windows), "start": round(start, 3), "end": round(end, 3), "tokens": _window_tokens(start, end if end < duration else duration + 1)})
        if end >= duration:
            break
        start = max(start + 1.0, end - overlap_seconds)
    for window in windows:
        window["count"] = len(windows)
    return windows


# ---------------------------------------------------------------------------
# Prompt / response
# ---------------------------------------------------------------------------


def build_windowed_prompt(
    *,
    window: Mapping[str, Any],
    recording_title: str,
    hint_block: str,
    lines: list[Mapping[str, Any]],
) -> str:
    window_lines = [line for line in lines if float(window["start"]) <= float(line["start"]) < float(window["end"]) + 1]
    return f"""TALK BOUNDARY DISCOVERY (window {int(window['index']) + 1} of {int(window['count'])})

This is part of a long recording that contains several talks (keynotes,
paper talks, panels, breaks). Identify each distinct talk or segment that
appears in this window.

Return only valid JSON:
{{
  "talks": [
    {{
      "talk_title": "Human-readable title",
      "talk_speaker": "Speaker name or Unknown",
      "talk_kind": "one of {', '.join(TALK_KINDS)}",
      "timestamp_start": 0.0,
      "timestamp_end": 300.0,
      "continues_before_window": false,
      "continues_after_window": false,
      "description": "One sentence topic summary"
    }}
  ]
}}

Rules:
- Timestamps are absolute seconds from the start of the recording, as in the
  line prefixes. This window covers {float(window['start']):.0f}s to {float(window['end']):.0f}s.
- If a talk was already running at the window start, set
  continues_before_window true and timestamp_start to the window start. If it
  is still running at the window end, set continues_after_window true and
  timestamp_end to the window end.
- Prefer a boundary when the speaker, a moderator introduction, a title slide
  mention, or the topic clearly changes. Do not split one talk at its Q&A;
  keep Q&A inside the talk unless it is a separate session.
- Each paper presentation is its own entry, even a 1-3 minute spotlight or
  lightning talk; never group several presentations into one entry such as
  "Spotlight presentations".
- Never invent speaker names: use "Unknown" unless the transcript says it.
- Program hints below may help name talks. Use a hint only when the
  transcript clearly matches it; never force one.
- Cover the window without overlapping talks.

PROGRAM HINTS:
{hint_block or 'none'}

RECORDING: {recording_title}

TRANSCRIPT:
{format_lines(window_lines)}
"""


@dataclass
class TalkCandidate:
    start: float
    end: float
    title: str
    speaker: str | None
    kind: str
    description: str | None
    continues_before: bool
    continues_after: bool
    window_index: int

    @property
    def interior(self) -> bool:
        return not self.continues_before and not self.continues_after


def parse_window_response(text: str, window: Mapping[str, Any]) -> list[TalkCandidate]:
    payload = _extract_json_payload(text)
    window_start = float(window["start"])
    window_end = float(window["end"])
    candidates: list[TalkCandidate] = []
    for raw in _iter_raw_talks(payload):
        try:
            start = float(raw.get("timestamp_start", raw.get("start", window_start)))
            end = float(raw.get("timestamp_end", raw.get("end", window_end)))
        except (TypeError, ValueError):
            continue
        start = min(max(start, window_start), window_end)
        end = min(max(end, window_start), window_end)
        if end <= start:
            continue
        kind = str(raw.get("talk_kind") or "talk").strip().lower()
        speaker = str(raw.get("talk_speaker") or "").strip()
        candidates.append(
            TalkCandidate(
                start=start,
                end=end,
                title=str(raw.get("talk_title") or raw.get("title") or "").strip(),
                speaker=None if not speaker or speaker.lower() == "unknown" else speaker,
                kind=kind if kind in TALK_KINDS else "talk",
                description=str(raw.get("description") or "").strip() or None,
                continues_before=bool(raw.get("continues_before_window")),
                continues_after=bool(raw.get("continues_after_window")),
                window_index=int(window["index"]),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Merge and validation
# ---------------------------------------------------------------------------


def _overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _same_talk(group: Mapping[str, Any], candidate: TalkCandidate) -> bool:
    item = {"start": candidate.start, "end": candidate.end}
    shorter = max(1.0, min(group["end"] - group["start"], candidate.end - candidate.start))
    if _overlap(group, item) / shorter >= MERGE_OVERLAP_RATIO:
        return True
    return (
        abs(group["start"] - candidate.start) <= MERGE_START_DELTA_SECONDS
        and title_similarity(group["title"], candidate.title) >= MERGE_TITLE_SIMILARITY
    )


def merge_window_talks(
    candidates: list[TalkCandidate],
    duration: float,
    *,
    workshop_id: str | None = None,
    workshop_title: str | None = None,
) -> list[dict[str, Any]]:
    """Merge per-window candidates into one monotonic, gap-free talk list."""

    groups: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        target = next((group for group in reversed(groups[-3:]) if _same_talk(group, candidate)), None)
        if target is None:
            groups.append(
                {
                    "start": candidate.start,
                    "end": candidate.end,
                    "title": candidate.title,
                    "members": [candidate],
                }
            )
            continue
        target["start"] = min(target["start"], candidate.start)
        target["end"] = max(target["end"], candidate.end)
        target["members"].append(candidate)

    talks: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda item: item["start"]):
        members: list[TalkCandidate] = group["members"]
        best = sorted(members, key=lambda item: (not item.interior, -(item.end - item.start)))[0]
        talks.append(
            {
                "timestamp_start": group["start"],
                "timestamp_end": group["end"],
                "talk_title": best.title,
                "talk_speaker": best.speaker,
                "talk_kind": best.kind,
                "description": best.description,
                "boundary_confidence": "high",
                "merged_from": sorted({member.window_index for member in members}),
            }
        )

    # Consecutive entries naming the same talk (split by a window edge or a break) collapse.
    collapsed: list[dict[str, Any]] = []
    for talk in talks:
        previous = collapsed[-1] if collapsed else None
        same_kind = previous is not None and (
            talk["talk_kind"] == previous["talk_kind"]
            # Windows disagree on "keynote" vs "talk" for the same session.
            or (talk["talk_kind"] in NAMED_SEGMENT_KINDS and previous["talk_kind"] in NAMED_SEGMENT_KINDS)
        )
        if (
            previous is not None
            and same_kind
            and talk["talk_title"]
            and title_similarity(talk["talk_title"], previous["talk_title"]) >= 0.9
        ):
            previous["timestamp_end"] = max(previous["timestamp_end"], talk["timestamp_end"])
            if talk["talk_kind"] == "keynote":
                previous["talk_kind"] = "keynote"
            previous["talk_speaker"] = previous.get("talk_speaker") or talk.get("talk_speaker")
            continue
        collapsed.append(talk)

    resolved: list[dict[str, Any]] = []
    for talk in collapsed:
        if resolved:
            previous = resolved[-1]
            if talk["timestamp_end"] <= previous["timestamp_end"]:
                continue  # nested inside the previous talk
            overlap = previous["timestamp_end"] - talk["timestamp_start"]
            if overlap > 0:
                if overlap < SNAP_OVERLAP_SECONDS:
                    previous["timestamp_end"] = talk["timestamp_start"]
                else:
                    midpoint = (talk["timestamp_start"] + previous["timestamp_end"]) / 2
                    previous["timestamp_end"] = midpoint
                    talk["timestamp_start"] = midpoint
                    previous["boundary_confidence"] = talk["boundary_confidence"] = "low"
        resolved.append(talk)

    def _unattributed(start: float, end: float) -> dict[str, Any]:
        return {
            "timestamp_start": start,
            "timestamp_end": end,
            "talk_title": "Unattributed segment",
            "talk_speaker": None,
            "talk_kind": "unattributed",
            "description": None,
            "boundary_confidence": "low",
            "merged_from": [],
        }

    filled: list[dict[str, Any]] = []
    cursor = 0.0
    for talk in resolved:
        gap = talk["timestamp_start"] - cursor
        if gap > 0:
            if gap < GAP_EXTEND_SECONDS:
                if filled:
                    filled[-1]["timestamp_end"] = talk["timestamp_start"]
                else:
                    talk["timestamp_start"] = cursor
            else:
                filled.append(_unattributed(cursor, talk["timestamp_start"]))
        filled.append(talk)
        cursor = talk["timestamp_end"]
    if filled and duration - cursor > 0:
        if duration - cursor < GAP_EXTEND_SECONDS:
            filled[-1]["timestamp_end"] = duration
        else:
            filled.append(_unattributed(cursor, duration))

    def _foldable(talk: Mapping[str, Any]) -> bool:
        """Only unnamed fragments fold into a neighbour; a distinct named talk survives."""

        length = float(talk["timestamp_end"]) - float(talk["timestamp_start"])
        named = talk.get("talk_kind") in NAMED_SEGMENT_KINDS and bool(str(talk.get("talk_title") or "").strip())
        if named:
            return length < MIN_NAMED_TALK_SECONDS
        return length < MIN_TALK_SECONDS

    merged: list[dict[str, Any]] = []
    for talk in filled:
        if _foldable(talk) and merged:
            merged[-1]["timestamp_end"] = talk["timestamp_end"]
            continue
        merged.append(talk)
    if len(merged) > 1 and _foldable(merged[0]):
        merged[1]["timestamp_start"] = merged[0]["timestamp_start"]
        merged.pop(0)

    for index, talk in enumerate(merged, start=1):
        talk["talk_id"] = f"talk_{index:03d}"
        talk["timestamp_start"] = round(float(talk["timestamp_start"]), 3)
        talk["timestamp_end"] = round(float(talk["timestamp_end"]), 3)
        if workshop_id:
            talk["workshop_id"] = workshop_id
        if workshop_title:
            talk["workshop_title"] = workshop_title
        if not talk.get("talk_title"):
            talk["talk_title"] = "Untitled segment"
    return merged


def as_workshop_talks(talks: list[Mapping[str, Any]]) -> list[WorkshopTalk]:
    return [
        WorkshopTalk(
            talk_id=str(talk["talk_id"]),
            talk_title=str(talk.get("talk_title") or talk["talk_id"]),
            talk_speaker=talk.get("talk_speaker"),
            timestamp_start=float(talk["timestamp_start"]),
            timestamp_end=float(talk["timestamp_end"]),
            workshop_id=talk.get("workshop_id"),
            workshop_title=talk.get("workshop_title"),
            description=talk.get("description"),
        )
        for talk in talks
    ]


def talk_map_quality_issues(talks: list[Mapping[str, Any]], duration: float) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = list(invalid_talk_boundary_reasons(as_workshop_talks(talks)))
    if not talks:
        return [{"reason": "no_talks"}]
    if float(talks[0]["timestamp_start"]) > COVERAGE_START_SLACK_SECONDS:
        issues.append({"reason": "coverage_starts_late"})
    if float(talks[-1]["timestamp_end"]) < duration - COVERAGE_END_SLACK_SECONDS:
        issues.append({"reason": "coverage_ends_early"})
    for previous, current in zip(talks, talks[1:]):
        if float(current["timestamp_start"]) - float(previous["timestamp_end"]) > 1.0:
            issues.append({"reason": "uncovered_gap", "talk_id": current["talk_id"]})
    limit = min(MAX_TALKS, max(1, int(math.ceil(duration / 60.0))))
    if len(talks) > limit:
        issues.append({"reason": "too_many_talks", "count": len(talks), "limit": limit})
    named = [talk for talk in talks if talk.get("talk_kind") in NAMED_TALK_KINDS]
    titles = [str(talk.get("talk_title") or "").strip().lower() for talk in named]
    if any(not title or title == "untitled segment" for title in titles):
        issues.append({"reason": "named_talk_without_title"})
    if len(set(titles)) != len(titles):
        issues.append({"reason": "duplicate_talk_titles"})
    for talk in talks:
        if float(talk["timestamp_start"]) < 0 or float(talk["timestamp_end"]) > duration + COVERAGE_START_SLACK_SECONDS:
            issues.append({"reason": "timestamp_out_of_range", "talk_id": talk["talk_id"]})
    return issues


def deterministic_whole_recording_talks(
    *, recording_title: str, duration: float, workshop_id: str | None, workshop_title: str | None
) -> list[dict[str, Any]]:
    talk: dict[str, Any] = {
        "talk_id": "talk_001",
        "talk_title": recording_title or "Recording",
        "talk_speaker": None,
        "talk_kind": "recording",
        "timestamp_start": 0.0,
        "timestamp_end": round(max(duration, 1.0), 3),
        "boundary_confidence": "none",
        "merged_from": [],
    }
    if workshop_id:
        talk["workshop_id"] = workshop_id
    if workshop_title:
        talk["workshop_title"] = workshop_title
    return [talk]


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def estimate_window_cost(
    prompt_tokens: int,
    *,
    model: str,
    provider: str,
    output_tokens: int = ASSUMED_OUTPUT_TOKENS_PER_WINDOW,
) -> float:
    from conference_pipeline.costs.estimator import estimate_text_model_cost

    return float(estimate_text_model_cost(model, prompt_tokens, output_tokens, provider=provider))


# ---------------------------------------------------------------------------
# One recording
# ---------------------------------------------------------------------------


@dataclass
class RecordingInput:
    video_id: str
    lane_id: str
    transcript_path: Path
    title: str
    duration_seconds: float | None
    timing_kind: str
    hints: Mapping[str, Any]


def load_transcript_segments(path: Path) -> list[dict[str, Any]]:
    """Timed segments from a segment JSON transcript or a caption (VTT/SRT) file.

    Caption files are normalized exactly as the corpus index normalizes them,
    so talk boundaries and indexed text share one timeline.
    """

    raw = path.read_text(encoding="utf-8", errors="replace")
    if looks_like_caption_file(raw):
        payload = caption_segments_payload(raw) or {}
        return [dict(segment) for segment in payload.get("segments") or []]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = payload.get("segments") or payload.get("transcript_segments") or []
    else:
        rows = []
    return [dict(segment) for segment in rows if isinstance(segment, Mapping)]


def plan_recording(recording: RecordingInput, *, synthetic_basis: str = SYNTHETIC_TIMING_CHAR_WEIGHT) -> dict[str, Any]:
    segments = load_transcript_segments(recording.transcript_path)
    duration = effective_duration(segments, recording.duration_seconds)
    cues = expand_coarse_segments(segments, basis=synthetic_basis)
    lines = prepare_discovery_lines(cues)
    windows = tile_windows(lines, duration)
    for window in windows:
        window["prompt"] = build_windowed_prompt(
            window=window,
            recording_title=recording.title,
            hint_block=str(recording.hints.get("hint_block") or ""),
            lines=lines,
        )
        window["prompt_tokens"] = token_count(window["prompt"])
    return {
        "segments": segments,
        "cues": cues,
        "lines": lines,
        "windows": windows,
        "duration": duration,
        "synthetic_basis": synthetic_basis,
    }


def write_recording_outputs(
    *,
    workspace: Path,
    recording: RecordingInput,
    talks_json: Path,
    annotated_json: Path,
    batches_json: Path,
    receipt_path: Path,
    plan: Mapping[str, Any],
    talks: list[dict[str, Any]],
    discovery_method: str,
    provider: str,
    model: str,
    window_receipts: list[dict[str, Any]],
    estimated_cost_usd: float,
    fallback_reason: str | None,
    validation_issues: list[dict[str, Any]],
    window_candidates: list[TalkCandidate] | None = None,
) -> dict[str, Any]:
    """Write talks, annotated transcript, batches, then the receipt (last).

    Every file is written atomically and names the video and the digests it
    depends on, so a crash leaves either the previous complete generation or a
    detectably incomplete one (receipt digests do not match), never a mix.
    """

    transcript_sha256 = sha256_file(recording.transcript_path)
    # Unattributed gaps stay unstamped: chunks there carry no invented talk identity.
    workshop_talks = as_workshop_talks([talk for talk in talks if talk.get("talk_kind") != "unattributed"])
    annotation_segments = plan["cues"] if any(cue.get("synthetic_timestamp") for cue in plan["cues"]) else plan["segments"]
    annotated = annotate_segments_with_talks([dict(segment) for segment in annotation_segments], workshop_talks)
    batches = build_talk_context_batches(annotated, workshop_talks, batch_seconds=WORKSHOP_TALK_DISCOVERY_DEFAULT_BATCH_SECONDS)
    hints = dict(recording.hints)
    hints.pop("hint_block", None)
    talks_payload = {
        "schema_version": 2,
        "discovery_algorithm_version": DISCOVERY_ALGORITHM_VERSION,
        "synthetic_timing_basis": plan.get("synthetic_basis") if any(cue.get("synthetic_timestamp") for cue in plan["cues"]) else None,
        "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
        "video_id": recording.video_id,
        "lane_id": recording.lane_id,
        "source_transcript_path": workspace_relpath(workspace, recording.transcript_path),
        "source_transcript_sha256": transcript_sha256,
        "duration_seconds": round(float(plan["duration"]), 3),
        "timing_kind": recording.timing_kind,
        "boundary_resolution_seconds": SYNTHETIC_CUE_SECONDS if recording.timing_kind != "caption_cue" else PROMPT_LINE_SECONDS,
        "discovery_method": discovery_method,
        "fallback_reason": fallback_reason,
        "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
        "cost_task_id": WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
        # Named whenever windows were actually sent, even if the map then fell back.
        "provider": provider if window_receipts else None,
        "model": model if window_receipts else None,
        "windows": window_receipts,
        "program_hints": hints,
        "workshop_association": hints.get("workshop_association"),
        "window_candidates": [
            {
                "window_index": candidate.window_index,
                "start": candidate.start,
                "end": candidate.end,
                "title": candidate.title,
                "speaker": candidate.speaker,
                "kind": candidate.kind,
                "description": candidate.description,
                "continues_before": candidate.continues_before,
                "continues_after": candidate.continues_after,
            }
            for candidate in window_candidates or []
        ],
        "talks": talks,
        "validation": {"issues": validation_issues},
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "secrets_copied": False,
    }
    _atomic_write_json(talks_json, talks_payload, compact=True)
    talks_sha256 = sha256_file(talks_json)
    _atomic_write_json(
        annotated_json,
        {
            "schema_version": 2,
            "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
            "video_id": recording.video_id,
            "source_transcript_sha256": transcript_sha256,
            "talks_sha256": talks_sha256,
            "segments": annotated,
        },
        compact=True,
    )
    _atomic_write_json(
        batches_json,
        {
            "schema_version": 2,
            "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
            "video_id": recording.video_id,
            "talks_sha256": talks_sha256,
            "batches": batches,
        },
        compact=True,
    )
    receipt = {
        "receipt_type": WINDOWED_DISCOVERY_RECEIPT_TYPE,
        "schema_version": 1,
        "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
        "generated_at": utc_now_iso(),
        "status": "succeeded",
        "video_id": recording.video_id,
        "lane_id": recording.lane_id,
        "discovery_method": discovery_method,
        "fallback_reason": fallback_reason,
        "talk_count": len(talks),
        "window_count": len(plan["windows"]),
        "provider_calls_made": bool(window_receipts and any(item.get("provider_receipt_path") for item in window_receipts)),
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "hint_sha256": recording.hints.get("hint_sha256"),
        "talks_json": workspace_relpath(workspace, talks_json),
        "talks_json_sha256": sha256_file(talks_json),
        "annotated_transcript_json": workspace_relpath(workspace, annotated_json),
        "annotated_transcript_json_sha256": sha256_file(annotated_json),
        "source_transcript_sha256": transcript_sha256,
        "validation_issues": validation_issues,
        "secrets_copied": False,
    }
    receipt["discovery_algorithm_version"] = DISCOVERY_ALGORITHM_VERSION
    _atomic_write_json(receipt_path, receipt, compact=False)
    return receipt


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if compact
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def discover_recording(
    *,
    workspace: Path,
    recording: RecordingInput,
    output_dir: Path,
    client: Any | None,
    provider: str,
    model: str,
    remaining_budget_usd: float | None,
    call_window: Callable[[Any, str, str], tuple[str, str | None, float | None]] | None = None,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Discover one recording's talks; never raises for provider problems.

    ``fallback_reason`` forces the deterministic fallback (for example when
    provider cost was not approved). ``call_window(client, model, prompt)``
    returns ``(response_text, provider_receipt_path, estimated_cost_usd)``.
    """

    plan = plan_recording(recording)
    stem = recording.video_id
    talks_json = output_dir / f"{stem}_talks.json"
    annotated_json = output_dir / f"{stem}_annotated.json"
    batches_json = output_dir / f"{stem}_batches.json"
    receipt_path = output_dir / f"{stem}_discovery_receipt.json"
    workshop_id = recording.hints.get("workshop_id")
    workshop_title = recording.hints.get("workshop_title")
    window_receipts: list[dict[str, Any]] = []
    spent = 0.0
    candidates: list[TalkCandidate] = []
    reason = fallback_reason
    if reason is None and (client is None or call_window is None):
        reason = "provider_not_configured"
    if reason is None:
        for window in plan["windows"]:
            estimate = 0.0
            if remaining_budget_usd is not None:
                try:
                    estimate = estimate_window_cost(int(window["prompt_tokens"]), model=model, provider=provider)
                except KeyError:
                    reason = "model_unpriced_budget_unenforceable"
                    break
                if spent + estimate > remaining_budget_usd:
                    reason = "budget_halt_before_window"
                    break
            try:
                text, receipt_ref, cost = call_window(client, model, str(window["prompt"]))
            except Exception as exc:  # noqa: BLE001 - provider failures become a recorded fallback
                provider_receipt = getattr(exc, "provider_receipt", None)
                spent += float(getattr(provider_receipt, "estimated_cost_usd", 0.0) or 0.0)
                window_receipts.append(
                    {"index": window["index"], "start": window["start"], "end": window["end"], "error": safe_receipt_error(exc)}
                )
                reason = "provider_call_failed"
                break
            spent += float(cost if cost is not None else estimate)
            if receipt_ref:
                receipt_ref = workspace_relpath(workspace, Path(receipt_ref))
            window_receipts.append(
                {
                    "index": window["index"],
                    "start": window["start"],
                    "end": window["end"],
                    "prompt_tokens": window["prompt_tokens"],
                    "provider_receipt_path": receipt_ref,
                    "estimated_cost_usd": cost,
                }
            )
            try:
                candidates.extend(parse_window_response(text, window))
            except Exception as exc:  # noqa: BLE001 - malformed model output
                window_receipts[-1]["error"] = safe_receipt_error(exc)
                reason = "window_response_unparseable"
                break
    issues: list[dict[str, Any]] = []
    talks: list[dict[str, Any]] = []
    if reason is None:
        talks = merge_window_talks(candidates, plan["duration"], workshop_id=workshop_id, workshop_title=workshop_title)
        issues = talk_map_quality_issues(talks, plan["duration"])
        if issues:
            reason = "talk_map_failed_validation"
    method = DISCOVERY_METHOD_WINDOWED
    if reason is not None:
        method = DISCOVERY_METHOD_FALLBACK
        talks = deterministic_whole_recording_talks(
            recording_title=recording.title,
            duration=plan["duration"],
            workshop_id=workshop_id,
            workshop_title=workshop_title,
        )
    receipt = write_recording_outputs(
        workspace=workspace,
        recording=recording,
        talks_json=talks_json,
        annotated_json=annotated_json,
        batches_json=batches_json,
        receipt_path=receipt_path,
        plan=plan,
        talks=talks,
        discovery_method=method,
        provider=provider,
        model=model,
        window_receipts=window_receipts,
        estimated_cost_usd=spent,
        fallback_reason=reason,
        validation_issues=issues,
        window_candidates=candidates,
    )
    receipt["spent_usd"] = spent
    return receipt


def remerge_recording(
    *,
    workspace: Path,
    recording: RecordingInput,
    output_dir: Path,
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    """Re-apply the current merge and validation rules to stored window candidates.

    No provider call: the talk map keeps every window's raw candidates and its
    provider receipts, so a merge-rule fix is applied offline. Returns the new
    receipt, or None when there is no model map (with candidates) to re-merge
    or it was made from a different transcript.
    """

    talks_json = output_dir / f"{recording.video_id}_talks.json"
    try:
        previous = json.loads(talks_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = previous.get("window_candidates")
    windows = previous.get("windows") or []
    if not raw or not windows or any(window.get("error") for window in windows):
        return None
    if previous.get("source_transcript_sha256") != sha256_file(recording.transcript_path):
        return None
    candidates = [
        TalkCandidate(
            start=float(item["start"]),
            end=float(item["end"]),
            title=str(item.get("title") or ""),
            speaker=item.get("speaker"),
            kind=str(item.get("kind") or "talk"),
            description=item.get("description"),
            continues_before=bool(item.get("continues_before")),
            continues_after=bool(item.get("continues_after")),
            window_index=int(item.get("window_index") or 0),
        )
        for item in raw
    ]
    plan = plan_recording(recording, synthetic_basis=str(previous.get("synthetic_timing_basis") or SYNTHETIC_TIMING_CHAR_WEIGHT))
    workshop_id = recording.hints.get("workshop_id")
    workshop_title = recording.hints.get("workshop_title")
    talks = merge_window_talks(candidates, plan["duration"], workshop_id=workshop_id, workshop_title=workshop_title)
    issues = talk_map_quality_issues(talks, plan["duration"])
    method = DISCOVERY_METHOD_WINDOWED
    reason = None
    if issues:
        method = DISCOVERY_METHOD_FALLBACK
        reason = "talk_map_failed_validation"
        talks = deterministic_whole_recording_talks(
            recording_title=recording.title, duration=plan["duration"], workshop_id=workshop_id, workshop_title=workshop_title
        )
    return write_recording_outputs(
        workspace=workspace,
        recording=recording,
        talks_json=talks_json,
        annotated_json=output_dir / f"{recording.video_id}_annotated.json",
        batches_json=output_dir / f"{recording.video_id}_batches.json",
        receipt_path=output_dir / f"{recording.video_id}_discovery_receipt.json",
        plan=plan,
        talks=talks,
        discovery_method=method,
        provider=str(previous.get("provider") or provider),
        model=str(previous.get("model") or model),
        window_receipts=list(windows),
        estimated_cost_usd=float(previous.get("estimated_cost_usd") or 0.0),
        fallback_reason=reason,
        validation_issues=issues,
        window_candidates=candidates,
    )


def gemini_call_window(client: Any, model: str, prompt: str) -> tuple[str, str | None, float | None]:
    """Default window call through the receipted Gemini wrapper."""

    from conference_pipeline.provider_receipts import gemini_generate_content

    result = gemini_generate_content(
        client,
        caller=__file__,
        model=model,
        contents=prompt,
        # Thinking tokens count against the output budget; the segmentation
        # module's 8k default leaves no room for the JSON answer on long windows.
        config=_provider_config(max_output_tokens=WINDOW_MAX_OUTPUT_TOKENS),
    )
    receipt = result.receipt
    return (
        str(getattr(result, "text", "") or ""),
        str(getattr(receipt, "receipt_path", "") or "") or None,
        getattr(receipt, "estimated_cost_usd", None),
    )


__all__ = [
    "DISCOVERY_METHOD_FALLBACK",
    "DISCOVERY_METHOD_WINDOWED",
    "RecordingInput",
    "TalkCandidate",
    "build_windowed_prompt",
    "deterministic_whole_recording_talks",
    "discover_recording",
    "effective_duration",
    "estimate_window_cost",
    "expand_coarse_segments",
    "gemini_call_window",
    "merge_window_talks",
    "parse_window_response",
    "plan_recording",
    "prepare_discovery_lines",
    "qualifies_for_talk_splitting",
    "remerge_recording",
    "talk_map_quality_issues",
    "tile_windows",
]
