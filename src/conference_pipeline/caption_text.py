"""Caption file (WebVTT/SRT) parsing and normalization for indexing.

Pure text functions shared by the YouTube harvest and the corpus index. The
corpus chunker digest covers this module, so any change here re-chunks (and
re-embeds only the chunks whose text actually changes).

YouTube auto captions scroll: each cue repeats the tail of the previous one
(``A`` / ``A B`` / ``B`` / ``B C``) and carries inline word timing tags. Raw
auto-caption text therefore reads every phrase two or three times;
``collapse_rolling_caption_segments`` removes that overlap.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

VTT_TIMING_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})"
)
VTT_TAG_RE = re.compile(r"<[^>]+>")
# Inline word timing (``<00:00:01.199><c> We</c>``) marks YouTube auto captions.
AUTO_CAPTION_WORD_TIMING_RE = re.compile(r"<\d{1,2}:\d{2}:\d{2}[.,]\d{3}><c>")

ROLLING_CAPTION_OVERLAP_WINDOW = 60
# Rolling cues abut or overlap the previous cue; a repeat that starts after a
# real pause is speech said again, not caption scroll.
ROLLING_CAPTION_MAX_GAP_SECONDS = 1.0

CAPTION_SEGMENTS_KIND = "caption_segments_json"


def vtt_time_to_seconds(stamp: str) -> float:
    stamp = stamp.replace(",", ".")
    parts = [float(part) for part in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def looks_like_caption_file(text: str) -> bool:
    head = text.lstrip("﻿ \t\r\n")[:2048]
    return head.startswith("WEBVTT") or bool(VTT_TIMING_RE.search(head))


def is_rolling_auto_caption(text: str) -> bool:
    return bool(AUTO_CAPTION_WORD_TIMING_RE.search(text[:200_000]))


def parse_caption_text(raw_text: str) -> list[dict[str, Any]]:
    """Parse VTT/SRT text into timed segments; exact consecutive repeats merge."""

    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        timing = VTT_TIMING_RE.search(line)
        if timing:
            if current and current["text"]:
                segments.append(current)
            current = {
                "start": vtt_time_to_seconds(timing.group("start")),
                "end": vtt_time_to_seconds(timing.group("end")),
                "text": "",
            }
            continue
        if current is None:
            continue
        if not line or line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "Kind:", "Language:")):
            continue
        if line.isdigit():
            continue
        text = VTT_TAG_RE.sub("", line).strip()
        if not text:
            continue
        current["text"] = (current["text"] + " " + text).strip() if current["text"] else text
    if current and current["text"]:
        segments.append(current)

    deduped: list[dict[str, Any]] = []
    for segment in segments:
        if deduped and segment["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = segment["end"]
            continue
        deduped.append(segment)
    return deduped


def parse_caption_file(path: Path) -> list[dict[str, Any]]:
    return parse_caption_text(path.read_text(encoding="utf-8", errors="replace"))


def _segment_seconds(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collapse_rolling_caption_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove YouTube auto-caption rolling overlap from parsed cue segments.

    For each cue, keep only the words beyond the longest prefix that equals the
    tail of what was already emitted, and drop cues that add nothing. Overlap is
    only removed between cues that abut in time: a phrase repeated after a pause
    longer than ``ROLLING_CAPTION_MAX_GAP_SECONDS`` is kept. Timing of kept cues
    is preserved; order stays monotonic.
    """

    emitted: list[str] = []
    collapsed: list[dict[str, Any]] = []
    previous_end: float | None = None
    for segment in segments:
        words = str(segment.get("text") or "").split()
        if not words:
            continue
        start = _segment_seconds(segment.get("start"))
        end = _segment_seconds(segment.get("end"))
        rolling = (
            previous_end is None
            or start is None
            or start <= previous_end + ROLLING_CAPTION_MAX_GAP_SECONDS
        )
        if end is not None:
            previous_end = end if previous_end is None else max(previous_end, end)
        tail = emitted[-ROLLING_CAPTION_OVERLAP_WINDOW:] if rolling else []
        overlap = 0
        for size in range(min(len(words), len(tail)), 0, -1):
            if tail[-size:] == words[:size]:
                overlap = size
                break
        new_words = words[overlap:]
        if not new_words:
            continue
        emitted.extend(new_words)
        collapsed.append({**segment, "text": " ".join(new_words)})
    return collapsed


def caption_segments_payload(raw_text: str) -> dict[str, Any] | None:
    """Return ``{"segments", "text", "normalization"}`` for caption text, or None.

    Rolling auto captions are collapsed; manual captions keep their cues.
    Returns None when the text is not a caption file or has no speech.
    """

    if not looks_like_caption_file(raw_text):
        return None
    raw_segments = parse_caption_text(raw_text)
    rolling = is_rolling_auto_caption(raw_text)
    segments = collapse_rolling_caption_segments(raw_segments) if rolling else raw_segments
    kept = [
        {"start": round(float(s["start"]), 3), "end": round(max(float(s["end"]), float(s["start"])), 3), "text": s["text"]}
        for s in segments
        if str(s.get("text") or "").strip()
    ]
    if not kept:
        return None
    return {
        "segments": kept,
        "text": " ".join(segment["text"] for segment in kept),
        "normalization": {
            "method": "collapse_rolling_captions" if rolling else "none",
            "raw_segment_count": len(raw_segments),
        },
    }


def caption_segments_json(raw_text: str) -> str | None:
    payload = caption_segments_payload(raw_text)
    return None if payload is None else json.dumps(payload, ensure_ascii=False)


__all__ = [
    "CAPTION_SEGMENTS_KIND",
    "ROLLING_CAPTION_MAX_GAP_SECONDS",
    "ROLLING_CAPTION_OVERLAP_WINDOW",
    "VTT_TAG_RE",
    "VTT_TIMING_RE",
    "caption_segments_json",
    "caption_segments_payload",
    "collapse_rolling_caption_segments",
    "is_rolling_auto_caption",
    "looks_like_caption_file",
    "parse_caption_file",
    "parse_caption_text",
    "vtt_time_to_seconds",
]
