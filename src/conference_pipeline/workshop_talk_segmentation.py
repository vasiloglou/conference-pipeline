"""Reusable workshop talk segmentation helpers.

This module owns two reusable workshop transcript steps: provider-backed
boundary discovery that writes governed ``*_talks.json`` files, and offline
annotation that applies those talk records to timestamped transcript segments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from conference_pipeline.audio_chunker import get_audio_duration
from conference_pipeline.model_governance import default_model, default_provider, validate_selection
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    provider_receipt_context,
    sanitize_for_receipt,
)

WORKSHOP_TALK_SEGMENTATION_OWNER_ID = "transcription_ocr.workshop_talk_segmentation"
WORKSHOP_TALK_DISCOVERY_OWNER_ID = "transcription_ocr.workshop_talk_boundary_discovery"
WORKSHOP_TALK_SEGMENTATION_RECEIPT_TYPE = "conference_pipeline.workshop_talk_segmentation"
WORKSHOP_TALK_DISCOVERY_RECEIPT_TYPE = "conference_pipeline.workshop_talk_boundary_discovery"
PROVIDER_NOT_CONFIGURED_CODE = "talk_segmentation_provider_not_configured"
PROVIDER_COST_NOT_APPROVED_CODE = "workshop_talk_boundary_discovery_cost_not_approved"
PROVIDER_DISCOVERY_FAILED_CODE = "workshop_talk_boundary_discovery_failed"
WORKSHOP_TALK_DISCOVERY_TASK_ID = "transcription_ocr.workshop_talk_boundary_discovery"
WORKSHOP_TALK_DISCOVERY_COST_TASK_ID = "workshop_talk_boundary_discovery"
WORKSHOP_TALK_DISCOVERY_DEFAULT_BATCH_SECONDS = 300.0
WORKSHOP_TALK_DISCOVERY_DEFAULT_MAX_CONTEXT_TOKENS = 40_000
SYNTHETIC_TRANSCRIPT_SEGMENT_SECONDS = 300.0
SYNTHETIC_TRANSCRIPT_CHARS_PER_SEGMENT = 3_500


@dataclass(frozen=True)
class WorkshopTalk:
    talk_id: str
    talk_title: str
    talk_speaker: str | None
    timestamp_start: float
    timestamp_end: float
    workshop_id: str | None = None
    workshop_title: str | None = None
    description: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in (None, "")}


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_receipt_error(error: object | None, fallback: str = "error") -> str:
    text = str(error or fallback)
    return str(sanitize_for_receipt(text))[:500]


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_workspace_member(workspace: Path, path_value: object) -> tuple[Path | None, str | None]:
    """Resolve a single, pipeline-recorded artifact path safely.

    Containment is checked lexically (``os.path.abspath`` normalizes
    ``.``/``..`` without following symlinks), not via a symlink-following
    resolve, so a workspace-owned symlink -- for example storage/
    transcription-ocr's Drive-backed on-demand artifact leaf -- is not
    rejected just because it resolves outside the repository tree. A ``..``
    escape attempt embedded in the path text is still rejected, since
    normpath collapses it before the comparison.
    """
    path_text = str(path_value or "").strip()
    if not path_text:
        return None, "missing_path"
    raw = Path(path_text)
    candidate = raw if raw.is_absolute() else workspace / raw
    workspace_root = Path(os.path.abspath(str(workspace)))
    resolved = Path(os.path.abspath(str(candidate)))
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, "path_outside_workspace"
    return resolved, None


def _float_value(record: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _string_value(record: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return ""


def _iter_raw_talks(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("talks", "records", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _iter_raw_segments(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("segments", "transcript_segments", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _iter_raw_slides(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("slides", "records", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def normalize_talk_record(
    record: dict[str, Any],
    *,
    index: int,
    default_workshop_id: str | None = None,
    default_workshop_title: str | None = None,
) -> WorkshopTalk:
    talk_id = _string_value(record, "talk_id", "id", "uid") or f"talk_{index + 1:03d}"
    talk_title = _string_value(record, "talk_title", "title", "name") or talk_id
    talk_speaker = _string_value(record, "talk_speaker", "speaker", "speakers") or None
    return WorkshopTalk(
        talk_id=talk_id,
        talk_title=talk_title,
        talk_speaker=talk_speaker,
        timestamp_start=_float_value(record, "timestamp_start", "start_sec", "start", "start_time"),
        timestamp_end=_float_value(record, "timestamp_end", "end_sec", "end", "end_time"),
        workshop_id=_string_value(record, "workshop_id") or default_workshop_id,
        workshop_title=_string_value(record, "workshop_title") or default_workshop_title,
        description=_string_value(record, "description", "abstract") or None,
    )


def normalize_transcript_segment(record: dict[str, Any]) -> dict[str, Any]:
    start = _float_value(record, "timestamp_start", "start_sec", "start")
    end = _float_value(record, "timestamp_end", "end_sec", "end")
    text = _string_value(record, "text", "transcript", "utterance")
    out = dict(record)
    out["start"] = start
    out["end"] = end if end >= start else start
    out["text"] = text
    return out


def _resolve_payload_path(
    *,
    workspace: Path | None,
    anchor_path: Path,
    path_value: object,
) -> Path | None:
    if not path_value:
        return None
    if workspace is not None:
        path, error = resolve_workspace_member(workspace, path_value)
        if error is None:
            return path
        return None
    raw = Path(str(path_value))
    return raw if raw.is_absolute() else anchor_path.parent / raw


def _read_transcript_text_from_payload(
    *,
    payload: Any,
    transcript_json_path: Path,
    workspace: Path | None,
) -> str:
    if isinstance(payload, dict):
        for key in ("text", "transcript", "full_text"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        transcript_path = _resolve_payload_path(
            workspace=workspace,
            anchor_path=transcript_json_path,
            path_value=payload.get("transcript_path"),
        )
        if transcript_path is not None and transcript_path.exists():
            return transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    return ""


def _duration_from_payload(
    *,
    payload: Any,
    transcript_json_path: Path,
    workspace: Path | None,
) -> float | None:
    if not isinstance(payload, dict):
        return None
    duration = _float_value(
        payload,
        "duration_seconds",
        "source_duration_seconds",
        "audio_duration_seconds",
        "end_time",
    )
    if duration > 0:
        return duration
    source_path = _resolve_payload_path(
        workspace=workspace,
        anchor_path=transcript_json_path,
        path_value=payload.get("source_path") or payload.get("local_path"),
    )
    if source_path is None or not source_path.exists():
        return None
    return get_audio_duration(source_path)


def _split_text_evenly(text: str, segment_count: int) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    segment_count = max(1, min(segment_count, max(1, len(normalized))))
    chunks: list[str] = []
    start = 0
    for index in range(segment_count):
        if index == segment_count - 1:
            end = len(normalized)
        else:
            target = round(len(normalized) * (index + 1) / segment_count)
            end = normalized.rfind(" ", start, max(target, start + 1))
            if end <= start:
                end = target
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def synthesize_transcript_segments(
    text: str,
    *,
    duration_seconds: float | None = None,
    segment_seconds: float = SYNTHETIC_TRANSCRIPT_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """Evenly timed segments for an untimed transcript.

    ``segment_seconds`` is the resolution a talk boundary can be placed at: a
    segment is attributed whole to one talk, so short talks (a 15-minute oral)
    need finer segments than a workshop's 30-minute invited talks.
    """

    text = text.strip()
    if not text:
        return []
    if duration_seconds is not None and duration_seconds > 0:
        segment_count = max(
            1,
            int(math.ceil(duration_seconds / segment_seconds)),
        )
        segment_duration = duration_seconds / segment_count
    else:
        segment_count = max(
            1,
            int(math.ceil(len(text) / SYNTHETIC_TRANSCRIPT_CHARS_PER_SEGMENT)),
        )
        segment_duration = SYNTHETIC_TRANSCRIPT_SEGMENT_SECONDS
    chunks = _split_text_evenly(text, segment_count)
    return [
        {
            "start": round(index * segment_duration, 3),
            "end": round((index + 1) * segment_duration, 3),
            "timestamp_start": round(index * segment_duration, 3),
            "timestamp_end": round((index + 1) * segment_duration, 3),
            "text": chunk,
            "synthetic_timestamp": True,
            "synthetic_timestamp_basis": (
                "audio_duration_and_text_position"
                if duration_seconds is not None and duration_seconds > 0
                else "text_position"
            ),
        }
        for index, chunk in enumerate(chunks)
        if chunk
    ]


def load_transcript_segments(
    path: Path,
    *,
    workspace: Path | None = None,
    synthetic_segment_seconds: float = SYNTHETIC_TRANSCRIPT_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = [
        normalize_transcript_segment(segment)
        for segment in _iter_raw_segments(payload)
        if _string_value(segment, "text", "transcript", "utterance")
    ]
    if segments:
        return segments
    text = _read_transcript_text_from_payload(
        payload=payload,
        transcript_json_path=path,
        workspace=workspace,
    )
    duration = _duration_from_payload(
        payload=payload,
        transcript_json_path=path,
        workspace=workspace,
    )
    return synthesize_transcript_segments(
        text, duration_seconds=duration, segment_seconds=synthetic_segment_seconds
    )


def load_slide_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _iter_raw_slides(payload)


def load_talks_json(path: Path) -> list[WorkshopTalk]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_workshop_id = _string_value(payload, "workshop_id") if isinstance(payload, dict) else None
    default_workshop_title = (
        _string_value(payload, "workshop_title", "title") if isinstance(payload, dict) else None
    )
    return [
        normalize_talk_record(
            record,
            index=index,
            default_workshop_id=default_workshop_id,
            default_workshop_title=default_workshop_title,
        )
        for index, record in enumerate(_iter_raw_talks(payload))
    ]


def find_overlapping_talks(
    talks: list[WorkshopTalk],
    *,
    timestamp_start: float,
    timestamp_end: float,
) -> list[WorkshopTalk]:
    return [
        talk
        for talk in talks
        if talk.timestamp_start < timestamp_end and talk.timestamp_end > timestamp_start
    ]


def invalid_talk_boundary_reasons(talks: list[WorkshopTalk]) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    previous_end: float | None = None
    for talk in sorted(talks, key=lambda item: (item.timestamp_start, item.timestamp_end, item.talk_id)):
        if not math.isfinite(talk.timestamp_start) or not math.isfinite(talk.timestamp_end):
            issues.append({"talk_id": talk.talk_id, "reason": "non_finite_timestamp"})
            continue
        if talk.timestamp_start < 0 or talk.timestamp_end < 0:
            issues.append({"talk_id": talk.talk_id, "reason": "negative_timestamp"})
            continue
        if talk.timestamp_end <= talk.timestamp_start:
            issues.append({"talk_id": talk.talk_id, "reason": "non_positive_duration"})
            continue
        if previous_end is not None and talk.timestamp_start < previous_end:
            issues.append({"talk_id": talk.talk_id, "reason": "overlapping_or_non_monotonic"})
        previous_end = max(previous_end or talk.timestamp_end, talk.timestamp_end)
    return issues


TALK_IDENTITY_METADATA_KEYS = (
    "talk_id",
    "talk_title",
    "talk_speaker",
    "workshop_id",
    "workshop_title",
    "description",
)


def _talk_identity_metadata(talk: WorkshopTalk) -> dict[str, Any]:
    """Return talk identity fields only, excluding timestamp_start/end.

    ``WorkshopTalk.to_record()`` includes the talk's own
    ``timestamp_start``/``timestamp_end``. Merging those into a transcript
    segment would overwrite the segment's own timing (e.g., a 5-8s utterance
    would be rewritten with the talk's 0-300s span). Restrict the merge to
    identity fields so segment timing is preserved.
    """
    record = talk.to_record()
    return {key: record[key] for key in TALK_IDENTITY_METADATA_KEYS if key in record}


def annotate_segments_with_talks(
    segments: list[dict[str, Any]],
    talks: list[WorkshopTalk],
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for segment in segments:
        start = _float_value(segment, "timestamp_start", "start")
        end = _float_value(segment, "timestamp_end", "end")
        matches = find_overlapping_talks(
            talks,
            timestamp_start=start,
            timestamp_end=end,
        )
        if not matches:
            annotated.append(dict(segment))
            continue
        talk = matches[0]
        annotated.append({**segment, **_talk_identity_metadata(talk)})
    return annotated


def build_talk_context_batches(
    segments: list[dict[str, Any]],
    talks: list[WorkshopTalk],
    *,
    batch_seconds: float = WORKSHOP_TALK_DISCOVERY_DEFAULT_BATCH_SECONDS,
) -> list[dict[str, Any]]:
    normalized = [normalize_transcript_segment(segment) for segment in segments]
    if not normalized:
        return []
    first_start = min(float(segment["start"]) for segment in normalized)
    last_end = max(float(segment["end"]) for segment in normalized)
    if last_end <= first_start:
        last_end = first_start + batch_seconds
    batch_count = max(1, int(math.ceil((last_end - first_start) / batch_seconds)))
    batches: list[dict[str, Any]] = []
    for index in range(batch_count):
        start = first_start + index * batch_seconds
        end = min(last_end, start + batch_seconds)
        batch_segments = [
            segment
            for segment in normalized
            if float(segment["start"]) < end and float(segment["end"]) > start
        ]
        if not batch_segments:
            continue
        overlapping_talks = find_overlapping_talks(talks, timestamp_start=start, timestamp_end=end)
        batches.append(
            {
                "batch_id": f"batch_{len(batches) + 1:03d}",
                "timestamp_start": start,
                "timestamp_end": end,
                "text": " ".join(str(segment.get("text") or "").strip() for segment in batch_segments).strip(),
                "segments": batch_segments,
                "talks": [talk.to_record() for talk in overlapping_talks],
                "talk_ids": [talk.talk_id for talk in overlapping_talks],
            }
        )
    return batches


def _format_segments_for_prompt(segments: list[dict[str, Any]]) -> str:
    lines = []
    for segment in segments:
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        text = str(segment.get("text") or "").strip()
        if text:
            lines.append(f"[{start:.2f}-{end:.2f}] {text}")
    return "\n".join(lines)


def _format_slides_for_prompt(slides: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, slide in enumerate(slides[:300]):
        timestamp = _float_value(slide, "timestamp_sec", "timestamp_start", "timestamp", "start")
        text = _string_value(slide, "ocr_text", "slide_text", "text", "title")
        if text:
            lines.append(f"slide_{index + 1:03d} @ {timestamp:.2f}s: {text}")
    return "\n".join(lines)


def _format_scheduled_talks_for_prompt(scheduled_talks: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for index, talk in enumerate(scheduled_talks, start=1):
        title = str(talk.get("title") or "").strip()
        if not title:
            continue
        speakers = str(talk.get("speakers") or "").strip()
        lines.append(f"{index}. {title}" + (f" -- {speakers}" if speakers else ""))
    return "\n".join(lines)


def build_workshop_talk_discovery_prompt(
    *,
    workshop_id: str,
    workshop_title: str,
    segments: list[dict[str, Any]],
    slides: list[dict[str, Any]] | None = None,
    scheduled_talks: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Prompt for one recording.

    ``scheduled_talks`` is the event's own program (title, optional speakers,
    in program order) when the event page lists its talks -- an oral session
    lists its orals. The model then names each talk it finds with the
    scheduled title verbatim, which is what indexing and embedding key on.
    """

    slide_block = _format_slides_for_prompt(slides or [])
    schedule_block = _format_scheduled_talks_for_prompt(scheduled_talks or [])
    schedule_section = (
        f"""
PROGRAM (the talks this event scheduled, in program order):
{schedule_block}

Program rules:
- When a talk in the transcript is one of the scheduled talks, set talk_title to the
  scheduled title EXACTLY as listed above (same words, same casing).
- Scheduled talks may be missing (cancelled, not recorded) or out of order: only
  return talks that are actually in the transcript, with their real positions.
- Anything that is not a scheduled talk (opening, chair introduction, Q&A panel,
  break) gets a short descriptive title of its own.
"""
        if schedule_block
        else ""
    )
    return f"""WORKSHOP TALK SEGMENTATION TASK

Analyze this workshop transcript and identify each distinct talk or presentation.
Use slide timestamps as supporting evidence when available.

Return only valid JSON with this schema:
{{
  "workshop_id": "{workshop_id}",
  "workshop_title": "{workshop_title}",
  "talks": [
    {{
      "talk_id": "talk_001",
      "talk_title": "Human-readable title",
      "talk_speaker": "Speaker name or Unknown",
      "timestamp_start": 0.0,
      "timestamp_end": 300.0,
      "description": "One sentence topic summary"
    }}
  ]
}}

Rules:
- Use monotonic timestamps in seconds.
- Do not invent precise speaker names when the transcript does not identify them; use "Unknown".
- Prefer a talk boundary when the speaker, title slide, moderator transition, or topic clearly changes.
- Cover the full workshop duration without overlapping talks unless the source truly overlaps.

WORKSHOP: {workshop_title} ({workshop_id})
{schedule_section}
TRANSCRIPT:
{_format_segments_for_prompt(segments)}

SLIDES:
{slide_block or "No slide evidence supplied."}
"""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_payload(text: str) -> Any:
    match = _JSON_FENCE_RE.search(text)
    if match:
        text = match.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                text = candidate
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                text = candidate
        try:
            from json_repair import loads as repair_json_loads  # type: ignore

            return repair_json_loads(text)
        except Exception:
            raise


def parse_talk_discovery_response(
    text: str,
    *,
    default_workshop_id: str,
    default_workshop_title: str,
) -> list[WorkshopTalk]:
    payload = _extract_json_payload(text)
    talks = [
        normalize_talk_record(
            record,
            index=index,
            default_workshop_id=default_workshop_id,
            default_workshop_title=default_workshop_title,
        )
        for index, record in enumerate(_iter_raw_talks(payload))
    ]
    deduped: list[WorkshopTalk] = []
    seen: set[tuple[str, float, float]] = set()
    for talk in sorted(talks, key=lambda item: (item.timestamp_start, item.timestamp_end, item.talk_id)):
        key = (talk.talk_id, round(talk.timestamp_start, 3), round(talk.timestamp_end, 3))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(talk)
    return deduped


# The boundary-discovery model reasons before it answers, and reasoning tokens
# count against the output cap. At 8,000 a long session's reasoning used the
# whole budget and the response carried no text at all (ICML 2026 rebuild,
# 2026-09-28: 3 of 53 recordings, total_tokens = input + ~8,000).
WORKSHOP_TALK_DISCOVERY_MAX_OUTPUT_TOKENS = 32_768


def _provider_config(max_output_tokens: int = WORKSHOP_TALK_DISCOVERY_MAX_OUTPUT_TOKENS) -> Any:
    try:
        from google.genai import types  # type: ignore

        return types.GenerateContentConfig(max_output_tokens=max_output_tokens, temperature=0.0)
    except Exception:  # pragma: no cover - exercised only when google-genai is absent.
        return {"max_output_tokens": max_output_tokens, "temperature": 0.0}


def _build_gemini_client() -> Any:
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional SDK install.
        raise RuntimeError("google-genai SDK is required for live workshop talk discovery") from exc
    return genai.Client()


def discover_talks_with_gemini(
    *,
    client: Any,
    model: str,
    workshop_id: str,
    workshop_title: str,
    segments: list[dict[str, Any]],
    slides: list[dict[str, Any]] | None = None,
    scheduled_talks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[WorkshopTalk], list[str], float | None]:
    prompt = build_workshop_talk_discovery_prompt(
        workshop_id=workshop_id,
        workshop_title=workshop_title,
        segments=segments,
        slides=slides,
        scheduled_talks=scheduled_talks,
    )
    result = gemini_generate_content(
        client,
        caller=__file__,
        model=model,
        contents=prompt,
        config=_provider_config(),
    )
    text = str(getattr(result, "text", "") or "")
    talks = parse_talk_discovery_response(
        text,
        default_workshop_id=workshop_id,
        default_workshop_title=workshop_title,
    )
    estimated_cost = result.receipt.estimated_cost_usd
    return talks, [str(result.receipt.receipt_path)], estimated_cost


def write_talk_segmentation_receipt(
    *,
    path: Path,
    workspace: Path,
    status: str,
    talks: list[WorkshopTalk],
    output_path: Path | None = None,
    blocker_code: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "receipt_type": WORKSHOP_TALK_SEGMENTATION_RECEIPT_TYPE,
        "schema_version": 1,
        "owner_id": WORKSHOP_TALK_SEGMENTATION_OWNER_ID,
        "status": status,
        "generated_at": utc_now_iso(),
        "workspace": str(workspace),
        "talk_count": len(talks),
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
    }
    if output_path is not None:
        receipt["output_path"] = workspace_relpath(workspace, output_path)
        if output_path.exists():
            receipt["output_sha256"] = sha256_file(output_path)
    if blocker_code:
        receipt["blocker_code"] = blocker_code
    if error:
        receipt["error"] = safe_receipt_error(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def write_talk_discovery_receipt(
    *,
    path: Path,
    workspace: Path,
    status: str,
    workshop_id: str,
    workshop_title: str,
    provider: str,
    model: str,
    talks: list[WorkshopTalk],
    batch_count: int = 0,
    output_talks_json: Path | None = None,
    output_batches_json: Path | None = None,
    annotated_transcript_json: Path | None = None,
    provider_receipt_paths: list[str] | None = None,
    estimated_cost_usd: float | None = None,
    blocker_code: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "receipt_type": WORKSHOP_TALK_DISCOVERY_RECEIPT_TYPE,
        "schema_version": 1,
        "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
        "status": status,
        "generated_at": utc_now_iso(),
        "workspace": str(workspace),
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
        "cost_task_id": WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
        "provider": provider,
        "model": model,
        "talk_count": len(talks),
        "batch_count": batch_count,
        "provider_calls_made": bool(provider_receipt_paths),
        "live_provider_calls_made": bool(provider_receipt_paths),
        "provider_receipt_paths": provider_receipt_paths or [],
        "estimated_cost_usd": estimated_cost_usd,
        "secrets_copied": False,
    }
    for key, output_path in (
        ("output_talks_json", output_talks_json),
        ("output_batches_json", output_batches_json),
        ("annotated_transcript_json", annotated_transcript_json),
    ):
        if output_path is None:
            continue
        receipt[key] = workspace_relpath(workspace, output_path)
        if output_path.exists():
            receipt[f"{key}_sha256"] = sha256_file(output_path)
    if blocker_code:
        receipt["blocker_code"] = blocker_code
    if error:
        receipt["error"] = safe_receipt_error(error)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def run_workshop_talk_segmentation(
    *,
    workspace: Path,
    input_transcript_json: Path | None = None,
    talks_json: Path | None = None,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if talks_json is None:
        return write_talk_segmentation_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            talks=[],
            blocker_code=PROVIDER_NOT_CONFIGURED_CODE,
            error="No talks.json or governed talk-segmentation provider was supplied.",
        )

    talks = load_talks_json(talks_json)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "owner_id": WORKSHOP_TALK_SEGMENTATION_OWNER_ID,
        "talks": [talk.to_record() for talk in talks],
        "provider_calls_made": False,
        "secrets_copied": False,
    }
    if input_transcript_json is not None and input_transcript_json.exists():
        segments = load_transcript_segments(input_transcript_json, workspace=workspace)
        if segments:
            payload["segments"] = annotate_segments_with_talks(segments, talks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return write_talk_segmentation_receipt(
        path=receipt_path,
        workspace=workspace,
        status="succeeded",
        talks=talks,
        output_path=output_path,
    )


def run_workshop_talk_boundary_discovery(
    *,
    workspace: Path,
    input_transcript_json: Path,
    output_talks_json: Path,
    output_batches_json: Path,
    receipt_path: Path,
    workshop_id: str,
    workshop_title: str,
    slides_json: Path | None = None,
    annotated_transcript_json: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    provider_cost_approved: bool = False,
    run_id: str | None = None,
    stage_id: str = "transcription-ocr",
    client: Any | None = None,
    batch_seconds: float = WORKSHOP_TALK_DISCOVERY_DEFAULT_BATCH_SECONDS,
    scheduled_talks: Sequence[Mapping[str, Any]] | None = None,
    talks_extra: Mapping[str, Any] | None = None,
    annotated_extra: Mapping[str, Any] | None = None,
    synthetic_segment_seconds: float = SYNTHETIC_TRANSCRIPT_SEGMENT_SECONDS,
) -> dict[str, Any]:
    """Discover a recording's talks and write talks / batches / annotated JSON.

    ``talks_extra`` / ``annotated_extra`` are merged into the talks and
    annotated payloads -- a talk map (``indexing.talk_maps``) records its
    ``video_id`` and ``source_transcript_sha256`` there. The annotated
    transcript always names the digest of the talks file it was built from
    (``talks_sha256``), so indexing can verify the chain.
    """

    workspace = workspace.resolve()
    provider = provider or default_provider(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    model = model or default_model(WORKSHOP_TALK_DISCOVERY_TASK_ID)
    validate_selection(WORKSHOP_TALK_DISCOVERY_TASK_ID, provider=provider, model=model)

    if not provider_cost_approved:
        return write_talk_discovery_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            workshop_id=workshop_id,
            workshop_title=workshop_title,
            provider=provider,
            model=model,
            talks=[],
            blocker_code=PROVIDER_COST_NOT_APPROVED_CODE,
            error="Workshop talk boundary discovery requires explicit provider cost approval.",
        )

    segments = load_transcript_segments(
        input_transcript_json, workspace=workspace, synthetic_segment_seconds=synthetic_segment_seconds
    )
    slides = load_slide_records(slides_json)
    provider_receipt_paths: list[str] = []
    estimated_cost_usd: float | None = None
    if not segments:
        return write_talk_discovery_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            workshop_id=workshop_id,
            workshop_title=workshop_title,
            provider=provider,
            model=model,
            talks=[],
            blocker_code=PROVIDER_DISCOVERY_FAILED_CODE,
            error="No transcript text or timestamped segments were available for workshop talk discovery.",
        )
    try:
        with provider_receipt_context(
            workspace_root=workspace,
            run_id=run_id,
            stage_id=stage_id,
            caller=__file__,
        ):
            talks, provider_receipt_paths, estimated_cost_usd = discover_talks_with_gemini(
                client=client or _build_gemini_client(),
                model=model,
                workshop_id=workshop_id,
                workshop_title=workshop_title,
                segments=segments,
                slides=slides,
                scheduled_talks=scheduled_talks,
            )
    except Exception as exc:  # noqa: BLE001 - receipts must preserve provider failures.
        receipt = getattr(exc, "provider_receipt", None)
        if receipt is not None and getattr(receipt, "receipt_path", None):
            provider_receipt_paths.append(str(receipt.receipt_path))
            estimated_cost_usd = getattr(receipt, "estimated_cost_usd", None)
        return write_talk_discovery_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            workshop_id=workshop_id,
            workshop_title=workshop_title,
            provider=provider,
            model=model,
            talks=[],
            provider_receipt_paths=provider_receipt_paths,
            estimated_cost_usd=estimated_cost_usd,
            blocker_code=PROVIDER_DISCOVERY_FAILED_CODE,
            error=safe_receipt_error(exc, "Workshop talk boundary discovery failed."),
        )

    if not talks:
        return write_talk_discovery_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            workshop_id=workshop_id,
            workshop_title=workshop_title,
            provider=provider,
            model=model,
            talks=[],
            provider_receipt_paths=provider_receipt_paths,
            estimated_cost_usd=estimated_cost_usd,
            blocker_code=PROVIDER_DISCOVERY_FAILED_CODE,
            error="Gemini response did not include any talk records.",
        )

    invalid_boundaries = invalid_talk_boundary_reasons(talks)
    if invalid_boundaries:
        return write_talk_discovery_receipt(
            path=receipt_path,
            workspace=workspace,
            status="blocked",
            workshop_id=workshop_id,
            workshop_title=workshop_title,
            provider=provider,
            model=model,
            talks=talks,
            provider_receipt_paths=provider_receipt_paths,
            estimated_cost_usd=estimated_cost_usd,
            blocker_code=PROVIDER_DISCOVERY_FAILED_CODE,
            error=f"invalid_talk_boundaries:{len(invalid_boundaries)}",
        )

    batches = build_talk_context_batches(segments, talks, batch_seconds=batch_seconds)
    annotated = annotate_segments_with_talks(segments, talks)
    output_talks_json.parent.mkdir(parents=True, exist_ok=True)
    output_talks_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
                "workshop_id": workshop_id,
                "workshop_title": workshop_title,
                "model_governance_task_id": WORKSHOP_TALK_DISCOVERY_TASK_ID,
                "cost_task_id": WORKSHOP_TALK_DISCOVERY_COST_TASK_ID,
                "provider": provider,
                "model": model,
                "talks": [talk.to_record() for talk in talks],
                "provider_receipt_paths": provider_receipt_paths,
                "estimated_cost_usd": estimated_cost_usd,
                "secrets_copied": False,
                **dict(talks_extra or {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_batches_json.parent.mkdir(parents=True, exist_ok=True)
    output_batches_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
                "workshop_id": workshop_id,
                "workshop_title": workshop_title,
                "batch_seconds": batch_seconds,
                "batches": batches,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if annotated_transcript_json is not None:
        annotated_transcript_json.parent.mkdir(parents=True, exist_ok=True)
        annotated_transcript_json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "owner_id": WORKSHOP_TALK_DISCOVERY_OWNER_ID,
                    "talks_sha256": sha256_file(output_talks_json),
                    **dict(annotated_extra or {}),
                    "segments": annotated,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    return write_talk_discovery_receipt(
        path=receipt_path,
        workspace=workspace,
        status="succeeded",
        workshop_id=workshop_id,
        workshop_title=workshop_title,
        provider=provider,
        model=model,
        talks=talks,
        batch_count=len(batches),
        output_talks_json=output_talks_json,
        output_batches_json=output_batches_json,
        annotated_transcript_json=annotated_transcript_json,
        provider_receipt_paths=provider_receipt_paths,
        estimated_cost_usd=estimated_cost_usd,
    )
