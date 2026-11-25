"""LLM full-read extraction lane for topic evidence.

This lane is intentionally separate from the heuristic-first 14A classifier.
It uses the same upstream walker and local-LLM backend infrastructure, but it
asks a local model to read every usable transcript segment and, later, every
paper abstract. The heuristic output remains attached as comparison evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from conference_pipeline.extractions.backends import select_local_llm_backend
from conference_pipeline.extractions.records import (
    BY_TOPIC_SCHEMA_VERSION,
    safe_artifact_filename,
)
from conference_pipeline.extractions.walker import (
    ExtractionInput,
    iterate_extraction_inputs,
)
from conference_pipeline.iclr2025_live import (
    COMPLETED_MEDIA_PROCESSING_STATUSES,
    ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL,
    TopicFamily,
    load_topic_families,
    sanitize_for_receipt,
    sha256_text,
    utc_now,
)
from conference_pipeline.receipt_lineage import (
    sanitize_receipt_for_write,
    stamp_substep_receipt,
    unbind_substep_receipt,
)
from conference_pipeline.workflow_runs import write_json_atomic

LLM_FULL_READ_SCHEMA_VERSION = 1
DEFAULT_SEGMENT_SECONDS = 180
DEFAULT_OVERLAP_SECONDS = 60
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_LOCAL_MODEL = ICLR_QWEN35B_A3B_EXHAUSTIVE_MODEL
DEFAULT_PRIORITY_TEXT_CHARS = 4000
PRIORITY_STOPWORDS = {
    "2026",
    "about",
    "abstract",
    "accepted",
    "analysis",
    "conference",
    "icml",
    "learning",
    "machine",
    "model",
    "models",
    "paper",
    "papers",
    "research",
    "talk",
    "topic",
    "workshop",
    "workshops",
}


@dataclass(frozen=True)
class FullReadInput:
    """One model-readable unit in transcript-first order."""

    unit_id: str
    source_artifact_id: str
    input_kind: str
    text: str
    source_artifact_path: str
    source_artifact_digest: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullReadResult:
    processed_count: int = 0
    skipped_existing_count: int = 0
    metadata_refreshed_count: int = 0
    skipped_empty_count: int = 0
    important_count: int = 0
    not_important_count: int = 0
    failed_count: int = 0
    transcript_unit_count: int = 0
    paper_abstract_count: int = 0
    elapsed_seconds: float = 0.0
    input_chars: int = 0
    output_tokens: int = 0
    input_tokens: int = 0
    provider_receipt_paths: list[str] = field(default_factory=list)


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _topic_prompt_block(topics: Iterable[TopicFamily], workspace_root: Path) -> str:
    strategy_path = workspace_root / "analysis" / "topic_strategy.json"
    rationales: dict[str, str] = {}
    queries: dict[str, list[str]] = {}
    payload = _load_json(strategy_path)
    if payload:
        for topic in payload.get("final_topics") or []:
            if not isinstance(topic, dict):
                continue
            topic_id = str(topic.get("topic_id") or topic.get("id") or "")
            rationales[topic_id] = str(topic.get("rationale") or "")
            starter_queries = topic.get("starter_queries")
            if isinstance(starter_queries, list):
                queries[topic_id] = [str(item) for item in starter_queries[:3]]

    lines: list[str] = []
    for topic in topics:
        detail = rationales.get(topic.id, "")
        starter = "; ".join(queries.get(topic.id, []))
        suffix = ""
        if detail:
            suffix += f" Rationale: {detail}"
        if starter:
            suffix += f" Search cues: {starter}"
        lines.append(f"- {topic.id}: {topic.title}.{suffix}")
    return "\n".join(lines)


def _topic_priority_terms(
    workspace_root: Path,
    topic_ids: Iterable[str],
) -> list[str]:
    wanted = {str(topic_id).strip() for topic_id in topic_ids if str(topic_id).strip()}
    if not wanted:
        return []
    strategy_path = workspace_root / "analysis" / "topic_strategy.json"
    payload = _load_json(strategy_path)
    if not payload:
        return []
    terms: list[str] = []
    for topic in payload.get("final_topics") or []:
        if not isinstance(topic, dict):
            continue
        topic_id = str(topic.get("topic_id") or topic.get("id") or "").strip()
        if topic_id not in wanted:
            continue
        for field_name in ("title", "rationale"):
            value = topic.get(field_name)
            if isinstance(value, str) and value.strip():
                terms.append(value)
        starter_queries = topic.get("starter_queries")
        if isinstance(starter_queries, list):
            terms.extend(str(item) for item in starter_queries if str(item).strip())
    return terms


def _priority_phrases(terms: Iterable[str]) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()
    for term in terms:
        text = str(term or "").lower().strip()
        if not text:
            continue
        candidates = [text]
        candidates.extend(part.strip() for part in text.replace("/", " ").split())
        for candidate in candidates:
            if len(candidate) < 4:
                continue
            if candidate in PRIORITY_STOPWORDS or candidate.isdigit():
                continue
            if candidate not in seen:
                seen.add(candidate)
                phrases.append(candidate)
    return phrases


def _unit_priority_text(unit: FullReadInput) -> str:
    metadata_parts: list[str] = []
    for key in (
        "title",
        "keywords",
        "workshop_title",
        "talk_title",
        "talk_speaker",
        "description",
        "paper_id",
    ):
        value = unit.metadata.get(key)
        if isinstance(value, list):
            metadata_parts.extend(str(item) for item in value)
        elif value is not None:
            metadata_parts.append(str(value))
    metadata_parts.append(unit.source_artifact_id)
    metadata_parts.append(unit.source_artifact_path)
    metadata_parts.append(unit.text[:DEFAULT_PRIORITY_TEXT_CHARS])
    return "\n".join(part for part in metadata_parts if part).lower()


def _prioritize_units(
    units: list[FullReadInput],
    *,
    workspace_root: Path,
    priority_topic_ids: Iterable[str] = (),
    priority_queries: Iterable[str] = (),
) -> list[FullReadInput]:
    topic_terms = _topic_priority_terms(workspace_root, priority_topic_ids)
    phrases = _priority_phrases([*topic_terms, *priority_queries])
    if not phrases:
        return units

    indexed_units = list(enumerate(units))

    def sort_key(item: tuple[int, FullReadInput]) -> tuple[int, int, int, int]:
        index, unit = item
        haystack = _unit_priority_text(unit)
        score = sum(1 for phrase in phrases if phrase in haystack)
        # For priority matches, surface paper abstracts before generic talk
        # segments so paper evidence can arrive early without changing the
        # all-scope completion contract.
        paper_rank = 0 if unit.input_kind == "paper_abstract" else 1
        return (0 if score else 1, -score, paper_rank, index)

    return [unit for _, unit in sorted(indexed_units, key=sort_key)]


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                text = candidate
                break
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _provider_metadata(response: Any, *, model: str, elapsed_seconds: float) -> dict[str, Any]:
    receipt = getattr(response, "receipt", None)
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(receipt, "input_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(receipt, "output_tokens", None)
    return {
        "provider": getattr(receipt, "provider", None) or "local",
        "model": getattr(receipt, "model", None) or model,
        "operation": getattr(receipt, "operation", None) or "chat.completions",
        "provider_receipt_path": getattr(receipt, "receipt_path", None),
        "request_id": getattr(receipt, "response_id", None) or getattr(response, "id", None),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": 0.0,
        "estimated_paid_cost_usd": 0.0,
        "cost_status": "local_no_paid_provider",
        "elapsed_seconds": elapsed_seconds,
    }


def _heuristic_record(
    workspace_root: Path,
    run_id: str,
    artifact_id: str,
) -> dict[str, Any] | None:
    path = (
        workspace_root
        / "storage"
        / "extractions"
        / run_id
        / "by-artifact"
        / f"{safe_artifact_filename(artifact_id)}.json"
    )
    return _load_json(path)


def _enrich_transcript_metadata(
    workspace_root: Path,
    input_: ExtractionInput,
    block: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(input_.metadata)
    if block:
        for key in (
            "talk_id",
            "talk_title",
            "talk_speaker",
            "workshop_id",
            "workshop_title",
            "description",
            "synthetic_timestamp",
            "synthetic_timestamp_basis",
        ):
            if key in block:
                metadata[key] = block[key]
    transcript_json = metadata.get("transcript_json_path")
    if isinstance(transcript_json, str) and transcript_json.strip():
        payload = _load_json(workspace_root / transcript_json)
        if payload:
            for key in ("source_url", "media_url", "local_path", "source_path"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    metadata[key] = value
    metadata["source_artifact_path"] = input_.source_path
    return metadata


def _slice_text_by_time(
    *,
    text: str,
    block_start: float,
    block_end: float,
    window_start: float,
    window_end: float,
) -> str:
    if block_end <= block_start:
        return text
    start_ratio = max(0.0, min(1.0, (window_start - block_start) / (block_end - block_start)))
    end_ratio = max(0.0, min(1.0, (window_end - block_start) / (block_end - block_start)))
    start_idx = int(len(text) * start_ratio)
    end_idx = int(len(text) * end_ratio)
    return text[start_idx:end_idx].strip()


_TALK_IDENTITY_KEYS = ("talk_id", "talk_title", "workshop_id")


def _coalesce_annotated_talks(segments: Iterable[Any]) -> list[dict[str, Any]]:
    """Merge consecutive annotated blocks that belong to the same talk.

    Returns one entry per talk run: ``text`` (all block text joined), ``start``
    and ``end`` (seconds), ``cues`` (``(start, text)`` per source block, used to
    cut time windows on real boundaries), and ``meta`` (the first block, which
    carries the talk identity and description).
    """

    talks: list[dict[str, Any]] = []
    previous_identity: tuple[Any, ...] | None = None
    for block in segments:
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        start = float(block.get("timestamp_start") or block.get("start") or 0.0)
        end = float(block.get("timestamp_end") or block.get("end") or start)
        identity = tuple(block.get(key) for key in _TALK_IDENTITY_KEYS)
        if talks and identity == previous_identity:
            talk = talks[-1]
            talk["cues"].append((start, text))
            talk["end"] = max(talk["end"], end)
        else:
            talks.append({"meta": block, "start": start, "end": end, "cues": [(start, text)]})
        previous_identity = identity
    for talk in talks:
        talk["text"] = " ".join(cue_text for _, cue_text in talk["cues"])
    return talks


def _char_windows(text: str, *, segment_seconds: int, step: int, min_chars: int) -> Iterator[str]:
    max_chars = max(min_chars, segment_seconds * 35)
    step_chars = max(min_chars, int(max_chars * (step / segment_seconds)))
    for start_idx in range(0, len(text), step_chars):
        chunk = text[start_idx : start_idx + max_chars].strip()
        if len(chunk) >= min_chars:
            yield chunk


def _transcript_units_for_input(
    workspace_root: Path,
    input_: ExtractionInput,
    *,
    segment_seconds: int,
    overlap_seconds: int,
    min_chars: int,
) -> Iterator[FullReadInput]:
    annotated_rel = input_.metadata.get("annotated_transcript_json_path")
    annotated_path = workspace_root / str(annotated_rel) if isinstance(annotated_rel, str) else None
    annotated = _load_json(annotated_path) if annotated_path else None
    step = max(1, segment_seconds - overlap_seconds)

    if annotated and isinstance(annotated.get("segments"), list):
        segment_index = 0
        # Annotated transcripts come at any granularity: one block per talk, one
        # per sentence, or one per caption cue (~40 chars). ``min_chars`` is a
        # floor on a read unit, never on a source block, so blocks are coalesced
        # into talks before it applies.
        for block in _coalesce_annotated_talks(annotated["segments"]):
            text = block["text"]
            if len(text) < min_chars:
                continue
            cues = block["cues"]
            start = float(block["start"])
            end = float(block["end"])
            if len(cues) > 1 and end <= start:
                # Several blocks and no usable timing: window by characters and
                # keep null timestamps rather than inventing them.
                for chunk_text in _char_windows(
                    text, segment_seconds=segment_seconds, step=step, min_chars=min_chars
                ):
                    metadata = _enrich_transcript_metadata(workspace_root, input_, block["meta"])
                    metadata.update(
                        {
                            "timestamp_start": None,
                            "timestamp_end": None,
                            "segment_seconds": segment_seconds,
                            "overlap_seconds": overlap_seconds,
                            "segmentation_policy": "text_window_no_timestamps",
                        }
                    )
                    yield FullReadInput(
                        unit_id=f"{input_.artifact_id}:segment:{segment_index:05d}",
                        source_artifact_id=input_.artifact_id,
                        input_kind="transcript_segment",
                        text=chunk_text,
                        source_artifact_path=input_.source_path,
                        source_artifact_digest=input_.digest,
                        metadata=metadata,
                    )
                    segment_index += 1
                continue
            if end <= start:
                end = start + segment_seconds
            window_start = start
            while window_start < end:
                window_end = min(end, window_start + segment_seconds)
                if len(cues) > 1:
                    chunk_text = " ".join(
                        cue_text
                        for cue_start, cue_text in cues
                        if window_start <= cue_start < window_end
                    ).strip()
                else:
                    chunk_text = _slice_text_by_time(
                        text=text,
                        block_start=start,
                        block_end=end,
                        window_start=window_start,
                        window_end=window_end,
                    )
                if len(chunk_text) >= min_chars:
                    metadata = _enrich_transcript_metadata(workspace_root, input_, block["meta"])
                    metadata.update(
                        {
                            "timestamp_start": round(window_start, 3),
                            "timestamp_end": round(window_end, 3),
                            "segment_seconds": segment_seconds,
                            "overlap_seconds": overlap_seconds,
                            "segmentation_policy": "three_minutes_with_one_minute_overlap",
                        }
                    )
                    unit_id = f"{input_.artifact_id}:segment:{segment_index:05d}"
                    yield FullReadInput(
                        unit_id=unit_id,
                        source_artifact_id=input_.artifact_id,
                        input_kind="transcript_segment",
                        text=chunk_text,
                        source_artifact_path=input_.source_path,
                        source_artifact_digest=input_.digest,
                        metadata=metadata,
                    )
                    segment_index += 1
                if window_end >= end:
                    break
                window_start += step
        if segment_index:
            return
        # An annotated transcript that yields nothing must not hide the plain
        # text the input also carries; fall through to text windowing.

    text = input_.text_content or ""
    if len(text) < min_chars:
        return
    # No timestamps available. Keep a deterministic text window with null time
    # metadata rather than inventing timestamps.
    max_chars = max(min_chars, segment_seconds * 35)
    step_chars = max(min_chars, int(max_chars * (step / segment_seconds)))
    segment_index = 0
    for start_idx in range(0, len(text), step_chars):
        chunk = text[start_idx : start_idx + max_chars].strip()
        if len(chunk) < min_chars:
            continue
        metadata = _enrich_transcript_metadata(workspace_root, input_)
        metadata.update(
            {
                "timestamp_start": None,
                "timestamp_end": None,
                "segment_seconds": segment_seconds,
                "overlap_seconds": overlap_seconds,
                "segmentation_policy": "text_window_no_timestamps",
            }
        )
        yield FullReadInput(
            unit_id=f"{input_.artifact_id}:segment:{segment_index:05d}",
            source_artifact_id=input_.artifact_id,
            input_kind="transcript_segment",
            text=chunk,
            source_artifact_path=input_.source_path,
            source_artifact_digest=input_.digest,
            metadata=metadata,
        )
        segment_index += 1


def _annotated_text_chars(workspace_root: Path, input_: ExtractionInput) -> int:
    annotated_rel = input_.metadata.get("annotated_transcript_json_path")
    if not isinstance(annotated_rel, str):
        return 0
    annotated = _load_json(workspace_root / annotated_rel)
    if not annotated or not isinstance(annotated.get("segments"), list):
        return 0
    return sum(
        len(str(block.get("text") or "").strip()) + 1
        for block in annotated["segments"]
        if isinstance(block, dict)
    )


def iter_transcript_units(
    workspace_root: Path,
    *,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    coverage: list[dict[str, Any]] | None = None,
) -> Iterator[FullReadInput]:
    """Yield every transcript read unit.

    ``coverage`` (when given) receives one row per transcript input: how much
    text the input carries and how many units it produced. The unit count a
    run "expects" is derived from this same iterator, so an input that yields
    nothing is invisible to completed==expected; the coverage rows are what
    let the caller see it.
    """

    for input_ in iterate_extraction_inputs(workspace_root):
        if input_.kind != "transcript":
            continue
        unit_count = 0
        for unit in _transcript_units_for_input(
            workspace_root,
            input_,
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            min_chars=min_chars,
        ):
            unit_count += 1
            yield unit
        if coverage is not None:
            coverage.append(
                {
                    "source_artifact_id": input_.artifact_id,
                    "source_artifact_path": input_.source_path,
                    "storage_lane": str(input_.metadata.get("storage_lane") or ""),
                    "indexed_text_kind": str(input_.metadata.get("indexed_text_kind") or ""),
                    "text_chars": max(
                        len(input_.text_content or ""),
                        _annotated_text_chars(workspace_root, input_),
                    ),
                    "unit_count": unit_count,
                }
            )


def transcript_source_coverage(
    coverage: Iterable[Mapping[str, Any]], *, min_chars: int
) -> dict[str, Any]:
    """Summarize per-input coverage; ``unread`` lists inputs with text and no unit."""

    rows = list(coverage)
    unread = [
        row
        for row in rows
        if int(row.get("unit_count") or 0) == 0 and int(row.get("text_chars") or 0) >= min_chars
    ]
    by_lane: dict[str, dict[str, int]] = {}
    for row in rows:
        lane = by_lane.setdefault(
            str(row.get("storage_lane") or "unknown"),
            {"input_count": 0, "unread_input_count": 0, "unit_count": 0, "text_chars": 0},
        )
        lane["input_count"] += 1
        lane["unit_count"] += int(row.get("unit_count") or 0)
        lane["text_chars"] += int(row.get("text_chars") or 0)
    for row in unread:
        by_lane[str(row.get("storage_lane") or "unknown")]["unread_input_count"] += 1
    return {
        "input_count": len(rows),
        "read_input_count": sum(1 for row in rows if int(row.get("unit_count") or 0) > 0),
        "unread_input_count": len(unread),
        "unread_text_chars": sum(int(row.get("text_chars") or 0) for row in unread),
        "unread_inputs": sorted(unread, key=lambda row: -int(row.get("text_chars") or 0))[:50],
        "by_storage_lane": dict(sorted(by_lane.items())),
    }


def iter_paper_abstract_units(
    workspace_root: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Iterator[FullReadInput]:
    """Paper-abstract units: the OpenReview export when the conference has one
    (ICLR, ICML), otherwise the corpus inventory's paper texts (CVF and
    workshop papers), whose abstracts are cut from the extracted PDF text."""

    yielded = False
    for unit in _iter_openreview_abstract_units(workspace_root, min_chars=min_chars):
        yielded = True
        yield unit
    if not yielded:
        yield from iter_corpus_paper_abstract_units(workspace_root, min_chars=min_chars)


# The abstract of an extracted PDF text: after an "Abstract" heading, up to
# the first section heading. The heading may open a line ("Abstract",
# "1. Abstract", run-in "Abstract—We ..."), or sit mid-line after the author
# block ("...uark.edu  Abstract We present ..."), and review copies glue line
# numbers to it ("Abstract001").
_ABSTRACT_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:\d{1,2}\.?[ \t]*)?abstract(?:\d{3})?\b[ \t]*[.:\u2014\u2013-]?[ \t]*"
)
_ABSTRACT_INLINE_RE = re.compile(r"(?<![A-Za-z])(?:Abstract|ABSTRACT)(?:\d{3})?[ \t]*[.:\u2014\u2013-]?\s+(?=[A-Z])")
# Section headings are case-sensitive ("1. Introduction", "I. INTRODUCTION"):
# a line such as "1 predictions, yet ..." (a hyphenated "top-1") is not one.
_ABSTRACT_END_RE = re.compile(
    r"(?m)^[ \t]*(?:(?:1|I)\.?[ \t]+[A-Z][A-Za-z][^\n]{0,80}$"
    r"|(?:(?:1|I)\.?[ \t]+)?(?i:introduction|keywords?|index terms|ccs concepts)\b)"
)
# Without a heading the abstract follows the author block: start after the
# last email address, URL or affiliation line near the top.
_AUTHOR_BLOCK_END_RE = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\{[^}\n]{1,200}\}@\S+"
    r"|https?://\S+"
    r"|(?m:^[^\n]{0,120}\b(?:University|Universit\w+|Institute|College|Laborator(?:y|ies)|Academy|"
    r"Inc\.|Ltd\.?|Corporation|Research)\b[^\n]{0,120}$)"
)
ABSTRACT_SEARCH_CHARS = 5000
ABSTRACT_MAX_CHARS = 4000
ABSTRACT_AUTHOR_BLOCK_CHARS = 2000
ABSTRACT_FALLBACK_CHARS = 2000


def _join_pdf_lines(text: str) -> str:
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _abstract_body(text: str) -> str:
    body = text[:ABSTRACT_MAX_CHARS]
    end = _ABSTRACT_END_RE.search(body)
    if end is not None and end.start() > 0:
        body = body[: end.start()]
    return _join_pdf_lines(body)


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _heading_match(head: str, title: str) -> re.Match[str] | None:
    """First Abstract heading that is not the word "Abstract" in the title."""

    normalized_title = _normalized_words(title)
    for pattern in (_ABSTRACT_HEADING_RE, _ABSTRACT_INLINE_RE):
        for match in pattern.finditer(head):
            line_start = head.rfind("\n", 0, match.start()) + 1
            line_end = head.find("\n", match.start())
            line = _normalized_words(head[line_start : line_end if line_end != -1 else len(head)])
            is_bare_heading = len(line.split()) <= 2  # "abstract", "1 abstract", "abstract001"
            if normalized_title and line and not is_bare_heading and (
                normalized_title.startswith(line) or line.startswith(normalized_title[:40])
            ):
                continue
            return match
    return None


def extract_paper_abstract(text: str, *, title: str = "") -> tuple[str, str]:
    """``(abstract, source)`` from extracted paper text.

    ``source`` is ``abstract_heading`` when an Abstract heading was found in
    the first pages; ``after_author_block`` when there is none and the text
    after the last email/URL/affiliation near the top is used; otherwise
    ``leading_text_fallback`` (the opening text, title block included).
    ``title`` lets a title that contains the word ("Abstract 3D Perception
    ...") be told apart from the heading.
    """

    head = text[:ABSTRACT_SEARCH_CHARS]
    match = _heading_match(head, title)
    if match:
        return _abstract_body(text[match.end() :]), "abstract_heading"
    anchor = None
    for block_end in _AUTHOR_BLOCK_END_RE.finditer(text[:ABSTRACT_AUTHOR_BLOCK_CHARS]):
        anchor = block_end.end()
    if anchor is not None:
        body = _abstract_body(text[anchor:])
        if len(body) >= DEFAULT_MIN_CHARS:
            return body, "after_author_block"
    return _join_pdf_lines(text[:ABSTRACT_FALLBACK_CHARS]), "leading_text_fallback"


def _paper_reference_body(artifact_id: str) -> str:
    """The ``paper_abstract:<body>`` id a corpus paper canonicalizes to (see
    ``topic_reference_validation.canonicalize_reference_id``)."""

    _, _, rest = artifact_id.partition(":")
    rest = re.sub(r":\d{1,8}$", "", rest)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", rest).strip("_")


def _corpus_paper_records(workspace_root: Path) -> dict[str, Mapping[str, Any]]:
    for relpath in ("storage/corpus/v15_corpus_inventory.json", "storage/corpus/corpus_inventory.json"):
        payload = _load_json(workspace_root / relpath)
        if payload:
            records = payload.get("artifact_records")
            if isinstance(records, list):
                return {
                    str(record.get("artifact_id")): record
                    for record in records
                    if isinstance(record, dict) and record.get("artifact_id")
                }
    return {}


def iter_corpus_paper_abstract_units(
    workspace_root: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Iterator[FullReadInput]:
    """Abstract units for every completed paper text in the corpus inventory."""

    records = _corpus_paper_records(workspace_root)
    for input_ in iterate_extraction_inputs(workspace_root):
        if input_.kind != "paper_pdf" or input_.status not in COMPLETED_MEDIA_PROCESSING_STATUSES:
            continue
        text = input_.text_content
        if not text and input_.text_path:
            text_path = Path(input_.text_path)
            if not text_path.is_absolute():
                text_path = workspace_root / text_path
            try:
                text = text_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
        record = records.get(input_.artifact_id, {})
        title = str(record.get("paper_title") or record.get("title") or "").strip()
        abstract, abstract_source = extract_paper_abstract(text or "", title=title)
        if len(abstract) < min_chars:
            continue
        body = _paper_reference_body(input_.artifact_id)
        if not body:
            continue
        authors = record.get("authors") if isinstance(record.get("authors"), list) else []
        unit_text = f"Title: {title}\n\nAbstract:\n{abstract}".strip() if title else f"Abstract:\n{abstract}"
        yield FullReadInput(
            unit_id=f"paper_abstract:{body}",
            source_artifact_id=input_.artifact_id,
            input_kind="paper_abstract",
            text=unit_text,
            source_artifact_path=str(input_.text_path or input_.source_path or ""),
            source_artifact_digest=input_.digest,
            metadata={
                "paper_id": input_.artifact_id,
                "title": title,
                "authors": authors,
                "source_family_id": record.get("source_family_id") or input_.metadata.get("source_family_id"),
                "abstract_source": abstract_source,
                "source_policy": "abstract_only",
            },
        )


def count_indexable_corpus_papers(workspace_root: Path) -> int:
    """Completed paper texts in the corpus inventory (the fail-closed check)."""

    return sum(
        1
        for input_ in iterate_extraction_inputs(workspace_root)
        if input_.kind == "paper_pdf" and input_.status in COMPLETED_MEDIA_PROCESSING_STATUSES
    )


def _iter_openreview_abstract_units(
    workspace_root: Path,
    *,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Iterator[FullReadInput]:
    payload = _load_json(workspace_root / "storage" / "download-corpus" / "openreview_accepted_papers.json")
    papers = payload.get("papers") if payload else None
    if not isinstance(papers, list):
        return
    for paper in papers:
        if not isinstance(paper, dict):
            continue
        paper_id = str(paper.get("id") or paper.get("forum") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        title = str(paper.get("title") or "").strip()
        if not paper_id or len(abstract) < min_chars:
            continue
        text = f"Title: {title}\n\nAbstract:\n{abstract}".strip()
        digest = sha256_text(text)
        yield FullReadInput(
            unit_id=f"paper_abstract:{paper_id}",
            source_artifact_id=f"paper_abstract:{paper_id}",
            input_kind="paper_abstract",
            text=text,
            source_artifact_path=str(paper.get("forum_url") or paper.get("pdf_url") or ""),
            source_artifact_digest=digest,
            metadata={
                "paper_id": paper_id,
                "title": title,
                "authors": paper.get("authors") if isinstance(paper.get("authors"), list) else [],
                "keywords": paper.get("keywords") if isinstance(paper.get("keywords"), list) else [],
                "forum_url": paper.get("forum_url"),
                "pdf_url": paper.get("pdf_url"),
                "source_policy": "abstract_only",
            },
        )


def _unit_output_path(workspace_root: Path, run_id: str, unit_id: str) -> Path:
    return (
        workspace_root
        / "storage"
        / "extractions"
        / run_id
        / "llm-full-read"
        / "by-unit"
        / f"{safe_artifact_filename(unit_id)}.json"
    )


def _conference_label(workspace_root: Path) -> str:
    config_path = workspace_root / "config.yaml"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return workspace_root.name.upper()

    in_conference_block = False
    for line in lines:
        if line.startswith("conference:"):
            in_conference_block = True
            continue
        if in_conference_block and line and not line.startswith((" ", "\t")):
            break
        if in_conference_block and line.lstrip().startswith("name:"):
            value = line.split(":", 1)[1].strip().strip("\"'")
            if value:
                return value
    return workspace_root.name.upper()


def _build_messages(
    *,
    workspace_root: Path,
    unit: FullReadInput,
    topics: list[TopicFamily],
    topic_block: str,
) -> list[dict[str, str]]:
    metadata = unit.metadata
    context_lines = [
        f"Input kind: {unit.input_kind}",
        f"Workshop title: {metadata.get('workshop_title') or ''}",
        f"Talk title: {metadata.get('talk_title') or ''}",
        f"Speaker: {metadata.get('talk_speaker') or metadata.get('speaker') or ''}",
        f"Timestamp start: {metadata.get('timestamp_start')}",
        f"Timestamp end: {metadata.get('timestamp_end')}",
        f"Source URL: {metadata.get('source_url') or metadata.get('forum_url') or ''}",
    ]
    conference_label = _conference_label(workspace_root)
    system = (
        f"You are classifying {conference_label} conference evidence into the approved "
        "12-topic analysis taxonomy. Read the whole provided segment. Ignore "
        "logistics, greetings, sponsor thanks, room instructions, repetition, "
        "and other non-substantive noise by returning important=false. If the "
        "segment contains substantive research content, return the best matching "
        "topic_ids from the approved list. Membership means the segment's PRIMARY "
        "research contribution belongs to that topic: a passing mention, a shared "
        "keyword, or an application of the topic's tools to another field's problem "
        "(a wireless-networks talk saying its stack will be 'written by agents', a "
        "chemistry talk that uses a transformer) is NOT membership in the mentioned "
        "topic. Substantive research whose primary domain lies outside every "
        "approved topic returns important=false with reason 'outside_taxonomy'. "
        "Return only a JSON object."
    )
    user = (
        "Approved topics:\n"
        f"{topic_block}\n\n"
        "Metadata:\n"
        + "\n".join(context_lines)
        + "\n\n"
        "Return JSON with this exact shape:\n"
        "{\n"
        '  "important": true or false,\n'
        '  "topic_ids": ["topic_id"],\n'
        '  "confidence": 0.0,\n'
        '  "summary": "one sentence or empty string",\n'
        '  "evidence_phrases": ["short phrase copied or paraphrased from the input"],\n'
        '  "reason": "short reason"\n'
        "}\n\n"
        "Input text:\n"
        f"{unit.text}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _coerce_topic_ids(value: Any, valid_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        topic_id = str(item or "").strip()
        if topic_id in valid_ids and topic_id not in result:
            result.append(topic_id)
    return result


def _record_for_unit(
    *,
    workspace_root: Path,
    run_id: str,
    unit: FullReadInput,
    topics: list[TopicFamily],
    topic_block: str,
    backend: Any,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    valid_ids = {topic.id for topic in topics}
    heuristic = _heuristic_record(workspace_root, run_id, unit.source_artifact_id)
    started = time.monotonic()
    response = backend.chat_completion(
        caller=__file__,
        model=model,
        messages=_build_messages(
            workspace_root=workspace_root,
            unit=unit,
            topics=topics,
            topic_block=topic_block,
        ),
        response_format={"type": "json_object"},
        max_tokens=max_output_tokens,
    )
    elapsed = time.monotonic() - started
    content = response.choices[0].message.content if response.choices else "{}"
    parsed = _parse_json_object(content)
    topic_ids = _coerce_topic_ids(parsed.get("topic_ids"), valid_ids)
    important = bool(parsed.get("important")) and bool(topic_ids)
    confidence_raw = parsed.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0
    provider = _provider_metadata(response, model=model, elapsed_seconds=elapsed)
    payload = {
        "schema_version": LLM_FULL_READ_SCHEMA_VERSION,
        "lane": "llm_full_read",
        "unit_id": unit.unit_id,
        "source_artifact_id": unit.source_artifact_id,
        "source_artifact_path": unit.source_artifact_path,
        "source_artifact_digest": unit.source_artifact_digest,
        "input_kind": unit.input_kind,
        "text_sha256": sha256_text(unit.text),
        "topic_prompt_sha256": sha256_text(topic_block),
        "text_length": len(unit.text),
        "topic_ids": topic_ids if important else [],
        "important": important,
        "confidence": confidence if important else 0.0,
        "summary": str(parsed.get("summary") or "")[:1000],
        "reason": str(parsed.get("reason") or "")[:1000],
        "evidence_phrases": [
            str(item)[:300]
            for item in (parsed.get("evidence_phrases") or [])
            if isinstance(item, str) and item.strip()
        ][:5],
        "skipped_reason": None if important else "not_important",
        "metadata": unit.metadata,
        "heuristic_comparison": {
            "artifact_topic_ids": heuristic.get("topic_ids") if heuristic else [],
            "artifact_confidence": heuristic.get("confidence") if heuristic else None,
            "artifact_skipped_reason": heuristic.get("skipped_reason") if heuristic else None,
        },
        "provider_metadata": provider,
        "run_id": run_id,
        "generated_at": utc_now(),
    }
    sanitized = sanitize_for_receipt(payload)
    if not isinstance(sanitized, dict):
        raise RuntimeError("LLM full-read record sanitation failed")
    return sanitized


def _add_int_field(result: FullReadResult, field_name: str, value: Any) -> None:
    if isinstance(value, bool):
        return
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return
    setattr(result, field_name, getattr(result, field_name) + parsed)


def _accumulate_record_metrics(result: FullReadResult, record: dict[str, Any]) -> None:
    _add_int_field(result, "input_chars", record.get("text_length"))
    if record.get("skipped_reason") == "llm_failed":
        result.failed_count += 1
    if record.get("important") and record.get("topic_ids"):
        result.important_count += 1
    elif record.get("skipped_reason") == "not_important":
        result.not_important_count += 1

    provider = record.get("provider_metadata")
    if isinstance(provider, dict):
        provider_path = provider.get("provider_receipt_path")
        if isinstance(provider_path, str) and provider_path:
            result.provider_receipt_paths.append(provider_path)
        _add_int_field(result, "input_tokens", provider.get("input_tokens"))
        _add_int_field(result, "output_tokens", provider.get("output_tokens"))


def _prune_obsolete_unit_records(
    workspace_root: Path,
    run_id: str,
    units: Iterable[Any],
    *,
    protected_source_ids: frozenset[str] = frozenset(),
) -> int:
    """Move by-unit records for units no longer in the corpus out of the outputs.

    Only a complete (scope ``all``, no limit) run knows the full unit set.
    A corpus rebuild that drops a unit would otherwise leave its record in
    the by-topic aggregates and in the sub-step output digest, so consumers
    of the full read would still see evidence the corpus no longer has. The
    records are moved (not deleted) to ``llm-full-read.pruned/<stamp>/``.
    Records of ``protected_source_ids`` are never pruned.
    """

    root = workspace_root / "storage" / "extractions" / run_id / "llm-full-read" / "by-unit"
    if not root.is_dir():
        return 0
    current = {f"{safe_artifact_filename(unit.unit_id)}.json" for unit in units}
    obsolete = [path for path in sorted(root.glob("*.json")) if path.name not in current]
    if protected_source_ids:
        # A source the corpus still has, which now yields no unit, has not been
        # dropped from the corpus: its reader broke. Keep what was read from it.
        obsolete = [
            path
            for path in obsolete
            if str((_load_json(path) or {}).get("source_artifact_id") or "")
            not in protected_source_ids
        ]
    if not obsolete:
        return 0
    stamp = utc_now().replace(":", "").replace("-", "")
    archive = root.parent.parent / "llm-full-read.pruned" / stamp / "by-unit"
    archive.mkdir(parents=True, exist_ok=True)
    for path in obsolete:
        path.replace(archive / path.name)
    return len(obsolete)


def _aggregate_by_topic(workspace_root: Path, run_id: str) -> dict[str, int]:
    root = workspace_root / "storage" / "extractions" / run_id / "llm-full-read"
    by_unit = root / "by-unit"
    by_topic = root / "by-topic"
    by_topic.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(by_unit.glob("*.json")):
        payload = _load_json(path)
        if not payload:
            continue
        for topic_id in payload.get("topic_ids") or []:
            if not isinstance(topic_id, str) or not topic_id:
                continue
            grouped.setdefault(topic_id, []).append(
                {
                    "unit_id": payload.get("unit_id"),
                    "source_artifact_id": payload.get("source_artifact_id"),
                    "input_kind": payload.get("input_kind"),
                    "confidence": payload.get("confidence"),
                    "summary": payload.get("summary"),
                    "source_artifact_path": payload.get("source_artifact_path"),
                    "metadata": payload.get("metadata"),
                    "heuristic_comparison": payload.get("heuristic_comparison"),
                }
            )
    for old in by_topic.glob("*.json"):
        old.unlink()
    counts: dict[str, int] = {}
    for topic_id, items in grouped.items():
        out = by_topic / f"{safe_artifact_filename(topic_id)}.json"
        write_json_atomic(
            out,
            {
                "schema_version": BY_TOPIC_SCHEMA_VERSION,
                "lane": "llm_full_read",
                "topic_id": topic_id,
                "unit_count": len(items),
                "items": sorted(items, key=lambda item: str(item.get("unit_id") or "")),
            },
        )
        counts[topic_id] = len(items)
    return counts


def iter_units_for_scope(
    workspace_root: Path,
    *,
    scope: str,
    segment_seconds: int,
    overlap_seconds: int,
    min_chars: int,
    transcript_coverage: list[dict[str, Any]] | None = None,
) -> Iterator[FullReadInput]:
    if scope in {"transcripts", "all"}:
        yield from iter_transcript_units(
            workspace_root,
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            min_chars=min_chars,
            coverage=transcript_coverage,
        )
    if scope in {"papers", "all"}:
        yield from iter_paper_abstract_units(workspace_root, min_chars=min_chars)


def _record_matches_unit_input(record: Mapping[str, Any], *, unit: Any, topic_block: str) -> bool:
    """Whether a cached unit record was computed from this unit's current input.

    The sub-step lineage stamps the receipt as current against today's corpus
    index, so resume may keep a record only when its text is unchanged (same
    unit id, new text or metadata is re-read). The topic prompt is compared when the
    record carries its digest; records written before that field existed are
    judged on text alone.
    """

    if record.get("text_sha256") != sha256_text(unit.text):
        return False
    # The prompt also carries the unit's metadata (talk title, speaker,
    # timestamps, source URL) and input kind; every record stores both, so a
    # metadata-only corpus correction is caught without a new field.
    if record.get("input_kind") != unit.input_kind:
        return False
    if json.dumps(record.get("metadata") or {}, sort_keys=True, default=str) != json.dumps(
        unit.metadata or {}, sort_keys=True, default=str
    ):
        return False
    recorded_prompt = record.get("topic_prompt_sha256")
    return recorded_prompt is None or recorded_prompt == sha256_text(topic_block)


def _record_reusable_with_new_metadata(record: Mapping[str, Any], *, unit: Any, topic_block: str) -> bool:
    """Same text, kind and topic prompt; only the unit's labels changed.

    Titles, authors, item kinds and path forms are attached to units after a
    corpus re-index. The model's reading of unchanged text still stands, so the
    record is carried forward with the new labels instead of re-read.
    """

    if record.get("text_sha256") != sha256_text(unit.text):
        return False
    if record.get("input_kind") != unit.input_kind:
        return False
    recorded_prompt = record.get("topic_prompt_sha256")
    return recorded_prompt is None or recorded_prompt == sha256_text(topic_block)


def _carry_forward_record(record: Mapping[str, Any], *, unit: Any, run_id: str) -> dict[str, Any]:
    previous = record.get("metadata") or {}
    return {
        **record,
        "metadata": unit.metadata,
        "run_id": run_id,
        "metadata_refreshed": {
            "at": utc_now(),
            "previous_metadata_sha256": sha256_text(json.dumps(previous, sort_keys=True, default=str)),
            "reason": "text_unchanged_labels_changed",
            "read_generated_at": record.get("generated_at"),
        },
    }


def run_llm_full_read(
    *,
    workspace_root: Path,
    run_id: str,
    scope: str = "transcripts",
    limit: int | None = None,
    resume: bool = True,
    model: str = DEFAULT_LOCAL_MODEL,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    priority_topic_ids: Iterable[str] = (),
    priority_queries: Iterable[str] = (),
    carry_forward_metadata_only_changes: bool = True,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    topics = load_topic_families(workspace_root)
    topic_block = _topic_prompt_block(topics, workspace_root)

    started_at = utc_now()
    start_time = time.monotonic()
    unbind_substep_receipt(workspace_root, run_id, "llm-full-read")
    result = FullReadResult()
    root = workspace_root / "storage" / "extractions" / run_id / "llm-full-read"
    (root / "by-unit").mkdir(parents=True, exist_ok=True)
    transcript_coverage_rows: list[dict[str, Any]] = []
    units = list(
        iter_units_for_scope(
            workspace_root,
            scope=scope,
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            min_chars=min_chars,
            transcript_coverage=transcript_coverage_rows,
        )
    )
    source_coverage = transcript_source_coverage(transcript_coverage_rows, min_chars=min_chars)
    unread_source_ids = frozenset(
        str(row["source_artifact_id"])
        for row in transcript_coverage_rows
        if int(row.get("unit_count") or 0) == 0
    )
    priority_topic_id_list = [
        str(topic_id).strip() for topic_id in priority_topic_ids if str(topic_id).strip()
    ]
    priority_query_list = [
        str(query).strip() for query in priority_queries if str(query).strip()
    ]
    priority_ordering_enabled = bool(priority_topic_id_list or priority_query_list)
    if priority_ordering_enabled:
        units = _prioritize_units(
            units,
            workspace_root=workspace_root,
            priority_topic_ids=priority_topic_id_list,
            priority_queries=priority_query_list,
        )
    expected_counts = Counter(unit.input_kind for unit in units)
    total_expected_units = len(units)
    units_to_process = units if limit is None else units[: max(0, limit)]
    backend = select_local_llm_backend(model) if units_to_process else None

    legacy_prompt_unverified = 0
    for unit in units_to_process:
        if unit.input_kind == "transcript_segment":
            result.transcript_unit_count += 1
        elif unit.input_kind == "paper_abstract":
            result.paper_abstract_count += 1
        if len(unit.text) < min_chars:
            result.skipped_empty_count += 1
            continue
        out_path = _unit_output_path(workspace_root, run_id, unit.unit_id)
        if resume and out_path.exists():
            existing_record = _load_json(out_path)
            # A record from a failed attempt (llm_failed) or an unreadable/
            # corrupt file is not "done" -- resume must retry it rather than
            # silently keeping the failure forever. This is exactly what
            # happened live: 1,614 units failed with a transient backend
            # bug (unrelated to these specific units), got written as
            # skipped_reason=llm_failed records, and every subsequent
            # --resume run skipped them as "already processed" because only
            # file existence was checked.
            if (
                existing_record
                and existing_record.get("skipped_reason") != "llm_failed"
                and _record_matches_unit_input(existing_record, unit=unit, topic_block=topic_block)
            ):
                if existing_record.get("topic_prompt_sha256") is None:
                    legacy_prompt_unverified += 1
                result.skipped_existing_count += 1
                _accumulate_record_metrics(result, existing_record)
                continue
            if (
                carry_forward_metadata_only_changes
                and existing_record
                and existing_record.get("skipped_reason") != "llm_failed"
                and _record_reusable_with_new_metadata(existing_record, unit=unit, topic_block=topic_block)
            ):
                refreshed = _carry_forward_record(existing_record, unit=unit, run_id=run_id)
                write_json_atomic(out_path, refreshed)
                result.skipped_existing_count += 1
                result.metadata_refreshed_count += 1
                _accumulate_record_metrics(result, refreshed)
                continue
        try:
            if backend is None:
                raise RuntimeError("local LLM backend unavailable")
            record = _record_for_unit(
                workspace_root=workspace_root,
                run_id=run_id,
                unit=unit,
                topics=topics,
                topic_block=topic_block,
                backend=backend,
                model=model,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - receipt boundary
            record = {
                "schema_version": LLM_FULL_READ_SCHEMA_VERSION,
                "lane": "llm_full_read",
                "unit_id": unit.unit_id,
                "source_artifact_id": unit.source_artifact_id,
                "input_kind": unit.input_kind,
                "text_sha256": sha256_text(unit.text),
                "text_length": len(unit.text),
                "topic_ids": [],
                "important": False,
                "confidence": 0.0,
                "skipped_reason": "llm_failed",
                "error": str(exc)[:500],
                "metadata": unit.metadata,
                "run_id": run_id,
                "generated_at": utc_now(),
            }
        write_json_atomic(out_path, record)
        result.processed_count += 1
        _accumulate_record_metrics(result, record)

    pruned_unit_count = 0
    if scope == "all" and limit is None:
        pruned_unit_count = _prune_obsolete_unit_records(
            workspace_root, run_id, units, protected_source_ids=unread_source_ids
        )
    topic_counts = _aggregate_by_topic(workspace_root, run_id)
    result.elapsed_seconds = time.monotonic() - start_time
    completed_units = result.processed_count + result.skipped_existing_count
    completion_blockers: list[dict[str, Any]] = []
    if scope != "all":
        completion_blockers.append(
            {
                "code": "llm_full_read_scope_not_all",
                "scope": scope,
                "required_scope": "all",
            }
        )
    if limit is not None:
        completion_blockers.append(
            {
                "code": "llm_full_read_limit_set",
                "limit": limit,
                "expected_unit_count": total_expected_units,
            }
        )
    if completed_units < total_expected_units:
        completion_blockers.append(
            {
                "code": "llm_full_read_incomplete_units",
                "completed_unit_count": completed_units,
                "expected_unit_count": total_expected_units,
            }
        )
    if result.failed_count:
        completion_blockers.append(
            {
                "code": "llm_full_read_failed_units",
                "failed_count": result.failed_count,
            }
        )
    if scope in {"transcripts", "all"} and source_coverage["unread_input_count"]:
        # completed==expected cannot see this: an input that yields no unit
        # lowers both sides. CVPR's September full read reported complete while
        # reading none of the talk-annotated keynotes, tutorials, workshops, or
        # oral sessions.
        completion_blockers.append(
            {
                "code": "llm_full_read_transcript_source_unread",
                "unread_input_count": source_coverage["unread_input_count"],
                "unread_text_chars": source_coverage["unread_text_chars"],
                "sample_source_artifact_ids": [
                    row["source_artifact_id"] for row in source_coverage["unread_inputs"][:10]
                ],
            }
        )
    if scope in {"papers", "all"} and result.paper_abstract_count == 0:
        # The corpus has papers but no paper unit was produced: never report
        # the paper lane complete (CVPR's July full read did, reading 0 papers).
        corpus_papers = count_indexable_corpus_papers(workspace_root)
        if corpus_papers:
            completion_blockers.append(
                {
                    "code": "llm_full_read_paper_source_missing",
                    "corpus_paper_count": corpus_papers,
                    "paper_abstract_count": 0,
                }
            )
    completion_state = "complete" if not completion_blockers else "partial"
    status = "succeeded" if completion_state == "complete" else "partial"
    if result.failed_count:
        status = "blocked"
    receipt = {
        "schema_version": LLM_FULL_READ_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.llm_full_read_extraction",
        "status": status,
        "run_id": run_id,
        "workspace": str(workspace_root),
        "scope": scope,
        "limit": limit,
        "started_at": started_at,
        "ended_at": utc_now(),
        "model": model,
        "backend": os.environ.get("EXTRACTION_LOCAL_LLM_BACKEND") or "auto",
        "provider_calls_made": result.processed_count > 0,
        "live_provider_calls_made": False,
        "paid_provider_calls_made": False,
        "secrets_copied": False,
        "policy": {
            "transcripts_first": not priority_ordering_enabled,
            "papers_last": not priority_ordering_enabled,
            "paper_policy": "abstract_only",
            "transcript_segment_seconds": segment_seconds,
            "transcript_overlap_seconds": overlap_seconds,
            "not_important_segments_are_ignored": True,
            "priority_ordering_enabled": priority_ordering_enabled,
            "priority_topic_ids": priority_topic_id_list,
            "priority_queries": priority_query_list,
        },
        "checks": {
            "completion_state": completion_state,
            "completion_blockers": completion_blockers,
            "expected_unit_count": total_expected_units,
            "expected_input_kind_counts": dict(sorted(expected_counts.items())),
            "completed_unit_count": completed_units,
            "processed_count": result.processed_count,
            "skipped_existing_count": result.skipped_existing_count,
            "metadata_refreshed_count": result.metadata_refreshed_count,
            "skipped_empty_count": result.skipped_empty_count,
            "important_count": result.important_count,
            "not_important_count": result.not_important_count,
            "failed_count": result.failed_count,
            "transcript_unit_count": result.transcript_unit_count,
            "paper_abstract_count": result.paper_abstract_count,
            "by_topic_count": len(topic_counts),
            "transcript_source_coverage": source_coverage,
            "elapsed_seconds": result.elapsed_seconds,
            "units_per_minute": (
                result.processed_count / (result.elapsed_seconds / 60.0)
                if result.elapsed_seconds > 0
                else 0.0
            ),
            "chars_per_second": (
                result.input_chars / result.elapsed_seconds
                if result.elapsed_seconds > 0
                else 0.0
            ),
            "input_chars": result.input_chars,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
        },
        "artifacts": [
            {"kind": "llm-full-read-by-unit-dir", "path": _relpath(root / "by-unit", workspace_root)},
            {"kind": "llm-full-read-by-topic-dir", "path": _relpath(root / "by-topic", workspace_root)},
        ],
        "provider_receipt_paths": result.provider_receipt_paths,
    }
    receipt_path = (
        workspace_root
        / "storage"
        / "workflow-runs"
        / run_id
        / "14A-llm-full-read-extraction-receipt.json"
    )
    receipt["receipt_path"] = str(receipt_path)
    if isinstance(receipt.get("checks"), dict):
        receipt["checks"]["pruned_obsolete_unit_count"] = pruned_unit_count
        # Reused records written before topic_prompt_sha256 existed were
        # matched on text and metadata only; a taxonomy change since then
        # would not re-read them (re-run without --resume to force).
        receipt["checks"]["legacy_prompt_unverified_count"] = legacy_prompt_unverified
    stamp_substep_receipt(receipt, workspace=workspace_root, run_id=run_id, substep_id="llm-full-read")
    write_json_atomic(receipt_path, sanitize_receipt_for_write(receipt))
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local LLM full-read extraction.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--scope", choices=("transcripts", "papers", "all"), default="transcripts")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--segment-seconds", type=int, default=DEFAULT_SEGMENT_SECONDS)
    parser.add_argument("--overlap-seconds", type=int, default=DEFAULT_OVERLAP_SECONDS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument(
        "--priority-topic",
        action="append",
        default=[],
        help=(
            "Prioritize units whose metadata/text matches this topic's title, "
            "rationale, or starter queries. May be repeated. Processing remains all-scope."
        ),
    )
    parser.add_argument(
        "--priority-query",
        action="append",
        default=[],
        help=(
            "Additional case-insensitive phrase/keyword used only for processing order. "
            "May be repeated. Processing remains all-scope."
        ),
    )
    parser.add_argument(
        "--reread-label-changes",
        action="store_true",
        help=(
            "Re-read units whose text is unchanged but whose labels (title, authors, "
            "item kind, paths) changed. Default: carry the earlier reading forward "
            "with the new labels (recorded as metadata_refreshed)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_llm_full_read(
        workspace_root=args.workspace,
        run_id=args.run_id,
        scope=args.scope,
        limit=args.limit,
        resume=args.resume,
        model=args.model,
        segment_seconds=args.segment_seconds,
        overlap_seconds=args.overlap_seconds,
        min_chars=args.min_chars,
        max_output_tokens=args.max_output_tokens,
        priority_topic_ids=args.priority_topic,
        priority_queries=args.priority_query,
        carry_forward_metadata_only_changes=not args.reread_label_changes,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") == "succeeded" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
