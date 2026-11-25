"""Canonical source-aware chunking for conference vector stores.

This module promotes the NeurIPS 2025 enhanced-vector-store strategy into a
conference-general implementation: preserve source context, use timestamp/page
locators when available, and split long text into overlapping semantic windows
instead of embedding short evidence previews.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from conference_pipeline.indexing.token_policy import (
    DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_TOKENS,
    TokenBudgetPolicy,
    token_count,
)

TRANSCRIPT_CHUNK_SECONDS = 120.0


@dataclass(frozen=True)
class SourceChunk:
    """Text plus provenance for one vector-store chunk candidate."""

    text: str
    strategy: str
    chunk_index: int
    chunk_count: int
    char_offsets: dict[str, int]
    locator: dict[str, Any]


def _metadata_dict(record: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("source_context", "quality_metadata", "metadata"):
        value = record.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _first_string(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    metadata = _metadata_dict(record)
    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    for key in keys:
        for source in (record, metadata, evidence):
            value = source.get(key) if isinstance(source, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _token_count(text: str) -> int:
    return token_count(text)


# Scripts written without spaces (Japanese, Chinese) arrive as one enormous
# "word"; a unit larger than the chunk budget used to be truncated, silently
# dropping speech. Runs longer than _MAX_UNIT_CHARS split into fixed pieces so
# every character lands in some chunk.
_MAX_UNIT_CHARS = 256
_UNIT_PIECE_CHARS = 64


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\S+", text):
        word = match.group(0)
        if len(word) <= _MAX_UNIT_CHARS:
            spans.append((word, match.start(), match.end()))
            continue
        for offset in range(0, len(word), _UNIT_PIECE_CHARS):
            piece = word[offset : offset + _UNIT_PIECE_CHARS]
            start = match.start() + offset
            spans.append((piece, start, start + len(piece)))
    return spans


def _span_text(spans: list[tuple[str, int, int]], start: int, end: int) -> str:
    return " ".join(word for word, _start, _end in spans[start:end]).strip()


def _max_end_within_token_budget(
    spans: list[tuple[str, int, int]],
    *,
    start: int,
    budget_tokens: int,
) -> int:
    if start >= len(spans):
        return start
    low = start + 1
    high = len(spans)
    best = start
    while low <= high:
        mid = (low + high) // 2
        candidate = _span_text(spans, start, mid)
        if _token_count(candidate) <= budget_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best if best > start else start + 1


def _overlap_start_for_budget(
    spans: list[tuple[str, int, int]],
    *,
    chunk_start: int,
    chunk_end: int,
    overlap_tokens: int,
) -> int:
    if overlap_tokens <= 0:
        return chunk_end
    overlap_start = chunk_end
    while overlap_start > chunk_start:
        candidate = _span_text(spans, overlap_start - 1, chunk_end)
        if _token_count(candidate) > overlap_tokens:
            break
        overlap_start -= 1
    return max(chunk_start + 1, overlap_start)


def _split_text_spans(
    text: str,
    *,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[tuple[int, int, int, int, str]]:
    spans = _word_spans(text)
    out: list[tuple[int, int, int, int, str]] = []
    start = 0
    while start < len(spans):
        end = _max_end_within_token_budget(
            spans,
            start=start,
            budget_tokens=max(1, chunk_tokens),
        )
        end = min(len(spans), max(start + 1, end))
        chunk_text = _span_text(spans, start, end)
        if not chunk_text:
            start = end
            continue
        char_start = spans[start][1]
        char_end = spans[end - 1][2]
        chunk_text = text[char_start:char_end].strip()
        out.append((char_start, char_end, start, end, chunk_text))
        if end >= len(spans):
            break
        start = _overlap_start_for_budget(
            spans,
            chunk_start=start,
            chunk_end=end,
            overlap_tokens=overlap_tokens,
        )
    return out


def _truncate_text_to_token_budget(text: str, budget_tokens: int) -> str:
    text = _clean_text(text)
    if budget_tokens <= 0 or not text:
        return ""
    if _token_count(text) <= budget_tokens:
        return text
    spans = _word_spans(text)
    if spans:
        end = _max_end_within_token_budget(
            spans,
            start=0,
            budget_tokens=budget_tokens,
        )
        candidate = _span_text(spans, 0, min(end, len(spans)))
        if candidate and _token_count(candidate) <= budget_tokens:
            return candidate
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip()
        if _token_count(candidate) <= budget_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _reindex_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    total = len(chunks)
    return [
        SourceChunk(
            text=chunk.text,
            strategy=chunk.strategy,
            chunk_index=index,
            chunk_count=total,
            char_offsets=chunk.char_offsets,
            locator=chunk.locator,
        )
        for index, chunk in enumerate(chunks)
    ]


def _chunk_by_words(
    text: str,
    *,
    strategy: str,
    locator_base: dict[str, Any] | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    max_embedding_tokens: int = DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
) -> list[SourceChunk]:
    text = _clean_text(text)
    if not text:
        return []
    policy = TokenBudgetPolicy(
        chunk_tokens=chunk_tokens,
        chunk_overlap_tokens=overlap_tokens,
        hard_embedding_tokens=max_embedding_tokens,
    )
    split_budget = min(policy.chunk_tokens, policy.hard_embedding_tokens)
    spans = _split_text_spans(
        text,
        chunk_tokens=split_budget,
        overlap_tokens=policy.chunk_overlap_tokens,
    )
    chunks = [
        SourceChunk(
            text=chunk_text,
            strategy=strategy,
            chunk_index=index,
            chunk_count=len(spans),
            char_offsets={"start": char_start, "end": char_end},
            locator={
                **(locator_base or {}),
                "kind": strategy,
                "word_start": word_start,
                "word_end": word_end,
            },
        )
        for index, (char_start, char_end, word_start, word_end, chunk_text) in enumerate(spans)
    ]
    return _reindex_chunks(chunks)


def paper_metadata_text(record: dict[str, Any]) -> str:
    """Build the canonical paper-metadata text chunk."""
    metadata = _metadata_dict(record)
    title = _first_string(record, ("paper_title", "title", "source_title"))
    abstract = _first_string(record, ("abstract", "paper_abstract", "summary"))
    venue = _first_string(record, ("venue", "primary_area"))
    primary_area = _first_string(record, ("primary_area",))
    authors = metadata.get("authors")
    keywords = metadata.get("keywords")
    author_line = ", ".join(str(item) for item in authors if str(item).strip()) if isinstance(authors, list) else ""
    keyword_line = ", ".join(str(item) for item in keywords if str(item).strip()) if isinstance(keywords, list) else ""
    if not any((title, abstract, author_line, keyword_line)):
        return ""
    lines = [
        f"Paper Title: {title}",
        f"Venue: {venue}" if venue else "",
        f"Primary Area: {primary_area}" if primary_area else "",
        f"Authors: {author_line}" if author_line else "",
        f"Keywords: {keyword_line}" if keyword_line else "",
        "",
        "Abstract:",
        abstract,
    ]
    return _clean_text("\n".join(line for line in lines if line is not None))


PREFIX_MAX_AUTHORS = 10
PREFIX_MAX_KEYWORDS = 8


def _name_list(value: Any, limit: int) -> str:
    if isinstance(value, str):
        items = [part.strip() for part in re.split(r";|,", value) if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        return ""
    shown = ", ".join(items[:limit])
    return f"{shown}, et al." if len(items) > limit else shown


def paper_text_prefix(record: dict[str, Any]) -> str:
    """Header lines every passage of a paper carries into its embedding and BM25 text.

    ``Paper Title`` (and, when known, ``Authors`` -- first ten -- ``Keywords``,
    the workshop it was presented at, and ``Material`` for a paper's slides or
    poster file) so retrieval can attribute and match a passage by the paper's
    own words, not only its body text.
    """

    title = _first_string(record, ("paper_title", "title", "source_title"))
    section = _first_string(record, ("section_title", "section", "heading"))
    subsection = _first_string(record, ("subsection_title", "subsection", "subheading"))
    lines = []
    if title:
        lines.append(f"Paper Title: {title}")
    authors = _name_list(record.get("authors") or _metadata_dict(record).get("authors"), PREFIX_MAX_AUTHORS)
    if authors:
        lines.append(f"Authors: {authors}")
    keywords = _name_list(record.get("keywords") or _metadata_dict(record).get("keywords"), PREFIX_MAX_KEYWORDS)
    if keywords:
        lines.append(f"Keywords: {keywords}")
    workshop = _first_string(record, ("workshop_title",))
    if workshop:
        lines.append(f"Workshop: {workshop}")
    venue = _first_string(record, ("venue",))
    if venue and _first_string(record, ("venue_track",)):
        # Only a named track (e.g. Findings) is stamped; the main track stays unmarked.
        lines.append(f"Venue: {venue}")
    material = _first_string(record, ("material_type",))
    if material and material != "paper":
        lines.append(f"Material: {material}")
    if section:
        lines.append(f"Section: {section}")
    if subsection:
        lines.append(f"Subsection: {subsection}")
    return "\n".join(lines)


TRANSCRIPT_CONTEXT_FIELDS = (
    "workshop_id",
    "workshop_title",
    "session_title",
    "source_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
)


def _segment_context(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: segment[key]
        for key in TRANSCRIPT_CONTEXT_FIELDS
        if segment.get(key) not in (None, "")
    }


def _unique_context_values(
    records: list[dict[str, Any]],
    key: str,
    fallback: dict[str, Any] | None = None,
) -> list[str]:
    values: list[str] = []
    for record in [*records, fallback or {}]:
        value = record.get(key) if isinstance(record, dict) else None
        if value in (None, ""):
            continue
        if isinstance(value, list):
            candidates = [str(item).strip() for item in value if str(item).strip()]
        else:
            candidates = [str(value).strip()]
        for candidate in candidates:
            if candidate and candidate not in values:
                values.append(candidate)
    return values


def _context_line(label: str, values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return f"{label}: {values[0]}"
    return f"{label}s: {'; '.join(values)}"


def transcript_context_prefix(
    *,
    title: str,
    timestamp_start: float | None = None,
    timestamp_end: float | None = None,
    context_records: list[dict[str, Any]] | None = None,
    fallback_context: dict[str, Any] | None = None,
) -> str:
    records = context_records or []
    fallback = fallback_context or {}
    presentation = title or (fallback.get("source_title") if isinstance(fallback, dict) else "")
    lines = [
        _context_line("Workshop", _unique_context_values(records, "workshop_title", fallback)),
        _context_line("Session", _unique_context_values(records, "session_title", fallback)),
        _context_line("Talk", _unique_context_values(records, "talk_title", fallback)),
        _context_line("Speaker", _unique_context_values(records, "talk_speaker", fallback)),
    ]
    if presentation:
        lines.append(f"Presentation: {presentation}")
    if timestamp_start is not None and timestamp_end is not None:
        lines.append(f"Timestamp: {timestamp_start:.2f}-{timestamp_end:.2f}s")
    return _clean_text("\n".join(line for line in lines if line))


def transcript_context_locator(
    context_records: list[dict[str, Any]],
    fallback_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    locator: dict[str, Any] = {}
    fallback = fallback_context or {}
    for key in ("workshop_id", "workshop_title"):
        values = _unique_context_values(context_records, key, fallback)
        if values:
            locator[key] = values[0]
    for source_key, plural_key, singular_key in (
        ("talk_id", "talk_ids", "talk_id"),
        ("talk_title", "talk_titles", "talk_title"),
        ("talk_speaker", "talk_speakers", "talk_speaker"),
    ):
        values = _unique_context_values(context_records, source_key, fallback)
        if not values:
            continue
        locator[plural_key] = values
        if len(values) == 1:
            locator[singular_key] = values[0]
    return locator


def _chunks_with_prefix(
    body: str,
    *,
    prefix: str,
    strategy: str,
    locator_base: dict[str, Any] | None = None,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    max_embedding_tokens: int = DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
) -> list[SourceChunk]:
    body = _clean_text(body)
    prefix = _clean_text(prefix)
    if not prefix:
        return _chunk_by_words(
            body,
            strategy=strategy,
            locator_base=locator_base,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            max_embedding_tokens=max_embedding_tokens,
        )

    out: list[SourceChunk] = []
    prefix_block = f"{prefix}\n\n"
    prefix_tokens = _token_count(prefix_block)
    if prefix_tokens > max_embedding_tokens:
        truncated_prefix = _truncate_text_to_token_budget(prefix, max_embedding_tokens)
        out.append(
            SourceChunk(
                text=truncated_prefix,
                strategy=strategy,
                chunk_index=0,
                chunk_count=1,
                char_offsets={"start": 0, "end": len(truncated_prefix)},
                locator={
                    **(locator_base or {}),
                    "kind": strategy,
                    "budget_warning": "prefix_exceeds_embedding_budget",
                },
            )
        )
        body_chunks = _chunk_by_words(
            body,
            strategy=strategy,
            locator_base=locator_base,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            max_embedding_tokens=max_embedding_tokens,
        )
        return _reindex_chunks(out + body_chunks)

    body_budget = max(1, chunk_tokens - prefix_tokens)
    body_chunks = _chunk_by_words(
        body,
        strategy=strategy,
        locator_base=locator_base,
        chunk_tokens=body_budget,
        overlap_tokens=overlap_tokens,
        max_embedding_tokens=max(1, max_embedding_tokens - prefix_tokens),
    )
    for chunk in body_chunks:
        body_text = chunk.text
        final_text = _clean_text(f"{prefix_block}{body_text}")
        if _token_count(final_text) > max_embedding_tokens:
            remaining_budget = max(0, max_embedding_tokens - prefix_tokens)
            body_text = _truncate_text_to_token_budget(body_text, remaining_budget)
            final_text = _clean_text(f"{prefix_block}{body_text}")
        out.append(
            SourceChunk(
                text=final_text,
                strategy=chunk.strategy,
                chunk_index=chunk.chunk_index,
                chunk_count=chunk.chunk_count,
                char_offsets=chunk.char_offsets,
                locator=chunk.locator,
            )
        )
    return _reindex_chunks(out)


def _segment_word_times(
    segments: list[dict[str, Any]],
) -> list[tuple[str, float, float]]:
    timed_words: list[tuple[str, float, float]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        words = [word for word, _start, _end in _word_spans(text)]
        if not words:
            continue
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end") or start)
        duration = max(0.0, end - start)
        for index, word in enumerate(words):
            word_start = start + duration * index / len(words)
            word_end = start + duration * (index + 1) / len(words)
            timed_words.append((word, word_start, word_end))
    return timed_words


def transcript_chunks_from_segments(
    segments: list[dict[str, Any]],
    *,
    title: str,
    source_context: dict[str, Any] | None = None,
    chunk_seconds: float = TRANSCRIPT_CHUNK_SECONDS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP,
    max_embedding_tokens: int = DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS,
) -> list[SourceChunk]:
    """Group timestamped transcript segments, then split dense windows by token budget."""
    windows: list[tuple[float, float, list[dict[str, Any]]]] = []
    chunk_start: float | None = None
    chunk_end = 0.0
    window_segments: list[dict[str, Any]] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        try:
            start = float(segment.get("start") or 0.0)
            end = float(segment.get("end") or start)
        except (TypeError, ValueError):
            start = chunk_end
            end = chunk_end
        normalized = {"start": start, "end": end, "text": text, **_segment_context(segment)}
        if segment.get("synthetic_timestamp"):
            normalized["_synthetic_timestamp"] = True
        if chunk_start is None:
            chunk_start = start
        if window_segments and (end - chunk_start) > chunk_seconds:
            windows.append((chunk_start, chunk_end, window_segments))
            chunk_start = start
            window_segments = [normalized]
        else:
            window_segments.append(normalized)
        chunk_end = end
    if window_segments and chunk_start is not None:
        windows.append((chunk_start, chunk_end, window_segments))

    out: list[SourceChunk] = []
    for start, end, window in windows:
        text = _clean_text(" ".join(str(segment["text"]) for segment in window))
        prefix = transcript_context_prefix(
            title=title,
            timestamp_start=start,
            timestamp_end=end,
            context_records=window,
            fallback_context=source_context,
        )
        prefix_block = f"{prefix}\n\n"
        chunk_text = _clean_text(f"{prefix_block}{text}")
        context_locator = transcript_context_locator(window, source_context)
        if any(segment.get("_synthetic_timestamp") for segment in window):
            # Times interpolated inside a coarse model window, not real cues.
            context_locator["timestamp_synthetic"] = True
        if _token_count(chunk_text) <= chunk_tokens and _token_count(chunk_text) <= max_embedding_tokens:
            out.append(
                SourceChunk(
                    text=chunk_text,
                    strategy="transcript_120s_segments",
                    chunk_index=len(out),
                    chunk_count=0,
                    char_offsets={"start": 0, "end": len(chunk_text)},
                    locator={
                        "kind": "transcript_time_range",
                        "timestamp_start": start,
                        "timestamp_end": end,
                        **context_locator,
                    },
                )
            )
            continue

        prefix_tokens = _token_count(prefix_block)
        body_budget = max(1, chunk_tokens - prefix_tokens)
        spans = _split_text_spans(
            text,
            chunk_tokens=body_budget,
            overlap_tokens=overlap_tokens,
        )
        timed_words = _segment_word_times(window)
        subchunk_count = len(spans)
        for subchunk_index, (_char_start, _char_end, word_start, word_end, body_text) in enumerate(spans):
            final_text = _clean_text(f"{prefix_block}{body_text}")
            if _token_count(final_text) > max_embedding_tokens:
                remaining_budget = max(0, max_embedding_tokens - prefix_tokens)
                body_text = _truncate_text_to_token_budget(body_text, remaining_budget)
                final_text = _clean_text(f"{prefix_block}{body_text}")
            timestamp_start = (
                timed_words[word_start][1]
                if word_start < len(timed_words)
                else start
            )
            timestamp_end = (
                timed_words[word_end - 1][2]
                if 0 <= word_end - 1 < len(timed_words)
                else end
            )
            out.append(
                SourceChunk(
                    text=final_text,
                    strategy="transcript_120s_token_window",
                    chunk_index=len(out),
                    chunk_count=0,
                    char_offsets={"start": 0, "end": len(final_text)},
                    locator={
                        "kind": "transcript_time_range",
                        "timestamp_start": timestamp_start,
                        "timestamp_end": timestamp_end,
                        "window_timestamp_start": start,
                        "window_timestamp_end": end,
                        "subchunk_index": subchunk_index,
                        "subchunk_count": subchunk_count,
                        **context_locator,
                    },
                )
            )
    return _reindex_chunks(out)


def _chunks_from_json_text(raw_text: str, record: dict[str, Any]) -> list[SourceChunk]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    title = _first_string(record, ("source_title", "title", "paper_title"))
    source_context = _metadata_dict(record)
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return transcript_chunks_from_segments(payload, title=title, source_context=source_context)
    if isinstance(payload, dict):
        for key in ("segments", "transcript_segments"):
            segments = payload.get(key)
            if isinstance(segments, list) and all(isinstance(item, dict) for item in segments):
                return transcript_chunks_from_segments(
                    segments,
                    title=title,
                    source_context=source_context,
                )
        for key in ("text", "ocr_text", "transcript"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _chunk_by_words(
                    value,
                    strategy="json_text_token_window",
                    locator_base={"json_key": key},
                )
        slide_text = payload.get("slide_text")
        if isinstance(slide_text, dict):
            parts: list[str] = []
            for key in ("title", "equations", "bullet_points", "other_text"):
                value = slide_text.get(key)
                if isinstance(value, list):
                    parts.extend(str(item) for item in value if str(item).strip())
                elif isinstance(value, str) and value.strip():
                    parts.append(value)
            text = _clean_text("\n".join(parts))
            if text:
                slide_index = payload.get("slide_index")
                chunk_text = f"Presentation: {title}\nSlide {slide_index}\n\n{text}"
                return [
                    SourceChunk(
                        text=chunk_text,
                        strategy="slide_ocr_page",
                        chunk_index=0,
                        chunk_count=1,
                        char_offsets={"start": 0, "end": len(chunk_text)},
                        locator={"kind": "slide", "slide_index": slide_index},
                    )
                ]
    return []


def chunks_for_record(
    record: dict[str, Any],
    *,
    source_text: str | None,
    source_family: str,
) -> list[SourceChunk]:
    """Return canonical vector-store chunks for one extraction record."""
    source_text = _clean_text(source_text or "")
    input_kind = _first_string(record, ("input_kind", "source_family", "source_type"))
    source_id = str(record.get("source_artifact_id") or record.get("artifact_id") or "")
    locator_base = {"source_family": source_family, "source_artifact_id": source_id}

    if source_family == "paper_metadata_fallback":
        text = paper_metadata_text(record)
        return _chunk_by_words(
            text,
            strategy="paper_metadata_title_abstract",
            locator_base=locator_base,
        )

    if source_text:
        json_chunks = _chunks_from_json_text(source_text, record)
        if json_chunks:
            return json_chunks
        if source_family in {"paper_pdf", "paper_text"} or input_kind in {"paper_pdf", "paper_text", "pdf_text"}:
            return _chunks_with_prefix(
                source_text,
                prefix=paper_text_prefix(record),
                strategy="paper_text_token_window",
                locator_base=locator_base,
            )
        if input_kind == "transcript":
            title = _first_string(record, ("source_title", "title", "paper_title"))
            return _chunks_with_prefix(
                source_text,
                prefix=transcript_context_prefix(title=title, fallback_context=_metadata_dict(record)),
                strategy="transcript_token_window",
                locator_base=locator_base,
            )
        if input_kind in {"ocr_json", "page_image"}:
            return _chunk_by_words(
                source_text,
                strategy="slide_or_image_text",
                locator_base=locator_base,
            )
        return _chunk_by_words(
            source_text,
            strategy="generic_text_token_window",
            locator_base=locator_base,
        )

    evidence = record.get("evidence") if isinstance(record.get("evidence"), dict) else {}
    snippet = str(evidence.get("snippet") or "").strip()
    if not snippet:
        return []
    if source_family in {"paper_pdf", "paper_text"}:
        prefix = paper_text_prefix(record)
        if prefix:
            snippet = f"{prefix}\n\n{snippet}"
    return _chunk_by_words(
        snippet,
        strategy="snippet_fallback_blocked_if_long",
        locator_base={**locator_base, "snippet_fallback": True},
    )


__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_MAX_EMBEDDING_TOKENS",
    "DEFAULT_CHUNK_TOKENS",
    "SourceChunk",
    "TRANSCRIPT_CHUNK_SECONDS",
    "chunks_for_record",
    "paper_metadata_text",
    "transcript_chunks_from_segments",
]
