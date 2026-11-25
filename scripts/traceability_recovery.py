#!/usr/bin/env python3
"""Generate traceability addenda for analysis documents."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class QuoteEntry:
    quote: str
    source_title: str | None = None
    presentation_id: str | None = None
    slide_number: int | None = None
    timestamp: str | None = None
    transcript_match: bool = False
    slide_match: bool = False
    paper_id: str | None = None
    paper_url: str | None = None
    paper_match: bool = False


MIN_SLIDE_SCORE = 0.62
MIN_TRANSCRIPT_SCORE = 0.62
MIN_PAPER_SCORE = 0.62
MIN_TITLE_SIMILARITY = 0.65
MIN_PRESENTATION_SCORE = 0.66


@dataclass(frozen=True)
class PresentationIndex:
    exact: dict[str, str]
    normalized: dict[str, str]
    titles: list[tuple[str, str]]
    id_to_title: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate traceability addenda for analysis docs.")
    parser.add_argument(
        "input",
        help="Path to a markdown file or a directory containing analysis docs.",
    )
    parser.add_argument(
        "--presentations-root",
        default="conferences/neurips2025/decks",
        help="Root directory for presentations data.",
    )
    parser.add_argument(
        "--output-dir",
        default="conferences/neurips2025/analysis/traceability_addenda",
        help="Directory for addendum files.",
    )
    parser.add_argument(
        "--extensions",
        default=".md",
        help="Comma-separated file extensions when input is a directory.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="conferences/neurips2025/cache/vector-store/chroma_db",
        help="Path to the ChromaDB directory for vector search.",
    )
    parser.add_argument(
        "--no-chromadb",
        action="store_true",
        help="Disable ChromaDB vector lookup.",
    )
    parser.add_argument(
        "--vector-top-k",
        type=int,
        default=8,
        help="Top-K results to consider from ChromaDB (default: 8).",
    )
    parser.add_argument(
        "--use-gpt-researcher",
        action="store_true",
        help="Use GPT Researcher as a fallback for unresolved entries.",
    )
    parser.add_argument(
        "--gpt-researcher-limit",
        type=int,
        default=0,
        help="Max unresolved entries per doc to send to GPT Researcher (default: 0).",
    )
    parser.add_argument(
        "--gpt-researcher-web",
        action="store_true",
        help="Allow GPT Researcher to use web search (default: local only).",
    )
    return parser.parse_args()


def collect_docs(path: Path, extensions: str) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {ext.strip().lower() for ext in extensions.split(",") if ext.strip()}
    exts = {ext if ext.startswith(".") else f".{ext}" for ext in exts}
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def title_similarity(source_title: str | None, candidate_title: str | None) -> float:
    if not source_title or not candidate_title:
        return 0.0
    source_norm = normalize_text(source_title)
    candidate_norm = normalize_text(candidate_title)
    if not source_norm or not candidate_norm:
        return 0.0
    return difflib.SequenceMatcher(None, source_norm, candidate_norm).ratio()


def is_generic_source_title(source_title: str | None) -> bool:
    if not source_title:
        return True
    normalized = normalize_text(source_title)
    if normalized in {"openreview", "openreviewnet", "arxiv", "openreviewnetforum"}:
        return True
    return len(normalized) < 8


def build_presentation_index(presentations_root: Path) -> PresentationIndex:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    titles: list[tuple[str, str]] = []
    id_to_title: dict[str, str] = {}
    for meta_path in presentations_root.glob("*/meta.json"):
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        title = str(data.get("title", "")).strip()
        if title:
            title_key = title.lower()
            exact[title_key] = meta_path.parent.name
            normalized[normalize_text(title)] = meta_path.parent.name
            titles.append((title_key, meta_path.parent.name))
            id_to_title[meta_path.parent.name] = title
    return PresentationIndex(exact=exact, normalized=normalized, titles=titles, id_to_title=id_to_title)


def resolve_presentation_id(source_title: str | None, index: PresentationIndex) -> str | None:
    if not source_title:
        return None
    cleaned = source_title.strip().strip("\"").strip("'")
    key = cleaned.lower()
    if key in index.exact:
        return index.exact[key]
    normalized = normalize_text(cleaned)
    if normalized in index.normalized:
        return index.normalized[normalized]
    for title, pid in index.titles:
        if key in title or title in key:
            return pid
    if normalized:
        matches = difflib.get_close_matches(normalized, index.normalized.keys(), n=1, cutoff=0.82)
        if matches:
            return index.normalized[matches[0]]
    return None


def strip_markdown(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def normalize_timestamp_value(value: str) -> str | None:
    times = re.findall(r"\d{1,2}:\d{2}(?::\d{2})?", value)
    if not times:
        return None

    def to_hms(ts: str) -> str:
        parts = [int(p) for p in ts.split(":")]
        if len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = parts[0], parts[1], parts[2]
        return f"{h:02d}:{m:02d}:{s:02d}"

    if len(times) >= 2:
        return f"{to_hms(times[0])} - {to_hms(times[1])}"
    return to_hms(times[0])


def parse_slide_value(value: str) -> int | None:
    match = re.search(r"(?:slide[_\s]*)(\d+)", value, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d{1,3})\b", value)
    if match:
        return int(match.group(1))
    return None


def parse_timestamp_to_seconds(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    times = re.findall(r"\d{1,2}:\d{2}(?::\d{2})?", timestamp)
    if not times:
        return None
    ts = times[0]
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        h, m, s = parts[0], parts[1], parts[2]
    return h * 3600 + m * 60 + s


def parse_source_fields(line: str) -> tuple[str | None, str | None, int | None, str | None]:
    cleaned = strip_markdown(line)
    if cleaned.startswith(("-", "*")):
        cleaned = cleaned.lstrip("-* ").strip()
    if cleaned.startswith("|") and cleaned.endswith("|"):
        parts = [p.strip() for p in cleaned.strip("|").split("|")]
        if len(parts) >= 2:
            return parse_source_fields(f"{parts[0]}: {parts[1]}")
    if ":" not in cleaned:
        return None, None, None, None
    label, raw_value = cleaned.split(":", 1)
    value = raw_value.strip().strip("`").strip().strip("\"").strip("'")
    label_norm = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()

    source_title = None
    presentation_id = None
    slide_number = None
    timestamp = None

    if label_norm in {"source", "sources", "presentation"}:
        source_title = value
    elif label_norm == "presentation id":
        presentation_id = value
    elif label_norm.startswith("slide"):
        slide_number = parse_slide_value(value)
    elif label_norm.startswith("timestamp"):
        timestamp = normalize_timestamp_value(value) or value

    return source_title, presentation_id, slide_number, timestamp


def score_ok(item: dict[str, Any] | None, threshold: float) -> bool:
    if not item:
        return False
    return float(item.get("score", 0.0)) >= threshold


def parse_citation_line(raw_line: str) -> tuple[str | None, int | None, str | None]:
    stripped = raw_line.strip()
    match = re.match(r"^>\s*[\u2014-]\s*\[(.+?)\]\s*$", stripped)
    if match:
        content = match.group(1).strip()
    else:
        match = re.match(r"^>\s*[\u2014-]\s*(.+?)\s*$", stripped)
        if not match:
            return None, None, None
        content = match.group(1).strip()
    parts = [p.strip() for p in content.split(",") if p.strip()]
    if not parts:
        return None, None, None
    title = parts[0]
    slide_number = None
    timestamp = None
    for part in parts[1:]:
        if slide_number is None and re.search(r"slide", part, re.IGNORECASE):
            slide_number = parse_slide_value(part)
        if timestamp is None:
            timestamp = normalize_timestamp_value(part) or timestamp
    return title, slide_number, timestamp


def split_field_segments(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|") and stripped.endswith("|"):
        return [line]
    if "|" in line:
        return [segment.strip() for segment in line.split("|") if segment.strip()]
    return [line]


def parse_doc_quotes(doc_path: Path) -> list[QuoteEntry]:
    lines = doc_path.read_text(encoding="utf-8").splitlines()
    entries: list[QuoteEntry] = []

    current_source: str | None = None
    current_slide: int | None = None
    current_timestamp: str | None = None
    current_presentation_id: str | None = None

    quote_buffer: list[str] = []
    last_context_line: int | None = None

    def flush_quote(line_index: int) -> None:
        nonlocal quote_buffer, current_source, current_slide, current_timestamp, current_presentation_id
        if not quote_buffer:
            return
        use_context = (
            last_context_line is not None
            and (line_index - last_context_line) <= 18
            and any(
                [
                    current_source,
                    current_presentation_id,
                    current_slide,
                    current_timestamp,
                ]
            )
        )
        quote = " ".join([q.strip() for q in quote_buffer]).strip()
        entries.append(
            QuoteEntry(
                quote=quote.strip("\""),
                source_title=current_source if use_context else None,
                presentation_id=current_presentation_id if use_context else None,
                slide_number=current_slide if use_context else None,
                timestamp=current_timestamp if use_context else None,
            )
        )
        quote_buffer = []

    for line_idx, raw_line in enumerate(lines):
        line = raw_line.strip()

        if line.startswith(">"):
            citation_title, citation_slide, citation_timestamp = parse_citation_line(raw_line)
            if citation_title:
                current_source = citation_title
                current_presentation_id = None
                current_slide = citation_slide or current_slide
                current_timestamp = citation_timestamp or current_timestamp
                last_context_line = line_idx
                continue
            quote_buffer.append(line.lstrip(">").strip())
            continue

        flush_quote(line_idx)

        if re.match(r"^#{2,6}\s", line):
            current_source = None
            current_presentation_id = None
            current_slide = None
            current_timestamp = None
            last_context_line = None
            continue

        for segment in split_field_segments(line):
            source_title, presentation_id, slide_number, timestamp = parse_source_fields(segment)
            if source_title:
                current_source = source_title
                current_presentation_id = None
                current_slide = None
                current_timestamp = None
                last_context_line = line_idx
                continue
            if presentation_id:
                current_presentation_id = presentation_id
                last_context_line = line_idx
            if slide_number:
                current_slide = slide_number
                last_context_line = line_idx
            if timestamp:
                current_timestamp = timestamp
                last_context_line = line_idx

    flush_quote(len(lines))
    return entries


def load_ocr_text(ocr_path: Path) -> str:
    data = json.loads(ocr_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    slide_text = data.get("slide_text", {})
    parts.append(str(slide_text.get("title", "")))
    parts.extend(slide_text.get("bullet_points", []) or [])
    other_text = slide_text.get("other_text")
    if other_text:
        parts.append(str(other_text))
    for item in data.get("visual_elements", []) or []:
        parts.append(str(item.get("description", "")))
        parts.append(str(item.get("data", "")))
        parts.extend(item.get("labels", []) or [])
    return " ".join([p for p in parts if p]).lower()


def find_slide_by_quote(presentations_root: Path, presentation_id: str, quote: str) -> int | None:
    ocr_dir = presentations_root / presentation_id / "ocr"
    if not ocr_dir.exists():
        return None
    snippet = quote.lower().strip()
    if len(snippet) > 120:
        snippet = snippet[:120]
    for ocr_path in sorted(ocr_dir.glob("slide_*.json")):
        content = load_ocr_text(ocr_path)
        if snippet and snippet in content:
            data = json.loads(ocr_path.read_text(encoding="utf-8"))
            return int(data.get("slide_index"))
    return None


def find_timestamp_from_slide(presentations_root: Path, presentation_id: str, slide_number: int) -> str | None:
    ocr_path = presentations_root / presentation_id / "ocr" / f"slide_{slide_number:03d}.json"
    if not ocr_path.exists():
        ocr_path = presentations_root / presentation_id / "ocr" / f"slide_{slide_number:04d}.json"
    if not ocr_path.exists():
        return None
    data = json.loads(ocr_path.read_text(encoding="utf-8"))
    ts = data.get("timestamp_sec")
    if ts is None:
        return None
    return format_timestamp(ts)


def find_timestamp_from_transcript(presentations_root: Path, presentation_id: str, quote: str) -> str | None:
    transcript_path = presentations_root / presentation_id / "transcripts" / "segments.json"
    if not transcript_path.exists():
        return None
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    snippet = quote.lower().strip()
    if len(snippet) > 120:
        snippet = snippet[:120]
    for seg in segments:
        text = str(seg.get("text", "")).lower()
        if snippet and snippet in text:
            start = seg.get("start", 0)
            end = seg.get("end", start)
            return f"{format_timestamp(start)} - {format_timestamp(end)}"
    return None


def format_timestamp(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_slide_by_timestamp(presentations_root: Path, presentation_id: str, timestamp: str) -> int | None:
    target = parse_timestamp_to_seconds(timestamp)
    if target is None:
        return None
    ocr_dir = presentations_root / presentation_id / "ocr"
    if not ocr_dir.exists():
        return None
    best_slide = None
    best_delta = None
    for ocr_path in sorted(ocr_dir.glob("slide_*.json")):
        data = json.loads(ocr_path.read_text(encoding="utf-8"))
        ts = data.get("timestamp_sec")
        if ts is None:
            continue
        delta = abs(ts - target)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_slide = data.get("slide_index")
    return int(best_slide) if best_slide is not None else None


def load_chromadb_collection(chroma_dir: Path) -> Any | None:
    try:
        import chromadb
    except Exception:
        return None
    if not chroma_dir.exists():
        return None
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        return client.get_collection("neurips_presentations")
    except Exception:
        return None


def load_gpt_researcher_module() -> Any | None:
    module_path = Path(__file__).parent / "query_gpt_researcher_hybrid.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("query_gpt_researcher_hybrid", module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpt_researcher_sources(
    query: str,
    max_results: int,
    use_web: bool,
) -> list[dict[str, Any]]:
    module = load_gpt_researcher_module()
    if module is None:
        return []
    try:
        module.setup_environment()
    except Exception:
        pass
    try:
        report, _context, sources = asyncio.run(
            module.research_hybrid(
                query,
                use_web=use_web,
                use_local=True,
                max_results=max_results,
            )
        )
        _ = report
    except Exception:
        return []
    return sources or []


def chromadb_query(collection: Any, query: str, top_k: int, where: dict | None) -> list[dict[str, Any]]:
    if not query:
        return []
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )
    output = []
    if results.get("ids") and results["ids"][0]:
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if not meta:
                continue
            output.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "score": 1 - distance if distance is not None else 0.0,
                }
            )
    output.sort(key=lambda item: item["score"], reverse=True)
    return output


def filter_match_types(results: Iterable[dict[str, Any]], types: set[str]) -> list[dict[str, Any]]:
    return [item for item in results if item.get("metadata", {}).get("type") in types]


def best_match(results: list[dict[str, Any]], source_title: str | None) -> dict[str, Any] | None:
    if not results:
        return None
    if not source_title:
        return results[0]
    scored = []
    for item in results:
        meta_title = str(item.get("metadata", {}).get("title", ""))
        title_score = title_similarity(source_title, meta_title)
        combined = item.get("score", 0.0) + (0.2 * title_score)
        scored.append((combined, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def extract_timestamp_from_meta(meta: dict[str, Any]) -> str | None:
    if meta.get("timestamp_range"):
        normalized = normalize_timestamp_value(str(meta["timestamp_range"]))
        return normalized or str(meta["timestamp_range"])
    if meta.get("timestamp_formatted"):
        normalized = normalize_timestamp_value(str(meta["timestamp_formatted"]))
        return normalized or str(meta["timestamp_formatted"])
    if meta.get("timestamp") is not None:
        return format_timestamp(meta["timestamp"])
    if meta.get("timestamp_start") is not None:
        start = float(meta["timestamp_start"])
        end = float(meta.get("timestamp_end", start))
        return f"{format_timestamp(start)} - {format_timestamp(end)}"
    return None


def extract_timestamp_from_source(source: dict[str, Any]) -> str | None:
    if not source:
        return None
    for key in ("timestamp_range", "timestamp_formatted", "timestamp"):
        if source.get(key):
            value = source[key]
            if isinstance(value, (int, float)):
                return format_timestamp(float(value))
            normalized = normalize_timestamp_value(str(value))
            return normalized or str(value)
    if source.get("timestamp_start") not in (None, "", "N/A"):
        try:
            start = float(source.get("timestamp_start", 0))
        except (TypeError, ValueError):
            return None
        end_value = source.get("timestamp_end", start)
        try:
            end = float(end_value)
        except (TypeError, ValueError):
            end = start
        return f"{format_timestamp(start)} - {format_timestamp(end)}"
    return None


def pick_gpt_researcher_match(
    sources: list[dict[str, Any]],
    source_title: str | None,
) -> dict[str, Any] | None:
    if not sources:
        return None
    local_sources = [s for s in sources if s.get("source") == "local_chromadb"]
    if not local_sources:
        return None
    scored = []
    for src in local_sources:
        score = 0.0
        if "similarity_score" in src:
            try:
                score = float(src["similarity_score"])
            except Exception:
                score = 0.0
        elif "distance" in src:
            try:
                score = 1 - float(src["distance"])
            except Exception:
                score = 0.0
        title_score = title_similarity(source_title, src.get("title"))
        combined = score + (0.2 * title_score)
        scored.append((combined, src))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def write_addendum(doc_path: Path, entries: list[QuoteEntry], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_path.stem}_traceability_addendum.md"
    lines: list[str] = [
        f"# Traceability Addendum: {doc_path.name}",
        "",
        f"Source document: `{doc_path.as_posix()}`",
        "",
    ]

    unresolved = 0
    for idx, entry in enumerate(entries, start=1):
        lines.append(f"## Entry {idx}")
        lines.append(f"Quote: \"{entry.quote}\"")
        lines.append(f"Source Title: {entry.source_title or 'UNKNOWN'}")
        lines.append(f"Presentation ID: {entry.presentation_id or 'UNKNOWN'}")
        lines.append(f"Slide: {entry.slide_number or 'UNKNOWN'}")
        lines.append(f"Timestamp: {entry.timestamp or 'UNKNOWN'}")
        lines.append(f"Paper ID: {entry.paper_id or 'UNKNOWN'}")
        lines.append(f"Paper URL: {entry.paper_url or 'UNKNOWN'}")
        lines.append(f"Slide Match: {'Yes' if entry.slide_match else 'No'}")
        lines.append(f"Transcript Match: {'Yes' if entry.transcript_match else 'No'}")
        lines.append(f"Paper Match: {'Yes' if entry.paper_match else 'No'}")
        lines.append("")
        if entry.presentation_id:
            if not (entry.slide_number and entry.timestamp):
                unresolved += 1
        elif not entry.paper_id:
            unresolved += 1

    lines.append("## Summary")
    lines.append(f"- Total entries: {len(entries)}")
    lines.append(f"- Unresolved entries: {unresolved}")
    lines.append("")

    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    presentations_root = Path(args.presentations_root)
    chroma_dir = Path(args.chroma_dir)

    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")
    if not presentations_root.exists():
        raise SystemExit(f"Presentations root not found: {presentations_root}")

    docs = collect_docs(input_path, args.extensions)
    if not docs:
        raise SystemExit(f"No documents found in: {input_path}")

    index = build_presentation_index(presentations_root)
    collection = None
    if not args.no_chromadb:
        collection = load_chromadb_collection(chroma_dir)
    presentation_types = {"slide", "transcript"}
    paper_types = {"paper"}

    for doc_path in docs:
        if doc_path.name == "NeurIPS_Analysis_Plan.md":
            continue
        if "pilot" in doc_path.name.lower():
            continue

        entries = parse_doc_quotes(doc_path)
        gpt_remaining = args.gpt_researcher_limit if args.use_gpt_researcher else 0
        for entry in entries:
            if not entry.presentation_id:
                entry.presentation_id = resolve_presentation_id(entry.source_title, index)
            if entry.presentation_id and (
                not entry.source_title
                or normalize_text(entry.source_title) == normalize_text(entry.presentation_id)
            ):
                entry.source_title = index.id_to_title.get(entry.presentation_id, entry.source_title)

            if collection and (entry.quote or entry.source_title):
                where_filter = {"type": {"$in": list(presentation_types)}}
                if entry.presentation_id:
                    where_filter["presentation_id"] = entry.presentation_id
                try:
                    results = chromadb_query(
                        collection,
                        entry.quote or entry.source_title or "",
                        args.vector_top_k,
                        where_filter,
                    )
                    results = filter_match_types(results, presentation_types)
                    if results and entry.source_title and not entry.presentation_id:
                        if not is_generic_source_title(entry.source_title):
                            filtered = [
                                item
                                for item in results
                                if title_similarity(
                                    entry.source_title, item.get("metadata", {}).get("title")
                                )
                                >= MIN_TITLE_SIMILARITY
                            ]
                            results = filtered
                except Exception:
                    results = []

                if results:
                    best_any = best_match(results, entry.source_title)
                    if best_any and score_ok(best_any, MIN_PRESENTATION_SCORE):
                        entry.presentation_id = (
                            entry.presentation_id
                            or best_any["metadata"].get("presentation_id")
                        )
                        if not entry.source_title or is_generic_source_title(entry.source_title):
                            entry.source_title = best_any["metadata"].get("title")

                if not entry.slide_number:
                    slide_results = filter_match_types(results, {"slide"})
                    slide_match = best_match(slide_results, entry.source_title)
                    if slide_match and score_ok(slide_match, MIN_SLIDE_SCORE):
                        entry.presentation_id = (
                            entry.presentation_id
                            or slide_match["metadata"].get("presentation_id")
                        )
                        entry.slide_number = slide_match["metadata"].get("slide_index")
                        entry.timestamp = entry.timestamp or extract_timestamp_from_meta(
                            slide_match["metadata"]
                        )
                        entry.slide_match = entry.slide_number is not None
                        if not entry.source_title or is_generic_source_title(entry.source_title):
                            entry.source_title = slide_match["metadata"].get("title")

                transcript_results = filter_match_types(results, {"transcript"})
                transcript_match = best_match(transcript_results, entry.source_title)
                if transcript_match and score_ok(transcript_match, MIN_TRANSCRIPT_SCORE):
                    entry.presentation_id = (
                        entry.presentation_id
                        or transcript_match["metadata"].get("presentation_id")
                    )
                    if not entry.timestamp:
                        entry.timestamp = extract_timestamp_from_meta(
                            transcript_match["metadata"]
                        )
                    entry.transcript_match = True
                    if not entry.source_title or is_generic_source_title(entry.source_title):
                        entry.source_title = transcript_match["metadata"].get("title")

                if collection and not entry.presentation_id:
                    try:
                        paper_results = chromadb_query(
                            collection,
                            entry.quote or entry.source_title or "",
                            args.vector_top_k,
                            {"type": {"$in": list(paper_types)}},
                        )
                        paper_results = filter_match_types(paper_results, paper_types)
                        if entry.source_title and not is_generic_source_title(entry.source_title):
                            paper_results = [
                                item
                                for item in paper_results
                                if title_similarity(
                                    entry.source_title, item.get("metadata", {}).get("title")
                                )
                                >= MIN_TITLE_SIMILARITY
                            ]
                    except Exception:
                        paper_results = []

                    best_paper = best_match(paper_results, entry.source_title)
                    if best_paper and score_ok(best_paper, MIN_PAPER_SCORE):
                        meta = best_paper["metadata"]
                        entry.paper_id = meta.get("paper_id")
                        entry.paper_url = meta.get("url")
                        entry.paper_match = True
                        if not entry.source_title or is_generic_source_title(entry.source_title):
                            entry.source_title = meta.get("title")

            if (
                args.use_gpt_researcher
                and gpt_remaining > 0
                and not entry.presentation_id
                and not entry.paper_id
                and (entry.quote or entry.source_title)
            ):
                sources = gpt_researcher_sources(
                    entry.quote or entry.source_title or "",
                    max_results=args.vector_top_k,
                    use_web=args.gpt_researcher_web,
                )
                match = pick_gpt_researcher_match(sources, entry.source_title)
                if match:
                    meta = match.get("metadata", {}) if isinstance(match.get("metadata"), dict) else {}
                    entry.presentation_id = match.get("presentation_id") or meta.get("presentation_id")
                    entry.paper_id = entry.paper_id or match.get("paper_id") or meta.get("paper_id")
                    if entry.paper_id:
                        entry.paper_url = entry.paper_url or match.get("url") or meta.get("url")
                    if not entry.source_title or is_generic_source_title(entry.source_title):
                        entry.source_title = match.get("title") or meta.get("title") or entry.source_title
                    slide_value = (
                        match.get("slide_number")
                        or match.get("slide_index")
                        or meta.get("slide_index")
                        or meta.get("slide_number")
                    )
                    if slide_value and not entry.slide_number:
                        entry.slide_number = parse_slide_value(str(slide_value))
                    if not entry.timestamp:
                        entry.timestamp = extract_timestamp_from_source(match) or extract_timestamp_from_meta(
                            meta
                        )
                    source_type = match.get("type") or meta.get("type")
                    if source_type == "slide":
                        entry.slide_match = entry.slide_number is not None or entry.slide_match
                    elif source_type == "transcript":
                        entry.transcript_match = True
                    elif source_type == "paper" and entry.paper_id:
                        entry.paper_match = True
                gpt_remaining -= 1

            if entry.presentation_id:
                if not entry.slide_number and entry.quote:
                    entry.slide_number = find_slide_by_quote(
                        presentations_root, entry.presentation_id, entry.quote
                    )
                    entry.slide_match = entry.slide_number is not None or entry.slide_match
                if entry.slide_number and not entry.timestamp:
                    entry.timestamp = find_timestamp_from_slide(
                        presentations_root, entry.presentation_id, entry.slide_number
                    )
                    entry.slide_match = entry.timestamp is not None or entry.slide_match
                if entry.timestamp and not entry.slide_number:
                    entry.slide_number = find_slide_by_timestamp(
                        presentations_root, entry.presentation_id, entry.timestamp
                    )
                    entry.slide_match = entry.slide_number is not None or entry.slide_match
                if entry.quote and not entry.timestamp:
                    ts = find_timestamp_from_transcript(
                        presentations_root, entry.presentation_id, entry.quote
                    )
                    if ts:
                        entry.timestamp = ts
                        entry.transcript_match = True

            if entry.presentation_id and (
                not entry.source_title
                or normalize_text(entry.source_title) == normalize_text(entry.presentation_id)
            ):
                entry.source_title = index.id_to_title.get(entry.presentation_id, entry.source_title)

        write_addendum(doc_path, entries, output_dir)


if __name__ == "__main__":
    main()
