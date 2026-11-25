"""Traceability addendum generation with injected vector/research boundaries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.analysis.steps.gpt_researcher_hybrid import Researcher


class VectorStore(Protocol):
    def query(self, text: str, *, top_k: int) -> list[dict[str, Any]]: ...


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


@dataclass(frozen=True)
class PresentationIndex:
    exact: dict[str, str]
    normalized: dict[str, str]
    titles: list[tuple[str, str]]
    id_to_title: dict[str, str]


@dataclass(frozen=True)
class TraceabilityInputs:
    analysis_dir: Path
    presentations_root: Path
    output_dir: Path
    vector_store_dir: Path
    use_gpt_researcher: bool = True
    vector_top_k: int = 5
    overwrite_existing: bool = True


@dataclass(frozen=True)
class TraceabilityOutputs:
    addenda_written: tuple[Path, ...]
    digest: str


def normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def build_presentation_index(presentations_root: Path) -> PresentationIndex:
    exact: dict[str, str] = {}
    normalized: dict[str, str] = {}
    titles: list[tuple[str, str]] = []
    id_to_title: dict[str, str] = {}
    for meta_path in sorted(presentations_root.glob("*/meta.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        title = str(data.get("title") or "").strip()
        if not title:
            continue
        exact[title.lower()] = meta_path.parent.name
        normalized[normalize_text(title)] = meta_path.parent.name
        titles.append((title.lower(), meta_path.parent.name))
        id_to_title[meta_path.parent.name] = title
    return PresentationIndex(exact=exact, normalized=normalized, titles=titles, id_to_title=id_to_title)


def _parse_source_field(line: str) -> tuple[str | None, str | None, int | None, str | None]:
    cleaned = line.strip().lstrip("-* ").replace("**", "")
    if ":" not in cleaned:
        return None, None, None, None
    label, value = cleaned.split(":", 1)
    label_norm = re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()
    value = value.strip().strip("`").strip("\"").strip("'")
    if label_norm in {"source", "sources", "presentation"}:
        return value, None, None, None
    if label_norm == "presentation id":
        return None, value, None, None
    if label_norm.startswith("slide"):
        match = re.search(r"\d+", value)
        return None, None, int(match.group(0)) if match else None, None
    if label_norm.startswith("timestamp"):
        return None, None, None, value
    return None, None, None, None


def parse_doc_quotes(doc_path: Path) -> list[QuoteEntry]:
    entries: list[QuoteEntry] = []
    source_title: str | None = None
    presentation_id: str | None = None
    slide_number: int | None = None
    timestamp: str | None = None

    for raw_line in doc_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed_source, parsed_id, parsed_slide, parsed_timestamp = _parse_source_field(line)
        source_title = parsed_source or source_title
        presentation_id = parsed_id or presentation_id
        slide_number = parsed_slide or slide_number
        timestamp = parsed_timestamp or timestamp
        if line.startswith(">"):
            quote = line.lstrip(">").strip().strip("\"")
            if quote:
                entries.append(
                    QuoteEntry(
                        quote=quote,
                        source_title=source_title,
                        presentation_id=presentation_id,
                        slide_number=slide_number,
                        timestamp=timestamp,
                    )
                )
    return entries


def _apply_vector_match(entry: QuoteEntry, vector_store: VectorStore, top_k: int) -> None:
    matches = vector_store.query(entry.quote, top_k=top_k)
    if not matches:
        return
    best = matches[0]
    metadata = best.get("metadata") if isinstance(best, dict) else None
    metadata = metadata if isinstance(metadata, dict) else best
    if not entry.presentation_id and metadata.get("presentation_id"):
        entry.presentation_id = str(metadata["presentation_id"])
    if not entry.source_title and metadata.get("title"):
        entry.source_title = str(metadata["title"])
    if not entry.timestamp:
        for key in ("timestamp_range", "timestamp_formatted", "timestamp"):
            if metadata.get(key):
                entry.timestamp = str(metadata[key])
                break
    if not entry.slide_number and metadata.get("slide_number"):
        try:
            entry.slide_number = int(metadata["slide_number"])
        except (TypeError, ValueError):
            pass
    entry.transcript_match = True


def write_addendum(
    doc_path: Path,
    entries: list[QuoteEntry],
    output_dir: Path,
    *,
    overwrite_existing: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_path.stem}_traceability_addendum.md"
    if output_path.exists() and not overwrite_existing:
        return output_path

    lines = [
        f"# Traceability Addendum: {doc_path.name}",
        "",
        f"Source document: `{doc_path.name}`",
        "",
    ]
    unresolved = 0
    for index, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"## Entry {index}",
                f"Quote: \"{entry.quote}\"",
                f"Source Title: {entry.source_title or 'UNKNOWN'}",
                f"Presentation ID: {entry.presentation_id or 'UNKNOWN'}",
                f"Slide: {entry.slide_number or 'UNKNOWN'}",
                f"Timestamp: {entry.timestamp or 'UNKNOWN'}",
                f"Paper ID: {entry.paper_id or 'UNKNOWN'}",
                f"Paper URL: {entry.paper_url or 'UNKNOWN'}",
                f"Slide Match: {'Yes' if entry.slide_match else 'No'}",
                f"Transcript Match: {'Yes' if entry.transcript_match else 'No'}",
                f"Paper Match: {'Yes' if entry.paper_match else 'No'}",
                "",
            ]
        )
        if not (entry.presentation_id or entry.paper_id):
            unresolved += 1
    lines.extend(["## Summary", f"- Total entries: {len(entries)}", f"- Unresolved entries: {unresolved}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _combined_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for item in sorted(sha256_file(path) for path in paths):
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


def step_traceability(
    inputs: TraceabilityInputs,
    *,
    researcher: Researcher | None = None,
    vector_store: VectorStore,
) -> TraceabilityOutputs:
    _ = researcher
    _ = build_presentation_index(inputs.presentations_root)
    docs = [
        path
        for path in sorted(inputs.analysis_dir.glob("*.md"))
        if not path.name.startswith(("NeurIPS_Analysis_Plan", "coverage_report", "qa_report", "pilot_"))
    ]
    written: list[Path] = []
    for doc_path in docs:
        entries = parse_doc_quotes(doc_path)
        if not entries:
            continue
        for entry in entries:
            _apply_vector_match(entry, vector_store, inputs.vector_top_k)
        written.append(
            write_addendum(
                doc_path,
                entries,
                inputs.output_dir,
                overwrite_existing=inputs.overwrite_existing,
            )
        )
    paths = tuple(written)
    return TraceabilityOutputs(addenda_written=paths, digest=_combined_digest(paths))
