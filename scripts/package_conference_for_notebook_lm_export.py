#!/usr/bin/env python3
"""Package a conference workspace for NotebookLM source imports.

The exporter is intentionally filesystem-only. It reads already-produced
conference artifacts and writes chunked Markdown files plus a manifest:

- speech transcripts from main conference and workshop media, formatted as
  title/author plus transcript text only
- accepted paper abstracts with useful metadata
- workshop paper records, enriched with abstracts when local metadata can join
- analysis Markdown documents, copied without hash dereferencing
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WORKSPACE = Path("conferences/neurips2025")
DEFAULT_MAX_BYTES = 4_000_000
DEFAULT_MAX_WORDS = 450_000
DEFAULT_ANALYSIS_EXTENSIONS = ".md"
DEFAULT_PAPER_MODE = "abstract"
PAPER_MODES = ("abstract", "full", "both")
UNCATEGORIZED_TOPIC = "00_uncategorized"

WORD_RE = re.compile(r"\S+")
TALK_ID_RE = re.compile(r"(talk_media_\d+_[A-Za-z0-9]+)")
POSTER_ID_RE = re.compile(r"/(poster-[^/]+)/")

# Full-text paper cleaning. The extracted paper text is raw OCR/text-extraction
# output, so it carries running page headers, hyphenated line breaks, and large
# reference lists that add little value for NotebookLM grounding while consuming
# most of the per-source word budget. We strip references (but keep appendices),
# drop the recurring conference header, de-hyphenate, and collapse whitespace.
ICLR_BOILERPLATE_RE = re.compile(
    r"^\s*published as a (?:conference|workshop) paper at iclr\b.*$", re.IGNORECASE
)
PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,4}\s*$")
REFERENCES_HEADING_RE = re.compile(r"^\s*(?:references|bibliography)\s*$", re.IGNORECASE)
APPENDIX_HEADING_RE = re.compile(
    r"^\s*(?:appendix|appendices|supplementary materials?)\b", re.IGNORECASE
)
HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])-\n\s*([a-z])")
PDF_HASH_RE = re.compile(r"/([0-9a-f]{40})(?:[-.])", re.IGNORECASE)

# Populated per-run from the workspace's analysis_manifest.json (see
# populate_topic_display_names). The literal defaults below are a fallback for
# workspaces predating that manifest and are not conference-specific in intent,
# only in fact — do not rely on them for a specific conference.
TOPIC_DISPLAY_NAMES: dict[str, str] = {
    "01_reasoning_post_training": "Reasoning & Post-training",
    "02_agentic_systems": "Agentic Systems",
    "03_evaluation_benchmarks_data": "Evaluation, Benchmarks & Data",
    "04_safety_alignment_privacy": "Safety, Alignment & Privacy",
    "05_generative_models": "Generative Models",
    "06_multimodal_spatial_vlm": "Multimodal & Spatial VLMs",
    "07_ai_for_science": "AI for Science",
    "08_theory_optimization_structure": "Theory, Optimization & Structure",
    "09_efficiency_systems_architectures": "Efficiency, Systems & Architectures",
    "10_interpretability_representation": "Interpretability & Representation",
    "11_rl_robotics_world_models": "RL, Robotics & World Models",
    "12_open_science_society_post_agi": "Open Science, Society & Post-AGI",
    UNCATEGORIZED_TOPIC: "Uncategorized",
}


def populate_topic_display_names(workspace: Path) -> None:
    """Replace TOPIC_DISPLAY_NAMES in place with this workspace's actual topic
    ids/titles from analysis_manifest.json, ordered by topic number. Mutating
    in place (not reassigning) means every existing reference to the module
    dict — sort-key lookups, membership tests, iteration — picks up the
    current conference's topics without threading a parameter through every
    call site. Falls back to the literal defaults above only when the
    manifest is missing or unreadable."""
    manifest_path = workspace / "analysis/analysis_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        payload = load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return
    topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(topics, list) or not topics:
        return
    ordered = sorted(
        (t for t in topics if isinstance(t, dict) and t.get("id")),
        key=lambda t: t.get("number", 999),
    )
    fresh = {str(t["id"]): str(t.get("title") or t["id"]) for t in ordered}
    if not fresh:
        return
    fresh[UNCATEGORIZED_TOPIC] = "Uncategorized"
    TOPIC_DISPLAY_NAMES.clear()
    TOPIC_DISPLAY_NAMES.update(fresh)


@dataclass
class ExportItem:
    kind: str
    title: str
    source_id: str
    source_path: str
    text: str
    topic: str = ""
    topic_source: str = ""


@dataclass
class TalkInfo:
    title: str
    author: str = ""
    workshop_title: str = ""


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_json(path: Path) -> Any:
    return json.loads(safe_read_text(path))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_workspace_path(workspace: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return workspace / path


def normalize_title(title: str) -> str:
    title = clean_text(title).lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


def stringify_list(values: Any) -> str:
    if not values:
        return ""
    if isinstance(values, list):
        return ", ".join(clean_text(value) for value in values if clean_text(value))
    return clean_text(values)


def first_present(values: Iterable[Any]) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def append_metadata_line(lines: list[str], label: str, value: Any) -> None:
    if isinstance(value, list):
        value = stringify_list(value)
    text = clean_text(value)
    if text:
        lines.append(f"{label}: {text}")


class ChunkWriter:
    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_words: int = DEFAULT_MAX_WORDS,
        extension: str = ".md",
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.max_words = max_words
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.part_index = 0
        self.current_path: Path | None = None
        self.current_lines: list[str] = []
        self.current_bytes = 0
        self.current_words = 0
        self.current_items = 0
        self.parts: list[dict[str, Any]] = []

    def _open_new(self) -> None:
        self.part_index += 1
        self.current_path = (
            self.output_dir / f"{self.prefix}_part_{self.part_index:03d}{self.extension}"
        )
        self.current_lines = []
        self.current_bytes = 0
        self.current_words = 0
        self.current_items = 0

    def _flush(self) -> None:
        if self.current_path is None or not self.current_lines:
            return
        content = "\n".join(self.current_lines).strip() + "\n"
        self.current_path.write_text(content, encoding="utf-8")
        self.parts.append(
            {
                "path": str(self.current_path),
                "byte_count": len(content.encode("utf-8")),
                "word_count": count_words(content),
                "item_count": self.current_items,
            }
        )
        self.current_lines = []

    def add(self, title: str, text: str) -> Path:
        block = text.strip()
        if not block.startswith("#"):
            block = f"# {title}\n\n{block}"
        block += "\n"
        block_bytes = len(block.encode("utf-8"))
        block_words = count_words(block)

        if self.current_path is None:
            self._open_new()
        if self.current_lines and (
            self.current_bytes + block_bytes > self.max_bytes
            or self.current_words + block_words > self.max_words
        ):
            self._flush()
            self._open_new()

        assert self.current_path is not None
        self.current_lines.append(block)
        self.current_bytes += block_bytes
        self.current_words += block_words
        self.current_items += 1
        return self.current_path

    def finalize(self) -> list[dict[str, Any]]:
        self._flush()
        return self.parts


def find_first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_main_papers(workspace: Path, papers_file: Path | None) -> list[dict[str, Any]]:
    candidate = papers_file or find_first_existing(
        [
            workspace / "storage/download-corpus/openreview_accepted_papers.json",
            workspace / "data/papers.json",
            workspace / "papers.json",
        ]
    )
    if candidate is None or not candidate.exists():
        return []

    payload = load_json(candidate)
    raw_papers = payload.get("papers", payload) if isinstance(payload, dict) else payload
    papers: list[dict[str, Any]] = []
    if not isinstance(raw_papers, list):
        return papers

    for raw in raw_papers:
        if not isinstance(raw, dict):
            continue
        paper = dict(raw)
        paper.setdefault("source_kind", "main_conference")
        paper.setdefault("metadata_file", str(candidate))
        if paper.get("forum_url") and not paper.get("openreview_url"):
            paper["openreview_url"] = paper["forum_url"]
        papers.append(paper)
    return papers


def load_workshop_titles(workspace: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    inventory_path = workspace / "storage/workshops/inventory.json"
    if inventory_path.exists():
        payload = load_json(inventory_path)
        for workshop in payload.get("workshops", []) if isinstance(payload, dict) else []:
            if not isinstance(workshop, dict):
                continue
            venueid = clean_text(workshop.get("venueid"))
            title = first_present(
                [
                    workshop.get("title"),
                    workshop.get("name"),
                    workshop.get("workshop_title"),
                    venueid,
                ]
            )
            if venueid and title:
                titles[venueid] = title

    talk_inventory = workspace / "storage/workshops/talk_media_inventory.json"
    if talk_inventory.exists():
        payload = load_json(talk_inventory)
        for record in payload.get("records", []) if isinstance(payload, dict) else []:
            if not isinstance(record, dict):
                continue
            venueid = clean_text(record.get("workshop_id"))
            title = clean_text(record.get("workshop_title"))
            if venueid and title:
                titles[venueid] = title
    return titles


def build_talk_metadata(workspace: Path) -> dict[str, TalkInfo]:
    talk_inventory = workspace / "storage/workshops/talk_media_inventory.json"
    if not talk_inventory.exists():
        return {}
    payload = load_json(talk_inventory)
    metadata: dict[str, TalkInfo] = {}
    for record in payload.get("records", []) if isinstance(payload, dict) else []:
        if not isinstance(record, dict):
            continue
        talk_id = clean_text(record.get("talk_id"))
        if not talk_id:
            continue
        talk_title = clean_text(record.get("talk_title"))
        workshop_title = clean_text(record.get("workshop_title"))
        speaker = clean_text(record.get("talk_speaker"))
        if talk_title and talk_title.lower() != "slideslive embedded media":
            title = talk_title
        elif workshop_title:
            title = f"{workshop_title} embedded media"
        else:
            title = talk_id
        current = metadata.get(talk_id)
        if current is None or current.title == talk_id or "embedded media" in current.title.lower():
            metadata[talk_id] = TalkInfo(
                title=title,
                author=speaker,
                workshop_title=workshop_title,
            )
    return metadata


def build_main_presentation_metadata(workspace: Path) -> dict[str, TalkInfo]:
    probe_root = workspace / "cache/acquisition-downloads/item-page-probes"
    if not probe_root.exists():
        return {}

    metadata: dict[str, TalkInfo] = {}
    for html_path in sorted(probe_root.glob("virtual-*-poster-*.html")):
        poster_match = re.search(r"poster-(\d+)", html_path.name)
        if not poster_match:
            continue
        poster_key = f"poster-{poster_match.group(1)}"
        payload = extract_json_ld(safe_read_text(html_path))
        if not payload:
            continue
        title = clean_text(payload.get("name"))
        authors = payload.get("author") if isinstance(payload.get("author"), list) else []
        author_names = [
            clean_text(author.get("name"))
            for author in authors
            if isinstance(author, dict) and clean_text(author.get("name"))
        ]
        if title:
            metadata[poster_key] = TalkInfo(title=title, author=", ".join(author_names))
    return metadata


def extract_json_ld(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>\s*(\{.*?\})\s*</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def paper_lookup_maps(papers: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_title: dict[str, dict[str, Any]] = {}
    for paper in papers:
        paper_id = clean_text(paper.get("id") or paper.get("forum") or paper.get("note_id"))
        title = clean_text(paper.get("title"))
        if paper_id:
            by_id[paper_id] = paper
        if title:
            by_title[normalize_title(title)] = paper
    return by_id, by_title


def load_workshop_papers(
    workspace: Path,
    main_by_id: dict[str, dict[str, Any]],
    main_by_title: dict[str, dict[str, Any]],
    workshop_titles: dict[str, str],
    workshop_papers_dir: Path | None = None,
) -> list[dict[str, Any]]:
    root = workshop_papers_dir or workspace / "storage/workshops/papers"
    if not root.exists():
        return []

    papers: list[dict[str, Any]] = []
    for receipt_path in sorted(root.glob("**/receipts/*.json")):
        receipt = load_json(receipt_path)
        if not isinstance(receipt, dict):
            continue
        note_id = clean_text(receipt.get("note_id") or receipt.get("id"))
        title = clean_text(receipt.get("title"))
        match = main_by_id.get(note_id) if note_id else None
        if match is None and title:
            match = main_by_title.get(normalize_title(title))

        paper = dict(match or {})
        paper.update(
            {
                "source_kind": "workshop",
                "note_id": note_id,
                "id": note_id or paper.get("id"),
                "title": title or paper.get("title"),
                "authors": receipt.get("authors") or paper.get("authors"),
                "venueid": receipt.get("venueid") or paper.get("venueid"),
                "workshop_title": workshop_titles.get(clean_text(receipt.get("venueid")), ""),
                "pdf_path": receipt.get("pdf_path") or paper.get("pdf_path"),
                "status": receipt.get("status"),
                "receipt_path": str(receipt_path),
            }
        )
        if note_id and not paper.get("openreview_url"):
            paper["openreview_url"] = f"https://openreview.net/forum?id={note_id}"
        papers.append(paper)
    return papers


def format_paper(paper: dict[str, Any]) -> ExportItem:
    title = clean_text(paper.get("title") or paper.get("id") or paper.get("note_id") or "Untitled paper")
    paper_id = clean_text(paper.get("id") or paper.get("note_id") or paper.get("forum"))
    kind = clean_text(paper.get("source_kind") or "paper")
    lines = [f"# Paper: {title}", ""]
    append_metadata_line(lines, "Source kind", kind)
    append_metadata_line(lines, "Paper ID", paper_id)
    append_metadata_line(lines, "OpenReview", paper.get("openreview_url") or paper.get("forum_url"))
    append_metadata_line(lines, "Forum", paper.get("forum"))
    append_metadata_line(lines, "Venue", paper.get("venue"))
    append_metadata_line(lines, "Venue ID", paper.get("venueid"))
    append_metadata_line(lines, "Workshop", paper.get("workshop_title"))
    append_metadata_line(lines, "Tier", paper.get("tier"))
    append_metadata_line(lines, "Status", paper.get("status"))
    append_metadata_line(lines, "Authors", paper.get("authors"))
    append_metadata_line(lines, "Primary area", paper.get("primary_area"))
    append_metadata_line(lines, "Keywords", paper.get("keywords"))
    append_metadata_line(lines, "PDF", paper.get("pdf_url") or paper.get("pdf_path"))
    append_metadata_line(lines, "Source URL", paper.get("source_url"))
    lines.append("")
    lines.append("## Abstract")
    abstract = clean_text(paper.get("abstract"))
    lines.append(abstract or "[Abstract not available in local metadata.]")
    return ExportItem(
        kind=f"{kind}_paper",
        title=title,
        source_id=paper_id,
        source_path=clean_text(paper.get("metadata_file") or paper.get("receipt_path") or ""),
        text="\n".join(lines).strip() + "\n",
    )


def strip_references_block(text: str) -> str:
    """Remove the references/bibliography list, preserving any appendix that
    follows it. References are matched from the first standalone References or
    Bibliography heading up to the next Appendix heading (or end of text)."""
    lines = text.split("\n")
    ref_idx = next(
        (i for i, line in enumerate(lines) if REFERENCES_HEADING_RE.match(line)),
        None,
    )
    if ref_idx is None:
        return text
    app_idx = next(
        (
            j
            for j in range(ref_idx + 1, len(lines))
            if APPENDIX_HEADING_RE.match(lines[j])
        ),
        None,
    )
    if app_idx is None:
        kept = lines[:ref_idx]
    else:
        kept = lines[:ref_idx] + lines[app_idx:]
    return "\n".join(kept)


def clean_paper_full_text(raw: str) -> str:
    """Reduce raw extracted paper text to lower-noise Markdown body text."""
    # Join words split across a line break by a trailing hyphen.
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", raw)
    kept_lines: list[str] = []
    for line in text.split("\n"):
        if ICLR_BOILERPLATE_RE.match(line):
            continue
        if PAGE_NUMBER_LINE_RE.match(line):
            continue
        kept_lines.append(line.rstrip())
    text = "\n".join(kept_lines)
    text = strip_references_block(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title_from_text(raw: str, max_words: int = 30) -> str:
    """Best-effort title recovery from a paper's text head, for records that do
    not join to local metadata. Reads the leading non-empty lines until an
    author/affiliation/email line appears."""
    collected: list[str] = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped:
            if collected:
                break
            continue
        if ICLR_BOILERPLATE_RE.match(stripped):
            continue
        if "@" in stripped:
            break
        if collected and re.search(r"\d", stripped):
            break
        collected.append(stripped)
        if sum(len(part.split()) for part in collected) >= max_words:
            break
        if len(collected) >= 3:
            break
    return clean_text(" ".join(collected))[:300]


def build_pdf_hash_map(papers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map OpenReview PDF hash -> paper metadata for joining extracted text to
    titles. The PDF hash also prefixes the extraction artifact's source path."""
    by_hash: dict[str, dict[str, Any]] = {}
    for paper in papers:
        pdf_url = clean_text(paper.get("pdf_url") or paper.get("pdf_path"))
        match = re.search(r"/([0-9a-f]{40})\.pdf", pdf_url, re.IGNORECASE)
        if match:
            by_hash[match.group(1).lower()] = paper
    return by_hash


def heuristic_primary_topic(topic_ids: Any, quality_metadata: dict[str, Any]) -> str:
    """Pick a paper's primary topic from the keyword-heuristic extraction record.

    The record's ``topic_ids`` is already ordered by the classifier's confidence
    (``score / keyword_count``), so ``topic_ids[0]`` is the confidence-ranked
    primary. This normalizes away topics with large/generic keyword lists, unlike
    a raw ``topic_scores`` argmax. Raw scores are only a last-resort tiebreak.

    Classifier metadata can predate a later topic-taxonomy reorganization (a
    workshop-paper classification pass that was never rerun, for example), so
    any candidate is checked against the current TOPIC_DISPLAY_NAMES before
    being trusted — a stale/foreign topic id must not silently create its own
    output bucket alongside the conference's actual topics."""
    if isinstance(topic_ids, list) and topic_ids:
        first = clean_text(topic_ids[0])
        if first and first in TOPIC_DISPLAY_NAMES:
            return first
    scores = quality_metadata.get("topic_scores") if quality_metadata else None
    if isinstance(scores, dict) and scores:
        ranked = sorted(
            (kv for kv in scores.items() if clean_text(kv[0]) in TOPIC_DISPLAY_NAMES),
            key=lambda kv: (-float(kv[1] or 0.0), kv[0]),
        )
        if ranked:
            return clean_text(ranked[0][0])
    return UNCATEGORIZED_TOPIC


def build_llm_topic_map(extractions_dir: Path) -> dict[str, str]:
    """Map OpenReview forum id -> LLM full-read primary topic.

    The exhaustive ``llm_full_read`` lane classifies each paper semantically and
    is far less skewed than the keyword heuristic, but only covers main-conference
    papers. Returns ``topic_ids[0]`` per paper keyed by forum id."""
    topic_by_forum: dict[str, str] = {}
    for unit_path in sorted(
        extractions_dir.glob("*/llm-full-read/by-unit/paper_abstract_*.json")
    ):
        record = load_json(unit_path)
        if not isinstance(record, dict) or record.get("skipped_reason"):
            continue
        metadata = record.get("metadata") or {}
        forum_id = clean_text(metadata.get("paper_id"))
        if not forum_id:
            forum_id = clean_text(record.get("unit_id")).split(":")[-1]
        topic_ids = record.get("topic_ids")
        if forum_id and isinstance(topic_ids, list) and topic_ids:
            topic_by_forum.setdefault(forum_id, clean_text(topic_ids[0]))
    return topic_by_forum


def index_paper_text_artifacts(
    extractions_dir: Path,
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Index ``paper_text``/``workshop_paper_text`` extraction records by the
    sha256 of their extracted text (``text_sha256``). The on-disk text path in
    each record is retention-redacted, so content hashing is the join key."""
    by_text_sha: dict[str, tuple[str, dict[str, Any]]] = {}
    for artifact_path in sorted(extractions_dir.glob("*/by-artifact/*.json")):
        name = artifact_path.name
        if name.startswith("paper_text_"):
            source_kind = "main_conference"
        elif name.startswith("workshop_paper_text_"):
            source_kind = "workshop"
        else:
            continue
        record = load_json(artifact_path)
        if not isinstance(record, dict) or record.get("skipped_reason"):
            continue
        quality_metadata = record.get("quality_metadata") or {}
        text_sha = clean_text(quality_metadata.get("text_sha256")).lower()
        if text_sha:
            by_text_sha.setdefault(text_sha, (source_kind, record))
    return by_text_sha


def load_full_text_paper_items(
    workspace: Path,
    extractions_dir: Path,
    paper_text_dir: Path,
    pdf_hash_map: dict[str, dict[str, Any]],
    llm_topic_map: dict[str, str] | None = None,
) -> list[ExportItem]:
    """Build cleaned full-text paper export items, tagged with a primary topic.

    Each extracted text file is content-hashed and joined to its extraction
    record and, where possible, to local paper metadata (for a clean title) via
    the OpenReview PDF hash. Primary topic prefers the exhaustive LLM full-read
    assignment (``llm_topic_map``, keyed by forum id) and falls back to the
    confidence-ranked keyword heuristic. Text is cleaned and grouped downstream
    by primary topic."""
    if not extractions_dir.exists() or not paper_text_dir.exists():
        return []

    llm_topic_map = llm_topic_map or {}
    artifacts_by_sha = index_paper_text_artifacts(extractions_dir)
    items: list[ExportItem] = []
    for text_path in sorted(paper_text_dir.rglob("*.txt")):
        data = text_path.read_bytes()
        text_sha = hashlib.sha256(data).hexdigest()
        joined = artifacts_by_sha.get(text_sha)
        if joined is None:
            continue
        source_kind, record = joined
        quality_metadata = record.get("quality_metadata") or {}

        raw = data.decode("utf-8", errors="replace")
        cleaned = clean_paper_full_text(raw)
        if not cleaned:
            continue

        source_path = clean_text(record.get("source_artifact_path"))
        hash_match = PDF_HASH_RE.search(source_path)
        meta = (
            pdf_hash_map.get(hash_match.group(1).lower()) if hash_match else None
        ) or {}

        title = clean_text(meta.get("title")) or extract_title_from_text(raw)
        if not title:
            title = text_path.stem
        forum_id = clean_text(meta.get("forum") or meta.get("id"))
        llm_topic = llm_topic_map.get(forum_id) if forum_id else None
        if llm_topic and llm_topic in TOPIC_DISPLAY_NAMES:
            topic, topic_source = llm_topic, "llm_full_read"
        else:
            topic = heuristic_primary_topic(
                record.get("topic_ids"), quality_metadata
            )
            topic_source = "heuristic_confidence"
        paper_id = clean_text(
            meta.get("id") or meta.get("forum") or record.get("source_artifact_id")
        )

        lines = [f"# Paper: {title}", ""]
        append_metadata_line(lines, "Source kind", source_kind)
        append_metadata_line(lines, "Primary topic", TOPIC_DISPLAY_NAMES.get(topic, topic))
        append_metadata_line(lines, "Topic source", topic_source)
        append_metadata_line(lines, "Paper ID", paper_id)
        append_metadata_line(
            lines, "OpenReview", meta.get("forum_url") or meta.get("openreview_url")
        )
        append_metadata_line(lines, "Venue", meta.get("venue"))
        append_metadata_line(lines, "Authors", meta.get("authors"))
        append_metadata_line(lines, "Primary area", meta.get("primary_area"))
        append_metadata_line(lines, "Keywords", meta.get("keywords"))
        append_metadata_line(lines, "PDF", meta.get("pdf_url"))
        lines.append("")
        lines.append("## Full text")
        lines.append(cleaned)

        items.append(
            ExportItem(
                kind=f"{source_kind}_paper_full",
                title=title,
                source_id=paper_id,
                source_path=display_path(text_path, workspace),
                text="\n".join(lines).strip() + "\n",
                topic=topic,
                topic_source=topic_source,
            )
        )
    return items


def guess_transcript_metadata(
    path: Path,
    sidecar: dict[str, Any],
    talk_metadata: dict[str, TalkInfo],
) -> tuple[str, str]:
    source_path = clean_text(sidecar.get("source_path"))
    searchable = f"{path.name} {source_path}"
    talk_match = TALK_ID_RE.search(searchable)
    if talk_match:
        talk_id = talk_match.group(1)
        if talk_id in talk_metadata:
            talk = talk_metadata[talk_id]
            return talk.title, talk.author

    poster_match = POSTER_ID_RE.search(f"/{source_path}/")
    if poster_match:
        poster_key = poster_match.group(1)
        if poster_key in talk_metadata:
            talk = talk_metadata[poster_key]
            return talk.title, talk.author
        return f"Main conference presentation {poster_key}", ""

    slideslive_id = clean_text(sidecar.get("slideslive_id"))
    if slideslive_id:
        return f"SlidesLive {slideslive_id}", ""

    source_sha = clean_text(sidecar.get("source_sha256"))
    if source_sha:
        return f"Speech source {source_sha[:12]}", ""
    return path.stem, ""


def load_transcript_items(
    workspace: Path,
    transcripts_dir: Path | None,
    talk_metadata: dict[str, TalkInfo],
) -> list[ExportItem]:
    root = transcripts_dir or workspace / "storage/transcription-ocr/speech-transcripts"
    if not root.exists():
        return []

    items: list[ExportItem] = []
    for transcript_path in sorted(root.glob("**/*.txt")):
        sidecar_path = transcript_path.with_suffix(".json")
        sidecar: dict[str, Any] = {}
        if sidecar_path.exists():
            payload = load_json(sidecar_path)
            if isinstance(payload, dict):
                sidecar = payload

        title, author = guess_transcript_metadata(transcript_path, sidecar, talk_metadata)
        source_id = first_present(
            [
                sidecar.get("transcript_sha256"),
                sidecar.get("source_sha256"),
                transcript_path.stem,
            ]
        )
        lines = [f"# {title}", ""]
        if author:
            lines.append(f"Author: {author}")
            lines.append("")
        lines.append(safe_read_text(transcript_path).strip())
        items.append(
            ExportItem(
                kind="speech_transcript",
                title=title,
                source_id=source_id,
                source_path=display_path(transcript_path, workspace),
                text="\n".join(lines).strip() + "\n",
            )
        )
    return items


def analysis_extensions(value: str) -> set[str]:
    extensions = set()
    for raw in value.split(","):
        extension = raw.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.add(extension)
    return extensions or {".md"}


def load_analysis_items(
    workspace: Path,
    analysis_dir: Path,
    extensions: set[str],
) -> list[ExportItem]:
    if not analysis_dir.exists():
        return []
    items: list[ExportItem] = []
    for path in sorted(analysis_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.parts):
            continue
        raw = safe_read_text(path)
        rel_path = display_path(path, workspace)
        title = f"Analysis Document: {rel_path}"
        lines = [f"# {title}", "", raw.strip()]
        items.append(
            ExportItem(
                kind="analysis_document",
                title=title,
                source_id=rel_path,
                source_path=rel_path,
                text="\n".join(lines).strip() + "\n",
            )
        )
    return items


def write_items(
    items: list[ExportItem],
    notebooklm_dir: Path,
    prefix: str,
    max_bytes: int,
    max_words: int,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    chunker = ChunkWriter(notebooklm_dir, prefix, max_bytes=max_bytes, max_words=max_words)
    index_rows: list[dict[str, str]] = []
    for item in items:
        part_path = chunker.add(item.title, item.text)
        index_rows.append(
            {
                "kind": item.kind,
                "title": item.title,
                "source_id": item.source_id,
                "source_path": item.source_path,
                "output_file": display_path(part_path, output_root),
            }
        )
    return chunker.finalize(), index_rows


def write_items_grouped_by_topic(
    items: list[ExportItem],
    notebooklm_dir: Path,
    base_prefix: str,
    max_bytes: int,
    max_words: int,
    output_root: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Pack items into max-size parts within each primary topic so every
    NotebookLM source stays topically coherent. Topics are emitted in their
    numeric taxonomy order; the per-topic prefix keeps part files self-describing."""
    by_topic: dict[str, list[ExportItem]] = {}
    for item in items:
        by_topic.setdefault(item.topic or UNCATEGORIZED_TOPIC, []).append(item)

    def topic_sort_key(topic: str) -> tuple[int, str]:
        order = list(TOPIC_DISPLAY_NAMES).index(topic) if topic in TOPIC_DISPLAY_NAMES else 999
        return (order, topic)

    parts_by_topic: dict[str, list[dict[str, Any]]] = {}
    index_rows: list[dict[str, str]] = []
    for topic in sorted(by_topic, key=topic_sort_key):
        prefix = f"{base_prefix}_{topic}"
        chunker = ChunkWriter(notebooklm_dir, prefix, max_bytes=max_bytes, max_words=max_words)
        for item in by_topic[topic]:
            part_path = chunker.add(item.title, item.text)
            index_rows.append(
                {
                    "kind": item.kind,
                    "title": item.title,
                    "source_id": item.source_id,
                    "source_path": item.source_path,
                    "output_file": display_path(part_path, output_root),
                }
            )
        parts_by_topic[topic] = chunker.finalize()
    return parts_by_topic, index_rows


def prepare_notebooklm_dir(notebooklm_dir: Path) -> None:
    notebooklm_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ("transcripts", "paper_abstracts", "papers_full", "slides", "analysis_docs"):
        for old_part in notebooklm_dir.glob(f"{prefix}_*part_*.txt"):
            old_part.unlink()
        for old_part in notebooklm_dir.glob(f"{prefix}_*part_*.md"):
            old_part.unlink()
    for old_overview in notebooklm_dir.glob("00_conference_overview.md"):
        old_overview.unlink()


def _parse_markdown_sections(md_text: str) -> dict[str, list[str]]:
    """Split a Markdown doc into ``{lowercased H2 heading: [body lines]}``."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in md_text.splitlines():
        heading = re.match(r"^##\s+(.*)$", line)
        if heading:
            current = heading.group(1).strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _section_bullets(sections: dict[str, list[str]], heading: str, limit: int) -> list[str]:
    bullets: list[str] = []
    for line in sections.get(heading.lower(), []):
        stripped = line.strip()
        if stripped.startswith("- "):
            text = clean_text(stripped[2:])
            if text:
                bullets.append(text)
    return bullets[:limit]


def _load_topic_titles(workspace: Path) -> dict[str, str]:
    """Rich per-topic display titles from topic_strategy.json (fallback to the
    built-in short names)."""
    titles = dict(TOPIC_DISPLAY_NAMES)
    strategy_path = workspace / "analysis/topic_strategy.json"
    if strategy_path.exists():
        payload = load_json(strategy_path)
        final_topics = payload.get("final_topics") if isinstance(payload, dict) else None
        for entry in final_topics or []:
            if not isinstance(entry, dict):
                continue
            topic_id = clean_text(entry.get("topic_id"))
            title = clean_text(entry.get("title"))
            if topic_id and title:
                titles[topic_id] = title
    return titles


def _conference_label(workspace: Path) -> str:
    strategy_path = workspace / "analysis/topic_strategy.json"
    if strategy_path.exists():
        payload = load_json(strategy_path)
        if isinstance(payload, dict):
            label = clean_text(payload.get("conference_label"))
            if label:
                # Render a year suffix readably, e.g. "ICLR2026" -> "ICLR 2026".
                return re.sub(r"([A-Za-z])(\d{4})\b", r"\1 \2", label)
    return "the conference"


def _executive_summary_text(workspace: Path) -> str:
    exec_path = workspace / "analysis/executive_summary.md"
    return safe_read_text(exec_path) if exec_path.exists() else ""


def _executive_cross_themes(exec_text: str, limit: int = 8) -> list[str]:
    """The ``N) ...`` ranked cross-topic conclusion headers from the exec summary."""
    themes: list[str] = []
    for line in exec_text.splitlines():
        match = re.match(r"^\d+\)\s+(.*)$", line.strip())
        if match:
            theme = clean_text(match.group(1))
            if theme:
                themes.append(theme)
    return themes[:limit]


def _executive_thesis(exec_text: str) -> str:
    match = re.search(r"actionable thesis:\s*(.+)", exec_text, re.IGNORECASE)
    if not match:
        return ""
    thesis = clean_text(match.group(1))
    return thesis[:1].upper() + thesis[1:] if thesis else ""


def _topic_one_liner(workspace: Path, topic_id: str, max_chars: int = 320) -> str:
    """First sentence of a topic's exhaustive paper summary, length-capped."""
    summary_path = workspace / f"analysis/exhaustive_summaries/{topic_id}__papers.md"
    if not summary_path.exists():
        return ""
    sections = _parse_markdown_sections(safe_read_text(summary_path))
    summary = clean_text(" ".join(sections.get("summary", [])))
    if not summary:
        return ""
    sentence_match = re.match(r"(.+?\.)(?:\s|$)", summary)
    sentence = sentence_match.group(1) if sentence_match else summary
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return sentence


def load_topic_slide_items(
    workspace: Path,
    titles: Mapping[str, str],
    decks_dir: Path | None = None,
) -> list[ExportItem]:
    """Load each topic's reviewed slide deck (``decks/<topic_id>/slides.md``) as
    a topic-tagged export item, so the slides can be grouped per topic
    alongside the papers."""
    root = decks_dir or workspace / "decks"
    if not root.exists():
        return []
    items: list[ExportItem] = []
    for topic_id in TOPIC_DISPLAY_NAMES:
        if topic_id == UNCATEGORIZED_TOPIC:
            continue
        slide_path = root / topic_id / "slides.md"
        if not slide_path.exists():
            continue
        text = safe_read_text(slide_path).strip()
        if not text:
            continue
        title = titles.get(topic_id, TOPIC_DISPLAY_NAMES.get(topic_id, topic_id))
        items.append(
            ExportItem(
                kind="topic_slides",
                title=f"{title} — slide deck",
                source_id=topic_id,
                source_path=display_path(slide_path, workspace),
                text=text + "\n",
                topic=topic_id,
            )
        )
    return items


def build_conference_overview(
    workspace: Path,
    counts: Mapping[str, Any],
    full_text_topic_counts: Mapping[str, int],
    titles: Mapping[str, str],
) -> str:
    """Build a single orientation source summarizing the conference and the
    notebook's contents, grounded in the executive summary, the per-topic
    exhaustive summaries, and the real artifact counts."""
    label = _conference_label(workspace)
    exec_text = _executive_summary_text(workspace)
    thesis = _executive_thesis(exec_text)
    themes = _executive_cross_themes(exec_text)

    main = int(counts.get("full_text_main_papers", 0) or 0)
    workshop = int(counts.get("full_text_workshop_papers", 0) or 0)
    transcripts = int(counts.get("speech_transcripts", 0) or 0)
    analysis_docs = int(counts.get("analysis_documents", 0) or 0)

    lines = [f"# {label}: Conference Overview", ""]
    lines.append(
        f"This is an orientation to the {label} research corpus assembled for "
        "deep-dive study. Read it first to navigate the other sources in this "
        "notebook."
    )
    lines += ["", "## Scale", ""]
    if main or workshop:
        lines.append(
            f"- Papers (full text): {main + workshop:,} "
            f"({main:,} main-conference, {workshop:,} workshop)"
        )
    if transcripts:
        lines.append(f"- Talk and poster transcripts: {transcripts:,}")
    if analysis_docs:
        lines.append(f"- Analysis documents: {analysis_docs:,}")
    lines.append("- Organized into 12 topic areas.")

    if thesis:
        lines += ["", "## The throughline", "", thesis]

    if themes:
        lines += ["", "## Cross-topic conclusions", ""]
        lines += [f"{index}. {theme}" for index, theme in enumerate(themes, start=1)]

    lines += ["", "## The 12 topic areas", ""]
    for topic_id in TOPIC_DISPLAY_NAMES:
        if topic_id == UNCATEGORIZED_TOPIC:
            continue
        title = titles.get(topic_id, TOPIC_DISPLAY_NAMES.get(topic_id, topic_id))
        count = int(full_text_topic_counts.get(topic_id, 0) or 0)
        suffix = f" — {count:,} papers" if count else ""
        lines.append(f"### {title}{suffix}")
        one_liner = _topic_one_liner(workspace, topic_id)
        if one_liner:
            lines.append(one_liner)
        lines.append("")

    lines += [
        "## What's in this notebook",
        "",
        "- Per-topic full paper text, grouped by primary topic (sources prefixed "
        "`papers_full_<NN_topic>`).",
        "- Per-topic slide decks (sources prefixed `slides_<NN_topic>`).",
        "- Talk and poster transcripts.",
        "- Analysis documents, including the executive summary and the per-topic "
        "exhaustive summaries (one for papers, one for transcripts).",
        "- Presentation prompts (in the separate `presentation_prompts/` folder): "
        "paste-in NotebookLM instructions to generate a long (~60 min) and a short "
        "(~20 min) presentation for each topic, plus conference-wide variants.",
        "",
        "For the deepest synthesis, read the executive summary and the per-topic "
        "exhaustive summaries; for primary evidence, use the full-text paper "
        "sources and slide decks.",
    ]
    return "\n".join(lines).strip() + "\n"


# Length presets for the per-topic/global presentation prompts. Each maps to a
# target duration and how many grounding bullets to include (long = exhaustive,
# short = headline only).
PRESENTATION_LENGTHS: dict[str, dict[str, Any]] = {
    "long": {"minutes": 60, "key_points": 8, "tensions": 5, "open_questions": 5},
    "short": {"minutes": 20, "key_points": 3, "tensions": 2, "open_questions": 2},
}


def _compose_topic_presentation_prompt(
    label: str,
    title: str,
    length: str,
    key_points: list[str],
    tensions: list[str],
    open_questions: list[str],
) -> str:
    preset = PRESENTATION_LENGTHS[length]
    minutes = preset["minutes"]
    lines = [
        f"Presentation prompt — {label}: {title} ({length}, ~{minutes} minutes)",
        "",
        f'Produce a ~{minutes}-minute presentation on "{title}" as covered at {label}.',
        "",
        "Audience: ML researchers and practitioners who already know the "
        "fundamentals. Be technical and specific — name papers, methods, datasets, "
        "and numbers from the sources. Avoid hype and generic filler.",
        "",
    ]
    if length == "long":
        lines.append(
            f"Target length: a full ~{minutes}-minute talk. Structure it in clear "
            "sections — motivation and framing, the main lines of work, methods and "
            "results, debates and failure modes, and outlook — and spend real time "
            "on each theme rather than skimming."
        )
        cover_label = "Cover these themes in depth:"
        tension_label = "Devote a section to debating these tensions:"
        question_label = "Address these open questions:"
        closing = (
            "Use the topic's slide deck and full-text paper sources as the backbone; "
            'trace every claim to a specific paper. End with a "what to watch next" '
            "section."
        )
    else:
        lines.append(
            f"Target length: a tight ~{minutes}-minute talk. Cover only the most "
            "important developments — what changed, why it matters, and the single "
            "biggest open debate. Prioritize signal over coverage."
        )
        cover_label = "Center on these headline themes:"
        tension_label = "Surface the most important tension:"
        question_label = "Close with these open questions:"
        closing = (
            "Use the topic's slide deck and full-text sources; keep to "
            f"~{minutes} minutes and end with a three-point takeaway."
        )
    if key_points:
        lines += ["", cover_label] + [f"- {point}" for point in key_points]
    if tensions:
        lines += ["", tension_label] + [f"- {point}" for point in tensions]
    if open_questions:
        lines += ["", question_label] + [f"- {point}" for point in open_questions]
    lines += ["", closing]
    return "\n".join(lines).strip() + "\n"


def _compose_global_presentation_prompt(
    label: str,
    length: str,
    cross_themes: list[str],
    topic_titles: list[str],
) -> str:
    preset = PRESENTATION_LENGTHS[length]
    minutes = preset["minutes"]
    lines = [
        f"Presentation prompt — {label}: conference-wide overview ({length}, ~{minutes} minutes)",
        "",
        f"Produce a ~{minutes}-minute presentation that surveys {label} as a whole.",
        "",
        "Audience: ML researchers and research leadership. Be technical and "
        "concrete; cite specific papers, benchmarks, and results from the sources. "
        "No hype.",
        "",
    ]
    if length == "long":
        lines.append(
            f"Target length: a full ~{minutes}-minute talk. Tell the story of the "
            "conference in sections: the dominant throughlines, how the topic areas "
            "connect, the biggest surprises, and the open risks."
        )
        themes_label = "Center the narrative on these cross-topic conclusions:"
        closing = (
            "Connect the topic areas rather than listing them; where they reinforce "
            'or contradict each other, say so. End with a "what to watch next" '
            "section for the year ahead."
        )
    else:
        lines.append(
            f"Target length: a tight ~{minutes}-minute talk. Give the headline story "
            "of the conference: the few dominant throughlines and the biggest risks."
        )
        themes_label = "Lead with these cross-topic conclusions:"
        closing = (
            "Keep it fast-moving and synthetic; end with a three-point "
            '"what to watch next."'
        )
    if cross_themes:
        lines += ["", themes_label] + [f"- {theme}" for theme in cross_themes]
    if length == "long" and topic_titles:
        lines += [
            "",
            "Touch each of the main topic areas, but connect them rather than "
            "listing them:",
        ] + [f"- {title}" for title in topic_titles]
    lines += ["", closing]
    return "\n".join(lines).strip() + "\n"


def build_presentation_prompt_items(workspace: Path) -> list[tuple[str, str]]:
    """Build NotebookLM presentation prompts: a long (~60 min) and a short
    (~20 min) variant for each topic, plus long and short conference-wide
    variants. Each prompt is grounded in the exhaustive LLM summaries (topic
    Key Points / Tensions / Open Questions) and the executive summary.

    Returns ``[(filename, prompt_text), ...]``; these are instructions to paste
    into NotebookLM, not sources to upload."""
    label = _conference_label(workspace)
    titles = _load_topic_titles(workspace)
    summaries_dir = workspace / "analysis/exhaustive_summaries"
    items: list[tuple[str, str]] = []

    topic_ids = [tid for tid in TOPIC_DISPLAY_NAMES if tid != UNCATEGORIZED_TOPIC]
    for topic_id in topic_ids:
        title = titles.get(topic_id, TOPIC_DISPLAY_NAMES.get(topic_id, topic_id))
        summary_path = summaries_dir / f"{topic_id}__papers.md"
        key_points: list[str] = []
        tensions: list[str] = []
        open_questions: list[str] = []
        if summary_path.exists():
            sections = _parse_markdown_sections(safe_read_text(summary_path))
            key_points = _section_bullets(sections, "Key Points", 8)
            tensions = _section_bullets(sections, "Tensions Or Contrasts", 5)
            open_questions = _section_bullets(sections, "Open Questions", 5)
        for length, preset in PRESENTATION_LENGTHS.items():
            prompt = _compose_topic_presentation_prompt(
                label,
                title,
                length,
                key_points[: preset["key_points"]],
                tensions[: preset["tensions"]],
                open_questions[: preset["open_questions"]],
            )
            items.append((f"{topic_id}_{length}.txt", prompt))

    # Conference-wide prompts grounded in the executive summary's cross-topic
    # conclusions (numbered "N) ..." headers).
    cross_themes = _executive_cross_themes(_executive_summary_text(workspace))
    topic_titles = [titles.get(tid, TOPIC_DISPLAY_NAMES.get(tid, tid)) for tid in topic_ids]
    for length, preset in PRESENTATION_LENGTHS.items():
        prompt = _compose_global_presentation_prompt(
            label,
            length,
            cross_themes[: preset["key_points"]],
            topic_titles,
        )
        items.append((f"00_global_conference_{length}.txt", prompt))
    return items


def write_index(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["kind", "title", "source_id", "source_path", "output_file"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package conference transcripts, paper abstracts, and analysis docs for NotebookLM."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Conference workspace root, for example conferences/iclr2026.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output package root. Defaults to <workspace>/exports/notebooklm_export.",
    )
    parser.add_argument(
        "--papers-file",
        type=Path,
        default=None,
        help="Optional main-conference paper metadata JSON override.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Analysis directory. Defaults to <workspace>/analysis.",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=None,
        help="Speech transcript directory. Defaults to <workspace>/storage/transcription-ocr/speech-transcripts.",
    )
    parser.add_argument(
        "--workshop-papers-dir",
        type=Path,
        default=None,
        help="Workshop paper root. Defaults to <workspace>/storage/workshops/papers.",
    )
    parser.add_argument(
        "--analysis-extensions",
        default=DEFAULT_ANALYSIS_EXTENSIONS,
        help=(
            "Comma-separated analysis file extensions to include. "
            f"Default: {DEFAULT_ANALYSIS_EXTENSIONS}."
        ),
    )
    parser.add_argument(
        "--paper-mode",
        choices=PAPER_MODES,
        default=DEFAULT_PAPER_MODE,
        help=(
            "Paper export mode. 'abstract' (default) writes metadata + abstract "
            "chunks; 'full' writes cleaned full-text chunks grouped by topic; "
            "'both' writes both."
        ),
    )
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        default=None,
        help=(
            "Extraction artifact root for full-text paper mode. Defaults to "
            "<workspace>/storage/extractions."
        ),
    )
    parser.add_argument(
        "--paper-text-dir",
        type=Path,
        default=None,
        help=(
            "Extracted paper text root for full-text paper mode. Defaults to "
            "<workspace>/storage/transcription-ocr/text-extraction."
        ),
    )
    parser.add_argument(
        "--paper-topic-source",
        choices=("llm", "heuristic"),
        default="llm",
        help=(
            "Primary-topic source for full-text grouping. 'llm' (default) prefers "
            "the exhaustive LLM full-read assignment and falls back to the "
            "confidence-ranked keyword heuristic; 'heuristic' forces the keyword "
            "heuristic for all papers."
        ),
    )
    parser.add_argument(
        "--max-bytes",
        "--notebooklm-max-bytes",
        dest="max_bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum bytes per NotebookLM chunk before starting a new part.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=DEFAULT_MAX_WORDS,
        help="Maximum words per NotebookLM chunk before starting a new part.",
    )
    parser.add_argument(
        "--papers-max-bytes",
        type=int,
        default=None,
        help="Optional paper-specific byte chunk limit. Defaults to --max-bytes.",
    )
    parser.add_argument(
        "--papers-max-words",
        type=int,
        default=None,
        help="Optional paper-specific word chunk limit. Defaults to --max-words.",
    )
    parser.add_argument(
        "--skip-presentation-prompts",
        "--skip-podcast-prompts",
        dest="skip_presentation_prompts",
        action="store_true",
        help=(
            "Do not emit the per-topic and conference-wide NotebookLM presentation "
            "prompt text files (long ~60 min and short ~20 min variants)."
        ),
    )
    parser.add_argument(
        "--skip-slides",
        action="store_true",
        help="Do not include the per-topic slide decks as NotebookLM sources.",
    )
    parser.add_argument(
        "--decks-dir",
        type=Path,
        default=None,
        help=(
            "Presentation decks root for slide-deck sources. Defaults to "
            "<workspace>/decks."
        ),
    )
    parser.add_argument(
        "--skip-conference-overview",
        action="store_true",
        help=(
            "Do not emit the 00_conference_overview.md orientation source that "
            "summarizes the conference and the notebook's contents."
        ),
    )
    parser.add_argument(
        "--skip-corpus-inventory",
        action="store_true",
        help="Deprecated no-op. Hashes are preserved as-is and no longer dereferenced.",
    )
    parser.add_argument(
        "--presentations-dir",
        type=Path,
        default=None,
        help="Deprecated legacy option accepted for compatibility; use --workspace instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace
    populate_topic_display_names(workspace)
    output_dir = args.output_dir or workspace / "exports/notebooklm_export"
    analysis_dir = args.analysis_dir or workspace / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebooklm_dir = output_dir / "notebooklm"
    prepare_notebooklm_dir(notebooklm_dir)

    workshop_titles = load_workshop_titles(workspace)
    talk_metadata = build_talk_metadata(workspace)
    talk_metadata.update(build_main_presentation_metadata(workspace))
    main_papers = load_main_papers(workspace, args.papers_file)
    main_by_id, main_by_title = paper_lookup_maps(main_papers)
    workshop_papers = load_workshop_papers(
        workspace,
        main_by_id,
        main_by_title,
        workshop_titles,
        args.workshop_papers_dir,
    )
    all_papers = main_papers + workshop_papers

    write_abstracts = args.paper_mode in ("abstract", "both")
    write_full = args.paper_mode in ("full", "both")

    transcript_items = load_transcript_items(workspace, args.transcripts_dir, talk_metadata)
    paper_items = [format_paper(paper) for paper in all_papers] if write_abstracts else []
    full_text_items: list[ExportItem] = []
    if write_full:
        extractions_dir = args.extractions_dir or workspace / "storage/extractions"
        paper_text_dir = (
            args.paper_text_dir
            or workspace / "storage/transcription-ocr/text-extraction"
        )
        llm_topic_map = (
            build_llm_topic_map(extractions_dir)
            if args.paper_topic_source == "llm"
            else {}
        )
        full_text_items = load_full_text_paper_items(
            workspace,
            extractions_dir,
            paper_text_dir,
            build_pdf_hash_map(main_papers),
            llm_topic_map,
        )
    analysis_items = load_analysis_items(
        workspace,
        analysis_dir,
        analysis_extensions(args.analysis_extensions),
    )
    slide_items: list[ExportItem] = []
    if not args.skip_slides:
        slide_items = load_topic_slide_items(
            workspace, _load_topic_titles(workspace), args.decks_dir
        )

    index_rows: list[dict[str, str]] = []
    transcript_parts, rows = write_items(
        transcript_items,
        notebooklm_dir,
        "transcripts",
        args.max_bytes,
        args.max_words,
        output_dir,
    )
    index_rows.extend(rows)

    papers_max_bytes = args.papers_max_bytes or args.max_bytes
    papers_max_words = args.papers_max_words or args.max_words

    paper_parts: list[dict[str, Any]] = []
    if write_abstracts:
        paper_parts, rows = write_items(
            paper_items,
            notebooklm_dir,
            "paper_abstracts",
            papers_max_bytes,
            papers_max_words,
            output_dir,
        )
        index_rows.extend(rows)

    papers_full_parts: dict[str, list[dict[str, Any]]] = {}
    if write_full:
        papers_full_parts, rows = write_items_grouped_by_topic(
            full_text_items,
            notebooklm_dir,
            "papers_full",
            papers_max_bytes,
            papers_max_words,
            output_dir,
        )
        index_rows.extend(rows)

    slides_parts: dict[str, list[dict[str, Any]]] = {}
    if slide_items:
        slides_parts, rows = write_items_grouped_by_topic(
            slide_items,
            notebooklm_dir,
            "slides",
            papers_max_bytes,
            papers_max_words,
            output_dir,
        )
        index_rows.extend(rows)

    analysis_parts, rows = write_items(
        analysis_items,
        notebooklm_dir,
        "analysis_docs",
        args.max_bytes,
        args.max_words,
        output_dir,
    )
    index_rows.extend(rows)

    conference_overview_path: str | None = None
    if not args.skip_conference_overview:
        topic_titles = _load_topic_titles(workspace)
        full_text_topic_counts: dict[str, int] = {}
        for item in full_text_items:
            full_text_topic_counts[item.topic] = (
                full_text_topic_counts.get(item.topic, 0) + 1
            )
        overview_main = sum(
            1 for item in full_text_items if item.kind.startswith("main_")
        )
        overview_text = build_conference_overview(
            workspace,
            {
                "speech_transcripts": len(transcript_items),
                "full_text_main_papers": overview_main,
                "full_text_workshop_papers": len(full_text_items) - overview_main,
                "analysis_documents": len(analysis_items),
            },
            full_text_topic_counts,
            topic_titles,
        )
        overview_file = notebooklm_dir / "00_conference_overview.md"
        overview_file.write_text(overview_text, encoding="utf-8")
        conference_overview_path = display_path(overview_file, output_dir)
        index_rows.insert(
            0,
            {
                "kind": "conference_overview",
                "title": f"{_conference_label(workspace)}: Conference Overview",
                "source_id": "conference_overview",
                "source_path": "analysis/executive_summary.md",
                "output_file": conference_overview_path,
            },
        )

    index_path = output_dir / "index.csv"
    write_index(index_path, index_rows)

    # Legacy single-length podcast prompts are superseded by presentation prompts.
    legacy_prompts_dir = output_dir / "podcast_prompts"
    if legacy_prompts_dir.exists():
        for old_prompt in legacy_prompts_dir.glob("*.txt"):
            old_prompt.unlink()
        try:
            legacy_prompts_dir.rmdir()
        except OSError:
            pass

    presentation_prompt_paths: list[str] = []
    if not args.skip_presentation_prompts:
        prompts_dir = output_dir / "presentation_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        for old_prompt in prompts_dir.glob("*.txt"):
            old_prompt.unlink()
        for filename, text in build_presentation_prompt_items(workspace):
            prompt_path = prompts_dir / filename
            prompt_path.write_text(text, encoding="utf-8")
            presentation_prompt_paths.append(display_path(prompt_path, output_dir))

    full_text_main = sum(1 for item in full_text_items if item.kind.startswith("main_"))
    full_text_workshop = len(full_text_items) - full_text_main
    full_text_topic_llm = sum(
        1 for item in full_text_items if item.topic_source == "llm_full_read"
    )
    full_text_topic_heuristic = len(full_text_items) - full_text_topic_llm
    papers_full_source_count = sum(len(parts) for parts in papers_full_parts.values())
    slides_source_count = sum(len(parts) for parts in slides_parts.values())

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "notebooklm_dir": str(notebooklm_dir),
        "paper_mode": args.paper_mode,
        "chunk_limits": {
            "max_bytes": args.max_bytes,
            "max_words": args.max_words,
            "papers_max_bytes": args.papers_max_bytes or args.max_bytes,
            "papers_max_words": args.papers_max_words or args.max_words,
        },
        "counts": {
            "speech_transcripts": len(transcript_items),
            "main_conference_papers": len(main_papers),
            "workshop_papers": len(workshop_papers),
            "full_text_main_papers": full_text_main,
            "full_text_workshop_papers": full_text_workshop,
            "full_text_topic_from_llm_full_read": full_text_topic_llm,
            "full_text_topic_from_heuristic": full_text_topic_heuristic,
            "papers_full_source_files": papers_full_source_count,
            "topic_slide_decks": len(slide_items),
            "slides_source_files": slides_source_count,
            "analysis_documents": len(analysis_items),
            "reference_map_entries": 0,
        },
        "parts": {
            "transcripts": transcript_parts,
            "paper_abstracts": paper_parts,
            "papers_full": papers_full_parts,
            "slides": slides_parts,
            "analysis_docs": analysis_parts,
        },
        "conference_overview": conference_overview_path,
        "presentation_prompts": presentation_prompt_paths,
        "index": str(index_path),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Packaged NotebookLM sources to: {output_dir}")
    print(f"  paper mode: {args.paper_mode}")
    print(f"  transcripts: {len(transcript_items)}")
    print(f"  main papers: {len(main_papers)}")
    print(f"  workshop papers: {len(workshop_papers)}")
    if write_full:
        print(
            f"  full-text papers: {len(full_text_items)} "
            f"(main {full_text_main}, workshop {full_text_workshop}) "
            f"-> {papers_full_source_count} topic-grouped source files"
        )
        print(
            f"  topic source: LLM full-read {full_text_topic_llm}, "
            f"heuristic fallback {full_text_topic_heuristic}"
        )
    if slide_items:
        print(
            f"  topic slide decks: {len(slide_items)} "
            f"-> {slides_source_count} topic-grouped source files"
        )
    print(f"  analysis docs: {len(analysis_items)}")
    print(f"  conference overview: {'yes' if conference_overview_path else 'skipped'}")
    print(f"  presentation prompts: {len(presentation_prompt_paths)}")
    print(f"  manifest: {manifest_path}")
    print(f"  index: {index_path}")


if __name__ == "__main__":
    main()
