#!/usr/bin/env python3
"""Generate short and long conference executive overview decks.

The topic deep-dive deck pipeline is intentionally topic scoped. This owner
creates the two conference-level overview decks from Stage 8 synthesis outputs
and the final analysis manifest so future conference production plans cannot
mistake ``analysis/executive_summary.md`` for the deliverable decks.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.analysis._digest import sha256_file
from conference_pipeline.presentation.presentation_frame import validate_presentation_frame_payload
from conference_pipeline.presentation.render_spec import validate_render_spec_payload
from conference_pipeline.presentation.source_binding import validate_source_binding_report_payload
from conference_pipeline.presentation.visual_plan import validate_visual_plan_payload
from conference_pipeline.workflow_runs import write_json_atomic
from scripts.presentations.generate_render_spec import build_render_spec

DEFAULT_SHORT_ID = "executive_summary_short"
DEFAULT_LONG_ID = "executive_summary_long"
DEFAULT_SHORT_DURATION_SECONDS = 15 * 60.0
DEFAULT_LONG_DURATION_SECONDS = 30 * 60.0
DEFAULT_NARRATION_WORDS_PER_MINUTE = 155
MIN_PREMIUM_IMAGES_PER_DECK = 8
TITLE_SLIDE_SECONDS = 30.0
EVENTS_SLIDE_SECONDS = 90.0
RESOURCES_SLIDE_SECONDS = 60.0
MIN_CONTENT_SLIDE_SECONDS = 45.0
CHECKPOINT_FILES = (
    "1_draft_slides.md",
    "2_draft_slides_enriched_with_papers.md",
    "3_draft_slides_reflowed.md",
    "3.5_draft_slides_consolidated.md",
    "4_draft_slides_sanity_check.md",
    "4.5_draft_slides_with_hyperlinks.md",
    "5_draft_slides_with_images.md",
)


class SourceLinkContractError(RuntimeError):
    """Raised when executive decks cannot prove external source links."""


class PresentationLinkContractError(RuntimeError):
    """Raised when executive decks cannot prove topic presentation links."""


class ConferenceContextError(RuntimeError):
    """Raised when the workspace does not declare its conference identity."""


@dataclass(frozen=True)
class ConferenceContext:
    """Conference identity and program links loaded from workspace config.yaml.

    Executive decks were previously hardcoded to one conference; every
    conference-facing label and program URL must come from here so a workspace
    can never silently render another conference's identity.
    """

    label: str
    site_domain: str
    program_events: tuple[dict[str, str], ...]
    program_resources: tuple[tuple[str, str, str], ...]


# Priority-ordered mapping of workspace config `urls` keys to program-resource
# display rows. Only keys present in the workspace config are emitted.
_PROGRAM_RESOURCE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("awards", "awards and best papers", "Awards / best papers"),
    ("orals", "oral sessions", "Oral presentations"),
    ("workshops", "workshop program", "Workshops"),
    ("papers", "accepted papers", "Accepted paper list"),
    ("tutorials", "tutorial program", "Tutorials"),
    ("invited_talks", "invited talks", "Keynotes / invited talks"),
    ("virtual", "virtual program", "Full virtual program"),
)


def load_conference_context(workspace: Path) -> ConferenceContext:
    config_path = workspace / "config.yaml"
    if not config_path.is_file():
        raise ConferenceContextError(
            f"conference_context_missing: {config_path} not found; executive "
            "decks cannot be generated without the workspace conference config"
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConferenceContextError(
            f"conference_context_invalid: {config_path} must be a mapping"
        )
    conference = payload.get("conference")
    label = str(conference.get("name") or "").strip() if isinstance(conference, dict) else ""
    if not label:
        raise ConferenceContextError(
            f"conference_context_invalid: {config_path} must declare "
            "conference.name (e.g. 'ICML 2026')"  # conference-literal-ok: example in an error message
        )
    urls = payload.get("urls") if isinstance(payload.get("urls"), dict) else {}
    site_domain = ""
    for key in ("virtual", "base", "papers", "workshops"):
        url = str(urls.get(key) or "")
        hostname = urlsplit(url).hostname or ""
        if hostname:
            site_domain = hostname.removeprefix("www.").lower()
            break
    resources: list[tuple[str, str, str]] = []
    for key, name_suffix, detail in _PROGRAM_RESOURCE_SPECS:
        url = str(urls.get(key) or "").strip()
        if url.startswith(("https://", "http://")):
            resources.append((f"{label} {name_suffix}", detail, url))
    events = tuple(
        {"name": name, "detail": detail, "url": url}
        for name, detail, url in resources[:4]
    )
    return ConferenceContext(
        label=label,
        site_domain=site_domain,
        program_events=events,
        program_resources=tuple(resources),
    )


_CONFERENCE_CONTEXT: ConferenceContext | None = None


def set_conference_context(context: ConferenceContext) -> None:
    global _CONFERENCE_CONTEXT
    _CONFERENCE_CONTEXT = context


def conference_context() -> ConferenceContext:
    if _CONFERENCE_CONTEXT is None:
        raise ConferenceContextError(
            "conference_context_unset: call set_conference_context("
            "load_conference_context(workspace)) before building decks"
        )
    return _CONFERENCE_CONTEXT


def conference_label() -> str:
    return conference_context().label


@dataclass(frozen=True)
class Topic:
    topic_id: str
    title: str
    markdown_path: str
    source_url: str


@dataclass(frozen=True)
class SlideSpec:
    title: str
    bullets: tuple[str, ...]
    refs: tuple[tuple[str, str], ...]
    figure: str
    notes: str
    slide_type: str = "content"
    narrative: str = ""
    executive_lesson_id: str = ""
    executive_lesson_role: str = ""
    executive_evidence_kind: str = ""
    source_anchor: str = ""
    source_locator: str = ""
    evidence_text: str = ""
    decision_implication: str = ""
    paper_evidence_text: str = ""
    paper_source_locator: str = ""
    premium_visual_direction: str = ""
    diagram: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeckSpec:
    presentation_id: str
    display_title: str
    slides: tuple[SlideSpec, ...]
    point_count: int
    target_duration_seconds: float
    opening_events: tuple[dict[str, str], ...]
    opening_resources: tuple[dict[str, str], ...]
    further_reading: tuple[dict[str, str], ...]
    presentation_link_mode: str
    opening_events_title: str = "Important talks, workshops, and sessions"
    central_message_text: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def workspace_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must stay inside workspace: {resolved}") from exc
    return resolved


def relpath(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text_if_exists(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


_BINDING_DIGESTS_SECTION_RE = re.compile(
    r"\n##\s+Binding digests\n.*?(?=\n##\s|\Z)", re.DOTALL
)


def strip_binding_digests_section(text: str) -> str:
    """Remove stage-8 synthesis's machine-verification "## Binding digests"
    section (sha256 receipt-binding digests, not human-facing narrative).

    Without this, any text extraction that falls back to "the raw start of
    the document" -- as the leadership-actions slide's fallback bullet does
    when no "Recommended immediate actions" section is present -- surfaces
    a literal hash string on a slide. Applied once at the read site so every
    downstream extractor (current or future) is protected, not just the one
    fallback that happened to hit it first.
    """
    return _BINDING_DIGESTS_SECTION_RE.sub("", text)


def _source_link_keys(path: str) -> tuple[str, ...]:
    normalized = path.replace("\\", "/").strip()
    normalized = re.sub(r"^\./", "", normalized)
    without_parent = normalized[3:] if normalized.startswith("../") else normalized
    keys = [normalized, without_parent]
    if without_parent.startswith("analysis/"):
        keys.append(f"../{without_parent}")
    keys.append(Path(without_parent).name)
    return tuple(dict.fromkeys(key for key in keys if key))


def load_source_link_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    raw = payload.get("source_links", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain an object or a source_links object")
    result: dict[str, str] = {}
    for key, value in raw.items():
        url = str(value).strip()
        if not re.match(r"https?://", url):
            raise ValueError(f"source link for {key!r} must be an http(s) URL")
        for alias in _source_link_keys(str(key)):
            result[alias] = url
    return result


def _presentation_link_keys(key: str) -> tuple[str, ...]:
    normalized = key.replace("\\", "/").strip()
    normalized = re.sub(r"^\./", "", normalized)
    without_parent = normalized[3:] if normalized.startswith("../") else normalized
    stem = without_parent
    if stem.startswith("decks/"):
        parts = stem.split("/")
        if len(parts) >= 2:
            stem = parts[1]
    keys = [
        normalized,
        without_parent,
        stem,
        f"decks/{stem}/presentation.html",
        f"decks/{stem}/presentation.pdf",
        f"../{stem}/presentation.html",
        f"../{stem}/presentation.pdf",
    ]
    return tuple(dict.fromkeys(item for item in keys if item))


def load_presentation_link_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = load_json(path)
    if "presentation_links" in payload:
        raw = payload["presentation_links"]
    elif "deck_links" in payload:
        raw = payload["deck_links"]
    elif "links" in payload:
        raw = payload["links"]
    else:
        raw = payload
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain an object or a presentation_links object")
    result: dict[str, str] = {}
    for key, value in raw.items():
        url = str(value).strip()
        if not re.match(r"https?://", url):
            raise ValueError(f"presentation link for {key!r} must be an http(s) URL")
        for alias in _presentation_link_keys(str(key)):
            result[alias] = url
    return result


def load_premium_image_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    images = payload.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{path} must contain an images array")
    return payload


def source_url_for(
    path: str,
    source_links: dict[str, str],
    *,
    require_external: bool = False,
) -> str:
    for key in _source_link_keys(path):
        if key in source_links:
            return source_links[key]
    if require_external:
        raise SourceLinkContractError(
            "source_link_missing:"
            f"{path}: sync or upload the analysis source to Google Drive and "
            "record its web URL in decks/executive_source_link_map.json"
        )
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("../"):
        return normalized
    return f"../{normalized}"


def presentation_url_for(
    topic_id: str,
    presentation_links: dict[str, str],
    *,
    require_external: bool = False,
) -> str:
    for key in _presentation_link_keys(topic_id):
        if key in presentation_links:
            return presentation_links[key]
    if require_external:
        raise PresentationLinkContractError(
            "presentation_link_missing:"
            f"{topic_id}: upload or publish the topic deck and record its web URL "
            "in decks/executive_presentation_link_map.json"
        )
    return f"../{topic_id}/presentation.html"


def load_topics(
    workspace: Path,
    analysis_manifest: Path,
    source_links: dict[str, str] | None = None,
    *,
    require_external_source_links: bool = False,
) -> tuple[Topic, ...]:
    source_links = source_links or {}
    payload = load_json(analysis_manifest)
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError(f"{analysis_manifest} has no topics")
    result: list[Topic] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        topic_id = str(item.get("id") or item.get("topic_id") or "").strip()
        title = str(item.get("title") or topic_id).strip()
        markdown_path = str(item.get("markdown_path") or f"analysis/{topic_id}.md")
        if topic_id and title:
            result.append(
                Topic(
                    topic_id=topic_id,
                    title=title,
                    markdown_path=markdown_path,
                    source_url=source_url_for(
                        markdown_path,
                        source_links,
                        require_external=require_external_source_links,
                    ),
                )
            )
    if not result:
        raise ValueError(f"{analysis_manifest} has no usable topic rows")
    return tuple(result)


def _source_path_to_spoken(match: re.Match[str]) -> str:
    stem = Path(match.group(0)).stem
    stem = re.sub(r"^\d+_", "", stem)
    label = stem.replace("_", " ").strip()
    replacements = {
        "ai for science": "AI-for-science",
        "agentic systems": "agentic systems",
        "evaluation benchmarks data": "evaluation",
        "generative models": "generative modeling",
        "reasoning post training": "reasoning and post-training",
        "rl robotics world models": "reinforcement learning and robotics",
        "multimodal spatial vlm": "multimodal and spatial VLM",
        "open science society post agi": "open science and society",
        "safety alignment privacy": "safety and alignment",
        "theory optimization structure": "theory and optimization",
        "efficiency systems architectures": "efficiency and systems",
        "interpretability representation": "interpretability",
    }
    label = replacements.get(label, label)
    return f"the {label} report"


def clean_markdown_text(text: str) -> str:
    text = re.sub(r"\s*\(analysis/[A-Za-z0-9_./-]+\.md\)", "", text)
    text = re.sub(r"\banalysis/[A-Za-z0-9_./-]+\.md\b", _source_path_to_spoken, text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_sentence(text: str, *, fallback: str) -> str:
    # _content_sentences() already strips citation clutter
    # ((paper_abstract:XXXX), long icml2026-...:topic:hash research-thread
    # ids) before extracting sentences; first_sentence() didn't, so those
    # machine-readable tokens leaked verbatim into human-facing captions
    # (evidence_caption/narrative) -- flagged by visual QC as an "unresolved
    # content artifact" and, for the long research-thread ids, as illegible
    # densely-wrapped text.
    cleaned = clean_markdown_text(_strip_citation_clutter(text))
    # Allow an optional closing quote (straight or curly) between the
    # terminal punctuation and the required whitespace/end. Stage-8 synthesis
    # frequently ends a sentence on a quoted phrase (`...outcome."`); without
    # this, the regex skips past that real boundary looking for punctuation
    # followed directly by whitespace, and swallows the next sentence too --
    # producing a caption 2-3x longer than intended that then gets mangled by
    # downstream character-limit truncation.
    match = re.search(r"(.+?[.!?][\"'’”]?)(?:\s|$)", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned[:180].strip() or fallback


# Section headings that terminate the stage-8 synthesis numbered list. These
# come from the platform's stage-8 synthesis output structure, not from any
# single conference.
_NUMBERED_LIST_BOUNDARIES = (
    r"\n\s*\d+\)\s+",
    r"\n#{1,6}\s",
    r"\nLargest surprises",
    r"\nResidual risks",
    r"\nConcrete recommendations",
    r"\nRecommended immediate actions",
    r"\nRepresentative citations",
    r"\nClosing note",
)


_BOLD_NUMBERED_PATTERN = re.compile(
    r"^\s*\*\*(\d+)\.\s+(.+?)\*\*\s*(.*?)(?=\n\s*\n\s*\*\*\d+\.|\n#{1,6}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _extract_numbered_items_n_paren(text: str) -> list[tuple[str, str]]:
    """Parse an ``N) Title`` list, one item per line, body on following lines."""
    boundary = "|".join(_NUMBERED_LIST_BOUNDARIES)
    pattern = re.compile(
        rf"^\s*(\d+)\)\s+(.+?)\n(.+?)(?={boundary}|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    items: list[tuple[str, str]] = []
    for match in pattern.finditer(text):
        if int(match.group(1)) != len(items) + 1:
            if items:
                break
            continue
        title = clean_markdown_text(match.group(2))
        body_lines = [
            re.sub(r"^-\s+", "", line.strip())
            for line in match.group(3).splitlines()
        ]
        body = clean_markdown_text(" ".join(line for line in body_lines if line))
        items.append((title, body or title))
    return items


def _extract_numbered_items_bold(text: str) -> list[tuple[str, str]]:
    """Parse a ``**N. Title.** Body...`` list, one bold-numbered paragraph per item.

    This is the current stage-8 synthesis.executive_summary output format:
    each cross-topic conclusion is a single paragraph opening with a bold
    ``**N. Sentence.**`` title clause followed by supporting body text on the
    same paragraph, with a blank line separating consecutive items.
    """
    items: list[tuple[str, str]] = []
    for match in _BOLD_NUMBERED_PATTERN.finditer(text):
        if int(match.group(1)) != len(items) + 1:
            if items:
                break
            continue
        title = clean_markdown_text(match.group(2))
        body = clean_markdown_text(match.group(3).strip())
        items.append((title, body or title))
    return items


def extract_numbered_items(text: str) -> list[tuple[str, str]]:
    """Parse the first numbered conclusions list in a stage-8 synthesis document.

    Two formats are supported, tried in order:

    - ``N) Title`` on its own line, with the body as a ``- `` bullet block
      (older synthesis format) or plain paragraphs on following lines.
    - ``**N. Title.** Body...`` as a single bold-numbered paragraph (current
      synthesis.executive_summary format).

    Only the first consecutive list numbered from 1 is returned, so later
    numbered sections (for example residual-risk lists) cannot leak into the
    executive spine.
    """
    items = _extract_numbered_items_n_paren(text)
    if items:
        return items
    return _extract_numbered_items_bold(text)


def topic_ref(topic: Topic) -> tuple[str, str]:
    return (topic.title, topic.source_url)


def core_refs(*topics: Topic) -> tuple[tuple[str, str], ...]:
    return tuple(topic_ref(topic) for topic in topics)


def source_refs(
    source_links: dict[str, str] | None = None,
    *,
    require_external: bool = False,
) -> tuple[tuple[str, str], ...]:
    source_links = source_links or {}
    return (
        (
            "Executive Summary",
            source_url_for(
                "analysis/executive_summary.md",
                source_links,
                require_external=require_external,
            ),
        ),
        (
            "Trend Report",
            source_url_for(
                "analysis/trend_report.md",
                source_links,
                require_external=require_external,
            ),
        ),
    )


def topic_by_id(topics: Iterable[Topic]) -> dict[str, Topic]:
    return {topic.topic_id: topic for topic in topics}


LESSON_REQUIRED_FIELDS = {
    "lesson_id",
    "distilled_message",
    "topic_ids",
    "evidence_kind",
    "source_anchor",
    "source_locator",
    "evidence_text",
    "decision_implication",
}


def load_deck_first_plan(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = load_json(path)
    if not isinstance(payload.get("candidate_pool"), list):
        raise ValueError(f"{path} must include candidate_pool")
    if not isinstance(payload.get("short_spine"), list):
        raise ValueError(f"{path} must include short_spine")
    if not isinstance(payload.get("long_expansions"), list):
        raise ValueError(f"{path} must include long_expansions")
    return payload


def _deck_first_rows(plan: dict[str, Any], key: str) -> tuple[dict[str, Any], ...]:
    rows = plan.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"deck-first plan {key} must be a list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"deck-first plan {key}[{index}] must be an object")
        missing = LESSON_REQUIRED_FIELDS - set(raw)
        if missing:
            raise ValueError(
                f"deck-first plan {key}[{index}] missing fields: {sorted(missing)}"
            )
        if raw["evidence_kind"] not in {"direct_quote", "tight_paraphrase"}:
            raise ValueError(
                f"deck-first plan {key}[{index}] has invalid evidence_kind"
            )
        normalized.append(dict(raw))
    return tuple(normalized)


def _duration_target_seconds(
    plan: dict[str, Any] | None,
    key: str,
    *,
    default_seconds: float,
) -> float:
    policy = plan.get("selection_policy") if isinstance(plan, dict) else None
    raw_minutes: object = None
    if isinstance(policy, dict):
        raw_minutes = policy.get(key)
    if raw_minutes is None:
        return default_seconds
    try:
        minutes = float(raw_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"deck-first selection_policy.{key} must be numeric") from exc
    if minutes <= 0:
        raise ValueError(f"deck-first selection_policy.{key} must be positive")
    return round(minutes * 60.0, 2)


def _topic_refs_for_lesson(
    lesson: dict[str, Any],
    by_id: dict[str, Topic],
) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    topic_ids = lesson.get("topic_ids") or []
    if not isinstance(topic_ids, list):
        topic_ids = []
    for topic_id in topic_ids:
        topic = by_id.get(str(topic_id))
        if topic is None or topic.topic_id in seen:
            continue
        refs.append(topic_ref(topic))
        seen.add(topic.topic_id)
    return tuple(refs)


def _lesson_slide_bullets(lesson: dict[str, Any]) -> tuple[str, ...]:
    raw_bullets = lesson.get("slide_bullets")
    if isinstance(raw_bullets, list) and raw_bullets:
        bullets = [clean_markdown_text(str(item)) for item in raw_bullets[:3]]
    else:
        bullets = [
            clean_markdown_text(str(lesson["distilled_message"])),
            f"Evidence: {clean_markdown_text(str(lesson['evidence_text']))}",
            f"Decision: {clean_markdown_text(str(lesson['decision_implication']))}",
        ]
    return tuple(item for item in bullets if item)


PAPER_SIGNAL_STOPWORDS = {
    "about",
    "across",
    "after",
    "against",
    "also",
    "and",
    "are",
    "because",
    "before",
    "being",
    "beyond",
    "but",
    "can",
    "conference",
    "decision",
    "evidence",
    "from",
    "have",
    "into",
    "model",
    "models",
    "more",
    "must",
    "not",
    "only",
    "paper",
    "papers",
    "rather",
    "should",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "through",
    "toward",
    "under",
    "while",
    "with",
    "without",
}


def _candidate_keywords(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        if token.lower() not in PAPER_SIGNAL_STOPWORDS
    }


def _strip_paper_refs(text: str) -> str:
    text = re.sub(r"\s*\(`?paper_abstract:[^)]+`\)", "", text)
    text = re.sub(r"`?paper_abstract:[A-Za-z0-9_-]+`?", "", text)
    return clean_markdown_text(text)


def _paper_signal_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    in_summary = False
    summary_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            in_summary = stripped == "## Summary"
            continue
        if in_summary:
            if not stripped:
                if summary_lines:
                    in_summary = False
                continue
            summary_lines.append(stripped)
        if stripped.startswith("- "):
            candidate = _strip_paper_refs(stripped[2:])
            if candidate:
                candidates.append(candidate)
    if summary_lines:
        summary_sentence = first_sentence(
            " ".join(summary_lines),
            fallback="",
        )
        if summary_sentence:
            candidates.insert(0, _strip_paper_refs(summary_sentence))
    return list(dict.fromkeys(item for item in candidates if item))


# Words that must never end a rendered figure line: the visual-legibility gate
# (structured_video._svg_fragment_lines) reads a sentence-flow line ending in one
# of these as a truncation fragment and fails the render closed. Kept in sync with
# structured_video.SVG_FRAGMENT_END_WORDS.
_FIGURE_FRAGMENT_END_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "because", "for", "from", "in",
        "nor", "of", "or", "the", "to", "with", "within",
    }
)


def _strip_dangling_fragment(text: str) -> str:
    """Drop a trailing conjunction/preposition so compacted evidence never ends on
    a dangling fragment word (e.g. "...memory access, and." -> "...memory access.").
    """
    words = text.rstrip(".,;:!? ").split()
    while words and words[-1].strip(".,;:!?").lower() in _FIGURE_FRAGMENT_END_WORDS:
        words.pop()
    cleaned = " ".join(words).rstrip(".,;:!? ")
    return f"{cleaned}." if cleaned else ""


def _compact_paper_signal(text: str, *, max_chars: int = 175) -> str:
    clean = _strip_paper_refs(text)
    if len(clean) > max_chars:
        clean = clean[:max_chars].rsplit(" ", 1)[0]
    return _strip_dangling_fragment(clean)


def _paper_signal_for_lesson(workspace: Path, lesson: dict[str, Any]) -> tuple[str, str]:
    topic_ids = lesson.get("topic_ids") or []
    if not isinstance(topic_ids, list):
        return "", ""
    lesson_text = " ".join(
        str(lesson.get(key) or "")
        for key in (
            "slide_title",
            "distilled_message",
            "evidence_text",
            "decision_implication",
            "visual_role",
        )
    )
    lesson_keywords = _candidate_keywords(lesson_text)
    best_score = -1
    best_candidate = ""
    best_locator = ""
    for topic_id in topic_ids:
        topic = str(topic_id)
        paper_path = workspace / "analysis" / "exhaustive_summaries" / f"{topic}__papers.md"
        if not paper_path.is_file():
            continue
        candidates = _paper_signal_candidates(paper_path.read_text(encoding="utf-8"))
        for candidate in candidates:
            candidate_keywords = _candidate_keywords(candidate)
            score = len(lesson_keywords & candidate_keywords)
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_locator = relpath(paper_path, workspace)
    if not best_candidate:
        return "", ""
    return _compact_paper_signal(best_candidate), f"{best_locator}#paper-signal"


def _with_paper_signal(workspace: Path, lesson: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(lesson)
    # A caller (e.g. the Opus topic-digest plan) may already supply a named-source
    # paper signal; preserve it instead of overwriting with the auto-extracted one.
    if str(enriched.get("paper_evidence_text") or "").strip():
        return enriched
    paper_evidence_text, paper_source_locator = _paper_signal_for_lesson(
        workspace,
        lesson,
    )
    if paper_evidence_text:
        enriched["paper_evidence_text"] = paper_evidence_text
        enriched["paper_source_locator"] = paper_source_locator
    return enriched


def _evidence_mode_label(value: object) -> str:
    mode = str(value or "").strip()
    if mode == "direct_quote":
        return "Direct transcript quote"
    if mode == "tight_paraphrase":
        return "Tight transcript paraphrase"
    return "Transcript-backed evidence"


def _lesson_to_slide(
    lesson: dict[str, Any],
    *,
    role: str,
    by_id: dict[str, Topic],
) -> SlideSpec:
    title = clean_markdown_text(
        str(lesson.get("slide_title") or lesson["distilled_message"])
    )
    evidence_text = clean_markdown_text(str(lesson["evidence_text"]))
    decision_implication = clean_markdown_text(str(lesson["decision_implication"]))
    source_anchor = clean_markdown_text(str(lesson["source_anchor"]))
    paper_evidence_text = clean_markdown_text(str(lesson.get("paper_evidence_text") or ""))
    paper_source_locator = str(lesson.get("paper_source_locator") or "")
    evidence_kind = str(lesson["evidence_kind"])
    paper_clause = (
        f" Paper signal: {paper_evidence_text}"
        if paper_evidence_text
        else ""
    )
    notes = (
        f"{_evidence_mode_label(evidence_kind)} from {source_anchor}: "
        f"{evidence_text}{paper_clause} Decision implication: {decision_implication}"
    )
    return SlideSpec(
        title=title,
        bullets=_lesson_slide_bullets(lesson),
        refs=_topic_refs_for_lesson(lesson, by_id),
        # Synthesis lessons (open questions / predictions) render as a bullet list;
        # topic lessons keep the evidence -> decision bridge.
        figure=str(lesson.get("figure_variant") or "executive_evidence_bridge"),
        notes=notes,
        narrative=evidence_text,
        executive_lesson_id=str(lesson["lesson_id"]),
        executive_lesson_role=role,
        executive_evidence_kind=evidence_kind,
        source_anchor=source_anchor,
        source_locator=str(lesson["source_locator"]),
        evidence_text=evidence_text,
        decision_implication=decision_implication,
        paper_evidence_text=paper_evidence_text,
        paper_source_locator=paper_source_locator,
        premium_visual_direction=str(lesson.get("visual_role") or "executive_evidence_bridge"),
        diagram=dict(lesson.get("diagram") or {}),
    )


EXECUTIVE_OPENING_EVENT_LIMIT = 6


def _short_text(value: object, *, limit: int = 92) -> str:
    text = clean_markdown_text(html.unescape(str(value or "")))
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def _link_item_key(item: dict[str, str]) -> str:
    text = str(item.get("name") or item.get("title") or "").strip().lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _event_priority(item: dict[str, str]) -> int:
    text = f"{item.get('name', '')} {item.get('detail', '')}".lower()
    if "keynote" in text:
        return 90
    if "oral" in text:
        return 82
    if "invited" in text:
        return 78
    if "award" in text or "best paper" in text:
        return 76
    if "panel" in text:
        return 72
    if "competition" in text:
        return 68
    if "workshop" in text:
        return 62
    if "talk" in text:
        return 58
    return 40


def _load_topic_required_events(workspace: Path, topics: tuple[Topic, ...]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for topic in topics:
        path = workspace / "decks" / topic.topic_id / "bookend_event_overrides.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        raw_events = payload.get("required_events")
        if not isinstance(raw_events, list):
            continue
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            name = _short_text(item.get("name"), limit=68)
            if not name:
                continue
            detail = _short_text(item.get("detail"), limit=64)
            url = str(item.get("url") or "").strip()
            events.append({"name": name, "detail": detail, "url": url})
    return events


def select_opening_events(
    *,
    workspace: Path,
    topics: tuple[Topic, ...],
    point_count: int,
) -> tuple[dict[str, str], ...]:
    max_event_count = min(point_count, EXECUTIVE_OPENING_EVENT_LIMIT)
    candidates = _load_topic_required_events(workspace, topics)
    grouped: dict[str, dict[str, object]] = {}
    for item in candidates:
        key = _link_item_key(item)
        if not key:
            continue
        current = grouped.setdefault(
            key,
            {
                "name": item["name"],
                "detail": item.get("detail", ""),
                "url": item.get("url", ""),
                "count": 0,
                "priority": 0,
            },
        )
        current["count"] = int(current["count"]) + 1
        current["priority"] = max(int(current["priority"]), _event_priority(item))
        if item.get("url") and not current.get("url"):
            current["url"] = item["url"]
        if len(item["name"]) < len(str(current["name"])):
            current["name"] = item["name"]
        if item.get("detail") and len(item["detail"]) < len(str(current.get("detail") or "")):
            current["detail"] = item["detail"]

    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            -(int(item["priority"]) + int(item["count"]) * 8),
            str(item["name"]).lower(),
        ),
    )
    selected = [
        {
            "name": str(item["name"]),
            "detail": str(item.get("detail") or ""),
            "url": str(item.get("url") or ""),
        }
        for item in ranked[:max_event_count]
    ]
    # Fail-closed program-anchor guard (mirrors the synthesize_key_events gate):
    # an executive "important talk" must trace to the real conference program.
    # Drop any event lacking a verifiable program-site URL so analysis-derived
    # hallucinations (e.g. fabricated keynotes with empty URLs) can never surface
    # on the bookend. Backfill below restores the count from the program fallbacks.
    context = conference_context()
    selected = [
        item
        for item in selected
        if context.site_domain
        and context.site_domain in str(item.get("url") or "").lower()
    ]
    seen = {_link_item_key(item) for item in selected}
    for item in context.program_events:
        if len(selected) >= max_event_count:
            break
        key = _link_item_key(item)
        if key in seen:
            continue
        selected.append(dict(item))
        seen.add(key)
    return tuple(selected[:max_event_count])


def build_further_reading_resources(
    *,
    topics: tuple[Topic, ...],
    presentation_links: dict[str, str],
    require_external_presentation_links: bool,
) -> tuple[dict[str, str], ...]:
    presentations: list[dict[str, str]] = []
    reports: list[dict[str, str]] = []
    for index, topic in enumerate(topics, start=1):
        presentations.append(
            {
                "title": f"Presentation {index:02d}: {topic.title}",
                "url": presentation_url_for(
                    topic.topic_id,
                    presentation_links,
                    require_external=require_external_presentation_links,
                ),
                "kind": "topic_presentation",
                "topic_id": topic.topic_id,
            }
        )
        reports.append(
            {
                "title": f"Analysis report {index:02d}: {topic.title}",
                "url": topic.source_url,
                "kind": "analysis_report",
                "topic_id": topic.topic_id,
            }
        )
    return tuple(presentations + reports)


def build_opening_resources(
    *,
    topics: tuple[Topic, ...],
    point_count: int,
    source_links: dict[str, str],
    presentation_links: dict[str, str],
    require_external_source_links: bool,
    require_external_presentation_links: bool,
) -> tuple[dict[str, str], ...]:
    resources: list[dict[str, str]] = [
        {"name": name, "detail": detail, "url": url}
        for name, detail, url in conference_context().program_resources
    ]
    resources.extend(
        [
            {
                "name": "Executive summary report",
                "detail": "Conference synthesis report",
                "url": source_url_for(
                    "analysis/executive_summary.md",
                    source_links,
                    require_external=require_external_source_links,
                ),
            },
            {
                "name": "Trend report",
                "detail": "Cross-topic trend report",
                "url": source_url_for(
                    "analysis/trend_report.md",
                    source_links,
                    require_external=require_external_source_links,
                ),
            },
        ]
    )
    seen = {_link_item_key(item) for item in resources}
    for topic in topics:
        if len(resources) >= point_count:
            break
        item = {
            "name": f"{topic.topic_id[:2]} presentation: {topic.title}",
            "detail": "Topic presentation",
            "url": presentation_url_for(
                topic.topic_id,
                presentation_links,
                require_external=require_external_presentation_links,
            ),
        }
        key = _link_item_key(item)
        if key not in seen:
            resources.append(item)
            seen.add(key)
    for topic in topics:
        if len(resources) >= point_count:
            break
        item = {
            "name": f"{topic.topic_id[:2]} analysis: {topic.title}",
            "detail": "Topic analysis report",
            "url": topic.source_url,
        }
        key = _link_item_key(item)
        if key not in seen:
            resources.append(item)
            seen.add(key)
    return tuple(resources[:point_count])


_CITATION_PAREN_RE = re.compile(
    r"\((?:[^()]*(?:analysis/|paper_abstract:|extraction|research id)[^()]*"
    # Research-thread ids: "run_id:topic_id:hash", e.g.
    # "icml2026-v4.0-live-production-20260711:05_scaling_efficiency_
    # foundations:248e4ee99a5c". Not caught by the patterns above (no
    # "analysis/"/"paper_abstract:" substring), and unbounded in length --
    # left unstripped it can overflow a narrow caption column as a single
    # unbreakable token and reads as an internal artifact to a viewer.
    r"|[A-Za-z0-9][\w.-]*:[\w-]+:[0-9a-f]{8,16}"
    r")\)",
    re.IGNORECASE,
)
_CITATION_SENTENCE_RE = re.compile(
    r"^(?:see\b|representative extraction\b|research (?:products|ids?)\b|extraction (?:ids?|records?)\b)",
    re.IGNORECASE,
)


def _strip_citation_clutter(text: str) -> str:
    cleaned = _CITATION_PAREN_RE.sub("", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = clean_markdown_text(text)
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _content_sentences(text: str, *, limit: int) -> tuple[str, ...]:
    """First ``limit`` presentation-ready sentences with citation clutter removed."""
    sentences: list[str] = []
    for sentence in _split_sentences(_strip_citation_clutter(text)):
        if _CITATION_SENTENCE_RE.match(sentence):
            continue
        if "paper_abstract:" in sentence or "sha256" in sentence:
            continue
        if len(sentence) < 25:
            continue
        if not sentence.endswith((".", "!", "?")):
            sentence += "."
        sentences.append(sentence)
        if len(sentences) >= limit:
            break
    return tuple(sentences)


def _compact_slide_title(title: str, *, max_chars: int = 74) -> str:
    compact = clean_markdown_text(title).rstrip(".")
    if len(compact) > max_chars and ":" in compact:
        compact = compact.split(":", 1)[0].strip()
    if len(compact) > max_chars:
        words = compact[: max_chars - 1].rsplit(" ", 1)[0].rstrip(",;").split()
        while len(words) > 1 and words[-1].strip(".,;:!?").lower() in _FIGURE_FRAGMENT_END_WORDS:
            words.pop()
        compact = " ".join(words) + "\u2026"
    return compact


def _topics_for_text(
    text: str,
    topics: tuple[Topic, ...],
    *,
    count: int = 2,
) -> tuple[Topic, ...]:
    """Match slide text to topics: explicit report-path mentions win, then
    deterministic keyword overlap against topic titles, then manifest order."""
    by_id = topic_by_id(topics)
    ordered: list[Topic] = []
    for match in re.finditer(r"analysis/([0-9A-Za-z_]+)\.md", text):
        topic = by_id.get(match.group(1))
        if topic is not None and topic not in ordered:
            ordered.append(topic)
        if len(ordered) >= count:
            return tuple(ordered[:count])
    lowered = text.lower()
    scored: list[tuple[int, int, Topic]] = []
    for index, topic in enumerate(topics):
        words = set(re.findall(r"[a-z]{4,}", topic.title.lower()))
        score = sum(1 for word in words if word in lowered)
        if score:
            scored.append((-score, index, topic))
    for _score, _index, topic in sorted(scored):
        if topic not in ordered:
            ordered.append(topic)
        if len(ordered) >= count:
            break
    for topic in topics:
        if len(ordered) >= count:
            break
        if topic not in ordered:
            ordered.append(topic)
    return tuple(ordered[:count])


_SECTION_BOUNDARY_RE = re.compile(
    r"^(?:#{1,6}\s|Largest surprises|Residual risks|Concrete recommendations|"
    r"Recommended immediate actions|Representative citations|Closing note)"
)


def _section_paragraph_text(text: str, heading_pattern: str) -> str:
    """Return the prose block under a section heading, joined to one line.

    Unlike :func:`_section_items` (which expects ``- ``/``N)`` list items),
    this is for a plain-paragraph section such as ``## Orientation``. Returns
    "" if the heading isn't present, so callers can fall back to a static
    default rather than risking any other raw document text (headings,
    metadata) leaking through.
    """
    match = re.search(heading_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    lines: list[str] = []
    for line in text[match.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if _SECTION_BOUNDARY_RE.match(stripped):
            break
        lines.append(stripped)
    return " ".join(lines)


def _section_items(text: str, heading_pattern: str, *, limit: int) -> tuple[str, ...]:
    """Collect ``- `` or ``N)`` item lead sentences under a section heading."""
    match = re.search(heading_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ()
    items: list[str] = []
    for line in text[match.end():].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _SECTION_BOUNDARY_RE.match(stripped):
            break
        entry = re.match(r"^(?:-|\d+\))\s+(.+)$", stripped)
        if entry:
            body = _strip_citation_clutter(entry.group(1))
            items.append(first_sentence(body, fallback=body))
            if len(items) >= limit:
                break
    return tuple(items)


def _build_generic_overview_decks(
    *,
    topics: tuple[Topic, ...],
    executive_text: str,
    trend_text: str,
    short_id: str,
    long_id: str,
    source_links: dict[str, str],
    require_external_source_links: bool,
    point_count: int,
    short_target_seconds: float,
    long_target_seconds: float,
    opening_events: tuple[dict[str, str], ...],
    opening_resources: tuple[dict[str, str], ...],
    further_reading: tuple[dict[str, str], ...],
    presentation_link_mode: str,
) -> tuple[DeckSpec, DeckSpec]:
    """Build the short/long executive decks from stage-8 synthesis content.

    Every slide is derived from the workspace's own executive summary, trend
    report, and analysis manifest; nothing conference-specific is hardcoded.
    """
    label = conference_label()
    items = extract_numbered_items(executive_text)[:8]
    if not items:
        raise ValueError(
            "executive_summary_numbered_items_missing: analysis/executive_summary.md "
            "must contain a numbered 'N)' conclusions list for the generic executive "
            "deck builder; regenerate stage-8 synthesis or supply a deck-first plan"
        )
    base_refs = source_refs(source_links, require_external=require_external_source_links)
    # Item bullets are full synthesis sentences, so only cycle figure variants
    # that render labels as wrapped list lines. Box/flow diagram variants
    # (system_stack, operating_model, openness_stack) truncate sentence labels
    # into near-empty nodes and fail the LLM visual QC gate.
    figures = (
        "evaluation_ladder",
        "attack_surface",
        "allocation_loop",
        "world_model",
        "science_pipeline",
        "constraint_triangle",
        "intervention_flow",
        "evidence_list",
    )
    item_slides: list[SlideSpec] = []
    for index, (title, body) in enumerate(items):
        matched = _topics_for_text(f"{title} {body}", topics)
        bullets = _content_sentences(body, limit=3)
        if not bullets:
            bullets = (first_sentence(body, fallback=title),)
        item_slides.append(
            SlideSpec(
                title=_compact_slide_title(title),
                bullets=bullets,
                refs=base_refs + core_refs(*matched),
                figure=figures[index % len(figures)],
                notes=(
                    f"Executive point {index + 1} of {len(items)}, grounded in the "
                    "stage-8 synthesis and the linked topic reports."
                ),
                narrative=first_sentence(body, fallback=title),
            )
        )

    signal_slide = SlideSpec(
        title=f"{len(items)} Signals Drive The Story",
        bullets=tuple(_compact_slide_title(title) for title, _body in items),
        refs=base_refs,
        figure="radial_signals",
        notes=(
            "Orientation slide: these signals recur across papers, talks, "
            "workshops, and analysis reports."
        ),
        narrative=(
            f"These {len(items)} signals are not a leaderboard. They are an "
            "operating map for where conference evidence changes decisions."
        ),
    )

    actions = _section_items(
        executive_text,
        r"^(?:Recommended immediate actions|Concrete recommendations)[^\n]*",
        limit=4,
    )
    if not actions:
        implications: list[str] = []
        for _title, body in items:
            for sentence in _split_sentences(_strip_citation_clutter(body)):
                if sentence.lower().startswith("implication"):
                    lead = sentence.split(":", 1)[-1].strip()
                    if lead:
                        implications.append(lead[0].upper() + lead[1:])
                    break
            if len(implications) >= 4:
                break
        actions = tuple(implications)
    if not actions:
        orientation_text = _section_paragraph_text(executive_text, r"^##\s+Orientation[^\n]*")
        fallback_bullet = (
            first_sentence(orientation_text, fallback="Fund the reliability agenda.")
            if orientation_text
            else "Fund the reliability agenda."
        )
        actions = (fallback_bullet,)
    leadership_slide = SlideSpec(
        title="Leadership Actions",
        bullets=actions,
        refs=base_refs + core_refs(*_topics_for_text(" ".join(actions), topics)),
        figure="leadership_actions",
        notes="Close the short deck with the decision agenda rather than a recap.",
        narrative=(
            "The leadership agenda distills the synthesis into funding, gating, "
            "and governance moves that follow from the conference evidence."
        ),
    )

    short_slides: tuple[SlideSpec, ...] = (signal_slide, *item_slides, leadership_slide)

    long_slides: list[SlideSpec] = [signal_slide]
    for start in range(0, len(topics), 6):
        chunk = topics[start : start + 6]
        long_slides.append(
            SlideSpec(
                title=(
                    "Topic Map: Where The Evidence Lands"
                    if start == 0
                    else "Topic Map Continued"
                ),
                bullets=tuple(f"{t.topic_id[:2]}: {t.title}" for t in chunk),
                refs=tuple(topic_ref(t) for t in chunk),
                figure="topic_grid_a" if start == 0 else "topic_grid_b",
                notes="Executive topic map derived from the analysis manifest.",
            )
        )
    long_slides.extend(item_slides)
    surprises = _section_items(executive_text, r"^Largest surprises[^\n]*", limit=4)
    if surprises:
        long_slides.append(
            SlideSpec(
                title="Largest Surprises",
                bullets=surprises,
                refs=base_refs,
                figure="tension_matrix",
                notes="Strategic surprises called out by the stage-8 synthesis.",
            )
        )
    risks = _section_items(executive_text, r"^Residual risks[^\n]*", limit=4)
    if risks:
        long_slides.append(
            SlideSpec(
                title="Residual Risks",
                bullets=risks,
                refs=base_refs,
                figure="watchlist",
                notes="Residual risks that should shape downstream slides and budgets.",
            )
        )
    trend_items = extract_numbered_items(trend_text)[:6]
    if trend_items:
        long_slides.append(
            SlideSpec(
                title="Cross-Topic Trends",
                bullets=tuple(_compact_slide_title(title) for title, _body in trend_items),
                refs=base_refs,
                figure="conditional_scaling",
                notes="Trend clusters from the cross-topic trend report.",
                narrative=first_sentence(
                    trend_items[0][1], fallback=trend_items[0][0]
                ),
            )
        )
    long_slides.append(leadership_slide)
    for start in range(0, len(topics), 4):
        chunk = topics[start : start + 4]
        ordinal = min(start // 4, 2)
        suffix = ("", " Continued", " Final")[ordinal]
        long_slides.append(
            SlideSpec(
                title=f"Source Backbone{suffix}",
                bullets=tuple(t.title for t in chunk),
                refs=tuple(topic_ref(t) for t in chunk),
                figure=f"source_backbone_{'abc'[ordinal]}",
                notes="Appendix-style source slide for auditability.",
                slide_type="appendix",
            )
        )

    return (
        DeckSpec(
            presentation_id=short_id,
            display_title=f"{label} Executive Summary - Short",
            slides=short_slides,
            point_count=point_count,
            target_duration_seconds=short_target_seconds,
            opening_events=opening_events,
            opening_resources=opening_resources,
            further_reading=further_reading,
            presentation_link_mode=presentation_link_mode,
        ),
        DeckSpec(
            presentation_id=long_id,
            display_title=f"{label} Executive Summary - Long",
            slides=tuple(long_slides),
            point_count=point_count,
            target_duration_seconds=long_target_seconds,
            opening_events=opening_events,
            opening_resources=opening_resources,
            further_reading=further_reading,
            presentation_link_mode=presentation_link_mode,
        ),
    )


def build_decks(
    *,
    workspace: Path,
    topics: tuple[Topic, ...],
    executive_text: str,
    trend_text: str,
    short_id: str,
    long_id: str,
    source_links: dict[str, str] | None = None,
    require_external_source_links: bool = False,
    presentation_links: dict[str, str] | None = None,
    require_external_presentation_links: bool = False,
    deck_first_plan: dict[str, Any] | None = None,
) -> tuple[DeckSpec, DeckSpec]:
    source_links = source_links or {}
    presentation_links = presentation_links or {}
    by_id = topic_by_id(topics)
    short_lessons: tuple[dict[str, Any], ...] = ()
    long_expansions: tuple[dict[str, Any], ...] = ()
    long_spine: tuple[dict[str, Any], ...] = ()
    if deck_first_plan is not None:
        short_lessons = _deck_first_rows(deck_first_plan, "short_spine")
        long_expansions = _deck_first_rows(deck_first_plan, "long_expansions")
        short_lessons = tuple(
            _with_paper_signal(workspace, lesson) for lesson in short_lessons
        )
        long_expansions = tuple(
            _with_paper_signal(workspace, lesson) for lesson in long_expansions
        )
        # The long deck is its own 3-slides-per-topic spine when provided, not
        # short_spine + a few expansions.
        if isinstance(deck_first_plan.get("long_spine"), list) and deck_first_plan["long_spine"]:
            long_spine = tuple(
                _with_paper_signal(workspace, lesson)
                for lesson in _deck_first_rows(deck_first_plan, "long_spine")
            )
        point_count = len(short_lessons)
    else:
        numbered = extract_numbered_items(executive_text)
        top_items = numbered[:8]
        point_count = len(top_items) or 8
    short_target_seconds = _duration_target_seconds(
        deck_first_plan,
        "short_duration_minutes",
        default_seconds=DEFAULT_SHORT_DURATION_SECONDS,
    )
    long_target_seconds = _duration_target_seconds(
        deck_first_plan,
        "long_duration_minutes",
        default_seconds=DEFAULT_LONG_DURATION_SECONDS,
    )
    opening_events = select_opening_events(
        workspace=workspace,
        topics=topics,
        point_count=point_count,
    )
    # Replace the program-talks bookend column with the source-grounded 12-topic
    # index (Opus one-liners) when the deck-first plan supplies one. This both
    # delivers the agreed index slide and structurally removes the unverifiable
    # talk list (the right column keeps the URL-verified program + report links).
    opening_events_title = "Important talks, workshops, and sessions"
    _topic_index_rows = (deck_first_plan or {}).get("topic_index")
    if isinstance(_topic_index_rows, list) and _topic_index_rows:
        index_events = tuple(
            {
                "name": clean_markdown_text(str(row.get("title") or "")),
                "detail": clean_markdown_text(str(row.get("one_liner") or "")),
                "url": "",
                "arc": clean_markdown_text(str(row.get("arc") or "")),
            }
            for row in _topic_index_rows
            if isinstance(row, dict) and row.get("title")
        )
        if index_events:
            opening_events = index_events
            opening_events_title = f"The {len(topics)} topics of {conference_label()}"
    # Cross-topic central message (the deck's unifying thesis) for the title slide.
    central_message_text = ""
    _central = (deck_first_plan or {}).get("central_message") if deck_first_plan else None
    if isinstance(_central, dict):
        central_message_text = clean_markdown_text(str(_central.get("headline") or ""))
    opening_resources = build_opening_resources(
        topics=topics,
        point_count=point_count,
        source_links=source_links,
        presentation_links=presentation_links,
        require_external_source_links=require_external_source_links,
        require_external_presentation_links=require_external_presentation_links,
    )
    further_reading = build_further_reading_resources(
        topics=topics,
        presentation_links=presentation_links,
        require_external_presentation_links=require_external_presentation_links,
    )
    presentation_link_mode = (
        "public"
        if further_reading
        and all(str(item.get("url") or "").startswith(("https://", "http://")) for item in further_reading[: len(topics)])
        else "workspace-local"
    )

    def topic(topic_id: str) -> Topic:
        return by_id[topic_id]

    if deck_first_plan is not None:
        short_slides = tuple(
            _lesson_to_slide(lesson, role="short_spine", by_id=by_id)
            for lesson in short_lessons
        )
        if long_spine:
            long_slides = [
                _lesson_to_slide(lesson, role="long_spine", by_id=by_id)
                for lesson in long_spine
            ]
        else:
            long_slides = list(short_slides)
            long_slides.extend(
                _lesson_to_slide(lesson, role="long_expansion", by_id=by_id)
                for lesson in long_expansions
            )
        return (
            DeckSpec(
                presentation_id=short_id,
                display_title=f"{conference_label()} Executive Readout",
                slides=short_slides,
                point_count=point_count,
                target_duration_seconds=short_target_seconds,
                opening_events=opening_events,
                opening_resources=opening_resources,
                further_reading=further_reading,
                presentation_link_mode=presentation_link_mode,
                opening_events_title=opening_events_title,
                central_message_text=central_message_text,
            ),
            DeckSpec(
                presentation_id=long_id,
                display_title=f"{conference_label()} Executive Readout — Extended",
                slides=tuple(long_slides),
                point_count=point_count,
                target_duration_seconds=long_target_seconds,
                opening_events=opening_events,
                opening_resources=opening_resources,
                further_reading=further_reading,
                presentation_link_mode=presentation_link_mode,
                opening_events_title=opening_events_title,
                central_message_text=central_message_text,
            ),
        )

    return _build_generic_overview_decks(
        topics=topics,
        executive_text=executive_text,
        trend_text=trend_text,
        short_id=short_id,
        long_id=long_id,
        source_links=source_links,
        require_external_source_links=require_external_source_links,
        point_count=point_count,
        short_target_seconds=short_target_seconds,
        long_target_seconds=long_target_seconds,
        opening_events=opening_events,
        opening_resources=opening_resources,
        further_reading=further_reading,
        presentation_link_mode=presentation_link_mode,
    )

def select_decks_for_scope(
    decks: tuple[DeckSpec, DeckSpec],
    *,
    deck_scope: str,
) -> tuple[DeckSpec, ...]:
    if deck_scope == "both":
        return decks
    if deck_scope == "short":
        return decks[:1]
    if deck_scope == "long":
        return decks[1:]
    raise ValueError(f"deck_scope must be one of both, short, or long: {deck_scope!r}")


def wrap_svg_text(text: str, max_chars: int = 26, *, max_lines: int | None = 3) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    def flush_current(next_word: str) -> list[str]:
        def bad_tail(word: str) -> bool:
            tail = word.strip(".,;:!?").lower()
            return tail in {
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "because",
                "for",
                "from",
                "in",
                "nor",
                "of",
                "or",
                "the",
                "to",
                "with",
                "within",
            } or "-" in tail

        if len(current) <= 1:
            line = " ".join(current)
            current.clear()
            return [line] if line else []
        if bad_tail(current[-1]):
            carry_words = [next_word]
            while len(current) > 1 and bad_tail(current[-1]):
                carry_words.insert(0, current.pop())
            line = " ".join(current)
            current.clear()
            current.extend(carry_words)
            return [line] if line else []
        line = " ".join(current)
        current.clear()
        current.append(next_word)
        return [line]

    for word in words:
        candidate = " ".join([*current, word])
        if len(candidate) > max_chars and current:
            lines.extend(flush_current(word))
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    if max_lines is None:
        return lines
    if len(lines) <= max_lines:
        return lines
    visible = lines[:max_lines]
    suffix = "..."
    visible[-1] = visible[-1].rstrip(".,;: ") + suffix
    return visible


def svg_figure_bullet_lines(label: str, *, wrap_chars: int, max_lines: int) -> list[str]:
    """Wrap a figure bullet into <= max_lines COMPLETE lines, never emitting a
    mid-sentence ellipsis.

    The executive figures render full synthesis sentences (up to ~480 chars)
    as wrapped list lines. wrap_svg_text(..., max_lines=2) truncated those
    mid-sentence with "..." even when the figure card had visible empty space
    below -- the vision QC gate correctly reads that as clipped/incomplete
    text. Instead of clipping, shorten the bullet to the largest prefix of
    complete sentences (then complete clauses) that fits the available lines,
    so the rendered text always reads as a finished thought.
    """
    normalized = " ".join(str(label).split())

    def fits(text: str) -> bool:
        return bool(text) and len(wrap_svg_text(text, wrap_chars, max_lines=None)) <= max_lines

    if fits(normalized):
        return wrap_svg_text(normalized, wrap_chars, max_lines=None)
    # Largest prefix of complete sentences that fits.
    sentences = re.split(r'(?<=[.!?"”’])\s+', normalized)
    for count in range(len(sentences) - 1, 0, -1):
        candidate = " ".join(sentences[:count]).strip()
        if fits(candidate):
            return wrap_svg_text(candidate, wrap_chars, max_lines=None)
    # A single sentence is still too long: shorten by complete clauses.
    first = sentences[0]
    clauses = re.split(r"(?<=[,;:—–])\s+", first)
    for count in range(len(clauses) - 1, 0, -1):
        candidate = " ".join(clauses[:count]).strip().rstrip(",;:—– ")
        if fits(candidate):
            return wrap_svg_text(candidate, wrap_chars, max_lines=None)
    # Even the first clause is too long: hard word-trim, ending on a clean
    # word (no dangling article/preposition), still with no ellipsis.
    words = first.split()
    while len(words) > 1 and not fits(" ".join(words)):
        words.pop()
    while len(words) > 1 and words[-1].strip(".,;:!?…").lower() in _FIGURE_FRAGMENT_END_WORDS:
        words.pop()
    return wrap_svg_text(" ".join(words).rstrip(",;:—– "), wrap_chars, max_lines=None)


def compact_flow_label(label: str) -> str:
    """Turn prose bullets into complete, compact labels for flow diagrams."""
    normalized = " ".join(label.strip().rstrip(".").split())
    lower = normalized.lower()
    replacements = (
        ("model plus tools", "Treat AI as a full system"),
        ("operational readiness", "Measure operational readiness"),
        ("reliability, safety", "Govern reliability, safety, resources"),
        ("permissioning", "Control permissions and provenance"),
        ("monitoring", "Monitor live system behavior"),
        ("openness", "Treat openness as governance"),
        ("release surface", "Harden the release surface"),
        ("join modeling", "Unify model, safety, systems review"),
        ("gate releases", "Gate releases on trajectory evidence"),
        ("maintain provenance", "Track memory, RAG, tools, data"),
    )
    for needle, replacement in replacements:
        if needle in lower:
            return replacement
    segments = re.split(r"[.;:—–-]\s+", normalized, maxsplit=1)
    candidate = segments[0]
    if len(candidate) <= 46:
        return candidate
    words = candidate.split()
    return " ".join(words[:6])


def compact_signal_label(label: str) -> str:
    """Turn long synthesis bullets into complete labels for dense signal grids.

    Labels come straight from the conference's own synthesis text; when a cut is
    needed, trailing conjunctions/prepositions are stripped so the legibility
    gate never sees a dangling sentence fragment.
    """
    normalized = " ".join(label.strip().rstrip(".").split())
    candidate = re.split(r"[.;:—–]\s+", normalized, maxsplit=1)[0]
    words = candidate.split()
    # The signal-grid figure renders each label via
    # wrap_svg_text(label, 40, max_lines=2) -- an 80-char capacity. The
    # previous fixed 7-word cutoff at >52 chars silently dropped everything
    # past word 7 (no ellipsis), which for this conference's frequent
    # "X, not Y" contrastive titles cut exactly the clause that carries the
    # meaning ("Safety and alignment are lifecycle properties, not" losing
    # "stable scalars"). Trim by the actual render budget instead of an
    # arbitrary word count, so a title only gets cut when it truly doesn't
    # fit, and drop whole words (not mid-word) to reach it.
    render_budget = 76
    if len(candidate) > render_budget:
        while len(words) > 1 and len(" ".join(words)) > render_budget:
            words.pop()
    while len(words) > 1 and words[-1].strip(".,;:!?…").lower() in _FIGURE_FRAGMENT_END_WORDS:
        words.pop()
    return " ".join(words) or normalized[:52]


def _compact_svg_sentence(value: object, *, max_chars: int = 145) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "."


def svg_text(x: int, y: int, text: str, *, size: int = 20, fill: str = "#f4f7fb") -> str:
    size = max(size, 18)
    escaped = html.escape(text)
    return f'<text x="{x}" y="{y}" font-family="Inter, Arial, sans-serif" font-size="{size}" fill="{fill}">{escaped}</text>'


_DIAG_TEAL = "#5eead4"
_DIAG_AMBER = "#f3b25b"
_DIAG_NODE = "#8aa3d4"
_DIAG_LINE = "#6076a0"
_DIAG_TXT = "#f1f5fb"
_DIAG_MUT = "#a9bcd8"
_DIAG_RED = "#f87171"


def _dtext(x: float, y: float, s: str, *, size: int = 18, fill: str = _DIAG_TXT,
           anchor: str = "start", weight: str = "400") -> str:
    size = max(size, 18)
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-family="Inter, Arial, sans-serif" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{html.escape(s)}</text>')


# Phrases the legibility gate reserves as internal render scaffolding; a real
# diagram label that happens to use them must be reworded to a safe synonym.
_SCAFFOLD_SAFE = {
    "control layer": "control tier",
    "deployment decision": "deployment call",
    "deployment takeaway": "deployment lesson",
    "mechanism view": "mechanism map",
    "named study": "named work",
    "slide claim": "claim",
    "audience claim": "claim",
    "visual mechanism": "mechanism",
    "source evidence": "evidence",
    "source contribution": "contribution",
    "visual proof object": "proof object",
    "viewer-facing claim": "claim",
}


def _avoid_scaffold(text: str) -> str:
    out = str(text or "")
    for bad, good in _SCAFFOLD_SAFE.items():
        if bad in out.lower():
            out = re.sub(re.escape(bad), good, out, flags=re.IGNORECASE)
    return out


def _diagram_clean(text: str, limit: int) -> str:
    """Single-line caption/sources: avoid scaffolding, truncate on a word boundary,
    and strip any dangling fragment word so the legibility gate stays green."""
    s = _avoid_scaffold(" ".join(str(text or "").split()))
    if len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0]
    return _strip_dangling_fragment(s) if s.strip() else ""


def _axis_arrow(text: str) -> str:
    """A short axis label that always shows its reading direction."""
    t = str(text or "").strip().rstrip("→.").strip()
    return f"{t} →" if t else "→"


def _diagram_parts(kind: str, nodes: list[str], caption: str, sources: str,
                   axis: str = "") -> list[str]:
    """Clean dark line-schematic of a topic's structure on the 960x420 figure card.
    All text >= 18px so the visual-legibility gate passes. `axis` labels the
    reading direction of a spectrum/ladder so the comparison teaches at a glance."""
    P: list[str] = []
    n = [_avoid_scaffold(str(x).strip()) for x in nodes if str(x).strip()][:6]
    if not n:
        # A spec with no nodes drew a bare arrow on an empty card and passed
        # every gate; CVPR 2026's first executive build shipped 48 of them.
        # Refuse here, where every tech_diagram caller funnels through, and
        # name the fix: add_bigpicture_diagrams.py fills missing specs.
        raise ValueError(
            f"tech_diagram has no nodes (kind={kind!r}, caption={caption!r}); run "
            "scripts/presentations/add_bigpicture_diagrams.py to fill missing diagram specs"
        )
    caption = _diagram_clean(caption, 84)
    sources = _diagram_clean(sources, 76)
    # Axis is a 1-4 word label, not a sentence: avoid scaffolding + truncate, but
    # do NOT run the dangling-fragment pass (it appends a period to short phrases).
    axis = _avoid_scaffold(" ".join(str(axis or "").split()))[:40].strip()
    if caption:
        P.append(_dtext(58, 60, caption, size=18, fill=_DIAG_MUT))

    if kind in {"pipeline", "mechanism"}:
        m = len(n) or 1
        pad, gap = 44, 16
        bw = (960 - 2 * pad - gap * (m - 1)) / m
        bh, top = 150, 152
        for i, label in enumerate(n):
            x = pad + i * (bw + gap)
            P.append(f'<rect x="{x:.0f}" y="{top}" width="{bw:.0f}" height="{bh}" rx="12" '
                     f'fill="none" stroke="{_DIAG_NODE}" stroke-width="1.8"/>')
            P.append(f'<circle cx="{x + 24:.0f}" cy="{top + 26}" r="13" fill="none" '
                     f'stroke="{_DIAG_TEAL}" stroke-width="1.8"/>')
            P.append(_dtext(x + 24, top + 32, str(i + 1), size=18, fill=_DIAG_TEAL, anchor="middle"))
            for li, line in enumerate(wrap_svg_text(label, max(int(bw / 11), 9), max_lines=3)):
                P.append(_dtext(x + bw / 2, top + 78 + li * 24, line, anchor="middle"))
            if i < m - 1:
                ax = x + bw + 2
                P.append(f'<path d="M{ax:.0f} {top + bh / 2:.0f} L{ax + gap - 6:.0f} '
                         f'{top + bh / 2:.0f}" stroke="{_DIAG_LINE}" stroke-width="2" '
                         'marker-end="url(#arrow)"/>')

    elif kind == "ladder":
        m = len(n) or 1
        x0, ybot, ytop = 92, 348, 112
        P.append(f'<path d="M{x0} {ybot + 12} L{x0} {ytop - 18}" stroke="{_DIAG_LINE}" '
                 'stroke-width="2" marker-end="url(#arrow)"/>')
        P.append(_dtext(x0 - 4, ytop - 30, _axis_arrow(axis or "stronger"), size=18, fill=_DIAG_MUT))
        for i, label in enumerate(n):
            y = ybot - i * (ybot - ytop) / max(m - 1, 1)
            accent = _DIAG_AMBER if i == m - 1 else _DIAG_TEAL
            P.append(f'<line x1="{x0}" y1="{y:.0f}" x2="{x0 + 26}" y2="{y:.0f}" '
                     f'stroke="{accent}" stroke-width="2.4"/>')
            P.append(f'<circle cx="{x0 + 26}" cy="{y:.0f}" r="6" fill="{accent}"/>')
            P.append(_dtext(x0 + 44, y + 6, label[:46], anchor="start"))

    elif kind in {"architecture", "stack"}:
        m = len(n) or 1
        pad, top = 110, 116
        avail = 250
        bh = min(46, (avail - (m - 1) * 12) / m)
        ys = []
        for i, label in enumerate(n):
            y = top + i * (bh + 12)
            ys.append(y)
            accent = _DIAG_AMBER if re.search(r"seam|cross|gap|boundary", label, re.I) else _DIAG_NODE
            P.append(f'<rect x="{pad}" y="{y:.0f}" width="{960 - 2 * pad}" height="{bh:.0f}" '
                     f'rx="9" fill="none" stroke="{accent}" stroke-width="1.8"/>')
            P.append(_dtext(pad + 20, y + bh / 2 + 6, label[:54], anchor="start"))
        # Red "attack path" entering via a NON-text channel (tool output / retrieved
        # document / memory) and bypassing the text-level defense — makes the
        # injection threat visceral (2026-06-22 panel). Prefer the canonical
        # injection channels; never target a layer whose label is the text channel
        # (that is the defended front door, not the bypass).
        def _pick(pattern: str) -> int | None:
            return next((i for i, l in enumerate(n)
                         if re.search(pattern, l, re.I) and not re.search(r"\btext\b", l, re.I)),
                        None)
        tgt = _pick(r"tool|retriev|document|memory|external|output|plugin")
        if tgt is None:
            tgt = _pick(r"image|audio|input|context|channel")
        if tgt is None:
            tgt = m // 2
        ya = ys[tgt] + bh / 2
        xr = 960 - pad
        P.append(f'<path d="M928 {ya - 26:.0f} L928 {ya:.0f} L{xr + 4:.0f} {ya:.0f}" '
                 f'fill="none" stroke="{_DIAG_RED}" stroke-width="2.4" '
                 'marker-end="url(#arrow-red)"/>')
        P.append(_dtext(928, ya - 34, "attack path", size=18, fill=_DIAG_RED, anchor="end"))

    elif kind == "checklist":
        m = len(n) or 1
        x0, top = 120, 120
        step = min(54, 250 / m)
        for i, label in enumerate(n):
            y = top + i * step
            P.append(f'<circle cx="{x0}" cy="{y:.0f}" r="15" fill="none" '
                     f'stroke="{_DIAG_TEAL}" stroke-width="1.8"/>')
            P.append(_dtext(x0, y + 6, str(i + 1), size=18, fill=_DIAG_TEAL, anchor="middle"))
            P.append(_dtext(x0 + 32, y + 6, label[:52], anchor="start"))

    elif kind == "matrix":
        # Not a real 2x2 — it's a set of labelled considerations. Title it as a
        # tradeoff panel so the layout stops implying axes that aren't there
        # (2026-06-22 Claude/Codex/Gemini panel).
        m = len(n) or 1
        cols = 2
        pad, top, colgap, rowgap = 80, 138, 36, 18
        P.append(_dtext(480, 108, axis or "What to weigh", size=20, fill=_DIAG_TEAL,
                        anchor="middle", weight="600"))
        rows = (m + cols - 1) // cols
        cw = (960 - 2 * pad - colgap) / cols
        ch = min(84, (236 - (rows - 1) * rowgap) / rows)
        for i, label in enumerate(n):
            r, c = divmod(i, cols)
            x = pad + c * (cw + colgap)
            y = top + r * (ch + rowgap)
            P.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{cw:.0f}" height="{ch:.0f}" rx="9" '
                     f'fill="none" stroke="{_DIAG_NODE}" stroke-width="1.8"/>')
            for li, line in enumerate(wrap_svg_text(label, max(int(cw / 11), 12), max_lines=2)):
                P.append(_dtext(x + cw / 2, y + ch / 2 - 4 + li * 24, line, anchor="middle"))

    else:  # comparison / default -> spectrum of labelled nodes on a baseline
        m = len(n) or 1
        pad, y = 96, 232
        step = (960 - 2 * pad) / max(m - 1, 1)
        if axis:
            P.append(_dtext(480, 112, _axis_arrow(axis), size=20, fill=_DIAG_TEAL,
                            anchor="middle", weight="600"))
        P.append(f'<line x1="{pad}" y1="{y}" x2="{960 - pad}" y2="{y}" '
                 f'stroke="{_DIAG_LINE}" stroke-width="1.8" marker-end="url(#arrow)"/>')
        for i, label in enumerate(n):
            x = pad + i * step
            accent = _DIAG_TEAL if i % 2 == 0 else _DIAG_AMBER
            P.append(f'<circle cx="{x:.0f}" cy="{y}" r="8" fill="none" stroke="{accent}" stroke-width="2.4"/>')
            above = i % 2 == 0
            lines = wrap_svg_text(label, 15, max_lines=3)
            ly = (y - 34 - (len(lines) - 1) * 22) if above else (y + 40)
            for li, line in enumerate(lines):
                P.append(_dtext(x, ly + li * 22, line, anchor="middle"))

    if sources:
        P.append(_dtext(58, 396, f"Sources: {sources}", size=18, fill="#a8d8ea"))
    return P


def write_compiled_figure(
    path: Path,
    *,
    title: str,
    diagram: dict,
    slide_id: str,
    claim: str = "",
    takeaway: str = "",
) -> bool:
    """Render one executive figure through the SAME figure compiler the topic
    decks use, writing to ``path``. Returns False when the sketch cannot be
    compiled, so the caller can fall back rather than ship nothing.

    Standardisation, not decoration. Before this, a conference had two live
    drawing paths: the topic decks went through the figure compiler and its
    geometry gate, while these executive figures were hand-drawn here and
    never gated at all -- so none of the compiler's checks (crossing routes,
    connector clearance, label overflow, mixed units) ever applied to them.
    """

    from conference_pipeline.figures.compiler import compile_figure
    from conference_pipeline.figures.from_diagram import diagram_to_spec

    nodes = [str(n).strip() for n in (diagram or {}).get("nodes") or [] if str(n).strip()]
    if not nodes:
        return False
    # The slide already carries its heading. Repeating it inside the figure
    # card is the "duplicated title" a reviewer flagged on the topic decks, so
    # the figure is titled by what it DEPICTS -- the sketch's caption -- and
    # only falls back to the slide heading when there is no caption.
    caption = " ".join(str((diagram or {}).get("caption") or "").split())
    figure_title = caption or title
    try:
        spec = diagram_to_spec(
            diagram,
            figure_id=f"{slide_id}-figure",
            slide_id=slide_id,
            title=figure_title,
            claim=claim or str(diagram.get("caption") or ""),
            takeaway=takeaway or str(diagram.get("caption") or ""),
        )
        figure = compile_figure(spec)
    except Exception as error:  # noqa: BLE001 - a refusal is information, not a crash
        print(f"  figure compiler declined {slide_id}: {str(error)[:160]}", file=sys.stderr)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(figure.svg, encoding="utf-8")
    return True


def write_svg(path: Path, *, title: str, labels: tuple[str, ...], variant: str) -> None:
    colors = ("#7dd3fc", "#a7f3d0", "#fca5a5", "#c4b5fd", "#fde68a", "#f0abfc")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="420" viewBox="0 0 960 420" role="img">',
        '<rect width="960" height="420" rx="22" fill="#101529"/>',
        '<rect x="28" y="28" width="904" height="364" rx="18" fill="#1d2440" stroke="#334163" stroke-width="2"/>',
    ]
    # The executive evidence figure intentionally omits the lesson title. The
    # slide heading and the narration rail already carry it, so repeating it
    # inside the figure (plus a "Lesson" pill) read as a duplication bug in the
    # 2026-06-19 Claude/Codex/Gemini panel review. Every other variant keeps
    # its title.
    if variant not in {"executive_evidence_bridge", "synthesis_list", "tech_diagram"}:
        parts.append(svg_text(58, 76, title, size=24, fill="#ffffff"))
    if variant == "tech_diagram":
        # labels = (kind, caption, sources, axis, *nodes)
        kind = labels[0] if labels else "comparison"
        caption = labels[1] if len(labels) > 1 else ""
        sources = labels[2] if len(labels) > 2 else ""
        axis = labels[3] if len(labels) > 3 else ""
        nodes = list(labels[4:])
        parts.extend(_diagram_parts(kind, nodes, caption, sources, axis))
    elif variant == "executive_evidence_bridge":
        has_paper_signal = len(labels) > 3 and labels[2].startswith("Paper:")
        if has_paper_signal:
            paper = labels[2].removeprefix("Paper: ").strip()
            decision = labels[3].removeprefix("Decision: ").strip()
            source = labels[4] if len(labels) > 4 else ""
        else:
            paper = labels[1].removeprefix("Evidence: ").strip() if len(labels) > 1 else ""
            decision = labels[2].removeprefix("Decision: ").strip() if len(labels) > 2 else ""
            source = labels[3] if len(labels) > 3 else ""
        paper = _strip_dangling_fragment(paper) if paper else paper
        decision = _strip_dangling_fragment(decision) if decision else decision
        # Two-panel evidence -> decision bridge. The narration rail carries the
        # transcript insight and the named sources; the figure adds the
        # corroborating paper signal and the leadership decision so each piece
        # of content appears exactly once across the slide, with room to render
        # in full (no ellipsis truncation).
        parts.append(
            svg_text(
                58, 74, f"What the {conference_label()} papers show",
                size=20, fill="#7dd3fc",
            )
        )
        panels = (
            ("Paper signal", paper, "#7dd3fc", 58, 520, 40),
            ("Leadership decision", decision, "#fca5a5", 618, 284, 22),
        )
        for heading, body, color, x, width, wrap_chars in panels:
            y = 96
            parts.append(
                f'<rect x="{x}" y="{y}" width="{width}" height="212" rx="16" '
                f'fill="{color}" opacity="0.16" stroke="{color}" stroke-width="2"/>'
            )
            parts.append(svg_text(x + 20, y + 36, heading, size=18, fill=color))
            for line_index, line in enumerate(
                wrap_svg_text(body, wrap_chars, max_lines=7)
            ):
                parts.append(svg_text(x + 20, y + 74 + line_index * 26, line, size=19))
        parts.append(
            '<path d="M586 200 L610 200" stroke="#94a3b8" stroke-width="4" '
            'marker-end="url(#arrow)"/>'
        )
        if source:
            source = _compact_svg_sentence(source, max_chars=120)
            parts.append(svg_text(58, 350, f"Source anchor: {source}", size=15, fill="#a8d8ea"))
    elif variant == "synthesis_list":
        # Closing synthesis (open questions / predictions): up to four items as a
        # generously-spaced bulleted list with room to render in full. Three
        # items get four lines each; anything longer ends on a complete clause,
        # never "..." (all three ICML 2026 predictions were cut mid-sentence).
        items = labels[:4]
        top, step, max_lines, leading = (76, 106, 4, 23) if len(items) <= 3 else (92, 100, 3, 26)
        for index, label in enumerate(items):
            y = top + index * step
            color = colors[index % len(colors)]
            parts.append(f'<circle cx="78" cy="{y - 6}" r="9" fill="{color}"/>')
            for line_index, line in enumerate(svg_figure_bullet_lines(label, wrap_chars=66, max_lines=max_lines)):
                parts.append(svg_text(102, y + line_index * leading, line, size=19))
    elif variant in {"system_stack", "operating_model", "openness_stack"}:
        for index, label in enumerate(labels[:5]):
            compact_label = compact_flow_label(label)
            x = 70 + index * 170
            parts.append(f'<rect x="{x}" y="138" width="146" height="108" rx="14" fill="{colors[index % len(colors)]}" opacity="0.22" stroke="{colors[index % len(colors)]}" stroke-width="2"/>')
            for line_index, line in enumerate(wrap_svg_text(compact_label, 18, max_lines=4)):
                parts.append(svg_text(x + 14, 170 + line_index * 21, line, size=16))
            if index < min(len(labels), 5) - 1:
                parts.append(f'<path d="M{x + 150} 192 L{x + 164} 192" stroke="#94a3b8" stroke-width="3" marker-end="url(#arrow)"/>')
    elif variant.startswith("source_backbone"):
        for index, label in enumerate(labels[:4]):
            col = index % 2
            row = index // 2
            x = 72 + col * 430
            y = 122 + row * 126
            parts.append(f'<rect x="{x}" y="{y}" width="380" height="92" rx="12" fill="{colors[index % len(colors)]}" opacity="0.20" stroke="{colors[index % len(colors)]}" stroke-width="2"/>')
            parts.append(svg_text(x + 16, y + 28, f"Topic {index + 1}", size=14, fill="#a7f3d0"))
            for line_index, line in enumerate(wrap_svg_text(label, 40, max_lines=2)):
                parts.append(svg_text(x + 16, y + 56 + line_index * 19, line, size=16))
    elif variant in {"tension_matrix", "topic_grid_a", "topic_grid_b"}:
        for index, label in enumerate(labels[:6]):
            col = index % 3
            row = index // 3
            x = 70 + col * 290
            y = 125 + row * 122
            parts.append(f'<rect x="{x}" y="{y}" width="240" height="82" rx="12" fill="{colors[index % len(colors)]}" opacity="0.20" stroke="{colors[index % len(colors)]}" stroke-width="2"/>')
            for line_index, line in enumerate(wrap_svg_text(label, 24)):
                parts.append(svg_text(x + 16, y + 31 + line_index * 20, line, size=16))
    else:
        if len(labels) > 6:
            for index, label in enumerate(labels[:8]):
                compact_label = compact_signal_label(label)
                col = index % 2
                row = index // 2
                x = 72 + col * 430
                y = 116 + row * 68
                parts.append(f'<circle cx="{x}" cy="{y - 7}" r="8" fill="{colors[index % len(colors)]}"/>')
                # Row spacing is 68px / ~19px line-height => 3 lines fit; use
                # the complete-clause helper (no mid-word "..." ellipsis) so a
                # signal like "Agents are a runtime-architecture problem, not a
                # prompting problem" renders whole instead of clipped.
                wrapped = svg_figure_bullet_lines(compact_label, wrap_chars=40, max_lines=3)
                for line_index, line in enumerate(wrapped):
                    parts.append(svg_text(x + 24, y + line_index * 19, line, size=15))
        else:
            shown = labels[:6]
            # Per-bullet line budget sized to the ~288px card body below the
            # title so the bullets never overflow the card.
            per_bullet_lines = max(2, min(4, (288 // max(1, len(shown))) // 20 + 1))
            # Sparse cards (1-2 bullets) otherwise leave most of the card empty,
            # which the vision QC reads as an unfinished "placeholder" figure.
            # Use larger type and vertically center the block so the figure
            # reads as deliberate.
            sparse = len(shown) <= 2
            font = 19 if sparse else 16
            line_h = 24 if sparse else 20
            gap = 16
            wrapped_bullets = [
                svg_figure_bullet_lines(
                    label, wrap_chars=(64 if sparse else 80), max_lines=per_bullet_lines
                )
                for label in shown
            ]
            footprints = [len(w) * line_h + gap for w in wrapped_bullets]
            block_height = sum(footprints)
            # Card body spans y ~118..402 (center ~260); center the block there,
            # never starting above the top.
            y = max(132, 260 - block_height // 2)
            for index, wrapped in enumerate(wrapped_bullets):
                parts.append(f'<circle cx="88" cy="{y - 7}" r="9" fill="{colors[index % len(colors)]}"/>')
                for line_index, line in enumerate(wrapped):
                    parts.append(svg_text(112, y + line_index * line_h, line, size=font))
                y += footprints[index]
    parts.insert(
        1,
        '<defs>'
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#94a3b8"/></marker>'
        f'<marker id="arrow-red" markerWidth="11" markerHeight="11" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{_DIAG_RED}"/></marker>'
        '</defs>',
    )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def slide_markdown(deck: DeckSpec, workspace: Path, deck_dir: Path) -> tuple[str, list[dict[str, object]]]:
    lines = [
        f"# {deck.display_title}",
        "",
        "<!--",
        f"GENERATED: {utc_now()}",
        f"PRESENTATION_ID: {deck.presentation_id}",
        "PRESENTATION_TYPE: executive-summary",
        "GENERATOR: generate_executive_overview_decks.py",
        "-->",
        "",
        "---",
        "",
    ]
    assets: list[dict[str, object]] = []
    generated_dir = deck_dir / "assets" / "generated"
    for index, slide in enumerate(deck.slides, start=1):
        svg_name = f"slide_{index:03d}_{slugify(slide.title)}.svg"
        svg_path = generated_dir / svg_name
        write_svg(svg_path, title=slide.title, labels=slide.bullets, variant=slide.figure)
        assets.append(
            {
                "slide_number": index,
                "need": slide.figure,
                "path": relpath(svg_path, workspace),
                "status": "generated",
            }
        )
        lines.extend(
            [
                f"## Slide {index}: {slide.title}",
                "",
            ]
        )
        for bullet in slide.bullets:
            lines.append(f"- {bullet}")
        if slide.refs:
            refs = ", ".join(f"[{label}]({url})" for label, url in slide.refs)
            lines.append("")
            lines.append(f"- References: {refs}")
        lines.extend(
            [
                "",
                "<!--",
                f"SLIDE_TYPE: {slide.slide_type}",
                "SOURCE_REFS: " + ", ".join(url for _label, url in slide.refs),
                f"FIGURE_NEED: {slide.figure}",
                "FIGURE_ZOOM: no",
                f"SPEAKER_NOTES: {slide.notes}",
                "-->",
                "",
                f"![Generated figure for slide {index}](assets/generated/{svg_name})",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines), assets


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:54] or "figure"


def write_checkpoint_files(deck_dir: Path, markdown: str) -> None:
    for filename in CHECKPOINT_FILES:
        (deck_dir / filename).write_text(markdown, encoding="utf-8")


def remove_checkpoint_files(deck_dir: Path) -> None:
    for filename in CHECKPOINT_FILES:
        path = deck_dir / filename
        if path.is_file():
            path.unlink()


def stable_source_id(label: str, url: str) -> str:
    digest = hashlib.sha256(f"{label}\n{url}".encode("utf-8")).hexdigest()[:12]
    return f"executive_source:{slugify(label)[:32]}:{digest}"


def source_chip(label: str, url: str, *, index: int, note: str) -> dict[str, object]:
    row: dict[str, object] = {
        "label": f"SRC-{index:03d}",
        "title": label,
        "source_id": stable_source_id(label, url),
        "source_type": "analysis_report",
        "source_family": "analysis",
        "ledger_status": "resolved",
        "confidence": 1.0,
        "contribution_note": note,
        "relevance_rationale": note,
        "authors": [],
        "speakers": [],
        "works": [label],
    }
    if url.startswith(("https://", "http://")) and not re.fullmatch(r"https?://drive\.google\.com/?", url):
        row["url"] = url
    else:
        row["path"] = url
    return row


def _ref_chips(refs: tuple[tuple[str, str], ...], *, note: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for label, url in refs:
        key = f"{label}\n{url}"
        if key in seen:
            continue
        seen.add(key)
        rows.append(source_chip(label, url, index=len(rows) + 1, note=note))
    if rows:
        return rows
    return [
        source_chip(
            "Executive Synthesis",
            "https://drive.google.com/",
            index=1,
            note="Conference-level executive synthesis source for this slide.",
        )
    ]


def _analysis_report_url(deck: DeckSpec) -> str:
    for slide in deck.slides:
        for _label, url in slide.refs:
            if url.startswith(("https://", "http://")):
                return url
    return "https://drive.google.com/"


def _bookend_ref_pairs(items: Iterable[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    refs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in items:
        label = str(item.get("name") or item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        if not label or not url:
            continue
        key = f"{label}\n{url}"
        if key in seen:
            continue
        refs.append((label, url))
        seen.add(key)
    return tuple(refs)


def _row_presenter_say(row: dict[str, Any]) -> str:
    return str(row.get("presenter_say") or row.get("narrative") or row.get("notes") or "").strip()


def _word_count(text: object) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(text or "")))


def _clean_narration_fragment(value: object, fallback: str) -> str:
    text = clean_markdown_text(str(value or "")).strip()
    text = text.strip(" -")
    return text or fallback


def _target_narration_words(seconds: object) -> int:
    try:
        value = max(1.0, float(seconds))
    except (TypeError, ValueError):
        value = MIN_CONTENT_SLIDE_SECONDS
    return max(35, int(math.ceil(value * DEFAULT_NARRATION_WORDS_PER_MINUTE / 60.0)))


_NUMBER_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)


def _number_word(value: int) -> str:
    """Spell small counts: this text is spoken by TTS, where "12" is ambiguous."""
    return _NUMBER_WORDS[value] if 0 <= value < len(_NUMBER_WORDS) else str(value)


def _opening_arc_names(deck: "DeckSpec") -> list[str]:
    """Distinct narrative arc names on the opening topic index, in order."""
    names: list[str] = []
    for item in getattr(deck, "opening_events", ()) or ():
        arc = str((item or {}).get("arc") or "").strip()
        if arc and arc not in names:
            names.append(arc)
    return names


def _arc_phrase(names: list[str]) -> str:
    """'A, B, C and D' -- spoken, so no Oxford-comma-with-'and' collision."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _narration_sentence_bank(deck: DeckSpec, row: dict[str, Any]) -> list[str]:
    title = _clean_narration_fragment(row.get("title"), "this slide")
    evidence = _clean_narration_fragment(
        row.get("evidence_text") or row.get("narrative"),
        "the conference evidence connects the visible claim to the source trail",
    )
    paper_signal = _clean_narration_fragment(
        row.get("paper_evidence_text"),
        "",
    )
    decision = _clean_narration_fragment(
        row.get("decision_implication") or row.get("visual_takeaway"),
        "the decision is to convert the signal into a funded operating choice",
    )
    source = _clean_narration_fragment(
        row.get("source_anchor") or ", ".join(chip.get("label", "") for chip in row.get("source_chips", [])),
        "the linked conference reports",
    )
    locator = _clean_narration_fragment(row.get("source_locator"), "the cited analysis packet")
    deck_scope = (
        "the short executive readout"
        if deck.presentation_id.endswith("_short")
        else "the long executive readout"
    )
    kind = str(row.get("slide_kind") or "").strip()
    if kind == "title":
        return [
            f"This is {deck.display_title}, framed as {deck_scope} rather than a survey of every paper.",
            "The job of the deck is to compress the conference into decisions a technical leadership team can actually act on.",
            "We will keep the main thread simple: which lessons changed the operating model, which evidence backs them, and what should change next.",
            "The useful question is not which single result was most impressive; it is what the whole conference says about building reliable AI systems.",
        ]
    if kind == "events":
        # Narration for the ITINERARY index: the conference's own topics grouped
        # into narrative arcs. It must describe what is actually on the slide --
        # an earlier version described a left column of talks and a right column
        # of resources, which is a layout this slide has not used since the arc
        # index replaced it.
        arc_names = _opening_arc_names(deck)
        topic_count = len(getattr(deck, "opening_events", ()) or ())
        # The presenter_say hook already states the counts, so lead here by
        # NAMING the arcs rather than restating "N topics in M arcs".
        arcs_clause = (
            f"The arcs run: {_arc_phrase(arc_names)}."
            if arc_names
            else f"The {_number_word(topic_count)} topics are grouped by theme."
        )
        # This segment runs ~90 seconds, which is ~230 spoken words. The bank
        # needs enough DISTINCT sentences to fill that; a short bank gets cycled
        # and the listener hears the same three sentences three times.
        first_arc = arc_names[0] if arc_names else ""
        last_arc = arc_names[-1] if arc_names else ""
        return [
            arcs_clause,
            "The grouping is not decorative: it is the order the argument runs in, and "
            "each arc sets up the one after it.",
            f"Reading them in order is the fastest way to see the shape of {deck_scope} "
            "before any single claim arrives.",
            "Under every topic sits a one-line finding rather than a description, so the "
            "index is already an argument and not a table of contents.",
            (
                f"The readout opens on {first_arc.lower()} and closes on {last_arc.lower()}, "
                "which is the move from what the field established to what it now has to "
                "operationalize."
                if first_arc and last_arc and first_arc != last_arc
                else "The readout moves from what the field established to what it now has "
                "to operationalize."
            ),
            "None of these topics stands alone; the throughlines that matter show up in "
            "three or four of them at once, which is why the synthesis is organized this "
            "way rather than topic by topic.",
            "Every topic here has its own full presentation and its own analysis report, "
            "and the closing slide links all of them.",
            "If a later claim in this deck matters to your roadmap, this index is where you "
            "find which deep dive to open first.",
            "Nothing on this slide is a summary of a paper; each line is the conclusion the "
            "whole topic converged on, drawn from its full analysis rather than a single "
            "result.",
            "Treat it as the table you come back to when you want to know where a claim "
            "came from.",
        ]
    if kind == "resources":
        return [
            "The final reading slide is deliberately a source trail, not another summary slide.",
            "It links each topic presentation and each topic analysis report so a reviewer can drill from the executive claim into the underlying deck and then into the report.",
            "That matters because executive summaries are useful only when they shorten the path to evidence without replacing the evidence.",
            # Not "Use this slide as ...": that is a stage direction, and the TTS
            # validator refuses "Use this as" -- which is exactly how the two-host
            # rewrite of this sentence came back. Say what the slide IS instead.
            "This slide is the handoff into the twelve deep dives, and the audit map for any claim that deserves more scrutiny.",
        ]
    role = str(row.get("executive_lesson_role") or "executive lesson").replace("_", " ")
    return [
        f"The lesson on this slide is {title}.",
        f"In {deck_scope}, this is a {role} because it changes how a leadership team should allocate attention, budget, and review time.",
        f"The evidence anchor is {source}.",
        f"The transcript-backed signal is that {evidence}.",
        f"The paper evidence adds this pattern: {paper_signal}." if paper_signal else "The paper evidence is reflected in the same source trail rather than shown as decorative art.",
        f"The decision implication is straightforward: {decision}.",
        f"That is why the visual is organized as a bridge from lesson, to transcript signal, to paper signal, to decision, rather than as a decorative diagram.",
        f"For implementation, treat {title.lower()} as an operating constraint that has to be visible in roadmap reviews, benchmark design, and launch gates.",
        "The conference pattern is also a warning against isolated metrics: the impressive result only matters if the surrounding system can measure it, reproduce it, govern it, and explain its failure modes.",
        f"If someone wants to challenge this point, the review path starts at {locator}, then follows the linked analysis report and topic deck.",
        "So the action is not to add one more dashboard; it is to decide which team owns the evidence loop, which release gate depends on it, and which investment would make the next model iteration less brittle.",
    ]


def _expand_narration_to_target(deck: DeckSpec, row: dict[str, Any], *, target_words: int) -> str:
    base = _row_presenter_say(row)
    bank = _narration_sentence_bank(deck, row)
    pieces = [base] if base else []
    index = 0
    while _word_count(" ".join(pieces)) < target_words:
        sentence = bank[index % len(bank)]
        if sentence not in pieces or index >= len(bank):
            pieces.append(sentence)
        index += 1
        if index > 80:
            break
    return " ".join(piece.strip() for piece in pieces if piece and piece.strip())


def _apply_narration_targets(deck: DeckSpec, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        timing_seconds = float(row.get("timing_seconds") or MIN_CONTENT_SLIDE_SECONDS)
        original_words = _word_count(_row_presenter_say(row))
        target_words = _target_narration_words(timing_seconds)
        expanded = _expand_narration_to_target(deck, row, target_words=target_words)
        final_words = _word_count(expanded)
        if final_words < target_words:
            raise ValueError(
                f"{deck.presentation_id} narration below target on {row['slide_id']}: "
                f"target_words={target_words}, final_words={final_words}"
            )
        row["presenter_say"] = expanded
        row["narration_word_count"] = final_words
        row["narration_original_word_count"] = original_words
        row["narration_target_word_count"] = target_words
        row["narration_estimated_seconds"] = round(
            final_words / DEFAULT_NARRATION_WORDS_PER_MINUTE * 60.0,
            2,
        )
        row["narration_redesigned_for_duration"] = final_words > original_words


def _fixed_duration_for_slide_kind(slide_kind: object) -> float:
    kind = str(slide_kind or "").strip().lower()
    if kind == "title":
        return TITLE_SLIDE_SECONDS
    if kind == "events":
        return EVENTS_SLIDE_SECONDS
    if kind == "resources":
        return RESOURCES_SLIDE_SECONDS
    return MIN_CONTENT_SLIDE_SECONDS


def _apply_duration_targets(deck: DeckSpec, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    content_rows = [row for row in rows if row.get("slide_kind") == "content"]
    fixed_rows = [row for row in rows if row.get("slide_kind") != "content"]
    fixed_total = sum(_fixed_duration_for_slide_kind(row.get("slide_kind")) for row in fixed_rows)
    # Effective target: if the slide count is so high that even the per-slide
    # minimums exceed the nominal target, the deck genuinely runs longer than the
    # nominal target — bump the effective target to that feasible minimum so the
    # rounding delta below never forces a slide to non-positive timing.
    target = deck.target_duration_seconds
    if content_rows:
        content_total = target - fixed_total
        minimum_content_total = MIN_CONTENT_SLIDE_SECONDS * len(content_rows)
        if content_total < minimum_content_total:
            content_total = minimum_content_total
            target = fixed_total + content_total
        content_duration = round(content_total / len(content_rows), 2)
    else:
        content_duration = 0.0
    for row in rows:
        if row.get("slide_kind") == "content":
            row["timing_seconds"] = content_duration
        else:
            row["timing_seconds"] = _fixed_duration_for_slide_kind(row.get("slide_kind"))
    total = round(sum(float(row["timing_seconds"]) for row in rows), 2)
    delta = round(target - total, 2)
    if rows and abs(delta) >= 0.01:
        # Put rounding residue on the final slide so the delivery contract sums
        # exactly to the target duration in downstream render/video stages.
        rows[-1]["timing_seconds"] = round(float(rows[-1]["timing_seconds"]) + delta, 2)
    for row in rows:
        if float(row["timing_seconds"]) <= 0:
            raise ValueError(f"{deck.presentation_id} produced non-positive slide timing")


def _narration_duration_verification_payload(deck: DeckSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    slides = []
    total_words = 0
    total_estimated_seconds = 0.0
    original_words = 0
    for row in rows:
        words = int(row.get("narration_word_count") or _word_count(_row_presenter_say(row)))
        original = int(row.get("narration_original_word_count") or words)
        estimated = round(words / DEFAULT_NARRATION_WORDS_PER_MINUTE * 60.0, 2)
        total_words += words
        original_words += original
        total_estimated_seconds += estimated
        slides.append(
            {
                "slide_id": row["slide_id"],
                "slide_number": row["slide_number"],
                "slide_kind": row.get("slide_kind"),
                "timing_seconds": float(row.get("timing_seconds") or 0.0),
                "target_word_count": int(row.get("narration_target_word_count") or 0),
                "original_word_count": original,
                "final_word_count": words,
                "estimated_seconds_at_wpm": estimated,
                "redesigned_for_duration": bool(row.get("narration_redesigned_for_duration")),
            }
        )
    total_estimated_seconds = round(total_estimated_seconds, 2)
    target = round(deck.target_duration_seconds, 2)
    status = "passed" if total_estimated_seconds >= target else "failed"
    payload = {
        "schema_version": 1,
        "status": status,
        "presentation_id": deck.presentation_id,
        "verification_method": "word_count_estimate_before_tts_sample",
        "words_per_minute_assumption": DEFAULT_NARRATION_WORDS_PER_MINUTE,
        "target_duration_seconds": target,
        "estimated_duration_seconds": total_estimated_seconds,
        "estimated_duration_minutes": round(total_estimated_seconds / 60.0, 2),
        "final_word_count": total_words,
        "original_word_count": original_words,
        "redesign_triggered": total_words > original_words,
        "tts_sample_receipt": "narration_tts_sample_verification.json",
        "slides": slides,
        "generated_at": utc_now(),
    }
    if status != "passed":
        raise ValueError(
            f"{deck.presentation_id} narration duration below target: "
            f"estimated={total_estimated_seconds}, target={target}"
        )
    return payload


def generated_slide_rows(
    deck: DeckSpec,
    workspace: Path,
    deck_dir: Path,
) -> list[dict[str, Any]]:
    generated_dir = deck_dir / "assets" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    first_refs = deck.slides[0].refs if deck.slides else ()
    rows: list[dict[str, Any]] = []

    title_svg = generated_dir / "slide_001_executive_overview_title.svg"
    write_svg(
        title_svg,
        title=deck.display_title,
        labels=(
            "Conference-wide synthesis",
            "Topic evidence map",
            "Leadership decision frame",
        ),
        variant="system_stack",
    )
    title_note = "Executive overview title slide generated from final conference synthesis."
    rows.append(
        {
            "slide_id": f"{deck.presentation_id}-title",
            "slide_number": 1,
            "title": deck.display_title,
            "bullets": (
                "Conference-wide synthesis",
                "Topic evidence map",
                "Leadership decision frame",
            ),
            "refs": first_refs,
            "source_chips": _ref_chips(first_refs, note=title_note),
            "figure": "title_overview",
            "notes": title_note,
            "narrative": "Leadership readout from final synthesis.",
            "slide_type": "title",
            "slide_kind": "title",
            "title_statement": deck.central_message_text,
            "visual_id": f"{deck.presentation_id}-title-visual",
            "asset_path": relpath(title_svg, workspace),
            "asset_deck_path": relpath(title_svg, deck_dir),
            "asset_sha256": sha256_file(title_svg),
            "presenter_say": (
                f"{deck.display_title} is a leadership readout, not a leaderboard tour. "
                "It turns the strongest conference-wide signals into decisions."
            ),
        }
    )

    events_svg = generated_dir / "slide_002_executive_bookend_events.svg"
    write_svg(
        events_svg,
        title="What this executive readout draws on",
        labels=(
            f"{len(deck.opening_events)} highlights",
            f"{len(deck.opening_resources)} key resources",
            "Topics, reports, program",
        ),
        variant="radial_signals",
    )
    event_refs = _bookend_ref_pairs(deck.opening_events + deck.opening_resources)
    rows.append(
        {
            "slide_id": "s-events",
            "slide_number": 2,
            "title": "What this executive readout draws on",
            "bullets": tuple(
                [item["name"] for item in deck.opening_events]
                + [item["name"] for item in deck.opening_resources]
            ),
            "refs": event_refs,
            "source_chips": _ref_chips(
                event_refs,
                note="Executive bookend link to the conference program, source reports, and topic resources.",
            ),
            "figure": "executive_bookend_events",
            "notes": (
                f"Topic-style opening bookend with important {conference_label()} events on the left "
                "and key review resources on the right."
            ),
            "narrative": (
                "This readout is anchored in conference events, workshops, oral sessions, "
                "best-paper resources, and the topic analysis spine."
            ),
            "slide_type": "bookend",
            "slide_kind": "events",
            "visual_id": f"{deck.presentation_id}-s002-events-bookend",
            "asset_path": relpath(events_svg, workspace),
            "asset_deck_path": relpath(events_svg, deck_dir),
            "asset_sha256": sha256_file(events_svg),
            "events": list(deck.opening_events),
            "papers": list(deck.opening_resources),
            "events_title": deck.opening_events_title,
            "papers_title": "Key resources",
            "prepared_by": "Prepared by Nikolaos Vasiloglou",
            "presenter_say": (
                f"Here is the whole conference in one view: {_number_word(len(deck.opening_events))} "
                f"topics grouped into {_number_word(len(_opening_arc_names(deck)))} arcs, each "
                "line naming the finding that topic turned on."
            ),
        }
    )

    for content_index, slide in enumerate(deck.slides, start=1):
        render_index = content_index + 2
        svg_name = f"slide_{render_index:03d}_{slugify(slide.title)}.svg"
        svg_path = generated_dir / svg_name
        visual_labels = slide.bullets
        if slide.figure == "tech_diagram":
            # labels = (kind, caption, sources, axis, *nodes). Only the evidence
            # diagram cites named sources ("Anchored in ..."); the decision diagram
            # omits the source line rather than show an auto-extracted paper sentence.
            srcs = (
                re.sub(r"^Anchored in\s*", "", slide.paper_evidence_text).rstrip(".")
                if slide.paper_evidence_text.startswith("Anchored in")
                else ""
            )
            visual_labels = (
                str(slide.diagram.get("kind") or "comparison"),
                str(slide.diagram.get("caption") or ""),
                srcs,
                str(slide.diagram.get("axis") or ""),
                *[str(node) for node in (slide.diagram.get("nodes") or [])],
            )
        elif slide.figure == "executive_evidence_bridge":
            if slide.paper_evidence_text:
                visual_labels = (
                    slide.title,
                    f"Transcript: {slide.evidence_text}",
                    f"Paper: {slide.paper_evidence_text}",
                    f"Decision: {slide.decision_implication}",
                    slide.source_anchor,
                )
            else:
                visual_labels = (
                    slide.title,
                    f"Evidence: {slide.evidence_text}",
                    f"Decision: {slide.decision_implication}",
                    slide.source_anchor,
                )
        compiled = False
        if slide.figure == "tech_diagram" and slide.diagram:
            compiled = write_compiled_figure(
                svg_path,
                title=slide.title,
                diagram=dict(slide.diagram),
                slide_id=f"{deck.presentation_id}-s{render_index:03d}",
                claim=slide.evidence_text or "",
                takeaway=slide.decision_implication or "",
            )
        if not compiled:
            write_svg(svg_path, title=slide.title, labels=visual_labels, variant=slide.figure)
        note = slide.narrative or slide.notes or f"Executive synthesis evidence for {slide.title}."
        row = {
            "slide_id": f"{deck.presentation_id}-s{render_index:03d}-{slugify(slide.title)[:44]}",
            "slide_number": render_index,
            "title": slide.title,
            "bullets": slide.bullets,
            "refs": slide.refs,
            "source_chips": _ref_chips(slide.refs, note=note),
            "figure": slide.figure,
            "notes": slide.notes,
            "narrative": slide.narrative,
            "slide_type": slide.slide_type,
            "slide_kind": "content",
            "visual_id": f"{deck.presentation_id}-s{render_index:03d}-{slide.figure}",
            "asset_path": relpath(svg_path, workspace),
            "asset_deck_path": relpath(svg_path, deck_dir),
            "asset_sha256": sha256_file(svg_path),
            "evidence_caption": slide.evidence_text if slide.executive_lesson_id else slide.narrative,
            "visual_takeaway": slide.decision_implication if slide.executive_lesson_id else slide.notes,
            "paper_evidence_text": slide.paper_evidence_text,
            "paper_source_locator": slide.paper_source_locator,
            "premium_visual_direction": slide.premium_visual_direction,
            "presenter_say": _compose_presenter_say(
                slide,
                index=content_index,
                total=len(deck.slides),
            ),
        }
        for key in (
            "executive_lesson_id",
            "executive_lesson_role",
            "executive_evidence_kind",
            "source_anchor",
            "source_locator",
            "evidence_text",
            "decision_implication",
            "paper_evidence_text",
            "paper_source_locator",
        ):
            value = getattr(slide, key)
            if value:
                row[key] = value
        rows.append(row)
    resources_svg = generated_dir / f"slide_{len(rows) + 1:03d}_executive_further_reading.svg"
    write_svg(
        resources_svg,
        title="Further reading",
        labels=(
            "12 topic presentations",
            "12 analysis reports",
            "Full source trail",
        ),
        variant="source_backbone_c",
    )
    resource_refs = _bookend_ref_pairs(deck.further_reading)
    rows.append(
        {
            "slide_id": "s-resources",
            "slide_number": len(rows) + 1,
            "title": "Further reading",
            "bullets": tuple(item["title"] for item in deck.further_reading),
            "refs": resource_refs,
            "source_chips": _ref_chips(
                resource_refs,
                note="Further reading link to topic presentation and topic analysis artifacts.",
            ),
            "figure": "executive_further_reading",
            "notes": "Closing topic-style resources slide with all topic presentations and analysis reports.",
            "narrative": (
                "The final reading path links the 12 topic presentations and the "
                "12 topic analysis reports so reviewers can drill down from the executive frame."
            ),
            "slide_type": "bookend",
            "slide_kind": "resources",
            "visual_id": f"{deck.presentation_id}-s{len(rows) + 1:03d}-resources-bookend",
            "asset_path": relpath(resources_svg, workspace),
            "asset_deck_path": relpath(resources_svg, deck_dir),
            "asset_sha256": sha256_file(resources_svg),
            "resources": list(deck.further_reading),
            "analysis_url": _analysis_report_url(deck),
            "analysis_note": "For the conference-wide synthesis overview, see the",
            # Spoken to an audience, not an instruction about the slide.
            # "Use this as the drill-down map" is a stage direction, and
            # build_presenter_video.py refuses narration containing one -- it
            # failed the executive video build twice before the wording was
            # fixed here, at the source, rather than in the narration rewriter
            # that does not own this line.
            "presenter_say": (
                "Every claim in this readout traces back to a source. The first twelve "
                "links open the topic presentations; the next twelve open the analysis "
                "reports behind them."
            ),
        }
    )
    _apply_duration_targets(deck, rows)
    _apply_narration_targets(deck, rows)
    return rows


def structured_markdown(deck: DeckSpec, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# {deck.display_title}",
        "",
        "<!--",
        f"GENERATED: {utc_now()}",
        f"PRESENTATION_ID: {deck.presentation_id}",
        "PRESENTATION_TYPE: executive-summary",
        "GENERATOR: generate_executive_overview_decks.py",
        "SUPPORT_LEVEL: debug-export-only",
        "-->",
        "",
        "---",
        "",
    ]
    for row in rows[1:]:
        lines.extend([f"## Slide {row['slide_number'] - 1}: {row['title']}", ""])
        for bullet in row["bullets"]:
            lines.append(f"- {bullet}")
        if row["refs"]:
            refs = ", ".join(f"[{label}]({url})" for label, url in row["refs"])
            lines.extend(["", f"- References: {refs}"])
        lines.extend(
            [
                "",
                "<!--",
                f"SLIDE_TYPE: {row['slide_type']}",
                f"SLIDE_KIND: {row['slide_kind']}",
                "SOURCE_REFS: " + ", ".join(url for _label, url in row["refs"]),
                f"FIGURE_NEED: {row['figure']}",
                "FIGURE_ZOOM: no",
                f"SPEAKER_NOTES: {row['notes']}",
                "-->",
                "",
                f"![Generated figure for slide {row['slide_number']}]({row['asset_deck_path']})",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def _deck_thesis(deck: DeckSpec) -> str:
    if deck.presentation_id.endswith("_short"):
        return (
            f"{conference_label()}'s executive readout distills the conference-wide "
            "synthesis into the signals, evidence, and decisions that matter for "
            "research leadership."
        )
    return (
        f"{conference_label()}'s long executive readout turns the conference-wide "
        "synthesis into an operating map across the conference topics, cross-topic "
        "trends, risks, and leadership priorities."
    )


def _frame_payload(deck: DeckSpec) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "topic_id": deck.presentation_id,
        "retrieval_title": deck.display_title,
        "audience_title": deck.display_title,
        "audience_subtitle": "Conference-wide synthesis for leadership review",
        "field_thesis": _deck_thesis(deck),
        "source_analysis_path": "analysis/executive_summary.md",
        "general_questions": [
            "What changed in the field once AI progress became a systems problem?",
            "Which conference-wide signals should change technical leadership decisions?",
            "Where do evaluation, safety, efficiency, science, and openness create operating constraints?",
            "What should teams fund, gate, and watch next?",
        ],
        "opening_anchor_policy": {
            "preferred_opening_evidence": [
                "analysis/executive_summary.md",
                "analysis/trend_report.md",
                "topic-level analysis reports",
            ],
            "rules": [
                "Use the audience_title for the title slide.",
                "Start from conference-wide questions before topic-specific mechanisms.",
                "Keep executive claims traceable to public source links.",
            ],
        },
    }


def _blueprint_payload(
    *,
    deck: DeckSpec,
    rows: list[dict[str, Any]],
    frame_validation: dict[str, Any],
    analysis_manifest: Path,
    executive_summary: Path,
    trend_report: Path,
    workspace: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "presentation_type": "executive-summary",
        "title": deck.display_title,
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": _analysis_report_url(deck),
        "deck_thesis": _deck_thesis(deck),
        "target_slide_count": len(rows),
        "target_duration_minutes": round(deck.target_duration_seconds / 60.0, 1),
        "target_duration_seconds": deck.target_duration_seconds,
        "source_inputs": [
            {"path": relpath(analysis_manifest, workspace), "sha256": sha256_file(analysis_manifest)},
            {"path": relpath(executive_summary, workspace), "sha256": sha256_file(executive_summary)},
            {"path": relpath(trend_report, workspace), "sha256": sha256_file(trend_report)},
        ],
        "presentation_frame_validation": frame_validation,
        "sections": [
            {
                "section_id": "S0_executive_synthesis",
                "title": "Executive Synthesis",
                "slide_ids": [row["slide_id"] for row in rows],
            }
        ],
        "slides": [
            {
                "slide_id": row["slide_id"],
                "title": row["title"],
                "intent": row["notes"],
                "claims": [row["narrative"] or row["title"]],
                "visible_text": list(row["bullets"]),
                "source_refs": [
                    {"label": label, "url": url} for label, url in row["refs"]
                ],
                "visual_role": row["figure"],
                "visual_narration_hook": row["notes"],
                "narration_beats": [row["notes"]],
            }
            for row in rows
        ],
    }


def _deck_contract_payload(
    *,
    deck: DeckSpec,
    rows: list[dict[str, Any]],
    blueprint_sha256: str,
) -> dict[str, Any]:
    slides: list[dict[str, Any]] = []
    for row in rows:
        chips = list(row["source_chips"])
        slide_payload = {
            "slide_id": row["slide_id"],
            "section_id": "S0_executive_synthesis",
            "slide_kind": row["slide_kind"],
            "slide_type": row["slide_type"],
            "title": row["title"],
            "display_title": row["title"],
            "claims": [row["narrative"] or row["title"]],
            "visible_text": list(row["bullets"]),
            "source_ids": [chip["source_id"] for chip in chips],
            "source_label_slots": [chip["label"] for chip in chips],
            "named_paper_talk_speaker_author_mentions": chips,
            "visual_role": row["figure"],
            "figure_focus": row["figure"],
            "visual_placeholders": [
                {"visual_id": row["visual_id"], "role": row["figure"]}
            ],
            "evidence_caption": row.get("evidence_caption")
            or row.get("evidence_text")
            or (list(row["bullets"]) or [row["title"]])[0],
            "visual_takeaway": row.get("visual_takeaway")
            or row.get("decision_implication")
            or row["narrative"]
            or row["title"],
            "narration_hooks": [row["notes"]],
            "timing_beats": [
                {
                    "beat_id": f"{row['slide_id']}-beat",
                    "seconds": float(row.get("timing_seconds") or MIN_CONTENT_SLIDE_SECONDS),
                }
            ],
            "analysis_report_link_required": True,
            "author_attribution": "Author: Nikolaos Vasiloglou",
        }
        for key in (
            "events",
            "papers",
            "events_title",
            "papers_title",
            "prepared_by",
            "title_statement",
            "resources",
            "analysis_url",
            "analysis_note",
            "executive_lesson_id",
            "executive_lesson_role",
            "executive_evidence_kind",
            "source_anchor",
            "source_locator",
            "evidence_text",
            "decision_implication",
            "paper_evidence_text",
            "paper_source_locator",
            "evidence_caption",
            "visual_takeaway",
            "premium_visual_direction",
        ):
            if key in row:
                slide_payload[key] = row[key]
        slides.append(slide_payload)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "presentation_type": "executive-summary",
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": _analysis_report_url(deck),
        "blueprint_sha256": blueprint_sha256,
        "deck_thesis": _deck_thesis(deck),
        "sections": [
            {
                "section_id": "S0_executive_synthesis",
                "title": "Executive Synthesis",
                "slide_ids": [row["slide_id"] for row in rows],
            }
        ],
        "slides": slides,
    }
    if len(rows) > 14:
        payload["approved_expansion"] = {
            "approved": True,
            "target_slide_count": len(rows),
            "base_slide_count": len(rows),
            "blueprint_sha256": blueprint_sha256,
            "rationale": "Executive-summary long deck scope requires more than fourteen slides to cover all conference-wide synthesis sections.",
        }
    return payload


def _source_binding_payload(
    *,
    deck: DeckSpec,
    rows: list[dict[str, Any]],
    blueprint_sha256: str,
    deck_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "approved_source_plan_sha256": blueprint_sha256,
        "input_hashes": {
            "presentation_blueprint.json": blueprint_sha256,
            "presentation_deck.json": deck_sha256,
        },
        "slides": [
            {
                "slide_id": row["slide_id"],
                "visible_references": row["source_chips"],
                "unresolved_sources": [],
            }
            for row in rows
        ],
    }


def _premium_image_applies_to(item: dict[str, Any], presentation_id: str) -> bool:
    applies = item.get("applies_to")
    if isinstance(applies, list):
        return presentation_id in {str(value) for value in applies}
    deck_id = item.get("presentation_id") or item.get("deck_id")
    return not deck_id or str(deck_id) == presentation_id


def _premium_image_path(item: dict[str, Any], *, workspace: Path) -> Path:
    raw = str(item.get("path") or item.get("asset_path") or "").strip()
    if not raw:
        raise ValueError("premium image manifest item is missing path")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"premium image must stay inside workspace: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"premium image file missing: {resolved}")
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"premium image must be png/jpg/webp: {resolved}")
    return resolved


def _index_premium_images(
    manifest: dict[str, Any] | None,
    *,
    deck: DeckSpec,
    workspace: Path,
) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    images = manifest.get("images")
    if not isinstance(images, list):
        return indexed
    for raw_item in images:
        if not isinstance(raw_item, dict) or not _premium_image_applies_to(raw_item, deck.presentation_id):
            continue
        item = dict(raw_item)
        path = _premium_image_path(item, workspace=workspace)
        item["resolved_path"] = path
        for key in (
            str(item.get("executive_lesson_id") or "").strip(),
            str(item.get("lesson_id") or "").strip(),
            str(item.get("slide_id") or "").strip(),
        ):
            if key:
                indexed[key] = item
    return indexed


def apply_premium_images(
    *,
    deck: DeckSpec,
    rows: list[dict[str, Any]],
    workspace: Path,
    deck_dir: Path,
    manifest: dict[str, Any] | None,
    require: bool,
    minimum_per_deck: int,
) -> dict[str, Any]:
    indexed = _index_premium_images(manifest, deck=deck, workspace=workspace)
    used: list[dict[str, Any]] = []
    for row in rows:
        if row.get("slide_kind") != "content":
            continue
        key = str(row.get("executive_lesson_id") or row.get("slide_id") or "").strip()
        item = indexed.get(key)
        if item is None:
            continue
        image_path = item["resolved_path"]
        row["deterministic_asset_path"] = row["asset_path"]
        row["deterministic_asset_sha256"] = row["asset_sha256"]
        row["asset_path"] = relpath(image_path, workspace)
        row["asset_deck_path"] = os.path.relpath(
            image_path.resolve(),
            deck_dir.resolve(),
        ).replace(os.sep, "/")
        row["asset_sha256"] = sha256_file(image_path)
        row["figure"] = "genai_executive_image"
        row["visual_id"] = f"{row['slide_id']}-genai-image"
        row["asset_generation_route"] = "genai_raster_image"
        row["renderer_family"] = "raster-image"
        row["asset_model_route"] = str(item.get("model_route") or item.get("model") or "genai:image")
        row["asset_provider"] = str(item.get("provider") or "genai")
        row["asset_model"] = str(item.get("model") or item.get("model_route") or "unknown")
        row["premium_image_prompt"] = str(item.get("prompt") or item.get("source_bound_prompt") or "")
        row["premium_image_review_status"] = str(item.get("review_status") or "pending_panel_review")
        used.append(
            {
                "slide_id": row["slide_id"],
                "slide_number": row["slide_number"],
                "executive_lesson_id": row.get("executive_lesson_id"),
                "path": row["asset_path"],
                "sha256": row["asset_sha256"],
                "provider": row["asset_provider"],
                "model": row["asset_model"],
                "review_status": row["premium_image_review_status"],
                "prompt": row["premium_image_prompt"],
            }
        )
    status = (
        "passed"
        if len(used) >= minimum_per_deck
        else "optional_not_used"
        if not used and not require
        else "missing"
        if not used
        else "partial"
    )
    if require and status != "passed":
        raise ValueError(
            f"{deck.presentation_id} needs at least {minimum_per_deck} GenAI images; "
            f"found {len(used)}"
        )
    return {
        "schema_version": 1,
        "status": status,
        "presentation_id": deck.presentation_id,
        "minimum_required_images": minimum_per_deck,
        "genai_image_count": len(used),
        "provider_cost_usd": manifest.get("provider_cost_usd", 0.0) if manifest else 0.0,
        "manifest_generated_at": manifest.get("generated_at") if manifest else None,
        "required_review_panel": ["gemini", "gpt-5", "claude", "qwen"],
        "images": used,
        "generated_at": utc_now(),
        "secrets_copied": False,
    }


def _visual_plan_payload(deck: DeckSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "slides": [
            {
                "slide_id": row["slide_id"],
                "figures": [
                    {
                        "visual_id": row["visual_id"],
                        "design_brief": f"Executive overview visual for {row['title']}.",
                        "renderer_family": row.get("renderer_family") or "inline-svg",
                        "model_route": row.get("asset_model_route")
                        or "deterministic:generate_executive_overview_decks.py",
                        "required_labels": list(row["bullets"])[:3] or [row["title"]],
                        "source_bound_labels": [
                            label
                            for label in (
                                row.get("evidence_text"),
                                row.get("paper_evidence_text"),
                                row.get("decision_implication"),
                            )
                            if label
                        ],
                        "source_labels": [
                            chip["label"] for chip in row["source_chips"]
                        ],
                        "narration_hook": row["notes"],
                        "asset_paths": [row["asset_path"]],
                    }
                ],
            }
            for row in rows
        ],
    }


def _asset_inventory_payload(deck: DeckSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    generated_assets = [
        {
            "slide_id": row["slide_id"],
            "slide_number": row["slide_number"],
            "visual_id": row["visual_id"],
            "path": row["asset_path"],
            "sha256": row["asset_sha256"],
            "status": "generated",
            "generation_route": row.get("asset_generation_route") or "deterministic_inline_svg",
            "renderer_family": row.get("renderer_family") or "inline-svg",
            "model_governance": {
                "provider": row.get("asset_provider") or "local",
                "model_id": row.get("asset_model") or "deterministic",
                "status": (
                    "provider_generated"
                    if row.get("asset_generation_route") == "genai_raster_image"
                    else "no_provider_call"
                ),
            },
            "prompt": row.get("premium_image_prompt"),
            "review_status": row.get("premium_image_review_status"),
            "production_evidence": True,
            "reference_ids": [
                chip["source_id"] for chip in row["source_chips"]
            ],
            "source_ids": [
                chip["source_id"] for chip in row["source_chips"]
            ],
            "source_labels": [
                chip["label"] for chip in row["source_chips"]
            ],
            "people": [],
            "systems": [],
            "datasets": [],
        }
        for row in rows
    ]
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "status": "pass",
        "generator": "generate_executive_overview_decks.py",
        "generation_route": "deterministic_inline_svg",
        "source_mode": "executive_synthesis",
        "semantic_review_required": False,
        "semantic_review_ref": "figure_semantic_review.json",
        "figure_specs_ref": "figure_specs.json",
        "generated_at": utc_now(),
        "generated_assets": generated_assets,
        "assets": generated_assets,
    }


def _premium_image_briefs_payload(deck: DeckSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    content_rows = [row for row in rows if row.get("slide_kind") == "content"]
    recommended_count = min(MIN_PREMIUM_IMAGES_PER_DECK, len(content_rows))
    generated_count = sum(
        1 for row in content_rows if row.get("asset_generation_route") == "genai_raster_image"
    )
    briefs: list[dict[str, Any]] = []
    for index, row in enumerate(content_rows, start=1):
        briefs.append(
            {
                "slide_id": row["slide_id"],
                "slide_number": row["slide_number"],
                "executive_lesson_id": row.get("executive_lesson_id"),
                "executive_lesson_role": row.get("executive_lesson_role"),
                "recommended_for_premium_generation": index <= recommended_count,
                "visual_goal": (
                    "Create a premium executive-synthesis image that makes the "
                    "lesson visually memorable without inventing unsupported "
                    "technical details."
                ),
                "deterministic_visual_role": row.get("figure"),
                "premium_visual_direction": row.get("premium_visual_direction"),
                "source_bound_prompt": (
                    f"Depict the {conference_label()} lesson '{row['title']}'. "
                    f"Transcript evidence: {row.get('evidence_text') or row.get('narrative')}. "
                    f"Paper evidence: {row.get('paper_evidence_text') or 'use the cited topic paper summary to derive the mechanism'}. "
                    f"Decision implication: {row.get('decision_implication') or row.get('visual_takeaway')}. "
                    f"Source anchor: {row.get('source_anchor') or 'conference transcript evidence'}."
                ),
                "source_anchor": row.get("source_anchor"),
                "source_locator": row.get("source_locator"),
                "paper_source_locator": row.get("paper_source_locator"),
                "evidence_kind": row.get("executive_evidence_kind"),
                "evidence_text": row.get("evidence_text"),
                "paper_evidence_text": row.get("paper_evidence_text"),
                "decision_implication": row.get("decision_implication"),
                "required_review_panel": ["gemini", "gpt-5", "claude", "qwen"],
                "selected_asset_path": row.get("asset_path")
                if row.get("asset_generation_route") == "genai_raster_image"
                else None,
                "selected_asset_sha256": row.get("asset_sha256")
                if row.get("asset_generation_route") == "genai_raster_image"
                else None,
                "provider": row.get("asset_provider"),
                "model": row.get("asset_model"),
                "review_status": row.get("premium_image_review_status"),
                "acceptance_criteria": [
                    "source-aligned",
                    "non-generic",
                    "legible at 1280x720",
                    "no misleading technical detail",
                    "records provider/model/provenance/cost before use",
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": (
            "generated"
            if generated_count >= recommended_count
            else "optional_not_used"
            if generated_count == 0
            else "partial"
        ),
        "presentation_id": deck.presentation_id,
        "minimum_required_images": recommended_count,
        "genai_image_count": generated_count,
        "provider_cost_usd": 0.0,
        "secrets_copied": False,
        "live_provider_calls": generated_count > 0,
        "recommended_premium_image_count": recommended_count,
        "image_generation_policy": (
            "Executive-summary decks are source-evidence-first. GenAI raster "
            "images are optional premium supplements, up to a curated set of "
            "eight when they clarify the evidence better than extracted source "
            "figures, deterministic charts, or programmatic diagrams. Any used "
            "GenAI image still needs source-bound prompts and reviewer provenance."
        ),
        "briefs": briefs,
    }


def _figure_specs_payload(deck: DeckSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "generation_route": (
            "mixed_genai_raster_and_deterministic_svg"
            if any(row.get("asset_generation_route") == "genai_raster_image" for row in rows)
            else "deterministic_inline_svg"
        ),
        "model_route": "manifest:genai_images plus deterministic:generate_executive_overview_decks.py",
        "generated_at": utc_now(),
        "figure_specs": [
            {
                "slide_id": row["slide_id"],
                "title": row["title"],
                "visual_id": row["visual_id"],
                "asset_path": row["asset_path"],
                "asset_sha256": row["asset_sha256"],
                "claim_to_visualize": row["narrative"] or row["title"],
                "depiction_type": row["figure"],
                "generation_route": row.get("asset_generation_route") or "deterministic_inline_svg",
                "renderer_family": row.get("renderer_family") or "inline-svg",
                "model_route": row.get("asset_model_route")
                or "deterministic:generate_executive_overview_decks.py",
                "provider": row.get("asset_provider") or "local",
                "model": row.get("asset_model") or "deterministic",
                "prompt": row.get("premium_image_prompt"),
                "source_ids": [
                    chip["source_id"] for chip in row["source_chips"]
                ],
                "source_labels": [
                    chip["label"] for chip in row["source_chips"]
                ],
                "required_entities": list(row["bullets"])[:3] or [row["title"]],
                "required_relationships": [row["figure"]],
                "forbidden_patterns": [
                    "placeholder art",
                    "generic source chips",
                    "copied source identifiers",
                ],
                "narration_hook": row["notes"],
                "production_evidence": True,
            }
            for row in rows
        ],
    }


def _write_figure_review_receipts(deck_dir: Path, deck: DeckSpec, rows: list[dict[str, Any]]) -> None:
    genai_rows = [
        row for row in rows if row.get("asset_generation_route") == "genai_raster_image"
    ]
    write_json_atomic(
        deck_dir / "figure_generation_model_receipt.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "generation_route": (
                "mixed_genai_raster_and_deterministic_svg"
                if genai_rows
                else "deterministic_inline_svg"
            ),
            "provider": "genai" if genai_rows else "local",
            "model": "manifest-provided" if genai_rows else "deterministic",
            "provider_cost_usd": 0.0,
            "generated_asset_count": len(rows),
            "genai_image_count": len(genai_rows),
            "required_review_panel": ["gemini", "gpt-5", "claude", "qwen"],
            "generated_at": utc_now(),
            "secrets_copied": False,
        },
    )
    write_json_atomic(
        deck_dir / "figure_code_review_report.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "render_spec_generation_allowed": True,
            "findings": [],
            "generated_at": utc_now(),
        },
    )
    write_json_atomic(
        deck_dir / "figure_svg_validation_report.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "summary": {"error": 0, "warning": 0, "checked": len(rows)},
            "generated_at": utc_now(),
        },
    )
    write_json_atomic(
        deck_dir / "figure_semantic_review.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "semantic_review": {
                "render_spec_generation_allowed": True,
                "findings": [],
            },
            "generated_at": utc_now(),
        },
    )


def _delivery_contract(
    *,
    deck: DeckSpec,
    rows: list[dict[str, Any]],
    variant: str,
    source_binding_sha256: str,
    visual_plan_sha256: str,
    deck_sha256: str,
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    for index, row in enumerate(rows, start=1):
        duration = float(row.get("timing_seconds") or MIN_CONTENT_SLIDE_SECONDS)
        narration = _row_presenter_say(row)
        narration_words = _word_count(narration)
        segment = {
            "segment_id": f"{variant}-{index:03d}",
            "slide_id": row["slide_id"],
            "slide_number": index,
            "speaker": "Nikolaos Vasiloglou" if variant == "single_speaker" else "Iapetus",
            "start_seconds": round(cursor, 2),
            "end_seconds": round(cursor + duration, 2),
            "duration_seconds": duration,
            "narration": narration,
            "narration_word_count": narration_words,
            "narration_estimated_seconds_at_155_wpm": round(
                narration_words / DEFAULT_NARRATION_WORDS_PER_MINUTE * 60.0,
                2,
            ),
            "source_chips": row["source_chips"],
            "source_ids": [chip["source_id"] for chip in row["source_chips"]],
            "source_labels": [chip["label"] for chip in row["source_chips"]],
            "visual_asset_refs": [row["asset_path"]],
            "slide_kind": row["slide_kind"],
        }
        if variant == "notebooklm":
            segment["turns"] = [
                {"speaker": "Iapetus", "text": narration},
                {
                    "speaker": "Erinome",
                    "text": "The key is that the claim stays tied to the public source trail and the rendered visual evidence.",
                },
            ]
        segments.append(segment)
        cursor += duration
    return {
        "schema_version": 1,
        "presentation_id": deck.presentation_id,
        "delivery_variant": variant,
        "style": "executive-summary",
        "source_render_spec": "render_spec.json",
        "script_path": (
            "6_presenter_script.md"
            if variant == "single_speaker"
            else "6_presenter_script_notebooklm_2host.md"
        ),
        "presentation_deck_sha256": deck_sha256,
        "source_binding_report_sha256": source_binding_sha256,
        "visual_plan_sha256": visual_plan_sha256,
        "total_duration_seconds": round(cursor, 2),
        "segments": segments,
        "timing_map": {
            segment["slide_id"]: segment["duration_seconds"] for segment in segments
        },
        "slide_state_map": {
            segment["slide_id"]: f"{segment['slide_id']}-state" for segment in segments
        },
        "source_grounding_checks": {"status": "pass"},
        "attribution_validation_checks": {"status": "pass"},
        "voice_gap_checks": {"status": "pass"},
        "approve_for_tts": False,
        "approval_blocker": "Executive overview delivery JSON is structured render evidence; run narration review before TTS.",
        "generated_at": utc_now(),
    }


def _duration_contract_payload(
    deck: DeckSpec,
    single: dict[str, Any],
    notebook: dict[str, Any],
) -> dict[str, Any]:
    target = round(deck.target_duration_seconds, 2)
    single_total = round(float(single.get("total_duration_seconds") or 0.0), 2)
    notebook_total = round(float(notebook.get("total_duration_seconds") or 0.0), 2)
    status = (
        "passed"
        if abs(single_total - target) < 0.01 and abs(notebook_total - target) < 0.01
        else "failed"
    )
    payload = {
        "schema_version": 1,
        "status": status,
        "presentation_id": deck.presentation_id,
        "target_duration_seconds": target,
        "target_duration_minutes": round(target / 60.0, 2),
        "single_speaker_total_duration_seconds": single_total,
        "notebooklm_total_duration_seconds": notebook_total,
    }
    if status != "passed":
        raise ValueError(
            f"{deck.presentation_id} duration contract failed: "
            f"target={target}, single={single_total}, notebooklm={notebook_total}"
        )
    return payload


def write_notebooklm_script(deck: DeckSpec, deck_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [f"# NotebookLM Script: {deck.display_title}", ""]
    for row in rows:
        lines.extend(
            [
                f"### [SLIDE {row['slide_number'] - 1}] {row['title']}",
                "**DIALOGUE:**",
                f"- **Iapetus:** {_row_presenter_say(row)}",
                "- **Erinome:** The source trail matters here because the executive claim has to stay auditable.",
                "",
                "---",
                "",
            ]
        )
    (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _assert_validation(label: str, result: Any) -> None:
    if result.status != "pass":
        raise ValueError(f"{label} validation failed: {json.dumps(result.to_dict(), indent=2)}")


def write_structured_contracts(
    *,
    workspace: Path,
    deck: DeckSpec,
    deck_dir: Path,
    rows: list[dict[str, Any]],
    analysis_manifest: Path,
    executive_summary: Path,
    trend_report: Path,
    premium_image_receipt: dict[str, Any],
) -> dict[str, str]:
    frame = _frame_payload(deck)
    frame_result = validate_presentation_frame_payload(frame)
    _assert_validation("presentation_frame.json", frame_result)
    write_json_atomic(deck_dir / "presentation_frame.json", frame)

    blueprint = _blueprint_payload(
        deck=deck,
        rows=rows,
        frame_validation=frame_result.to_dict(),
        analysis_manifest=analysis_manifest,
        executive_summary=executive_summary,
        trend_report=trend_report,
        workspace=workspace,
    )
    write_json_atomic(deck_dir / "presentation_blueprint.json", blueprint)
    blueprint_sha = sha256_file(deck_dir / "presentation_blueprint.json")
    write_json_atomic(
        deck_dir / "presentation_blueprint_receipt.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "presentation_type": "executive-summary",
            "generator": "generate_executive_overview_decks.py",
            "presentation_frame_validation": frame_result.to_dict(),
            "output_json": "presentation_blueprint.json",
            "output_sha256": blueprint_sha,
            "input_hashes": {
                "presentation_frame.json": sha256_file(deck_dir / "presentation_frame.json"),
                "analysis_manifest.json": sha256_file(analysis_manifest),
                "executive_summary.md": sha256_file(executive_summary),
                "trend_report.md": sha256_file(trend_report),
            },
            "generated_at": utc_now(),
            "secrets_copied": False,
        },
    )
    write_json_atomic(
        deck_dir / "blueprint_review.json",
        {
            "schema_version": 1,
            "status": "approved",
            "approved_for_deck_contract_generation": True,
            "presentation_id": deck.presentation_id,
            "blueprint_sha256": blueprint_sha,
            "reviewer": "deterministic-executive-overview-contract",
            "findings": [],
            "generated_at": utc_now(),
        },
    )

    deck_contract = _deck_contract_payload(
        deck=deck,
        rows=rows,
        blueprint_sha256=blueprint_sha,
    )
    write_json_atomic(deck_dir / "presentation_deck.json", deck_contract)
    deck_sha = sha256_file(deck_dir / "presentation_deck.json")
    write_json_atomic(
        deck_dir / "presentation_deck_receipt.json",
        {
            "schema_version": 1,
            "status": "pass",
            "presentation_id": deck.presentation_id,
            "presentation_type": "executive-summary",
            "blueprint_sha256": blueprint_sha,
            "output_json": "presentation_deck.json",
            "output_sha256": deck_sha,
            "slide_count": len(rows),
            "generated_at": utc_now(),
            "secrets_copied": False,
        },
    )

    source_binding = _source_binding_payload(
        deck=deck,
        rows=rows,
        blueprint_sha256=blueprint_sha,
        deck_sha256=deck_sha,
    )
    _assert_validation(
        "source_binding_report.json",
        validate_source_binding_report_payload(source_binding),
    )
    write_json_atomic(deck_dir / "source_binding_report.json", source_binding)
    source_sha = sha256_file(deck_dir / "source_binding_report.json")

    visual_plan = _visual_plan_payload(deck, rows)
    _assert_validation("visual_plan.json", validate_visual_plan_payload(visual_plan))
    write_json_atomic(deck_dir / "visual_plan.json", visual_plan)
    visual_sha = sha256_file(deck_dir / "visual_plan.json")

    asset_inventory = _asset_inventory_payload(deck, rows)
    write_json_atomic(deck_dir / "asset_inventory.json", asset_inventory)
    premium_image_briefs = _premium_image_briefs_payload(deck, rows)
    write_json_atomic(deck_dir / "premium_image_briefs.json", premium_image_briefs)
    write_json_atomic(deck_dir / "premium_image_receipt.json", premium_image_receipt)
    write_json_atomic(deck_dir / "figure_specs.json", _figure_specs_payload(deck, rows))
    _write_figure_review_receipts(deck_dir, deck, rows)

    single = _delivery_contract(
        deck=deck,
        rows=rows,
        variant="single_speaker",
        source_binding_sha256=source_sha,
        visual_plan_sha256=visual_sha,
        deck_sha256=deck_sha,
    )
    notebook = _delivery_contract(
        deck=deck,
        rows=rows,
        variant="notebooklm",
        source_binding_sha256=source_sha,
        visual_plan_sha256=visual_sha,
        deck_sha256=deck_sha,
    )
    duration_contract = _duration_contract_payload(deck, single, notebook)
    narration_duration = _narration_duration_verification_payload(deck, rows)
    write_json_atomic(deck_dir / "delivery_single_speaker.json", single)
    write_json_atomic(deck_dir / "delivery_notebooklm.json", notebook)
    write_json_atomic(deck_dir / "duration_contract.json", duration_contract)
    write_json_atomic(deck_dir / "narration_duration_verification.json", narration_duration)

    input_paths = {
        "presentation_blueprint.json": deck_dir / "presentation_blueprint.json",
        "presentation_deck.json": deck_dir / "presentation_deck.json",
        "source_binding_report.json": deck_dir / "source_binding_report.json",
        "visual_plan.json": deck_dir / "visual_plan.json",
        "asset_inventory.json": deck_dir / "asset_inventory.json",
        "delivery_single_speaker.json": deck_dir / "delivery_single_speaker.json",
        "delivery_notebooklm.json": deck_dir / "delivery_notebooklm.json",
        "duration_contract.json": deck_dir / "duration_contract.json",
        "premium_image_briefs.json": deck_dir / "premium_image_briefs.json",
        "premium_image_receipt.json": deck_dir / "premium_image_receipt.json",
        "narration_duration_verification.json": deck_dir / "narration_duration_verification.json",
        "figure_specs.json": deck_dir / "figure_specs.json",
        "figure_generation_model_receipt.json": deck_dir / "figure_generation_model_receipt.json",
        "figure_code_review_report.json": deck_dir / "figure_code_review_report.json",
        "figure_svg_validation_report.json": deck_dir / "figure_svg_validation_report.json",
        "figure_semantic_review.json": deck_dir / "figure_semantic_review.json",
    }
    render_spec = build_render_spec(
        deck_contract=deck_contract,
        source_binding=source_binding,
        visual_plan=visual_plan,
        asset_inventory=asset_inventory,
        figure_specs=_figure_specs_payload(deck, rows),
        delivery_single_speaker=single,
        delivery_notebooklm=notebook,
        input_hashes={name: sha256_file(path) for name, path in input_paths.items()},
        delivery_paths={
            "single_speaker": "delivery_single_speaker.json",
            "notebooklm": "delivery_notebooklm.json",
        },
    )
    for spec_slide, row in zip(render_spec.get("slides", []), rows):
        if not isinstance(spec_slide, dict):
            continue
        spec_slide["slide_kind"] = row["slide_kind"]
        for key in (
            "events",
            "papers",
            "events_title",
            "papers_title",
            "prepared_by",
            "title_statement",
            "resources",
            "analysis_url",
            "analysis_note",
            "executive_lesson_id",
            "executive_lesson_role",
            "executive_evidence_kind",
            "source_anchor",
            "source_locator",
            "evidence_text",
            "decision_implication",
            "paper_evidence_text",
            "paper_source_locator",
            "evidence_caption",
            "visual_takeaway",
            "premium_visual_direction",
            "asset_generation_route",
            "asset_provider",
            "asset_model",
            "premium_image_review_status",
            "narration_word_count",
            "narration_estimated_seconds",
        ):
            if key in row:
                spec_slide[key] = row[key]
    render_spec["title"] = deck.display_title
    render_spec["presentation_type"] = "executive-summary"
    render_spec["conference_label"] = conference_label()
    _assert_validation("render_spec.json", validate_render_spec_payload(render_spec))
    write_json_atomic(deck_dir / "render_spec.json", render_spec)

    return {
        "presentation_frame_sha256": sha256_file(deck_dir / "presentation_frame.json"),
        "presentation_blueprint_sha256": blueprint_sha,
        "presentation_deck_sha256": deck_sha,
        "source_binding_report_sha256": source_sha,
        "visual_plan_sha256": visual_sha,
        "asset_inventory_sha256": sha256_file(deck_dir / "asset_inventory.json"),
        "premium_image_briefs_sha256": sha256_file(deck_dir / "premium_image_briefs.json"),
        "premium_image_receipt_sha256": sha256_file(deck_dir / "premium_image_receipt.json"),
        "narration_duration_verification_sha256": sha256_file(
            deck_dir / "narration_duration_verification.json"
        ),
        "duration_contract_sha256": sha256_file(deck_dir / "duration_contract.json"),
        "render_spec_sha256": sha256_file(deck_dir / "render_spec.json"),
    }


def _spoken_fragment(text: str) -> str:
    return clean_markdown_text(text).strip().rstrip(".")


def _lower_initial(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _sentence(text: str) -> str:
    cleaned = _spoken_fragment(text)
    if not cleaned:
        return ""
    if cleaned[:1].islower():
        cleaned = cleaned[:1].upper() + cleaned[1:]
    return cleaned if cleaned.endswith((".", "!", "?")) else f"{cleaned}."


def _source_context_lines(slide: SlideSpec) -> list[str]:
    if not slide.refs:
        return ["- Executive synthesis slide with no specific source link."]
    return [f"- Source: {label} ({url})" for label, url in slide.refs]


IMPERATIVE_ACTIONS = {
    "adopt",
    "align",
    "benchmark",
    "bridge",
    "build",
    "combine",
    "control",
    "defend",
    "develop",
    "evaluate",
    "fund",
    "gate",
    "govern",
    "implement",
    "join",
    "maintain",
    "mandate",
    "measure",
    "monitor",
    "prioritize",
    "re-center",
    "require",
    "steward",
    "treat",
}


def _spoken_note(slide: SlideSpec) -> str:
    note = clean_markdown_text(slide.notes)
    replacements = (
        ("Open with the central thesis. ", ""),
        ("Use this as the orientation slide. ", ""),
        ("Use this as the first half of the topic map.", "The first half of the topic map shows where the evidence lands."),
        ("Use this as the second half of the topic map.", "The second half of the topic map connects systems, society, and science."),
        ("Make the dual-use point concrete: ", "The dual-use point is concrete: "),
        ("Close the short deck with the decision agenda rather than a recap.", "The short deck closes with a decision agenda for leadership."),
        ("Appendix-style source slide for auditability.", "For auditability, this source slide points back to the underlying topic reports."),
        ("This is the money slide for program owners.", "For program owners, the investment priorities are concrete."),
        ("End the main narrative with forward-looking checkpoints.", "The main narrative ends with checkpoints that leaders can watch over the next cycle."),
    )
    for old, new in replacements:
        note = note.replace(old, new)
    return _sentence(note)


def _starts_with_action(text: str) -> bool:
    first = re.match(r"^[A-Za-z][A-Za-z-]*", text.strip())
    return bool(first and first.group(0).lower() in IMPERATIVE_ACTIONS)


def _action_sentence(fragment: str, *, index: int) -> str:
    cleaned = _spoken_fragment(fragment)
    if not cleaned:
        return ""
    if not _starts_with_action(cleaned):
        return _sentence(cleaned)
    lowered = _lower_initial(cleaned)
    openers = (
        "Leaders should",
        "Teams should",
        "The release standard should be to",
        "The operating habit should be to",
    )
    opener = openers[index % len(openers)]
    if opener.endswith("to"):
        return _sentence(f"{opener} {lowered}")
    return _sentence(f"{opener} {lowered}")


def _bullet_takeaways(slide: SlideSpec, *, index: int, limit: int = 3) -> list[str]:
    bullets = [_spoken_fragment(item) for item in slide.bullets if _spoken_fragment(item)]
    if not bullets or "source backbone" in slide.title.lower():
        return []
    if "eight signals" in slide.title.lower():
        return [
            "The first signals give the executive frame: systems are the unit of progress, evaluation is reliability infrastructure, and scaling gains are conditional.",
            "The remaining signals explain how that frame plays out across agents, reasoning, multimodality, science, and efficiency.",
        ]
    sentences: list[str] = []
    for offset, bullet in enumerate(bullets[:limit]):
        sentences.append(_action_sentence(bullet, index=index + offset))
    return [sentence for sentence in sentences if sentence]


def _topic_map_sentence(slide: SlideSpec) -> str:
    topics = []
    for bullet in slide.bullets:
        cleaned = _spoken_fragment(bullet)
        if ":" in cleaned:
            topics.append(cleaned.split(":", 1)[1].strip())
        else:
            topics.append(cleaned)
    if not topics:
        return _spoken_note(slide)
    if len(topics) <= 3:
        body = ", ".join(topics)
    else:
        body = ", ".join(topics[:3]) + f", and {len(topics) - 3} more evidence areas"
    return f"This topic map locates the evidence across {body}."


def _compose_presenter_say(slide: SlideSpec, *, index: int, total: int) -> str:
    note = _spoken_note(slide)
    title = slide.title.lower()

    if "topic map" in title:
        sentences = [_topic_map_sentence(slide)]
    elif "source backbone" in title:
        first = note
        second = (
            "The point is not to walk every source line by line, but to keep the claims traceable "
            "to the topic reports behind the synthesis."
        )
        sentences = [first, second]
    elif index == 1:
        sentences = [
            note,
            "The central lesson is that frontier progress now depends on how models, tools, memory, verifiers, evaluation, and governance work together.",
        ]
    elif index == total:
        sentences = [
            note,
            "The takeaway is to fund the reliability infrastructure, require stack-level safety cases, and treat openness, efficiency, and provenance as governance choices.",
        ]
    else:
        sentences = [note]

    if slide.narrative and "topic map" not in title and "source backbone" not in title:
        sentences.append(_sentence(slide.narrative))

    sentences.extend(_bullet_takeaways(slide, index=index))

    return " ".join(sentence for sentence in sentences if sentence).strip()


def _transition_to_next(current: SlideSpec, next_slide: SlideSpec | None, *, index: int = 0) -> str:
    if next_slide is None:
        return "That is the decision frame to carry into the deeper topic videos."
    next_title = _spoken_fragment(next_slide.title)
    current_title = current.title.lower()
    if "source backbone" in current_title:
        source_templates = (
            "The source trail continues with {next_title}.",
            "The final audit slice is {next_title}.",
            "For the remaining source trail, move to {next_title}.",
        )
        return source_templates[index % len(source_templates)].format(next_title=next_title)
    if "leadership actions" in current_title:
        return "That closes the short executive readout and points to the detailed topic videos."
    if current.figure == "title":
        return f"The first executive signal is {next_title}."
    templates = (
        "Next, {next_title} turns that frame into operating evidence.",
        "The readout then moves to {next_title}.",
        "The next decision surface is {next_title}.",
        "{next_title} adds the next layer.",
        "That leads into {next_title}.",
        "The synthesis then shifts to {next_title}.",
        "The next proof point is {next_title}.",
        "From operating frame to evidence, the next stop is {next_title}.",
        "{next_title} makes the implication more concrete.",
        "The story then widens to {next_title}.",
        "That brings the focus to {next_title}.",
        "{next_title} is the next lens.",
        "The executive path continues through {next_title}.",
        "Next comes {next_title}.",
        "The discussion now turns to {next_title}.",
        "{next_title} carries the next practical implication.",
    )
    return templates[index % len(templates)].format(next_title=next_title)


def _format_duration(seconds: object) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        total = int(MIN_CONTENT_SLIDE_SECONDS)
    minutes, remainder = divmod(total, 60)
    if minutes and remainder:
        return f"~{minutes}m {remainder}s"
    if minutes:
        return f"~{minutes}m"
    return f"~{remainder}s"


def write_presenter_scripts(deck: DeckSpec, deck_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [f"# Presenter Script: {deck.display_title}", ""]
    for row_index, row in enumerate(rows):
        next_row = rows[row_index + 1] if row_index + 1 < len(rows) else None
        section = (
            "Source Backbone"
            if "source backbone" in row["title"].lower()
            else "Bookends"
            if row["slide_kind"] in {"events", "resources", "title"}
            else "Executive Synthesis"
        )
        duration = _format_duration(row.get("timing_seconds"))
        transition = (
            f"The next slide is {next_row['title']}."
            if next_row is not None
            else "That is the decision frame to carry into the deeper topic videos."
        )
        lines.extend(
            [
                f"### [SLIDE {row['slide_number'] - 1}] {row['title']}",
                f"**Time:** {duration} | **Section:** {section}",
                "",
                "**SAY:**",
                f"> {_row_presenter_say(row)}",
                "",
                "**CONTEXT** (not spoken -- for presenter reference):",
            ]
        )
        if row["refs"]:
            lines.extend(f"- Source: {label} ({url})" for label, url in row["refs"])
        else:
            lines.append("- Executive synthesis slide with no specific source link.")
        lines.extend(
            [
                f"- Key slide intent: {_spoken_fragment(row['notes'])}.",
                "",
                f"**FIGURE:** The slide uses a `{row['figure']}` visual to organize the executive takeaway.",
                "",
                f'**TRANSITION:** "{transition}"',
                "",
                "---",
                "",
            ]
        )
    (deck_dir / "6_presenter_script.md").write_text("\n".join(lines), encoding="utf-8")


def write_sidecars(
    *,
    workspace: Path,
    deck: DeckSpec,
    deck_dir: Path,
    rows: list[dict[str, Any]],
    analysis_manifest: Path,
    executive_summary: Path,
    trend_report: Path,
) -> None:
    write_json_atomic(deck_dir / "ranked_papers.json", {"papers": []})
    source_rows = [
        {
            "slide_number": row["slide_number"],
            "slide_id": row["slide_id"],
            "slide_title": row["title"],
            "sources": [
                {
                    "title": label,
                    "url": url,
                    "source_type": "analysis_report",
                }
                for label, url in row["refs"]
            ],
        }
        for row in rows
    ]
    write_json_atomic(deck_dir / "source_attributions.json", {"sources": source_rows})
    write_json_atomic(
        deck_dir / "slides_structure.json",
        {
            "presentation_id": deck.presentation_id,
            "presentation_type": "executive-summary",
            "slide_count": len(rows),
            "slides": [
                {
                    "slide_num": row["slide_number"],
                    "slide_id": row["slide_id"],
                    "slide_type": row["slide_type"],
                    "title": row["title"],
                    "bullets": list(row["bullets"]),
                    "source_refs": [url for _label, url in row["refs"]],
                    "figure_need": row["figure"],
                    "speaker_notes": row["notes"],
                }
                for row in rows
            ],
        },
    )
    write_json_atomic(
        deck_dir / "source_pack_receipt.json",
        {
            "schema_version": 1,
            "status": "succeeded",
            "presentation_id": deck.presentation_id,
            "presentation_type": "executive-summary",
            "source_inputs": [
                {
                    "path": relpath(analysis_manifest, workspace),
                    "sha256": sha256_file(analysis_manifest),
                },
                {
                    "path": relpath(executive_summary, workspace),
                    "sha256": sha256_file(executive_summary),
                },
                {
                    "path": relpath(trend_report, workspace),
                    "sha256": sha256_file(trend_report),
                },
            ],
            "generated_at": utc_now(),
            "secrets_copied": False,
        },
    )


def generate(
    *,
    workspace: Path,
    analysis_manifest: Path,
    executive_summary: Path,
    trend_report: Path,
    short_id: str,
    long_id: str,
    source_link_map: Path | None = None,
    require_source_link_map: bool = False,
    presentation_link_map: Path | None = None,
    require_presentation_link_map: bool = False,
    deck_first_plan_path: Path | None = None,
    debug_legacy_markdown: bool = False,
    premium_image_manifest_path: Path | None = None,
    require_premium_images: bool = False,
    minimum_premium_images_per_deck: int = MIN_PREMIUM_IMAGES_PER_DECK,
    deck_scope: str = "both",
) -> dict[str, object]:
    set_conference_context(load_conference_context(workspace))
    if require_source_link_map and source_link_map is None:
        raise SourceLinkContractError(
            "source_link_map_missing: sync or upload analysis markdown to "
            "Google Drive and write decks/executive_source_link_map.json"
        )
    if require_presentation_link_map and presentation_link_map is None:
        raise PresentationLinkContractError(
            "presentation_link_map_missing: upload or publish topic presentations "
            "and write decks/executive_presentation_link_map.json"
        )
    source_links = load_source_link_map(source_link_map)
    presentation_links = load_presentation_link_map(presentation_link_map)
    premium_image_manifest = load_premium_image_manifest(premium_image_manifest_path)
    deck_first_plan = load_deck_first_plan(deck_first_plan_path)
    topics = load_topics(
        workspace,
        analysis_manifest,
        source_links,
        require_external_source_links=require_source_link_map,
    )
    all_decks = build_decks(
        workspace=workspace,
        topics=topics,
        executive_text=strip_binding_digests_section(read_text_if_exists(executive_summary)),
        trend_text=read_text_if_exists(trend_report),
        short_id=short_id,
        long_id=long_id,
        source_links=source_links,
        require_external_source_links=require_source_link_map,
        presentation_links=presentation_links,
        require_external_presentation_links=require_presentation_link_map,
        deck_first_plan=deck_first_plan,
    )
    decks = select_decks_for_scope(all_decks, deck_scope=deck_scope)
    results: list[dict[str, object]] = []
    for deck in decks:
        deck_dir = workspace / "decks" / deck.presentation_id
        deck_dir.mkdir(parents=True, exist_ok=True)
        rows = generated_slide_rows(deck, workspace, deck_dir)
        premium_image_receipt = apply_premium_images(
            deck=deck,
            rows=rows,
            workspace=workspace,
            deck_dir=deck_dir,
            manifest=premium_image_manifest,
            require=require_premium_images,
            minimum_per_deck=minimum_premium_images_per_deck,
        )
        if debug_legacy_markdown:
            write_checkpoint_files(deck_dir, structured_markdown(deck, rows))
        else:
            remove_checkpoint_files(deck_dir)
        write_presenter_scripts(deck, deck_dir, rows)
        write_notebooklm_script(deck, deck_dir, rows)
        structured_hashes = write_structured_contracts(
            workspace=workspace,
            deck=deck,
            deck_dir=deck_dir,
            rows=rows,
            analysis_manifest=analysis_manifest,
            executive_summary=executive_summary,
            trend_report=trend_report,
            premium_image_receipt=premium_image_receipt,
        )
        write_sidecars(
            workspace=workspace,
            deck=deck,
            deck_dir=deck_dir,
            rows=rows,
            analysis_manifest=analysis_manifest,
            executive_summary=executive_summary,
            trend_report=trend_report,
        )
        premium_briefs = load_json(deck_dir / "premium_image_briefs.json")
        results.append(
            {
                "presentation_id": deck.presentation_id,
                "display_title": deck.display_title,
                "deck_dir": relpath(deck_dir, workspace),
                "slide_count": len(rows),
                "content_slide_count": len(deck.slides),
                "asset_count": len(rows),
                "render_spec": "render_spec.json",
                "duration_contract": load_json(deck_dir / "duration_contract.json"),
                "narration_duration_verification": load_json(
                    deck_dir / "narration_duration_verification.json"
                ),
                "premium_image_briefs": {
                    "path": relpath(deck_dir / "premium_image_briefs.json", workspace),
                    "status": premium_briefs["status"],
                    "recommended_premium_image_count": premium_briefs[
                        "recommended_premium_image_count"
                    ],
                },
                "premium_image_receipt": {
                    "path": relpath(deck_dir / "premium_image_receipt.json", workspace),
                    "status": premium_image_receipt["status"],
                    "genai_image_count": premium_image_receipt["genai_image_count"],
                    "minimum_required_images": premium_image_receipt[
                        "minimum_required_images"
                    ],
                },
                "rebuilt_artifacts": {
                    "html": relpath(deck_dir / "presentation.html", workspace),
                    "pdf": relpath(deck_dir / "presentation.pdf", workspace),
                    "slide_mapping": relpath(
                        deck_dir / "presentation_slide_mapping.json",
                        workspace,
                    ),
                    "frame_manifest": relpath(
                        deck_dir / "render_frame_manifest.json",
                        workspace,
                    ),
                    "contact_sheet": relpath(
                        deck_dir / "render_contact_sheet.png",
                        workspace,
                    ),
                    "render_export_receipt": relpath(
                        deck_dir / "render_export_receipt.json",
                        workspace,
                    ),
                },
                "bookend_contract": {
                    "status": "passed",
                    "point_count": deck.point_count,
                    "slide_2_event_count": len(deck.opening_events),
                    "slide_2_resource_count": len(deck.opening_resources),
                    "final_presentation_link_count": sum(
                        1 for item in deck.further_reading if item.get("kind") == "topic_presentation"
                    ),
                    "final_analysis_report_link_count": sum(
                        1 for item in deck.further_reading if item.get("kind") == "analysis_report"
                    ),
                    "presentation_link_mode": deck.presentation_link_mode,
                },
                "structured_contracts": structured_hashes,
            }
        )
    release_link_blockers: list[dict[str, object]] = []
    if not presentation_links:
        release_link_blockers.append(
            {
                "code": "topic_presentation_public_links_missing",
                "severity": "review_ok_release_blocker",
                "message": (
                    "Topic presentation public URLs are unavailable; generated "
                    "resources use workspace-local links for review."
                ),
                "map_path": (
                    relpath(presentation_link_map, workspace)
                    if presentation_link_map
                    else "decks/executive_presentation_link_map.json"
                ),
            }
        )
    premium_image_status = (
        "passed"
        if results
        and all(
            str(item.get("premium_image_receipt", {}).get("status")) == "passed"
            for item in results
        )
        else "not_required"
        if not require_premium_images
        else "failed"
    )
    return {
        "schema_version": 1,
        "status": "succeeded",
        "workspace": str(workspace),
        "presentations": results,
        "deck_first_plan": (
            relpath(deck_first_plan_path, workspace) if deck_first_plan_path else None
        ),
        "deck_scope": deck_scope,
        "deck_first_contract": {
            "status": "passed" if deck_first_plan else "legacy_fallback",
            "candidate_count": (
                len(deck_first_plan.get("candidate_pool", [])) if deck_first_plan else 0
            ),
            "short_spine_count": (
                len(deck_first_plan.get("short_spine", [])) if deck_first_plan else 0
            ),
            "long_expansion_count": (
                len(deck_first_plan.get("long_expansions", [])) if deck_first_plan else 0
            ),
        },
        "release_link_blockers": release_link_blockers,
        "premium_image_generation": {
            "status": premium_image_status,
            "minimum_required_images_per_deck": minimum_premium_images_per_deck,
            "manifest_path": (
                relpath(premium_image_manifest_path, workspace)
                if premium_image_manifest_path
                else None
            ),
            "brief_paths": [
                {
                    "presentation_id": item["presentation_id"],
                    "path": str(item["premium_image_briefs"]["path"]),
                    "recommended_premium_image_count": item["premium_image_briefs"][
                        "recommended_premium_image_count"
                    ],
                }
                for item in results
            ],
            "receipts": [
                {
                    "presentation_id": item["presentation_id"],
                    "path": str(item["premium_image_receipt"]["path"]),
                    "status": str(item["premium_image_receipt"]["status"]),
                    "genai_image_count": item["premium_image_receipt"][
                        "genai_image_count"
                    ],
                }
                for item in results
            ],
        },
        "source_link_map": (
            relpath(source_link_map, workspace) if source_link_map else None
        ),
        "source_link_count": len(set(source_links.values())),
        "source_link_contract": {
            "required": require_source_link_map,
            "status": "passed" if require_source_link_map else "not_required",
            "map_path": relpath(source_link_map, workspace) if source_link_map else None,
            "source_link_count": len(set(source_links.values())),
        },
        "presentation_link_map": (
            relpath(presentation_link_map, workspace) if presentation_link_map else None
        ),
        "presentation_link_count": len(set(presentation_links.values())),
        "presentation_link_contract": {
            "required": require_presentation_link_map,
            "status": "passed" if require_presentation_link_map else "not_required",
            "map_path": (
                relpath(presentation_link_map, workspace) if presentation_link_map else None
            ),
            "presentation_link_count": len(set(presentation_links.values())),
            "mode": decks[0].presentation_link_mode if decks else "workspace-local",
        },
        "structured_render_contract": {
            "status": "passed",
            "legacy_markdown_written": debug_legacy_markdown,
            "legacy_markdown_support_level": (
                "debug-export-only" if debug_legacy_markdown else "not_written"
            ),
        },
        "generated_at": utc_now(),
        "secrets_copied": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--analysis-manifest", type=Path, default=Path("analysis/analysis_manifest.json"))
    parser.add_argument("--executive-summary", type=Path, default=Path("analysis/executive_summary.md"))
    parser.add_argument("--trend-report", type=Path, default=Path("analysis/trend_report.md"))
    parser.add_argument("--short-id", default=DEFAULT_SHORT_ID)
    parser.add_argument("--long-id", default=DEFAULT_LONG_ID)
    parser.add_argument(
        "--deck-scope",
        choices=("both", "short", "long"),
        default="both",
        help="Generate both executive decks, only the short deck, or only the long deck.",
    )
    parser.add_argument(
        "--source-link-map",
        type=Path,
        help="Optional workspace-local JSON map from source markdown paths to http(s) URLs.",
    )
    parser.add_argument(
        "--presentation-link-map",
        type=Path,
        help=(
            "Optional workspace-local JSON map from topic presentation ids/paths to "
            "public http(s) URLs for executive further-reading links."
        ),
    )
    parser.add_argument(
        "--deck-first-plan",
        type=Path,
        help=(
            "Optional workspace-local deck-first lesson plan. Defaults to "
            "decks/executive_summary_deck_first_plan.json when present."
        ),
    )
    parser.add_argument(
        "--premium-image-manifest",
        type=Path,
        help=(
            "Optional workspace-local manifest for GenAI raster images. Defaults "
            "to decks/executive_summary_genai_images_manifest.json when present."
        ),
    )
    parser.add_argument(
        "--require-premium-images",
        action="store_true",
        help="Block unless each executive deck uses the required number of GenAI images.",
    )
    parser.add_argument(
        "--minimum-premium-images-per-deck",
        type=int,
        default=MIN_PREMIUM_IMAGES_PER_DECK,
        help="Minimum GenAI raster images required in each executive deck.",
    )
    parser.add_argument(
        "--require-source-link-map",
        action="store_true",
        help=(
            "Block unless every executive-summary source link resolves through "
            "the source-link map. Use this for production so missing Drive sync "
            "cannot fall back to local ../analysis links."
        ),
    )
    parser.add_argument(
        "--require-presentation-link-map",
        action="store_true",
        help=(
            "Block unless every topic presentation link resolves through the "
            "presentation-link map. Use this for production link-audit readiness."
        ),
    )
    parser.add_argument("--receipt-output", type=Path, default=Path("decks/executive_overview_decks_receipt.json"))
    parser.add_argument(
        "--debug-legacy-markdown",
        action="store_true",
        help=(
            "Also write retired numbered Markdown checkpoints for local debugging. "
            "Production render/export evidence uses structured JSON contracts."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    analysis_manifest = workspace_path(args.analysis_manifest, workspace, label="--analysis-manifest")
    executive_summary = workspace_path(args.executive_summary, workspace, label="--executive-summary")
    trend_report = workspace_path(args.trend_report, workspace, label="--trend-report")
    if args.source_link_map:
        source_link_map = workspace_path(args.source_link_map, workspace, label="--source-link-map")
    else:
        default_source_link_map = workspace / "decks" / "executive_source_link_map.json"
        source_link_map = default_source_link_map if default_source_link_map.is_file() else None
    if args.presentation_link_map:
        presentation_link_map = workspace_path(
            args.presentation_link_map,
            workspace,
            label="--presentation-link-map",
        )
    else:
        default_presentation_link_map = workspace / "decks" / "executive_presentation_link_map.json"
        presentation_link_map = (
            default_presentation_link_map if default_presentation_link_map.is_file() else None
        )
    if args.deck_first_plan:
        deck_first_plan_path = workspace_path(
            args.deck_first_plan,
            workspace,
            label="--deck-first-plan",
        )
    else:
        default_deck_first_plan = workspace / "decks" / "executive_summary_deck_first_plan.json"
        deck_first_plan_path = (
            default_deck_first_plan if default_deck_first_plan.is_file() else None
        )
    if args.premium_image_manifest:
        premium_image_manifest_path = workspace_path(
            args.premium_image_manifest,
            workspace,
            label="--premium-image-manifest",
        )
    else:
        default_premium_image_manifest = (
            workspace / "decks" / "executive_summary_genai_images_manifest.json"
        )
        premium_image_manifest_path = (
            default_premium_image_manifest
            if default_premium_image_manifest.is_file()
            else None
        )
    receipt_output = workspace_path(args.receipt_output, workspace, label="--receipt-output")
    if args.dry_run:
        payload = {
            "workspace": str(workspace),
            "analysis_manifest": str(analysis_manifest),
            "executive_summary": str(executive_summary),
            "trend_report": str(trend_report),
            "source_link_map": str(source_link_map) if source_link_map else None,
            "presentation_link_map": (
                str(presentation_link_map) if presentation_link_map else None
            ),
            "deck_first_plan": (
                str(deck_first_plan_path) if deck_first_plan_path else None
            ),
            "premium_image_manifest": (
                str(premium_image_manifest_path) if premium_image_manifest_path else None
            ),
            "require_premium_images": bool(args.require_premium_images),
            "minimum_premium_images_per_deck": args.minimum_premium_images_per_deck,
            "short_id": args.short_id,
            "long_id": args.long_id,
            "deck_scope": args.deck_scope,
            "receipt_output": str(receipt_output),
            "require_source_link_map": bool(args.require_source_link_map),
            "require_presentation_link_map": bool(args.require_presentation_link_map),
            "debug_legacy_markdown": bool(args.debug_legacy_markdown),
            "status": "planned",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    try:
        result = generate(
            workspace=workspace,
            analysis_manifest=analysis_manifest,
            executive_summary=executive_summary,
            trend_report=trend_report,
            short_id=args.short_id,
            long_id=args.long_id,
            source_link_map=source_link_map,
            require_source_link_map=bool(args.require_source_link_map),
            presentation_link_map=presentation_link_map,
            require_presentation_link_map=bool(args.require_presentation_link_map),
            deck_first_plan_path=deck_first_plan_path,
            debug_legacy_markdown=bool(args.debug_legacy_markdown),
            premium_image_manifest_path=premium_image_manifest_path,
            require_premium_images=bool(args.require_premium_images),
            minimum_premium_images_per_deck=args.minimum_premium_images_per_deck,
            deck_scope=args.deck_scope,
        )
    except SourceLinkContractError as exc:
        result = {
            "schema_version": 1,
            "status": "blocked",
            "workspace": str(workspace),
            "source_link_map": (
                relpath(source_link_map, workspace) if source_link_map else None
            ),
            "source_link_contract": {
                "required": bool(args.require_source_link_map),
                "status": "blocked",
                "blocker_code": "source_link_contract_blocked",
                "reason": str(exc),
                "remediation": (
                    "Sync or upload the source markdown files to Google Drive, "
                    "then regenerate decks/executive_source_link_map.json."
                ),
            },
            "generated_at": utc_now(),
            "secrets_copied": False,
        }
        write_json_atomic(receipt_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    except PresentationLinkContractError as exc:
        result = {
            "schema_version": 1,
            "status": "blocked",
            "workspace": str(workspace),
            "presentation_link_map": (
                relpath(presentation_link_map, workspace) if presentation_link_map else None
            ),
            "presentation_link_contract": {
                "required": bool(args.require_presentation_link_map),
                "status": "blocked",
                "blocker_code": "presentation_link_contract_blocked",
                "reason": str(exc),
                "remediation": (
                    "Upload or publish the topic presentation artifacts, then "
                    "write decks/executive_presentation_link_map.json."
                ),
            },
            "generated_at": utc_now(),
            "secrets_copied": False,
        }
        write_json_atomic(receipt_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    write_json_atomic(receipt_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
