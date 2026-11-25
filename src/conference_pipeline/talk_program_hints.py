"""Program context for naming talks inside a long multi-talk recording.

Talk-boundary discovery works from the transcript itself; these hints only
help the model name what it finds (workshop title, accepted-paper titles,
keynote mentions scraped from the workshop site). Hints are optional and
never forced: the prompt tells the model to use a hint only when the
transcript clearly matches it.

Conference-neutral: the workshop schedule rows are matched by any id field
named ``workshop_id`` or ``<slug>_workshop_id``; event ids and categories are
read from the event-page URL path.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from conference_pipeline.acquisition.youtube_talks import title_similarity
from conference_pipeline.indexing.token_policy import token_count

WORKSHOP_SITES_RELATIVE = Path("storage/workshops/workshop_external_sites.json")
KEYNOTE_EXTRACTION_RELATIVE = Path("storage/workshops/workshop_keynote_extraction.json")
GAP_RESEARCH_RELATIVE = Path("storage/workshops/gap-research")
# Typed event pages name their category in the path; hyphenated slugs such as
# ``invited-talk``, ``affinity-event`` or ``town-hall`` count as one category.
# Bare-ID pages (``/virtual/<year>/<digits>``, how ICML 2026 links its
# tutorials and expo items) carry no category in the URL: they resolve to
# category ``item`` here and to a real kind from the cached page title
# (``program_coverage.classify_item_page_title_kind``) by the caller.
EVENT_URL_RE = re.compile(
    r"/(workshop|oral|tutorial|poster|session|meeting|invited-talk|invited-speaker|invited|keynote|spotlight|"
    r"affinity-event|affinity|town-hall|test-of-time|expo|event)s?/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
BARE_ITEM_URL_RE = re.compile(r"/virtual/(\d{4})/(\d+)/?(?:[?#]|$)")
MAX_PAPER_TITLES = 60
MAX_KEYNOTE_LINES = 20
MAX_KEYNOTE_LINE_CHARS = 200
MAX_HINT_TOKENS = 3000
WORKSHOP_TITLE_MATCH_THRESHOLD = 0.5
_TAG_RE = re.compile(r"<[^>]*>|[a-zA-Z-]+=\"[^\"]*\"")
_KEYNOTE_WORD_RE = re.compile(r"\b(keynote|invited|speaker|talk|panel)\b", re.IGNORECASE)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _rows(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        payload = payload.get("workshops") or payload.get("rows") or payload.get("items") or []
    return [row for row in payload or [] if isinstance(row, Mapping)]


def _row_workshop_id(row: Mapping[str, Any]) -> str:
    if row.get("workshop_id"):
        return str(row["workshop_id"])
    for key, value in row.items():
        if key.endswith("_workshop_id") and value:
            return str(value)
    return ""


def parse_event_url(source_url: object) -> tuple[str | None, str | None]:
    """``(category, event_id)`` from an event-page URL, or ``(None, None)``.

    Hyphenated categories are normalised to their head noun family the
    program hints already know (``invited-talk`` -> ``invited``,
    ``affinity-event`` -> ``affinity``); a bare-ID page returns ``("item",
    "<year>-<id>")`` so the caller can resolve its kind from the cached page.
    """

    text = str(source_url or "")
    match = EVENT_URL_RE.search(text)
    if match:
        category = match.group(1).lower()
        category = {"invited-talk": "invited", "invited-speaker": "invited", "affinity-event": "affinity"}.get(
            category, category
        )
        return category, match.group(2)
    bare = BARE_ITEM_URL_RE.search(text)
    if bare:
        return "item", f"{bare.group(1)}-{bare.group(2)}"
    return None, None


def _clean_keynote_context(text: str) -> str:
    cleaned = html.unescape(_TAG_RE.sub(" ", str(text)))
    cleaned = re.sub(r"[<>{}\[\]\"]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_KEYNOTE_LINE_CHARS]


def load_program_hints(
    workspace: Path,
    *,
    source_url: object = None,
    title: str | None = None,
    channel_category: str | None = None,
) -> dict[str, Any]:
    """Return hints for one recording plus the exact prompt block and its digest."""

    category, event_id = parse_event_url(source_url)
    category = category or channel_category
    sites = _rows(_read_json(workspace / WORKSHOP_SITES_RELATIVE))
    workshop: Mapping[str, Any] | None = None
    association: str | None = None
    if category == "workshop" and event_id:
        workshop = next((row for row in sites if _row_workshop_id(row) == event_id), None)
        association = "event_url" if workshop is not None else None
    if workshop is None and title:
        scored = sorted(
            ((title_similarity(title, str(row.get("title") or "")), row) for row in sites),
            key=lambda item: item[0],
            reverse=True,
        )
        if scored and scored[0][0] >= WORKSHOP_TITLE_MATCH_THRESHOLD:
            workshop = scored[0][1]
            association = "fuzzy_title"
    context_workshop_id = _row_workshop_id(workshop) if workshop else None
    context_workshop_title = str(workshop.get("title") or "") if workshop else None
    # Only an event-page association is identity. A fuzzy title match can
    # still feed the prompt (as an unverified guess) but is never stamped into
    # talk maps or chunks: "Mobile AI Workshop" once became "Biometrics".
    authoritative = association == "event_url"
    workshop_id = context_workshop_id if authoritative else None
    workshop_title = context_workshop_title if authoritative else None

    paper_titles: list[str] = []
    keynote_lines: list[str] = []
    if workshop_id:  # paper/keynote lists only for a verified workshop
        gap_dir = workspace / GAP_RESEARCH_RELATIVE
        for path in sorted(gap_dir.glob("*.json")) if gap_dir.is_dir() else []:
            payload = _read_json(path)
            if not isinstance(payload, Mapping) or str(payload.get("schedule_id") or "") != workshop_id:
                continue
            for paper in payload.get("papers") or []:
                paper_title = str((paper or {}).get("title") or "").strip() if isinstance(paper, Mapping) else ""
                if paper_title and paper_title not in paper_titles:
                    paper_titles.append(paper_title)
        paper_titles = paper_titles[:MAX_PAPER_TITLES]
        for row in _rows(_read_json(workspace / KEYNOTE_EXTRACTION_RELATIVE)):
            if _row_workshop_id(row) != workshop_id:
                continue
            for context in row.get("keynote_contexts") or []:
                line = _clean_keynote_context(context)
                if line and _KEYNOTE_WORD_RE.search(line) and line not in keynote_lines:
                    keynote_lines.append(line)
        keynote_lines = keynote_lines[:MAX_KEYNOTE_LINES]

    lines = [f"Recording title: {title or 'unknown'}"]
    if category:
        lines.append(f"Event category: {category}")
    if workshop_title:
        lines.append(f"Workshop: {workshop_title}")
    elif context_workshop_title:
        lines.append(
            f"Possibly related workshop (unverified title match, may be wrong): {context_workshop_title}"
        )
    if paper_titles:
        lines.append("Accepted paper titles of this workshop (use only if clearly presented):")
        lines.extend(f"- {item}" for item in paper_titles)
    if keynote_lines:
        lines.append("Keynote/speaker mentions scraped from the workshop site (noisy):")
        lines.extend(f"- {item}" for item in keynote_lines)
    block = "\n".join(lines)
    while token_count(block) > MAX_HINT_TOKENS and len(lines) > 1:
        lines.pop()
        block = "\n".join(lines)
    return {
        "event_category": category,
        "event_id": event_id,
        "workshop_id": workshop_id,
        "workshop_title": workshop_title,
        "workshop_association": association,
        "candidate_workshop_id": None if authoritative else context_workshop_id,
        "candidate_workshop_title": None if authoritative else context_workshop_title,
        "paper_title_count": len(paper_titles),
        "keynote_line_count": len(keynote_lines),
        "hint_coverage": (
            "workshop_context" if authoritative else "unverified_workshop_context" if context_workshop_id else "title_only"
        ),
        "hint_block": block,
        "hint_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
    }


__all__ = ["load_program_hints", "parse_event_url"]
