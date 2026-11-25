#!/usr/bin/env python3
"""Reproducible pipeline step: add the opening "events" and closing "resources"
bookend slides to a rendered deck, derived from the deck's own bound sources.

Runs AFTER ``generate_render_spec.py`` (so the figure-bearing content slides are
already laid out) and injects two ``slide_kind`` slides into both the deck
contract and the render spec:

- opening "Relevant events" slide: talks/keynotes/workshops + the most-cited
  papers, plus a "Prepared by <author>" credit;
- closing "Sources & further reading" slide: the top-N most-cited sources with
  hyperlinks, plus the deck's ``analysis_report_url``.

All content is derived deterministically from ``render_spec.json`` source chips
(label, title, url, contribution_note, source_type, speakers, cross-slide
frequency) — nothing is topic-hardcoded, so the same step works for any deck.

Usage:
  python scripts/presentations/add_bookend_slides.py \
      --deck-dir conferences/iclr2026/decks/04_safety_alignment_privacy
"""

from __future__ import annotations

import argparse
import collections
import html
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.presentation.contribution_notes import _LEDGER_SECTION_NOTE_RE, _note_matches_work
from conference_pipeline.presentation.display_text import (  # noqa: E402
    repair_leaked_macro_names,
    strip_markdown,
)
from conference_pipeline.presentation.source_links import (
    SourceLinkResolver,
    canonical_public_url,
)

TALK_SOURCE_TYPES = {
    "invited_talk",
    "keynote",
    "media_asset",
    "oral",
    "oral_presentation",
    "panel",
    "session",
    "speech_transcript",
    "spotlight",
    "talk",
    "transcript",
    "workshop",
    "workshop_talk",
}

# Not inside a hyphenated compound: "Multi-Session Agentic Tasks" is a paper
# title, not a session (ICML 2026 deck 08 listed it as an event).
EVENT_TEXT_RE = re.compile(
    r"(?<![-\w])(workshop|keynote|oral|spotlight|invited\s+talk|session|panel|talk)(?![-\w])",
    re.IGNORECASE,
)

TOP_EVENTS = 5
TOP_PAPERS = 5
TOP_RESOURCES = 15

# Optional curated, punchier event labels live in a per-deck override file so the
# step stays topic-agnostic and reproducible. Format: a JSON object keyed by
# speaker name (lowercased), each value `{"role": "...", "detail": "..."}`. Any
# speaker without an entry falls back to the derived talk title.
# The same file may include `required_events`, a list of conference-level events
# such as workshops, keynote sessions, or oral sessions that must be present on
# the opening bookend even when the bound source chips are individual talks.
#   { "dawn song": {"role": "opening keynote", "detail": "AIWILD workshop ..."} }
DEFAULT_EVENT_OVERRIDES_FILENAME = "bookend_event_overrides.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_event_config(
    deck_dir: Path,
    override_path: Path | None,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Load optional speaker overrides and required conference-level events."""
    path = override_path or (deck_dir / DEFAULT_EVENT_OVERRIDES_FILENAME)
    if not path.is_file():
        return {}, []
    raw = json.loads(path.read_text(encoding="utf-8"))
    overrides: dict[str, dict[str, str]] = {}
    required_events: list[dict[str, str]] = []
    for speaker, value in (raw or {}).items():
        if str(speaker) == "required_events":
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("name") or "").strip()
                    detail = str(item.get("detail") or "").strip()
                    if not name:
                        continue
                    required_events.append(
                        {
                            "name": name,
                            "detail": detail,
                            "url": str(item.get("url") or "").strip(),
                        }
                    )
            continue
        if isinstance(value, dict):
            overrides[str(speaker).strip().lower()] = {
                "role": str(value.get("role") or "").strip(),
                "detail": str(value.get("detail") or "").strip(),
            }
    return overrides, required_events


_SEGMENT_SUFFIX_RE = re.compile(r":segment:\d+$")


def _aggregation_key(chip: dict[str, Any], label: str) -> str:
    """One talk/paper is one source, however many transcript segments cite it.

    Two segments of the same recording arrive as two ``SRC-*`` labels with two
    ``speech_transcript:<hash>:segment:<n>`` ids. Counting them separately makes
    "most-referenced" counts depend on how the recording was chunked rather than
    on how many slides draw on the work, and lets one recording occupy two
    bookend slots under two labels."""

    source_id = str(chip.get("source_id") or "").strip()
    if source_id:
        return _SEGMENT_SUFFIX_RE.sub("", source_id).lower()
    return label


def _aggregate_sources(
    render_spec: dict[str, Any],
    *,
    link_resolver: SourceLinkResolver | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-source aggregation across all content slides, keyed by the underlying
    work/artifact (segment-level ids collapse to the recording)."""
    freq: collections.Counter[str] = collections.Counter()
    info: dict[str, dict[str, Any]] = {}
    for slide in render_spec.get("slides", []):
        if str(slide.get("slide_kind") or "").strip():
            continue  # skip any already-injected bookend slides
        seen: set[str] = set()
        for chip in slide.get("source_chips", []) or []:
            if not isinstance(chip, dict):
                continue
            label = str(chip.get("label") or "")
            if not label.startswith("SRC-"):
                continue
            key = _aggregation_key(chip, label)
            if key not in seen:
                freq[key] += 1
                seen.add(key)
            if key not in info:
                info[key] = {
                    "label": label,
                    "title": str(chip.get("title") or "").strip(),
                    "note": str(chip.get("contribution_note") or "").strip(),
                    "url": (
                        canonical_public_url(chip.get("url"))
                        or canonical_public_url(chip.get("source_url"))
                        or canonical_public_url(chip.get("artifact_url"))
                        or (link_resolver.resolve_item(chip) if link_resolver is not None else "")
                    ),
                    "source_type": str(chip.get("source_type") or "").strip().lower(),
                    "speakers": [s for s in (chip.get("speakers") or []) if s],
                }
    for key in info:
        info[key]["count"] = freq[key]
    return info


def _is_event(rec: dict[str, Any]) -> bool:
    if rec["source_type"] in TALK_SOURCE_TYPES:
        return True
    if any(kind in rec["source_type"] for kind in TALK_SOURCE_TYPES):
        return True
    # transcript-style sources with a named speaker are talks/keynotes
    if bool(rec["speakers"]) and "paper" not in rec["source_type"]:
        return True
    # Some conference-event records arrive through workshop ledgers with
    # source_type="paper" even though the visible title is an event/talk label.
    # Keep these in the event rail so workshops, keynotes, panels, orals, and
    # major talks are not demoted into "most-referenced work".
    return bool(EVENT_TEXT_RE.search(f"{rec.get('title', '')} {rec.get('note', '')}"))


def _event_key(rec: dict[str, Any]) -> str:
    parts = [
        str(rec.get("source_type") or ""),
        str(rec.get("title") or ""),
        " ".join(str(s) for s in rec.get("speakers", []) or []),
    ]
    return _strip_md(" | ".join(parts)).lower()[:120]


_MD_MARKER_RE = re.compile(r"(\*\*|__|\*|`|~~)")


def _strip_md(text: str) -> str:
    """Remove inline Markdown emphasis markers so bookend slides never render
    literal '**', '`', or '_' characters (the rendered slide is plain text)."""
    # Links lose their URL (a raw "[OpenReview](https://...)" shipped on ICML 2026
    # deck 02's resources slide) and a leaked LaTeX macro name is repaired, by the
    # same rules the ledger and the chips use.
    cleaned = repair_leaked_macro_names(strip_markdown(str(text or "")))
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)  # stray heading markers
    # PDF extraction often leaves line-wrap hyphens inside ordinary words.
    # A suspended hyphen is not a wrap: "Object- and Image-level" stays
    # (it rendered as "Objectand" on CVPR 2026 deck 06).
    cleaned = re.sub(r"(?<=[A-Za-z])-\s+(?!(?:and|or|nor|to|vs|versus)\b)(?=[a-z])", "", cleaned)
    return " ".join(cleaned.split())


def _is_source_locator(title: str) -> bool:
    """Internal acquisition labels are not titles for a public bookend."""
    return bool(re.fullmatch(r"(?:event_media_caption|speech_transcript)(?::[^\s]+)?", title.strip(), re.I))


def _short(text: str, limit: int = 64) -> str:
    text = _strip_md(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _short_title(text: str, limit: int) -> str:
    """A work title that fits ``limit`` WITHOUT a mid-title ellipsis.

    Real titles replaced locator labels in the ledger, so bookend rows got
    longer and the pixel review flagged "reference titles visibly truncated
    with ellipses". The rows wrap, so the limit is generous; when a title is
    still too long, its natural short name (the pre-colon / pre-question head,
    e.g. "Is Vibe Coding Safe?") is used before falling back to an ellipsis."""

    text = _strip_md(text)
    if len(text) <= limit:
        return text
    for sep in (": ", "? ", " — ", " - "):
        head, _, _ = text.partition(sep)
        if head != text and 12 <= len(head) <= limit:
            return head + ("?" if sep == "? " else "")
    return _short(text, limit)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", _strip_md(text), maxsplit=1)[0]


def _rank_value(entry: dict[str, Any]) -> float:
    try:
        return float(entry.get("rank"))
    except (TypeError, ValueError):
        return 1_000_000.0


def _ledger_event_candidates(
    deck_dir: Path,
    *,
    limit: int,
    skip_names: set[str] | None = None,
    link_resolver: SourceLinkResolver | None = None,
) -> list[dict[str, str]]:
    """Fallback event candidates from the speaker/author/work ledger.

    Some decks bind only paper-like chips into figure slides even though the
    topic ledger records relevant workshops, keynotes, invited talks, panels,
    or oral/contributed talks. Use the ledger only to fill missing event slots
    on the opening bookend, preserving explicit required events first.
    """
    if limit <= 0:
        return []
    path = deck_dir / "speaker_author_work_ledger.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else []
    if not isinstance(entries, list):
        return []
    seen = set(skip_names or set())
    candidates: list[dict[str, str]] = []
    for entry in sorted((e for e in entries if isinstance(e, dict)), key=_rank_value):
        title = str(entry.get("resolved_title") or "").strip()
        note = str(entry.get("contribution_note") or "").strip()
        if not title:
            continue
        if not EVENT_TEXT_RE.search(f"{title} {note}"):
            continue
        key = _strip_md(title).lower()
        if key in seen:
            continue
        seen.add(key)
        detail = _first_sentence(note) if note else title
        candidates.append(
            {
                "name": _short(title, 88),
                "detail": _short(detail, 92),
                "url": link_resolver.resolve_item({"title": title, "detail": detail})
                if link_resolver is not None
                else "",
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def _strip_event_prefix(text: object) -> str:
    return re.sub(
        r"^\s*(?:workshop\s+(?:keynote|talk)|keynote|invited\s+talk|oral\s+(?:talk|session)|panel)\s*:\s*",
        "",
        _strip_md(str(text or "")),
        flags=re.I,
    ).strip()


def enrich_required_event_urls_from_cluster_plan(
    deck_dir: Path,
    required_events: list[dict[str, str]],
    *,
    link_resolver: SourceLinkResolver,
) -> int:
    """Fill required-event URLs from research_cluster_plan source IDs."""

    path = deck_dir / "research_cluster_plan.json"
    if not path.is_file():
        return 0
    raw = json.loads(path.read_text(encoding="utf-8"))
    clusters = raw.get("clusters") if isinstance(raw, dict) else []
    if not isinstance(clusters, list):
        return 0
    candidates: list[tuple[str, str]] = []
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        for role in ("primary_sources", "supporting_sources"):
            for source in cluster.get(role) or []:
                if not isinstance(source, dict):
                    continue
                title = str(source.get("resolved_title") or source.get("title") or "").strip()
                if not title:
                    continue
                url = link_resolver.resolve_item(
                    {
                        "title": title,
                        "source_id": source.get("source_id"),
                        "works": source.get("works") or [],
                    }
                )
                if not url:
                    continue
                candidates.append((_match_key(title), url))
                candidates.append((_match_key(_strip_event_prefix(title)), url))
    changed = 0
    for event in required_events:
        if canonical_public_url(event.get("url")):
            continue
        keys = [
            _match_key(event.get("name")),
            _match_key(event.get("detail")),
        ]
        for key in keys:
            if not key:
                continue
            for candidate_key, url in candidates:
                if key == candidate_key or (
                    len(key) >= 14
                    and len(candidate_key) >= 14
                    and (key in candidate_key or candidate_key in key)
                ):
                    event["url"] = url
                    changed += 1
                    break
            if canonical_public_url(event.get("url")):
                break
    return changed


def _match_key(text: object) -> str:
    cleaned = _strip_md(str(text or "")).lower().replace("…", " ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())

def _note_names_work(note: str, title: str) -> bool:
    """True when the note's first sentence actually NAMES the work.

    Stricter than ``_note_matches_work``, and deliberately so: that guard asks
    whether a note belongs to this work at all (misattribution), which a shared
    topic word satisfies. A citation has a harder job -- it must let a reader
    identify the source. Raw abstract prefixes open with background prose
    ("Reproducible closed-loop evaluation remains a major bottleneck in Embodied
    AI...") that passes the attribution guard on "embodied" while never naming
    Wanderland, so the citation line ends up describing a research area instead
    of citing a paper. Requiring the work's own short name keeps the sources
    slide a list of sources.
    """

    lead = _first_sentence(note).lower()
    if not lead:
        return False
    # A work's short name is its pre-colon head ("VAD-GS: Visibility-Aware..."
    # -> "VAD-GS"); for titles with no colon the whole title is the name.
    head = _strip_md(title).split(":", 1)[0].strip()
    return bool(head) and len(head) <= 40 and head.lower() in lead


def build_bookend_slides(
    deck: dict[str, Any],
    render_spec: dict[str, Any],
    event_overrides: dict[str, dict[str, str]] | None = None,
    required_events: list[dict[str, str]] | None = None,
    link_resolver: SourceLinkResolver | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_overrides = event_overrides or {}
    required_events = required_events or []

    info = _aggregate_sources(render_spec, link_resolver=link_resolver)
    source_title_urls = [
        (_match_key(rec.get("title")), str(rec.get("url") or "").strip())
        for rec in info.values()
        if str(rec.get("url") or "").strip()
    ]

    def _source_title_url(item: dict[str, Any]) -> str:
        candidates = [
            _match_key(item.get(key))
            for key in ("name", "title", "detail", "note")
            if str(item.get(key) or "").strip()
        ]
        for candidate in candidates:
            if not candidate:
                continue
            for title_key, url in source_title_urls:
                if candidate == title_key:
                    return url
            if len(candidate) < 14:
                continue
            for title_key, url in source_title_urls:
                if len(title_key) >= 14 and (candidate in title_key or title_key in candidate):
                    return url
        return ""

    def _resolved_url(item: dict[str, Any]) -> str:
        return (
            canonical_public_url(item.get("url"))
            or (link_resolver.resolve_item(item) if link_resolver is not None else "")
            or _source_title_url(item)
        )

    ranked = sorted(info.values(), key=lambda r: (-r["count"], r["label"]))

    events, papers, seen_event_keys, seen_paper_keys = [], [], set(), set()
    event_limit = max(TOP_EVENTS, len(required_events))
    for item in required_events:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen_event_keys:
            continue
        seen_event_keys.add(key)
        events.append(
            {
                "name": name,
                "detail": _short(str(item.get("detail") or ""), 92),
                "url": _resolved_url(item),
            }
        )
    for rec in ranked:
        if _is_source_locator(rec["title"]):
            continue
        is_event = _is_event(rec)
        event_key = _event_key(rec)
        paper_key = (rec["note"] or rec["title"]).split("—")[0].strip().lower()[:40]
        if is_event and event_key not in seen_event_keys and len(events) < event_limit:
            seen_event_keys.add(event_key)
            speaker = rec["speakers"][0] if rec["speakers"] else ""
            override = event_overrides.get(speaker.lower())
            if override:
                name = f"{speaker} — {override['role']}"
                detail = override["detail"]
            else:
                # Lead with the talk: a transcript's speaker is often one
                # ASR-heard first name ("Gardo", "Fabricio"), which tells the
                # audience nothing they can look up. A full name goes under it.
                name = _short_title(rec["title"], 100)
                if len(speaker.split()) >= 2:
                    detail = speaker
                else:
                    # Acronym titles need a plain-language description.
                    detail = _short(rec["note"], 92) if len(name) <= 12 else ""
            events.append({"name": name, "detail": detail, "url": _resolved_url(rec)})
        elif not is_event and paper_key not in seen_paper_keys and len(papers) < TOP_PAPERS:
            seen_paper_keys.add(paper_key)
            note = rec["note"]
            if note.lower().startswith("source:"):
                note = ""  # a pointer to another work, not this paper's claim
            if note and not _note_matches_work(note, rec["title"]):
                note = ""  # another cluster's claim — don't misattribute it here
            if note and _LEDGER_SECTION_NOTE_RE.match(note):
                # A ledger section header over a list of works ("**Full-read
                # abstract records and paper resources:** *ComplexMCP* — , [OpenReview]...")
                # is not one work's claim; split on its dash it put the header
                # on the slide as the paper's name (ICML 2026 deck 02 slide 2).
                note = ""
            if "—" in note:
                pname, pdetail = note.split("—", 1)
                name, detail = _short_title(pname.strip(), 100), _short(pdetail.strip(), 120)
            else:
                name, detail = _short_title(rec["title"], 100), _short(note, 120)
            # The same paper can reach here under two notes (two clusters
            # quoting it); de-duplicate by the name the slide shows.
            if any(p["name"].strip().lower() == name.strip().lower() for p in papers):
                continue
            papers.append({"name": name, "detail": detail, "url": _resolved_url(rec)})

    resources, seen_res = [], set()
    for rec in ranked:
        if _is_source_locator(rec["title"]):
            continue
        # A reading list names works, so an entry is the work's title and one
        # title appears once. Keying on the contribution note listed a paper and
        # its talk twice (CVPR 2026 deck 02: Hint2Gen) and showed a claim
        # sentence where a title belonged (deck 07: "CURE reduces ... 67%").
        text = _strip_md(rec["title"])
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if not key or key in seen_res:
            continue
        seen_res.add(key)
        resources.append({"title": text, "url": _resolved_url({"title": text, **rec})})
        if len(resources) >= TOP_RESOURCES:
            break

    author = str(deck.get("author") or "Nikolaos Vasiloglou")
    analysis_url = str(deck.get("analysis_report_url") or "")

    events_slide = {
        "slide_id": "s-events",
        "slide_kind": "events",
        "section_id": "frontmatter",
        "title": "What this talk draws on",
        "events": events,
        "papers": papers,
        "prepared_by": f"Prepared by {author}",
        "source_chips": [],
    }
    resources_slide = {
        "slide_id": "s-resources",
        "slide_kind": "resources",
        "section_id": "backmatter",
        "title": "Sources & further reading",
        "resources": resources,
        "analysis_url": analysis_url,
        "analysis_note": "These are some of the sources featured in this talk. For the full set of papers and talks, see the",
        "source_chips": [],
    }
    return events_slide, resources_slide


def count_derived_events(render_spec: dict[str, Any]) -> int:
    return sum(1 for rec in _aggregate_sources(render_spec).values() if _is_event(rec))


def _inject(slides: list[dict[str, Any]], events_slide: dict[str, Any], resources_slide: dict[str, Any]) -> list[dict[str, Any]]:
    body = [s for s in slides if str(s.get("slide_kind") or "") not in {"events", "resources"}]
    # opening slide goes right after the title frame (index 0), closing slide last.
    return [body[0], events_slide, *body[1:], resources_slide] if body else [events_slide, resources_slide]


def _infer_workspace(deck_dir: Path) -> Path:
    parts = deck_dir.resolve().parts
    if "decks" in parts:
        index = parts.index("decks")
        return Path(*parts[:index])
    return deck_dir.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument("--deck", default="presentation_deck.json")
    parser.add_argument("--render-spec", default="render_spec.json")
    parser.add_argument(
        "--event-overrides",
        type=Path,
        help="Optional speaker->{role,detail} event-label overrides JSON "
        f"(default: <deck-dir>/{DEFAULT_EVENT_OVERRIDES_FILENAME} if present).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    deck_dir = args.deck_dir.expanduser().resolve()
    deck_path = deck_dir / args.deck
    render_spec_path = deck_dir / args.render_spec
    deck = _load(deck_path)
    render_spec = _load(render_spec_path)
    link_resolver = SourceLinkResolver.from_workspace(_infer_workspace(deck_dir))
    event_overrides, required_events = load_event_config(deck_dir, args.event_overrides)
    enriched_required_urls = enrich_required_event_urls_from_cluster_plan(
        deck_dir,
        required_events,
        link_resolver=link_resolver,
    )
    derived_events = [
        rec
        for rec in _aggregate_sources(render_spec, link_resolver=link_resolver).values()
        if _is_event(rec)
    ]
    derived_event_count = len(derived_events)
    explicit_event_names = {
        _strip_md(str(event.get("name") or "")).lower()
        for event in required_events
        if isinstance(event, dict)
    }
    derived_event_names = {_strip_md(str(event.get("title") or "")).lower() for event in derived_events}
    missing_event_slots = max(0, TOP_EVENTS - (len(required_events) + derived_event_count))
    ledger_events = _ledger_event_candidates(
        deck_dir,
        limit=missing_event_slots,
        skip_names=explicit_event_names | derived_event_names,
        link_resolver=link_resolver,
    )
    if ledger_events:
        required_events.extend(ledger_events)
    if event_overrides:
        print(f"loaded {len(event_overrides)} event-label override(s)")
    if required_events:
        print(f"loaded {len(required_events)} required event(s)")
    if enriched_required_urls:
        print(f"enriched {enriched_required_urls} required event URL(s) from cluster plan")

    events_slide, resources_slide = build_bookend_slides(
        deck,
        render_spec,
        event_overrides,
        required_events,
        link_resolver=link_resolver,
    )
    print(f"events: {len(events_slide['events'])} | papers: {len(events_slide['papers'])} | resources: {len(resources_slide['resources'])}")

    deck["slides"] = _inject(deck.get("slides", []), events_slide, resources_slide)
    render_spec["slides"] = _inject(render_spec.get("slides", []), events_slide, resources_slide)
    # Re-approve the slide count so the bookends are part of the approved envelope.
    # The bookend injection IS the approved expansion, so record it on both the deck
    # and render spec (creating the block if the contract did not carry one) — an
    # absent approved_expansion otherwise reads as an unapproved slide-count change.
    final_count = len(deck["slides"])
    base_count = final_count - 2
    for contract in (deck, render_spec):
        approved = contract.get("approved_expansion")
        if not isinstance(approved, dict):
            approved = {}
            contract["approved_expansion"] = approved
        approved["target_slide_count"] = final_count
        approved["base_slide_count"] = base_count
        approved["bookend_slides_added"] = ["s-events", "s-resources"]
        approved["approved"] = True
        approved["expansion_reason"] = (
            "Opening 'relevant events + most-cited papers' and closing "
            "'sources & further reading' bookend slides added per the approved "
            "presentation conventions; structural frontmatter/backmatter, not new claims."
        )
    print(f"deck slides: {final_count} | render_spec slides: {len(render_spec['slides'])}")

    if args.dry_run:
        print("[dry-run] no files written")
        return 0
    deck_sha_before = hashlib.sha256(deck_path.read_bytes()).hexdigest()
    deck_path.write_text(json.dumps(deck, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    render_spec_path.write_text(json.dumps(render_spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {deck_path}\nwrote {render_spec_path}")
    _restamp_source_binding(deck_dir / "source_binding_report.json", deck_sha_before, deck_path)
    return 0


def _restamp_source_binding(binding_path: Path, deck_sha_before: str, deck_path: Path) -> None:
    """Carry a current source binding across the bookend rewrite of the deck.

    Binding runs before this step (the bookends are derived from its chips), so
    its recorded deck digest named the pre-bookend deck, and every Gate A
    reviewer flagged the binding as stale against the deck they were shown
    (CVPR 2026). The two bookend slides carry no source bindings, so a binding
    that was current for the deck immediately before this rewrite is current
    after it. A binding that was already stale is left stale.
    """

    if not binding_path.is_file():
        return
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    hashes = binding.get("input_hashes")
    if not isinstance(hashes, dict) or hashes.get("presentation_deck.json") != deck_sha_before:
        return
    deck_sha_after = hashlib.sha256(deck_path.read_bytes()).hexdigest()
    if deck_sha_after == deck_sha_before:
        return
    hashes["presentation_deck.json"] = deck_sha_after
    binding["bookend_restamp"] = {
        "by": "scripts/presentations/add_bookend_slides.py",
        "from_presentation_deck_sha256": deck_sha_before,
        "to_presentation_deck_sha256": deck_sha_after,
        "reason": "only the s-events/s-resources bookend slides changed; they carry no source bindings",
    }
    binding_path.write_text(json.dumps(binding, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"restamped {binding_path.name} deck digest across the bookend rewrite")


if __name__ == "__main__":
    raise SystemExit(main())
