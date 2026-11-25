#!/usr/bin/env python3
"""Synthesize top conference-program events for presentation bookends.

This step combines:

1. deterministic deck evidence (source chips, required overrides, work ledger),
2. all-program evidence from cached ICLR program pages and workshop inventory,
3. optional GPT-researcher output produced from the generated prompt.

It does not call live providers. The GPT-researcher lane is an input contract:
generate a prompt, run it through the approved researcher lane if budget allows,
then pass the JSON/Markdown result back into this script.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline.conference_identity import (
    ConferenceIdentityError,
    conference_name,
    load_conference_identity,
)
from conference_pipeline.presentation.source_links import (
    SourceLinkResolver,
    canonical_public_url,
)

import add_bookend_slides as bookends


EVENT_TEXT_RE = re.compile(
    r"\b(workshop|keynote|oral|spotlight|invited\s+talk|invited|session|panel|talk|tutorial)\b",
    re.IGNORECASE,
)
GENERIC_EVENT_TEXT_RE = re.compile(
    r"^(?:workshops?|posters?|accepted papers?|oral sessions?|invited talks?|keynotes?|"
    r"panels?|tutorials?|competitions?|schedule|program|calendar|downloads?|home)$",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<[^>]+>")
ANCHOR_RE = re.compile(r"<a\b[^>]*?href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
JSON_TITLE_RE = re.compile(
    r'"(?:title|name|event_name|session_title|workshop_title|talk_title)"\s*:\s*"([^"]{6,180})"',
    re.I,
)

EVENT_KIND_WEIGHTS = {
    "keynote": 42,
    "invited_talk": 38,
    "oral": 32,
    "panel": 30,
    "workshop": 28,
    "spotlight": 26,
    "tutorial": 22,
    "competition": 22,
    "session": 20,
    "talk": 18,
    "unknown": 8,
}

PROGRAM_SALIENCE_EVENT_KINDS = {
    "keynote",
    "invited_talk",
    "oral",
    "panel",
    "workshop",
    "competition",
    "spotlight",
}

# Speaker/talk kinds that an LLM can fabricate with no program URL. When a
# candidate of one of these kinds is NOT anchored to the real program
# (program_hits>0 or a URL on the current conference's own program_host, see
# _is_program_anchored), it must be dropped before it can reach the slide-2
# bookend. Workshop/competition/tutorial are deliberately excluded: those come
# from the workshop inventory / program HTML with real program_host URLs and
# are anchored by source.
UNVERIFIABLE_TALK_KINDS = frozenset(
    {
        "keynote",
        "invited_talk",
        "talk",
        "oral",
        "panel",
        "spotlight",
        "session",
    }
)

PROGRAM_EVENT_SALIENCE_WEIGHTS = {
    "keynote": 84,
    "invited_talk": 78,
    "oral": 70,
    "competition": 64,
    "panel": 62,
    "workshop": 60,
    "spotlight": 58,
    "tutorial": 42,
    "session": 36,
    "talk": 24,
    "unknown": 0,
}

PLACEHOLDER_DETAILS = {
    "",
    "conference event",
    "conference program candidate",
    "workshop in acquired iclr program inventory",
}

EVENT_DETAIL_LABELS = {
    "keynote": "Keynote",
    "invited_talk": "Invited talk",
    "oral": "Oral talk",
    "panel": "Panel",
    "workshop": "Workshop",
    "spotlight": "Spotlight",
    "tutorial": "Tutorial",
    "competition": "Competition",
    "session": "Session",
    "talk": "Talk",
    "unknown": "Conference event",
}

COVERAGE_STRAND_TERMS = {
    "biology": {
        "antibody",
        "bio",
        "biological",
        "biology",
        "biomolecular",
        "cell",
        "cellular",
        "dna",
        "drug",
        "enzyme",
        "gene",
        "genome",
        "genomics",
        "methylation",
        "molecular",
        "protein",
        "rna",
    },
    "materials": {
        "accelerated",
        "atomistic",
        "crystal",
        "crystalline",
        "density",
        "diffraction",
        "electron",
        "hamiltonian",
        "interatomic",
        "material",
        "materials",
        "mof",
        "superconductor",
        "synthesis",
        "xrd",
    },
    "physics_pde": {
        "boundary",
        "conservation",
        "equivariant",
        "function",
        "geometry",
        "neural",
        "operator",
        "operators",
        "pde",
        "pdes",
        "physics",
        "simulation",
        "solver",
        "solvers",
    },
    "earth_weather": {
        "climate",
        "earth",
        "flood",
        "forecast",
        "forecasting",
        "geospatial",
        "precipitation",
        "remote",
        "sensing",
        "spectral",
        "weather",
    },
    "agents_evaluation": {
        "agent",
        "agentic",
        "agents",
        "autonomous",
        "benchmark",
        "benchmarks",
        "evaluation",
        "experiment",
        "experiments",
        "lab",
        "laboratory",
        "reasoning",
        "verifier",
        "verifiers",
    },
    "safety": {
        "dual",
        "hazard",
        "misuse",
        "risk",
        "safety",
        "trust",
        "trustworthy",
    },
}

DEFAULT_TOP_N = 5


@dataclass
class EventCandidate:
    name: str
    detail: str = ""
    url: str = ""
    event_kind: str = "unknown"
    source_family: str = "unknown"
    source_path: str = ""
    deterministic_score: float = 0.0
    gpt_score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    program_hits: int = 0
    deck_source_count: int = 0
    ledger_rank: float | None = None
    event_salience_score: float = 0.0
    deck_resonance_score: float = 0.0
    deck_resonance: dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        return round(
            self.deterministic_score
            + self.gpt_score
            + self.deck_resonance_score
            + self.event_salience_score,
            3,
        )


def _clean_text(text: str) -> str:
    text = TAG_RE.sub(" ", str(text or ""))
    text = html.unescape(text)
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = re.sub(r"^\s*(?:talk|session|event)\s*:\s*", "", text, flags=re.I)
    return " ".join(text.split())


def _norm(text: str) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if len(t) >= 4}


def _looks_like_event(text: str) -> bool:
    cleaned = _clean_text(text)
    if len(cleaned) < 6 or len(cleaned) > 180:
        return False
    if GENERIC_EVENT_TEXT_RE.match(cleaned):
        return False
    return bool(EVENT_TEXT_RE.search(cleaned)) or _is_competition_event_text(cleaned)


def _is_competition_event_text(text: str, source_type: str = "") -> bool:
    haystack = f"{source_type} {text}".lower()
    source = str(source_type or "").lower()
    if "competition" in source:
        return True
    if "price competition" in haystack:
        return False
    if re.search(r"\b(?:challenge|contest|shared task)\b", haystack):
        return True
    if not re.search(r"\bcompetition\b", haystack):
        return False
    return any(
        re.search(pattern, haystack)
        for pattern in (
            r"\barena\b",
            r"\btrack\b",
            r"\bsession\b",
            r"\bevent\b",
            r"prompt injection competition",
            r"competition workshop",
        )
    )


def _event_kind(text: str, source_type: str = "") -> str:
    haystack = f"{source_type} {text}".lower()
    if "keynote" in haystack:
        return "keynote"
    if "invited" in haystack:
        return "invited_talk"
    if "oral" in haystack:
        return "oral"
    if "panel" in haystack:
        return "panel"
    if "workshop" in haystack:
        return "workshop"
    if "spotlight" in haystack:
        return "spotlight"
    if "tutorial" in haystack:
        return "tutorial"
    if _is_competition_event_text(text, source_type):
        return "competition"
    if "session" in haystack:
        return "session"
    if "talk" in haystack or "transcript" in haystack:
        return "talk"
    return "unknown"


def _is_placeholder_detail(detail: str) -> bool:
    return _clean_text(detail).lower() in PLACEHOLDER_DETAILS


def _event_detail_label(kind: str, text: str = "") -> str:
    haystack = _clean_text(text).lower()
    if "workshop keynote" in haystack:
        return "Workshop keynote"
    if "workshop talk" in haystack:
        return "Workshop talk"
    if "oral session" in haystack:
        return "Oral session"
    if "oral talk" in haystack:
        return "Oral talk"
    return EVENT_DETAIL_LABELS.get(kind, EVENT_DETAIL_LABELS["unknown"])


def _display_detail(detail: str, kind: str, text: str = "") -> str:
    if _is_placeholder_detail(detail):
        return _event_detail_label(kind, text)
    return _clean_text(detail)


def _clean_prefixed_event_name(title: str) -> tuple[str, str | None]:
    cleaned = _clean_text(title)
    prefixes = [
        (r"^workshop\s+keynote\s*:\s*", "Workshop keynote"),
        (r"^workshop\s+talk\s*:\s*", "Workshop talk"),
        (r"^oral\s+talk\s*:\s*", "Oral talk"),
        (r"^oral\s+session\s*:\s*", "Oral session"),
        (r"^keynote\s*:\s*", "Keynote"),
        (r"^invited\s+talk\s*:\s*", "Invited talk"),
        (r"^panel\s*:\s*", "Panel"),
    ]
    for pattern, detail in prefixes:
        stripped = re.sub(pattern, "", cleaned, flags=re.I).strip()
        if stripped != cleaned:
            return stripped, detail
    return cleaned, None


def _rank_value(value: Any, default: float = 1_000_000.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number):
        return default
    return number


def _candidate_key(candidate: EventCandidate) -> str:
    return _norm(candidate.name)[:120]


def _same_event(left: EventCandidate, right: EventCandidate) -> bool:
    left_norm = _norm(left.name)
    right_norm = _norm(right.name)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    if len(left_norm) >= 14 and left_norm in right_norm:
        return True
    if len(right_norm) >= 14 and right_norm in left_norm:
        return True
    lt = _tokens(left.name)
    rt = _tokens(right.name)
    if not lt or not rt:
        return False
    return len(lt & rt) / max(len(lt | rt), 1) >= 0.62


def _add_or_merge(
    merged: list[EventCandidate],
    candidate: EventCandidate,
) -> None:
    if not _norm(candidate.name):
        return
    candidate.url = canonical_public_url(candidate.url)
    candidate.detail = _display_detail(
        candidate.detail,
        candidate.event_kind,
        f"{candidate.name} {candidate.detail}",
    )
    for existing in merged:
        if _same_event(existing, candidate):
            if candidate.deterministic_score > existing.deterministic_score:
                existing.deterministic_score = candidate.deterministic_score
            existing.gpt_score += candidate.gpt_score
            existing.program_hits += candidate.program_hits
            existing.deck_source_count = max(existing.deck_source_count, candidate.deck_source_count)
            if existing.ledger_rank is None:
                existing.ledger_rank = candidate.ledger_rank
            elif candidate.ledger_rank is not None:
                existing.ledger_rank = min(existing.ledger_rank, candidate.ledger_rank)
            if (not existing.detail or _is_placeholder_detail(existing.detail)) and candidate.detail:
                existing.detail = candidate.detail
            if not existing.url and candidate.url:
                existing.url = candidate.url
            if existing.event_kind == "unknown" and candidate.event_kind != "unknown":
                existing.event_kind = candidate.event_kind
            for item in candidate.evidence:
                if item and item not in existing.evidence:
                    existing.evidence.append(item)
            return
    merged.append(candidate)


def _topic_label(topic_id: str) -> str:
    topic_number = topic_id.split("_", 1)[0]
    readable = topic_id.split("_", 1)[1] if "_" in topic_id else topic_id
    return f"{topic_number}: {readable.replace('_', ' ')}"


def _topic_keywords(topic_id: str, deck_dir: Path, workspace: Path) -> set[str]:
    pieces = [topic_id.replace("_", " ")]
    for path in (
        workspace / "analysis" / f"{topic_id}.md",
        workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md",
        deck_dir / "presentation_blueprint.md",
    ):
        if path.is_file():
            pieces.append(path.read_text(encoding="utf-8", errors="replace")[:4000])
    stop = {
        "with", "from", "that", "this", "have", "into", "using", "model", "models",
        "learning", "paper", "papers", "talk", "talks", "workshop", "iclr", "2026",
        "slide", "source", "sources", "research", "presentation",
    }
    words = set()
    for piece in pieces:
        words.update(
            token
            for token in _tokens(piece)
            if token not in stop
        )
    return words


def _topic_overlap_score(candidate: EventCandidate, keywords: set[str]) -> float:
    if not keywords:
        return 0.0
    ct = _tokens(f"{candidate.name} {candidate.detail}")
    overlap = ct & keywords
    return min(20.0, len(overlap) * 4.0)


DECK_RESONANCE_STOPWORDS = {
    "agent", "agents", "analysis", "based", "benchmark", "benchmarks", "candidate",
    "conference", "data", "deep", "event", "events", "from", "iclr", "invited",
    "keynote", "large", "learning", "model", "models", "paper", "papers",
    "panel", "presentation", "research", "session", "slide", "slides", "source",
    "sources", "talk", "talks", "theory", "this", "through", "using", "where",
    "with", "workshop", "workshops",
    # Topic-family words are too broad to prove that an event belongs in this
    # specific deck narrative.
    "alignment", "agentic", "language", "llms", "reasoning", "systems",
}


def _signal_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if token not in DECK_RESONANCE_STOPWORDS}


def _deck_text_parts(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield value
    elif isinstance(value, list):
        for item in value:
            yield from _deck_text_parts(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            yield from _deck_text_parts(item)


def _load_deck_resonance_context(deck_dir: Path) -> dict[str, Any]:
    """Collect the deck's actual narrative surface for event/deck alignment.

    The event slide should resonate with what the deck argues, not only with a
    broad topic label. Use slide copy and source chips from the structured deck
    contracts, skipping existing bookend slides so the event rail cannot support
    itself circularly.
    """
    slide_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    seen_slides: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for path in (deck_dir / "presentation_deck.json", deck_dir / "render_spec.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        slides = payload.get("slides") if isinstance(payload, dict) else []
        if not isinstance(slides, list):
            continue
        for index, slide in enumerate(slides, start=1):
            if not isinstance(slide, dict):
                continue
            if str(slide.get("slide_kind") or "") in {"events", "resources"}:
                continue
            slide_id = str(slide.get("slide_id") or f"slide-{index:03d}")
            title = str(slide.get("title") or slide.get("headline") or "").strip()
            slide_key = f"{slide_id}:{title}"
            slide_text = " ".join(_deck_text_parts({k: slide.get(k) for k in ("title", "headline", "subtitle", "claim", "body", "bullets", "takeaway", "visual_takeaway")}))
            if slide_key not in seen_slides and slide_text.strip():
                seen_slides.add(slide_key)
                slide_records.append(
                    {
                        "slide_id": slide_id,
                        "title": title or slide_id,
                        "text": slide_text,
                        "tokens": _signal_tokens(slide_text),
                    }
                )
            for chip in slide.get("source_chips") or []:
                if not isinstance(chip, dict):
                    continue
                label = str(chip.get("label") or chip.get("source_label") or "").strip()
                chip_text = " ".join(
                    _deck_text_parts(
                        {
                            "title": chip.get("title"),
                            "note": chip.get("contribution_note") or chip.get("relevance_rationale"),
                            "source_type": chip.get("source_type"),
                            "speakers": chip.get("speakers"),
                            "authors": chip.get("authors"),
                            "systems": chip.get("systems"),
                            "works": chip.get("works"),
                        }
                    )
                )
                key = (label, _norm(chip_text)[:160])
                if chip_text.strip() and key not in seen_sources:
                    seen_sources.add(key)
                    source_records.append(
                        {
                            "label": label,
                            "title": str(chip.get("title") or "").strip(),
                            "text": chip_text,
                            "tokens": _signal_tokens(chip_text),
                        }
                    )
    return {"slides": slide_records, "sources": source_records}


def _resonance_hit(event_tokens: set[str], text: str, tokens: set[str]) -> tuple[int, bool]:
    overlap = event_tokens & tokens
    event_phrase = _norm(" ".join(sorted(event_tokens)))
    text_norm = _norm(text)
    # Phrase matching by full candidate text happens in the caller; this helper
    # intentionally scores distinctive token overlap only.
    return len(overlap), bool(event_phrase and event_phrase in text_norm)


def score_deck_resonance(candidate: EventCandidate, deck_context: dict[str, Any]) -> dict[str, Any]:
    event_text = f"{candidate.name} {candidate.detail}"
    event_tokens = _signal_tokens(event_text)
    candidate_norm = _norm(event_text)
    slide_hits: list[dict[str, Any]] = []
    source_hits: list[dict[str, Any]] = []
    overlap_terms: set[str] = set()
    exact_bonus = 0.0

    for slide in deck_context.get("slides", []):
        tokens = set(slide.get("tokens") or set())
        overlap = event_tokens & tokens
        text_norm = _norm(str(slide.get("text") or ""))
        exact = bool(candidate_norm and candidate_norm in text_norm)
        if len(overlap) >= 2 or exact:
            overlap_terms.update(overlap)
            slide_hits.append(
                {
                    "slide_id": slide.get("slide_id"),
                    "title": slide.get("title"),
                    "overlap_terms": sorted(overlap)[:8],
                }
            )
            if exact:
                exact_bonus += 4.0

    for source in deck_context.get("sources", []):
        tokens = set(source.get("tokens") or set())
        overlap = event_tokens & tokens
        text_norm = _norm(str(source.get("text") or ""))
        exact = bool(candidate_norm and candidate_norm in text_norm)
        if len(overlap) >= 2 or exact:
            overlap_terms.update(overlap)
            source_hits.append(
                {
                    "label": source.get("label"),
                    "title": source.get("title"),
                    "overlap_terms": sorted(overlap)[:8],
                }
            )
            if exact:
                exact_bonus += 6.0

    score = min(18.0, len(slide_hits) * 5.0) + min(30.0, len(source_hits) * 10.0) + min(12.0, exact_bonus)
    if candidate.deck_source_count:
        score += min(12.0, candidate.deck_source_count * 4.0)
    if source_hits:
        status = "strong" if score >= 24 else "moderate"
    elif slide_hits:
        status = "moderate" if score >= 10 else "weak"
    else:
        status = "weak"
    return {
        "status": status,
        "score": round(score, 3),
        "action": "keep" if status in {"strong", "moderate"} else "review_or_replace",
        "supporting_slides": slide_hits[:5],
        "supporting_sources": source_hits[:5],
        "overlap_terms": sorted(overlap_terms)[:12],
    }


def score_event_salience(candidate: EventCandidate) -> float:
    """Rank actual conference events above paper-like source mentions.

    Slide 2 is a conference-program guide. Keynotes, workshops, competitions,
    panels, and oral/spotlight sessions should therefore get explicit ranking
    force, while the deck-resonance gate still decides whether they belong in
    this particular deck.
    """
    kind = candidate.event_kind
    score = PROGRAM_EVENT_SALIENCE_WEIGHTS.get(kind, 0)
    if kind not in PROGRAM_SALIENCE_EVENT_KINDS:
        return float(score)
    evidence_text = " ".join(candidate.evidence).lower()
    if candidate.source_family in {
        "full_program_html",
        "workshop_inventory",
        "workshop_talk_inventory",
    } or "program page=" in evidence_text or "workshop inventory" in evidence_text:
        score += 22
    if candidate.source_family == "gpt_researcher":
        score += 18
    if candidate.source_family == "deck_source_chip" and candidate.deck_source_count:
        score += min(14, candidate.deck_source_count * 4)
    if "required event override" in evidence_text:
        score += 8
    score += min(10, candidate.program_hits * 3)
    return float(score)


def _coverage_text_for_candidate(candidate: EventCandidate) -> str:
    # Coverage is about what the visible event communicates to the audience.
    # Supporting source-chip terms are used for resonance, but counting them as
    # event-domain terms lets a broad workshop falsely satisfy unrelated strands.
    return f"{candidate.name} {candidate.detail}"


def _coverage_strands_for_text(text: str) -> set[str]:
    tokens = _signal_tokens(text)
    return {
        strand
        for strand, terms in COVERAGE_STRAND_TERMS.items()
        if tokens & terms
    }


def _candidate_coverage_strands(candidate: EventCandidate) -> set[str]:
    return _coverage_strands_for_text(_coverage_text_for_candidate(candidate))


def _event_kind_priority(candidate: EventCandidate) -> float:
    return PROGRAM_EVENT_SALIENCE_WEIGHTS.get(candidate.event_kind, 0)


def _deck_coverage_strands(deck_context: dict[str, Any]) -> list[str]:
    counts: collections.Counter[str] = collections.Counter()
    for family in ("slides", "sources"):
        for record in deck_context.get(family, []) or []:
            text = f"{record.get('title') or ''} {record.get('text') or ''}"
            tokens = _signal_tokens(text)
            for strand, terms in COVERAGE_STRAND_TERMS.items():
                hit_count = len(tokens & terms)
                if hit_count:
                    counts[strand] += hit_count
    return [strand for strand, _ in counts.most_common()]


def _replaceable_selected_index(
    selected: list[EventCandidate],
    *,
    deck_strands: list[str],
    replacement_strand: str,
) -> int:
    deck_strand_set = set(deck_strands)
    selected_strands = [_candidate_coverage_strands(candidate) & deck_strand_set for candidate in selected]
    strand_counts: collections.Counter[str] = collections.Counter()
    for strands in selected_strands:
        strand_counts.update(strands)

    ranked_indexes = sorted(
        range(len(selected)),
        key=lambda index: (
            _event_kind_priority(selected[index]),
            selected[index].event_salience_score,
            selected[index].total_score,
            selected[index].name.lower(),
        ),
    )
    for index in ranked_indexes:
        strands = selected_strands[index]
        if replacement_strand in strands:
            continue
        if any(strand_counts[strand] <= 1 for strand in strands):
            continue
        return index
    return ranked_indexes[0]


def _is_program_anchored(candidate: EventCandidate, *, program_host: str) -> bool:
    """True when a candidate is tied to the real conference program.

    The program anchor is either a positive program-hit count (set by the
    program-HTML / workshop-inventory loaders) or a URL on ``program_host``,
    the accepted program-domain source-of-truth host emitted by those same
    loaders for the current conference workspace (e.g. ``iclr.cc`` for ICLR,
    ``cvpr.thecvf.com`` for CVPR) -- never a single hardcoded host, since this
    gate runs for whichever conference's deck is being synthesized.
    """
    if candidate.program_hits > 0:
        return True
    return bool(program_host) and program_host in (candidate.url or "").lower()


def _passes_program_gate(candidate: EventCandidate, *, program_host: str) -> bool:
    """Fail-closed gate for fabricated speaker/talk events.

    Non-talk kinds always pass. Talk/speaker kinds pass only when they are
    program-anchored, which keeps LLM-invented keynotes (no URL, no program hit)
    out of the slide-2 bookend.
    """
    if candidate.event_kind not in UNVERIFIABLE_TALK_KINDS:
        return True
    return _is_program_anchored(candidate, program_host=program_host)


def select_key_events(
    ranked: list[EventCandidate],
    *,
    top_n: int,
    deck_context: dict[str, Any],
    program_host: str = "",
) -> tuple[list[EventCandidate], dict[str, Any]]:
    """Select top events with conference salience and deck-strand coverage.

    Scalar score alone can over-select the densest source-chip cluster. For
    broad decks, keep the top ranking but backfill missing deck strands when a
    salient program event is available and passes the resonance gate.

    ``program_host`` is the current conference's own program-domain host
    (e.g. ``iclr.cc``, ``cvpr.thecvf.com``), used by the program-anchor gate
    below; an empty value (the default) means only ``program_hits`` can
    anchor a candidate, never a hardcoded conference's host.
    """
    eligible = [
        candidate
        for candidate in ranked
        if candidate.deck_resonance.get("action") == "keep"
    ]
    # Fail-closed program-anchor gate: drop fabricated speaker/talk events
    # (no program hit, no program_host URL) BEFORE the top_n slice and before
    # the coverage-backfill loop, which iterates this same `eligible` list and
    # could otherwise reintroduce a dropped candidate.
    gate_dropped = [
        candidate
        for candidate in eligible
        if not _passes_program_gate(candidate, program_host=program_host)
    ]
    eligible = [
        candidate
        for candidate in eligible
        if _passes_program_gate(candidate, program_host=program_host)
    ]
    selected = list(eligible[:top_n])
    selected_ids = {id(candidate) for candidate in selected}
    deck_strands = _deck_coverage_strands(deck_context)
    promoted: list[dict[str, Any]] = []
    max_candidate_rank = max(top_n * 8, 40)
    rank_lookup = {id(candidate): index for index, candidate in enumerate(ranked)}

    for strand in deck_strands[:top_n]:
        if any(strand in _candidate_coverage_strands(candidate) for candidate in selected):
            continue
        replacement: EventCandidate | None = None
        replacement_rank: int | None = None
        for candidate in eligible:
            rank = rank_lookup.get(id(candidate), 1_000_000) + 1
            if rank > max_candidate_rank:
                break
            if id(candidate) in selected_ids:
                continue
            if candidate.event_kind not in PROGRAM_SALIENCE_EVENT_KINDS:
                continue
            if strand not in _candidate_coverage_strands(candidate):
                continue
            replacement = candidate
            replacement_rank = rank
            break
        if replacement is None:
            continue
        replace_index = _replaceable_selected_index(
            selected,
            deck_strands=deck_strands[:top_n],
            replacement_strand=strand,
        )
        removed = selected[replace_index]
        if _event_kind_priority(removed) > _event_kind_priority(replacement):
            continue
        if (
            _event_kind_priority(removed) == _event_kind_priority(replacement)
            and removed.event_salience_score > replacement.event_salience_score
        ):
            continue
        selected_ids.discard(id(removed))
        selected[replace_index] = replacement
        selected_ids.add(id(replacement))
        promoted.append(
            {
                "strand": strand,
                "promoted": replacement.name,
                "promoted_rank": replacement_rank,
                "removed": removed.name,
            }
        )

    selected.sort(key=lambda candidate: rank_lookup.get(id(candidate), 1_000_000))
    return selected, {
        "policy": (
            "program_salience_score promotes keynotes, invited/oral talks, "
            "workshops, competitions, panels, and spotlights; coverage repair "
            "backfills missing deck strands only with resonance-approved events"
        ),
        "deck_strands": deck_strands,
        "coverage_promotions": promoted,
        "eligible_keep_candidate_count": len(eligible),
        "skipped_non_keep_candidate_count": len(ranked) - len(eligible),
        "program_gate_dropped": [
            {"name": candidate.name, "event_kind": candidate.event_kind}
            for candidate in gate_dropped
        ],
        "program_gate_dropped_count": len(gate_dropped),
    }


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_required_event_candidates(deck_dir: Path, workspace: Path) -> list[EventCandidate]:
    _, required_events = bookends.load_event_config(deck_dir, None)
    resolver = SourceLinkResolver.from_workspace(workspace)
    out: list[EventCandidate] = []
    for index, item in enumerate(required_events):
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        raw_detail = str(item.get("detail") or "").strip()
        kind = _event_kind(f"{name} {raw_detail}")
        detail = _display_detail(raw_detail, kind, f"{name} {raw_detail}")
        out.append(
            EventCandidate(
                name=name,
                detail=detail,
                url=canonical_public_url(item.get("url")) or resolver.resolve_item(item),
                event_kind=kind,
                source_family="required_event_override",
                source_path=_rel(deck_dir / bookends.DEFAULT_EVENT_OVERRIDES_FILENAME, workspace),
                deterministic_score=145 - index * 3 + EVENT_KIND_WEIGHTS.get(kind, 0),
                evidence=["required event override"],
            )
        )
    return out


def load_deck_source_candidates(deck_dir: Path, workspace: Path) -> list[EventCandidate]:
    render_spec_path = deck_dir / "render_spec.json"
    if not render_spec_path.is_file():
        return []
    render_spec = json.loads(render_spec_path.read_text(encoding="utf-8"))
    resolver = SourceLinkResolver.from_workspace(workspace)
    info = bookends._aggregate_sources(render_spec, link_resolver=resolver)
    out: list[EventCandidate] = []
    for rec in info.values():
        if not bookends._is_event(rec):
            continue
        title = _clean_text(str(rec.get("title") or "").strip())
        note = _clean_text(str(rec.get("note") or "").strip())
        speaker = (rec.get("speakers") or [""])[0]
        name = f"{speaker} — {bookends._short(title, 80)}" if speaker and title else speaker or bookends._short(title, 88)
        detail = bookends._short(title, 110)
        kind = _event_kind(f"{name} {detail}", str(rec.get("source_type") or ""))
        count = int(rec.get("count") or 0)
        out.append(
            EventCandidate(
                name=name,
                detail=detail or note,
                url=canonical_public_url(rec.get("url")) or resolver.resolve_item(rec),
                event_kind=kind,
                source_family="deck_source_chip",
                source_path=_rel(render_spec_path, workspace),
                deterministic_score=70 + count * 12 + EVENT_KIND_WEIGHTS.get(kind, 0),
                evidence=[f"deck source chip count={count}", f"source_type={rec.get('source_type') or ''}"],
                deck_source_count=count,
            )
        )
    return out


def load_ledger_candidates(deck_dir: Path, workspace: Path) -> list[EventCandidate]:
    path = deck_dir / "speaker_author_work_ledger.json"
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else []
    if not isinstance(entries, list):
        return []
    resolver = SourceLinkResolver.from_workspace(workspace)
    out: list[EventCandidate] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("resolved_title") or entry.get("title") or "").strip()
        note = str(entry.get("contribution_note") or "").strip()
        if not _looks_like_event(f"{title} {note}"):
            continue
        rank = _rank_value(entry.get("rank"))
        kind = _event_kind(f"{title} {note}")
        rank_bonus = max(0.0, 35.0 - min(rank, 35.0)) if rank < 1_000_000 else 0.0
        out.append(
            EventCandidate(
                name=bookends._short(_clean_text(title), 100),
                detail=bookends._short(bookends._first_sentence(note) if note else title, 120),
                url=canonical_public_url(entry.get("url"))
                or canonical_public_url(entry.get("source_url"))
                or resolver.resolve_item({"title": title, "detail": note}),
                event_kind=kind,
                source_family="speaker_author_work_ledger",
                source_path=_rel(path, workspace),
                deterministic_score=45 + rank_bonus + EVENT_KIND_WEIGHTS.get(kind, 0),
                evidence=[f"ledger rank={rank if rank < 1_000_000 else 'missing'}"],
                ledger_rank=rank if rank < 1_000_000 else None,
            )
        )
    return out


def load_cluster_plan_event_candidates(deck_dir: Path, workspace: Path) -> tuple[list[EventCandidate], dict[str, Any]]:
    path = deck_dir / "research_cluster_plan.json"
    if not path.is_file():
        return [], {"cluster_plan_present": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    clusters = raw.get("clusters") if isinstance(raw, dict) else []
    if not isinstance(clusters, list):
        return [], {"cluster_plan_present": True, "cluster_plan_event_candidate_count": 0}

    resolver = SourceLinkResolver.from_workspace(workspace)
    out: list[EventCandidate] = []
    seen: set[str] = set()
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        cluster_id = str(cluster.get("cluster_id") or "").strip()
        for source_role, role_bonus in (("primary_sources", 12.0), ("supporting_sources", 4.0)):
            sources = cluster.get(source_role)
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                raw_title = str(
                    source.get("resolved_title")
                    or source.get("title")
                    or ((source.get("works") or [""])[0] if isinstance(source.get("works"), list) else "")
                    or ""
                ).strip()
                note = str(source.get("contribution_note") or "").strip()
                source_id = str(source.get("source_id") or "").strip()
                if not raw_title:
                    continue
                has_event_prefix = bool(
                    re.match(
                        r"^\s*(?:workshop\s+(?:keynote|talk)|keynote|invited\s+talk|oral\s+(?:talk|session)|panel)\s*:",
                        raw_title,
                        flags=re.I,
                    )
                )
                if not has_event_prefix and not _looks_like_event(f"{raw_title} {note}"):
                    continue
                name, prefix_detail = _clean_prefixed_event_name(raw_title)
                kind = _event_kind(f"{raw_title} {note}", source_id)
                detail = prefix_detail or _event_detail_label(kind, raw_title)
                key = _norm(f"{name} {detail}")
                if not key or key in seen:
                    continue
                seen.add(key)
                rank = _rank_value(source.get("rank"))
                rank_bonus = max(0.0, 75.0 - min(rank, 75.0)) / 3.0 if rank < 1_000_000 else 0.0
                confidence = _rank_value(source.get("confidence"), 0.0)
                out.append(
                    EventCandidate(
                        name=bookends._short(name, 112),
                        detail=detail,
                        url=canonical_public_url(source.get("url"))
                        or canonical_public_url(source.get("source_url"))
                        or resolver.resolve_item(
                            {
                                "title": name,
                                "detail": note,
                                "source_id": source_id,
                                "works": source.get("works") or [],
                            }
                        ),
                        event_kind=kind,
                        source_family="research_cluster_plan",
                        source_path=_rel(path, workspace),
                        deterministic_score=82 + role_bonus + rank_bonus + confidence * 10 + EVENT_KIND_WEIGHTS.get(kind, 0),
                        evidence=[
                            f"cluster plan {source_role.rstrip('s')}",
                            f"cluster_id={cluster_id}",
                            f"source_id={source_id}",
                        ],
                        ledger_rank=rank if rank < 1_000_000 else None,
                    )
                )
    return out, {
        "cluster_plan_present": True,
        "cluster_plan_event_candidate_count": len(out),
    }


def load_program_html_candidates(workspace: Path) -> tuple[list[EventCandidate], dict[str, Any]]:
    cache_root = workspace / "cache" / "acquisition-downloads"
    files = []
    files.extend(sorted((cache_root / "source-pages").glob("*.html")))
    files.extend(sorted((cache_root / "item-page-probes").glob("*.html")))
    out: list[EventCandidate] = []
    seen: collections.Counter[str] = collections.Counter()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        path_rel = _rel(path, workspace)
        local_candidates: list[tuple[str, str]] = []
        for href, body in ANCHOR_RE.findall(text):
            title = _clean_text(body)
            if _looks_like_event(title) or "/event" in href or "/workshop" in href:
                local_candidates.append((title, href))
        for raw_title in JSON_TITLE_RE.findall(text):
            title = _clean_text(raw_title)
            if _looks_like_event(title):
                local_candidates.append((title, ""))
        for title, href in local_candidates:
            if not title or GENERIC_EVENT_TEXT_RE.match(title):
                continue
            key = _norm(title)
            seen[key] += 1
            if seen[key] > 1:
                continue
            kind = _event_kind(title)
            detail = _event_detail_label(kind, title)
            out.append(
                EventCandidate(
                    name=bookends._short(title, 110),
                    detail=detail,
                    url=canonical_public_url(href),
                    event_kind=kind,
                    source_family="full_program_html",
                    source_path=path_rel,
                    deterministic_score=18 + EVENT_KIND_WEIGHTS.get(kind, 0),
                    evidence=[f"program page={path.name}"],
                    program_hits=1,
                )
            )
    coverage = {
        "source_page_count": len(list((cache_root / "source-pages").glob("*.html"))),
        "item_page_probe_count": len(list((cache_root / "item-page-probes").glob("*.html"))),
        "html_files_scanned": len(files),
        "program_candidate_count": len(out),
    }
    return out, coverage


def load_workshop_inventory_candidates(workspace: Path) -> tuple[list[EventCandidate], dict[str, Any]]:
    path = workspace / "storage" / "workshops" / "talk_media_inventory.json"
    if not path.is_file():
        return [], {"workshop_inventory_present": False}
    raw = json.loads(path.read_text(encoding="utf-8"))
    workshops = raw.get("workshops") if isinstance(raw, dict) else []
    records = raw.get("records") if isinstance(raw, dict) else []
    resolver = SourceLinkResolver.from_workspace(workspace)
    out: list[EventCandidate] = []
    seen = set()
    for workshop in workshops if isinstance(workshops, list) else []:
        title = str(workshop.get("workshop_title") or workshop.get("title") or "").strip()
        if not title:
            continue
        key = _norm(title)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            EventCandidate(
                name=bookends._short(title, 112),
                detail="Workshop",
                url=resolver.resolve_item({**workshop, "title": title}),
                event_kind="workshop",
                source_family="workshop_inventory",
                source_path=_rel(path, workspace),
                deterministic_score=48 + EVENT_KIND_WEIGHTS["workshop"],
                evidence=["workshop inventory"],
                program_hits=1,
            )
        )
    for rec in records if isinstance(records, list) else []:
        title = str(rec.get("talk_title") or "").strip()
        speaker = str(rec.get("talk_speaker") or "").strip()
        workshop_title = str(rec.get("workshop_title") or "").strip()
        if not title or title.lower() == "slideslive embedded media":
            continue
        name = f"{speaker} — {title}" if speaker else title
        key = _norm(f"{workshop_title} {name}")
        if key in seen:
            continue
        seen.add(key)
        kind = _event_kind(f"{name} {workshop_title}", "workshop_talk")
        out.append(
            EventCandidate(
                name=bookends._short(name, 112),
                detail=bookends._short(workshop_title, 120),
                url=resolver.resolve_item({**rec, "title": title, "name": name}),
                event_kind=kind,
                source_family="workshop_talk_inventory",
                source_path=_rel(path, workspace),
                deterministic_score=42 + EVENT_KIND_WEIGHTS.get(kind, 0),
                evidence=["workshop talk media inventory"],
                program_hits=1,
            )
        )
    coverage = {
        "workshop_inventory_present": True,
        "workshop_count": raw.get("workshop_count"),
        "media_record_count": raw.get("media_record_count"),
        "slide_image_record_count": raw.get("slide_image_record_count"),
        "workshop_candidate_count": len(out),
    }
    return out, coverage


def load_gpt_researcher_candidates(path: Path | None, workspace: Path) -> tuple[list[EventCandidate], dict[str, Any]]:
    if path is None:
        return [], {"status": "staged_no_gpt_researcher_result"}
    if not path.is_file():
        return [], {"status": "missing_gpt_researcher_result", "path": path.as_posix()}
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[EventCandidate] = []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        items = (
            payload.get("key_events")
            or payload.get("events")
            or payload.get("top_key_events")
            or []
        )
        if isinstance(items, list):
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("title") or "").strip()
                if not name:
                    continue
                detail = str(item.get("detail") or item.get("rationale") or item.get("why") or "").strip()
                score = _rank_value(item.get("score"), default=max(0, 100 - index * 8))
                kind = _event_kind(f"{name} {detail}")
                out.append(
                    EventCandidate(
                        name=bookends._short(name, 112),
                        detail=bookends._short(detail, 140),
                        url=canonical_public_url(item.get("url")),
                        event_kind=kind,
                        source_family="gpt_researcher",
                        source_path=_rel(path, workspace),
                        gpt_score=70 + score + EVENT_KIND_WEIGHTS.get(kind, 0),
                        evidence=["GPT-researcher JSON result"],
                    )
                )
        return out, {"status": "loaded_json", "path": _rel(path, workspace), "candidate_count": len(out)}
    lines = [line.strip() for line in text.splitlines()]
    for index, line in enumerate(lines):
        if not re.match(r"^(?:[-*]|\d+[.)])\s+", line):
            continue
        clean = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line)
        if not _looks_like_event(clean):
            continue
        name = clean.split(" — ", 1)[0].split(":", 1)[0]
        kind = _event_kind(clean)
        out.append(
            EventCandidate(
                name=bookends._short(name, 112),
                detail=bookends._short(clean, 140),
                event_kind=kind,
                source_family="gpt_researcher",
                source_path=_rel(path, workspace),
                gpt_score=140 - min(index, 80) + EVENT_KIND_WEIGHTS.get(kind, 0),
                evidence=["GPT-researcher Markdown result"],
            )
        )
    return out, {"status": "loaded_markdown", "path": _rel(path, workspace), "candidate_count": len(out)}


def _load_transcript_summary_excerpt(workspace: Path, topic_id: str, limit: int = 6000) -> str:
    path = workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__transcripts.md"
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    path = workspace / "analysis" / "exhaustive_summaries" / f"{topic_id}__transcripts.json"
    if path.is_file():
        raw = json.loads(path.read_text(encoding="utf-8"))
        parts = [
            str(raw.get("title") or ""),
            str(raw.get("summary") or ""),
            "\n".join(str(x) for x in raw.get("key_points") or []),
            "\n".join(str(x) for x in raw.get("notable_evidence") or []),
        ]
        return "\n\n".join(part for part in parts if part)[:limit]
    return ""


def build_gpt_researcher_prompt(
    *,
    topic_id: str,
    deterministic_candidates: list[EventCandidate],
    coverage: dict[str, Any],
    transcript_excerpt: str,
    deck_resonance_summary: str,
    conference_label: str = "the conference",
) -> str:
    candidate_lines = []
    for index, candidate in enumerate(deterministic_candidates[:30], start=1):
        candidate_lines.append(
            f"{index}. {candidate.name} | kind={candidate.event_kind} | "
            f"source={candidate.source_family} | detail={candidate.detail} | "
            f"score={candidate.total_score} | "
            f"deck_resonance={candidate.deck_resonance.get('status', 'unknown')} "
            f"({candidate.deck_resonance_score:.1f})"
        )
    return f"""You are selecting the most important {conference_label} conference-program events for a topic presentation.

Topic: {_topic_label(topic_id)}

Goal:
Find the top 5 key events that should be visible on slide 2 under "Relevant events at {conference_label}".
You are not selecting the most-cited papers. You are looking for important talks and conference events
from the program and transcript evidence: workshops, keynotes, invited talks, oral talks/sessions,
spotlights, panels, tutorials, competitions, and workshop talks.

Use these priorities:
1. Conference-program salience: keynote, invited talk, oral/session, panel, workshop, spotlight.
2. Deck resonance: the event must help explain this deck's actual thesis, slide sections, source chips,
   or claims. A broad topic match is not enough.
3. Topic relevance: the event materially explains this topic's thesis or recurring evidence.
4. Transcript substance: prefer talks whose transcript/summary contains a concrete argument, framing,
   method, benchmark, or controversy, not just a generic event page.
5. Coverage balance: avoid picking five near-duplicates from the same workshop unless the topic truly demands it.
6. Evidence honesty: if the available transcript or deck-resonance evidence is weak, say so in the rationale.

Full-program coverage already scanned by the deterministic pass:
{json.dumps(coverage, indent=2, sort_keys=True)}

Deterministic candidates to audit and improve:
{chr(10).join(candidate_lines) if candidate_lines else "(none)"}

Deck-resonance audit summary:
{deck_resonance_summary or "(no deck-resonance audit available)"}

Transcript/exhaustive-summary excerpt for this topic:
\"\"\"
{transcript_excerpt or "(no transcript summary excerpt found)"}
\"\"\"

Return STRICT JSON only:
{{
  "topic_id": "{topic_id}",
  "key_events": [
    {{
      "rank": 1,
      "name": "speaker or event name",
      "detail": "short slide-ready detail",
      "event_kind": "keynote|invited_talk|oral|panel|workshop|session|talk|spotlight|tutorial|competition",
      "url": "https://...",
      "score": 0-100,
      "rationale": "why this belongs in the top 5",
      "evidence": ["program evidence", "transcript evidence"]
    }}
  ],
  "coverage_warnings": []
}}
"""


def _candidate_payload(candidate: EventCandidate, workspace: Path) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "detail": candidate.detail,
        "url": candidate.url,
        "event_kind": candidate.event_kind,
        "source_family": candidate.source_family,
        "source_path": candidate.source_path,
        "deterministic_score": round(candidate.deterministic_score, 3),
        "gpt_score": round(candidate.gpt_score, 3),
        "event_salience_score": round(candidate.event_salience_score, 3),
        "deck_resonance_score": round(candidate.deck_resonance_score, 3),
        "total_score": candidate.total_score,
        "program_hits": candidate.program_hits,
        "deck_source_count": candidate.deck_source_count,
        "ledger_rank": candidate.ledger_rank,
        "deck_resonance": candidate.deck_resonance,
        "evidence": candidate.evidence[:10],
    }


def write_bookend_overrides_from_synthesis(deck_dir: Path, payload: dict[str, Any]) -> Path:
    """Promote selected key events into the bookend override contract."""

    path = deck_dir / bookends.DEFAULT_EVENT_OVERRIDES_FILENAME
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                existing = {key: value for key, value in raw.items() if key != "required_events"}
        except json.JSONDecodeError:
            existing = {}
    required_events: list[dict[str, str]] = []
    for event in payload.get("top_key_events") or []:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name") or "").strip()
        if not name:
            continue
        required_events.append(
            {
                "name": name,
                "detail": str(event.get("detail") or "").strip(),
                "url": canonical_public_url(event.get("url")),
            }
        )
    existing["required_events"] = required_events
    write_json_atomic(path, existing)
    return path


def _deck_resonance_summary(candidates: list[EventCandidate], limit: int = 20) -> str:
    lines = []
    for index, candidate in enumerate(candidates[:limit], start=1):
        resonance = candidate.deck_resonance or {}
        support = []
        for slide in resonance.get("supporting_slides", [])[:2]:
            support.append(f"slide {slide.get('slide_id')}: {slide.get('title')}")
        for source in resonance.get("supporting_sources", [])[:2]:
            support.append(f"source {source.get('label')}: {source.get('title')}")
        lines.append(
            f"{index}. {candidate.name} -> {resonance.get('status', 'unknown')} "
            f"({candidate.deck_resonance_score:.1f}); support: "
            f"{'; '.join(support) if support else 'none'}"
        )
    return "\n".join(lines)


def synthesize_topic(
    *,
    workspace: Path,
    deck_dir: Path,
    topic_id: str,
    output_dir: Path,
    gpt_researcher_result: Path | None = None,
    top_n: int = DEFAULT_TOP_N,
    include_required_events: bool = True,
) -> dict[str, Any]:
    keywords = _topic_keywords(topic_id, deck_dir, workspace)
    candidates: list[EventCandidate] = []
    if include_required_events:
        candidates.extend(load_required_event_candidates(deck_dir, workspace))
    candidates.extend(load_deck_source_candidates(deck_dir, workspace))
    candidates.extend(load_ledger_candidates(deck_dir, workspace))
    cluster_candidates, cluster_coverage = load_cluster_plan_event_candidates(deck_dir, workspace)
    candidates.extend(cluster_candidates)
    program_candidates, program_coverage = load_program_html_candidates(workspace)
    workshop_candidates, workshop_coverage = load_workshop_inventory_candidates(workspace)
    gpt_candidates, gpt_status = load_gpt_researcher_candidates(gpt_researcher_result, workspace)

    topic_filtered_program: list[EventCandidate] = []
    for candidate in [*program_candidates, *workshop_candidates]:
        overlap = _topic_overlap_score(candidate, keywords)
        if overlap <= 0:
            continue
        candidate.deterministic_score += overlap
        candidate.evidence.append(f"topic keyword overlap score={overlap:.1f}")
        topic_filtered_program.append(candidate)

    candidates.extend(topic_filtered_program)
    candidates.extend(gpt_candidates)

    merged: list[EventCandidate] = []
    for candidate in candidates:
        _add_or_merge(merged, candidate)
    deck_context = _load_deck_resonance_context(deck_dir)
    for candidate in merged:
        resonance = score_deck_resonance(candidate, deck_context)
        candidate.deck_resonance = resonance
        candidate.deck_resonance_score = float(resonance.get("score") or 0.0)
        candidate.event_salience_score = score_event_salience(candidate)
        candidate.evidence.append(
            f"deck resonance {resonance['status']} score={candidate.deck_resonance_score:.1f}"
        )
        if candidate.event_salience_score:
            candidate.evidence.append(
                f"program event salience score={candidate.event_salience_score:.1f}"
            )
    ranked = sorted(
        merged,
        key=lambda item: (
            -item.total_score,
            -item.event_salience_score,
            -item.gpt_score,
            -item.deterministic_score,
            item.name.lower(),
        ),
    )
    try:
        program_host = load_conference_identity(workspace).site_domain.lower()
    except ConferenceIdentityError:
        # Fails closed to program_hits-only anchoring (see select_key_events),
        # never to another conference's host.
        program_host = ""
    selected, selection_audit = select_key_events(
        ranked,
        top_n=top_n,
        deck_context=deck_context,
        program_host=program_host,
    )

    transcript_excerpt = _load_transcript_summary_excerpt(workspace, topic_id)
    prompt = build_gpt_researcher_prompt(
        topic_id=topic_id,
        deterministic_candidates=ranked,
        coverage={**program_coverage, **workshop_coverage, **cluster_coverage},
        transcript_excerpt=transcript_excerpt,
        deck_resonance_summary=_deck_resonance_summary(ranked),
        conference_label=conference_name(workspace, default="the conference"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "gpt_researcher_prompt.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    payload = {
        "schema_version": 1,
        "topic_id": topic_id,
        "top_n": top_n,
        "status": "succeeded",
        "gpt_researcher": gpt_status,
        "program_coverage": {**program_coverage, **workshop_coverage, **cluster_coverage},
        "inputs": {
            "deck_dir": _rel(deck_dir, workspace),
            "required_event_overrides_loaded": include_required_events,
            "gpt_researcher_prompt": _rel(prompt_path, workspace),
            "gpt_researcher_result": _rel(gpt_researcher_result, workspace)
            if gpt_researcher_result
            else None,
        },
        "top_key_events": [_candidate_payload(candidate, workspace) for candidate in selected],
        "deck_resonance_gate": {
            "status_counts": dict(collections.Counter(
                candidate.deck_resonance.get("status", "unknown")
                for candidate in selected
            )),
            "weak_top_key_events": [
                candidate.name
                for candidate in selected
                if candidate.deck_resonance.get("status") == "weak"
            ],
            "promotion_rule": (
                "Do not promote weak events into bookend_event_overrides.json "
                "without explicit editorial approval or a documented replacement search."
            ),
        },
        "selection_audit": selection_audit,
        "candidate_count": len(ranked),
        "deterministic_candidate_count": len(candidates) - len(gpt_candidates),
        "gpt_candidate_count": len(gpt_candidates),
    }
    write_json_atomic(output_dir / "key_event_synthesis.json", payload)
    return payload


def _topic_dirs(workspace: Path, topic: str) -> list[Path]:
    decks = workspace / "decks"
    if topic != "all":
        return [decks / topic]
    return sorted(path for path in decks.glob("[0-9][0-9]_*") if path.is_dir())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--topic", default="all", help="Topic id or 'all'.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--gpt-researcher-result",
        type=Path,
        help="Optional GPT-researcher JSON/Markdown result for a single topic.",
    )
    parser.add_argument(
        "--output-subdir",
        default="key_event_synthesis",
        help="Deck-local output directory name.",
    )
    parser.add_argument(
        "--ignore-required-events",
        action="store_true",
        help=(
            "Do not load existing bookend_event_overrides.json required_events "
            "as candidates. Use when rerunning after an editorial audit so stale "
            "slide-2 selections do not bias the replacement search."
        ),
    )
    parser.add_argument(
        "--write-bookend-overrides",
        action="store_true",
        help=(
            "Write selected top_key_events to each deck's bookend_event_overrides.json "
            "after synthesis succeeds."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"workspace does not exist: {workspace}")
    topics = _topic_dirs(workspace, args.topic)
    if args.gpt_researcher_result is not None and len(topics) != 1:
        raise ValueError("--gpt-researcher-result is only supported with a single --topic")
    results = []
    for deck_dir in topics:
        if not deck_dir.is_dir():
            raise FileNotFoundError(f"deck directory does not exist: {deck_dir}")
        topic_id = deck_dir.name
        payload = synthesize_topic(
            workspace=workspace,
            deck_dir=deck_dir,
            topic_id=topic_id,
            output_dir=deck_dir / args.output_subdir,
            gpt_researcher_result=args.gpt_researcher_result.expanduser().resolve()
            if args.gpt_researcher_result
            else None,
            top_n=max(1, int(args.top_n)),
            include_required_events=not bool(args.ignore_required_events),
        )
        override_path = None
        if args.write_bookend_overrides:
            override_path = write_bookend_overrides_from_synthesis(deck_dir, payload)
        results.append(
            {
                "topic_id": topic_id,
                "output": payload["inputs"]["gpt_researcher_prompt"],
                "bookend_overrides": _rel(override_path, workspace) if override_path else None,
                "top_key_events": [
                    {"name": item["name"], "score": item["total_score"]}
                    for item in payload["top_key_events"]
                ],
                "gpt_researcher_status": payload["gpt_researcher"]["status"],
            }
        )
    print(json.dumps({"status": "succeeded", "topics": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
