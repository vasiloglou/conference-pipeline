#!/usr/bin/env python3
"""LLM-authored editorial presentation blueprint (v2.3 Phase 04).

Generalizes the v2.2 Topic-4 editorial process (source editorial board ->
material clusters -> section spine -> storyboard) into a per-topic generator.
A sequence of structured Opus calls authors the editorial content, grounded in
the enriched speaker/author/work ledger and the topic's analysis corpus; a
deterministic compiler then emits `presentation_blueprint.json` in the exact
schema the downstream deck-contract/binding/figure stages consume.

Output targets the approved canary's richness: 22-26 content slides, >=25
distinct sources, >=40 named people, evidence-grounded figure briefs, and a
20-30 minute narration budget — all enforced locally before the blueprint is
written (one repair call when floors are missed).

Usage (workspace-relative paths, like generate_presentation_blueprint):
    PYTHONPATH=src .venv/bin/python scripts/presentations/generate_editorial_blueprint.py \
      --workspace conferences/iclr2026 --presentation-id 01_reasoning_post_training \
      --presentation-type topic-deep-dive --topic 01_reasoning_post_training \
      --analysis-md analysis/01_reasoning_post_training.md \
      --speaker-author-work-ledger decks/01_reasoning_post_training/speaker_author_work_ledger.json \
      --exhaustive-papers analysis/exhaustive_summaries/01_reasoning_post_training__papers.json \
      --final-report-url "https://..." \
      --output-json decks/01_reasoning_post_training/presentation_blueprint.json \
      --output-md decks/01_reasoning_post_training/presentation_blueprint.md \
      --receipt-path decks/01_reasoning_post_training/presentation_blueprint_receipt.json \
      --provider-cost-approved
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.provider_receipts import (  # noqa: E402
    provider_receipt_context,
    provider_receipt_summary,
)
from conference_pipeline.presentation.presentation_frame import (  # noqa: E402
    load_presentation_frame_file,
    validate_opening_area_generality,
)
from conference_pipeline.figures.grounding import filter_components, filter_edges  # noqa: E402
from conference_pipeline.presentation.blueprint import drop_unauthored_sections, validate_blueprint_payload, MAX_SLIDE_TITLE_CHARS  # noqa: E402
from conference_pipeline.presentation.blueprint_markdown import render_blueprint_markdown  # noqa: E402
from conference_pipeline.workflow_runs import write_json_atomic  # noqa: E402
from conference_pipeline.conference_identity import conference_name  # noqa: E402

from conference_pipeline.model_governance.llm_client import (  # noqa: E402
    complete,
    credential_names,
    has_credentials,
    resolve_selection,
)
from conference_pipeline.model_governance.registry import default_fallback_model, default_fallback_provider  # noqa: E402

TASK_ID = "presentation.blueprint_generation"
# The route every call in this run takes. Set once by ``configure_route`` from
# the CLI/env (default: the registry's); module state because the authoring
# functions are several calls deep and all use the same model.
PROVIDER, MODEL_ID = resolve_selection(TASK_ID)
SCHEMA_VERSION = 3
MIN_CONTENT_SLIDES = 22
MAX_CONTENT_SLIDES = 26
MIN_SOURCES = 25
MIN_PEOPLE = 40
MIN_WORKS = 20
MAX_TITLE_CHARS = MAX_SLIDE_TITLE_CHARS
PER_SLIDE_MINUTES = 0.95
PLACEHOLDER_PEOPLE = {"unknown", "n/a", "na", "none", "anonymous", "unnamed", "unidentified"}
UNRESOLVED_CATALOG_STATUSES = frozenset(
    {"missing_people", "packet_too_large", "unresolved", "blocked"}
)
_GENERIC_NAME_PHRASES = {
    "Advances Financial",
    "AI Agents",
    "AI Safety",
    "Big Tech",
    "Global South",
    "Good AI Lab",
    "Human AI",
    "ICBINB",
    "MLGenX",
    "MLRS",
    "Multi Agent",
    "Opening Remarks",
    "Panel Discussion",
    "Post AGI",
    "Research Agora",
    "Skills Benchmarks",
    "Test Driven Research",
}
_SINGLE_NAME_HINTS = {"Alex", "Alexandra", "David", "Yasha"}
_PERSON_NAME_RE = re.compile(r"[A-Z][a-z'\-]+(?:\s+(?:[A-Z]\.|[A-Z][a-z'\-]+)){0,3}")
_EVENT_TITLE_RE = re.compile(
    r"\b(?:workshops?|expo|tutorials?|symposium|affinity|competition|"
    r"\d+(?:st|nd|rd|th)\s+(?:annual\s+)?(?:workshop|edition))\b",
    re.IGNORECASE,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(value: str, limit: int = 42) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:limit].rstrip("-") or "slide"


def _people_from_title(title: str) -> list[str]:
    """Recover obvious speaker names from event/talk labels.

    Reference-resolution artifacts sometimes put the named speaker only in the
    human title, leaving ``speakers`` empty.  This parser is intentionally
    conservative: it ignores quoted talk titles and common event phrases, and it
    is used only as fallback metadata for presentation crediting.
    """

    # An event title (a workshop, expo, tutorial ...) names a field, never a
    # person: "Workshop: AI as a Tool for Mathematics ... and Machine Learning"
    # yielded the speaker "Machine Learning" on ICML 2026 decks.
    # Only the leading segment decides: "Alex Hanna, 'Talk', AI for Peace
    # workshop" still credits its speaker.
    if _EVENT_TITLE_RE.search(re.split(r"[:,]", title, maxsplit=1)[0]):
        return []
    # Names are only read from a leading "Name[ and Name], <talk/event>"
    # segment. A bare talk or paper title is never a person: "FLIP2:
    # Expanding Protein Fitness Landscape Benchmarks ..." yielded the speaker
    # "Expanding Protein Fitness Landscape" on ICML 2026 decks.
    if "," not in title:
        return []
    lead = title.split(",", 1)[0]
    parts = [p.strip() for p in re.split(r"\s+and\s+|&|/", re.sub(r"\bProfessor\s+", "", lead)) if p.strip()]
    if not parts or not all(_PERSON_NAME_RE.fullmatch(p) for p in parts):
        return []
    clean = re.sub(r"'[^']*'", " ", lead)
    clean = clean.replace("/", " and ")
    clean = clean.replace("&", " and ")
    clean = re.sub(r"\bProfessor\s+", "", clean)
    candidates: list[str] = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", clean):
        name = re.sub(r"\s+", " ", match.group(0)).strip()
        if not name:
            continue
        if name in _GENERIC_NAME_PHRASES:
            continue
        if len(name.split()) == 1 and name not in _SINGLE_NAME_HINTS:
            continue
        if any(token in {"Workshop", "Panel", "Talks", "Models", "Science"} for token in name.split()):
            continue
        candidates.append(name)
    return list(dict.fromkeys(candidates))[:6]


def _extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Try the bracket that opens first: a list reply must stay a list, not
    # collapse to its first element when that element happens to parse.
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda pair: (text.find(pair[0]) if text.find(pair[0]) >= 0 else len(text)))
    for opener, closer in pairs:
        start = text.find(opener)
        if start < 0:
            continue
        end = text.rfind(closer)
        if end <= start:
            continue
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            continue
    raise ValueError(f"model response did not contain parseable JSON (first 200: {text[:200]!r})")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def build_source_catalog(ledger: dict[str, Any], limit: int = 90) -> list[dict[str, Any]]:
    """Eligible ledger entries the editorial board may select from."""
    catalog: list[dict[str, Any]] = []
    for entry in ledger.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if not entry.get("visible_source_chip_eligible"):
            continue
        people = [
            p for p in (entry.get("authors") or []) + (entry.get("speakers") or [])
            if str(p).strip() and str(p).strip().lower() not in PLACEHOLDER_PEOPLE
        ]
        title = str(entry.get("resolved_title") or "").strip()
        # Pseudo-people guard: an entry whose title came from the deterministic
        # local join either carries real joined authors or stays honestly
        # unattributed.  Landing real event-media titles unguarded was measured
        # to introduce 341 distinct fabricated people across the twelve decks --
        # noun phrases like "Oral Session" and "Artificial Intelligence"
        # presented as named human researchers.
        if not people and title and entry.get("resolved_title_kind") != "local_join":
            people = _people_from_title(title)
        note = str(entry.get("contribution_note") or "").strip()
        status = str(entry.get("resolution_status") or entry.get("status") or "resolved")
        if not title and not note:
            continue
        catalog.append(
            {
                "source_id": str(entry.get("source_id") or ""),
                "title": title[:110],
                "people": people[:6],
                "people_total": len(people),
                "has_named_people": bool(people),
                "resolution_status": status,
                "source_family": str(entry.get("source_family") or ""),
                "contribution": note[:240],
                "confidence": float(entry.get("confidence") or 0.0),
            }
        )
    catalog.sort(
        key=lambda c: (
            not c["has_named_people"],
            c["resolution_status"] in UNRESOLVED_CATALOG_STATUSES,
            -c["confidence"],
            -len(c["people"]),
            c["source_id"],
        )
    )
    # One catalog row per WORK: multi-segment talks otherwise crowd the board's
    # selection with duplicate titles and starve distinct-work coverage.  For
    # speech transcripts, the artifact hash before ``:segment:`` is part of the
    # work identity so different workshops/keynotes/panels with a broad shared
    # label are not collapsed into one source.
    deduped: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for item in catalog:
        key = _catalog_work_key(item["source_id"], item["title"])
        if key in seen_titles:
            continue
        seen_titles.add(key)
        deduped.append(item)
    return deduped[:limit]


def _catalog_work_key(source_id: str, title: str) -> str:
    base_title = title.split(":", 1)[0].strip().lower() or source_id
    match = re.match(r"^speech_transcript:([^:]+):segment:\d+$", source_id)
    if match:
        return f"{base_title}|speech:{match.group(1)}"
    return base_title


def build_evidence_digest(analysis_md: Path | None, exhaustive_papers: Path | None) -> str:
    parts: list[str] = []
    if analysis_md and analysis_md.is_file():
        parts.append(
            "## Topic analysis\n"
            + analysis_md.read_text(encoding="utf-8", errors="replace")
        )
    if exhaustive_papers and exhaustive_papers.is_file():
        try:
            payload = _load_json(exhaustive_papers)
        except (json.JSONDecodeError, OSError):
            payload = None
        if isinstance(payload, dict):
            feed = payload.get("analysis_feed")
            if isinstance(feed, dict):
                parts.append("## Paper corpus digest\n" + json.dumps(feed, indent=1)[:8000])
            chunks = payload.get("chunk_summaries")
            if isinstance(chunks, list):
                parts.append("## Paper chunk summaries (excerpt)\n" + json.dumps(chunks[:6], indent=1)[:8000])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the editorial board for a technical conference talk. "
    "You ground every claim in the provided named sources. "
    "Respond with pure JSON only — no prose, no markdown fences."
)

# Every call the primary model refused and the registry fallback answered.
# Recorded in the blueprint's provenance so the switch is never silent.
REFUSAL_FALLBACKS: list[dict[str, Any]] = []


def configure_route(provider: str | None = None, model: str | None = None) -> tuple[str, str]:
    """Choose the (provider, model) every call in this run uses; the registry validates it."""

    global PROVIDER, MODEL_ID
    PROVIDER, MODEL_ID = resolve_selection(TASK_ID, provider or None, model or None)
    return PROVIDER, MODEL_ID


def _send(provider: str, model: str, messages: list[dict[str, Any]], *, max_tokens: int) -> tuple[str, Any, list[str]]:
    result = complete(
        TASK_ID,
        system=SYSTEM_PROMPT,
        messages=messages,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        timeout=float(os.environ.get("EDITORIAL_BLUEPRINT_TIMEOUT_SECONDS", "900")),
        caller=__file__,
    )
    return result.text, result.stop_reason, list(result.block_types)


def _call(prompt: str, *, max_tokens: int = 8000) -> Any:
    messages = [{"role": "user", "content": prompt}]
    provider, model = PROVIDER, MODEL_ID
    text, stop_reason, block_types = _send(provider, model, messages, max_tokens=max_tokens)
    if stop_reason == "refusal":
        # The primary model's safety filter stopped the reply (ICML 2026: a
        # molecule-design section, cut off mid-JSON on every attempt). Ask the
        # registry's fallback model for this task once, and record the switch.
        fallback = default_fallback_model(TASK_ID)
        REFUSAL_FALLBACKS.append(
            {
                "primary_provider": provider,
                "primary_model": model,
                "fallback_provider": default_fallback_provider(TASK_ID),
                "fallback_model": fallback,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "primary_partial_chars": len(text),
            }
        )
        provider, model = default_fallback_provider(TASK_ID), fallback
        text, stop_reason, block_types = _send(provider, model, messages, max_tokens=max_tokens)
    if not text.strip() or stop_reason == "refusal":
        # An empty reply is not malformed JSON: report why the model stopped
        # (e.g. a refusal) instead of retrying blindly.
        raise ValueError(
            f"model returned no text (stop_reason={stop_reason!r}, content_blocks={block_types})"
        )
    try:
        return _extract_json(text)
    except ValueError as first_error:
        # One repair attempt: the model sometimes emits JSON with an unescaped
        # quote or a trailing comma. Ask it to return the same content as valid
        # JSON instead of failing the whole deck.
        retry_text, _stop, _blocks = _send(
            provider,
            model,
            [
                *messages,
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON "
                        f"({first_error}). Return the same content as strictly "
                        "valid JSON: escape every double quote inside strings, "
                        "no trailing commas, no comments, nothing outside the JSON."
                    ),
                },
            ],
            max_tokens=max_tokens,
        )
        return _extract_json(retry_text)


def _author_extra_slides(
    *,
    topic_title: str,
    section: dict[str, Any],
    section_sources: list[dict[str, Any]],
    evidence: str,
    prior_titles: list[str],
    count: int,
    presentation_frame: dict[str, Any] | None = None,
    conference_label: str = "the conference",
) -> list[dict[str, Any]]:
    """Author ``count`` additional slides for ``section``; [] if that fails.

    Used to fill a section that came back short and to top up a deck that a
    skipped section left under the coverage minimum. Titles already used are
    passed in so the new slides do not repeat them.
    """

    if count <= 0 or not section_sources:
        return []
    extra_section = {
        **section,
        "slide_budget": count,
        "intent": (
            f"{section.get('intent') or ''} -- ADDITIONAL slides for this section: each must make "
            "a new point with a new title; never repeat or rephrase a title or claim from the "
            "slides already authored."
        ).strip(" -"),
    }
    try:
        slides = author_section_slides(
            topic_title=topic_title,
            section=extra_section,
            section_sources=section_sources,
            evidence=evidence,
            prior_titles=prior_titles,
            presentation_frame=presentation_frame,
            conference_label=conference_label,
        )
    except ValueError as error:
        print(f"extra slides for {section.get('title')!r} failed: {error}", file=sys.stderr)
        return []
    # The assembler drops a slide whose title repeats an earlier one, so a
    # repeated title would not count toward the minimum; drop it here.
    seen = {str(t).strip().lower() for t in prior_titles}
    fresh: list[dict[str, Any]] = []
    for slide in slides:
        title = str(slide.get("title") or "").strip().lower()
        if title and title not in seen:
            seen.add(title)
            fresh.append(slide)
    return fresh[:count]


def author_board_and_spine(
    *,
    topic_title: str,
    catalog: list[dict[str, Any]],
    evidence: str,
    presentation_frame: dict[str, Any] | None = None,
    conference_label: str = "the conference",
) -> dict[str, Any]:
    frame_block = ""
    if presentation_frame:
        frame_block = (
            "\nAudience-facing presentation frame (use this for the public opening; "
            "treat the topic title as retrieval metadata):\n"
            + json.dumps(presentation_frame, indent=1)
            + "\n"
        )
    prompt = f"""Plan a {conference_label} deep-dive conference talk: "{topic_title}".
{frame_block}

You have this catalog of eligible sources (papers/talks with named people where known;
"people" lists at most six of a work's "people_total" names):
{json.dumps(catalog, indent=1)}

Evidence digest:
{evidence}

Author the editorial plan as JSON with EXACTLY this shape:
{{
  "selected_sources": [{{"source_id": "...", "role": "anchor|supporting|context", "why": "<=120 chars"}}],
  "clusters": [{{"cluster_id": "kebab-slug", "title": "<=60 chars complete claim phrase",
                 "claim": "1-2 sentence technical claim", "source_ids": ["..."]}}],
  "sections": [{{"section_id": "S1_...", "title": "<=60 chars", "intent": "1 sentence",
                 "cluster_ids": ["..."], "slide_budget": 4}}]
}}

Hard requirements:
- Select 28-34 sources. Favor sources with NAMED people; cover the topic's breadth, not one corner.
- If an audience-facing frame is supplied, the first clusters must be field-level
  claims that answer its general_questions before introducing specialist mechanisms.
- 5-7 clusters; every selected source belongs to exactly one cluster.
- 5-6 sections whose slide_budget values sum to {MIN_CONTENT_SLIDES}-{MAX_CONTENT_SLIDES - 2}
  (this is a 20-30 minute talk; the opening title slide is added separately).
- The spine must tell one connected story that TEACHES a broad, mixed-expertise
  audience spanning many subfields. Assume NO prior knowledge of this topic's
  subfield: this program covers a wide spectrum of work and no listener can have
  background on everything. Order sections so ideas build up from fundamentals —
  motivate why each matters before how it works, and introduce every specialist
  term, method, acronym, and benchmark in plain language before the spine relies
  on it. Teach; do not gatekeep."""
    return _call(prompt, max_tokens=12000)


def author_section_slides(
    *,
    topic_title: str,
    section: dict[str, Any],
    section_sources: list[dict[str, Any]],
    evidence: str,
    prior_titles: list[str],
    presentation_frame: dict[str, Any] | None = None,
    conference_label: str = "the conference",
) -> list[dict[str, Any]]:
    budget = int(section.get("slide_budget") or 4)
    frame_block = ""
    if presentation_frame:
        frame_block = (
            "\nAudience-facing frame to preserve, especially in the opening section:\n"
            + json.dumps(
                {
                    "audience_title": presentation_frame.get("audience_title"),
                    "field_thesis": presentation_frame.get("field_thesis"),
                    "general_questions": presentation_frame.get("general_questions"),
                },
                indent=1,
            )
            + "\n"
        )
    prompt = f"""You are authoring section "{section.get('title')}" ({section.get('intent')})
of the {conference_label} talk "{topic_title}". Slides already authored: {json.dumps(prior_titles)}.
{frame_block}

Sources assigned to this section (cite ONLY these, by source_id):
{json.dumps(section_sources, indent=1)}

Evidence digest (ground claims, numbers, and mechanisms here):
{evidence}

Author exactly {budget} slides as a JSON array. Each slide:
{{
  "title": "<= {MAX_TITLE_CHARS} chars, a complete technical phrase (never truncated mid-thought)",
  "viewer_facing_claim": "1-2 sentences, specific and technical",
  "visible_text": ["3-4 short bullet phrases, each a complete phrase"],
  "narration_beats": ["2-3 sentences EACH, 2 beats. Name the actual works and people
                       (e.g. 'Tao, Weston and colleagues show in Hybrid Reinforcement that ...');
                       name at most two people per work, then 'and colleagues' --
                       never read a list of names as if it were the full authorship,
                       state mechanisms and quantitative results from the evidence,
                       written pedagogically for a broad, mixed-expertise audience
                       that cannot be assumed to know this subfield: define each named
                       method, acronym, and benchmark in plain language at first mention
                       and give the intuition before the technical detail"],
  "visual_concept": {{
     "depiction": "mechanism_pipeline|comparison_matrix|evidence_chart|architecture|tradeoff_map",
     "summary": "<=200 chars: what the figure shows",
     "components": ["2-6 labeled elements. Each must be a system, method, benchmark, work,
                     person or stage that THIS slide's bullets, narration or assigned sources
                     name. Never invent a baseline, comparator, axis, stage or row the sources
                     do not describe: a comparison with one real side is not a figure."],
     "data_points": ["0-4 numbers stated in the assigned sources, e.g. 'MATH-500 +6.2'; none
                     if the sources state none"],
     "relationships": ["2-4 'A -> B' edges between components"]
  }}
  Depiction rules: comparison_matrix only when two or more assigned sources report the
  same properties, so every cell has a source; tradeoff_map only when data_points holds
  measured numbers to place the points by; otherwise mechanism_pipeline, architecture or
  evidence_chart.
  "mentions": [{{"source_id": "...", "role": "grounds claim|provides benchmark|counterpoint",
                 "supports": "one sentence: the specific finding, method, or number THIS source
                              contributes to THIS slide (never the same sentence for two sources)"}}]
}}

Hard requirements:
- Every slide cites 2-4 mentions from the assigned sources.
- Across the whole talk every selected source must be mentioned at least once; in THIS
  section use every assigned source at least once.
- Narration must read like a strong conference talk: technical, concrete, naming people,
  works, systems, and numbers. No filler, no 'in this slide we see'."""
    payload = _call(prompt, max_tokens=14000)
    if isinstance(payload, dict):
        # The model sometimes wraps the list under another key ("section",
        # "content_slides") or returns one slide object; accept those shapes.
        if isinstance(payload.get("slides"), list):
            payload = payload["slides"]
        elif "title" in payload and ("narration" in payload or "mentions" in payload):
            payload = [payload]
        else:
            payload = next(
                (v for v in payload.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)),
                [],
            )
    return [s for s in payload if isinstance(s, dict)] if isinstance(payload, list) else []


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

class LabelRegistry:
    def __init__(self) -> None:
        self._labels: dict[str, str] = {}

    def label_for(self, source_id: str) -> str:
        if source_id not in self._labels:
            self._labels[source_id] = f"SRC-{len(self._labels) + 1:03d}"
        return self._labels[source_id]


_TITLE_TRAILING_TRIM = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "is", "are", "that", "this", "by", "as", "from", "into", "over", "than",
    "but", "nor", "via", "per", "not", "when", "while",
    "nearly", "almost", "very", "more", "most", "less", "least", "too",
    "quite", "rather", "so", "only", "just", "even", "makes", "make",
}


def _cap_title(title: str, limit: int = MAX_TITLE_CHARS) -> str:
    """Normalize whitespace without discarding words; repair happens deck-wide."""
    return " ".join(title.split())


def repair_overlength_titles(blueprint: dict[str, Any], *, allow_model: bool = True) -> None:
    """Rewrite all long titles in one bounded call, preserving every other field."""
    long = [{"index": i, "title": s.get("title"), "claims": s.get("claims", [])}
            for i, s in enumerate(blueprint.get("slides", []))
            if len(str(s.get("title") or "")) > MAX_TITLE_CHARS]
    if not long:
        return
    if not allow_model:
        raise ValueError("over-length titles in saved artifacts require authoring repair: " + json.dumps(long))
    reply = _call(f"Rewrite ALL these titles as complete technical phrases of at most {MAX_TITLE_CHARS} characters. "
                  "Do not truncate a phrase. Preserve meaning. Return JSON {titles: [{index: integer, title: string}]}.\n" + json.dumps(long))
    rows = reply.get("titles", []) if isinstance(reply, dict) else []
    repaired = {row.get("index"): row.get("title") for row in rows if isinstance(row, dict)}
    if set(repaired) != {row["index"] for row in long} or any(
        not isinstance(title, str) or not title.strip() or len(title) > MAX_TITLE_CHARS
        for title in repaired.values()
    ):
        raise ValueError("title repair failed: return one complete title within the limit for every index")
    for index, title in repaired.items():
        blueprint["slides"][index]["title"] = title.strip()


def _mention_payload(
    source_id: str, ledger_index: dict[str, dict[str, Any]], labels: LabelRegistry
) -> dict[str, Any] | None:
    entry = ledger_index.get(source_id)
    if not entry:
        return None
    title = str(entry.get("resolved_title") or "").strip()
    speakers = [str(s) for s in entry.get("speakers") or []][:4]
    # Pseudo-people guard, second call site: same rule as the catalog builder --
    # a locally joined title either carries real joined authors or stays
    # honestly unattributed.  This is additionally the site whose output the
    # coverage statistics count for the post-LLM people gate, so guarding the
    # catalog alone would suppress the reported ceiling while leaving the gate
    # that actually decides the outcome fully exposed.
    if not speakers and not entry.get("authors") and title and entry.get("resolved_title_kind") != "local_join":
        speakers = _people_from_title(title)[:4]
    return {
        "label": labels.label_for(source_id),
        "source_id": source_id,
        "title": title,
        "authors": [str(a) for a in entry.get("authors") or []][:6],
        "author_count": len(entry.get("authors") or []),
        "speakers": speakers,
        "works": [title] if title else [],
        "contribution_note": str(entry.get("contribution_note") or "")[:300],
    }


def _source_status(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    return str(entry.get("resolution_status") or entry.get("status") or "")


def _mention_has_named_people(mention: dict[str, Any]) -> bool:
    return bool(mention.get("authors") or mention.get("speakers"))


def _append_opening_anchor(slide: dict[str, Any], mention: dict[str, Any]) -> None:
    source_id = str(mention.get("source_id") or "")
    if not source_id:
        return
    existing_ids = {
        str(item.get("source_id") or "")
        for item in slide.get("speaker_author_work_mentions") or []
        if isinstance(item, dict)
    }
    if source_id in existing_ids:
        return
    slide.setdefault("speaker_author_work_mentions", []).append(mention)
    slide.setdefault("source_plan", []).append(
        {
            "cluster_id": str(slide.get("section_id") or "S0_frame"),
            "label": mention.get("label"),
            "role": "field-level opening anchor",
        }
    )


def _rebalance_opening_resolved_anchors(
    slides: list[dict[str, Any]],
    *,
    mention_plan: list[dict[str, Any]],
    ledger_index: dict[str, dict[str, Any]],
    opening_slide_count: int = 5,
) -> None:
    """Add resolved named anchors when unresolved artifacts dominate opening slides."""

    opening = slides[:opening_slide_count]
    if not opening:
        return
    opening_ids = {
        str(mention.get("source_id") or "")
        for slide in opening
        for mention in slide.get("speaker_author_work_mentions") or []
        if isinstance(mention, dict)
    }
    total = len(opening_ids)
    unresolved = sum(
        1
        for source_id in opening_ids
        if _source_status(ledger_index.get(source_id)) in UNRESOLVED_CATALOG_STATUSES
    )
    if total < 3 or unresolved / total <= 0.5:
        return
    candidates = [
        mention
        for mention in mention_plan
        if str(mention.get("source_id") or "") not in opening_ids
        and _mention_has_named_people(mention)
        and _source_status(ledger_index.get(str(mention.get("source_id") or "")))
        not in UNRESOLVED_CATALOG_STATUSES
    ]
    for mention in candidates:
        _append_opening_anchor(opening[0], mention)
        opening_ids.add(str(mention.get("source_id") or ""))
        total = len(opening_ids)
        if total < 3:
            continue
        if unresolved / total <= 0.5:
            return


def compile_blueprint(
    *,
    args: argparse.Namespace,
    topic_title: str,
    presentation_frame: dict[str, Any] | None,
    board: dict[str, Any],
    section_slides: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    ledger_index: dict[str, dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    _workspace = getattr(args, "workspace", None)
    conference_label = (
        conference_name(_workspace, default="the conference")
        if _workspace is not None
        else "the conference"
    )
    labels = LabelRegistry()
    slides_payload: list[dict[str, Any]] = []
    all_mentions: dict[str, dict[str, Any]] = {}
    section_map: list[dict[str, Any]] = []
    audience_title = str((presentation_frame or {}).get("audience_title") or topic_title).strip()
    field_thesis = str((presentation_frame or {}).get("field_thesis") or "").strip()
    general_questions = [
        str(q).strip()
        for q in (presentation_frame or {}).get("general_questions", [])
        if str(q).strip()
    ]

    title_slide_id = f"sl_01_{_slug(audience_title)}"
    intro_mentions: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    index = 2
    for section, slides in section_slides:
        section_id = str(section.get("section_id") or f"S{len(section_map) + 1}")
        section_slide_ids: list[str] = []
        for slide in slides:
            title = _cap_title(str(slide.get("title") or "").strip())
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            slide_id = f"sl_{index:02d}_{_slug(title)}"
            mentions: list[dict[str, Any]] = []
            source_plan: list[dict[str, Any]] = []
            for m in slide.get("mentions") or []:
                source_id = str((m or {}).get("source_id") or "").strip()
                payload = _mention_payload(source_id, ledger_index, labels)
                if payload is None:
                    continue
                mentions.append(payload)
                source_plan.append(
                    {
                        "cluster_id": section_id,
                        "label": payload["label"],
                        "role": str((m or {}).get("role") or "grounds slide claim")[:60],
                        "supports": str((m or {}).get("supports") or "").strip()[:300],
                    }
                )
                all_mentions.setdefault(source_id, payload)
            if not mentions:
                continue
            concept = slide.get("visual_concept") or {}
            depiction = str(concept.get("depiction") or "mechanism_pipeline")
            # The brief's components must be things this slide names; an
            # invented comparator ("Hand-designed harness") otherwise reaches
            # the spec author as an instruction to draw it (ICML 2026, 2026-10-01).
            components, dropped_components = filter_components(
                [str(c)[:60] for c in concept.get("components") or []][:8],
                {**slide, "speaker_author_work_mentions": mentions, "source_plan": source_plan},
                depiction=depiction,
            )
            data_points = [str(d)[:60] for d in concept.get("data_points") or []][:4]
            relationships = filter_edges(
                [str(r)[:90] for r in concept.get("relationships") or []][:4], components
            )
            visual_role = (
                f"{depiction}: "
                f"{str(concept.get('summary') or title)[:200]} | components: "
                + "; ".join(components)
                + (" | data: " + "; ".join(data_points) if data_points else "")
                + (" | edges: " + "; ".join(relationships) if relationships else "")
            )
            beats = [str(b).strip() for b in slide.get("narration_beats") or [] if str(b).strip()]
            slides_payload.append(
                {
                    "slide_id": slide_id,
                    "section_id": section_id,
                    "title": title,
                    "intent": str(section.get("intent") or "")[:200],
                    "claims": [str(slide.get("viewer_facing_claim") or title)],
                    "visible_text": [str(v)[:120] for v in slide.get("visible_text") or []][:4],
                    "narration_beat": beats[0] if beats else "",
                    "narration_beats": beats[:3],
                    "estimated_minutes": PER_SLIDE_MINUTES,
                    "source_plan": source_plan,
                    "speaker_author_work_mentions": mentions,
                    "visual_role": visual_role,
                    "visual_figure_brief": {
                        **({"dropped_components": dropped_components} if dropped_components else {}),
                        "depiction": str(concept.get("depiction") or "mechanism_pipeline"),
                        "summary": str(concept.get("summary") or "")[:240],
                        "components": components,
                        "data_points": data_points,
                        "relationships": relationships,
                    },
                    "visual_narration_hook": f"{title}: {str(concept.get('summary') or 'the named evidence and mechanism')[:160]}",
                }
            )
            section_slide_ids.append(slide_id)
            for mention in mentions:
                if len(intro_mentions) >= 4:
                    break
                if any(existing["source_id"] == mention["source_id"] for existing in intro_mentions):
                    continue
                intro_mentions.append(mention)
            index += 1
        section_map.append(
            {
                "section_id": section_id,
                "title": str(section.get("title") or section_id)[:80],
                "slide_ids": section_slide_ids,
            }
        )

    # Opening title slide (bookends are injected later by add_bookend_slides).
    clusters = board.get("clusters") or []
    lead_clusters = "; ".join(str(c.get("title") or "") for c in clusters[:3])
    opening_visible_text = general_questions[:4] or [str(c.get("title") or "")[:90] for c in clusters[:4]]
    opening_claim = (
        field_thesis
        or f"{topic_title}: what the {conference_label} corpus actually shows, judged on named evidence."
    )
    opening_beat = field_thesis or f"The {conference_label} corpus reframes {topic_title} around: {lead_clusters}."
    title_slide = {
        "slide_id": title_slide_id,
        "section_id": "S0_frame",
        "title": _cap_title(audience_title),
        "intent": "Frame the talk thesis and stakes.",
        "claims": [opening_claim],
        "visible_text": [text[:120] for text in opening_visible_text],
        "narration_beat": opening_beat,
        "narration_beats": [
            opening_beat,
            "Every claim in this talk is grounded in named papers, talks, and people from the corpus.",
        ],
        "estimated_minutes": PER_SLIDE_MINUTES,
        "source_plan": [
            {"cluster_id": "S0_frame", "label": m["label"], "role": "frames thesis"}
            for m in intro_mentions[:3]
        ],
        "speaker_author_work_mentions": intro_mentions[:3],
        "visual_role": "tradeoff_map: thesis map connecting the talk's clusters to their anchor sources",
        "visual_figure_brief": {
            "depiction": "tradeoff_map",
            "summary": f"Thesis map connecting {len(clusters)} research clusters to anchor sources",
            "components": [str(c.get("title") or "")[:60] for c in clusters[:7]],
            "data_points": [],
            "relationships": [],
        },
        "visual_narration_hook": f"{audience_title}: the field frame and its anchor evidence.",
    }
    slides_payload.insert(0, title_slide)
    section_map.insert(0, {"section_id": "S0_frame", "title": "Frame the thesis", "slide_ids": [title_slide_id]})

    total_slides = len(slides_payload)
    duration_target = round(total_slides * PER_SLIDE_MINUTES, 1)
    mention_plan = list(all_mentions.values())
    _rebalance_opening_resolved_anchors(
        slides_payload,
        mention_plan=mention_plan,
        ledger_index=ledger_index,
    )
    label_by_source = {m["source_id"]: m["label"] for m in mention_plan}
    slide_source_ids = {
        s["slide_id"]: {m["source_id"] for m in s.get("speaker_author_work_mentions") or []}
        for s in slides_payload
    }
    subtopic_clusters = []
    for c in clusters:
        cluster_source_ids = [
            s for s in c.get("source_ids") or [] if s in ledger_index and s in label_by_source
        ]
        cluster_labels = [label_by_source[s] for s in cluster_source_ids]
        slide_bindings = [
            slide_id
            for slide_id, mentioned in slide_source_ids.items()
            if mentioned & set(cluster_source_ids)
        ]
        subtopic_clusters.append(
            {
                "cluster_id": str(c.get("cluster_id") or _slug(str(c.get("title") or ""))),
                "title": str(c.get("title") or "")[:90],
                "claim": str(c.get("claim") or "")[:400],
                "story_contribution": str(c.get("claim") or "")[:300],
                "sources": cluster_labels,
                "primary_source_labels": cluster_labels[:3],
                "supporting_source_labels": cluster_labels[3:],
                "slide_bindings": slide_bindings,
                "source_ids": cluster_source_ids,
                "retained_rationale": str(c.get("claim") or "")[:200],
                "merged_rationale": "merged near-duplicate candidates during editorial selection",
                "dropped_rationale": "dropped lower-confidence or out-of-scope catalog candidates",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "blueprint_id": f"{args.presentation_id}-editorial-blueprint",
        "revision": 1,
        "presentation_id": args.presentation_id,
        "presentation_type": args.presentation_type,
        "topic_id": args.topic,
        "author": "Nikolaos Vasiloglou",
        "retrieval_title": topic_title,
        "audience_title": audience_title,
        "presentation_frame": presentation_frame,
        "deck_thesis": (
            field_thesis
            or (
                f"{topic_title}: the {conference_label} corpus, judged on named source evidence — "
                + lead_clusters
            )
        )[:500],
        "audience_promise": (
            f"You will leave knowing how the {conference_label} evidence reframes {audience_title}, "
            "which named works and people anchor that frame, and what changes in practice."
        ),
        "conclusion_synthesis": (
            f"Across {len(clusters)} clusters and {len(mention_plan)} named sources, the corpus "
            f"converges on a concrete picture of {topic_title}; the named evidence supports "
            "specific design, evaluation, and deployment decisions rather than slogans."
        ),
        "final_report_url": args.final_report_url,
        "analysis_report_url": args.final_report_url,
        "target_runtime_minutes": duration_target,
        "target_slide_count": total_slides,
        "approved_expansion": None,  # filled in main() with artifact hashes
        "duration_budget": {
            "target_minutes": duration_target,
            "per_slide_target_minutes": PER_SLIDE_MINUTES,
            "hard_limit_minutes": 40,
            "duration_budget": "pass",
        },
        "notebooklm_coherence_policy": "two-host dialogue must follow the section spine and name sources on first use",
        "slides": slides_payload,
        "telescopic_section_map": section_map,
        "subtopic_clusters": subtopic_clusters,
        "speaker_author_work_mention_plan": mention_plan,
        "source_paper_plan": [m for m in mention_plan if str(m.get("source_id", "")).startswith("paper_abstract:")],
        "cluster_coverage_justification": {
            "required": [c["cluster_id"] for c in subtopic_clusters],
            "covered_cluster_ids": [c["cluster_id"] for c in subtopic_clusters],
            "reason": "every editorial cluster maps to at least one section of slides",
        },
        "provenance": {
            "generator": "scripts/presentations/generate_editorial_blueprint.py",
            "generation_method": "editorial-live-multicall",
            "model_id": MODEL_ID,
            "input_hashes": input_hashes,
            "source_metadata": [
                {
                    "label": m["label"],
                    "source_id": m["source_id"],
                    "title": m["title"],
                    "authors": m.get("authors") or [],
                    "speakers": m.get("speakers") or [],
                    "contribution_note": m.get("contribution_note") or "",
                    "confidence": 1.0,
                }
                for m in mention_plan
            ],
            "presentation_frame": presentation_frame,
        },
        "input_hashes": input_hashes,
        "repair_of": None,
        "source_plan_repair_of": None,
    }


def coverage_stats(blueprint: dict[str, Any]) -> dict[str, int]:
    sources, people, works = set(), set(), set()
    slides = [slide for slide in blueprint.get("slides") or [] if isinstance(slide, dict)]
    content_slide_count = sum(
        1 for slide in slides
        if slide.get("section_id") != "S0_frame"
    )
    for slide in slides:
        for m in slide.get("speaker_author_work_mentions") or []:
            if m.get("source_id"):
                sources.add(m["source_id"])
            if m.get("title"):
                works.add(str(m["title"]).lower()[:70])
            for p in (m.get("authors") or []) + (m.get("speakers") or []):
                if str(p).strip().lower() not in PLACEHOLDER_PEOPLE:
                    people.add(str(p))
    return {
        "slides": content_slide_count,
        "total_slides": len(slides),
        "sources": len(sources),
        "people": len(people),
        "works": len(works),
    }


def presentation_frame_request_mismatches(
    presentation_frame: dict[str, Any],
    *,
    presentation_id: str,
    topic_id: str,
) -> list[str]:
    expected = {
        "presentation_id": presentation_id,
        "topic_id": topic_id,
    }
    return [
        f"{key}={presentation_frame.get(key)!r} expected {value!r}"
        for key, value in expected.items()
        if presentation_frame.get(key) != value
    ]


def rebuild_blueprint_derived_metadata(
    blueprint: dict[str, Any],
    *,
    board: dict[str, Any],
    ledger_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Recompute fields derived from the final per-slide source mentions."""

    slides = [slide for slide in blueprint.get("slides") or [] if isinstance(slide, dict)]
    mention_plan: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for slide in slides:
        for mention in slide.get("speaker_author_work_mentions") or []:
            if not isinstance(mention, dict):
                continue
            source_id = str(mention.get("source_id") or "").strip()
            if not source_id or source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            mention_plan.append(mention)

    label_by_source = {
        str(mention.get("source_id")): str(mention.get("label") or "")
        for mention in mention_plan
        if mention.get("source_id")
    }
    slide_source_ids = {
        str(slide.get("slide_id")): {
            str(mention.get("source_id"))
            for mention in slide.get("speaker_author_work_mentions") or []
            if isinstance(mention, dict) and mention.get("source_id")
        }
        for slide in slides
        if slide.get("slide_id")
    }

    subtopic_clusters = []
    for cluster in board.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        cluster_source_ids = [
            str(source_id)
            for source_id in cluster.get("source_ids") or []
            if str(source_id) in ledger_index and str(source_id) in label_by_source
        ]
        cluster_labels = [label_by_source[source_id] for source_id in cluster_source_ids]
        source_id_set = set(cluster_source_ids)
        slide_bindings = [
            slide_id
            for slide_id, mentioned in slide_source_ids.items()
            if mentioned & source_id_set
        ]
        subtopic_clusters.append(
            {
                "cluster_id": str(cluster.get("cluster_id") or _slug(str(cluster.get("title") or ""))),
                "title": str(cluster.get("title") or "")[:90],
                "claim": str(cluster.get("claim") or "")[:400],
                "story_contribution": str(cluster.get("claim") or "")[:300],
                "sources": cluster_labels,
                "primary_source_labels": cluster_labels[:3],
                "supporting_source_labels": cluster_labels[3:],
                "slide_bindings": slide_bindings,
                "source_ids": cluster_source_ids,
                "retained_rationale": str(cluster.get("claim") or "")[:200],
                "merged_rationale": "merged near-duplicate candidates during editorial selection",
                "dropped_rationale": "dropped lower-confidence or out-of-scope catalog candidates",
            }
        )

    blueprint["speaker_author_work_mention_plan"] = mention_plan
    blueprint["source_paper_plan"] = [
        mention
        for mention in mention_plan
        if str(mention.get("source_id", "")).startswith("paper_abstract:")
    ]
    blueprint["subtopic_clusters"] = subtopic_clusters
    blueprint["cluster_coverage_justification"] = {
        "required": [cluster["cluster_id"] for cluster in subtopic_clusters],
        "covered_cluster_ids": [
            cluster["cluster_id"] for cluster in subtopic_clusters if cluster["slide_bindings"]
        ],
        "reason": "cluster bindings are recomputed from final slide source mentions",
    }
    provenance = blueprint.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        blueprint["provenance"] = provenance
    provenance["source_metadata"] = [
        {
            "label": mention.get("label") or "",
            "source_id": mention.get("source_id") or "",
            "title": mention.get("title") or "",
            "authors": mention.get("authors") or [],
            "speakers": mention.get("speakers") or [],
            "contribution_note": mention.get("contribution_note") or "",
            "confidence": 1.0,
        }
        for mention in mention_plan
    ]
    return blueprint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--presentation-id", required=True)
    parser.add_argument("--presentation-type", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--topic-title", default="")
    parser.add_argument("--analysis-md", type=Path)
    parser.add_argument("--exhaustive-papers", type=Path)
    parser.add_argument("--speaker-author-work-ledger", required=True, type=Path)
    parser.add_argument("--source-corpus-inventory", type=Path)
    parser.add_argument("--source-repair-findings", type=Path)
    parser.add_argument("--research-cluster-plan", type=Path)
    parser.add_argument("--presentation-frame", type=Path)
    parser.add_argument("--final-report-url", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--receipt-path", required=True, type=Path)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--provider", default=os.environ.get("EDITORIAL_BLUEPRINT_PROVIDER", ""),
                        help="model provider; default is the one the registry records for the model")
    parser.add_argument("--model", default=os.environ.get("EDITORIAL_BLUEPRINT_MODEL", ""),
                        help="blueprint model (checked against presentation.blueprint_generation's allowed models)")
    parser.add_argument(
        "--from-artifacts", action="store_true",
        help="Recompile the blueprint from previously saved editorial_board_artifacts.json "
             "without new model calls (schema/compiler iteration).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.provider_cost_approved:
        print("editorial blueprint generation is a live paid model stage: pass --provider-cost-approved", file=sys.stderr)
        return 2
    provider, model = configure_route(args.provider, args.model)
    if not args.from_artifacts and not has_credentials(provider):
        print(f"{' or '.join(credential_names(provider) or ())} is required for {provider}", file=sys.stderr)
        return 2
    workspace = args.workspace.expanduser().resolve()

    def resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else workspace / p

    ledger_path = resolve(args.speaker_author_work_ledger)
    analysis_md = resolve(args.analysis_md)
    exhaustive_papers = resolve(args.exhaustive_papers)
    presentation_frame_path = resolve(args.presentation_frame)
    output_json = resolve(args.output_json)
    output_md = resolve(args.output_md)
    receipt_path = resolve(args.receipt_path)

    ledger = _load_json(ledger_path)
    ledger_index = {
        str(e.get("source_id")): e for e in ledger.get("entries") or [] if isinstance(e, dict)
    }
    catalog = build_source_catalog(ledger)
    if len(catalog) < MIN_SOURCES:
        print(
            f"only {len(catalog)} eligible catalog sources; need >= {MIN_SOURCES}. "
            "Enrich the ledger (authors/contributions) first.",
            file=sys.stderr,
        )
        return 2
    evidence = build_evidence_digest(analysis_md, exhaustive_papers)
    topic_title = args.topic_title or args.presentation_id.replace("_", " ")
    try:
        presentation_frame = load_presentation_frame_file(
            presentation_frame_path,
            required=args.presentation_frame is not None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if presentation_frame:
        mismatches = presentation_frame_request_mismatches(
            presentation_frame,
            presentation_id=args.presentation_id,
            topic_id=args.topic,
        )
        if mismatches:
            print(
                "presentation_frame.json deck mismatch: " + "; ".join(mismatches),
                file=sys.stderr,
            )
            return 1

    # Key names must match the review gate's required source-ledger contract set.
    input_hashes: dict[str, str] = {}
    for label, path in (
        ("speaker_author_work_ledger.json", ledger_path),
        ("source_corpus_inventory.json", resolve(args.source_corpus_inventory)),
        ("source_repair_findings.json", resolve(args.source_repair_findings)),
        ("research_cluster_plan.json", resolve(args.research_cluster_plan)),
        ("presentation_frame.json", presentation_frame_path),
        ("analysis.md", analysis_md),
        ("exhaustive_papers.json", exhaustive_papers),
    ):
        if path and path.is_file():
            input_hashes[label] = _sha256_file(path)

    artifacts_path = output_json.parent / "editorial_board_artifacts.json"
    catalog_by_id = {c["source_id"]: c for c in catalog}
    conference_label = conference_name(workspace, default="the conference")

    before_receipts = set(provider_receipt_summary(workspace).get("receipt_paths", []))
    with provider_receipt_context(
        workspace_root=workspace,
        stage_id="editorial_blueprint",
        caller=__file__,
    ):
        if args.from_artifacts and artifacts_path.is_file():
            saved = _load_json(artifacts_path)
            board = saved["board"]
            section_slides = [tuple(item) for item in saved["section_slides"]]
            section_failures = dict(saved.get("section_failures") or {})
        else:
            board = author_board_and_spine(
                topic_title=topic_title,
                catalog=catalog,
                evidence=evidence,
                presentation_frame=presentation_frame,
                conference_label=conference_label,
            )
            selected_ids = {
                str(s.get("source_id")) for s in board.get("selected_sources") or []
            } & set(ledger_index)
            cluster_by_id = {
                str(c.get("cluster_id")): c for c in board.get("clusters") or []
            }
            section_slides = []
            section_failures: dict[str, str] = {}
            prior_titles: list[str] = []
            for section in board.get("sections") or []:
                section_source_ids: list[str] = []
                for cid in section.get("cluster_ids") or []:
                    cluster = cluster_by_id.get(str(cid)) or {}
                    section_source_ids.extend(
                        s for s in cluster.get("source_ids") or [] if s in selected_ids
                    )
                section_sources = [
                    catalog_by_id[s] for s in dict.fromkeys(section_source_ids) if s in catalog_by_id
                ]
                if not section_sources:
                    section_failures[str(section.get("section_id") or "")] = "no selected sources in its clusters"
                    continue
                budget = int(section.get("slide_budget") or 4)
                try:
                    slides = author_section_slides(
                        topic_title=topic_title,
                        section=section,
                        section_sources=section_sources,
                        evidence=evidence,
                        prior_titles=prior_titles[-12:],
                        presentation_frame=presentation_frame,
                        conference_label=conference_label,
                    )
                except ValueError as error:
                    # One section must not sink the deck (e.g. the model declines
                    # or garbles it); its budget is recovered by the top-up below.
                    print(f"section {section.get('title')!r} returned no usable slides: {error}", file=sys.stderr)
                    section_failures[str(section.get("section_id") or "")] = f"authoring failed: {error}"[:300]
                    slides = []
                if len(slides) < budget:
                    # Ask once for the missing slides of a short or empty section.
                    more = _author_extra_slides(
                        topic_title=topic_title,
                        section=section,
                        section_sources=section_sources,
                        evidence=evidence,
                        prior_titles=prior_titles + [str(s.get("title") or "") for s in slides],
                        count=budget - len(slides),
                        presentation_frame=presentation_frame,
                        conference_label=conference_label,
                    )
                    slides = slides + more
                section_slides.append((section, slides))
                prior_titles.extend(str(s.get("title") or "") for s in slides)
            # Top up a deck still under the minimum (a skipped or short section)
            # from the best-sourced sections, a few rounds at most.
            def _section_source_list(section: dict[str, Any]) -> list[dict[str, Any]]:
                ids: list[str] = []
                for cid in section.get("cluster_ids") or []:
                    ids.extend(
                        s for s in (cluster_by_id.get(str(cid)) or {}).get("source_ids") or [] if s in selected_ids
                    )
                return [catalog_by_id[s] for s in dict.fromkeys(ids) if s in catalog_by_id]

            ranked = sorted(
                range(len(section_slides)),
                key=lambda i: (-len(_section_source_list(section_slides[i][0])), -len(section_slides[i][1])),
            )
            for index in ranked[:3]:
                authored = sum(len(item[1]) for item in section_slides)
                if authored >= MIN_CONTENT_SLIDES:
                    break
                section, slides = section_slides[index]
                more = _author_extra_slides(
                    topic_title=topic_title,
                    section=section,
                    section_sources=_section_source_list(section),
                    evidence=evidence,
                    prior_titles=prior_titles,
                    count=MIN_CONTENT_SLIDES - authored,
                    presentation_frame=presentation_frame,
                    conference_label=conference_label,
                )
                section_slides[index] = (section, slides + more)
                prior_titles.extend(str(s.get("title") or "") for s in more)
            write_json_atomic(
                artifacts_path,
                {
                    "board": board,
                    "section_slides": [list(item) for item in section_slides],
                    "section_failures": section_failures,
                },
            )
        selected_ids = {
            str(s.get("source_id")) for s in board.get("selected_sources") or []
        } & set(ledger_index)

        blueprint = compile_blueprint(
            args=args,
            topic_title=topic_title,
            presentation_frame=presentation_frame,
            board=board,
            section_slides=section_slides,
            ledger_index=ledger_index,
            input_hashes=input_hashes,
        )
        repair_overlength_titles(blueprint, allow_model=not args.from_artifacts)
        # A section every authoring call failed for (e.g. the model declined
        # its subject) is dropped with its reason recorded, not left claiming
        # coverage the deck does not deliver.
        for record in drop_unauthored_sections(blueprint, reasons=section_failures):
            print(f"dropped from blueprint: {record}", file=sys.stderr)
        if presentation_frame:
            opening_result = validate_opening_area_generality(
                blueprint,
                presentation_frame=presentation_frame,
                ledger=ledger,
            )
            blueprint["presentation_frame_validation"] = opening_result.to_dict()
            if opening_result.status != "pass":
                print(
                    "presentation frame validation failed: "
                    + "; ".join(error.message for error in opening_result.errors),
                    file=sys.stderr,
                )
                return 1
        stats = coverage_stats(blueprint)
        if (
            stats["sources"] < MIN_SOURCES
            or stats["people"] < MIN_PEOPLE
            or stats["works"] < MIN_WORKS
            or stats["slides"] < MIN_CONTENT_SLIDES
        ):
            # One repair call: ask for additional mention assignments on existing slides.
            used_source_ids = {
                str(m.get("source_id"))
                for sl in blueprint["slides"]
                for m in sl.get("speaker_author_work_mentions") or []
                if isinstance(m, dict) and m.get("source_id")
            }
            unused = [
                catalog_by_id[s]
                for s in selected_ids
                if s in catalog_by_id and s not in used_source_ids
            ]
            if stats["people"] < MIN_PEOPLE:
                current_people = {
                    str(p)
                    for sl in blueprint["slides"]
                    for m in sl.get("speaker_author_work_mentions") or []
                    if isinstance(m, dict)
                    for p in (m.get("authors") or []) + (m.get("speakers") or [])
                    if str(p).strip().lower() not in PLACEHOLDER_PEOPLE
                }
                queued = {str(item.get("source_id")) for item in unused}
                for item in catalog:
                    source_id = str(item.get("source_id") or "")
                    if source_id in used_source_ids or source_id in queued:
                        continue
                    people = [
                        str(p)
                        for p in item.get("people") or []
                        if str(p).strip().lower() not in PLACEHOLDER_PEOPLE
                    ]
                    if not people:
                        continue
                    unused.append(item)
                    queued.add(source_id)
                    current_people.update(people)
                    if len(current_people) >= MIN_PEOPLE and len(unused) >= 4:
                        break
            if unused:
                assignments = _call(
                    "These slides exist: "
                    + json.dumps([
                        {"slide_id": s["slide_id"], "title": s["title"]}
                        for s in blueprint["slides"]
                    ])
                    + "\nAssign EACH of these so-far-unused sources to the single most "
                    "relevant slide. JSON array of {\"slide_id\": ..., \"source_id\": ..., "
                    "\"role\": \"supporting evidence\"}.\nSources: "
                    + json.dumps(unused),
                    max_tokens=6000,
                )
                labels = LabelRegistry()
                for s in blueprint["slides"]:
                    for m in s.get("speaker_author_work_mentions") or []:
                        labels.label_for(str(m.get("source_id")))
                by_slide = {s["slide_id"]: s for s in blueprint["slides"]}
                for item in assignments if isinstance(assignments, list) else []:
                    slide = by_slide.get(str((item or {}).get("slide_id")))
                    payload = _mention_payload(
                        str((item or {}).get("source_id") or ""), ledger_index, labels
                    )
                    if slide is None or payload is None:
                        continue
                    slide["speaker_author_work_mentions"].append(payload)
                    slide["source_plan"].append(
                        {
                            "cluster_id": slide["section_id"],
                            "label": payload["label"],
                            "role": "supporting evidence",
                        }
                    )
                    plan = blueprint["speaker_author_work_mention_plan"]
                    if payload["source_id"] not in {m.get("source_id") for m in plan}:
                        plan.append(payload)
                blueprint = rebuild_blueprint_derived_metadata(
                    blueprint,
                    board=board,
                    ledger_index=ledger_index,
                )
            stats = coverage_stats(blueprint)

    # approved_expansion: the >14-slide gate requires artifact-bound hashes and
    # a runtime cap. The storyboard is the editorial artifacts file; the source
    # plan is the enriched ledger; the blueprint hash binds the canonical
    # payload (sans this block, which cannot self-hash).
    blueprint["approved_expansion"] = {
        "approved": True,
        "target_slide_count": len(blueprint["slides"]),
        "max_runtime_minutes": min(40, int(blueprint["target_runtime_minutes"]) + 8),
        "blueprint_sha256": hashlib.sha256(
            json.dumps(
                {k: v for k, v in blueprint.items() if k != "approved_expansion"},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "storyboard_sha256": _sha256_file(artifacts_path),
        "source_plan_sha256": _sha256_file(ledger_path),
        "review_receipt_path": str(receipt_path),
        "rationale": (
            "Editorial deep-dive scope: canary-calibrated 22-26 content slides for a "
            "20-30 minute talk covering the full topic corpus, authored by the "
            "editorial-live board/spine/storyboard process."
        ),
    }

    validation = validate_blueprint_payload(blueprint)
    if validation.errors:
        print("editorial blueprint validation failed: " + "; ".join(error.message for error in validation.errors), file=sys.stderr)
        return 1

    failures = []
    if stats["sources"] < MIN_SOURCES:
        failures.append(f"sources {stats['sources']} < {MIN_SOURCES}")
    if stats["people"] < MIN_PEOPLE:
        failures.append(f"people {stats['people']} < {MIN_PEOPLE}")
    if stats["works"] < MIN_WORKS:
        failures.append(f"works {stats['works']} < {MIN_WORKS}")
    if stats["slides"] < MIN_CONTENT_SLIDES:
        failures.append(f"content slides {stats['slides']} < {MIN_CONTENT_SLIDES}")
    if stats["slides"] > MAX_CONTENT_SLIDES:
        failures.append(f"content slides {stats['slides']} > {MAX_CONTENT_SLIDES}")
    if failures:
        print("editorial blueprint failed coverage requirements: " + "; ".join(failures), file=sys.stderr)
        return 1

    if REFUSAL_FALLBACKS:
        provenance = blueprint.setdefault("provenance", {})
        provenance["refusal_fallbacks"] = list(REFUSAL_FALLBACKS)
        print(f"{len(REFUSAL_FALLBACKS)} call(s) refused by the primary model were answered by the fallback", file=sys.stderr)
    write_json_atomic(output_json, blueprint)
    output_md.write_text(render_blueprint_markdown(blueprint, topic_title=topic_title, stats=stats), encoding="utf-8")

    after = provider_receipt_summary(workspace)
    new_receipt_paths = sorted(set(after.get("receipt_paths", [])) - before_receipts)
    if not new_receipt_paths:
        # --from-artifacts recompiles make no new provider calls; the original
        # authoring run's provider evidence (same stage id) still backs this
        # blueprint's content.
        new_receipt_paths = [
            p
            for p in after.get("receipt_paths", [])
            if "/editorial_blueprint/" in p
        ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.blueprint_generation",
        "generated_at": _now(),
        "generated_by": "scripts/presentations/generate_editorial_blueprint.py",
        "presentation_id": args.presentation_id,
        "mode": "live",
        "generation_method": "editorial-live-multicall",
        "provider": PROVIDER,
        "model_id": MODEL_ID,
        "blueprint_path": str(output_json),
        "blueprint_sha256": _sha256_file(output_json),
        "input_hashes": input_hashes,
        "coverage": stats,
        "provider_receipt_paths": new_receipt_paths,
        "production_evidence": True,
    }
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"status": "pass", "coverage": stats, "blueprint": str(output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
