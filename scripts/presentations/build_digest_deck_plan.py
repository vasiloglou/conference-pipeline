#!/usr/bin/env python3
"""Adapt Opus topic digests into a deck-first plan the executive generator can render.

Reads ``decks/executive_topic_digests.json`` and writes ``decks/<out>`` (a deck-first
plan) so the existing, fully-gated ``generate_executive_overview_decks.py`` renders the
Opus-derived content. This is the FIRST content-proof cut: one evidence slide per topic
(named papers land on each slide). The full bespoke structure (central-message slide,
12-topic index, 2 slides/topic with a big-picture GenAI slide, open-questions/predictions)
is a follow-on that extends the generator's slide-sequence assembly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402


def load_topic_count(workspace: Path, digests: dict) -> int:
    """Topic count from analysis_manifest.json, falling back to the digest topics."""
    manifest_path = workspace / "analysis" / "analysis_manifest.json"
    if manifest_path.is_file():
        try:
            topics = json.loads(manifest_path.read_text(encoding="utf-8")).get("topics")
        except (ValueError, OSError):
            topics = None
        if isinstance(topics, list) and topics:
            return len(topics)
    return len(digests.get("topics", []))


_FRAGMENT_END_WORDS = frozenset(
    {"a", "an", "and", "are", "as", "at", "because", "but", "by", "for", "from",
     "in", "instead", "into", "is", "nor", "not", "of", "on", "onto", "or",
     "over", "rather", "than", "that", "the", "their", "these", "this", "those",
     "to", "toward", "towards", "versus", "vs", "while", "whereas", "with",
     "within", "without"}
)


def _strip_dangling(text: str) -> str:
    words = text.rstrip(".,;:!?—– ").split()
    while words and (
        words[-1].strip(".,;:!?—–").lower() in _FRAGMENT_END_WORDS
        or words[-1].strip(".,;:!?") in {"—", "–", "-"}
    ):
        words.pop()
    cleaned = " ".join(words).rstrip(".,;:!?—– ")
    return f"{cleaned}." if cleaned else ""


def first_sentences(text: str, *, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    first = parts[0]
    if len(first) > max_chars:
        # A single long first sentence was word-cut and re-punctuated, which
        # changes its meaning ("...reaches state-of-the-art manipulation with
        # about 20x less." -- the claim was "20x less robot data"; Opus 5.5
        # digests, CVPR 2026). Keep it whole when it is merely long, else end
        # it at the last clause boundary that fits; never mid-phrase.
        if len(first) <= int(max_chars * 1.6):
            return first
        clause = max(first.rfind(sep, 0, max_chars) for sep in (", and ", "; ", " — ", ", while ", ", so "))
        if clause > max_chars // 3:
            return first[:clause].rstrip(" ,;—") + "."
    out = ""
    for part in parts:
        candidate = (out + " " + part).strip() if out else part
        if len(candidate) > max_chars and out:
            break
        out = candidate
        if len(out) >= max_chars:
            break
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0]
    if out.rstrip(".,;:!? ").split() and out.rstrip(".,;:!? ").split()[-1].strip(".,;:!?").lower() in _FRAGMENT_END_WORDS:
        return _strip_dangling(out)
    if out.endswith(("!", "?")):
        return out
    return _strip_dangling(out)


def short_title(title: str) -> str:
    head = str(title or "").split(":", 1)[0].strip()
    words = head.split()
    return " ".join(words[:6]) if len(words) > 6 else head


def paper_signal(named_sources: list[dict]) -> str:
    titles = [short_title(s.get("title", "")) for s in (named_sources or []) if s.get("title")]
    titles = [t for t in titles if t][:3]
    if not titles:
        return ""
    if len(titles) == 1:
        body = titles[0]
    elif len(titles) == 2:
        body = f"{titles[0]} and {titles[1]}"
    else:
        body = f"{titles[0]}, {titles[1]}, and {titles[2]}"
    return f"Anchored in {body}."


def _evidence_lesson(t: dict) -> dict:
    """Short-deck slide: one evidence -> decision bridge per topic."""
    tid = t["topic_id"]
    ev = t.get("short", {}).get("evidence_slide", {})
    bp = t.get("short", {}).get("bigpicture_slide", {})
    return {
        "lesson_id": f"{tid}_evidence",
        "slide_title": ev.get("headline") or t.get("title"),
        "distilled_message": ev.get("headline") or t.get("central_takeaway", ""),
        "topic_ids": [tid],
        "evidence_kind": "tight_paraphrase",
        "source_anchor": t.get("title", tid),
        "source_locator": f"analysis/exhaustive_summaries/{tid}__transcripts.md#summary",
        "evidence_text": first_sentences(ev.get("synthesis", ""), max_chars=150),
        # The big-picture headline is the punchy, action-shaped "so what"; fall
        # back to the first clause of the longer implication.
        "decision_implication": bp.get("headline") or first_sentences(bp.get("implication", ""), max_chars=90),
        "paper_evidence_text": paper_signal(ev.get("named_sources", [])),
        "paper_source_locator": f"analysis/exhaustive_summaries/{tid}__papers.md#paper-signal",
        "visual_role": ev.get("diagram", {}).get("kind", "comparison"),
        # Render the mechanism as a clean dark line-schematic diagram.
        "diagram": ev.get("diagram", {}),
        "figure_variant": "tech_diagram",
    }


def _bigpicture_lesson(t: dict) -> dict:
    """Short-deck 2nd slide per topic: the big-picture/implication slide whose
    figure is swapped for a GenAI hero image (executive_lesson_id '{tid}_bigpicture')."""
    tid = t["topic_id"]
    bp = t.get("short", {}).get("bigpicture_slide", {})
    return {
        "lesson_id": f"{tid}_bigpicture",
        "slide_title": bp.get("headline") or t.get("title"),
        "distilled_message": bp.get("headline") or t.get("title"),
        "topic_ids": [tid],
        "evidence_kind": "tight_paraphrase",
        "source_anchor": t.get("title", tid),
        "source_locator": f"analysis/exhaustive_summaries/{tid}__transcripts.md#summary",
        "evidence_text": first_sentences(bp.get("implication", ""), max_chars=150),
        "decision_implication": bp.get("headline") or "Act on this evidence.",
        "paper_evidence_text": "",
        "paper_source_locator": "",
        "visual_role": "bigpicture",
        # The 2nd slide is a distinct decision/strategy diagram (Opus-generated).
        "diagram": bp.get("diagram", {}),
        "figure_variant": "tech_diagram",
    }


def _long_lessons(t: dict) -> list[dict]:
    """Long-deck slides: three deeper sub-theme slides per topic (different zoom),
    sharing the topic's leadership decision as the through-line."""
    tid = t["topic_id"]
    decision = t.get("short", {}).get("bigpicture_slide", {}).get("headline") or "Act on this evidence."
    lessons = []
    for idx, sl in enumerate(t.get("long", {}).get("slides", [])[:3], start=1):
        lessons.append({
            "lesson_id": f"{tid}_long_{idx}",
            "slide_title": sl.get("headline") or t.get("title"),
            "distilled_message": sl.get("headline") or t.get("title"),
            "topic_ids": [tid],
            "evidence_kind": "tight_paraphrase",
            "source_anchor": t.get("title", tid),
            "source_locator": f"analysis/exhaustive_summaries/{tid}__transcripts.md#summary",
            "evidence_text": first_sentences(sl.get("body", ""), max_chars=150),
            # The slide's own takeaway (add_bigpicture_diagrams.py); the topic
            # decision under every sub-theme figure read as unrelated to two of them.
            "decision_implication": sl.get("takeaway") or decision,
            "paper_evidence_text": paper_signal(sl.get("named_sources", [])),
            "paper_source_locator": f"analysis/exhaustive_summaries/{tid}__papers.md#paper-signal",
            "visual_role": sl.get("diagram", {}).get("kind", "comparison"),
            "diagram": sl.get("diagram", {}),
            "figure_variant": "tech_diagram",
        })
    return lessons


def _synthesis_lessons(digests: dict, *, conference_label: str, topic_count: int) -> list[dict]:
    """Closing open-questions + predictions slides (bullet lists), shared by both decks."""
    topic_ids = [t["topic_id"] for t in digests.get("topics", [])]
    open_q = [q for q in digests.get("open_questions", []) if q.get("question")]
    preds = [p for p in digests.get("predictions", []) if p.get("prediction")]
    cross_topic_anchor = f"Cross-topic synthesis across the {topic_count} {conference_label} topics"
    lessons: list[dict] = []
    if open_q:
        lessons.append({
            "lesson_id": "synthesis_open_questions",
            "slide_title": f"Open questions {conference_label} surfaced",
            "distilled_message": f"Open questions {conference_label} surfaced",
            "topic_ids": topic_ids[:3],
            "evidence_kind": "tight_paraphrase",
            "source_anchor": cross_topic_anchor,
            "source_locator": "analysis/executive_summary.md#open-questions",
            "evidence_text": f"The unresolved questions {conference_label} leaves on the table for leadership.",
            "decision_implication": "Track these as the open frontier.",
            "slide_bullets": [first_sentences(q["question"], max_chars=150) for q in open_q[:3]],
            "figure_variant": "synthesis_list",
        })
    if preds:
        lessons.append({
            "lesson_id": "synthesis_predictions",
            "slide_title": "What we predict comes next",
            "distilled_message": "What we predict comes next",
            "topic_ids": topic_ids[3:6],
            "evidence_kind": "tight_paraphrase",
            "source_anchor": cross_topic_anchor,
            "source_locator": "analysis/trend_report.md#predictions",
            "evidence_text": "Near-term predictions grounded in the conference evidence.",
            "decision_implication": "Plan against these trajectories.",
            "slide_bullets": [first_sentences(p["prediction"], max_chars=150) for p in preds[:3]],
            "figure_variant": "synthesis_list",
        })
    return lessons


# The opening index slide groups the topics into narrative arcs so it reads as a
# guided itinerary, not a flat directory (2026-06-22 Claude/Codex/Gemini panel).
#
# Those arcs are produced per-conference by the cross-topic Opus call in
# generate_executive_topic_digests.py and validated there against this
# conference's own topic ids. This module used to carry a hardcoded table of
# ICLR 2026 slugs instead; every other conference matched none of them, every
# row fell through with arc="", and the renderer silently swapped its
# two-column itinerary for the 5-item topic-deck bookend -- which then had to
# hold 12 topics and 26 links. Read the arcs, never re-derive them.


def _arc_topic_index(digests: dict) -> list[dict]:
    """Topic index reordered into narrative arcs, each row tagged with its `arc`.

    Refuses rather than emitting a partially-arced index: the renderer treats
    "every row has an arc" as the signal to use the itinerary layout, so a
    partial grouping is silently mis-rendered instead of reported.
    """

    index_rows = [r for r in digests.get("topic_index", []) if r.get("topic_id")]
    arcs = digests.get("arcs")
    if not isinstance(arcs, list) or not arcs:
        raise ValueError(
            "executive_topic_digests.json has no 'arcs'; regenerate the digests with "
            "generate_executive_topic_digests.py (the arcs come from its cross-topic "
            "synthesis call, validated against this conference's topic ids)"
        )

    rows = {r["topic_id"]: r for r in index_rows}
    ordered: list[dict] = []
    seen: set[str] = set()
    for entry in arcs:
        arc_name = str((entry or {}).get("name") or "").strip()
        topic_ids = (entry or {}).get("topic_ids") or []
        if not arc_name:
            raise ValueError(f"arc without a name in executive_topic_digests.json: {entry!r}")
        for tid in topic_ids:
            row = rows.get(str(tid))
            if row is None:
                raise ValueError(
                    f"arc {arc_name!r} cites topic_id {tid!r}, which is not in this "
                    "conference's topic_index"
                )
            ordered.append({**row, "arc": arc_name})
            seen.add(str(tid))

    unassigned = [r["topic_id"] for r in index_rows if r["topic_id"] not in seen]
    if unassigned:
        raise ValueError(
            "these topics are in no arc, so the executive index slide would silently "
            f"fall back to the crowded bookend layout: {unassigned}"
        )
    return ordered


def build_plan(
    digests: dict,
    *,
    short_minutes: float,
    with_bigpicture: bool = False,
    conference_label: str = "the conference",
    topic_count: int = 0,
) -> dict:
    topics = digests.get("topics", [])
    topic_count = topic_count or len(topics)
    synthesis = _synthesis_lessons(
        digests, conference_label=conference_label, topic_count=topic_count
    )
    short_spine: list[dict] = []
    for t in topics:
        short_spine.append(_evidence_lesson(t))
        if with_bigpicture:
            short_spine.append(_bigpicture_lesson(t))
    short_spine = short_spine + synthesis
    # Scale the short-deck duration to its content count (~50s/content slide +
    # ~180s of bookends) but never below the requested floor.
    short_minutes = max(short_minutes, (180 + len(short_spine) * 50) // 60 + 1)
    long_spine: list[dict] = []
    for t in topics:
        long_spine.extend(_long_lessons(t))
    long_spine = long_spine + synthesis
    # Scale the long-deck duration to its slide count (~58s/content slide + ~180s
    # of bookends) so the 3-slides-per-topic deck is comfortably paced, not crammed.
    long_minutes = max(30, (180 + len(long_spine) * 58) // 60 + 1)
    return {
        "schema_version": 1,
        "source": "executive_topic_digests.json (Opus claude-opus-4-8)",
        "selection_policy": {
            "short_duration_minutes": short_minutes,
            "long_duration_minutes": long_minutes,
            "short_spine_count": len(short_spine),
            "long_spine_count": len(long_spine),
            "long_expansion_count": 0,
            "evidence_policy": "Opus topic-narration digests; named sources from each topic's blueprint/narration.",
            "image_policy": "SVG-first content-proof cut; GenAI per-topic images deferred to the gated image phase.",
        },
        "candidate_pool": short_spine,
        "short_spine": short_spine,
        "long_spine": long_spine,
        "long_expansions": [],
        # Source-grounded 12-topic index + cross-topic synthesis for the bookend
        # and (future) central-message / open-questions / predictions slides.
        "topic_index": _arc_topic_index(digests),
        "central_message": digests.get("central_message", {}),
        "open_questions": digests.get("open_questions", []),
        "predictions": digests.get("predictions", []),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", type=Path, required=True,
                   help="Conference workspace dir (e.g. conferences/<slug>).")
    p.add_argument("--digests", type=Path, default=Path("decks/executive_topic_digests.json"))
    p.add_argument("--out", type=Path, default=Path("decks/executive_summary_digest_plan.json"))
    p.add_argument("--short-minutes", type=float, default=16.0)
    p.add_argument("--with-bigpicture", action="store_true",
                   help="add the 2nd slide per topic (big-picture/GenAI hero); pair with generated images")
    args = p.parse_args()
    ws = args.workspace.expanduser().resolve()
    digests = json.loads((ws / args.digests).read_text())
    conference_label = conference_name(ws, default="the conference")
    topic_count = load_topic_count(ws, digests)
    plan = build_plan(
        digests,
        short_minutes=args.short_minutes,
        with_bigpicture=args.with_bigpicture,
        conference_label=conference_label,
        topic_count=topic_count,
    )
    out = ws / args.out
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out} — {len(plan['short_spine'])} short / {len(plan['long_spine'])} long lessons, "
          f"short {plan['selection_policy']['short_duration_minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
