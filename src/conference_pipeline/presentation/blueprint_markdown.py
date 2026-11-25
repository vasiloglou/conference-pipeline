"""Human review view of a presentation blueprint (Markdown).

The blueprint JSON is the contract; this is what a reviewer reads before
approving it: the story, then every slide in full with its sources. Shared by
the editorial generator and the versioned blueprint repair, so a repair never
replaces the review view with a stub.
"""

from __future__ import annotations

import json
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def render_blueprint_markdown(blueprint: dict[str, Any], *, topic_title: str, stats: dict[str, Any]) -> str:
    """Human review view of a blueprint: the story, then every slide in full.

    The JSON is the contract; this file is what a reviewer reads before
    approving, so it shows each slide's purpose, on-slide text, claims,
    narration, figure plan and every source with what it contributes.
    """

    sections: dict[str, str] = {}
    for item in _as_list(blueprint.get("telescopic_section_map")):
        if isinstance(item, dict):
            sid = str(item.get("section_id") or item.get("id") or "")
            sections[sid] = str(item.get("title") or item.get("section_title") or sid)
    lines = [f"# Blueprint: {blueprint.get('audience_title') or topic_title}", ""]
    lines.append(
        f"{stats.get('slides')} content slides · {stats.get('sources')} sources · "
        f"{stats.get('works')} works · {stats.get('people')} people named"
    )
    lines.append("")
    for label, key in (("Thesis", "deck_thesis"), ("Promise to the audience", "audience_promise"), ("Conclusion", "conclusion_synthesis")):
        value = blueprint.get(key)
        if isinstance(value, dict):
            value = value.get("text") or value.get("summary") or json.dumps(value, ensure_ascii=False)
        if value:
            lines += [f"**{label}.** {value}", ""]

    dropped = [d for d in blueprint.get("dropped_sections") or [] if isinstance(d, dict) and d.get("section_id")]
    if dropped:
        lines.append("**Dropped sections** (planned, but no slides could be authored)")
        lines += [f"- {d.get('title') or d.get('section_id')}: {d.get('reason')}" for d in dropped]
        lines.append("")

    slides = [s for s in blueprint.get("slides") or [] if isinstance(s, dict)]
    lines += ["## Outline", ""]
    current = None
    for number, slide in enumerate(slides, start=1):
        sid = str(slide.get("section_id") or "")
        if sid and sid != current:
            current = sid
            lines.append(f"- **{sections.get(sid) or sid}**")
        sources = len(slide.get("speaker_author_work_mentions") or [])
        lines.append(f"  {number}. {slide.get('title')} ({sources} sources)")
    lines.append("")

    for number, slide in enumerate(slides, start=1):
        lines += [f"## {number}. {slide.get('title')}", ""]
        sid = str(slide.get("section_id") or "")
        if sid:
            lines.append(f"*Section: {sections.get(sid) or sid}* · ~{slide.get('estimated_minutes', '?')} min")
            lines.append("")
        if slide.get("intent"):
            lines += [f"**Section goal.** {slide['intent']}", ""]
        visible = [str(v) for v in _as_list(slide.get("visible_text")) if str(v).strip()]
        if visible:
            lines.append("**On the slide**")
            lines += [f"- {v}" for v in visible]
            lines.append("")
        claims = [str(c) for c in _as_list(slide.get("claims")) if str(c).strip()]
        if claims:
            lines.append("**Claims**")
            lines += [f"- {c}" for c in claims]
            lines.append("")
        beats = [str(b) for b in _as_list(slide.get("narration_beats") or slide.get("narration_beat")) if str(b).strip()]
        if beats:
            lines.append("**Narration**")
            lines += [f"> {b}" for b in beats]
            lines.append("")
        figure = slide.get("visual_figure_brief") or {}
        if isinstance(figure, dict) and (figure.get("summary") or figure.get("components")):
            lines.append(f"**Figure** ({figure.get('depiction') or 'figure'}): {figure.get('summary') or ''}")
            parts = [str(c) for c in _as_list(figure.get("components"))]
            if parts:
                lines.append(f"- Parts: {'; '.join(parts)}")
            links = [str(r) for r in _as_list(figure.get("relationships"))]
            if links:
                lines.append(f"- Links: {'; '.join(links)}")
            numbers = [str(d) for d in _as_list(figure.get("data_points"))]
            if numbers:
                lines.append(f"- Numbers: {'; '.join(numbers)}")
            lines.append("")
        plan = {str(p.get("label")): p for p in _as_list(slide.get("source_plan")) if isinstance(p, dict)}
        roles = {label: str(p.get("role") or "") for label, p in plan.items()}
        supports = {label: str(p.get("supports") or "").strip() for label, p in plan.items()}
        mentions = [m for m in slide.get("speaker_author_work_mentions") or [] if isinstance(m, dict)]
        if mentions:
            lines.append("**Sources**")
            for m in mentions:
                people = [str(p) for p in (m.get("speakers") or []) + (m.get("authors") or []) if str(p).strip()]
                kind = "talk" if str(m.get("source_id") or "").startswith("speech_transcript") else "paper"
                who = ", ".join(people[:6]) if people else "no named people"
                role = roles.get(str(m.get("label")), "")
                lines.append(f"- {m.get('label')} ({kind}) *{m.get('title')}* — {who}" + (f" — role: {role}" if role else ""))
                # What the source does on THIS slide; the ledger's topic-level
                # note is only a fallback (it repeats across sources).
                note = supports.get(str(m.get("label")), "") or str(m.get("contribution_note") or "").strip()
                if note:
                    lines.append(f"  - {note}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def blueprint_stats(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Coverage counts recomputed from the slides (title slide excluded)."""

    slides = [s for s in blueprint.get("slides") or [] if isinstance(s, dict)]
    mentions = [m for s in slides for m in s.get("speaker_author_work_mentions") or [] if isinstance(m, dict)]
    people = {
        str(p) for m in mentions for p in (m.get("speakers") or []) + (m.get("authors") or []) if str(p).strip()
    }
    return {
        "slides": max(len(slides) - 1, 0),
        "sources": len({str(m.get("source_id")) for m in mentions if m.get("source_id")}),
        "works": len({str(m.get("title")) for m in mentions if m.get("title")}),
        "people": len(people),
    }
