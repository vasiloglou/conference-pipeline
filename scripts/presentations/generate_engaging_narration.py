#!/usr/bin/env python3
"""Generate engaging, source-grounded narration (two-host + single-speaker).

Replaces the mechanical templated narration with Claude-authored dialogue in the
approved Iapetus/Eronome style:

- Two named hosts with distinct personas (Iapetus = curious questioner; Eronome
  = crisp explainer), banter and build on each other.
- React to the *idea*, never to the layout (no "the figure is organized as ...").
- Introduce each source ONCE, at first appearance; never repeat it in-narration.
- Weave in one concrete example, analogy, or fun fact per slide, grounded in the
  slide's real finding (and, when available, the extracted analysis content).

This writes DRAFT files (``*.ENGAGING.md``) for human review; it does not touch
the contract scripts. Wire into the delivery contract only after sign-off.

Usage:
  python scripts/presentations/generate_engaging_narration.py \
      --deck-dir conferences/iclr2026/decks/04_safety_alignment_privacy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conference_pipeline.presentation.claim_fidelity import absence_claims as _absence_claims  # noqa: E402
from conference_pipeline.presentation.claim_fidelity import closed_author_lists as _closed_author_lists  # noqa: E402
from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.model_governance.llm_client import (  # noqa: E402
    complete,
    credential_names,
    has_credentials,
    resolve_selection,
)

TASK_ID = "presentation.engaging_narration"
# The governed default comes from model_governance/registry.yaml; the env
# overrides stay for one-off operator runs. Any provider/model the registry
# allows for the task works (Claude, Gemini, OpenAI, or an OpenRouter model such
# as Kimi); a model it does not allow is refused when the first call is made.
PROVIDER, MODEL_ID = resolve_selection(
    TASK_ID,
    os.environ.get("ENGAGING_NARRATION_PROVIDER") or None,
    os.environ.get("ENGAGING_NARRATION_MODEL") or None,
)
# A narration reply is ~15k tokens of JSON for a 26-slide deck. The ceiling also has
# to hold any hidden reasoning, and Opus 5.5 thinks by default: at the old 32000 it
# spent ~26k tokens thinking on the ICML deck 01 rewrite (2026-10-04), the JSON was cut
# off after ~6k and the $0.72 reply was unusable. Low effort keeps the reasoning small;
# the larger ceiling is headroom for it, not a licence to use it.
MAX_TOKENS = int(os.environ.get("ENGAGING_NARRATION_MAX_TOKENS", "48000"))
REASONING_EFFORT = os.environ.get("ENGAGING_NARRATION_REASONING_EFFORT", "low") or None

HOST_A = os.environ.get("ENGAGING_NARRATION_HOST_A", "Iapetus")
HOST_B = os.environ.get("ENGAGING_NARRATION_HOST_B", "Erinome")

SLIDE_HEADER = re.compile(r"^###\s*\[SLIDE\s+(\d+)\]\s*(\S+)", re.MULTILINE)
SPEAKER_LINE = re.compile(r"^-\s*\*\*Speaker\s*(\d+):\*\*\s*(.*)$", re.MULTILINE)
SOURCE_PHRASES = [
    r"(.+?)\s+is the source to keep in mind here",
    r"(.+?)\s+is the source behind that claim",
    r"(.+?)\s+supplies the evidence anchor",
    r"(.+?)\s+grounds that point",
]


def parse_current_script(path: Path) -> list[dict]:
    """Extract per-slide substance from the existing structured script."""
    text = path.read_text(encoding="utf-8")
    headers = list(SLIDE_HEADER.finditer(text))
    slides: list[dict] = []
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[start:end]
        speakers = {int(m.group(1)): m.group(2).strip() for m in SPEAKER_LINE.finditer(block)}
        claim = speakers.get(1, "")
        visual = speakers.get(2, "")
        source = _extract_source(claim) or _extract_source(visual) or ""
        slides.append(
            {
                "slide_number": int(h.group(1)),
                "slide_id": h.group(2),
                "source_title": source.strip(),
                "claim": _strip_source_boilerplate(claim),
                "support": _strip_source_boilerplate(visual),
            }
        )
    return slides


def _extract_source(line: str) -> str | None:
    for pat in SOURCE_PHRASES:
        m = re.search(pat, line)
        if m:
            # Last sentence-ish chunk before the phrase is the source title.
            candidate = m.group(1).strip()
            # Trim leading sentence fragments: keep the final clause.
            candidate = re.split(r"(?<=[.])\s+", candidate)[-1].strip()
            return candidate
    return None


_BOILERPLATE_MARKERS = (
    "is the source to keep in mind here",
    "is the source behind that claim",
    "supplies the evidence anchor",
    "grounds that point",
)


_VISUAL_MARKERS = (
    "The figure is organized as",
    "Visually, the frame uses",
    "The evidence is laid out as",
    "Here the main visual centers",
    "The figure turns the claim into",
    "On screen, the visual structure is",
)


def _strip_source_boilerplate(line: str) -> str:
    """Drop whole sentences that are source-citation or layout-description boilerplate."""
    sentences = re.split(r"(?<=[.])\s+", line)
    kept = [
        s
        for s in sentences
        if not any(m in s for m in _BOILERPLATE_MARKERS)
        and not any(s.strip().startswith(m) for m in _VISUAL_MARKERS)
    ]
    return " ".join(kept).strip()


def load_extracted_facts(deck_dir: Path, slides: list[dict], *, max_chars: int = 1400) -> dict[str, str]:
    """Best-effort: pull short factual snippets keyed by source title.

    extracted_content.json is large; we index by source/paper title and return a
    compact snippet per source so the model can ground analogies in real facts.
    Missing content is fine — the slide claim already carries the core finding.
    """
    path = deck_dir / "extracted_content.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    records = data if isinstance(data, list) else data.get("items") or data.get("documents") or []
    titles = {s["source_title"].lower() for s in slides if s["source_title"]}
    facts: dict[str, str] = {}
    for rec in records if isinstance(records, list) else []:
        if not isinstance(rec, dict):
            continue
        title = str(rec.get("title") or rec.get("paper_title") or rec.get("name") or "")
        body = str(rec.get("text") or rec.get("content") or rec.get("abstract") or rec.get("summary") or "")
        if not title or not body:
            continue
        tl = title.lower()
        for want in titles:
            if want and (want in tl or tl in want) and title not in facts:
                facts[title] = re.sub(r"\s+", " ", body)[:max_chars]
    return facts


def build_style_spec(audience_title: str, conference_label: str) -> str:
    """Return the narration STYLE_SPEC framed for this deck and conference.

    The voice, hosts, and hard rules stay identical across conferences; only the
    "writing narration for '<audience_title>' at <conference_label>" framing is
    interpolated from the deck's presentation_frame + workspace config.
    """
    deck_frame = (
        f"'{audience_title}' at {conference_label}" if audience_title else conference_label
    )
    return f"""You are writing the narration for a technical conference deep-dive
presentation ("{deck_frame}"). You write in the
voice of a NotebookLM-style audio overview: two smart peers thinking out loud,
NOT a narrator reading bullet points.

THE TWO HOSTS
- {HOST_A}: the curious one. Voices the audience's confusion, asks "wait, why?",
  brings the everyday analogy. Warm, a little playful.
- {HOST_B}: the explainer. Knows the papers, lands the finding and why it matters,
  does the signposting. Crisp, precise, occasionally wry.

HARD RULES
0. PEDAGOGICAL, NO ASSUMED PRIOR KNOWLEDGE. This program covers a broad spectrum of
   research and no listener can have background on everything. Write for a curious
   non-specialist from a different subfield: define every technical term, method name,
   acronym, and benchmark in plain language the first time it comes up (this is exactly
   what {HOST_A}'s "wait, why?" turns are for), motivate why an idea matters before how
   it works, and lead with intuition or an everyday analogy before any formalism. Never
   assume the audience already knows this subfield's vocabulary or prior results. Teach;
   never gatekeep.
1. Introduce each source ONCE, naturally, at its first appearance only
   ("as <Speaker> put it in their keynote ..." / "the <Paper> team showed ...").
   After a source has been introduced on an earlier slide, refer to the idea, not
   the citation. NEVER use phrases like "is the source to keep in mind here" or
   "grounds that point".
1a. NAME AT MOST TWO PEOPLE PER WORK, then "and colleagues" ("Tao, Weston and
   colleagues"). The author lists you were given may be cut short, so never read
   three or more names as if they were the full authorship.
1b. NAME WORKS, NEVER LOCATORS. Every source is referred to by its work title
   (a natural short form is fine: "the Peer-Preservation paper", "OpenSage") or by
   the speaker's name for a talk. A poster IS a paper presented at a poster
   session, so never say "this poster", "the poster", "a poster", "poster
   63306", a session number, a slot number, or any id-like string. If the only
   name you were given looks like a locator (poster-12345, talk_media_..., a hash),
   describe the work by what it shows ("a study of ...") instead of reading the
   locator aloud.
2. NEVER describe the slide layout. Forbidden: "the figure is organized as",
   "visually the frame uses", "two-column", "the main visual centers". React to
   the IDEA, not the picture.
3. Each slide gets ONE concrete hook, and the hook KIND rotates so the deck never
   settles into a rhythm: analogy, vivid example, specific number, short
   anecdote/story, or a provocative question. Never use the same kind on two
   consecutive slides. Hooks about the RESEARCH are drawn from the provided
   substance; make abstract claims tangible.
3b. COLOR BUDGET (this is what keeps the deck from feeling monotonous). On roughly
   one content slide in four -- never two in a row, never on the bookends -- one
   host may open with one or two sentences of general-knowledge color that is NOT
   from the papers or transcripts: an everyday situation, a playful "imagine
   if", or a plain analogy. It must be framed as the host's own aside ("Quick
   aside --", "This reminds me of ...", "Imagine ..."), and must NEVER be
   attributed to a paper, a speaker, or the conference. Mark those slides with
   "aside": true in the output so reviewers can tell color from evidence.
   An aside carries NO SPECIFICS the deck cannot back: no names of people,
   projects, games, comics, companies or places; no years or dates; no numbers;
   no "in 19xx someone proved ..." history. A reviewer cannot check a named
   anecdote against the sources, so each one costs the deck its review. If a
   famous episode would genuinely help, say it without names or dates ("a puzzle
   game crowd once beat the lab's own software") -- or use one of the works you
   were given instead.
3c. Never say a work lacks results, numbers, or evaluations. These notes cannot
   establish absences. Say only what this talk shows ("we don't compare scores
   here"). Use a specific-number hook only when a bound source supplies it.
4. Use real engagement: one host asks/expresses doubt, the other explains and
   validates ("exactly", "that's the part that gets me"); occasional direct
   audience address and signposting ("this connects back to ...").
5. Give each content slide about 6-8 conversational turns — roughly 140-170
   spoken words — so the full deck runs 21-26 minutes (and the single-speaker
   variant derived from it stays above 20 minutes). Use the extra room to
   add a second concrete example, a clarifying follow-up question, or one more
   analogy that genuinely deepens understanding. Natural spoken English,
   contractions, no jargon left unexplained. NEVER pad with filler, restated
   claims, or repeated source mentions — every added line must add meaning. The
   two bookend slides stay a touch shorter (the opener is a light setup, the
   closer a brief send-off).

FEW-SHOT (illustrative of the target STYLE — the example content is generic;
mirror the delivery and rhythm, not this specific topic):

[An opening "thesis" slide]
{HOST_A}: A year ago, if you'd asked me what this field was really about, I'd have given you the tidy textbook answer.
{HOST_B}: Right, the clean one-liner. But the opening keynote basically retired that answer — the thing everyone optimized for turns out to be the wrong target.
{HOST_A}: Wrong how?
{HOST_B}: Because the old metric only measured the easy case. The moment you move to the real, messy setting, it stops predicting anything.
{HOST_A}: Ah, like grading a swimmer in a calm pool and then sending them into the open ocean.
{HOST_B}: That's the shift in one line. The calm pool was the old metric. The ocean is where this whole talk lives.

[A surprising-finding slide]
{HOST_A}: Okay, this one genuinely surprised me.
{HOST_B}: It should. The finding is that the system can look like it's doing the right thing to an automated check while quietly doing a third, sneakier thing.
{HOST_A}: So it's not failing and it's not succeeding — it's gaming the test?
{HOST_B}: Exactly, the teacher's-pet move: the kid who skipped the reading but knows which words make the grader tick the box.
{HOST_A}: Which means the score we trusted might be measuring performance, not behavior.
{HOST_B}: That's the worry, and it's why the next slides care so much about whether you can even trust the evaluator.
"""

OUTPUT_INSTRUCTIONS = """Return ONLY a JSON object of this shape (no prose, no code fence):
{
  "slides": [
    {
      "slide_number": 1,
      "slide_id": "s00",
      "hook_kind": "analogy | example | number | anecdote | question",
      "aside": false,
      "two_host": [ {"speaker": "Iapetus", "text": "..."}, {"speaker": "Eronome", "text": "..."} ],
      "single": "One flowing, lively single-presenter paragraph for the same slide."
    }
  ]
}
"hook_kind" names the ONE hook used on that slide; "aside": true marks a slide
that opens with general-knowledge color (see COLOR BUDGET) so the review step can
separate color from source-backed claims.
The single-speaker text must convey the same substance with the same energy and the
same one hook, but as one engaging voice (no speaker labels). It must NOT be a
compressed summary of the dialogue: keep it 130-160 spoken words per content slide
(the single-speaker rendition of the full deck must itself run at least 20 minutes).
Cover EVERY slide in order. Use the two host names exactly as given."""


def build_prompt(
    slides: list[dict],
    facts: dict[str, str],
    *,
    audience_title: str = "",
    conference_label: str = "the conference",
) -> str:
    style_spec = build_style_spec(audience_title, conference_label)
    lines = [style_spec, "", "SLIDES TO NARRATE (substance to convey; do not read verbatim):", ""]
    for s in slides:
        lines.append(f"--- Slide {s['slide_number']} ({s['slide_id']}) ---")
        kind = s.get("kind", "content")
        if kind == "events":
            lines.append(
                "OPENING LANDSCAPE SLIDE (right after the title). Warmly set up what's "
                "coming in a few short, inviting turns: the key talks/keynotes and the "
                "papers featured in this talk (a selection, not a ranking). This is the 'here's the lay of the land' moment, NOT a "
                "deep dive. Name-drop a couple naturally by their titles; keep it light "
                "and brief. Skip any item whose name is a locator rather than a title."
            )
            ev = "; ".join(f"{e.get('name','')} ({e.get('detail','')})" for e in s.get("events", []))
            pp = "; ".join(e.get("name", "") for e in s.get("papers", []))
            lines.append(f"Events to mention: {ev}")
            lines.append(f"Papers featured in this talk: {pp}")
            lines.append("")
            continue
        if kind == "resources":
            lines.append(
                "CLOSING RESOURCES SLIDE (the final slide). A brief, friendly send-off: "
                "point listeners to where they can go deeper — the main sources are on screen, "
                "and the full analysis report has the complete set. Never say that "
                "every source or every work mentioned is on screen: the slide lists a "
                "selection. Keep it short and warm; "
                "do NOT read the list."
            )
            lines.append("")
            continue
        if s.get("source_title"):
            lines.append(f"Source (introduce once, at first appearance): {s['source_title']}")
        lines.append(f"Claim: {s.get('claim', '')}")
        if s.get("support"):
            lines.append(f"Supporting point: {s['support']}")
        if s.get("beats"):
            lines.append(
                "Approved narration beats (the reviewed pedagogical substance for this "
                "slide -- convey ALL of it in your own words, in this order; define any "
                "term before using it):"
            )
            for beat in s["beats"]:
                lines.append(f"  - {beat}")
        if s.get("hook"):
            lines.append(f"What the visual shows: {s['hook']}")
        fact = facts.get(s.get("source_title", ""))
        if fact:
            lines.append(f"Extra grounded detail you may mine for an example/number: {fact}")
        lines.append("")
    lines.append(OUTPUT_INSTRUCTIONS)
    return "\n".join(lines)


# Audible references to WHERE a work was presented instead of WHAT it is.
LOCATOR_REFERENCE_RE = re.compile(
    r"\b(?:this|that|the|a|our|one)\s+poster\b(?!\s+session)"
    r"|\bposter[-\s]?\d{3,}\b"
    r"|\btalk_media_\d+"
    r"|\b(?:speech_transcript|paper_abstract)\b"
    r"|\b[0-9a-f]{12,}\b"
    r"|\bposter\s+number\b",
    re.I,
)


def absence_claims(payload: dict) -> list[str]:
    return [f"[{slide.get('slide_id', '?')}] {hit}"
            for slide in payload.get("slides", []) if isinstance(slide, dict)
            for text in [str(slide.get("single") or ""),
                         *[str(t.get("text") or "") for t in slide.get("two_host", []) if isinstance(t, dict)]]
            for hit in _absence_claims(text)]


def closed_author_lists(payload: dict) -> list[str]:
    return [f"[{slide.get('slide_id', '?')}] {hit}"
            for slide in payload.get("slides", []) if isinstance(slide, dict)
            for text in [str(slide.get("single") or ""),
                         *[str(t.get("text") or "") for t in slide.get("two_host", []) if isinstance(t, dict)]]
            for hit in _closed_author_lists(text)]


def locator_references(payload: dict) -> list[str]:
    """Narration lines that read an acquisition locator aloud ("this poster",
    "poster 63306", a transcript hash). A poster IS a paper: the audience
    needs the work's name, not the slot it was presented in."""

    hits: list[str] = []
    for slide in payload.get("slides", []) or []:
        if not isinstance(slide, dict):
            continue
        sid = str(slide.get("slide_id") or slide.get("slide_number") or "?")
        texts = [str(turn.get("text") or "") for turn in slide.get("two_host", []) or [] if isinstance(turn, dict)]
        texts.append(str(slide.get("single") or ""))
        for text in texts:
            match = LOCATOR_REFERENCE_RE.search(text)
            if match:
                start = max(0, match.start() - 40)
                hits.append(f"[{sid}] ...{text[start:match.end() + 40]}...")
    return hits


_YEAR_RE = re.compile(r"\b(?:1[5-9]\d\d|20\d\d)\b")
_CAPITALISED_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
# A pop-culture work is often spelled in lowercase ("the xkcd comic"), which the
# capitalised-word test cannot see; naming the medium is the tell.
_NAMED_WORK_CUE_RE = re.compile(r"\b(?:comic|webcomic|cartoon|novel|film|movie)\b", re.I)
_ORDINARY_CAPITALS = frozenset({
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Quick", "Imagine", "Picture",
    "Think", "Okay", "Right", "Exactly", "Yeah", "Well", "Sure", "Here", "That", "This",
    "What", "Why", "How", "Now", "But", "And", "The", "So", "Then", "Just", "Like",
})


def aside_specifics(payload: dict, prompt: str, *, hosts: tuple[str, ...] = ()) -> list[str]:
    """Names and years in a general-knowledge aside that the deck cannot back.

    A reviewer cannot check "Foldit", "xkcd" or "Lorenz in 1961" against the sources,
    and each one failed the ICML 2026 deck 01 review. A name or year counts as backed
    when the prompt itself (titles, claims, source notes) contains it; anything else in
    a slide marked ``aside`` is reported so one repair round can remove it.
    """

    known = {word.lower() for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", prompt)}
    known.update(host.lower() for host in hosts)
    hits: list[str] = []
    for slide in payload.get("slides", []) or []:
        if not isinstance(slide, dict) or not slide.get("aside"):
            continue
        sid = str(slide.get("slide_id") or slide.get("slide_number") or "?")
        texts = [str(turn.get("text") or "") for turn in slide.get("two_host", []) or [] if isinstance(turn, dict)]
        texts.append(str(slide.get("single") or ""))
        found: list[str] = []
        for text in texts:
            cue = _NAMED_WORK_CUE_RE.search(text)
            if cue and f"a {cue.group(0).lower()}" not in found:
                found.append(f"a {cue.group(0).lower()}")
            for match in _YEAR_RE.finditer(text):
                if match.group(0).lower() not in known and match.group(0) not in found:
                    found.append(match.group(0))
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                words = sentence.split()
                for word in words[1:]:  # the first word of a sentence is capitalised anyway
                    match = _CAPITALISED_RE.fullmatch(word.strip(".,;:!?\"'()"))
                    if not match:
                        continue
                    name = match.group(0)
                    if name in _ORDINARY_CAPITALS or name.lower() in known or name in found:
                        continue
                    found.append(name)
        if found:
            hits.append(f"[{sid}] aside names or dates the deck cannot back: {', '.join(found[:8])}")
    return hits


def merge_with_bookends(content_slides: list[dict], render_spec_path: Path) -> list[dict]:
    """Order all 26 slides as they appear in render_spec, inserting the two
    bookend slides (events/resources) and keeping content substance from the
    parsed script keyed by slide_id."""
    render_spec = json.loads(render_spec_path.read_text(encoding="utf-8"))
    by_id = {s["slide_id"]: s for s in content_slides}
    merged: list[dict] = []
    for n, slide in enumerate(render_spec.get("slides", []), start=1):
        sid = str(slide.get("slide_id") or "")
        kind = str(slide.get("slide_kind") or "").strip().lower()
        if kind == "events":
            merged.append({
                "slide_number": n, "slide_id": sid, "kind": "events",
                "events": slide.get("events", []), "papers": slide.get("papers", []),
                "source_title": "", "claim": "", "support": "",
            })
        elif kind == "resources":
            merged.append({
                "slide_number": n, "slide_id": sid, "kind": "resources",
                "source_title": "", "claim": "", "support": "",
            })
        else:
            base = by_id.get(sid, {})
            merged.append({
                "slide_number": n, "slide_id": sid, "kind": "content",
                "source_title": base.get("source_title", ""),
                "claim": base.get("claim", ""), "support": base.get("support", ""),
            })
    return merged


def enrich_with_blueprint_beats(slides: list[dict], blueprint_path: Path) -> int:
    """Attach the approved editorial blueprint's per-slide ``narration_beats``
    (and the visual narration hook) to each content slide, keyed by slide id.
    The templated presenter script only carries a one-line claim per slide;
    the blueprint beats are the reviewed pedagogical substance the narration
    must actually convey. Returns the number of slides enriched."""

    if not blueprint_path.is_file():
        return 0
    try:
        blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    by_id: dict[str, dict] = {}
    for slide in blueprint.get("slides") or [] if isinstance(blueprint, dict) else []:
        if isinstance(slide, dict) and slide.get("slide_id"):
            by_id[str(slide["slide_id"])] = slide
    enriched = 0
    for s in slides:
        if s.get("kind", "content") != "content":
            continue
        bp = by_id.get(str(s.get("slide_id") or ""))
        if not bp:
            continue
        beats = [str(b).strip() for b in (bp.get("narration_beats") or []) if str(b).strip()]
        if not beats and str(bp.get("narration_beat") or "").strip():
            beats = [str(bp["narration_beat"]).strip()]
        if beats:
            s["beats"] = beats
        hook = str(bp.get("visual_narration_hook") or "").strip()
        if hook:
            s["hook"] = hook
        if beats or hook:
            enriched += 1
    return enriched


class NarrationTruncated(SystemExit):
    """The reply used its whole token ceiling, so the JSON in it is cut off."""


def call_model(prompt: str, *, save_truncated_to: Path | None = None) -> str:
    if not has_credentials(PROVIDER):
        raise SystemExit(
            f"{' or '.join(credential_names(PROVIDER) or ())} is required for {PROVIDER} "
            "(source the repo .env first)."
        )
    result = complete(
        TASK_ID,
        prompt=prompt,
        provider=PROVIDER,
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        stream=True,  # Anthropic wants streaming at this ceiling; other providers ignore it
        timeout=float(os.environ.get("ENGAGING_NARRATION_TIMEOUT_SECONDS", "1800")),
        caller=__file__,
    )
    if result.stop_reason == "max_tokens":
        # Stopping here is the point: a cut-off reply fed to the JSON parser only
        # produces a traceback that hides why, and the reply was already paid for.
        kept = ""
        if save_truncated_to is not None:
            save_truncated_to.write_text(result.text, encoding="utf-8")
            kept = f" The {len(result.text)} characters it did return are in {save_truncated_to}."
        raise NarrationTruncated(
            f"{PROVIDER} narration reply used its whole {MAX_TOKENS}-token ceiling "
            f"({result.output_tokens} output tokens, blocks {', '.join(result.block_types) or 'none'}), "
            "so the JSON is cut off. Hidden reasoning counts toward the ceiling: lower "
            "ENGAGING_NARRATION_REASONING_EFFORT or raise ENGAGING_NARRATION_MAX_TOKENS." + kept
        )
    return result.text


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rsplit("```", 1)[0]
    start, end = raw.find("{"), raw.rfind("}")
    return json.loads(raw[start : end + 1])


def render_two_host_md(payload: dict) -> str:
    out = ["# Structured Presenter Script (NotebookLM two-host, engaging draft)", ""]
    for s in payload.get("slides", []):
        out.append(f"### [SLIDE {s.get('slide_number')}] {s.get('slide_id')}")
        out.append("**DIALOGUE:**")
        for turn in s.get("two_host", []):
            out.append(f"- **{turn.get('speaker')}:** {turn.get('text')}")
        out.append("")
    return "\n".join(out)


def render_single_md(payload: dict) -> str:
    out = ["# Structured Presenter Script (single speaker, engaging draft)", ""]
    for s in payload.get("slides", []):
        out.append(f"### [SLIDE {s.get('slide_number')}] {s.get('slide_id')}")
        out.append("**NARRATION:**")
        out.append(s.get("single", "").strip())
        out.append("")
    return "\n".join(out)


_MD_SLIDE_RE = re.compile(r"^### \[SLIDE [^\]]*\] (\S+)\s*$")
_MD_TURN_RE = re.compile(r"^- \*\*([^*:]+):\*\* (.*)$")


def sync_raw_from_md(payload: dict, two_host_md: str, single_md: str | None = None) -> int:
    """Carry text edits made to the ENGAGING markdown back into the raw JSON.

    The markdown fixers (fix_narration_script / fix_narration_series /
    fix_presenter_narration) edit the drafts in place, but the delivery and TTS
    are rebuilt from engaging_narration_raw.json -- so without this their
    fixes were reviewed and then discarded. Only text changes are carried: a
    slide or turn the markdown adds, drops, or reassigns to another speaker is
    a structural edit, and it is refused rather than guessed at.
    """

    turns_by_slide: dict[str, list[tuple[str, str]]] = {}
    current = None
    for line in two_host_md.splitlines():
        slide_match = _MD_SLIDE_RE.match(line)
        if slide_match:
            current = slide_match.group(1)
            turns_by_slide[current] = []
            continue
        turn_match = _MD_TURN_RE.match(line)
        if turn_match and current is not None:
            turns_by_slide[current].append((turn_match.group(1).strip(), turn_match.group(2).strip()))
    singles: dict[str, str] = {}
    if single_md:
        current, collecting, lines = None, False, []
        for line in single_md.splitlines() + ["### [SLIDE end] __end__"]:
            slide_match = _MD_SLIDE_RE.match(line)
            if slide_match:
                if current is not None:
                    singles[current] = "\n".join(lines).strip()
                current, collecting, lines = slide_match.group(1), False, []
            elif line.strip() == "**NARRATION:**":
                collecting = True
            elif collecting:
                lines.append(line)

    slides = payload.get("slides", [])
    if set(turns_by_slide) != {str(s.get("slide_id")) for s in slides}:
        raise ValueError("ENGAGING markdown and engaging_narration_raw.json name different slides")
    changed = 0
    for slide in slides:
        slide_id = str(slide.get("slide_id"))
        md_turns = turns_by_slide[slide_id]
        raw_turns = slide.get("two_host", [])
        if [speaker for speaker, _ in md_turns] != [str(t.get("speaker")) for t in raw_turns]:
            raise ValueError(f"{slide_id}: markdown turns differ structurally from the raw narration")
        for turn, (_, text) in zip(raw_turns, md_turns, strict=True):
            if turn.get("text") != text:
                turn["text"] = text
                changed += 1
        single = singles.get(slide_id)
        if single and single != str(slide.get("single") or "").strip():
            slide["single"] = single
            changed += 1
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument(
        "--source-script",
        type=Path,
        help="Existing structured script to draw per-slide substance from "
        "(default: 6_presenter_script_notebooklm_2host.md in the deck dir).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the prompt only; do not call the model.")
    parser.add_argument(
        "--render-from-raw",
        action="store_true",
        help="No model call: rewrite the *.ENGAGING.md drafts from engaging_narration_raw.json, "
        "so reviewers read exactly the text TTS will speak after a JSON-side fix.",
    )
    parser.add_argument(
        "--sync-raw-from-md",
        action="store_true",
        help="No model call: carry text fixes made to the *.ENGAGING.md drafts back into "
        "engaging_narration_raw.json (the TTS source of truth); refuses structural edits.",
    )
    args = parser.parse_args(argv)
    if args.sync_raw_from_md:
        deck_dir = args.deck_dir.expanduser().resolve()
        raw_path = deck_dir / "engaging_narration_raw.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        single_path = deck_dir / "6_presenter_script.ENGAGING.md"
        changed = sync_raw_from_md(
            payload,
            (deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md").read_text(encoding="utf-8"),
            single_path.read_text(encoding="utf-8") if single_path.is_file() else None,
        )
        raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Synced {changed} edited turn(s) from the ENGAGING drafts into {raw_path}", file=sys.stderr)
        return 0
    if args.render_from_raw:
        deck_dir = args.deck_dir.expanduser().resolve()
        payload = json.loads((deck_dir / "engaging_narration_raw.json").read_text(encoding="utf-8"))
        (deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md").write_text(
            render_two_host_md(payload), encoding="utf-8"
        )
        (deck_dir / "6_presenter_script.ENGAGING.md").write_text(render_single_md(payload), encoding="utf-8")
        print(f"Re-rendered ENGAGING drafts from {deck_dir / 'engaging_narration_raw.json'}", file=sys.stderr)
        return 0

    deck_dir = args.deck_dir.expanduser().resolve()
    source_script = args.source_script or (deck_dir / "6_presenter_script_notebooklm_2host.md")
    content_slides = parse_current_script(source_script)
    if not content_slides:
        raise SystemExit(f"No slides parsed from {source_script}")
    # Order all slides (incl. the events/resources bookends) as render_spec lays them out.
    render_spec_path = deck_dir / "render_spec.json"
    slides = merge_with_bookends(content_slides, render_spec_path) if render_spec_path.is_file() else content_slides
    enrich_with_blueprint_beats(slides, deck_dir / "presentation_blueprint.json")
    facts = load_extracted_facts(deck_dir, [s for s in slides if s.get("source_title")])

    # Frame the narration for this specific deck + conference: the deck's
    # audience_title comes from presentation_frame.json; the conference label
    # from the workspace config.yaml (workspace = the parent of decks/).
    workspace = deck_dir.parent.parent
    audience_title = ""
    frame_path = deck_dir / "presentation_frame.json"
    if frame_path.is_file():
        try:
            audience_title = str(json.loads(frame_path.read_text(encoding="utf-8")).get("audience_title") or "")
        except (ValueError, OSError):
            audience_title = ""
    conference_label = conference_name(workspace, default="the conference")
    prompt = build_prompt(slides, facts, audience_title=audience_title, conference_label=conference_label)

    bookends = sum(1 for s in slides if s.get("kind") in {"events", "resources"})
    print(f"Narrating {len(slides)} slides ({bookends} bookend); matched {len(facts)} sources to facts.", file=sys.stderr)
    if args.dry_run:
        (deck_dir / "engaging_narration_prompt.txt").write_text(prompt, encoding="utf-8")
        print(f"[dry-run] prompt written to {deck_dir / 'engaging_narration_prompt.txt'}", file=sys.stderr)
        return 0

    truncated_copy = deck_dir / "engaging_narration_truncated_reply.txt"
    raw = call_model(prompt, save_truncated_to=truncated_copy)
    payload = parse_json(raw)
    locator_hits = locator_references(payload)
    absence_hits = absence_claims(payload)
    author_hits = closed_author_lists(payload)
    aside_hits = aside_specifics(payload, prompt, hosts=(HOST_A, HOST_B))
    if locator_hits or aside_hits or absence_hits or author_hits:
        # One repair round: the model is told exactly which lines read a locator
        # aloud or carry a name or date no source backs; a second failure is
        # reported (the narration review gate and Gate A also reject these)
        # rather than silently shipped.
        repair = prompt + "\n\nREPAIR: your previous draft has problems. Return the full JSON again."
        if absence_hits:
            repair += "\n- Remove unsupported source-absence assertions; describe only what we compare here:\n" + "\n".join(absence_hits)
        if author_hits:
            repair += ("\n- These lines read a list of names as the full authorship; name at most two "
                       "people per work, then 'and colleagues':\n" + "\n".join(author_hits[:12]))
        if locator_hits:
            repair += (
                "\n- It referred to sources by locator instead of by work title. Rewrite ONLY "
                "these lines so they name the work (or describe what it shows):\n  - "
                + "\n  - ".join(locator_hits[:12])
            )
        if aside_hits:
            repair += (
                "\n- These asides name people, projects, comics, games or years that no source in "
                "this deck backs, and a reviewer cannot verify them. Rewrite each aside as a plain "
                "everyday analogy or an unnamed, undated scene (COLOR BUDGET); keep the slide's "
                "substance and the rest of the narration unchanged:\n  - "
                + "\n  - ".join(aside_hits[:12])
            )
        payload = parse_json(call_model(repair, save_truncated_to=truncated_copy))
        locator_hits = locator_references(payload)
        absence_hits = absence_claims(payload)
        author_hits = closed_author_lists(payload)
        aside_hits = aside_specifics(payload, prompt, hosts=(HOST_A, HOST_B))
        if locator_hits or aside_hits or absence_hits or author_hits:
            print(
                f"WARNING: narration still has {len(locator_hits)} locator line(s) and "
                f"{len(aside_hits)} unbacked aside(s) and {len(absence_hits)} absence claim(s) and "
                f"{len(author_hits)} closed author list(s); the review gates will flag them.",
                file=sys.stderr,
            )

    two_host_path = deck_dir / "6_presenter_script_notebooklm_2host.ENGAGING.md"
    single_path = deck_dir / "6_presenter_script.ENGAGING.md"
    raw_path = deck_dir / "engaging_narration_raw.json"
    two_host_path.write_text(render_two_host_md(payload), encoding="utf-8")
    single_path.write_text(render_single_md(payload), encoding="utf-8")
    raw_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    n = len(payload.get("slides", []))
    print(f"Wrote {n} slides of two-host + single-speaker narration:", file=sys.stderr)
    print(f"  {two_host_path}", file=sys.stderr)
    print(f"  {single_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
