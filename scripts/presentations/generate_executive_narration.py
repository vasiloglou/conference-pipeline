#!/usr/bin/env python3
"""Opus-generate clean, slide-matched narration for the executive-summary video.

The deterministic narration in the delivery contracts leaks internal scaffolding
("Tight transcript paraphrase", "Paper signal:") and no longer matches the
redesigned slides. This regenerates natural spoken prose per slide from the
render_spec + topic digests, writes it into both delivery contracts
(single_speaker + notebooklm), and flips approve_for_tts on.

Needs ANTHROPIC_API_KEY: `set -a; source <repo>/.env; set +a`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402

TARGET_WORDS = {"title": 70, "events": 130, "resources": 60,
                "evidence": 150, "bigpicture": 110, "synthesis": 130}

# The delivery gate (validate_video_delivery.py) fails a video over 40 minutes.
# Per-slide targets alone never bounded the deck: Opus runs ~15% over them, and
# the ICML 2026 long cut came out at 40.5 min single / 46.4 min two-host. So the
# deck has a word budget under that gate, measured at the TTS's slower pace.
MAX_VIDEO_MINUTES = 38
SPOKEN_WPM = 155
WORD_BUDGET = MAX_VIDEO_MINUTES * SPOKEN_WPM
# Opus overshoots its targets; aim below the budget so the first pass fits.
TARGET_OVERSHOOT = 1.15



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def load_topic_count(workspace: Path) -> int:
    """Number of analysis topics for this workspace (from analysis_manifest.json)."""
    manifest_path = workspace / "analysis" / "analysis_manifest.json"
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return 0
    topics = manifest.get("topics")
    return len(topics) if isinstance(topics, list) else 0

def build_system(conference_label: str) -> str:
    return (
    f"You script the spoken narration for an {conference_label} executive-readout VIDEO for technical "
    "leaders (research directors, eng VPs, founders), in a TEACHING style — explain so a smart "
    "non-specialist follows along and enjoys it. Write natural, fluent spoken prose in the first "
    "person plural ('we'). For each CONTENT slide: (1) open by posing the listener's natural "
    "question, then answer it (ask-and-answer rhythm like 'So what actually breaks?', 'Why does "
    "that matter?') without faking a two-person dialogue; (2) define any jargon in plain words "
    "BEFORE you name the technical term; (3) weave in ONE concrete anecdote, result, or number "
    "from the provided TEACHING material (a real finding from the papers/talks) and ONE "
    "plain-language analogy where it helps the idea land; (4) then land the decision/implication. "
    "When a new topic starts, add a one-line 'why you should care' stake. Name real papers, works, "
    "and speakers naturally. NEVER use internal labels such as 'Tight transcript paraphrase', "
    "'Paper signal', 'Transcript signal', 'Decision:', 'Source anchor', 'slide N', or 'this "
    "slide'. No bullet syntax, no markup. "
    # build_presenter_video.py refuses narration containing a stage direction --
    # text telling someone how to USE a slide rather than words spoken to the
    # audience. It is a fail-closed guard, and it caught the long executive
    # deck's "Use this as the drill-down map" only at the video build, after
    # the narration had already been paid for. The generator now knows the rule.
    "NEVER write stage directions -- instructions about how to use or present a "
    "slide rather than words spoken aloud. In particular never begin a sentence "
    "with 'Use this as', 'Open with', or 'Close the deck', never say 'not spoken', "
    "'presenter reference', or 'appendix-style source slide', and never read out a "
    "file path or an [arXiv] marker. Say what the slide MEANS, as speech. "
    "Keep each slide near its target word count (slightly "
    'over is fine for teaching). Return ONLY a JSON array of {"slide_number": int, "narration": str}.'
)


def topic_lookup(digests: dict) -> dict:
    return {t["topic_id"]: t for t in digests.get("topics", [])}


def diagram_for(lesson_id: str, topics: dict) -> dict:
    m = re.match(r"(.+)_(evidence|bigpicture)$", lesson_id or "")
    if m:
        t = topics.get(m.group(1))
        if not t:
            return {}
        key = "evidence_slide" if m.group(2) == "evidence" else "bigpicture_slide"
        return t.get("short", {}).get(key, {}).get("diagram", {})
    ml = re.match(r"(.+)_long_(\d+)$", lesson_id or "")
    if ml:
        t = topics.get(ml.group(1))
        slides = t.get("long", {}).get("slides", []) if t else []
        idx = int(ml.group(2)) - 1
        if 0 <= idx < len(slides):
            return slides[idx].get("diagram", {})
    return {}


def teaching_for(lesson_id: str, topics: dict) -> str:
    """Concrete anecdotes + analogies for this lesson's topic (from add_teaching_anecdotes)."""
    m = re.match(r"(.+)_(evidence|bigpicture|long_\d+)$", lesson_id or "")
    teaching = topics.get(m.group(1), {}).get("teaching", {}) if m else {}
    if not teaching:
        return ""
    anec = "; ".join(f"{a.get('point')} — {a.get('detail')}" for a in teaching.get("anecdotes", [])[:5])
    anal = "; ".join(f"{a.get('concept')} = {a.get('analogy')}" for a in teaching.get("analogies", [])[:3])
    parts = []
    if anec:
        parts.append(f"TEACHING ANECDOTES/RESULTS (weave in ONE that fits): {anec}")
    if anal:
        parts.append(f"ANALOGIES (use ONE if it helps): {anal}")
    return (" " + " || ".join(parts)) if parts else ""


def slide_brief(
    slide: dict,
    idx: int,
    topics: dict,
    digests: dict,
    *,
    conference_label: str = "the conference",
    topic_count: int = 0,
) -> tuple[str, int]:
    kind = str(slide.get("slide_kind") or "")
    title = str(slide.get("title") or "")
    lesson = str(slide.get("executive_lesson_id") or "")
    count = topic_count or len(topics)
    if kind == "title":
        return (f"OPENING TITLE. Deck: '{conference_label} Executive Readout'. The one-line thesis on "
                f"screen: '{slide.get('title_statement','')}'. Set up the readout: a leadership "
                "synthesis of the conference, organized as decisions, not a paper tour.",
                TARGET_WORDS["title"])
    if kind == "events":
        # The index is grouped into narrative arcs; narrate it as a guided
        # itinerary (name each arc, then sweep its topics) rather than a flat list.
        arcs: dict[str, list[str]] = {}
        order: list[str] = []
        for e in slide.get("events", []):
            arc = str(e.get("arc") or "Topics")
            if arc not in arcs:
                arcs[arc] = []
                order.append(arc)
            arcs[arc].append(str(e.get("name") or ""))
        if len(order) > 1:
            grouped = " | ".join(f"ARC '{a}': {', '.join(arcs[a])}" for a in order)
            return (f"GUIDED ITINERARY for the readout: the {count} {conference_label} topics are grouped on "
                    f"screen into {len(order)} arcs. Walk the audience through the arcs in order, "
                    f"naming each arc and the journey it represents (don't read all {count} titles): "
                    f"{grouped}. Frame these arcs as the route the rest of the readout will travel.",
                    TARGET_WORDS["events"])
        items = "; ".join(f"{e.get('name')} ({e.get('detail')})" for e in slide.get("events", []))
        return (f"INDEX of the {count} {conference_label} topics shown on screen, each with a one-line takeaway: "
                f"{items}. Frame these as the map of the readout.", TARGET_WORDS["events"])
    if kind == "resources":
        return ("CLOSING slide pointing to the 12 topic presentations and analysis reports for "
                "drill-down. Wrap up the readout and invite the team to go deeper.", TARGET_WORDS["resources"])
    # content
    if lesson.startswith("synthesis_open_questions"):
        qs = "; ".join(q.get("question", "") for q in digests.get("open_questions", []))
        return (f"OPEN QUESTIONS the conference leaves for leaders, shown as a list: {qs}.",
                TARGET_WORDS["synthesis"])
    if lesson.startswith("synthesis_predictions"):
        ps = "; ".join(p.get("prediction", "") for p in digests.get("predictions", []))
        return (f"PREDICTIONS for what comes next, shown as a list: {ps}.", TARGET_WORDS["synthesis"])
    role = "bigpicture" if lesson.endswith("_bigpicture") else "evidence"
    diag = diagram_for(lesson, topics)
    diag_desc = (f" On screen is a {diag.get('kind')} diagram: {diag.get('caption')} "
                 f"(elements: {', '.join(diag.get('nodes', []))})." if diag else "")
    src = slide.get("paper_evidence_text", "")
    src_desc = f" Named sources: {re.sub(r'^Anchored in ', '', src).rstrip('.')}." \
        if str(src).startswith("Anchored in") else ""
    teach = teaching_for(lesson, topics)
    if role == "evidence":
        return (f"TOPIC EVIDENCE slide. Headline: '{title}'. What the conference showed: "
                f"{slide.get('evidence_text','')}.{src_desc}{diag_desc} Then point to the decision: "
                f"{slide.get('decision_implication','')}.{teach}", TARGET_WORDS["evidence"])
    return (f"TOPIC DECISION slide. Headline: '{title}'. The leadership implication: "
            f"{slide.get('evidence_text','')}.{diag_desc} Make it a concrete call to action.{teach}",
            TARGET_WORDS["bigpicture"])


# Slides per Opus call. Batching keeps every response well under max_tokens so a
# large deck (the long executive cut is ~38 slides) never truncates the JSON
# array mid-stream — which previously failed parsing on the long deck.
NARRATION_BATCH = 8


def _call_opus_batch(client, model, briefs, system):
    from conference_pipeline.provider_receipts import anthropic_messages_create
    user = ("Write narration for these slides. The number in brackets is a word LIMIT for that "
            "slide; stay at or under it.\n\n") + "\n\n".join(
        f"SLIDE {sn} [at most {tw} words]: {brief}" for sn, brief, tw in briefs
    )
    # A reply occasionally comes back as prose or bare objects instead of the array
    # (ICML 2026, 2026-11-06); one reminder retry recovers it, a second miss fails closed.
    for attempt in range(2):
        if attempt:
            user += "\n\nReturn ONLY the JSON array, starting with [ and ending with ]."
        # Opus 5.5 thinks by default inside max_tokens, so 8000 truncated the long
        # deck's array mid-stream; a cut-off reply is never parsed.
        result = anthropic_messages_create(client, caller="generate_executive_narration",
            model=model, max_tokens=20000, system=system,
            messages=[{"role": "user", "content": user}])
        if getattr(result.response, "stop_reason", None) == "max_tokens":
            raise ValueError("narration response hit max_tokens; batch is too large")
        text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            return {int(o["slide_number"]): o["narration"].strip() for o in json.loads(m.group(0))}
        print(f"  ! no JSON array in narration response (got {len(text)} chars): {text[:300]!r}",
              file=sys.stderr)
    raise ValueError(f"no JSON array in narration response after a retry (got {len(text)} chars)")


def call_opus(client, model, briefs, system):
    narr: dict[int, str] = {}
    for i in range(0, len(briefs), NARRATION_BATCH):
        narr.update(_call_opus_batch(client, model, briefs[i:i + NARRATION_BATCH], system))
    return narr


def _scaled(briefs, factor: float):
    return [(sn, brief, max(40, int(tw * factor))) for sn, brief, tw in briefs]


def narrate_within_budget(client, model, briefs, system, budget: int = WORD_BUDGET):
    """Narrate the deck inside the word budget: scale targets to fit, retry once
    tighter if the result still runs over, then fail closed before any TTS spend."""
    planned = sum(tw for _, _, tw in briefs)
    briefs = _scaled(briefs, min(1.0, budget / (planned * TARGET_OVERSHOOT)))
    narr = call_opus(client, model, briefs, system)
    total = sum(len(text.split()) for text in narr.values())
    if total > budget:
        print(f"  ! narration {total} words over the {budget}-word budget; retrying tighter",
              file=sys.stderr)
        narr = call_opus(client, model, _scaled(briefs, 0.95 * budget / total), system)
        total = sum(len(text.split()) for text in narr.values())
    if total > budget:
        raise SystemExit(f"narration is {total} words, over the {budget}-word budget "
                         f"(~{MAX_VIDEO_MINUTES} min at {SPOKEN_WPM} wpm)")
    return narr


def write_contract(path: Path, narr: dict, model: str) -> int:
    d = json.loads(path.read_text())
    for seg in d.get("segments", []):
        n = narr.get(int(seg.get("slide_number")))
        if n:
            seg["narration"] = n
            wc = len(n.split())
            seg["narration_word_count"] = wc
            seg["narration_estimated_seconds_at_155_wpm"] = round(wc / 155 * 60, 2)
    d["approve_for_tts"] = True
    d["approval_blocker"] = None
    d["narration_source"] = f"opus_{model}_slide_matched"
    # The two-host build needs style to be a host/voice dict (the exec-summary
    # generator mistakenly wrote the presentation type string here).
    if "notebooklm" in path.name and not isinstance(d.get("style"), dict):
        d["style"] = {"mode": "notebooklm", "two_host": True,
                      "host_a": "Iapetus", "host_b": "Erinome",
                      "voice_a": "Iapetus", "voice_b": "Erinome"}
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return sum(len(str(s.get("narration", "")).split()) for s in d.get("segments", []))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True,
                   help="Conference workspace dir (e.g. conferences/<slug>).")
    p.add_argument("--deck-id", default="executive_summary_short")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    args = p.parse_args(argv)
    from anthropic import Anthropic

    workspace = REPO / args.workspace
    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_executive_narration")
    conference_label = conference_name(workspace, default="the conference")
    topic_count = load_topic_count(workspace)

    deck = workspace / "decks" / args.deck_id
    rs = json.loads((deck / "render_spec.json").read_text())
    digests = json.loads((workspace / "decks" / "executive_topic_digests.json").read_text())
    topics = topic_lookup(digests)

    briefs = []
    for idx, slide in enumerate(rs["slides"], start=1):
        brief, tw = slide_brief(slide, idx, topics, digests,
                                conference_label=conference_label, topic_count=topic_count)
        briefs.append((idx, brief, tw))

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    narr = narrate_within_budget(client, args.model, briefs, build_system(conference_label))
    if len(narr) < len(briefs):
        print(f"WARN: only {len(narr)}/{len(briefs)} narrations returned", file=sys.stderr)

    total = write_contract(deck / "delivery_single_speaker.json", narr, args.model)
    write_contract(deck / "delivery_notebooklm.json", narr, args.model)
    print(f"narration written to both contracts: {len(narr)} slides, ~{total} words "
          f"(~{round(total/155)} min). approve_for_tts=True", file=sys.stderr)
    print(f"\nSAMPLE slide 1: {narr.get(1,'')[:300]}", file=sys.stderr)
    print(f"SAMPLE slide 4: {narr.get(4,'')[:300]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
