#!/usr/bin/env python3
"""Turn the single-narrator executive narration into a two-host conversation.

The NotebookLM video build renders whatever narration is in delivery_notebooklm.json;
topic decks put already-transformed dialogue there ("Iapetus: ... Erinome: ..."). This
reads the clean single-speaker narration and asks Opus to rewrite each slide as a lively
two-host exchange, then writes it into delivery_notebooklm.json so the build produces a
real multi-voice conversation.

Needs ANTHROPIC_API_KEY.
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

HOST_A, HOST_B = "Iapetus", "Erinome"

# The video builder owns the list of text TTS must never speak; this script is
# on that builder's sys.path (same directory), so it imports the list rather
# than carrying a copy that would drift.
from build_presenter_video import FORBIDDEN_TTS_TEXT_PATTERNS  # noqa: E402
from generate_executive_narration import SPOKEN_WPM, WORD_BUDGET  # noqa: E402

FORBIDDEN_PHRASE_EXAMPLES = (
    "Use this as", "Open with", "Close the deck", "Make this point concrete",
    "Appendix-style source slide", "not spoken", "presenter reference",
)


SYSTEM = (
    f"You turn a single-narrator executive-readout script into a natural two-host conversation "
    f"for a NotebookLM-style video. Host '{HOST_A}' is the curious generalist who asks and "
    f"reframes; host '{HOST_B}' is the expert who explains. Keep ALL the substance of each slide "
    "— the same points, the named papers/people, and any 'on screen' description — but make it "
    "lively and conversational with short, natural turns. Cover each slide in 3 to 6 alternating "
    f"turns, starting with {HOST_A}. Format strictly as lines 'Iapetus: ...' and 'Erinome: ...' "
    "joined by spaces (no line breaks needed, no stage directions, no markdown). Return ONLY a "
    'JSON array of {"slide_number": int, "dialogue": str}. '
    "NEVER write stage directions -- text addressed to the presenter rather than the listener. "
    "The TTS gate refuses these phrases outright, so never use them, even when the source "
    "narration does: " + ", ".join(f"'{example}'" for example in FORBIDDEN_PHRASE_EXAMPLES) + ". "
    "Say what a slide IS ('this slide is the handoff'), never what to DO with it ('use this as')."
)



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def non_spoken_findings(dialogue: dict[int, str]) -> list[tuple[int, str]]:
    """(slide_number, pattern_name) for every dialogue line the TTS gate would refuse.

    Same patterns as build_presenter_video.FORBIDDEN_TTS_TEXT_PATTERNS, checked
    HERE so a bad rewrite is caught at generation ($0.10 to redo one batch)
    rather than at video build, after every earlier step has run.
    """

    findings: list[tuple[int, str]] = []
    for slide_number, text in sorted(dialogue.items()):
        for name, pattern in FORBIDDEN_TTS_TEXT_PATTERNS:
            if pattern.search(text):
                findings.append((slide_number, name))
    return findings


def write_dialogue(path: Path, dialogue: dict) -> int:
    d = json.loads(path.read_text())
    for seg in d.get("segments", []):
        text = dialogue.get(int(seg.get("slide_number")))
        if text:
            seg["narration"] = text
            seg["narration_word_count"] = len(re.sub(r"\b(Iapetus|Erinome):", "", text).split())
    d["approve_for_tts"] = True
    d["approval_blocker"] = None
    d["narration_source"] = "opus_two_host_dialogue"
    if not isinstance(d.get("style"), dict):
        d["style"] = {"mode": "notebooklm", "two_host": True, "host_a": HOST_A,
                      "host_b": HOST_B, "voice_a": HOST_A, "voice_b": HOST_B}
    path.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
    return len(dialogue)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="conferences/iclr2026")
    p.add_argument("--deck-id", default="executive_summary_short")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    args = p.parse_args(argv)
    from anthropic import Anthropic
    from conference_pipeline.provider_receipts import anthropic_messages_create

    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(REPO / args.workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_executive_dialogue")

    deck = REPO / args.workspace / "decks" / args.deck_id
    single = json.loads((deck / "delivery_single_speaker.json").read_text())
    mono = [(int(s["slide_number"]), s.get("narration", "")) for s in single["segments"]]

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    # Batch so a large deck (the long cut is ~38 slides, and dialogue expands each
    # slide into several turns) never truncates the JSON array mid-stream.
    BATCH = 6
    dialogue: dict[int, str] = {}
    for i in range(0, len(mono), BATCH):
        chunk = mono[i:i + BATCH]
        # A conversation runs longer than the monologue it came from (ICML 2026's
        # long cut: 7,278 words from 6,727), so each slide is capped at its
        # source's length and the deck is checked against the shared budget.
        user = ("Rewrite each slide's narration as a two-host exchange. The number in brackets is "
                "a word LIMIT for that slide (host names not counted): do not run longer than the "
                "source.\n\n") + "\n\n".join(
            f"SLIDE {sn} [at most {len(text.split())} words]: {text}" for sn, text in chunk
        )
        batch: dict[int, str] = {}
        # One retry, naming the exact phrase, then fail closed: a contract the
        # TTS gate will refuse is not a contract worth writing.
        for attempt in range(2):
            result = anthropic_messages_create(client, caller="generate_executive_dialogue",
                model=args.model, max_tokens=20000, system=SYSTEM,
                messages=[{"role": "user", "content": user}])
            if getattr(result.response, "stop_reason", None) == "max_tokens":
                raise ValueError("dialogue response hit max_tokens; batch is too large")
            text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                if attempt:
                    raise ValueError(f"no JSON array in dialogue response (got {len(text)} chars)")
                print(f"  ! no JSON array in dialogue response, retrying once: {text[:300]!r}",
                      file=sys.stderr)
                user += "\n\nReturn ONLY the JSON array, starting with [ and ending with ]."
                continue
            batch = {int(o["slide_number"]): o["dialogue"].strip() for o in json.loads(m.group(0))}
            findings = non_spoken_findings(batch)
            if not findings:
                break
            described = "; ".join(f"slide {sn}: {name}" for sn, name in findings)
            print(f"  ! non-spoken text in dialogue, retrying once: {described}", file=sys.stderr)
            user = (
                user
                + "\n\nYour previous rewrite contained stage directions the TTS gate refuses ("
                + described
                + "). Rewrite those slides so no such phrase appears."
            )
        else:
            raise ValueError(f"dialogue still contains non-spoken text after retry: {described}")
        dialogue.update(batch)

    spoken = sum(len(re.sub(r"\b(Iapetus|Erinome):", "", text).split()) for text in dialogue.values())
    if spoken > WORD_BUDGET:
        raise SystemExit(f"two-host dialogue is {spoken} words, over the {WORD_BUDGET}-word budget "
                         f"(~{WORD_BUDGET // SPOKEN_WPM} min); rerun or shorten the narration")
    n = write_dialogue(deck / "delivery_notebooklm.json", dialogue)
    print(f"two-host dialogue written for {n} slides", file=sys.stderr)
    print("SAMPLE slide 3:", dialogue.get(3, "")[:320], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
