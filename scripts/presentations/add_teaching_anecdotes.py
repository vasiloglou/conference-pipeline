#!/usr/bin/env python3
"""Extract vivid teaching material (anecdotes + analogies) for each topic from the
real paper/transcript analysis, so narration can ground claims in concrete detail.

A 3-AI panel found the narration names papers without telling their story. The
concrete results/anecdotes live in analysis/exhaustive_summaries/<topic>__papers.md
and __transcripts.md — which the narration pass never read. This Opus pass reads
those per topic and stores `digest.topics[i].teaching = {anecdotes, analogies}` for
the narration/dialogue generators to use.

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

CHAR_CAP = 20000

SYSTEM = (
    "You mine conference paper summaries and talk transcripts for the most VIVID, CONCRETE, "
    "MEMORABLE material a presenter could use to make ideas land for a smart non-specialist. "
    "Pull specifics: a surprising result, a concrete number or before/after, a named author's "
    "claim or quote, a striking failure case or example. Avoid vague generalities — every item "
    "must be specific and grounded in the provided text (do not invent numbers). Also propose "
    "plain-language ANALOGIES for the topic's hardest concepts (everyday comparisons a "
    "non-expert gets instantly). Return ONLY JSON: "
    '{"anecdotes":[{"point":"<short claim>","detail":"<concrete, grounded specifics, 1-2 sentences>"}],'
    '"analogies":[{"concept":"<jargon/idea>","analogy":"<plain everyday comparison>"}]}. '
    "Give 4-6 anecdotes and 2-3 analogies."
)



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def read_capped(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")[:CHAR_CAP] if path.is_file() else ""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="conferences/iclr2026")
    p.add_argument("--digests", default="decks/executive_topic_digests.json")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    args = p.parse_args(argv)
    from anthropic import Anthropic
    from conference_pipeline.provider_receipts import anthropic_messages_create

    ws = REPO / args.workspace
    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(ws))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "add_teaching_anecdotes")
    path = ws / args.digests
    digests = json.loads(path.read_text())
    exh = ws / "analysis" / "exhaustive_summaries"
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for t in digests.get("topics", []):
        tid = t["topic_id"]
        papers = read_capped(exh / f"{tid}__papers.md")
        transcripts = read_capped(exh / f"{tid}__transcripts.md")
        if not papers and not transcripts:
            print(f"  ! {tid}: no analysis files", file=sys.stderr)
            continue
        user = (f"TOPIC: {t.get('title')}\n\n=== PAPER SUMMARIES ===\n{papers}\n\n"
                f"=== TALK TRANSCRIPTS ===\n{transcripts}\n\nExtract the teaching material as JSON.")
        try:
            result = anthropic_messages_create(
                client, caller="add_teaching_anecdotes", model=args.model, max_tokens=2600,
                system=SYSTEM, messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            teaching = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {tid}: parse/call failed ({str(exc)[:80]}), skipping", file=sys.stderr)
            continue
        t["teaching"] = teaching
        print(f"  + {tid}: {len(teaching.get('anecdotes',[]))} anecdotes, "
              f"{len(teaching.get('analogies',[]))} analogies", file=sys.stderr)

    path.write_text(json.dumps(digests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
