#!/usr/bin/env python3
"""Add a technical diagram spec to each long-deck sub-theme slide.

The long deck renders 3 deeper sub-theme slides per topic (digest.topics[i].long.slides).
This focused Opus pass adds a `diagram` (kind/caption/nodes) to each so the long deck
renders clean dark line-schematic diagrams like the short deck, instead of the plain
2-panel evidence bridge. One call per topic returns a diagram for each of its 3 slides.
Merges into executive_topic_digests.json.

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

KINDS = "comparison | pipeline | mechanism | architecture | ladder | matrix | checklist"

SYSTEM = (
    "You design clean technical diagrams for an executive readout. For EACH of a topic's "
    "deeper sub-theme slides, output ONE diagram spec that depicts that slide's structure. "
    f"Pick the best kind per slide from: {KINDS} ('pipeline'/'mechanism' for a process, 'ladder' "
    "for a hierarchy, 'architecture' for a stack/components, 'comparison' for contrasted options, "
    "'matrix' for factors, 'checklist' for steps). Nodes are 3-5 SHORT labels (<= 5 words each), "
    "concrete and specific to that slide. Return ONLY a JSON array (one object per slide, in order) "
    'of {"kind","caption" (one short line),"nodes" (list of 3-5)}.'
)



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def build_user(topic: dict) -> str:
    slides = topic.get("long", {}).get("slides", [])[:3]
    rows = [f"SLIDE {i+1}: headline='{s.get('headline')}' | body={s.get('body')}"
            for i, s in enumerate(slides)]
    return (f"TOPIC: {topic.get('title')}\n\nSub-theme slides:\n" + "\n".join(rows)
            + "\n\nReturn one diagram spec per slide, in order.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="conferences/iclr2026")
    p.add_argument("--digests", default="decks/executive_topic_digests.json")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    args = p.parse_args(argv)
    from anthropic import Anthropic
    from conference_pipeline.provider_receipts import anthropic_messages_create

    path = REPO / args.workspace / args.digests
    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(REPO / args.workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "add_long_diagrams")
    digests = json.loads(path.read_text())
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    for t in digests.get("topics", []):
        slides = t.get("long", {}).get("slides", [])[:3]
        if not slides:
            continue
        # 900 tokens fit Opus 4.8's three specs; Opus 5.5 writes longer ones. A
        # cut-off reply is retried once with twice the budget, never parsed.
        max_tokens = 3600
        for _attempt in range(2):
            result = anthropic_messages_create(
                client, caller="add_long_diagrams", model=args.model, max_tokens=max_tokens,
                system=SYSTEM, messages=[{"role": "user", "content": build_user(t)}])
            text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m and getattr(result.response, "stop_reason", None) != "max_tokens":
                break
            max_tokens *= 2
        if not m:
            raise SystemExit(f"long-diagram pass returned no JSON array: {text[:200]!r}")
        specs = json.loads(m.group(0))
        for slide, spec in zip(slides, specs):
            slide["diagram"] = spec
        print(f"  + {t['topic_id']}: {[s.get('kind') for s in specs]}", file=sys.stderr)

    path.write_text(json.dumps(digests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
