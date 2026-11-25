#!/usr/bin/env python3
"""Add a short axis label to every spectrum/ladder diagram in the executive digests.

A 3-AI panel (2026-06-22) found the comparison/spectrum diagrams teach poorly
because the axis meaning is implicit — the nodes sit on an unlabeled line. This
Opus pass reads each spectrum/ladder diagram (kind in comparison/spectrum/ladder)
and writes a 1-4 word `axis` naming the dimension that increases left->right (or
bottom->top for a ladder), e.g. "Signal richness", "Evaluation realism". The
renderer appends the arrow and draws it. Matrix/pipeline/checklist/architecture
diagrams are left untouched (the renderer titles the matrix "What to weigh").

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

AXIS_KINDS = {"comparison", "spectrum", "ladder"}

SYSTEM = (
    "You label the reading axis of a technical comparison diagram. Each diagram is a row "
    "(or ladder) of nodes ordered along one hidden dimension. Given the caption and the "
    "ordered node labels, name that dimension in 1-4 words, as it INCREASES along the arrow "
    "(left->right for a spectrum, bottom->top for a ladder). Examples: 'Signal richness', "
    "'Evaluation realism', 'Autonomy', 'Supervision density', 'Robustness'. No arrow, no "
    "punctuation, Title Case. Return ONLY a JSON object mapping each given id (as a string) "
    'to its axis label: {"0":"Signal richness","1":"Evaluation realism"}.'
)



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def collect(digests: dict) -> list[dict]:
    items = []
    for t in digests.get("topics", []):
        short = t.get("short", {})
        for key in ("evidence_slide", "bigpicture_slide"):
            d = short.get(key, {}).get("diagram")
            if isinstance(d, dict):
                items.append(d)
        for sl in t.get("long", {}).get("slides", []):
            d = sl.get("diagram")
            if isinstance(d, dict):
                items.append(d)
    return items


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", default="conferences/iclr2026")
    p.add_argument("--digests", default="decks/executive_topic_digests.json")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    p.add_argument("--force", action="store_true", help="relabel even if axis already set")
    args = p.parse_args(argv)
    from anthropic import Anthropic
    from conference_pipeline.provider_receipts import anthropic_messages_create

    path = REPO / args.workspace / args.digests
    # D-QUICK-260715-rz1: establish a workspace-rooted receipt context so
    # standalone invocations don't fall back to Path.cwd() (repo root).
    os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(REPO / args.workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "enrich_diagram_axes")
    digests = json.loads(path.read_text())
    diagrams = collect(digests)
    targets = [d for d in diagrams
               if str(d.get("kind") or "").lower() in AXIS_KINDS
               and (args.force or not str(d.get("axis") or "").strip())]
    if not targets:
        print("no spectrum/ladder diagrams need an axis", file=sys.stderr)
        return 0

    listing = "\n".join(
        f'{i}: kind={d.get("kind")}; caption={d.get("caption","")}; '
        f'nodes={" | ".join(str(x) for x in (d.get("nodes") or []))}'
        for i, d in enumerate(targets))
    user = f"Label the axis for each diagram:\n\n{listing}"
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = anthropic_messages_create(client, caller="enrich_diagram_axes",
        model=args.model, max_tokens=2000, system=SYSTEM,
        messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
    mapping = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))

    labelled = 0
    for i, d in enumerate(targets):
        axis = str(mapping.get(str(i), "")).strip().rstrip("→").strip()
        if axis:
            d["axis"] = axis
            labelled += 1
    path.write_text(json.dumps(digests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"labelled {labelled}/{len(targets)} spectrum/ladder diagrams with an axis", file=sys.stderr)
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
