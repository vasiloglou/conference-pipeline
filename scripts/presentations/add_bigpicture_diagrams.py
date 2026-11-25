#!/usr/bin/env python3
"""Fill every missing diagram spec in the executive topic digests.

Every executive slide renders as a deterministic technical diagram, so every
slot the plan builder reads needs a spec: `short.evidence_slide.diagram` (the
digest pass authors this one), `short.bigpicture_slide.diagram` (a DISTINCT
leadership/decision view) and `long.slides[i].diagram` (each sub-theme's own
mechanism). It also fills `long.slides[i].takeaway`, the one-line lesson
under that slide's figure: the long deck used to print the topic's decision
headline under all three sub-theme figures, and the vision judge failed slides
whose figure it did not describe (ICML 2026). Slots already filled are left
alone, so the pass is idempotent and
free when nothing is missing. A slot left empty renders as a bare arrow on an
empty card -- CVPR 2026's first build shipped 48 of those -- so the deck
generator now refuses an empty spec, and this pass is what fills them. Merges
into executive_topic_digests.json.

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

# Shared with generate_executive_topic_digests.py so the two authoring prompts
# teach one vocabulary and one set of label conventions.
from generate_executive_topic_digests import DIAGRAM_KINDS, DIAGRAM_NODE_RULES  # noqa: E402

KINDS = DIAGRAM_KINDS.split(":", 1)[1].split(".", 1)[0].strip()

SYSTEM = (
    "You design clean technical diagrams for an executive readout. Given a conference topic's "
    "leadership implication, output ONE diagram spec for a DECISION/STRATEGY view — what leaders "
    "should change or weigh — that is DISTINCT from the supplied evidence (mechanism) diagram. "
    f"Pick the best kind from: {KINDS}. Use 'checklist' for ordered actions, 'comparison' for "
    "old-vs-new or competing options, 'architecture' for an operating-model/stack, 'matrix' for "
    "factors to weigh. Nodes are 3-5 SHORT labels (<= 5 words each), concrete and specific to this "
    "topic. " + DIAGRAM_NODE_RULES + " Return ONLY JSON: {\"kind\",\"caption\" (one short line),"
    "\"nodes\" (list of 3-5), optional \"hub\"}."
)


LONG_SYSTEM = (
    "You design clean technical diagrams for an executive readout. Given one sub-theme slide "
    "from a conference topic (headline, body, named sources), output ONE diagram spec that "
    "depicts THIS sub-theme's mechanism or structure, grounded only in the supplied text. "
    f"Pick the best kind from: {KINDS}. Nodes are 3-5 SHORT labels (<= 5 words each), concrete "
    "and specific to the sub-theme, never generic. Do not repeat the topic-level evidence "
    "diagram. " + DIAGRAM_NODE_RULES + " Return ONLY JSON: {\"kind\",\"caption\" (one short line),"
    "\"nodes\" (list of 3-5), optional \"hub\"}."
)

TAKEAWAY_SYSTEM = (
    "You write the one-line takeaway printed under a sub-theme slide's diagram in an executive "
    "readout. State the lesson THIS slide's diagram and body support, in at most 12 words, as a "
    "complete phrase (no trailing ellipsis, no dangling word). Use only the supplied text: no new "
    "numbers, sources or claims, and no stronger claim than the body makes. Return ONLY JSON: "
    "{\"takeaway\": \"...\"}."
)


def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def build_user(t: dict) -> str:
    ev = t.get("short", {}).get("evidence_slide", {})
    bp = t.get("short", {}).get("bigpicture_slide", {})
    return (
        f"TOPIC: {t.get('title')}\n"
        f"EVIDENCE diagram (do NOT repeat this view): {json.dumps(ev.get('diagram', {}))}\n"
        f"LEADERSHIP HEADLINE: {bp.get('headline')}\n"
        f"LEADERSHIP IMPLICATION: {bp.get('implication')}\n"
        f"CENTRAL TAKEAWAY: {t.get('central_takeaway')}\n\n"
        "Design the decision/strategy diagram spec as JSON."
    )


def build_long_user(t: dict, sl: dict) -> str:
    ev = t.get("short", {}).get("evidence_slide", {})
    return (
        f"TOPIC: {t.get('title')}\n"
        f"TOPIC-LEVEL EVIDENCE diagram (do NOT repeat this view): {json.dumps(ev.get('diagram', {}))}\n"
        f"SUB-THEME HEADLINE: {sl.get('headline')}\n"
        f"SUB-THEME BODY: {sl.get('body')}\n"
        f"NAMED SOURCES: {json.dumps(sl.get('named_sources', []))}\n\n"
        "Design this sub-theme's diagram spec as JSON."
    )


def build_takeaway_user(t: dict, sl: dict) -> str:
    return (
        f"TOPIC: {t.get('title')}\n"
        f"SLIDE HEADLINE: {sl.get('headline')}\n"
        f"SLIDE BODY: {sl.get('body')}\n"
        f"SLIDE DIAGRAM: {json.dumps(sl.get('diagram', {}))}\n\n"
        "Write this slide's takeaway as JSON."
    )


def _has_diagram(slot: dict) -> bool:
    spec = slot.get("diagram")
    return isinstance(spec, dict) and bool([n for n in (spec.get("nodes") or []) if str(n).strip()])


def main(argv: list[str] | None = None) -> int:
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
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "add_bigpicture_diagrams")
    digests = json.loads(path.read_text())
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def author(system: str, user: str, *, require_nodes: bool = True) -> dict:
        # 600 tokens fit Opus 4.8's specs; Opus 5.5 writes longer ones and was
        # cut off mid-object (2026-09-25). A cut-off reply is retried once with
        # twice the budget rather than parsed.
        max_tokens = 2400
        for _attempt in range(2):
            result = anthropic_messages_create(
                client, caller="add_bigpicture_diagrams", model=args.model, max_tokens=max_tokens,
                system=system, messages=[{"role": "user", "content": user}],
            )
            text = "".join(b.text for b in result.response.content if getattr(b, "type", "") == "text")
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m and getattr(result.response, "stop_reason", None) != "max_tokens":
                break
            max_tokens *= 2
        if not m:
            raise SystemExit(f"diagram pass returned no JSON object: {text[:200]!r}")
        spec = json.loads(m.group(0))
        if require_nodes and not [n for n in (spec.get("nodes") or []) if str(n).strip()]:
            raise SystemExit(f"diagram pass returned no nodes: {text[:200]!r}")
        return spec

    filled = skipped = 0
    for t in digests.get("topics", []):
        bp = t.setdefault("short", {}).setdefault("bigpicture_slide", {})
        if _has_diagram(bp):
            skipped += 1
        else:
            bp["diagram"] = spec = author(SYSTEM, build_user(t))
            filled += 1
            print(f"  + {t['topic_id']} bigpicture: {spec.get('kind')} — {spec.get('nodes')}", file=sys.stderr)
        for idx, sl in enumerate(t.get("long", {}).get("slides", []), start=1):
            if _has_diagram(sl):
                skipped += 1
            else:
                sl["diagram"] = spec = author(LONG_SYSTEM, build_long_user(t, sl))
                filled += 1
                print(f"  + {t['topic_id']} long_{idx}: {spec.get('kind')} — {spec.get('nodes')}", file=sys.stderr)
            if str(sl.get("takeaway") or "").strip():
                skipped += 1
            else:
                reply = author(TAKEAWAY_SYSTEM, build_takeaway_user(t, sl), require_nodes=False)
                takeaway = " ".join(str(reply.get("takeaway") or "").split()).rstrip(".… ")
                if not takeaway or len(takeaway.split()) > 12:
                    raise SystemExit(f"takeaway pass returned an unusable takeaway: {reply!r}")
                sl["takeaway"] = takeaway
                filled += 1
                print(f"  + {t['topic_id']} long_{idx} takeaway: {takeaway}", file=sys.stderr)
    print(f"diagram/takeaway slots filled={filled} already_present={skipped}", file=sys.stderr)

    path.write_text(json.dumps(digests, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
