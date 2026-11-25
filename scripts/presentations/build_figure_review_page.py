#!/usr/bin/env python3
"""Build the canary review page for one deck: every rendered slide screenshot
with its vision-judge findings, its adversarial figure-critic findings still
open after three rounds, and any operator outcome already recorded.

This is the page the operator reads to decide each open figure (accept, patch,
replace -- docs/operations/figure-fidelity-policy.md item 9). The runner writes
it beside the deck at the canary stop; it can also be rebuilt by hand.

Usage: build_figure_review_page.py --deck-dir <deck> [--output <html>]
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SEMANTIC_CODES = frozenset({
    "unsupported_number", "unsupported_label", "misattribution", "overclaim", "claim_not_encoded",
    "depiction_mismatch", "sequence_implies_causality", "alternatives_as_sequence", "incommensurable_axes",
})


def build(deck: Path, out: Path) -> dict[str, int]:
    contract = json.loads((deck / "presentation_deck.json").read_text())
    slide_ids = [s.get("slide_id") for s in contract.get("slides") or []]
    titles = {s.get("slide_id"): s.get("title") for s in contract.get("slides") or []}
    qc = json.loads((deck / "assets/generated/visual_llm_qc_report.json").read_text())
    qc_by_number = {}
    for row in qc.get("slides") or []:
        qc_by_number[row.get("slide_number")] = row
    spec = json.loads((deck / "figure_spec_receipt.json").read_text())
    critic_by_slide = {}
    for row in spec.get("specs") or []:
        c = row.get("critic") or {}
        sidecar = deck / "figure_specs" / f"{row.get('figure_id')}.critic.json"
        if not c and sidecar.is_file():
            c = json.loads(sidecar.read_text())
        critic_by_slide[row["slide_id"]] = {
            "archetype": row.get("archetype"),
            "resolution": c.get("operator_resolution"),
            "unresolved": [] if c.get("operator_resolution") else (c.get("unresolved_major") or []),
            "calls": c.get("critic_calls"),
            "reauthored": c.get("reauthored"),
        }
    shots = sorted((deck / "assets/generated/qc/screenshots").glob("slide_*.png"))

    cards = []
    n_vfail = n_open = 0
    for shot in shots:
        number = int(shot.stem.split("_")[1])
        q = qc_by_number.get(number) or {}
        sid = slide_ids[number - 1] if number - 1 < len(slide_ids) else None
        cr = critic_by_slide.get(sid) or {}
        status = str(q.get("status") or "")
        vfail = status in {"fail", "failed"}
        n_vfail += vfail
        n_open += bool(cr.get("unresolved"))
        issues = "".join(
            f"<li><b>{html.escape(str(i.get('severity')))}</b> {html.escape(str(i.get('description')))}</li>"
            for i in q.get("issues") or []
        )
        crit = "".join(
            f"<li><b>{html.escape(str(i.get('code')))}</b> {html.escape(str(i.get('detail')))}"
            f"<br><i>suggested:</i> {html.escape(str(i.get('suggested_change')))}</li>"
            for i in cr.get("unresolved") or []
        )
        img = base64.b64encode(shot.read_bytes()).decode()
        badge = "fail" if vfail else ("warn" if status == "warn" else "ok")
        cards.append(
            f"<section class='card {badge}'><h2>Slide {number}: {html.escape(str(titles.get(sid) or ''))}</h2>"
            f"<img src='data:image/png;base64,{img}' alt='slide {number}'>"
            f"<div class='notes'><p class='meta'>vision check: <b>{html.escape(status or 'n/a')}</b>"
            + (f" &middot; figure type: {html.escape(str(cr.get('archetype')))} &middot; critic rounds: {cr.get('calls')}"
               f"{' (rewritten)' if cr.get('reauthored') else ''}" if cr else "")
            + "</p>"
            + (f"<h3>Vision findings</h3><ul>{issues}</ul>" if issues else "")
            + (f"<h3>Figure check: major issues left after 3 rounds</h3><ul>{crit}</ul>" if crit else "")
            + (f"<p class='meta'>operator: <b>{html.escape(str(cr['resolution'].get('outcome')))}</b> &mdash; {html.escape(str(cr['resolution'].get('reason')))}</p>" if cr.get("resolution") else "")
            + "</div></section>"
        )

    title = contract.get("audience_title") or contract.get("title") or deck.name
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Canary review: {html.escape(deck.name)}</title>
    <style>
    :root{{--bg:#f6f6f4;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--fail:#b3261e;--warn:#9a6700;--ok:#2e7d32;--line:#e2e2dd}}
    @media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#ececea;--muted:#a3a39d;--line:#33332f}}}}
    body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}}
    main{{max-width:1100px;margin:0 auto;padding:16px}}
    .card{{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--ok);border-radius:8px;margin:18px 0;padding:14px}}
    .card.fail{{border-left-color:var(--fail)}} .card.warn{{border-left-color:var(--warn)}}
    img{{width:100%;height:auto;border:1px solid var(--line);border-radius:4px}}
    h1{{font-size:22px}} h2{{font-size:17px;margin:0 0 10px}} h3{{font-size:14px;margin:12px 0 4px}}
    .meta{{color:var(--muted);margin:8px 0 0}} li{{margin:4px 0}}
    </style></head><body><main>
    <h1>Canary review &mdash; {html.escape(str(title))}</h1>
    <p>{len(shots)} slides &middot; {n_vfail} fail the vision check &middot; {n_open} figures still carry a major figure-check issue after 3 rounds.
    Decide per slide: accept, fix a named item, or replace the figure with a text layout.</p>
    {''.join(cards)}
    </main></body></html>"""
    out.write_text(page)
    return {"slides": len(shots), "vision_fails": n_vfail, "open_figure_findings": n_open}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="default: <deck-dir>/figure_review_page.html")
    args = parser.parse_args(argv)
    out = args.output or (args.deck_dir / "figure_review_page.html")
    summary = build(args.deck_dir, out)
    print(json.dumps({"output": str(out), **summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
