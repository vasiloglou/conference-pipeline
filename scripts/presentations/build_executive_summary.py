#!/usr/bin/env python3
"""One-command rebuild of the digest-driven executive-summary deck(s).

This is the single automated entry point for the ICLR executive readout. It
chains the five stages that were previously run by hand:

  1. (optional) Opus topic digests   generate_executive_topic_digests.py
  2. deck-first plan from digests     build_digest_deck_plan.py
  3. deck contracts + figures         generate_executive_overview_decks.py
  4. Reveal.js HTML                   md_to_revealjs.py  (per deck in scope)
  5. PDF + frame export               export_revealjs_pdf.py (per deck in scope)

The resulting short deck is: title + central message -> 12-topic index ->
one evidence slide per topic (named sources) -> open questions -> predictions ->
further reading. Content is derived from each topic's narration by Opus
(the catalog's `opus` model); the event bookend is the source-grounded topic index, never
the unverifiable analysis-derived talk list.

Examples
--------
  # Fast rebuild reusing existing decks/executive_topic_digests.json:
  python scripts/presentations/build_executive_summary.py --scope short

  # Full rebuild including the Opus digest pass (needs ANTHROPIC_API_KEY —
  # `set -a; source <repo>/.env; set +a` first):
  python scripts/presentations/build_executive_summary.py --scope short --regenerate-digests
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
SCRIPTS = REPO / "scripts" / "presentations"
PY = sys.executable

DECK_IDS = {"short": "executive_summary_short", "long": "executive_summary_long"}



def _opus() -> str:
    """The catalog's current Opus model: this script wants the Opus tier, not a pinned generation."""
    from conference_pipeline.model_governance import model_for_alias

    return model_for_alias("opus")

def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    result = subprocess.run([PY, *cmd], cwd=str(REPO))
    if result.returncode != 0:
        raise SystemExit(f"executive-summary build failed at: {label} (exit {result.returncode})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", default="conferences/iclr2026")
    p.add_argument("--scope", choices=["short", "long", "both"], default="short")
    p.add_argument("--regenerate-digests", action="store_true",
                   help="re-run the Opus topic-digest pass (expensive; needs ANTHROPIC_API_KEY)")
    p.add_argument("--with-genai-images", action="store_true",
                   help="DEPRECATED no-op: the 2nd slide per topic is now a deterministic decision "
                        "diagram, not a GenAI image.")
    p.add_argument("--digests", default="decks/executive_topic_digests.json")
    p.add_argument("--plan", default="decks/executive_summary_digest_plan.json")
    p.add_argument("--model", default=_opus(), help="default: the catalog's current `opus` model")
    p.add_argument("--short-minutes", type=float, default=16.0)
    p.add_argument("--allow-local-links", action="store_true",
                   help="review builds only: let topic-deck and source links fall back to "
                        "workspace-local paths when decks/executive_*_link_map.json is missing")
    args = p.parse_args(argv)
    ws = args.workspace

    if args.regenerate_digests:
        run([str(SCRIPTS / "generate_executive_topic_digests.py"),
             "--workspace", ws, "--model", args.model, "--output", args.digests],
            "1/5 Opus topic digests")
    # Every slide renders as a tech_diagram, so every diagram slot must be
    # filled before the plan is built. This pass only authors what is missing,
    # so it is free when the digests already carry every spec. It used to be a
    # separate hand-run step, and CVPR 2026's first build shipped 48 empty
    # figure cards because nobody ran it.
    run([str(SCRIPTS / "add_bigpicture_diagrams.py"),
         "--workspace", ws, "--digests", args.digests, "--model", args.model],
        "1b/5 fill missing diagram specs")
    # Executive decks have no Gate A, so their claims are checked against each topic
    # deck's reviewed evidence here; ICML 2026 shipped 71 blockers (wrong credit, merged
    # works, stronger results than the topic deck claimed) past a layout-only judge.
    # Unchanged topics reuse their passing result, so rebuilds do not re-spend.
    run([str(SCRIPTS / "review_executive_claim_fidelity.py"),
         "--workspace", ws, "--digests", args.digests],
        "1c/5 executive claims vs topic evidence")

    # The short deck carries two clean technical diagrams per topic (an evidence
    # mechanism diagram + a decision diagram). GenAI heroes were dropped after a
    # review found them too cheesy/futuristic; --with-genai-images is deprecated.
    plan_cmd = [str(SCRIPTS / "build_digest_deck_plan.py"),
                "--workspace", ws, "--digests", args.digests, "--out", args.plan,
                "--short-minutes", str(args.short_minutes), "--with-bigpicture"]
    run(plan_cmd, "2/5 deck-first plan from digests")

    # Without a link map the generator links each topic deck as ../<topic>/presentation.html,
    # which the PDF export turns into a file:///Users/... path and the cross-deck link audit
    # rejects. ICML 2026 shipped both executive decks that way because this path never asked
    # for the maps, so a release build fails here instead.
    link_flags = [] if args.allow_local_links else [
        "--require-source-link-map", "--require-presentation-link-map"]
    run([str(SCRIPTS / "generate_executive_overview_decks.py"),
         "--workspace", ws, "--deck-scope", args.scope, "--deck-first-plan", args.plan,
         "--receipt-output", "decks/executive_overview_decks_receipt.json", *link_flags],
        "3/5 deck contracts + figures")

    scopes = ["short", "long"] if args.scope == "both" else [args.scope]
    for sc in scopes:
        deck_id = DECK_IDS[sc]
        d = f"decks/{deck_id}"
        run([str(SCRIPTS / "md_to_revealjs.py"),
             "--workspace", ws, "--presentation-id", deck_id,
             "--presentation-type", "executive-summary",
             "--render-spec", f"{d}/render_spec.json",
             "--output-html", f"{d}/presentation.html",
             "--slide-mapping", f"{d}/presentation_slide_mapping.json"],
            f"4/5 {sc}: Reveal.js HTML (legibility gate)")
        run([str(SCRIPTS / "export_revealjs_pdf.py"),
             "--workspace", ws, "--presentation-id", deck_id,
             "--presentation-type", "executive-summary",
             "--render-spec", f"{d}/render_spec.json",
             "--input-html", f"{d}/presentation.html",
             "--output-pdf", f"{d}/presentation.pdf",
             "--slide-mapping", f"{d}/presentation_slide_mapping.json",
             "--frame-manifest", f"{d}/render_frame_manifest.json",
             "--frames-dir", f"{d}/render_frames",
             "--contact-sheet", f"{d}/render_contact_sheet.png",
             "--receipt-output", f"{d}/render_export_receipt.json"],
            f"5/5 {sc}: PDF + frame export")
        # The presenter-video build reads slide_frame_manifest.json; mirror the
        # export's render_frame_manifest.json so the deck is video-ready.
        deck_dir = REPO / ws / "decks" / deck_id
        shutil.copyfile(deck_dir / "render_frame_manifest.json",
                        deck_dir / "slide_frame_manifest.json")

    print(f"\n✓ Executive summary rebuilt (scope={args.scope}).")
    for sc in scopes:
        print(f"  {sc}: {args.workspace}/decks/{DECK_IDS[sc]}/presentation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
