#!/usr/bin/env python3
"""Generate per-topic GenAI hero images for the executive-summary big-picture slides.

Reads executive_topic_digests.json and, for each topic, generates one
source-grounded hero image from the digest's bigpicture.image_concept via Imagen 4.
Writes a premium-image manifest that the deck generator consumes
(apply_premium_images) to swap the image into the topic's big-picture slide,
matched by executive_lesson_id "{tid}_bigpicture".

Needs GOOGLE_API_KEY (or GEMINI_API_KEY): `set -a; source <repo>/.env; set +a`.
This SPENDS on a live image provider; cost is recorded in the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from conference_pipeline.conference_identity import conference_name

DEFAULT_MODEL = "imagen-4.0-generate-001"
COST_PER_IMAGE_USD = 0.04  # Imagen 4 standard, single 1K image (estimate; recorded for the receipt)

STYLE = (
    " Editorial conceptual technology illustration. Deep navy background, teal and "
    "warm-amber accents, clean minimal composition, soft volumetric glow, abstract "
    "and metaphor-forward. ABSOLUTELY NO TEXT, no words, no letters, no labels, no "
    "typography, no numbers, no charts, no graphs, no UI panels, and no captioned "
    "diagrams anywhere in the image. 16:9 widescreen."
)
# Note: concepts that name labelled artifacts (leaderboards, token/cache tiers,
# data/recipe layers) tend to make Imagen render text even with the directive
# above; a source-fit panel + abstract regeneration is the reliable backstop.


def build_prompt(topic_title: str, concept: str, *, conference_label: str) -> str:
    return f"A hero image evoking the {conference_label} theme '{topic_title}'. Concept: {concept}.{STYLE}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True, help="e.g. conferences/cvpr2026")
    p.add_argument("--digests", default="decks/executive_topic_digests.json")
    p.add_argument("--deck-id", default="executive_summary_short")
    p.add_argument("--out-manifest", default="decks/executive_summary_genai_images_manifest.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None, help="only the first N topics (smoke runs)")
    args = p.parse_args(argv)

    from google import genai
    from google.genai import types

    ws = REPO / args.workspace
    conference_label = conference_name(ws, default="the conference")
    digests = json.loads((ws / args.digests).read_text())
    topics = digests.get("topics", [])
    if args.limit:
        topics = topics[: args.limit]

    img_dir = ws / "decks" / args.deck_id / "genai_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY/GEMINI_API_KEY required (source the repo .env first)")
    client = genai.Client(api_key=api_key)

    images: list[dict] = []
    for t in topics:
        tid = t["topic_id"]
        bp = t.get("short", {}).get("bigpicture_slide", {})
        concept = bp.get("image_concept") or bp.get("implication") or t.get("central_takeaway", "")
        prompt = build_prompt(t.get("title", tid), concept, conference_label=conference_label)
        resp = client.models.generate_images(
            model=args.model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio="16:9"),
        )
        data = resp.generated_images[0].image.image_bytes
        png = img_dir / f"{tid}.png"
        png.write_bytes(data)
        images.append({
            "presentation_id": args.deck_id,
            "path": f"decks/{args.deck_id}/genai_images/{tid}.png",
            "asset_path": f"decks/{args.deck_id}/genai_images/{tid}.png",
            "executive_lesson_id": f"{tid}_bigpicture",
            "topic_id": tid,
            "source_bound_prompt": prompt,
            "provider": "google-genai",
            "model": args.model,
            "review_status": "pending_panel_review",
        })
        print(f"  + {tid}: {len(data)} bytes", file=sys.stderr)

    manifest = {
        "schema_version": 1,
        "provider_cost_usd": round(len(images) * COST_PER_IMAGE_USD, 4),
        "cost_basis": f"{COST_PER_IMAGE_USD} USD/image x {len(images)} (estimate)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "images": images,
    }
    out = ws / args.out_manifest
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {out}: {len(images)} images, est ${manifest['provider_cost_usd']} ({args.model})",
        file=sys.stderr,
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
