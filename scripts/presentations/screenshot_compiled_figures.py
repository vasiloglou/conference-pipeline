#!/usr/bin/env python3
"""Render a deck's compiled figure SVGs to slide-sized PNGs for the vision judge.

The judge grades pictures, not markup, so every QC round needs a raster of
each figure at the size a viewer sees. Rounds 1-4 were shot by a throwaway
snippet, which is why round 4's images were a render behind the code that
had already fixed six of its findings. This is that snippet, kept.

    python scripts/presentations/screenshot_compiled_figures.py \
        --deck-dir conferences/<c>/decks/<id> --out-dir <deck>/compiled_qc/screenshots_r5

Names are `<position>__<figure_id>.png`, zero-padded, so the vision QC's
position join lines up with the deck's slide order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SLIDE_WIDTH = 1280
SLIDE_HEIGHT = 720

PAGE = """<!doctype html><meta charset="utf-8">
<style>
  html, body {{ margin: 0; padding: 0; width: {w}px; height: {h}px;
                background: #ffffff; overflow: hidden; }}
  body {{ display: flex; align-items: center; justify-content: center; }}
  svg {{ max-width: 100%; max-height: 100%; }}
</style>
{svg}
"""


def render(svg_paths: list[Path], out_dir: Path) -> list[Path]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": SLIDE_WIDTH, "height": SLIDE_HEIGHT})
        for position, svg_path in enumerate(svg_paths, start=1):
            page.set_content(
                PAGE.format(w=SLIDE_WIDTH, h=SLIDE_HEIGHT,
                            svg=svg_path.read_text(encoding="utf-8")),
                wait_until="load",
            )
            target = out_dir / f"{position:02d}__{svg_path.stem}.png"
            page.screenshot(path=str(target))
            written.append(target)
        browser.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path,
                        help="default: <deck-dir>/compiled_qc/screenshots")
    parser.add_argument("--svg-dir", type=Path,
                        help="default: <deck-dir>/assets/compiled")
    args = parser.parse_args()

    svg_dir = args.svg_dir or args.deck_dir / "assets" / "compiled"
    out_dir = args.out_dir or args.deck_dir / "compiled_qc" / "screenshots"
    svgs = sorted(svg_dir.glob("*.svg"))
    if not svgs:
        print(f"no SVGs under {svg_dir}", file=sys.stderr)
        return 1
    written = render(svgs, out_dir)
    print(f"{len(written)} screenshots -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
