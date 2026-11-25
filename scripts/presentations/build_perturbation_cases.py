"""Build by-construction perturbation cases for PANEL-10.

Each injection mechanically corrupts a known-good accept-anchor render so that a
correct judge MUST flag it, providing a non-circular correctness signal for the
voting-rule selection in Phase 02.

CLI usage:
    python scripts/presentations/build_perturbation_cases.py \\
        --deck-dir path/to/deck \\
        --output-dir path/to/perturbation_cases \\
        --manifest-out path/to/02-perturbation-manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Injection functions — all non-destructive (operate on a copy)
# ---------------------------------------------------------------------------

def inject_oversized_arrow(img: Image.Image) -> Image.Image:
    """Draw a filled red 200×200 rectangle centered on the image plus an upward arrowhead."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy = w // 2, h // 2

    rect_w, rect_h = 200, 200
    x0, y0 = cx - rect_w // 2, cy - rect_h // 2
    x1, y1 = cx + rect_w // 2, cy + rect_h // 2
    draw.rectangle([x0, y0, x1, y1], fill=(255, 0, 0))

    # Arrowhead: upward-pointing triangle, 40 px above rectangle top
    tip_y = y0 - 40
    half = rect_w // 4
    draw.polygon([(cx, tip_y), (cx - half, y0), (cx + half, y0)], fill=(255, 0, 0))

    return out


def inject_text_clip(img: Image.Image) -> Image.Image:
    """Draw a solid black bar across the full width covering the bottom 15% of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    bar_top = int(h * 0.85)
    draw.rectangle([0, bar_top, w, h], fill=(0, 0, 0))
    return out


def inject_off_palette(img: Image.Image) -> Image.Image:
    """Flood a center rectangle with (0, 255, 128) — a colour outside the brand palette."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    cx, cy = w // 2, h // 2
    rw = int(w * 0.25)
    rh = int(h * 0.25)
    draw.rectangle([cx - rw, cy - rh, cx + rw, cy + rh], fill=(0, 255, 128))
    return out


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def build_manifest(cases: list[dict]) -> dict:
    return {"perturbation_cases": cases}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CASES: list[tuple] = [
    # (filename_pattern, slide_prefix, inject_fn, case_id, inject_name, failure_mode, failure_mode_alt)
    ("slide_005.png", "sl_05", inject_oversized_arrow, "perturb_01", "oversized_arrow", "style-drift", "legibility"),
    ("slide_006.png", "sl_06", inject_text_clip,       "perturb_02", "text_clip",       "legibility",   None),
    ("slide_010.png", "sl_10", inject_off_palette,     "perturb_03", "off_palette",     "style-drift",  None),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PANEL-10 perturbation cases.")
    parser.add_argument("--deck-dir", required=True, type=Path, help="Path to deck directory (parent of rendered_frames/)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory where perturbed PNGs are written")
    parser.add_argument("--manifest-out", required=True, type=Path, help="Path for 02-perturbation-manifest.json")
    args = parser.parse_args()

    frames_dir = args.deck_dir / "rendered_frames"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict] = []

    for filename, slide_prefix, inject_fn, case_id, inject_name, failure_mode, failure_mode_alt in _CASES:
        src_path = frames_dir / filename
        src_img = Image.open(src_path).convert("RGB")
        perturbed = inject_fn(src_img)

        out_filename = f"{slide_prefix}_{inject_name}.png"
        out_path = args.output_dir / out_filename
        perturbed.save(out_path, format="PNG")

        case: dict = {
            "case_id": case_id,
            "source_slide": filename,
            "injection_type": inject_name,
            "output_path": str(out_path),
            "expected_failure_mode": failure_mode,
        }
        if failure_mode_alt is not None:
            case["expected_failure_mode_alt"] = failure_mode_alt
        manifest_cases.append(case)

        print(f"BUILT {case_id}: {out_path}")

    manifest = build_manifest(manifest_cases)

    # Prefer atomic write from the platform library; fall back to plain json.dump.
    try:
        from conference_pipeline.workflow_runs import write_json_atomic
        write_json_atomic(args.manifest_out, manifest)
    except ImportError:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w") as fh:
            json.dump(manifest, fh, indent=2)


if __name__ == "__main__":
    main()
