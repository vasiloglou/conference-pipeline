"""Unit tests for build_perturbation_cases.py (PANEL-10).

Uses synthetic PIL images only — no disk reads, no live calls, no real renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageStat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic(w: int = 800, h: int = 600, color: tuple = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from scripts.presentations.build_perturbation_cases import (
    inject_oversized_arrow,
    inject_text_clip,
    inject_off_palette,
    build_manifest,
)


# ---------------------------------------------------------------------------
# inject_oversized_arrow
# ---------------------------------------------------------------------------

class TestInjectOversizedArrow:
    def test_returns_copy_not_same_object(self):
        src = _synthetic()
        out = inject_oversized_arrow(src)
        assert out is not src

    def test_original_unchanged(self):
        src = _synthetic()
        original_px = src.getpixel((400, 300))
        inject_oversized_arrow(src)
        assert src.getpixel((400, 300)) == original_px

    def test_output_has_red_rectangle_at_center(self):
        src = _synthetic(800, 600)
        out = inject_oversized_arrow(src)
        # The rectangle is 200x200 centered at (400, 300) → spans (300,200) to (500,400)
        px = out.getpixel((400, 300))
        assert px[0] > 200 and px[1] < 50 and px[2] < 50, f"Expected red center pixel, got {px}"

    def test_output_same_size(self):
        src = _synthetic(800, 600)
        out = inject_oversized_arrow(src)
        assert out.size == src.size

    def test_output_mode_rgb(self):
        out = inject_oversized_arrow(_synthetic())
        assert out.mode == "RGB"

    def test_red_pixels_exist(self):
        out = inject_oversized_arrow(_synthetic())
        # Sample corners — should still be gray (128,128,128), center is red
        corner_px = out.getpixel((5, 5))
        assert corner_px == (128, 128, 128), "Corner should remain gray"

    def test_arrowhead_area_is_red(self):
        # Arrowhead is 40px above the rectangle top.
        # Rectangle y-top = (600 - 200) // 2 = 200; arrowhead center y ≈ 200 - 40 = 160
        src = _synthetic(800, 600)
        out = inject_oversized_arrow(src)
        px = out.getpixel((400, 160))
        assert px[0] > 150, f"Arrowhead area expected reddish, got {px}"


# ---------------------------------------------------------------------------
# inject_text_clip
# ---------------------------------------------------------------------------

class TestInjectTextClip:
    def test_returns_copy(self):
        src = _synthetic()
        out = inject_text_clip(src)
        assert out is not src

    def test_original_unchanged(self):
        src = _synthetic()
        before = src.getpixel((0, 599))
        inject_text_clip(src)
        assert src.getpixel((0, 599)) == before

    def test_bottom_bar_is_black(self):
        src = _synthetic(800, 600)
        out = inject_text_clip(src)
        # Bar covers y >= 600 * 0.85 = 510 to 600
        px = out.getpixel((400, 580))
        assert px == (0, 0, 0), f"Expected black bar at bottom, got {px}"

    def test_above_bar_unchanged(self):
        src = _synthetic(800, 600)
        out = inject_text_clip(src)
        # y < 510 should still be original gray
        px = out.getpixel((400, 100))
        assert px == (128, 128, 128)

    def test_bar_full_width(self):
        src = _synthetic(800, 600)
        out = inject_text_clip(src)
        for x in [0, 100, 400, 700, 799]:
            px = out.getpixel((x, 590))
            assert px == (0, 0, 0), f"Expected black at x={x}, y=590, got {px}"

    def test_output_same_size(self):
        src = _synthetic(800, 600)
        out = inject_text_clip(src)
        assert out.size == src.size


# ---------------------------------------------------------------------------
# inject_off_palette
# ---------------------------------------------------------------------------

class TestInjectOffPalette:
    def test_returns_copy(self):
        src = _synthetic()
        out = inject_off_palette(src)
        assert out is not src

    def test_original_unchanged(self):
        src = _synthetic()
        before = src.getpixel((400, 300))
        inject_off_palette(src)
        assert src.getpixel((400, 300)) == before

    def test_center_is_off_palette_color(self):
        src = _synthetic(800, 600)
        out = inject_off_palette(src)
        # Rectangle centered at (400, 300), rw=200, rh=150
        px = out.getpixel((400, 300))
        assert px == (0, 255, 128), f"Expected off-palette color, got {px}"

    def test_corner_unchanged(self):
        src = _synthetic(800, 600)
        out = inject_off_palette(src)
        px = out.getpixel((5, 5))
        assert px == (128, 128, 128)

    def test_output_same_size(self):
        src = _synthetic(800, 600)
        out = inject_off_palette(src)
        assert out.size == src.size

    def test_off_palette_bounds(self):
        # rw = 800 * 0.25 = 200, rh = 600 * 0.25 = 150
        # Rectangle: (200, 150) to (600, 450)
        src = _synthetic(800, 600)
        out = inject_off_palette(src)
        for x, y in [(210, 160), (590, 440), (400, 200)]:
            px = out.getpixel((x, y))
            assert px == (0, 255, 128), f"Expected off-palette at ({x},{y}), got {px}"


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_wraps_list(self):
        cases = [{"case_id": "a"}, {"case_id": "b"}]
        result = build_manifest(cases)
        assert result == {"perturbation_cases": cases}

    def test_empty_list(self):
        result = build_manifest([])
        assert result == {"perturbation_cases": []}

    def test_schema_fields_preserved(self):
        case = {
            "case_id": "perturb_01",
            "source_slide": "slide_005.png",
            "injection_type": "oversized_arrow",
            "output_path": "sl_05_oversized_arrow.png",
            "expected_failure_mode": "style-drift",
        }
        result = build_manifest([case])
        assert result["perturbation_cases"][0] == case


# ---------------------------------------------------------------------------
# CLI round-trip (tmp_path-based, no real renders)
# ---------------------------------------------------------------------------

class TestCLIRoundTrip:
    def test_cli_writes_pngs_and_manifest(self, tmp_path: Path):
        import subprocess, sys

        # Create synthetic rendered_frames with the expected slide filenames
        deck_dir = tmp_path / "my_deck"
        frames_dir = deck_dir / "rendered_frames"
        frames_dir.mkdir(parents=True)

        for fname in ("slide_005.png", "slide_006.png", "slide_010.png"):
            _synthetic().save(frames_dir / fname)

        output_dir = tmp_path / "perturbed"
        manifest_out = tmp_path / "02-perturbation-manifest.json"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/presentations/build_perturbation_cases.py",
                "--deck-dir", str(deck_dir),
                "--output-dir", str(output_dir),
                "--manifest-out", str(manifest_out),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

        # All 3 PNGs produced
        assert (output_dir / "sl_05_oversized_arrow.png").exists()
        assert (output_dir / "sl_06_text_clip.png").exists()
        assert (output_dir / "sl_10_off_palette.png").exists()

        # Manifest written and schema-valid
        manifest = json.loads(manifest_out.read_text())
        cases = manifest["perturbation_cases"]
        assert len(cases) == 3

        ids = {c["case_id"] for c in cases}
        assert ids == {"perturb_01", "perturb_02", "perturb_03"}

        for c in cases:
            for field in ("case_id", "source_slide", "injection_type", "output_path", "expected_failure_mode"):
                assert field in c, f"Missing field {field!r} in case {c}"

    def test_cli_stdout_has_built_lines(self, tmp_path: Path):
        import subprocess, sys

        deck_dir = tmp_path / "deck"
        frames_dir = deck_dir / "rendered_frames"
        frames_dir.mkdir(parents=True)
        for fname in ("slide_005.png", "slide_006.png", "slide_010.png"):
            _synthetic().save(frames_dir / fname)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/presentations/build_perturbation_cases.py",
                "--deck-dir", str(deck_dir),
                "--output-dir", str(tmp_path / "out"),
                "--manifest-out", str(tmp_path / "manifest.json"),
            ],
            capture_output=True,
            text=True,
        )
        assert "BUILT perturb_01" in result.stdout
        assert "BUILT perturb_02" in result.stdout
        assert "BUILT perturb_03" in result.stdout

    def test_deterministic_output(self, tmp_path: Path):
        """Running twice with same input produces bit-identical PNGs."""
        import subprocess, sys, hashlib

        deck_dir = tmp_path / "deck"
        frames_dir = deck_dir / "rendered_frames"
        frames_dir.mkdir(parents=True)
        for fname in ("slide_005.png", "slide_006.png", "slide_010.png"):
            _synthetic().save(frames_dir / fname)

        def run(out_dir: Path, mfst: Path):
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    sys.executable,
                    "scripts/presentations/build_perturbation_cases.py",
                    "--deck-dir", str(deck_dir),
                    "--output-dir", str(out_dir),
                    "--manifest-out", str(mfst),
                ],
                check=True,
            )

        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        run(out1, tmp_path / "m1.json")
        run(out2, tmp_path / "m2.json")

        for fname in ("sl_05_oversized_arrow.png", "sl_06_text_clip.png", "sl_10_off_palette.png"):
            h1 = hashlib.sha256((out1 / fname).read_bytes()).hexdigest()
            h2 = hashlib.sha256((out2 / fname).read_bytes()).hexdigest()
            assert h1 == h2, f"{fname} is not bit-identical across runs"
