#!/usr/bin/env python3
"""One-command build of the executive-summary videos (single-presenter + two-host).

Chains: Opus slide-matched narration -> Opus two-host dialogue -> single-presenter MP4
(Gemini TTS + ffmpeg) -> two-host NotebookLM MP4. Assumes the deck is already built
(build_executive_summary.py, which also writes slide_frame_manifest.json).

Needs ANTHROPIC_API_KEY (narration/dialogue) and GOOGLE_API_KEY/GEMINI_API_KEY (TTS):
  set -a; source <repo>/.env; set +a
  python scripts/presentations/build_executive_videos.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "presentations"
PY = sys.executable

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402


def run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:{env.get('PYTHONPATH', '')}"
    if subprocess.run([PY, *cmd], cwd=str(REPO), env=env).returncode != 0:
        raise SystemExit(f"executive-video build failed at: {label}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", required=True,
                   help="Conference workspace dir (e.g. conferences/<slug>).")
    p.add_argument("--deck-id", default="executive_summary_short")
    p.add_argument("--presentation-type", default="executive-overview")
    p.add_argument("--skip-narration", action="store_true",
                   help="reuse the narration/dialogue already in the delivery contracts")
    p.add_argument("--single-only", action="store_true")
    p.add_argument("--two-host-only", action="store_true")
    args = p.parse_args(argv)
    ws, dk = args.workspace, args.deck_id
    d = f"decks/{dk}"
    conference_label = conference_name(REPO / ws, default="the conference")

    if not args.skip_narration:
        run([str(SCRIPTS / "generate_executive_narration.py"), "--workspace", ws, "--deck-id", dk],
            "1/4 Opus slide-matched narration")
        run([str(SCRIPTS / "generate_executive_dialogue.py"), "--workspace", ws, "--deck-id", dk],
            "2/4 Opus two-host dialogue")

    common = ["--workspace", ws, "--presentation-id", dk,
              "--presentation-type", args.presentation_type,
              "--slides-html", f"{d}/presentation.html",
              "--slide-mapping", f"{d}/presentation_slide_mapping.json",
              "--blueprint", f"{d}/presentation_blueprint.json",
              "--blueprint-review", f"{d}/blueprint_review.json"]

    if not args.two_host_only:
        run([str(SCRIPTS / "build_presenter_video.py"), *common,
             "--delivery-contract", f"{d}/delivery_single_speaker.json",
             "--build-dir", f"{d}/video_build",
             "--output", f"{d}/presentation_video.mp4"],
            "3/4 single-presenter video")

    if not args.single_only:
        run([str(SCRIPTS / "build_notebooklm_style_presenter_video.py"), *common,
             "--delivery-contract", f"{d}/delivery_notebooklm.json",
             "--build-dir", f"{d}/video_build_notebooklm",
             "--output", f"{d}/presentation_video_notebooklm.mp4",
             "--host-a", "Iapetus", "--host-b", "Erinome",
             "--voice-a", "Iapetus", "--voice-b", "Erinome",
             "--conference-label", conference_label],
            "4/4 two-host NotebookLM video")

    # Slide/video sync validation is not topic-deck-specific: it compares the
    # render spec's slides against the delivery contracts and the rendered
    # MP4s, and Stage 10's done gate wants it for every deck. Runs only when
    # both videos exist, since it validates both.
    if not args.single_only and not args.two_host_only:
        run([str(SCRIPTS / "validate_video_sync.py"),
             "--workspace", ws, "--presentation-id", dk,
             "--presentation-type", args.presentation_type, "--topic", dk,
             # Absolute: validate_video_sync.py resolves these against the
             # process CWD, not --workspace, unlike the build steps above.
             "--render-spec", str(REPO / ws / d / "render_spec.json"),
             "--delivery-single-speaker", str(REPO / ws / d / "delivery_single_speaker.json"),
             "--delivery-notebooklm", str(REPO / ws / d / "delivery_notebooklm.json"),
             "--single-speaker-video", str(REPO / ws / d / "presentation_video.mp4"),
             "--notebooklm-video", str(REPO / ws / d / "presentation_video_notebooklm.mp4"),
             "--single-speaker-report", str(REPO / ws / d / "video_sync_validation_single_speaker.json"),
             "--notebooklm-report", str(REPO / ws / d / "video_sync_validation_notebooklm.json")],
            "5/5 slide/video sync validation")

    print(f"\n✓ Executive videos built under {ws}/{d}/:")
    if not args.two_host_only:
        print("  presentation_video.mp4 (single-presenter)")
    if not args.single_only:
        print("  presentation_video_notebooklm.mp4 (two-host)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
