#!/usr/bin/env python3
"""Generate a video from slides and audio with proper synchronization."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def render_slides_to_images(slides_md: Path, output_dir: Path) -> list[Path]:
    """Render Marp slides to PNG images using marp-cli."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use marp to export slides as PNG images
    result = subprocess.run(
        [
            "marp",
            str(slides_md),
            "--images", "png",
            "--output", str(output_dir / "slide.png"),
            "--allow-local-files",
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error rendering slides: {result.stderr}")
        raise RuntimeError(f"marp failed: {result.stderr}")

    # Find all generated slide images
    images = sorted(output_dir.glob("slide.*.png"), key=lambda p: int(p.stem.split(".")[-1]))
    print(f"Rendered {len(images)} slide images")
    return images


def create_video_from_slides(
    slide_images: list[Path],
    audio_path: Path,
    sync_data: dict,
    output_path: Path,
    fps: int = 30
) -> Path:
    """Create video by combining slide images with audio using FFmpeg."""

    # Create a temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create concat file for FFmpeg
        concat_file = tmpdir / "concat.txt"
        slide_entries = []

        for entry in sync_data["sync"]:
            slide_idx = entry["slide"] - 1  # Convert to 0-indexed
            if slide_idx < len(slide_images):
                duration = entry["duration"]
                img_path = slide_images[slide_idx].resolve()  # Use absolute path
                slide_entries.append(f"file '{img_path}'\nduration {duration}")

        # Add final slide (FFmpeg needs it for the last duration)
        if slide_entries:
            last_slide_idx = sync_data["sync"][-1]["slide"] - 1
            if last_slide_idx < len(slide_images):
                slide_entries.append(f"file '{slide_images[last_slide_idx].resolve()}'")

        concat_file.write_text("\n".join(slide_entries))

        # Create video with slides
        slides_video = tmpdir / "slides.mp4"
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-vsync", "vfr",
                "-pix_fmt", "yuv420p",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                str(slides_video)
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"FFmpeg concat error: {result.stderr}")
            print(f"Concat file content:\n{concat_file.read_text()[:2000]}")
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        # Combine with audio
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(slides_video),
                "-i", str(audio_path),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(output_path)
            ],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"FFmpeg merge error: {result.stderr}")
            raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

    print(f"Video created: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate video from slides and audio")
    parser.add_argument("slides_dir", type=Path, help="Directory containing slides.md and sync_data.json")
    parser.add_argument("podcast_dir", type=Path, help="Directory containing podcast.mp3")
    parser.add_argument("--output", "-o", type=Path, help="Output video file")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    slides_md = args.slides_dir / "slides.md"
    sync_json = args.slides_dir / "sync_data.json"
    audio_path = args.podcast_dir / "podcast.mp3"

    if not slides_md.exists():
        raise FileNotFoundError(f"slides.md not found in {args.slides_dir}")
    if not sync_json.exists():
        raise FileNotFoundError(f"sync_data.json not found in {args.slides_dir}")
    if not audio_path.exists():
        raise FileNotFoundError(f"podcast.mp3 not found in {args.podcast_dir}")

    # Load sync data
    with open(sync_json) as f:
        sync_data = json.load(f)

    print(f"Loaded sync data: {len(sync_data['sync'])} segments, {sync_data['total_duration']:.1f}s total")

    # Render slides to images
    images_dir = args.slides_dir / "slide_images"
    slide_images = render_slides_to_images(slides_md, images_dir)

    # Create video
    output_path = args.output or args.slides_dir / "presentation_video.mp4"
    create_video_from_slides(slide_images, audio_path, sync_data, output_path, args.fps)

    print(f"\nVideo generation complete!")
    print(f"Output: {output_path}")
    print(f"Duration: {sync_data['total_duration']:.1f}s")

    return output_path


if __name__ == "__main__":
    main()
