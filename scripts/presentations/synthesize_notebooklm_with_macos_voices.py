#!/usr/bin/env python3
"""Fill a NotebookLM video build cache with local macOS dual-voice audio."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-json", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--voice-a", default="Samantha")
    parser.add_argument("--voice-b", default="Daniel")
    parser.add_argument("--rate", type=int, default=190)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data = json.loads(args.script_json.read_text(encoding="utf-8"))
    host_a = data["hosts"]["host_a"]
    host_b = data["hosts"]["host_b"]
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    def render(segment: dict) -> Path:
        index = int(segment["index"])
        output = args.audio_dir / f"segment_{index:03d}_01.mp3"
        if output.exists() and output.stat().st_size:
            return output
        voice = args.voice_b if segment["speaker"] == host_b else args.voice_a
        with tempfile.TemporaryDirectory(prefix=f"local_tts_{index:03d}_") as tmp:
            aiff = Path(tmp) / "speech.aiff"
            subprocess.run(
                ["say", "-v", voice, "-r", str(args.rate), "-o", str(aiff), segment["text"]],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", str(aiff),
                    "-codec:a", "libmp3lame", "-b:a", "192k", str(output),
                ],
                check=True,
                capture_output=True,
            )
        return output

    segments = data["segments"]
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(render, segment) for segment in segments]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 20 == 0 or completed == len(segments):
                print(f"Synthesized {completed}/{len(segments)} local audio segments.")
    print(f"Local dual-voice cache ready: {args.audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
