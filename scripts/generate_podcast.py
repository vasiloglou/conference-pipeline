#!/usr/bin/env python3
"""Generate a podcast script and audio from a text document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conference_pipeline.podcast import (
    DEFAULT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE_A,
    DEFAULT_VOICE_B,
    PodcastConfig,
    PodcastGenerator,
)
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a podcast from a text document.")
    parser.add_argument("input", help="Path to a text or markdown file.")
    parser.add_argument("--output-dir", help="Output directory for the podcast assets.")
    parser.add_argument("--title", help="Episode title (defaults to input filename).")
    parser.add_argument("--single-host", action="store_true", help="Use a single host.")
    parser.add_argument("--host-a", default="Host A", help="Primary host name.")
    parser.add_argument("--host-b", default="Host B", help="Secondary host name.")
    parser.add_argument("--voice-a", default=DEFAULT_VOICE_A, help="TTS voice for host A.")
    parser.add_argument("--voice-b", default=DEFAULT_VOICE_B, help="TTS voice for host B.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for script generation.")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL, help="TTS model for audio.")
    parser.add_argument("--target-minutes", type=int, default=8, help="Target episode length.")
    parser.add_argument("--max-source-chars", type=int, default=60000, help="Max chars fed to LLM.")
    parser.add_argument("--max-segment-chars", type=int, default=800, help="Max chars per script segment.")
    parser.add_argument("--max-tts-chars", type=int, default=900, help="Max chars per TTS call.")
    parser.add_argument("--skip-tts", action="store_true", help="Generate script only.")
    parser.add_argument("--no-merge", action="store_true", help="Skip audio concatenation.")
    parser.add_argument(
        "--intro-audio",
        help="Path to intro music (optional). Defaults to conferences/neurips2025/assets/intro.mp3 if present.",
    )
    parser.add_argument(
        "--outro-audio",
        help="Path to outro music (optional). Defaults to conferences/neurips2025/assets/outro.mp3 if present.",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable loudness normalization.")
    parser.add_argument(
        "--extensions",
        default=".md,.txt",
        help="Comma-separated file extensions when input is a directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when input is a directory.",
    )
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Override OpenAI API base URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(name="podcast")

    default_assets = Path("conferences/neurips2025/assets")
    default_intro = default_assets / "intro.mp3"
    default_outro = default_assets / "outro.mp3"

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input path not found: {input_path}")

    source_paths = collect_sources(input_path, args.extensions, args.recursive)
    if not source_paths:
        raise SystemExit(f"No source files found in: {input_path}")

    title = args.title or input_path.stem.replace("_", " ").title()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("podcasts") / input_path.stem
    )

    config = PodcastConfig(
        title=title,
        host_a=args.host_a,
        host_b=args.host_b,
        two_hosts=not args.single_host,
        target_minutes=args.target_minutes,
        model=args.model,
        tts_model=args.tts_model,
        voice_a=args.voice_a,
        voice_b=args.voice_b,
        max_source_chars=args.max_source_chars,
        max_segment_chars=args.max_segment_chars,
        max_tts_chars=args.max_tts_chars,
        output_dir=output_dir,
        source_path=source_paths[0] if len(source_paths) == 1 else None,
        source_paths=source_paths if len(source_paths) > 1 else [],
        openai_base_url=args.openai_base_url,
        intro_path=resolve_media_path(args.intro_audio, default_intro),
        outro_path=resolve_media_path(args.outro_audio, default_outro),
        normalize_audio=not args.no_normalize,
    )

    generator = PodcastGenerator(config=config)
    if len(source_paths) == 1:
        source_text = generator.read_source(source_paths[0])
    else:
        source_text = generator.read_sources(source_paths)
    script = generator.generate_script(source_text)
    outputs = generator.write_outputs(script)

    print(f"Wrote script JSON: {outputs['script_json']}")
    print(f"Wrote script Markdown: {outputs['script_markdown']}")
    print(f"Wrote show notes: {outputs['show_notes']}")

    if args.skip_tts:
        return

    segment_paths = generator.synthesize_audio(script)
    print(f"Generated {len(segment_paths)} audio segments in {output_dir / 'audio_segments'}")

    if args.no_merge:
        return

    merged = generator.assemble_audio(segment_paths)
    if merged:
        print(f"Merged podcast audio: {merged}")
    else:
        print("Audio merge skipped (ffmpeg not available).")


def collect_sources(path: Path, extensions: str, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    exts = {ext.strip().lower() for ext in extensions.split(",") if ext.strip()}
    exts = {ext if ext.startswith(".") else f".{ext}" for ext in exts}

    if recursive:
        candidates = path.rglob("*")
    else:
        candidates = path.iterdir()

    files = [p for p in candidates if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def resolve_media_path(arg: str | None, default_path: Path) -> Path | None:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise SystemExit(f"Media file not found: {path}")
        return path
    if default_path.exists():
        return default_path
    return None


if __name__ == "__main__":
    main()
