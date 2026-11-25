#!/usr/bin/env python3
"""Generate a narrated slide video and PDF deck from a topic document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from conference_pipeline.topic_video import (
    DEFAULT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    TopicVideoBuilder,
    TopicVideoConfig,
)
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a topic video from a document.")
    parser.add_argument("input", help="Path to a topic document (markdown or text).")
    parser.add_argument("--output-dir", help="Output directory for assets.")
    parser.add_argument("--title", help="Override title.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for scene planning.")
    parser.add_argument("--tts-model", default=DEFAULT_TTS_MODEL, help="TTS model.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice for narration.")
    parser.add_argument("--voice-b", default="nova", help="Secondary voice.")
    parser.add_argument("--two-voices", action="store_true", help="Alternate two voices.")
    parser.add_argument("--max-source-chars", type=int, default=60000)
    parser.add_argument("--max-tts-chars", type=int, default=900)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-normalize", action="store_true", help="Disable loudness normalization.")
    parser.add_argument("--no-playwright", action="store_true", help="Disable HTML filler rendering.")
    parser.add_argument("--llm-scenes", action="store_true", help="Use LLM to plan scenes.")
    parser.add_argument(
        "--context-slides",
        type=int,
        default=1,
        help="Include +/- N context slides around referenced slides.",
    )
    parser.add_argument(
        "--transcript-window-sec",
        type=int,
        default=18,
        help="Transcript context window size in seconds.",
    )
    parser.add_argument("--intro-audio", help="Optional intro audio.")
    parser.add_argument("--outro-audio", help="Optional outro audio.")
    parser.add_argument("--music-bed", help="Optional background music bed.")
    parser.add_argument("--music-gain", type=float, default=0.18, help="Music bed gain.")
    parser.add_argument(
        "--openai-base-url",
        default="https://api.openai.com/v1",
        help="Override OpenAI API base URL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(name="topic_video")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("topic_videos") / input_path.stem

    intro_path = resolve_media_path(args.intro_audio)
    outro_path = resolve_media_path(args.outro_audio)
    music_bed_path = resolve_media_path(args.music_bed)

    config = TopicVideoConfig(
        title=args.title or input_path.stem.replace("_", " ").title(),
        model=args.model,
        tts_model=args.tts_model,
        voice=args.voice,
        voice_b=args.voice_b,
        two_voices=args.two_voices,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        max_source_chars=args.max_source_chars,
        max_tts_chars=args.max_tts_chars,
        normalize_audio=not args.no_normalize,
        prefer_playwright=not args.no_playwright,
        intro_path=intro_path,
        outro_path=outro_path,
        music_bed_path=music_bed_path,
        music_bed_gain=args.music_gain,
        openai_base_url=args.openai_base_url,
        use_llm_scenes=args.llm_scenes,
        context_slide_radius=args.context_slides,
        transcript_window_sec=args.transcript_window_sec,
    )

    builder = TopicVideoBuilder(config=config)
    title, evidence = builder.parse_topic_document(input_path)
    presentations_root = Path("conferences/neurips2025/decks")
    builder.map_presentations(evidence, presentations_root)

    scenes = builder.build_scenes(title, evidence)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    builder.write_scenes_json(scenes)
    builder.write_narration(scenes)

    slide_paths = builder.render_slides(evidence, scenes)
    deck_path = builder.assemble_deck_pdf(slide_paths)
    print(f"Wrote PDF deck: {deck_path}")

    audio_paths = builder.synthesize_audio(scenes)
    if len(audio_paths) != len(slide_paths):
        raise SystemExit("Audio/slide count mismatch after synthesis.")

    video_path = builder.assemble_video(slide_paths, audio_paths)
    print(f"Wrote video: {video_path}")


def resolve_media_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"Media file not found: {path}")
    return path


if __name__ == "__main__":
    main()
