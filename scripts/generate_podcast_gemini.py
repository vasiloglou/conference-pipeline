#!/usr/bin/env python3
"""Generate a podcast script and audio from NeurIPS 2025 analysis topics using Gemini."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from conference_pipeline.audio_bytes import looks_like_raw_pcm
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    infer_conference_workspace,
)

try:
    from google import genai
except ImportError as exc:  # pragma: no cover - runtime guard for missing dependency
    raise SystemExit(
        "Missing dependency: google-genai. Install with `pip install google-genai`."
    ) from exc


DEFAULT_MODEL = default_model_for_provider("podcast.longform_script_generation", "gemini")
DEFAULT_AUDIO_MODEL = default_model_for_provider("podcast.tts", "gemini")
DEFAULT_VOICE_A = "Puck"
DEFAULT_VOICE_B = "Puck"
DEFAULT_OUTPUT_DIR = Path("conferences/neurips2025/audio/neurips2025_topics_01_20_gemini")
DEFAULT_MAX_TTS_CHARS = 900
DEFAULT_LOUDNORM_I = "-16"
DEFAULT_LOUDNORM_TP = "-1.5"
DEFAULT_LOUDNORM_LRA = "11"
TOPIC_RANGE = range(1, 21)


@dataclass
class ScriptConfig:
    title: str
    host_a: str
    host_b: str
    two_hosts: bool
    target_minutes: int
    max_segment_chars: int
    max_source_chars: int
    temperature: float
    max_output_tokens: int | None


@dataclass
class SourceEntry:
    path: Path
    title: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a podcast script from NeurIPS 2025 topic analyses using Gemini."
    )
    parser.add_argument(
        "--analysis-dir",
        default="conferences/neurips2025/analysis",
        help="Directory containing topic analysis markdown files.",
    )
    parser.add_argument(
        "--input",
        help="Optional path to a single markdown file to use as the only source.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the generated script.",
    )
    parser.add_argument(
        "--title",
        default="NeurIPS 2025 Topics 01-20 Podcast",
        help="Episode title.",
    )
    parser.add_argument("--host-a", default="Host A", help="Primary host name.")
    parser.add_argument("--host-b", default="Host B", help="Secondary host name.")
    parser.add_argument("--single-host", action="store_true", help="Use a single host.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name.",
    )
    parser.add_argument(
        "--api-key",
        help="Google GenAI API key (defaults to GOOGLE_API_KEY env var).",
    )
    parser.add_argument(
        "--target-minutes",
        type=int,
        default=15,
        help="Target episode length in minutes.",
    )
    parser.add_argument(
        "--max-segment-chars",
        type=int,
        default=800,
        help="Max characters per script segment.",
    )
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=1_000_000,
        help="Max characters fed to Gemini (combined sources).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum output tokens (omit to use model default).",
    )
    parser.add_argument(
        "--audio-model",
        default=DEFAULT_AUDIO_MODEL,
        help="Gemini model name for speech generation.",
    )
    parser.add_argument("--voice-a", default=DEFAULT_VOICE_A, help="Voice for host A.")
    parser.add_argument("--voice-b", default=DEFAULT_VOICE_B, help="Voice for host B.")
    parser.add_argument(
        "--max-tts-chars",
        type=int,
        default=DEFAULT_MAX_TTS_CHARS,
        help="Max characters per speech generation call.",
    )
    parser.add_argument("--skip-audio", action="store_true", help="Skip speech generation.")
    parser.add_argument("--no-merge", action="store_true", help="Skip audio concatenation.")
    parser.add_argument(
        "--intro-audio",
        help="Path to intro audio (optional).",
    )
    parser.add_argument(
        "--outro-audio",
        help="Path to outro audio (optional).",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Disable loudness normalization.")
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Write raw model response to raw_response.txt.",
    )
    parser.add_argument(
        "--max-expansion-attempts",
        type=int,
        default=2,
        help="Maximum expansion attempts if script is under target word count.",
    )
    parser.add_argument(
        "--no-expansion",
        action="store_true",
        help="Disable automatic expansion of under-length scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    # D-QUICK-260715-rz1: this script has no --workspace flag, so derive one
    # from --analysis-dir/--output-dir to avoid falling back to Path.cwd()
    # (repo root).
    inferred_workspace = infer_conference_workspace(
        args.analysis_dir
    ) or infer_conference_workspace(args.output_dir)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "generate_podcast_gemini")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_assets = Path("conferences/neurips2025/assets")
    default_intro = default_assets / "intro.mp3"
    default_outro = default_assets / "outro.mp3"

    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY not set and --api-key not provided.")

    config = ScriptConfig(
        title=args.title,
        host_a=args.host_a,
        host_b=args.host_b,
        two_hosts=not args.single_host,
        target_minutes=args.target_minutes,
        max_segment_chars=args.max_segment_chars,
        max_source_chars=args.max_source_chars,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise SystemExit(f"Input path not found: {input_path}")
        source_paths = [input_path]
        enforce_topic_order = False
    else:
        analysis_dir = Path(args.analysis_dir)
        if not analysis_dir.exists():
            raise SystemExit(f"Analysis directory not found: {analysis_dir}")
        source_paths = collect_topic_paths(analysis_dir)
        enforce_topic_order = True
    sources = load_sources(source_paths)
    prompt, truncated = build_prompt(sources, config, enforce_topic_order)

    if truncated:
        print(
            f"Warning: sources truncated to {config.max_source_chars:,} characters.",
            file=sys.stderr,
        )

    client = genai.Client(api_key=api_key)
    response = generate_content_with_retry(
        client=client,
        model=args.model,
        contents=prompt,
        config=build_generation_config(
            config=config,
            response_mime_type="application/json",
        ),
        label="script",
    )

    raw = response.text or ""
    if not raw:
        raise SystemExit("Gemini returned an empty response.")

    if args.write_raw:
        (output_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

    payload = parse_json_payload(raw)
    script = coerce_script(payload, config)

    # Check word count and expand if under target
    target_words = int(config.target_minutes * 130)
    min_words = int(target_words * 0.9)
    word_count = count_script_words(script)
    print(f"Initial script: {word_count} words (target: {target_words}, minimum: {min_words})")

    if not args.no_expansion:
        max_expansion_attempts = args.max_expansion_attempts
        expansion_attempt = 0
        while word_count < min_words and expansion_attempt < max_expansion_attempts:
            expansion_attempt += 1
            print(f"Script under target. Requesting expansion (attempt {expansion_attempt}/{max_expansion_attempts})...")
            script = expand_script(
                client=client,
                model=args.model,
                script=script,
                sources=sources,
                config=config,
                current_words=word_count,
            )
            word_count = count_script_words(script)
            print(f"After expansion: {word_count} words")

        if word_count < min_words:
            print(f"Warning: Script still under target ({word_count}/{min_words} words) after {max_expansion_attempts} expansion attempts.", file=sys.stderr)

    write_outputs(output_dir, script, sources)
    print(f"Wrote script JSON: {output_dir / 'script.json'}")
    print(f"Wrote script Markdown: {output_dir / 'script.md'}")
    print(f"Wrote show notes: {output_dir / 'show_notes.md'}")

    if args.skip_audio:
        return

    intro_path = resolve_media_path(args.intro_audio, default_intro)
    outro_path = resolve_media_path(args.outro_audio, default_outro)
    segment_paths = synthesize_audio(
        client=client,
        script=script,
        host_a=config.host_a,
        host_b=config.host_b,
        two_hosts=config.two_hosts,
        voice_a=args.voice_a,
        voice_b=args.voice_b,
        max_tts_chars=args.max_tts_chars,
        output_dir=output_dir,
        model=args.audio_model,
    )
    print(f"Generated {len(segment_paths)} audio segments in {output_dir / 'audio_segments'}")

    if args.no_merge:
        return

    merged = assemble_audio(
        segment_paths,
        output_dir=output_dir,
        intro_path=intro_path,
        outro_path=outro_path,
        normalize_audio=not args.no_normalize,
        loudnorm_i=DEFAULT_LOUDNORM_I,
        loudnorm_tp=DEFAULT_LOUDNORM_TP,
        loudnorm_lra=DEFAULT_LOUDNORM_LRA,
    )
    if merged:
        print(f"Merged podcast audio: {merged}")
    else:
        print("Audio merge skipped (ffmpeg not available).")


def collect_topic_paths(analysis_dir: Path) -> list[Path]:
    missing: list[str] = []
    paths: list[Path] = []
    for idx in TOPIC_RANGE:
        matches = sorted(analysis_dir.glob(f"{idx:02d}_*.md"))
        if not matches:
            missing.append(f"{idx:02d}_*.md")
            continue
        paths.append(matches[0])
    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing topic files: {missing_text}")
    return paths


def load_sources(paths: list[Path]) -> list[SourceEntry]:
    sources: list[SourceEntry] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        title = extract_title(text, fallback=path.stem)
        sources.append(SourceEntry(path=path, title=title, text=text))
    return sources


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return fallback


def build_prompt(
    sources: list[SourceEntry],
    config: ScriptConfig,
    enforce_topic_order: bool,
) -> tuple[str, bool]:
    target_words = int(config.target_minutes * 130)
    min_words = int(target_words * 0.9)
    if config.two_hosts:
        host_line = f"Hosts: {config.host_a} and {config.host_b}."
        speaker_rule = "Use both hosts and alternate every 1-2 segments."
    else:
        host_line = f"Single host: {config.host_a}."
        speaker_rule = f"Use only {config.host_a} in every segment."

    topic_lines = []
    if enforce_topic_order:
        for idx, entry in zip(TOPIC_RANGE, sources):
            topic_lines.append(f"{idx:02d} - {entry.title} ({entry.path.name})")
        ordering_rule = "- Cover topics 01-20 in order and name each topic when transitioning."
    else:
        for entry in sources:
            topic_lines.append(f"- {entry.title} ({entry.path.name})")
        ordering_rule = "- Focus only on the provided source(s)."

    combined, truncated = build_source_text(sources, config.max_source_chars)

    prompt = f"""Create a podcast script based ONLY on the sources below.

{host_line}

CRITICAL: The script MUST contain at least {min_words} words (target: {target_words} words).
This is a {config.target_minutes}-minute podcast. Do NOT summarize aggressively.

Max segment length: {config.max_segment_chars} characters.
Tone: engaging, informative, and thorough.

Rules:
- Do not add facts not present in the sources.
- If quoting the source, quote exactly.
- Each segment should be 2-5 sentences to allow for thorough discussion.
- Cover ALL major points from the source material, not just highlights.
- Include specific examples, statistics, and details from the sources.
- Output JSON only.
- {ordering_rule}
- {speaker_rule}

Topic list:
{chr(10).join(topic_lines)}

JSON schema:
{{
  "title": "...",
  "summary": "...",
  "segments": [
    {{"speaker": "{config.host_a}", "text": "..."}},
    {{"speaker": "{config.host_b}", "text": "..."}}
  ],
  "show_notes": ["..."]
}}

SOURCES:
{combined}
"""
    return prompt, truncated


def build_generation_config(
    *,
    config: ScriptConfig,
    response_mime_type: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"temperature": config.temperature}
    if config.max_output_tokens is not None:
        payload["max_output_tokens"] = config.max_output_tokens
    if response_mime_type:
        payload["response_mime_type"] = response_mime_type
    return payload


def build_source_text(sources: list[SourceEntry], max_chars: int) -> tuple[str, bool]:
    parts: list[str] = []
    total = 0
    truncated = False
    for entry in sources:
        chunk = f"[SOURCE: {entry.path.name}]\n{entry.text}\n"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 0:
                parts.append(chunk[:remaining])
            truncated = True
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts), truncated


def parse_json_payload(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*", "", raw).strip()
        raw = raw.strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"{.*}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise SystemExit("Failed to parse JSON from Gemini response.")


def coerce_script(payload: dict[str, Any], config: ScriptConfig) -> dict[str, Any]:
    title = str(payload.get("title") or config.title).strip()
    summary = str(payload.get("summary") or "").strip()
    segments_raw = payload.get("segments", [])

    if not isinstance(segments_raw, list) or not segments_raw:
        raise SystemExit("No segments returned in script payload.")

    segments: list[dict[str, str]] = []
    for seg in segments_raw:
        speaker = str(seg.get("speaker") or config.host_a).strip()
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        segments.append({"speaker": speaker, "text": text})

    if not segments:
        raise SystemExit("All segments were empty after parsing.")

    show_notes = payload.get("show_notes", [])
    if not isinstance(show_notes, list):
        show_notes = []

    return {
        "title": title,
        "summary": summary,
        "segments": segments,
        "show_notes": show_notes,
    }


def count_script_words(script: dict[str, Any]) -> int:
    """Count total words in all script segments."""
    total = 0
    for seg in script.get("segments", []):
        text = str(seg.get("text", "")).strip()
        total += len(text.split())
    return total


def build_expansion_prompt(
    script: dict[str, Any],
    sources: list[SourceEntry],
    config: ScriptConfig,
    current_words: int,
) -> str:
    """Build a prompt to expand an under-length script."""
    target_words = int(config.target_minutes * 130)
    additional_needed = target_words - current_words

    combined, _ = build_source_text(sources, config.max_source_chars)

    current_script_json = json.dumps(script, indent=2)

    return f"""The following podcast script is too short. It has {current_words} words but needs at least {target_words} words.

Please EXPAND the script by adding {additional_needed}+ more words. You can:
1. Add more detail and examples to existing segments
2. Add new segments that cover additional points from the source material
3. Include more specific statistics, quotes, and technical details

IMPORTANT:
- Keep the same structure (title, summary, segments, show_notes)
- Do not remove any existing content, only expand
- All additions must be factual and based on the source material
- Maintain the same hosts and speaking style
- Output the COMPLETE expanded script as JSON only

Current script:
{current_script_json}

Source material for reference:
{combined}
"""


def expand_script(
    *,
    client: Any,
    model: str,
    script: dict[str, Any],
    sources: list[SourceEntry],
    config: ScriptConfig,
    current_words: int,
) -> dict[str, Any]:
    """Request an expansion of the script from Gemini."""
    prompt = build_expansion_prompt(script, sources, config, current_words)

    response = generate_content_with_retry(
        client=client,
        model=model,
        contents=prompt,
        config=build_generation_config(
            config=config,
            response_mime_type="application/json",
        ),
        label="expansion",
    )

    raw = response.text or ""
    if not raw:
        return script  # Return original if expansion fails

    try:
        payload = parse_json_payload(raw)
        return coerce_script(payload, config)
    except SystemExit:
        return script  # Return original if parsing fails


def write_outputs(output_dir: Path, script: dict[str, Any], sources: list[SourceEntry]) -> None:
    json_path = output_dir / "script.json"
    json_path.write_text(json.dumps(script, indent=2), encoding="utf-8")

    md_path = output_dir / "script.md"
    md_path.write_text(script_to_markdown(script), encoding="utf-8")

    notes_path = output_dir / "show_notes.md"
    notes_path.write_text(show_notes(script, sources), encoding="utf-8")


def script_to_markdown(script: dict[str, Any]) -> str:
    lines = [
        f"# {script.get('title', '').strip()}",
        "",
        "## Summary",
        script.get("summary", "").strip(),
        "",
        "## Script",
        "",
    ]
    for seg in script.get("segments", []):
        speaker = str(seg.get("speaker", "")).strip()
        text = str(seg.get("text", "")).strip()
        if speaker and text:
            lines.append(f"{speaker}: {text}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def show_notes(script: dict[str, Any], sources: list[SourceEntry]) -> str:
    lines = ["# Show Notes", "", "Sources:"]
    for entry in sources:
        lines.append(f"- {entry.path.as_posix()}")
    lines.append("")
    notes = script.get("show_notes") or []
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- Generated from the source documents.")
    return "\n".join(lines).strip() + "\n"


def resolve_media_path(arg: str | None, default_path: Path) -> Path | None:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise SystemExit(f"Media file not found: {path}")
        return path
    if default_path.exists():
        return default_path
    return None


def synthesize_audio(
    *,
    client: genai.Client,
    script: dict[str, Any],
    host_a: str,
    host_b: str,
    two_hosts: bool,
    voice_a: str,
    voice_b: str,
    max_tts_chars: int,
    output_dir: Path,
    model: str,
) -> list[Path]:
    audio_dir = output_dir / "audio_segments"
    audio_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    segments = script.get("segments", [])
    for idx, segment in enumerate(segments, start=1):
        speaker = str(segment.get("speaker", "")).strip()
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        voice = voice_for_speaker(
            speaker=speaker,
            host_b=host_b,
            voice_a=voice_a,
            voice_b=voice_b,
            two_hosts=two_hosts,
        )
        chunks = split_text(text, max_tts_chars)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            filename = f"segment_{idx:03d}_{chunk_idx:02d}.mp3"
            path = audio_dir / filename
            audio_bytes, mime_type = generate_speech(
                client=client,
                model=model,
                text=chunk,
                voice=voice,
            )
            output_path = write_audio_file(
                audio_bytes=audio_bytes,
                mime_type=mime_type,
                output_path=path,
            )
            outputs.append(output_path)
    return outputs


def generate_speech(
    *, client: genai.Client, model: str, text: str, voice: str
) -> tuple[bytes, str | None]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        response = generate_content_with_retry(
            client=client,
            model=model,
            contents=text,
            config={
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": voice,
                        }
                    },
                },
                "response_modalities": ["AUDIO"],
            },
            label="audio",
        )
        try:
            return extract_audio_payload(response)
        except SystemExit as exc:
            last_error = exc
            if attempt >= 3:
                break
            time.sleep(3 * attempt)
    if last_error:
        raise last_error
    raise SystemExit("Gemini response did not include audio data.")


def extract_audio_payload(response: Any) -> tuple[bytes, str | None]:
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", []) or []
        for part in parts:
            data, mime_type = extract_inline_data(part)
            if data is None:
                continue
            if mime_type and not mime_type.startswith("audio"):
                continue
            audio = coerce_bytes(data)
            if audio:
                return audio, mime_type
    raise SystemExit("Gemini response did not include audio data.")


def generate_content_with_retry(
    *,
    client: genai.Client,
    model: str,
    contents: str,
    config: dict[str, Any],
    label: str,
    max_attempts: int = 4,
    base_sleep: float = 5.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return gemini_generate_content(client, caller=__file__,
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_for = base_sleep * attempt
            print(
                f"Retrying {label} request ({attempt}/{max_attempts}) after error: {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    if last_error:
        raise last_error
    raise SystemExit(f"Failed to generate {label} response.")


def extract_inline_data(part: Any) -> tuple[Any | None, str | None]:
    if isinstance(part, dict):
        inline = part.get("inline_data") or part.get("inlineData")
        if isinstance(inline, dict):
            return inline.get("data"), inline.get("mime_type") or inline.get("mimeType")
        return None, None
    inline = getattr(part, "inline_data", None)
    if inline is None:
        return None, None
    data = getattr(inline, "data", None)
    mime_type = getattr(inline, "mime_type", None) or getattr(inline, "mimeType", None)
    return data, mime_type


def coerce_bytes(data: Any) -> bytes | None:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except Exception:
            try:
                return base64.b64decode(data)
            except Exception:
                return None
    return None


def write_audio_file(*, audio_bytes: bytes, mime_type: str | None, output_path: Path) -> Path:
    if looks_like_raw_pcm(audio_bytes, mime_type):
        return convert_pcm_to_mp3(audio_bytes, mime_type, output_path)
    output_path.write_bytes(audio_bytes)
    return output_path


def convert_pcm_to_mp3(audio_bytes: bytes, mime_type: str | None, output_path: Path) -> Path:
    rate = parse_pcm_rate(mime_type) or 24000
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-ar",
            "44100",
            "-ac",
            "1",
            str(output_path),
        ]
        subprocess.run(cmd, input=audio_bytes, check=True, capture_output=True)
        return output_path

    wav_path = output_path.with_suffix(".wav")
    write_wav(audio_bytes=audio_bytes, rate=rate, output_path=wav_path)
    return wav_path


def parse_pcm_rate(mime_type: str | None) -> int | None:
    if not mime_type:
        return None
    for part in mime_type.split(";"):
        part = part.strip()
        if part.startswith("rate="):
            value = part.split("=", 1)[1]
            try:
                return int(value)
            except ValueError:
                return None
    return None


def write_wav(*, audio_bytes: bytes, rate: int, output_path: Path) -> None:
    import wave

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(audio_bytes)


def voice_for_speaker(
    *,
    speaker: str,
    host_b: str,
    voice_a: str,
    voice_b: str,
    two_hosts: bool,
) -> str:
    if two_hosts and speaker.strip().lower() == host_b.strip().lower():
        return voice_b
    return voice_a


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current).strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[i : i + max_chars])
    return final_chunks


def assemble_audio(
    segment_paths: list[Path],
    *,
    output_dir: Path,
    intro_path: Path | None,
    outro_path: Path | None,
    normalize_audio: bool,
    loudnorm_i: str,
    loudnorm_tp: str,
    loudnorm_lra: str,
) -> Path | None:
    if not segment_paths:
        return None

    if not shutil.which("ffmpeg"):
        return None

    inputs: list[Path] = []
    if intro_path:
        inputs.append(intro_path)
    inputs.extend(segment_paths)
    if outro_path:
        inputs.append(outro_path)

    output_path = output_dir / "podcast.mp3"
    filter_inputs = "".join([f"[{idx}:a]" for idx in range(len(inputs))])
    concat_filter = f"{filter_inputs}concat=n={len(inputs)}:v=0:a=1[a]"

    if normalize_audio:
        loudnorm = f"loudnorm=I={loudnorm_i}:TP={loudnorm_tp}:LRA={loudnorm_lra}"
        filter_complex = f"{concat_filter};[a]{loudnorm}[aout]"
        map_target = "[aout]"
    else:
        filter_complex = concat_filter
        map_target = "[a]"

    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            map_target,
            "-ar",
            "44100",
            "-ac",
            "2",
            str(output_path),
        ]
    )
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


if __name__ == "__main__":
    main()
