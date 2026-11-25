#!/usr/bin/env python3
"""Render podcast script JSON to MP3 audio with provider receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.model_governance import default_model, validate_selection
from conference_pipeline.provider_receipts import (
    provider_receipt_context,
    record_provider_call,
    sanitize_for_receipt,
    utc_now,
)

DEFAULT_MODEL = default_model("podcast.tts")
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_VOICE_A = "alloy"
DEFAULT_VOICE_B = "nova"
DEFAULT_MAX_TTS_CHARS = 900
WORDS_PER_MINUTE = 150


@dataclass(frozen=True)
class RenderSegment:
    index: int
    speaker: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a podcast script.json file to audio segments and an assembled MP3."
    )
    parser.add_argument("--script-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--voice-a", default=DEFAULT_VOICE_A)
    parser.add_argument("--voice-b", default=DEFAULT_VOICE_B)
    parser.add_argument("--host-a", default="Host A")
    parser.add_argument("--host-b", default="Host B")
    parser.add_argument("--max-tts-chars", type=int, default=DEFAULT_MAX_TTS_CHARS)
    parser.add_argument("--openai-base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-id", default="podcast")
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    validate_selection("podcast.tts", provider="openai", model=args.model)

    script_path = args.script_json
    if not script_path.is_file():
        raise SystemExit(f"script JSON not found: {script_path}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    script = json.loads(script_path.read_text(encoding="utf-8"))
    segments = coerce_segments(script)
    if not segments:
        raise SystemExit("script JSON did not contain any non-empty segments")

    workspace = args.workspace.resolve() if args.workspace else None
    audio_paths: list[Path] = []
    receipts: list[dict[str, Any]] = []
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=args.run_id,
        stage_id=args.stage_id,
        caller=__file__,
    ):
        for segment in segments:
            voice = voice_for_speaker(
                segment.speaker,
                host_a=args.host_a,
                host_b=args.host_b,
                voice_a=args.voice_a,
                voice_b=args.voice_b,
            )
            for chunk_index, chunk in enumerate(
                split_text(segment.text, args.max_tts_chars),
                start=1,
            ):
                audio_path, receipt = render_chunk(
                    chunk=chunk,
                    output_dir=output_dir / "audio_segments",
                    segment_index=segment.index,
                    chunk_index=chunk_index,
                    model=args.model,
                    voice=voice,
                    api_key=api_key,
                    base_url=args.openai_base_url,
                )
                audio_paths.append(audio_path)
                receipts.append(receipt)

    podcast_path = assemble_audio(audio_paths, output_dir / "podcast.mp3")
    manifest = build_manifest(
        script_path=script_path,
        output_dir=output_dir,
        segments=segments,
        audio_paths=audio_paths,
        podcast_path=podcast_path,
        model=args.model,
        receipts=receipts,
    )
    manifest_path = output_dir / "audio_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(f"Rendered podcast audio: {podcast_path}")
    return 0


def coerce_segments(script: dict[str, Any]) -> list[RenderSegment]:
    raw_segments = script.get("segments")
    if not isinstance(raw_segments, list):
        return []
    segments: list[RenderSegment] = []
    for index, raw in enumerate(raw_segments, start=1):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        speaker = str(raw.get("speaker") or "Host A").strip() or "Host A"
        segments.append(RenderSegment(index=index, speaker=speaker, text=text))
    return segments


def voice_for_speaker(
    speaker: str,
    *,
    host_a: str,
    host_b: str,
    voice_a: str,
    voice_b: str,
) -> str:
    if speaker.strip().lower() == host_b.strip().lower():
        return voice_b
    return voice_a


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        if not sentence:
            continue
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += addition
    if current:
        chunks.append(" ".join(current).strip())

    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            final.extend(chunk[index : index + max_chars] for index in range(0, len(chunk), max_chars))
    return [chunk for chunk in final if chunk.strip()]


def render_chunk(
    *,
    chunk: str,
    output_dir: Path,
    segment_index: int,
    chunk_index: int,
    model: str,
    voice: str,
    api_key: str,
    base_url: str,
) -> tuple[Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"segment_{segment_index:03d}_{chunk_index:02d}.mp3"
    payload = {
        "model": model,
        "voice": voice,
        "input": chunk,
        "response_format": "mp3",
    }

    def invoke() -> bytes:
        response = httpx.post(
            f"{base_url.rstrip('/')}/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        return response.content

    result = record_provider_call(
        invoke,
        provider="openai",
        model=model,
        operation="tts",
        caller=__file__,
        request_kwargs=payload,
        tts_minutes=estimate_tts_minutes(chunk),
    )
    output_path.write_bytes(result.response)
    return output_path, result.receipt.to_dict()


def estimate_tts_minutes(text: str) -> float:
    words = max(1, len(text.split()))
    return round(words / WORDS_PER_MINUTE, 6)


def assemble_audio(segment_paths: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        command = ["ffmpeg", "-y"]
        for path in segment_paths:
            command.extend(["-i", str(path)])
        filter_inputs = "".join(f"[{index}:a]" for index in range(len(segment_paths)))
        command.extend(
            [
                "-filter_complex",
                f"{filter_inputs}concat=n={len(segment_paths)}:v=0:a=1[a]",
                "-map",
                "[a]",
                "-ar",
                "44100",
                "-ac",
                "2",
                str(output_path),
            ]
        )
        subprocess.run(command, check=True, capture_output=True)
    else:
        with output_path.open("wb") as merged:
            for path in segment_paths:
                merged.write(path.read_bytes())
    return output_path


def build_manifest(
    *,
    script_path: Path,
    output_dir: Path,
    segments: list[RenderSegment],
    audio_paths: list[Path],
    podcast_path: Path,
    model: str,
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    return sanitize_for_receipt(
        {
            "schema_version": 1,
            "receipt_type": "conference_pipeline.podcast_audio_manifest",
            "generated_at": utc_now(),
            "script_json_path": str(script_path),
            "script_json_sha256": sha256_file(script_path),
            "output_dir": str(output_dir),
            "podcast_path": str(podcast_path),
            "podcast_sha256": sha256_file(podcast_path),
            "podcast_byte_count": podcast_path.stat().st_size,
            "segment_count": len(segments),
            "audio_file_count": len(audio_paths),
            "audio_files": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "byte_count": path.stat().st_size,
                }
                for path in audio_paths
            ],
            "tts_model": model,
            "provider_receipt_paths": [
                receipt.get("receipt_path") for receipt in receipts if receipt.get("receipt_path")
            ],
            "provider_call_count": len(receipts),
            "estimated_cost_usd": round(
                sum(float(receipt.get("estimated_cost_usd") or 0.0) for receipt in receipts),
                6,
            ),
            "secrets_copied": False,
        }
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
