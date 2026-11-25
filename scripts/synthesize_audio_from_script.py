#!/usr/bin/env python3
"""Synthesize audio from an existing podcast script.json file."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import multiprocessing
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from conference_pipeline.audio_bytes import DEFAULT_PCM_MIME, looks_like_raw_pcm
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    gemini_interactions_create,
    infer_conference_workspace,
)

try:
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: google-genai. Install with `pip install google-genai`."
    ) from exc


DEFAULT_AUDIO_MODEL = default_model_for_provider("podcast.tts", "gemini")
DEFAULT_VOICE = "Puck"
DEFAULT_LOUDNORM_I = "-16"
DEFAULT_LOUDNORM_TP = "-1.5"
DEFAULT_LOUDNORM_LRA = "11"
REPO_ROOT = Path(__file__).resolve().parents[1]


class SegmentSynthesisTimeout(TimeoutError):
    """Raised when a single TTS segment exceeds the local watchdog timeout."""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_SMALL_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
    "13": "thirteen",
    "14": "fourteen",
    "15": "fifteen",
    "16": "sixteen",
    "17": "seventeen",
    "18": "eighteen",
    "19": "nineteen",
    "20": "twenty",
}
_TENS_WORDS = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_LARGE_NUMBER_WORDS = (
    (1_000_000_000, "billion"),
    (1_000_000, "million"),
    (1_000, "thousand"),
)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _spoken_digits(text: str) -> str:
    return " ".join(_SMALL_NUMBER_WORDS.get(ch, ch) for ch in text)


def _spoken_int(value: int) -> str:
    if value < 0:
        return f"minus {_spoken_int(abs(value))}"
    if value <= 20:
        return _SMALL_NUMBER_WORDS[str(value)]
    if value < 100:
        tens, ones = divmod(value, 10)
        spoken = _TENS_WORDS[tens * 10]
        return f"{spoken} {_spoken_int(ones)}" if ones else spoken
    if value < 1_000:
        hundreds, remainder = divmod(value, 100)
        spoken = f"{_spoken_int(hundreds)} hundred"
        return f"{spoken} {_spoken_int(remainder)}" if remainder else spoken
    for divisor, label in _LARGE_NUMBER_WORDS:
        if value >= divisor:
            whole, remainder = divmod(value, divisor)
            spoken = f"{_spoken_int(whole)} {label}"
            return f"{spoken} {_spoken_int(remainder)}" if remainder else spoken
    return str(value)


def _spoken_comma_number(match: re.Match[str]) -> str:
    return _spoken_int(int(match.group(0).replace(",", "")))


def _spoken_signed_decimal(match: re.Match[str]) -> str:
    sign = "plus" if match.group(1) == "+" else "minus"
    whole = _spoken_int(int(match.group(2)))
    fractional = _spoken_digits(match.group(3))
    return f"{sign} {whole} point {fractional}"


def _spoken_scale(match: re.Match[str]) -> str:
    whole = match.group(1)
    fractional = match.group(2)
    scale = {"K": "thousand", "M": "million", "B": "billion", "T": "trillion"}[
        match.group(3).upper()
    ]
    whole_spoken = _SMALL_NUMBER_WORDS.get(whole, whole)
    if fractional:
        return f"{whole_spoken} point {_spoken_digits(fractional)} {scale}"
    return f"{whole_spoken} {scale}"


def prepare_text_for_tts(text: str) -> str:
    """Make reviewed script text easier for TTS without changing script.json."""
    text = _normalize_space(text)
    text = re.sub(r"\b(\d+)(?:\.(\d+))?\s*([KMBT])\b", _spoken_scale, text)
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _spoken_comma_number, text)
    text = re.sub(r"(?<![\w.])([+-])\s*(\d+)\.(\d+)\b", _spoken_signed_decimal, text)
    text = text.replace("->", " to ")
    return _normalize_space(text)


def parse_speaker_voices(values: list[str]) -> list[tuple[str, str]]:
    """Parse ``NAME=VOICE`` mappings into ordered (speaker, voice) pairs."""
    pairs: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in str(value):
            continue
        name, voice = str(value).split("=", 1)
        name, voice = name.strip(), voice.strip()
        if name and voice:
            pairs.append((name, voice))
    return pairs


def segment_fingerprint(
    text: str,
    *,
    audio_model: str,
    voice: str,
    speaker_voices: list[tuple[str, str]] | None = None,
) -> str:
    """Fingerprint the exact segment inputs that make an audio file reusable."""
    payload = {
        "audio_model": audio_model,
        "text": text,
        "voice": voice,
        "speaker_voices": [list(pair) for pair in (speaker_voices or [])],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_audio_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    segments = data.get("segments")
    return segments if isinstance(segments, dict) else {}


def write_audio_manifest(path: Path, segments: dict[str, dict[str, object]]) -> None:
    payload = {
        "schema_version": 1,
        "segments": segments,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def reusable_segment_audio(
    path: Path,
    *,
    manifest_segments: dict[str, dict[str, object]],
    segment_index: int,
    fingerprint: str,
    fingerprint_aliases: list[str] | None = None,
) -> bool:
    if not path.exists():
        return False
    record = manifest_segments.get(str(segment_index))
    if not isinstance(record, dict):
        return False
    accepted_fingerprints = {fingerprint, *(fingerprint_aliases or [])}
    return (
        record.get("filename") == path.name
        and record.get("fingerprint_sha256") in accepted_fingerprints
        and record.get("byte_count") == path.stat().st_size
    )


def record_segment_audio(
    path: Path,
    *,
    manifest_segments: dict[str, dict[str, object]],
    segment_index: int,
    fingerprint: str,
) -> None:
    manifest_segments[str(segment_index)] = {
        "filename": path.name,
        "fingerprint_sha256": fingerprint,
        "byte_count": path.stat().st_size,
    }


@contextmanager
def segment_timeout(seconds: int, *, segment_index: int):
    """Bound one provider call so a hung TTS request cannot stall the stage."""
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001 - signal handler signature
        raise SegmentSynthesisTimeout(
            f"TTS segment {segment_index} exceeded {seconds}s watchdog timeout"
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _write_segment_audio(audio_bytes: bytes, mime_type: str | None, path: Path) -> Path:
    """Write one TTS segment as MP3 whatever container the provider returned.

    The governed ``gemini_tts`` catalog model returns a WAV file (RIFF header) where earlier models
    returned raw PCM; written verbatim under an .mp3 name it broke the
    copy-concat into the .m4a narration track ("pcm_s16le not supported").
    """
    if audio_bytes[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio_bytes)) as wav:
            rate = wav.getframerate()
            pcm = wav.readframes(wav.getnframes())
        return convert_pcm_to_mp3(pcm, f"audio/L16;rate={rate}", path)
    if _looks_like_raw_pcm(audio_bytes, mime_type):
        return convert_pcm_to_mp3(audio_bytes, mime_type or DEFAULT_PCM_MIME, path)
    path.write_bytes(audio_bytes)
    return path


def split_speaker_turns(text: str, speakers: list[str]) -> list[tuple[str, str]]:
    """Split ``"A: ... B: ..."`` dialogue into (speaker, text) turns.

    Text before the first speaker label belongs to the first speaker.
    """
    names = [re.escape(name) for name in speakers if name]
    pattern = re.compile(r"(?:^|(?<=\s))(" + "|".join(names) + r")\s*:\s*")
    turns: list[tuple[str, str]] = []
    current = speakers[0]
    position = 0
    for match in pattern.finditer(text):
        chunk = text[position:match.start()].strip()
        if chunk:
            turns.append((current, chunk))
        current = match.group(1)
        position = match.end()
    tail = text[position:].strip()
    if tail:
        turns.append((current, tail))
    return turns


def _synthesize_turns_via_interactions(
    client: Any,
    text: str,
    *,
    audio_model: str,
    speaker_voices: list[tuple[str, str]],
    output_path: Path,
    tts_minutes: float,
) -> Path:
    turns = split_speaker_turns(text, [name for name, _voice in speaker_voices])
    interaction = gemini_interactions_create(
        client,
        caller=__file__,
        model=audio_model,
        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": turn_text,
                        "annotations": [{"type": "speech_metadata", "speaker": speaker}],
                    }
                    for speaker, turn_text in turns
                ],
            }
        ],
        response_format={"type": "audio"},
        generation_config={
            # No "mode": the Interactions API now rejects it ("Unknown parameter
            # 'mode' at 'generation_config.structured_speech_config'", 2026-10-31)
            # and speaker-annotated turns already make the request conversational.
            "speech_config": {
                "speakers": [{"speaker": name, "voice": voice} for name, voice in speaker_voices],
            }
        },
        tts_minutes=tts_minutes,
    )
    payload = interaction.model_dump() if hasattr(interaction, "model_dump") else interaction
    for step in payload.get("steps") or []:
        for item in step.get("content") or []:
            if item.get("type") != "audio" or not item.get("data"):
                continue
            return _write_segment_audio(base64.b64decode(item["data"]), item.get("mime_type"), output_path)
    raise RuntimeError("no audio in interaction response")


def _synthesize_segment_worker(
    text: str,
    *,
    audio_model: str,
    voice: str,
    timeout_ms: int,
    output_path: str,
    result_queue: multiprocessing.Queue,
    speaker_voices: list[tuple[str, str]] | None = None,
) -> None:
    """Run one Gemini TTS call in a killable child process."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            result_queue.put({"ok": False, "error": "GOOGLE_API_KEY or GEMINI_API_KEY not set."})
            return

        client = genai.Client(
            api_key=api_key,
            http_options=genai.types.HttpOptions(timeout=max(1000, timeout_ms)),
        )
        if speaker_voices and len(speaker_voices) >= 2:
            speech_config = {
                "multi_speaker_voice_config": {
                    "speaker_voice_configs": [
                        {
                            "speaker": name,
                            "voice_config": {"prebuilt_voice_config": {"voice_name": vname}},
                        }
                        for name, vname in speaker_voices
                    ]
                }
            }
        else:
            speech_config = {"voice_config": {"prebuilt_voice_config": {"voice_name": voice}}}
        # D-12 (Pitfall 2): pass tts_minutes proxy so receipt records cost dimension
        # rather than falling back to token branch (cost_status="unknown").
        # char→minute proxy: 1000 chars ≈ 1 minute of speech at ~167 words/min avg.
        _tts_minutes_estimate = len(text) / 1000 / 3.0
        try:
            response = gemini_generate_content(
                client,
                caller=__file__,
                model=audio_model,
                contents=text,
                config={
                    "speech_config": speech_config,
                    "response_modalities": ["AUDIO"],
                },
                tts_minutes=_tts_minutes_estimate,
            )
        except Exception as exc:  # noqa: BLE001 - narrowed by message below
            # gemini-3.8-flash-tts refuses a multi-speaker generate_content
            # request ("must specify speaker names for each part", later
            # reworded to "must specify speech_metadata.speaker for each text
            # part"); it takes the Interactions API with one speaker-annotated
            # part per turn.
            message = str(exc)
            speaker_refusal = "speaker names" in message or "speech_metadata.speaker" in message
            if not (speaker_voices and len(speaker_voices) >= 2 and speaker_refusal):
                raise
            path = _synthesize_turns_via_interactions(
                client,
                text,
                audio_model=audio_model,
                speaker_voices=speaker_voices,
                output_path=Path(output_path),
                tts_minutes=_tts_minutes_estimate,
            )
            result_queue.put({"ok": True, "path": str(path)})
            return

        for candidate in response.candidates or []:
            content = candidate.content
            for part in content.parts or []:
                inline = getattr(part, "inline_data", None)
                if not inline:
                    continue
                data = getattr(inline, "data", None)
                if isinstance(data, str):
                    audio_bytes = base64.b64decode(data)
                else:
                    audio_bytes = bytes(data) if data else None
                if not audio_bytes:
                    continue
                path = Path(output_path)
                mime_type = getattr(inline, "mime_type", None)
                # Gemini TTS returns raw 16-bit PCM. The SDK does not always
                # label it (mime_type missing/None), and raw PCM written under
                # an .mp3 name is unreadable to ffprobe ("N/A" duration), which
                # broke both ICML 2026 topic-02 video builds after 54 paid
                # segments. Decide by the bytes, not only by the label.
                path = _write_segment_audio(audio_bytes, mime_type, path)
                result_queue.put({"ok": True, "path": str(path)})
                return
        result_queue.put({"ok": False, "error": "no audio in response"})
    except BaseException as exc:  # noqa: BLE001 - child must report failures to parent
        result_queue.put({"ok": False, "error": repr(exc)})


def synthesize_segment_audio(
    text: str,
    path: Path,
    *,
    audio_model: str,
    voice: str,
    timeout_ms: int,
    timeout_seconds: int,
    segment_index: int,
    speaker_voices: list[tuple[str, str]] | None = None,
) -> Path:
    """Synthesize one segment and enforce timeout by terminating a child process."""
    temp_path = path.with_name(f"{path.stem}.part{path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(
        target=_synthesize_segment_worker,
        kwargs={
            "text": text,
            "audio_model": audio_model,
            "voice": voice,
            "timeout_ms": timeout_ms,
            "output_path": str(temp_path),
            "result_queue": result_queue,
            "speaker_voices": speaker_voices,
        },
    )
    proc.start()
    proc.join(None if timeout_seconds <= 0 else timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        if temp_path.exists():
            temp_path.unlink()
        raise SegmentSynthesisTimeout(
            f"TTS segment {segment_index} exceeded {timeout_seconds}s subprocess timeout"
        )

    try:
        result = result_queue.get_nowait()
    except queue.Empty as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(
            f"TTS segment {segment_index} worker exited without returning a result "
            f"(exit code {proc.exitcode})"
        ) from exc

    if not result.get("ok"):
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(str(result.get("error", "unknown TTS worker failure")))

    produced_path = Path(result["path"])
    if not produced_path.exists():
        raise RuntimeError(f"TTS segment {segment_index} worker did not create {produced_path}")
    produced_path.replace(path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize audio from a script.json file.")
    parser.add_argument("script_json", help="Path to the script.json file.")
    parser.add_argument("--audio-model", default=DEFAULT_AUDIO_MODEL, help="Gemini TTS model.")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name for TTS.")
    parser.add_argument(
        "--speaker-voice",
        action="append",
        default=[],
        metavar="NAME=VOICE",
        help="Multi-speaker mapping, e.g. --speaker-voice Iapetus=Iapetus. Repeat per speaker; "
        "when 2+ are given, Gemini multi-speaker TTS is used and per-segment text must label turns "
        "with the speaker names.",
    )
    parser.add_argument("--no-merge", action="store_true", help="Skip merging segments.")
    parser.add_argument("--no-normalize", action="store_true", help="Skip loudness normalization.")
    parser.add_argument(
        "--segment-timeout-seconds",
        type=int,
        default=_env_int("PRESENTATION_GEMINI_TTS_SEGMENT_TIMEOUT_SECONDS", 240),
        help="Local watchdog timeout for one Gemini TTS segment; <=0 disables it.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_env_int("PRESENTATION_GEMINI_TTS_WORKERS", 1),
        help="Number of segment TTS workers. Defaults to PRESENTATION_GEMINI_TTS_WORKERS or 1.",
    )
    parser.add_argument(
        "--intro-audio",
        help="Path to intro audio.",
    )
    parser.add_argument(
        "--outro-audio",
        help="Path to outro audio.",
    )
    return parser.parse_args()


def synthesize_segment_with_retries(
    *,
    idx: int,
    segment_count: int,
    text: str,
    path: Path,
    audio_model: str,
    voice: str,
    timeout_ms: int,
    timeout_seconds: int,
    max_retries: int,
    speaker_voices: list[tuple[str, str]] | None = None,
) -> tuple[int, Path, Exception | None]:
    print(f"  [{idx}/{segment_count}] Generating {path.name}...", end=" ", flush=True)
    for attempt in range(max_retries):
        try:
            produced = synthesize_segment_audio(
                text,
                path,
                audio_model=audio_model,
                voice=voice,
                timeout_ms=timeout_ms,
                timeout_seconds=timeout_seconds,
                segment_index=idx,
                speaker_voices=speaker_voices,
            )
            print("done")
            return idx, produced, None
        except Exception as exc:  # noqa: BLE001 - report provider/chunk failures to caller
            if attempt < max_retries - 1:
                print(f"retry ({exc})", end=" ", flush=True)
                time.sleep(5 * (attempt + 1))
            else:
                print(f"failed: {exc}")
                return idx, path, exc
    return idx, path, RuntimeError("unreachable TTS retry state")


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")

    speaker_voices = parse_speaker_voices(args.speaker_voice)
    if speaker_voices and len(speaker_voices) >= 2:
        print(f"Multi-speaker TTS: {', '.join(f'{n}->{v}' for n, v in speaker_voices)}")

    script_path = Path(args.script_json)
    if not script_path.exists():
        raise SystemExit(f"Script file not found: {script_path}")

    # D-QUICK-260715-rz1: this script previously had no workspace-rooted
    # provider_receipt_context, so standalone invocations wrote TTS receipts
    # under <cwd>/storage/provider-calls/... (frequently the repo root)
    # instead of the owning conference workspace. Derive the workspace from
    # the script.json path so receipts land under conferences/<slug>/storage.
    inferred_workspace = infer_conference_workspace(script_path)
    if inferred_workspace is not None:
        os.environ.setdefault("PROVIDER_RECEIPT_WORKSPACE", str(inferred_workspace))
    os.environ.setdefault("PROVIDER_RECEIPT_STAGE_ID", "synthesize_audio_from_script")
    os.environ.setdefault("PROVIDER_RECEIPT_CALLER", __file__)

    output_dir = script_path.parent
    audio_dir = output_dir / "audio_segments"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = audio_dir / "audio_manifest.json"
    manifest_segments = load_audio_manifest(manifest_path)

    script = json.loads(script_path.read_text())

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GOOGLE_API_KEY or GEMINI_API_KEY not set.")

    timeout_ms = int(
        os.getenv(
            "PRESENTATION_GEMINI_TTS_TIMEOUT_MS",
            os.getenv("PRESENTATION_GEMINI_TIMEOUT_MS", "180000"),
        )
    )
    max_retries = int(os.getenv("PRESENTATION_GEMINI_TTS_MAX_RETRIES", "3"))
    segments = script.get("segments", [])
    print(f"Generating audio for {len(segments)} segments...")

    segment_paths_by_index: dict[int, Path] = {}
    failed_segments: list[int] = []
    pending: list[dict[str, object]] = []

    for idx, segment in enumerate(segments, start=1):
        text = prepare_text_for_tts(segment.get("text", "").strip())
        if not text:
            continue

        filename = f"segment_{idx:03d}_01.mp3"
        path = audio_dir / filename
        fingerprint = segment_fingerprint(
            text, audio_model=args.audio_model, voice=args.voice, speaker_voices=speaker_voices
        )
        fingerprint_aliases: list[str] = []
        if speaker_voices and len(speaker_voices) >= 2:
            # Gemini ignores the outer single-speaker voice when a multi-speaker
            # voice map is supplied. Older manifests may still include that
            # unused value in their cache fingerprint, so accept those aliases
            # while rewriting the manifest to the current canonical fingerprint.
            alias_voices = {DEFAULT_VOICE, args.voice}
            alias_voices.update(vname for _, vname in speaker_voices)
            fingerprint_aliases = [
                segment_fingerprint(
                    text,
                    audio_model=args.audio_model,
                    voice=alias_voice,
                    speaker_voices=speaker_voices,
                )
                for alias_voice in sorted(alias_voices)
                if alias_voice != args.voice
            ]

        if reusable_segment_audio(
            path,
            manifest_segments=manifest_segments,
            segment_index=idx,
            fingerprint=fingerprint,
            fingerprint_aliases=fingerprint_aliases,
        ):
            print(f"  [{idx}/{len(segments)}] Skipping {filename} (exists)")
            record = manifest_segments.get(str(idx), {})
            if isinstance(record, dict) and record.get("fingerprint_sha256") != fingerprint:
                record_segment_audio(
                    path,
                    manifest_segments=manifest_segments,
                    segment_index=idx,
                    fingerprint=fingerprint,
                )
                write_audio_manifest(manifest_path, manifest_segments)
            segment_paths_by_index[idx] = path
            continue
        if path.exists():
            print(f"  [{idx}/{len(segments)}] Regenerating {filename} (stale or unverified)")
            path.unlink()
        pending.append(
            {
                "idx": idx,
                "text": text,
                "path": path,
                "fingerprint": fingerprint,
            }
        )

    workers = max(1, int(args.workers or 1))
    if pending and workers == 1:
        for item in pending:
            idx, path, error = synthesize_segment_with_retries(
                idx=int(item["idx"]),
                segment_count=len(segments),
                text=str(item["text"]),
                path=item["path"],
                audio_model=args.audio_model,
                voice=args.voice,
                timeout_ms=timeout_ms,
                timeout_seconds=args.segment_timeout_seconds,
                max_retries=max_retries,
                speaker_voices=speaker_voices,
            )
            if error is None:
                segment_paths_by_index[idx] = path
                record_segment_audio(
                    path,
                    manifest_segments=manifest_segments,
                    segment_index=idx,
                    fingerprint=str(item["fingerprint"]),
                )
                write_audio_manifest(manifest_path, manifest_segments)
            else:
                failed_segments.append(idx)
            time.sleep(0.5)
    elif pending:
        print(f"Using {workers} TTS segment workers for {len(pending)} pending segments.")
        by_index = {int(item["idx"]): item for item in pending}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    synthesize_segment_with_retries,
                    idx=int(item["idx"]),
                    segment_count=len(segments),
                    text=str(item["text"]),
                    path=item["path"],
                    audio_model=args.audio_model,
                    voice=args.voice,
                    timeout_ms=timeout_ms,
                    timeout_seconds=args.segment_timeout_seconds,
                    max_retries=max_retries,
                    speaker_voices=speaker_voices,
                ): int(item["idx"])
                for item in pending
            }
            for future in as_completed(futures):
                idx, path, error = future.result()
                if error is None:
                    item = by_index[idx]
                    segment_paths_by_index[idx] = path
                    record_segment_audio(
                        path,
                        manifest_segments=manifest_segments,
                        segment_index=idx,
                        fingerprint=str(item["fingerprint"]),
                    )
                    write_audio_manifest(manifest_path, manifest_segments)
                else:
                    failed_segments.append(idx)

    segment_paths = [segment_paths_by_index[index] for index in sorted(segment_paths_by_index)]
    print(f"\nGenerated {len(segment_paths)} audio segments in {audio_dir}")
    if failed_segments:
        preview = ", ".join(str(i) for i in failed_segments[:20])
        suffix = "" if len(failed_segments) <= 20 else f", ... (+{len(failed_segments) - 20} more)"
        raise SystemExit(f"Failed to synthesize audio segment(s): {preview}{suffix}")

    if args.no_merge:
        return

    # Resolve intro/outro
    default_assets = Path("conferences/neurips2025/assets")
    intro_path = resolve_media_path(args.intro_audio, default_assets / "intro.mp3")
    outro_path = resolve_media_path(args.outro_audio, default_assets / "outro.mp3")

    merged = assemble_audio(
        segment_paths,
        output_dir=output_dir,
        intro_path=intro_path,
        outro_path=outro_path,
        normalize_audio=not args.no_normalize,
    )
    if merged:
        print(f"Merged podcast audio: {merged}")
    else:
        print("Audio merge skipped (ffmpeg not available).")


def _looks_like_raw_pcm(audio_bytes: bytes, mime_type: str | None) -> bool:
    return looks_like_raw_pcm(audio_bytes, mime_type)


def looks_like_encoded_audio(data: bytes) -> bool:
    """True for a container/codec we can hand to ffmpeg as-is (MP3, WAV, Ogg,
    FLAC, M4A); False for headerless PCM such as Gemini's ``audio/L16``.

    Kept as the module's public name for callers and tests; the container
    sniffing itself lives in ``conference_pipeline.audio_bytes`` so every TTS
    writer in the repo shares one implementation."""

    return not looks_like_raw_pcm(data, None)


def convert_pcm_to_mp3(audio_bytes: bytes, mime_type: str | None, output_path: Path) -> Path:
    rate = 24000
    if mime_type:
        for part in mime_type.split(";"):
            part = part.strip()
            if part.startswith("rate="):
                try:
                    rate = int(part.split("=", 1)[1])
                except ValueError:
                    pass

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-f", "s16le", "-ar", str(rate), "-ac", "1",
            "-i", "pipe:0", "-ar", "44100", "-ac", "1", str(output_path),
        ]
        subprocess.run(cmd, input=audio_bytes, check=True, capture_output=True)
        return output_path

    # Fallback to WAV
    import wave
    wav_path = output_path.with_suffix(".wav")
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(audio_bytes)
    return wav_path


def resolve_media_path(arg: str | None, default_path: Path) -> Path | None:
    if arg:
        path = Path(arg)
        if not path.exists():
            raise SystemExit(f"Media file not found: {path}")
        return path
    if default_path.exists():
        return default_path
    return None


def assemble_audio(
    segment_paths: list[Path],
    *,
    output_dir: Path,
    intro_path: Path | None,
    outro_path: Path | None,
    normalize_audio: bool,
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
        loudnorm = f"loudnorm=I={DEFAULT_LOUDNORM_I}:TP={DEFAULT_LOUDNORM_TP}:LRA={DEFAULT_LOUDNORM_LRA}"
        filter_complex = f"{concat_filter};[a]{loudnorm}[aout]"
        map_target = "[aout]"
    else:
        filter_complex = concat_filter
        map_target = "[a]"

    cmd = ["ffmpeg", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", map_target,
        "-ar", "44100",
        "-ac", "2",
        str(output_path),
    ])
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


if __name__ == "__main__":
    main()
