"""TTS segments must be written as decodable audio regardless of how the
provider labelled the response.

Live discovery (ICML 2026 topic 02): Gemini TTS returned raw 16-bit PCM with
no usable mime type, the script saved the bytes under an .mp3 name, ffprobe
reported duration 'N/A', and both video builds failed after 54 paid segments.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_looks_like_encoded_audio_recognises_containers_and_rejects_raw_pcm() -> None:
    m = _load("synthesize_audio_from_script.py")
    assert m.looks_like_encoded_audio(b"ID3\x04\x00" + b"\x00" * 20)
    assert m.looks_like_encoded_audio(b"\xff\xfb\x90\x00" + b"\x00" * 20)  # MPEG-1 layer III sync
    assert m.looks_like_encoded_audio(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    assert m.looks_like_encoded_audio(b"OggS" + b"\x00" * 12)
    assert not m.looks_like_encoded_audio(b"\x00" * 258 + b"\x12\x34" * 100)  # leading silence PCM
    assert not m.looks_like_encoded_audio(b"\x01\x02\x03\x04" * 8)
    assert not m.looks_like_encoded_audio(b"")


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_convert_pcm_to_mp3_produces_probeable_audio(tmp_path: Path) -> None:
    m = _load("synthesize_audio_from_script.py")
    pcm = b"\x00\x00" * 24000  # one second of silence, s16le mono 24 kHz
    out = m.convert_pcm_to_mp3(pcm, None, tmp_path / "segment_001_01.mp3")
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert 0.9 <= float(probe) <= 1.2
