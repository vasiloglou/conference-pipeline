"""gemini-3.8-flash-tts: two-host turns go through the Interactions API; WAV segments become MP3.

Multi-speaker generate_content is refused by 3.8 ("must specify speaker names
for each part"), and 3.8 returns WAV where earlier models returned raw PCM --
written verbatim under .mp3 it broke the narration-track concat.
"""

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import synthesize_audio_from_script as tts  # noqa: E402


def test_dialogue_splits_into_speaker_turns() -> None:
    turns = tts.split_speaker_turns(
        "Opening line. Iapetus: Welcome! So what? Erinome: Fluent isn't enough. Iapetus: I like that.",
        ["Iapetus", "Erinome"],
    )
    assert turns == [
        ("Iapetus", "Opening line."),
        ("Iapetus", "Welcome! So what?"),
        ("Erinome", "Fluent isn't enough."),
        ("Iapetus", "I like that."),
    ]


def test_wav_segments_are_converted_not_written_verbatim(tmp_path: Path, monkeypatch) -> None:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x00\x01" * 100)
    seen = {}

    def fake_convert(pcm: bytes, mime: str, path: Path) -> Path:
        seen.update(pcm=pcm, mime=mime)
        return path

    monkeypatch.setattr(tts, "convert_pcm_to_mp3", fake_convert)
    out = tts._write_segment_audio(buffer.getvalue(), None, tmp_path / "s.mp3")
    assert out == tmp_path / "s.mp3"
    assert seen == {"pcm": b"\x00\x01" * 100, "mime": "audio/L16;rate=24000"}
    assert not (tmp_path / "s.mp3").exists()
