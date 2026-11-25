"""Gemini TTS raw-PCM detection must not depend on the response mime type."""

from __future__ import annotations

from conference_pipeline.audio_bytes import looks_like_raw_pcm as _looks_like_raw_pcm


def test_pcm_detected_by_mime_or_by_missing_container_signature() -> None:
    silence_then_pcm = b"\x00" * 200 + b"\x01\x00\xff\xff" * 50
    assert _looks_like_raw_pcm(silence_then_pcm, None)
    assert _looks_like_raw_pcm(silence_then_pcm, "audio/mpeg")  # mislabelled stream
    assert _looks_like_raw_pcm(b"ID3\x03" + b"\x00" * 40, "audio/L16;codec=pcm;rate=24000")


def test_real_containers_are_not_treated_as_pcm() -> None:
    assert not _looks_like_raw_pcm(b"ID3\x03\x00\x00\x00\x00\x00\x00" + b"\x00" * 30, None)
    assert not _looks_like_raw_pcm(b"\xff\xfb\x90\x64" + b"\x00" * 30, None)  # MP3 frame sync
    assert not _looks_like_raw_pcm(b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav")
    assert not _looks_like_raw_pcm(b"OggS" + b"\x00" * 30, None)
    assert not _looks_like_raw_pcm(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 20, None)
