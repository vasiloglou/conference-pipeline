"""Container sniffing for provider audio payloads.

Gemini TTS returns raw 16-bit PCM. Some model generations label it
``audio/L16`` / ``audio/pcm``; others leave the mime type unhelpful, so a PCM
stream written verbatim under a ``.mp3`` name only fails later, at
ffprobe/concat time, after the audio has been paid for.
Every TTS writer therefore decides by the bytes, not only by the label.
"""

from __future__ import annotations

_CONTAINER_SIGNATURES = (b"ID3", b"RIFF", b"OggS", b"fLaC")
DEFAULT_PCM_MIME = "audio/L16;rate=24000"


def looks_like_raw_pcm(audio_bytes: bytes, mime_type: str | None) -> bool:
    """True when ``audio_bytes`` is a raw PCM stream that must be wrapped.

    Trusts a mime type that names PCM; otherwise sniffs for a real container:
    MP3 (ID3 tag or frame sync), WAV, Ogg, FLAC, or an MP4/M4A ``ftyp`` box.
    """

    lowered = (mime_type or "").lower()
    if "pcm" in lowered or "l16" in lowered:
        return True
    head = bytes(audio_bytes[:12])
    if head.startswith(_CONTAINER_SIGNATURES) or head[4:8] == b"ftyp":
        return False
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return False
    return True
