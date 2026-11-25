"""Synthetic shared utilities the orchestrators shadow."""

import hashlib
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_speech_lane(record):
    kind = str(record.get("kind") or "")
    if kind == "caption":
        return "caption-transcript"
    if kind == "audio":
        return "speech-transcription"
    if record.get("slides"):
        return "slide-ocr"
    return "unsupported"
