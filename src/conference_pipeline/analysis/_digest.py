"""Streaming sha256 helper for analysis manifest digests."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 65536  # 64 KiB


def sha256_file(path: Path) -> str:
    """Streaming sha256 so large fact_ranks files do not blow memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
