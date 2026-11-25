#!/usr/bin/env python3
"""Generate per-deck NotebookLM podcasts (Audio Overviews) for a conference workspace.

Thin CLI entry point; the implementation lives in
``conference_pipeline.podcast_notebooklm``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.podcast_notebooklm import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
