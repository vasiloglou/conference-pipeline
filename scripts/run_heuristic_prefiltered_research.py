#!/usr/bin/env python3
"""Build heuristic keyword-prefilter packs for GPT-Researcher deep dives."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conference_pipeline.research.heuristic_prefilter import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
