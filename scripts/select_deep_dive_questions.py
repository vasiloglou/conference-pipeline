#!/usr/bin/env python3
"""Select bounded deep-dive questions from exhaustive topic summaries."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conference_pipeline.research.deep_dive_lanes import select_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(select_main())
