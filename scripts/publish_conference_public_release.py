#!/usr/bin/env python3
"""Thin CLI wrapper for reusable conference public-release orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.publishing.public_release import main


if __name__ == "__main__":
    raise SystemExit(main())
