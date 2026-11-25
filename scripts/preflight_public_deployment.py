#!/usr/bin/env python3
"""Thin CLI wrapper for public deployment preflight checks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.publishing.live_deployment import preflight_main


if __name__ == "__main__":
    raise SystemExit(preflight_main())
