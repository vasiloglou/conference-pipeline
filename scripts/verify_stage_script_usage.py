#!/usr/bin/env python3
"""Verify plan, skill, and run evidence against the stage script registry."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.stage_script_audit import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
