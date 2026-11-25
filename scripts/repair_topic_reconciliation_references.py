#!/usr/bin/env python3
"""Repair topic reconciliation references with local LLM assistance."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conference_pipeline.analysis.topic_reference_repair import run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_main())
