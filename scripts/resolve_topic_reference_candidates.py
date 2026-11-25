#!/usr/bin/env python3
"""Resolve topic reconciliation references with bounded LLM candidate selection."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conference_pipeline.analysis.topic_reference_candidate_resolution import (  # noqa: E402
    run_main,
)


if __name__ == "__main__":
    raise SystemExit(run_main())
