#!/usr/bin/env python3
"""Build a workspace's ontology inputs: roster, splits, leakage report, catalog (v6.1 Phase 2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conference_pipeline.ontology.build_inputs import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
