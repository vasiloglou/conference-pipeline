#!/usr/bin/env python3
"""Run the offline GND capability probe (v6.1 Phase 1); see conference_pipeline.ontology.gnd_probe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conference_pipeline.ontology.gnd_probe import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
