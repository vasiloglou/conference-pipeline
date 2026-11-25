#!/usr/bin/env python3
"""The v6.1 Phase 3 gold lane: select, estimate, label, queue, build (see conference_pipeline.ontology.gold_run)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from conference_pipeline.ontology.gold_run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
