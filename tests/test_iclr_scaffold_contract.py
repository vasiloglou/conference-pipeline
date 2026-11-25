"""Phase 26 ICLR presentation scaffold contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.mark.repo_data("conferences/iclr2025/decks/presentation_manifest.json")
def test_backfill_iclr_scaffold_records_all_missing_for_empty_canonical_topic():
    manifest_path = Path("conferences/iclr2025/decks/presentation_manifest.json")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["presentations"]) == 1
    entry = payload["presentations"][0]
    if entry["id"] != "01_iclr_2025_proof":
        pytest.skip("ICLR scaffold was replaced by a production presentation manifest")
    assert entry["id"] == "01_iclr_2025_proof"
    assert entry["presentation_type"] == "topic-deep-dive"
    assert {state["status"] for state in entry["stage_states"].values()} == {"missing"}
    assert entry["source_analysis_artifacts"][0]["path"] == "analysis/01_iclr_2025_proof.md"
    deck_files = {
        path.name for path in Path("conferences/iclr2025/decks/01_iclr_2025_proof").iterdir()
    }
    assert deck_files == {".gitkeep"}
