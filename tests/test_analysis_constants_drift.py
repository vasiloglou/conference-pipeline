"""Drift guard for checked-in NeurIPS analysis topic constants."""

from __future__ import annotations

import pytest

from pathlib import Path

from conference_pipeline.analysis.constants import NEURIPS_TOPIC_IDS


@pytest.mark.repo_data("conferences/neurips2025/analysis")
def test_neurips_topic_ids_matches_disk():
    analysis_dir = Path("conferences/neurips2025/analysis")
    disk_stems = tuple(sorted(path.stem for path in analysis_dir.glob("[0-9][0-9]_*.md")))
    constant_stems = tuple(sorted(NEURIPS_TOPIC_IDS))
    added = sorted(set(disk_stems) - set(constant_stems))
    removed = sorted(set(constant_stems) - set(disk_stems))

    assert disk_stems == constant_stems, (
        "NEURIPS_TOPIC_IDS drifted from conferences/neurips2025/analysis. "
        f"added_on_disk={added}; removed_from_disk={removed}"
    )
