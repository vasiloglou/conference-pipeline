from __future__ import annotations
import pytest

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CLOSEOUT = ROOT / "conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-closeout-receipt.json"
PUBLIC_REPO = ROOT / "conferences/iclr2026/exports/public-release/iclr2026-v1.0.0/public-repo"


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-closeout-receipt.json")
def test_iclr_closeout_is_redacted_and_live():
    payload = json.loads(CLOSEOUT.read_text(encoding="utf-8"))

    assert payload["status"] == "succeeded"
    assert payload["secret_values_recorded"] is False
    assert payload["safety"]["status"] == "clean"
    assert payload["safety"]["private_marker_count"] == 0
    assert payload["liveness"]["checked_count"] == 37
    assert payload["liveness"]["passed_count"] == 37
    assert payload["liveness"]["failed_count"] == 0
    assert payload["metadata_agreement"]["status"] == "succeeded"
    assert payload["provider_blockers"] == []
    assert "/Users/" not in CLOSEOUT.read_text(encoding="utf-8")
    assert "Google Drive" not in CLOSEOUT.read_text(encoding="utf-8")


def test_iclr_staged_public_repo_has_no_text_private_markers():
    markers = ("file://", "/Users/", "/Volumes/", "Google Drive", "drive.google.com")
    for path in PUBLIC_REPO.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        assert not any(marker in text for marker in markers), path
