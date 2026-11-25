from __future__ import annotations
import pytest

import json
from pathlib import Path


RECEIPT = Path(__file__).resolve().parents[1] / "conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-phase34-receipt.json"


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-phase34-receipt.json")
def test_phase34_live_publication_is_fail_closed_and_secret_free():
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["status"] == "operator_blocked"
    assert payload["exit_code"] == 2
    assert payload["secret_values_recorded"] is False
    stages = {row["stage"]: row for row in payload["blocked_stages"]}
    assert {"live_github", "live_r2", "live_pages"} <= stages.keys()
    assert "ack_live_github" in stages["live_github"]["missing_approvals"]
    assert "ack_live_r2" in stages["live_r2"]["missing_approvals"]
    assert "verified_r2_live_upload_support" in stages["live_r2"]["missing_receipts"]
    assert "ack_live_pages" in stages["live_pages"]["missing_approvals"]
    assert payload["command_results"] == []

