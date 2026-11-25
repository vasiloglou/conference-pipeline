from __future__ import annotations
import pytest

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "conferences/iclr2026/storage/workflow-runs"


def read_json(name: str) -> dict:
    return json.loads((RECEIPTS / name).read_text(encoding="utf-8"))


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-repo-iclr2026-v1.0.0-dry-run-receipt.json")
def test_iclr_public_repo_dry_run_is_clean_and_history_free():
    receipt = read_json("public-repo-iclr2026-v1.0.0-dry-run-receipt.json")
    assert receipt["status"] == "succeeded"
    assert receipt["repo_name"] == "iclr2026-data"
    assert receipt["public_safety"]["status"] == "clean"
    assert receipt["public_safety"]["source_repo_history_published"] is False
    assert receipt["secret_gate"]["status"] == "passed"
    assert receipt["secret_values_recorded"] is False


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-preflight-receipt.json")
def test_iclr_provider_preflight_is_explicitly_blocked_without_live_receipts():
    receipt = read_json("public-release-iclr2026-v1.0.0-preflight-receipt.json")
    assert receipt["status"] == "operator_blocked"
    assert receipt["exit_code"] == 2
    assert receipt["secret_values_recorded"] is False
    missing = receipt["blocked_stages"][0]["missing_receipts"]
    assert "r2_upload_receipt" in missing
    assert "r2_liveness_receipt" in missing


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-cdn-iclr2026-v1.0.0-dry-run-receipt.json")
def test_iclr_non_live_provider_receipts_are_redacted_and_targeted():
    for name in (
        "public-cdn-iclr2026-v1.0.0-dry-run-receipt.json",
        "hf-iclr2026-v1.0.0-dry-run-receipt.json",
        "zenodo-iclr2026-v1.0.0-sandbox-receipt.json",
    ):
        receipt = read_json(name)
        assert receipt["status"] == "succeeded"
        assert receipt["dry_run"] is True
        assert receipt["live"] is False
        assert receipt.get("secret_values_recorded") in {False, None}
        assert "neurips2025" not in json.dumps(receipt).lower()

