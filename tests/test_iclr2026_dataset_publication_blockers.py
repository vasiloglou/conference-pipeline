from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "conferences/iclr2026/exports/public-release/iclr2026-v1.0.0"
RECEIPTS = ROOT / "conferences/iclr2026/storage/workflow-runs"

# The public-release bundle is an export artifact that lives only in the
# checkout that built it (and on Drive); it is not tracked. In any other
# worktree these tests have nothing to inspect, and "fail" would mean nothing.


def _bundle_unreadable() -> bool:
    # ``exports`` is a symlink into the Google Drive backup leaf; a sandboxed
    # run (macOS TCC on ~/Library/CloudStorage) can stat it but not read it.
    try:
        (BUNDLE / "RELEASE.json").read_bytes()
    except PermissionError:
        return True
    except OSError:
        return False
    return False


# Only the test that reads the bundle needs it; the receipt test reads tracked files and
# runs everywhere (v6.0-Public-Release Phase 3, O26).
_requires_bundle = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="ICLR 2026 public-release bundle not present in this checkout",
)
_requires_readable_bundle = pytest.mark.skipif(
    BUNDLE.exists() and _bundle_unreadable(),
    reason="ICLR 2026 public-release bundle is a Drive symlink this environment cannot read (PermissionError)",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@_requires_bundle
@_requires_readable_bundle
def test_iclr_dataset_metadata_binds_local_release_identity():
    release = read_json(BUNDLE / "RELEASE.json")
    manifest = read_json(BUNDLE / "metadata/dataset_manifest.json")
    datapackage = read_json(BUNDLE / "metadata/datapackage.json")
    assert manifest["bundle_id"] == release["bundle_id"] == "iclr2026-v1.0.0"
    assert manifest["source_sha"] == release["source_sha"]
    assert manifest["export_manifest_sha256"] == release["export_manifest_sha256"]
    assert datapackage["bundle_id"] == release["bundle_id"]
    assert datapackage["source_sha"] == release["source_sha"]
    assert datapackage["dataset_identifier"] == "metadata/dataset_manifest.json"


@pytest.mark.repo_data("conferences/iclr2026/storage/workflow-runs/public-release-iclr2026-v1.0.0-phase35-receipt.json")
def test_iclr_hf_and_zenodo_publication_are_explicitly_non_live_or_blocked():
    aggregate = read_json(RECEIPTS / "public-release-iclr2026-v1.0.0-phase35-receipt.json")
    assert aggregate["status"] == "operator_blocked"
    assert aggregate["command_results"] == []
    assert aggregate["secret_values_recorded"] is False
    for name in (
        "hf-iclr2026-v1.0.0-dry-run-receipt.json",
        "zenodo-iclr2026-v1.0.0-sandbox-receipt.json",
    ):
        receipt = read_json(RECEIPTS / name)
        assert receipt["dry_run"] is True
        assert receipt["live"] is False
        assert receipt.get("secret_values_recorded") is False

