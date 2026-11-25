from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE = REPO_ROOT / "conferences/iclr2026/exports/public-release/iclr2026-v1.0.0"
RECEIPTS = REPO_ROOT / "conferences/iclr2026/storage/workflow-runs"

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


pytestmark = [
    pytest.mark.skipif(
        not BUNDLE.exists(),
        reason="ICLR 2026 public-release bundle not present in this checkout",
    ),
    pytest.mark.skipif(
        BUNDLE.exists() and _bundle_unreadable(),
        reason="ICLR 2026 public-release bundle is a Drive symlink this environment cannot read (PermissionError)",
    ),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_iclr_public_index_provenance_and_capabilities_are_current():
    release = read_json(BUNDLE / "RELEASE.json")
    manifest = read_json(BUNDLE / "index/index_manifest.json")
    receipt = read_json(RECEIPTS / "public-index-iclr2026-v1.0.0-receipt.json")

    assert receipt["status"] == "succeeded"
    assert manifest["bundle_id"] == "iclr2026-v1.0.0"
    assert manifest["source_sha"] == release["source_sha"]
    assert manifest["public_export_manifest_sha256"] == release["export_manifest_sha256"]
    assert manifest["chunk_count"] > 0
    assert manifest["output_checksums"]
    assert "browser_capabilities" in manifest
    assert receipt["chunk_count"] == manifest["chunk_count"]


def test_iclr_site_and_byok_self_host_metadata_bind_bundle_index_and_dataset():
    release = read_json(BUNDLE / "RELEASE.json")
    index_digest = hashlib.sha256((BUNDLE / "index/index_manifest.json").read_bytes()).hexdigest()

    site = read_json(BUNDLE / "site/site_manifest.json")
    byok = read_json(BUNDLE / "metadata/byok_tools_manifest.json")
    self_host = read_json(BUNDLE / "self-host/self_host_manifest.json")
    for manifest in (site, byok, self_host):
        assert manifest["bundle_id"] == release["bundle_id"] == "iclr2026-v1.0.0"
        assert manifest["source_sha"] == release["source_sha"]
        assert manifest["public_export_manifest_sha256"] == release["export_manifest_sha256"]
        assert manifest["index_identifier"] == "index/index_manifest.json"
        assert manifest["index_manifest_sha256"] == index_digest
        assert manifest["dataset_identifier"]

    generated = [
        path
        for root in (BUNDLE / "site", BUNDLE / "self-host", BUNDLE / "metadata")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".js", ".json", ".yaml", ".yml", ".md", ".env", ".cff"}
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in generated)
    lowered = text.lower()
    for forbidden in ("neurips2025", "neurips 2025", "google drive", "/volumes/", "openai_api_key", "anthropic_api_key"):
        assert forbidden not in lowered

