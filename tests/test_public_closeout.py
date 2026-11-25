from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.publishing.public_closeout import record_closeout, scan_public_tree


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "exports" / "public-release" / "iclr2026-v1.0.0"
    bundle.mkdir(parents=True)
    write_json(
        bundle / "RELEASE.json",
        {
            "bundle_id": "iclr2026-v1.0.0",
            "source_sha": "source-sha",
            "export_manifest_sha256": "manifest-sha",
        },
    )
    write_json(bundle / "CHECKSUMS.json", {"RELEASE.json": {"sha256": "release-sha"}})
    payload = tmp_path / "public-repo"
    payload.mkdir()
    (payload / "README.md").write_text("safe public payload\n", encoding="utf-8")
    return bundle, payload


def test_scan_public_tree_skips_binary_and_rejects_private_text(tmp_path: Path):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "artifact.bin").write_bytes(b"Google Drive\x00binary")
    (clean / "README.md").write_text("safe\n", encoding="utf-8")
    assert scan_public_tree(clean)["status"] == "clean"

    (clean / "bad.md").write_text("/Users/private\n", encoding="utf-8")
    result = scan_public_tree(clean)
    assert result["status"] == "failed"
    assert result["private_marker_count"] == 1


def test_record_closeout_is_operator_blocked_without_live_surfaces(tmp_path: Path):
    bundle, payload_root = make_bundle(tmp_path)
    liveness = tmp_path / "liveness.json"
    metadata = tmp_path / "metadata.json"
    provider = tmp_path / "provider.json"
    write_json(liveness, {"status": "failed", "checked_count": 2, "passed_count": 0, "failed_count": 2})
    write_json(metadata, {"status": "succeeded", "mismatch_count": 0})
    write_json(
        provider,
        {
            "status": "operator_blocked",
            "blocked_stages": [
                {"stage": "live_github", "missing_approvals": ["ack_live_github"], "missing_receipts": []}
            ],
        },
    )
    receipt = tmp_path / "closeout.json"
    result = record_closeout(
        bundle_root=bundle,
        public_payload_root=payload_root,
        liveness_receipt=liveness,
        metadata_receipt=metadata,
        provider_receipts=[("github", provider)],
        receipt_path=receipt,
        focused_tests=["pytest tests/test_public_closeout.py"],
    )
    assert result["status"] == "operator_blocked"
    assert result["provider_blockers"][0]["provider"] == "github"
    assert result["secret_values_recorded"] is False
    assert "/Users/" not in receipt.read_text(encoding="utf-8")
