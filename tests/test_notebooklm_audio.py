from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.publishing.byok_tools import ByokToolsConfig, generate_byok_tooling
from conference_pipeline.publishing.notebooklm_audio import (
    NotebookLMAudioConfig,
    NotebookLMAudioError,
    generate_notebooklm_audio,
)
from conference_pipeline.publishing.self_host_kit import SelfHostKitConfig, package_self_host_tools


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(root / "RELEASE.json", {"bundle_id": "bundle-test", "conference": "neurips2025", "source_sha": "abcdef"})
    write_json(root / "CHECKSUMS.json", {"analysis/topic.md": {"sha256": "x"}})
    (root / "index").mkdir()
    (root / "index" / "corpus_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "pubchunk:1",
                "source_path": "analysis/topic.md",
                "content_type": "markdown",
                "text": "Alpha public notebook source.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_vendor_pin_records_notebooklm_commit() -> None:
    pin = Path("vendor/notebooklm-py/PINNED_VERSION.txt").read_text(encoding="utf-8")
    readme = Path("vendor/notebooklm-py/README.md").read_text(encoding="utf-8")

    assert "https://github.com/teng-lin/notebooklm-py" in pin
    assert "1712d76cc2906d96a9386177e554529d677567a8" in pin
    for expected in ("unofficial", "Playwright", "Google account", "cookies", "not a hosted public endpoint"):
        assert expected in readme


def test_notebooklm_dry_run_receipt_and_manifest_are_gated(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "notebooklm-receipt.json"
    result = generate_notebooklm_audio(
        NotebookLMAudioConfig(bundle_root=bundle, receipt_path=receipt, dry_run=True)
    )

    assert result.planned_count == 1
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.notebooklm_audio"
    assert data["dry_run"] is True
    assert data["live"] is False
    assert data["operator_live_step_required"] is True
    assert manifest["entries"][0]["redistribution_status"] == "caution"
    assert manifest["entries"][0]["license"] == "review_required_before_publication"


def test_notebooklm_live_requires_ack_before_credentials(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    with pytest.raises(NotebookLMAudioError):
        generate_notebooklm_audio(NotebookLMAudioConfig(bundle_root=bundle, live=True))


def test_byok_tool_references_notebooklm_manifest_when_present(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    package_self_host_tools(SelfHostKitConfig(bundle_root=bundle))
    generate_notebooklm_audio(NotebookLMAudioConfig(bundle_root=bundle, dry_run=True))
    generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))
    html = (bundle / "tools.html").read_text(encoding="utf-8")
    manifest = json.loads((bundle / "metadata" / "byok_tools_manifest.json").read_text(encoding="utf-8"))

    assert "metadata/notebooklm_audio_manifest.json" in html
    assert "self-host/podcastfy_config.yaml" in html
    assert manifest["notebooklm_audio_status"] == "available"

