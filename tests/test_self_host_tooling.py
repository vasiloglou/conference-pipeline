from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.publishing.byok_tools import ByokToolsConfig, generate_byok_tooling
from conference_pipeline.publishing.self_host_kit import (
    SelfHostKitConfig,
    package_self_host_tools,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(root / "RELEASE.json", {"bundle_id": "bundle-test", "conference": "neurips2025", "source_sha": "abcdef"})
    write_json(root / "CHECKSUMS.json", {"decks/a.md": {"sha256": "x"}})
    (root / "index").mkdir()
    (root / "index" / "corpus_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "pubchunk:1",
                "source_path": "decks/a.md",
                "content_type": "markdown",
                "text": "Alpha public conference result.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_self_host_kit_generates_required_configs(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "self-host-receipt.json"
    result = package_self_host_tools(SelfHostKitConfig(bundle_root=bundle, receipt_path=receipt))

    assert result.kit_dir == bundle / "self-host"
    for rel in (
        "README.md",
        "gpt-researcher.env.example",
        "deck-builder-config.json",
        "report-builder-config.json",
        "podcastfy_config.yaml",
        "docker-compose.yml",
        "self_host_manifest.json",
    ):
        assert (bundle / "self-host" / rel).exists()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.self_host_tools"
    assert data["web_search_enabled"] is False


def test_self_host_configs_are_local_source_and_byok(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    package_self_host_tools(SelfHostKitConfig(bundle_root=bundle))

    env_text = (bundle / "self-host" / "gpt-researcher.env.example").read_text(encoding="utf-8")
    report = json.loads((bundle / "self-host" / "report-builder-config.json").read_text(encoding="utf-8"))
    deck = json.loads((bundle / "self-host" / "deck-builder-config.json").read_text(encoding="utf-8"))
    podcast = (bundle / "self-host" / "podcastfy_config.yaml").read_text(encoding="utf-8")

    assert "WEB_SEARCH_ENABLED=false" in env_text
    assert "report_source=local" in env_text
    assert report["WEB_SEARCH_ENABLED"] is False
    assert report["report_source"] == "local"
    assert deck["input_bundle"] == ".."
    assert deck["llm"]["mode"] == "byok-or-self-hosted"
    assert "provider: edge" in podcast
    assert "api_key_required: false" in podcast


def test_self_host_kit_has_no_private_paths(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    package_self_host_tools(SelfHostKitConfig(bundle_root=bundle))
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (bundle / "self-host").rglob("*")
        if path.is_file()
    )

    for marker in ("/Users/", "file://", "/Volumes/", "Google Drive", "drive.google.com"):
        assert marker not in combined


def test_byok_tool_links_to_generated_self_host_kit(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    package_self_host_tools(SelfHostKitConfig(bundle_root=bundle))
    generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))
    html = (bundle / "tools.html").read_text(encoding="utf-8")
    manifest = json.loads((bundle / "metadata" / "byok_tools_manifest.json").read_text(encoding="utf-8"))

    assert "self-host/README.md" in html
    assert "self-host/gpt-researcher.env.example" in html
    assert manifest["self_host_status"] == "available"

