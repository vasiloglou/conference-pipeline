from __future__ import annotations

import json
from pathlib import Path

import pytest

from conference_pipeline.publishing.byok_tools import (
    ByokToolsConfig,
    ByokToolsError,
    generate_byok_tooling,
)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(tmp_path: Path, *, include_private_chunk: bool = False) -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(root / "RELEASE.json", {"bundle_id": "bundle-test", "conference": "neurips2025", "source_sha": "abcdef"})
    write_json(root / "CHECKSUMS.json", {"decks/a.md": {"sha256": "x"}})
    (root / "index").mkdir()
    rows = [
        {
            "chunk_id": "pubchunk:1",
            "source_path": "decks/a.md",
            "content_type": "markdown",
            "text": "Alpha public conference result about retrieval.",
        }
    ]
    if include_private_chunk:
        rows.append(
            {
                "chunk_id": "pubchunk:secret",
                "source_path": "/Users/name/private.md",
                "content_type": "markdown",
                "text": "private",
            }
        )
    (root / "index" / "corpus_chunks.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    write_json(root / "index" / "index_manifest.json", {"chunk_count": len(rows)})
    return root


def test_generates_browser_byok_tool_assets(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    receipt = tmp_path / "byok-receipt.json"
    result = generate_byok_tooling(ByokToolsConfig(bundle_root=bundle, receipt_path=receipt))

    assert (bundle / "tools.html").exists()
    assert (bundle / "assets" / "byok-tools.js").exists()
    assert (bundle / "assets" / "byok-tools.css").exists()
    assert (bundle / "assets" / "byok-corpus.json").exists()
    assert result.manifest_path == bundle / "metadata" / "byok_tools_manifest.json"
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["receipt_type"] == "conference_pipeline.byok_tools"
    assert data["status"] == "succeeded"
    assert data["chunk_count"] == 1


def test_html_contains_required_tool_controls(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))
    html = (bundle / "tools.html").read_text(encoding="utf-8")

    assert 'type="password"' in html
    assert "Ask the release" in html
    assert "Deep search" in html
    assert "Open self-host kit" in html or "Self-host kit not generated yet" in html
    assert "podcastfy Edge TTS config" in html


def test_generated_browser_assets_do_not_persist_or_route_keys(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            bundle / "tools.html",
            bundle / "assets" / "byok-tools.js",
            bundle / "assets" / "byok-tools.css",
            bundle / "assets" / "byok-corpus.json",
        )
    )

    for forbidden in (
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "URLSearchParams",
        "window.location.search",
        "?key=",
        "api_key=",
        "/api/llm",
        "/api/chat",
    ):
        assert forbidden not in combined
    assert "docs/operations/public-byok-cost-safety.md" in combined


def test_private_marker_chunks_are_filtered(tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, include_private_chunk=True)
    generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))
    corpus = json.loads((bundle / "assets" / "byok-corpus.json").read_text(encoding="utf-8"))

    assert len(corpus) == 1
    assert corpus[0]["source_path"] == "decks/a.md"
    assert "/Users/" not in json.dumps(corpus)


def test_rejects_non_public_bundle_root(tmp_path: Path) -> None:
    root = tmp_path / "conferences" / "neurips2025"
    root.mkdir(parents=True)
    with pytest.raises(Exception):
        generate_byok_tooling(ByokToolsConfig(bundle_root=root))


def test_policy_scan_fails_on_forbidden_storage_pattern(tmp_path: Path, monkeypatch) -> None:
    bundle = make_bundle(tmp_path)
    from conference_pipeline.publishing import byok_tools

    monkeypatch.setattr(byok_tools, "js_text", lambda: "localStorage.setItem('x', 'y')")
    with pytest.raises(ByokToolsError):
        generate_byok_tooling(ByokToolsConfig(bundle_root=bundle))

