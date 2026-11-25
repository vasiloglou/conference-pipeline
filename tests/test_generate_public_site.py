from __future__ import annotations

import json

from conference_pipeline.publishing.search_shards import read_all_rows
from pathlib import Path

from conference_pipeline.publishing.static_site import PublicSiteConfig, build_public_site


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_indexed_bundle(tmp_path: Path, *, sqlite_status: str = "unavailable") -> Path:
    root = tmp_path / "conferences" / "neurips2025" / "exports" / "public-release" / "bundle-test"
    root.mkdir(parents=True)
    write_json(
        root / "RELEASE.json",
        {
            "bundle_id": "bundle-test",
            "conference": "neurips2025",
            "source_sha": "abcdef1234567890",
            "export_manifest_sha256": "manifestdigest",
        },
    )
    (root / "decks").mkdir()
    (root / "decks" / "a.md").write_text("Alpha public text\n", encoding="utf-8")
    _write_checksums(root)
    write_json(
        root / "index" / "index_manifest.json",
        {
            "bundle_id": "bundle-test",
            "source_sha": "abcdef1234567890",
            "chunk_count": 1,
            "browser_capabilities": {
                "parquet": {"status": "written", "path": "corpus_chunks.parquet"},
                "sqlite_vec": {"status": sqlite_status, "reason": "sqlite_vec_not_installed"},
            },
        },
    )
    (root / "index" / "corpus_chunks.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "pubchunk:1",
                "source_path": "decks/a.md",
                "content_type": "markdown",
                "text": "Alpha public text",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_index_checksums(root)
    return root


def _write_checksums(root: Path) -> None:
    """A bundle's CHECKSUMS.json lists every exported file with its real sha256 (v6.0 Phase 4)."""
    import hashlib

    entries = {
        p.relative_to(root).as_posix(): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "CHECKSUMS.json" and not p.relative_to(root).as_posix().startswith("index/")
    }
    write_json(root / "CHECKSUMS.json", entries)


def _write_index_checksums(root: Path) -> None:
    """As the index stage does: index/ files are recorded in the index manifest, keyed relative to index/."""
    import hashlib

    path = root / "index" / "index_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["output_checksums"] = {
        p.relative_to(root / "index").as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root / "index").rglob("*")) if p.is_file() and p.name != "index_manifest.json"
    }
    write_json(path, manifest)


def test_site_manifest_records_release_and_index_capabilities(tmp_path: Path):
    """Working-lane fixture (sqlite_status='written' + browser_index + blob):
    semantic_search.status must be 'available' and Semantic/Both buttons must not be disabled.
    FLIPPED from 'unavailable' in 12-02 (PUBWEB-03 / D-08 runtime contract replaces build-time stub).
    """
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert (out / "index.html").exists()
    assert (out / "search.html").exists()
    assert (out / "assets" / "site.css").exists()
    assert (out / "assets" / "search.js").exists()
    assert (out / "site_manifest.json").exists()
    assert manifest["bundle_id"] == "bundle-test"
    assert manifest["source_sha"] == "abcdef1234567890"
    # v6.0 Phase 4: the modes the page offers, and semantic search gated on the browser blob.
    assert manifest["search_modes"] == ["text", "semantic", "both"]
    assert manifest["semantic_search"]["status"] == "available"
    html = (out / "search.html").read_text(encoding="utf-8")
    assert "This release has full-text search and semantic search." in html


def test_site_manifest_echoes_public_identifiers(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    release = json.loads((bundle / "RELEASE.json").read_text(encoding="utf-8"))
    release["public_repo_url"] = "https://github.com/vasiloglou/neurips2025-data"
    release["cdn_base_url"] = "https://pub-0c8355dc0a634ac6985680523ed7f40f.r2.dev/conferences/neurips2025"
    write_json(bundle / "RELEASE.json", release)
    _write_checksums(bundle)  # as the live restamp does: RELEASE.json is re-checksummed

    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=tmp_path / "site"))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    # v6.0 Phase 4: the release's public links are on the front page for readers.
    assert manifest["bundle_id"] == release["bundle_id"]
    assert f'href="{release["public_repo_url"]}"' in (tmp_path / "site" / "index.html").read_text(encoding="utf-8")


def test_generated_ui_contains_expected_public_catalog_copy(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    html = (out / "index.html").read_text(encoding="utf-8") + (out / "search.html").read_text(encoding="utf-8")

    # v6.0 Phase 4: the reader-facing copy (topics first, a data-files page, a working citation).
    for expected in ("Search this release", "Topics", "Copy citation", "Data files"):
        assert expected in html


def test_generated_site_contains_no_secret_or_private_hosted_llm_copy(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    combined = "\n".join(path.read_text(encoding="utf-8") for path in out.rglob("*") if path.is_file())

    for forbidden in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "hosted LLM",
        "maintainer key",
        "file://",
        "Google Drive",
    ):
        assert forbidden not in combined


def test_css_avoids_forbidden_visual_patterns(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    css = (out / "assets" / "site.css").read_text(encoding="utf-8")

    assert "linear-gradient" not in css
    assert "radial-gradient" not in css
    assert "letter-spacing: -" not in css
    assert "vw" not in css


def test_semantic_unavailable_copy_comes_from_manifest_capability(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path, sqlite_status="unavailable")
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    html = (out / "search.html").read_text(encoding="utf-8")

    # v6.0 Phase 4: an unavailable mode is not shown at all, and the page says so (spec 6.5).
    assert "This release has full-text search only." in html
    assert 'data-mode="semantic"' not in html and 'data-mode="both"' not in html


def test_search_assets_are_client_side_and_bundle_relative(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")
    rows = read_all_rows(out)

    assert "fetch('/api" not in js
    assert "https://api." not in js
    assert rows[0]["source_path"] == "decks/a.md"
    assert rows[0]["kind"] == "deck"
    assert rows[0]["id"] == "pubchunk:1"
    assert "/Users/" not in json.dumps(rows)


def test_site_build_clears_stale_output_and_keeps_receipt_outside_deploy_root(tmp_path: Path):
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    out.mkdir()
    (out / "public_site_receipt.json").write_text(
        '{"output_dir": "/Users/example/private/site"}\n',
        encoding="utf-8",
    )
    (out / "stale.html").write_text("stale\n", encoding="utf-8")

    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))

    assert not (out / "public_site_receipt.json").exists()
    assert not (out / "stale.html").exists()
    assert result.receipt_path == out.parent / "public_site_receipt.json"
    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in out.rglob("*")
        if path.is_file()
    )
    assert "/Users/" not in combined


# ---------------------------------------------------------------------------
# Task 1 (12-02): semantic JS module assertions
# ---------------------------------------------------------------------------

def test_emitted_js_contains_semantic_module_identifiers(tmp_path: Path):
    """search.js must contain the feature-extraction pipeline + embeddinggemma model id
    and the gemma query prompt prefix (PUBWEB-03 / D-03)."""
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")
    config = json.loads((out / "search-config.json").read_text(encoding="utf-8"))

    assert "feature-extraction" in js, "JS must call pipeline('feature-extraction', ...)"
    # v6.0 Phase 4 (R10): the model id and prompt come from the bundle, not a JS literal.
    assert "embeddinggemma" not in js
    assert "config.semantic.model" in js
    assert config["semantic"] is None  # this fixture has no browser index


def test_emitted_js_uses_pinned_cdn_not_latest(tmp_path: Path):
    """search.js must import Transformers.js at an exact pinned version (no @latest)."""
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    assert "@latest" not in js, "CDN import must not use @latest (OPFS cache stability)"
    assert "cdn.jsdelivr.net/npm/@huggingface/transformers@" in js, (
        "JS must import Transformers.js from jsDelivr with an exact pinned version"
    )


def test_emitted_js_is_client_side_key_free_extended(tmp_path: Path):
    """search.js must contain no API key strings or server-side fetch patterns (T-12-01)."""
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY" not in js
    assert "https://api." not in js
    assert "fetch('/api" not in js
    assert "/Users/" not in js


def test_emitted_js_preserves_matchrows_full_text_floor(tmp_path: Path):
    """Full-text ranking must remain in search.js as the D-08 degradation floor, reachable
    without the semantic lane (text mode + the semantic-failure fallback both use it)."""
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    assert "fullTextRanked" in js, "Full-text ranking must be preserved as the D-08 floor"
    assert "renderRanked" in js, "Full-text results must render via renderRanked"
    assert "if (mode === 'text')" in js, "Text mode must render full-text without the semantic lane"


def test_emitted_js_has_lazy_ensuremodel_and_retry_path(tmp_path: Path):
    """search.js must contain lazy ensureModel() and a visible retry/error path (D-08/D-09)."""
    bundle = make_indexed_bundle(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    assert "ensureModel" in js, "Lazy model loader must be named ensureModel (D-09)"
    assert "pipeline(" in js, "Model must use pipeline() not eager construction"
    # retry / visible-error: look for a retry UI control or retry keyword
    assert "retry" in js.lower() or "Retry" in js, (
        "JS must include a visible retry path for model/blob load failure (D-08)"
    )


# ---------------------------------------------------------------------------
# Task 2 (12-02): blob copy + browser_index echo tests (RED phase)
# ---------------------------------------------------------------------------

import struct  # noqa: E402 (local import for fixture helpers)

# One vector per indexed chunk: make_indexed_bundle writes one chunk. The blob used to
# have 3 rows, a shape the site now refuses to advertise (phase-4-r1 F7).
N_BLOB_ROWS = 1
BLOB_DIM = 768


def make_indexed_bundle_with_blob(tmp_path: Path, *, sqlite_status: str = "written") -> Path:
    """Like make_indexed_bundle but also writes a tiny embeddings_browser.f32 and
    a browser_index block in the index manifest — simulating Plan 01 output."""
    root = make_indexed_bundle(tmp_path, sqlite_status=sqlite_status)
    # Write a tiny deterministic Float32 blob: N_BLOB_ROWS × BLOB_DIM floats
    blob_bytes = struct.pack(f"{N_BLOB_ROWS * BLOB_DIM}f", *([0.5] * (N_BLOB_ROWS * BLOB_DIM)))
    (root / "index" / "embeddings_browser.f32").write_bytes(blob_bytes)
    # Re-write the index manifest to include the browser_index block
    import hashlib
    digest = hashlib.sha256(blob_bytes).hexdigest()
    index_manifest = json.loads((root / "index" / "index_manifest.json").read_text(encoding="utf-8"))
    index_manifest["browser_index"] = {
        "model": "onnx-community/embeddinggemma-300m-ONNX",
        "dim": BLOB_DIM,
        "blob": "embeddings_browser.f32",
        "chunk_order": "chunk_ids.json",
        "digest": digest,
    }
    (root / "index" / "index_manifest.json").write_text(
        json.dumps(index_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_index_checksums(root)
    return root


def test_blob_is_copied_to_site_assets_as_raw_bytes(tmp_path: Path):
    """embeddings_browser.f32 must be copied byte-identical into site/assets/ (Task 2)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))

    # Phase 7.1 (D8): the site carries int8 shards, one float32 scale per row.
    blob_out = out / "assets" / "vectors" / "0.bin"
    assert blob_out.exists(), "site/assets/vectors/0.bin must exist"
    expected_size = N_BLOB_ROWS * (BLOB_DIM + 4)
    assert blob_out.stat().st_size == expected_size, (
        f"Shard must be {expected_size} bytes ({N_BLOB_ROWS}×({BLOB_DIM}+4))"
    )


def test_blob_bytes_not_in_search_data_json(tmp_path: Path):
    """The binary blob must NOT appear in search-data.json (excluded from text-scrub)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))

    search_data = b"".join(p.read_bytes() for p in sorted((out / "search").rglob("*.json")))
    blob_bytes = (bundle / "index" / "embeddings_browser.f32").read_bytes()
    # The blob bytes (raw floats) must not appear in the search index
    assert blob_bytes[:8] not in search_data, "Blob bytes must not appear in the search index"


def test_site_manifest_has_browser_index_echo_when_present(tmp_path: Path):
    """site_manifest.json must carry a browser_index echo {model,dim,blob,digest} (Task 2)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    # v6.0 Phase 4: the site manifest records the semantic model; search-config.json carries
    # the block the JS uses (model, dim, digest).
    assert manifest["semantic_search"] == {"status": "available", "model": "onnx-community/embeddinggemma-300m-ONNX", "dim": BLOB_DIM}
    config = json.loads((out / "search-config.json").read_text(encoding="utf-8"))
    assert config["semantic"]["dim"] == BLOB_DIM and config["semantic"]["digest"]


def test_build_succeeds_without_browser_index_no_crash(tmp_path: Path):
    """When index manifest has no browser_index, build_public_site must succeed (capability-blocker degradation)."""
    # make_indexed_bundle without blob — standard fixture, sqlite_status=unavailable
    bundle = make_indexed_bundle(tmp_path, sqlite_status="unavailable")
    out = tmp_path / "site"
    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["semantic_search"]["status"] == "not_in_this_release"
    assert manifest["search_modes"] == ["text"]


# ---------------------------------------------------------------------------
# Task 3 (12-02): FLIP + EXTEND tests (RED phase)
# ---------------------------------------------------------------------------

def test_working_lane_manifest_semantic_status_available(tmp_path: Path):
    """FLIP: with sqlite_status='written' + browser_index + blob, semantic_search.status == 'available'."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    result = build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["semantic_search"]["status"] == "available", (
        "Working lane must set semantic_search.status to 'available'"
    )


def test_working_lane_buttons_are_not_disabled(tmp_path: Path):
    """FLIP: with working-lane fixture, the Semantic/Both buttons must NOT be disabled."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    html = (out / "search.html").read_text(encoding="utf-8")

    assert "This release has full-text search and semantic search." in html
    # The Semantic/Both buttons must not carry the disabled attribute
    # (they appear once each in the mode bar)
    import re
    mode_section = re.search(r'<div class="mode"[^>]*>(.*?)</div>', html, re.DOTALL)
    assert mode_section is not None
    mode_html = mode_section.group(0)
    # Disabled buttons would have the literal string 'disabled' in the mode div
    assert "disabled" not in mode_html, (
        "Semantic/Both buttons must not be disabled on the working lane"
    )


def test_search_assets_extended_client_side_and_blob_assertions(tmp_path: Path):
    """EXTEND test_search_assets_are_client_side_and_bundle_relative with Task 2 assertions."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    # JS module identifiers; the model and prompt come from search-config.json (v6.0 Phase 4)
    assert "feature-extraction" in js
    config = json.loads((out / "search-config.json").read_text(encoding="utf-8"))
    assert "embeddinggemma" in config["semantic"]["model"]
    assert "task: search result | query:" in config["semantic"]["query_prefix"]
    assert "OPENAI_API_KEY" not in js

    # Blob byte length on working-lane fixture
    blob_path = out / "assets" / "vectors" / "0.bin"
    assert blob_path.exists(), "the vector shard must be in site/assets/vectors"
    expected_size = N_BLOB_ROWS * (BLOB_DIM + 4)
    assert blob_path.stat().st_size == expected_size

    # Blob bytes must NOT appear in the search index
    search_data_bytes = b"".join(p.read_bytes() for p in sorted((out / "search").rglob("*.json")))
    blob_bytes = (bundle / "index" / "embeddings_browser.f32").read_bytes()
    assert blob_bytes[:8] not in search_data_bytes


def test_no_floating_cdn_tag_in_full_site(tmp_path: Path):
    """No floating/unpinned CDN tag must appear in any TEXT file in the site (T-12-05)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    # Skip binary .f32 — read only text files
    text_files = [p for p in out.rglob("*") if p.is_file() and p.suffix not in (".f32", ".bin")]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in text_files)
    assert "@latest" not in combined, "No floating CDN tag must appear in any site text file"


def test_generated_site_no_secret_excludes_binary_blob(tmp_path: Path):
    """The secret-scrub test must succeed even when the binary blob is present in site/."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    # Read only text files (skip .f32 binary)
    text_files = [p for p in out.rglob("*") if p.is_file() and p.suffix not in (".f32", ".bin")]
    combined = "\n".join(p.read_text(encoding="utf-8") for p in text_files)

    for forbidden in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "hosted LLM",
        "maintainer key",
        "file://",
        "Google Drive",
    ):
        assert forbidden not in combined, f"Forbidden string {forbidden!r} found in site text files"


# ---------------------------------------------------------------------------
# 12-04 gap closure: the client-side search must actually LOAD and WIRE UP, not
# merely contain the right substrings. These guard the code-review findings
# (CR-01 module load, WR-04/WR-05 unwired Semantic/Both buttons, WR-02/03 XSS).
# ---------------------------------------------------------------------------

import shutil  # noqa: E402
import subprocess  # noqa: E402

import pytest  # noqa: E402


def test_search_js_parses_as_valid_es_module(tmp_path: Path):
    """search.js must parse as a valid ES module — catches the CR-01 class of defect (a
    top-level `import` in a non-module script is a SyntaxError that kills ALL JS). This is a
    real parser-level execution check, not a substring assertion."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available to syntax-check the emitted module")
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js_src = (out / "assets" / "search.js").read_text(encoding="utf-8")
    # Force MODULE-mode parsing via stdin (--input-type is rejected with a file argument), so a
    # top-level static `import` in a non-module-safe form is caught as the SyntaxError it is.
    proc = subprocess.run(
        [node, "--check", "--input-type=module"],
        input=js_src,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"search.js is not a valid ES module:\n{proc.stderr}"


def test_search_script_tag_is_module(tmp_path: Path):
    """The page must load search.js as type=module so its import()/module scope work (CR-01)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    # v6.0 Phase 4: search runs on search.html; the other pages load only site.js.
    html = (out / "search.html").read_text(encoding="utf-8")
    assert '<script type="module" src="assets/search.js"></script>' in html
    assert '<script type="module" src="assets/site.js"></script>' in (out / "index.html").read_text(encoding="utf-8")


def test_mode_buttons_are_wired_to_semantic_lane(tmp_path: Path):
    """Semantic/Both buttons must carry data-mode and the JS must wire click handlers that
    drive a mode-aware submit calling semanticSearch + rrf (WR-04/WR-05: previously the
    buttons had no handlers and semanticSearch/rrf were defined but never invoked)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    html = (out / "search.html").read_text(encoding="utf-8")
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")

    assert 'data-mode="semantic"' in html and 'data-mode="both"' in html
    # buttons are wired
    assert ".mode button[data-mode]" in js
    assert ".addEventListener('click'" in js
    # mode-aware submit actually invokes the semantic lane + RRF fusion (not just defines them)
    assert "await semanticSearch(" in js, "Semantic/Both submit must invoke semanticSearch"
    assert "rrf(textRanked, semantic" in js, "Both mode must fuse full-text + semantic via RRF"


def test_semantic_search_is_invoked_not_only_defined(tmp_path: Path):
    """Regression guard for WR-04/05: semanticSearch must be CALLED, not merely declared."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")
    # one declaration + at least one call site
    assert js.count("semanticSearch(") >= 2


def test_transformers_js_loaded_lazily_via_dynamic_import(tmp_path: Path):
    """Transformers.js must be a lazy dynamic import() inside ensureModel(), never a top-level
    static import — so a CDN outage disables only semantic search, not the whole page (CR-01)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")
    assert "await import(TRANSFORMERS_CDN)" in js, "Transformers.js must be lazy-imported"
    assert 'import { pipeline } from "https://' not in js, "No top-level static CDN import"


def test_result_rendering_escapes_html(tmp_path: Path):
    """Bundle text must be escaped before reaching innerHTML (WR-02/03 XSS hardening)."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    js = (out / "assets" / "search.js").read_text(encoding="utf-8")
    assert "function escapeHtml(" in js
    # v6.0 Phase 4: every row field reaching innerHTML is escaped; status text uses textContent.
    assert "escapeHtml(row.title || row.source_path || row.id)" in js and "escapeHtml(row.href)" in js
    assert "status.innerHTML" not in js


def test_semantic_digest_threaded_to_results(tmp_path: Path):
    """When a browser index exists, #results must carry data-semantic-digest so the JS can
    cache-bust the blob fetch on the index digest."""
    bundle = make_indexed_bundle_with_blob(tmp_path)
    out = tmp_path / "site"
    build_public_site(PublicSiteConfig(bundle_root=bundle, output_dir=out))
    # v6.0 Phase 4: the digest travels in search-config.json, beside the blob path it keys.
    config = json.loads((out / "search-config.json").read_text(encoding="utf-8"))
    assert config["semantic"]["digest"] and config["semantic"]["vectors"]["shards"] == ["assets/vectors/0.bin"]
