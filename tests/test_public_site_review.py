"""Regression tests for review round phase-4-r1 (notes/reviews/phase-4-r1/FINDINGS.md).

Each test reproduces one reviewer finding on disk and failed on 3a92b0d46, before its fix.
Private-marker strings are assembled from pieces so the tracked-tree scan stays clean.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tests.public_site_fixtures import _finish, make_alpha, make_beta

HOME = "/" + "Users" + "/qzxprobe/private"
DRIVE = "https://drive." + "google.com/private"


def site():
    from conference_pipeline.publishing import static_site

    return static_site


def hub():
    from conference_pipeline.publishing import hub_site

    return hub_site


def build(bundle: Path, out: Path):
    module = site()
    return module.build_public_site(module.PublicSiteConfig(bundle_root=bundle, output_dir=out, receipt_path=out.parent / f"{out.name}-receipt.json"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_checksum(bundle: Path, rel: str, target: Path) -> None:
    """Add a CHECKSUMS.json entry under ``rel`` (any string) for the bytes of ``target``."""
    checksums = json.loads((bundle / "CHECKSUMS.json").read_text(encoding="utf-8"))
    checksums[rel] = {"artifact_type": "inline", "sha256": _sha(target), "size_bytes": target.stat().st_size}
    (bundle / "CHECKSUMS.json").write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reindex(bundle: Path) -> None:
    """Re-record index/ files in the index manifest, as the index stage does."""
    manifest_path = bundle / "index" / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_checksums"] = {
        p.relative_to(bundle / "index").as_posix(): _sha(p)
        for p in sorted((bundle / "index").rglob("*")) if p.is_file() and p.name != "index_manifest.json"
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- F1: checksum keys that leave the bundle or dodge the copy exclusions --------------


def test_r1_f1_a_checksum_key_outside_the_bundle_is_refused(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    sibling = bundle.parent / "sibling.txt"
    sibling.write_text("outside the bundle\n", encoding="utf-8")
    _add_checksum(bundle, "../sibling.txt", sibling)
    out = tmp_path / "f1a-site"
    with pytest.raises(site().StaticSiteError, match="unsafe path"):
        build(bundle, out)
    assert not out.exists()


def test_r1_f1_a_dotted_key_cannot_copy_the_staged_repository(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    notes = bundle / "public-repo" / "notes.txt"
    notes.write_text("staged repository content\n", encoding="utf-8")
    _add_checksum(bundle, "./public-repo/notes.txt", notes)
    out = tmp_path / "f1b-site"
    with pytest.raises(site().StaticSiteError, match="unsafe path"):
        build(bundle, out)
    assert not out.exists()


# --- F2: topic ids that become paths outside the site --------------------------------


def test_r1_f2_a_topic_id_cannot_write_outside_the_site(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    manifest = bundle / "decks" / "control" / "presentation_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["presentations"][1]["id"] = "../../escaped-topic"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    _finish(bundle)
    out = tmp_path / "f2-site"
    with pytest.raises(site().StaticSiteError, match="unsafe topic id"):
        build(bundle, out)
    assert not any(tmp_path.rglob("escaped-topic.html"))
    assert not out.exists()


# --- F3: the hub's marker gate --------------------------------------------------------


def test_r1_f3_the_hub_refuses_a_private_marker(tmp_path: Path) -> None:
    alpha = make_alpha(tmp_path)
    beta = make_beta(tmp_path)
    papers = alpha / "catalog" / "papers.jsonl"
    rows = [json.loads(line) for line in papers.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows[0]["title"] = f"A title with {HOME}"
    papers.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _finish(alpha)
    out = tmp_path / "f3-hub"
    with pytest.raises(hub().HubSiteError, match="private marker"):
        hub().build_hub_site([alpha, beta], out, hub_title="Hub")
    assert not out.exists()


# --- F4: binary files are scanned too ---------------------------------------------------


def test_r1_f4_a_marker_in_a_pdf_stops_the_build(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    pdf = bundle / "decks" / "payloads" / "01_reasoning" / "presentation.pdf"
    pdf.write_bytes(b"%PDF-1.4\n1 0 obj << /URI (" + DRIVE.encode() + b") >> endobj\n\xff\xfe\n")
    _finish(bundle)
    out = tmp_path / "f4-site"
    with pytest.raises(site().StaticSiteError, match="private marker"):
        build(bundle, out)
    assert not out.exists()


def test_r1_f4_random_bytes_that_look_like_an_email_do_not_stop_the_build(tmp_path: Path) -> None:
    """The binary scan uses the literal tokens only; the rule set false-matches compressed data."""
    bundle = make_alpha(tmp_path)
    email_shaped = b"q" + b"@" + b"zxcv.qa"  # split so the tracked-tree scan stays clean
    (bundle / "audio" / "01_reasoning" / "podcast.mp3").write_bytes(b"ID3\x00\x9a\xfe" + email_shaped + b"\x01\xff" * 4)
    _finish(bundle)
    build(bundle, tmp_path / "f4b-site")


# --- F5: excluded catalog rows never leave through a copied file ------------------------


def test_r1_f5_excluded_rows_are_not_downloadable(tmp_path: Path) -> None:
    out = tmp_path / "f5-site"
    build(make_beta(tmp_path), out)
    for path in out.rglob("*"):
        if path.is_file() and path.suffix in {".jsonl", ".json", ".html"}:
            assert "openreview:B3" not in path.read_text(encoding="utf-8"), path.relative_to(out).as_posix()
    public = [json.loads(line) for line in (out / "papers.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["paper_id"] for r in public) == ["openreview:B1", "openreview:B2"]


# --- F6 and F7: semantic search over a filtered or malformed index ----------------------


def _ranked_by_node(site_dir: Path) -> list:
    """Run the emitted search.js's semanticSearch in Node with a fake model, return its order."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node_not_installed")
    js = (site_dir / "assets" / "search.js").read_text(encoding="utf-8")
    start, end = js.index("async function semanticSearch"), js.index("function rrf")
    config = json.loads((site_dir / "search-config.json").read_text(encoding="utf-8"))
    # v6.0 Phase 7.1: the vectors are int8 shards, each row a float32 scale then dim codes.
    block, dim = config["semantic"]["vectors"], config["semantic"]["dim"]
    rows = []
    for name in block["shards"]:
        data = (site_dir / name).read_bytes()
        for offset in range(0, len(data), 4 + dim):
            scale = struct.unpack_from("<f", data, offset)[0]
            rows.append([scale, list(struct.unpack_from(f"<{dim}b", data, offset + 4))])
    harness = (
        f"const config = {json.dumps(config)};\n"
        f"const shardRows = {json.dumps(rows)}.map(([s, c]) => [s, Int8Array.from(c)]);\n"
        "async function loadVectors() { return shardRows; }\n"
        "async function ensureModel() { return async () => ({ data: new Float32Array(config.semantic.dim).fill(1) }); }\n"
        + js[start:end]
        + "\nsemanticSearch('q').then((r) => console.log(JSON.stringify(r)));\n"
    )
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_r1_f6_a_dropped_chunk_does_not_break_semantic_results(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    chunks = bundle / "index" / "corpus_chunks.jsonl"
    rows = [json.loads(line) for line in chunks.read_text(encoding="utf-8").splitlines()]
    rows[0]["text"] = f"see {HOME}"
    chunks.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (bundle / "index" / "chunk_ids.json").write_text(json.dumps([r["chunk_id"] for r in rows]), encoding="utf-8")
    _reindex(bundle)
    out = tmp_path / "f6-site"
    build(bundle, out)
    config = json.loads((out / "search-config.json").read_text(encoding="utf-8"))
    assert config["semantic"]["order"][0] is None
    ranked = _ranked_by_node(out)
    assert ranked and all(isinstance(i, int) for i in ranked)


def test_r1_f7_a_malformed_vector_blob_is_not_advertised(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    (bundle / "index" / "embeddings_browser.f32").write_bytes(b"\x00")
    _reindex(bundle)
    out = tmp_path / "f7-site"
    build(bundle, out)
    manifest = json.loads((out / "site_manifest.json").read_text(encoding="utf-8"))
    assert manifest["search_modes"] == ["text"]
    assert manifest["semantic_search"]["status"] == "not_in_this_release"
    assert "does not match" in manifest["semantic_search"]["reason"]


# --- F8: a failed build never leaves a partial site --------------------------------------


def test_r1_f8_a_failed_rebuild_keeps_the_previous_site_whole(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    out = tmp_path / "f8-site"
    build(bundle, out)
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    (bundle / "decks" / "markdown" / "01_reasoning" / "slides.md").write_text(f"see {HOME}\n", encoding="utf-8")
    _finish(bundle)
    with pytest.raises(site().StaticSiteError):
        build(bundle, out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before
    assert [p.name for p in tmp_path.iterdir() if ".staging-" in p.name] == []


def test_r1_f8_the_hub_replaces_its_directory_and_cleans_up_on_failure(tmp_path: Path, monkeypatch) -> None:
    alpha, beta = make_alpha(tmp_path), make_beta(tmp_path)
    out = tmp_path / "f8-hub"
    out.mkdir()
    (out / "stale.html").write_text("left from an older hub\n", encoding="utf-8")
    hub().build_hub_site([alpha, beta], out, hub_title="Hub")
    assert not (out / "stale.html").exists()
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(hub(), "write_json", boom)
    with pytest.raises(OSError):
        hub().build_hub_site([alpha, beta], out, hub_title="Hub")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before
    assert [p.name for p in tmp_path.iterdir() if ".staging-" in p.name] == []


# --- F9: hrefs are judged as a browser reads them -----------------------------------------


@pytest.mark.parametrize("href", [" //example.test/x", "\t//example.test/x", "/\\example.test/x", "\\\\example.test/x", "\n//example.test/x", "java\nscript:alert(1)"])
def test_r1_f9_disguised_links_are_dropped(href: str) -> None:
    from conference_pipeline.publishing.site_checks import sanitize_html

    out = sanitize_html(f'<a href="{href}">x</a>')
    assert "href" not in out, out


def test_r1_f9_ordinary_links_are_kept() -> None:
    from conference_pipeline.publishing.site_checks import sanitize_html

    assert sanitize_html('<a href="https://example.test/x">x</a>') == '<a href="https://example.test/x">x</a>'
    assert sanitize_html('<a href="../topics/a.html#s">x</a>') == '<a href="../topics/a.html#s">x</a>'


# === round phase-4-r2 (notes/reviews/phase-4-r2/FINDINGS.md); each failed on 0a981710d ===


def test_r2_f1_an_index_directory_symlinked_outside_is_refused(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    # The reviewer's case: the external manifest names this release and lists no files, so
    # no per-file containment check ever runs; only the manifest is read from outside.
    outside = tmp_path / "outside-index"
    shutil.move(str(bundle / "index"), outside)
    manifest = json.loads((outside / "index_manifest.json").read_text(encoding="utf-8"))
    manifest["output_checksums"] = {}
    manifest.pop("browser_index", None)
    (outside / "index_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "index").symlink_to(outside, target_is_directory=True)
    out = tmp_path / "r2f1-site"
    with pytest.raises(site().StaticSiteError, match="unsafe path"):
        build(bundle, out)
    assert not out.exists()


def test_r2_f2_a_path_to_a_dot_env_in_a_pdf_stops_the_build(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    pdf = bundle / "decks" / "payloads" / "01_reasoning" / "presentation.pdf"
    pdf.write_bytes(b"%PDF-1.4\n(see conf/" + b".env" + b" for keys)\n\xff\xfe\n")
    _finish(bundle)
    with pytest.raises(site().StaticSiteError, match="private marker"):
        build(bundle, tmp_path / "r2f2a-site")


def test_r2_f2_text_in_a_binary_suffix_file_meets_the_full_rules(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    email = "someone" + "@" + "realmail.example-corp.com"
    (bundle / "decks" / "payloads" / "01_reasoning" / "notes.png").write_text(f"contact {email}\n", encoding="utf-8")
    _finish(bundle)
    with pytest.raises(site().StaticSiteError, match="private marker"):
        build(bundle, tmp_path / "r2f2b-site")


def test_r2_f3_index_chunks_of_unpublished_files_are_not_searchable(tmp_path: Path) -> None:
    bundle = make_beta(tmp_path)
    chunks = [
        {"chunk_id": "pubchunk:v1", "source_path": "decks/payloads/01_vision/presentation.html", "content_type": "html", "text": "Vision deck text"},
        {"chunk_id": "pubchunk:c1", "source_path": "catalog/papers.jsonl", "content_type": "jsonl", "text": "openreview:B3 An excluded diffusion submission"},
        {"chunk_id": "pubchunk:x1", "source_path": "catalog/excluded_ids.json", "content_type": "json", "text": '{"excluded": ["openreview:B3"]}'},
    ]
    (bundle / "index" / "corpus_chunks.jsonl").write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
    _reindex(bundle)
    out = tmp_path / "r2f3-site"
    build(bundle, out)
    data = "".join(p.read_text(encoding="utf-8") for p in sorted((out / "search").rglob("*.json")))
    assert "openreview:B3" not in data and "excluded diffusion" not in data
    assert "pubchunk:v1" in data
    # The index holds no text, so its row ids say what is searchable (Phase 7.1).
    from conference_pipeline.publishing.search_shards import read_all_rows, read_ranked

    ids = {row.get("id") for row in read_all_rows(out)}
    assert "pubchunk:c1" not in ids and "pubchunk:x1" not in ids
    assert not [row for row in read_ranked(out, "excluded") if str(row.get("id", "")).startswith("pubchunk:")]


@pytest.mark.parametrize("dim", [0, "0", 0.5, -8, True, "8"])
def test_r2_f4_a_non_positive_or_non_integer_dim_is_not_advertised(tmp_path: Path, dim) -> None:
    bundle = make_alpha(tmp_path)
    (bundle / "index" / "embeddings_browser.f32").write_bytes(b"")
    manifest_path = bundle / "index" / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["browser_index"]["dim"] = dim
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _reindex(bundle)
    out = tmp_path / "r2f4-site"
    build(bundle, out)
    assert json.loads((out / "site_manifest.json").read_text(encoding="utf-8"))["search_modes"] == ["text"]


def test_r2_f5_a_failed_swap_restores_the_previous_site(tmp_path: Path, monkeypatch) -> None:
    bundle = make_alpha(tmp_path)
    out = tmp_path / "r2f5-site"
    build(bundle, out)
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    real_rename = Path.rename

    def flaky(self, target):
        if ".staging-" in self.name:
            raise OSError("rename failed")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        build(bundle, out)
    monkeypatch.undo()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before
    assert [p.name for p in tmp_path.iterdir() if ".staging-" in p.name or ".old-" in p.name] == []


def test_r2_f6_an_unwritable_receipt_leaves_the_previous_site(tmp_path: Path) -> None:
    bundle = make_alpha(tmp_path)
    out = tmp_path / "r2f6-site"
    build(bundle, out)
    sentinel = out / "previous-build.txt"  # a rebuild is byte-identical; this shows a swap
    sentinel.write_text("from the previous build\n", encoding="utf-8")
    receipt_dir = tmp_path / "locked"
    receipt_dir.mkdir()
    receipt_dir.chmod(0o500)
    try:
        module = site()
        with pytest.raises(OSError):
            module.build_public_site(module.PublicSiteConfig(bundle_root=bundle, output_dir=out, receipt_path=receipt_dir / "r.json"))
    finally:
        receipt_dir.chmod(0o700)
    assert sentinel.exists(), "the previous site was replaced although the build failed"
    assert [p.name for p in tmp_path.iterdir() if ".staging-" in p.name or ".old-" in p.name] == []
