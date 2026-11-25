"""Fixture bundles for the public-site corpus (tests/fixtures/public_site/README.md).

Both bundles follow the layout of the real ICLR 2026 v1.0.0 bundle (spec D6): deck payloads
under ``decks/payloads/<topic>/``, video sidecars, ``decks/control/presentation_manifest.json``,
and the reader families of spec section 6.1 (``narratives/``, ``catalog/``).
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

ALPHA = "alphaconf-v1.0.0"
BETA = "betaconf-v1.0.0"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _put(root: Path, rel: str, data: bytes | str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)


def _blank_pdf() -> bytes:
    """A real one-page PDF: the marker check reads a PDF's text and fails closed on one it
    cannot read (review phase-6b-r1 F1), so a fixture deck needs a readable one."""
    import io

    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def _sidecar(bundle_id: str, rel: str) -> str:
    return _json({
        "cdn_url": f"https://cdn.example.test/{bundle_id}/{rel.removesuffix('.cdn.json')}",
        "mime_type": "video/mp4", "object_key": rel, "sha256": "0" * 64,
        "size_bytes": 1234, "source_filename": Path(rel.removesuffix(".cdn.json")).name,
    })


def _finish(root: Path, *, not_in_checksums: tuple[str, ...] = ()) -> None:
    checksums = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel != "CHECKSUMS.json" and rel not in not_in_checksums and not rel.startswith("index/"):
            data = path.read_bytes()
            checksums[rel] = {"artifact_type": "inline", "sha256": _sha(data), "size_bytes": len(data)}
    _put(root, "CHECKSUMS.json", _json(checksums))


def _release(root: Path, bundle_id: str, slug: str, name: str) -> None:
    _put(root, "RELEASE.json", _json({
        "bundle_id": bundle_id, "conference": slug, "source_sha": "a" * 40,
        "export_manifest_sha256": "b" * 64, "included_families": [], "checksum_file": "CHECKSUMS.json",
        "release": {"conference_name": name, "source_sha": "a" * 40},
        "zenodo_doi": f"10.5281/zenodo.{len(slug)}000", "hf_dataset_url": f"https://huggingface.co/datasets/example/{slug}",
        "public_site_url": f"https://{slug}.example.test", "public_repo_url": f"https://github.com/example/{slug}-data",
    }))


def _index(root: Path, bundle_id: str, chunks: list[dict[str, Any]], *, browser_dim: int | None) -> None:
    _put(root, "index/corpus_chunks.jsonl", "".join(json.dumps(c) + "\n" for c in chunks))
    manifest: dict[str, Any] = {"bundle_id": bundle_id, "chunk_count": len(chunks),
                                "browser_capabilities": {"parquet": {"status": "unavailable"}, "sqlite_vec": {"status": "unavailable"}}}
    if browser_dim:
        blob = b"".join(struct.pack("<f", 0.1 * (i % 7)) for i in range(browser_dim * len(chunks)))
        _put(root, "index/embeddings_browser.f32", blob)
        manifest["browser_index"] = {"model": "fixture-embed-model", "dim": browser_dim,
                                     "blob": "embeddings_browser.f32", "digest": _sha(blob)}
    # As the real index stage does: index/ files are recorded in the index manifest's
    # output_checksums (relative to index/), not in the bundle's CHECKSUMS.json.
    manifest["output_checksums"] = {
        p.relative_to(root / "index").as_posix(): _sha(p.read_bytes())
        for p in sorted((root / "index").rglob("*")) if p.is_file()
    }
    _put(root, "index/index_manifest.json", _json(manifest))


def _bundle_dir(base: Path, slug: str, bundle_id: str) -> Path:
    return base / "conferences" / slug / "exports" / "public-release" / bundle_id


def make_alpha(base: Path) -> Path:
    root = _bundle_dir(base, "alphaconf", ALPHA)
    _release(root, ALPHA, "alphaconf", "Alpha Conference 2026")
    _put(root, "decks/control/presentation_manifest.json", _json({"schema_version": 1, "presentations": [
        {"id": "01_reasoning", "title": "Reasoning at test time", "presentation_type": "topic-deep-dive"},
        {"id": "02_agents", "title": "Agents and tools", "presentation_type": "topic-deep-dive"},
    ]}))
    p = "decks/payloads/01_reasoning/"
    _put(root, p + "presentation.html", "<!doctype html><html lang=en><title>Reasoning deck</title><h1>Reasoning</h1></html>")
    _put(root, p + "presentation.pdf", _blank_pdf())
    for video in ("presentation_video.mp4.cdn.json", "presentation_video_notebooklm.mp4.cdn.json"):
        _put(root, p + video, _sidecar(ALPHA, p + video))
    _put(root, "decks/markdown/01_reasoning/slides.md", "# Reasoning slides\n")
    _put(root, "decks/payloads/02_agents/presentation.html", "<!doctype html><html lang=en><title>Agents deck</title><h1>Agents</h1></html>")
    _put(root, "audio/01_reasoning/podcast.mp3", b"ID3fixture")
    narrative = "# Reasoning at test time\n\nThe models behave in a quokkalike way under search.\n\n## Findings\n\n- One finding.\n"
    _put(root, "narratives/01_reasoning.md", narrative)
    rows = [
        {"paper_id": "openreview:A1", "title": "Scaling test time search", "authors": ["Ada One"], "track": "main", "venue": "Alpha 2026", "workshop": None, "official_url": "https://openreview.net/forum?id=A1", "topics": ["01_reasoning"]},
        {"paper_id": "openreview:A2", "title": "Diffusion planners for reasoning", "authors": ["Ben Two"], "track": "main", "venue": "Alpha 2026", "workshop": None, "official_url": "https://openreview.net/forum?id=A2", "topics": ["01_reasoning"]},
        {"paper_id": "openreview:A3", "title": "Tool use benchmarks", "authors": ["Cy Three"], "track": "workshop", "venue": "Alpha 2026", "workshop": "Agents WS", "official_url": "https://openreview.net/forum?id=A3", "topics": ["02_agents"]},
        {"paper_id": "site:s1", "title": "A site-only workshop paper", "authors": ["Di Four"], "track": "workshop", "venue": "Alpha 2026", "workshop": "Agents WS", "official_url": None, "topics": []},
    ]
    _put(root, "catalog/papers.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _put(root, "catalog/topic_papers.json", _json({"01_reasoning": ["openreview:A1", "openreview:A2"], "02_agents": []}))
    _put(root, "catalog/statistics.json", _json({"papers": {"total": 4}, "corpus_version": {"label": "alphaconf-corpus-000000000000"}}))
    _put(root, "public-repo/.git/config", "[core]\n")  # a staged repo inside the bundle: never copied
    _index(root, ALPHA, [
        {"chunk_id": "pubchunk:n1", "source_path": "narratives/01_reasoning.md", "content_type": "markdown", "text": narrative},
        {"chunk_id": "pubchunk:d1", "source_path": "decks/markdown/01_reasoning/slides.md", "content_type": "markdown", "text": "Reasoning slides about search"},
    ], browser_dim=8)
    _finish(root)
    return root


def make_beta(base: Path) -> Path:
    root = _bundle_dir(base, "betaconf", BETA)
    _release(root, BETA, "betaconf", "Beta Conference 2025")
    _put(root, "decks/control/presentation_manifest.json", _json({"schema_version": 1, "presentations": [
        {"id": "01_vision", "title": "Vision models", "presentation_type": "topic-deep-dive"},
    ]}))
    _put(root, "decks/payloads/01_vision/presentation.html", "<!doctype html><html lang=en><title>Vision deck</title><h1>Vision</h1></html>")
    rows = [
        {"paper_id": "openreview:B1", "title": "Latent diffusion at scale", "authors": ["Ed Five"], "track": "main", "venue": "Beta 2025", "workshop": None, "official_url": "https://openreview.net/forum?id=B1", "topics": ["01_vision"]},
        {"paper_id": "openreview:B2", "title": "Vision transformers revisited", "authors": ["Fay Six"], "track": "main", "venue": "Beta 2025", "workshop": None, "official_url": "https://openreview.net/forum?id=B2", "topics": ["01_vision"]},
        {"paper_id": "openreview:B3", "title": "An excluded diffusion submission", "authors": ["Gil Seven"], "track": "workshop", "venue": "Beta 2025", "workshop": "WS", "official_url": "https://openreview.net/forum?id=B3", "topics": []},
    ]
    _put(root, "catalog/papers.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    _put(root, "catalog/excluded_ids.json", _json({"excluded": ["openreview:B3"]}))
    _put(root, "catalog/extra_papers.jsonl", json.dumps({"paper_id": "openreview:B9", "title": "A stray diffusion row", "authors": [], "track": "main", "venue": "Beta 2025", "workshop": None, "official_url": None, "topics": []}) + "\n")
    _put(root, "catalog/topic_papers.json", _json({"01_vision": ["openreview:B1", "openreview:B2"]}))
    _index(root, BETA, [{"chunk_id": "pubchunk:v1", "source_path": "decks/payloads/01_vision/presentation.html", "content_type": "html", "text": "Vision deck text"}], browser_dim=None)
    _finish(root, not_in_checksums=("catalog/extra_papers.jsonl",))
    return root
