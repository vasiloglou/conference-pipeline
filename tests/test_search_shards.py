"""v6.0 Phase 7.1: the sharded search index ranks exactly as the reference BM25 (T5-T8)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from conference_pipeline.publishing import site_templates as tpl
from conference_pipeline.publishing.search_shards import (
    SearchShardError,
    quantize_rows,
    read_all_rows,
    read_scored,
    read_vector_shards,
    write_search_shards,
    write_vector_shards,
)
from conference_pipeline.publishing.site_checks import BM25_B, BM25_K1, bm25_rank, tokenize
from conference_pipeline.publishing.static_site import search_body

WORDS = "diffusion model vision language benchmark reinforcement learning graph agent robot video".split()


def _rows(count: int = 240, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        words = [rng.choice(WORDS) for _ in range(rng.randint(0, 30))]
        rows.append({"id": f"r{index}", "kind": "paper", "title": f"T{index}", "text": " ".join(words), "href": None})
    # Exact duplicates give equal scores: the tie order must be row order in both rankers.
    rows += [dict(rows[3], id="dup-a"), dict(rows[3], id="dup-b")]
    return rows


QUERIES = [
    "diffusion",
    "vision language model",
    "reinforcement learning",
    "benchmark benchmark",  # a repeated term counts twice in bm25_rank
    "quokka",  # no row has it
    "",
    "Graph-Agent!",
    "robot video diffusion",
    "learning",
    "model",
    "agent quokka",
    "t3",  # titles are searched too
]


def _reference_scores(rows: list[dict], query: str) -> list[tuple[float, str]]:
    import math
    from collections import Counter

    terms = tokenize(query)
    docs = [tokenize(f"{r.get('title') or ''} {r.get('text') or ''}") for r in rows]
    if not terms or not docs:
        return []
    avgdl = sum(len(d) for d in docs) / len(docs) or 1.0
    df = Counter(t for d in docs for t in set(d))
    out = []
    for row, doc in zip(rows, docs):
        tf = Counter(doc)
        score = 0.0
        for term in terms:
            if tf[term]:
                idf = math.log(1 + (len(docs) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (BM25_K1 + 1) / (tf[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avgdl))
        if score > 0:
            out.append((score, row["id"]))
    out.sort(key=lambda item: -item[0])
    return out


@pytest.mark.parametrize("query", QUERIES)
def test_t5_the_shards_rank_exactly_as_bm25_rank(tmp_path: Path, query: str) -> None:
    rows = _rows()
    write_search_shards(rows, tmp_path, max_shard_bytes=4096)
    scored = read_scored(tmp_path, query)
    got = [(score, rows[index]["id"]) for score, index in scored]
    assert [rid for _s, rid in got] == [row["id"] for row in bm25_rank(rows, query)]
    reference = _reference_scores(rows, query)
    assert len(got) == len(reference)
    for (score, rid), (expected, eid) in zip(got, reference):
        assert rid == eid and abs(score - expected) <= 1e-9


def test_t6_shards_respect_the_cap_and_rebuild_byte_identical(tmp_path: Path) -> None:
    rows = _rows(1500)
    first, second = tmp_path / "a", tmp_path / "b"
    meta = write_search_shards(rows, first, max_shard_bytes=8192)
    write_search_shards(rows, second, max_shard_bytes=8192)
    assert meta["buckets"] > 1 and meta["doc_blocks"] == 2
    for path in (first / "search" / "terms").iterdir():
        assert path.stat().st_size <= 8192
    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    for rel in files:
        assert (first / rel).read_bytes() == (second / rel).read_bytes()
    assert [r["id"] for r in read_all_rows(first)] == [r["id"] for r in rows]
    assert "text" not in read_all_rows(first)[0]  # display fields only


def test_t6_a_term_too_big_for_any_shard_is_refused(tmp_path: Path) -> None:
    rows = [{"id": str(i), "text": "same"} for i in range(5000)]
    with pytest.raises(SearchShardError):
        write_search_shards(rows, tmp_path, max_shard_bytes=1024)


def test_t8_int8_vectors_keep_the_float_ranking(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(400, 64)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    block = write_vector_shards(vectors, tmp_path, max_shard_bytes=4096)
    assert len(block["shards"]) > 1
    for name in block["shards"]:
        assert (tmp_path / name).stat().st_size <= 4096
    decoded = read_vector_shards(tmp_path, block)
    assert decoded.shape == vectors.shape
    overlaps = []
    for _ in range(10):
        query = rng.normal(size=64).astype(np.float32)
        exact = set(np.argsort(-(vectors @ query))[:10])
        approx = set(np.argsort(-(decoded @ query))[:10])
        overlaps.append(len(exact & approx))
    assert sum(overlaps) / len(overlaps) >= 9


def test_t8_a_zero_vector_quantises_without_dividing_by_zero() -> None:
    codes, scales = quantize_rows(np.zeros((2, 4), dtype=np.float32))
    assert not codes.any() and not scales.any()


def _site(tmp_path: Path, rows: list[dict]) -> Path:
    site = tmp_path / "site"
    write_search_shards(rows, site, max_shard_bytes=4096)
    (site / "assets").mkdir(parents=True, exist_ok=True)
    (site / "assets" / "search.js").write_text(tpl.SEARCH_JS, encoding="utf-8")
    (site / "assets" / "site.css").write_text(tpl.CSS, encoding="utf-8")
    (site / "search-config.json").write_text(json.dumps({"modes": ["text"], "semantic": None}), encoding="utf-8")
    (site / "search.html").write_text(
        tpl.page(title="Search", heading="Search", body=search_body(["text"]), site_name="t", scripts=("assets/search.js",)),
        encoding="utf-8",
    )
    return site


@pytest.mark.parametrize("query", ["diffusion", "vision language model", "benchmark benchmark", "t3"])
def test_t7_the_browser_ranks_like_the_reference_and_loads_only_its_shards(tmp_path: Path, query: str) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed")
    import http.server
    import socketserver
    import threading

    rows = _rows()
    site = _site(tmp_path, rows)
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(site), **k)  # noqa: E731
    expected = bm25_rank(rows, query)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        requested: list[str] = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.on("request", lambda request: requested.append(request.url.split(f":{port}/", 1)[1]))
            page.goto(f"http://127.0.0.1:{port}/search.html")
            page.fill("#q", query)
            page.click("#search-form button[type=submit]")
            page.wait_for_function("document.getElementById('search-status').textContent.includes('result')")
            shown = page.eval_on_selector_all("#results li p:first-child", "els => els.map(e => e.textContent)")
            status = page.inner_text("#search-status")
            browser.close()
        server.shutdown()
    assert shown == [row["title"] for row in expected[:50]]
    assert status == f"{len(expected)} result(s)."
    meta = json.loads((site / "search" / "meta.json").read_text(encoding="utf-8"))
    from conference_pipeline.publishing.search_shards import fnv1a

    wanted = {f"search/terms/{fnv1a(t) & (meta['buckets'] - 1)}.json" for t in tokenize(query)}
    assert {r for r in requested if r.startswith("search/terms/")} == wanted
    blocks = {f"search/docs/{rows.index(row) // meta['doc_block']}.json" for row in expected[:50]}
    assert {r for r in requested if r.startswith("search/docs/")} == blocks
