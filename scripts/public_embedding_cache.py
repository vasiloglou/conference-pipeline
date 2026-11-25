#!/usr/bin/env python3
"""Make a public index's vectors on another machine (v6.0 Phase 7.1, embedding_cache).

  export  --bundle-root B --out chunks.jsonl   the bundle's chunk texts, each once
  embed   --chunks chunks.jsonl --out DIR      run the governed models over them

``conference-pipeline release build <slug> --embedding-cache DIR`` then builds the index
from DIR, refusing any chunk DIR does not hold. The models are the registry's
``public_search.canonical_embedding`` and ``public_search.browser_embedding`` tasks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from conference_pipeline.publishing import public_index as pi  # noqa: E402
from conference_pipeline.publishing.embedding_cache import (  # noqa: E402
    embed_texts,
    export_chunk_texts,
    read_chunk_texts,
)


def _export(args: argparse.Namespace) -> int:
    chunks, _files, _skipped = pi.build_public_chunks(pi.PublicIndexConfig(bundle_root=args.bundle_root))
    count = export_chunk_texts((chunk.text for chunk in chunks), args.out)
    print(f"exported {count} distinct text(s) from {len(chunks)} chunk(s) to {args.out}")
    return 0


def _canonical() -> tuple[str, object]:
    from conference_pipeline.indexing.embedding_resolver import build_resolved_embedding_client

    resolved, client = build_resolved_embedding_client(
        {"provider": pi.DEFAULT_EMBEDDING_PROVIDER, "model": pi.DEFAULT_EMBEDDING_MODEL, "embedding_dim": pi.DEFAULT_EMBEDDING_DIM}
    )

    def fn(texts: list[str]) -> np.ndarray:
        out = client.embed_texts_array(texts) if hasattr(client, "embed_texts_array") else client.embed_texts(texts)
        return np.asarray(out, dtype=np.float32)

    return pi.DEFAULT_EMBEDDING_MODEL, fn


def _browser() -> tuple[str, object]:
    def fn(texts: list[str]) -> np.ndarray:
        vectors = pi._embed_embeddinggemma_onnx(pi.DEFAULT_BROWSER_EMBEDDING_MODEL, texts, is_query=False)
        if vectors is None:
            raise SystemExit("the browser embedding runtime is not available here")
        return vectors

    return pi.DEFAULT_BROWSER_EMBEDDING_MODEL, fn


def _embed(args: argparse.Namespace) -> int:
    rows = read_chunk_texts(args.chunks)
    lanes = {}
    if "canonical" in args.lanes:
        lanes["canonical"] = _canonical()
    if "browser" in args.lanes:
        lanes["browser"] = _browser()
    started = time.time()

    def progress(lane: str, done: int, total: int) -> None:
        print(f"{lane}: {done}/{total} ({time.time() - started:.0f}s)", flush=True)

    embed_texts(rows, args.out, lanes, batch=args.batch, progress=progress)
    print(f"cache written: {args.out} ({len(rows)} text(s), lanes {', '.join(lanes)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--bundle-root", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--chunks", type=Path, required=True)
    embed.add_argument("--out", type=Path, required=True)
    embed.add_argument("--lanes", default="canonical,browser")
    embed.add_argument("--batch", type=int, default=256)
    args = parser.parse_args(argv)
    return _export(args) if args.command == "export" else _embed(args)


if __name__ == "__main__":
    raise SystemExit(main())
