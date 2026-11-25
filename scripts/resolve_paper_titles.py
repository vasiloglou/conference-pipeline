#!/usr/bin/env python3
"""Resolve paper titles on OpenAlex and arXiv: open copies and other venues.

Input is either a JSON list of ``{key, title, authors?, workshop?}`` records
(``--input``) or a workshop gap-research directory (``--research-dir``, one
JSON file per workshop with a ``papers`` list). Resumable: rerun with the same
``--output`` to continue. Default backends arXiv + Crossref are free
and credential-free (~4 s per title, ~7 s when arXiv has no copy under that
exact title and a second arXiv query is sent); OpenAlex and Semantic Scholar
are opt-in and need their API key env vars.

    PYTHONPATH=src .venv/bin/python scripts/resolve_paper_titles.py \\
        --research-dir <dir> --output <resolutions.json> --workspace conferences/<slug>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from conference_pipeline.conference_identity import load_conference_identity  # noqa: E402
from conference_pipeline.indexing import paper_title_resolver as resolver  # noqa: E402


def items_from_research_dir(research_dir: Path) -> list[dict]:
    items: dict[str, dict] = {}
    for path in sorted(research_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Skip anything that is not a workshop research object, such as the
        # resolutions list a previous run wrote with --output in this directory.
        if not isinstance(payload, dict) or not isinstance(payload.get("papers"), list):
            continue
        for paper in payload["papers"]:
            title = str((paper or {}).get("title") or "").strip()
            if not title:
                continue
            key = f"{payload.get('workshop')}:{hashlib.sha1(resolver.normalize(title).encode()).hexdigest()[:12]}"
            items.setdefault(key, {"key": key, "workshop": payload.get("workshop"), "schedule_id": payload.get("schedule_id"),
                                   "title": title, "authors": paper.get("authors") or [], "site_link": paper.get("link")})
    return list(items.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--research-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=None, help="Adds the conference acronym as a home-venue pattern.")
    parser.add_argument("--home-venue-pattern", action="append", default=[])
    parser.add_argument(
        "--backends", default=",".join(resolver.DEFAULT_BACKENDS),
        help="Comma-separated: arxiv, crossref (free defaults); openalex, semantic_scholar (need API key env vars).",
    )
    args = parser.parse_args(argv)

    try:
        backends = resolver.check_backends(b.strip() for b in args.backends.split(",") if b.strip())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    items = json.loads(args.input.read_text(encoding="utf-8")) if args.input else items_from_research_dir(args.research_dir)
    patterns = list(args.home_venue_pattern)
    if args.workspace:
        acronym = re.findall(r"[A-Za-z]{2,}", load_conference_identity(args.workspace).name)
        patterns += [rf"\b{re.escape(acronym[0])}\b"] if acronym else []
    results = resolver.resolve_titles(
        items, output_path=args.output, backends=backends, home_venue_patterns=patterns,
        progress=lambda i, n: print(f"{i}/{n}", flush=True) if i % 20 == 0 or i == n else None,
    )
    print("resolve_paper_titles:", json.dumps(resolver.summarize(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
