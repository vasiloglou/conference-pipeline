#!/usr/bin/env python3
"""Recover missing paper authors by exact-title lookup, verified against the PDF.

Conference-neutral. Reads the corpus inventory, selects paper records whose
authors are unresolved (their source receipt had none), looks each title up on
free backends (arXiv, Crossref; ``paper_title_resolver``), and accepts a hit
only when the normalized titles match and at least half of the looked-up
surnames appear on the paper's own first page. The first-page header parse is
never used as the author list itself (it misreads titles and figure text).

Writes ``storage/workshops/paper_author_recovery.json`` (resumable); Stage 6
applies verified entries with provenance ``title_lookup_verified``.

    scripts/recover_paper_authors_by_title.py --workspace conferences/<slug>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.indexing import paper_title_resolver as resolver  # noqa: E402
from conference_pipeline.indexing.corpus_vector_store import resolve_workspace_member  # noqa: E402
from conference_pipeline.indexing.paper_source_metadata import (  # noqa: E402
    PROVENANCE_UNRESOLVED,
    verify_authors_against_header,
)

DEFAULT_INVENTORY = Path("storage/corpus/corpus_inventory.json")
DEFAULT_OUTPUT = Path("storage/workshops/paper_author_recovery.json")
LOOKUPS_NAME = "paper_author_recovery.lookups.json"
DEFAULT_FAMILIES = ("workshop_paper_text",)
RECEIPT_TYPE = "conference_pipeline.paper_author_recovery"


def _unresolved_records(inventory: dict[str, Any], families: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for record in inventory.get("artifact_records") or []:
        if not isinstance(record, dict) or record.get("source_family_id") not in families:
            continue
        if record.get("authors"):
            continue
        provenance = record.get("source_metadata_provenance") or {}
        if provenance.get("authors") not in (None, PROVENANCE_UNRESOLVED):
            continue
        if not record.get("paper_title") or not record.get("text_path"):
            continue
        out.append(record)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--family", action="append", default=[], help="source_family_id to recover (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="list candidates; no network")
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    inventory_path = args.inventory or workspace / DEFAULT_INVENTORY
    output_path = args.output or workspace / DEFAULT_OUTPUT
    families = tuple(args.family) or DEFAULT_FAMILIES
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    records = _unresolved_records(inventory, families)
    if args.dry_run:
        print(json.dumps({"candidates": len(records), "families": families}, indent=2))
        return 0

    lookups = resolver.resolve_titles(
        [{"key": record["artifact_id"], "title": record["paper_title"]} for record in records],
        output_path=output_path.with_name(LOOKUPS_NAME),
        backends=("arxiv", "crossref"),
    )
    by_key = {str(item["key"]): item for item in lookups}
    entries: dict[str, dict[str, Any]] = {}
    for record in records:
        lookup = by_key.get(record["artifact_id"]) or {}
        text_path, error = resolve_workspace_member(workspace, record["text_path"])
        header = "" if error or text_path is None else text_path.read_text(encoding="utf-8", errors="replace")
        entry: dict[str, Any] = {"paper_title": record["paper_title"], "verified": False}
        for backend in ("arxiv", "crossref"):
            hit = lookup.get(backend)
            if not isinstance(hit, dict) or not hit.get("authors"):
                continue
            verified, matched = verify_authors_against_header(hit["authors"], header)
            candidate = {
                "backend": backend,
                "reference": hit.get("arxiv_id") or hit.get("doi"),
                "match_score": hit.get("match_score"),
                "authors": hit["authors"],
                "surnames_on_first_page": matched,
                "verified": verified,
            }
            entry.setdefault("candidates", []).append(candidate)
            if verified and not entry["verified"]:
                entry.update({key: candidate[key] for key in ("backend", "reference", "match_score", "authors")})
                entry["verified"] = True
        if lookup.get("error"):
            entry["lookup_error"] = lookup["error"]
        entries[record["artifact_id"]] = entry

    payload = {
        "receipt_type": RECEIPT_TYPE,
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inventory_path": str(inventory_path.relative_to(workspace)) if inventory_path.is_relative_to(workspace) else str(inventory_path),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "families": list(families),
        "candidate_count": len(records),
        "verified_count": sum(1 for entry in entries.values() if entry["verified"]),
        "acceptance_rule": "normalized title similarity >= 0.9 and >= half of surnames on the PDF first page",
        "entries": entries,
        "provider_calls_made": False,
        "secrets_copied": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("candidate_count", "verified_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
