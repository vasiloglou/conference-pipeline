#!/usr/bin/env python3
"""Offline, zero-network re-derivation of CVF openaccess author evidence.

``source_discovery.py``'s ``_extract_cvf_openaccess_paper_records`` now
captures per-paper author lists from the CVF day-listing HTML it already
fetches and caches (see the extractor change landed alongside this script).
An already-frozen ``source_discovery_inventory.json`` produced *before* that
change has no author evidence -- but the HTML bodies it was built from are
still sitting in ``cache/source-discovery/<run_id>/...`` on disk. This
command upgrades such a frozen inventory in place by re-parsing those
already-cached bodies, WITHOUT re-fetching anything and WITHOUT changing any
existing ``source_record_id`` or ``snapshot_sha256`` -- both are derived by
``_source_record`` from ``native_identifier``/``title``/``aliases`` and the
snapshot's own digest, never from the ``authors`` field this command adds.

``snapshot_family`` (``conference_pipeline.source_discovery``) always calls
the fetcher unless a pre-built response is supplied -- it is NOT an offline
re-parse path. This script is the separate, genuinely offline command that
resolves that gap: it never imports or calls anything that opens a socket.

Usage:
    python scripts/rederive_source_discovery_authors.py \\
        --workspace conferences/cvpr2026 \\
        --run-id cvpr2026-live-production-r10-main-paper-corpus \\
        --dry-run --json

Exit codes: 0 on success (including a clean ``--dry-run``), 1 on any
fail-closed condition (see the ``Rederive*Error`` codes below).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.source_discovery import (  # noqa: E402
    _authors_by_pdf_stem,
    _CVF_OPENACCESS_FAMILY_ID_PREFIX,
    _CVF_OPENACCESS_PDF_HREF_RE,
    _atomic_write_json,
    canonical_json_digest,
)

INVENTORY_RELATIVE_PATH = Path("storage/source-discovery/source_discovery_inventory.json")
CACHE_RELATIVE_ROOT = Path("cache/source-discovery")

# The downstream digest-binding scan (see `_downstream_digest_bindings`) reads
# this production runner file, the only known consumer that pins the whole
# inventory's own digest. If a future conference runner introduces a second
# pinning site it must be added here, not assumed absent.
DOWNSTREAM_PRODUCTION_SCRIPTS: tuple[Path, ...] = (Path("scripts/run_cvpr2026_production.py"),)
_INVENTORY_PATH_CONSTANT_NAME = "SOURCE_DISCOVERY_INVENTORY_PATH"
_INVENTORY_SHA_BINDING_MARKERS = ("inventory_sha256", "SOURCE_DISCOVERY_INVENTORY_PATH")


class RederiveError(RuntimeError):
    """A fail-closed condition. ``code`` is the stable machine-readable id."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _resolve_contained(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it stays inside ``root``.

    Rejects a resolved path escaping the workspace (including via a symlink)
    before any read is attempted -- refusal happens purely on path math, the
    file is never opened first.
    """

    resolved_root = root.resolve()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RederiveError(
            "rederive_path_escapes_workspace",
            f"{candidate} resolves outside the permitted root {root}",
        ) from exc
    return resolved


def _safe_run_id(run_id: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
        raise RederiveError(
            "rederive_invalid_run_id", "run id must be a non-empty path-safe identifier"
        )
    return run_id


def _stem_from_pdf_alias(alias: str) -> str | None:
    """Extract the same filename-stem join key the extractor uses, from a
    source record's own pdf alias url -- never re-derived by string
    substitution."""

    if not alias:
        return None
    match = _CVF_OPENACCESS_PDF_HREF_RE.match(urlsplit(alias).path)
    return match.group("stem") if match else None


def _collect_cached_authors_by_stem(
    workspace: Path, cache_root: Path
) -> tuple[dict[str, list[str]], list[str]]:
    """Read every already-cached openaccess-family ``body.bin`` under
    ``cache_root`` and return the merged stem -> authors mapping plus the
    relative body paths actually read (for the JSON summary). Never opens a
    socket; only ever reads bytes already on disk."""

    authors_by_stem: dict[str, list[str]] = {}
    bodies_read: list[str] = []
    if not cache_root.is_dir():
        return authors_by_stem, bodies_read
    for family_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        if not family_dir.name.startswith(_CVF_OPENACCESS_FAMILY_ID_PREFIX):
            continue
        for snapshot_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
            body_path = snapshot_dir / "body.bin"
            if not body_path.is_file():
                continue
            contained = _resolve_contained(cache_root, body_path)
            body = contained.read_bytes()
            html_text = body.decode("utf-8", errors="replace")
            for stem, authors in _authors_by_pdf_stem(html_text).items():
                authors_by_stem.setdefault(stem, authors)
            bodies_read.append(str(body_path.relative_to(workspace)))
    return authors_by_stem, bodies_read


def _identity_snapshot(records: list[Mapping[str, Any]]) -> tuple[set[str], list[str]]:
    ids = {str(record["source_record_id"]) for record in records}
    digests = sorted(str(record["snapshot_sha256"]) for record in records)
    return ids, digests


def assert_identity_preserved(
    pre_records: list[Mapping[str, Any]], post_records: list[Mapping[str, Any]]
) -> None:
    """Fail closed if the ``source_record_id`` set or the ``snapshot_sha256``
    multiset moved between the pre-image and the post-image.

    This is the invariant that lets this command run against a frozen,
    downstream-pinned corpus without invalidating any existing digest
    binding to a *record*: only new ``authors`` values are ever added.
    """

    pre_ids, pre_digests = _identity_snapshot(pre_records)
    post_ids, post_digests = _identity_snapshot(post_records)
    if pre_ids != post_ids or pre_digests != post_digests:
        raise RederiveError(
            "rederive_identity_violation",
            "source_record_id set or snapshot_sha256 multiset changed during re-derivation",
        )


def _downstream_digest_bindings(repo_root: Path) -> list[dict[str, Any]]:
    """Grep the known production runner(s) for the inventory path constant
    and any nearby ``*_sha256`` binding, so a rewrite's blast radius is
    enumerated by a real scan rather than assumed absent."""

    bindings: list[dict[str, Any]] = []
    for relative_script in DOWNSTREAM_PRODUCTION_SCRIPTS:
        script_path = repo_root / relative_script
        if not script_path.is_file():
            continue
        for lineno, line in enumerate(script_path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(marker in line for marker in _INVENTORY_SHA_BINDING_MARKERS):
                bindings.append(
                    {
                        "file": str(relative_script),
                        "line": lineno,
                        "text": line.strip()[:200],
                    }
                )
    return bindings


def run_rederive(*, workspace: Path, run_id: str, dry_run: bool) -> dict[str, Any]:
    """Core, fully offline re-derivation. Returns the JSON-summary dict on
    success; raises ``RederiveError`` on any fail-closed condition."""

    safe_run_id = _safe_run_id(run_id)
    resolved_workspace = workspace.resolve()
    cache_root = _resolve_contained(
        resolved_workspace, resolved_workspace / CACHE_RELATIVE_ROOT / safe_run_id
    )
    inventory_path = _resolve_contained(
        resolved_workspace, resolved_workspace / INVENTORY_RELATIVE_PATH
    )
    if not inventory_path.is_file():
        raise RederiveError("rederive_inventory_missing", f"no inventory at {inventory_path}")

    pre_bytes = inventory_path.read_bytes()
    pre_inventory_sha256 = hashlib.sha256(pre_bytes).hexdigest()
    inventory = json.loads(pre_bytes.decode("utf-8"))
    if not isinstance(inventory, dict):
        raise RederiveError("rederive_inventory_invalid", "inventory must be a JSON object")

    source_records = inventory.get("source_records")
    program_items = inventory.get("program_items")
    if not isinstance(source_records, list) or not isinstance(program_items, list):
        raise RederiveError(
            "rederive_inventory_invalid", "inventory is missing source_records/program_items"
        )

    pre_records_snapshot = [dict(record) for record in source_records]

    authors_by_stem, bodies_read = _collect_cached_authors_by_stem(resolved_workspace, cache_root)

    records_by_id: dict[str, dict[str, Any]] = {}
    records_authors_added = 0
    for record in source_records:
        family_id = str(record.get("family_id", ""))
        source_record_id = str(record.get("source_record_id", ""))
        records_by_id[source_record_id] = record
        if not family_id.startswith(_CVF_OPENACCESS_FAMILY_ID_PREFIX):
            continue
        stem = _stem_from_pdf_alias(str((record.get("aliases") or [""])[0]))
        if not stem:
            continue
        authors = authors_by_stem.get(stem)
        if not authors:
            continue
        if record.get("authors") != authors:
            record["authors"] = list(authors)
            records_authors_added += 1

    program_items_authors_added = 0
    for item in program_items:
        source_record_ids = item.get("source_record_ids")
        if not isinstance(source_record_ids, list):
            continue
        field_evidence = item.get("field_evidence")
        if not isinstance(field_evidence, dict):
            continue
        title_evidence = field_evidence.get("title")
        if not isinstance(title_evidence, dict):
            continue
        for source_record_id in source_record_ids:
            record = records_by_id.get(str(source_record_id))
            if record is None or not record.get("authors"):
                continue
            new_evidence = {
                "authority": record["family_id"],
                "source_record_id": record["source_record_id"],
                "snapshot_sha256": record["snapshot_sha256"],
            }
            if field_evidence.get("authors") != new_evidence:
                field_evidence["authors"] = new_evidence
                program_items_authors_added += 1
            break

    assert_identity_preserved(pre_records_snapshot, source_records)

    changed = records_authors_added > 0 or program_items_authors_added > 0
    post_inventory_sha256 = pre_inventory_sha256
    if changed:
        digest_payload = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
        inventory["inventory_sha256"] = canonical_json_digest(digest_payload)
        serialized = (json.dumps(inventory, indent=2, sort_keys=True) + "\n").encode("utf-8")
        post_inventory_sha256 = hashlib.sha256(serialized).hexdigest()
        if not dry_run:
            _atomic_write_json(inventory_path, inventory)

    pre_ids, pre_digests = _identity_snapshot(pre_records_snapshot)
    post_ids, post_digests = _identity_snapshot(source_records)

    return {
        "status": "success",
        "dry_run": dry_run,
        "workspace": str(resolved_workspace),
        "run_id": safe_run_id,
        "cached_bodies_read": bodies_read,
        "records_scanned": len(source_records),
        "records_authors_added": records_authors_added,
        "program_items_authors_added": program_items_authors_added,
        "record_ids_stable": pre_ids == post_ids,
        "snapshot_digests_stable": pre_digests == post_digests,
        "network_calls": 0,
        "pre_inventory_sha256": pre_inventory_sha256,
        "post_inventory_sha256": post_inventory_sha256,
        "wrote_file": changed and not dry_run,
        "downstream_digest_bindings": _downstream_digest_bindings(REPO_ROOT),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True, help="conference workspace root, e.g. conferences/cvpr2026")
    parser.add_argument("--run-id", required=True, help="source-discovery cache run id to re-parse")
    parser.add_argument("--dry-run", action="store_true", help="report counts; write nothing")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    try:
        summary = run_rederive(
            workspace=Path(args.workspace), run_id=args.run_id, dry_run=args.dry_run
        )
    except RederiveError as exc:
        if args.json:
            print(json.dumps({"status": "error", "code": exc.code, "message": str(exc)}, indent=2))
        else:
            print(f"FAIL [{exc.code}]: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{'DRY-RUN: ' if summary['dry_run'] else ''}"
            f"scanned {summary['records_scanned']} records, "
            f"added authors to {summary['records_authors_added']} source_records / "
            f"{summary['program_items_authors_added']} program_items "
            f"(network_calls={summary['network_calls']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
