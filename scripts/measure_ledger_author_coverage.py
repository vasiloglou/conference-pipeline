#!/usr/bin/env python3
"""Measure ledger-level author coverage per source family, at $0.

Phase 10.1-10 measured ledger-level author coverage with an uncommitted
scratch script that was then deleted -- that measurement cannot be
reproduced, and it must not happen twice. This script is the committed
replacement.

ROADMAP done gate 2 requires this phase's fix to be measured at the LEDGER
level, per family -- a source-pack-level or chunk-level claim is explicitly
rejected evidence, because a fix that lands in the source pack can still be
lost before it reaches the ledger a deck actually builds slides from
(``decks/<id>/speaker_author_work_ledger.json``). This script walks every
deck's ledger under a workspace and reports, PER FAMILY (main-conference
papers, workshop papers, artifacts, and an explicit ``unclassified`` bucket
for anything that cannot be resolved), how many entries carry at least one
resolved author or speaker and how many carry a resolved title -- never
blended into one aggregate rate.

Family resolution
------------------
A ledger entry's own ``source_family`` field is only two-valued
(``"papers"`` / ``"artifacts"``) -- the canonical reference-id space collapses
main-conference and workshop papers to the same ``paper_abstract:`` prefix
(see ``conference_pipeline.analysis.topic_reference_validation``'s own
module docstring), and neither the topic-reference-ledger builder nor the
narrative-pack compaction step (``COMPACT_METADATA_KEYS`` in
``conference_pipeline.analysis.narrative``) currently propagates the chunk-
level ``conference_scope`` field onto ledger entries. So an entry's
``source_family`` alone cannot answer main-vs-workshop.

The one signal that DOES survive to the deck-level ledger is
``evidence_paths`` -- the on-disk artifact path(s) each entry's evidence
came from. Workshop papers are consistently stored under a ``workshops/``
path segment (mirroring
``conference_pipeline.indexing.corpus_vector_store.derive_conference_scope``'s
own path-based check); main-conference papers are not. An entry whose only
evidence path is the generic topic-reference-ledger file itself (true for
title-matched "matched_paper" entries, which carry no chunk-level evidence at
all) has NO positive scope signal either way, so it is placed in
``unclassified`` -- never silently assumed main-conference, because an
assumed default is exactly how a coverage number gets inflated (this phase's
own threat register, T-10.1.1-13).

This module defines no coverage floor and enforces no threshold --
``scripts/audit_corpus_source_metadata.py`` owns floors and is not modified
or reused here. This script only measures and reports.

$0 / no provider calls: refuses to run if ``anthropic``/``openai`` is already
imported, imports no provider SDK, and the receipt records
``provider_calls_made: false`` / ``live_provider_calls_made: false``.

Usage::

    .venv/bin/python scripts/measure_ledger_author_coverage.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Refuse, at import time, to have any provider SDK anywhere on the path this
# script pulls in -- if a future edit accidentally reintroduces a provider
# call, this trips immediately rather than silently spending money. Mirrors
# scripts/measure_source_pack_family_volumes.py verbatim in shape.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the $0 ledger author-coverage measurement path alongside a live provider SDK."
        )

from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402

# The four buckets every ledger entry lands in -- exactly one each, never
# zero, so bucket totals always sum to the entries scanned (no entry is ever
# silently dropped from a denominator).
BUCKET_MAIN_CONFERENCE_PAPERS = "main_conference_papers"
BUCKET_WORKSHOP_PAPERS = "workshop_papers"
BUCKET_ARTIFACTS = "artifacts"
BUCKET_UNCLASSIFIED = "unclassified"
BUCKETS = (
    BUCKET_MAIN_CONFERENCE_PAPERS,
    BUCKET_WORKSHOP_PAPERS,
    BUCKET_ARTIFACTS,
    BUCKET_UNCLASSIFIED,
)

# A ledger entry's evidence_paths always includes this generic self-reference
# (the topic_reference_ledgers file the entry was drafted from) alongside any
# real, per-entry evidence path. Stripping it out is what lets us tell "no
# real evidence at all" (matched_paper entries) apart from "real evidence,
# just not under a workshop path" (main-conference chunk-derived entries).
_LEDGER_SELF_REFERENCE_PATH_PREFIX = "analysis/topic_reference_ledgers/"

# Mirrors scripts/presentations/generate_editorial_blueprint.py's/
# scripts/presentations/verify_presentation_content.py's own
# PLACEHOLDER_PEOPLE vocabulary (not imported from either -- this script must
# not depend on, or be modified alongside, the audit/blueprint files this
# plan explicitly leaves untouched).
_PLACEHOLDER_PERSON_NAMES = frozenset({"unknown", "n/a", "na", "none", "anonymous", "unnamed", "unidentified", ""})


def _is_resolved_person(name: Any) -> bool:
    if not isinstance(name, str):
        return False
    return name.strip().lower() not in _PLACEHOLDER_PERSON_NAMES


def _has_resolved_person(names: Any) -> bool:
    if not isinstance(names, list):
        return False
    return any(_is_resolved_person(name) for name in names)


def resolve_entry_bucket(entry: dict[str, Any]) -> str:
    """Resolve one ledger entry's per-family bucket.

    ``artifacts``-family entries bucket as ``artifacts`` outright (no
    main/workshop split applies to that family). ``papers``-family entries
    are split by evidence-path signal: a workshop path segment -> workshop
    papers; some other real (non-self-reference) evidence path -> main-
    conference papers; no real evidence path at all -> unclassified. Any
    other/missing ``source_family`` value -> unclassified.
    """

    source_family = str(entry.get("source_family") or "")
    if source_family == BUCKET_ARTIFACTS:
        return BUCKET_ARTIFACTS
    if source_family != "papers":
        return BUCKET_UNCLASSIFIED

    evidence_paths = entry.get("evidence_paths")
    real_paths = [
        path
        for path in (evidence_paths if isinstance(evidence_paths, list) else [])
        if isinstance(path, str) and not path.startswith(_LEDGER_SELF_REFERENCE_PATH_PREFIX)
    ]
    if any("workshops/" in path for path in real_paths):
        return BUCKET_WORKSHOP_PAPERS
    if real_paths:
        return BUCKET_MAIN_CONFERENCE_PAPERS
    return BUCKET_UNCLASSIFIED


def _new_bucket() -> dict[str, int]:
    return {"total": 0, "authors_present": 0, "title_present": 0}


def measure_ledger(entries: Any) -> dict[str, dict[str, int]]:
    """Count one ledger's entries into the four per-family buckets.

    Never drops an entry: every dict-shaped entry lands in exactly one of
    ``BUCKETS``, so ``sum(bucket["total"] for bucket in result.values())``
    always equals the number of dict-shaped entries scanned.
    """

    buckets = {name: _new_bucket() for name in BUCKETS}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        bucket = buckets[resolve_entry_bucket(entry)]
        bucket["total"] += 1
        if _has_resolved_person(entry.get("authors")) or _has_resolved_person(entry.get("speakers")):
            bucket["authors_present"] += 1
        title = entry.get("resolved_title")
        if isinstance(title, str) and title.strip():
            bucket["title_present"] += 1
    return buckets


def roll_up_buckets(per_deck: dict[str, dict[str, dict[str, int]]]) -> dict[str, dict[str, int]]:
    """Sum every deck's per-bucket counts into one workspace-level total per bucket."""

    totals = {name: _new_bucket() for name in BUCKETS}
    for deck_buckets in per_deck.values():
        for name in BUCKETS:
            deck_bucket = deck_buckets.get(name, {})
            for key in ("total", "authors_present", "title_present"):
                totals[name][key] += int(deck_bucket.get(key) or 0)
    return totals


def measure_workspace(workspace: Path) -> dict[str, Any]:
    """Walk every ``decks/<id>/speaker_author_work_ledger.json`` under a workspace.

    Tolerates a missing or unreadable ledger by recording it in
    ``skipped_decks`` rather than raising -- a deck with no ledger yet
    contributes nothing to any bucket's denominator (it is not scanned at
    all, so it cannot silently deflate or inflate a rate).
    """

    workspace = workspace.expanduser().resolve()
    decks_dir = workspace / "decks"
    per_deck: dict[str, dict[str, dict[str, int]]] = {}
    skipped_decks: list[dict[str, str]] = []

    deck_dirs = sorted(child for child in decks_dir.iterdir() if child.is_dir()) if decks_dir.is_dir() else []
    for deck_dir in deck_dirs:
        ledger_path = deck_dir / "speaker_author_work_ledger.json"
        if not ledger_path.is_file():
            skipped_decks.append({"deck_id": deck_dir.name, "reason": "ledger_missing"})
            continue
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            skipped_decks.append({"deck_id": deck_dir.name, "reason": f"ledger_unreadable:{exc}"})
            continue
        entries = payload.get("entries") if isinstance(payload, dict) else None
        per_deck[deck_dir.name] = measure_ledger(entries)

    return {
        "per_deck": per_deck,
        "workspace_totals": roll_up_buckets(per_deck),
        "skipped_decks": skipped_decks,
    }


def _measured_input_paths(workspace: Path) -> list[str]:
    decks_dir = workspace / "decks"
    if not decks_dir.is_dir():
        return []
    return sorted(
        platform_corpus_vector_store.workspace_relpath(workspace, path)
        for path in decks_dir.glob("*/speaker_author_work_ledger.json")
    )


def write_coverage_receipt(
    path: Path,
    *,
    workspace: Path,
    run_id: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    """Write an atomic JSON coverage receipt. Never includes author names,
    titles, or any other corpus free text -- counts and paths/digests only."""

    workspace = workspace.expanduser().resolve()
    inputs = {
        relpath: platform_corpus_vector_store.sha256_file(workspace / relpath)
        for relpath in _measured_input_paths(workspace)
    }
    receipt = {
        "schema_version": 1,
        "generated_at": platform_corpus_vector_store.utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "buckets": list(BUCKETS),
        "per_deck": measurement.get("per_deck", {}),
        "workspace_totals": measurement.get("workspace_totals", {}),
        "skipped_decks": measurement.get("skipped_decks", []),
        "inputs": inputs,
    }
    platform_corpus_vector_store.atomic_write_json(path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=None,
        help=(
            "Receipt path (default: storage/workflow-runs/<run-id>/"
            "ledger_author_coverage.json under the workspace)"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    measurement = measure_workspace(workspace)

    output_receipt = args.output_receipt
    if output_receipt is None:
        output_receipt = workspace / "storage" / "workflow-runs" / args.run_id / "ledger_author_coverage.json"
    elif not output_receipt.is_absolute():
        output_receipt = workspace / output_receipt

    receipt = write_coverage_receipt(
        output_receipt,
        workspace=workspace,
        run_id=args.run_id,
        measurement=measurement,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    else:
        print(f"Wrote ledger author-coverage receipt for {len(measurement['per_deck'])} deck(s) to {output_receipt}")
        for bucket_name in BUCKETS:
            bucket = receipt["workspace_totals"][bucket_name]
            print(f"  {bucket_name}: authors {bucket['authors_present']}/{bucket['total']}  title {bucket['title_present']}/{bucket['total']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
