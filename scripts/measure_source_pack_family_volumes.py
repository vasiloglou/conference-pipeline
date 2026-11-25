#!/usr/bin/env python3
"""Measure per-topic, per-source-family keyword-match volumes, at $0.

Phase 10.1.1's `MAIN_CONFERENCE_FLOOR` quota constant (set in a later plan)
must be derived from measured per-topic keyword-match volumes, not a guessed
constant (CONTEXT.md, ROADMAP done gate 2). This script produces that
evidence: for every topic and every vector-store ("papers"/"artifacts"), it
counts how many chunks match the topic's own keyword set, stratified by
`conference_scope` family ("main_conference" vs. "workshop"), and reports
both the raw matching-chunk count and the distinct-source count (the pack
selector de-dups by source artifact, so a topic's true "how many independent
papers could I draw from" number is the distinct-source count, not the
chunk count).

This is a measurement CLI, not a selection-logic change: it mirrors
`build_analysis_source_packs`' own scan/dedup semantics (same keyword
derivation via `load_analysis_topic_specs`, same lowercase substring test,
same per-topic `extraction_record_id` de-dup, same `_analysis_compact_chunk`
quote text) but counts rather than composes a bounded pack, and it never
writes to `analysis/narrative_source_packs/` or any selection-facing path.

Family classification is delegated entirely to the shared, already-computed
`conference_scope` field stamped by `derive_conference_scope()`
(`src/conference_pipeline/indexing/corpus_vector_store.py`) -- this script
never re-derives it. `resolve_chunk_family` reads the top-level field first,
falls back to `source_context.conference_scope`, and fails OPEN to
"main_conference" for any missing or unrecognized value, so an
unclassifiable chunk is never silently dropped from any denominator. The
receipt also tracks `raw_unclassified_count` per store as an explicit,
separate diagnostic of how many chunks landed in "main_conference" via
fail-open rather than a true stamped value, so the two family totals can
always be checked to sum to the store's total chunk count.

$0 / no provider calls: refuses to run if `anthropic`/`openai` is already
imported, imports no provider SDK, and the receipt records
`provider_calls_made: false` / `live_provider_calls_made: false`.

Usage::

    .venv/bin/python scripts/measure_source_pack_family_volumes.py \\
        --workspace conferences/<slug> --run-id <run-id> --json

Add ``--output-receipt <path>`` to override the default receipt location
(``storage/workflow-runs/<run-id>/source_pack_family_volumes.json`` under
the workspace).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Refuse, at import time, to have any provider SDK anywhere on the path this
# script pulls in -- if a future edit accidentally reintroduces a provider
# call, this trips immediately rather than silently spending money. Mirrors
# scripts/regenerate_narrative_ledgers_without_synthesis.py verbatim in shape.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the $0 family-volume measurement path alongside a live provider SDK."
        )

from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402

RUNNER_PATH = REPO_ROOT / "scripts" / "run_cvpr2026_production.py"

# The two conference_scope families derive_conference_scope() ever stamps
# (src/conference_pipeline/indexing/corpus_vector_store.py:451-461). Anything
# else -- missing, blank, or an unrecognized string -- fails open to
# "main_conference" per resolve_chunk_family below.
CONFERENCE_SCOPE_FAMILIES = ("main_conference", "workshop")


def resolve_chunk_family(chunk: dict[str, Any]) -> str:
    """Resolve a chunk's conference-scope family, failing open to main_conference.

    Reads the shared, already-computed ``conference_scope`` value verbatim
    (never re-derives it): top-level field first, then
    ``source_context.conference_scope``. Any missing or unrecognized value
    fails OPEN to "main_conference" rather than being silently dropped from
    any denominator -- an unclassifiable chunk must never disappear from the
    corpus (RESEARCH Security Domain V5).
    """

    value = chunk.get("conference_scope")
    if not value:
        source_context = chunk.get("source_context")
        if isinstance(source_context, dict):
            value = source_context.get("conference_scope")
    value = str(value or "")
    if value not in CONFERENCE_SCOPE_FAMILIES:
        return "main_conference"
    return value


def _raw_conference_scope_is_missing_or_unrecognized(chunk: dict[str, Any]) -> bool:
    """True when the chunk's own stamped value required the fail-open rule."""

    value = chunk.get("conference_scope")
    if not value:
        source_context = chunk.get("source_context")
        if isinstance(source_context, dict):
            value = source_context.get("conference_scope")
    return str(value or "") not in CONFERENCE_SCOPE_FAMILIES


def measure_store_family_volumes(
    chunks: Iterable[dict[str, Any]],
    *,
    topic_specs: tuple[dict[str, Any], ...],
    compact_chunk_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Pure, dependency-injected per-store measurement (no file I/O, no live workspace).

    Mirrors ``build_analysis_source_packs``' scan semantics -- same lowercase
    substring keyword test against the same compacted ``quote`` text, same
    per-topic ``extraction_record_id`` de-dup for the distinct-source count --
    but counts every match (stratified by family) instead of stopping at a
    bounded per-topic cap, and never composes or writes a pack.

    ``compact_chunk_fn`` is injected (rather than importing the CVPR2026
    runner directly) so this function -- and its unit tests -- never require
    a live workspace; the CLI wires in the runner's own
    ``_analysis_compact_chunk`` for the real measurement run.
    """

    specs_by_id = {str(spec["id"]): spec for spec in topic_specs}
    matching_chunk_counts: dict[str, dict[str, int]] = {
        topic_id: {family: 0 for family in CONFERENCE_SCOPE_FAMILIES}
        for topic_id in specs_by_id
    }
    distinct_sources: dict[str, dict[str, set[str]]] = {
        topic_id: {family: set() for family in CONFERENCE_SCOPE_FAMILIES}
        for topic_id in specs_by_id
    }
    first_match_index: dict[str, dict[str, int | None]] = {
        topic_id: {family: None for family in CONFERENCE_SCOPE_FAMILIES}
        for topic_id in specs_by_id
    }
    family_chunk_totals: dict[str, int] = {family: 0 for family in CONFERENCE_SCOPE_FAMILIES}
    raw_unclassified_count = 0
    total_chunk_count = 0

    for chunk_index, chunk in enumerate(chunks):
        total_chunk_count += 1
        family = resolve_chunk_family(chunk)
        family_chunk_totals[family] += 1
        if _raw_conference_scope_is_missing_or_unrecognized(chunk):
            raw_unclassified_count += 1
        compact = compact_chunk_fn(chunk)
        text_lower = str(compact.get("quote") or "").lower()
        source_id = str(compact.get("extraction_record_id") or "")
        for topic_id, spec in specs_by_id.items():
            keywords = [str(item).lower() for item in (spec.get("keywords") or [])]
            if any(keyword and keyword in text_lower for keyword in keywords):
                matching_chunk_counts[topic_id][family] += 1
                if source_id:
                    distinct_sources[topic_id][family].add(source_id)
                if first_match_index[topic_id][family] is None:
                    first_match_index[topic_id][family] = chunk_index

    # Denominator-integrity invariant: every chunk lands in exactly one
    # family bucket (fail-open still counts), so the two family totals must
    # sum to the total scanned -- coverage can never be inflated by a
    # shrinking denominator.
    assert sum(family_chunk_totals.values()) == total_chunk_count

    per_topic = {
        topic_id: {
            family: {
                "matching_chunk_count": matching_chunk_counts[topic_id][family],
                "distinct_source_count": len(distinct_sources[topic_id][family]),
                "first_match_chunk_index": first_match_index[topic_id][family],
            }
            for family in CONFERENCE_SCOPE_FAMILIES
        }
        for topic_id in specs_by_id
    }
    return {
        "per_topic": per_topic,
        "total_chunk_count": total_chunk_count,
        "family_chunk_counts": dict(family_chunk_totals),
        "raw_unclassified_count": raw_unclassified_count,
    }


def _load_runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "cvpr2026_runner_for_family_volume_measurement", RUNNER_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure_family_volumes(
    *,
    runner: Any,
    workspace: Path,
    index_digest: dict[str, Any],
    topic_specs: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Measure every ("papers", "artifacts") store named in the index digest."""

    stores = index_digest.get("stores")
    if not isinstance(stores, dict):
        raise RuntimeError("analysis_index_digest_missing_stores")

    store_results: dict[str, Any] = {}
    for store_name in ("papers", "artifacts"):
        store = stores.get(store_name)
        if not isinstance(store, dict):
            continue
        chunks_path, error = platform_corpus_vector_store.resolve_workspace_member(
            workspace, store.get("chunks_path")
        )
        if error or chunks_path is None or not chunks_path.exists():
            raise RuntimeError(f"analysis_{store_name}_chunks_missing")
        store_results[store_name] = measure_store_family_volumes(
            platform_corpus_vector_store.iter_chunk_jsonl(chunks_path),
            topic_specs=topic_specs,
            compact_chunk_fn=runner._analysis_compact_chunk,
        )

    topics: dict[str, Any] = {}
    for spec in topic_specs:
        topic_id = str(spec["id"])
        topics[topic_id] = {
            "title": str(spec["title"]),
            "stores": {
                store_name: result["per_topic"][topic_id]
                for store_name, result in store_results.items()
            },
        }

    store_totals = {
        store_name: {
            "total_chunk_count": result["total_chunk_count"],
            "family_chunk_counts": result["family_chunk_counts"],
            "raw_unclassified_count": result["raw_unclassified_count"],
        }
        for store_name, result in store_results.items()
    }
    return {"topics": topics, "store_totals": store_totals}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        default=None,
        help=(
            "Receipt path (default: storage/workflow-runs/<run-id>/"
            "source_pack_family_volumes.json under the workspace)"
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runner = _load_runner_module()
    workspace = args.workspace.expanduser().resolve()
    run_id = runner.validate_run_id(args.run_id)
    target = runner.run_dir(workspace, run_id)

    # --- Mirror the regeneration script's own Stage 6 admission gate -------
    stage6 = runner.STAGES["corpus-inventory-index"]
    stage6_receipt_path = target / stage6.receipt_filename
    stage6_receipt, stage6_error = runner._analysis_load_json(stage6_receipt_path)
    if stage6_error or not isinstance(stage6_receipt, dict):
        print(f"ERROR: stage6 receipt missing or malformed at {stage6_receipt_path}", file=sys.stderr)
        return 2
    stage6_checks = stage6_receipt.get("checks") if isinstance(stage6_receipt.get("checks"), dict) else {}
    if (
        stage6_receipt.get("status") != "succeeded"
        or not stage6_checks.get("vector_embeddings_written")
        or stage6_checks.get("chunk_counts") != stage6_checks.get("embedded_chunk_counts")
    ):
        print(
            "ERROR: stage6 vector evidence required (status/vector_embeddings_written/chunk_counts mismatch)",
            file=sys.stderr,
        )
        return 2

    index_digest_path = workspace / runner.VECTOR_STORE_DIR / run_id / "index_digest.json"
    index_digest, index_digest_error = runner._analysis_load_json(index_digest_path)
    if (
        index_digest_error
        or not isinstance(index_digest, dict)
        or runner._file_sha256(index_digest_path) != str(stage6_checks.get("index_digest_sha256") or "")
    ):
        print(f"ERROR: stage6 index_digest stale or missing at {index_digest_path}", file=sys.stderr)
        return 2

    try:
        topic_specs = runner.load_analysis_topic_specs(workspace, run_id=run_id)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    measurement = measure_family_volumes(
        runner=runner,
        workspace=workspace,
        index_digest=index_digest,
        topic_specs=topic_specs,
    )

    output_receipt = args.output_receipt
    if output_receipt is None:
        output_receipt = target / "source_pack_family_volumes.json"
    elif not output_receipt.is_absolute():
        output_receipt = workspace / output_receipt

    receipt = {
        "schema_version": 1,
        "generated_at": platform_corpus_vector_store.utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "conference_scope_families": list(CONFERENCE_SCOPE_FAMILIES),
        "topics": measurement["topics"],
        "store_totals": measurement["store_totals"],
    }
    runner.atomic_write_json(output_receipt, receipt)

    if args.json:
        print(json.dumps(receipt, indent=2, default=str))
    else:
        print(
            f"Wrote family-volume receipt for {len(topic_specs)} topics to {output_receipt}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
