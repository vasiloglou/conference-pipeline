#!/usr/bin/env python3
"""Measure per-deck main-conference source diversity, at $0.

Phase 10.1.1.1 (PROD-15) found that the shared narrative source-pack
selector was admitting the *same 5 main-conference papers* into every one
of the 12 topic decks -- the first 5 distinct main-conference chunks in
physical file order, regardless of topical relevance (union 5,
intersection 5 across all 12 decks: zero diversity). This script is the
committed, reusable, conference-general measurement of that diversity, so
a pre-fix baseline and a post-fix re-measurement can be directly and
honestly compared from the same code, rather than from two different
throwaway calculations.

Method, mirroring ``scripts/measure_blueprint_content_sufficiency.py``'s own
precedent (load real on-disk artifacts, call real shared functions, read off
real output -- never hand-derive the numbers):

1. Discover topic decks by globbing
   ``<workspace>/analysis/narrative_source_packs/*.json``, using each file's
   stem as the topic id. Stems beginning with ``executive_summary`` are
   summary decks, not per-topic decks (they have no per-topic
   ``matched_papers.json`` ranking), and are skipped. No topic list,
   conference name, or year is ever hardcoded here.
2. Per topic, collect every ``chunks[].extraction_record_id`` beginning
   with the ``paper:`` prefix from that deck's own narrative source pack --
   this is the deck's *selected* set, exactly as written by the selector.
3. Resolve which of those ids are ``main_conference``-scoped independently
   of the selector: stream ``<workspace>/storage/vector-store/<run-id>/
   stores/papers/chunks.jsonl`` **once**, via the platform's own
   ``iter_chunk_jsonl``, classifying each raw chunk with the platform's own
   ``chunk_conference_scope`` -- the same function the selector itself
   uses to build its per-family buckets. This script never reads a
   self-reported scope or provenance field the selector wrote about its
   own behaviour; it re-derives scope from the corpus chunk record every
   time, so a bug in the selector's own bookkeeping could not silently
   inflate this metric. The stream stops as soon as every id collected
   across every topic has been resolved (bounded by the handful of ids the
   decks actually selected, not by store size).
4. Per topic, join against ``storage/paper-matching/<run-id>/<topic-id>/
   matched_papers.json`` (already-paid GPT-Researcher topical matching)
   directly off disk -- never through a field the selector emitted -- to
   split the deck's selected main-conference ids into ``priority_source_ids``
   (present in that topic's matched-papers set) and
   ``keyword_backfill_source_ids`` (the remainder). ``matched_papers.json``'s
   ids are a mix of bare sha256 hex64 hashes, arXiv ids, and slugs; only the
   64-character lowercase hex ids join to the vector store's ``paper:<hash>``
   identity, so only those are used for the join.

**What this measures, and what it does not:** a diversity improvement
reported here is a *topical-correctness and attribution* result -- it
proves the selector stopped handing every deck the same 5 arbitrary
papers, and shows which of that gain came from the authoritative
topically-matched signal versus the weaker keyword-backfill path. It is
**not** evidence about ledger eligibility, the minimum-eligible-source
threshold, or the named-people ceiling -- those are separate,
already-committed measurements
(``scripts/measure_blueprint_content_sufficiency.py``,
``scripts/measure_ledger_author_coverage.py``) with their own thresholds,
which this script never restates or re-derives.

$0 / no provider calls: refuses to run if ``anthropic``/``openai`` is
already imported, imports no provider SDK, and the receipt records
``provider_calls_made: false`` / ``live_provider_calls_made: false``.

Usage::

    .venv/bin/python scripts/measure_source_pack_topic_diversity.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production --json

    .venv/bin/python scripts/measure_source_pack_topic_diversity.py \\
        --workspace conferences/cvpr2026 --run-id cvpr2026-live-production \\
        --baseline-receipt storage/workflow-runs/cvpr2026-live-production/source_pack_topic_diversity_pre_fix.json \\
        --fail-under-baseline
"""

from __future__ import annotations

import argparse
import json
import re
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
# scripts/measure_blueprint_content_sufficiency.py and
# scripts/measure_ledger_author_coverage.py verbatim in shape.
_FORBIDDEN_PROVIDER_MODULES = ("anthropic", "openai")
for _forbidden in _FORBIDDEN_PROVIDER_MODULES:
    if _forbidden in sys.modules:
        raise RuntimeError(
            f"{_forbidden} is already imported in this process -- refusing to run "
            "the $0 source-pack topic-diversity measurement path alongside a live "
            "provider SDK."
        )

from conference_pipeline.analysis.source_pack_selection import chunk_conference_scope  # noqa: E402
from conference_pipeline.indexing import corpus_vector_store as platform_corpus_vector_store  # noqa: E402
from conference_pipeline.indexing.corpus_vector_store import iter_chunk_jsonl  # noqa: E402

SCHEMA_VERSION = 1

# The prefix a papers-store extraction_record_id carries, both in a
# narrative source pack's compact chunk records and in the raw
# chunks.jsonl's own ``artifact_id`` field -- shared vocabulary, not
# re-derived independently for the two reads.
PAPER_ID_PREFIX = "paper:"

# 64-character lowercase hex is the only matched_papers.json paper_id shape
# that joins to a papers-store paper:<hash> identity; arXiv ids, OpenReview
# slugs, and workshop-prefixed ids never match this and are excluded from
# the join, not silently coerced.
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

# Topic-pack stems that are summary decks, not per-topic decks -- they have
# no per-topic matched_papers.json ranking to join against, and are skipped
# during discovery rather than reported as zero-diversity topics.
_SUMMARY_STEM_PREFIX = "executive_summary"


def discover_topic_ids(workspace: Path) -> list[str]:
    """Glob the workspace's narrative source packs for topic ids.

    Never hardcodes a topic list, conference name, or year -- any workspace
    with the same on-disk shape is measured identically.
    """

    packs_dir = workspace / "analysis" / "narrative_source_packs"
    if not packs_dir.is_dir():
        return []
    return sorted(
        path.stem
        for path in packs_dir.glob("*.json")
        if not path.stem.startswith(_SUMMARY_STEM_PREFIX)
    )


def _load_source_pack_chunks(workspace: Path, topic_id: str) -> list[dict[str, Any]]:
    path = workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []
    chunks = payload.get("chunks")
    return chunks if isinstance(chunks, list) else []


def _collect_paper_prefixed_ids(chunks: list[dict[str, Any]]) -> set[str]:
    """Collect every extraction_record_id this deck actually selected.

    Kept in the deck's own full ``paper:<hash>`` form here (not yet stripped
    or scope-filtered) so the same id can be looked up directly against the
    raw chunk's own ``artifact_id`` field during scope resolution below.
    """

    ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        record_id = chunk.get("extraction_record_id")
        if isinstance(record_id, str) and record_id.startswith(PAPER_ID_PREFIX):
            ids.add(record_id)
    return ids


def resolve_conference_scopes(chunks_path: Path, collected_ids: set[str]) -> dict[str, str]:
    """Resolve each collected id's conference_scope from the raw corpus chunks.

    Streams ``chunks_path`` once via ``iter_chunk_jsonl`` and classifies each
    raw chunk with ``chunk_conference_scope`` -- the platform's own
    classification function, re-derived from the corpus record itself, never
    read as a self-report from the selector or the narrative pack. Only
    records a scope for ids already in ``collected_ids`` (the union of ids
    every topic actually selected), and stops as soon as every one of them
    has been resolved.
    """

    resolved: dict[str, str] = {}
    if not collected_ids or not chunks_path.is_file():
        return resolved
    for raw_chunk in iter_chunk_jsonl(chunks_path):
        artifact_id = raw_chunk.get("artifact_id")
        if (
            isinstance(artifact_id, str)
            and artifact_id in collected_ids
            and artifact_id not in resolved
        ):
            resolved[artifact_id] = chunk_conference_scope(raw_chunk)
            if len(resolved) == len(collected_ids):
                break
    return resolved


def load_matched_paper_hex_ids(workspace: Path, run_id: str, topic_id: str) -> set[str]:
    """Load a topic's joinable (hex64) matched-papers candidate ids, tolerantly.

    A missing file, unreadable JSON, non-dict payload, or a missing/non-list
    ``papers`` key (falling back to ``matched_papers``) each yield an empty
    set -- never an exception. This is deliberately independent of the
    selector: it re-reads ``matched_papers.json`` from disk directly rather
    than trusting any priority-set field the selector or runner may have
    already computed and passed along.
    """

    path = workspace / "storage" / "paper-matching" / run_id / topic_id / "matched_papers.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(payload, dict):
        return set()
    papers = payload.get("papers")
    if not isinstance(papers, list):
        papers = payload.get("matched_papers")
    if not isinstance(papers, list):
        return set()
    hex_ids: set[str] = set()
    for entry in papers:
        if not isinstance(entry, dict):
            continue
        paper_id = entry.get("paper_id")
        if isinstance(paper_id, str) and _HEX64_RE.match(paper_id):
            hex_ids.add(paper_id)
    return hex_ids


def _measure_topic(
    topic_id: str,
    *,
    collected_ids: set[str],
    resolved_scopes: dict[str, str],
    matched_hex_ids: set[str],
) -> dict[str, Any]:
    # Fail-open to "main_conference" for any collected id this stream never
    # actually saw in chunks.jsonl, mirroring chunk_conference_scope's own
    # fail-open philosophy for a missing/unrecognized scope value -- a
    # narrative pack entry with no corpus record to classify is not silently
    # dropped from the metric.
    main_conference_ids = sorted(
        record_id[len(PAPER_ID_PREFIX) :]
        for record_id in collected_ids
        if resolved_scopes.get(record_id, "main_conference") == "main_conference"
    )
    priority_source_ids = sorted(
        source_id for source_id in main_conference_ids if source_id in matched_hex_ids
    )
    keyword_backfill_source_ids = sorted(
        source_id for source_id in main_conference_ids if source_id not in matched_hex_ids
    )
    return {
        "topic_id": topic_id,
        "main_conference_source_ids": main_conference_ids,
        "main_conference_source_count": len(main_conference_ids),
        "priority_source_ids": priority_source_ids,
        "keyword_backfill_source_ids": keyword_backfill_source_ids,
        "joinable_matched_paper_count": len(matched_hex_ids),
    }


def _compute_diversity(topics: list[dict[str, Any]]) -> dict[str, Any]:
    non_empty_topics = [topic for topic in topics if topic["main_conference_source_count"] > 0]

    union_ids = sorted(
        set().union(*(set(topic["main_conference_source_ids"]) for topic in topics))
    )
    # Intersection is computed only across topics with at least one
    # main-conference id -- an empty-selection topic must not be allowed to
    # collapse the intersection to the empty set by definitional accident,
    # which would misreport a real zero-diversity finding as a data gap (or
    # vice versa). intersection_basis_topic_count records exactly how many
    # topics that intersection was computed over.
    intersection_ids = (
        sorted(set.intersection(*(set(topic["main_conference_source_ids"]) for topic in non_empty_topics)))
        if non_empty_topics
        else []
    )

    return {
        "union_count": len(union_ids),
        "union_source_ids": union_ids,
        "intersection_count": len(intersection_ids),
        "intersection_source_ids": intersection_ids,
        "intersection_basis_topic_count": len(non_empty_topics),
        "topics_with_zero_main_conference": sorted(
            topic["topic_id"] for topic in topics if topic["main_conference_source_count"] == 0
        ),
        "topics_entirely_backfilled": sorted(
            topic["topic_id"]
            for topic in topics
            if topic["main_conference_source_count"] > 0 and not topic["priority_source_ids"]
        ),
        "topics_fully_priority_seeded": sorted(
            topic["topic_id"]
            for topic in topics
            if topic["main_conference_source_count"] > 0 and not topic["keyword_backfill_source_ids"]
        ),
        "priority_selected_total": sum(len(topic["priority_source_ids"]) for topic in topics),
        "keyword_backfill_total": sum(len(topic["keyword_backfill_source_ids"]) for topic in topics),
    }


def measure_workspace(workspace: Path, run_id: str) -> dict[str, Any]:
    """Measure every topic deck under a workspace, at $0, from disk only."""

    workspace = workspace.expanduser().resolve()
    topic_ids = discover_topic_ids(workspace)

    per_topic_chunks = {topic_id: _load_source_pack_chunks(workspace, topic_id) for topic_id in topic_ids}
    per_topic_collected = {
        topic_id: _collect_paper_prefixed_ids(chunks) for topic_id, chunks in per_topic_chunks.items()
    }
    collected_superset: set[str] = set().union(*per_topic_collected.values()) if per_topic_collected else set()

    chunks_path = workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    resolved_scopes = resolve_conference_scopes(chunks_path, collected_superset)

    topics = [
        _measure_topic(
            topic_id,
            collected_ids=per_topic_collected[topic_id],
            resolved_scopes=resolved_scopes,
            matched_hex_ids=load_matched_paper_hex_ids(workspace, run_id, topic_id),
        )
        for topic_id in topic_ids
    ]

    return {
        "topic_count": len(topics),
        "topics": topics,
        "diversity": _compute_diversity(topics),
    }


def _measured_input_relpaths(workspace: Path, run_id: str, topic_ids: list[str]) -> list[str]:
    paths: list[Path] = []
    packs_dir = workspace / "analysis" / "narrative_source_packs"
    for topic_id in topic_ids:
        pack_path = packs_dir / f"{topic_id}.json"
        if pack_path.is_file():
            paths.append(pack_path)
        matched_path = workspace / "storage" / "paper-matching" / run_id / topic_id / "matched_papers.json"
        if matched_path.is_file():
            paths.append(matched_path)
    chunks_path = workspace / "storage" / "vector-store" / run_id / "stores" / "papers" / "chunks.jsonl"
    if chunks_path.is_file():
        paths.append(chunks_path)
    return sorted(platform_corpus_vector_store.workspace_relpath(workspace, path) for path in paths)


def write_diversity_receipt(
    path: Path,
    *,
    workspace: Path,
    run_id: str,
    measurement: dict[str, Any],
    baseline_receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    """Write an atomic JSON diversity receipt, optionally compared to a baseline.

    ``diversity_improved_vs_baseline`` is true only when this measurement's
    ``union_count`` is strictly greater than the baseline's, AND this
    measurement's ``intersection_count`` is strictly less than the
    baseline's. Without a baseline it is left ``None`` (null in the written
    JSON) rather than defaulted to a value that could be mistaken for a
    real comparison.
    """

    workspace = workspace.expanduser().resolve()
    diversity = measurement["diversity"]

    diversity_improved_vs_baseline: bool | None = None
    baseline_union_count: int | None = None
    baseline_intersection_count: int | None = None
    if baseline_receipt is not None:
        baseline_diversity = baseline_receipt.get("diversity") or {}
        baseline_union_count = baseline_diversity.get("union_count")
        baseline_intersection_count = baseline_diversity.get("intersection_count")
        diversity_improved_vs_baseline = bool(
            isinstance(baseline_union_count, int)
            and isinstance(baseline_intersection_count, int)
            and diversity["union_count"] > baseline_union_count
            and diversity["intersection_count"] < baseline_intersection_count
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": platform_corpus_vector_store.utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "topic_count": measurement["topic_count"],
        "topics": measurement["topics"],
        "diversity": diversity,
        "diversity_improved_vs_baseline": diversity_improved_vs_baseline,
        "baseline_union_count": baseline_union_count,
        "baseline_intersection_count": baseline_intersection_count,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "inputs": _measured_input_relpaths(
            workspace, run_id, [topic["topic_id"] for topic in measurement["topics"]]
        ),
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
            "source_pack_topic_diversity.json under the workspace)"
        ),
    )
    parser.add_argument(
        "--baseline-receipt",
        type=Path,
        default=None,
        help="Path to a previously written receipt of this same schema to compare against.",
    )
    parser.add_argument(
        "--fail-under-baseline",
        action="store_true",
        help=(
            "Exit 1 unless diversity_improved_vs_baseline is true (requires "
            "--baseline-receipt; without one the comparison is null and this exits 1)."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()

    measurement = measure_workspace(workspace, args.run_id)

    baseline_receipt: dict[str, Any] | None = None
    if args.baseline_receipt is not None:
        baseline_path = args.baseline_receipt
        if not baseline_path.is_absolute():
            baseline_path = workspace / baseline_path
        baseline_receipt = json.loads(baseline_path.read_text(encoding="utf-8"))

    output_receipt = args.output_receipt
    if output_receipt is None:
        output_receipt = workspace / "storage" / "workflow-runs" / args.run_id / "source_pack_topic_diversity.json"
    elif not output_receipt.is_absolute():
        output_receipt = workspace / output_receipt

    receipt = write_diversity_receipt(
        output_receipt,
        workspace=workspace,
        run_id=args.run_id,
        measurement=measurement,
        baseline_receipt=baseline_receipt,
    )

    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    else:
        diversity = receipt["diversity"]
        print(
            f"Wrote source-pack topic-diversity receipt for {receipt['topic_count']} topic(s) to {output_receipt}"
        )
        print(
            f"  union={diversity['union_count']} intersection={diversity['intersection_count']}"
            f" (basis={diversity['intersection_basis_topic_count']} topics)"
            f" priority_selected_total={diversity['priority_selected_total']}"
            f" keyword_backfill_total={diversity['keyword_backfill_total']}"
        )
        if receipt["diversity_improved_vs_baseline"] is not None:
            print(
                f"  vs baseline union={receipt['baseline_union_count']}"
                f" intersection={receipt['baseline_intersection_count']}:"
                f" diversity_improved_vs_baseline={receipt['diversity_improved_vs_baseline']}"
            )

    if args.fail_under_baseline:
        return 0 if receipt["diversity_improved_vs_baseline"] is True else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
