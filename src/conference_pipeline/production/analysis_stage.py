"""Stage 7: analysis (deterministic source packs + governed deep narratives), shared by every runner.

Gates, does not orchestrate. The post-full-read chain scripts (topic
strategy, 14A full-read extraction, 07B question selection, 07C deep-dive
research, 07D/07E/07H topic reconciliation) run as their own commands; this
stage only reads their fixed-path receipts in the run directory and blocks
with a zero-spend receipt whenever one is missing, stale or not succeeded:

1. the receipt convention's start halt;
2. a readable, succeeded Stage 6 receipt with matching embedded chunk
   counts, and the index digest it names;
3. the 14A heuristic exhaustive-extraction receipt and its by-topic /
   by-artifact output families;
4. the live title-spine topic strategy (receipt and JSON agreeing);
5. a succeeded, complete 14A full-read extraction (before any source pack is
   built);
6. the deterministic ($0) source packs -- built *before* the 07H/07B/07C
   gates because topic reconciliation consumes and enriches them;
7. the 07H reconciliation reference workflow, 07B question selection and
   07C dual deep-dive research receipts;
8. the pre-spend lineage gate;
9. the governed deep-narrative synthesis (paid; the client is constructed
   only with provider-cost approval), then the analysis manifest and the
   convention's success receipt with per-owner execution evidence.

Everything a conference decides comes in through the
:class:`~conference_pipeline.production.context.ProductionRunnerContext`,
the :class:`~conference_pipeline.production.receipts.StageReceiptConvention`
(receipt and summary shape), :class:`AnalysisSettings` (including the
:class:`SourcePackPolicy`) and :class:`AnalysisOwners` (the source-pack
builder, topic-spec loader, narrative client and renderer, bound at call
time so a runner's test doubles reach the stage). The module carries no
conference literal.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from conference_pipeline.analysis.evidence_importance import (
    ProgramProminence,
    build_importance_scorer,
    load_program_prominence,
)
from conference_pipeline.analysis.full_read_evidence_selection import select_full_read_evidence
from conference_pipeline.analysis.manifest import emit_analysis_manifest, load_and_validate_manifest
from conference_pipeline.analysis.narrative import (
    COMPACT_METADATA_KEYS,
    DEFAULT_ANALYSIS_PROFILE,
    DEFAULT_DEEP_NARRATIVE_MODEL,
    MAX_COMPACT_AUTHORS,
    MAX_COMPACT_TITLE_CHARS,
    OpenAIDeepNarrativeClient,
    compact_matched_papers,
    load_narrative_source_packs,
    narrative_report_is_current,
    render_narrative_outputs_from_source_packs,
)
from conference_pipeline.analysis.source_pack_selection import (
    conference_keyword_terms,
    select_topic_source_chunks,
)
from conference_pipeline.indexing.corpus_vector_store import (
    VECTOR_STORE_DIR,
    workspace_relpath,
)
from conference_pipeline.paper_matching.matcher import MATCHED_PAPERS_SUPERSEDED_INFIX
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.receipts import (
    DEFAULT_STAGE_IO,
    StageIO,
    StageReceiptConvention,
    StrictStageConvention,
)
from conference_pipeline.provider_receipts import sanitize_for_receipt
from conference_pipeline.receipt_lineage import sha256_file

STAGE_ID = "analysis"
STAGE6_ID = "corpus-inventory-index"
GPT_RESEARCHER_TASK_ID = "analysis.gpt_researcher_stage"

# Post-full-read chain receipts (run directory) this stage gates on.
EXHAUSTIVE_EXTRACTION_RECEIPT = "14A-exhaustive-topic-extraction-receipt.json"
LLM_FULL_READ_RECEIPT = "14A-llm-full-read-extraction-receipt.json"
TOPIC_RECONCILIATION_RECEIPT = "07D-topic-reconciliation.json"
TOPIC_REFERENCE_VALIDATION_RECEIPT = "07E-topic-reference-validation.json"
RECONCILIATION_REFERENCE_RECEIPT = "07H-topic-reconciliation-reference-workflow.json"
DEEP_DIVE_QUESTION_SELECTION_RECEIPT = "07B-deep-dive-question-selection.json"
DUAL_DEEP_DIVE_RESEARCH_RECEIPT = "07C-dual-deep-dive-research-lanes.json"

TOPIC_STRATEGY_FILENAME = "topic_strategy.json"
TOPIC_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_PRIORITY_PAPER_ID_RE = re.compile(r"^[0-9a-f]{64}$")

# Compaction windows. ``quote`` is selector-visible (the shared selector
# ranks and admits on it), so its width is frozen; the abstract window is
# for note derivation only and the selector never reads it.
COMPACT_QUOTE_CHARS = 700
COMPACT_ABSTRACT_WINDOW_CHARS = 3000
COMPACT_ABSTRACT_WINDOW_KEY = "abstract_window"

ZERO_SPEND_FLAGS: Mapping[str, bool] = {
    "provider_calls_made": False,
    "live_provider_calls_made": False,
    "secrets_copied": False,
}

# Blocker code -> (reason, remediation) recorded by conventions that write a
# blocker receipt (the strict convention records the code alone).
BLOCKER_REASONS: Mapping[str, tuple[str, str]] = {
    "stage6_receipt_missing_or_malformed": (
        "analysis requires a readable Stage 6 receipt",
        "rerun corpus-inventory-index before analysis",
    ),
    "stage6_vector_evidence_required": (
        "analysis requires fresh Stage 6 vector evidence with matching embedded counts",
        "rerun corpus-inventory-index and BK-06 before analysis",
    ),
    "stage6_index_digest_stale_or_missing": (
        "analysis requires the Stage 6 index digest named in the receipt",
        "rerun corpus-inventory-index before analysis",
    ),
    "analysis_exhaustive_extraction_required": (
        "analysis requires fresh exhaustive extraction receipt and output families",
        "rerun exhaustive-topic-extraction / Phase 14A before analysis",
    ),
    "analysis_topic_scope_invalid": (
        "analysis topic scope names unknown topics or is empty",
        "pass topic ids from the run's topic strategy",
    ),
    "analysis_llm_full_read_extraction_required": (
        "analysis requires a succeeded, complete Qwen full-read extraction receipt before paid narrative synthesis",
        "rerun the Phase 14A LLM full-read extraction before analysis",
    ),
    "analysis_source_packs_identical": (
        "analysis source-pack selection gave different topics the same chunks",
        "make each topic's title and evidence patterns select its own evidence, then rerun analysis",
    ),
    "analysis_source_pack_insufficient": (
        "analysis source-pack construction did not find enough cited chunks",
        "inspect Stage 6 sidecars and analysis topic keywords",
    ),
    "analysis_reconciliation_reference_required": (
        "analysis requires a succeeded topic-reconciliation reference workflow with "
        "final_validation_status succeeded before paid narrative synthesis",
        "rerun the topic reconciliation reference workflow before analysis",
    ),
    "analysis_deep_dive_question_selection_required": (
        "analysis requires a succeeded 07B deep-dive question selection receipt before paid narrative synthesis",
        "run scripts/select_deep_dive_questions.py before analysis",
    ),
    "analysis_dual_deep_dive_research_required": (
        "analysis requires a succeeded 07C dual deep-dive research lanes receipt before paid narrative synthesis",
        "run scripts/run_dual_deep_dive_research.py before analysis",
    ),
    "analysis_provider_cost_approval_required": (
        "analysis requires provider-cost approval before deep narrative synthesis",
        "rerun analysis with provider-cost approval and a valid narrative model",
    ),
    "analysis_deep_narrative_synthesis_blocked": (
        "deep narrative synthesis did not produce validation-passing reports",
        "rerun analysis with provider-cost approval and a valid narrative model",
    ),
}
TOPIC_STRATEGY_REMEDIATION = "rerun scripts/build_topic_strategy.py for this run before analysis"


def _default_load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    return DEFAULT_STAGE_IO.load_json(path)


# --------------------------------------------------------------------------
# Topic strategy
# --------------------------------------------------------------------------


def topic_strategy_keywords(topic: Mapping[str, Any]) -> list[str]:
    """Derive deterministic retrieval terms from a title-strategy topic."""

    raw_terms: list[str] = []
    for key in ("title", "topic_id"):
        value = str(topic.get(key) or "").replace("_", " ").strip()
        if value:
            raw_terms.append(value)
    for key in ("starter_queries", "evidence_title_patterns"):
        values = topic.get(key)
        if isinstance(values, list):
            raw_terms.extend(str(item).strip() for item in values if str(item).strip())
    normalized: list[str] = []
    seen: set[str] = set()
    for term in raw_terms:
        for candidate in re.split(r"[,;:/|]", term):
            candidate = " ".join(candidate.lower().split())
            if len(candidate) < 3 or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
            for token in re.findall(r"[a-z][a-z0-9-]{2,}", candidate):
                if token not in seen:
                    seen.add(token)
                    normalized.append(token)
    return normalized


def topic_strategy_receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage/analysis/topic-strategy" / run_id / "topic_strategy_receipt.json"


def load_analysis_topic_specs(
    workspace: Path,
    *,
    run_id: str,
    expected_topic_count: int | None = None,
    analysis_dir: Path = Path("analysis"),
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json,
) -> tuple[dict[str, Any], ...]:
    """Topic specs from the live title-spine topic strategy (receipt and JSON must agree).

    ``expected_topic_count`` pins the number of final topics; a mismatch
    raises ``analysis_topic_strategy_requires_<n>_final_topics``. With
    ``None`` the count comes from the strategy itself: ``final_topics`` must
    be a non-empty list, and a ``topic_count`` the strategy declares must
    equal its length.
    """

    receipt, receipt_error = load_json(topic_strategy_receipt_path(workspace, run_id))
    if receipt_error or not isinstance(receipt, dict):
        raise RuntimeError("analysis_topic_strategy_receipt_missing_or_malformed")
    if receipt.get("status") != "succeeded" or receipt.get("mode") != "live":
        raise RuntimeError("analysis_topic_strategy_live_receipt_required")
    provider_receipt_paths = receipt.get("provider_receipt_paths")
    if not isinstance(provider_receipt_paths, list) or not provider_receipt_paths:
        raise RuntimeError("analysis_topic_strategy_provider_receipts_required")
    payload, error = load_json(workspace / analysis_dir / TOPIC_STRATEGY_FILENAME)
    if error or not isinstance(payload, dict):
        raise RuntimeError("analysis_topic_strategy_missing_or_malformed")
    if payload.get("final_topics") != receipt.get("final_topics"):
        raise RuntimeError("analysis_topic_strategy_receipt_json_mismatch")
    final_topics = receipt.get("final_topics")
    if expected_topic_count is not None and (
        not isinstance(final_topics, list) or len(final_topics) != expected_topic_count
    ):
        raise RuntimeError(f"analysis_topic_strategy_requires_{expected_topic_count}_final_topics")
    if expected_topic_count is None:
        if not isinstance(final_topics, list) or not final_topics:
            raise RuntimeError("analysis_topic_strategy_final_topics_missing")
        declared = payload.get("topic_count")
        if declared is not None and (isinstance(declared, bool) or declared != len(final_topics)):
            raise RuntimeError("analysis_topic_strategy_topic_count_mismatch")
    if not isinstance(final_topics, list):
        raise RuntimeError("analysis_topic_strategy_topic_malformed")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for topic in final_topics:
        if not isinstance(topic, dict):
            raise RuntimeError("analysis_topic_strategy_topic_malformed")
        topic_id = str(topic.get("topic_id") or "").strip()
        title = str(topic.get("title") or "").strip()
        if not topic_id or not title or topic_id in seen:
            raise RuntimeError("analysis_topic_strategy_topic_id_or_title_invalid")
        if not TOPIC_ID_RE.fullmatch(topic_id):
            raise RuntimeError("analysis_topic_strategy_topic_id_invalid")
        seen.add(topic_id)
        keywords = topic_strategy_keywords(topic)
        if not keywords:
            raise RuntimeError("analysis_topic_strategy_topic_keywords_missing")
        specs.append({"id": topic_id, "title": title, "keywords": tuple(keywords)})
    return tuple(specs)


# --------------------------------------------------------------------------
# Deterministic source packs
# --------------------------------------------------------------------------


def compact_chunk_quote(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """A vector-store chunk as ``{chunk_id, extraction_record_id, source_artifact_path, quote}``."""

    text = " ".join(str(chunk.get("text") or "").split())
    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "extraction_record_id": str(chunk.get("artifact_id") or ""),
        "source_artifact_path": str(chunk.get("source_path") or ""),
        "quote": text[:COMPACT_QUOTE_CHARS],
    }


def compact_chunk_with_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`compact_chunk_quote` plus a wider abstract window and the structured
    title/author metadata (top-level first, then ``source_context``).

    Mirrors ``compact_chunk_evidence`` in ``analysis/narrative.py``: change what
    structured metadata survives compaction in both places together.
    """

    compact = compact_chunk_quote(chunk)
    text = " ".join(str(chunk.get("text") or "").split())
    compact[COMPACT_ABSTRACT_WINDOW_KEY] = text[:COMPACT_ABSTRACT_WINDOW_CHARS]
    source_context = chunk.get("source_context")
    source_context = source_context if isinstance(source_context, dict) else {}
    for key in COMPACT_METADATA_KEYS:
        raw_value = chunk.get(key)
        if raw_value in (None, "", [], {}):
            raw_value = source_context.get(key)
        if key == "authors":
            authors = [
                str(author).strip()
                for author in (raw_value if isinstance(raw_value, list) else [])
                if str(author).strip()
            ]
            if authors:
                compact["authors"] = authors[:MAX_COMPACT_AUTHORS]
            continue
        title_value = str(raw_value or "").strip()
        if title_value:
            compact[key] = title_value[:MAX_COMPACT_TITLE_CHARS]
    return compact


@dataclass(frozen=True)
class SourcePackPolicy:
    """How a conference composes its per-topic narrative source packs.

    The defaults are the shared selector's defaults with plain quote
    compaction. ``matched_paper_priority`` feeds each topic's already-paid
    ``matched_papers.json`` ranks into selection; ``evidence_importance``
    orders full-read confidence ties by program prominence, matched rank and
    talk corroboration; ``record_source_families`` records the selector's
    per-family counts in each query plan; ``scoped_rebuilds`` accepts a
    ``topic_scope`` (out-of-scope packs are carried forward byte-for-byte);
    ``reject_identical_selections`` refuses packs that select the same chunks
    for different topics.
    """

    min_chunks: int = 3
    max_chunks_per_topic: int = 8
    paper_target: int = 5
    main_conference_floor: int | None = None
    full_read_max_items_per_kind: int = 10
    compact_chunk: Callable[[Mapping[str, Any]], dict[str, Any]] = compact_chunk_quote
    matched_paper_priority: bool = False
    evidence_importance: bool = False
    record_source_families: bool = False
    scoped_rebuilds: bool = False
    reject_identical_selections: bool = False


DEFAULT_SOURCE_PACK_POLICY = SourcePackPolicy()


def source_pack_paths(workspace: Path, topic_id: str, *, analysis_dir: Path = Path("analysis")) -> tuple[Path, Path]:
    """``(query_plan_path, source_pack_path)`` for a topic."""

    query_plan_path = workspace / analysis_dir / "provenance" / "query_plans" / f"{topic_id}.json"
    source_pack_path = workspace / analysis_dir / "narrative_source_packs" / f"{topic_id}.json"
    return query_plan_path, source_pack_path


def compact_full_read_item(item: Mapping[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    evidence_phrases = item.get("evidence_phrases")
    if not isinstance(evidence_phrases, list):
        evidence_phrases = []
    return {
        "unit_id": str(item.get("unit_id") or ""),
        "input_kind": str(item.get("input_kind") or ""),
        "confidence": item.get("confidence"),
        "summary": " ".join(str(item.get("summary") or "").split())[:900],
        "reason": " ".join(str(item.get("reason") or "").split())[:500],
        "evidence_phrases": [str(phrase)[:240] for phrase in evidence_phrases[:5]],
        "source_artifact_path": str(item.get("source_artifact_path") or ""),
        "paper_title": str(metadata.get("title") or ""),
        "paper_id": str(metadata.get("paper_id") or ""),
        "workshop_id": str(metadata.get("workshop_id") or ""),
        "workshop_title": str(metadata.get("workshop_title") or ""),
        "talk_id": str(metadata.get("talk_id") or ""),
        "talk_title": str(metadata.get("talk_title") or ""),
        "talk_speaker": str(metadata.get("talk_speaker") or ""),
        "timestamp_start": metadata.get("timestamp_start"),
        "timestamp_end": metadata.get("timestamp_end"),
    }


def load_llm_full_read_evidence(
    workspace: Path,
    *,
    run_id: str,
    topic_id: str,
    max_items_per_kind: int = 10,
    importance_ranking: bool = False,
    prominence: ProgramProminence | None = None,
    priority_ranks: Mapping[str, int] | None = None,
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json,
) -> dict[str, Any]:
    """A topic's completed 14A full-read units, compacted and selected order-independently."""

    receipt_path = workspace / "storage" / "workflow-runs" / run_id / LLM_FULL_READ_RECEIPT
    topic_path = workspace / "storage" / "extractions" / run_id / "llm-full-read" / "by-topic" / f"{topic_id}.json"
    receipt, receipt_error = load_json(receipt_path)
    topic_payload, topic_error = load_json(topic_path)
    if receipt_error or topic_error or not isinstance(receipt, dict) or not isinstance(topic_payload, dict):
        return {
            "status": "missing",
            "receipt_path": workspace_relpath(workspace, receipt_path),
            "topic_path": workspace_relpath(workspace, topic_path),
            "unit_count": 0,
            "selected_items": [],
        }
    items = topic_payload.get("items")
    if not isinstance(items, list):
        items = []
    compact_items = [
        compact_full_read_item(item) for item in items if isinstance(item, dict) and item.get("important") is not False
    ]
    # Confidence ties are the norm, so the selection must not depend on the
    # by-topic file order (see full_read_evidence_selection); with importance
    # ranking, program tier, matched rank and talk corroboration order a tie
    # before the digest draw.
    importance_counts: dict[str, int] | None = None
    if importance_ranking:
        importance = build_importance_scorer(
            compact_items, prominence=prominence or ProgramProminence(), priority_ranks=priority_ranks
        )
        selected = select_full_read_evidence(
            compact_items, max_items_per_kind=max_items_per_kind, importance=importance
        )
        importance_counts = {"program_tier": 0, "matched_rank": 0, "talk_corroborated": 0}
        for item in selected:
            tier, matched, talked = importance(item)
            importance_counts["program_tier"] += int(tier > 0)
            importance_counts["matched_rank"] += int(matched > 0)
            importance_counts["talk_corroborated"] += int(talked > 0)
    else:
        selected = select_full_read_evidence(compact_items, max_items_per_kind=max_items_per_kind)
    kind_counts: dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            kind = str(item.get("input_kind") or "unknown")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
    checks = receipt.get("checks") if isinstance(receipt.get("checks"), dict) else {}
    evidence: dict[str, Any] = {
        "status": "succeeded" if receipt.get("status") == "succeeded" else "blocked",
        "receipt_path": workspace_relpath(workspace, receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "topic_path": workspace_relpath(workspace, topic_path),
        "topic_sha256": sha256_file(topic_path),
        "model": receipt.get("model"),
        "scope": receipt.get("scope"),
        "completion_state": checks.get("completion_state") if isinstance(checks, dict) else None,
        "expected_unit_count": checks.get("expected_unit_count") if isinstance(checks, dict) else None,
        "completed_unit_count": checks.get("completed_unit_count") if isinstance(checks, dict) else None,
        "failed_count": checks.get("failed_count") if isinstance(checks, dict) else None,
        "unit_count": int(topic_payload.get("unit_count") or len(items)),
        "input_kind_counts": kind_counts,
    }
    if importance_counts is not None:
        evidence["selected_importance_counts"] = importance_counts
    evidence["selected_items"] = selected
    return evidence


def load_matched_papers(
    workspace: Path,
    *,
    run_id: str,
    topic_id: str,
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json,
) -> dict[str, Any]:
    """A topic's already-paid ``matched_papers.json``, compacted."""

    matched_path = workspace / "storage" / "paper-matching" / run_id / topic_id / "matched_papers.json"
    payload, error = load_json(matched_path)
    if error or not isinstance(payload, dict):
        return {
            "status": "missing",
            "path": workspace_relpath(workspace, matched_path),
            "paper_count": 0,
            "papers": [],
        }
    papers = payload.get("papers")
    if not isinstance(papers, list):
        papers = payload.get("matched_papers")
    if not isinstance(papers, list):
        papers = []
    compact = compact_matched_papers([paper for paper in papers if isinstance(paper, dict)], limit=20)
    return {
        "status": "available" if compact else "empty",
        "path": workspace_relpath(workspace, matched_path),
        "sha256": sha256_file(matched_path),
        "paper_count": len(compact),
        "papers": compact,
    }


def priority_source_ranks(
    workspace: Path,
    *,
    run_id: str,
    topic_specs: Sequence[Mapping[str, Any]],
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json,
) -> dict[str, dict[str, int]]:
    """Each topic's matched-paper ranks as ``{"paper:<hash>": rank}`` for the selector.

    Reads ``matched_papers.json`` directly (``compact_matched_papers`` drops
    ``rank`` and truncates). Earlier results the matcher superseded rank after
    every current one, newest first. Never raises: an unusable file yields no
    ranks for that topic.
    """

    priority_by_topic: dict[str, dict[str, int]] = {}
    for spec in topic_specs:
        topic_id = str(spec["id"])
        topic_dir = workspace / "storage" / "paper-matching" / run_id / topic_id
        matched_paths = [topic_dir / "matched_papers.json"] + sorted(
            topic_dir.glob(f"matched_papers{MATCHED_PAPERS_SUPERSEDED_INFIX}*.json"), reverse=True
        )
        ranks: dict[str, int] = {}
        for matched_path in matched_paths:
            payload, error = load_json(matched_path)
            if error or not isinstance(payload, dict):
                continue
            papers = payload.get("papers")
            if not isinstance(papers, list):
                papers = payload.get("matched_papers")
            if not isinstance(papers, list):
                continue
            offset = max(ranks.values(), default=0)
            for position, paper in enumerate(papers, start=1):
                if not isinstance(paper, dict):
                    continue
                paper_id = str(paper.get("paper_id") or "")
                if not _PRIORITY_PAPER_ID_RE.match(paper_id):
                    continue
                source_id = f"paper:{paper_id}"
                if source_id in ranks:
                    continue
                raw_rank = paper.get("rank")
                rank = raw_rank if isinstance(raw_rank, int) and raw_rank > 0 else position
                ranks[source_id] = offset + rank
        priority_by_topic[topic_id] = ranks
    return priority_by_topic


def preserve_current_topic_reconciliation(
    source_pack_path: Path,
    source_pack: dict[str, Any],
    *,
    load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]] = _default_load_json,
) -> None:
    """Carry topic reconciliation through a deterministic source-pack rebuild.

    ``analysis.topic_reconciliation`` enriches packs after they exist; a
    rebuild keeps that enrichment only while the pack body still matches the
    body that was reconciled.
    """

    if not source_pack_path.exists():
        return
    existing, _error = load_json(source_pack_path)
    if not isinstance(existing, dict):
        return
    reconciliation = existing.get("topic_reconciliation")
    if not isinstance(reconciliation, dict) or reconciliation.get("status") != "succeeded":
        return
    try:
        from conference_pipeline.analysis.topic_reconciliation import source_pack_input_sha256
    except Exception:  # noqa: BLE001 - optional enrichment
        return
    if source_pack_input_sha256(existing) == source_pack_input_sha256(source_pack):
        source_pack["topic_reconciliation"] = reconciliation


class IdenticalSourcePacksError(RuntimeError):
    """Different topics selected the same chunks: a selection failure, never agreement."""

    def __init__(self, groups: Sequence[Sequence[str]]) -> None:
        self.groups = [list(group) for group in groups]
        super().__init__("analysis_source_packs_identical:" + ";".join(",".join(group) for group in self.groups))


def build_analysis_source_packs(
    *,
    workspace: Path,
    run_id: str,
    stage6_receipt: Mapping[str, Any],
    index_digest: Mapping[str, Any],
    topic_specs: tuple[dict[str, Any], ...],
    policy: SourcePackPolicy = DEFAULT_SOURCE_PACK_POLICY,
    topic_scope: frozenset[str] | None = None,
    analysis_dir: Path = Path("analysis"),
    io: StageIO = DEFAULT_STAGE_IO,
    selector: Callable[..., Any] = select_topic_source_chunks,
) -> dict[str, Any]:
    """Deterministic per-topic source packs and query plans from the final Stage 6 sidecars.

    A keyword-substring scan over the ``papers``/``artifacts`` ``chunks.jsonl``
    sidecars through the shared, order-independent, family-stratified
    selector -- no embeddings query and no provider call -- paired with each
    topic's 14A full-read evidence and matched papers.
    """

    query_plan_dir = workspace / analysis_dir / "provenance" / "query_plans"
    source_pack_dir = workspace / analysis_dir / "narrative_source_packs"
    query_plan_dir.mkdir(parents=True, exist_ok=True)
    source_pack_dir.mkdir(parents=True, exist_ok=True)
    stores = index_digest.get("stores")
    if not isinstance(stores, dict):
        raise RuntimeError("analysis_index_digest_missing_stores")
    store_chunk_paths: list[tuple[str, Path]] = []
    for store_name in ("papers", "artifacts"):
        store = stores.get(store_name)
        if not isinstance(store, dict):
            continue
        chunks_path, error = io.resolve_member(workspace, store.get("chunks_path"))
        if error or chunks_path is None or not chunks_path.exists():
            raise RuntimeError(f"analysis_{store_name}_chunks_missing")
        store_chunk_paths.append((store_name, chunks_path))

    priority_ranks: dict[str, dict[str, int]] = {}
    selector_options: dict[str, Any] = {}
    if policy.main_conference_floor is not None:
        selector_options["main_conference_floor"] = policy.main_conference_floor
    if policy.matched_paper_priority:
        priority_ranks = priority_source_ranks(
            workspace, run_id=run_id, topic_specs=topic_specs, load_json=io.load_json
        )
        selector_options["priority_source_ranks"] = priority_ranks
    selection = selector(
        store_chunk_paths=store_chunk_paths,
        topic_specs=topic_specs,
        compact_chunk=policy.compact_chunk,
        ignored_keyword_terms=conference_keyword_terms(workspace),
        max_chunks_per_topic=policy.max_chunks_per_topic,
        paper_target=policy.paper_target,
        **selector_options,
    )
    if policy.reject_identical_selections and selection.identical_selection_groups:
        # Every pack holding the same chunks is a selection failure (the
        # 2026-07-22 packs all carried the opening keynote), never topics that
        # agree; the stage blocks before any paid call.
        raise IdenticalSourcePacksError(selection.identical_selection_groups)
    selected = selection.selected
    fallback = selection.fallback

    index_digest_sha256 = sha256_file(workspace / VECTOR_STORE_DIR / run_id / "index_digest.json")
    stage6_checks = stage6_receipt.get("checks") if isinstance(stage6_receipt.get("checks"), dict) else {}
    output_inventory = stage6_checks.get("output_inventory") if isinstance(stage6_checks, dict) else {}
    corpus_inventory_sha256 = output_inventory.get("sha256") if isinstance(output_inventory, dict) else None
    prominence = load_program_prominence(workspace) if policy.evidence_importance else None
    source_pack_path_values: list[str] = []
    query_plan_path_values: list[str] = []
    insufficient_topics: list[str] = []
    carried_forward_topics: list[str] = []
    for spec in topic_specs:
        topic_id = str(spec["id"])
        title = str(spec["title"])
        query_plan_path, source_pack_path = source_pack_paths(workspace, topic_id, analysis_dir=analysis_dir)
        if policy.scoped_rebuilds and topic_scope is not None and topic_id not in topic_scope:
            # Out of scope: the existing pack (and the narrative rendered from
            # it) is carried forward byte-for-byte; the renderer refuses to
            # carry a topic whose pack is missing.
            carried_forward_topics.append(topic_id)
            source_pack_path_values.append(workspace_relpath(workspace, source_pack_path))
            query_plan_path_values.append(workspace_relpath(workspace, query_plan_path))
            continue
        chunks = selected[topic_id] or fallback[: policy.min_chunks]
        if len(chunks) < policy.min_chunks:
            insufficient_topics.append(topic_id)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks if chunk.get("chunk_id")]
        research_digest = hashlib.sha256(json.dumps(chunk_ids, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        llm_full_read_evidence = load_llm_full_read_evidence(
            workspace,
            run_id=run_id,
            topic_id=topic_id,
            max_items_per_kind=policy.full_read_max_items_per_kind,
            importance_ranking=policy.evidence_importance,
            prominence=prominence,
            priority_ranks=priority_ranks.get(topic_id),
            load_json=io.load_json,
        )
        matched_paper_evidence = load_matched_papers(
            workspace, run_id=run_id, topic_id=topic_id, load_json=io.load_json
        )
        query_plan: dict[str, Any] = {
            "schema_version": 1,
            "generated_at": io.now(),
            "run_id": run_id,
            "topic_id": topic_id,
            "title": title,
            "retrieval_mode": "deterministic_keyword_sidecar_scan",
            "keywords": list(spec.get("keywords") or []),
            "chunk_limit": policy.max_chunks_per_topic,
            "selected_chunk_ids": chunk_ids,
        }
        if policy.record_source_families:
            query_plan["source_families"] = dict(selection.family_counts.get(topic_id) or {})
        query_plan["llm_full_read"] = {
            "status": llm_full_read_evidence.get("status"),
            "receipt_path": llm_full_read_evidence.get("receipt_path"),
            "receipt_sha256": llm_full_read_evidence.get("receipt_sha256"),
            "topic_path": llm_full_read_evidence.get("topic_path"),
            "topic_sha256": llm_full_read_evidence.get("topic_sha256"),
            "unit_count": llm_full_read_evidence.get("unit_count"),
            "input_kind_counts": llm_full_read_evidence.get("input_kind_counts"),
            "selected_unit_ids": [
                item.get("unit_id")
                for item in (llm_full_read_evidence.get("selected_items") or [])
                if isinstance(item, dict) and item.get("unit_id")
            ],
        }
        query_plan["paper_matching"] = {
            "status": matched_paper_evidence.get("status"),
            "path": matched_paper_evidence.get("path"),
            "sha256": matched_paper_evidence.get("sha256"),
            "paper_count": matched_paper_evidence.get("paper_count"),
        }
        if policy.matched_paper_priority:
            query_plan["paper_matching"]["priority_paper_count"] = len(priority_ranks.get(topic_id) or {})
        if policy.evidence_importance and prominence is not None:
            query_plan["evidence_importance"] = {
                "program_prominence": prominence.summary(),
                "selected_importance_counts": llm_full_read_evidence.get("selected_importance_counts"),
            }
        query_plan.update(
            {
                "index_digest_sha256": index_digest_sha256,
                "owner_ref": GPT_RESEARCHER_TASK_ID,
                **ZERO_SPEND_FLAGS,
            }
        )
        io.write_json(query_plan_path, query_plan)
        finding_text = (
            f"Deterministic Phase 7 query planning selected {len(chunks)} "
            f"source chunks for {title} from the final Stage 6 paper and artifact "
            "sidecars and paired them with the completed Qwen full-read evidence "
            f"for {llm_full_read_evidence.get('unit_count') or 0} topic-assigned "
            "transcript windows and paper abstracts. The source pack is intended "
            "for governed deep narrative synthesis and preserves chunk ids, "
            "full-read unit ids, extraction record ids, and source artifact paths "
            "for citation validation."
        )
        full_read_unit_ids = [
            str(item.get("unit_id") or "")
            for item in (llm_full_read_evidence.get("selected_items") or [])
            if isinstance(item, dict) and item.get("unit_id")
        ]
        source_pack = {
            "schema_version": 1,
            "generated_at": io.now(),
            "run_id": run_id,
            "topic_id": topic_id,
            "title": title,
            "status": DEFAULT_ANALYSIS_PROFILE,
            "research_id": f"{run_id}:{topic_id}:{research_digest}",
            "research_path": workspace_relpath(workspace, query_plan_path),
            "finding_text": finding_text,
            "chunks": chunks,
            "llm_full_read_evidence": llm_full_read_evidence,
            "matched_papers": matched_paper_evidence.get("papers") or [],
            "paper_matching_evidence": {key: value for key, value in matched_paper_evidence.items() if key != "papers"},
            "extraction_record_ids": sorted(
                {chunk["extraction_record_id"] for chunk in chunks if chunk.get("extraction_record_id")}
                | set(full_read_unit_ids)
            ),
            "corpus_inventory_sha256": corpus_inventory_sha256,
            "index_digest_sha256": index_digest_sha256,
            "owner_ref": GPT_RESEARCHER_TASK_ID,
            **ZERO_SPEND_FLAGS,
        }
        preserve_current_topic_reconciliation(source_pack_path, source_pack, load_json=io.load_json)
        io.write_json(source_pack_path, source_pack)
        source_pack_path_values.append(workspace_relpath(workspace, source_pack_path))
        query_plan_path_values.append(workspace_relpath(workspace, query_plan_path))

    result: dict[str, Any] = {
        "source_pack_count": len(source_pack_path_values),
        "source_pack_paths": source_pack_path_values,
        "query_plan_paths": query_plan_path_values,
        "insufficient_topics": insufficient_topics,
    }
    if policy.scoped_rebuilds:
        result["topic_scope"] = sorted(topic_scope) if topic_scope is not None else None
        result["carried_forward_topics"] = carried_forward_topics
    return result


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisSettings:
    """What a conference decides about its analysis stage.

    ``manifest_generator`` is the analysis manifest's ``generator`` field
    (default ``conference_pipeline.<slug>_production.analysis@v1``);
    ``expected_topic_count`` pins the final topic count (default ``None``:
    the count the conference's topic strategy holds); ``load_env_file``
    loads ``--env-file`` and records its status flags in the receipt.
    """

    source_pack_policy: SourcePackPolicy = DEFAULT_SOURCE_PACK_POLICY
    manifest_generator: str | None = None
    expected_topic_count: int | None = None
    load_env_file: bool = True


DEFAULT_SETTINGS = AnalysisSettings()


@dataclass(frozen=True)
class AnalysisOwners:
    """The callables the stage runs, bound by the runner at call time.

    Each defaults to the shared implementation; a runner passes its own
    module-level names so a test double installed on the runner reaches the
    stage. ``build_source_packs(**kwargs)`` receives ``topic_scope`` only
    under a ``scoped_rebuilds`` policy.
    """

    build_source_packs: Callable[..., Mapping[str, Any]] | None = None
    load_topic_specs: Callable[..., tuple[dict[str, Any], ...]] | None = None
    narrative_client: Callable[[], Any] | None = None
    render_narratives: Callable[..., Any] | None = None


def _receipt_gate(
    workspace: Path, path: Path, prefix: str, *, load_json: Callable[[Path], tuple[dict[str, Any] | None, str | None]]
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any]]:
    payload, error = load_json(path)
    checks = {
        f"{prefix}_receipt_path": workspace_relpath(workspace, path),
        f"{prefix}_receipt_present": path.exists(),
        f"{prefix}_receipt_status": payload.get("status") if isinstance(payload, dict) else None,
    }
    return payload, error, checks


def _succeeded(payload: Any, error: str | None) -> bool:
    return error is None and isinstance(payload, dict) and payload.get("status") == "succeeded"


@dataclass
class _Run:
    ctx: ProductionRunnerContext
    convention: StageReceiptConvention
    settings: AnalysisSettings

    def block(self, code: str, checks: Mapping[str, Any], *, reason: str | None = None) -> Any:
        default_reason, remediation = BLOCKER_REASONS.get(code, (code, "rerun analysis"))
        return self.convention.block(code, reason=reason or default_reason, remediation=remediation, checks=checks)


def run_analysis_stage(
    ctx: ProductionRunnerContext,
    *,
    convention: StageReceiptConvention | None = None,
    resume: bool,
    provider_cost_approved: bool,
    env_file: Path | None = None,
    topic_scope: frozenset[str] | None = None,
    settings: AnalysisSettings = DEFAULT_SETTINGS,
    owners: AnalysisOwners | None = None,
    io: StageIO = DEFAULT_STAGE_IO,
) -> Any:
    """Run Stage 7 under ``convention`` (default: the strict shared convention).

    On ``resume`` (whole-run, no ``topic_scope``) a succeeded receipt that
    still verifies, over fresh upstreams, is returned as is. Otherwise the
    gates re-run and, on resume, every topic whose existing narrative is
    current for its rebuilt pack, the narrative model and the analysis
    profile is carried forward instead of re-synthesized. Topics left to
    synthesize need provider-cost approval, checked before the renderer is
    touched.
    """

    convention = convention or StrictStageConvention(ctx, STAGE_ID, io=io)
    owners = owners or AnalysisOwners()
    policy = settings.source_pack_policy
    analysis_dir = ctx.analysis_manifest.parent
    run = _Run(ctx, convention, settings)
    workspace, run_id = ctx.workspace, ctx.run_id
    target = ctx.run_dir()
    target.mkdir(parents=True, exist_ok=True)

    halted = convention.start()
    if halted is not None:
        return halted
    if resume and topic_scope is None and ctx.upstream_lineage_checks(STAGE_ID) is None:
        resumed = convention.resumed()
        if resumed is not None:
            return resumed

    approval = ctx.provider_cost_approval(provider_cost_approved)
    checks: dict[str, Any] = {}
    if settings.load_env_file:
        checks["env_file"] = ctx.load_env(env_file)
    checks.update({"provider_cost_approval": approval, **ZERO_SPEND_FLAGS})

    # Gate: Stage 6 receipt and its vector evidence.
    stage6_receipt_path = ctx.receipt_path(STAGE6_ID)
    stage6_receipt, stage6_error = io.load_json(stage6_receipt_path)
    checks.update(
        {
            "stage6_receipt_path": workspace_relpath(workspace, stage6_receipt_path),
            "stage6_receipt_present": stage6_receipt_path.exists(),
        }
    )
    if stage6_error or not isinstance(stage6_receipt, dict):
        return run.block("stage6_receipt_missing_or_malformed", checks)
    stage6_checks = stage6_receipt.get("checks") if isinstance(stage6_receipt.get("checks"), dict) else {}
    checks.update(
        {
            "stage6_status": stage6_receipt.get("status"),
            "stage6_vector_embeddings_written": stage6_checks.get("vector_embeddings_written"),
            "stage6_chunk_counts": stage6_checks.get("chunk_counts"),
            "stage6_embedded_chunk_counts": stage6_checks.get("embedded_chunk_counts"),
            "stage6_index_digest_sha256": stage6_checks.get("index_digest_sha256"),
            "stage6_output_inventory": stage6_checks.get("output_inventory"),
        }
    )
    if (
        stage6_receipt.get("status") != "succeeded"
        or not stage6_checks.get("vector_embeddings_written")
        or stage6_checks.get("chunk_counts") != stage6_checks.get("embedded_chunk_counts")
    ):
        return run.block("stage6_vector_evidence_required", checks)

    index_digest_path = workspace / VECTOR_STORE_DIR / run_id / "index_digest.json"
    index_digest, index_digest_error = io.load_json(index_digest_path)
    checks["index_digest_path"] = workspace_relpath(workspace, index_digest_path)
    checks["index_digest_present"] = index_digest_path.exists()
    if (
        index_digest_error
        or not isinstance(index_digest, dict)
        or sha256_file(index_digest_path) != str(stage6_checks.get("index_digest_sha256") or "")
    ):
        return run.block("stage6_index_digest_stale_or_missing", checks)

    # Gate: the 14A heuristic exhaustive extraction (Stage 7A) and its outputs.
    extractions = workspace / "storage" / "extractions" / run_id
    exhaustive_receipt_path = target / EXHAUSTIVE_EXTRACTION_RECEIPT
    by_topic_glob = sorted((extractions / "by-topic").glob("*.json"))
    by_artifact_glob = sorted((extractions / "by-artifact").glob("*.json"))
    exhaustive_receipt, exhaustive_error = io.load_json(exhaustive_receipt_path)
    checks.update(
        {
            "exhaustive_extraction_receipt_path": workspace_relpath(workspace, exhaustive_receipt_path),
            "exhaustive_extraction_receipt_present": exhaustive_receipt_path.exists(),
            "exhaustive_extraction_receipt_status": (
                exhaustive_receipt.get("status") if isinstance(exhaustive_receipt, dict) else None
            ),
            "exhaustive_extraction_by_topic_count": len(by_topic_glob),
            "exhaustive_extraction_by_artifact_count": len(by_artifact_glob),
        }
    )
    if not _succeeded(exhaustive_receipt, exhaustive_error) or not by_topic_glob or not by_artifact_glob:
        return run.block("analysis_exhaustive_extraction_required", checks)

    # Gate: the live title-spine topic strategy.
    topic_strategy_path = workspace / analysis_dir / TOPIC_STRATEGY_FILENAME
    strategy_receipt_path = topic_strategy_receipt_path(workspace, run_id)
    checks["topic_strategy_path"] = workspace_relpath(workspace, topic_strategy_path)
    checks["topic_strategy_present"] = topic_strategy_path.exists()
    checks["topic_strategy_receipt_path"] = workspace_relpath(workspace, strategy_receipt_path)
    checks["topic_strategy_receipt_present"] = strategy_receipt_path.exists()
    try:
        if owners.load_topic_specs is not None:
            topic_specs = owners.load_topic_specs(workspace, run_id=run_id)
        else:
            topic_specs = load_analysis_topic_specs(
                workspace,
                run_id=run_id,
                expected_topic_count=settings.expected_topic_count,
                analysis_dir=analysis_dir,
                load_json=io.load_json,
            )
    except RuntimeError as exc:
        count = settings.expected_topic_count
        sized = f"{count}-topic " if count is not None else ""
        return convention.block(
            str(exc),
            reason=f"analysis requires a fresh {sized}title-spine topic strategy",
            remediation=TOPIC_STRATEGY_REMEDIATION,
            checks=checks,
        )
    topic_ids = [str(spec["id"]) for spec in topic_specs]
    checks["topic_strategy_topic_ids"] = topic_ids
    checks["topic_strategy_topic_count"] = len(topic_specs)
    if topic_scope is not None:
        unknown_scope = sorted(topic_scope - set(topic_ids))
        checks["analysis_topic_scope"] = {
            "rebuilt_topic_ids": sorted(topic_scope),
            "carried_forward_topic_ids": sorted(set(topic_ids) - topic_scope),
            "unknown_topic_ids": unknown_scope,
        }
        if unknown_scope or not topic_scope:
            return run.block("analysis_topic_scope_invalid", checks)

    # Pre-spend gate: the 14A full-read extraction must be complete before
    # any source pack is built.
    llm_full_read_receipt_path = target / LLM_FULL_READ_RECEIPT
    llm_full_read_receipt, llm_full_read_error = io.load_json(llm_full_read_receipt_path)
    llm_full_read_checks = (
        llm_full_read_receipt.get("checks")
        if isinstance(llm_full_read_receipt, dict) and isinstance(llm_full_read_receipt.get("checks"), dict)
        else {}
    )
    checks.update(
        {
            "llm_full_read_receipt_path": workspace_relpath(workspace, llm_full_read_receipt_path),
            "llm_full_read_receipt_present": llm_full_read_receipt_path.exists(),
            "llm_full_read_receipt_status": (
                llm_full_read_receipt.get("status") if isinstance(llm_full_read_receipt, dict) else None
            ),
            "llm_full_read_completion_state": llm_full_read_checks.get("completion_state"),
        }
    )
    if (
        not _succeeded(llm_full_read_receipt, llm_full_read_error)
        or llm_full_read_checks.get("completion_state") != "complete"
    ):
        return run.block("analysis_llm_full_read_extraction_required", checks)

    # Deterministic source packs ($0): before the reconciliation gates, which
    # consume and enrich them; the paid call below stays gated on them.
    pack_kwargs: dict[str, Any] = {
        "workspace": workspace,
        "run_id": run_id,
        "stage6_receipt": stage6_receipt,
        "index_digest": index_digest,
        "topic_specs": topic_specs,
    }
    if policy.scoped_rebuilds:
        pack_kwargs["topic_scope"] = topic_scope
    try:
        if owners.build_source_packs is not None:
            source_pack_result = owners.build_source_packs(**pack_kwargs)
        else:
            source_pack_result = build_analysis_source_packs(
                **pack_kwargs, policy=policy, analysis_dir=analysis_dir, io=io
            )
    except IdenticalSourcePacksError as exc:
        # A selection failure, never topics that agree: block with zero spend.
        checks["identical_source_pack_groups"] = exc.groups
        return run.block("analysis_source_packs_identical", checks)
    checks["source_packs"] = source_pack_result
    if source_pack_result.get("insufficient_topics"):
        return run.block("analysis_source_pack_insufficient", checks)

    reconciliation_path = target / RECONCILIATION_REFERENCE_RECEIPT
    reconciliation, reconciliation_error, reconciliation_checks = _receipt_gate(
        workspace, reconciliation_path, "reconciliation_reference", load_json=io.load_json
    )
    reconciliation_inner = (
        reconciliation.get("checks")
        if isinstance(reconciliation, dict) and isinstance(reconciliation.get("checks"), dict)
        else {}
    )
    checks.update(reconciliation_checks)
    checks["reconciliation_reference_final_validation_status"] = reconciliation_inner.get("final_validation_status")
    if (
        not _succeeded(reconciliation, reconciliation_error)
        or reconciliation_inner.get("final_validation_status") != "succeeded"
    ):
        return run.block("analysis_reconciliation_reference_required", checks)

    for receipt_name, prefix, code in (
        (DEEP_DIVE_QUESTION_SELECTION_RECEIPT, "deep_dive_question_selection", "analysis_deep_dive_question_selection_required"),
        (DUAL_DEEP_DIVE_RESEARCH_RECEIPT, "dual_deep_dive_research", "analysis_dual_deep_dive_research_required"),
    ):
        payload, error, gate_checks = _receipt_gate(workspace, target / receipt_name, prefix, load_json=io.load_json)
        checks.update(gate_checks)
        if not _succeeded(payload, error):
            return run.block(code, checks)

    gated = convention.lineage_gate(checks)
    if gated is not None:
        return gated

    # All pre-spend gates passed: only now may a paid client be constructed,
    # and only with provider-cost approval.
    approval = ctx.provider_cost_approval(provider_cost_approved)
    client_factory = owners.narrative_client or OpenAIDeepNarrativeClient.from_env
    llm_client = client_factory() if approval["approved"] else None
    narrative_model = llm_client.model if llm_client is not None else DEFAULT_DEEP_NARRATIVE_MODEL
    scope_carried = (
        frozenset(checks["analysis_topic_scope"]["carried_forward_topic_ids"]) if topic_scope is not None else frozenset()
    )
    reused = (
        current_narrative_topic_ids(workspace, run_id=run_id, narrative_model=narrative_model) - scope_carried
        if resume
        else frozenset()
    )
    pending = sorted(set(topic_ids) - scope_carried - reused)
    if resume:
        checks["narrative_reuse"] = {
            "narrative_model": narrative_model,
            "reused_topic_ids": sorted(reused),
            "pending_topic_ids": pending,
        }
    if pending and not approval["approved"]:
        # Never touch the renderer without approval: with no client it would
        # overwrite every topic's report with a blocked placeholder.
        return run.block("analysis_provider_cost_approval_required", checks)
    if not pending:
        llm_client = None
    render = owners.render_narratives or render_narrative_outputs_from_source_packs
    scope_kwargs: dict[str, Any] = {}
    if topic_scope is not None or reused:
        scope_kwargs["carry_forward_topic_ids"] = scope_carried | reused
    narrative_result = render(
        workspace,
        run_id=run_id,
        llm_client=llm_client,
        allow_deterministic_fallback=False,
        narrative_model=narrative_model,
        analysis_profile=DEFAULT_ANALYSIS_PROFILE,
        reuse_existing_pass_reports=False,
        **scope_kwargs,
    )
    provider_request_count = int(getattr(llm_client, "request_count", 0) or 0)
    checks.update(
        {
            "narrative_status": narrative_result.status,
            "narrative_model": narrative_model,
            "analysis_profile": DEFAULT_ANALYSIS_PROFILE,
            "report_paths": [workspace_relpath(workspace, path) for path in narrative_result.report_paths],
            "slide_brief_paths": [workspace_relpath(workspace, path) for path in narrative_result.slide_brief_paths],
            "provenance_paths": [workspace_relpath(workspace, path) for path in narrative_result.provenance_paths],
            "provenance_json_paths": [
                workspace_relpath(workspace, path) for path in narrative_result.provenance_json_paths
            ],
            "citation_manifest_path": (
                workspace_relpath(workspace, narrative_result.citation_manifest_path)
                if narrative_result.citation_manifest_path
                else None
            ),
            "citation_summaries": sanitize_for_receipt(narrative_result.citation_summaries),
            "narrative_blockers": sanitize_for_receipt(narrative_result.blockers),
            "provider_calls_made": provider_request_count > 0,
            "live_provider_calls_made": provider_request_count > 0,
        }
    )
    if narrative_result.status != "succeeded":
        code = (
            "analysis_provider_cost_approval_required"
            if not approval["approved"]
            else "analysis_deep_narrative_synthesis_blocked"
        )
        return run.block(code, checks)

    summaries = [item for item in narrative_result.citation_summaries if isinstance(item, dict) and item.get("topic_id")]
    manifest_path, manifest_sha256, manifest_content_sha256 = emit_analysis_manifest(
        SimpleNamespace(root=workspace),
        generator=settings.manifest_generator or f"conference_pipeline.{ctx.slug}_production.analysis@v1",
        citations_by_topic={
            str(item.get("topic_id")): {
                "extraction_record_ids": len(item.get("extraction_record_ids") or []),
                "vector_store_chunk_ids": len(item.get("vector_store_chunk_ids") or []),
                "research_ids": len(item.get("research_ids") or []),
            }
            for item in summaries
        },
        provenance_by_topic={
            str(item.get("topic_id")): {
                "provenance_path": item.get("provenance_path"),
                "provenance_json_path": item.get("provenance_json_path"),
            }
            for item in summaries
        },
        narrative_metadata_by_topic={
            str(item.get("topic_id")): {
                "slide_brief_path": item.get("slide_brief_path"),
                "analysis_profile": item.get("analysis_profile"),
                "narrative_model": item.get("narrative_model"),
                "deep_validation": item.get("deep_validation"),
            }
            for item in summaries
        },
    )
    loaded_manifest = load_and_validate_manifest(SimpleNamespace(root=workspace))
    checks.update(
        {
            "analysis_manifest_path": workspace_relpath(workspace, manifest_path),
            "analysis_manifest_sha256": manifest_sha256,
            "analysis_manifest_content_sha256": manifest_content_sha256,
            "analysis_manifest_topic_count": len(loaded_manifest.manifest.topics),
            "analysis_manifest_status": "succeeded",
        }
    )
    return convention.succeed(checks, owner_evidence=owner_execution_evidence(ctx, checks, source_pack_result))


def current_narrative_topic_ids(workspace: Path, *, run_id: str, narrative_model: str) -> frozenset[str]:
    """Topics whose existing narrative is current for the pack on disk, the model and the profile."""

    packs = load_narrative_source_packs(workspace / "analysis")
    return frozenset(
        str(pack.get("topic_id"))
        for pack in packs
        if narrative_report_is_current(
            workspace,
            run_id=run_id,
            source_pack=pack,
            narrative_model=narrative_model,
            analysis_profile=DEFAULT_ANALYSIS_PROFILE,
        )
    )


def owner_execution_evidence(
    ctx: ProductionRunnerContext, checks: Mapping[str, Any], source_pack_result: Mapping[str, Any]
) -> dict[str, Any]:
    """Each Stage 7 owner id -> a receipt or output path this stage checked or produced."""

    target = ctx.run_dir()
    return {
        "analysis.topic_strategy": checks.get("topic_strategy_receipt_path"),
        "analysis.exhaustive_topic_extraction": checks.get("exhaustive_extraction_receipt_path"),
        "analysis.llm_full_read_extraction": checks.get("llm_full_read_receipt_path"),
        "analysis.gpt_researcher_stage": next(iter(source_pack_result.get("source_pack_paths") or []), None),
        "analysis.gpt_researcher_heuristic_prefilter": checks.get("dual_deep_dive_research_receipt_path"),
        "analysis.deep_narrative_module": checks.get("citation_manifest_path")
        or next(iter(checks.get("report_paths") or []), None),
        "analysis.topic_reconciliation": workspace_relpath(ctx.workspace, target / TOPIC_RECONCILIATION_RECEIPT),
        "analysis.topic_reconciliation_reference_workflow": checks.get("reconciliation_reference_receipt_path"),
        "analysis.topic_reference_validation": workspace_relpath(
            ctx.workspace, target / TOPIC_REFERENCE_VALIDATION_RECEIPT
        ),
        "analysis.deep_dive_question_selection": checks.get("deep_dive_question_selection_receipt_path"),
        "analysis.dual_deep_dive_research_lanes": checks.get("dual_deep_dive_research_receipt_path"),
    }


__all__ = [
    "AnalysisOwners",
    "AnalysisSettings",
    "IdenticalSourcePacksError",
    "SourcePackPolicy",
    "build_analysis_source_packs",
    "compact_chunk_quote",
    "compact_chunk_with_metadata",
    "load_analysis_topic_specs",
    "run_analysis_stage",
]
