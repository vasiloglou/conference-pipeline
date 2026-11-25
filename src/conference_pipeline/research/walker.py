"""Per-topic walker for Phase 14C — iterates production topics, builds plans,
runs research, persists outputs, and accumulates resume metadata.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.iclr2025_live import (
    load_json,
    load_topic_families,
    sha256_file,
    stage_path,
)
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.research.config import GPTResearcherParameters
from conference_pipeline.research.query_planner import (
    QueryExpansionLLMClient,
    QueryPlanLLMClient,
    build_plan,
)
from conference_pipeline.research.records import (
    Citation,
    QueryPlan,
    ResearchRecord,
    safe_research_filename,
    write_query_plan,
    write_rebuild_inputs,
    write_research_record,
)
from conference_pipeline.research.retrieval_bundle import RetrievalBundle
from conference_pipeline.research.researcher import (
    Researcher,
    TavilyClient,
    VectorStoreRetriever,
    run_topic_research,
)


class WalkerError(RuntimeError):
    """Boundary error for the walker; messages are sanitized."""


@dataclass
class WalkerResult:
    expected_count: int
    records: tuple[ResearchRecord, ...]
    by_topic_counts: dict[str, dict[str, int]]
    bound_extraction_receipt_digest: str
    bound_vector_store_digest: str
    resumed_count: int = 0
    researched_count: int = 0
    cache_hit_count: int = 0
    llm_token_usage: dict[str, int] = field(default_factory=dict)
    web_search_credits: dict[str, int] = field(default_factory=dict)
    retrieval_bundle: dict[str, Any] = field(default_factory=dict)
    parallelism_receipt: dict[str, Any] = field(default_factory=dict)


def _load_by_topic(by_topic_dir: Path, topic_id: str) -> dict[str, Any]:
    path = by_topic_dir / f"{topic_id}.json"
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:  # noqa: BLE001 - fallback path
        return {}


def _load_entity_index(workspace_root: Path, run_id: str) -> dict[str, Any]:
    path = (
        workspace_root
        / "storage"
        / "vector-store"
        / run_id
        / "entity_index.json"
    )
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:  # noqa: BLE001 - fallback path
        return {}


def _load_rebuild_inputs(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        payload = load_json(path)
    except Exception:  # noqa: BLE001 - fallback path
        return {}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    out: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            topic_id = entry.get("topic_id")
            if isinstance(topic_id, str):
                out[topic_id] = entry
    return out


def _load_existing_record(
    receipts_dir: Path, topic_id: str
) -> ResearchRecord | None:
    path = receipts_dir / f"{topic_id}-research-receipt.json"
    if not path.exists():
        return None
    try:
        payload = load_json(path)
    except Exception:  # noqa: BLE001 - fallback path
        return None
    if not isinstance(payload, dict):
        return None
    citations: list[Citation] = []
    for raw in payload.get("citations") or []:
        if not isinstance(raw, dict):
            continue
        citations.append(
            Citation(
                source_artifact_id=str(raw.get("source_artifact_id") or ""),
                source_artifact_path=str(raw.get("source_artifact_path") or ""),
                evidence_span_or_chunk_id=str(
                    raw.get("evidence_span_or_chunk_id") or ""
                ),
                verbatim_quote=str(raw.get("verbatim_quote") or ""),
                source_type=str(raw.get("source_type") or "local"),
            )
        )
    return ResearchRecord(
        research_id=str(payload.get("research_id") or topic_id),
        topic_id=str(payload.get("topic_id") or topic_id),
        run_id=str(payload.get("run_id") or ""),
        status=str(payload.get("status") or "skipped"),
        coverage_class=str(payload.get("coverage_class") or "skipped"),
        query_plan_path=str(payload.get("query_plan_path") or ""),
        synthesized_findings_path=str(
            payload.get("synthesized_findings_path") or ""
        ),
        retrieved_evidence_ids=tuple(
            str(x) for x in (payload.get("retrieved_evidence_ids") or [])
        ),
        source_paths=tuple(
            str(x) for x in (payload.get("source_paths") or [])
        ),
        citations=tuple(citations),
        confidence_metadata=dict(payload.get("confidence_metadata") or {}),
        provider_usage=dict(payload.get("provider_usage") or {}),
        blockers=tuple(
            dict(b) for b in (payload.get("blockers") or []) if isinstance(b, dict)
        ),
        bound_extraction_receipt_digest=str(
            payload.get("bound_extraction_receipt_digest") or ""
        ),
        bound_vector_store_digest=str(
            payload.get("bound_vector_store_digest") or ""
        ),
        web_search_enabled=bool(payload.get("web_search_enabled")),
        timestamp=str(payload.get("timestamp") or ""),
        retrieval_bundle=dict(payload.get("retrieval_bundle") or {}),
        expansion_mode=str(payload.get("expansion_mode") or "disabled"),
        expanded_query_count=int(payload.get("expanded_query_count") or 0),
        dedup_policy=str(payload.get("dedup_policy") or "passthrough"),
        dedup_removed_count=int(payload.get("dedup_removed_count") or 0),
        dedup_removed_chunk_ids=tuple(
            str(x) for x in (payload.get("dedup_removed_chunk_ids") or [])
        ),
        dedup_kept_source_families=tuple(
            str(x) for x in (payload.get("dedup_kept_source_families") or [])
        ),
    )


def existing_research_digest_matches(
    rebuild_inputs: dict[str, dict[str, Any]],
    topic_id: str,
    bound_vector_store_digest: str,
    research_parameters_digest: str = "",
) -> bool:
    """Return True iff the cached ``topic_id`` is bound to the current vs digest."""
    entry = rebuild_inputs.get(topic_id)
    if not entry:
        return False
    cached = str(entry.get("bound_vector_store_digest") or "")
    if not (bool(cached) and cached == bound_vector_store_digest):
        return False
    if research_parameters_digest:
        cached_parameters = str(entry.get("research_parameters_digest") or "")
        return bool(cached_parameters) and cached_parameters == research_parameters_digest
    return True


def walk_and_research(
    workspace_root: Path,
    *,
    run_id: str,
    retriever: VectorStoreRetriever,
    researcher: Researcher,
    tavily: TavilyClient | None = None,
    llm_client: QueryPlanLLMClient | None = None,
    expansion_llm_client: QueryExpansionLLMClient | None = None,
    force_rebuild: bool = False,
    research_parameters: GPTResearcherParameters | None = None,
    retrieval_bundle: RetrievalBundle | None = None,
) -> WalkerResult:
    """Iterate production topics; persist plans + findings + receipts."""
    workspace_root = Path(workspace_root).resolve()
    conference_label = conference_name(workspace_root, default="the conference")
    parameters = research_parameters or GPTResearcherParameters()
    parameters_digest = parameters.behavioral_digest()
    research_provider = str(getattr(researcher, "provider", "openai") or "openai")
    research_model = str(
        getattr(researcher, "model", None)
        or default_model_for_provider("research.gpt_researcher_fast", research_provider)
    )
    output_root = workspace_root / "storage" / "research" / run_id
    plans_dir = output_root / "query-plans"
    findings_dir = output_root / "findings"
    receipts_dir = (
        workspace_root / "storage" / "workflow-runs" / run_id / "14C-research"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir.mkdir(parents=True, exist_ok=True)

    extraction_receipt = stage_path(
        workspace_root, run_id, "exhaustive-topic-extraction"
    )
    vector_store_receipt = stage_path(
        workspace_root, run_id, "enhanced-indexing-vector-store"
    )
    bound_extraction_digest = (
        sha256_file(extraction_receipt) if extraction_receipt.exists() else ""
    )
    bound_vector_store_digest = (
        sha256_file(vector_store_receipt)
        if vector_store_receipt.exists()
        else ""
    )

    rebuild_inputs_path = output_root / "rebuild_inputs.json"
    existing_inputs = _load_rebuild_inputs(rebuild_inputs_path)

    by_topic_dir = workspace_root / "storage" / "extractions" / run_id / "by-topic"
    entity_index = _load_entity_index(workspace_root, run_id)

    topics = load_topic_families(workspace_root)
    raw_topic_filter = os.environ.get("QUERY_RESEARCH_TOPIC_IDS", "").strip()
    if raw_topic_filter:
        selected_topic_ids = {
            item.strip()
            for item in raw_topic_filter.split(",")
            if item.strip()
        }
        topics = [topic for topic in topics if topic.id in selected_topic_ids]
    records: list[ResearchRecord] = []
    by_topic_counts: dict[str, dict[str, int]] = {}
    resumed_count = 0
    researched_count = 0
    web_search_credits = 0
    llm_input_tokens = 0
    llm_output_tokens = 0
    llm_request_count = 0
    retrieval_bundle_receipt: dict[str, Any] = {}

    tavily_enabled = tavily is not None
    topic_cap = max(
        1,
        int(
            parameters.topic_concurrency
            if parameters.topic_concurrency is not None
            else parameters.deep_research_concurrency
        ),
    )
    question_cap = max(
        1,
        int(
            parameters.question_concurrency
            if parameters.question_concurrency is not None
            else parameters.deep_research_concurrency
        ),
    )

    # Timing instrumentation for parallelism receipt.
    _run_start = time.monotonic()
    _topic_timestamps: dict[str, dict[str, float]] = {}
    _timestamps_lock = threading.Lock()
    _throttle_events: list[dict[str, Any]] = []

    # Phase 1: partition — resumed topics skip worker slots; only remaining
    # topics are dispatched to the bounded thread pool.
    resumed_records: dict[str, ResearchRecord] = {}
    to_research = []
    for topic in topics:
        previous = existing_inputs.get(topic.id)
        if (
            not force_rebuild
            and previous
            and existing_research_digest_matches(
                existing_inputs,
                topic.id,
                bound_vector_store_digest,
                parameters_digest,
            )
        ):
            cached = _load_existing_record(receipts_dir, topic.id)
            if cached is not None:
                resumed_records[topic.id] = cached
                resumed_count += 1
                continue
        to_research.append(topic)

    # Phase 2: parallel research execution bounded by topic_cap.
    def _research_one_topic(topic: Any) -> ResearchRecord:
        t_start = time.monotonic() - _run_start
        plan = build_plan(
            topic=topic,
            by_topic_seeds=_load_by_topic(by_topic_dir, topic.id),
            entity_index=entity_index,
            bound_vector_store_digest=bound_vector_store_digest,
            run_id=run_id,
            llm_client=llm_client,
            expansion_llm_client=expansion_llm_client,
            tavily_enabled=tavily_enabled,
            research_parameters=parameters,
            research_provider=research_provider,
            research_model=research_model,
            conference_label=conference_label,
        )
        plan_filename = (
            f"{safe_research_filename(plan.topic_id, plan.plan_digest)}-plan.json"
        )
        write_query_plan(plans_dir / plan_filename, plan)
        rec = run_topic_research(
            workspace_root,
            plan=plan,
            retriever=retriever,
            researcher=researcher,
            tavily=tavily,
            findings_path=findings_dir / f"{topic.id}.md",
            bound_extraction_receipt_digest=bound_extraction_digest,
            bound_vector_store_digest=bound_vector_store_digest,
            run_id=run_id,
            research_parameters=parameters,
            retrieval_bundle=retrieval_bundle,
        )
        write_research_record(
            receipts_dir / f"{topic.id}-research-receipt.json", rec
        )
        t_end = time.monotonic() - _run_start
        with _timestamps_lock:
            _topic_timestamps[topic.id] = {
                "start_offset_seconds": round(t_start, 4),
                "end_offset_seconds": round(t_end, 4),
                "duration_seconds": round(t_end - t_start, 4),
            }
        return rec

    researched_map: dict[str, ResearchRecord] = {}
    if len(to_research) > 1 and topic_cap > 1:
        with ThreadPoolExecutor(max_workers=topic_cap) as _ex:
            _futures = {_ex.submit(_research_one_topic, t): t for t in to_research}
        for _fut, _topic in _futures.items():
            researched_map[_topic.id] = _fut.result()
    else:
        for topic in to_research:
            researched_map[topic.id] = _research_one_topic(topic)

    # Phase 3: reassemble in original deterministic topic order.
    for topic in topics:
        if topic.id in resumed_records:
            cached = resumed_records[topic.id]
            records.append(cached)
            if cached.retrieval_bundle and not retrieval_bundle_receipt:
                retrieval_bundle_receipt = dict(cached.retrieval_bundle)
            by_topic_counts[topic.id] = {
                "local_hits": len(cached.retrieved_evidence_ids),
                "citations": len(cached.citations),
            }
        elif topic.id in researched_map:
            record = researched_map[topic.id]
            if record.retrieval_bundle and not retrieval_bundle_receipt:
                retrieval_bundle_receipt = dict(record.retrieval_bundle)
            records.append(record)
            researched_count += 1
            by_topic_counts[topic.id] = {
                "local_hits": len(record.retrieved_evidence_ids),
                "citations": len(record.citations),
            }
            web_search_credits += int(record.provider_usage.get("web_searches") or 0)
            llm_input_tokens += int(record.provider_usage.get("input_tokens") or 0)
            llm_output_tokens += int(record.provider_usage.get("output_tokens") or 0)
            llm_request_count += 1

    write_rebuild_inputs(
        rebuild_inputs_path,
        records,
        bound_vector_store_digest,
        research_parameters_digest=parameters_digest,
    )

    # Build parallelism receipt from timing instrumentation.
    # Resumed topics did not consume worker slots; record them with zero duration.
    for topic in topics:
        if topic.id in resumed_records and topic.id not in _topic_timestamps:
            _topic_timestamps[topic.id] = {
                "start_offset_seconds": 0.0,
                "end_offset_seconds": 0.0,
                "duration_seconds": 0.0,
            }
    serial_estimate = sum(
        v["duration_seconds"] for v in _topic_timestamps.values()
    )
    actual_topic_workers = min(len(to_research), topic_cap) if to_research else 0
    parallelism_receipt: dict[str, Any] = {
        "configured_topic_cap": parameters.topic_concurrency,
        "configured_question_cap": parameters.question_concurrency,
        "effective_topic_cap": topic_cap,
        "effective_question_cap": question_cap,
        "topic_worker_count": actual_topic_workers,
        "question_worker_count": question_cap,
        "per_topic_timestamps": dict(_topic_timestamps),
        "skipped_count": resumed_count,
        "resumed_count": resumed_count,
        "throttle_events": list(_throttle_events),
        "serial_wall_clock_estimate_seconds": round(serial_estimate, 4),
    }

    return WalkerResult(
        expected_count=len(topics),
        records=tuple(records),
        by_topic_counts=by_topic_counts,
        bound_extraction_receipt_digest=bound_extraction_digest,
        bound_vector_store_digest=bound_vector_store_digest,
        resumed_count=resumed_count,
        researched_count=researched_count,
        llm_token_usage={
            "input_tokens": llm_input_tokens,
            "output_tokens": llm_output_tokens,
            "request_count": llm_request_count,
        },
        web_search_credits={"used": web_search_credits},
        retrieval_bundle=retrieval_bundle_receipt,
        parallelism_receipt=parallelism_receipt,
    )


__all__ = [
    "WalkerError",
    "WalkerResult",
    "existing_research_digest_matches",
    "walk_and_research",
]
