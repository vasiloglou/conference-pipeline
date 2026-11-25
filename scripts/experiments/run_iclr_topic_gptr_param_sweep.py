#!/usr/bin/env python3
"""Experiment-only GPT-Researcher parameter sweep for ICLR topic reports.

This harness is deliberately outside the production workflow. It keeps the
source run, vector store, local endpoint, model, and temperature fixed while
varying the GPT-Researcher-style research knobs requested for topic 02.
Outputs are confined to:

    conferences/iclr2025/storage/experiments/topic2-gptr-param-sweep/

The repository's current ICLR analysis stage uses a local retriever plus LLM
synthesis wrapper rather than instantiating upstream ``GPTResearcher``
directly. This script therefore writes GPT-Researcher-compatible config files
for each variant and applies the knobs to the local query expansion and report
structure harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline._security.loopback import is_loopback_url
from conference_pipeline.analysis.narrative import validate_narrative_topic_report
from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
    REFERENCE_QUALITY_PROMPT,
)
from conference_pipeline.iclr2025_live import (
    load_topic_families,
    relpath,
    sanitize_for_receipt,
    sha256_file,
    stage_path,
    utc_now,
)
from conference_pipeline.indexing.embeddings import build_embedding_client
from conference_pipeline.paper_matching.matcher import (
    LocalKeypointExtractor,
    PaperMatchingConfig,
    StoreBackedResearchRunner,
    run_paper_matching,
)
from conference_pipeline.research.records import (
    QueryPlan,
    compute_plan_digest,
    safe_research_filename,
    write_query_plan,
    write_research_record,
)
from conference_pipeline.research.researcher import (
    HybridChromaBM25Retriever,
    RetrievedChunk,
    build_researcher,
    run_topic_research,
)
from conference_pipeline.provider_receipts import provider_receipt_context
from conference_pipeline.workflow_runs import write_json_atomic


EXPERIMENT_SLUG = "topic2-gptr-param-sweep"
EXPERIMENT_RELATIVE_ROOT = Path("storage") / "experiments" / EXPERIMENT_SLUG
DEFAULT_TOPIC = "02_model_architectures"
DEFAULT_SOURCE_RUN_ID = "v151-iclr2025-topic2-local-analysis"
DEFAULT_EMBEDDING_MODEL = "mlx-community/bge-m3-mlx-8bit"  # model-literal-ok: the finished topic-2 sweep pins the embedding model it measured
DEFAULT_LLM_MODEL = "mlx-community/Qwen3.5-35B-A3B-OptiQ-4bit"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_TOTAL_WORDS = 3000
DEFAULT_SMART_TOKEN_LIMIT = 8192
DEFAULT_STRATEGIC_TOKEN_LIMIT = 8192
DEFAULT_FAST_TOKEN_LIMIT = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_RETRIEVAL_K = 18
DEFAULT_MAX_EVIDENCE_CHUNKS = 96
DEFAULT_CANDIDATE_MULTIPLIER = 3
DEFAULT_RRF_K = 60
VALID_LOCAL_LLM_RUNTIME_KINDS = ("mlx", "ollama")


class SweepError(RuntimeError):
    """Experiment harness boundary error."""


@dataclass(frozen=True)
class SweepVariant:
    """One controlled sweep variant."""

    variant_id: str
    max_iterations: int
    breadth: int
    depth: int
    max_subtopics: int
    purpose: str


@dataclass(frozen=True)
class SweepControls:
    """Fixed controls shared across all variants."""

    topic: str
    source_run_id: str
    vector_store_run_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dim: int | None
    llm_runtime_kind: str
    llm_model: str
    fast_model: str
    local_base_url: str
    temperature: float
    server_max_tokens: int
    web_search_enabled: bool
    total_words: int
    retrieval_k: int
    candidate_multiplier: int
    rrf_k: int
    bm25_weight: float | None
    vector_weight: float | None
    paper_matching_retrieval_k: int


class BudgetedDedupRetriever:
    """Global evidence budget wrapper for the per-question retriever.

    The production ``run_topic_research`` function retrieves per question. For
    wide/deep variants that can produce many overlapping queries, this wrapper
    keeps the local context inside a repeatable global chunk budget while still
    preserving the sweep's relative differences.
    """

    def __init__(
        self,
        inner: Any,
        *,
        max_chunks: int = DEFAULT_MAX_EVIDENCE_CHUNKS,
    ) -> None:
        self.inner = inner
        self.max_chunks = max(1, int(max_chunks))
        self.seen: set[str] = set()

    def retrieve(self, *, topic_id: str, query: str, k: int) -> list[RetrievedChunk]:
        if len(self.seen) >= self.max_chunks:
            return []
        chunks = self.inner.retrieve(topic_id=topic_id, query=query, k=k)
        out: list[RetrievedChunk] = []
        for chunk in chunks:
            if chunk.chunk_id in self.seen:
                continue
            self.seen.add(chunk.chunk_id)
            out.append(chunk)
            if len(self.seen) >= self.max_chunks or len(out) >= max(1, int(k)):
                break
        return out


def default_variants() -> tuple[SweepVariant, ...]:
    """Return the controlled matrix requested by the experiment plan."""

    return (
        SweepVariant("i3-b1-d1-s3", 3, 1, 1, 3, "Local shallow baseline"),
        SweepVariant("i5-b1-d1-s3", 5, 1, 1, 3, "Isolate more initial queries"),
        SweepVariant("i3-b3-d1-s3", 3, 3, 1, 3, "Isolate wider branch coverage"),
        SweepVariant("i3-b1-d2-s3", 3, 1, 2, 3, "Isolate recursive depth"),
        SweepVariant("i3-b1-d1-s6", 3, 1, 1, 6, "Isolate report/subtopic structure"),
        SweepVariant("i5-b3-d2-s6", 5, 3, 2, 6, "Balanced likely-best config"),
        SweepVariant("i5-b4-d3-s8", 5, 4, 3, 8, "Expensive local stress test"),
    )


def select_variants(raw: str | None) -> tuple[SweepVariant, ...]:
    """Select variants by comma-separated id list."""

    variants = default_variants()
    if not raw:
        return variants
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    known = {variant.variant_id: variant for variant in variants}
    unknown = sorted(wanted - set(known))
    if unknown:
        raise SweepError(f"unknown variant id(s): {', '.join(unknown)}")
    return tuple(known[item] for item in known if item in wanted)


def ensure_loopback_base_url(base_url: str) -> str:
    """Return a normalized local endpoint or raise for non-loopback URLs."""

    normalized = str(base_url or "").rstrip("/")
    if not is_loopback_url(normalized):
        raise SweepError(
            "local-only guard rejected non-loopback base URL: "
            f"{normalized!r}"
        )
    return normalized


def experiment_root(workspace_root: Path) -> Path:
    return Path(workspace_root).resolve() / EXPERIMENT_RELATIVE_ROOT


def ensure_under_experiment_root(path: Path, root: Path) -> Path:
    """Resolve ``path`` and prove it stays under ``root``."""

    root_resolved = Path(root).resolve()
    path_resolved = Path(path).resolve()
    try:
        path_resolved.relative_to(root_resolved)
    except ValueError:
        raise SweepError(
            "refusing to write outside experiment root: "
            f"{path_resolved}"
        ) from None
    return path_resolved


def safe_experiment_child(root: Path, *parts: str) -> Path:
    candidate = Path(root).joinpath(*parts)
    return ensure_under_experiment_root(candidate, root)


def variant_dir(root: Path, variant_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", variant_id):
        raise SweepError(f"unsafe variant id: {variant_id!r}")
    return safe_experiment_child(root, "variants", variant_id)


def config_for_variant(
    variant: SweepVariant,
    controls: SweepControls,
) -> dict[str, Any]:
    """Build the per-variant GPT-Researcher-style config."""

    return {
        "schema_version": 1,
        "experiment": EXPERIMENT_SLUG,
        "variant_id": variant.variant_id,
        "purpose": variant.purpose,
        "topic": controls.topic,
        "source_run_id": controls.source_run_id,
        "vector_store_run_id": controls.vector_store_run_id,
        "RETRIEVER": "custom",
        "RETRIEVAL_MODE": "hybrid",
        "REPORT_SOURCE": "local",
        "WEB_SEARCH_ENABLED": False,
        "MAX_ITERATIONS": variant.max_iterations,
        "DEEP_RESEARCH_BREADTH": variant.breadth,
        "DEEP_RESEARCH_DEPTH": variant.depth,
        "DEEP_RESEARCH_CONCURRENCY": 2,
        "MAX_SUBTOPICS": variant.max_subtopics,
        "TOTAL_WORDS": controls.total_words,
        "SERVER_MAX_TOKENS": controls.server_max_tokens,
        "SMART_TOKEN_LIMIT": DEFAULT_SMART_TOKEN_LIMIT,
        "STRATEGIC_TOKEN_LIMIT": DEFAULT_STRATEGIC_TOKEN_LIMIT,
        "FAST_TOKEN_LIMIT": DEFAULT_FAST_TOKEN_LIMIT,
        "TEMPERATURE": controls.temperature,
        "RETRIEVAL_K": controls.retrieval_k,
        "CANDIDATE_MULTIPLIER": controls.candidate_multiplier,
        "RRF_K": controls.rrf_k,
        "BM25_WEIGHT": controls.bm25_weight,
        "VECTOR_WEIGHT": controls.vector_weight,
        "PAPER_MATCHING_RETRIEVAL_K": controls.paper_matching_retrieval_k,
        "EMBEDDING_PROVIDER": controls.embedding_provider,
        "EMBEDDING_MODEL": controls.embedding_model,
        "EMBEDDING_DIM": controls.embedding_dim,
        "LOCAL_LLM_RUNTIME_KIND": controls.llm_runtime_kind,
        "LOCAL_LLM_BASE_URL": controls.local_base_url,
        "LOCAL_LLM_SMART_MODEL": controls.llm_model,
        "LOCAL_LLM_FAST_MODEL": controls.fast_model,
        "SMART_LLM": f"local:{controls.llm_model}",
        "FAST_LLM": f"local:{controls.fast_model}",
        "STRATEGIC_LLM": f"local:{controls.llm_model}",
        "MAX_SEARCH_RESULTS_PER_QUERY": controls.retrieval_k,
        "CURATE_SOURCES": False,
        "BROWSE_CHUNK_MAX_LENGTH": DEFAULT_SMART_TOKEN_LIMIT,
        "SUMMARY_TOKEN_LIMIT": 700,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_json_if_exists(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def topic_title(workspace_root: Path, topic_id: str) -> str:
    try:
        for topic in load_topic_families(workspace_root):
            if topic.id == topic_id:
                return topic.title
    except Exception:
        pass
    return topic_id.replace("_", " ").title()


def evidence_terms(workspace_root: Path, source_run_id: str, topic_id: str) -> list[str]:
    payload = load_json_if_exists(
        workspace_root
        / "storage"
        / "extractions"
        / source_run_id
        / "by-topic"
        / f"{topic_id}.json"
    )
    out: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        cleaned = re.sub(r"\s+", " ", value).strip()
        if cleaned and cleaned.lower() not in {item.lower() for item in out}:
            out.append(cleaned)

    terms = payload.get("evidence_terms")
    if isinstance(terms, dict):
        for term, _count in sorted(
            terms.items(), key=lambda item: -int(item[1] or 0)
        ):
            add(term)
            if len(out) >= 10:
                return out
    if isinstance(terms, list):
        for term in terms:
            add(term)
            if len(out) >= 10:
                return out
    for entry in payload.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            continue
        for term in evidence.get("terms") or []:
            add(term)
            if len(out) >= 10:
                return out
    return out


def build_variant_questions(
    *,
    topic: str,
    terms: list[str],
    variant: SweepVariant,
) -> tuple[str, ...]:
    """Map iteration/breadth/depth knobs into deterministic local queries."""

    term_phrase = ", ".join(terms[:5]) if terms else "architecture, scaling, retrieval, memory"
    initial_questions = [
        f"What are the most-discussed methods, results, and limitations in {topic} at ICLR 2025?",
        f"Which datasets, benchmarks, and evaluation settings anchor {topic} claims in this source corpus?",
        f"What concrete failure modes, open problems, or tradeoffs recur across {topic} sessions?",
        f"How do the sources connect {topic} to recurring mechanisms such as {term_phrase}?",
        f"Which numeric claims, paper comparisons, and provenance details need stronger evidence for {topic}?",
    ]
    base_questions = initial_questions[: max(1, int(variant.max_iterations))]
    lenses = [
        "methods and mechanisms",
        "evaluation evidence",
        "limitations and failure modes",
        "cross-paper patterns",
        "deployment and efficiency tradeoffs",
        "retrieval, memory, and routing details",
        "data and benchmark coverage",
        "source provenance and unsupported claims",
    ]
    questions: list[str] = []
    for base in base_questions:
        questions.append(base)
        for depth_index in range(max(1, int(variant.depth))):
            for lens in lenses[: max(1, int(variant.breadth))]:
                questions.append(
                    f"{base} Depth pass {depth_index + 1}: focus on {lens}."
                )
    return tuple(_dedupe(questions))


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", str(value)).strip()
        key = normalized.casefold()
        if normalized and key not in seen:
            out.append(normalized)
            seen.add(key)
    return out


def build_variant_plan(
    *,
    workspace_root: Path,
    controls: SweepControls,
    variant: SweepVariant,
    bound_vector_store_digest: str,
    run_id: str,
) -> QueryPlan:
    title = topic_title(workspace_root, controls.topic)
    terms = evidence_terms(workspace_root, controls.source_run_id, controls.topic)
    questions = build_variant_questions(topic=title, terms=terms, variant=variant)
    synonyms = _dedupe([title, *terms])[:8]
    retrieval_filters = {"topic_ids": [controls.topic], "min_confidence": 0.6}
    source_gap_policy = {
        "min_local_hits": 5,
        "fallback": "none",
        "tavily_max_results": 0,
        "web_search_enabled": False,
    }
    plan_body = {
        "topic_id": controls.topic,
        "topic_title": title,
        "research_questions": list(questions),
        "synonym_expansion": synonyms,
        "entity_seeds": [],
        "retrieval_filters": retrieval_filters,
        "model": controls.llm_model,
        "provider": "local",
        "source_gap_policy": source_gap_policy,
        "bound_vector_store_digest": bound_vector_store_digest,
        "run_id": run_id,
        "sweep_variant": asdict(variant),
        "research_parameters": {
            "MAX_ITERATIONS": variant.max_iterations,
            "DEEP_RESEARCH_BREADTH": variant.breadth,
            "DEEP_RESEARCH_DEPTH": variant.depth,
            "DEEP_RESEARCH_CONCURRENCY": 2,
            "MAX_SUBTOPICS": variant.max_subtopics,
            "TOTAL_WORDS": controls.total_words,
            "SMART_TOKEN_LIMIT": DEFAULT_SMART_TOKEN_LIMIT,
            "STRATEGIC_TOKEN_LIMIT": DEFAULT_STRATEGIC_TOKEN_LIMIT,
            "FAST_TOKEN_LIMIT": DEFAULT_FAST_TOKEN_LIMIT,
            "TEMPERATURE": controls.temperature,
            "retrieval_mode": "hybrid",
            "retrieval_k": controls.retrieval_k,
            "candidate_multiplier": controls.candidate_multiplier,
            "rrf_k": controls.rrf_k,
            "bm25_weight": controls.bm25_weight,
            "vector_weight": controls.vector_weight,
        },
    }
    digest = compute_plan_digest(plan_body)
    return QueryPlan(
        topic_id=controls.topic,
        topic_title=title,
        research_questions=questions,
        synonym_expansion=tuple(synonyms),
        entity_seeds=(),
        retrieval_filters=retrieval_filters,
        model=controls.llm_model,
        provider="local",
        output_schema=REFERENCE_QUALITY_PROMPT,
        source_gap_policy=source_gap_policy,
        bound_vector_store_digest=bound_vector_store_digest,
        plan_digest=digest,
        generated_at=utc_now(),
        run_id=run_id,
        research_parameters=dict(plan_body["research_parameters"]),
    )


def embedding_client_config(controls: SweepControls) -> dict[str, Any]:
    """Return explicit embedding-client config for the active sweep lane."""

    config: dict[str, Any] = {
        "provider": controls.embedding_provider,
        "model": controls.embedding_model,
    }
    if controls.embedding_dim is not None:
        config["embedding_dim"] = controls.embedding_dim
    return config


def local_llm_endpoint_label(controls: SweepControls) -> str:
    """Return redacted provider provenance for the configured local runtime."""

    runtime_kind = controls.llm_runtime_kind.strip().lower()
    if runtime_kind not in VALID_LOCAL_LLM_RUNTIME_KINDS:
        raise SweepError(
            "llm_runtime_kind must be one of "
            f"{', '.join(VALID_LOCAL_LLM_RUNTIME_KINDS)}; got "
            f"{controls.llm_runtime_kind!r}"
        )
    return f"{runtime_kind}://{controls.llm_model}"


def required_receipt_digest(path: Path, *, label: str) -> str:
    """Return a required stage-receipt digest or fail closed."""

    if not path.exists():
        raise SweepError(f"missing required {label} receipt: {path}")
    return sha256_file(path)


def provider_receipt_snapshot(repo_root: Path) -> set[str]:
    root = repo_root / "storage"
    if not root.exists():
        return set()
    return {
        path.relative_to(repo_root).as_posix()
        for path in root.rglob("provider-calls/**/*.json")
        if path.is_file()
    }


def write_source_manifest(path: Path, record_payload: dict[str, Any]) -> None:
    citations = record_payload.get("citations") or []
    sources = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        sources.append(
            {
                "source_artifact_id": citation.get("source_artifact_id", ""),
                "source_artifact_path": citation.get("source_artifact_path", ""),
                "evidence_span_or_chunk_id": citation.get(
                    "evidence_span_or_chunk_id", ""
                ),
                "source_type": citation.get("source_type", "local"),
            }
        )
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "retrieved_evidence_ids": record_payload.get("retrieved_evidence_ids", []),
            "source_paths": record_payload.get("source_paths", []),
            "citations": sources,
            "source_count": len(sources),
        },
    )


def run_variant(
    *,
    workspace_root: Path,
    root: Path,
    controls: SweepControls,
    variant: SweepVariant,
    skip_paper_matching: bool,
    max_evidence_chunks: int,
) -> dict[str, Any]:
    """Execute one live local-only variant and return its metrics."""

    start_wall = time.time()
    start_perf = time.perf_counter()
    out_dir = variant_dir(root, variant.variant_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = config_for_variant(variant, controls)
    write_json_atomic(out_dir / "config.json", config)

    extraction_receipt = stage_path(
        workspace_root, controls.source_run_id, "exhaustive-topic-extraction"
    )
    vector_store_receipt = stage_path(
        workspace_root, controls.vector_store_run_id, "enhanced-indexing-vector-store"
    )
    bound_extraction_digest = required_receipt_digest(
        extraction_receipt,
        label="exhaustive-topic-extraction",
    )
    bound_vector_digest = required_receipt_digest(
        vector_store_receipt,
        label="enhanced-indexing-vector-store",
    )

    run_id = f"{controls.source_run_id}__{EXPERIMENT_SLUG}__{variant.variant_id}"
    plan = build_variant_plan(
        workspace_root=workspace_root,
        controls=controls,
        variant=variant,
        bound_vector_store_digest=bound_vector_digest,
        run_id=run_id,
    )

    research_dir = out_dir / "research"
    plans_dir = research_dir / "query-plans"
    findings_dir = research_dir / "findings"
    plans_dir.mkdir(parents=True, exist_ok=True)
    findings_dir.mkdir(parents=True, exist_ok=True)
    plan_path = (
        plans_dir
        / f"{safe_research_filename(plan.topic_id, plan.plan_digest)}-plan.json"
    )
    write_query_plan(plan_path, plan)
    write_json_atomic(out_dir / "query_plan.json", plan.as_payload())

    receipts_before = provider_receipt_snapshot(out_dir)
    with provider_receipt_context(
        workspace_root=out_dir,
        run_id=run_id,
        stage_id=EXPERIMENT_SLUG,
        caller=__file__,
    ):
        embedding_client = build_embedding_client(embedding_client_config(controls))
        retriever = BudgetedDedupRetriever(
            HybridChromaBM25Retriever(
                workspace_root,
                run_id=controls.vector_store_run_id,
                embedding_client=embedding_client,
                bm25_weight=controls.bm25_weight,
                vector_weight=controls.vector_weight,
                rrf_k=controls.rrf_k,
                candidate_multiplier=controls.candidate_multiplier,
            ),
            max_chunks=max_evidence_chunks,
        )
        researcher = build_researcher(
            {
                "provider": "local",
                "model": controls.llm_model,
                "fast_model": controls.fast_model,
                "smart_model": controls.llm_model,
                "base_url": controls.local_base_url,
                "endpoint_label": local_llm_endpoint_label(controls),
            }
        )

        findings_path = findings_dir / f"{controls.topic}.md"
        record = run_topic_research(
            workspace_root,
            plan=plan,
            retriever=retriever,
            researcher=researcher,
            tavily=None,
            findings_path=findings_path,
            bound_extraction_receipt_digest=bound_extraction_digest,
            bound_vector_store_digest=bound_vector_digest,
            run_id=run_id,
        )
    receipt_path = out_dir / "research_receipt.json"
    write_research_record(receipt_path, record)
    record_payload = record.as_payload()
    write_source_manifest(out_dir / "retrieved_source_manifest.json", record_payload)

    findings_text = read_text_if_exists(findings_path)
    write_text(out_dir / "gpt_researcher_findings.md", findings_text)

    paper_matching_summary: dict[str, Any] = {
        "status": "skipped",
        "reason": "skip_paper_matching",
    }
    if not skip_paper_matching:
        with provider_receipt_context(
            workspace_root=out_dir,
            run_id=run_id,
            stage_id=EXPERIMENT_SLUG,
            caller=__file__,
        ):
            try:
                paper_matching_summary = run_variant_paper_matching(
                    workspace_root=workspace_root,
                    controls=controls,
                    variant=variant,
                    run_id=run_id,
                    variant_output_dir=out_dir,
                    artifact_findings_path=out_dir / "gpt_researcher_findings.md",
                )
            except Exception as exc:  # noqa: BLE001 - experiment should keep artifacts.
                paper_matching_summary = {
                    "status": "failed",
                    "error": sanitize_for_receipt(str(exc)),
                    "keypoint_count": 0,
                    "matched_paper_count": 0,
                    "matched_paper_titles": [],
                    "token_counts": {},
                }
                write_json_atomic(
                    out_dir / "paper_matching" / "error.json",
                    {
                        "schema_version": 1,
                        "variant_id": variant.variant_id,
                        "status": "failed",
                        "error": paper_matching_summary["error"],
                    },
                )

    final_report = compose_final_report(
        variant=variant,
        controls=controls,
        findings_text=findings_text,
        paper_matching_summary=paper_matching_summary,
    )
    final_report_path = out_dir / "final_report.md"
    write_text(final_report_path, final_report)

    receipts_after = provider_receipt_snapshot(out_dir)
    provider_receipts = sorted(receipts_after - receipts_before)
    provider_receipts.extend(record_payload.get("provider_usage", {}).get("provider_receipt_paths", []))
    provider_receipts = sorted(set(str(item) for item in provider_receipts if item))
    write_json_atomic(
        out_dir / "provider_call_receipts.json",
        {
            "schema_version": 1,
            "variant_id": variant.variant_id,
            "provider_receipt_paths": provider_receipts,
        },
    )

    elapsed_s = round(time.perf_counter() - start_perf, 3)
    timing = {
        "schema_version": 1,
        "variant_id": variant.variant_id,
        "started_at_epoch": start_wall,
        "finished_at": utc_now(),
        "elapsed_s": elapsed_s,
    }
    write_json_atomic(out_dir / "timing.json", timing)

    token_counts = {
        "schema_version": 1,
        "variant_id": variant.variant_id,
        "input_tokens": int(record_payload.get("provider_usage", {}).get("input_tokens") or 0),
        "output_tokens": int(record_payload.get("provider_usage", {}).get("output_tokens") or 0),
        "request_count": int(getattr(researcher, "request_count", 0) or 0),
        "paper_matching": paper_matching_summary.get("token_counts", {}),
    }
    write_json_atomic(out_dir / "token_counts.json", token_counts)

    metrics = measure_report(
        final_report,
        variant_id=variant.variant_id,
        report_path=final_report_path,
        record_payload=record_payload,
        paper_matching_summary=paper_matching_summary,
        timing=timing,
        token_counts=token_counts,
        provider_receipts=provider_receipts,
    )
    write_json_atomic(out_dir / "metrics.json", metrics)
    return metrics


def run_variant_paper_matching(
    *,
    workspace_root: Path,
    controls: SweepControls,
    variant: SweepVariant,
    run_id: str,
    variant_output_dir: Path,
    artifact_findings_path: Path,
) -> dict[str, Any]:
    """Run paper matching for a variant with local-only components."""

    embedding_client = build_embedding_client(embedding_client_config(controls))
    paper_researcher = build_researcher(
        {
            "provider": "local",
            "model": controls.llm_model,
            "fast_model": controls.fast_model,
            "smart_model": controls.llm_model,
            "base_url": controls.local_base_url,
            "endpoint_label": local_llm_endpoint_label(controls),
        }
    )
    paper_runner = StoreBackedResearchRunner(
        workspace_root=workspace_root,
        run_id=controls.vector_store_run_id,
        store_kind="papers",
        retriever=HybridChromaBM25Retriever(
            workspace_root,
            run_id=controls.vector_store_run_id,
            store_kind="papers",
            embedding_client=embedding_client,
            bm25_weight=controls.bm25_weight,
            vector_weight=controls.vector_weight,
            rrf_k=controls.rrf_k,
            candidate_multiplier=controls.candidate_multiplier,
        ),
        researcher=paper_researcher,
        k=controls.paper_matching_retrieval_k,
    )
    config = PaperMatchingConfig(
        enabled=True,
        keypoint_provider="local",
        keypoint_model=controls.llm_model,
        research_provider="local",
        research_model=controls.llm_model,
        output_top_n=20,
    )
    result = run_paper_matching(
        workspace_root=workspace_root,
        run_id=run_id,
        topic_id=controls.topic,
        artifact_findings_path=artifact_findings_path,
        output_dir=variant_output_dir / "paper_matching",
        decks_dir=variant_output_dir / "deck_compat",
        config=config,
        keypoint_extractor=LocalKeypointExtractor(
            model=controls.llm_model,
            base_url=controls.local_base_url,
            max_keypoints=20,
        ),
        paper_research_runner=paper_runner,
        output_top_n=20,
        write_compat_ranked=False,
    )
    keypoints_payload = load_json_if_exists(result.keypoints_path)
    matched_payload = load_json_if_exists(result.matched_papers_path)
    token_counts = paper_researcher.usage() if hasattr(paper_researcher, "usage") else {}
    return {
        "status": "succeeded",
        "keypoints_path": str(result.keypoints_path),
        "paper_research_path": str(result.paper_research_path),
        "matched_papers_path": str(result.matched_papers_path),
        "keypoint_count": len(keypoints_payload.get("keypoints") or []),
        "matched_paper_count": len(matched_payload.get("papers") or []),
        "matched_paper_titles": [
            str(paper.get("title") or "")
            for paper in matched_payload.get("papers") or []
            if isinstance(paper, dict)
        ],
        "token_counts": token_counts,
        "max_subtopics": variant.max_subtopics,
    }


def compose_final_report(
    *,
    variant: SweepVariant,
    controls: SweepControls,
    findings_text: str,
    paper_matching_summary: dict[str, Any],
) -> str:
    papers = paper_matching_summary.get("matched_paper_titles") or []
    paper_lines = "\n".join(f"- {title}" for title in papers[:20]) or "- Not run."
    return (
        f"# GPT-Researcher Parameter Sweep: {controls.topic} / {variant.variant_id}\n\n"
        "## Experiment Controls\n\n"
        f"- Source run: `{controls.source_run_id}`\n"
        f"- Local model: `{controls.llm_model}`\n"
        f"- Embedding model: `{controls.embedding_model}`\n"
        f"- Temperature: `{controls.temperature}`\n"
        "- Web search: disabled\n\n"
        "## Variant Parameters\n\n"
        f"- MAX_ITERATIONS: `{variant.max_iterations}`\n"
        f"- DEEP_RESEARCH_BREADTH: `{variant.breadth}`\n"
        f"- DEEP_RESEARCH_DEPTH: `{variant.depth}`\n"
        f"- MAX_SUBTOPICS: `{variant.max_subtopics}`\n\n"
        "## GPT-Researcher Findings\n\n"
        f"{findings_text.strip()}\n\n"
        "## Paper Matching Summary\n\n"
        f"- Status: `{paper_matching_summary.get('status', 'unknown')}`\n"
        f"- Keypoints: `{paper_matching_summary.get('keypoint_count', 0)}`\n"
        f"- Matched papers: `{paper_matching_summary.get('matched_paper_count', 0)}`\n\n"
        f"{paper_lines}\n"
    )


def measure_report(
    text: str,
    *,
    variant_id: str,
    report_path: Path,
    record_payload: dict[str, Any] | None = None,
    paper_matching_summary: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    token_counts: dict[str, Any] | None = None,
    provider_receipts: list[str] | None = None,
) -> dict[str, Any]:
    record_payload = record_payload or {}
    paper_matching_summary = paper_matching_summary or {}
    timing = timing or {}
    token_counts = token_counts or {}
    provider_receipts = provider_receipts or []
    narrative = validate_narrative_topic_report(text)
    source_paths = record_payload.get("source_paths") or []
    citations = record_payload.get("citations") or []
    openreview_urls = sorted(set(re.findall(r"https://openreview\.net/forum\?id=[A-Za-z0-9_-]+", text)))
    extraction_ids = sorted(
        set(
            re.findall(
                r"(?:media|paper_metadata_fallback|[0-9]{2}_[a-z0-9_]+)[A-Za-z0-9_.:/#-]*",
                text,
            )
        )
    )
    malformed_heading_count = len(
        re.findall(r"^#{1,6}\s+#{1,6}\s+", text, flags=re.MULTILINE)
    )
    unsupported_numeric_claims = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"\b\d+(?:\.\d+)?(?:%|x|X|\s*(?:tokens|papers|chunks|hours))\b", line)
        and not re.search(r"(source|chunk|citation|reference|\[|`)", line, re.IGNORECASE)
    ]
    metrics = {
        "schema_version": 1,
        "variant_id": variant_id,
        "report_path": str(report_path),
        "word_count": len(re.findall(r"\b[\w'-]+\b", text)),
        "section_count": len(re.findall(r"^##\s+", text, flags=re.MULTILINE)),
        "subsection_count": len(re.findall(r"^###\s+", text, flags=re.MULTILINE)),
        "unique_paper_title_count": len(set(paper_matching_summary.get("matched_paper_titles") or [])),
        "openreview_url_count": len(openreview_urls),
        "source_path_count": len(set(str(path) for path in source_paths)),
        "extraction_id_count": len(extraction_ids),
        "citation_count": len(citations),
        "vector_chunk_count": len(record_payload.get("retrieved_evidence_ids") or []),
        "transcript_media_reference_count": len(
            [
                path
                for path in source_paths
                if "media_" in str(path) or "audio" in str(path) or "transcript" in str(path)
            ]
        ),
        "keypoint_count": int(paper_matching_summary.get("keypoint_count") or 0),
        "matched_paper_count": int(paper_matching_summary.get("matched_paper_count") or 0),
        "malformed_heading_count": malformed_heading_count,
        "forbidden_headings": narrative.get("forbidden_headings", []),
        "narrative_validation": narrative,
        "unsupported_numeric_claim_count": len(unsupported_numeric_claims),
        "unsupported_numeric_claim_examples": unsupported_numeric_claims[:10],
        "missing_provenance": len(citations) == 0,
        "elapsed_s": timing.get("elapsed_s"),
        "input_tokens": token_counts.get("input_tokens", 0),
        "output_tokens": token_counts.get("output_tokens", 0),
        "model_call_count": token_counts.get("request_count", 0),
        "provider_receipt_count": len(provider_receipts),
    }
    metrics["reference_or_source_item_count"] = max(
        int(metrics["citation_count"]),
        int(metrics["source_path_count"]),
        int(narrative.get("reference_item_count") or 0),
    )
    metrics["decision_rule_pass"] = decision_rule_pass(metrics)
    metrics["score"] = score_metrics(metrics)
    return metrics


def decision_rule_pass(metrics: dict[str, Any]) -> bool:
    narrative = metrics.get("narrative_validation") or {}
    return (
        int(metrics.get("word_count") or 0) >= 3500
        and int(metrics.get("matched_paper_count") or 0) >= 8
        and int(metrics.get("keypoint_count") or 0) >= 16
        and int(metrics.get("reference_or_source_item_count") or 0) >= 40
        and narrative.get("status") == "pass"
    )


def score_metrics(metrics: dict[str, Any]) -> float:
    narrative = metrics.get("narrative_validation") or {}
    score = 0.0
    score += min(int(metrics.get("word_count") or 0), 3500) / 35.0
    score += 15.0 * min(int(metrics.get("matched_paper_count") or 0), 8) / 8.0
    score += 15.0 * min(int(metrics.get("keypoint_count") or 0), 16) / 16.0
    score += 20.0 * min(int(metrics.get("reference_or_source_item_count") or 0), 40) / 40.0
    score += 10.0 if narrative.get("status") == "pass" else 0.0
    score -= 10.0 * int(metrics.get("malformed_heading_count") or 0)
    score -= 5.0 * len(metrics.get("forbidden_headings") or [])
    score -= 2.0 * int(metrics.get("unsupported_numeric_claim_count") or 0)
    if metrics.get("missing_provenance"):
        score -= 15.0
    return round(score, 3)


def baseline_metrics(
    *,
    workspace_root: Path,
    label: str,
    report_path: Path,
) -> dict[str, Any] | None:
    text = read_text_if_exists(report_path)
    if not text:
        return None
    metrics = measure_report(text, variant_id=label, report_path=report_path)
    metrics["baseline"] = True
    return metrics


def load_completed_variant_metrics(root: Path) -> list[dict[str, Any]]:
    variants_root = root / "variants"
    if not variants_root.exists():
        return []
    out: list[dict[str, Any]] = []
    for metrics_path in sorted(variants_root.glob("*/metrics.json")):
        payload = load_json_if_exists(metrics_path)
        if payload:
            out.append(payload)
    return out


def write_comparison(
    *,
    workspace_root: Path,
    root: Path,
    topic: str,
) -> dict[str, Any]:
    """Create comparison.json and comparison.md."""

    variants = load_completed_variant_metrics(root)
    baselines = [
        item
        for item in (
            baseline_metrics(
                workspace_root=workspace_root,
                label="current_local_report",
                report_path=workspace_root / "analysis" / f"{topic}.md",
            ),
            baseline_metrics(
                workspace_root=workspace_root,
                label="before_local_report",
                report_path=workspace_root
                / "storage"
                / "experiments"
                / "topic2-local-analysis"
                / f"{topic}.before-local.md",
            ),
        )
        if item is not None
    ]
    ranked_variants = sorted(
        variants,
        key=lambda item: (bool(item.get("decision_rule_pass")), float(item.get("score") or 0.0)),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_SLUG,
        "topic": topic,
        "generated_at": utc_now(),
        "decision_rule": {
            "min_words": 3500,
            "min_matched_papers": 8,
            "min_keypoints": 16,
            "min_references_or_source_items": 40,
            "narrative_validation": "pass",
        },
        "ranked_variants": ranked_variants,
        "baselines": baselines,
        "conclusion": comparison_conclusion(ranked_variants),
    }
    write_json_atomic(root / "comparison.json", payload)
    write_text(root / "comparison.md", comparison_markdown(payload))
    return payload


def comparison_conclusion(ranked_variants: list[dict[str, Any]]) -> str:
    if not ranked_variants:
        return "No completed variant metrics yet. Run without --dry-run to evaluate quality."
    best = ranked_variants[0]
    if best.get("decision_rule_pass"):
        return (
            f"Best local config is {best.get('variant_id')} and it passes the "
            "decision rule."
        )
    return (
        "No completed local config reaches the decision rule yet. Local budget "
        "may improve coverage, but this result points to better scaffolding or "
        "a stronger local model before claiming parity."
    )


def comparison_markdown(payload: dict[str, Any]) -> str:
    rows = []
    for row in payload.get("ranked_variants") or []:
        rows.append(
            "| {variant} | {score} | {pass_} | {words} | {papers} | {keys} | {refs} | {narr} |".format(
                variant=row.get("variant_id", ""),
                score=row.get("score", ""),
                pass_="yes" if row.get("decision_rule_pass") else "no",
                words=row.get("word_count", ""),
                papers=row.get("matched_paper_count", ""),
                keys=row.get("keypoint_count", ""),
                refs=row.get("reference_or_source_item_count", ""),
                narr=(row.get("narrative_validation") or {}).get("status", ""),
            )
        )
    if not rows:
        rows.append("| none | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    baseline_rows = []
    for row in payload.get("baselines") or []:
        baseline_rows.append(
            "| {variant} | {score} | {words} | {papers} | {keys} | {refs} | {narr} |".format(
                variant=row.get("variant_id", ""),
                score=row.get("score", ""),
                words=row.get("word_count", ""),
                papers=row.get("matched_paper_count", ""),
                keys=row.get("keypoint_count", ""),
                refs=row.get("reference_or_source_item_count", ""),
                narr=(row.get("narrative_validation") or {}).get("status", ""),
            )
        )
    if not baseline_rows:
        baseline_rows.append("| none | n/a | n/a | n/a | n/a | n/a | n/a |")
    return (
        f"# GPT-Researcher Parameter Sweep Comparison\n\n"
        f"Topic: `{payload.get('topic')}`\n\n"
        "## Ranked Variants\n\n"
        "| Variant | Score | Decision Rule | Words | Papers | Keypoints | Refs/Sources | Narrative |\n"
        "|---|---:|---|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\n"
        "## Baselines\n\n"
        "| Baseline | Score | Words | Papers | Keypoints | Refs/Sources | Narrative |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(baseline_rows)
        + "\n\n"
        "## Conclusion\n\n"
        f"{payload.get('conclusion')}\n"
    )


def write_dry_run_plan(
    *,
    workspace_root: Path,
    root: Path,
    controls: SweepControls,
    variants: tuple[SweepVariant, ...],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        root / "plan.json",
        {
            "schema_version": 1,
            "experiment": EXPERIMENT_SLUG,
            "status": "planned",
            "dry_run": True,
            "controls": asdict(controls),
            "variants": [asdict(variant) for variant in variants],
        },
    )
    for variant in variants:
        out_dir = variant_dir(root, variant.variant_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out_dir / "config.json", config_for_variant(variant, controls))
        write_json_atomic(
            out_dir / "dry_run.json",
            {
                "schema_version": 1,
                "variant_id": variant.variant_id,
                "status": "planned",
                "provider_calls": 0,
                "writes_production_analysis": False,
            },
        )
    write_comparison(workspace_root=workspace_root, root=root, topic=controls.topic)


def guard_live_local_only(
    controls: SweepControls, *, allow_openai_embeddings: bool = False
) -> None:
    ensure_loopback_base_url(controls.local_base_url)
    if controls.embedding_provider == "openai" and not allow_openai_embeddings:
        raise SweepError(
            "OpenAI vector-store queries require paid OpenAI embeddings. "
            "Pass --allow-openai-embeddings to acknowledge this."
        )
    if controls.embedding_provider != "openai" and os.environ.get("OPENAI_API_KEY"):
        raise SweepError(
            "local-only experiment requires OPENAI_API_KEY to be unset. "
            "Run with env -u OPENAI_API_KEY ..."
        )
    if controls.web_search_enabled:
        raise SweepError("web search must remain disabled for this experiment")


def build_controls(args: argparse.Namespace) -> SweepControls:
    base_url = ensure_loopback_base_url(
        args.local_base_url
        or os.environ.get("GPTR_LOCAL_LLM_BASE_URL")
        or DEFAULT_LOCAL_BASE_URL
    )
    smart_model = (
        args.smart_model
        or os.environ.get("GPTR_LOCAL_LLM_SMART_MODEL")
        or DEFAULT_LLM_MODEL
    )
    fast_model = (
        args.fast_model
        or os.environ.get("GPTR_LOCAL_LLM_FAST_MODEL")
        or smart_model
    )
    llm_runtime_kind = (
        args.llm_runtime_kind
        or os.environ.get("GPTR_LOCAL_LLM_RUNTIME_KIND")
        or "mlx"
    ).strip().lower()
    if llm_runtime_kind not in VALID_LOCAL_LLM_RUNTIME_KINDS:
        raise SweepError(
            "llm_runtime_kind must be one of "
            f"{', '.join(VALID_LOCAL_LLM_RUNTIME_KINDS)}; got "
            f"{llm_runtime_kind!r}"
        )
    embedding_model = (
        args.embedding_model
        or os.environ.get("EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    embedding_dim: int | None = None
    raw_embedding_dim = args.embedding_dim or os.environ.get("EMBEDDING_DIM") or ""
    if raw_embedding_dim:
        try:
            embedding_dim = int(raw_embedding_dim)
        except (TypeError, ValueError):
            raise SweepError(
                f"embedding dimension must be a positive integer; got {raw_embedding_dim!r}"
            ) from None
        if embedding_dim <= 0:
            raise SweepError(
                f"embedding dimension must be a positive integer; got {raw_embedding_dim!r}"
            )
    retrieval_k = _positive_int_arg(
        args.retrieval_k or os.environ.get("GPTR_RETRIEVAL_K") or os.environ.get("RETRIEVAL_K"),
        DEFAULT_RETRIEVAL_K,
        "retrieval_k",
    )
    candidate_multiplier = _positive_int_arg(
        args.candidate_multiplier
        or os.environ.get("GPTR_CANDIDATE_MULTIPLIER")
        or os.environ.get("CANDIDATE_MULTIPLIER"),
        DEFAULT_CANDIDATE_MULTIPLIER,
        "candidate_multiplier",
    )
    rrf_k = _positive_int_arg(
        args.rrf_k or os.environ.get("GPTR_RRF_K") or os.environ.get("RRF_K"),
        DEFAULT_RRF_K,
        "rrf_k",
    )
    bm25_weight = _optional_nonnegative_float_arg(
        args.bm25_weight
        if args.bm25_weight is not None
        else os.environ.get("GPTR_BM25_WEIGHT") or os.environ.get("BM25_WEIGHT"),
        "bm25_weight",
    )
    vector_weight = _optional_nonnegative_float_arg(
        args.vector_weight
        if args.vector_weight is not None
        else os.environ.get("GPTR_VECTOR_WEIGHT") or os.environ.get("VECTOR_WEIGHT"),
        "vector_weight",
    )
    paper_matching_retrieval_k = _positive_int_arg(
        args.paper_matching_retrieval_k
        or os.environ.get("PAPER_MATCHING_RETRIEVAL_K")
        or retrieval_k,
        retrieval_k,
        "paper_matching_retrieval_k",
    )
    return SweepControls(
        topic=args.topic,
        source_run_id=args.source_run_id,
        vector_store_run_id=args.vector_store_run_id or args.source_run_id,
        embedding_provider=args.embedding_provider
        or os.environ.get("EMBEDDING_PROVIDER")
        or "mlx",
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        llm_runtime_kind=llm_runtime_kind,
        llm_model=smart_model,
        fast_model=fast_model,
        local_base_url=base_url,
        temperature=float(args.temperature),
        server_max_tokens=int(args.server_max_tokens),
        web_search_enabled=False,
        total_words=int(args.total_words),
        retrieval_k=retrieval_k,
        candidate_multiplier=candidate_multiplier,
        rrf_k=rrf_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        paper_matching_retrieval_k=paper_matching_retrieval_k,
    )


def _positive_int_arg(value: Any, default: int, label: str) -> int:
    if value is None or value == "":
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise SweepError(f"{label} must be a positive integer; got {value!r}") from None
    if parsed <= 0:
        raise SweepError(f"{label} must be a positive integer; got {value!r}")
    return parsed


def _optional_nonnegative_float_arg(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise SweepError(f"{label} must be non-negative; got {value!r}") from None
    if parsed < 0:
        raise SweepError(f"{label} must be non-negative; got {value!r}")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an experiment-only local GPT-Researcher parameter sweep."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--vector-store-run-id", default="")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--variants", default="")
    parser.add_argument("--experiment-slug", default=EXPERIMENT_SLUG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--skip-paper-matching", action="store_true")
    parser.add_argument("--embedding-provider", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-dim", type=int, default=0)
    parser.add_argument("--llm-runtime-kind", default="")
    parser.add_argument("--local-base-url", default="")
    parser.add_argument("--smart-model", default="")
    parser.add_argument("--fast-model", default="")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--server-max-tokens", type=int, default=DEFAULT_SMART_TOKEN_LIMIT)
    parser.add_argument("--total-words", type=int, default=DEFAULT_TOTAL_WORDS)
    parser.add_argument("--max-evidence-chunks", type=int, default=DEFAULT_MAX_EVIDENCE_CHUNKS)
    parser.add_argument("--retrieval-k", type=int, default=0)
    parser.add_argument("--candidate-multiplier", type=int, default=0)
    parser.add_argument("--rrf-k", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=None)
    parser.add_argument("--vector-weight", type=float, default=None)
    parser.add_argument("--paper-matching-retrieval-k", type=int, default=0)
    parser.add_argument("--allow-openai-embeddings", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    global EXPERIMENT_SLUG, EXPERIMENT_RELATIVE_ROOT
    args = parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", args.experiment_slug):
        raise SweepError(f"unsafe experiment slug: {args.experiment_slug!r}")
    EXPERIMENT_SLUG = args.experiment_slug
    EXPERIMENT_RELATIVE_ROOT = Path("storage") / "experiments" / EXPERIMENT_SLUG
    workspace_root = args.workspace.resolve()
    controls = build_controls(args)
    variants = select_variants(args.variants)
    root = experiment_root(workspace_root)

    if args.compare_only:
        write_comparison(workspace_root=workspace_root, root=root, topic=controls.topic)
        return 0

    if args.dry_run:
        write_dry_run_plan(
            workspace_root=workspace_root,
            root=root,
            controls=controls,
            variants=variants,
        )
        return 0

    guard_live_local_only(
        controls, allow_openai_embeddings=bool(args.allow_openai_embeddings)
    )
    root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    for variant in variants:
        completed.append(
            run_variant(
                workspace_root=workspace_root,
                root=root,
                controls=controls,
                variant=variant,
                skip_paper_matching=bool(args.skip_paper_matching),
                max_evidence_chunks=int(args.max_evidence_chunks),
            )
        )
    write_comparison(workspace_root=workspace_root, root=root, topic=controls.topic)
    print(
        json.dumps(
            {
                "status": "succeeded",
                "experiment_root": relpath(root, workspace_root),
                "completed_variants": [item.get("variant_id") for item in completed],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
