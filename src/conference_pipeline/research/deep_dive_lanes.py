"""Bounded deep-dive question selection and dual GPT-Researcher lanes.

This module turns broad exhaustive topic summaries into a small, slide-usable
set of deep-dive questions per topic, then runs two research lanes for each
selected question:

* a global, no-topic-id hybrid vector-store lane over stable corpus metadata;
* a keyword-prefiltered lane using the explicit heuristic evidence pack owner.

The selector uses the governed provider/model for
``analysis.deep_dive_question_selection``. The research lanes default
to a loopback local Qwen 3.6 35B-A3B OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.extractions.records import safe_artifact_filename
from conference_pipeline.iclr2025_live import relpath
from conference_pipeline.model_governance import default_model, default_provider, validate_selection
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_chat_completions_create,
    provider_receipt_context,
    sanitize_for_receipt,
)
from conference_pipeline.receipt_lineage import (
    archive_obsolete_outputs,
    current_stage_output_digest,
    sanitize_receipt_for_write,
    stamp_substep_receipt,
    unbind_substep_receipt,
)
from conference_pipeline.research.engine_choice import (
    ENGINE_GPT_RESEARCHER,
    ENGINE_INHOUSE,
    VALID_ENGINES,
    ResearchEngineNotChosenError,
    resolve_research_engine,
)
from conference_pipeline.research.heuristic_prefilter import (
    DEFAULT_STORES,
    build_prefiltered_evidence_pack,
    write_prefiltered_evidence_pack,
)
from conference_pipeline.research.researcher import (
    HybridChromaBM25Retriever,
    RetrievedChunk,
    build_researcher,
)
from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline.model_governance.model_families import uses_openai_reasoning_params

OWNER_SELECTION = "analysis.deep_dive_question_selection"
OWNER_DUAL_LANES = "analysis.dual_deep_dive_research_lanes"
RECEIPT_TYPE_SELECTION = "conference_pipeline.deep_dive_question_selection"
RECEIPT_TYPE_DUAL_LANES = "conference_pipeline.dual_deep_dive_research_lanes"
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_SELECTION_PROVIDER = default_provider(OWNER_SELECTION)
DEFAULT_SELECTION_MODEL = default_model(OWNER_SELECTION)
# The lane model comes from the registry task, never a literal.
DEFAULT_QWEN36_MODEL = default_model_for_provider("research.gpt_researcher_smart", "local")
DEFAULT_MAX_QUESTIONS_PER_TOPIC = 5
DEFAULT_MIN_QUESTIONS_PER_TOPIC = 4
DEFAULT_MAX_OUTPUT_TOKENS = 8_000
DEFAULT_RETRIEVAL_K_PER_STORE = 12
DEFAULT_PREFILTER_LIMIT = 32
DEFAULT_RESEARCH_MAX_TOKENS = 24_000
VALID_LANES = ("global", "heuristic")
# Which engine researches a lane: the GPT-Researcher package on our hybrid
# index and local model (default), or the in-house single-call synthesis.
# No default engine: each conference's operator chooses (research.engine_choice).
DEFAULT_ENGINE: str | None = None


class DeepDiveLaneError(RuntimeError):
    """Raised when the deep-dive workflow cannot safely continue."""


class RetrieverLike(Protocol):
    def retrieve(self, *, topic_id: str, query: str, k: int) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class SummaryPair:
    topic_id: str
    title: str
    summaries: dict[str, dict[str, Any]]
    summary_paths: dict[str, str]


@dataclass(frozen=True)
class DeepDiveQuestion:
    deep_dive_id: str
    topic_id: str
    topic_title: str
    question: str
    rationale: str
    keywords: tuple[str, ...]
    source_categories: tuple[str, ...] = ("transcripts", "papers")
    priority_score: float = 0.0
    slide_usefulness: str = ""
    evidence_handles: tuple[str, ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "deep_dive_id": self.deep_dive_id,
            "topic_id": self.topic_id,
            "topic_title": self.topic_title,
            "question": self.question,
            "rationale": self.rationale,
            "keywords": list(self.keywords),
            "source_categories": list(self.source_categories),
            "priority_score": self.priority_score,
            "slide_usefulness": self.slide_usefulness,
            "evidence_handles": list(self.evidence_handles),
        }


@dataclass(frozen=True)
class LaneOutput:
    deep_dive_id: str
    lane: str
    status: str
    output_markdown: str | None
    evidence_json: str | None
    evidence_count: int
    provider_receipt_paths: tuple[str, ...] = ()
    blockers: tuple[dict[str, Any], ...] = ()

    def as_payload(self) -> dict[str, Any]:
        return {
            "deep_dive_id": self.deep_dive_id,
            "lane": self.lane,
            "status": self.status,
            "output_markdown": self.output_markdown,
            "evidence_json": self.evidence_json,
            "evidence_count": self.evidence_count,
            "provider_receipt_paths": list(self.provider_receipt_paths),
            "blockers": [dict(item) for item in self.blockers],
        }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepDiveLaneError(f"json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise DeepDiveLaneError(f"json_not_object:{path}")
    return payload


def _summary_index_path(workspace: Path) -> Path:
    return workspace / "analysis" / "exhaustive_summaries" / "index.json"


def _selection_output_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "analysis" / "deep-dive-topics" / run_id / "selected_deep_dives.json"


def _selection_receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07B-deep-dive-question-selection.json"


def _dual_lanes_receipt_path(workspace: Path, run_id: str) -> Path:
    return workspace / "storage" / "workflow-runs" / run_id / "07C-dual-deep-dive-research-lanes.json"


def _deep_dive_root(workspace: Path, run_id: str, deep_dive_id: str) -> Path:
    return workspace / "storage" / "analysis" / "deep-dive-topics" / run_id / deep_dive_id


def _slugify(value: str, *, limit: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized[:limit].strip("_") or "deep_dive"


def _deep_dive_id_for(topic_id: str, question: str, *, index: int) -> str:
    topic_stem = safe_artifact_filename(topic_id)[:32].strip("._") or "topic"
    prefix = f"{topic_stem}__dd{index:02d}_"
    # Provider receipt sanitization redacts 80+ character opaque-token-shaped
    # strings. Keep ids shorter than that without giving up deterministic names.
    slug_limit = max(16, 62 - len(prefix))
    return f"{prefix}{_slugify(question, limit=slug_limit)}"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def load_summary_pairs(workspace: Path, *, run_id: str) -> list[SummaryPair]:
    workspace = workspace.resolve()
    index = _load_json(_summary_index_path(workspace))
    if index.get("run_id") != run_id:
        raise DeepDiveLaneError("exhaustive_summary_index_run_id_mismatch")
    grouped: dict[str, SummaryPair] = {}
    for rel_summary in index.get("summaries") or []:
        path = workspace / str(rel_summary)
        payload = _load_json(path)
        topic_id = str(payload.get("topic_id") or "").strip()
        category = str(payload.get("category") or "").strip()
        title = str(payload.get("title") or topic_id).strip()
        if not topic_id or category not in {"transcripts", "papers"}:
            continue
        existing = grouped.get(topic_id)
        summaries = dict(existing.summaries) if existing else {}
        paths = dict(existing.summary_paths) if existing else {}
        summaries[category] = payload
        paths[category] = relpath(path, workspace)
        grouped[topic_id] = SummaryPair(
            topic_id=topic_id,
            title=title or topic_id,
            summaries=summaries,
            summary_paths=paths,
        )
    pairs = sorted(grouped.values(), key=lambda item: item.topic_id)
    if not pairs:
        raise DeepDiveLaneError("exhaustive_summary_pairs_missing")
    return pairs


def _compact_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "category": payload.get("category"),
        "summary": payload.get("summary"),
        "key_points": _as_list(payload.get("key_points"))[:16],
        "tensions_or_contrasts": _as_list(payload.get("tensions_or_contrasts"))[:12],
        "open_questions": _as_list(payload.get("open_questions"))[:10],
        "analysis_feed": payload.get("analysis_feed") if isinstance(payload.get("analysis_feed"), dict) else {},
        "notable_evidence": _as_list(payload.get("notable_evidence"))[:12],
        "input_tokens": payload.get("input_tokens"),
        "unit_count": payload.get("unit_count"),
    }


def build_selection_prompt(
    pair: SummaryPair,
    *,
    min_questions: int,
    max_questions: int,
    conference_label: str = "the conference",
) -> str:
    compact = {
        category: _compact_summary(summary)
        for category, summary in sorted(pair.summaries.items())
    }
    return (
        f"You are selecting a bounded set of deep-research questions for a {conference_label} "
        "conference analysis topic. The source summaries are already exhaustive "
        "topic/category synthesis packs. Do not turn every bullet into a research "
        "task. Cluster overlapping ideas and select only the questions that are "
        "most useful for a later slide narrative.\n\n"
        f"Topic id: {pair.topic_id}\n"
        f"Topic title: {pair.title}\n"
        f"Select between {min_questions} and {max_questions} deep dives.\n\n"
        "Return only JSON with this shape:\n"
        "{\n"
        "  \"deep_dives\": [\n"
        "    {\n"
        "      \"question\": \"one research question\",\n"
        "      \"rationale\": \"why this deserves deep research\",\n"
        "      \"keywords\": [\"6-16 search terms or phrases\"],\n"
        "      \"source_categories\": [\"transcripts\", \"papers\"],\n"
        "      \"priority_score\": 0.0,\n"
        "      \"slide_usefulness\": \"how it could shape slides\",\n"
        "      \"evidence_handles\": [\"unit ids or source labels from notable evidence\"]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Selection rules:\n"
        "- Prefer questions that integrate papers and talks when possible.\n"
        "- Preserve tensions and contrasts, but do not create separate questions for near-duplicates.\n"
        "- Favor concrete methods, evaluation shifts, systems, datasets, and scientific consequences.\n"
        "- Avoid broad topic restatements and purely logistical questions.\n\n"
        "SOURCE SUMMARIES:\n"
        f"{json.dumps(compact, ensure_ascii=True, sort_keys=True)}"
    )


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = str(text or "").strip()
    match = _JSON_FENCE_RE.search(candidate)
    if match:
        candidate = match.group(1).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _openai_options(model: str, max_completion_tokens: int) -> dict[str, Any]:
    if uses_openai_reasoning_params(model):
        return {"max_completion_tokens": max_completion_tokens}
    return {"max_tokens": max_completion_tokens, "temperature": 0.1}


def _build_openai_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise DeepDiveLaneError("openai_sdk_unavailable") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise DeepDiveLaneError("missing_openai_api_key")
    timeout = float(os.environ.get("DEEP_DIVE_SELECTION_OPENAI_TIMEOUT_SECONDS", "300"))
    return OpenAI(api_key=api_key, timeout=max(1.0, timeout))


def _build_anthropic_client() -> Any:
    try:
        from anthropic import Anthropic  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise DeepDiveLaneError("anthropic_sdk_unavailable") from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise DeepDiveLaneError("missing_anthropic_api_key")
    timeout = float(os.environ.get("DEEP_DIVE_SELECTION_ANTHROPIC_TIMEOUT_SECONDS", "300"))
    return Anthropic(api_key=api_key, timeout=max(1.0, timeout))


def _build_selector_client(provider: str) -> Any:
    if provider == "anthropic":
        return _build_anthropic_client()
    if provider == "openai":
        return _build_openai_client()
    raise DeepDiveLaneError(f"unsupported_selection_provider:{provider}")


def _call_openai_selector(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None, float | None]:
    response = openai_chat_completions_create(
        client,
        caller=__file__,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON only. Select bounded, non-duplicative deep dives.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        **_openai_options(model, max_output_tokens),
    )
    text = str(response.choices[0].message.content or "")
    return parse_json_object(text), response.receipt.receipt_path, response.receipt.estimated_cost_usd


def _call_anthropic_selector(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None, float | None]:
    response = anthropic_messages_create(
        client,
        caller=__file__,
        model=model,
        max_tokens=max_output_tokens,
        system="Return strict JSON only. Select bounded, non-duplicative deep dives.",
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        str(block.get("text") or "")
        if isinstance(block, dict)
        else str(getattr(block, "text", "") or "")
        for block in (getattr(response, "content", None) or [])
    )
    return parse_json_object(text), response.receipt.receipt_path, response.receipt.estimated_cost_usd


def _call_selector(
    *,
    provider: str,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None, float | None]:
    if provider == "anthropic":
        return _call_anthropic_selector(
            client=client,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return _call_openai_selector(
            client=client,
            model=model,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
        )
    raise DeepDiveLaneError(f"unsupported_selection_provider:{provider}")


def _question_from_raw(
    *,
    pair: SummaryPair,
    raw: Mapping[str, Any],
    index: int,
) -> DeepDiveQuestion | None:
    question = str(raw.get("question") or "").strip()
    if not question:
        return None
    keywords = tuple(
        sorted(
            {
                str(item).strip()
                for item in _as_list(raw.get("keywords"))
                if str(item).strip()
            }
        )
    )
    if not keywords:
        keywords = tuple(_slugify(question, limit=80).replace("_", " ").split()[:10])
    score_raw = raw.get("priority_score")
    try:
        priority_score = float(score_raw)
    except (TypeError, ValueError):
        priority_score = 0.0
    return DeepDiveQuestion(
        deep_dive_id=_deep_dive_id_for(pair.topic_id, question, index=index),
        topic_id=pair.topic_id,
        topic_title=pair.title,
        question=question,
        rationale=str(raw.get("rationale") or "").strip(),
        keywords=keywords,
        source_categories=tuple(
            str(item).strip()
            for item in _as_list(raw.get("source_categories"))
            if str(item).strip()
        )
        or ("transcripts", "papers"),
        priority_score=priority_score,
        slide_usefulness=str(raw.get("slide_usefulness") or "").strip(),
        evidence_handles=tuple(
            str(item).strip()
            for item in _as_list(raw.get("evidence_handles"))
            if str(item).strip()
        ),
    )


def _dry_run_questions(
    pair: SummaryPair,
    *,
    max_questions: int,
) -> list[DeepDiveQuestion]:
    raw_candidates: list[dict[str, Any]] = []
    for category in ("transcripts", "papers"):
        summary = pair.summaries.get(category) or {}
        for value in _as_list(summary.get("tensions_or_contrasts")):
            raw_candidates.append(
                {
                    "question": f"What does the evidence imply about this tension: {value}",
                    "rationale": "Deterministic dry-run question from tensions_or_contrasts.",
                    "keywords": str(value).split()[:12],
                    "source_categories": [category],
                    "priority_score": 0.7,
                }
            )
        for value in _as_list(summary.get("key_points")):
            raw_candidates.append(
                {
                    "question": f"How important is this pattern for the topic: {value}",
                    "rationale": "Deterministic dry-run question from key_points.",
                    "keywords": str(value).split()[:12],
                    "source_categories": [category],
                    "priority_score": 0.5,
                }
            )
    questions: list[DeepDiveQuestion] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        key = _slugify(str(raw.get("question") or ""))
        if key in seen:
            continue
        seen.add(key)
        question = _question_from_raw(pair=pair, raw=raw, index=len(questions) + 1)
        if question is not None:
            questions.append(question)
        if len(questions) >= max_questions:
            break
    return questions


def run_deep_dive_question_selection(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    provider: str = DEFAULT_SELECTION_PROVIDER,
    model: str = DEFAULT_SELECTION_MODEL,
    min_questions_per_topic: int = DEFAULT_MIN_QUESTIONS_PER_TOPIC,
    max_questions_per_topic: int = DEFAULT_MAX_QUESTIONS_PER_TOPIC,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    provider_cost_approved: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    if max_questions_per_topic < 1:
        raise ValueError("max_questions_per_topic must be positive")
    if min_questions_per_topic < 1 or min_questions_per_topic > max_questions_per_topic:
        raise ValueError("min_questions_per_topic must be between 1 and max_questions_per_topic")
    validate_selection(OWNER_SELECTION, provider=provider, model=model)
    generated_at = utc_now()
    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "deep-dive-question-selection")
    conference_label = conference_name(workspace, default="the conference")
    receipt_path = _selection_receipt_path(workspace, run_id)
    selection_path = _selection_output_path(workspace, run_id)

    if mode == "live" and not provider_cost_approved:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": RECEIPT_TYPE_SELECTION,
            "owner_id": OWNER_SELECTION,
            "status": "blocked",
            "blocker_code": "provider_cost_approval_required",
            "generated_at": generated_at,
            "workspace": str(workspace),
            "run_id": run_id,
            "mode": mode,
            "provider": provider,
            "model": model,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            "paid_provider_calls_made": False,
            "secrets_copied": False,
            "receipt_path": str(receipt_path),
        }
        write_json_atomic(receipt_path, receipt)
        return receipt

    pairs = load_summary_pairs(workspace, run_id=run_id)
    selector_client = client if mode == "live" else None
    if mode == "live" and selector_client is None:
        selector_client = _build_selector_client(provider)

    provider_receipts: list[str] = []
    estimated_cost = 0.0
    topics_payload: list[dict[str, Any]] = []
    flat_questions: list[dict[str, Any]] = []
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=run_id,
        stage_id=OWNER_SELECTION,
        caller=__file__,
    ):
        for pair in pairs:
            if mode == "live":
                prompt = build_selection_prompt(
                    pair,
                    min_questions=min_questions_per_topic,
                    max_questions=max_questions_per_topic,
                    conference_label=conference_label,
                )
                payload, receipt, cost = _call_selector(
                    provider=provider,
                    client=selector_client,
                    model=model,
                    prompt=prompt,
                    max_output_tokens=max_output_tokens,
                )
                if receipt:
                    provider_receipts.append(str(receipt))
                if isinstance(cost, (int, float)):
                    estimated_cost += float(cost)
                raw_questions = _as_list(payload.get("deep_dives"))
            else:
                raw_questions = [q.as_payload() for q in _dry_run_questions(pair, max_questions=max_questions_per_topic)]

            questions: list[DeepDiveQuestion] = []
            seen_questions: set[str] = set()
            for raw in raw_questions:
                if not isinstance(raw, Mapping):
                    continue
                normalized_question = _slugify(str(raw.get("question") or ""))
                if not normalized_question or normalized_question in seen_questions:
                    continue
                seen_questions.add(normalized_question)
                question = _question_from_raw(
                    pair=pair,
                    raw=raw,
                    index=len(questions) + 1,
                )
                if question is not None:
                    questions.append(question)
                if len(questions) >= max_questions_per_topic:
                    break
            topics_payload.append(
                {
                    "topic_id": pair.topic_id,
                    "title": pair.title,
                    "summary_paths": dict(pair.summary_paths),
                    "selected_count": len(questions),
                    "selected_questions": [question.as_payload() for question in questions],
                }
            )
            flat_questions.extend(question.as_payload() for question in questions)

    selection = {
        "schema_version": SCHEMA_VERSION,
        "owner_id": OWNER_SELECTION,
        "run_id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "generated_at": generated_at,
        "min_questions_per_topic": min_questions_per_topic,
        "max_questions_per_topic": max_questions_per_topic,
        "topic_count": len(topics_payload),
        "question_count": len(flat_questions),
        "topics": topics_payload,
        "questions": flat_questions,
    }
    write_json_atomic(selection_path, sanitize_for_receipt(selection))

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_SELECTION,
        "owner_id": OWNER_SELECTION,
        "status": "succeeded" if mode == "live" else "planned",
        "generated_at": generated_at,
        "ended_at": utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "provider_cost_approved": provider_cost_approved,
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": mode == "live",
        "secrets_copied": False,
        "topic_count": len(topics_payload),
        "question_count": len(flat_questions),
        "min_questions_per_topic": min_questions_per_topic,
        "max_questions_per_topic": max_questions_per_topic,
        "selection_path": relpath(selection_path, workspace),
        "selection_sha256": sha256_text(json.dumps(selection, sort_keys=True, ensure_ascii=True)),
        "estimated_cost_usd": estimated_cost if mode == "live" else None,
        "provider_receipt_paths": provider_receipts,
        "elapsed_seconds": time.monotonic() - started,
        "receipt_path": str(receipt_path),
    }
    stamp_substep_receipt(receipt, workspace=workspace, run_id=run_id, substep_id="deep-dive-question-selection")
    write_json_atomic(receipt_path, sanitize_receipt_for_write(receipt))
    return receipt


def _load_selection(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("owner_id") != OWNER_SELECTION:
        raise DeepDiveLaneError("selection_owner_mismatch")
    return payload


def _chunk_payload(chunk: RetrievedChunk, *, store: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "store": store,
        "source_artifact_id": chunk.source_artifact_id,
        "source_artifact_path": chunk.source_artifact_path,
        "score": chunk.score,
        "topic_ids": list(chunk.topic_ids),
        "chunk_strategy": chunk.chunk_strategy,
        "text": chunk.text,
    }


def _retrieve_global_evidence(
    *,
    question: str,
    retrievers: Mapping[str, RetrieverLike],
    k_per_store: int,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for store, retriever in retrievers.items():
        chunks = retriever.retrieve(topic_id="", query=question, k=k_per_store)
        for chunk in chunks:
            key = f"{store}:{chunk.chunk_id}"
            if key in seen:
                continue
            seen.add(key)
            evidence.append(_chunk_payload(chunk, store=store))
    return evidence


def _build_default_retrievers(
    workspace: Path,
    *,
    run_id: str,
    stores: Sequence[str],
) -> dict[str, RetrieverLike]:
    return {
        store: HybridChromaBM25Retriever(
            workspace,
            run_id=run_id,
            store_kind=store,
        )
        for store in stores
    }


def _evidence_snippets(evidence: Iterable[Mapping[str, Any]], *, limit_chars: int = 900) -> str:
    parts: list[str] = []
    for index, item in enumerate(evidence, start=1):
        text = " ".join(str(item.get("text") or "").split())
        if len(text) > limit_chars:
            text = text[:limit_chars].rstrip() + "..."
        parts.append(
            "[EVIDENCE #{index}] store={store} chunk_id={chunk_id} source={source}\n{text}".format(
                index=index,
                store=item.get("store") or "",
                chunk_id=item.get("chunk_id") or "",
                source=item.get("source_artifact_path") or item.get("source_path") or "",
                text=text,
            )
        )
    return "\n\n".join(parts) or "(no retrieved evidence)"


def build_research_prompt(
    question: Mapping[str, Any],
    *,
    lane: str,
    evidence: Iterable[Mapping[str, Any]],
    conference_label: str = "the conference",
) -> str:
    return (
        f"You are running a corpus-grounded GPT-Researcher deep dive for a {conference_label} "
        "conference analysis. Write a concise, slide-usable research report. "
        "Use only the supplied local evidence. Preserve titles, talks, papers, "
        "workshops, timestamps, and source paths when present. Cite every claim "
        "with the number of the evidence passage it comes from, written as "
        "[EVIDENCE #n]. Name uncertainty rather than inventing support.\n\n"
        f"Lane: {lane}\n"
        f"Topic: {question.get('topic_title')} ({question.get('topic_id')})\n"
        f"Deep dive id: {question.get('deep_dive_id')}\n"
        f"Question: {question.get('question')}\n"
        f"Rationale: {question.get('rationale')}\n"
        f"Keywords: {', '.join(str(item) for item in question.get('keywords') or [])}\n\n"
        "Return Markdown with these sections:\n"
        "## Answer\n"
        "## Evidence\n"
        "## Tensions And Caveats\n"
        "## Slide Implications\n\n"
        "LOCAL EVIDENCE:\n"
        f"{_evidence_snippets(evidence)}"
    )


# "[EVIDENCE #3]", "Evidence #9, #10", "[Evidence #1 and #4]" -- the in-house
# prompt numbers passages; the numbers mean nothing outside this one report.
_EVIDENCE_NUMBER_RE = re.compile(
    r"(?:\[\s*)?evidence\s*#\s*(\d+)((?:\s*(?:,|;|and|&)\s*(?:evidence\s*)?#?\s*\d+)*)\s*\]?",
    re.IGNORECASE,
)


def resolve_evidence_numbers(report: str, evidence: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Rewrite the in-house engine's "Evidence #n" citations to passage links.

    ``build_research_prompt`` numbers the passages in the order of ``evidence``,
    the same list the lane saves, so #n is exactly ``evidence[n - 1]``. The
    rewritten report cites ``chunk://<store>/<chunk_id>`` like GPT-Researcher's
    reports, so citations survive the topic merge, which sees only the text.
    A number outside the list is marked, never guessed.
    """

    stats: dict[str, Any] = {"resolved": 0, "unresolved": []}

    def rewrite(match: re.Match[str]) -> str:
        numbers = [int(match.group(1))] + [int(n) for n in re.findall(r"\d+", match.group(2) or "")]
        links: list[str] = []
        for number in numbers:
            item = evidence[number - 1] if 1 <= number <= len(evidence) else None
            chunk_id = str((item or {}).get("chunk_id") or "")
            if item is None or not chunk_id:
                stats["unresolved"].append(f"Evidence #{number}")
                links.append(f"[unresolved citation: Evidence #{number}]")
                continue
            stats["resolved"] += 1
            links.append(f"chunk://{item.get('store') or 'papers'}/{chunk_id}")
        return "(" + "; ".join(links) + ")"

    return _EVIDENCE_NUMBER_RE.sub(rewrite, report), stats


def build_lane_instructions(
    question: Mapping[str, Any],
    *,
    lane: str,
    conference_label: str = "the conference",
) -> str:
    """Report instructions for the GPT-Researcher engine (it gathers its own evidence).

    Same sections as ``build_research_prompt``; sources are cited by the
    ``chunk://<store>/<chunk_id>`` URL each retrieved passage carries, so a
    citation always resolves to an indexed passage.
    """

    return (
        f"Write a corpus-grounded deep-dive research report for a {conference_label} "
        "conference analysis, concise and slide-usable. Use only the research "
        "context below, which comes from the conference's own papers, talks and "
        "workshops. Preserve paper titles, talk titles, speakers and workshops as "
        "they appear. Cite every claim with the source URL of the passage it comes "
        "from (chunk://...). Name uncertainty rather than inventing support.\n\n"
        f"Lane: {lane}\n"
        f"Topic: {question.get('topic_title')} ({question.get('topic_id')})\n"
        f"Deep dive id: {question.get('deep_dive_id')}\n"
        f"Question: {question.get('question')}\n"
        f"Rationale: {question.get('rationale')}\n\n"
        "Return Markdown with these sections:\n"
        "## Answer\n## Evidence\n## Tensions And Caveats\n## Slide Implications\n"
    )


def _run_coroutine(result: Any) -> str:
    """Run a researcher coroutine to completion, letting its own errors surface.

    Only the "an event loop is already running" case gets a private loop; an
    exception raised *by* the research must propagate, never trigger a second
    await of the same coroutine (which masked the real error as "cannot reuse
    already awaited coroutine").
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return str(asyncio.run(result))
    loop = asyncio.new_event_loop()
    try:
        return str(loop.run_until_complete(result))
    finally:
        loop.close()


def _engine_research(engine: Any, question: Mapping[str, Any], *, instructions: str) -> str:
    result = engine.run(str(question.get("question") or ""), custom_prompt=instructions)
    if not asyncio.iscoroutine(result):
        return str(result)
    return _run_coroutine(result)


def _engine_evidence(engine: Any) -> list[dict[str, Any]]:
    """The passages GPT-Researcher's retriever returned, in lane evidence shape."""

    out: list[dict[str, Any]] = []
    for item in engine.evidence():
        out.append(
            {
                "store": item.get("store"),
                "chunk_id": item.get("chunk_id"),
                "source_artifact_id": item.get("source_artifact_id"),
                "source_artifact_path": item.get("source_artifact_path"),
                "score": item.get("score"),
                "url": item.get("url"),
                "text": item.get("text"),
            }
        )
    return out


def _await_researcher(researcher: Any, prompt: str, context_path: Path) -> str:
    result = researcher.run(prompt, context_path=context_path)
    if not asyncio.iscoroutine(result):
        return str(result)
    return _run_coroutine(result)


def _researcher_usage_delta(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, ...]:
    before_paths = set(str(item) for item in before.get("provider_receipt_paths") or [])
    return tuple(
        str(item)
        for item in after.get("provider_receipt_paths") or []
        if str(item) not in before_paths
    )


def _existing_lane_output(
    *,
    workspace: Path,
    deep_dive_id: str,
    lane: str,
    markdown_path: Path,
    evidence_path: Path,
    lane_input_sha256: str | None = None,
) -> LaneOutput | None:
    if not markdown_path.exists() or not evidence_path.exists():
        return None
    try:
        evidence_payload = _load_json(evidence_path)
    except DeepDiveLaneError:
        return None
    if lane_input_sha256 is not None and evidence_payload.get("lane_input_sha256") != lane_input_sha256:
        # The question, lane settings or corpus index moved since this lane ran.
        return None
    if lane == "heuristic":
        evidence_count = int(evidence_payload.get("candidate_count") or 0)
    else:
        evidence_count = int(
            evidence_payload.get("evidence_count")
            or len(_as_list(evidence_payload.get("evidence")))
        )
    return LaneOutput(
        deep_dive_id=deep_dive_id,
        lane=lane,
        status="succeeded",
        output_markdown=relpath(markdown_path, workspace),
        evidence_json=relpath(evidence_path, workspace),
        evidence_count=evidence_count,
        provider_receipt_paths=(),
    )


def _obsolete_lane_outputs(workspace: Path, run_id: str, current_ids: set[str]) -> list[Path]:
    """Lane reports and evidence directories of deep dives not in the selection."""

    obsolete: list[Path] = []
    reports = workspace / "analysis" / "deep_dive_topics"
    if reports.is_dir():
        for path in sorted(reports.glob("*__*.md")):
            deep_dive_id, _, lane = path.stem.rpartition("__")
            if lane in VALID_LANES and deep_dive_id not in current_ids:
                obsolete.append(path)
    evidence_root = workspace / "storage" / "analysis" / "deep-dive-topics" / run_id
    if evidence_root.is_dir():
        obsolete.extend(
            path for path in sorted(evidence_root.iterdir()) if path.is_dir() and path.name not in current_ids
        )
    return obsolete


def lane_input_sha256(
    *,
    question: Mapping[str, Any],
    lane: str,
    corpus_index_output_digest: str | None,
    settings: Mapping[str, Any],
) -> str:
    """Digest of what one research lane is computed from (its resume key)."""

    encoded = json.dumps(
        {
            "question": dict(question),
            "lane": lane,
            "corpus_index_output_digest": corpus_index_output_digest,
            "settings": dict(settings),
        },
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_dual_deep_dive_research_lanes(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    selection_path: Path | None = None,
    lanes: Sequence[str] = VALID_LANES,
    stores: Sequence[str] = DEFAULT_STORES,
    max_questions: int | None = None,
    retrieval_k_per_store: int = DEFAULT_RETRIEVAL_K_PER_STORE,
    prefilter_limit: int = DEFAULT_PREFILTER_LIMIT,
    researcher: Any | None = None,
    retrievers: Mapping[str, RetrieverLike] | None = None,
    local_model: str = DEFAULT_QWEN36_MODEL,
    local_base_url: str | None = None,
    research_max_tokens: int = DEFAULT_RESEARCH_MAX_TOKENS,
    resume: bool = False,
    engine: str | None = DEFAULT_ENGINE,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    # Refuses (asking the operator) when neither --engine nor the conference
    # config chose one; never falls back to a default engine.
    engine_choice = resolve_research_engine(workspace, engine)
    engine = engine_choice.engine
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    selected_lanes = tuple(lane for lane in lanes if lane in VALID_LANES)
    if not selected_lanes:
        raise ValueError("at least one valid lane is required")
    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "dual-deep-dive-research")
    generated_at = utc_now()
    conference_label = conference_name(workspace, default="the conference")
    canonical_selection_path = _selection_output_path(workspace, run_id)
    selection_path = selection_path or canonical_selection_path
    selection = _load_selection(selection_path)
    questions = [q for q in selection.get("questions") or [] if isinstance(q, dict)]
    selected_question_count = len(questions)
    if max_questions is not None:
        questions = questions[: max(0, int(max_questions))]

    receipt_path = _dual_lanes_receipt_path(workspace, run_id)
    lane_outputs: list[LaneOutput] = []
    engine_settings: dict[str, Any] = {}
    if engine == ENGINE_GPT_RESEARCHER:
        from conference_pipeline.research.gptr_engine import GPTResearcherEngine, default_settings

        gptr_settings = default_settings(
            stores=tuple(stores),
            max_search_results_per_query=retrieval_k_per_store,
            base_url=local_base_url or None,
            smart_token_limit=research_max_tokens,
        )
        engine_settings = gptr_settings.to_payload()
    if mode == "live":
        if engine == ENGINE_GPT_RESEARCHER:
            active_researcher = researcher or GPTResearcherEngine(
                workspace,
                run_id=run_id,
                settings=gptr_settings,
                retriever_factory=(lambda store: retrievers[store]) if retrievers else None,
            )
        else:
            active_researcher = researcher or build_researcher(
                {
                    "provider": "local",
                    "model": local_model,
                    "base_url": local_base_url or os.environ.get("GPTR_LOCAL_LLM_BASE_URL"),
                    "endpoint_label": f"mlx://{local_model}",
                    "max_tokens": research_max_tokens,
                }
            )
        active_retrievers = dict(retrievers or _build_default_retrievers(workspace, run_id=run_id, stores=stores))
    else:
        active_researcher = researcher
        active_retrievers = dict(retrievers or {})
    egress_reports: list[dict[str, Any]] = []
    citation_reports: list[dict[str, Any]] = []

    corpus_index_digest = current_stage_output_digest(workspace, run_id, "corpus-inventory-index")
    for raw_question in questions:
        question = dict(raw_question)
        deep_dive_id = str(question.get("deep_dive_id") or "")
        if not deep_dive_id:
            continue
        global_input_sha256 = lane_input_sha256(
            question=question,
            lane="global",
            corpus_index_output_digest=corpus_index_digest,
            settings={
                "stores": list(stores),
                "k_per_store": retrieval_k_per_store,
                "model": local_model,
                "engine": engine,
                "engine_settings": engine_settings,
            },
        )
        heuristic_input_sha256 = lane_input_sha256(
            question=question,
            lane="heuristic",
            corpus_index_output_digest=corpus_index_digest,
            settings={
                "stores": list(stores),
                "prefilter_limit": prefilter_limit,
                "model": local_model,
                "engine": engine,
                "engine_settings": engine_settings,
            },
        )
        output_root = _deep_dive_root(workspace, run_id, deep_dive_id)
        output_root.mkdir(parents=True, exist_ok=True)
        keywords = [str(item) for item in question.get("keywords") or [] if str(item).strip()]
        if "global" in selected_lanes:
            evidence_path = output_root / "global_evidence.json"
            markdown_path = workspace / "analysis" / "deep_dive_topics" / f"{deep_dive_id}__global.md"
            existing = None
            if resume:
                existing = _existing_lane_output(
                    workspace=workspace,
                    deep_dive_id=deep_dive_id,
                    lane="global",
                    markdown_path=markdown_path,
                    evidence_path=evidence_path,
                    lane_input_sha256=global_input_sha256,
                )
            if existing is not None:
                lane_outputs.append(existing)
            elif mode == "live" and engine == ENGINE_GPT_RESEARCHER:
                before = active_researcher.usage()
                body = _engine_research(
                    active_researcher,
                    question,
                    instructions=build_lane_instructions(
                        question, lane="global_no_topic_id", conference_label=conference_label
                    ),
                )
                evidence = _engine_evidence(active_researcher)
                egress_reports.append(active_researcher.last_run.egress)
                citation_reports.append(active_researcher.last_run.citations)
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(body.rstrip() + "\n", encoding="utf-8")
                write_json_atomic(
                    evidence_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "owner_id": OWNER_DUAL_LANES,
                        "retrieval_scope": "global_no_topic_id",
                        "retrieval_engine": ENGINE_GPT_RESEARCHER,
                        "deep_dive_id": deep_dive_id,
                        "question": question.get("question"),
                        "stores_searched": list(stores),
                        "evidence_count": len(evidence),
                        "evidence": sanitize_for_receipt(evidence),
                        "egress": active_researcher.last_run.egress,
                        "citations": active_researcher.last_run.citations,
                        "lane_input_sha256": global_input_sha256,
                    },
                )
                after = active_researcher.usage()
                lane_outputs.append(
                    LaneOutput(
                        deep_dive_id=deep_dive_id,
                        lane="global",
                        status="succeeded",
                        output_markdown=relpath(markdown_path, workspace),
                        evidence_json=relpath(evidence_path, workspace),
                        evidence_count=len(evidence),
                        provider_receipt_paths=_researcher_usage_delta(before, after),
                    )
                )
            elif mode == "live":
                evidence = _retrieve_global_evidence(
                    question=str(question.get("question") or ""),
                    retrievers=active_retrievers,
                    k_per_store=retrieval_k_per_store,
                )
                # Written without its resume key: an interrupted lane leaves an
                # evidence file that no resume will accept.
                global_evidence_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "owner_id": OWNER_DUAL_LANES,
                    "retrieval_scope": "global_no_topic_id",
                    "deep_dive_id": deep_dive_id,
                    "question": question.get("question"),
                    "stores_searched": list(active_retrievers),
                    "evidence_count": len(evidence),
                    "evidence": sanitize_for_receipt(evidence),
                }
                write_json_atomic(evidence_path, global_evidence_payload)
                before = active_researcher.usage() if hasattr(active_researcher, "usage") else {}
                prompt = build_research_prompt(
                    question,
                    lane="global_no_topic_id",
                    evidence=evidence,
                    conference_label=conference_label,
                )
                body = _await_researcher(active_researcher, prompt, output_root)
                body, citations = resolve_evidence_numbers(body, evidence)
                citation_reports.append(citations)
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(body.rstrip() + "\n", encoding="utf-8")
                # Publish the resume key only now that the report matches it.
                write_json_atomic(
                    evidence_path,
                    {**global_evidence_payload, "citations": citations, "lane_input_sha256": global_input_sha256},
                )
                after = active_researcher.usage() if hasattr(active_researcher, "usage") else {}
                lane_outputs.append(
                    LaneOutput(
                        deep_dive_id=deep_dive_id,
                        lane="global",
                        status="succeeded",
                        output_markdown=relpath(markdown_path, workspace),
                        evidence_json=relpath(evidence_path, workspace),
                        evidence_count=len(evidence),
                        provider_receipt_paths=_researcher_usage_delta(before, after),
                    )
                )
            else:
                lane_outputs.append(
                    LaneOutput(
                        deep_dive_id=deep_dive_id,
                        lane="global",
                        status="planned",
                        output_markdown=relpath(markdown_path, workspace),
                        evidence_json=relpath(evidence_path, workspace),
                        evidence_count=0,
                    )
                )

        if "heuristic" in selected_lanes:
            pack_path = output_root / "heuristic_prefilter_evidence.json"
            markdown_path = workspace / "analysis" / "deep_dive_topics" / f"{deep_dive_id}__heuristic.md"
            existing = None
            if resume:
                existing = _existing_lane_output(
                    workspace=workspace,
                    deep_dive_id=deep_dive_id,
                    lane="heuristic",
                    markdown_path=markdown_path,
                    evidence_path=pack_path,
                    lane_input_sha256=heuristic_input_sha256,
                )
            if existing is not None:
                lane_outputs.append(existing)
            elif mode == "live":
                pack = build_prefiltered_evidence_pack(
                    workspace=workspace,
                    run_id=run_id,
                    deep_dive_id=deep_dive_id,
                    question=str(question.get("question") or ""),
                    keywords=keywords,
                    keyword_seed_source=relpath(selection_path, workspace),
                    stores=stores,
                    limit=prefilter_limit,
                )
                write_prefiltered_evidence_pack(pack, pack_path)
                evidence = [
                    {
                        **item,
                        "store": item.get("store", ""),
                    }
                    for item in pack.get("evidence") or []
                    if isinstance(item, dict)
                ]
                before = active_researcher.usage() if hasattr(active_researcher, "usage") else {}
                if engine == ENGINE_GPT_RESEARCHER:
                    # GPT-Researcher researches within the prefiltered pack only.
                    pack_engine = active_researcher.with_pack(evidence)
                    body = _engine_research(
                        pack_engine,
                        question,
                        instructions=build_lane_instructions(
                            question, lane="heuristic_keyword_prefilter", conference_label=conference_label
                        ),
                    )
                    egress_reports.append(pack_engine.last_run.egress)
                    citation_reports.append(pack_engine.last_run.citations)
                    pack = {
                        **pack,
                        "retrieval_engine": ENGINE_GPT_RESEARCHER,
                        "egress": pack_engine.last_run.egress,
                        "citations": pack_engine.last_run.citations,
                    }
                else:
                    prompt = build_research_prompt(
                        question,
                        lane="heuristic_keyword_prefilter",
                        evidence=evidence,
                        conference_label=conference_label,
                    )
                    body = _await_researcher(active_researcher, prompt, output_root)
                    body, citations = resolve_evidence_numbers(body, evidence)
                    citation_reports.append(citations)
                    pack = {**pack, "citations": citations}
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(body.rstrip() + "\n", encoding="utf-8")
                write_prefiltered_evidence_pack({**pack, "lane_input_sha256": heuristic_input_sha256}, pack_path)
                after = active_researcher.usage() if hasattr(active_researcher, "usage") else {}
                lane_outputs.append(
                    LaneOutput(
                        deep_dive_id=deep_dive_id,
                        lane="heuristic",
                        status="succeeded",
                        output_markdown=relpath(markdown_path, workspace),
                        evidence_json=relpath(pack_path, workspace),
                        evidence_count=int(pack.get("candidate_count") or 0),
                        provider_receipt_paths=_researcher_usage_delta(before, after),
                    )
                )
            else:
                lane_outputs.append(
                    LaneOutput(
                        deep_dive_id=deep_dive_id,
                        lane="heuristic",
                        status="planned",
                        output_markdown=relpath(markdown_path, workspace),
                        evidence_json=relpath(pack_path, workspace),
                        evidence_count=0,
                    )
                )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_DUAL_LANES,
        "owner_id": OWNER_DUAL_LANES,
        "status": "succeeded" if mode == "live" else "planned",
        "generated_at": generated_at,
        "ended_at": utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "workflow": "workflow.research.dual_deep_dive_gpt_researcher_lanes",
        "selection_path": relpath(selection_path, workspace),
        "selection_sha256": sha256_text(selection_path.read_text(encoding="utf-8")),
        "question_count": len(questions),
        "lane_count": len(lane_outputs),
        "lanes": list(selected_lanes),
        "stores": list(stores),
        "local_researcher_model": local_model,
        "research_engine": engine,
        "research_engine_choice": engine_choice.to_payload(),
        "research_engine_settings": engine_settings,
        "egress": {
            "runs": len(egress_reports),
            "blocked_destinations": sorted(
                {item for report in egress_reports for item in (report or {}).get("blocked_destinations") or []}
            ),
        },
        "citations": {
            "resolved": sum(int(report.get("resolved") or 0) for report in citation_reports),
            "unresolved": sorted(
                {url for report in citation_reports for url in report.get("unresolved") or []}
            ),
        },
        "retrieval_k_per_store": retrieval_k_per_store,
        "prefilter_limit": prefilter_limit,
        "resume": resume,
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": False,
        "secrets_copied": False,
        "outputs": [item.as_payload() for item in lane_outputs],
        "elapsed_seconds": time.monotonic() - started,
        "receipt_path": str(receipt_path),
    }
    # Whole scope: every lane over every question of the canonical 07B
    # selection this receipt's lineage binds (not an alternate file).
    complete = (
        set(selected_lanes) == set(VALID_LANES)
        and len(questions) == selected_question_count
        and Path(selection_path).resolve() == canonical_selection_path.resolve()
    )
    if complete and mode == "live":
        # Reports for questions 07B no longer selects would still be read by
        # 07D's glob and certified by this stamp: move them out first.
        receipt["pruned_outputs"] = archive_obsolete_outputs(
            workspace,
            _obsolete_lane_outputs(workspace, run_id, {str(q.get("deep_dive_id") or "") for q in questions}),
            label="dual-deep-dive-research",
        )
    stamp_substep_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        substep_id="dual-deep-dive-research",
        complete=complete,
    )
    write_json_atomic(receipt_path, sanitize_receipt_for_write(receipt))
    return receipt


def build_select_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select bounded deep-dive questions.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--provider", choices=("anthropic", "openai"), default=DEFAULT_SELECTION_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_SELECTION_MODEL)
    parser.add_argument("--min-questions-per-topic", type=int, default=DEFAULT_MIN_QUESTIONS_PER_TOPIC)
    parser.add_argument("--max-questions-per-topic", type=int, default=DEFAULT_MAX_QUESTIONS_PER_TOPIC)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def select_main(argv: Sequence[str] | None = None) -> int:
    args = build_select_arg_parser().parse_args(argv)
    receipt = run_deep_dive_question_selection(
        workspace=args.workspace,
        run_id=args.run_id,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        min_questions_per_topic=args.min_questions_per_topic,
        max_questions_per_topic=args.max_questions_per_topic,
        max_output_tokens=args.max_output_tokens,
        provider_cost_approved=args.provider_cost_approved,
    )
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "question_count": receipt.get("question_count"),
                    "topic_count": receipt.get("topic_count"),
                    "selection_path": receipt.get("selection_path"),
                    "receipt_path": receipt.get("receipt_path"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"succeeded", "planned"} else 1


def build_run_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dual deep-dive GPT-Researcher lanes.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--lane", action="append", choices=VALID_LANES, default=[])
    parser.add_argument("--store", action="append", default=[])
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--retrieval-k-per-store", type=int, default=DEFAULT_RETRIEVAL_K_PER_STORE)
    parser.add_argument("--prefilter-limit", type=int, default=DEFAULT_PREFILTER_LIMIT)
    parser.add_argument("--local-model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--local-base-url")
    parser.add_argument("--research-max-tokens", type=int, default=DEFAULT_RESEARCH_MAX_TOKENS)
    parser.add_argument(
        "--engine",
        choices=VALID_ENGINES,
        default=None,
        help=(
            "gpt_researcher: the GPT-Researcher package on our hybrid index and local model; "
            "inhouse: one retrieval and one local-model synthesis per lane. No default: when omitted, "
            "workflow.research_engine in the conference config.yaml is used, and without it the "
            "stage stops and asks the operator."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: Sequence[str] | None = None) -> int:
    args = build_run_arg_parser().parse_args(argv)
    try:
        resolve_research_engine(args.workspace.resolve(), args.engine)
    except ResearchEngineNotChosenError as exc:
        print(f"research engine not chosen: {exc}", file=sys.stderr)
        return 2
    receipt = run_dual_deep_dive_research_lanes(
        workspace=args.workspace,
        run_id=args.run_id,
        mode=args.mode,
        selection_path=args.selection_path,
        lanes=tuple(args.lane) if args.lane else VALID_LANES,
        stores=tuple(args.store) if args.store else DEFAULT_STORES,
        max_questions=args.max_questions,
        retrieval_k_per_store=args.retrieval_k_per_store,
        prefilter_limit=args.prefilter_limit,
        local_model=args.local_model,
        local_base_url=args.local_base_url,
        research_max_tokens=args.research_max_tokens,
        resume=args.resume,
        engine=args.engine,
    )
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "question_count": receipt.get("question_count"),
                    "lane_count": receipt.get("lane_count"),
                    "local_researcher_model": receipt.get("local_researcher_model"),
                    "receipt_path": receipt.get("receipt_path"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"succeeded", "planned"} else 1


__all__ = [
    "DEFAULT_QWEN36_MODEL",
    "DeepDiveLaneError",
    "DeepDiveQuestion",
    "LaneOutput",
    "SummaryPair",
    "build_research_prompt",
    "build_selection_prompt",
    "load_summary_pairs",
    "run_deep_dive_question_selection",
    "run_dual_deep_dive_research_lanes",
    "run_main",
    "select_main",
]
