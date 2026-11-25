"""Query-plan generator for Phase 14C.

Heuristic-first query plan generation from taxonomy + by-topic seeds +
entity_index. Optional LLM-assist refinement is env-gated
(``QUERY_PLANNER_LLM_ASSIST=1``) so the default path is provider-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from conference_pipeline.analysis.steps.gpt_researcher_hybrid import (
    REFERENCE_QUALITY_PROMPT,
)
from conference_pipeline.iclr2025_live import sanitize_for_receipt, utc_now
from conference_pipeline.model_governance.registry import default_model_for_provider
from conference_pipeline.research.config import GPTResearcherParameters
from conference_pipeline.research.records import (
    QueryPlan,
    compute_plan_digest,
)


class QueryPlannerError(RuntimeError):
    """Boundary error for the query-plan generation path; messages are pre-sanitized."""


class QueryExpansionError(QueryPlannerError):
    """Typed, topic-tagged hard-block error for enabled query expansion (Phase 11 D-06).

    Raised by :func:`build_plan` when ``query_expansion_enabled`` is True, an
    ``expansion_llm_client`` is present, and a per-topic ``expand_queries`` call
    fails. Unlike the Phase 04 deterministic-fallback behavior (D-10), an enabled
    expansion failure is a hard operator promise and must propagate so the stage
    can emit a sanitized, topic-specific blocker (D-07).

    The message is pre-sanitized by the caller. ``topic_id`` survives both the
    planner's outer ``except Exception`` wrapper (via a dedicated
    ``except QueryExpansionError: raise``) and the walker's ThreadPoolExecutor
    ``.result()`` re-raise, because it is carried as an attribute.
    """

    def __init__(
        self, *, topic_id: str, message: str, retryable: bool = True
    ) -> None:
        super().__init__(message)
        self.topic_id = topic_id
        self.retryable = retryable


class QueryPlanLLMClient(Protocol):
    """Protocol for the optional LLM-assist boundary."""

    def generate(self, prompt: str) -> dict[str, Any]: ...


@dataclass
class FakeQueryPlanLLMClient:
    """Deterministic test boundary — keys responses on prompt hash, never calls OpenAI."""

    seed: int = 0

    def generate(self, prompt: str) -> dict[str, Any]:
        # Deterministic mock: rotate question wording based on prompt length parity.
        flavor = "in-context" if len(prompt) % 2 == 0 else "out-of-context"
        return {
            "research_questions": [
                f"What {flavor} evidence does the corpus show about this topic?",
            ],
            "synonym_expansion": ["model", "method"],
            "entity_seeds": [],
        }


class QueryExpansionLLMClient(Protocol):
    """Protocol for the local-LLM query expansion boundary (D-09)."""

    def expand_queries(
        self, topic_title: str, base_questions: list[str]
    ) -> list[str]: ...


@dataclass
class FakeQueryExpansionLLMClient:
    """Deterministic test double for local-LLM query expansion. Never calls a provider."""

    seed: int = 0

    def expand_queries(
        self, topic_title: str, base_questions: list[str]
    ) -> list[str]:
        return [
            f"Alternate framing: {q.rstrip('?')} in a different context?"
            for q in base_questions[:2]
        ]


def _topic_keywords(topic) -> tuple[str, ...]:
    """Derive a reasonable synonym-seed set from a TopicFamily title.

    The TOPIC_KEYWORDS map in ``iclr2025_live`` is the canonical fallback;
    we re-use it lazily to avoid a hard import dependency at module load.
    """
    try:
        # Legacy fallback: TOPIC_KEYWORDS lives in the conference-specific
        # ``iclr2025_live`` module and is intentionally kept as a best-effort
        # synonym seed source. A separate effort owns ``iclr2025_live``; do not
        # rip this out here. When absent (any other conference) we return ().
        from conference_pipeline.iclr2025_live import TOPIC_KEYWORDS

        return TOPIC_KEYWORDS.get(getattr(topic, "id", ""), ())
    except Exception:  # noqa: BLE001 - fallback path
        return ()


def _evidence_terms(by_topic_seeds: dict[str, Any], limit: int = 10) -> list[str]:
    """Best-effort extraction of frequent terms from a by-topic seed payload."""
    if not isinstance(by_topic_seeds, dict):
        return []
    terms_field = by_topic_seeds.get("evidence_terms") or []
    if isinstance(terms_field, dict):
        # ``{term: count}`` style mapping — pick top-N by count.
        terms = sorted(terms_field.items(), key=lambda kv: -int(kv[1] or 0))
        return [str(t) for t, _ in terms[:limit]]
    if isinstance(terms_field, list):
        out: list[str] = []
        seen: set[str] = set()
        for item in terms_field:
            if isinstance(item, str) and item.strip() and item not in seen:
                out.append(item.strip())
                seen.add(item)
            if len(out) >= limit:
                break
        return out
    # Fallback: try `entries[*].evidence_terms` shape.
    entries = by_topic_seeds.get("entries") or []
    if isinstance(entries, list):
        out_set: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            evid = entry.get("evidence") or {}
            terms = (evid or {}).get("terms") if isinstance(evid, dict) else None
            if isinstance(terms, list):
                for term in terms:
                    if isinstance(term, str) and term.strip():
                        out_set[term.strip()] = out_set.get(term.strip(), 0) + 1
        return [
            term for term, _ in sorted(out_set.items(), key=lambda kv: -kv[1])[:limit]
        ]
    return []


def _entity_seeds_for_topic(
    entity_index: dict[str, Any], topic_id: str, limit: int = 15
) -> list[str]:
    """Pull top entity names that mention ``topic_id`` in the entity_index."""
    if not isinstance(entity_index, dict):
        return []
    entities = entity_index.get("entities")
    seeds: list[tuple[str, int]] = []
    if isinstance(entities, dict):
        # ``{entity_type: {entity_name: {sources: [...], topic_ids: [...]}}}`` shape.
        for _etype, names in entities.items():
            if not isinstance(names, dict):
                continue
            for name, payload in names.items():
                if not isinstance(name, str) or not isinstance(payload, dict):
                    continue
                topics = payload.get("topic_ids") or payload.get("topics") or []
                if isinstance(topics, list) and topic_id in topics:
                    seeds.append((name, len(payload.get("sources") or [])))
    elif isinstance(entities, list):
        for item in entities:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if isinstance(name, str) and topic_id in (item.get("topic_ids") or []):
                seeds.append((name, int(item.get("count") or 0)))
    seeds.sort(key=lambda kv: -kv[1])
    out: list[str] = []
    seen: set[str] = set()
    for name, _ in seeds:
        if name not in seen:
            out.append(name)
            seen.add(name)
        if len(out) >= limit:
            break
    return out


def _heuristic_questions(
    topic, evidence_terms: list[str], *, conference_label: str = "the conference"
) -> list[str]:
    """3–5 research questions seeded from the topic title + top evidence terms."""
    title = getattr(topic, "title", "") or getattr(topic, "id", "")
    questions = [
        f"What are the most-discussed methods, results, and limitations in {title} at {conference_label}?",
        f"Which datasets and benchmarks anchor the {title} contributions in this corpus?",
        f"What concrete failure modes or open problems are reported across {title} sessions?",
    ]
    # Augment with top evidence terms when available.
    if evidence_terms:
        terms_phrase = ", ".join(evidence_terms[:5])
        questions.append(
            f"How do papers in {title} treat the recurring concepts: {terms_phrase}?"
        )
    if len(evidence_terms) >= 6:
        questions.append(
            "What cross-paper patterns emerge for "
            + ", ".join(evidence_terms[5:8])
            + "?"
        )
    return questions[:5]


def _heuristic_synonyms(
    topic, evidence_terms: list[str], questions: list[str]
) -> list[str]:
    """3–8 synonym-expansion strings from topic keywords + evidence terms."""
    base = list(_topic_keywords(topic)) + list(evidence_terms)
    out: list[str] = []
    seen: set[str] = set()
    for term in base:
        if not isinstance(term, str):
            continue
        norm = term.strip().lower()
        if not norm or norm in seen:
            continue
        out.append(term.strip())
        seen.add(norm)
        if len(out) >= 8:
            break
    # Pad to at least 3 entries with question-seeded terms when sparse.
    for question in questions:
        if len(out) >= 3:
            break
        seed = question.split()[0:3]
        candidate = " ".join(seed).strip(" ?")
        if candidate and candidate.lower() not in seen:
            out.append(candidate)
            seen.add(candidate.lower())
    return out


def _expand_research_questions(
    *,
    topic_title: str,
    base_questions: list[str],
    synonyms: list[str],
    entity_seeds: list[str],
    parameters: GPTResearcherParameters,
) -> list[str]:
    """Map GPT-Researcher breadth/depth knobs onto deterministic queries."""

    max_iterations = max(1, int(parameters.max_iterations))
    breadth = max(1, int(parameters.deep_research_breadth))
    depth = max(1, int(parameters.deep_research_depth))
    base = [q for q in base_questions if isinstance(q, str) and q.strip()]
    if not base:
        base = [f"What source-backed evidence exists for {topic_title}?"]
    selected = base[:max_iterations]

    seeds: list[str] = []
    seen: set[str] = set()
    for seed in [*synonyms, *entity_seeds]:
        norm = str(seed).strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        seeds.append(norm)

    expanded: list[str] = []
    seen_questions: set[str] = set()

    def add(question: str) -> None:
        normalized = " ".join(str(question).strip().split())
        if not normalized:
            return
        key = normalized.lower()
        if key in seen_questions:
            return
        seen_questions.add(key)
        expanded.append(normalized)

    for question in selected:
        add(question)
        if breadth <= 1:
            continue
        for seed in seeds[: breadth - 1]:
            stem = question.rstrip(" ?")
            add(f"{stem}, with specific evidence about {seed}?")

    if depth > 1:
        depth_seeds = seeds[:breadth] or [topic_title]
        for level in range(2, depth + 1):
            for seed in depth_seeds:
                add(
                    "What deeper source-backed relationships, limitations, "
                    f"or counterexamples connect {seed} to {topic_title} "
                    f"(depth {level})?"
                )

    cap = max_iterations * breadth * depth
    return expanded[: max(1, cap)]


def build_plan(
    *,
    topic,
    by_topic_seeds: dict[str, Any],
    entity_index: dict[str, Any],
    bound_vector_store_digest: str,
    run_id: str,
    llm_client: QueryPlanLLMClient | None = None,
    expansion_llm_client: QueryExpansionLLMClient | None = None,
    tavily_enabled: bool = True,
    research_parameters: GPTResearcherParameters | None = None,
    research_provider: str = "openai",
    research_model: str | None = None,
    conference_label: str = "the conference",
) -> QueryPlan:
    """Build a single per-topic query plan.

    The default heuristic path emits one well-formed plan per topic without
    any provider call. Optional LLM-assist refinement is env-gated by
    ``QUERY_PLANNER_LLM_ASSIST=1`` AND a non-None ``llm_client``.
    """
    try:
        parameters = research_parameters or GPTResearcherParameters()
        evidence_terms = _evidence_terms(by_topic_seeds, limit=10)
        research_questions = _heuristic_questions(
            topic, evidence_terms, conference_label=conference_label
        )
        synonyms = _heuristic_synonyms(topic, evidence_terms, research_questions)
        entity_seeds = _entity_seeds_for_topic(
            entity_index, getattr(topic, "id", ""), limit=15
        )

        if llm_client is not None and os.environ.get("QUERY_PLANNER_LLM_ASSIST") == "1":
            try:
                refinement = llm_client.generate(
                    f"Refine {conference_label} query plan for topic "
                    f"{getattr(topic, 'id', '')}: questions={research_questions}, "
                    f"evidence_terms={evidence_terms}"
                )
                if isinstance(refinement, dict):
                    extra_questions = refinement.get("research_questions") or []
                    if isinstance(extra_questions, list):
                        for question in extra_questions:
                            if isinstance(question, str) and question.strip():
                                if question not in research_questions:
                                    research_questions.append(question.strip())
                                if len(research_questions) >= 5:
                                    break
                    extra_syn = refinement.get("synonym_expansion") or []
                    if isinstance(extra_syn, list):
                        for syn in extra_syn:
                            if isinstance(syn, str) and syn.strip():
                                if syn not in synonyms:
                                    synonyms.append(syn.strip())
                                if len(synonyms) >= 8:
                                    break
            except Exception as exc:  # noqa: BLE001 - boundary catch
                # Sanitize error and continue with heuristic plan.
                _ = sanitize_for_receipt(str(exc))

        # D-09: local-LLM assisted query expansion when enabled and a client is available.
        expansion_receipt: dict[str, Any] = {}
        topic_title_for_expansion = getattr(topic, "title", "") or getattr(topic, "id", "")
        if parameters.query_expansion_enabled and expansion_llm_client is not None:
            try:
                variants = expansion_llm_client.expand_queries(
                    topic_title_for_expansion, list(research_questions)
                )
                added = 0
                seen_lower = {q.lower() for q in research_questions}
                max_variants = max(1, int(parameters.query_expansion_max_variants))
                for variant in variants[:max_variants]:
                    if not isinstance(variant, str) or not variant.strip():
                        continue
                    norm = variant.strip()
                    if norm.lower() not in seen_lower:
                        research_questions.append(norm)
                        seen_lower.add(norm.lower())
                        added += 1
                expansion_receipt = {
                    "expansion_mode": "local_assisted",
                    "variants_added": added,
                    "failure_reason": None,
                    "fallback_mode": None,
                }
            except Exception as exc:  # noqa: BLE001 - D-06: enabled expansion failure HARD-BLOCKS
                # Phase 11 D-06 (intentional override of Phase 04 D-10): when
                # expansion is explicitly enabled and a client is present, a
                # per-topic expand_queries failure is a hard operator promise.
                # Re-raise a typed, topic-tagged error instead of silently
                # degrading to deterministic_fallback. See the dedicated
                # ``except QueryExpansionError: raise`` below (Pitfall 1) which
                # lets it bypass the outer wrapper with topic_id intact.
                sanitized = sanitize_for_receipt(str(exc))
                msg = (
                    sanitized if isinstance(sanitized, str) else "expansion_failed"
                )
                raise QueryExpansionError(
                    topic_id=getattr(topic, "id", ""),
                    message=msg,
                    retryable=True,
                ) from None
        else:
            expansion_receipt = {
                "expansion_mode": "disabled",
                "variants_added": 0,
                "failure_reason": (
                    None
                    if not parameters.query_expansion_enabled
                    else "no_expansion_client"
                ),
                "fallback_mode": None,
            }

        research_questions = _expand_research_questions(
            topic_title=topic_title_for_expansion,
            base_questions=research_questions,
            synonyms=synonyms,
            entity_seeds=entity_seeds,
            parameters=parameters,
        )

        retrieval_filters = {
            "topic_ids": [getattr(topic, "id", "")],
            "min_confidence": 0.6,
        }
        source_gap_policy = {
            "min_local_hits": 5,
            "fallback": "tavily" if tavily_enabled else "none",
            "tavily_max_results": 5,
        }
        resolved_provider = str(research_provider or "openai")
        resolved_model = str(
            research_model
            or default_model_for_provider("research.gpt_researcher_fast", resolved_provider)
        )
        plan_body = {
            "topic_id": getattr(topic, "id", ""),
            "topic_title": getattr(topic, "title", ""),
            "research_questions": research_questions,
            "synonym_expansion": synonyms,
            "entity_seeds": entity_seeds,
            "retrieval_filters": retrieval_filters,
            "model": resolved_model,
            "provider": resolved_provider,
            "source_gap_policy": source_gap_policy,
            "bound_vector_store_digest": bound_vector_store_digest,
            "run_id": run_id,
            "research_parameters": parameters.to_payload(),
        }
        plan_digest = compute_plan_digest(plan_body)
        return QueryPlan(
            topic_id=plan_body["topic_id"],
            topic_title=plan_body["topic_title"],
            research_questions=tuple(research_questions),
            synonym_expansion=tuple(synonyms),
            entity_seeds=tuple(entity_seeds),
            retrieval_filters=retrieval_filters,
            model=resolved_model,
            provider=resolved_provider,
            output_schema=REFERENCE_QUALITY_PROMPT,
            source_gap_policy=source_gap_policy,
            bound_vector_store_digest=bound_vector_store_digest,
            plan_digest=plan_digest,
            generated_at=utc_now(),
            run_id=run_id,
            research_parameters=parameters.to_payload(),
            expansion_receipt=expansion_receipt,
        )
    except QueryExpansionError:
        # Phase 11 D-06/D-07 (Pitfall 1): the enabled-expansion hard-block error
        # carries topic_id and must bypass the generic re-wrap below, which would
        # otherwise flatten it into a bare QueryPlannerError(... ) from None and
        # discard both the type and the topic id.
        raise
    except Exception as exc:  # noqa: BLE001 - boundary catch
        sanitized = sanitize_for_receipt(str(exc))
        msg = sanitized if isinstance(sanitized, str) else "query_planner_failed"
        raise QueryPlannerError(msg) from None


__all__ = [
    "FakeQueryExpansionLLMClient",
    "FakeQueryPlanLLMClient",
    "QueryExpansionError",
    "QueryExpansionLLMClient",
    "QueryPlanLLMClient",
    "QueryPlannerError",
    "build_plan",
]
