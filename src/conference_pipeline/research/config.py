"""GPT-Researcher runtime parameters for query-planning research stages."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_int(value: Any, fallback: int | None, *, minimum: int | None = None) -> int | None:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    if minimum is not None and parsed < minimum:
        return fallback
    return parsed


def _coerce_float(
    value: Any,
    fallback: float | None,
    *,
    minimum: float | None = None,
) -> float | None:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if minimum is not None and parsed < minimum:
        return fallback
    return parsed


def _coerce_str(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    candidate = str(value).strip()
    return candidate or fallback


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if value is None or value == "":
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return fallback


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
        upper = name.upper()
        if upper in mapping:
            return mapping[upper]
    return None


def _env_value(env: Mapping[str, str], *names: str) -> Any:
    for name in names:
        if name in env:
            return env[name]
    return None



# Bumped when hybrid fusion changes behavior, so research-cache keys and
# retrieval bundles made under the old fusion are not reused.
RRF_FUSION_VERSION = "equal_weight_rrf.v2"

@dataclass(frozen=True)
class GPTResearcherParameters:
    """Operator-facing GPT-Researcher-style knobs.

    The production stage does not instantiate upstream ``GPTResearcher``
    directly; it maps these familiar knobs onto the local query-plan,
    retrieval, and synthesis boundaries.
    """

    max_iterations: int = 5
    deep_research_breadth: int = 4
    deep_research_depth: int = 3
    deep_research_concurrency: int = 2
    max_subtopics: int = 8
    total_words: int = 3000
    smart_token_limit: int | None = 8192
    strategic_token_limit: int | None = 8192
    fast_token_limit: int | None = 4096
    temperature: float | None = 0.0
    retrieval_mode: str = "hybrid"
    retrieval_k: int = 18
    bm25_weight: float | None = None
    vector_weight: float | None = None
    rrf_k: int = 60
    candidate_multiplier: int = 3
    topic_concurrency: int | None = None
    question_concurrency: int | None = None
    query_expansion_enabled: bool = False
    query_expansion_max_variants: int = 3
    semantic_dedup_enabled: bool = False
    semantic_dedup_threshold: float = 0.82

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "GPTResearcherParameters":
        """Build parameters from config mapping + env overrides.

        Config keys accept either snake_case or uppercase GPT-Researcher names.
        Environment supports prefixed names (``GPTR_MAX_ITERATIONS``) and the
        uppercase package names used by upstream GPT-Researcher
        (``MAX_ITERATIONS``).
        """

        config = _as_mapping(mapping)
        env_map: Mapping[str, str] = env if env is not None else os.environ
        defaults = cls()

        def pick(*config_names: str, env_names: str = "") -> Any:
            aliases = tuple(name.strip() for name in env_names.split(",") if name.strip())
            env_picked = _env_value(env_map, *aliases)
            if env_picked is not None:
                return env_picked
            return _mapping_value(config, *config_names)

        max_iterations = _coerce_int(
            pick("max_iterations", env_names="GPTR_MAX_ITERATIONS,MAX_ITERATIONS"),
            defaults.max_iterations,
            minimum=1,
        )
        breadth = _coerce_int(
            pick(
                "deep_research_breadth",
                "breadth",
                env_names="GPTR_DEEP_RESEARCH_BREADTH,DEEP_RESEARCH_BREADTH",
            ),
            defaults.deep_research_breadth,
            minimum=1,
        )
        depth = _coerce_int(
            pick(
                "deep_research_depth",
                "depth",
                env_names="GPTR_DEEP_RESEARCH_DEPTH,DEEP_RESEARCH_DEPTH",
            ),
            defaults.deep_research_depth,
            minimum=1,
        )
        concurrency = _coerce_int(
            pick(
                "deep_research_concurrency",
                "concurrency",
                env_names="GPTR_DEEP_RESEARCH_CONCURRENCY,DEEP_RESEARCH_CONCURRENCY",
            ),
            defaults.deep_research_concurrency,
            minimum=1,
        )
        max_subtopics = _coerce_int(
            pick("max_subtopics", env_names="GPTR_MAX_SUBTOPICS,MAX_SUBTOPICS"),
            defaults.max_subtopics,
            minimum=1,
        )
        total_words = _coerce_int(
            pick("total_words", env_names="GPTR_TOTAL_WORDS,TOTAL_WORDS"),
            defaults.total_words,
            minimum=1,
        )
        smart_token_limit = _coerce_int(
            pick(
                "smart_token_limit",
                env_names="GPTR_SMART_TOKEN_LIMIT,SMART_TOKEN_LIMIT",
            ),
            defaults.smart_token_limit,
            minimum=1,
        )
        strategic_token_limit = _coerce_int(
            pick(
                "strategic_token_limit",
                env_names="GPTR_STRATEGIC_TOKEN_LIMIT,STRATEGIC_TOKEN_LIMIT",
            ),
            defaults.strategic_token_limit,
            minimum=1,
        )
        fast_token_limit = _coerce_int(
            pick("fast_token_limit", env_names="GPTR_FAST_TOKEN_LIMIT,FAST_TOKEN_LIMIT"),
            defaults.fast_token_limit,
            minimum=1,
        )
        temperature = _coerce_float(
            pick("temperature", env_names="GPTR_TEMPERATURE,TEMPERATURE"),
            defaults.temperature,
            minimum=0.0,
        )
        retrieval_mode = _coerce_str(
            pick("retrieval_mode", env_names="GPTR_RETRIEVAL_MODE,RETRIEVAL_MODE"),
            defaults.retrieval_mode,
        ).lower()
        if retrieval_mode not in {"hybrid", "vector"}:
            retrieval_mode = defaults.retrieval_mode
        retrieval_k = _coerce_int(
            pick("retrieval_k", env_names="GPTR_RETRIEVAL_K,RETRIEVAL_K"),
            defaults.retrieval_k,
            minimum=1,
        )
        bm25_weight = _coerce_float(
            pick("bm25_weight", env_names="GPTR_BM25_WEIGHT,BM25_WEIGHT"),
            defaults.bm25_weight,
            minimum=0.0,
        )
        vector_weight = _coerce_float(
            pick("vector_weight", env_names="GPTR_VECTOR_WEIGHT,VECTOR_WEIGHT"),
            defaults.vector_weight,
            minimum=0.0,
        )
        rrf_k = _coerce_int(
            pick("rrf_k", env_names="GPTR_RRF_K,RRF_K"),
            defaults.rrf_k,
            minimum=1,
        )
        candidate_multiplier = _coerce_int(
            pick(
                "candidate_multiplier",
                env_names="GPTR_CANDIDATE_MULTIPLIER,CANDIDATE_MULTIPLIER",
            ),
            defaults.candidate_multiplier,
            minimum=1,
        )
        resolved_concurrency = int(concurrency or defaults.deep_research_concurrency)
        topic_concurrency = _coerce_int(
            pick(
                "topic_concurrency",
                env_names="GPTR_TOPIC_CONCURRENCY,TOPIC_CONCURRENCY",
            ),
            None,
            minimum=1,
        )
        if topic_concurrency is None:
            topic_concurrency = resolved_concurrency
        question_concurrency = _coerce_int(
            pick(
                "question_concurrency",
                env_names="GPTR_QUESTION_CONCURRENCY,QUESTION_CONCURRENCY",
            ),
            None,
            minimum=1,
        )
        if question_concurrency is None:
            question_concurrency = resolved_concurrency

        query_expansion_enabled = _coerce_bool(
            pick(
                "query_expansion_enabled",
                env_names="GPTR_QUERY_EXPANSION_ENABLED,QUERY_EXPANSION_ENABLED",
            ),
            defaults.query_expansion_enabled,
        )
        query_expansion_max_variants = _coerce_int(
            pick(
                "query_expansion_max_variants",
                env_names="GPTR_QUERY_EXPANSION_MAX_VARIANTS,QUERY_EXPANSION_MAX_VARIANTS",
            ),
            defaults.query_expansion_max_variants,
            minimum=1,
        )
        semantic_dedup_enabled = _coerce_bool(
            pick(
                "semantic_dedup_enabled",
                env_names="GPTR_SEMANTIC_DEDUP_ENABLED,SEMANTIC_DEDUP_ENABLED",
            ),
            defaults.semantic_dedup_enabled,
        )
        semantic_dedup_threshold = _coerce_float(
            pick(
                "semantic_dedup_threshold",
                env_names="GPTR_SEMANTIC_DEDUP_THRESHOLD,SEMANTIC_DEDUP_THRESHOLD",
            ),
            defaults.semantic_dedup_threshold,
            minimum=0.0,
        )

        return cls(
            max_iterations=int(max_iterations or defaults.max_iterations),
            deep_research_breadth=int(breadth or defaults.deep_research_breadth),
            deep_research_depth=int(depth or defaults.deep_research_depth),
            deep_research_concurrency=int(
                concurrency or defaults.deep_research_concurrency
            ),
            max_subtopics=int(max_subtopics or defaults.max_subtopics),
            total_words=int(total_words or defaults.total_words),
            smart_token_limit=smart_token_limit,
            strategic_token_limit=strategic_token_limit,
            fast_token_limit=fast_token_limit,
            temperature=temperature,
            retrieval_mode=retrieval_mode,
            retrieval_k=int(retrieval_k or defaults.retrieval_k),
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            rrf_k=int(rrf_k or defaults.rrf_k),
            candidate_multiplier=int(
                candidate_multiplier or defaults.candidate_multiplier
            ),
            topic_concurrency=topic_concurrency,
            question_concurrency=question_concurrency,
            query_expansion_enabled=query_expansion_enabled,
            query_expansion_max_variants=int(
                query_expansion_max_variants or defaults.query_expansion_max_variants
            ),
            semantic_dedup_enabled=semantic_dedup_enabled,
            semantic_dedup_threshold=float(
                semantic_dedup_threshold
                if semantic_dedup_threshold is not None
                else defaults.semantic_dedup_threshold
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "MAX_ITERATIONS": self.max_iterations,
            "DEEP_RESEARCH_BREADTH": self.deep_research_breadth,
            "DEEP_RESEARCH_DEPTH": self.deep_research_depth,
            "DEEP_RESEARCH_CONCURRENCY": self.deep_research_concurrency,
            "MAX_SUBTOPICS": self.max_subtopics,
            "TOTAL_WORDS": self.total_words,
            "SMART_TOKEN_LIMIT": self.smart_token_limit,
            "STRATEGIC_TOKEN_LIMIT": self.strategic_token_limit,
            "FAST_TOKEN_LIMIT": self.fast_token_limit,
            "TEMPERATURE": self.temperature,
            "retrieval_mode": self.retrieval_mode,
            "retrieval_k": self.retrieval_k,
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "rrf_k": self.rrf_k,
            "candidate_multiplier": self.candidate_multiplier,
            "rrf_fusion": RRF_FUSION_VERSION,
        }
        if self.topic_concurrency is not None:
            payload["topic_concurrency"] = self.topic_concurrency
        if self.question_concurrency is not None:
            payload["question_concurrency"] = self.question_concurrency
        payload["query_expansion_enabled"] = self.query_expansion_enabled
        payload["query_expansion_max_variants"] = self.query_expansion_max_variants
        payload["semantic_dedup_enabled"] = self.semantic_dedup_enabled
        payload["semantic_dedup_threshold"] = self.semantic_dedup_threshold
        return payload

    def receipt_only_payload(self) -> dict[str, Any]:
        """Parameters accepted for parity but not currently execution controls."""

        return {
            "DEEP_RESEARCH_CONCURRENCY": self.deep_research_concurrency,
            "STRATEGIC_TOKEN_LIMIT": self.strategic_token_limit,
            "FAST_TOKEN_LIMIT": self.fast_token_limit,
        }

    def behavioral_payload(self) -> dict[str, Any]:
        """Payload fields that change runtime behavior or synthesis prompts.

        ``DEEP_RESEARCH_CONCURRENCY``, ``STRATEGIC_TOKEN_LIMIT``, and
        ``FAST_TOKEN_LIMIT`` are recorded for operator parity with upstream
        GPT-Researcher, but this stage does not execute separate strategic or
        fast-model calls. Excluding them from the behavioral digest avoids
        spurious cache invalidations. ``topic_concurrency`` and
        ``question_concurrency`` are explicit fanout controls that directly
        govern parallel execution; they are included when set.
        """

        payload = self.to_payload()
        for key in self.receipt_only_payload():
            payload.pop(key, None)
        return payload

    def digest(self) -> str:
        payload = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def behavioral_digest(self) -> str:
        payload = json.dumps(
            self.behavioral_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_gpt_researcher_parameters_from_workflow_config(
    config_data: Mapping[str, Any] | None,
    *,
    env: Mapping[str, str] | None = None,
) -> GPTResearcherParameters:
    """Resolve ``workflow.research.gpt_researcher`` from workspace config."""

    data = _as_mapping(config_data)
    workflow_data = _as_mapping(data.get("workflow"))
    research_data = _as_mapping(workflow_data.get("research"))
    nested = _as_mapping(
        research_data.get("gpt_researcher")
        or research_data.get("gpt_researcher_parameters")
        or research_data.get("parameters")
    )
    return GPTResearcherParameters.from_mapping(nested, env=env)


__all__ = [
    "GPTResearcherParameters",
    "load_gpt_researcher_parameters_from_workflow_config",
]
