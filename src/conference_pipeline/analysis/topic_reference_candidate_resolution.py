"""Bounded LLM adjudication for topic reference repair candidates.

This module is the safer first-line repair lane after deterministic topic
reference validation blocks a local reconciliation. Deterministic code finds a
small set of plausible canonical references; local Qwen may only choose among
those candidates; deterministic validation remains the final gate before any
markdown is applied.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from conference_pipeline.analysis.topic_reconciliation import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    estimate_tokens,
)
from conference_pipeline.analysis.topic_reference_validation import (
    ReferenceEntry,
    ReferenceValidationResult,
    TopicReferenceLedger,
    TopicReferenceValidationError,
    build_reference_ledger,
    normalize_text,
    render_validation_markdown,
    sha256_text,
    validate_reconciliation_references,
    workspace_relpath,
)
from conference_pipeline.receipt_lineage import refresh_substep_outputs
from conference_pipeline.research.deep_dive_lanes import DEFAULT_QWEN36_MODEL
from conference_pipeline.research.local_researcher import LocalLLMResearcher
from conference_pipeline.workflow_runs import write_json_atomic

OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION = (
    "analysis.topic_reference_candidate_resolution"
)
RECEIPT_TYPE_TOPIC_REFERENCE_CANDIDATE_RESOLUTION = (
    "conference_pipeline.topic_reference_candidate_resolution"
)
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_MAX_CANDIDATES_PER_REFERENCE = 12
# Hard stop on packet growth. ``max_candidates_per_reference`` is the starting
# size; a packet may keep more candidates while it still fits the Qwen input
# budget (see ``_fit_candidates_to_token_budget``), because a packet truncated
# purely by the count cap blocks EVERY candidate in it from ever being a visible
# source -- four ICML 2026 topic-01 packets held 16 candidates at ~4.2k tokens
# against an 8k budget and blocked five real papers for no budget reason.
DEFAULT_MAX_CANDIDATE_HARD_CEILING = 40
DEFAULT_MAX_SOURCE_SNIPPETS_PER_CANDIDATE = 3
DEFAULT_MAX_SNIPPET_CHARS = 600
DEFAULT_MAX_PACKET_INPUT_TOKENS = 8000
DEFAULT_MAX_REFERENCES_PER_QWEN_CALL = 8
DEFAULT_MIN_CANDIDATE_SCORE = 0.45
DEFAULT_AMBIGUITY_MARGIN = 0.10
ALLOWED_DECISIONS = {
    "replace_with_candidate",
    "drop_reference",
    "needs_escalation",
    "no_safe_match",
}
ALLOWED_CONFIDENCE_LEVELS = {"high", "medium", "low"}
REFERENCE_TOKEN_RE = re.compile(
    r"\b(?:paper_abstract|speech_transcript):[A-Za-z0-9._-]+(?::segment:\d{1,8})?\b"
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")


class TopicReferenceCandidateResolutionError(RuntimeError):
    """Raised when candidate resolution cannot safely continue."""


class AsyncResearcherLike(Protocol):
    provider_receipt_paths: list[str]
    model: str

    async def run(self, query: str, *, context_path: Path | None = None) -> str: ...

    def usage(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CandidateResolutionConfig:
    max_candidates_per_reference: int = DEFAULT_MAX_CANDIDATES_PER_REFERENCE
    max_source_snippets_per_candidate: int = DEFAULT_MAX_SOURCE_SNIPPETS_PER_CANDIDATE
    max_snippet_chars: int = DEFAULT_MAX_SNIPPET_CHARS
    max_packet_input_tokens: int = DEFAULT_MAX_PACKET_INPUT_TOKENS
    max_references_per_qwen_call: int = DEFAULT_MAX_REFERENCES_PER_QWEN_CALL
    min_candidate_score: float = DEFAULT_MIN_CANDIDATE_SCORE
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN
    max_candidate_hard_ceiling: int = DEFAULT_MAX_CANDIDATE_HARD_CEILING

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates_per_reference": self.max_candidates_per_reference,
            "max_candidate_hard_ceiling": self.max_candidate_hard_ceiling,
            "max_source_snippets_per_candidate": self.max_source_snippets_per_candidate,
            "max_snippet_chars": self.max_snippet_chars,
            "max_packet_input_tokens": self.max_packet_input_tokens,
            "max_references_per_qwen_call": self.max_references_per_qwen_call,
            "min_candidate_score": self.min_candidate_score,
            "ambiguity_margin": self.ambiguity_margin,
        }


@dataclass(frozen=True)
class CandidateFeature:
    name: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": self.score, "reason": self.reason}


@dataclass
class ReferenceCandidate:
    canonical_id: str
    source_family: str
    titles_or_labels: list[str] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    snippets: list[dict[str, Any]] = field(default_factory=list)
    features: list[CandidateFeature] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.features:
            return 0.0
        base = max(feature.score for feature in self.features)
        bonus = min(0.12, 0.03 * max(0, len(self.features) - 1))
        return round(min(1.0, base + bonus), 4)

    @property
    def has_direct_evidence_feature(self) -> bool:
        return any(
            feature.name
            in {
                "exact_canonical_id",
                "exact_bare_id",
                "validator_foreign_alias",
                "normalized_alias",
            }
            for feature in self.features
        )

    def add_feature(self, name: str, score: float, reason: str) -> None:
        self.features.append(CandidateFeature(name=name, score=score, reason=reason))

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "source_family": self.source_family,
            "score": self.score,
            "has_direct_evidence_feature": self.has_direct_evidence_feature,
            "titles_or_labels": self.titles_or_labels,
            "source_paths": self.source_paths,
            "metadata": self.metadata,
            "snippets": self.snippets,
            "features": [feature.to_dict() for feature in self.features],
        }


@dataclass(frozen=True)
class ReferenceWorkItem:
    work_item_id: str
    issue_type: str
    line: int | None
    reference_id: str | None
    line_text: str
    foreign_title_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "issue_type": self.issue_type,
            "line": self.line,
            "reference_id": self.reference_id,
            "line_text": self.line_text,
            "foreign_title_aliases": {
                str(key): list(value) for key, value in self.foreign_title_aliases.items()
            },
            "message": self.message,
        }


@dataclass(frozen=True)
class ReferenceCandidatePacket:
    topic_id: str
    title: str
    work_item: ReferenceWorkItem
    candidates: tuple[ReferenceCandidate, ...]
    blockers: tuple[dict[str, Any], ...]
    token_estimate: int
    packet_truncated: bool
    config: CandidateResolutionConfig
    raw_candidate_count: int = 0

    @property
    def candidate_ids(self) -> set[str]:
        return {candidate.canonical_id for candidate in self.candidates}

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "work_item": self.work_item.to_dict(),
            "candidate_count": len(self.candidates),
            "raw_candidate_count": self.raw_candidate_count,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "blockers": list(self.blockers),
            "packet_truncated": self.packet_truncated,
            "token_estimate": self.token_estimate,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_prompt_dict()
        payload["config"] = self.config.to_dict()
        return payload


@dataclass(frozen=True)
class QwenCandidateDecision:
    work_item_id: str
    decision: str
    candidate_id: str | None = None
    confidence: str = "unknown"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "work_item_id": self.work_item_id,
            "decision": self.decision,
            "candidate_id": self.candidate_id,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TopicReferenceCandidateResolutionError(f"json_unreadable:{path}") from exc
    if not isinstance(payload, dict):
        raise TopicReferenceCandidateResolutionError(f"json_not_object:{path}")
    return payload


def _reconciliation_markdown_path(workspace: Path, topic_id: str) -> Path:
    return workspace / "analysis" / "topic_reconciliations" / f"{topic_id}.md"


def _output_paths(workspace: Path, topic_id: str) -> dict[str, Path]:
    root = workspace / "analysis" / "topic_reference_candidate_resolution"
    return {
        "packets": root / f"{topic_id}__packets.json",
        "decisions": root / f"{topic_id}__decisions.json",
        "sidecar": root / f"{topic_id}.json",
        "repaired_markdown": root / f"{topic_id}__repaired.md",
        "summary_markdown": root / f"{topic_id}.md",
        "validation_json": root / f"{topic_id}__validation.json",
        "validation_md": root / f"{topic_id}__validation.md",
    }


def _receipt_path(workspace: Path, run_id: str) -> Path:
    return (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07G-topic-reference-candidate-resolution.json"
    )


def _topic_ids_from_source_packs(workspace: Path) -> list[str]:
    root = workspace / "analysis" / "narrative_source_packs"
    return [path.stem for path in sorted(root.glob("*.json"))]


def _line_at(text: str, lineno: int | None) -> str:
    if lineno is None or lineno <= 0:
        return ""
    lines = text.splitlines()
    if lineno > len(lines):
        return ""
    return lines[lineno - 1]


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else stripped


def _entry_labels(entry: ReferenceEntry) -> list[str]:
    return sorted(entry.titles | entry.source_labels)


def _entry_metadata_text(entry: ReferenceEntry) -> str:
    values = []
    for key in (
        "talk_speaker",
        "talk_id",
        "workshop_id",
        "input_kind",
        "claim",
    ):
        value = entry.metadata.get(key)
        if isinstance(value, str):
            values.append(value)
    return " ".join(values)


def _source_pack_path(workspace: Path, topic_id: str) -> Path:
    return workspace / "analysis" / "narrative_source_packs" / f"{topic_id}.json"


def _summary_paths(workspace: Path, topic_id: str) -> tuple[Path, ...]:
    root = workspace / "analysis" / "exhaustive_summaries"
    return tuple(
        path
        for path in (
            root / f"{topic_id}__papers.json",
            root / f"{topic_id}__transcripts.json",
        )
        if path.exists()
    )


def _add_snippet(
    snippets: dict[str, list[dict[str, Any]]],
    reference_id: Any,
    *,
    text: Any,
    source: str,
    label: Any = None,
    max_chars: int,
) -> None:
    if not isinstance(reference_id, str) or not reference_id:
        return
    if not isinstance(text, str) or not text.strip():
        return
    snippet = {
        "source": source,
        "label": label if isinstance(label, str) else None,
        "text": " ".join(text.split())[:max_chars],
    }
    rows = snippets.setdefault(reference_id, [])
    if snippet not in rows:
        rows.append(snippet)


def collect_source_snippets(
    workspace: Path,
    topic_id: str,
    *,
    max_chars: int = DEFAULT_MAX_SNIPPET_CHARS,
) -> dict[str, list[dict[str, Any]]]:
    snippets: dict[str, list[dict[str, Any]]] = {}
    source_pack_path = _source_pack_path(workspace, topic_id)
    if source_pack_path.exists():
        source_pack = _load_json_object(source_pack_path)
        for chunk in _as_dicts(source_pack.get("chunks")):
            labels = [
                chunk.get("title"),
                chunk.get("paper_title"),
                chunk.get("talk_title"),
                chunk.get("session_title"),
                chunk.get("workshop_title"),
            ]
            text = " ".join(str(item) for item in labels if isinstance(item, str))
            for key in ("chunk_id", "extraction_record_id", "unit_id"):
                _add_snippet(
                    snippets,
                    chunk.get(key),
                    text=text or json.dumps(chunk, sort_keys=True),
                    source="narrative_source_pack.chunks",
                    label=chunk.get("source_artifact_path"),
                    max_chars=max_chars,
                )
        llm_evidence = source_pack.get("llm_full_read_evidence")
        if isinstance(llm_evidence, Mapping):
            for item in _as_dicts(llm_evidence.get("selected_items")):
                labels = [
                    item.get("paper_title"),
                    item.get("talk_title"),
                    item.get("session_title"),
                    item.get("workshop_title"),
                    item.get("talk_speaker"),
                ]
                text = " ".join(str(value) for value in labels if isinstance(value, str))
                _add_snippet(
                    snippets,
                    item.get("unit_id"),
                    text=text or json.dumps(item, sort_keys=True),
                    source="narrative_source_pack.llm_full_read_evidence",
                    label=item.get("source_artifact_path"),
                    max_chars=max_chars,
                )
    for summary_path in _summary_paths(workspace, topic_id):
        summary = _load_json_object(summary_path)
        for item in _as_dicts(summary.get("notable_evidence")):
            text = str(item.get("claim") or "")
            labels = _as_strings(item.get("source_labels"))
            if labels:
                text = f"{text} Labels: {', '.join(labels)}"
            for unit_id in _as_strings(item.get("unit_ids")):
                _add_snippet(
                    snippets,
                    unit_id,
                    text=text,
                    source=workspace_relpath(workspace, summary_path),
                    label=", ".join(labels),
                    max_chars=max_chars,
                )
        for chunk in _as_dicts(summary.get("chunk_summaries")):
            for item in _as_dicts(chunk.get("notable_evidence")):
                text = str(item.get("claim") or "")
                labels = _as_strings(item.get("source_labels"))
                if labels:
                    text = f"{text} Labels: {', '.join(labels)}"
                for unit_id in _as_strings(item.get("unit_ids")):
                    _add_snippet(
                        snippets,
                        unit_id,
                        text=text,
                        source=workspace_relpath(workspace, summary_path),
                        label=", ".join(labels),
                        max_chars=max_chars,
                    )
    return snippets


def extract_reference_work_items(
    *,
    validation: ReferenceValidationResult,
    markdown_text: str,
) -> list[ReferenceWorkItem]:
    work_items: list[ReferenceWorkItem] = []
    for index, reference_id in enumerate(validation.unknown_reference_ids):
        line_no = None
        line_text = ""
        for lineno, line in enumerate(markdown_text.splitlines(), start=1):
            if reference_id in line:
                line_no = lineno
                line_text = line.strip()
                break
        work_items.append(
            ReferenceWorkItem(
                work_item_id=f"unknown-{index}",
                issue_type="unknown_id",
                line=line_no,
                reference_id=reference_id,
                line_text=line_text,
                message="Referenced id is not present in the deterministic ledger.",
            )
        )
    for index, issue in enumerate(validation.bare_reference_ids):
        if not isinstance(issue, Mapping):
            continue
        line_no = issue.get("line") if isinstance(issue.get("line"), int) else None
        work_items.append(
            ReferenceWorkItem(
                work_item_id=f"bare-{index}",
                issue_type="bare_id",
                line=line_no,
                reference_id=str(issue.get("canonical_id") or ""),
                line_text=_line_at(markdown_text, line_no).strip(),
                message="Bare id should be canonicalized.",
            )
        )
    for index, issue in enumerate(validation.title_mismatch_errors):
        if not isinstance(issue, Mapping):
            continue
        raw_foreign = issue.get("foreign_title_aliases")
        foreign_aliases: dict[str, list[str]] = {}
        if isinstance(raw_foreign, Mapping):
            foreign_aliases = {
                str(alias): [str(item) for item in ids if isinstance(item, str)]
                for alias, ids in raw_foreign.items()
                if isinstance(ids, list)
            }
        line_no = issue.get("line") if isinstance(issue.get("line"), int) else None
        work_items.append(
            ReferenceWorkItem(
                work_item_id=f"title-mismatch-{index}",
                issue_type="title_mismatch",
                line=line_no,
                reference_id=str(issue.get("canonical_id") or ""),
                line_text=str(issue.get("line_text") or _line_at(markdown_text, line_no)).strip(),
                foreign_title_aliases=foreign_aliases,
                message=str(issue.get("message") or "Title/id mismatch."),
            )
        )
    return work_items


def _candidate_from_entry(
    entry: ReferenceEntry,
    *,
    snippets_by_id: Mapping[str, list[dict[str, Any]]],
    config: CandidateResolutionConfig,
) -> ReferenceCandidate:
    return ReferenceCandidate(
        canonical_id=entry.canonical_id,
        source_family=entry.source_family,
        titles_or_labels=_entry_labels(entry)[:8],
        source_paths=sorted(entry.source_paths)[:6],
        metadata=entry.metadata,
        snippets=list(snippets_by_id.get(entry.canonical_id, []))[
            : config.max_source_snippets_per_candidate
        ],
    )


def _add_candidate_feature(
    candidates: dict[str, ReferenceCandidate],
    entry: ReferenceEntry,
    *,
    snippets_by_id: Mapping[str, list[dict[str, Any]]],
    config: CandidateResolutionConfig,
    name: str,
    score: float,
    reason: str,
) -> None:
    candidate = candidates.get(entry.canonical_id)
    if candidate is None:
        candidate = _candidate_from_entry(entry, snippets_by_id=snippets_by_id, config=config)
        candidates[entry.canonical_id] = candidate
    candidate.add_feature(name, score, reason)


def _line_anchor_terms(line_text: str) -> set[str]:
    terms = {
        normalize_text(term)
        for term in re.findall(
            r'"([^"]{8,120})"|`([^`]{8,120})`|\b([A-Z][A-Za-z0-9-]{3,})\b',
            line_text,
        )
        for term in term
        if term
    }
    words = [word for word in WORD_RE.findall(line_text) if len(word) >= 5]
    for index in range(0, max(0, len(words) - 2)):
        phrase = normalize_text(" ".join(words[index : index + 3]))
        if len(phrase) >= 12:
            terms.add(phrase)
    return {term for term in terms if len(term) >= 6}


def build_candidate_packet(
    *,
    topic_id: str,
    title: str,
    work_item: ReferenceWorkItem,
    ledger: TopicReferenceLedger,
    snippets_by_id: Mapping[str, list[dict[str, Any]]],
    config: CandidateResolutionConfig = CandidateResolutionConfig(),
) -> ReferenceCandidatePacket:
    candidates: dict[str, ReferenceCandidate] = {}
    line_normalized = normalize_text(work_item.line_text)
    reference_id = work_item.reference_id or ""
    for entry in ledger.entries.values():
        if reference_id and reference_id == entry.canonical_id:
            _add_candidate_feature(
                candidates,
                entry,
                snippets_by_id=snippets_by_id,
                config=config,
                name="exact_canonical_id",
                score=0.95,
                reason="The validator referenced this canonical id directly.",
            )
        if reference_id and reference_id == entry.bare_id:
            _add_candidate_feature(
                candidates,
                entry,
                snippets_by_id=snippets_by_id,
                config=config,
                name="exact_bare_id",
                score=0.95,
                reason="The invalid reference is the bare id for this canonical row.",
            )
        if reference_id and reference_id.endswith(entry.bare_id):
            _add_candidate_feature(
                candidates,
                entry,
                snippets_by_id=snippets_by_id,
                config=config,
                name="suffix_bare_id",
                score=0.82,
                reason="The invalid reference suffix matches this canonical row.",
            )
        if reference_id and ":" in reference_id:
            ratio = difflib.SequenceMatcher(
                None,
                reference_id.lower(),
                entry.canonical_id.lower(),
            ).ratio()
            if ratio >= 0.76:
                _add_candidate_feature(
                    candidates,
                    entry,
                    snippets_by_id=snippets_by_id,
                    config=config,
                    name="close_id_match",
                    score=round(0.35 + (ratio * 0.45), 4),
                    reason=f"Canonical id is close to invalid id (ratio={ratio:.3f}).",
                )
        for ids in work_item.foreign_title_aliases.values():
            if entry.canonical_id in ids:
                _add_candidate_feature(
                    candidates,
                    entry,
                    snippets_by_id=snippets_by_id,
                    config=config,
                    name="validator_foreign_alias",
                    score=0.92,
                    reason="The validator found this row owns a title alias on the blocked line.",
                )
        for label in _entry_labels(entry):
            alias = normalize_text(label)
            if alias and len(alias) >= 6 and re.search(
                rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])",
                line_normalized,
            ):
                _add_candidate_feature(
                    candidates,
                    entry,
                    snippets_by_id=snippets_by_id,
                    config=config,
                    name="normalized_alias",
                    score=0.86,
                    reason=f"Blocked line contains this row's normalized label: {label}",
                )
        metadata_text = normalize_text(_entry_metadata_text(entry))
        if metadata_text:
            for term in _line_anchor_terms(work_item.line_text):
                if term in metadata_text or metadata_text in term:
                    _add_candidate_feature(
                        candidates,
                        entry,
                        snippets_by_id=snippets_by_id,
                        config=config,
                        name="metadata_anchor",
                        score=0.68,
                        reason=f"Blocked line anchor matches metadata: {term}",
                    )
        snippet_text = normalize_text(
            " ".join(
                str(snippet.get("text") or "")
                for snippet in snippets_by_id.get(entry.canonical_id, [])
            )
        )
        if snippet_text:
            for term in _line_anchor_terms(work_item.line_text):
                if len(term) >= 10 and term in snippet_text:
                    _add_candidate_feature(
                        candidates,
                        entry,
                        snippets_by_id=snippets_by_id,
                        config=config,
                        name="source_snippet_anchor",
                        score=0.72,
                        reason=f"Blocked line anchor appears in source snippet: {term}",
                    )
    ranked = sorted(candidates.values(), key=lambda item: (-item.score, item.canonical_id))
    raw_candidate_count = len(ranked)

    def _payload_for(kept: list[ReferenceCandidate]) -> dict[str, Any]:
        return {
            "topic_id": topic_id,
            "title": title,
            "work_item": work_item.to_dict(),
            "candidates": [candidate.to_dict() for candidate in kept],
            "packet_truncated": raw_candidate_count > len(kept),
            "config": config.to_dict(),
        }

    ranked = _fit_candidates_to_token_budget(ranked, config=config, payload_for=_payload_for)
    packet_truncated = raw_candidate_count > len(ranked)
    blockers: list[dict[str, Any]] = []
    if not ranked:
        blockers.append(
            {
                "code": "candidate_resolution_no_candidates",
                "work_item_id": work_item.work_item_id,
                "reason": "No candidate rows matched this validation error.",
                "retryable": True,
            }
        )
    elif ranked[0].score < config.min_candidate_score:
        blockers.append(
            {
                "code": "candidate_resolution_low_confidence",
                "work_item_id": work_item.work_item_id,
                "top_score": ranked[0].score,
                "min_candidate_score": config.min_candidate_score,
                "reason": "Top candidate score is below the configured minimum.",
                "retryable": True,
            }
        )
    elif (
        len(ranked) > 1
        and ranked[0].score - ranked[1].score < config.ambiguity_margin
        and not ranked[0].has_direct_evidence_feature
    ):
        blockers.append(
            {
                "code": "candidate_resolution_ambiguous",
                "work_item_id": work_item.work_item_id,
                "top_score": ranked[0].score,
                "second_score": ranked[1].score,
                "ambiguity_margin": config.ambiguity_margin,
                "reason": "Top two candidates are too close without direct evidence.",
                "retryable": True,
            }
        )
    packet_payload = {
        "topic_id": topic_id,
        "title": title,
        "work_item": work_item.to_dict(),
        "candidates": [candidate.to_dict() for candidate in ranked],
        "packet_truncated": packet_truncated,
        "config": config.to_dict(),
    }
    token_estimate = estimate_tokens(json.dumps(packet_payload, sort_keys=True))
    if token_estimate > config.max_packet_input_tokens:
        blockers.append(
            {
                "code": "candidate_resolution_packet_too_large",
                "work_item_id": work_item.work_item_id,
                "token_estimate": token_estimate,
                "max_packet_input_tokens": config.max_packet_input_tokens,
                "reason": "Candidate packet exceeds the configured Qwen input budget.",
                "retryable": True,
            }
        )
    if packet_truncated or token_estimate > config.max_packet_input_tokens:
        top_candidates = ", ".join(
            f"{candidate.canonical_id}"
            f"[{candidate.titles_or_labels[0] if candidate.titles_or_labels else '?'}]"
            f"(score={candidate.score:.2f})"
            for candidate in ranked[:5]
        )
        print(
            "[candidate_resolution] "
            f"topic={topic_id} work_item={work_item.work_item_id} "
            f"reference={work_item.reference_id!r} line={work_item.line} "
            f"line_text={work_item.line_text[:160]!r} "
            f"raw_candidates={raw_candidate_count} kept_candidates={len(ranked)} "
            f"max_candidates_per_reference={config.max_candidates_per_reference} "
            f"token_estimate={token_estimate} max_packet_input_tokens={config.max_packet_input_tokens} "
            f"packet_truncated={packet_truncated} top_candidates=[{top_candidates}]",
            file=sys.stderr,
        )
    return ReferenceCandidatePacket(
        topic_id=topic_id,
        title=title,
        work_item=work_item,
        candidates=tuple(ranked),
        blockers=tuple(blockers),
        token_estimate=token_estimate,
        packet_truncated=packet_truncated,
        config=config,
        raw_candidate_count=raw_candidate_count,
    )


def _fit_candidates_to_token_budget(
    ranked: list[ReferenceCandidate],
    *,
    config: CandidateResolutionConfig,
    payload_for: Callable[[list[ReferenceCandidate]], dict[str, Any]],
) -> list[ReferenceCandidate]:
    """Keep as many ranked candidates as the Qwen input budget really allows.

    ``max_candidates_per_reference`` is the starting size, not a hard cut: a
    packet grows one candidate at a time (highest score first) while it still
    fits ``max_packet_input_tokens``, up to ``max_candidate_hard_ceiling``.
    ``packet_truncated`` then means the budget genuinely bound, which is what
    the downstream visible-source ledger treats as blocking."""

    start = max(1, min(config.max_candidates_per_reference, len(ranked)))
    kept = ranked[:start]
    ceiling = min(len(ranked), max(start, config.max_candidate_hard_ceiling))
    index = start
    while index < ceiling:
        trial = ranked[: index + 1]
        if estimate_tokens(json.dumps(payload_for(trial), sort_keys=True)) > config.max_packet_input_tokens:
            break
        kept = trial
        index += 1
    return kept


def build_candidate_packets(
    *,
    workspace: Path,
    topic_id: str,
    config: CandidateResolutionConfig = CandidateResolutionConfig(),
) -> tuple[TopicReferenceLedger, ReferenceValidationResult, str, list[ReferenceCandidatePacket]]:
    workspace = workspace.resolve()
    markdown_path = _reconciliation_markdown_path(workspace, topic_id)
    if not markdown_path.exists():
        raise TopicReferenceCandidateResolutionError(
            f"reconciliation_markdown_missing:{topic_id}"
        )
    markdown = markdown_path.read_text(encoding="utf-8", errors="replace")
    ledger = build_reference_ledger(workspace, topic_id)
    validation = validate_reconciliation_references(
        text=markdown,
        ledger=ledger,
        require_citations=True,
    )
    snippets_by_id = collect_source_snippets(
        workspace,
        topic_id,
        max_chars=config.max_snippet_chars,
    )
    packets = [
        build_candidate_packet(
            topic_id=topic_id,
            title=ledger.title,
            work_item=work_item,
            ledger=ledger,
            snippets_by_id=snippets_by_id,
            config=config,
        )
        for work_item in extract_reference_work_items(
            validation=validation,
            markdown_text=markdown,
        )
    ]
    truncated = [packet for packet in packets if packet.packet_truncated]
    if truncated:
        token_estimates = [packet.token_estimate for packet in truncated]
        print(
            "[candidate_resolution] "
            f"topic={topic_id} summary: {len(truncated)} of {len(packets)} packets truncated "
            f"(raw candidates > {config.max_candidates_per_reference}); "
            f"token_estimate range for truncated packets: "
            f"{min(token_estimates)}-{max(token_estimates)} "
            f"(budget={config.max_packet_input_tokens})",
            file=sys.stderr,
        )
    return ledger, validation, markdown, packets


def build_qwen_candidate_prompt(packets: Sequence[ReferenceCandidatePacket]) -> str:
    packet_rows = [packet.to_prompt_dict() for packet in packets]
    return (
        "You are resolving invalid citations in a conference analysis report. "
        "Choose only from the provided candidate rows. Do not invent ids, titles, "
        "or sources. Return strict JSON only.\n\n"
        "Allowed decision values: replace_with_candidate, drop_reference, "
        "needs_escalation, no_safe_match.\n\n"
        "Use replace_with_candidate only when a candidate clearly backs the "
        "blocked line. Use needs_escalation or no_safe_match when evidence is "
        "ambiguous. Use drop_reference only when no candidate can support the "
        "citation and removing the citation is safer than guessing.\n\n"
        "Required JSON schema:\n"
        '{"decisions":[{"work_item_id":"...","decision":"replace_with_candidate",'
        '"candidate_id":"paper_abstract:...","confidence":"high|medium|low",'
        '"rationale":"short evidence-based reason"}]}\n\n'
        "Candidate packets:\n"
        f"{json.dumps(packet_rows, ensure_ascii=True, indent=2, sort_keys=True)}\n"
    )


def parse_qwen_candidate_decisions(
    text: str,
    *,
    packets: Sequence[ReferenceCandidatePacket],
) -> list[QwenCandidateDecision]:
    packet_by_id = {packet.work_item.work_item_id: packet for packet in packets}
    allowed_ids = {
        packet.work_item.work_item_id: packet.candidate_ids for packet in packets
    }
    try:
        payload = json.loads(_strip_code_fence(text))
    except json.JSONDecodeError as exc:
        raise TopicReferenceCandidateResolutionError("qwen_decision_json_invalid") from exc
    if not isinstance(payload, Mapping):
        raise TopicReferenceCandidateResolutionError("qwen_decision_json_not_object")
    rows = payload.get("decisions")
    if not isinstance(rows, list):
        raise TopicReferenceCandidateResolutionError("qwen_decisions_missing")
    decisions: list[QwenCandidateDecision] = []
    seen: set[str] = set()
    all_candidate_ids = set().union(*allowed_ids.values()) if allowed_ids else set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TopicReferenceCandidateResolutionError("qwen_decision_not_object")
        work_item_id = row.get("work_item_id")
        decision = row.get("decision")
        candidate_id = row.get("candidate_id")
        rationale = str(row.get("rationale") or "").strip()
        confidence = str(row.get("confidence") or "").strip().lower()
        if not isinstance(work_item_id, str) or work_item_id not in packet_by_id:
            raise TopicReferenceCandidateResolutionError("qwen_decision_unknown_work_item")
        if work_item_id in seen:
            raise TopicReferenceCandidateResolutionError("qwen_decision_duplicate_work_item")
        seen.add(work_item_id)
        if decision not in ALLOWED_DECISIONS:
            raise TopicReferenceCandidateResolutionError("qwen_decision_type_invalid")
        if confidence not in ALLOWED_CONFIDENCE_LEVELS:
            raise TopicReferenceCandidateResolutionError("qwen_decision_confidence_invalid")
        if decision == "replace_with_candidate":
            if confidence == "low":
                raise TopicReferenceCandidateResolutionError("qwen_decision_low_confidence")
            if not isinstance(candidate_id, str) or candidate_id not in allowed_ids[work_item_id]:
                raise TopicReferenceCandidateResolutionError("qwen_decision_candidate_not_in_packet")
        elif candidate_id not in (None, ""):
            raise TopicReferenceCandidateResolutionError("qwen_decision_unexpected_candidate")
        if decision == "drop_reference" and not rationale:
            raise TopicReferenceCandidateResolutionError("qwen_decision_drop_without_rationale")
        for mentioned_id in REFERENCE_TOKEN_RE.findall(rationale):
            if mentioned_id not in all_candidate_ids:
                raise TopicReferenceCandidateResolutionError("qwen_decision_rationale_invented_id")
        decisions.append(
            QwenCandidateDecision(
                work_item_id=work_item_id,
                decision=str(decision),
                candidate_id=candidate_id if isinstance(candidate_id, str) else None,
                confidence=confidence,
                rationale=rationale,
            )
        )
    missing = set(packet_by_id) - seen
    if missing:
        raise TopicReferenceCandidateResolutionError("qwen_decision_missing_work_items")
    return decisions


def _replace_reference_on_line(line: str, old: str, new: str) -> tuple[str, bool]:
    if not old:
        return line, False
    updated = line.replace(f"`{old}`", f"`{new}`")
    if updated != line:
        return updated, True
    pattern = re.compile(rf"(?<![A-Za-z0-9._:-]){re.escape(old)}(?![A-Za-z0-9._:-])")
    updated = pattern.sub(new, line)
    return updated, updated != line


def _remove_reference_on_line(line: str, reference_id: str) -> tuple[str, bool]:
    updated = line.replace(f"`{reference_id}`", "")
    if updated == line:
        updated = line.replace(reference_id, "")
    updated = re.sub(r"\s{2,}", " ", updated)
    updated = re.sub(r"\s+([,.;:])", r"\1", updated)
    return updated.rstrip(), updated != line


def apply_candidate_decisions(
    *,
    markdown: str,
    packets: Sequence[ReferenceCandidatePacket],
    decisions: Sequence[QwenCandidateDecision],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    packet_by_id = {packet.work_item.work_item_id: packet for packet in packets}
    lines = markdown.splitlines()
    applied: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for decision in decisions:
        packet = packet_by_id[decision.work_item_id]
        work_item = packet.work_item
        if decision.decision in {"needs_escalation", "no_safe_match"}:
            blockers.append(
                {
                    "code": f"candidate_resolution_{decision.decision}",
                    "work_item_id": decision.work_item_id,
                    "reason": decision.rationale or "Qwen did not select a safe replacement.",
                    "retryable": True,
                }
            )
            continue
        if work_item.line is None or not (1 <= work_item.line <= len(lines)):
            blockers.append(
                {
                    "code": "candidate_resolution_line_missing",
                    "work_item_id": decision.work_item_id,
                    "reason": "Cannot apply a line-scoped edit without a valid line number.",
                    "retryable": True,
                }
            )
            continue
        before = lines[work_item.line - 1]
        if decision.decision == "replace_with_candidate":
            replacement = decision.candidate_id or ""
            old = work_item.reference_id or ""
            if work_item.issue_type == "bare_id" and old.startswith(("paper_abstract:", "speech_transcript:")):
                old = old.split(":", 1)[1]
            updated, changed = _replace_reference_on_line(before, old, replacement)
            if not changed and work_item.issue_type == "title_mismatch" and old:
                updated, changed = _replace_reference_on_line(before, old, replacement)
            if not changed:
                blockers.append(
                    {
                        "code": "candidate_resolution_edit_not_applied",
                        "work_item_id": decision.work_item_id,
                        "reason": "Selected candidate could not be applied to the blocked line.",
                        "retryable": True,
                    }
                )
                continue
            lines[work_item.line - 1] = updated
            applied.append(
                {
                    "work_item_id": decision.work_item_id,
                    "action": "replace_with_candidate",
                    "line": work_item.line,
                    "candidate_id": replacement,
                }
            )
        elif decision.decision == "drop_reference":
            old = work_item.reference_id or ""
            updated, changed = _remove_reference_on_line(before, old)
            if not changed:
                blockers.append(
                    {
                        "code": "candidate_resolution_drop_not_applied",
                        "work_item_id": decision.work_item_id,
                        "reason": "Reference could not be removed from the blocked line.",
                        "retryable": True,
                    }
                )
                continue
            lines[work_item.line - 1] = updated
            applied.append(
                {
                    "work_item_id": decision.work_item_id,
                    "action": "drop_reference",
                    "line": work_item.line,
                }
            )
    trailing_newline = "\n" if markdown.endswith("\n") else ""
    return "\n".join(lines) + trailing_newline, applied, blockers


def _candidate_has_direct_evidence(
    packet: ReferenceCandidatePacket,
    candidate_id: str | None,
) -> bool:
    if not candidate_id:
        return False
    for candidate in packet.candidates:
        if candidate.canonical_id == candidate_id:
            return candidate.has_direct_evidence_feature
    return False


def _ambiguous_packet_blockers_after_qwen(
    *,
    packets: Sequence[ReferenceCandidatePacket],
    decisions: Sequence[QwenCandidateDecision],
) -> list[dict[str, Any]]:
    decisions_by_work_item = {decision.work_item_id: decision for decision in decisions}
    blockers: list[dict[str, Any]] = []
    for packet in packets:
        ambiguity_blockers = [
            blocker
            for blocker in packet.blockers
            if blocker.get("code") == "candidate_resolution_ambiguous"
        ]
        if not ambiguity_blockers:
            continue
        decision = decisions_by_work_item.get(packet.work_item.work_item_id)
        if (
            decision is not None
            and decision.decision == "replace_with_candidate"
            and _candidate_has_direct_evidence(packet, decision.candidate_id)
        ):
            continue
        blockers.extend(ambiguity_blockers)
    return blockers


def _render_markdown_summary(
    *,
    topic_id: str,
    status: str,
    packets: Sequence[ReferenceCandidatePacket],
    decisions: Sequence[QwenCandidateDecision],
    validation: ReferenceValidationResult,
    blockers: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"# Topic Reference Candidate Resolution: {topic_id}",
        "",
        f"Status: `{status}`",
        "",
        "## Counts",
        "",
        f"- Work items: `{len(packets)}`",
        f"- Qwen decisions: `{len(decisions)}`",
        f"- Final invalid references: `{validation.invalid_reference_count}`",
        f"- Final bare-id warnings: `{validation.bare_reference_warning_count}`",
        "",
    ]
    if blockers:
        lines.extend(["## Blockers", ""])
        for blocker in blockers:
            lines.append(f"- `{blocker.get('code')}`: {blocker.get('reason')}")
        lines.append("")
    lines.extend(["## Decisions", ""])
    for decision in decisions:
        lines.append(
            f"- `{decision.work_item_id}`: `{decision.decision}`"
            + (f" -> `{decision.candidate_id}`" if decision.candidate_id else "")
        )
    return "\n".join(lines).rstrip() + "\n"


def _enrich_source_pack(
    *,
    workspace: Path,
    topic_id: str,
    summary: Mapping[str, Any],
) -> bool:
    path = _source_pack_path(workspace, topic_id)
    if not path.exists():
        return False
    source_pack = _load_json_object(path)
    reconciliation = source_pack.get("topic_reconciliation")
    if not isinstance(reconciliation, dict):
        reconciliation = {}
    reconciliation["reference_candidate_resolution"] = dict(summary)
    if summary.get("applied") and summary.get("status") == "succeeded":
        markdown_path = _reconciliation_markdown_path(workspace, topic_id)
        if markdown_path.exists():
            text = markdown_path.read_text(encoding="utf-8", errors="replace")
            reconciliation["reconciliation_text"] = text
            reconciliation["markdown_sha256"] = sha256_text(text)
            reconciliation["markdown_path"] = workspace_relpath(workspace, markdown_path)
    source_pack["topic_reconciliation"] = reconciliation
    write_json_atomic(path, source_pack)
    return True


def _chunked(items: Sequence[ReferenceCandidatePacket], size: int) -> list[list[ReferenceCandidatePacket]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def resolve_topic_reference_candidates(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    model: str = DEFAULT_QWEN36_MODEL,
    topics: list[str] | None = None,
    config: CandidateResolutionConfig = CandidateResolutionConfig(),
    dry_run: bool = False,
    apply: bool = False,
    enrich_source_packs: bool = True,
    researcher: AsyncResearcherLike | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    started = time.monotonic()
    generated_at = utc_now()
    topic_ids = topics or _topic_ids_from_source_packs(workspace)
    if not topic_ids:
        raise TopicReferenceCandidateResolutionError("narrative_source_packs_missing")
    if mode == "live" and researcher is None:
        researcher = LocalLLMResearcher(smart_model=model, fast_model=model, max_tokens=DEFAULT_MAX_OUTPUT_TOKENS)

    outputs: list[dict[str, Any]] = []
    all_blockers: list[dict[str, Any]] = []
    enriched_count = 0
    applied_count = 0
    for topic_id in topic_ids:
        paths = _output_paths(workspace, topic_id)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            ledger, initial_validation, markdown, packets = build_candidate_packets(
                workspace=workspace,
                topic_id=topic_id,
                config=config,
            )
            packet_blockers = [
                blocker
                for packet in packets
                for blocker in packet.blockers
                if blocker.get("code") != "candidate_resolution_ambiguous"
            ]
            qwen_decisions: list[QwenCandidateDecision] = []
            qwen_errors: list[dict[str, Any]] = []
            if initial_validation.status == "succeeded":
                repaired_markdown = markdown
                applied_edits: list[dict[str, Any]] = []
                edit_blockers: list[dict[str, Any]] = []
                action = "skipped_already_valid"
            elif mode == "live" and not packet_blockers:
                if researcher is None:
                    raise TopicReferenceCandidateResolutionError("local_researcher_missing")
                for batch in _chunked(packets, max(1, config.max_references_per_qwen_call)):
                    prompt = build_qwen_candidate_prompt(batch)
                    response = asyncio.run(researcher.run(prompt))
                    try:
                        qwen_decisions.extend(
                            parse_qwen_candidate_decisions(response, packets=batch)
                        )
                    except TopicReferenceCandidateResolutionError as exc:
                        retry_prompt = (
                            f"The previous response failed schema validation: {exc}. "
                            "Return strict JSON only for the same candidate packets.\n\n"
                            f"{prompt}"
                        )
                        retry_response = asyncio.run(researcher.run(retry_prompt))
                        try:
                            qwen_decisions.extend(
                                parse_qwen_candidate_decisions(retry_response, packets=batch)
                            )
                        except TopicReferenceCandidateResolutionError as retry_exc:
                            qwen_errors.append(
                                {
                                    "code": str(retry_exc).split(":", 1)[0],
                                    "topic_id": topic_id,
                                    "reason": str(retry_exc),
                                    "retryable": True,
                                }
                            )
                repaired_markdown, applied_edits, edit_blockers = apply_candidate_decisions(
                    markdown=markdown,
                    packets=packets,
                    decisions=qwen_decisions,
                )
                action = "qwen_candidate_resolution"
            else:
                qwen_decisions = [
                    QwenCandidateDecision(
                        work_item_id=packet.work_item.work_item_id,
                        decision="needs_escalation",
                        confidence="unknown",
                        rationale="Dry-run or packet blocker prevented Qwen adjudication.",
                    )
                    for packet in packets
                ]
                repaired_markdown = markdown
                applied_edits = []
                edit_blockers = []
                action = "dry_run_no_mutation" if mode == "dry-run" else "blocked_before_qwen"

            final_validation = validate_reconciliation_references(
                text=repaired_markdown,
                ledger=ledger,
                require_citations=True,
            )
            unresolved_decision_blockers = [
                {
                    "code": f"candidate_resolution_{decision.decision}",
                    "topic_id": topic_id,
                    "work_item_id": decision.work_item_id,
                    "reason": decision.rationale or "Qwen did not choose a safe candidate.",
                    "retryable": True,
                }
                for decision in qwen_decisions
                if decision.decision in {"needs_escalation", "no_safe_match"}
            ]
            topic_blockers = (
                list(packet_blockers)
                + qwen_errors
                + edit_blockers
                + _ambiguous_packet_blockers_after_qwen(
                    packets=packets,
                    decisions=qwen_decisions,
                )
                + unresolved_decision_blockers
                + list(final_validation.blockers)
            )
            topic_status = "blocked" if topic_blockers else "succeeded"
            write_json_atomic(
                paths["packets"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "owner_id": OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
                    "topic_id": topic_id,
                    "generated_at": generated_at,
                    "config": config.to_dict(),
                    "initial_validation": initial_validation.to_dict(),
                    "packets": [packet.to_dict() for packet in packets],
                    "packet_count": len(packets),
                    "packet_input_token_estimate": sum(packet.token_estimate for packet in packets),
                    "packet_truncated_count": sum(1 for packet in packets if packet.packet_truncated),
                },
            )
            write_json_atomic(
                paths["decisions"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "owner_id": OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
                    "topic_id": topic_id,
                    "generated_at": generated_at,
                    "model": model,
                    "mode": mode,
                    "decisions": [decision.to_dict() for decision in qwen_decisions],
                    "qwen_errors": qwen_errors,
                    "applied_edits": applied_edits,
                },
            )
            validation_payload = final_validation.to_dict()
            validation_payload.update(
                {
                    "generated_at": generated_at,
                    "run_id": run_id,
                    "candidate_resolution_markdown_path": workspace_relpath(
                        workspace,
                        paths["repaired_markdown"],
                    ),
                    "candidate_resolution_markdown_sha256": sha256_text(repaired_markdown),
                }
            )
            write_json_atomic(paths["validation_json"], validation_payload)
            paths["validation_md"].write_text(
                render_validation_markdown(final_validation),
                encoding="utf-8",
            )
            paths["repaired_markdown"].write_text(repaired_markdown, encoding="utf-8")
            applied = False
            if apply and not dry_run and final_validation.status == "succeeded" and not topic_blockers:
                _reconciliation_markdown_path(workspace, topic_id).write_text(
                    repaired_markdown,
                    encoding="utf-8",
                )
                applied = True
                applied_count += 1
            summary = {
                "status": topic_status,
                "owner_id": OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
                "topic_id": topic_id,
                "title": ledger.title,
                "generated_at": generated_at,
                "run_id": run_id,
                "mode": mode,
                "model": model,
                "action": action,
                "dry_run": dry_run,
                "applied": applied,
                "config": config.to_dict(),
                "packet_path": workspace_relpath(workspace, paths["packets"]),
                "decision_path": workspace_relpath(workspace, paths["decisions"]),
                "repaired_markdown_path": workspace_relpath(
                    workspace,
                    paths["repaired_markdown"],
                ),
                "summary_markdown_path": workspace_relpath(
                    workspace,
                    paths["summary_markdown"],
                ),
                "validation_json_path": workspace_relpath(workspace, paths["validation_json"]),
                "validation_markdown_path": workspace_relpath(workspace, paths["validation_md"]),
                "packet_count": len(packets),
                "packet_truncated_count": sum(1 for packet in packets if packet.packet_truncated),
                "packet_input_token_estimate": sum(packet.token_estimate for packet in packets),
                "initial_validation": initial_validation.to_dict(),
                "final_validation": final_validation.to_dict(),
                "decision_counts": {
                    decision_type: sum(
                        1 for decision in qwen_decisions if decision.decision == decision_type
                    )
                    for decision_type in sorted(ALLOWED_DECISIONS)
                },
                "applied_edit_count": len(applied_edits),
                "blockers": topic_blockers,
            }
            write_json_atomic(paths["sidecar"], summary)
            paths["summary_markdown"].write_text(
                _render_markdown_summary(
                    topic_id=topic_id,
                    status=topic_status,
                    packets=packets,
                    decisions=qwen_decisions,
                    validation=final_validation,
                    blockers=topic_blockers,
                ),
                encoding="utf-8",
            )
            if enrich_source_packs:
                source_summary = {
                    "status": topic_status,
                    "owner_id": OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
                    "generated_at": generated_at,
                    "model": model,
                    "packet_path": summary["packet_path"],
                    "decision_path": summary["decision_path"],
                    "validation_json_path": summary["validation_json_path"],
                    "repaired_markdown_path": summary["repaired_markdown_path"],
                    "summary_markdown_path": summary["summary_markdown_path"],
                    "applied": applied,
                    "unresolved_blocker_count": len(topic_blockers),
                    "final_invalid_reference_count": final_validation.invalid_reference_count,
                    "final_bare_reference_warning_count": final_validation.bare_reference_warning_count,
                }
                if _enrich_source_pack(
                    workspace=workspace,
                    topic_id=topic_id,
                    summary=source_summary,
                ):
                    enriched_count += 1
            outputs.append(summary)
            all_blockers.extend(topic_blockers)
        except Exception as exc:  # noqa: BLE001 - receipt boundary.
            all_blockers.append(
                {
                    "code": str(exc).split(":", 1)[0] or "topic_reference_candidate_resolution_failed",
                    "topic_id": topic_id,
                    "reason": str(exc),
                    "retryable": True,
                }
            )

    usage = researcher.usage() if researcher is not None else {}
    receipt_path = _receipt_path(workspace, run_id)
    checks = {
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "applied_topic_count": applied_count,
        "packet_count": sum(int(item.get("packet_count", 0)) for item in outputs),
        "packet_truncated_count": sum(
            int(item.get("packet_truncated_count", 0)) for item in outputs
        ),
        "packet_input_token_estimate": sum(
            int(item.get("packet_input_token_estimate", 0)) for item in outputs
        ),
        "decision_count": sum(
            sum(int(count) for count in item.get("decision_counts", {}).values())
            for item in outputs
        ),
        "unresolved_blocker_count": len(all_blockers),
        "final_invalid_reference_count": sum(
            int(item.get("final_validation", {}).get("invalid_reference_count", 0))
            for item in outputs
        ),
        "final_unknown_reference_count": sum(
            int(item.get("final_validation", {}).get("unknown_reference_count", 0))
            for item in outputs
        ),
        "final_title_mismatch_count": sum(
            int(item.get("final_validation", {}).get("title_mismatch_count", 0))
            for item in outputs
        ),
        "final_bare_reference_warning_count": sum(
            int(item.get("final_validation", {}).get("bare_reference_warning_count", 0))
            for item in outputs
        ),
        "source_packs_enriched_count": enriched_count,
    }
    status = "blocked" if all_blockers else "succeeded"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
        "owner_id": OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION,
        "status": status,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "model": model,
        "dry_run": dry_run,
        "apply": apply,
        "topic_count": len(topic_ids),
        "completed_topic_count": len(outputs),
        "output_paths": [item["summary_markdown_path"] for item in outputs],
        "repaired_markdown_paths": [item["repaired_markdown_path"] for item in outputs],
        "sidecar_paths": [
            workspace_relpath(workspace, _output_paths(workspace, str(item["topic_id"]))["sidecar"])
            for item in outputs
        ],
        "packet_paths": [item["packet_path"] for item in outputs],
        "decision_paths": [item["decision_path"] for item in outputs],
        "validation_paths": [item["validation_json_path"] for item in outputs],
        "provider_calls_made": mode == "live",
        "live_provider_calls_made": mode == "live",
        "paid_provider_calls_made": False,
        "usage": usage,
        "local_provider_receipt_paths": list(usage.get("provider_receipt_paths") or []),
        "checks": checks,
        "blockers": all_blockers,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "receipt_path": workspace_relpath(workspace, receipt_path),
    }
    if apply and not dry_run:
        # Applied resolutions rewrote 07D's Markdown (see refresh_substep_outputs).
        refresh_substep_outputs(workspace, run_id, "topic-reconciliation")
    write_json_atomic(receipt_path, receipt)
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve blocked topic reconciliation references with bounded Qwen candidate selection."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument("--model", default=DEFAULT_QWEN36_MODEL)
    parser.add_argument("--topic", action="append", dest="topics")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no-enrich-source-packs", action="store_true")
    parser.add_argument("--max-candidates-per-reference", type=int, default=DEFAULT_MAX_CANDIDATES_PER_REFERENCE)
    parser.add_argument("--max-source-snippets-per-candidate", type=int, default=DEFAULT_MAX_SOURCE_SNIPPETS_PER_CANDIDATE)
    parser.add_argument("--max-snippet-chars", type=int, default=DEFAULT_MAX_SNIPPET_CHARS)
    parser.add_argument("--max-packet-input-tokens", type=int, default=DEFAULT_MAX_PACKET_INPUT_TOKENS)
    parser.add_argument("--max-references-per-qwen-call", type=int, default=DEFAULT_MAX_REFERENCES_PER_QWEN_CALL)
    parser.add_argument("--min-candidate-score", type=float, default=DEFAULT_MIN_CANDIDATE_SCORE)
    parser.add_argument("--ambiguity-margin", type=float, default=DEFAULT_AMBIGUITY_MARGIN)
    parser.add_argument("--summary", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = CandidateResolutionConfig(
        max_candidates_per_reference=args.max_candidates_per_reference,
        max_source_snippets_per_candidate=args.max_source_snippets_per_candidate,
        max_snippet_chars=args.max_snippet_chars,
        max_packet_input_tokens=args.max_packet_input_tokens,
        max_references_per_qwen_call=args.max_references_per_qwen_call,
        min_candidate_score=args.min_candidate_score,
        ambiguity_margin=args.ambiguity_margin,
    )
    try:
        receipt = resolve_topic_reference_candidates(
            workspace=args.workspace,
            run_id=args.run_id,
            mode=args.mode,
            model=args.model,
            topics=args.topics,
            config=config,
            dry_run=args.dry_run,
            apply=args.apply,
            enrich_source_packs=not args.no_enrich_source_packs,
        )
    except (TopicReferenceCandidateResolutionError, TopicReferenceValidationError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2, sort_keys=True))
        return 1
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "topic_count": receipt.get("topic_count"),
                    "completed_topic_count": receipt.get("completed_topic_count"),
                    "checks": receipt.get("checks"),
                    "receipt_path": receipt.get("receipt_path"),
                    "blockers": receipt.get("blockers"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if receipt.get("status") == "succeeded" else 1


__all__ = [
    "ALLOWED_DECISIONS",
    "CandidateFeature",
    "CandidateResolutionConfig",
    "QwenCandidateDecision",
    "ReferenceCandidate",
    "ReferenceCandidatePacket",
    "ReferenceWorkItem",
    "TopicReferenceCandidateResolutionError",
    "OWNER_TOPIC_REFERENCE_CANDIDATE_RESOLUTION",
    "build_candidate_packet",
    "build_candidate_packets",
    "build_qwen_candidate_prompt",
    "collect_source_snippets",
    "extract_reference_work_items",
    "parse_qwen_candidate_decisions",
    "resolve_topic_reference_candidates",
    "run_main",
]
