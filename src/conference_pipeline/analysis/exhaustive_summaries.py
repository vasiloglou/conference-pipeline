"""Summarize exhaustive full-read topic evidence.

This module is a provisional analysis owner. It consumes the completed
``llm-full-read`` by-topic outputs, rebuilds the underlying transcript-window
and paper-abstract text, and produces per-topic/category synthesis packs that
can be inspected before promotion into the formal Stage 7 narrative path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.costs.estimator import InputCeiling, tier_aware_input_ceiling
from conference_pipeline.extractions.backends import select_local_llm_backend
from conference_pipeline.extractions.llm_full_read import (
    DEFAULT_MIN_CHARS,
    DEFAULT_OVERLAP_SECONDS,
    DEFAULT_SEGMENT_SECONDS,
    FullReadInput,
    iter_units_for_scope,
)
from conference_pipeline.extractions.records import safe_artifact_filename
from conference_pipeline.indexing.token_policy import token_count
from conference_pipeline.model_governance import validate_selection
from conference_pipeline.model_governance.registry import default_model_for_provider, default_provider
from conference_pipeline.analysis.summary_model_choice import resolve_summary_model
from conference_pipeline.provider_receipts import (
    gemini_generate_content,
    openai_chat_completions_create,
    openrouter_chat_completions_create,
    provider_receipt_context,
    sanitize_for_receipt,
)
from conference_pipeline.receipt_lineage import (
    archive_obsolete_outputs,
    sanitize_receipt_for_write,
    stamp_substep_receipt,
    unbind_substep_receipt,
)
from conference_pipeline.workflow_runs import write_json_atomic
from conference_pipeline.model_governance.model_families import uses_openai_reasoning_params

OWNER_ID = "analysis.exhaustive_topic_summarization"
RECEIPT_TYPE = "conference_pipeline.exhaustive_topic_summaries"
SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "iclr2026-live-production"
DEFAULT_PROVIDER = default_provider(OWNER_ID)
# One governed default per provider. The local default must come from the
# task's registered local selection: it was once the task's *fallback* model,
# an OpenAI model, which --provider local would have sent to the local backend.
DEFAULT_GEMINI_MODEL = default_model_for_provider(OWNER_ID, "gemini")
DEFAULT_OPENAI_MODEL = default_model_for_provider(OWNER_ID, "openai")
# Kimi via OpenRouter: the benchmarked quality option (denser packs, ~5x the latency).
DEFAULT_OPENROUTER_MODEL = default_model_for_provider(OWNER_ID, "openrouter")
DEFAULT_LOCAL_MODEL = default_model_for_provider(OWNER_ID, "local")
# Provider upper bounds. When --max-input-tokens is not passed, the effective
# ceiling is further lowered below the selected model's high price tier (read
# from the pricing catalog) by resolve_max_input_tokens().
DEFAULT_GEMINI_MAX_INPUT_TOKENS = 400_000
DEFAULT_OPENAI_MAX_INPUT_TOKENS = 400_000
DEFAULT_OPENROUTER_MAX_INPUT_TOKENS = 400_000
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCAL_MAX_INPUT_TOKENS = 120_000
DEFAULT_MAX_OUTPUT_TOKENS = 16_384
VALID_CATEGORIES = ("transcripts", "papers")
KIND_BY_CATEGORY = {
    "transcripts": "transcript_segment",
    "papers": "paper_abstract",
}


@dataclass(frozen=True)
class TopicInfo:
    topic_id: str
    title: str


@dataclass(frozen=True)
class EvidenceItem:
    unit_id: str
    input_kind: str
    text: str
    text_tokens: int
    text_chars: int
    source_artifact_id: str
    source_artifact_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_summary: str = ""
    llm_reason: str = ""
    evidence_phrases: tuple[str, ...] = ()
    confidence: float | None = None


@dataclass(frozen=True)
class CorpusChunk:
    topic_id: str
    category: str
    chunk_index: int
    chunk_count: int
    unit_ids: tuple[str, ...]
    input_tokens: int
    input_chars: int
    content: str


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def workspace_relpath(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_topic_info(workspace: Path) -> dict[str, TopicInfo]:
    strategy = _load_json(workspace / "analysis" / "topic_strategy.json") or {}
    topics: dict[str, TopicInfo] = {}
    for raw in strategy.get("final_topics") or []:
        if not isinstance(raw, Mapping):
            continue
        topic_id = str(raw.get("topic_id") or raw.get("id") or "").strip()
        title = str(raw.get("title") or raw.get("name") or topic_id).strip()
        if topic_id:
            topics[topic_id] = TopicInfo(topic_id=topic_id, title=title or topic_id)
    return topics


def _unit_output_path(workspace: Path, run_id: str, unit_id: str) -> Path:
    return (
        workspace
        / "storage"
        / "extractions"
        / run_id
        / "llm-full-read"
        / "by-unit"
        / f"{safe_artifact_filename(unit_id)}.json"
    )


def build_unit_index(
    workspace: Path,
    *,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> dict[str, FullReadInput]:
    return {
        unit.unit_id: unit
        for unit in iter_units_for_scope(
            workspace,
            scope="all",
            segment_seconds=segment_seconds,
            overlap_seconds=overlap_seconds,
            min_chars=min_chars,
        )
    }


def _item_from_unit(
    *,
    workspace: Path,
    run_id: str,
    unit: FullReadInput,
) -> EvidenceItem:
    record = _load_json(_unit_output_path(workspace, run_id, unit.unit_id)) or {}
    phrases = tuple(
        str(item).strip()
        for item in (record.get("evidence_phrases") or [])
        if isinstance(item, str) and item.strip()
    )
    confidence_raw = record.get("confidence")
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    return EvidenceItem(
        unit_id=unit.unit_id,
        input_kind=unit.input_kind,
        text=unit.text,
        text_tokens=token_count(unit.text or ""),
        text_chars=len(unit.text or ""),
        source_artifact_id=unit.source_artifact_id,
        source_artifact_path=unit.source_artifact_path,
        metadata=dict(unit.metadata),
        llm_summary=str(record.get("summary") or ""),
        llm_reason=str(record.get("reason") or ""),
        evidence_phrases=phrases,
        confidence=confidence,
    )


def collect_topic_category_items(
    workspace: Path,
    *,
    run_id: str,
    topic_id: str,
    category: str,
    unit_index: Mapping[str, FullReadInput] | None = None,
) -> list[EvidenceItem]:
    if category not in KIND_BY_CATEGORY:
        raise ValueError(f"unsupported category: {category}")
    topic_path = (
        workspace
        / "storage"
        / "extractions"
        / run_id
        / "llm-full-read"
        / "by-topic"
        / f"{safe_artifact_filename(topic_id)}.json"
    )
    payload = _load_json(topic_path)
    if not payload:
        raise RuntimeError(f"llm_full_read_topic_missing:{topic_id}")
    index = unit_index or build_unit_index(workspace)
    expected_kind = KIND_BY_CATEGORY[category]
    items: list[EvidenceItem] = []
    for raw in payload.get("items") or []:
        if not isinstance(raw, Mapping):
            continue
        if raw.get("input_kind") != expected_kind:
            continue
        unit_id = str(raw.get("unit_id") or "").strip()
        unit = index.get(unit_id)
        if unit is None:
            continue
        items.append(_item_from_unit(workspace=workspace, run_id=run_id, unit=unit))
    return items


def _metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "workshop_id",
        "workshop_title",
        "talk_id",
        "talk_title",
        "talk_speaker",
        "timestamp_start",
        "timestamp_end",
        "source_url",
        "media_url",
        "local_path",
        "paper_id",
        "title",
        "forum_url",
        "pdf_url",
    )
    return {key: metadata.get(key) for key in keys if metadata.get(key) not in (None, "")}


def format_evidence_block(item: EvidenceItem, *, ordinal: int) -> str:
    metadata = _metadata_summary(item.metadata)
    header = {
        "ordinal": ordinal,
        "unit_id": item.unit_id,
        "input_kind": item.input_kind,
        "source_artifact_id": item.source_artifact_id,
        "source_artifact_path": item.source_artifact_path,
        "tokens": item.text_tokens,
        "chars": item.text_chars,
        "confidence": item.confidence,
        "metadata": metadata,
        "qwen_full_read_summary": item.llm_summary,
        "qwen_full_read_reason": item.llm_reason,
        "evidence_phrases": list(item.evidence_phrases),
    }
    return (
        f"### Evidence Unit {ordinal:05d}\n"
        f"{json.dumps(header, ensure_ascii=True, sort_keys=True)}\n\n"
        "TEXT:\n"
        f"{item.text.strip()}\n"
    )


def provider_default_max_input_tokens(provider: str) -> int:
    if provider == "gemini":
        return DEFAULT_GEMINI_MAX_INPUT_TOKENS
    if provider == "openai":
        return DEFAULT_OPENAI_MAX_INPUT_TOKENS
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_MAX_INPUT_TOKENS
    return DEFAULT_LOCAL_MAX_INPUT_TOKENS


def resolve_max_input_tokens(
    *,
    provider: str,
    model: str,
    explicit: int | None = None,
) -> InputCeiling:
    """Chunk ceiling for one summary call, tier-aware unless set explicitly.

    Prompts above a model's high-tier threshold bill every token at the higher
    rate (on ICML 2026, 49 of 76 calls crossed it and cost $81.87 of $84.78),
    so the default keeps chunks below the threshold the pricing catalog lists
    for the selected model.
    """

    return tier_aware_input_ceiling(
        model=model,
        provider=provider,
        provider_default=provider_default_max_input_tokens(provider),
        explicit=explicit,
    )


def chunk_evidence_items(
    *,
    topic_id: str,
    category: str,
    items: Iterable[EvidenceItem],
    max_input_tokens: int,
) -> list[CorpusChunk]:
    chunks: list[CorpusChunk] = []
    current_blocks: list[str] = []
    current_unit_ids: list[str] = []
    current_tokens = 0
    current_chars = 0
    for ordinal, item in enumerate(items, start=1):
        block = format_evidence_block(item, ordinal=ordinal)
        block_tokens = token_count(block)
        if current_blocks and current_tokens + block_tokens > max_input_tokens:
            chunks.append(
                CorpusChunk(
                    topic_id=topic_id,
                    category=category,
                    chunk_index=len(chunks) + 1,
                    chunk_count=0,
                    unit_ids=tuple(current_unit_ids),
                    input_tokens=current_tokens,
                    input_chars=current_chars,
                    content="\n\n".join(current_blocks),
                )
            )
            current_blocks = []
            current_unit_ids = []
            current_tokens = 0
            current_chars = 0
        current_blocks.append(block)
        current_unit_ids.append(item.unit_id)
        current_tokens += block_tokens
        current_chars += len(block)
    if current_blocks:
        chunks.append(
            CorpusChunk(
                topic_id=topic_id,
                category=category,
                chunk_index=len(chunks) + 1,
                chunk_count=0,
                unit_ids=tuple(current_unit_ids),
                input_tokens=current_tokens,
                input_chars=current_chars,
                content="\n\n".join(current_blocks),
            )
        )
    total = len(chunks)
    return [
        CorpusChunk(
            topic_id=chunk.topic_id,
            category=chunk.category,
            chunk_index=chunk.chunk_index,
            chunk_count=total,
            unit_ids=chunk.unit_ids,
            input_tokens=chunk.input_tokens,
            input_chars=chunk.input_chars,
            content=chunk.content,
        )
        for chunk in chunks
    ]


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
        return {"raw_text": text}
    return payload if isinstance(payload, dict) else {"raw_text": text}


def _summary_schema_hint() -> str:
    return """Return only JSON with this shape:
{
  "summary": "one dense paragraph",
  "key_points": ["bullet"],
  "narrative": "source-grounded narrative suitable for downstream analysis",
  "notable_evidence": [
    {
      "claim": "claim",
      "unit_ids": ["unit id"],
      "source_labels": ["talk/workshop/paper label"]
    }
  ],
  "tensions_or_contrasts": ["bullet"],
  "open_questions": ["bullet"],
  "analysis_feed": {
    "what_to_emphasize": ["bullet"],
    "representative_examples": ["bullet"],
    "risks_or_caveats": ["bullet"]
  }
}"""


def build_chunk_prompt(
    *,
    topic: TopicInfo,
    category: str,
    chunk: CorpusChunk,
    conference_label: str = "the conference",
) -> str:
    return (
        f"You are creating a source-grounded synthesis pack from exhaustive {conference_label} "
        "LLM full-read evidence. Read every evidence unit below. Ignore logistics and "
        "duplicated noise. Preserve concrete methods, claims, datasets, systems, "
        "benchmarks, scientific domains, and failure modes. Use unit_ids for evidence.\n\n"
        f"Topic: {topic.title}\n"
        f"Topic id: {topic.topic_id}\n"
        f"Category: {category}\n"
        f"Chunk: {chunk.chunk_index} / {chunk.chunk_count}\n"
        f"Evidence units in this chunk: {len(chunk.unit_ids)}\n\n"
        f"{_summary_schema_hint()}\n\n"
        "EVIDENCE:\n"
        f"{chunk.content}"
    )


def build_merge_prompt(
    *,
    topic: TopicInfo,
    category: str,
    chunk_payloads: list[dict[str, Any]],
) -> str:
    return (
        "Merge these chunk-level synthesis packs into one final per-topic/category "
        "analysis input. De-duplicate repeated claims, preserve the best evidence "
        "unit_ids, and make the narrative useful to a downstream deep-analysis writer.\n\n"
        f"Topic: {topic.title}\n"
        f"Topic id: {topic.topic_id}\n"
        f"Category: {category}\n\n"
        f"{_summary_schema_hint()}\n\n"
        "CHUNK SUMMARIES:\n"
        f"{json.dumps(chunk_payloads, ensure_ascii=True, sort_keys=True)}"
    )


def _provider_config(max_output_tokens: int, *, response_json: bool = True) -> Any:
    try:
        from google.genai import types  # type: ignore

        kwargs: dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1,
        }
        if response_json:
            kwargs["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kwargs)
    except Exception:
        config: dict[str, Any] = {"max_output_tokens": max_output_tokens, "temperature": 0.1}
        if response_json:
            config["response_mime_type"] = "application/json"
        return config


def _build_gemini_client() -> Any:
    try:
        from google import genai  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError("google-genai SDK is required for Gemini summary synthesis") from exc
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


def _build_openai_client() -> Any:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError("openai SDK is required for OpenAI summary synthesis") from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI summary synthesis")
    raw_timeout = os.environ.get("EXHAUSTIVE_SUMMARY_OPENAI_TIMEOUT_SECONDS", "600")
    try:
        timeout = max(1.0, float(raw_timeout))
    except ValueError:
        timeout = 600.0
    return OpenAI(api_key=api_key, timeout=timeout)


def _build_openrouter_client() -> Any:
    """An OpenAI-compatible client pointed at OpenRouter (``OPENROUTER_API_KEY``)."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency boundary.
        raise RuntimeError("openai SDK is required for OpenRouter summary synthesis") from exc
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for OpenRouter summary synthesis")
    raw_timeout = os.environ.get("EXHAUSTIVE_SUMMARY_OPENROUTER_TIMEOUT_SECONDS", "1800")
    try:
        timeout = max(1.0, float(raw_timeout))
    except ValueError:
        timeout = 1800.0
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, timeout=timeout)


def _openai_options(model: str, max_completion_tokens: int) -> dict[str, Any]:
    if uses_openai_reasoning_params(model):
        return {"max_completion_tokens": max_completion_tokens}
    return {"max_tokens": max_completion_tokens, "temperature": 0.1}


def _call_openai(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str, float | None]:
    result = openai_chat_completions_create(
        client,
        caller=__file__,
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        **_openai_options(model, max_output_tokens),
    )
    text = str(result.choices[0].message.content or "")
    payload = parse_json_object(text)
    return payload, str(result.receipt.receipt_path), result.receipt.estimated_cost_usd


def _call_openrouter(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str, float | None]:
    """Chat completion through OpenRouter, receipted under provider ``openrouter``.

    ``model`` is the registry's routed id (``openrouter/<vendor>/<model>``); the
    receipt wrapper strips the prefix for the API call and keeps it on the receipt
    so the pricing catalog's OpenRouter entry prices it.
    """
    messages = [
        {"role": "system", "content": "Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    result = openrouter_chat_completions_create(
        client,
        caller=__file__,
        model=model,
        messages=messages,
        max_tokens=max_output_tokens,
        response_format={"type": "json_object"},
    )
    text = str(result.choices[0].message.content or "")
    payload = parse_json_object(text)
    return payload, str(result.receipt.receipt_path), result.receipt.estimated_cost_usd


def _call_gemini(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str, float | None]:
    result = gemini_generate_content(
        client,
        caller=__file__,
        model=model,
        contents=prompt,
        config=_provider_config(max_output_tokens),
    )
    text = str(getattr(result, "text", "") or "")
    payload = parse_json_object(text)
    return payload, str(result.receipt.receipt_path), result.receipt.estimated_cost_usd


def _call_local(
    *,
    backend: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None, float | None]:
    response = backend.chat_completion(
        caller=__file__,
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_output_tokens,
    )
    text = response.choices[0].message.content if response.choices else "{}"
    receipt_path = getattr(getattr(response, "receipt", None), "receipt_path", None)
    return parse_json_object(text), str(receipt_path) if receipt_path else None, 0.0


def _summary_paths(workspace: Path, topic_id: str, category: str) -> tuple[Path, Path]:
    root = workspace / "analysis" / "exhaustive_summaries"
    stem = f"{safe_artifact_filename(topic_id)}__{category}"
    return root / f"{stem}.json", root / f"{stem}.md"


def pair_input_sha256(
    items: Sequence[EvidenceItem],
    *,
    max_input_tokens: int,
    topic_title: str = "",
    category: str = "",
) -> str:
    """Digest of everything a topic x category summary is computed from.

    A resumed run reuses a pair only when this matches, so a changed full-read
    record (new text, new summary) is re-summarized rather than re-stamped as
    current by the sub-step lineage.
    """

    # Hash exactly what the summarizer is shown for each item (text, metadata,
    # source path, full-read summary fields), so a metadata-only correction
    # is re-summarized too.
    payload = [
        hashlib.sha256(format_evidence_block(item, ordinal=index).encode("utf-8")).hexdigest()
        for index, item in enumerate(items, start=1)
    ]
    encoded = json.dumps(
        {"items": payload, "max_input_tokens": max_input_tokens, "topic_title": topic_title, "category": category},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reusable_summary_payload(
    *,
    json_path: Path,
    markdown_path: Path,
    run_id: str,
    provider: str,
    model: str,
    topic_id: str,
    category: str,
    input_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Return a completed pair only when its provenance matches this run.

    The post-14A supervisor is intentionally restartable.  A restart must not
    spend another paid synthesis pass when the pair outputs already exist, but
    it also must not silently reuse a pair from a different run/provider/model.
    """

    if not json_path.exists() or not markdown_path.exists():
        return None
    payload = _load_json(json_path)
    if not payload:
        return None
    if (
        payload.get("run_id") != run_id
        or payload.get("provider") != provider
        or payload.get("model") != model
        or payload.get("topic_id") != topic_id
        or payload.get("category") != category
    ):
        return None
    if input_sha256 is not None and payload.get("input_sha256") != input_sha256:
        # Its inputs moved (or it predates input digests): summarize again.
        return None
    if input_sha256 is not None and payload.get("markdown_sha256") != hashlib.sha256(
        markdown_path.read_bytes()
    ).hexdigest():
        # The Markdown is not the one this JSON was written with (an
        # interrupted re-run between the two writes).
        return None
    if not isinstance(payload.get("chunk_summaries"), list):
        return None
    if int(payload.get("chunk_count") or 0) != len(payload["chunk_summaries"]):
        return None
    if not markdown_path.read_text(encoding="utf-8", errors="replace").strip():
        return None
    return payload


def _render_markdown(payload: Mapping[str, Any]) -> str:
    title = str(payload.get("title") or payload.get("topic_id") or "Topic")
    category = str(payload.get("category") or "")
    lines = [f"# {title} - {category}", ""]
    for key, heading in (
        ("summary", "Summary"),
        ("narrative", "Narrative"),
    ):
        value = str(payload.get(key) or "").strip()
        if value:
            lines.extend([f"## {heading}", "", value, ""])
    for key, heading in (
        ("key_points", "Key Points"),
        ("tensions_or_contrasts", "Tensions Or Contrasts"),
        ("open_questions", "Open Questions"),
    ):
        values = payload.get(key) or []
        if isinstance(values, list) and values:
            lines.extend([f"## {heading}", ""])
            lines.extend(f"- {str(item)}" for item in values)
            lines.append("")
    analysis_feed = payload.get("analysis_feed")
    if isinstance(analysis_feed, Mapping):
        lines.extend(["## Analysis Feed", ""])
        for feed_key, values in analysis_feed.items():
            lines.append(f"### {str(feed_key).replace('_', ' ').title()}")
            lines.append("")
            if isinstance(values, list):
                lines.extend(f"- {str(item)}" for item in values)
            elif values:
                lines.append(str(values))
            lines.append("")
    evidence = payload.get("notable_evidence") or []
    if isinstance(evidence, list) and evidence:
        lines.extend(["## Notable Evidence", ""])
        for item in evidence:
            if isinstance(item, Mapping):
                claim = str(item.get("claim") or "").strip()
                ids = ", ".join(str(unit_id) for unit_id in (item.get("unit_ids") or []))
                lines.append(f"- {claim} (`{ids}`)")
            else:
                lines.append(f"- {str(item)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write_pair_outputs(
    *,
    workspace: Path,
    run_id: str,
    topic: TopicInfo,
    category: str,
    chunks: list[CorpusChunk],
    chunk_payloads: list[dict[str, Any]],
    final_payload: dict[str, Any],
    provider: str,
    model: str,
    mode: str,
    input_sha256: str | None = None,
) -> tuple[Path, Path]:
    json_path, md_path = _summary_paths(workspace, topic.topic_id, category)
    enriched = {
        "schema_version": SCHEMA_VERSION,
        "owner_id": OWNER_ID,
        "run_id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "topic_id": topic.topic_id,
        "title": topic.title,
        "category": category,
        "chunk_count": len(chunks),
        "unit_count": sum(len(chunk.unit_ids) for chunk in chunks),
        "input_tokens": sum(chunk.input_tokens for chunk in chunks),
        "input_chars": sum(chunk.input_chars for chunk in chunks),
        "input_sha256": input_sha256,
        "chunk_summaries": chunk_payloads,
        **final_payload,
        "generated_at": utc_now(),
    }
    # Markdown first: the JSON carries the resume key (input_sha256), so an
    # interrupted write never pairs a new key with an old report; the JSON also
    # records the Markdown's hash, checked before reuse.
    markdown = _render_markdown(enriched)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    enriched["markdown_sha256"] = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    write_json_atomic(json_path, sanitize_for_receipt(enriched))
    return json_path, md_path


def run_exhaustive_topic_summaries(
    *,
    workspace: Path,
    run_id: str = DEFAULT_RUN_ID,
    mode: str = "dry-run",
    provider: str | None = None,
    model: str | None = None,
    categories: Iterable[str] = VALID_CATEGORIES,
    topic_ids: Iterable[str] | None = None,
    max_input_tokens: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    provider_cost_approved: bool = False,
    resume: bool = False,
    client: Any | None = None,
    local_backend: Any | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    if mode not in {"dry-run", "live"}:
        raise ValueError("mode must be dry-run or live")
    # Stop point: the 07A model is the operator's call (cost vs. pack density),
    # never a default. Refuses, asking the operator, when neither --provider nor
    # workflow.exhaustive_summary_model in config.yaml chose one -- before any
    # receipt is touched or any backend is built.
    provider = provider.strip().lower() if provider else None
    if provider is not None and provider not in {"gemini", "openai", "openrouter", "local"}:
        raise ValueError("provider must be gemini, openai, openrouter, or local")
    model_choice = resolve_summary_model(workspace, provider=provider, model=model)
    provider = model_choice.provider
    model = model_choice.model
    # Fail closed on a provider/model pair the registry does not allow for this
    # task, before any receipt is touched or any backend is built.
    validate_selection(OWNER_ID, provider=provider, model=model)
    input_ceiling = resolve_max_input_tokens(
        provider=provider, model=model, explicit=max_input_tokens
    )
    max_input_tokens = input_ceiling.max_input_tokens
    selected_categories = tuple(category for category in categories if category in VALID_CATEGORIES)
    if not selected_categories:
        raise ValueError("at least one valid category is required")

    started = time.monotonic()
    unbind_substep_receipt(workspace, run_id, "exhaustive-topic-summaries")
    generated_at = utc_now()
    conference_label = conference_name(workspace, default="the conference")
    topics = _load_topic_info(workspace)
    all_topic_ids = set(topics)
    if topic_ids is not None:
        wanted = {str(topic_id) for topic_id in topic_ids}
        topics = {topic_id: topic for topic_id, topic in topics.items() if topic_id in wanted}
    if not topics:
        raise RuntimeError("topic_strategy_final_topics_missing")

    unit_index = build_unit_index(workspace)
    receipt_path = (
        workspace
        / "storage"
        / "workflow-runs"
        / run_id
        / "07A-exhaustive-topic-summaries.json"
    )
    plan_dir = workspace / "storage" / "analysis" / "exhaustive-summaries" / run_id
    plan_dir.mkdir(parents=True, exist_ok=True)

    provider_receipt_paths: list[str] = []
    estimated_cost_usd = 0.0
    pair_records: list[dict[str, Any]] = []
    pruned_outputs: list[str] = []
    json_outputs: list[str] = []
    md_outputs: list[str] = []
    reused_pair_count = 0
    live_calls_made = False

    paid_provider = provider in {"gemini", "openai", "openrouter"}
    if mode == "live" and paid_provider and not provider_cost_approved:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "receipt_type": RECEIPT_TYPE,
            "owner_id": OWNER_ID,
            "status": "blocked",
            "blocker_code": "provider_cost_approval_required",
            "generated_at": generated_at,
            "workspace": str(workspace),
            "run_id": run_id,
            "mode": mode,
            "provider": provider,
            "model": model,
            "summary_model_choice": model_choice.to_payload(),
            "secrets_copied": False,
            "provider_calls_made": False,
            "live_provider_calls_made": False,
            **input_ceiling.to_receipt(),
        }
        receipt["receipt_path"] = str(receipt_path)
        write_json_atomic(receipt_path, receipt)
        return receipt

    gemini_client = client if provider == "gemini" and mode == "live" else None
    openai_client = client if provider == "openai" and mode == "live" else None
    openrouter_client = client if provider == "openrouter" and mode == "live" else None
    local = local_backend if provider == "local" and mode == "live" else None
    if mode == "live" and provider == "gemini" and gemini_client is None:
        gemini_client = _build_gemini_client()
    if mode == "live" and provider == "openai" and openai_client is None:
        openai_client = _build_openai_client()
    if mode == "live" and provider == "openrouter" and openrouter_client is None:
        openrouter_client = _build_openrouter_client()
    if mode == "live" and provider == "local" and local is None:
        local = select_local_llm_backend(model)
        local.readiness()

    for topic in sorted(topics.values(), key=lambda item: item.topic_id):
        for category in selected_categories:
            items = collect_topic_category_items(
                workspace,
                run_id=run_id,
                topic_id=topic.topic_id,
                category=category,
                unit_index=unit_index,
            )
            chunks = chunk_evidence_items(
                topic_id=topic.topic_id,
                category=category,
                items=items,
                max_input_tokens=max_input_tokens,
            )
            json_path, md_path = _summary_paths(workspace, topic.topic_id, category)
            input_sha256 = pair_input_sha256(
                items, max_input_tokens=max_input_tokens, topic_title=topic.title, category=category
            )
            if mode == "live" and resume:
                reusable = _reusable_summary_payload(
                    json_path=json_path,
                    markdown_path=md_path,
                    run_id=run_id,
                    provider=provider,
                    model=model,
                    topic_id=topic.topic_id,
                    category=category,
                    input_sha256=input_sha256,
                )
                if reusable is not None:
                    json_outputs.append(workspace_relpath(workspace, json_path))
                    md_outputs.append(workspace_relpath(workspace, md_path))
                    pair_records.append(
                        {
                            "topic_id": topic.topic_id,
                            "title": topic.title,
                            "category": category,
                            "unit_count": len(items),
                            "chunk_count": int(reusable.get("chunk_count") or 0),
                            "input_tokens": int(reusable.get("input_tokens") or 0),
                            "input_chars": int(reusable.get("input_chars") or 0),
                            "max_input_tokens": max_input_tokens,
                            "status": "succeeded",
                            "output_json": json_outputs[-1],
                            "output_markdown": md_outputs[-1],
                            "reused": True,
                        }
                    )
                    reused_pair_count += 1
                    continue
            chunk_payloads: list[dict[str, Any]] = []
            if mode == "live" and chunks:
                live_calls_made = True
                print(
                    json.dumps(
                        {
                            "event": "summary_pair_start",
                            "topic_id": topic.topic_id,
                            "category": category,
                            "chunk_count": len(chunks),
                            "unit_count": len(items),
                            "input_tokens": sum(chunk.input_tokens for chunk in chunks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                with provider_receipt_context(
                    workspace_root=workspace,
                    run_id=run_id,
                    stage_id=OWNER_ID,
                    caller=__file__,
                ):
                    for chunk in chunks:
                        prompt = build_chunk_prompt(
                            topic=topic,
                            category=category,
                            chunk=chunk,
                            conference_label=conference_label,
                        )
                        if provider == "gemini":
                            payload, receipt, cost = _call_gemini(
                                client=gemini_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        elif provider == "openai":
                            payload, receipt, cost = _call_openai(
                                client=openai_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        elif provider == "openrouter":
                            payload, receipt, cost = _call_openrouter(
                                client=openrouter_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        else:
                            payload, receipt, cost = _call_local(
                                backend=local,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        payload.update(
                            {
                                "topic_id": topic.topic_id,
                                "category": category,
                                "chunk_index": chunk.chunk_index,
                                "chunk_count": chunk.chunk_count,
                                "unit_count": len(chunk.unit_ids),
                            }
                        )
                        chunk_payloads.append(payload)
                        if receipt:
                            provider_receipt_paths.append(receipt)
                        if isinstance(cost, (int, float)):
                            estimated_cost_usd += float(cost)
                        print(
                            json.dumps(
                                {
                                    "event": "summary_chunk_done",
                                    "topic_id": topic.topic_id,
                                    "category": category,
                                    "chunk_index": chunk.chunk_index,
                                    "chunk_count": chunk.chunk_count,
                                    "estimated_cost_usd": round(estimated_cost_usd, 6),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if len(chunk_payloads) > 1:
                        prompt = build_merge_prompt(
                            topic=topic,
                            category=category,
                            chunk_payloads=chunk_payloads,
                        )
                        if provider == "gemini":
                            final_payload, receipt, cost = _call_gemini(
                                client=gemini_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        elif provider == "openai":
                            final_payload, receipt, cost = _call_openai(
                                client=openai_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        elif provider == "openrouter":
                            final_payload, receipt, cost = _call_openrouter(
                                client=openrouter_client,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        else:
                            final_payload, receipt, cost = _call_local(
                                backend=local,
                                model=model,
                                prompt=prompt,
                                max_output_tokens=max_output_tokens,
                            )
                        if receipt:
                            provider_receipt_paths.append(receipt)
                        if isinstance(cost, (int, float)):
                            estimated_cost_usd += float(cost)
                    elif chunk_payloads:
                        final_payload = dict(chunk_payloads[0])
                    else:
                        final_payload = {
                            "summary": "",
                            "key_points": [],
                            "narrative": "",
                            "notable_evidence": [],
                            "open_questions": [],
                        }
                json_path, md_path = _write_pair_outputs(
                    workspace=workspace,
                    run_id=run_id,
                    topic=topic,
                    category=category,
                    chunks=chunks,
                    chunk_payloads=chunk_payloads,
                    final_payload=final_payload,
                    provider=provider,
                    model=model,
                    mode=mode,
                    input_sha256=input_sha256,
                )
                json_outputs.append(workspace_relpath(workspace, json_path))
                md_outputs.append(workspace_relpath(workspace, md_path))
            if mode == "live" and not chunks:
                # The category has no evidence now: a summary left from an
                # earlier corpus would still be read downstream and certified
                # current by this receipt's stamp, so move it out.
                pruned_outputs.extend(
                    archive_obsolete_outputs(workspace, [json_path, md_path], label="exhaustive-topic-summaries")
                )
            pair_records.append(
                {
                    "topic_id": topic.topic_id,
                    "title": topic.title,
                    "category": category,
                    "unit_count": len(items),
                    "chunk_count": len(chunks),
                    "input_tokens": sum(chunk.input_tokens for chunk in chunks),
                    "input_chars": sum(chunk.input_chars for chunk in chunks),
                    "max_input_tokens": max_input_tokens,
                    "status": "planned" if mode == "dry-run" else ("succeeded" if chunks else "empty"),
                    "output_json": json_outputs[-1] if mode == "live" and chunks else None,
                    "output_markdown": md_outputs[-1] if mode == "live" and chunks else None,
                }
            )

    plan_path = plan_dir / "summary_plan.json"
    index_path = workspace / "analysis" / "exhaustive_summaries" / "index.json"
    plan_payload = {
        "schema_version": SCHEMA_VERSION,
        "owner_id": OWNER_ID,
        "generated_at": generated_at,
        "run_id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "summary_model_choice": model_choice.to_payload(),
        **input_ceiling.to_receipt(),
        "pair_count": len(pair_records),
        "pairs": pair_records,
    }
    write_json_atomic(plan_path, plan_payload)
    if mode == "live":
        write_json_atomic(
            index_path,
            {
                "schema_version": SCHEMA_VERSION,
                "owner_id": OWNER_ID,
                "generated_at": utc_now(),
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "summary_count": len(json_outputs),
                "summaries": json_outputs,
                "markdown_summaries": md_outputs,
            },
        )

    receipt_status = "planned" if mode == "dry-run" else "succeeded"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "owner_id": OWNER_ID,
        "status": receipt_status,
        "generated_at": generated_at,
        "ended_at": utc_now(),
        "workspace": str(workspace),
        "run_id": run_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "provider_calls_made": live_calls_made,
        "live_provider_calls_made": live_calls_made and paid_provider,
        "paid_provider_calls_made": live_calls_made and paid_provider,
        "reused_pair_count": reused_pair_count,
        "resume": resume,
        "provider_cost_approved": provider_cost_approved,
        "secrets_copied": False,
        **input_ceiling.to_receipt(),
        "max_output_tokens": max_output_tokens,
        "topic_count": len(topics),
        "category_count": len(selected_categories),
        "pair_count": len(pair_records),
        "chunk_count": sum(int(record["chunk_count"]) for record in pair_records),
        "unit_assignment_count": sum(int(record["unit_count"]) for record in pair_records),
        "input_tokens": sum(int(record["input_tokens"]) for record in pair_records),
        "input_chars": sum(int(record["input_chars"]) for record in pair_records),
        "estimated_cost_usd": estimated_cost_usd if mode == "live" else None,
        "provider_receipt_paths": provider_receipt_paths,
        "plan_path": workspace_relpath(workspace, plan_path),
        "index_path": workspace_relpath(workspace, index_path) if mode == "live" else None,
        "summary_json_paths": json_outputs,
        "summary_markdown_paths": md_outputs,
        "pruned_outputs": pruned_outputs,
        "elapsed_seconds": time.monotonic() - started,
        "receipt_path": str(receipt_path),
    }
    stamp_substep_receipt(
        receipt,
        workspace=workspace,
        run_id=run_id,
        substep_id="exhaustive-topic-summaries",
        complete=set(topics) == all_topic_ids and set(selected_categories) == set(VALID_CATEGORIES),
    )
    write_json_atomic(receipt_path, sanitize_receipt_for_write(receipt))
    return receipt


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exhaustive topic/category summaries.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=("dry-run", "live"), default="dry-run")
    parser.add_argument(
        "--provider",
        choices=("gemini", "openai", "openrouter", "local"),
        default=None,
        help=(
            "The operator's model choice for this stage (stop point). Without it the "
            "conference config.yaml must set workflow.exhaustive_summary_model; "
            "otherwise the stage refuses and asks."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--topic", action="append", dest="topics", default=None)
    parser.add_argument("--category", action="append", choices=VALID_CATEGORIES, default=None)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=(
            "Per-chunk input token ceiling. Default: the provider default, lowered "
            "below the selected model's high-tier price threshold from the pricing catalog."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--provider-cost-approved", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    receipt = run_exhaustive_topic_summaries(
        workspace=args.workspace,
        run_id=args.run_id,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        categories=args.category or VALID_CATEGORIES,
        topic_ids=args.topics,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        provider_cost_approved=args.provider_cost_approved,
        resume=args.resume,
    )
    if args.summary:
        print(
            json.dumps(
                {
                    "status": receipt.get("status"),
                    "provider": receipt.get("provider"),
                    "model": receipt.get("model"),
                    "pair_count": receipt.get("pair_count"),
                    "chunk_count": receipt.get("chunk_count"),
                    "unit_assignment_count": receipt.get("unit_assignment_count"),
                    "input_tokens": receipt.get("input_tokens"),
                    "receipt_path": receipt.get("receipt_path"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"succeeded", "planned"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
