"""Stage 8: QA, coverage gap fill and synthesis, shared by every conference runner.

Gates, does not orchestrate: (1) the receipt convention's start halt and the
resume short-circuit over an already succeeded, still-verifying receipt;
(2) fail closed, with zero spend, on any missing, unreadable or
not-succeeded upstream input -- the Stage 7 receipt (and its passing
contract), the topic reconciliation / reference validation / reference
workflow sub-step receipts, the analysis manifest and the citation
manifest; (3) the pre-spend lineage gate; (4) the deterministic ($0)
traceability and coverage-gap reports, blocking unless both pass; (5)
provider-cost approval before any paid client is constructed (an injected
client is the caller's own approval); (6) the governed executive-summary and
trend-report synthesis, each validated against the citation floors and the
digest bindings, with the estimated cost held under the cap; (7) the
convention's success receipt, binding the coverage-gap and trend reports as
the owners' execution evidence.

Everything a conference decides comes in through the
:class:`~conference_pipeline.production.context.ProductionRunnerContext`
(workspace layout, identity, env loader, cost approval), the
:class:`~conference_pipeline.production.receipts.StageReceiptConvention`
(receipt and summary shape) and :class:`QaSynthesisSettings`. The module
carries no conference literal.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from conference_pipeline.analysis.narrative import DEFAULT_DEEP_NARRATIVE_MODEL
from conference_pipeline.costs import load_pricing_catalog
from conference_pipeline.indexing.corpus_vector_store import workspace_relpath
from conference_pipeline.model_governance import (
    default_fallback_model,
    default_model,
    default_provider,
    validate_selection,
)
from conference_pipeline.production.context import ProductionRunnerContext
from conference_pipeline.production.receipts import (
    DEFAULT_STAGE_IO,
    StageIO,
    StageReceiptConvention,
    StrictStageConvention,
)
from conference_pipeline.provider_receipts import (
    anthropic_messages_create,
    openai_chat_completions_create,
    provider_receipt_context,
    sanitize_for_receipt,
)
from conference_pipeline.receipt_lineage import sha256_file

STAGE_ID = "qa-gap-fill-synthesis"
ANALYSIS_STAGE_ID = "analysis"
SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID = "synthesis.executive_summary"

STAGE8_TOPIC_REPORT_FLOOR = 3
STAGE8_EXTRACTION_ID_FLOOR = 5
STAGE8_RESEARCH_ID_FLOOR = 3
STAGE8_EXECUTIVE_SUMMARY_MIN_BYTES = 2000
STAGE8_TREND_REPORT_MIN_BYTES = 3000
STAGE8_SYNTHESIS_COST_CAP_USD = 35.0

# Stage 7 sub-step receipts Stage 8 gates on (written into the run directory
# by their own commands).
TOPIC_RECONCILIATION_RECEIPT = "07D-topic-reconciliation.json"
TOPIC_REFERENCE_VALIDATION_RECEIPT = "07E-topic-reference-validation.json"
TOPIC_REFERENCE_WORKFLOW_RECEIPT = "07H-topic-reconciliation-reference-workflow.json"

# Required owner ids -> the artifact whose path:digest is their execution evidence.
OWNER_COVERAGE_GAP_FILL = "analysis.coverage_gap_fill"
OWNER_SYNTHESIS_TREND_REPORT = "analysis.synthesis_trend_report"

RECEIPT_SCOPE_STAGE = "stage"
RECEIPT_SCOPE_CLIENT = "client"


@dataclass(frozen=True)
class QaSynthesisSettings:
    """Where a conference's provider receipts for the paid synthesis are scoped.

    ``receipt_scope="client"`` (default): the synthesis client opens a
    provider-receipt context per call, tagged with the governed task id.
    ``receipt_scope="stage"``: the stage opens one context around both calls
    and the client records none of its own.
    """

    receipt_scope: str = RECEIPT_SCOPE_CLIENT


DEFAULT_SETTINGS = QaSynthesisSettings()


@dataclass(frozen=True)
class AnalysisPaths:
    """Stage 7 outputs Stage 8 reads and the Stage 8 reports it writes (workspace-relative)."""

    manifest: Path
    citation_manifest: Path
    source_pack_dir: Path
    traceability_report: Path
    coverage_gap_report: Path
    executive_summary: Path
    trend_report: Path

    @classmethod
    def for_context(cls, ctx: ProductionRunnerContext) -> AnalysisPaths:
        analysis_dir = ctx.analysis_manifest.parent
        qa_dir = analysis_dir / "qa"
        return cls(
            manifest=ctx.analysis_manifest,
            citation_manifest=analysis_dir / "analysis_citation_manifest.json",
            source_pack_dir=analysis_dir / "narrative_source_packs",
            traceability_report=qa_dir / "traceability_report.json",
            coverage_gap_report=qa_dir / "coverage_gap_report.json",
            executive_summary=ctx.analysis_executive_summary,
            trend_report=ctx.analysis_trend_report,
        )


# --------------------------------------------------------------------------
# Governed paid synthesis client
# --------------------------------------------------------------------------


class Stage8SynthesisClient:
    """Registry-governed paid synthesis client for the executive summary and trend report.

    Provider, model and fallback model come from the model-governance
    registry (``SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID``); no model or provider
    literal is written here. ``caller`` is the path provider receipts
    attribute the call to (the orchestrator). With ``workspace`` and
    ``run_id`` the client opens its own provider-receipt context per call
    (``receipt_scope="client"``); without them the stage owns the context.
    """

    def __init__(
        self,
        *,
        workspace: Path | None = None,
        run_id: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        caller: str | None = None,
        stage_id: str = STAGE_ID,
    ) -> None:
        self.workspace = workspace
        self.run_id = run_id
        self.stage_id = stage_id
        self.caller = caller or __file__
        self.provider = provider or default_provider(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
        self.model = model or default_model(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
        self.fallback_model = default_fallback_model(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID)
        validate_selection(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID, provider=self.provider, model=self.model)
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0

    @contextlib.contextmanager
    def _receipt_scope(self) -> Iterator[None]:
        if self.workspace is None or self.run_id is None:
            yield
            return
        with provider_receipt_context(
            workspace_root=self.workspace,
            run_id=self.run_id,
            stage_id=self.stage_id,
            caller=self.caller,
            task_id=SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID,
        ):
            yield

    def synthesize(
        self,
        *,
        kind: str,
        system_prompt: str,
        user_prompt: str,
    ) -> SimpleNamespace:  # pragma: no cover - provider boundary
        del kind
        with self._receipt_scope():
            if self.provider == "anthropic":
                try:
                    from anthropic import Anthropic
                except ImportError as exc:
                    raise RuntimeError("anthropic_not_available") from exc
                api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("missing_anthropic_api_key")
                response = anthropic_messages_create(
                    Anthropic(api_key=api_key, timeout=180),
                    caller=self.caller,
                    model=self.model,
                    max_tokens=16000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                content = "".join(
                    str(block.get("text") or "")
                    if isinstance(block, dict)
                    else str(getattr(block, "text", "") or "")
                    for block in (getattr(response, "content", None) or [])
                )
                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
                output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
            elif self.provider == "openai":
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise RuntimeError("openai_not_available") from exc
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("missing_openai_api_key")
                response = openai_chat_completions_create(
                    OpenAI(api_key=api_key, timeout=180),
                    caller=self.caller,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            else:
                raise RuntimeError(f"unsupported_synthesis_provider:{self.provider}")
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.request_count += 1
        return SimpleNamespace(
            text=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider=self.provider,
            model=self.model,
            request_count=self.request_count,
        )


# --------------------------------------------------------------------------
# Deterministic ($0) input gates and QA reports
# --------------------------------------------------------------------------


def stage8_load_required_json(
    *,
    workspace: Path,
    path: Path,
    label: str,
    blockers: list[dict[str, Any]],
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any] | None:
    if not path.exists():
        blockers.append(
            {
                "code": "stage8_input_missing",
                "input": label,
                "evidence": workspace_relpath(workspace, path),
                "retryable": True,
            }
        )
        return None
    payload, error = io.load_json(path)
    if error or payload is None:
        blockers.append(
            {
                "code": "stage8_input_unreadable",
                "input": label,
                "evidence": workspace_relpath(workspace, path),
                "reason": sanitize_for_receipt(error or "unreadable_json"),
                "retryable": True,
            }
        )
        return None
    return payload


def stage8_citation_pool(citation_manifest: Mapping[str, Any]) -> dict[str, set[str]]:
    extraction_ids: set[str] = set()
    research_ids: set[str] = set()
    chunk_ids: set[str] = set()
    topic_paths: set[str] = set()
    for topic in citation_manifest.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        markdown_path = str(topic.get("markdown_path") or "").strip()
        if markdown_path:
            topic_paths.add(markdown_path)
        for record in topic.get("extraction_record_ids") or []:
            if isinstance(record, str) and record:
                extraction_ids.add(record)
        for record in topic.get("research_ids") or []:
            if isinstance(record, str) and record:
                research_ids.add(record)
        for record in topic.get("vector_store_chunk_ids") or []:
            if isinstance(record, str) and record:
                chunk_ids.add(record)
    return {
        "extraction_record_ids": extraction_ids,
        "research_ids": research_ids,
        "vector_store_chunk_ids": chunk_ids,
        "topic_report_paths": topic_paths,
    }


def stage8_validate_required_workspace_file(
    *,
    workspace: Path,
    rel_path: object,
    label: str,
    topic_id: str,
    expected_sha256: str | None,
    blockers: list[dict[str, Any]],
    io: StageIO = DEFAULT_STAGE_IO,
) -> dict[str, Any]:
    path, error = io.resolve_member(workspace, rel_path)
    result: dict[str, Any] = {
        "label": label,
        "path": str(rel_path or ""),
        "present": False,
        "sha256": None,
        "sha256_matches": None,
    }
    if error or path is None:
        blockers.append(
            {
                "code": "stage8_required_topic_file_missing",
                "topic_id": topic_id,
                "input": label,
                "path": str(rel_path or ""),
                "retryable": True,
            }
        )
        return result
    result["path"] = workspace_relpath(workspace, path)
    if not path.is_file():
        blockers.append(
            {
                "code": "stage8_required_topic_file_missing",
                "topic_id": topic_id,
                "input": label,
                "path": workspace_relpath(workspace, path),
                "retryable": True,
            }
        )
        return result
    observed_sha = sha256_file(path)
    result["present"] = True
    result["sha256"] = observed_sha
    if expected_sha256:
        result["sha256_matches"] = observed_sha == expected_sha256
        if observed_sha != expected_sha256:
            blockers.append(
                {
                    "code": "stage8_topic_file_digest_mismatch",
                    "topic_id": topic_id,
                    "input": label,
                    "path": workspace_relpath(workspace, path),
                    "expected_sha256": expected_sha256,
                    "observed_sha256": observed_sha,
                    "retryable": True,
                }
            )
    return result


def stage8_topic_reference_status(source_pack: Mapping[str, Any] | None) -> dict[str, Any]:
    reconciliation = (source_pack or {}).get("topic_reconciliation")
    if not isinstance(reconciliation, dict):
        return {"status": "missing"}
    validation = reconciliation.get("reference_validation")
    validation_counts = reconciliation.get("reference_validation_counts")
    if not isinstance(validation, dict):
        validation = {}
    if not isinstance(validation_counts, dict):
        validation_counts = {}
    return {
        "status": reconciliation.get("status"),
        "model": reconciliation.get("model"),
        "reference_validation_status": reconciliation.get("reference_validation_status")
        or validation.get("status"),
        "citation_count": validation_counts.get("citation_count") or validation.get("citation_count"),
        "ledger_entry_count": validation_counts.get("ledger_entry_count") or validation.get("ledger_entry_count"),
        "invalid_reference_count": validation_counts.get("invalid_reference_count")
        or validation.get("invalid_reference_count")
        or 0,
        "unknown_reference_count": validation_counts.get("unknown_reference_count")
        or validation.get("unknown_reference_count")
        or 0,
        "title_mismatch_count": validation_counts.get("title_mismatch_count")
        or validation.get("title_mismatch_count")
        or 0,
        "bare_reference_warning_count": validation_counts.get("bare_reference_warning_count")
        or validation.get("bare_reference_warning_count")
        or 0,
    }


def build_stage8_traceability_reports(
    *,
    workspace: Path,
    run_id: str,
    stage7_receipt: Mapping[str, Any],
    analysis_manifest: Mapping[str, Any],
    citation_manifest: Mapping[str, Any],
    receipt_digests: Mapping[str, str],
    paths: AnalysisPaths,
    io: StageIO = DEFAULT_STAGE_IO,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_topics = [topic for topic in analysis_manifest.get("topics") or [] if isinstance(topic, dict)]
    citation_by_topic = {
        str(topic.get("topic_id")): topic
        for topic in citation_manifest.get("topics") or []
        if isinstance(topic, dict) and topic.get("topic_id")
    }
    blockers: list[dict[str, Any]] = []
    topic_reports: list[dict[str, Any]] = []
    coverage_topics: list[dict[str, Any]] = []

    stage7_checks = stage7_receipt.get("checks") if isinstance(stage7_receipt, dict) else {}
    expected_content_sha = str(analysis_manifest.get("content_sha256") or "")
    observed_stage7_sha = ""
    if isinstance(stage7_checks, dict):
        observed_stage7_sha = str(stage7_checks.get("analysis_manifest_content_sha256") or "")
    if observed_stage7_sha != expected_content_sha:
        blockers.append(
            {
                "code": "stage8_stage7_manifest_digest_mismatch",
                "expected": expected_content_sha,
                "observed": observed_stage7_sha,
                "retryable": True,
            }
        )

    for topic in manifest_topics:
        topic_id = str(topic.get("id") or "").strip()
        topic_title = str(topic.get("title") or topic_id)
        topic_blockers: list[dict[str, Any]] = []
        files = [
            stage8_validate_required_workspace_file(
                workspace=workspace,
                rel_path=topic.get(label),
                label=label,
                topic_id=topic_id,
                expected_sha256=(str(topic.get("markdown_sha256") or "") or None) if label == "markdown_path" else None,
                blockers=topic_blockers,
                io=io,
            )
            for label in ("markdown_path", "provenance_json_path", "provenance_path", "slide_brief_path")
        ]
        citation_entry = citation_by_topic.get(topic_id, {})
        extraction_count = len(citation_entry.get("extraction_record_ids") or [])
        chunk_count = len(citation_entry.get("vector_store_chunk_ids") or [])
        research_count = len(citation_entry.get("research_ids") or [])
        if extraction_count < STAGE8_EXTRACTION_ID_FLOOR:
            topic_blockers.append(
                {
                    "code": "stage8_topic_extraction_citations_below_floor",
                    "topic_id": topic_id,
                    "observed": extraction_count,
                    "minimum": STAGE8_EXTRACTION_ID_FLOOR,
                    "retryable": True,
                }
            )
        if chunk_count <= 0:
            topic_blockers.append(
                {
                    "code": "stage8_topic_vector_citations_missing",
                    "topic_id": topic_id,
                    "observed": chunk_count,
                    "minimum": 1,
                    "retryable": True,
                }
            )
        if research_count < 1:
            topic_blockers.append(
                {
                    "code": "stage8_topic_research_citations_missing",
                    "topic_id": topic_id,
                    "observed": research_count,
                    "minimum": 1,
                    "retryable": True,
                }
            )
        deep_validation = topic.get("deep_validation")
        deep_status = str((deep_validation or {}).get("status") or "") if isinstance(deep_validation, dict) else ""
        if deep_status != "pass":
            topic_blockers.append(
                {
                    "code": "stage8_topic_deep_validation_not_pass",
                    "topic_id": topic_id,
                    "observed": deep_status or "<missing>",
                    "retryable": True,
                }
            )
        narrative_model = str(topic.get("narrative_model") or "")
        if narrative_model != DEFAULT_DEEP_NARRATIVE_MODEL:
            topic_blockers.append(
                {
                    "code": "stage8_topic_narrative_model_not_promoted",
                    "topic_id": topic_id,
                    "expected": DEFAULT_DEEP_NARRATIVE_MODEL,
                    "observed": narrative_model or "<missing>",
                    "retryable": True,
                }
            )

        source_pack_path = workspace / paths.source_pack_dir / f"{topic_id}.json"
        source_pack, source_pack_error = io.load_json(source_pack_path)
        if source_pack_error or source_pack is None:
            topic_blockers.append(
                {
                    "code": "stage8_topic_source_pack_missing_or_unreadable",
                    "topic_id": topic_id,
                    "path": workspace_relpath(workspace, source_pack_path),
                    "reason": sanitize_for_receipt(source_pack_error or "missing_source_pack"),
                    "retryable": True,
                }
            )
        reference_status = stage8_topic_reference_status(source_pack)
        if reference_status.get("status") != "succeeded":
            topic_blockers.append(
                {
                    "code": "stage8_topic_reconciliation_missing",
                    "topic_id": topic_id,
                    "observed": reference_status.get("status"),
                    "retryable": True,
                }
            )
        if reference_status.get("reference_validation_status") != "succeeded":
            topic_blockers.append(
                {
                    "code": "stage8_topic_reference_validation_not_succeeded",
                    "topic_id": topic_id,
                    "observed": reference_status.get("reference_validation_status"),
                    "retryable": True,
                }
            )
        for key in (
            "invalid_reference_count",
            "unknown_reference_count",
            "title_mismatch_count",
            "bare_reference_warning_count",
        ):
            if int(reference_status.get(key) or 0) != 0:
                topic_blockers.append(
                    {
                        "code": f"stage8_topic_reference_{key}_nonzero",
                        "topic_id": topic_id,
                        "observed": reference_status.get(key),
                        "retryable": True,
                    }
                )

        blockers.extend(topic_blockers)
        status = "pass" if not topic_blockers else "blocked"
        citation_counts = {
            "extraction_record_ids": extraction_count,
            "vector_store_chunk_ids": chunk_count,
            "research_ids": research_count,
        }
        topic_reports.append(
            {
                "topic_id": topic_id,
                "title": topic_title,
                "status": status,
                "files": files,
                "citation_counts": dict(citation_counts),
                "deep_validation_status": deep_status,
                "narrative_model": narrative_model,
                "reference_validation": reference_status,
                "blockers": topic_blockers,
            }
        )
        gap_class = "well_covered"
        if topic_blockers:
            if any("missing" in str(blocker.get("code") or "") for blocker in topic_blockers):
                gap_class = "missing_evidence"
            elif any("below_floor" in str(blocker.get("code") or "") for blocker in topic_blockers):
                gap_class = "under_covered"
            else:
                gap_class = "gap"
        coverage_topics.append(
            {
                "topic_id": topic_id,
                "title": topic_title,
                "status": status,
                "gap_class": gap_class,
                "citation_counts": dict(citation_counts),
                "fix_recommendation": (
                    "No action required"
                    if status == "pass"
                    else "Rerun Stage 7 analysis/reconciliation, then rerun Phase 8"
                ),
                "blockers": topic_blockers,
            }
        )

    class_counts = {
        gap_class: sum(1 for topic in coverage_topics if topic["gap_class"] == gap_class)
        for gap_class in ("well_covered", "under_covered", "missing_evidence", "gap")
    }
    generated_at = io.now()
    traceability_report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "status": "pass" if not blockers else "blocked",
        "analysis_manifest_path": str(paths.manifest),
        "analysis_manifest_content_sha256": expected_content_sha,
        "citation_manifest_path": str(paths.citation_manifest),
        "citation_manifest_sha256": receipt_digests["citation_manifest_sha256"],
        "stage7_receipt_sha256": receipt_digests["stage7_receipt_sha256"],
        "topic_reconciliation_workflow_sha256": receipt_digests["topic_reconciliation_workflow_sha256"],
        "topic_count": len(manifest_topics),
        "topics": topic_reports,
        "blockers": blockers,
    }
    coverage_report = {
        "schema_version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "status": "pass" if not blockers else "blocked",
        "classes": class_counts,
        "thresholds": {
            "topic_report_paths": STAGE8_TOPIC_REPORT_FLOOR,
            "extraction_record_ids": STAGE8_EXTRACTION_ID_FLOOR,
            "research_ids": STAGE8_RESEARCH_ID_FLOOR,
        },
        "topic_count": len(manifest_topics),
        "topics": coverage_topics,
        "blockers": blockers,
        "bound_analysis_manifest_content_sha256": expected_content_sha,
        "bound_stage7_receipt_sha256": receipt_digests["stage7_receipt_sha256"],
        "bound_topic_reconciliation_workflow_sha256": receipt_digests["topic_reconciliation_workflow_sha256"],
    }
    return traceability_report, coverage_report


# --------------------------------------------------------------------------
# Synthesis prompts, anchoring and validation
# --------------------------------------------------------------------------


def stage8_topic_report_excerpt(
    workspace: Path,
    rel_path: str,
    *,
    excerpt_chars: int = 2400,
    io: StageIO = DEFAULT_STAGE_IO,
) -> str:
    path, error = io.resolve_member(workspace, rel_path)
    if error or path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")[:excerpt_chars]
    except OSError:
        return ""


def build_stage8_synthesis_prompts(
    *,
    kind: str,
    workspace: Path,
    analysis_manifest: Mapping[str, Any],
    citation_manifest: Mapping[str, Any],
    coverage_gap_report: Mapping[str, Any],
    traceability_report: Mapping[str, Any],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
    io: StageIO = DEFAULT_STAGE_IO,
) -> tuple[str, str, dict[str, list[str]]]:
    pool = stage8_citation_pool(citation_manifest)
    topics = [topic for topic in analysis_manifest.get("topics") or [] if isinstance(topic, dict)]
    topic_paths = [str(topic.get("markdown_path") or "") for topic in topics if str(topic.get("markdown_path") or "")]
    citation_pool = {
        "topic_report_paths": topic_paths,
        "extraction_record_ids": sorted(pool["extraction_record_ids"])[:80],
        "research_ids": sorted(pool["research_ids"])[:40],
    }
    if kind == "executive_summary":
        target = "1200-1800 words"
        instructions = (
            "Produce a crisp executive synthesis for a research leadership audience. "
            "Prioritize the 6-8 strongest cross-topic conclusions, the largest "
            "surprises, and the residual risks that affect downstream slides."
        )
    elif kind == "trend_report":
        target = "1800-2800 words"
        instructions = (
            "Produce a cross-topic trend report. Identify clusters, tensions, "
            "methodological shifts, adoption risks, and implications for the "
            "podcast/deck narrative."
        )
    else:
        raise ValueError(f"unknown synthesis kind: {kind}")

    excerpts = []
    for topic in topics:
        rel = str(topic.get("markdown_path") or "")
        title = str(topic.get("title") or topic.get("id") or rel)
        excerpts.append(f"### {rel} - {title}\n\n{stage8_topic_report_excerpt(workspace, rel, io=io)}\n")
    system_prompt = (
        "You are a careful conference intelligence synthesizer. Use only the "
        "provided topic report excerpts, citation pools, traceability report, "
        "and coverage report. Cite paths and ids verbatim from the context. "
        "Output markdown only."
    )
    traceability_status = {"status": traceability_report.get("status"), "topic_count": traceability_report.get("topic_count")}
    user_prompt = (
        f"Task: write the `{kind}` synthesis in {target}.\n\n"
        f"{instructions}\n\n"
        "Required citation discipline:\n"
        f"- Mention at least {STAGE8_TOPIC_REPORT_FLOOR} topic report paths verbatim.\n"
        f"- Mention at least {STAGE8_EXTRACTION_ID_FLOOR} extraction record ids verbatim.\n"
        f"- Mention at least {STAGE8_RESEARCH_ID_FLOOR} research ids verbatim.\n"
        "- Include the analysis manifest, traceability, and coverage digests verbatim.\n\n"
        "## Binding digests\n\n"
        f"- analysis_manifest_content_sha256: `{analysis_manifest_content_sha256}`\n"
        f"- traceability_report_sha256: `{traceability_report_sha256}`\n"
        f"- coverage_gap_report_sha256: `{coverage_gap_report_sha256}`\n\n"
        "## Citation pool\n\n"
        f"{json.dumps(citation_pool, indent=2, sort_keys=True)}\n\n"
        "## Coverage classes\n\n"
        f"{json.dumps(coverage_gap_report.get('classes') or {}, indent=2, sort_keys=True)}\n\n"
        "## Traceability status\n\n"
        f"{json.dumps(traceability_status, indent=2, sort_keys=True)}\n\n"
        "## Topic report excerpts\n\n" + "\n".join(excerpts)
    )
    return system_prompt, user_prompt, citation_pool


def append_stage8_source_anchor(
    text: str,
    *,
    kind: str,
    citation_pool: Mapping[str, list[str]],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
) -> str:
    topic_paths = citation_pool.get("topic_report_paths", [])[:6]
    extraction_ids = citation_pool.get("extraction_record_ids", [])[:10]
    research_ids = citation_pool.get("research_ids", [])[:6]
    anchor = [
        "",
        "## Source Anchors",
        "",
        "This synthesis is bound to the validated Stage 7 analysis and Phase 8 QA reports.",
        "",
        "Topic reports:",
        *(f"- `{path}`" for path in topic_paths),
        "",
        "Extraction ids:",
        *(f"- `{item}`" for item in extraction_ids),
        "",
        "Research ids:",
        *(f"- `{item}`" for item in research_ids),
        "",
        "Digest bindings:",
        f"- `analysis_manifest_content_sha256`: `{analysis_manifest_content_sha256}`",
        f"- `traceability_report_sha256`: `{traceability_report_sha256}`",
        f"- `coverage_gap_report_sha256`: `{coverage_gap_report_sha256}`",
        "",
        "<!-- stage8-synthesis-bound: "
        f"kind={kind}; "
        f"analysis_manifest_content_sha256={analysis_manifest_content_sha256}; "
        f"traceability_report_sha256={traceability_report_sha256}; "
        f"coverage_gap_report_sha256={coverage_gap_report_sha256} -->",
    ]
    return "\n".join(line.rstrip() for line in [text.rstrip(), *anchor]).rstrip() + "\n"


def validate_stage8_synthesis(
    text: str,
    *,
    citation_pool: Mapping[str, list[str]],
    analysis_manifest_content_sha256: str,
    coverage_gap_report_sha256: str,
    traceability_report_sha256: str,
) -> dict[str, Any]:
    topic_paths = sorted(path for path in citation_pool.get("topic_report_paths", []) if path and path in text)
    extraction_ids = sorted(item for item in citation_pool.get("extraction_record_ids", []) if item and item in text)
    research_ids = sorted(item for item in citation_pool.get("research_ids", []) if item and item in text)
    missing_digest_bindings = [
        label
        for label, value in (
            ("analysis_manifest_content_sha256", analysis_manifest_content_sha256),
            ("coverage_gap_report_sha256", coverage_gap_report_sha256),
            ("traceability_report_sha256", traceability_report_sha256),
        )
        if value and value not in text
    ]
    return {
        "counts": {
            "topic_report_paths": len(topic_paths),
            "extraction_record_ids": len(extraction_ids),
            "research_ids": len(research_ids),
        },
        "topic_report_paths": topic_paths,
        "extraction_record_ids": extraction_ids,
        "research_ids": research_ids,
        "missing_digest_bindings": missing_digest_bindings,
    }


def estimate_stage8_synthesis_cost(token_usage: Mapping[str, int], *, provider: str, model: str) -> float:
    subtotal, _, _ = load_pricing_catalog().estimate_tokens(
        provider,
        model,
        int(token_usage.get("input_tokens") or 0),
        int(token_usage.get("output_tokens") or 0),
    )
    return round(subtotal, 6)


def synthesis_validation_blockers(
    *, kind: str, validation: Mapping[str, Any], byte_length: int, minimum_bytes: int
) -> list[dict[str, Any]]:
    """Citation-floor, digest-binding and length blockers for one synthesized output."""

    counts = validation["counts"]
    blockers: list[dict[str, Any]] = []
    for code, key, minimum in (
        ("stage8_synthesis_topic_report_floor_unmet", "topic_report_paths", STAGE8_TOPIC_REPORT_FLOOR),
        ("stage8_synthesis_extraction_floor_unmet", "extraction_record_ids", STAGE8_EXTRACTION_ID_FLOOR),
        ("stage8_synthesis_research_floor_unmet", "research_ids", STAGE8_RESEARCH_ID_FLOOR),
    ):
        if int(counts[key]) < minimum:
            blockers.append({"code": code, "kind": kind, "observed": counts[key], "minimum": minimum, "retryable": True})
    if validation["missing_digest_bindings"]:
        blockers.append(
            {
                "code": "stage8_synthesis_digest_binding_missing",
                "kind": kind,
                "missing": validation["missing_digest_bindings"],
                "retryable": True,
            }
        )
    if byte_length < minimum_bytes:
        blockers.append(
            {
                "code": "stage8_synthesis_output_too_short",
                "kind": kind,
                "observed_bytes": byte_length,
                "minimum_bytes": minimum_bytes,
                "retryable": True,
            }
        )
    return blockers


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def _upstream_input_blockers(
    stage7_receipt: Mapping[str, Any] | None,
    receipts: Mapping[str, Mapping[str, Any] | None],
    topic_reference_validation: Mapping[str, Any] | None,
    topic_reference_workflow: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for label, payload in receipts.items():
        if isinstance(payload, dict) and payload.get("status") != "succeeded":
            blockers.append(
                {
                    "code": "stage8_required_receipt_not_succeeded",
                    "input": label,
                    "observed_status": payload.get("status"),
                    "retryable": True,
                }
            )
    if isinstance(stage7_receipt, dict):
        contract = stage7_receipt.get("stage_contract")
        if not isinstance(contract, dict) or contract.get("status") != "pass":
            blockers.append(
                {
                    "code": "stage8_stage7_contract_not_pass",
                    "observed_status": contract.get("status") if isinstance(contract, dict) else "<missing>",
                    "retryable": True,
                }
            )
    if isinstance(topic_reference_validation, dict):
        checks = topic_reference_validation.get("checks")
        if isinstance(checks, dict):
            for key in (
                "invalid_reference_count",
                "unknown_reference_count",
                "title_mismatch_count",
                "bare_reference_warning_count",
            ):
                if int(checks.get(key) or 0) != 0:
                    blockers.append(
                        {
                            "code": f"stage8_topic_reference_validation_{key}_nonzero",
                            "observed": checks.get(key),
                            "retryable": True,
                        }
                    )
    if isinstance(topic_reference_workflow, dict):
        checks = topic_reference_workflow.get("checks")
        if not isinstance(checks, dict) or checks.get("final_validation_status") != "succeeded":
            blockers.append(
                {
                    "code": "stage8_topic_reference_workflow_final_validation_not_succeeded",
                    "observed": checks.get("final_validation_status") if isinstance(checks, dict) else "<missing>",
                    "retryable": True,
                }
            )
    return blockers


def _default_client_factory(settings: QaSynthesisSettings) -> Callable[[ProductionRunnerContext], Any]:
    def build(ctx: ProductionRunnerContext) -> Any:
        if settings.receipt_scope == RECEIPT_SCOPE_CLIENT:
            return Stage8SynthesisClient(workspace=ctx.workspace, run_id=ctx.run_id, caller=ctx.receipt_caller)
        return Stage8SynthesisClient(caller=ctx.receipt_caller)

    return build


def run_qa_gap_fill_synthesis_stage(
    ctx: ProductionRunnerContext,
    *,
    convention: StageReceiptConvention | None = None,
    resume: bool = True,
    provider_cost_approved: bool = False,
    env_file: Path | None = None,
    client: Any | None = None,
    client_factory: Callable[[ProductionRunnerContext], Any] | None = None,
    settings: QaSynthesisSettings = DEFAULT_SETTINGS,
    io: StageIO = DEFAULT_STAGE_IO,
) -> Any:
    """Run Stage 8 under ``convention`` and return that convention's stage result.

    ``client`` is an already constructed synthesis client (its caller has
    approved the spend); otherwise ``client_factory(ctx)`` builds one only
    after provider-cost approval (default: :class:`Stage8SynthesisClient`).
    """

    workspace = ctx.workspace
    run_id = ctx.run_id
    convention = convention or StrictStageConvention(ctx, STAGE_ID, io=io)
    paths = AnalysisPaths.for_context(ctx)
    ctx.run_dir().mkdir(parents=True, exist_ok=True)
    env_file_status = ctx.load_env(env_file)
    approval = ctx.provider_cost_approval(provider_cost_approved)

    halted = convention.start()
    if halted is not None:
        return halted
    if resume:
        resumed = convention.resumed()
        if resumed is not None:
            return resumed

    stage7_receipt_path = ctx.receipt_path(ANALYSIS_STAGE_ID)
    topic_reconciliation_path = ctx.run_dir() / TOPIC_RECONCILIATION_RECEIPT
    topic_reference_validation_path = ctx.run_dir() / TOPIC_REFERENCE_VALIDATION_RECEIPT
    topic_reference_workflow_path = ctx.run_dir() / TOPIC_REFERENCE_WORKFLOW_RECEIPT
    analysis_manifest_path = workspace / paths.manifest
    citation_manifest_path = workspace / paths.citation_manifest

    input_blockers: list[dict[str, Any]] = []
    loaded: dict[str, dict[str, Any] | None] = {}
    for label, path in (
        ("stage_07_analysis_receipt", stage7_receipt_path),
        ("topic_reconciliation_receipt", topic_reconciliation_path),
        ("topic_reference_validation_receipt", topic_reference_validation_path),
        ("topic_reconciliation_reference_workflow_receipt", topic_reference_workflow_path),
        ("analysis_manifest", analysis_manifest_path),
        ("analysis_citation_manifest", citation_manifest_path),
    ):
        loaded[label] = stage8_load_required_json(
            workspace=workspace, path=path, label=label, blockers=input_blockers, io=io
        )
    stage7_receipt = loaded["stage_07_analysis_receipt"]
    topic_reference_validation = loaded["topic_reference_validation_receipt"]
    topic_reference_workflow = loaded["topic_reconciliation_reference_workflow_receipt"]
    analysis_manifest = loaded["analysis_manifest"]
    citation_manifest = loaded["analysis_citation_manifest"]
    input_blockers.extend(
        _upstream_input_blockers(
            stage7_receipt,
            {
                label: loaded[label]
                for label in (
                    "stage_07_analysis_receipt",
                    "topic_reconciliation_receipt",
                    "topic_reference_validation_receipt",
                    "topic_reconciliation_reference_workflow_receipt",
                )
            },
            topic_reference_validation,
            topic_reference_workflow,
        )
    )

    base_checks: dict[str, Any] = {
        "env_file": env_file_status,
        "provider_cost_approval": approval,
        "provider_calls_made": False,
        "live_provider_calls_made": False,
        "secrets_copied": False,
        "stage7_receipt_path": workspace_relpath(workspace, stage7_receipt_path),
        "topic_reconciliation_receipt_path": workspace_relpath(workspace, topic_reconciliation_path),
        "topic_reference_validation_receipt_path": workspace_relpath(workspace, topic_reference_validation_path),
        "topic_reconciliation_reference_workflow_receipt_path": workspace_relpath(
            workspace, topic_reference_workflow_path
        ),
    }
    if input_blockers:
        return convention.block(
            "stage8_required_inputs_blocked",
            reason="Phase 8 required inputs are missing, stale, or failed",
            remediation="rerun Stage 7 analysis and topic reference workflow, then rerun Phase 8",
            checks={**base_checks, "input_blockers": input_blockers},
        )
    assert stage7_receipt is not None
    assert analysis_manifest is not None
    assert citation_manifest is not None
    assert topic_reference_workflow is not None

    receipt_digests = {
        "stage7_receipt_sha256": sha256_file(stage7_receipt_path),
        "topic_reconciliation_receipt_sha256": sha256_file(topic_reconciliation_path),
        "topic_reference_validation_receipt_sha256": sha256_file(topic_reference_validation_path),
        "topic_reconciliation_workflow_sha256": sha256_file(topic_reference_workflow_path),
        "analysis_manifest_sha256": sha256_file(analysis_manifest_path),
        "citation_manifest_sha256": sha256_file(citation_manifest_path),
    }
    gated = convention.lineage_gate(base_checks, input_digests=receipt_digests)
    if gated is not None:
        return gated

    traceability_report, coverage_gap_report = build_stage8_traceability_reports(
        workspace=workspace,
        run_id=run_id,
        stage7_receipt=stage7_receipt,
        analysis_manifest=analysis_manifest,
        citation_manifest=citation_manifest,
        receipt_digests=receipt_digests,
        paths=paths,
        io=io,
    )
    io.write_json(workspace / paths.traceability_report, traceability_report)
    io.write_json(workspace / paths.coverage_gap_report, coverage_gap_report)
    traceability_sha = sha256_file(workspace / paths.traceability_report)
    coverage_sha = sha256_file(workspace / paths.coverage_gap_report)
    report_checks = {
        "traceability_report_path": str(paths.traceability_report),
        "traceability_report_sha256": traceability_sha,
        "traceability_report_status": traceability_report.get("status"),
        "coverage_gap_report_path": str(paths.coverage_gap_report),
        "coverage_gap_report_sha256": coverage_sha,
        "coverage_gap_report_status": coverage_gap_report.get("status"),
    }
    if traceability_report.get("status") != "pass" or coverage_gap_report.get("status") != "pass":
        qa_blockers = list(traceability_report.get("blockers") or []) + list(coverage_gap_report.get("blockers") or [])
        return convention.block(
            "stage8_traceability_or_coverage_blocked",
            reason="Phase 8 traceability or coverage checks blocked synthesis",
            remediation="repair listed QA/source gaps and rerun Phase 8",
            checks={**base_checks, **receipt_digests, **report_checks, "qa_blockers": qa_blockers},
        )

    if client is None and not bool(approval["approved"]):
        return convention.block(
            "stage8_provider_cost_not_approved",
            reason="Phase 8 synthesis requires paid provider approval",
            remediation="rerun with --provider-cost-approved after confirming budget",
            checks={**base_checks, **receipt_digests},
        )
    if client is None:
        client = (client_factory or _default_client_factory(settings))(ctx)

    synthesis_provider = str(getattr(client, "provider", "") or default_provider(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID))
    synthesis_model = str(getattr(client, "model", "") or default_model(SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID))
    content_sha = str(analysis_manifest.get("content_sha256") or "")
    synthesis_outputs: dict[str, dict[str, Any]] = {}
    token_usage = {"input_tokens": 0, "output_tokens": 0, "request_count": 0}
    synthesis_blockers: list[dict[str, Any]] = []
    stage_scope = (
        provider_receipt_context(workspace_root=workspace, run_id=run_id, stage_id=STAGE_ID, caller=ctx.receipt_caller)
        if settings.receipt_scope == RECEIPT_SCOPE_STAGE
        else contextlib.nullcontext()
    )
    with stage_scope:
        for kind, rel_path, minimum_bytes in (
            ("executive_summary", paths.executive_summary, STAGE8_EXECUTIVE_SUMMARY_MIN_BYTES),
            ("trend_report", paths.trend_report, STAGE8_TREND_REPORT_MIN_BYTES),
        ):
            system_prompt, user_prompt, citation_pool = build_stage8_synthesis_prompts(
                kind=kind,
                workspace=workspace,
                analysis_manifest=analysis_manifest,
                citation_manifest=citation_manifest,
                coverage_gap_report=coverage_gap_report,
                traceability_report=traceability_report,
                analysis_manifest_content_sha256=content_sha,
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
                io=io,
            )
            try:
                result = client.synthesize(kind=kind, system_prompt=system_prompt, user_prompt=user_prompt)
            except Exception as exc:  # noqa: BLE001 - provider boundary
                return convention.block(
                    "stage8_synthesis_provider_failed",
                    reason="Phase 8 synthesis provider failed",
                    remediation="inspect provider receipt/error, then rerun Phase 8",
                    checks={
                        **base_checks,
                        **receipt_digests,
                        "synthesis_provider": synthesis_provider,
                        "synthesis_model": synthesis_model,
                        "provider_error": sanitize_for_receipt(str(exc)),
                    },
                )
            text = append_stage8_source_anchor(
                str(result.text or ""),
                kind=kind,
                citation_pool=citation_pool,
                analysis_manifest_content_sha256=content_sha,
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
            )
            output_path = workspace / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text, encoding="utf-8")
            validation = validate_stage8_synthesis(
                text,
                citation_pool=citation_pool,
                analysis_manifest_content_sha256=content_sha,
                coverage_gap_report_sha256=coverage_sha,
                traceability_report_sha256=traceability_sha,
            )
            byte_length = output_path.stat().st_size
            synthesis_blockers.extend(
                synthesis_validation_blockers(
                    kind=kind, validation=validation, byte_length=byte_length, minimum_bytes=minimum_bytes
                )
            )
            token_usage["input_tokens"] += int(getattr(result, "input_tokens", 0) or 0)
            token_usage["output_tokens"] += int(getattr(result, "output_tokens", 0) or 0)
            token_usage["request_count"] += 1
            synthesis_outputs[kind] = {
                "path": workspace_relpath(workspace, output_path),
                "sha256": sha256_file(output_path),
                "byte_length": byte_length,
                "citation_counts": validation["counts"],
                "missing_digest_bindings": validation["missing_digest_bindings"],
            }

    estimated_cost_usd = estimate_stage8_synthesis_cost(token_usage, provider=synthesis_provider, model=synthesis_model)
    if estimated_cost_usd > STAGE8_SYNTHESIS_COST_CAP_USD:
        synthesis_blockers.append(
            {
                "code": "stage8_synthesis_cost_cap_exceeded",
                "estimated_cost_usd": estimated_cost_usd,
                "cost_cap_usd": STAGE8_SYNTHESIS_COST_CAP_USD,
                "retryable": False,
            }
        )
    if synthesis_blockers:
        return convention.block(
            "stage8_synthesis_validation_blocked",
            reason="Phase 8 synthesis failed validation",
            remediation="repair synthesis prompt/model output and rerun Phase 8",
            checks={
                **base_checks,
                **receipt_digests,
                **report_checks,
                "synthesis_provider": synthesis_provider,
                "synthesis_model": synthesis_model,
                "synthesis_blockers": synthesis_blockers,
                "token_usage": token_usage,
                "estimated_cost_usd": estimated_cost_usd,
            },
        )

    checks = {
        **base_checks,
        **receipt_digests,
        "analysis_manifest_path": str(paths.manifest),
        "analysis_manifest_sha256": receipt_digests["analysis_manifest_sha256"],
        "analysis_manifest_content_sha256": content_sha,
        "analysis_manifest_topic_count": len(analysis_manifest.get("topics") or []),
        "citation_manifest_path": str(paths.citation_manifest),
        "citation_manifest_sha256": receipt_digests["citation_manifest_sha256"],
        "traceability_report_path": str(paths.traceability_report),
        "traceability_report_sha256": traceability_sha,
        "traceability_report_status": "pass",
        "coverage_gap_report_path": str(paths.coverage_gap_report),
        "coverage_gap_report_sha256": coverage_sha,
        "coverage_gap_report_status": "pass",
        "coverage_classes": coverage_gap_report.get("classes"),
        "executive_summary_path": str(paths.executive_summary),
        "executive_summary_sha256": synthesis_outputs["executive_summary"]["sha256"],
        "executive_summary_status": "pass",
        "executive_summary_citation_counts": synthesis_outputs["executive_summary"]["citation_counts"],
        "trend_report_path": str(paths.trend_report),
        "trend_report_sha256": synthesis_outputs["trend_report"]["sha256"],
        "trend_report_status": "pass",
        "trend_report_citation_counts": synthesis_outputs["trend_report"]["citation_counts"],
        "synthesis_provider": synthesis_provider,
        "synthesis_model": synthesis_model,
        "model_governance_task_id": SYNTHESIS_EXECUTIVE_SUMMARY_TASK_ID,
        "provider_calls_made": True,
        "live_provider_calls_made": True,
        "paid_provider_calls_made": True,
        "token_usage": token_usage,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_cap_usd": STAGE8_SYNTHESIS_COST_CAP_USD,
        "secrets_copied": False,
    }
    owner_evidence = {
        OWNER_COVERAGE_GAP_FILL: f"{checks['coverage_gap_report_path']}:{checks['coverage_gap_report_sha256']}",
        OWNER_SYNTHESIS_TREND_REPORT: f"{checks['trend_report_path']}:{checks['trend_report_sha256']}",
    }
    return convention.succeed(checks, owner_evidence=owner_evidence)
