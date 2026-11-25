"""Presentation Blueprint generation owner implementation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from conference_pipeline.conference_identity import conference_name
from conference_pipeline.costs.estimator import estimate_text_model_cost, load_task_specs
from conference_pipeline.model_governance import load_registry, validate_selection
from conference_pipeline.model_governance.llm_client import (
    complete,
    credential_names,
    has_credentials,
    resolve_selection,
)
from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_SCHEMA_VERSION,
    validate_blueprint_file,
)
from conference_pipeline.provider_receipts import (
    provider_receipt_context,
    provider_receipt_summary,
)
from conference_pipeline.workflow_runs import write_json_atomic


BLUEPRINT_GENERATION_TASK_ID = "presentation.blueprint_generation"
BLUEPRINT_COST_TASK_ID = "presentation_blueprint_generation"
BLUEPRINT_PROVIDER_COST_APPROVAL_REQUIRED = "BLUEPRINT_PROVIDER_COST_APPROVAL_REQUIRED"
BLUEPRINT_PROVIDER_BUDGET_CAP_EXCEEDED = "BLUEPRINT_PROVIDER_BUDGET_CAP_EXCEEDED"
BLUEPRINT_PROVIDER_CREDENTIALS_MISSING = "BLUEPRINT_PROVIDER_CREDENTIALS_MISSING"
BLUEPRINT_LIVE_GENERATION_FAILED = "BLUEPRINT_LIVE_GENERATION_FAILED"
BLUEPRINT_LEGACY_PRESENTATIONS_LANE = "BLUEPRINT_LEGACY_PRESENTATIONS_LANE"
GLOBAL_PROVIDER_BUDGET_CAP_USD = 800.0
SOURCE_CONTENT_CHAR_LIMIT = 120_000

_LEGACY_PRESENTATIONS_MESSAGE = (
    "presentations/ is a legacy NeurIPS lane; use decks/<presentation>/... for reusable outputs"
)


class BlueprintGenerationError(RuntimeError):
    """Fail-closed blueprint generation error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class BlueprintGenerationConfig:
    workspace: Path
    presentation_id: str
    presentation_type: str
    topic: str | None = None
    run_id: str | None = None
    analysis_md: Path | None = None
    slide_brief: Path | None = None
    source_pack: Path | None = None
    matched_papers: Path | None = None
    source_metadata: Path | None = None
    source_corpus_inventory: Path | None = None
    speaker_author_work_ledger: Path | None = None
    source_repair_findings: Path | None = None
    research_cluster_plan: Path | None = None
    final_report_url: str | None = None
    provenance_inputs: tuple[Path, ...] = ()
    output_json: Path | None = None
    output_md: Path | None = None
    receipt_path: Path | None = None
    mode: str = "dry-run"
    provider_cost_approved: bool = False
    model: str | None = None
    provider: str | None = None
    summary: str | None = None
    mock_fixture: Path | None = None


@dataclass(frozen=True)
class BlueprintGenerationPaths:
    output_json: Path
    output_md: Path
    receipt_path: Path


def generate_presentation_blueprint(config: BlueprintGenerationConfig) -> dict[str, Any]:
    """Generate or plan a presentation blueprint and return the summary payload."""

    workspace = config.workspace.expanduser().resolve()
    paths = resolve_generation_paths(config)
    if config.mode == "source-ledger":
        provider = "deterministic"
        selected_model = config.model or "source-ledger-blueprint-v1"
        selection_provider = provider
        selection_model = selected_model
    else:
        provider, selected_model = resolve_selection(
            BLUEPRINT_GENERATION_TASK_ID, config.provider, config.model
        )
        selection = validate_selection(BLUEPRINT_GENERATION_TASK_ID, provider=provider, model=selected_model)
        selection_provider = selection.provider
        selection_model = selection.model
    input_records = _input_records(config, workspace)
    prompt = _build_prompt(config, input_records)
    prompt_sha256 = _sha256_text(prompt)
    input_sha256 = _sha256_json(input_records)
    estimated_cost = (
        0.0
        if config.mode == "source-ledger"
        else estimate_blueprint_generation_cost(selection_model, provider=selection_provider)
    )
    budget_guardrail = _budget_guardrail(
        workspace=workspace,
        provider_cost_approved=config.provider_cost_approved,
        estimated_cost_usd=estimated_cost,
    )

    if config.mode == "dry-run":
        return {
            "dry_run": True,
            "mode": "dry-run",
            "presentation_id": config.presentation_id,
            "presentation_type": config.presentation_type,
            "topic": config.topic,
            "model_governance_task_id": BLUEPRINT_GENERATION_TASK_ID,
            "provider": selection_provider,
            "model_id": selection_model,
            "prompt_sha256": prompt_sha256,
            "input_sha256": input_sha256,
            "provider_requirements": ["task:presentation.blueprint_generation"],
            "budget_guardrail": budget_guardrail,
            "would_write": [str(paths.output_json), str(paths.output_md), str(paths.receipt_path)],
        }

    if config.mode == "source-ledger":
        blueprint_payload = _build_source_ledger_blueprint_payload(
            config=config,
            workspace=workspace,
            input_records=input_records,
        )
        write_json_atomic(paths.output_json, blueprint_payload)
        _write_blueprint_markdown(paths.output_md, blueprint_payload)
        validation = validate_blueprint_file(paths.output_json)
        receipt = _receipt_payload(
            config=config,
            paths=paths,
            mode="source-ledger",
            production_evidence=True,
            provider=selection_provider,
            model_id=selection_model,
            prompt_sha256=prompt_sha256,
            input_sha256=input_sha256,
            input_records=input_records,
            provider_receipt_paths=[],
            estimated_cost_usd=0.0,
            budget_guardrail=budget_guardrail,
            validation=validation.to_dict(),
            extra={
                "generation_method": "deterministic_source_ledger",
                "input_hashes": blueprint_payload["input_hashes"],
            },
        )
        write_json_atomic(paths.receipt_path, receipt)
        return _summary(receipt)

    if config.mode == "mock":
        if config.mock_fixture is None:
            raise BlueprintGenerationError("BLUEPRINT_MOCK_FIXTURE_REQUIRED", "--mock-fixture is required in mock mode")
        fixture = _resolve_inside(workspace, config.mock_fixture, label="--mock-fixture")
        blueprint_payload = json.loads(fixture.read_text(encoding="utf-8"))
        write_json_atomic(paths.output_json, blueprint_payload)
        _write_blueprint_markdown(paths.output_md, blueprint_payload)
        validation = validate_blueprint_file(paths.output_json)
        receipt = _receipt_payload(
            config=config,
            paths=paths,
            mode="mock",
            production_evidence=False,
            provider=selection_provider,
            model_id=selection_model,
            prompt_sha256=prompt_sha256,
            input_sha256=input_sha256,
            input_records=input_records,
            provider_receipt_paths=[],
            estimated_cost_usd=0.0,
            budget_guardrail=budget_guardrail,
            validation=validation.to_dict(),
            extra={"mock_fixture": _relative_path(fixture, workspace)},
        )
        write_json_atomic(paths.receipt_path, receipt)
        return _summary(receipt)

    if config.mode != "live":
        raise BlueprintGenerationError("BLUEPRINT_MODE_INVALID", f"unknown blueprint generation mode: {config.mode}")

    _assert_live_guardrails(budget_guardrail)
    blueprint_payload, provider_receipt_paths, actual_cost = _live_generate_blueprint(
        config=config,
        workspace=workspace,
        provider=selection_provider,
        model_id=selection_model,
        prompt=prompt,
    )
    blueprint_payload = _normalize_provider_blueprint_payload(
        blueprint_payload,
        config=config,
        input_records=input_records,
    )
    write_json_atomic(paths.output_json, blueprint_payload)
    _write_blueprint_markdown(paths.output_md, blueprint_payload)
    validation = validate_blueprint_file(paths.output_json)
    receipt = _receipt_payload(
        config=config,
        paths=paths,
        mode="live",
        production_evidence=True,
        provider=selection_provider,
        model_id=selection_model,
        prompt_sha256=prompt_sha256,
        input_sha256=input_sha256,
        input_records=input_records,
        provider_receipt_paths=provider_receipt_paths,
        estimated_cost_usd=actual_cost if actual_cost is not None else estimated_cost,
        budget_guardrail=budget_guardrail,
        validation=validation.to_dict(),
    )
    write_json_atomic(paths.receipt_path, receipt)
    return _summary(receipt)


def resolve_generation_paths(config: BlueprintGenerationConfig) -> BlueprintGenerationPaths:
    workspace = config.workspace.expanduser().resolve()
    deck_dir = workspace / "decks" / config.presentation_id
    output_json = _resolve_inside(
        workspace,
        config.output_json or deck_dir / "presentation_blueprint.json",
        label="--output-json",
    )
    output_md = _resolve_inside(
        workspace,
        config.output_md or output_json.with_suffix(".md"),
        label="--output-md",
    )
    receipt_path = _resolve_inside(
        workspace,
        config.receipt_path or output_json.parent / "presentation_blueprint_receipt.json",
        label="--receipt-path",
    )
    for label, path in (
        ("--output-json", output_json),
        ("--output-md", output_md),
        ("--receipt-path", receipt_path),
    ):
        _reject_legacy_presentations_lane(path, workspace, label=label)
    return BlueprintGenerationPaths(output_json=output_json, output_md=output_md, receipt_path=receipt_path)


def estimate_blueprint_generation_cost(model: str, *, provider: str | None = None) -> float:
    specs = load_task_specs()
    input_tokens = 100_000
    output_tokens = 25_000
    for task in specs.get("tasks", []):
        if not isinstance(task, dict) or task.get("task_id") != BLUEPRINT_COST_TASK_ID:
            continue
        input_tokens = int(task.get("input_tokens_per_unit") or input_tokens)
        output_tokens = int(task.get("output_tokens_per_unit") or output_tokens)
        break
    try:
        return round(estimate_text_model_cost(model, input_tokens, output_tokens, provider=provider), 6)
    except Exception:
        return 0.0


def _input_records(config: BlueprintGenerationConfig, workspace: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for kind, path in (
        ("analysis_md", config.analysis_md),
        ("slide_brief", config.slide_brief),
        ("source_pack", config.source_pack),
        ("matched_papers", config.matched_papers),
        ("source_metadata", config.source_metadata),
        ("source_corpus_inventory", config.source_corpus_inventory),
        ("speaker_author_work_ledger", config.speaker_author_work_ledger),
        ("source_repair_findings", config.source_repair_findings),
        ("research_cluster_plan", config.research_cluster_plan),
    ):
        if path is None:
            continue
        resolved = _resolve_inside(workspace, path, label=f"--{kind.replace('_', '-')}")
        _reject_legacy_presentations_lane(resolved, workspace, label=f"--{kind.replace('_', '-')}")
        records.append(
            {
                "kind": kind,
                "path": _relative_path(resolved, workspace),
                "sha256": _sha256_file(resolved),
                "chars": len(resolved.read_text(encoding="utf-8")),
                "content": _source_content_for_prompt(resolved),
            }
        )
    for index, path in enumerate(config.provenance_inputs, start=1):
        resolved = _resolve_inside(workspace, path, label="--provenance-input")
        _reject_legacy_presentations_lane(resolved, workspace, label="--provenance-input")
        records.append(
            {
                "kind": "provenance_input",
                "path": _relative_path(resolved, workspace),
                "sha256": _sha256_file(resolved),
                "chars": len(resolved.read_text(encoding="utf-8")),
                "content": _source_content_for_prompt(resolved),
                "index": index,
            }
        )
    return records


def _build_prompt(config: BlueprintGenerationConfig, input_records: list[dict[str, Any]]) -> str:
    parts = [
        "Create a source-grounded Presentation Blueprint JSON artifact.",
        f"presentation_id: {config.presentation_id}",
        f"presentation_type: {config.presentation_type}",
        f"topic: {config.topic or config.presentation_id}",
        (
            "Return exactly one JSON object using these required top-level keys: "
            "blueprint_id, revision, deck_thesis, audience_promise, target_runtime_minutes, "
            "target_slide_count, telescopic_section_map, subtopic_clusters, slides, "
            "conclusion_synthesis, final_report_url, input_hashes, duration_budget, "
            "source_paper_plan, speaker_author_work_mention_plan, cluster_coverage_justification, "
            "provenance."
        ),
        (
            "Each slide object must include slide_id, section_id, title or intent, source_plan "
            "as a non-empty list of objects with label fields, visual_role, narration_beat or "
            "visual_narration_hook, narration_beats, and estimated_minutes."
        ),
        (
            "provenance.source_metadata must list every approved source label used by slide "
            "source_plan. Use labels SRC-001, SRC-002, etc. for supplied matched papers."
        ),
        "The blueprint must include thesis, audience promise, telescopic sections, clusters, source plan, slide intents, visual roles, narration beats, conclusion, and runtime budget.",
        "Use only supplied sources. Reject decorative visuals and late weak references.",
    ]
    if config.final_report_url:
        parts.append(f"final_report_url: {config.final_report_url}")
    if config.summary:
        parts.append(f"operator_summary: {config.summary}")
    for record in input_records:
        parts.extend(
            [
                "",
                f"## Source: {record['kind']} ({record['path']})",
                f"sha256: {record['sha256']}",
                str(record.get("content", "")),
            ]
        )
    return "\n".join(parts)


def _build_source_ledger_blueprint_payload(
    *,
    config: BlueprintGenerationConfig,
    workspace: Path,
    input_records: list[dict[str, Any]],
) -> dict[str, Any]:
    records_by_kind = _records_by_kind(input_records)
    required = {
        "source_corpus_inventory",
        "speaker_author_work_ledger",
        "source_repair_findings",
        "research_cluster_plan",
    }
    missing = sorted(kind for kind in required if not records_by_kind.get(kind))
    if missing:
        raise BlueprintGenerationError(
            "BLUEPRINT_SOURCE_LEDGER_INPUT_REQUIRED",
            f"source-ledger mode requires inputs: {', '.join(missing)}",
        )
    final_report_url = config.final_report_url or _analysis_report_url_from_summary(config.summary)
    if not final_report_url:
        raise BlueprintGenerationError(
            "BLUEPRINT_FINAL_REPORT_URL_REQUIRED",
            "source-ledger mode requires --final-report-url",
        )

    inventory_payload = _load_json_path(workspace, config.source_corpus_inventory, label="source_corpus_inventory")
    ledger_payload = _load_json_path(workspace, config.speaker_author_work_ledger, label="speaker_author_work_ledger")
    repair_payload = _load_json_path(workspace, config.source_repair_findings, label="source_repair_findings")
    cluster_payload = _load_json_path(workspace, config.research_cluster_plan, label="research_cluster_plan")
    clusters = cluster_payload.get("clusters") if isinstance(cluster_payload, dict) else None
    if not isinstance(clusters, list) or not clusters:
        raise BlueprintGenerationError(
            "BLUEPRINT_RESEARCH_CLUSTER_PLAN_EMPTY",
            "research_cluster_plan.json must contain non-empty clusters",
        )

    ledger_by_source_id = _ledger_by_source_id(ledger_payload)
    source_metadata, labels_by_source_id = _source_metadata_from_clusters(
        clusters,
        ledger_by_source_id=ledger_by_source_id,
    )
    if not source_metadata:
        raise BlueprintGenerationError(
            "BLUEPRINT_SOURCE_LEDGER_EMPTY",
            "source-ledger mode could not resolve any visible source metadata",
        )

    input_hashes = _direct_input_hashes(input_records)
    topic_id = config.topic or str(cluster_payload.get("topic_id") or config.presentation_id)
    topic_title = str(cluster_payload.get("topic_title") or "").strip() or _humanize_topic_id(topic_id)
    target_slide_count = max(11, min(14, len(clusters) + 4))
    duration_target = round(min(38.0, max(22.0, target_slide_count * 2.0)), 1)
    subtopic_clusters = _blueprint_clusters(clusters, labels_by_source_id=labels_by_source_id)
    slides = _source_ledger_slides(
        topic_id=topic_id,
        topic_title=topic_title,
        target_slide_count=target_slide_count,
        clusters=subtopic_clusters,
        source_metadata=source_metadata,
    )
    _bind_cluster_slides(subtopic_clusters, slides)
    _cluster_titles = [
        str(cluster.get("title") or "").strip()
        for cluster in subtopic_clusters
        if str(cluster.get("title") or "").strip()
    ]
    _lead_clusters = "; ".join(_cluster_titles[:3]) if _cluster_titles else topic_title
    conference_label = conference_name(workspace, default="conference")
    source_counts = inventory_payload.get("counts") if isinstance(inventory_payload, dict) else {}
    repair_counts = repair_payload.get("counts") if isinstance(repair_payload, dict) else {}
    return {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "blueprint_id": config.presentation_id,
        "repair_of": None,
        "revision": 1,
        "source_plan_repair_of": None,
        "presentation_id": config.presentation_id,
        "presentation_type": config.presentation_type,
        "topic_id": topic_id,
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": final_report_url,
        "final_report_url": final_report_url,
        "deck_thesis": (
            f"{topic_title}: the {conference_label} corpus reframes this area around {_lead_clusters}, "
            "judged against named source evidence rather than a single benchmark or claim."
        ),
        "audience_promise": (
            f"The audience will leave with a source-grounded map of {topic_title}, tracing each "
            "claim to the named speakers, authors, and works behind it."
        ),
        "target_runtime_minutes": duration_target,
        "target_slide_count": target_slide_count,
        "duration_budget": {
            "target_minutes": duration_target,
            "hard_limit_minutes": 40,
            "per_slide_target_minutes": round(duration_target / target_slide_count, 2),
            "duration_budget": "pass",
        },
        "cluster_coverage_justification": {
            "required": target_slide_count > 14,
            "reason": f"not required: {topic_id} slide count stays within the 11-14 source-cluster range",
            "covered_cluster_ids": [str(cluster.get("cluster_id") or "") for cluster in subtopic_clusters],
        },
        "telescopic_section_map": _telescopic_section_map(slides),
        "subtopic_clusters": subtopic_clusters,
        "source_paper_plan": _source_paper_plan(source_metadata),
        "speaker_author_work_mention_plan": _speaker_author_work_mention_plan(source_metadata),
        "slides": slides,
        "conclusion_synthesis": (
            f"The cluster evidence points to one practical conclusion for {topic_title}: progress "
            "is real but conditional, and each claim must stay auditable from the named source "
            "ledger through to the rendered deck, figures, and narration rather than resting on a "
            "single benchmark score."
        ),
        "notebooklm_coherence_policy": (
            "NotebookLM delivery must preserve the same source-ledger labels, mention the named "
            "speaker/author/work contributions, and keep each visual explanation tied to a slide state."
        ),
        "input_hashes": input_hashes,
        "provenance": {
            "source_metadata": source_metadata,
            "input_hashes": input_hashes,
            "input_sha256": _sha256_json(input_records),
            "source_corpus_inventory_counts": source_counts,
            "source_repair_findings_counts": repair_counts,
            "research_cluster_plan_input_hashes": cluster_payload.get("input_hashes", {}),
            "generation_method": "deterministic_source_ledger",
            "provider_schema_normalized": False,
        },
    }


def _records_by_kind(input_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for record in input_records:
        by_kind.setdefault(str(record.get("kind")), []).append(record)
    return by_kind


def _json_from_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(str(record.get("content") or "{}"))
    if not isinstance(payload, dict):
        raise BlueprintGenerationError(
            "BLUEPRINT_SOURCE_LEDGER_JSON_INVALID",
            f"{record.get('kind')} must be a JSON object",
        )
    return payload


def _load_json_path(workspace: Path, path: Path | None, *, label: str) -> dict[str, Any]:
    if path is None:
        raise BlueprintGenerationError(
            "BLUEPRINT_SOURCE_LEDGER_INPUT_REQUIRED",
            f"source-ledger mode requires {label}",
        )
    resolved = _resolve_inside(workspace, path, label=f"--{label.replace('_', '-')}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlueprintGenerationError(
            "BLUEPRINT_SOURCE_LEDGER_JSON_INVALID",
            f"{label} must be a JSON object",
        )
    return payload


def _direct_input_hashes(input_records: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in input_records:
        path = str(record.get("path") or "")
        sha256 = str(record.get("sha256") or "")
        if not path or not sha256:
            continue
        basename = Path(path).name
        key = basename
        if key in hashes:
            key = f"{record.get('kind')}:{basename}"
        hashes[key] = sha256
    return hashes


def _ledger_by_source_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_id = entry.get("source_id")
        if isinstance(source_id, str) and source_id:
            result[source_id] = entry
    return result


def _source_metadata_from_clusters(
    clusters: list[Any],
    *,
    ledger_by_source_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    metadata: list[dict[str, Any]] = []
    labels_by_source_id: dict[str, str] = {}
    for cluster in clusters:
        if not isinstance(cluster, dict):
            continue
        for role in ("primary_sources", "supporting_sources"):
            sources = cluster.get(role)
            if not isinstance(sources, list):
                continue
            for source in sources:
                source_id = _source_id(source)
                if not source_id or source_id in labels_by_source_id:
                    continue
                merged = dict(ledger_by_source_id.get(source_id, {}))
                if isinstance(source, dict):
                    merged.update(source)
                label = f"SRC-{len(metadata) + 1:03d}"
                labels_by_source_id[source_id] = label
                metadata.append(
                    {
                        "label": label,
                        "source_id": source_id,
                        "title": str(
                            merged.get("resolved_title")
                            or merged.get("title")
                            or merged.get("work")
                            or source_id
                        ),
                        "authors": _string_list(merged.get("authors")),
                        "speakers": _string_list(merged.get("speakers")),
                        "works": _string_list(merged.get("works")),
                        "contribution_note": str(merged.get("contribution_note") or ""),
                        "source_family": str(merged.get("source_family") or ""),
                        "confidence": merged.get("confidence"),
                        "status": str(merged.get("status") or ""),
                        "eligibility_basis": str(merged.get("eligibility_basis") or "speaker_author_work_ledger"),
                        "evidence_paths": _string_list(merged.get("evidence_paths")),
                    }
                )
    return metadata, labels_by_source_id


def _source_id(source: Any) -> str | None:
    if isinstance(source, str) and source.strip():
        return source.strip()
    if isinstance(source, dict):
        value = source.get("source_id") or source.get("id") or source.get("label")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _blueprint_clusters(
    clusters: list[Any],
    *,
    labels_by_source_id: dict[str, str],
) -> list[dict[str, Any]]:
    blueprint_clusters: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        if not isinstance(cluster, dict):
            continue
        primary_labels = _labels_for_sources(cluster.get("primary_sources"), labels_by_source_id)
        supporting_labels = _labels_for_sources(cluster.get("supporting_sources"), labels_by_source_id)
        labels = primary_labels + [label for label in supporting_labels if label not in set(primary_labels)]
        cluster_id = str(cluster.get("cluster_id") or f"cluster-{index}")
        blueprint_clusters.append(
            {
                "cluster_id": cluster_id,
                "title": str(cluster.get("title") or cluster_id.replace("-", " ").title()),
                "story_contribution": str(cluster.get("story_contribution") or ""),
                "sources": labels,
                "primary_source_labels": primary_labels,
                "supporting_source_labels": supporting_labels,
                "slide_bindings": [],
                "retained_rationale": str(cluster.get("retained_rationale") or ""),
                "merged_rationale": str(cluster.get("merged_rationale") or ""),
                "dropped_rationale": str(cluster.get("dropped_rationale") or ""),
            }
        )
    return blueprint_clusters


def _labels_for_sources(value: Any, labels_by_source_id: dict[str, str]) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for source in value:
        source_id = _source_id(source)
        label = labels_by_source_id.get(source_id or "")
        if label and label not in labels:
            labels.append(label)
    return labels


def _source_paper_plan(source_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": item["label"],
            "source_id": item["source_id"],
            "title": item["title"],
            "authors": item["authors"],
            "speakers": item["speakers"],
            "works": item["works"],
            "contribution_note": item["contribution_note"],
        }
        for item in source_metadata
    ]


def _speaker_author_work_mention_plan(source_metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": item["label"],
            "source_id": item["source_id"],
            "speakers": item["speakers"],
            "authors": item["authors"],
            "works": item["works"],
            "contribution_note": item["contribution_note"],
        }
        for item in source_metadata
    ]


def _humanize_topic_id(topic_id: str) -> str:
    """Fallback display title for a topic when no narrative title is available
    (e.g. ``01_reasoning_post_training`` -> ``reasoning post training``)."""

    humanized = re.sub(r"^\d+[_-]", "", str(topic_id)).replace("_", " ").replace("-", " ").strip()
    return humanized or str(topic_id)


def _source_ledger_slides(
    *,
    topic_id: str,
    topic_title: str,
    target_slide_count: int,
    clusters: list[dict[str, Any]],
    source_metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metadata_by_label = {str(item["label"]): item for item in source_metadata}
    all_labels = [str(item["label"]) for item in source_metadata]
    slides: list[dict[str, Any]] = []

    def add_slide(
        *,
        title: str,
        intent: str,
        cluster: dict[str, Any] | None,
        section_id: str,
        visual_role: str,
        claim_override: str | None = None,
    ) -> None:
        labels = list(cluster.get("sources", [])) if cluster else all_labels[:3]
        if not labels:
            labels = all_labels[:1]
        slide_id = f"sl_{len(slides) + 1:02d}_{_slugify(title)[:36]}"
        mentions = [_mention_for_label(label, metadata_by_label) for label in labels]
        # Synthesis slides pass a distinct claim so they do not restate the cluster
        # slide's story verbatim (which reads as copy-paste body text).
        story = claim_override or (str(cluster.get("story_contribution") or intent) if cluster else intent)
        slides.append(
            {
                "slide_id": slide_id,
                "section_id": section_id,
                "title": title,
                "intent": intent,
                "claims": [story],
                "visible_text": [title, story],
                "source_plan": [
                    {
                        "label": label,
                        "role": "grounds slide claim",
                        "cluster_id": cluster.get("cluster_id") if cluster else "synthesis",
                    }
                    for label in labels
                ],
                "speaker_author_work_mentions": mentions,
                "visual_role": visual_role,
                "visual_narration_hook": (
                    f"{title}: the named evidence and what it changes."
                ),
                "narration_beat": (
                    f"Name the source-backed contribution for {title} and connect it to the visible evidence."
                ),
                "narration_beats": [
                    f"Introduce {title} as a source-backed step in the {topic_title} argument.",
                    "Name at least one speaker, author, or work from the source plan.",
                    "Explain what the audience sees and why that visual evidence matters.",
                ],
                "estimated_minutes": 2.0,
            }
        )

    # Keep the frame heading complete (not truncated mid-sentence in the render
    # band): only append the "what changed" tail when the whole title still fits.
    frame_title = f"{topic_title}: what changed and why it matters now"
    if len(frame_title) > 66:
        frame_title = topic_title
    add_slide(
        title=frame_title,
        intent=f"Frame {topic_title} around the source-grounded research clusters in this deck.",
        cluster=clusters[0] if clusters else None,
        section_id="S0_frame",
        visual_role=f"Thesis map connecting the {topic_title} clusters to their named source evidence",
    )
    for cluster in clusters:
        add_slide(
            title=str(cluster.get("title") or cluster.get("cluster_id")),
            intent=str(cluster.get("story_contribution") or "Explain the source cluster contribution."),
            cluster=cluster,
            section_id="S1_source_clusters",
            visual_role=f"Evidence map linking the {cluster.get('title')} cluster to named sources and contribution notes",
        )

    cluster_titles = [str(c.get("title") or c.get("cluster_id") or "") for c in clusters]
    # Concise, topic-neutral synthesis headings (the deck context supplies the
    # topic); embedding the full topic title here overflows the render band.
    synthesis_titles = [
        "How the research clusters connect",
        "What the evidence changes in practice",
        "Open questions for the next research cycle",
        "What the canary deck must prove",
        "From source ledger to rendered deck",
    ]

    _cap_connectors = {
        "a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
        "with", "at", "is", "are", "that", "this", "by", "as", "from",
        "into", "over", "than", "but", "nor", "via", "per",
    }

    def _cap_title(value: str, limit: int = 46) -> str:
        compact = " ".join(str(value or "").split())
        if len(compact) <= limit:
            return compact
        words: list[str] = []
        for word in compact.split():
            if len(" ".join([*words, word])) > limit:
                break
            words.append(word)
        # Never end a capped title on a dangling connector ("... is a concrete").
        while words and words[-1].lower().strip(".,;:'\"()") in _cap_connectors:
            words.pop()
        return " ".join(words) or compact[:limit].rstrip()

    index = 0
    while len(slides) < target_slide_count:
        cluster = clusters[index % len(clusters)] if clusters else None
        synthesis_title = synthesis_titles[index % len(synthesis_titles)]
        cluster_title = str(cluster.get("title") or "") if cluster else ""
        if cluster_titles and index < len(cluster_titles):
            synthesis_title = f"Implications: {_cap_title(cluster_titles[index])}"
            subject = cluster_title.rstrip(".").lower() or "this evidence"
            # Cluster titles are mixed grammatical types ("RLVR is central but
            # brittle" is a clause; "hybrid sparse/dense rewards" is a noun
            # phrase). Wrap each so it slots into a noun position grammatically.
            clause_verbs = {
                "is", "are", "was", "were", "becomes", "becoming", "remains",
                "has", "have", "can", "cannot", "must", "will", "may", "fails",
                "wins", "beats", "matters", "works", "outperforms", "changes",
            }
            subject_words = {w.strip(".,;:'\"()") for w in subject.split()}
            if subject_words & clause_verbs:
                subject_phrase = f"the finding that {subject}"
            else:
                subject_phrase = f"the evidence on {subject}"
            # Each implication slide takes a DIFFERENT forward-looking angle
            # (training, evaluation, deployment economics, operations) so the
            # synthesis slides do not read as one repeated mad-lib sentence.
            implication_angles = [
                (
                    f"Training implication: teams building {topic_title} systems should treat "
                    f"{subject_phrase} as a design constraint from day one — it changes which "
                    "post-training recipes are worth running at all."
                ),
                (
                    f"Evaluation implication: benchmarks that ignore {subject_phrase} will "
                    f"overstate progress; {topic_title} evaluations need to measure it "
                    "directly rather than assume it away."
                ),
                (
                    f"Deployment implication: because of {subject_phrase}, the cost and latency "
                    f"budget of a production {topic_title} system is a tunable policy, "
                    "not a fixed property of the model."
                ),
                (
                    f"Operations implication: {subject_phrase} will not stay static — production "
                    f"{topic_title} systems need monitoring and re-tuning loops that track it "
                    "as models and workloads drift."
                ),
            ]
            claim = implication_angles[index % len(implication_angles)]
        else:
            other = cluster_titles[(index + 1) % len(cluster_titles)] if len(cluster_titles) > 1 else topic_title
            claim = (
                f"Cross-cluster synthesis: the evidence on {cluster_title.rstrip('.').lower() or topic_title} "
                f"and {str(other).rstrip('.').lower()} converges into the deck's argument for {topic_title}, "
                "kept auditable from the named source ledger to the rendered figures and narration."
            )
        add_slide(
            title=synthesis_title,
            intent=f"Synthesize cluster evidence into a production-deck takeaway for {topic_title}.",
            cluster=cluster,
            section_id="S2_synthesis",
            visual_role="Synthesis matrix connecting source clusters, visual roles, and narration obligations",
            claim_override=claim,
        )
        index += 1
    return slides[:target_slide_count]


def _bind_cluster_slides(clusters: list[dict[str, Any]], slides: list[dict[str, Any]]) -> None:
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id")
        bindings = [
            slide["slide_id"]
            for slide in slides
            if any(source.get("cluster_id") == cluster_id for source in slide.get("source_plan", []))
        ]
        cluster["slide_bindings"] = bindings[:3] or [slides[0]["slide_id"]]


def _telescopic_section_map(slides: list[dict[str, Any]]) -> list[dict[str, Any]]:
    section_titles = {
        "S0_frame": "Frame the system-level thesis",
        "S1_source_clusters": "Expand through source-grounded research clusters",
        "S2_synthesis": "Compress into deployment implications and canary proof obligations",
    }
    sections: list[dict[str, Any]] = []
    for section_id, title in section_titles.items():
        slide_ids = [slide["slide_id"] for slide in slides if slide.get("section_id") == section_id]
        if slide_ids:
            sections.append({"section_id": section_id, "title": title, "slide_ids": slide_ids})
    return sections


def _mention_for_label(
    label: str,
    metadata_by_label: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item = metadata_by_label.get(label, {})
    title = str(item.get("title") or "")
    return {
        "label": label,
        "source_id": str(item.get("source_id") or ""),
        "title": title,
        "speakers": item.get("speakers", []),
        "authors": item.get("authors", []),
        # A title-only work (no resolved people) still names a work; carry the
        # title into works so the deck contract can render it as a work mention.
        "works": item.get("works") or ([title] if title else []),
        "contribution_note": item.get("contribution_note", ""),
    }


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "slide"


def _normalize_provider_blueprint_payload(
    payload: dict[str, Any],
    *,
    config: BlueprintGenerationConfig,
    input_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Coerce common provider synonym fields into the strict blueprint contract."""

    conference_label = conference_name(config.workspace, default="conference")
    source_metadata = _approved_source_metadata(input_records, conference_label=conference_label)
    source_labels = [str(item["label"]) for item in source_metadata]
    fallback_labels = source_labels[:3] or ["SRC-001"]
    raw_slides = payload.get("slides") if isinstance(payload.get("slides"), list) else []
    slides = [
        _normalize_slide(slide if isinstance(slide, dict) else {}, index=index, approved_labels=source_labels)
        for index, slide in enumerate(raw_slides, start=1)
    ]
    if not slides:
        slides = [
            _normalize_slide(
                {"title": str(payload.get("thesis") or payload.get("deck_thesis") or config.presentation_id)},
                index=1,
                approved_labels=source_labels,
            )
        ]
    sections = _normalize_sections(payload, slides, conference_label=conference_label)
    clusters = _normalize_clusters(payload, slides, fallback_labels, conference_label=conference_label)
    report_url = _analysis_report_url_from_summary(config.summary) or str(
        payload.get("analysis_report_url")
        or payload.get("final_report_url")
        or "https://drive.google.com/"
    )
    target_runtime = _number_from_runtime_budget(payload.get("runtime_budget"))
    if target_runtime is None:
        target_runtime = _optional_float(payload.get("target_runtime_minutes")) or min(36.0, max(12.0, len(slides) * 1.5))
    normalized = {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "blueprint_id": str(payload.get("blueprint_id") or f"{config.presentation_id}-blueprint"),
        "repair_of": payload.get("repair_of"),
        "revision": _positive_int(payload.get("revision")) or 1,
        "source_plan_repair_of": payload.get("source_plan_repair_of"),
        "presentation_id": config.presentation_id,
        "presentation_type": config.presentation_type,
        "topic_id": config.topic or config.presentation_id,
        "author": "Nikolaos Vasiloglou",
        "analysis_report_url": report_url,
        "final_report_url": report_url,
        "deck_thesis": str(
            payload.get("deck_thesis")
            or payload.get("thesis")
            or f"Source-grounded synthesis of the {conference_label} corpus evidence."
        ),
        "audience_promise": _stringify_audience_promise(
            payload.get("audience_promise"), conference_label=conference_label
        ),
        "target_runtime_minutes": float(target_runtime),
        "target_slide_count": len(slides),
        "telescopic_section_map": sections,
        "subtopic_clusters": clusters,
        "slides": slides,
        "conclusion_synthesis": _stringify_conclusion(
            payload.get("conclusion_synthesis") or payload.get("conclusion"),
            conference_label=conference_label,
        ),
        "notebooklm_coherence_policy": str(
            payload.get("notebooklm_coherence_policy")
            or payload.get("narration_beats_policy")
            or "NotebookLM delivery must stay slide-locked, avoid long voice gaps, and preserve source/visual coupling."
        ),
        "provenance": {
            "source_metadata": source_metadata,
            "input_sha256": _sha256_json(input_records),
            "provider_payload_keys": sorted(str(key) for key in payload.keys()),
            "provider_schema_normalized": True,
        },
    }
    return normalized


def _analysis_report_url_from_summary(summary: str | None) -> str | None:
    if not summary:
        return None
    match = re.search(r"https://drive\.google\.com/[^\s;,)]+", summary)
    return match.group(0) if match else None


def _approved_source_metadata(
    input_records: list[dict[str, Any]], *, conference_label: str = "conference"
) -> list[dict[str, Any]]:
    for record in input_records:
        if record.get("kind") != "matched_papers":
            continue
        try:
            payload = json.loads(str(record.get("content") or "{}"))
        except json.JSONDecodeError:
            continue
        papers = payload.get("papers") if isinstance(payload, dict) else None
        if not isinstance(papers, list):
            continue
        metadata: list[dict[str, Any]] = []
        for index, paper in enumerate(papers[:24], start=1):
            if not isinstance(paper, dict):
                continue
            metadata.append(
                {
                    "label": f"SRC-{index:03d}",
                    "title": str(paper.get("title") or f"Matched paper {index}"),
                    "url": str(paper.get("url") or "https://openreview.net/"),
                    "source_type": "paper",
                    "paper_id": str(paper.get("paper_id") or ""),
                    "rank": paper.get("rank", index),
                }
            )
        if metadata:
            return metadata
    return [
        {
            "label": "SRC-001",
            "title": f"The {conference_label} corpus analysis report",
            "url": "https://drive.google.com/",
            "source_type": "report",
        }
    ]


def _normalize_slide(
    slide: dict[str, Any],
    *,
    index: int,
    approved_labels: list[str],
) -> dict[str, Any]:
    slide_id = str(slide.get("slide_id") or f"slide-{index:03d}")
    title = str(slide.get("title") or slide.get("intent") or f"Slide {index}")
    labels = _source_labels_from_slide(slide, approved_labels)
    visual_role = str(
        slide.get("visual_role")
        or slide.get("visual_role_description")
        or slide.get("visual")
        or f"Evidence map showing how source claims support {title}"
    )
    hook = str(
        slide.get("visual_narration_hook")
        or slide.get("narration_beat")
        or slide.get("narration")
        or f"Use the visual to connect source evidence to {title}."
    )
    return {
        "slide_id": slide_id,
        "section_id": str(slide.get("section_id") or slide.get("section") or f"section-{max(1, min(index, 6))}"),
        "title": title,
        "intent": str(slide.get("intent") or title),
        "claims": _string_list(slide.get("claims")) or [title],
        "visible_text": _string_list(slide.get("visible_text")) or _string_list(slide.get("bullets")) or [title],
        "source_plan": [{"label": label, "role": "grounds slide claim"} for label in labels],
        "visual_role": visual_role,
        "visual_narration_hook": hook,
        "narration_beat": str(slide.get("narration_beat") or hook),
        "estimated_minutes": float(_optional_float(slide.get("estimated_minutes")) or 1.5),
    }


def _source_labels_from_slide(slide: dict[str, Any], approved_labels: list[str]) -> list[str]:
    labels: list[str] = []
    approved = set(approved_labels)
    for key in ("source_plan", "source_labels", "sources"):
        value = slide.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if isinstance(item, str) and item in approved:
                labels.append(item)
            elif isinstance(item, dict) and isinstance(item.get("label"), str) and item["label"] in approved:
                labels.append(item["label"])
    return labels or approved_labels[: min(3, len(approved_labels))]


def _normalize_sections(
    payload: dict[str, Any],
    slides: list[dict[str, Any]],
    *,
    conference_label: str = "conference",
) -> list[dict[str, Any]]:
    raw = payload.get("telescopic_section_map") or payload.get("telescopic_sections")
    sections: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for index, section in enumerate(raw, start=1):
            if not isinstance(section, dict):
                continue
            sections.append(
                {
                    "section_id": str(section.get("section_id") or section.get("id") or f"section-{index}"),
                    "title": str(section.get("title") or section.get("name") or f"Section {index}"),
                    "slide_ids": _string_list(section.get("slide_ids")) or [slide["slide_id"] for slide in slides[max(0, index - 1) : index]],
                }
            )
    if sections:
        return sections
    return [
        {
            "section_id": "section-1",
            "title": f"The {conference_label} corpus evidence",
            "slide_ids": [slide["slide_id"] for slide in slides],
        }
    ]


def _normalize_clusters(
    payload: dict[str, Any],
    slides: list[dict[str, Any]],
    fallback_labels: list[str],
    *,
    conference_label: str = "conference",
) -> list[dict[str, Any]]:
    raw = payload.get("subtopic_clusters") or payload.get("clusters")
    clusters: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for index, cluster in enumerate(raw, start=1):
            if not isinstance(cluster, dict):
                continue
            clusters.append(
                {
                    "cluster_id": str(cluster.get("cluster_id") or cluster.get("id") or f"cluster-{index}"),
                    "title": str(cluster.get("title") or cluster.get("name") or f"Cluster {index}"),
                    "sources": _string_list(cluster.get("sources")) or fallback_labels[:1],
                    "slide_bindings": _string_list(cluster.get("slide_bindings")) or [slide["slide_id"] for slide in slides[max(0, index - 1) : index]],
                }
            )
    if clusters:
        return clusters
    return [
        {
            "cluster_id": "cluster-1",
            "title": f"Structured evidence from the {conference_label} corpus",
            "sources": fallback_labels[:1],
            "slide_bindings": [slide["slide_id"] for slide in slides],
        }
    ]


def _stringify_audience_promise(value: Any, *, conference_label: str = "conference") -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        parts = [str(value.get(key) or "").strip() for key in ("who", "promise", "non_promises")]
        joined = " ".join(part for part in parts if part)
        if joined:
            return joined
    return (
        f"A source-grounded audience can understand the {conference_label} corpus evidence "
        "and its practical implications."
    )


def _stringify_conclusion(value: Any, *, conference_label: str = "conference") -> str:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        joined = " ".join(str(v).strip() for v in value.values() if str(v).strip())
        if joined:
            return joined
    return (
        f"The {conference_label} corpus evidence points to one practical conclusion: progress is "
        "real but conditional, and robust systems need source-grounded evaluation, layered "
        "defenses, and explicit threat models."
    )


def _number_from_runtime_budget(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("target_runtime_minutes", "minutes", "target_minutes"):
            parsed = _optional_float(value.get(key))
            if parsed is not None:
                return parsed
    return _optional_float(value)


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 1 else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            result.append(item.strip())
        elif isinstance(item, dict):
            label = item.get("label") or item.get("slide_id") or item.get("title") or item.get("id")
            if isinstance(label, str) and label.strip():
                result.append(label.strip())
    return result


def _live_generate_blueprint(
    *,
    config: BlueprintGenerationConfig,
    workspace: Path,
    provider: str,
    model_id: str,
    prompt: str,
) -> tuple[dict[str, Any], list[str], float | None]:
    if not has_credentials(provider):
        raise BlueprintGenerationError(
            BLUEPRINT_PROVIDER_CREDENTIALS_MISSING,
            f"{' or '.join(credential_names(provider) or ())} is required for live "
            f"presentation blueprint generation via {provider}",
        )

    before = set(provider_receipt_summary(workspace).get("receipt_paths", []))
    with provider_receipt_context(
        workspace_root=workspace,
        run_id=config.run_id,
        stage_id="presentation_blueprint",
        task_id=BLUEPRINT_GENERATION_TASK_ID,
        caller="generate_presentation_blueprint.py",
    ):
        result = complete(
            BLUEPRINT_GENERATION_TASK_ID,
            prompt=prompt,
            provider=provider,
            model=model_id,
            max_tokens=24000,
            stream=True,  # Anthropic wants streaming at this ceiling; other providers ignore it
            caller="scripts/presentations/generate_presentation_blueprint.py",
        )
    if result.stop_reason == "max_tokens":
        raise BlueprintGenerationError(
            BLUEPRINT_LIVE_GENERATION_FAILED,
            f"{provider} blueprint reply was cut off at the 24000-token ceiling",
        )
    response_text = result.text
    blueprint_payload = _extract_json_object(response_text)
    after_summary = provider_receipt_summary(workspace)
    after = set(after_summary.get("receipt_paths", []))
    provider_receipt_paths = sorted(after - before)
    cost = None
    if provider_receipt_paths:
        receipts = provider_receipt_summary(workspace).get("canonical_totals_by_model", {})
        if isinstance(receipts, dict):
            cost = None
    return blueprint_payload, provider_receipt_paths, cost


def _assert_live_guardrails(budget_guardrail: dict[str, Any]) -> None:
    if not budget_guardrail["provider_cost_approved"]:
        raise BlueprintGenerationError(
            BLUEPRINT_PROVIDER_COST_APPROVAL_REQUIRED,
            "rerun with --provider-cost-approved after confirming paid provider budget",
        )
    if budget_guardrail["status"] == "fail":
        raise BlueprintGenerationError(
            BLUEPRINT_PROVIDER_BUDGET_CAP_EXCEEDED,
            "budget cap halt-and-approve guardrail",
        )


def _budget_guardrail(
    *,
    workspace: Path,
    provider_cost_approved: bool,
    estimated_cost_usd: float,
) -> dict[str, Any]:
    registry = load_registry()
    policy = registry.get(BLUEPRINT_GENERATION_TASK_ID)
    summary = provider_receipt_summary(workspace)
    existing = float(summary.get("canonical_total_estimated_cost_usd") or 0.0)
    policy_cap = policy.budget_cap_usd
    total_after = existing + estimated_cost_usd
    failures: list[str] = []
    if policy_cap is not None and estimated_cost_usd > policy_cap:
        failures.append("task budget cap exceeded")
    if total_after > GLOBAL_PROVIDER_BUDGET_CAP_USD:
        failures.append("global budget cap exceeded")
    return {
        "status": "fail" if failures else "pass",
        "failures": failures,
        "provider_cost_approved": provider_cost_approved,
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "existing_recorded_spend_usd": round(existing, 6),
        "projected_recorded_spend_usd": round(total_after, 6),
        "task_budget_cap_usd": policy_cap,
        "global_budget_cap_usd": GLOBAL_PROVIDER_BUDGET_CAP_USD,
        "guardrail": "budget cap halt-and-approve guardrail",
    }


def _receipt_payload(
    *,
    config: BlueprintGenerationConfig,
    paths: BlueprintGenerationPaths,
    mode: str,
    production_evidence: bool,
    provider: str,
    model_id: str,
    prompt_sha256: str,
    input_sha256: str,
    input_records: list[dict[str, Any]],
    provider_receipt_paths: list[str],
    estimated_cost_usd: float,
    budget_guardrail: dict[str, Any],
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = config.workspace.expanduser().resolve()
    payload: dict[str, Any] = {
        "schema_version": BLUEPRINT_SCHEMA_VERSION,
        "receipt_type": "conference_pipeline.presentation.blueprint_generation",
        "presentation_id": config.presentation_id,
        "presentation_type": config.presentation_type,
        "topic": config.topic,
        "run_id": config.run_id,
        "mode": mode,
        "production_evidence": production_evidence,
        "generated_at": _utc_now(),
        "blueprint_path": _relative_path(paths.output_json, workspace),
        "blueprint_sha256": _sha256_file(paths.output_json),
        "markdown_path": _relative_path(paths.output_md, workspace),
        "receipt_path": _relative_path(paths.receipt_path, workspace),
        "model_governance_task_id": BLUEPRINT_GENERATION_TASK_ID,
        "provider": provider,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "input_sha256": input_sha256,
        "inputs": [_receipt_input_record(record) for record in input_records],
        "provider_receipt_paths": provider_receipt_paths,
        "estimated_cost_usd": round(float(estimated_cost_usd), 6),
        "budget_guardrail": budget_guardrail,
        "validation": validation,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_blueprint_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Presentation Blueprint",
        "",
        f"- Blueprint ID: `{payload.get('blueprint_id', '')}`",
        f"- Revision: `{payload.get('revision', '')}`",
        f"- Thesis: {payload.get('deck_thesis', '')}",
        f"- Audience promise: {payload.get('audience_promise', '')}",
        "",
        "## Telescopic Map",
    ]
    for section in payload.get("telescopic_section_map", []):
        if isinstance(section, dict):
            lines.append(f"- {section.get('section_id', '')}: {section.get('title', '')}")
    lines.extend(["", "## Slides"])
    for slide in payload.get("slides", []):
        if isinstance(slide, dict):
            lines.append(f"- {slide.get('slide_id', '')}: {slide.get('intent', '')}")
    lines.extend(["", "## Conclusion", str(payload.get("conclusion_synthesis", "")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_content_for_prompt(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    if len(content) <= SOURCE_CONTENT_CHAR_LIMIT:
        return content
    omitted = len(content) - SOURCE_CONTENT_CHAR_LIMIT
    return (
        content[:SOURCE_CONTENT_CHAR_LIMIT]
        + f"\n\n[TRUNCATED: {omitted} additional characters omitted from this source]\n"
    )


def _receipt_input_record(record: dict[str, Any]) -> dict[str, Any]:
    content = str(record.get("content") or "")
    sanitized = {
        key: value
        for key, value in record.items()
        if key != "content"
    }
    sanitized["content_included_chars"] = len(content)
    sanitized["content_truncated"] = "[TRUNCATED:" in content
    return sanitized


def _summary(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": receipt.get("validation", {}).get("status"),
        "mode": receipt.get("mode"),
        "production_evidence": receipt.get("production_evidence"),
        "presentation_id": receipt.get("presentation_id"),
        "blueprint_path": receipt.get("blueprint_path"),
        "receipt_path": receipt.get("receipt_path"),
        "model_governance_task_id": receipt.get("model_governance_task_id"),
        "provider": receipt.get("provider"),
        "model_id": receipt.get("model_id"),
        "estimated_cost_usd": receipt.get("estimated_cost_usd"),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise BlueprintGenerationError(BLUEPRINT_LIVE_GENERATION_FAILED, "provider response did not contain a JSON object")
    payload = json.loads(match.group())
    if not isinstance(payload, dict):
        raise BlueprintGenerationError(BLUEPRINT_LIVE_GENERATION_FAILED, "provider response JSON must be an object")
    return payload


def _resolve_inside(workspace: Path, candidate: str | Path, *, label: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace: {resolved}") from exc
    return resolved


def _reject_legacy_presentations_lane(path: Path, workspace: Path, *, label: str) -> None:
    try:
        relative = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        return
    if relative.parts and relative.parts[0] == "presentations":
        raise ValueError(f"{label}: {_LEGACY_PRESENTATIONS_MESSAGE}")


def _relative_path(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
