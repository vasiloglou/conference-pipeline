"""Deterministic Presentation Blueprint review and approval-token writing."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.blueprint import (
    APPROVED_FOR_DECK_CONTRACT_GENERATION,
    APPROVED_FOR_DRAFT_GENERATION,
    BLUEPRINT_RUNTIME_HARD_BUDGET_MINUTES,
    GATE_TYPES,
    validate_blueprint_file,
    validate_blueprint_receipt_file,
)
from conference_pipeline.workflow_runs import write_json_atomic


BLUEPRINT_RELEVANCE_LLM_JUDGE_PENDING = "BLUEPRINT_RELEVANCE_LLM_JUDGE_PENDING"
BLUEPRINT_RELEVANCE_LLM_JUDGE_APPROVED = "BLUEPRINT_RELEVANCE_LLM_JUDGE_APPROVED"
BLUEPRINT_RELEVANCE_LLM_JUDGE_REQUIRED = "BLUEPRINT_RELEVANCE_LLM_JUDGE_REQUIRED"
BLUEPRINT_HUMAN_APPROVAL_REQUIRED = "BLUEPRINT_HUMAN_APPROVAL_REQUIRED"
BLUEPRINT_HUMAN_APPROVAL_RECORDED = "BLUEPRINT_HUMAN_APPROVAL_RECORDED"
BLUEPRINT_RECEIPT_REQUIRED = "BLUEPRINT_RECEIPT_REQUIRED"
BLUEPRINT_SOURCE_COVERAGE_MISSING = "BLUEPRINT_SOURCE_COVERAGE_MISSING"
BLUEPRINT_FINAL_REPORT_MISSING = "BLUEPRINT_FINAL_REPORT_MISSING"
BLUEPRINT_CONTRIBUTION_PLAN_MISSING = "BLUEPRINT_CONTRIBUTION_PLAN_MISSING"
BLUEPRINT_CONCLUSION_WEAK = "BLUEPRINT_CONCLUSION_WEAK"
BLUEPRINT_VISUAL_ROLES_MISSING = "BLUEPRINT_VISUAL_ROLES_MISSING"
BLUEPRINT_NARRATION_BEATS_MISSING = "BLUEPRINT_NARRATION_BEATS_MISSING"
BLUEPRINT_DURATION_BUDGET_FAILED = "BLUEPRINT_DURATION_BUDGET_FAILED"


def review_presentation_blueprint(
    blueprint_path: str | Path,
    *,
    source_metadata_path: str | Path | None = None,
    blueprint_receipt_path: str | Path | None = None,
    output_json: str | Path,
    output_md: str | Path | None = None,
    approve_for_draft_generation: bool = False,
    approve_for_deck_contract_generation: bool = False,
    approved_by: str | None = None,
    review_artifact: str | None = None,
    llm_judge_approved: bool = False,
    llm_judge_artifact: str | None = None,
) -> dict[str, Any]:
    """Run deterministic blueprint review and write JSON/Markdown artifacts."""

    blueprint = Path(blueprint_path).expanduser().resolve()
    output = Path(output_json).expanduser().resolve()
    md_path = Path(output_md).expanduser().resolve() if output_md is not None else output.with_suffix(".md")
    validation = validate_blueprint_file(blueprint)
    findings = [record.to_dict() | {"severity": "fail"} for record in validation.errors]
    findings.extend(record.to_dict() | {"severity": "warn"} for record in validation.warnings)
    approval_requested = approve_for_draft_generation or approve_for_deck_contract_generation
    approval_status = (
        APPROVED_FOR_DECK_CONTRACT_GENERATION
        if approve_for_deck_contract_generation
        else APPROVED_FOR_DRAFT_GENERATION
    )
    blueprint_payload = _load_blueprint_payload(blueprint)
    findings.extend(_phase15_review_findings(blueprint_payload))

    receipt_errors = ()
    if blueprint_receipt_path is not None:
        receipt_validation = validate_blueprint_receipt_file(blueprint, blueprint_receipt_path)
        receipt_errors = receipt_validation.errors
        findings.extend(record.to_dict() | {"severity": "fail"} for record in receipt_validation.errors)
    elif approval_requested:
        findings.append(
            {
                "code": BLUEPRINT_RECEIPT_REQUIRED,
                "severity": "fail",
                "message": "approved blueprint review requires a hash-bound presentation_blueprint_receipt.json",
                "path": "$.blueprint_receipt",
                "gate_type": "deterministic",
            }
        )

    if llm_judge_approved:
        findings.append(
            {
                "code": BLUEPRINT_RELEVANCE_LLM_JUDGE_APPROVED,
                "severity": "info",
                "message": "LLM-judge source relevance and story-quality gate recorded",
                "path": "$.llm_judge",
                "gate_type": "LLM-judge",
                "artifact": llm_judge_artifact,
            }
        )
    else:
        findings.append(
            {
                "code": (
                    BLUEPRINT_RELEVANCE_LLM_JUDGE_REQUIRED
                    if approval_requested
                    else BLUEPRINT_RELEVANCE_LLM_JUDGE_PENDING
                ),
                "severity": "fail" if approval_requested else "info",
                "message": "source relevance and story quality require the configured LLM-judge gate before canary approval",
                "path": "$.llm_judge",
                "gate_type": "LLM-judge",
            }
        )

    deterministic_pass = not any(finding.get("severity") == "fail" for finding in findings)
    llm_pass = llm_judge_approved
    approval_allowed = approval_requested and deterministic_pass and llm_pass
    if approval_allowed:
        findings.append(
            {
                "code": BLUEPRINT_HUMAN_APPROVAL_RECORDED,
                "severity": "info",
                "message": f"human approval recorded for {approval_status}",
                "path": "$.approval",
                "gate_type": "human",
            }
        )
    else:
        findings.append(
            {
                "code": BLUEPRINT_HUMAN_APPROVAL_REQUIRED,
                "severity": "fail" if approval_requested else "info",
                "message": "human approval is required before production draft_generation or deck_contract_generation consumes this blueprint",
                "path": "$.approval",
                "gate_type": "human",
            }
        )

    status = _review_status(findings)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "review_type": "conference_pipeline.presentation.blueprint_review",
        "status": status,
        "blueprint_path": str(blueprint),
        "blueprint_receipt_path": str(Path(blueprint_receipt_path).expanduser().resolve())
        if blueprint_receipt_path
        else None,
        "source_metadata_path": str(Path(source_metadata_path).expanduser().resolve()) if source_metadata_path else None,
        "llm_judge_artifact": llm_judge_artifact,
        "reviewed_at": _utc_now(),
        "findings": findings,
        "gate_types": {gate_type: _gate_status(gate_type, findings) for gate_type in sorted(GATE_TYPES)},
        "validation": validation.to_dict(),
    }
    if approval_allowed and status == "pass":
        payload.update(
            {
                "approval": {"status": approval_status},
                "blueprint_sha256": _sha256_file(blueprint),
                "approved_at": _utc_now(),
                "approved_by": approved_by or "operator",
                "review_artifact": review_artifact or md_path.name,
            }
        )
    write_json_atomic(output, payload)
    _write_review_markdown(md_path, payload)
    return payload


def _load_blueprint_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _phase15_review_findings(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    findings: list[dict[str, Any]] = []
    if not _has_source_coverage(payload):
        findings.append(
            _finding(
                BLUEPRINT_SOURCE_COVERAGE_MISSING,
                "missing source coverage: source_paper_plan, source metadata, and source-ledger input_hashes are required",
                "$.source_paper_plan",
                "deterministic",
            )
        )
    if not _has_final_report_url(payload):
        findings.append(
            _finding(
                BLUEPRINT_FINAL_REPORT_MISSING,
                "missing final report link: final_report_url or analysis_report_url must be an external report URL",
                "$.final_report_url",
                "deterministic",
            )
        )
    if not _has_speaker_author_work_plan(payload):
        findings.append(
            _finding(
                BLUEPRINT_CONTRIBUTION_PLAN_MISSING,
                "speaker/author/work mention plan is required before blueprint approval",
                "$.speaker_author_work_mention_plan",
                "deterministic",
            )
        )
    if _weak_conclusion(payload.get("conclusion_synthesis")):
        findings.append(
            _finding(
                BLUEPRINT_CONCLUSION_WEAK,
                "weak conclusion: conclusion_synthesis must synthesize source clusters into a downstream deck argument",
                "$.conclusion_synthesis",
                "LLM-judge",
            )
        )
    if not _slides_have_visual_roles(payload):
        findings.append(
            _finding(
                BLUEPRINT_VISUAL_ROLES_MISSING,
                "visual roles are required for every slide intent",
                "$.slides",
                "LLM-judge",
            )
        )
    if not _slides_have_narration_beats(payload):
        findings.append(
            _finding(
                BLUEPRINT_NARRATION_BEATS_MISSING,
                "narration beats are required for every slide intent",
                "$.slides",
                "LLM-judge",
            )
        )
    duration = _duration_budget_minutes(payload)
    if duration is not None and duration > BLUEPRINT_RUNTIME_HARD_BUDGET_MINUTES:
        findings.append(
            _finding(
                BLUEPRINT_DURATION_BUDGET_FAILED,
                "duration above 40 minutes cannot be approved",
                "$.duration_budget",
                "deterministic",
            )
        )
    if _requires_cluster_coverage_justification(payload) and not _has_cluster_coverage_justification(payload):
        findings.append(
            _finding(
                BLUEPRINT_SOURCE_COVERAGE_MISSING,
                "missing source coverage: slide counts above 14 require cluster_coverage_justification and a duration-budget pass",
                "$.cluster_coverage_justification",
                "deterministic",
            )
        )
    return findings


def _finding(code: str, message: str, path: str, gate_type: str) -> dict[str, str]:
    return {
        "code": code,
        "severity": "fail",
        "message": message,
        "path": path,
        "gate_type": gate_type,
    }


def _has_source_coverage(payload: dict[str, Any]) -> bool:
    source_plan = payload.get("source_paper_plan")
    input_hashes = payload.get("input_hashes")
    provenance = payload.get("provenance")
    source_metadata = provenance.get("source_metadata") if isinstance(provenance, dict) else None
    required_hashes = {
        "source_corpus_inventory.json",
        "speaker_author_work_ledger.json",
        "source_repair_findings.json",
        "research_cluster_plan.json",
    }
    return (
        isinstance(source_plan, list)
        and bool(source_plan)
        and isinstance(source_metadata, list)
        and bool(source_metadata)
        and isinstance(input_hashes, dict)
        and required_hashes.issubset(set(input_hashes))
    )


def _has_final_report_url(payload: dict[str, Any]) -> bool:
    for key in ("final_report_url", "analysis_report_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return True
    return False


def _has_speaker_author_work_plan(payload: dict[str, Any]) -> bool:
    plan = payload.get("speaker_author_work_mention_plan")
    if not isinstance(plan, list) or not plan:
        return False
    for item in plan:
        if not isinstance(item, dict):
            return False
        has_people_or_work = any(
            isinstance(item.get(key), list) and item[key]
            for key in ("speakers", "authors", "works")
        )
        if not has_people_or_work or not str(item.get("contribution_note") or "").strip():
            return False
    return True


def _weak_conclusion(value: Any) -> bool:
    if not isinstance(value, str):
        return True
    words = [word for word in value.strip().split() if word]
    return len(words) < 12


def _slides_have_visual_roles(payload: dict[str, Any]) -> bool:
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        return False
    return all(isinstance(slide, dict) and str(slide.get("visual_role") or "").strip() for slide in slides)


def _slides_have_narration_beats(payload: dict[str, Any]) -> bool:
    slides = payload.get("slides")
    if not isinstance(slides, list) or not slides:
        return False
    for slide in slides:
        if not isinstance(slide, dict):
            return False
        beats = slide.get("narration_beats")
        if not isinstance(beats, list) or not any(str(beat).strip() for beat in beats):
            return False
    return True


def _duration_budget_minutes(payload: dict[str, Any]) -> float | None:
    duration_budget = payload.get("duration_budget")
    if isinstance(duration_budget, dict):
        for key in ("target_minutes", "duration_minutes", "minutes"):
            value = _optional_float(duration_budget.get(key))
            if value is not None:
                return value
    return _optional_float(payload.get("target_runtime_minutes"))


def _requires_cluster_coverage_justification(payload: dict[str, Any]) -> bool:
    slide_count = payload.get("target_slide_count")
    return isinstance(slide_count, int) and not isinstance(slide_count, bool) and slide_count > 14


def _has_cluster_coverage_justification(payload: dict[str, Any]) -> bool:
    value = payload.get("cluster_coverage_justification")
    return isinstance(value, dict) and bool(value) and _duration_budget_minutes(payload) is not None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _review_status(findings: list[dict[str, Any]]) -> str:
    if any(finding.get("severity") == "fail" for finding in findings):
        return "fail"
    if any(finding.get("severity") == "warn" for finding in findings):
        return "warn"
    return "pass"


def _gate_status(gate_type: str, findings: list[dict[str, Any]]) -> str:
    scoped = [finding for finding in findings if finding.get("gate_type") == gate_type]
    if any(finding.get("severity") == "fail" for finding in scoped):
        return "fail"
    if any(finding.get("severity") == "warn" for finding in scoped):
        return "warn"
    if gate_type == "deterministic":
        return "pass"
    if gate_type == "LLM-judge" and any(
        finding.get("code") == BLUEPRINT_RELEVANCE_LLM_JUDGE_APPROVED for finding in scoped
    ):
        return "pass"
    if gate_type == "human" and any(finding.get("code") == BLUEPRINT_HUMAN_APPROVAL_RECORDED for finding in scoped):
        return "pass"
    if scoped:
        return "pending" if gate_type in {"LLM-judge", "human"} else "pass"
    return "not_applicable"


def _write_review_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Presentation Blueprint Review",
        "",
        f"- Status: `{payload['status']}`",
        f"- Reviewed at: `{payload['reviewed_at']}`",
        "",
        "## Findings",
    ]
    for finding in payload["findings"]:
        lines.append(
            f"- `{finding['severity']}` `{finding['gate_type']}` `{finding['code']}`: {finding['message']}"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
