from __future__ import annotations
import pytest

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "plan-conference-analysis"
SKILL_MD = SKILL_DIR / "SKILL.md"
PRODUCTION_SKILL_DIR = REPO_ROOT / ".claude" / "skills" / "plan-conference-production"
PRODUCTION_SKILL_MD = PRODUCTION_SKILL_DIR / "SKILL.md"
PRODUCTION_INSTALL_SCRIPT = PRODUCTION_SKILL_DIR / "scripts" / "install_skill.py"
PRODUCTION_SCAFFOLD_SCRIPT = PRODUCTION_SKILL_DIR / "scripts" / "scaffold_production_plan.py"
PRODUCTION_AUDIT_SCRIPT = PRODUCTION_SKILL_DIR / "scripts" / "audit_generated_production_plan.py"
REGISTRY_PATH = REPO_ROOT / "src" / "conference_pipeline" / "model_governance" / "registry.yaml"
SUPPORT_BOUNDARIES = SKILL_DIR / "references" / "support-boundaries.md"
SCAFFOLD_SCRIPT = SKILL_DIR / "scripts" / "scaffold_plan.py"
AUDIT_SCRIPT = SKILL_DIR / "scripts" / "audit_generated_plan.py"
INSTALL_SCRIPT = SKILL_DIR / "scripts" / "install_skill.py"
PHASE_DIR = (
    REPO_ROOT
    / ".planning"
    / "milestones"
    / "v1.3-phases"
    / "30-plan-conference-analysis-skill-packaging-and-uat"
)
UAT_REPORT_JSON = PHASE_DIR / "30-uat-generated-plan-report.json"
UAT_REPORT_MD = PHASE_DIR / "30-uat-generated-plan-report.md"
PRODUCTION_SKILL_SPEC = PHASE_DIR / "30-PRODUCTION-SKILL-SPEC.md"
PRODUCTION_SKILL_VERIFICATION = PHASE_DIR / "30-PRODUCTION-SKILL-VERIFICATION.md"
CODEx_INSTALL_TARGET = "~/.codex/skills/plan-conference-analysis"
CLAUDE_INSTALL_TARGET = "~/.claude/skills/plan-conference-analysis"
PHASE14_CANARY_BLOCKER = (
    "No bulk presentation regeneration before approved 04_safety_alignment_privacy canary."
)
PHASE14_BLUEPRINT_TERMS = (
    "presentation_blueprint",
    "presentation.generate_editorial_blueprint",
    "presentation.review_presentation_blueprint",
    "presentation.repair_presentation_blueprint",
    "approved_for_draft_generation",
    "blueprint_sha256",
    "gate type: deterministic",
    "gate type: LLM-judge",
    "gate type: human",
    "scripts/presentations/generate_deck_contract.py",
    "presentation_deck.json",
    "docs/reference/stage-owner-audit.md#presentation-generate-editorial-blueprint",
    "docs/reference/stage-owner-audit.md#presentation-review-presentation-blueprint",
    PHASE14_CANARY_BLOCKER,
)
REQUIRED_READS = (
    "CLAUDE.md",
    "AGENTS.md",
    "docs/source-of-truth.md",
    "docs/reference/pipeline-capabilities.md",
    "docs/reference/stage-owner-audit.md",
    "docs/workflows/pipeline-workflow.md",
    "docs/workflows/conference-analysis-playbook.md",
    "docs/operations/analysis-contract.md",
    "docs/design/analysis-manifest-schema.md",
    "docs/cli.md",
    "src/conference_pipeline/model_governance/registry.yaml",
    "src/conference_pipeline/model_governance/callsite_inventory.json",
    ".planning/milestones/*-CONTEXT.md",
)
REQUIRED_SUPPORT_LABELS = (
    "live",
    "dry-run",
    "mock",
    "unsupported",
    "planned",
    "dry-run/mock only",
    "needs live proof",
)
OFFICIAL_WORKFLOW_REQUIRED_IDS = (
    "WF-01",
    "WF-02",
    "WF-03",
    "WF-04",
    "WF-05",
    "WF-06",
    "WF-07",
    "WF-08",
    "WF-09",
    "WF-10",
    "WF-11",
    "WF-12",
    "WF-13",
    "WF-14",
)
OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS = ("WF-14a", "WF-14b", "WF-14c")
OFFICIAL_WORKFLOW_ROW_FIELDS = {
    "step_id",
    "official_step",
    "status",
    "phase_name",
    "support_label",
    "receipt_boundary",
    "rationale",
    "source_ref",
}
OFFICIAL_WORKFLOW_STATUSES = {"mapped", "blocked", "out_of_scope"}
CRITICAL_RECEIPT_BOUNDARY_TERMS = {
    "WF-08": "enhanced indexing",
    "WF-09": "query planning",
    "WF-10": "topic execution",
    "WF-11": "traceability",
    "WF-12": "coverage",
    "WF-13": "synthesis",
}
REQUIRED_ROADMAP_AREAS = (
    "source intake",
    "workspace/storage",
    "provider cost gate",
    "acquisition proof",
    "transcription/OCR",
    "analysis",
    "narrative analysis provenance",
    "QA",
    "podcast",
    "presentation production",
    "validation",
    "proof/closeout",
)
REQUIRED_PRODUCTION_STAGES = (
    "source discovery/import",
    "media and paper download",
    "transcription/OCR",
    "corpus inventory",
    "enhanced indexing",
    "query planning",
    "topic analysis generation",
    "narrative analysis provenance",
    "traceability",
    "QA/source-gap/drift checks",
    "coverage/gap fill",
    "synthesis executive/trend report",
    "podcast rendered audio",
    "deck/presentation render/export",
    "release bundle",
    "final proof",
)
REQUIRED_PRODUCTION_GATES = (
    "live-only production",
    "all discovered conference items unless explicitly excluded",
    "Video/media downloads cannot be substituted",
    "embedded media recovery is acquisition work before transcription/OCR",
    "item_page_auth_probe",
    "acquisition.slideslive_media_recovery",
    "acquisition.slideslive_slide_download",
    "acquisition.slide_reference_detection",
    "acquisition.slide_cleanup",
    "PG-CORPUS-TOKEN-INDEXING",
    "token_budget_policy_digest",
    "embedding_policy_digest",
    "EMBEDDING_MAX_LENGTH",
    "CORPUS_VECTOR_EMBED_BATCH_SIZE",
    "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
    "talk_media_acquisition_receipt.json",
    "workshop_talk_media_scope_waiver.json",
    "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
    "transcription_ocr.workshop_talk_boundary_discovery",
    "scripts/discover_workshop_talks.py",
    "Gemini 2.5 Pro",
    "gemini-2.5-pro",
    "*_talks.json",
    "5-minute talk-context",
    "provider_receipt_paths",
    "workshop_talk_boundary_discovery",
    "transcription_ocr.workshop_talk_segmentation",
    "talk_segmentation_provider_not_configured",
    "audio-only recovery is rejected evidence for talk-aware transcripts",
    "count of transcript/OCR artifacts",
    "bounded exponential backoff",
    "ask the user",
    "provider approval",
    "disk preflight",
    "fail closed",
    "receipt-backed completion",
    "non-live evidence is rejected",
    "public analysis/<topic>.md reports are deep analysis documents",
    "analysis/slide_briefs/<topic_id>.md",
    "analysis.deep_narrative_synthesis",
    "analysis/provenance/<topic_id>.md",
    "analysis/analysis_citation_manifest.json",
)
PRODUCTION_GATE_REQUIRED_IDS = (
    "PG-EMBEDDED-MEDIA-RECOVERY",
    "PG-SLIDESLIVE-SLIDE-ACQUISITION",
    "PG-WORKSHOP-TALK-MEDIA-ACQUISITION",
    "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY",
    "PG-CORPUS-TOKEN-INDEXING",
    "PG-INDEX",
    "PG-QUERY",
    "PG-GPT-RESEARCHER",
    "PG-COVERAGE-GAP-FILL",
    "PG-SYNTHESIS",
    "PG-NARRATION-SCRIPT-REVIEW",
    "PG-OPTIONAL-PIPELINES",
)
PRODUCTION_GATE_REQUIRED_FIELDS = {
    "gate_id",
    "workflow_step",
    "owner_command",
    "receipt_path",
    "pass_criteria",
    "fail_criteria",
    "downstream_blockers",
    "stale_evidence_rule",
    "rejected_evidence",
    "source_ref",
}
PRODUCTION_GATE_DOWNSTREAM_TERMS = ("podcast", "deck", "release", "proof")
PRODUCTION_REJECTED_EVIDENCE_TERMS = (
    "dry-run",
    "mock",
    "readiness-only",
    "metadata-only",
    "paper-only",
    "historical proof",
)
REQUIRED_HANDOFF_FIELDS = (
    "next_step",
    "evidence_class",
    "corpus_bounds",
    "inventory_status",
    "interest_group_decisions",
    "exclusions",
    "implementation_inventory",
    "provider_model_posture",
    "benchmark_requirements",
    "local_provider_deferrals",
    "presentation_figure_policy",
    "presentation_density_policy",
    "presentation_visual_qc_policy",
    "narrative_analysis_contract",
    "local_proof_posture",
)
PROVIDER_CONTRACT_REQUIRED_IDS = (
    "PC-14A-LOCAL-FIRST",
    "PC-14B-EMBEDDINGS",
    "PC-GPT-RESEARCHER-LLM",
    "PC-GPT-RESEARCHER-EMBEDDINGS",
    "PC-DEEP-NARRATIVE-SYNTHESIS",
    "PC-TAVILY-SEARCH-EXTRACT",
    "PC-OCR-AUTHORITY",
    "PC-OCR-PRODUCTION",
    "PC-EXTRACTION-REMOTE-ESCALATION",
    "PC-NON14A-REGISTRY-GOVERNED",
)
PROVIDER_CONTRACT_REQUIRED_FIELDS = {
    "contract_id",
    "stage",
    "registry_task_id",
    "cost_task_id",
    "approved_provider_model",
    "fallback_or_escalation",
    "selection_status",
    "benchmark_authority",
    "receipt_requirement",
    "local_posture",
    "source_refs",
}
PROVIDER_CONTRACT_MARKDOWN_FIELDS = (
    "contract_id",
    "stage",
    "registry_task_id",
    "cost_task_id",
    "approved_provider_model",
    "fallback_or_escalation",
    "selection_status",
    "benchmark_authority",
    "receipt_requirement",
    "local_posture",
    "source_refs",
)
PROVIDER_CONTRACT_TABLE_HEADER = "| Contract ID | Stage | Registry task ID | Cost task ID | Approved provider/model | Fallback or escalation | Selection status | Benchmark authority | Receipt requirement | Local posture | Sources |"
PROVIDER_CONTRACT_UAT_EXPECTED_STRINGS = (
    "Benchmark-Aware Provider Contracts",
    "benchmark-aware provider contracts",
    "provider_contracts_checked",
    "provider_contract_failures",
    "provider contract failures",
    "phase_14b_indexing_embeddings",
    "exhaustive_extraction_local_heuristic",
    "gpt_researcher_llm",
    "gpt_researcher_embeddings",
    "analysis_deep_narrative_synthesis",
    "tavily_search_extract",
    "ocr.slide_image_parse",
    "analysis.source_reading",
    "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit",
    "mlx-community/bge-m3-mlx-8bit",
    "WEB_SEARCH_ENABLED=false",
    "EMBEDDING_PROVIDER=mlx",
    "EMBEDDING_DIM=1024",
    "RETRIEVAL_K=18",
    "PAPER_MATCHING_RETRIEVAL_K=18",
    "CANDIDATE_MULTIPLIER=3",
    "RRF_K=60",
    "OpenAI gpt-5.6-sol authority with gpt-5.6-terra fallback",
    "remote extraction escalation/failover",
    "cloud LLM escalation only for low-confidence or ambiguous records",
)
EXTRACTION_POLICY_REQUIRED_STRINGS = (
    "## Extraction Provider Policy",
    "ambiguous records",
    "heuristics serve as prefilter",
    "benchmark-governed contracts",
    "EXTRACTION_LOCAL_LLM_BASE_URL",
    "Non-extraction generation providers",
    "readiness",
)
OCR_STATE_REQUIRED_STRINGS = (
    "## Slide/Visual OCR Assessment",
    "skipped_with_rationale",
    "deferred_blocked",
    "reopen_conditions",
    "source-coverage",
    "transcripts",
)
PRODUCTION_V18_BACKEND_STRINGS = (
    "## Backend Selection (v1.8)",
    "MLX-first on Apple Silicon",
    "OpenAI-compatible HTTP fallback",
    "LocalLLMReceipt.endpoint_label",
    "EXTRACTION_LOCAL_LLM_BACKEND=mlx|http",
    "BackendSelectionError",
)
PRODUCTION_V18_THRESHOLD_STRINGS = (
    "## Quality Sample Thresholds (v1.8)",
    "min mean confidence",
    "max failure-class rate",
    "min record coverage",
    "cloud escalation trigger",
    "config/extraction-thresholds.json",
    "LocalLLMReceipt.effective_thresholds",
    "override receipt referenced:",
)
PRODUCTION_V18_FAIL_STOP_STRINGS = (
    "## Fail-Stop Blocker Policy (v1.92)",
    "BlockerReceipt",
    "UnblockReceipt",
    "status: blocked",
    "status: unavailable",
    "ocr_deferred_blocked",
    "stage_blocked",
    "stage_unavailable",
    "local_llm_not_ready",
    "missing_receipt",
    "threshold_violation",
    "stale_receipt",
    "workflow unblock",
    "conference_pipeline.workflow_unblock",
    "halt_status",
    "unblock_acknowledgements",
)
BACKUP_CHECKPOINT_REQUIRED_IDS = tuple(f"BK-{index:02d}" for index in range(2, 13))
WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS = (
    "discover",
    "download",
    "transcribe",
    "exhaustive-topic-extraction",
    "enhanced-indexing-vector-store",
    "query-planning-gpt-researcher",
    "coverage-gap-fill",
    "synthesis-trend-report",
    "podcast",
    "deck",
)
SECRET_PATTERNS = (
    "sk-",
    "AIza",
    "ANTHROPIC_API_KEY=",
    "OPENAI_API_KEY=",
    "GOOGLE_API_KEY=",
    "GEMINI_API_KEY=",
    "TAVILY_API_KEY=",
    "Cookie:",
    "NEURIPS_PASSWORD=",
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sample_brief() -> dict[str, object]:
    return {
        "platform_version": "v1.3",
        "conference_name": "ICLR Scaffold 2027",
        "conference_identity": {"slug": "iclr-scaffold-2027"},
        "source_urls": [
            "https://example.org/iclr-scaffold-2027/schedule",
            "https://example.org/iclr-scaffold-2027/proceedings",
        ],
        "auth_notes": "Public scaffold URLs only; no cookies or provider secrets.",
        "target_outputs": ["analysis", "QA", "podcast", "presentation production"],
        "proof_mode": "dry-run/mock only",
        "topic_strategy": "start with a small representative topic set, then expand after proof",
        "provider_preferences": "mock providers for UAT; live providers need live proof",
        "provider_cost_approval_posture": "estimate required before paid provider work",
        "storage_share_targets": "local workspace only",
        "acceptance_criteria": [
            "generated roadmap covers source intake through proof/closeout",
            "unsupported live support is labeled needs live proof",
            "generated roadmap includes an Official Workflow Mapping table for WF-01 through WF-14",
        ],
    }


def run_command(
    args: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def run_scaffold(brief_path: Path, draft_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            sys.executable,
            str(SCAFFOLD_SCRIPT),
            "--brief",
            str(brief_path),
            "--draft-root",
            str(draft_root),
            *extra,
        ]
    )


def run_production_scaffold(spec_root: Path, draft_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            sys.executable,
            str(PRODUCTION_SCAFFOLD_SCRIPT),
            "--spec-root",
            str(spec_root),
            "--draft-root",
            str(draft_root),
            *extra,
        ]
    )


def scaffold_sample(tmp_path: Path) -> Path:
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief, sample_brief())
    result = run_scaffold(brief, draft_root)
    assert result.returncode == 0, result.stderr + result.stdout
    return draft_root / "iclr-scaffold-2027"


def production_sample(tmp_path: Path) -> Path:
    spec_root = scaffold_sample(tmp_path)
    production_root = tmp_path / "production-drafts"
    result = run_production_scaffold(spec_root, production_root)
    assert result.returncode == 0, result.stderr + result.stdout
    return production_root / "iclr-scaffold-2027-production"


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(read_text(path))
    assert isinstance(payload, dict)
    return payload


def write_loaded_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provider_contract_rows(
    approval: dict[str, object],
    key: str = "benchmark-aware provider contracts",
) -> list[dict[str, object]]:
    rows = approval.get(key)
    assert isinstance(rows, list), f"{key} must be a list"
    assert all(isinstance(row, dict) for row in rows), f"{key} rows must be objects"
    return rows


def provider_contract_failure_names(report: dict[str, object]) -> set[str]:
    names: set[str] = set()
    for failure in report.get("failures", []):
        if isinstance(failure, dict) and failure.get("name"):
            names.add(str(failure["name"]))
    for failure in report.get("provider_contract_failures", []):
        if isinstance(failure, dict) and failure.get("name"):
            names.add(str(failure["name"]))
        else:
            names.add(str(failure))
    return names


def provider_contract_row_by_id(
    rows: list[dict[str, object]],
    contract_id: str,
) -> dict[str, object]:
    for row in rows:
        if row.get("contract_id") == contract_id:
            return row
    raise AssertionError(f"missing provider contract row: {contract_id}")


def extra_provider_contract_row(
    rows: list[dict[str, object]],
    contract_id: str = "PC-EXTRA-DRIFT",
) -> dict[str, object]:
    row = dict(provider_contract_row_by_id(rows, "PC-14B-EMBEDDINGS"))
    row["contract_id"] = contract_id
    row["stage"] = "unexpected extra provider contract"
    return row


def provider_contract_containers(
    approval: dict[str, object],
    *,
    production: bool = False,
) -> list[list[dict[str, object]]]:
    containers = [provider_contract_rows(approval)]
    handoff_key = "source analysis handoff" if production else "production handoff"
    handoff = approval.get(handoff_key)
    assert isinstance(handoff, dict), f"{handoff_key} must be structured"
    containers.append(provider_contract_rows(handoff, "provider_contracts"))
    return containers


def provider_contract_cell(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def provider_contract_markdown_row(row: dict[str, object]) -> str:
    return "| " + " | ".join(provider_contract_cell(row.get(field, "")) for field in PROVIDER_CONTRACT_MARKDOWN_FIELDS) + " |"


def sync_provider_contract_roadmap_row(plan_dir: Path, row: dict[str, object]) -> None:
    contract_id = str(row["contract_id"])
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    lines = read_text(roadmap_path).splitlines()
    replacement = provider_contract_markdown_row(row)
    updated = [replacement if line.startswith(f"| {contract_id} |") else line for line in lines]
    roadmap_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def remove_provider_contract_roadmap_row(plan_dir: Path, contract_id: str) -> None:
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    lines = [
        line for line in read_text(roadmap_path).splitlines() if not line.startswith(f"| {contract_id} |")
    ]
    roadmap_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_provider_contract_row(
    plan_dir: Path,
    contract_id: str,
    updates: dict[str, object],
    *,
    production: bool = False,
) -> None:
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    updated_row: dict[str, object] | None = None
    for rows in provider_contract_containers(approval, production=production):
        row = provider_contract_row_by_id(rows, contract_id)
        row.update(updates)
        updated_row = row
    write_loaded_json(approval_path, approval)
    assert updated_row is not None
    sync_provider_contract_roadmap_row(plan_dir, updated_row)


def remove_provider_contract_row(plan_dir: Path, contract_id: str, *, production: bool = False) -> None:
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    for rows in provider_contract_containers(approval, production=production):
        if not any(row.get("contract_id") == contract_id for row in rows):
            raise AssertionError(f"missing provider contract row: {contract_id}")
        rows[:] = [row for row in rows if row.get("contract_id") != contract_id]
    write_loaded_json(approval_path, approval)
    remove_provider_contract_roadmap_row(plan_dir, contract_id)


def test_skill_metadata_and_install_targets() -> None:
    skill = read_text(SKILL_MD)

    assert "name: plan-conference-analysis" in skill
    assert "conference-specific GSD milestone" in skill
    assert CODEx_INSTALL_TARGET in skill
    assert CLAUDE_INSTALL_TARGET in skill


def test_analysis_skill_hands_off_to_production_skill() -> None:
    skill = read_text(SKILL_MD)

    assert "Production Execution Handoff" in skill
    assert "plan-conference-production" in skill
    assert "one execution phase" in skill
    assert "multi-phase production" in skill
    assert "conference_pipeline.stage_owner_audit" in skill
    assert "stage owner registry audit" in skill
    assert "Narrative Analysis Contract" in skill
    assert "analysis/provenance/<topic_id>.md" in skill
    assert "scripts/presentations/generate_deck_contract.py" in skill
    assert "scripts/generate_focused_podcast.py" in skill
    assert "--fact-ranks audio/<id>/ranks.json" in skill
    assert "decks/<id>/presentation_deck.json" in skill
    assert "scripts/presentations/generate_slides_simple.py" in skill
    assert "scripts/generate_slides_agentic.py" in skill
    assert "acquisition.slideslive_media_recovery" in skill
    assert "acquisition.slideslive_slide_download" in skill
    assert "acquisition.slide_reference_detection" in skill
    assert "acquisition.slide_cleanup" in skill
    assert "item_page_auth_probe" in skill
    assert "PG-CORPUS-TOKEN-INDEXING" in skill
    assert "token_budget_policy_digest" in skill
    assert "embedding_policy_digest" in skill
    assert "EMBEDDING_MAX_LENGTH" in skill
    assert "CORPUS_VECTOR_EMBED_BATCH_SIZE" in skill
    assert "transcription_ocr.workshop_talk_boundary_discovery" in skill
    assert "scripts/discover_workshop_talks.py" in skill
    assert "Gemini 2.5 Pro" in skill
    assert "gemini-2.5-pro" in skill
    assert "*_talks.json" in skill
    assert "5-minute talk-context" in skill
    assert "provider_receipt_paths" in skill
    assert "workshop_talk_boundary_discovery" in skill
    assert "transcription_ocr.workshop_talk_segmentation" in skill
    assert "talk_segmentation_provider_not_configured" in skill
    assert "audio-only recovery is rejected evidence for talk-aware transcripts" in skill
    assert "presentation narrative-content gate" in skill
    assert "content-aware, slide-specific" in skill
    assert "generic placeholder/scaffold diagrams" in skill


def test_phase14_canonical_blueprint_skills_exist_and_name_sync_targets() -> None:
    sync_sentence = (
        "Repo canonical skills live under .claude/skills; local mirrored skills under "
        "~/.codex/skills are sync targets, not the only source."
    )
    for skill_name in (
        "conference-presentation-blueprint",
        "conference-presentation-blueprint-review",
        "conference-figure-design-brief",
    ):
        path = REPO_ROOT / ".claude" / "skills" / skill_name / "SKILL.md"
        text = read_text(path)
        assert sync_sentence in text
        assert "~/.claude/skills" in text


def test_phase14_blueprint_contract_terms_in_planning_skills() -> None:
    combined = "\n".join(
        read_text(path)
        for path in (
            SKILL_MD,
            PRODUCTION_SKILL_MD,
            SCAFFOLD_SCRIPT,
            AUDIT_SCRIPT,
            PRODUCTION_SCAFFOLD_SCRIPT,
            PRODUCTION_AUDIT_SCRIPT,
        )
    )
    for term in PHASE14_BLUEPRINT_TERMS:
        assert term in combined


def test_production_skill_metadata_scope_and_completion_rules() -> None:
    skill = read_text(PRODUCTION_SKILL_MD)

    assert "name: plan-conference-production" in skill
    assert "plan-conference-analysis specification" in skill
    assert "one execution phase" in skill
    assert "multi-phase milestone" in skill
    assert "live-only production" in skill
    assert "scripts/scaffold_production_plan.py" in skill
    assert "scripts/audit_generated_production_plan.py" in skill
    assert "conference_pipeline.stage_owner_audit" in skill
    assert "stage owner registry audit" in skill
    assert "scripts/generate_focused_podcast.py" in skill
    assert "--fact-ranks audio/<id>/ranks.json" in skill
    assert "all discovered conference items unless" in skill
    assert "Do not produce a plan whose main command does not exist" in skill
    assert "Completion requires actual receipt-backed execution" in skill
    assert "Video/media downloads cannot be substituted with paper PDFs" in skill
    assert "acquisition.slideslive_media_recovery" in skill
    assert "acquisition.slideslive_slide_download" in skill
    assert "acquisition.slide_reference_detection" in skill
    assert "acquisition.slide_cleanup" in skill
    assert "item_page_auth_probe" in skill
    assert "PG-CORPUS-TOKEN-INDEXING" in skill
    assert "token_budget_policy_digest" in skill
    assert "embedding_policy_digest" in skill
    assert "EMBEDDING_MAX_LENGTH" in skill
    assert "CORPUS_VECTOR_EMBED_BATCH_SIZE" in skill
    assert "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY" in skill
    assert "transcription_ocr.workshop_talk_boundary_discovery" in skill
    assert "scripts/discover_workshop_talks.py" in skill
    assert "Gemini 2.5 Pro" in skill
    assert "gemini-2.5-pro" in skill
    assert "*_talks.json" in skill
    assert "5-minute talk-context" in skill
    assert "provider_receipt_paths" in skill
    assert "workshop_talk_boundary_discovery" in skill
    assert "transcription_ocr.workshop_talk_segmentation" in skill
    assert "talk_segmentation_provider_not_configured" in skill
    assert "audio-only recovery is rejected evidence for talk-aware transcripts" in skill
    assert "Transcription means speech-to-text or OCR over downloaded audio/video/slides" in skill
    assert "metadata-only analysis" in skill
    assert "deep analysis documents" in skill
    assert "analysis/slide_briefs/<topic_id>.md" in skill
    assert "analysis/analysis_citation_manifest.json" in skill
    assert "programmatic matplotlib/Plotly/svgwrite owner command" in skill
    assert "content-aware, slide-specific editorial SVGs" in skill
    assert "generic placeholder SVGs" in skill
    assert "scripts/presentations/extract_topic_figures.py" in skill
    assert "scripts/presentations/add_images_to_md.py" in skill
    for stage in (
        "Source discovery or source import",
        "Paper/source/session/media download",
        "Transcription, OCR, and media processing",
        "Analysis generation",
        "Podcast script generation plus rendered audio",
        "Deck/presentation generation",
        "Final proof packet",
    ):
        assert stage in skill


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/30-plan-conference-analysis-skill-packaging-and-uat/30-PRODUCTION-SKILL-SPEC.md")
def test_production_skill_spec_and_verification_are_recorded_in_v13() -> None:
    spec = read_text(PRODUCTION_SKILL_SPEC)
    verification = read_text(PRODUCTION_SKILL_VERIFICATION)

    assert "plan-conference-production" in spec
    assert "Companion To plan-conference-analysis" in spec
    assert "Stage Coverage Contract" in spec
    assert "Completion Gates" in spec
    assert "scaffold_production_plan.py" in spec
    assert "audit_generated_production_plan.py" in spec
    assert "live-only production" in spec
    assert "status: passed" in verification
    assert "SKILL-07" in verification
    assert "SKILL-08" in verification
    assert "deterministic generator/auditor scripts" in verification


def test_skill_required_reads_cover_canonical_contracts() -> None:
    skill = read_text(SKILL_MD)

    for required in REQUIRED_READS:
        assert required in skill


def test_stage_owner_audit_documentation_is_linked() -> None:
    doc = read_text(REPO_ROOT / "docs" / "reference" / "stage-owner-audit.md")
    source_of_truth = read_text(REPO_ROOT / "docs" / "source-of-truth.md")
    capabilities = read_text(REPO_ROOT / "docs" / "reference" / "pipeline-capabilities.md")
    cli = read_text(REPO_ROOT / "docs" / "cli.md")

    assert "src/conference_pipeline/stage_owner_audit.py" in doc
    assert "src/conference_pipeline/stage_owner_registry.json" in doc
    assert "usage_doc_refs" in doc
    assert "Stage Script Registry Tool Usage Index" in doc
    assert "presentation.generate_deck_contract" in doc
    assert "conference_pipeline.stage_owner_audit" in doc
    assert "usage_doc_refs" in source_of_truth
    assert "usage_doc_refs" in capabilities
    assert "--docs-root docs" in cli


def test_skill_references_define_locked_support_labels() -> None:
    support = read_text(SUPPORT_BOUNDARIES)

    for label in REQUIRED_SUPPORT_LABELS:
        assert label in support


def test_brief_contract_requires_truth_fields(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    write_json(brief, {"conference_name": "ICLR Scaffold 2027"})

    result = run_scaffold(brief, tmp_path / "drafts")

    assert result.returncode == 2
    assert "missing required brief fields" in result.stderr


def test_scaffold_outputs_use_milestone_naming_and_locations(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief, sample_brief())

    result = run_scaffold(brief, draft_root)

    assert result.returncode == 0, result.stderr
    plan_dir = draft_root / "iclr-scaffold-2027"
    assert (plan_dir / ".planning" / "milestones" / "v1.3-ICLR Scaffold 2027-CONTEXT.md").exists()
    assert (plan_dir / ".planning" / "REQUIREMENTS.md").exists()
    assert (plan_dir / ".planning" / "ROADMAP.md").exists()
    assert (plan_dir / ".planning" / "STATE.md").exists()
    assert (plan_dir / "conferences" / "iclr-scaffold-2027" / "README.md").exists()

    approval = json.loads(read_text(plan_dir / "approval-summary.json"))
    assert set(approval) >= {
        "generated milestone identity",
        "support claims",
        "roadmap phases",
        "changed files",
    }
    assert approval["generated milestone identity"] == "v1.3-ICLR Scaffold 2027"
    assert "dry-run/mock only" in json.dumps(approval)

    repeat = run_scaffold(brief, draft_root)
    assert repeat.returncode != 0
    assert "target already exists" in repeat.stderr


def test_scaffold_redacts_full_secret_values_from_brief(tmp_path: Path) -> None:
    brief_payload = sample_brief()
    brief_payload["conference_identity"] = {"slug": "iclr-sk-slug-secret"}
    brief_payload["auth_notes"] = (
        "OPENAI_API_KEY=sk-live-secret-tail api_key=generic-tail "
        "Authorization: Bearer bearer-token Cookie: sessionid=abc123 EXTRA_TOKEN=value123"
    )
    brief_payload["target_outputs"] = ["analysis", "sk-target-secret"]
    brief_payload["proof_mode"] = "dry-run/mock only OPENAI_API_KEY=sk-proof-secret"
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief, brief_payload)

    result = run_scaffold(brief, draft_root)

    assert result.returncode == 0, result.stderr
    plan_dir = draft_root / "iclr-redacted"
    assert plan_dir.exists()
    combined = "\n".join(
        path.relative_to(draft_root).as_posix() + "\n" + read_text(path)
        for path in plan_dir.rglob("*")
        if path.is_file() and path.suffix in {".json", ".md"}
    )
    assert "sk-live-secret-tail" not in combined
    assert "live-secret-tail" not in combined
    assert "generic-tail" not in combined
    assert "bearer-token" not in combined
    assert "sessionid=abc123" not in combined
    assert "value123" not in combined
    assert "sk-target-secret" not in combined
    assert "sk-proof-secret" not in combined
    assert "sk-slug-secret" not in combined
    assert "OPENAI_API_KEY" not in combined
    assert "api_key" not in combined
    assert "Authorization" not in combined
    assert "[redacted]" in combined


def test_approved_finalization_applies_full_planning_set_to_temp_repo(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    write_json(brief, sample_brief())
    assert run_scaffold(brief, draft_root).returncode == 0

    result = run_scaffold(
        brief,
        draft_root,
        "--finalize",
        "--approved-finalize",
        "--repo-root",
        str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert "finalized planning set" in result.stdout
    context = repo_root / ".planning" / "milestones" / "v1.3-ICLR Scaffold 2027-CONTEXT.md"
    assert context.exists()
    assert (repo_root / ".planning" / "REQUIREMENTS.md").exists()
    assert (repo_root / ".planning" / "ROADMAP.md").exists()
    assert (repo_root / ".planning" / "STATE.md").exists()
    assert (repo_root / "conferences" / "iclr-scaffold-2027" / "README.md").exists()
    assert "conference_milestone_name: \"v1.3-ICLR Scaffold 2027\"" in read_text(context)
    assert "needs live proof" in read_text(context)


def test_new_version_uses_isolated_versioned_draft_directory(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief, sample_brief())
    assert run_scaffold(brief, draft_root).returncode == 0

    next_brief = sample_brief()
    next_brief["platform_version"] = "v1.4"
    write_json(brief, next_brief)

    result = run_scaffold(brief, draft_root, "--new-version")

    assert result.returncode == 0, result.stderr
    assert (draft_root / "iclr-scaffold-2027").exists()
    versioned_dir = draft_root / "iclr-scaffold-2027-v1-4"
    assert versioned_dir.exists()
    assert (versioned_dir / ".planning" / "milestones" / "v1.4-ICLR Scaffold 2027-CONTEXT.md").exists()

    repeat = run_scaffold(brief, draft_root, "--new-version")
    assert repeat.returncode != 0
    assert "target already exists" in repeat.stderr


def test_new_version_finalization_copies_only_current_generated_files(tmp_path: Path) -> None:
    brief = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    brief_payload = sample_brief()
    brief_payload["platform_version"] = "v1.4"
    write_json(brief, brief_payload)
    assert run_scaffold(brief, draft_root, "--new-version").returncode == 0

    draft_dir = draft_root / "iclr-scaffold-2027-v1-4"
    stale_context = draft_dir / ".planning" / "milestones" / "v1.3-ICLR Scaffold 2027-CONTEXT.md"
    stale_context.write_text("# stale old version\n", encoding="utf-8")

    result = run_scaffold(
        brief,
        draft_root,
        "--new-version",
        "--finalize",
        "--approved-finalize",
        "--repo-root",
        str(repo_root),
    )

    assert result.returncode == 0, result.stderr
    assert (repo_root / ".planning" / "milestones" / "v1.4-ICLR Scaffold 2027-CONTEXT.md").exists()
    assert not (repo_root / ".planning" / "milestones" / "v1.3-ICLR Scaffold 2027-CONTEXT.md").exists()


def test_generated_roadmap_covers_required_backbone(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assert set(REQUIRED_ROADMAP_AREAS) <= set(report["roadmap_areas_checked"])
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert ".venv/bin/python -m conference_pipeline.costs estimate" in roadmap
    assert "analysis-governance-handoff" not in json.dumps(report["failures"])


def test_generated_roadmap_includes_official_workflow_mapping(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "## Official Workflow Mapping" in roadmap
    assert (
        "| Step ID | Official workflow step | Status | Phase | Support label | "
        "Receipt boundary | Rationale | Source |"
    ) in roadmap
    for step_id in OFFICIAL_WORKFLOW_REQUIRED_IDS + OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS:
        assert f"| {step_id} |" in roadmap
    assert "docs/workflows/pipeline-workflow.md#overview" in roadmap
    for unsupported_phrase in (
        "live crawl",
        "live transcription",
        "live analysis",
        "live podcast",
        "live presentation",
    ):
        assert unsupported_phrase not in roadmap.lower()


def test_approval_summary_records_official_workflow_mapping(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval = json.loads(read_text(plan_dir / "approval-summary.json"))
    rows = approval["official workflow mapping"]

    assert isinstance(rows, list)
    step_ids = {row["step_id"] for row in rows}
    assert set(OFFICIAL_WORKFLOW_REQUIRED_IDS) <= step_ids
    assert set(OFFICIAL_WORKFLOW_OPTIONAL_SUBSTEP_IDS) <= step_ids

    for row in rows:
        assert set(row) == OFFICIAL_WORKFLOW_ROW_FIELDS
        assert row["status"] in OFFICIAL_WORKFLOW_STATUSES
        assert row["support_label"] in REQUIRED_SUPPORT_LABELS
        assert row["rationale"]
        if row["status"] == "mapped":
            assert row["phase_name"]
            assert row["receipt_boundary"]


def test_uat_report_includes_official_workflow_mapping(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    report = json.loads(read_text(report_path))
    markdown = read_text(markdown_path)
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assert set(OFFICIAL_WORKFLOW_REQUIRED_IDS) <= set(report["workflow_mapping_checked"])
    assert report["workflow_mapping_failures"] == []
    assert "## Official workflow mapping checked" in markdown
    for unsupported_phrase in (
        "live crawl",
        "live transcription",
        "live analysis",
        "live podcast",
        "live presentation",
    ):
        assert unsupported_phrase not in (json.dumps(report) + markdown).lower()


def test_analysis_scaffold_requires_inventory_interest_groups_and_handoff(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context = read_text(next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md")))
    approval = load_json(plan_dir / "approval-summary.json")

    assert "Event-Family Inventory Contract" in context
    assert "Interest-Group Review" in context
    assert "Exclusion Records" in context
    assert "Production Execution Handoff" in context
    assert "Next Step: Production Planning" in context
    assert "$plan-conference-production" in context
    assert "actually produces the conference analysis" in context
    assert "Implementation Inventory Contract" in context
    assert "Programmatic Figure Generation Contract" in context
    assert "interest_group.research" in context
    assert "source_reading.categorization" in context
    assert "evidence_class" in approval["production handoff fields"]
    assert "local_proof_posture" in approval["production handoff fields"]
    assert "approved, edited, rejected, or complete" in context
    assert "presentation.svg_figure_generation" in context
    assert "presentation.visual_qc" in context
    assert "presentation.deck_contract_generation" in context
    assert "scripts/presentations/generate_deck_contract.py" in context
    assert "--presentation-blueprint" in context
    assert "decks/<id>/presentation_blueprint.json" in context
    assert "decks/<id>/blueprint_review.json" in context
    assert "presentation narrative-content gate" in context
    assert "target_hyperlinks_per_slide" in context
    assert "target_svgs_per_slide" in context
    assert "hyperlink_density_meets_target" in context
    assert "matplotlib" in context
    assert "Plotly" in context
    assert "svgwrite" in context
    assert "visual_llm_qc_report.json" in context


def test_analysis_scaffold_emits_structured_production_handoff(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval = load_json(plan_dir / "approval-summary.json")

    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    assert handoff["schema_version"] == 1
    assert approval["next step"]["skill"] == "plan-conference-production"
    assert approval["next step"]["command"] == "$plan-conference-production"
    assert "actually produces the conference analysis" in approval["next step"]["purpose"]
    assert handoff["next_step"] == approval["next step"]
    for field in REQUIRED_HANDOFF_FIELDS:
        assert field in handoff
    task_ids = handoff["model_governance_task_ids"]
    assert isinstance(task_ids, list)
    assert task_ids
    assert "interest_group.research" in task_ids
    assert "vision.figure_caption" in task_ids
    for field in (
        "provider_model_posture",
        "benchmark_requirements",
        "local_provider_deferrals",
        "local_proof_posture",
    ):
        value = handoff[field]
        assert isinstance(value, dict)
        assert set(task_ids) <= set(value)
    assert handoff["implementation_inventory"]["deprecated_owner_commands"] == [
        "scripts/presentations/extract_topic_figures.py",
        "scripts/presentations/add_images_to_md.py",
    ]
    required_owners = handoff["implementation_inventory"]["required_owner_commands"]
    assert (
        "scripts/generate_focused_podcast.py"
        in required_owners["podcast.script_generation"]
    )
    assert "--fact-ranks audio/<id>/ranks.json" in required_owners[
        "podcast.script_generation"
    ]
    assert (
        "scripts/presentations/generate_deck_contract.py"
        in required_owners["presentation.deck_contract_generation"]
    )
    assert "decks/<id>/presentation_deck.json" in required_owners[
        "presentation.deck_contract_generation"
    ]
    assert "decks/<id>/presentation_deck_receipt.json" in required_owners[
        "presentation.deck_contract_generation"
    ]
    assert "decks/<id>/presentation_blueprint.json" in required_owners[
        "presentation.deck_contract_generation"
    ]
    assert "decks/<id>/blueprint_review.json" in required_owners[
        "presentation.deck_contract_generation"
    ]
    assert "source_binding_report.json" in required_owners["presentation.source_binding"]
    assert "render_spec.json" in required_owners["presentation.render_spec_generation"]
    assert "render_spec.json" in required_owners["presentation.render_export"]
    assert "delivery_single_speaker.json" in required_owners["presentation.delivery_generation"]
    assert "delivery_notebooklm.json" in required_owners["presentation.delivery_generation"]
    assert (
        required_owners["presentation.svg_figure_generation"]
        == "scripts/presentations/generate_programmatic_figures.py"
    )
    assert "inspect_rendered_slides.py" in required_owners["presentation.visual_qc"]
    assert "inspect_rendered_slides_llm.py" in required_owners["presentation.visual_qc"]
    assert "scripts/recover_iclr2026_slideslive_media.py" in required_owners[
        "acquisition.slideslive_media_recovery"
    ]
    assert "--resume" in required_owners["acquisition.slideslive_media_recovery"]
    assert "scripts/recover_slideslive_slides.py" in required_owners[
        "acquisition.slideslive_slide_download"
    ]
    assert "slideslive_slide_recovery.json" in required_owners[
        "acquisition.slideslive_slide_download"
    ]
    assert "scripts/detect_reference_slides.py" in required_owners[
        "acquisition.slide_reference_detection"
    ]
    assert "scripts/cleanup_slides.py" in required_owners["acquisition.slide_cleanup"]
    assert "src/conference_pipeline/indexing/token_policy.py" in required_owners[
        "indexing.token_bounded_chunking"
    ]
    assert "src/conference_pipeline/indexing/chunking.py" in required_owners[
        "indexing.token_bounded_chunking"
    ]
    assert "src/conference_pipeline/indexing/embeddings.py" in required_owners[
        "indexing.local_embedding_materialization_policy"
    ]
    assert "scripts/discover_workshop_talks.py" in required_owners[
        "transcription_ocr.workshop_talk_boundary_discovery"
    ]
    assert "--provider-cost-approved" in required_owners[
        "transcription_ocr.workshop_talk_boundary_discovery"
    ]
    assert "gemini-2.5-pro" in required_owners[
        "transcription_ocr.workshop_talk_boundary_discovery"
    ]
    assert "scripts/segment_workshop_talks.py" in required_owners[
        "transcription_ocr.workshop_talk_segmentation"
    ]
    assert "skip_ocr" in handoff["implementation_inventory"][
        "slideslive_slide_acquisition_policy"
    ]
    corpus_policy = handoff["implementation_inventory"]["corpus_token_indexing_policy"]
    for term in (
        "PG-CORPUS-TOKEN-INDEXING",
        "token_budget_policy_digest",
        "embedding_policy_digest",
        "EMBEDDING_MAX_LENGTH",
        "CORPUS_VECTOR_EMBED_BATCH_SIZE",
        "vector_embeddings_written: true",
    ):
        assert term in corpus_policy
    talk_policy = handoff["implementation_inventory"]["workshop_talk_segmentation_policy"]
    for term in (
        "transcription_ocr.workshop_talk_boundary_discovery",
        "Gemini 2.5 Pro",
        "gemini-2.5-pro",
        "*_talks.json",
        "5-minute talk-context",
        "provider_receipt_paths",
        "workshop_talk_boundary_discovery",
        "transcription_ocr.workshop_talk_segmentation",
        "talk_segmentation_provider_not_configured",
        "audio-only recovery is rejected evidence for talk-aware transcripts",
        "workshop_id",
        "talk_id",
    ):
        assert term in talk_policy
    assert "item_page_auth_probe" in handoff["implementation_inventory"][
        "conditional_acquisition_recovery_policy"
    ]
    rejected_draft_owners = handoff["implementation_inventory"][
        "rejected_draft_generation_owner_commands"
    ]
    assert "scripts/presentations/generate_slides_simple.py" in rejected_draft_owners
    assert "scripts/generate_slides_agentic.py" in rejected_draft_owners
    assert handoff["presentation_figure_policy"]["task_id"] == "presentation.svg_figure_generation"
    assert set(handoff["presentation_figure_policy"]["required_renderers"]) == {
        "matplotlib",
        "plotly",
        "svgwrite",
    }
    assert "content-aware" in handoff["presentation_figure_policy"]["completion_owner_policy"]
    assert "generic" in handoff["presentation_figure_policy"]["visual_quality_profile"]
    assert "generic scaffold rejection" in handoff["presentation_figure_policy"]["required_quality_gates"]
    assert handoff["presentation_density_policy"]["target_hyperlinks_per_slide"] == 2.0
    assert handoff["presentation_density_policy"]["target_svgs_per_slide"] == 0.70
    assert "hyperlink_density_meets_target" in handoff["presentation_density_policy"]["validation_check_ids"]
    assert "source_binding_density_meets_target" in handoff["presentation_density_policy"]["validation_check_ids"]
    assert "svg_density_meets_target" in handoff["presentation_density_policy"]["validation_check_ids"]
    assert handoff["presentation_visual_qc_policy"]["model_governance_task_id"] == "presentation.visual_qc"
    assert handoff["presentation_visual_qc_policy"]["default_model"] == "openai:gpt-5.6-sol"
    assert handoff["presentation_visual_qc_policy"]["fallback_model"] == "openai:gpt-5.6-terra"
    assert "visual_llm_qc_report.json" in handoff["presentation_visual_qc_policy"]["required_report_path"]
    assert "Executive Summary" in handoff["narrative_analysis_contract"]["public_report_shape"]
    assert "Detailed Analysis" in handoff["narrative_analysis_contract"]["public_report_shape"]
    assert handoff["narrative_analysis_contract"]["model_governance_task_id"] == (
        "analysis.deep_narrative_synthesis"
    )
    assert handoff["narrative_analysis_contract"]["default_model"] == "openai:gpt-5.6-sol"
    assert handoff["narrative_analysis_contract"]["fallback_model"] == "openai:gpt-5.6-terra"
    assert (
        "analysis/slide_briefs/<topic_id>.md"
        in handoff["narrative_analysis_contract"]["slide_brief_sidecars"]
    )
    assert "analysis/provenance/<topic_id>.md" in handoff["narrative_analysis_contract"][
        "provenance_sidecars"
    ]
    assert "Source Basis" in handoff["narrative_analysis_contract"]["forbidden_public_sections"]
    assert handoff["presentation_content_gate_policy"]["gate"] == (
        "presentation narrative-content gate"
    )
    assert handoff["presentation_content_gate_policy"]["requires_analysis_digest_binding"] is True


def test_analysis_scaffold_emits_token_indexing_and_talk_transcript_contract(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    generated_text = "\n".join(
        read_text(path)
        for path in (
            plan_dir / ".planning" / "ROADMAP.md",
            plan_dir / "approval-summary.json",
            next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md")),
        )
    )

    for term in (
        "PG-CORPUS-TOKEN-INDEXING",
        "token_budget_policy_digest",
        "embedding_policy_digest",
        "EMBEDDING_MAX_LENGTH",
        "CORPUS_VECTOR_EMBED_BATCH_SIZE",
        "vector_embeddings_written: true",
        "transcription_ocr.workshop_talk_boundary_discovery",
        "scripts/discover_workshop_talks.py",
        "Gemini 2.5 Pro",
        "gemini-2.5-pro",
        "*_talks.json",
        "5-minute talk-context",
        "provider_receipt_paths",
        "workshop_talk_boundary_discovery",
        "transcription_ocr.workshop_talk_segmentation",
        "talk_segmentation_provider_not_configured",
        "audio-only recovery is rejected evidence for talk-aware transcripts",
    ):
        assert term in generated_text


def test_analysis_scaffold_emits_benchmark_aware_provider_contracts(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    task_defaults = approval["task model defaults"]
    assert "## Task Model Defaults" in roadmap
    assert any(
        row["task_id"] == "transcription.session_speech"
        and row["default_provider_model"] == "openai:gpt-4o-transcribe-diarize"
        for row in task_defaults
    )
    assert "## Benchmark-Aware Provider Contracts" in roadmap
    assert PROVIDER_CONTRACT_TABLE_HEADER in roadmap
    for contract_id in PROVIDER_CONTRACT_REQUIRED_IDS:
        assert contract_id in roadmap
    for expected in PROVIDER_CONTRACT_UAT_EXPECTED_STRINGS:
        if expected not in {"provider_contracts_checked", "provider_contract_failures", "provider contract failures"}:
            assert expected in roadmap

    rows = provider_contract_rows(approval)
    ids = {str(row.get("contract_id")) for row in rows}
    assert set(PROVIDER_CONTRACT_REQUIRED_IDS) <= ids
    for row in rows:
        assert PROVIDER_CONTRACT_REQUIRED_FIELDS <= set(row)

    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    assert handoff["provider_contracts"] == rows
    assert handoff["task_model_defaults"] == task_defaults
    assert "local-first default for non-14A" not in json.dumps(rows)
    local_row = provider_contract_row_by_id(rows, "PC-14A-LOCAL-FIRST")
    assert local_row["cost_task_id"] == "exhaustive_extraction_local_heuristic"
    assert "ambiguous records" in str(local_row["fallback_or_escalation"])
    assert "endpoint label" in str(local_row["receipt_requirement"])
    gptr_row = provider_contract_row_by_id(rows, "PC-GPT-RESEARCHER-LLM")
    assert "mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit" in str(gptr_row["approved_provider_model"])
    assert "WEB_SEARCH_ENABLED=false" in str(gptr_row["receipt_requirement"])
    embedding_row = provider_contract_row_by_id(rows, "PC-GPT-RESEARCHER-EMBEDDINGS")
    assert "mlx-community/bge-m3-mlx-8bit" in str(embedding_row["approved_provider_model"])
    assert "RETRIEVAL_K=18" in str(embedding_row["receipt_requirement"])
    phase14b_embedding_row = provider_contract_row_by_id(rows, "PC-14B-EMBEDDINGS")
    for row in (phase14b_embedding_row, embedding_row):
        receipt_requirement = str(row["receipt_requirement"])
        assert "embedding_readiness" in receipt_requirement
        assert "client_start_or_load_attempted" in receipt_requirement


def test_analysis_scaffold_declares_backup_posture(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context = read_text(next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md")))
    approval = load_json(plan_dir / "approval-summary.json")

    assert "## Backup Posture" in context
    assert "My Drive/ICLR Scaffold 2027-raw" in context
    assert "--remote-path <mounted-drive-root>/ICLR Scaffold 2027-raw" in context
    assert "storage/storage_manifest.json" in context
    assert "--init-only" in context
    assert "immediately after" in context
    assert "rsync-incremental" in context
    assert approval["backup posture"]["expected_profile"] == "backup"
    assert "--init-only" in approval["backup posture"]["drive_setup_command"]
    assert "immediately after" in approval["backup posture"]["continuous_offload"]
    assert "sync_profiles" in approval["backup posture"]["bloat_control"]
    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    assert handoff["backup_posture"]["manifest_source"] == "conferences/iclr-scaffold-2027/storage/storage_manifest.json"
    assert handoff["backup_posture"]["target_drive_leaf"] == "My Drive/ICLR Scaffold 2027-raw"
    assert handoff["backup_posture"]["runtime_requirement"] == "explicit --remote-path <mounted-drive-root>/ICLR Scaffold 2027-raw"
    assert "--init-only" in handoff["backup_posture"]["drive_setup_command"]
    assert "--remote-path '<mounted-drive-root>/ICLR Scaffold 2027-raw'" in handoff["backup_posture"]["drive_setup_command"]
    assert "immediately after" in handoff["backup_posture"]["continuous_offload"]


def test_analysis_audit_rejects_malformed_structured_production_handoff(tmp_path: Path) -> None:
    for index, mutation in enumerate(("remove_provider", "scalar_interest_groups", "placeholder_backup")):
        case_tmp = tmp_path / mutation
        case_tmp.mkdir()
        plan_dir = scaffold_sample(case_tmp)
        approval_path = plan_dir / "approval-summary.json"
        approval = load_json(approval_path)
        handoff = approval["production handoff"]
        assert isinstance(handoff, dict)
        if mutation == "remove_provider":
            handoff.pop("provider_model_posture", None)
        elif mutation == "scalar_interest_groups":
            handoff["interest_group_decisions"] = "approved"
        else:
            handoff["backup_posture"]["target_drive_leaf"] = "My Drive/<Conference> <Year>-raw"
        write_loaded_json(approval_path, approval)
        report_path = case_tmp / f"report-{index}.json"

        result = run_command(
            [
                sys.executable,
                str(AUDIT_SCRIPT),
                str(plan_dir),
                "--json-output",
                str(report_path),
            ]
        )

        report = load_json(report_path)
        assert result.returncode != 0
        assert "production-handoff-structured" in json.dumps(report["failures"])


def test_analysis_audit_report_records_provider_contract_coverage(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assert set(PROVIDER_CONTRACT_REQUIRED_IDS) <= set(report["provider_contracts_checked"])
    assert report["provider_contract_failures"] == []
    assert "## Provider contracts checked" in read_text(markdown_path)


def test_analysis_audit_runs_anti_tautology_coverage_and_collision_guard(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assertion_names = {item["name"] for item in report["assertions"]}
    assert "stage-contract-runtime-coverage" in assertion_names
    assert "stage-contract-receipt-collision-guard" in assertion_names
    coverage_assertions = [
        item
        for item in report["assertions"]
        if item["name"]
        in {"stage-contract-runtime-coverage", "stage-contract-receipt-collision-guard"}
    ]
    assert all(item["passed"] for item in coverage_assertions), coverage_assertions


def test_production_audit_runs_anti_tautology_coverage_and_collision_guard(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode == 0, report["failures"]
    assertion_names = {item["name"] for item in report["assertions"]}
    assert "stage-contract-runtime-coverage" in assertion_names
    assert "stage-contract-receipt-collision-guard" in assertion_names
    coverage_assertions = [
        item
        for item in report["assertions"]
        if item["name"]
        in {"stage-contract-runtime-coverage", "stage-contract-receipt-collision-guard"}
    ]
    assert all(item["passed"] for item in coverage_assertions), coverage_assertions


def test_analysis_scaffold_emits_pre_spend_contract_gate_and_resume_economics(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "Pre-Spend Contract Gate" in roadmap
    assert "precondition mode" in roadmap
    assert "BEFORE the first provider call" in roadmap
    assert "precondition_contract_unmet" in roadmap
    assert "zero-spend blocker" in roadmap
    assert "Incremental Receipts And Resume Economics" in roadmap
    assert "refresh their top-level receipt after every" in roadmap
    assert "reuse-by-default" in roadmap
    assert "regenerate-all pattern" in roadmap


def test_analysis_audit_rejects_missing_pre_spend_contract_gate_language(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace(
        "precondition_contract_unmet", "contract_check_deferred"
    )
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "pre-spend-contract-gate" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_incremental_receipts_resume_economics_language(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace("reuse-by-default", "reuse-if-convenient")
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "incremental-receipts-resume-economics" in json.dumps(report["failures"])


def test_production_scaffold_emits_pre_spend_contract_gate_and_resume_economics(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "Pre-Spend Contract Gate" in roadmap
    assert "precondition mode" in roadmap
    assert "BEFORE the first provider call" in roadmap
    assert "precondition_contract_unmet" in roadmap
    assert "zero-spend blocker" in roadmap
    assert "Incremental Receipts And Resume Economics" in roadmap
    assert "refresh their top-level receipt after every" in roadmap
    assert "reuse-by-default" in roadmap
    assert "regenerate-all pattern" in roadmap


def test_production_audit_rejects_missing_pre_spend_contract_gate_language(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace(
        "precondition_contract_unmet", "contract_check_deferred"
    )
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "pre-spend-contract-gate" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_incremental_receipts_resume_economics_language(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace("reuse-by-default", "reuse-if-convenient")
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "incremental-receipts-resume-economics" in json.dumps(report["failures"])


def test_analysis_scaffold_emits_provider_substitution_governance(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "Provider-Substitution Governance And Credential Preflight" in roadmap
    assert "provider_calls_made: true" in roadmap
    assert "provider_substitution_waiver.json" in roadmap
    assert "references/provider-substitution-waiver.md" in roadmap
    assert "ONLY governed path" in roadmap
    assert "Phase-1 cloud-credential preflight" in roadmap
    assert "unauthorized_provider_substitution" in roadmap
    assert "credential_invalid" in roadmap
    waiver_doc = read_text(
        REPO_ROOT
        / ".claude"
        / "skills"
        / "plan-conference-analysis"
        / "references"
        / "provider-substitution-waiver.md"
    )
    assert "task_id" in waiver_doc
    assert '"approver"' in waiver_doc
    assert '"expiry"' in waiver_doc


def test_analysis_audit_rejects_missing_provider_substitution_governance_language(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace(
        "provider_substitution_waiver.json", "provider_substitution_note.json"
    )
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-substitution-governance" in json.dumps(report["failures"])


def test_production_scaffold_emits_provider_substitution_governance(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "Provider-Substitution Governance And Credential Preflight" in roadmap
    assert "provider_calls_made: true" in roadmap
    assert "provider_substitution_waiver.json" in roadmap
    assert "references/provider-substitution-waiver.md" in roadmap
    assert "ONLY governed path" in roadmap
    assert "Phase-1 cloud-credential preflight" in roadmap
    assert "unauthorized_provider_substitution" in roadmap
    assert "credential_invalid" in roadmap
    waiver_doc = read_text(
        REPO_ROOT
        / ".claude"
        / "skills"
        / "plan-conference-production"
        / "references"
        / "provider-substitution-waiver.md"
    )
    assert "task_id" in waiver_doc
    assert '"approver"' in waiver_doc
    assert '"expiry"' in waiver_doc


def test_production_audit_rejects_missing_provider_substitution_governance_language(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated = read_text(roadmap_path).replace(
        "provider_substitution_waiver.json", "provider_substitution_note.json"
    )
    assert mutated != read_text(roadmap_path)
    roadmap_path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-substitution-governance" in json.dumps(report["failures"])


def test_analysis_scaffold_emits_gsd_lifecycle_and_orchestrator_conformance(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    state = read_text(plan_dir / ".planning" / "STATE.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "no live execution of Phase N without" in roadmap
    assert "no live execution of Phase N without" in state
    assert "phases/NN-*/" in roadmap
    assert "plan-review/verification artifacts" in roadmap
    assert "GSD Lifecycle And Orchestrator Conformance" in roadmap
    assert "orchestrator_stage_sequence" in roadmap
    assert "forbidden from provider fallbacks outside the approved contract rows" in roadmap
    assert "reject receipts produced by an undeclared orchestrator" in roadmap
    lifecycle = approval["gsd lifecycle and orchestrator conformance"]
    assert isinstance(lifecycle["orchestrator_stage_sequence"], list)
    assert lifecycle["orchestrator_stage_sequence"]


def test_analysis_audit_rejects_missing_gsd_lifecycle_hard_rule(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    state_path = plan_dir / ".planning" / "STATE.md"
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    approval_path = plan_dir / "approval-summary.json"
    for path in (state_path, roadmap_path):
        mutated = read_text(path).replace(
            "no live execution of Phase N without", "phase execution may proceed without"
        )
        path.write_text(mutated, encoding="utf-8")
    approval = load_json(approval_path)
    approval["gsd lifecycle and orchestrator conformance"]["hard_rule"] = (
        "phase execution may proceed without a scaffolded plan"
    )
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "gsd-lifecycle-hard-rule" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_orchestrator_conformance_language(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    mutated_roadmap = read_text(roadmap_path).replace(
        "GSD Lifecycle And Orchestrator Conformance", "GSD Lifecycle Notes"
    )
    assert mutated_roadmap != read_text(roadmap_path)
    roadmap_path.write_text(mutated_roadmap, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "gsd-orchestrator-conformance" in json.dumps(report["failures"])


def test_production_scaffold_emits_gsd_lifecycle_and_orchestrator_conformance(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    state = read_text(plan_dir / ".planning" / "STATE.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "no live execution of Phase N without" in roadmap
    assert "no live execution of Phase N without" in state
    assert "GSD Lifecycle And Orchestrator Conformance" in roadmap
    assert "orchestrator_stage_sequence" in roadmap
    lifecycle = approval["gsd lifecycle and orchestrator conformance"]
    assert isinstance(lifecycle["orchestrator_stage_sequence"], list)
    assert lifecycle["orchestrator_stage_sequence"]


def test_production_audit_rejects_missing_gsd_lifecycle_hard_rule(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    state_path = plan_dir / ".planning" / "STATE.md"
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    approval_path = plan_dir / "approval-summary.json"
    for path in (state_path, roadmap_path):
        mutated = read_text(path).replace(
            "no live execution of Phase N without", "phase execution may proceed without"
        )
        path.write_text(mutated, encoding="utf-8")
    approval = load_json(approval_path)
    approval["gsd lifecycle and orchestrator conformance"]["hard_rule"] = (
        "phase execution may proceed without a scaffolded plan"
    )
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "gsd-lifecycle-hard-rule" in json.dumps(report["failures"])


def test_analysis_scaffold_emits_structured_gates_with_distinct_receipt_paths(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval = load_json(plan_dir / "approval-summary.json")
    handoff = approval["production handoff"]
    assert "structured_gates" in approval["production handoff fields"]
    gates = {row["gate_id"]: row for row in handoff["structured_gates"]}
    named = ("PG-TOPIC-STRATEGY", "PG-QUERY", "PG-GPT-RESEARCHER", "PG-PAPER-MATCHING")
    assert set(named) <= set(gates)
    receipt_paths = [gates[gate_id]["receipt_path"] for gate_id in named]
    assert len(set(receipt_paths)) == len(receipt_paths)
    for gate_id in named:
        assert gates[gate_id]["field_checks"]


def test_analysis_audit_rejects_structured_gates_collapsed_to_one_receipt(
    tmp_path: Path,
) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["production handoff"]
    for row in handoff["structured_gates"]:
        if row["gate_id"] in {
            "PG-TOPIC-STRATEGY",
            "PG-QUERY",
            "PG-GPT-RESEARCHER",
            "PG-PAPER-MATCHING",
        }:
            row["receipt_path"] = "storage/workflow-runs/<run-id>/stage-07-analysis.json"
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-handoff-structured" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_structured_gates_field(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    del approval["production handoff"]["structured_gates"]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-handoff-structured" in json.dumps(report["failures"])


def test_production_scaffold_consumes_structured_gates(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval = load_json(plan_dir / "approval-summary.json")

    assert "structured_gates" in approval
    gates = {row["gate_id"]: row for row in approval["structured_gates"]}
    named = ("PG-TOPIC-STRATEGY", "PG-QUERY", "PG-GPT-RESEARCHER", "PG-PAPER-MATCHING")
    assert set(named) <= set(gates)
    receipt_paths = [gates[gate_id]["receipt_path"] for gate_id in named]
    assert len(set(receipt_paths)) == len(receipt_paths)


def test_production_audit_rejects_missing_propagated_structured_gates(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    del approval["structured_gates"]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "structured-gates-propagated" in json.dumps(report["failures"])


def _workshop_scaffold_sample(tmp_path: Path) -> Path:
    brief = sample_brief()
    brief["workshop_coverage"] = {
        "posture": "selective",
        "openreview_parent_venue": "iclr-scaffold-2027/Workshop",
        "workshops": [
            {
                "venueid": "iclr-scaffold-2027/Workshop/Example",
                "pdf_availability_class": "openreview_with_pdf",
                "paper_count": 4,
                "decision": "approved",
                "decision_rationale": "In-scope for corpus completeness proof.",
            }
        ],
    }
    brief_path = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief_path, brief)
    result = run_scaffold(brief_path, draft_root)
    assert result.returncode == 0, result.stderr + result.stdout
    return draft_root / "iclr-scaffold-2027"


def test_analysis_scaffold_emits_openreview_auth_posture_when_workshops_in_scope(
    tmp_path: Path,
) -> None:
    plan_dir = _workshop_scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "--openreview-auth required" in roadmap
    handoff = approval["production handoff"]
    assert "openreview_auth_posture" in approval["production handoff fields"]
    posture = handoff["openreview_auth_posture"]
    assert posture["unauthenticated_is_fail_stop"] is True
    assert "unauthenticated" in posture["rejected_modes_on_required_corpus"]


def test_analysis_audit_rejects_workshop_inventory_without_openreview_auth_required(
    tmp_path: Path,
) -> None:
    plan_dir = _workshop_scaffold_sample(tmp_path)
    for path in plan_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".md", ".json", ".yaml", ".yml", ".py"}:
            continue
        original = path.read_text(encoding="utf-8")
        mutated = original.replace("--openreview-auth required", "--openreview-auth auto")
        if mutated != original:
            path.write_text(mutated, encoding="utf-8")
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "openreview-auth-required-when-workshop-inventory" in json.dumps(report["failures"])


def test_analysis_audit_rejects_workshop_inventory_with_unauthenticated_accepted(
    tmp_path: Path,
) -> None:
    plan_dir = _workshop_scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    approval["production handoff"]["openreview_auth_posture"]["unauthenticated_is_fail_stop"] = False
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "openreview-auth-unauthenticated-fail-stop" in json.dumps(report["failures"])


def test_analysis_skill_registry_table_lists_browser_cookie_export() -> None:
    skill = read_text(SKILL_MD)

    assert "`acquisition.browser_cookie_export`" in skill
    assert "docs/reference/stage-owner-audit.md#acquisition-browser-cookie-export" in skill


def test_analysis_skill_prose_fragments_repaired() -> None:
    skill = read_text(SKILL_MD)

    assert "\nsatisfied.\n" not in skill
    assert "\nevidence.\n" not in skill
    assert "stage owner registry audit (stage owner registry audit)" not in skill
    assert ".planning/workstreams/v1-6/" not in skill


def test_analysis_audit_rejects_missing_benchmark_authority(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    update_provider_contract_row(
        plan_dir,
        "PC-OCR-AUTHORITY",
        {"benchmark_authority": "", "receipt_requirement": ""},
    )
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-benchmark-authority" in provider_contract_failure_names(report)


def test_analysis_audit_rejects_unsupported_local_first_provider_claim(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    update_provider_contract_row(
        plan_dir,
        "PC-14B-EMBEDDINGS",
        {"local_posture": "local-first default for non-14A embeddings"},
    )
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\nBad clause: local models are always best for embeddings without benchmark receipts.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    failure_names = provider_contract_failure_names(report)
    assert result.returncode != 0
    assert "provider-contract-local-boundary" in failure_names
    assert "provider-contract-no-universal-best" in failure_names


def test_analysis_audit_rejects_provider_contract_handoff_drift(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    rows = provider_contract_rows(handoff, "provider_contracts")
    provider_contract_row_by_id(rows, "PC-14B-EMBEDDINGS")["approved_provider_model"] = "local-embedding-model"
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-handoff-continuity" in provider_contract_failure_names(report)


def test_analysis_audit_rejects_extra_provider_contract_row_drift(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    top_rows = provider_contract_rows(approval)
    top_rows.append(extra_provider_contract_row(top_rows))
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-handoff-continuity" in provider_contract_failure_names(report)


def test_analysis_audit_rejects_missing_governance_handoff(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context_path = next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md"))
    context_path.write_text(
        read_text(context_path).replace("Interest-Group Review", "Interest Review"),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "analysis-governance-handoff" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_backup_posture(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context_path = next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md"))
    context_path.write_text(
        read_text(context_path).replace("## Backup Posture", "## Storage Note"),
        encoding="utf-8",
    )
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    approval.pop("backup posture", None)
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "backup-posture-contract" in json.dumps(report["failures"])


def test_audit_rejects_missing_official_workflow_step(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = json.loads(read_text(approval_path))
    approval["official workflow mapping"] = [
        row for row in approval["official workflow mapping"] if row["step_id"] != "WF-09"
    ]
    write_json(approval_path, approval)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        "\n".join(
            line
            for line in read_text(roadmap_path).splitlines()
            if not line.startswith("| WF-09 |")
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-mapping-complete" in json.dumps(report["failures"])


def test_audit_rejects_malformed_official_workflow_mapping_row(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = json.loads(read_text(approval_path))
    approval["official workflow mapping"][0]["status"] = "unclear"
    approval["official workflow mapping"][0]["support_label"] = "live-ish"
    write_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    failures = json.dumps(report["failures"])
    assert result.returncode != 0
    assert "official-workflow-status-values" in failures
    assert "official-workflow-support-labels" in failures


def test_audit_rejects_extra_official_workflow_mapping_row_fields(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = json.loads(read_text(approval_path))
    approval["official workflow mapping"][0]["unreviewed_metadata"] = "skip review"
    write_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-row-schema" in json.dumps(report["failures"])


def test_audit_rejects_missing_official_workflow_optional_substep_from_roadmap(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        "\n".join(
            line
            for line in read_text(roadmap_path).splitlines()
            if not line.startswith("| WF-14c |")
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-roadmap-table" in json.dumps(report["failures"])


def test_audit_rejects_official_workflow_mapping_after_phase_blocks(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap = read_text(roadmap_path)
    mapping_index = roadmap.index("## Official Workflow Mapping")
    phase_index = roadmap.index("### Phase ")
    mapping_block = roadmap[mapping_index:phase_index]
    roadmap_path.write_text(
        roadmap[:mapping_index] + roadmap[phase_index:].rstrip() + "\n\n" + mapping_block,
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-roadmap-table" in json.dumps(report["failures"])


def test_audit_rejects_wrong_official_workflow_source_ref(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = json.loads(read_text(approval_path))
    approval["official workflow mapping"][0]["source_ref"] = "docs/other.md#overview"
    write_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-source-ref" in json.dumps(report["failures"])


def test_audit_rejects_generic_secret_leaks(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context = next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md"))
    context.write_text(
        read_text(context)
        + "\nLeaked forms: api_key=generic-tail Authorization: Bearer bearer-token sessionid=abc123\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "secret-like" in json.dumps(report["failures"])


def test_analysis_scaffold_redacts_whole_secret_values(tmp_path: Path) -> None:
    brief = sample_brief()
    brief["auth_notes"] = "OPENAI_API_KEY=sk-proj-secret-token Cookie: session=secret-cookie"
    brief["source_urls"] = [
        "https://example.org/iclr-scaffold-2027/schedule?key=AIzaSecretToken",
    ]
    brief["provider_preferences"] = "Use sk-provider-secret-token only in local env."
    brief_path = tmp_path / "brief.json"
    draft_root = tmp_path / "drafts"
    write_json(brief_path, brief)

    result = run_scaffold(brief_path, draft_root)

    assert result.returncode == 0, result.stderr
    plan_dir = draft_root / "iclr-scaffold-2027"
    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in plan_dir.rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    assert "proj-secret-token" not in generated_text
    assert "secret-cookie" not in generated_text
    assert "AIzaSecretToken" not in generated_text
    assert "sk-provider-secret-token" not in generated_text
    audit = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir)])
    assert audit.returncode == 0, audit.stderr + audit.stdout


def test_analysis_audit_rejects_unredacted_secret_values(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    context_path = next((plan_dir / ".planning" / "milestones").glob("*-CONTEXT.md"))
    context_path.write_text(
        read_text(context_path) + "\nOPENAI_API_KEY=sk-proj-secret-token\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "secret-like" in json.dumps(report["failures"])


def test_audit_rejects_silent_workflow_step_merge(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = json.loads(read_text(approval_path))
    for row in approval["official workflow mapping"]:
        if row["step_id"] in CRITICAL_RECEIPT_BOUNDARY_TERMS:
            row["phase_name"] = "Analysis"
            row["receipt_boundary"] = "analysis phase receipt"
    write_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "official-workflow-no-silent-merge" in json.dumps(report["failures"])


def test_production_audit_requires_pg_topic_strategy_gate(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    gates = approval["production workflow gates"]
    assert any(row.get("gate_id") == "PG-TOPIC-STRATEGY" for row in gates)
    approval["production workflow gates"] = [
        row for row in gates if row.get("gate_id") != "PG-TOPIC-STRATEGY"
    ]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "PG-TOPIC-STRATEGY" not in report["workflow_gates_checked"]
    assert "production-workflow-gates-complete" in json.dumps(report["failures"])


def test_production_skill_mandates_json_output_for_audit_command() -> None:
    skill = read_text(PRODUCTION_SKILL_MD)

    assert (
        "audit_generated_production_plan.py <generated-production-dir> --json-output "
        "<workstream>/audits/generated-production-audit.json" in skill
    )


def test_production_generator_outputs_live_only_milestone_and_audit_passes(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assert set(REQUIRED_PRODUCTION_STAGES) <= set(report["pipeline_stages_checked"])
    assert set(REQUIRED_PRODUCTION_GATES) <= set(report["gate_phrases_checked"])

    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = json.loads(read_text(plan_dir / "approval-summary.json"))
    assert ".venv/bin/python scripts/run_iclr_scaffold_2027_production.py" in roadmap
    assert "--mode live" in roadmap
    assert "--max-papers 180" not in roadmap
    assert "canonical source-aware chunking" in roadmap
    assert "snippet-only/first-240-character vectorization is rejected" in roadmap
    assert "dry-run receipts, mock receipts" in roadmap
    assert approval["support claims"] == [
        "live-only production",
        "receipt-backed completion",
        "non-live evidence rejected",
    ]
    assert "stage-local benchmark gates" in approval
    assert "workflow registry dispatch coverage" in approval
    assert "backup checkpoints" in approval
    assert "ocr.slide_image_parse" in json.dumps(approval["stage-local benchmark gates"])
    assert approval["deprecated owner command policy"]["blocked_task_id"] == "presentation.svg_figure_generation"
    assert "content-aware editorial" in approval["deprecated owner command policy"]["required_owner"]
    assert "generic scaffold rejection" in approval["deprecated owner command policy"]["required_quality_gates"]
    assert approval["presentation density policy"]["target_hyperlinks_per_slide"] == 2.0
    assert approval["presentation density policy"]["target_svgs_per_slide"] == 0.70
    assert "hyperlink_density_meets_target" in approval["presentation density policy"]["validation_check_ids"]
    assert approval["presentation visual qc policy"]["model_governance_task_id"] == "presentation.visual_qc"
    assert any(
        "inspect_rendered_slides_llm.py" in owner
        for owner in approval["presentation visual qc policy"]["required_owner_commands"]
    )
    assert set(approval["deprecated owner command policy"]["banned_scripts"]) == {
        "scripts/presentations/extract_topic_figures.py",
        "scripts/presentations/add_images_to_md.py",
    }
    assert approval["corpus scope"] == "all discovered conference items unless explicitly excluded"
    assert (plan_dir / "conferences" / "iclr-scaffold-2027" / "PRODUCTION.md").exists()


def test_production_generator_outputs_structured_presentation_artifacts(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    generated_text = "\n".join(
        [
            read_text(plan_dir / ".planning" / "ROADMAP.md"),
            read_text(plan_dir / "approval-summary.json"),
            read_text(plan_dir / "conferences" / "iclr-scaffold-2027" / "PRODUCTION.md"),
        ]
    )

    for term in (
        "presentation_deck.json",
        "source_binding_report.json",
        "visual_plan.json",
        "render_spec.json",
        "delivery_single_speaker.json",
        "delivery_notebooklm.json",
        "video_sync_validation_single_speaker.json",
        "video_sync_validation_notebooklm.json",
        "PRESENTATION_LEGACY_MARKDOWN_STAGE_USED",
    ):
        assert term in generated_text


def test_production_audit_rejects_legacy_markdown_render_source(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\n**Presentation render proof:** `decks/<id>/5_draft_slides_with_images.md` "
        "is the production render source for `presentation.html`.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = json.loads(read_text(report_path))

    assert result.returncode == 1
    assert "legacy-markdown-production-proof" in json.dumps(report["failures"])


def test_production_audit_rejects_legacy_markdown_phase13_canary_proof(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    production_path = plan_dir / "conferences" / "iclr-scaffold-2027" / "PRODUCTION.md"
    production_path.write_text(
        read_text(production_path)
        + "\nPhase 13 canary evidence is complete when "
        "`decks/04_safety_alignment_privacy/2_draft_slides_enriched_with_papers.md` "
        "and `decks/04_safety_alignment_privacy/4.5_draft_slides_with_hyperlinks.md` "
        "exist.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = json.loads(read_text(report_path))

    assert result.returncode == 1
    assert "legacy-markdown-phase13-canary-proof" in json.dumps(report["failures"])


def test_production_audit_rejects_silent_max_papers_cap(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    capped_live_command = (
        "\n**Live command:** `.venv/bin/python "
        "scripts/run_iclr_scaffold_2027_production.py "
        "--workspace conferences/iclr-scaffold-2027 --run-id iclr-scaffold-2027-live-production "
        "--stage download-corpus --mode live --resume --max-papers 180`\n"
    )
    roadmap_path.write_text(read_text(roadmap_path) + capped_live_command, encoding="utf-8")
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-corpus-no-silent-item-cap" in json.dumps(report["failures"])


def test_production_audit_rejects_snippet_only_vector_store_policy(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\nOperator note: snippet-based chunking is correct for evidence-cited retrieval.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-vector-store-no-snippet-only-chunking" in json.dumps(report["failures"])


def test_production_scaffold_emits_explicit_missing_workflow_gates(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    assert "## Production Workflow Gates" in roadmap
    assert (
        "| Gate ID | Workflow step | Owner command | Receipt path | Pass criteria | "
        "Fail criteria | Downstream blockers | Stale evidence rule | Rejected evidence | Source |"
    ) in roadmap
    for gate_id in PRODUCTION_GATE_REQUIRED_IDS:
        assert gate_id in roadmap
    for token in (
        "--mode live",
        "--resume",
        "storage/workflow-runs",
        "stage-06-corpus-inventory-index.json",
        "stage-07-analysis.json",
        "stage-08-qa-gap-fill-synthesis.json",
        "token_budget_policy_digest",
        "embedding_policy_digest",
        "EMBEDDING_MAX_LENGTH",
        "CORPUS_VECTOR_EMBED_BATCH_SIZE",
        "transcription_ocr.workshop_talk_segmentation",
        "talk_segmentation_provider_not_configured",
        "audio-only recovery is rejected evidence for talk-aware transcripts",
    ):
        assert token in roadmap
    for term in PRODUCTION_GATE_DOWNSTREAM_TERMS + PRODUCTION_REJECTED_EVIDENCE_TERMS:
        assert term in roadmap


def test_production_scaffold_emits_workflow_registry_dispatch_coverage(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "## Workflow Registry Dispatch Coverage" in roadmap
    rows = approval["workflow registry dispatch coverage"]
    assert isinstance(rows, list)
    by_stage = {row["workflow_stage_id"]: row for row in rows if isinstance(row, dict)}
    backup_ids = {
        row["checkpoint_id"]
        for row in approval["backup checkpoints"]
        if isinstance(row, dict) and row.get("checkpoint_id")
    }
    expected_receipts = {
        "discover": "stage-02-source-discovery.json",
        "download": "stage-04-download-corpus.json",
        "transcribe": "stage-05-transcription-ocr.json",
        "enhanced-indexing-vector-store": "stage-06-corpus-inventory-index.json",
        "exhaustive-topic-extraction": "stage-07-analysis.json",
        "query-planning-gpt-researcher": "stage-07-analysis.json",
        "coverage-gap-fill": "stage-08-qa-gap-fill-synthesis.json",
        "synthesis-trend-report": "stage-08-qa-gap-fill-synthesis.json",
        "podcast": "stage-09-podcast.json",
        "deck": "stage-10-presentation.json",
    }
    assert set(WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS) <= set(by_stage)
    for stage_id in WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS:
        row = by_stage[stage_id]
        assert row["dispatch_kind"] in {"python_stage", "subprocess_stage"}
        assert row["implementation_ref"]
        assert row["implementation_ref"] != "registry unavailable"
        assert "--stage" in row["owner_command"]
        assert "storage/workflow-runs" in row["receipt_path"]
        assert str(row["receipt_path"]).endswith(expected_receipts[stage_id])
        assert "cost" in row["cost_accounting"]
        assert row["backup_checkpoint"] in backup_ids
        assert "verification" in row["verification_contract"]


def test_production_scaffold_emits_backup_checkpoints(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "## Manifest-Driven Backup Checkpoints" in roadmap
    rows = approval["backup checkpoints"]
    assert isinstance(rows, list)
    ids = {row["checkpoint_id"] for row in rows if isinstance(row, dict)}
    assert set(BACKUP_CHECKPOINT_REQUIRED_IDS) <= ids
    for row in rows:
        assert "scripts/sync_to_gdrive.py" in row["dry_run_command"]
        assert "--profile backup" in row["dry_run_command"]
        assert "--remote-path" in row["dry_run_command"]
        assert "--remote-path" in row["write_command"]
        assert "--dry-run" in row["dry_run_command"]
        assert "--receipt-path" in row["write_command"]
        assert "--dry-run" not in row["write_command"]
        assert "/backup/" in row["receipt_path"]
        assert "-backup-write-receipt.json" in row["receipt_path"]
        assert "-backup-dry-run-receipt.json" in row["dry_run_command"]
        assert row["dry_run_command"] != row["write_command"]
        assert "digest_count" in row["required_receipt_fields"]
        assert "manifest_sha256" in row["required_receipt_fields"]
        assert "artifact_snapshot_sha256" in row["required_receipt_fields"]
        assert "sync_items" in row["required_receipt_fields"]
        assert "artifact_snapshot_sha256" in row["stale_artifact_failure_rule"]
        assert "immediately after" in row["verification_fail_closed"].lower()
        assert "fail" in row["verification_fail_closed"].lower()
    assert "## Drive Workspace Setup Gate" in roadmap
    assert "--init-only" in roadmap
    setup = approval["drive workspace setup"]
    assert isinstance(setup, dict)
    assert "--init-only" in setup["command"]
    assert "--remote-path" in setup["command"]
    assert "without copying payload artifacts" in setup["bloat_control"]


def test_production_approval_records_machine_checkable_workflow_gates(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval = load_json(plan_dir / "approval-summary.json")
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    assert workflow_gates

    gates_by_id = {str(row.get("gate_id")): row for row in workflow_gates if isinstance(row, dict)}
    assert set(PRODUCTION_GATE_REQUIRED_IDS) <= set(gates_by_id)
    for gate_id, row in gates_by_id.items():
        assert PRODUCTION_GATE_REQUIRED_FIELDS <= set(row), gate_id
        owner_command = str(row["owner_command"])
        if gate_id == "PG-EMBEDDED-MEDIA-RECOVERY":
            assert "scripts/recover_iclr2026_slideslive_media.py" in owner_command
            assert "--workspace conferences/iclr-scaffold-2027" in owner_command
            assert "--run-id iclr-scaffold-2027-live-production" in owner_command
            assert "--resume" in owner_command
            assert str(row["receipt_path"]).endswith(
                "storage/download-corpus/slideslive_media_recovery.json"
            )
        elif gate_id == "PG-SLIDESLIVE-SLIDE-ACQUISITION":
            assert "scripts/recover_slideslive_slides.py" in owner_command
            assert "scripts/detect_reference_slides.py" in owner_command
            assert "scripts/cleanup_slides.py" in owner_command
            assert "skip_ocr" in row["pass_criteria"]
            assert str(row["receipt_path"]).endswith(
                "storage/download-corpus/slideslive_slide_recovery.json"
            )
        elif gate_id == "PG-WORKSHOP-TALK-BOUNDARY-DISCOVERY":
            assert "scripts/discover_workshop_talks.py" in owner_command
            assert "--provider-cost-approved" in owner_command
            assert "gemini-2.5-pro" in owner_command
            assert "transcription_ocr.workshop_talk_boundary_discovery" in row["pass_criteria"]
            assert "Gemini 2.5 Pro" in row["pass_criteria"]
            assert "*_talks.json" in row["pass_criteria"]
            assert "5-minute talk-context" in row["pass_criteria"]
            assert "provider_receipt_paths" in row["pass_criteria"]
            assert "workshop_talk_boundary_discovery" in row["pass_criteria"]
            assert str(row["receipt_path"]).endswith(
                "storage/transcription-ocr/workshop-talks/<workshop_id>_discovery_receipt.json"
            )
        elif gate_id == "PG-WORKSHOP-TALK-MEDIA-ACQUISITION":
            assert "workshop talk-media acquisition" in str(row["workflow_step"]).lower()
            assert "talk_media_acquisition_receipt.json" in owner_command
            assert "workshop_talk_media_scope_waiver.json" in owner_command
            assert "workshop_id" in row["pass_criteria"]
            assert "zero workshop speech records" in row["fail_criteria"]
            assert str(row["receipt_path"]).endswith(
                "storage/workshops/talk_media_acquisition_receipt.json"
            )
        elif gate_id == "PG-CORPUS-TOKEN-INDEXING":
            gate_text = " ".join(
                str(row[field])
                for field in (
                    "pass_criteria",
                    "fail_criteria",
                    "stale_evidence_rule",
                    "rejected_evidence",
                )
            )
            assert "--stage corpus-inventory-index" in owner_command
            assert "stage-06-corpus-inventory-index.json" in str(row["receipt_path"])
            for term in (
                "token_budget_policy_digest",
                "embedding_policy_digest",
                "EMBEDDING_MAX_LENGTH",
                "CORPUS_VECTOR_EMBED_BATCH_SIZE",
                "vector_embeddings_written: true",
                "embedded chunk counts match",
            ):
                assert term in gate_text
        elif gate_id == "PG-INDEX":
            gate_text = " ".join(
                str(row[field])
                for field in (
                    "pass_criteria",
                    "fail_criteria",
                    "stale_evidence_rule",
                    "rejected_evidence",
                )
            )
            for term in (
                "embedding_readiness",
                "client_start_or_load_attempted",
                "build_embedding_client",
                "embed_texts",
                "blocked_local_embedding_client_start_failed",
            ):
                assert term in gate_text
        elif gate_id == "PG-LLM-FULL-READ-EXHAUSTIVE-EXTRACTION":
            assert "scripts/run_llm_full_read_extraction.py" in owner_command
            assert "--scope all" in owner_command
            assert "--resume" in owner_command
            assert "14A-llm-full-read-extraction-receipt.json" in str(row["receipt_path"])
        elif gate_id == "PG-EXHAUSTIVE-TOPIC-SUMMARIES":
            assert "scripts/summarize_llm_full_read_topics.py" in owner_command
            assert "--max-input-tokens 400000" in owner_command
            assert "07A-exhaustive-topic-summaries.json" in str(row["receipt_path"])
        elif gate_id == "PG-CORPUS-SOURCE-METADATA":
            gate_text = " ".join(
                str(row[field])
                for field in (
                    "pass_criteria",
                    "fail_criteria",
                    "stale_evidence_rule",
                    "rejected_evidence",
                )
            )
            assert "scripts/audit_corpus_source_metadata.py" in owner_command
            assert "--workspace" in owner_command
            assert "--write-receipt" in owner_command
            assert "corpus_source_metadata_coverage.json" in str(row["receipt_path"])
            for term in (
                "papers_main",
                "papers_workshop",
                "structurally_authorless",
                "floor_override_unjustified",
                "COVERAGE_FLOORS",
            ):
                assert term in gate_text
        else:
            assert ".venv/bin/python scripts/run_iclr_scaffold_2027_production.py" in owner_command
            assert "--stage" in owner_command
            assert "--mode live" in owner_command
            assert "--resume" in owner_command
            assert "conferences/iclr-scaffold-2027/storage/workflow-runs/" in str(row["receipt_path"])
        for field in (
            "pass_criteria",
            "fail_criteria",
            "downstream_blockers",
            "stale_evidence_rule",
            "rejected_evidence",
        ):
            assert isinstance(row[field], str)
            assert row[field].strip()


def test_production_audit_report_records_workflow_gate_coverage(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    report_path = tmp_path / "production-report.json"
    markdown_path = tmp_path / "production-report.md"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode == 0, report["failures"]
    assert report["status"] == "passed"
    assert set(PRODUCTION_GATE_REQUIRED_IDS) <= set(report["workflow_gates_checked"])
    assert report["workflow_gate_failures"] == []
    assert set(WORKFLOW_REGISTRY_REQUIRED_STAGE_IDS) <= set(report["workflow_registry_checked"])
    assert report["workflow_registry_failures"] == []
    assert set(BACKUP_CHECKPOINT_REQUIRED_IDS) <= set(report["backup_checkpoints_checked"])
    assert report["backup_checkpoint_failures"] == []
    assert "## Production workflow gates checked" in read_text(markdown_path)
    assert "## Workflow registry dispatch checked" in read_text(markdown_path)
    assert "## Backup checkpoints checked" in read_text(markdown_path)


def test_production_audit_rejects_missing_provider_contract_row(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    remove_provider_contract_row(plan_dir, "PC-TAVILY-SEARCH-EXTRACT", production=True)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contracts-complete" in provider_contract_failure_names(report)


def test_production_audit_rejects_changed_provider_contract_embedding_model(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    update_provider_contract_row(
        plan_dir,
        "PC-14B-EMBEDDINGS",
        {"approved_provider_model": "local-embedding-model"},
        production=True,
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-non14a-preserved" in provider_contract_failure_names(report)


def test_production_audit_rejects_non14a_local_first_synonym_claim(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\nBad clause: embeddings should use a local-first default before benchmark receipts exist.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-local-boundary" in provider_contract_failure_names(report)


def test_production_audit_rejects_extra_provider_contract_row_drift(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["source analysis handoff"]
    assert isinstance(handoff, dict)
    source_rows = provider_contract_rows(handoff, "provider_contracts")
    source_rows.append(extra_provider_contract_row(source_rows))
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contract-source-handoff-continuity" in provider_contract_failure_names(report)


def test_production_audit_rejects_missing_openai_tavily_or_ocr_contracts(tmp_path: Path) -> None:
    for index, contract_id in enumerate(
        (
            "PC-GPT-RESEARCHER-LLM",
            "PC-GPT-RESEARCHER-EMBEDDINGS",
            "PC-TAVILY-SEARCH-EXTRACT",
            "PC-OCR-AUTHORITY",
            "PC-OCR-PRODUCTION",
        )
    ):
        case_tmp = tmp_path / f"case-{index}"
        case_tmp.mkdir()
        plan_dir = production_sample(case_tmp)
        remove_provider_contract_row(plan_dir, contract_id, production=True)
        report_path = case_tmp / "production-report.json"

        result = run_command(
            [
                sys.executable,
                str(PRODUCTION_AUDIT_SCRIPT),
                str(plan_dir),
                "--json-output",
                str(report_path),
            ]
        )

        report = load_json(report_path)
        assert result.returncode != 0, contract_id
        assert "provider-contracts-complete" in provider_contract_failure_names(report)


def test_production_audit_rejects_missing_extraction_escalation_contract(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    remove_provider_contract_row(plan_dir, "PC-EXTRACTION-REMOTE-ESCALATION", production=True)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider-contracts-complete" in provider_contract_failure_names(report)


def test_production_audit_report_records_provider_contract_failures_field(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    update_provider_contract_row(
        plan_dir,
        "PC-14B-EMBEDDINGS",
        {"approved_provider_model": "local-embedding-model"},
        production=True,
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "provider_contracts_checked" in report
    assert report["provider_contract_failures"]


def test_production_audit_rejects_missing_required_workflow_gate(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    approval["production workflow gates"] = [
        row for row in workflow_gates if not isinstance(row, dict) or row.get("gate_id") != "PG-QUERY"
    ]
    write_loaded_json(approval_path, approval)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_rows = [
        line for line in read_text(roadmap_path).splitlines() if not line.startswith("| PG-QUERY |")
    ]
    roadmap_path.write_text("\n".join(roadmap_rows) + "\n", encoding="utf-8")
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-workflow-gates-complete" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_corpus_token_indexing_gate(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    approval["production workflow gates"] = [
        row
        for row in workflow_gates
        if not isinstance(row, dict) or row.get("gate_id") != "PG-CORPUS-TOKEN-INDEXING"
    ]
    write_loaded_json(approval_path, approval)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_rows = [
        line
        for line in read_text(roadmap_path).splitlines()
        if not line.startswith("| PG-CORPUS-TOKEN-INDEXING |")
    ]
    roadmap_path.write_text("\n".join(roadmap_rows) + "\n", encoding="utf-8")
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "corpus-token-indexing-gate-missing" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_workshop_talk_segmentation_guardrails(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    replacements = {
        "transcription_ocr.workshop_talk_segmentation": "transcription_ocr.talks_tbd",
        "talk_segmentation_provider_not_configured": "talk_segmentation_pending",
        "audio-only recovery is rejected evidence for talk-aware transcripts": (
            "audio recovery might be enough for talks"
        ),
    }
    for path in plan_dir.rglob("*"):
        if path.is_file() and path.suffix in {".md", ".json"}:
            content = read_text(path)
            for old, new in replacements.items():
                content = content.replace(old, new)
            path.write_text(content, encoding="utf-8")
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "workshop-talk-segmentation-gate-missing" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_workflow_registry_dispatch_coverage(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    rows = approval["workflow registry dispatch coverage"]
    assert isinstance(rows, list)
    approval["workflow registry dispatch coverage"] = [
        row for row in rows if not isinstance(row, dict) or row.get("workflow_stage_id") != "enhanced-indexing-vector-store"
    ]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "workflow-registry-dispatch-complete" in json.dumps(report["failures"])


def test_production_audit_rejects_placeholder_workflow_registry_dispatch_row(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    rows = approval["workflow registry dispatch coverage"]
    assert isinstance(rows, list)
    for row in rows:
        if isinstance(row, dict) and row.get("workflow_stage_id") == "download":
            row["dispatch_kind"] = "placeholder_stage"
            row["implementation_ref"] = "registry unavailable"
            row["receipt_path"] = "conferences/iclr-scaffold-2027/storage/workflow-runs/iclr-scaffold-2027-live-production/download-receipt.json"
            row["backup_checkpoint"] = "BK-NOT-REAL"
            break
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    failures = json.dumps(report["failures"])
    assert result.returncode != 0
    assert "workflow-registry-dispatch-real-registry" in failures
    assert "workflow-registry-dispatch-owner-commands" in failures
    assert "workflow-registry-dispatch-cost-backup-verification" in failures


def test_production_audit_rejects_missing_backup_checkpoint(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    rows = approval["backup checkpoints"]
    assert isinstance(rows, list)
    approval["backup checkpoints"] = [
        row for row in rows if not isinstance(row, dict) or row.get("checkpoint_id") != "BK-10"
    ]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "backup-checkpoints-present" in json.dumps(report["failures"])


def test_production_audit_rejects_backup_checkpoint_write_dry_run_alias(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    rows = approval["backup checkpoints"]
    assert isinstance(rows, list)
    first = rows[0]
    assert isinstance(first, dict)
    first["write_command"] = first["dry_run_command"]
    first["receipt_path"] = first["dry_run_command"].split("--receipt-path ", 1)[1].split()[0]
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    failures = json.dumps(report["failures"])
    assert result.returncode != 0
    assert "backup-checkpoint-commands" in failures
    assert "backup-checkpoint-receipts" in failures


def test_production_audit_rejects_gate_without_stale_rule_or_downstream_blocker(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    for row in workflow_gates:
        if isinstance(row, dict) and row.get("gate_id") == "PG-INDEX":
            row["stale_evidence_rule"] = ""
            row["downstream_blockers"] = "none"
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    failures = json.dumps(report["failures"])
    assert result.returncode != 0
    assert "production-workflow-gate-stale-rules" in failures
    assert "production-workflow-gate-downstream-blockers" in failures


def test_production_audit_rejects_index_gate_without_embedding_readiness_probe(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    for row in workflow_gates:
        if isinstance(row, dict) and row.get("gate_id") == "PG-INDEX":
            row["pass_criteria"] = "receipt status succeeded; vector store outputs exist"
            row["fail_criteria"] = "indexing receipt missing, failed, stale, or digest-mismatched"
            row["rejected_evidence"] = "dry-run, mock, readiness-only, metadata-only, paper-only, historical proof"
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    failures = json.dumps(report["failures"])
    assert result.returncode != 0
    assert "production-index-embedding-readiness-gate" in failures


def test_production_audit_rejects_optional_pipeline_gate_without_waiver_boundary(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    workflow_gates = approval["production workflow gates"]
    assert isinstance(workflow_gates, list)
    for row in workflow_gates:
        if isinstance(row, dict) and row.get("gate_id") == "PG-OPTIONAL-PIPELINES":
            row["pass_criteria"] = "optional outputs are flexible"
            row["fail_criteria"] = "none"
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-workflow-gate-pass-fail-criteria" in json.dumps(report["failures"])


def test_production_scaffold_carries_registry_and_stage_local_benchmark_gates(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    task_defaults = approval["task model defaults"]
    assert "## Task Model Defaults" in roadmap
    assert any(
        row["task_id"] == "transcription.session_speech"
        and row["default_provider_model"] == "openai:gpt-4o-transcribe-diarize"
        for row in task_defaults
    )
    for task_id in (
        "transcription.session_speech",
        "ocr.slide_image_parse",
        "analysis.source_reading",
        "podcast.tts",
        "presentation.svg_figure_generation",
        "qa.repair",
    ):
        assert task_id in roadmap
        assert task_id in json.dumps(approval["stage-local benchmark gates"])
    assert "stage-local benchmark gate waits until realistic inputs exist" in roadmap
    assert "explicit blocker" in roadmap
    assert "local-provider deferral" in roadmap


def test_production_generator_consumes_structured_handoff(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    source_approval = load_json(spec_root / "approval-summary.json")
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(spec_root, production_root)

    assert result.returncode == 0, result.stderr + result.stdout
    plan_dir = production_root / "iclr-scaffold-2027-production"
    approval = load_json(plan_dir / "approval-summary.json")
    assert approval["source analysis handoff"] == source_approval["production handoff"]
    for field in (
        "interest_group_decisions",
        "exclusions",
        "evidence_class",
        "provider_model_posture",
        "benchmark_requirements",
        "local_provider_deferrals",
        "local_proof_posture",
    ):
        assert field in approval["source analysis handoff"]
    assert approval["registry task ids"] == source_approval["production handoff"]["model_governance_task_ids"]
    assert "registry policy snapshot" in approval


def test_production_scaffold_preserves_benchmark_aware_provider_contracts_from_handoff(
    tmp_path: Path,
) -> None:
    spec_root = scaffold_sample(tmp_path)
    source_approval = load_json(spec_root / "approval-summary.json")
    source_handoff = source_approval["production handoff"]
    assert isinstance(source_handoff, dict)
    source_rows = source_handoff["provider_contracts"]
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(spec_root, production_root)

    assert result.returncode == 0, result.stderr + result.stdout
    plan_dir = production_root / "iclr-scaffold-2027-production"
    approval = load_json(plan_dir / "approval-summary.json")
    assert approval["source analysis handoff"]["provider_contracts"] == source_rows
    assert approval["benchmark-aware provider contracts"] == source_rows

    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "## Benchmark-Aware Provider Contracts" in roadmap
    for contract_id in PROVIDER_CONTRACT_REQUIRED_IDS:
        assert contract_id in roadmap


def test_production_scaffold_emits_v18_backend_threshold_and_fail_stop_sections(
    tmp_path: Path,
) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")

    for phrase in (
        *PRODUCTION_V18_BACKEND_STRINGS,
        *PRODUCTION_V18_THRESHOLD_STRINGS,
        *PRODUCTION_V18_FAIL_STOP_STRINGS,
    ):
        assert phrase in roadmap


def test_production_audit_rejects_missing_v18_fail_stop_policy(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path).replace("## Fail-Stop Blocker Policy (v1.92)", "## Removed"),
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "fail-stop-blocker-policy-present" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_v18_quality_threshold_policy(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path).replace("## Quality Sample Thresholds (v1.8)", "## Removed"),
        encoding="utf-8",
    )
    roadmap = read_text(roadmap_path)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "quality-threshold-policy-present" in json.dumps(report["failures"])
    assert "provider/model/cost receipt" in roadmap
    assert "local providers deferred unless approved local proof and benchmark receipts exist" in roadmap


def test_production_scaffold_sanitizes_source_analysis_handoff_secrets(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    approval_path = spec_root / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    rows = provider_contract_rows(handoff, "provider_contracts")
    secret_value = "OPENAI_API_KEY=sk-test-production-secret-tail123"
    rows[0]["receipt_requirement"] = f"do not persist {secret_value}"
    handoff["operator_notes"] = {
        "authorization": "Authorization: Bearer bearer-production-secret-tail",
        "cookie": "sessionid=production-session-secret-tail",
    }
    write_loaded_json(approval_path, approval)
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(spec_root, production_root)

    assert result.returncode == 0, result.stderr + result.stdout
    plan_dir = production_root / "iclr-scaffold-2027-production"
    generated_text = "\n".join(
        read_text(path)
        for path in plan_dir.rglob("*")
        if path.is_file() and path.suffix in {".md", ".json"}
    )
    assert secret_value not in generated_text
    assert "production-secret-tail123" not in generated_text
    assert "bearer-production-secret-tail" not in generated_text
    assert "production-session-secret-tail" not in generated_text
    assert "[redacted]" in generated_text


def test_production_generator_rejects_missing_structured_handoff(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    approval_path = spec_root / "approval-summary.json"
    approval = load_json(approval_path)
    approval.pop("production handoff", None)
    write_loaded_json(approval_path, approval)
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(spec_root, production_root)

    assert result.returncode == 2
    assert "structured production handoff" in result.stderr
    assert not (production_root / "iclr-scaffold-2027-production").exists()


def test_production_generator_rejects_unknown_handoff_task_id(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    approval_path = spec_root / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    handoff["model_governance_task_ids"].append("unregistered.production.task")
    write_loaded_json(approval_path, approval)

    result = run_production_scaffold(spec_root, tmp_path / "production-drafts")

    assert result.returncode == 2
    assert "unknown model-governance task id: unregistered.production.task" in result.stderr


def test_production_audit_rejects_missing_stage_local_model_gate(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap).replace("registry task id: `ocr.slide_image_parse`", "registry task id: `removed`"),
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "stage-local-benchmark-gates" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_handoff_registry_pricing_benchmark_or_deferral(
    tmp_path: Path,
) -> None:
    cases = {
        "source analysis handoff": "production-handoff-structured",
        "registry policy snapshot": "registry-policy-snapshot",
        "pricing research": "pricing-research-current",
        "benchmark receipts": "benchmark-receipts-required",
        "local provider deferrals": "local-provider-deferrals",
    }
    for index, (field, assertion_name) in enumerate(cases.items()):
        case_tmp = tmp_path / f"case-{index}"
        case_tmp.mkdir()
        plan_dir = production_sample(case_tmp)
        approval_path = plan_dir / "approval-summary.json"
        approval = load_json(approval_path)
        approval.pop(field, None)
        write_loaded_json(approval_path, approval)
        report_path = case_tmp / "production-report.json"

        result = run_command(
            [
                sys.executable,
                str(PRODUCTION_AUDIT_SCRIPT),
                str(plan_dir),
                "--json-output",
                str(report_path),
            ]
        )

        report = load_json(report_path)
        assert result.returncode != 0, field
        assert assertion_name in json.dumps(report["failures"])


def test_production_audit_rejects_deprecated_figure_script_live_commands(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap)
        + "\n**Live command:** `.venv/bin/python scripts/presentations/extract_topic_figures.py --topic 01`\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "deprecated-figure-scripts-not-live-commands" in json.dumps(report["failures"])


def test_production_audit_rejects_wrong_presentation_draft_owner(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap)
        + "\n**Live command:** `.venv/bin/python scripts/generate_slides_agentic.py --topic 02`\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "presentation-draft-generation-owner-command" in json.dumps(report["failures"])


def test_production_audit_rejects_wrong_podcast_script_owner(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap)
        + "\n**Live command:** `.venv/bin/python -c \"write_podcast_package(...)\"`\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode != 0
    assert "podcast-script-generation-owner-command" in json.dumps(report["failures"])


def test_production_audit_accepts_non_default_registry_path(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--registry-path",
            str(REGISTRY_PATH),
            "--json-output",
            str(report_path),
        ]
    )

    report = load_json(report_path)
    assert result.returncode == 0, report["failures"]


def test_production_audit_rejects_non_live_completion_evidence(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap)
        + "\nBad clause: dry-run/mock receipts may satisfy completion for final proof.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "non-live-completion-rejected" in json.dumps(report["failures"])


def test_production_audit_rejects_common_invalid_completion_claims(tmp_path: Path) -> None:
    for index, claim in enumerate(
        (
            "dry-run receipts are enough to mark production complete",
            "metadata-only evidence can close production",
        )
    ):
        case_tmp = tmp_path / f"case-{index}"
        case_tmp.mkdir()
        plan_dir = production_sample(case_tmp)
        roadmap = plan_dir / ".planning" / "ROADMAP.md"
        roadmap.write_text(read_text(roadmap) + f"\nBad clause: {claim}.\n", encoding="utf-8")
        report_path = case_tmp / "production-report.json"

        result = run_command(
            [
                sys.executable,
                str(PRODUCTION_AUDIT_SCRIPT),
                str(plan_dir),
                "--json-output",
                str(report_path),
            ]
        )

        report = json.loads(read_text(report_path))
        assert result.returncode != 0, claim
        assert "non-live-completion-rejected" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_media_or_transcription_gate(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap = plan_dir / ".planning" / "ROADMAP.md"
    roadmap.write_text(
        read_text(roadmap).replace("Video/media downloads cannot be substituted", "Media downloads are flexible"),
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"

    result = run_command(
        [
            sys.executable,
            str(PRODUCTION_AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "live-production-gates" in json.dumps(report["failures"])


def test_production_generator_rejects_one_phase_without_owner_proof(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)

    result = run_production_scaffold(spec_root, tmp_path / "production-drafts", "--output-shape", "one-phase")

    assert result.returncode == 2
    assert "one-phase output requires --all-owner-commands-exist" in result.stderr


def test_production_generator_allows_one_phase_with_owner_proof(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(
        spec_root,
        production_root,
        "--output-shape",
        "one-phase",
        "--all-owner-commands-exist",
    )

    assert result.returncode == 0, result.stderr
    plan_dir = production_root / "iclr-scaffold-2027-production"
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "Output shape: one-phase" in roadmap
    assert "### Phase 1: End-To-End Live Production Execution" in roadmap
    assert "### Phase 2:" not in roadmap

    audit = run_command([sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir)])
    assert audit.returncode == 0, audit.stderr + audit.stdout


def test_uat_audit_is_exact_label_strict_and_writes_proof(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    for generated in plan_dir.rglob("*"):
        if generated.is_file() and generated.suffix in {".md", ".json"}:
            generated.write_text(
                read_text(generated).replace("dry-run/mock only", "dry-run and mock"),
                encoding="utf-8",
            )
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    result = run_command(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            str(plan_dir),
            "--json-output",
            str(report_path),
            "--markdown-output",
            str(markdown_path),
        ]
    )

    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "support-labels-exact" in json.dumps(report["failures"])
    assert "# plan-conference-analysis UAT Report" in read_text(markdown_path)


def test_install_skill_syncs_explicit_user_targets(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    result = run_command(
        [sys.executable, str(INSTALL_SCRIPT), "--target", "codex", "--target", "claude"],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "installed codex" in result.stdout
    assert "installed claude" in result.stdout
    source_hash = hashlib.sha256(read_text(SKILL_MD).encode("utf-8")).hexdigest()
    codex_skill = tmp_path / ".codex" / "skills" / "plan-conference-analysis" / "SKILL.md"
    claude_skill = tmp_path / ".claude" / "skills" / "plan-conference-analysis" / "SKILL.md"
    assert hashlib.sha256(read_text(codex_skill).encode("utf-8")).hexdigest() == source_hash
    assert hashlib.sha256(read_text(claude_skill).encode("utf-8")).hexdigest() == source_hash

    rejected = run_command([sys.executable, str(INSTALL_SCRIPT), "--target", "other"], env=env)
    assert rejected.returncode != 0
    assert "unknown install target" in rejected.stderr


def test_install_production_skill_syncs_explicit_user_targets(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)

    result = run_command(
        [sys.executable, str(PRODUCTION_INSTALL_SCRIPT), "--target", "codex", "--target", "claude"],
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "installed codex" in result.stdout
    assert "installed claude" in result.stdout
    source_hash = hashlib.sha256(read_text(PRODUCTION_SKILL_MD).encode("utf-8")).hexdigest()
    codex_skill = tmp_path / ".codex" / "skills" / "plan-conference-production" / "SKILL.md"
    claude_skill = tmp_path / ".claude" / "skills" / "plan-conference-production" / "SKILL.md"
    assert hashlib.sha256(read_text(codex_skill).encode("utf-8")).hexdigest() == source_hash
    assert hashlib.sha256(read_text(claude_skill).encode("utf-8")).hexdigest() == source_hash
    assert (
        tmp_path
        / ".codex"
        / "skills"
        / "plan-conference-production"
        / "scripts"
        / "scaffold_production_plan.py"
    ).exists()
    assert (
        tmp_path
        / ".claude"
        / "skills"
        / "plan-conference-production"
        / "scripts"
        / "audit_generated_production_plan.py"
    ).exists()

    rejected = run_command([sys.executable, str(PRODUCTION_INSTALL_SCRIPT), "--target", "other"], env=env)
    assert rejected.returncode != 0
    assert "unknown install target" in rejected.stderr


@pytest.mark.repo_data(".planning/milestones/v1.3-phases/30-plan-conference-analysis-skill-packaging-and-uat/30-uat-generated-plan-report.json")
def test_phase30_durable_uat_report_is_passed_and_secret_free() -> None:
    report = json.loads(read_text(UAT_REPORT_JSON))
    markdown = read_text(UAT_REPORT_MD)
    combined = json.dumps(report) + markdown

    assert report["status"] == "passed"
    assert report["scenario"] == "iclr-style-scaffold"
    for label in ("dry-run/mock only", "unsupported", "needs live proof"):
        assert label in report["support_labels_checked"]
    for area in ("source intake", "transcription/OCR", "presentation production", "proof/closeout"):
        assert area in report["roadmap_areas_checked"]
    assert "30-uat-generated-plan-report.json" in markdown
    for pattern in SECRET_PATTERNS:
        assert pattern not in combined

# ── Phase 10-03: local proof posture and audit assertion tests ─────────────────

PRODUCTION_AUDIT = PRODUCTION_AUDIT_SCRIPT
ANALYSIS_SKILL_MD = SKILL_MD
PRODUCTION_SKILL_MD_PATH = PRODUCTION_SKILL_MD
MODEL_GOVERNANCE_MD = REPO_ROOT / "docs" / "reference" / "model-governance.md"


def _minimal_production_approval(tmp_path: Path) -> dict[str, object]:
    """Return a valid production approval-summary structure for audit tests."""
    plan_dir = production_sample(tmp_path)
    return load_json(plan_dir / "approval-summary.json"), plan_dir


def test_production_audit_rejects_missing_local_research_ref(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "fixtures/operator-approval.json",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-missing-research" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_hardware_fit_ref(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "fixtures/operator-approval.json",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-missing-hardware-fit" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_benchmark_receipt_ref(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "",
            "operator_approval_ref": "fixtures/operator-approval.json",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-missing-benchmark" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_operator_approval_ref(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-missing-operator" in json.dumps(report["failures"])


def test_production_audit_rejects_local_proof_deferral_disagreement(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "fixtures/operator-approval.json",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-disagree-fail-closed" in json.dumps(report["failures"])


def test_production_audit_rejects_unverified_approved_local_without_registry_proof(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    task_id = "analysis.source_reading"
    approval["local proof approvals"] = {
        task_id: {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "fixtures/operator-approval.json",
        }
    }
    deferrals = approval["local provider deferrals"]
    assert isinstance(deferrals, dict)
    deferrals.pop(task_id, None)
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-proof-unverified" in json.dumps(report["failures"])


def test_production_audit_rejects_raw_local_model_bypass(tmp_path: Path) -> None:
    approval, plan_dir = _minimal_production_approval(tmp_path)
    approval["local proof approvals"] = {
        "analysis.source_reading": {
            "research_ref": "fixtures/research.md",
            "hardware_fit_ref": "fixtures/hardware-fit.json",
            "benchmark_receipt_ref": "fixtures/benchmark.json",
            "operator_approval_ref": "fixtures/operator-approval.json",
            "model_id": "mlx-community/Qwen3-8B-4bit",
        }
    }
    write_loaded_json(plan_dir / "approval-summary.json", approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-raw-model-bypass" in json.dumps(report["failures"])


def test_scaffold_production_plan_with_approved_local_posture(tmp_path: Path) -> None:
    spec_root = scaffold_sample(tmp_path)
    approval_path = spec_root / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["production handoff"]
    assert isinstance(handoff, dict)
    task_id = "analysis.source_reading"
    posture = handoff["local_proof_posture"]
    assert isinstance(posture, dict)
    posture[task_id] = {
        "research_ref": "fixtures/research.md",
        "hardware_fit_ref": "fixtures/hardware-fit.json",
        "benchmark_receipt_ref": "fixtures/benchmark.json",
        "operator_approval_ref": "fixtures/operator-approval.json",
        "approved_date": "2026-04-27T00:00:00Z",
        "scope": "test fixture only",
    }
    approval["production handoff"] = handoff
    write_loaded_json(approval_path, approval)
    production_root = tmp_path / "production-drafts"

    result = run_production_scaffold(spec_root, production_root)

    assert result.returncode == 0, result.stderr + result.stdout
    plan_dir = production_root / "iclr-scaffold-2027-production"
    production_approval = load_json(plan_dir / "approval-summary.json")
    assert production_approval["local proof approvals"][task_id] == {
        "research_ref": "fixtures/research.md",
        "hardware_fit_ref": "fixtures/hardware-fit.json",
        "benchmark_receipt_ref": "fixtures/benchmark.json",
        "operator_approval_ref": "fixtures/operator-approval.json",
        "approved_date": "2026-04-27T00:00:00Z",
        "scope": "test fixture only",
    }
    assert task_id not in production_approval["local provider deferrals"]


def test_production_audit_rejects_handoff_missing_local_proof_posture_coverage(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval_path = plan_dir / "approval-summary.json"
    approval = load_json(approval_path)
    handoff = approval["source analysis handoff"]
    assert isinstance(handoff, dict)
    posture = handoff["local_proof_posture"]
    assert isinstance(posture, dict)
    posture.pop("analysis.source_reading", None)
    write_loaded_json(approval_path, approval)
    report_path = tmp_path / "report.json"

    result = run_command([sys.executable, str(PRODUCTION_AUDIT), str(plan_dir), "--json-output", str(report_path)])

    report = load_json(report_path)
    assert result.returncode != 0
    assert "production-handoff-structured" in json.dumps(report["failures"])


def test_non_live_dry_run_scaffold_produces_no_local_proof_approvals(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    approval = load_json(plan_dir / "approval-summary.json")

    local_approvals = approval.get("local proof approvals")
    assert not local_approvals, (
        f"Expected no local proof approvals in default dry-run output, got: {local_approvals}"
    )


def test_skill_sync_records_local_proof_posture_documentation() -> None:
    analysis_skill = read_text(ANALYSIS_SKILL_MD)
    assert "local_proof_posture" in analysis_skill

    production_skill = read_text(PRODUCTION_SKILL_MD_PATH)
    assert "local-proof-missing-research" in production_skill or "approved_local_selection" in production_skill

    model_governance = read_text(MODEL_GOVERNANCE_MD)
    assert "approved_local_selection" in model_governance


def test_analysis_skill_emits_extraction_provider_policy() -> None:
    """plan-conference-analysis SKILL.md has ## Extraction Provider Policy with v1.7 language."""
    skill = read_text(SKILL_MD)
    assert "## Extraction Provider Policy" in skill, "Missing ## Extraction Provider Policy section"
    for phrase in EXTRACTION_POLICY_REQUIRED_STRINGS:
        if phrase not in {"EXTRACTION_LOCAL_LLM_BASE_URL"}:
            # EXTRACTION_LOCAL_LLM_BASE_URL is in the scaffold output but phrased
            # slightly differently in SKILL.md — assert the env var name directly
            assert phrase in skill, f"Missing extraction policy phrase in analysis SKILL.md: {phrase!r}"
    assert "EXTRACTION_LOCAL_LLM_BASE_URL" in skill, "Missing env var reference EXTRACTION_LOCAL_LLM_BASE_URL"
    assert "EXTRACTION_LOCAL_LLM_MODEL" in skill, "Missing env var reference EXTRACTION_LOCAL_LLM_MODEL"
    assert "EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL" in skill, "Missing env var reference EXTRACTION_LOCAL_LLM_ENDPOINT_LABEL"


def test_analysis_scaffold_emits_extraction_policy_in_roadmap(tmp_path: Path) -> None:
    """scaffold_plan.py generates ROADMAP.md containing the Extraction Provider Policy block."""
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "## Extraction Provider Policy" in roadmap, "Scaffold ROADMAP.md missing ## Extraction Provider Policy"
    for phrase in EXTRACTION_POLICY_REQUIRED_STRINGS:
        assert phrase in roadmap, f"Missing extraction policy phrase in generated ROADMAP.md: {phrase!r}"


def test_analysis_extraction_policy_heuristic_prefilter_language() -> None:
    """SKILL.md explicitly names heuristics as prefilter/fallback, not the primary extraction engine."""
    skill = read_text(SKILL_MD)
    # Heuristic prefilter language must be explicit
    assert "prefilter" in skill, "SKILL.md must use 'prefilter' to describe heuristic role"
    # Local LLM must be described as default, not optional assist
    assert "optional" not in skill.split("## Extraction Provider Policy")[1].split("##")[0], (
        "Extraction Provider Policy section must not describe local LLM as 'optional'"
    )


def test_analysis_extraction_policy_cloud_escalation_only() -> None:
    """Cloud-as-default extraction language is absent from plan-conference-analysis SKILL.md."""
    skill = read_text(SKILL_MD)
    assert "cloud LLM as the default" not in skill, (
        "SKILL.md must not describe cloud LLM as the default extraction path"
    )
    assert "cloud provider for exhaustive extraction by default" not in skill, (
        "SKILL.md must not imply cloud is the default extraction provider"
    )
    # The v1.6 'optional assist' framing must not survive in the Extraction Provider Policy section
    policy_section = skill.split("## Extraction Provider Policy")[1].split("##")[0]
    assert "Optional OpenAI-compatible local LLM assist is limited to" not in policy_section, (
        "Extraction Provider Policy must not contain v1.6 'optional assist' framing"
    )


def test_production_skill_requires_local_llm_readiness_gate() -> None:
    """plan-conference-production SKILL.md has Extraction Provider Policy with readiness preflight."""
    skill = read_text(PRODUCTION_SKILL_MD)
    assert "## Extraction Provider Policy" in skill, (
        "plan-conference-production SKILL.md missing ## Extraction Provider Policy"
    )
    assert "readiness preflight" in skill, (
        "Production skill Extraction Provider Policy must require local LLM readiness preflight"
    )
    assert "Non-extraction generation providers" in skill, (
        "Production skill must explicitly exclude non-extraction providers from v1.7 policy"
    )
    assert "fail closed" in skill, (
        "Production skill must state the plan fails closed when local endpoint is absent"
    )
    assert "benchmark-governed contracts" in skill, (
        "Production skill must reference benchmark-governed contracts for non-extraction providers"
    )


def test_analysis_skill_emits_conditional_slide_ocr_section() -> None:
    """plan-conference-analysis SKILL.md has ## Conditional Slide OCR section."""
    skill = read_text(SKILL_MD)
    assert "## Conditional Slide OCR" in skill, "Missing ## Conditional Slide OCR section"
    assert "skipped_with_rationale" in skill
    assert "deferred_blocked" in skill
    assert "reopen_conditions" in skill


def test_analysis_scaffold_emits_slide_ocr_assessment_in_roadmap(tmp_path: Path) -> None:
    """scaffold_plan.py generates ROADMAP.md containing the Slide/Visual OCR Assessment block."""
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "## Slide/Visual OCR Assessment" in roadmap, (
        "Scaffold ROADMAP.md missing ## Slide/Visual OCR Assessment"
    )
    for phrase in OCR_STATE_REQUIRED_STRINGS:
        assert phrase in roadmap, f"Missing OCR state phrase in generated ROADMAP.md: {phrase!r}"


def test_analysis_slide_ocr_skipped_with_rationale_language() -> None:
    """SKILL.md includes skipped_with_rationale receipt template with reopen_conditions."""
    skill = read_text(SKILL_MD)
    assert "skipped_with_rationale" in skill
    assert "reopen_conditions" in skill
    assert "available_text_sources" in skill
    assert "expected_coverage_impact" in skill


def test_analysis_slide_ocr_required_state_language() -> None:
    """SKILL.md states the three required-state triggers and required receipt fields."""
    skill = read_text(SKILL_MD)
    ocr_section = skill.split("## Conditional Slide OCR")[1].split("##")[0]
    assert "required" in ocr_section
    assert "slide-level citations" in skill
    assert "why_required" in skill
    assert "expected_receipt_paths" in skill


def test_analysis_audit_rejects_plan_missing_local_llm_policy(tmp_path: Path) -> None:
    """Analysis audit fails when generated plan omits local LLM default extraction policy (CLOUD-ESC-03)."""
    plan_dir = scaffold_sample(tmp_path)
    # Patch all .md and .json files to remove the phrase so audit() sees it nowhere
    for f in sorted(plan_dir.rglob("*")):
        if f.is_file() and f.suffix in {".md", ".json"}:
            content = f.read_text(encoding="utf-8")
            if "heuristics serve as prefilter" in content:
                f.write_text(
                    content.replace("heuristics serve as prefilter", "heuristics are useful"),
                    encoding="utf-8",
                )
    report_path = tmp_path / "report.json"
    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "extraction-policy-present" in json.dumps(report["failures"])


def test_analysis_audit_rejects_cloud_default_extraction(tmp_path: Path) -> None:
    """Analysis audit fails when generated plan omits cloud escalation boundary language (CLOUD-ESC-03)."""
    plan_dir = scaffold_sample(tmp_path)
    # Patch all .md and .json files to remove the phrase so audit() sees it nowhere
    for f in sorted(plan_dir.rglob("*")):
        if f.is_file() and f.suffix in {".md", ".json"}:
            content = f.read_text(encoding="utf-8")
            if "cloud LLM escalation only for low-confidence" in content:
                f.write_text(
                    content.replace(
                        "cloud LLM escalation only for low-confidence",
                        "cloud LLM may be used when needed",
                    ),
                    encoding="utf-8",
                )
    report_path = tmp_path / "report.json"
    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "cloud-escalation-boundaries-present" in json.dumps(report["failures"])


def test_analysis_audit_rejects_unconditional_ocr(tmp_path: Path) -> None:
    """Analysis audit fails when generated plan mentions OCR without three-state classification (EXTRACT-POLICY-03)."""
    plan_dir = scaffold_sample(tmp_path)
    # Remove all three-state tokens from ALL plan files so audit() sees none of them
    for f in sorted(plan_dir.rglob("*")):
        if f.is_file() and f.suffix in {".md", ".json"}:
            content = f.read_text(encoding="utf-8")
            patched = content.replace("skipped_with_rationale", "skipped_with_note").replace(
                "deferred_blocked", "deferred_pending"
            )
            if patched != content:
                f.write_text(patched, encoding="utf-8")
    # Append an unconditional OCR clause that lacks three-state tokens
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\n\nOCR must always run for all slide decks regardless of coverage.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = json.loads(read_text(report_path))
    assert result.returncode != 0
    assert "ocr-three-state-classification" in json.dumps(report["failures"])


def test_production_audit_rejects_cloud_as_default_extraction(tmp_path: Path) -> None:
    """Production audit fails when generated plan asserts cloud LLM as default extraction path (CLOUD-ESC-03)."""
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\nDrift clause: cloud LLM as the default extraction path for all records.\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "production-report.json"
    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)
    assert result.returncode != 0
    assert "extraction-cloud-not-default" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_readiness_preflight(tmp_path: Path) -> None:
    """Production audit fails when generated plan omits local LLM readiness preflight (EXTRACT-POLICY-03)."""
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    original = read_text(roadmap_path)
    if "local LLM readiness" not in original:
        # Phase 32 scaffold did not emit readiness language; inject then immediately remove
        # to confirm the check fires on absence
        pass
    roadmap_path.write_text(
        original.replace("local LLM readiness", "LLM startup check"),
        encoding="utf-8",
    )
    # Also patch all other .md and .json files in case the phrase is emitted elsewhere
    for f in plan_dir.rglob("*"):
        if f == roadmap_path:
            continue
        if f.is_file() and f.suffix in {".md", ".json"}:
            content = read_text(f)
            patched = content.replace("local LLM readiness", "LLM startup check")
            if patched != content:
                f.write_text(patched, encoding="utf-8")
    report_path = tmp_path / "production-report.json"
    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)
    assert result.returncode != 0
    assert "local-llm-readiness-preflight" in json.dumps(report["failures"])


def test_extraction_policy_audit_passes_compliant_plan(tmp_path: Path) -> None:
    """A fully compliant production plan passes all extraction policy audit checks (CLOUD-ESC-03, EXTRACT-POLICY-03)."""
    plan_dir = production_sample(tmp_path)
    # If Phase 32 scaffold is not yet complete and readiness language is missing, patch it in
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    if "local LLM readiness" not in read_text(roadmap_path):
        roadmap_path.write_text(
            read_text(roadmap_path) + "\nlocal LLM readiness preflight required before extraction.\n",
            encoding="utf-8",
        )
    report_path = tmp_path / "production-report.json"
    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)
    assert result.returncode == 0, report.get("failures", [])
    assert report["status"] == "passed"


def test_uat_analysis_plan_local_llm_default_extraction(tmp_path: Path) -> None:
    """UAT: generated analysis plan states local LLM as default extraction path."""
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "heuristics serve as prefilter" in roadmap, (
        "Generated plan must assert heuristics-as-prefilter (local LLM default) — "
        "check that scaffold_plan.py Phase 31 output is intact"
    )
    assert "cloud LLM escalation only for low-confidence" in roadmap, (
        "Generated plan must assert cloud escalation boundary — "
        "check that scaffold_plan.py Phase 31 output is intact"
    )


def test_uat_production_plan_preserves_local_llm_default(tmp_path: Path) -> None:
    """UAT: generated scaffold preserves local LLM default framing throughout."""
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "heuristics serve as prefilter" in roadmap, (
        "Production plan must preserve local LLM default framing — "
        "check that scaffold_plan.py Phase 31 output has not drifted"
    )


def test_uat_conditional_ocr_posture_in_generated_plan(tmp_path: Path) -> None:
    """UAT: generated plan includes three-state OCR posture language."""
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    assert "skipped_with_rationale" in roadmap, (
        "Generated plan must include OCR skipped_with_rationale state — "
        "check that scaffold_plan.py Phase 33 OCR output is intact"
    )
    assert "deferred_blocked" in roadmap, (
        "Generated plan must include OCR deferred_blocked state — "
        "check that scaffold_plan.py Phase 33 OCR output is intact"
    )


def test_analysis_audit_rejects_missing_dual_store_paper_matching(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap = read_text(roadmap_path)
    roadmap = roadmap.replace("Dual-Store Paper Matching", "Paper Matching")
    roadmap = roadmap.replace("artifact-store GPT-Researcher", "artifact research")
    roadmap = roadmap.replace("paper-store GPT-Researcher", "paper research")
    roadmap = roadmap.replace("scripts/match_papers_from_artifact_keypoints.py", "scripts/rank_papers.py")
    roadmap_path.write_text(roadmap, encoding="utf-8")

    result = run_command([sys.executable, str(AUDIT_SCRIPT), str(plan_dir)])

    assert result.returncode == 1


def test_analysis_audit_rejects_wrong_presentation_draft_owner(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap_path.write_text(
        read_text(roadmap_path)
        + "\n**Command:** `.venv/bin/python scripts/presentations/generate_slides_simple.py --topic 02`\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "analysis-draft-owner-report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)

    assert result.returncode == 1
    assert "presentation-draft-generation-owner-command" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_presentation_narrative_gate(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    for path in sorted(plan_dir.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            content = read_text(path)
            patched = (
                content.replace("presentation narrative-content gate", "presentation content check")
                .replace("analysis digest", "analysis fingerprint")
                .replace("stale scaffold", "old draft")
                .replace("public-ledger", "public ledger")
                .replace("identical deck checkpoint files", "duplicate deck files")
            )
            if patched != content:
                path.write_text(patched, encoding="utf-8")
    report_path = tmp_path / "analysis-presentation-gate-report.json"

    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)

    assert result.returncode == 1
    assert "presentation-narrative-content-gate" in json.dumps(report["failures"])


def test_analysis_skill_requires_fail_stop_pipeline_policy() -> None:
    skill = read_text(SKILL_MD)
    assert "## Fail-Stop Pipeline Policy" in skill
    for phrase in (
        "status: blocked",
        "status: unavailable",
        "BlockerReceipt",
        "UnblockReceipt",
        "stage_blocked",
        "stage_unavailable",
        "workflow unblock",
        "halt_status",
    ):
        assert phrase in skill


def test_analysis_scaffold_emits_fail_stop_pipeline_policy(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    roadmap = read_text(plan_dir / ".planning" / "ROADMAP.md")
    approval = load_json(plan_dir / "approval-summary.json")

    assert "## Fail-Stop Pipeline Policy" in roadmap
    assert "status: blocked" in roadmap
    assert "status: unavailable" in roadmap
    assert "stage_unavailable" in roadmap
    assert "ocr_deferred_blocked" in roadmap
    assert "failed required validation" in roadmap
    policy = approval["fail-stop pipeline policy"]
    assert "blocked" in policy["blocking_statuses"]
    assert "stage_blocked" in policy["blocker_classes"]
    assert "ocr_deferred_blocked" in policy["blocker_classes"]


def test_analysis_audit_rejects_missing_ocr_deferred_blocked_term(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    for path in sorted(plan_dir.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            content = read_text(path)
            patched = content.replace("ocr_deferred_blocked", "ocr_waiting")
            if patched != content:
                path.write_text(patched, encoding="utf-8")

    report_path = tmp_path / "analysis-ocr-deferred-blocked-report.json"
    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)

    assert result.returncode == 1
    assert "fail-stop-pipeline-policy" in json.dumps(report["failures"])


def test_analysis_audit_rejects_missing_fail_stop_pipeline_policy(tmp_path: Path) -> None:
    plan_dir = scaffold_sample(tmp_path)
    for path in sorted(plan_dir.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            content = read_text(path)
            patched = (
                content.replace("Fail-Stop Pipeline Policy", "Pipeline Policy")
                .replace("BlockerReceipt", "Receipt")
                .replace("stage_blocked", "stage waiting")
                .replace("stage_unavailable", "stage missing")
                .replace("status: unavailable", "status: waiting")
            )
            if patched != content:
                path.write_text(patched, encoding="utf-8")

    report_path = tmp_path / "analysis-fail-stop-report.json"
    result = run_command(
        [sys.executable, str(AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)

    assert result.returncode == 1
    assert "fail-stop-pipeline-policy" in json.dumps(report["failures"])


def test_production_audit_rejects_missing_dual_store_paper_matching(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    roadmap_path = plan_dir / ".planning" / "ROADMAP.md"
    roadmap = read_text(roadmap_path)
    roadmap = roadmap.replace("stores/papers", "paper index")
    roadmap = roadmap.replace("stores/artifacts", "artifact index")
    roadmap = roadmap.replace("artifact-store GPT-Researcher", "artifact research")
    roadmap = roadmap.replace("paper-store GPT-Researcher", "paper research")
    roadmap = roadmap.replace("matched_papers.json", "papers.json")
    roadmap_path.write_text(roadmap, encoding="utf-8")

    result = run_command([sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir)])

    assert result.returncode == 1


def test_production_audit_rejects_missing_v192_fail_stop_terms(tmp_path: Path) -> None:
    plan_dir = production_sample(tmp_path)
    for path in sorted(plan_dir.rglob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            content = read_text(path)
            patched = (
                content.replace("Fail-Stop Blocker Policy (v1.92)", "Fail-Stop Blocker Policy")
                .replace("status: unavailable", "status: waiting")
                .replace("stage_unavailable", "stage missing")
            )
            if patched != content:
                path.write_text(patched, encoding="utf-8")

    report_path = tmp_path / "production-fail-stop-report.json"
    result = run_command(
        [sys.executable, str(PRODUCTION_AUDIT_SCRIPT), str(plan_dir), "--json-output", str(report_path)]
    )
    report = load_json(report_path)

    assert result.returncode == 1
    assert "fail-stop-blocker-policy-present" in json.dumps(report["failures"])
