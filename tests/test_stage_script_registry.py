from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from conference_pipeline.stage_script_audit import (
    audit_plan_stage_script_registry,
    audit_run_stage_script_usage,
    audit_skill_stage_script_contracts,
    audit_stage_script_registry_integrity,
    audit_stage_script_usage_docs,
)
from conference_pipeline.stage_script_registry import (
    StageScriptRegistryError,
    load_stage_script_registry,
    registry_sha256,
    validate_registered_usage_doc_refs,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

PHASE15_PRESENTATION_OWNER_COMMANDS = {
    "presentation.source_corpus_inventory": {
        "path": "scripts/presentations/build_presentation_source_corpus_inventory.py",
        "output": "source_corpus_inventory.json",
        "usage_anchor": "docs/reference/stage-owner-audit.md#presentation-source-corpus-inventory",
    },
    "presentation.speaker_author_work_ledger": {
        "path": "scripts/presentations/build_speaker_author_work_ledger.py",
        "output": "speaker_author_work_ledger.json",
        "usage_anchor": "docs/reference/stage-owner-audit.md#presentation-speaker-author-work-ledger",
    },
    "presentation.repair_presentation_sources": {
        "path": "scripts/presentations/repair_presentation_sources.py",
        "output": "source_repair_findings.json",
        "usage_anchor": "docs/reference/stage-owner-audit.md#presentation-repair-presentation-sources",
    },
    "presentation.research_cluster_plan": {
        "path": "scripts/presentations/build_research_cluster_plan.py",
        "output": "research_cluster_plan.json",
        "usage_anchor": "docs/reference/stage-owner-audit.md#presentation-research-cluster-plan",
    },
    "presentation.artifact_truth_validation": {
        "path": "scripts/presentations/validate_artifact_truth.py",
        "output": "artifact_truth_report.json",
        "usage_anchor": "docs/reference/stage-owner-audit.md#presentation-artifact-truth-validation",
    },
}

PHASE15_PRESENTATION_OWNER_USAGE_TERMS = {
    "presentation.source_corpus_inventory": (
        "source corpus inventory",
        "source_corpus_inventory_receipt.json",
    ),
    "presentation.speaker_author_work_ledger": (
        "speaker/author/work ledger",
        "speaker_author_work_ledger_receipt.json",
    ),
    "presentation.repair_presentation_sources": (
        "source repair findings",
        "source_repair_receipt.json",
    ),
    "presentation.research_cluster_plan": (
        "research cluster plan",
        "research_cluster_plan_receipt.json",
    ),
    "presentation.artifact_truth_validation": (
        "artifact-truth validation",
        "artifact truth",
    ),
}

PHASE15_PRESENTATION_OWNER_USAGE_COMMON_TERMS = (
    "Purpose:",
    "Command:",
    "Outputs:",
    "Diagnostics/recovery:",
    "Rejected evidence:",
    "canary acceptance",
    "[Presentation contract](../operations/presentation-contract.md)",
)


def _stage_owner_usage_row(anchor: str) -> str:
    docs_text = (REPO_ROOT / "docs" / "reference" / "stage-owner-audit.md").read_text(
        encoding="utf-8"
    )
    marker = f'id="{anchor}"'
    start = docs_text.index(marker)
    end = docs_text.find("\n| <a id=", start + len(marker))
    if end == -1:
        end = docs_text.find("\n\n###", start + len(marker))
    if end == -1:
        end = len(docs_text)
    return docs_text[start:end]


def test_stage_script_registry_loads_schema_v2_and_required_paths_exist() -> None:
    registry = load_stage_script_registry()

    assert registry["schema_version"] == 2
    assert "planning.workspace_gitignore_policy" in registry["planning_required_task_ids"]
    assert "analysis.gpt_researcher_hybrid_retrieval" in registry["planning_required_task_ids"]
    assert "analysis.gpt_researcher_heuristic_prefilter" in registry["planning_required_task_ids"]
    assert "analysis.topic_reconciliation_reference_workflow" in registry[
        "planning_required_task_ids"
    ]
    assert "analysis.topic_reference_validation" in registry["planning_required_task_ids"]
    assert "analysis.topic_reference_candidate_resolution" in registry["planning_required_task_ids"]
    assert "analysis.topic_reference_repair" in registry["planning_required_task_ids"]
    assert "gitignore.workspace_policy" in registry["scripts"]
    assert "analysis.gpt_researcher_stage" in registry["scripts"]
    assert "analysis.gpt_researcher_heuristic_prefilter" in registry["scripts"]
    assert "analysis.topic_reconciliation_reference_workflow" in registry["scripts"]
    assert "analysis.topic_reference_validation" in registry["scripts"]
    assert "analysis.topic_reference_candidate_resolution" in registry["scripts"]
    assert "analysis.topic_reference_repair" in registry["scripts"]
    assert "presentation.visual_qc" in registry["task_bindings"]
    assert "presentation.presentation_review" in registry["task_bindings"]
    assert "analysis.gpt_researcher_hybrid_retrieval" in registry["task_bindings"]
    assert "analysis.gpt_researcher_heuristic_prefilter" in registry["task_bindings"]
    assert "indexing.token_bounded_chunking" in registry["scripts"]
    assert "indexing.local_embedding_materialization_policy" in registry["scripts"]
    assert "indexing.token_bounded_chunking" in registry["task_bindings"]
    assert "indexing.local_embedding_materialization_policy" in registry["task_bindings"]
    assert "presentation.add_title_slide" in registry["scripts"]
    assert "presentation.generate_editorial_blueprint" in registry["scripts"]
    assert "presentation.inspect_rendered_slides_llm" in registry["scripts"]
    assert "presentation.presentation_review" in registry["scripts"]
    assert "presentation.phase13_canary_review_packet" in registry["scripts"]
    assert "presentation.phase13_rollout" in registry["scripts"]
    assert "transcription_ocr.local_media_processing" in registry["scripts"]
    assert "transcription_ocr.workshop_talk_boundary_discovery" in registry["scripts"]
    assert "transcription_ocr.workshop_talk_segmentation" in registry["scripts"]
    assert "acquisition.embedded_media_recovery" in registry["scripts"]
    assert "transcription-ocr" in registry["stage_bindings"]
    assert "transcribe" in registry["stage_bindings"]
    assert registry["stage_bindings"]["corpus-inventory-index"]["required_script_ids"] == [
        "indexing.token_bounded_chunking",
        "indexing.local_embedding_materialization_policy",
    ]
    assert registry["stage_bindings"]["enhanced-indexing-vector-store"]["required_script_ids"] == [
        "indexing.token_bounded_chunking",
        "indexing.local_embedding_materialization_policy",
    ]
    assert registry["stage_bindings"]["transcription-ocr"]["required_script_ids"] == [
        "transcription_ocr.local_media_processing"
    ]
    assert registry["stage_bindings"]["transcribe"]["required_script_ids"] == [
        "transcription_ocr.local_media_processing"
    ]
    assert "transcription_ocr.workshop_talk_segmentation" in registry["stage_bindings"][
        "transcription-ocr"
    ]["optional_script_ids"]
    assert "transcription_ocr.workshop_talk_boundary_discovery" in registry["stage_bindings"][
        "transcription-ocr"
    ]["optional_script_ids"]
    assert "transcription_ocr.workshop_talk_segmentation" in registry["stage_bindings"][
        "transcribe"
    ]["optional_script_ids"]
    assert "transcription_ocr.workshop_talk_boundary_discovery" in registry["task_bindings"]
    assert registry["task_bindings"]["acquisition.workshop_talk_media"]["required_script_ids"] == [
        "acquisition.workshop_talk_media",
        "acquisition.embedded_media_recovery",
    ]
    assert "presentation.video_generation" in registry["planning_required_task_ids"]
    assert "presentation.figure_validation" in registry["planning_required_task_ids"]
    assert "presentation.video_generation" in registry["task_bindings"]
    assert registry["task_bindings"]["presentation.figure_validation"]["required_command_terms"] == [
        "scripts/presentations/validate_svgs.py",
        "--workspace",
        "--presentation-id",
        "--dir",
        "decks/<id>/assets/generated",
        "--report-output",
        "decks/<id>/assets/generated/validation_report.md",
    ]
    assert "presentation" in registry["stage_bindings"]
    assert "analysis" in registry["stage_bindings"]
    assert "analysis.exhaustive_topic_extraction" in registry["scripts"]
    assert "analysis.exhaustive_topic_extraction" in registry["stage_bindings"]["analysis"][
        "required_script_ids"
    ]
    assert "analysis.gpt_researcher_stage" in registry["stage_bindings"]["analyze"]["required_script_ids"]
    assert "analysis.gpt_researcher_heuristic_prefilter" in registry["stage_bindings"]["analysis"][
        "required_script_ids"
    ]
    assert "analysis.topic_reference_validation" in registry["stage_bindings"]["analysis"][
        "required_script_ids"
    ]
    assert "analysis.topic_reconciliation_reference_workflow" in registry["stage_bindings"][
        "analysis"
    ]["required_script_ids"]
    assert "analysis.topic_reference_candidate_resolution" in registry["stage_bindings"][
        "analysis"
    ]["optional_script_ids"]
    assert "analysis.topic_reference_repair" in registry["stage_bindings"]["analysis"][
        "optional_script_ids"
    ]
    qa_gap_contract = registry["stage_bindings"]["qa-gap-fill-synthesis"][
        "runtime_contract"
    ]
    assert not qa_gap_contract.get("non_production")
    assert registry["stage_bindings"]["qa-gap-fill-synthesis"]["required_script_ids"] == [
        "analysis.coverage_gap_fill",
        "analysis.synthesis_trend_report",
    ]
    qa_gap_outputs = {
        item.get("path") for item in qa_gap_contract["required_outputs"]
    }
    assert {
        "analysis/qa/traceability_report.json",
        "analysis/qa/coverage_gap_report.json",
        "analysis/executive_summary.md",
        "analysis/trend_report.md",
    } <= qa_gap_outputs
    podcast_contract = registry["stage_bindings"]["podcast"]["runtime_contract"]
    assert registry["stage_bindings"]["podcast"]["required_script_ids"] == [
        "podcast.focused_script",
        "podcast.render_audio",
    ]
    podcast_outputs = {
        item.get("glob") or item.get("path") for item in podcast_contract["required_outputs"]
    }
    assert {
        "audio/*/focused_podcast/script.json",
        "audio/*/focused_podcast/audio_manifest.json",
        "audio/*/focused_podcast/audio_ready_package.json",
        "audio/*/focused_podcast/podcast.mp3",
    } <= podcast_outputs
    topic_reference_receipts = registry["stage_bindings"]["analysis"]["runtime_contract"][
        "required_receipts"
    ]
    topic_reference_receipt = next(
        item
        for item in topic_reference_receipts
        if item.get("path") == "storage/workflow-runs/<run-id>/07E-topic-reference-validation.json"
    )
    topic_reference_checks = {
        check["path"]: check for check in topic_reference_receipt["checks"] if "path" in check
    }
    assert topic_reference_checks["checks.ledger_entry_count"]["truthy"] is True
    assert topic_reference_checks["checks.citation_count"]["truthy"] is True
    assert topic_reference_checks["checks.bare_reference_warning_count"]["equals"] == 0
    topic_reference_workflow_receipt = next(
        item
        for item in topic_reference_receipts
        if item.get("path")
        == "storage/workflow-runs/<run-id>/07H-topic-reconciliation-reference-workflow.json"
    )
    topic_reference_workflow_checks = {
        check["path"]: check
        for check in topic_reference_workflow_receipt["checks"]
        if "path" in check
    }
    assert topic_reference_workflow_checks["checks.final_validation_status"]["equals"] == "succeeded"
    assert topic_reference_workflow_checks["checks.final_invalid_reference_count"]["equals"] == 0
    assert topic_reference_workflow_checks["checks.final_unknown_reference_count"]["equals"] == 0
    assert topic_reference_workflow_checks["checks.final_title_mismatch_count"]["equals"] == 0
    assert topic_reference_workflow_checks["checks.final_bare_reference_warning_count"][
        "equals"
    ] == 0
    assert registry["task_bindings"]["analysis.gpt_researcher_heuristic_prefilter"][
        "required_script_ids"
    ] == [
        "analysis.gpt_researcher_stage",
        "analysis.gpt_researcher_heuristic_prefilter",
    ]
    assert "presentation.md_to_revealjs" in registry["stage_bindings"]["render_export"][
        "required_script_ids"
    ]
    assert registry["stage_bindings"]["video_generation"]["required_script_ids"] == [
        "presentation.build_presenter_video",
        "presentation.build_notebooklm_video",
    ]
    assert {
        "gitignore.workspace_policy",
        "acquisition.workshop_talk_media",
        "acquisition.embedded_media_recovery",
        "analysis.topic_reconciliation_reference_workflow",
        "presentation.presenter_script",
        "presentation.phase13_canary_review_packet",
        "presentation.phase13_rollout",
        "presentation.narration_series_review",
        "presentation.build_presenter_video",
        "presentation.build_notebooklm_video",
    } <= set(registry["skill_contracts"]["plan-conference-production"]["required_script_ids"])

    report = audit_stage_script_registry_integrity()
    assert report["status"] == "pass", report


def test_runtime_stage_contract_registry_covers_workflow_and_production_stages() -> None:
    from conference_pipeline.workflow import WORKFLOW_STAGES

    registry = load_stage_script_registry()
    stage_bindings = registry["stage_bindings"]
    missing_workflow = [stage.id for stage in WORKFLOW_STAGES if stage.id not in stage_bindings]
    assert missing_workflow == []

    runner_text = (REPO_ROOT / "scripts" / "run_iclr2026_production.py").read_text(
        encoding="utf-8"
    )
    production_stage_ids = re.findall(
        r'"[^"]+": ProductionStage\(\n\s*id="([^"]+)"',
        runner_text,
    )
    assert production_stage_ids
    missing_production = [
        stage_id for stage_id in production_stage_ids if stage_id not in stage_bindings
    ]
    assert missing_production == []
    for stage_id in set(production_stage_ids).union(stage.id for stage in WORKFLOW_STAGES):
        binding = stage_bindings[stage_id]
        assert isinstance(binding.get("runtime_contract"), dict), stage_id


def test_stage_script_registry_usage_doc_refs_resolve() -> None:
    registry = load_stage_script_registry()

    for script_id, script in registry["scripts"].items():
        refs = script.get("usage_doc_refs")
        assert isinstance(refs, list) and refs, script_id
        assert all(str(ref).startswith("docs/reference/stage-owner-audit.md#") for ref in refs)

    report = audit_stage_script_usage_docs(REPO_ROOT / "docs")

    assert report["status"] == "pass", report


def test_stage_script_registry_diagnostic_profiles_cover_all_tools() -> None:
    registry = load_stage_script_registry()

    for script_id, script in registry["scripts"].items():
        profile = script.get("diagnostic_profile")
        assert isinstance(profile, dict), script_id
        assert profile.get("success_summary"), script_id
        assert profile.get("blocked_summary"), script_id
        assert profile.get("next_actions"), script_id
        assert profile.get("blocked_next_actions"), script_id
        assert "outputs" in profile, script_id
        if script.get("status") == "deprecated":
            replacements = profile.get("replacement_script_ids")
            assert replacements, script_id
            assert set(replacements) <= set(registry["scripts"]), script_id


def test_phase15_presentation_owner_commands_are_registry_owned() -> None:
    registry = load_stage_script_registry()
    stage_bindings = registry["stage_bindings"]

    for script_id, expected in PHASE15_PRESENTATION_OWNER_COMMANDS.items():
        script = registry["scripts"][script_id]
        assert script["path"] == expected["path"]
        assert script["kind"] == "script"
        assert script["status"] == "required"
        assert script["scope"] == "presentation"
        assert script["usage_doc_refs"] == [expected["usage_anchor"]]
        assert expected["path"] in script["refs"]
        assert expected["output"] in script["diagnostic_profile"]["outputs"]
        assert script["diagnostic_profile"]["success_summary"]
        assert script["diagnostic_profile"]["blocked_summary"]
        assert script["diagnostic_profile"]["next_actions"]
        assert script["diagnostic_profile"]["blocked_next_actions"]
        assert any(
            script_id in binding.get("required_script_ids", [])
            for binding in stage_bindings.values()
        )


def test_phase15_presentation_usage_doc_anchors_describe_evidence_contract() -> None:
    for script_id, expected in PHASE15_PRESENTATION_OWNER_COMMANDS.items():
        anchor = expected["usage_anchor"].rsplit("#", 1)[1]
        row = _stage_owner_usage_row(anchor)

        for term in (
            script_id,
            expected["path"],
            expected["output"],
            *PHASE15_PRESENTATION_OWNER_USAGE_TERMS[script_id],
            *PHASE15_PRESENTATION_OWNER_USAGE_COMMON_TERMS,
        ):
            assert term in row


def test_stage_script_registry_blocks_missing_diagnostic_profile(tmp_path: Path) -> None:
    registry = load_stage_script_registry()
    registry["scripts"]["presentation.validate_svgs"].pop("diagnostic_profile")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    try:
        load_stage_script_registry(registry_path)
    except StageScriptRegistryError as exc:
        assert "diagnostic_profile" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing diagnostic_profile should block registry loading")


def test_stage_script_registry_blocks_missing_usage_doc_refs(tmp_path: Path) -> None:
    registry = load_stage_script_registry()
    registry["scripts"]["presentation.validate_svgs"].pop("usage_doc_refs")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    try:
        load_stage_script_registry(registry_path)
    except StageScriptRegistryError as exc:
        assert "usage_doc_refs" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing usage_doc_refs should block registry loading")


def test_stage_script_registry_blocks_missing_usage_doc_anchor(tmp_path: Path) -> None:
    registry = load_stage_script_registry()
    registry["scripts"]["presentation.validate_svgs"]["usage_doc_refs"] = [
        "docs/reference/stage-owner-audit.md#missing-validate-anchor"
    ]
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = audit_stage_script_usage_docs(REPO_ROOT / "docs", registry_path=registry_path)

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"stage_script_registry_usage_doc_anchor_missing"}


def test_stage_script_registry_allows_claude_usage_doc_refs(tmp_path: Path) -> None:
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    claude_root = tmp_path / ".claude"
    claude_root.mkdir()
    (claude_root / "stage-usage.md").write_text(
        "## Example Usage\n",
        encoding="utf-8",
    )
    registry = {
        "scripts": {
            "example": {
                "status": "required",
                "usage_doc_refs": [".claude/stage-usage.md#example-usage"],
            }
        }
    }

    assert validate_registered_usage_doc_refs(
        registry,
        repo_root=tmp_path,
        docs_root=docs_root,
    ) == []


def test_stage_script_registry_skill_contracts_cover_explicit_skill_mentions() -> None:
    report = audit_skill_stage_script_contracts(REPO_ROOT / ".claude" / "skills")

    assert report["status"] == "pass", report


def test_plan_stage_script_registry_requires_digest_and_verifier(tmp_path: Path) -> None:
    digest = registry_sha256()
    plan_text = "\n".join(
        [
            "scripts/presentations/inspect_rendered_slides.py",
            "scripts/presentations/inspect_rendered_slides_llm.py",
            "visual_llm_qc_report.json",
            "presentation.visual_qc",
            "scripts/verify_stage_script_usage.py",
            digest,
        ]
    )
    (tmp_path / "ROADMAP.md").write_text(plan_text, encoding="utf-8")

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.visual_qc",),
        require_registry_digest=True,
        require_verifier_command=True,
    )

    assert report["status"] == "pass", report


def test_plan_stage_script_registry_requires_usage_doc_refs(tmp_path: Path) -> None:
    digest = registry_sha256()
    registry = load_stage_script_registry()
    visual_qc_docs = "\n".join(
        registry["scripts"][script_id]["usage_doc_refs"][0]
        for script_id in (
            "presentation.inspect_rendered_slides",
            "presentation.inspect_rendered_slides_llm",
        )
    )
    (tmp_path / "ROADMAP.md").write_text(
        "\n".join(
            [
                "scripts/presentations/inspect_rendered_slides.py",
                "scripts/presentations/inspect_rendered_slides_llm.py",
                "visual_llm_qc_report.json",
                "presentation.visual_qc",
                "scripts/verify_stage_script_usage.py",
                digest,
                visual_qc_docs,
            ]
        ),
        encoding="utf-8",
    )

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.visual_qc",),
        require_registry_digest=True,
        require_verifier_command=True,
        require_usage_doc_refs=True,
    )

    assert report["status"] == "pass", report


def test_plan_stage_script_registry_blocks_missing_usage_doc_refs(tmp_path: Path) -> None:
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/presentations/inspect_rendered_slides.py\n"
        "scripts/presentations/inspect_rendered_slides_llm.py\n"
        "visual_llm_qc_report.json\n"
        "presentation.visual_qc\n"
        "scripts/verify_stage_script_usage.py\n"
        f"{registry_sha256()}\n",
        encoding="utf-8",
    )

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.visual_qc",),
        require_registry_digest=True,
        require_verifier_command=True,
        require_usage_doc_refs=True,
    )

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"stage_script_registry_usage_doc_ref_missing_from_plan"}
    assert report["diagnostic"]["checked_inputs"]["plan_root"] == str(tmp_path)
    assert report["diagnostic"]["next_actions"]
    assert "next_action" in report["blockers"][0]
    assert "expected" in report["blockers"][0]
    assert "observed" in report["blockers"][0]


def test_plan_stage_script_registry_blocks_missing_digest(tmp_path: Path) -> None:
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/presentations/inspect_rendered_slides.py\n"
        "scripts/presentations/inspect_rendered_slides_llm.py\n"
        "visual_llm_qc_report.json\n"
        "presentation.visual_qc\n"
        "scripts/verify_stage_script_usage.py\n",
        encoding="utf-8",
    )

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.visual_qc",),
        require_registry_digest=True,
        require_verifier_command=True,
    )

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"stage_script_registry_digest_missing_from_plan"}


def test_plan_stage_script_registry_scans_historical_docs_for_active_forbidden_command(
    tmp_path: Path,
) -> None:
    registry = load_stage_script_registry()
    binding = registry["task_bindings"]["presentation.draft_generation"]
    (tmp_path / "ROADMAP.md").write_text(
        "# Historical deprecated owner migration notes\n"
        + "\n".join(binding["required_command_terms"])
        + "\nReject scripts/generate_slides_agentic.py as a completion owner.\n"
        + "python scripts/generate_slides_agentic.py --topic 02\n",
        encoding="utf-8",
    )

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_forbidden_script_used_in_plan"
        for blocker in report["blockers"]
    )


def test_plan_stage_script_registry_blocks_forbidden_command_in_policy_list(
    tmp_path: Path,
) -> None:
    registry = load_stage_script_registry()
    binding = registry["task_bindings"]["presentation.draft_generation"]
    (tmp_path / "ROADMAP.md").write_text(
        "\n".join(binding["required_command_terms"])
        + "\nDeprecated completion owners:\n"
        + "- python scripts/generate_slides_agentic.py --topic 02\n",
        encoding="utf-8",
    )

    report = audit_plan_stage_script_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_forbidden_script_used_in_plan"
        for blocker in report["blockers"]
    )


def test_run_stage_script_registry_blocks_missing_required_receipt_owner(tmp_path: Path) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": [
                    "presentation.inspect_rendered_slides",
                    "presentation.inspect_rendered_slides_llm",
                ],
                "commands": [
                    ["python", "scripts/presentations/inspect_rendered_slides.py"]
                ],
                "command_metadata": [
                    {
                        "owner_ref": "scripts/presentations/inspect_rendered_slides.py"
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "stage_script_registry_required_script_missing_from_receipt" in codes
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "stage_script_registry_required_script_missing_from_receipt"
    )
    assert "Rerun the stage" in blocker["next_action"]
    assert report["diagnostic"]["checked_inputs"]["run_id"] == "run-1"


def test_run_stage_script_registry_blocks_summary_completed_stage_without_receipt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"stages": [{"stage_id": "visual_qc", "status": "succeeded"}]}),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"stage_script_registry_completed_stage_receipt_missing"}


def test_run_stage_script_registry_blocks_unreadable_registry_receipt(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text("{bad json", encoding="utf-8")

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert {
        blocker["code"] for blocker in report["blockers"]
    } == {"stage_script_registry_receipt_unreadable"}


def test_run_stage_script_registry_scopes_manifest_evidence_to_stage(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    decks_dir = tmp_path / "decks"
    decks_dir.mkdir()
    (decks_dir / "presentation_manifest.json").write_text(
        json.dumps(
            {
                "stage_specs": [
                    {
                        "id": "video_generation",
                        "owner_refs": [
                            "scripts/presentations/inspect_rendered_slides_llm.py"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": [
                    "presentation.inspect_rendered_slides",
                    "presentation.inspect_rendered_slides_llm",
                ],
                "commands": [
                    ["python", "scripts/presentations/inspect_rendered_slides.py"]
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    blockers = report["blockers"]
    assert any(
        blocker["code"] == "stage_script_registry_required_script_missing_from_receipt"
        and blocker["missing_script_ids"] == ["presentation.inspect_rendered_slides_llm"]
        for blocker in blockers
    )


def test_run_stage_script_registry_execution_evidence_blocks_prose_only_mention(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "notes": (
                    "See scripts/presentations/inspect_rendered_slides.py and "
                    "scripts/presentations/inspect_rendered_slides_llm.py for details."
                ),
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(
        tmp_path, run_id="run-1", execution_evidence=True
    )

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "stage_script_registry_required_script_missing_from_receipt" in codes
    blocker = next(
        item
        for item in report["blockers"]
        if item["code"] == "stage_script_registry_required_script_missing_from_receipt"
    )
    assert set(blocker["missing_script_ids"]) == {
        "presentation.inspect_rendered_slides",
        "presentation.inspect_rendered_slides_llm",
    }


def test_run_stage_script_registry_execution_evidence_passes_with_real_command(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "commands": [
                    ["python", "scripts/presentations/inspect_rendered_slides.py"]
                ],
                "executed_owner_refs": [
                    "scripts/presentations/inspect_rendered_slides_llm.py"
                ],
                "notes": "unrelated prose that must not be needed for the pass",
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(
        tmp_path, run_id="run-1", execution_evidence=True
    )

    assert report["status"] == "pass"
    assert report["blockers"] == []


def test_run_stage_script_registry_resolves_workspace_specific_owner_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    required_id = "acquisition.source_discovery.cvpr2026"
    (run_dir / "stage-02-source-discovery.json").write_text(
        json.dumps(
            {
                "stage_id": "source-discovery",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": [required_id],
                "owner_execution_evidence": {
                    required_id: {
                        "owner_function": "run_cvpr_source_discovery_stage",
                        "inventory_path": "storage/source-discovery/source_discovery_inventory.json",
                    }
                },
                "stage_contract": {
                    "status": "pass",
                    "registry_stage_id": "source-discovery.cvpr2026",
                    "required_script_ids": [required_id],
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(workspace, run_id="run-1", strict=True)

    assert report["status"] == "pass", report
    assert report["blockers"] == []


def test_run_stage_script_registry_rejects_generic_owner_for_cvpr_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "conferences" / "cvpr2026"
    run_dir = workspace / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    generic_id = "acquisition.source_discovery"
    (run_dir / "stage-02-source-discovery.json").write_text(
        json.dumps(
            {
                "stage_id": "source-discovery",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": [generic_id],
                "owner_execution_evidence": {
                    generic_id: {"owner_function": "generic_source_discovery"}
                },
                "stage_contract": {
                    "status": "pass",
                    "registry_stage_id": "source-discovery.cvpr2026",
                    "required_script_ids": [generic_id],
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(workspace, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_required_script_missing_from_receipt"
        and blocker["missing_script_ids"] == ["acquisition.source_discovery.cvpr2026"]
        for blocker in report["blockers"]
    )


def test_run_stage_script_registry_strict_mode_rejects_static_manifest_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    decks_dir = tmp_path / "decks"
    decks_dir.mkdir()
    (decks_dir / "presentation_manifest.json").write_text(
        json.dumps(
            {
                "stage_specs": [
                    {
                        "id": "visual_qc",
                        "owner_refs": [
                            "scripts/presentations/inspect_rendered_slides_llm.py"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": [
                    "presentation.inspect_rendered_slides",
                    "presentation.inspect_rendered_slides_llm",
                ],
                "commands": [
                    ["python", "scripts/presentations/inspect_rendered_slides.py"]
                ],
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_required_script_missing_from_receipt"
        and blocker["missing_script_ids"] == ["presentation.inspect_rendered_slides_llm"]
        for blocker in report["blockers"]
    )


def test_run_stage_script_registry_strict_mode_rejects_stage_contract_as_execution(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    required_ids = [
        "presentation.inspect_rendered_slides",
        "presentation.inspect_rendered_slides_llm",
    ]
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "script_registry_sha256": registry_sha256(),
                "required_script_ids": required_ids,
                "stage_contract": {
                    "status": "pass",
                    "required_script_ids": required_ids,
                    "accepted_evidence": [
                        "scripts/presentations/inspect_rendered_slides.py",
                        "scripts/presentations/inspect_rendered_slides_llm.py",
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_required_script_missing_from_receipt"
        and set(blocker["missing_script_ids"]) == set(required_ids)
        for blocker in report["blockers"]
    )


def test_run_stage_script_registry_non_execution_evidence_mode_accepts_prose_mention(
    tmp_path: Path,
) -> None:
    """Default (execution_evidence=False) behavior is unchanged: mention-based."""
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "notes": (
                    "See scripts/presentations/inspect_rendered_slides.py and "
                    "scripts/presentations/inspect_rendered_slides_llm.py for details."
                ),
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1")

    assert report["status"] == "pass"
    assert report["blockers"] == []


def test_run_stage_script_registry_strict_mode_requires_execution_evidence(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "storage" / "workflow-runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "stage-13-visual_qc.json").write_text(
        json.dumps(
            {
                "stage_id": "visual_qc",
                "status": "succeeded",
                "notes": (
                    "See scripts/presentations/inspect_rendered_slides.py and "
                    "scripts/presentations/inspect_rendered_slides_llm.py for details."
                ),
            }
        ),
        encoding="utf-8",
    )

    report = audit_run_stage_script_usage(tmp_path, run_id="run-1", strict=True)

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "stage_script_registry_required_script_missing_from_receipt"
        for blocker in report["blockers"]
    )


def test_verify_stage_script_usage_cli_checks_skills() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_stage_script_usage.py",
            "--skills-root",
            ".claude/skills",
            "--docs-root",
            "docs",
            "--strict",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["diagnostic"]["status"] == "pass"
    assert payload["diagnostic"]["next_actions"]
    assert payload["reports"]["registry"]["diagnostic"]["accepted_evidence"]


def _lineage_test_registry(tmp_path, **contract_overrides):
    import json as _json

    payload = {
        "schema_version": 2,
        "registry_name": "test",
        "planning_required_task_ids": [],
        "scripts": {},
        "stage_bindings": {
            "upstream": {
                "approved_script_ids": [],
                "optional_script_ids": [],
                "required_script_ids": [],
                "forbidden_script_ids": [],
                "receipt_filename": "stage-01-upstream.json",
            },
            "stage": {
                "approved_script_ids": [],
                "optional_script_ids": [],
                "required_script_ids": [],
                "forbidden_script_ids": [],
                "runtime_contract": {
                    "required_receipts": [{"path": "storage/workflow-runs/<run-id>/stage-01-upstream.json"}],
                    **contract_overrides,
                },
            },
        },
        "task_bindings": {},
        "skill_contracts": {},
        "evidence_rules": {},
        "tasks": {},
    }
    path = tmp_path / "registry.json"
    path.write_text(_json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, payload


def test_registry_rejects_fresh_after_in_required_receipts(tmp_path):
    import json as _json

    import pytest as _pytest

    from conference_pipeline.stage_script_registry import (
        StageScriptRegistryError,
        load_stage_script_registry,
    )

    path, payload = _lineage_test_registry(tmp_path)
    assert load_stage_script_registry(path)["stage_bindings"]["upstream"]["receipt_filename"]
    payload["stage_bindings"]["stage"]["runtime_contract"]["required_receipts"][0]["fresh_after"] = "x.json"
    path.write_text(_json.dumps(payload) + "\n", encoding="utf-8")
    with _pytest.raises(StageScriptRegistryError, match="fresh_after"):
        load_stage_script_registry(path)


def test_registry_validates_lineage_vocabulary(tmp_path):
    import json as _json

    import pytest as _pytest

    from conference_pipeline.stage_script_registry import (
        StageScriptRegistryError,
        load_stage_script_registry,
    )

    path, payload = _lineage_test_registry(
        tmp_path,
        required_upstream_stages=["upstream"],
        lineage={"input_artifacts": [{"name": "m", "path": "analysis/m.json"}]},
        output_artifact_checks=[["index", "index_path", "index_sha256"]],
    )
    assert load_stage_script_registry(path)

    bad_cases = {
        "unknown upstream": {"required_upstream_stages": ["ghost"]},
        "bad lineage mode": {"required_receipts": [{"path": "p.json", "lineage": "mtime"}]},
        "input artifact without path": {"lineage": {"input_artifacts": [{"name": "m"}]}},
        "bad table row": {"output_artifact_checks": [["only-name"]]},
    }
    for label, override in bad_cases.items():
        broken = _json.loads(_json.dumps(payload))
        broken["stage_bindings"]["stage"]["runtime_contract"].update(override)
        path.write_text(_json.dumps(broken) + "\n", encoding="utf-8")
        with _pytest.raises(StageScriptRegistryError):
            load_stage_script_registry(path)
        del label

    duplicate = _json.loads(_json.dumps(payload))
    duplicate["stage_bindings"]["stage"]["receipt_filename"] = "stage-01-upstream.json"
    path.write_text(_json.dumps(duplicate) + "\n", encoding="utf-8")
    with _pytest.raises(StageScriptRegistryError, match="already used"):
        load_stage_script_registry(path)


def test_registry_receipt_filenames_match_runner_stage_tables():
    from pathlib import Path as _Path

    from conference_pipeline.receipt_lineage import receipt_filenames_from_registry
    from conference_pipeline.stage_script_registry import REPO_ROOT, load_stage_script_registry

    names = receipt_filenames_from_registry(load_stage_script_registry())
    assert len(names) >= 13
    for runner in ("run_cvpr2026_production.py", "run_iclr2026_production.py", "run_icml2026_production.py"):
        source = (_Path(REPO_ROOT) / "scripts" / runner).read_text(encoding="utf-8")
        for stage_id, filename in names.items():
            if stage_id == "notebooklm-podcast" and runner != "run_cvpr2026_production.py":
                continue  # CVPR-only stage
            if "stage_table.from_registry(" not in source:
                # A literal stage table must spell every registry receipt filename;
                # a registry-derived one gets them from the bindings (fail-closed).
                assert f'"{filename}"' in source, f"{runner} does not write {filename} for {stage_id}"
            assert f'"{stage_id}"' in source, f"{runner} does not name stage {stage_id}"
