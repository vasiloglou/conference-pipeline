from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline.model_governance import default_model, default_provider
from conference_pipeline.presentation.blueprint_generation import (
    BlueprintGenerationConfig,
    _build_prompt,
    _input_records,
    _receipt_input_record,
    generate_presentation_blueprint,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conference"
    deck_dir = workspace / "decks" / "01_demo_topic"
    _write_text(
        workspace / "analysis" / "01_demo_topic.md",
        "# Demo analysis\n\nUNIQUE_ANALYSIS_CLAIM safety telescoping plan\n",
    )
    _write_text(
        deck_dir / "source_pack_ranked_papers.json",
        '{"papers": [{"title": "UNIQUE_SOURCE_PACK_PAPER"}]}\n',
    )
    _write_text(
        deck_dir / "source_attributions.json",
        '{"sources": [{"label": "UNIQUE_SOURCE_METADATA_LABEL"}]}\n',
    )
    return workspace, deck_dir


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase15_inputs(workspace: Path, deck_dir: Path) -> dict[str, Path]:
    inventory = deck_dir / "source_corpus_inventory.json"
    ledger = deck_dir / "speaker_author_work_ledger.json"
    repair_findings = deck_dir / "source_repair_findings.json"
    cluster_plan = deck_dir / "research_cluster_plan.json"
    provenance = workspace / "analysis" / "provenance" / "04_safety_alignment_privacy.json"
    _write_json(
        inventory,
        {
            "schema_version": 1,
            "topic_id": "04_safety_alignment_privacy",
            "inputs": [
                {
                    "kind": "analysis_report",
                    "path": "analysis/04_safety_alignment_privacy.md",
                    "sha256": "analysis-hash",
                    "production_evidence": True,
                }
            ],
            "source_records": [],
        },
    )
    _write_json(
        ledger,
        {
            "schema_version": 1,
            "topic_id": "04_safety_alignment_privacy",
            "entries": [
                {
                    "source_id": "src-agent",
                    "resolved_title": "Agents in the Wild",
                    "speakers": ["Don Song"],
                    "authors": [],
                    "works": ["Opening Remarks"],
                    "contribution_note": "Agentic systems make memory, tools, and environments part of safety.",
                    "status": "resolved",
                    "confidence": 0.95,
                    "visible_source_chip_eligible": True,
                    "source_family": "artifacts",
                    "evidence_paths": ["analysis/topic_reference_ledgers/04_safety_alignment_privacy.json"],
                },
                {
                    "source_id": "src-privacy",
                    "resolved_title": "Fine-Grained Privacy Extraction from RAG",
                    "speakers": [],
                    "authors": ["Ada Example"],
                    "works": ["RAG privacy"],
                    "contribution_note": "RAG privacy turns retrieval context into a first-class leakage boundary.",
                    "status": "resolved",
                    "confidence": 0.9,
                    "visible_source_chip_eligible": True,
                    "source_family": "papers",
                    "evidence_paths": ["analysis/topic_reference_ledgers/04_safety_alignment_privacy.json"],
                },
            ],
        },
    )
    _write_json(
        repair_findings,
        {
            "schema_version": 1,
            "topic_id": "04_safety_alignment_privacy",
            "inventory_sha256": _sha256(inventory),
            "ledger_sha256": _sha256(ledger),
            "findings": [],
        },
    )
    _write_json(
        cluster_plan,
        {
            "schema_version": 1,
            "topic_id": "04_safety_alignment_privacy",
            "input_hashes": {
                "source_corpus_inventory.json": _sha256(inventory),
                "speaker_author_work_ledger.json": _sha256(ledger),
                "source_repair_findings.json": _sha256(repair_findings),
            },
            "clusters": [
                {
                    "cluster_id": "agent-system-safety",
                    "title": "Agent/system safety",
                    "story_contribution": "Safety shifts from refusal to deployed agent systems.",
                    "primary_sources": [{"source_id": "src-agent"}],
                    "supporting_sources": [{"source_id": "src-privacy"}],
                },
                {
                    "cluster_id": "privacy-unlearning",
                    "title": "Privacy and unlearning",
                    "story_contribution": "Privacy becomes part of the alignment operating boundary.",
                    "primary_sources": [{"source_id": "src-privacy"}],
                    "supporting_sources": [{"source_id": "src-agent"}],
                },
            ],
        },
    )
    _write_json(provenance, {"topic_id": "04_safety_alignment_privacy", "sha256": "provenance-hash"})
    return {
        "source_corpus_inventory": inventory,
        "speaker_author_work_ledger": ledger,
        "source_repair_findings": repair_findings,
        "research_cluster_plan": cluster_plan,
        "provenance": provenance,
    }


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/presentations/generate_presentation_blueprint.py", *args],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_registry_has_blueprint_generation_task() -> None:
    assert default_provider("presentation.blueprint_generation") == "anthropic"
    assert default_model("presentation.blueprint_generation")


def test_mock_generation_writes_blueprint_and_nonproduction_receipt(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    mock_fixture = workspace / "fixtures" / "valid_blueprint.json"
    _write_text(mock_fixture, (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    result = _run(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--analysis-md",
            str(workspace / "analysis" / "01_demo_topic.md"),
            "--output-json",
            str(deck_dir / "presentation_blueprint.json"),
            "--output-md",
            str(deck_dir / "presentation_blueprint.md"),
            "--receipt-path",
            str(deck_dir / "presentation_blueprint_receipt.json"),
            "--mode",
            "mock",
            "--mock-fixture",
            str(mock_fixture),
        ]
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads((deck_dir / "presentation_blueprint_receipt.json").read_text(encoding="utf-8"))
    assert (deck_dir / "presentation_blueprint.json").is_file()
    assert (deck_dir / "presentation_blueprint.md").is_file()
    assert receipt["mode"] == "mock"
    assert receipt["production_evidence"] is False
    assert receipt["model_governance_task_id"] == "presentation.blueprint_generation"
    assert receipt["provider"] == "anthropic"
    assert receipt["model_id"] == default_model("presentation.blueprint_generation")
    assert receipt["provider_receipt_paths"] == []
    assert receipt["estimated_cost_usd"] == 0.0
    assert receipt["budget_guardrail"]["provider_cost_approved"] is False
    assert receipt["prompt_sha256"]
    assert receipt["input_sha256"]
    assert "mock_fixture" in receipt


def test_live_generation_requires_explicit_provider_cost_approval(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    result = _run(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--analysis-md",
            str(workspace / "analysis" / "01_demo_topic.md"),
            "--output-json",
            str(deck_dir / "presentation_blueprint.json"),
            "--output-md",
            str(deck_dir / "presentation_blueprint.md"),
            "--receipt-path",
            str(deck_dir / "presentation_blueprint_receipt.json"),
            "--mode",
            "live",
        ]
    )

    assert result.returncode == 2
    assert "BLUEPRINT_PROVIDER_COST_APPROVAL_REQUIRED" in result.stderr


def test_live_blueprint_prompt_includes_source_contents_not_only_hashes(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    matched_papers = deck_dir / "matched_papers.json"
    _write_text(matched_papers, '{"papers": [{"title": "UNIQUE_MATCHED_PAPER"}]}\n')
    config = BlueprintGenerationConfig(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        analysis_md=workspace / "analysis" / "01_demo_topic.md",
        source_pack=deck_dir / "source_pack_ranked_papers.json",
        matched_papers=matched_papers,
        source_metadata=deck_dir / "source_attributions.json",
    )

    records = _input_records(config, workspace)
    prompt = _build_prompt(config, records)

    assert "UNIQUE_ANALYSIS_CLAIM" in prompt
    assert "UNIQUE_SOURCE_PACK_PAPER" in prompt
    assert "UNIQUE_MATCHED_PAPER" in prompt
    assert "UNIQUE_SOURCE_METADATA_LABEL" in prompt
    assert "## Source: analysis" in prompt
    assert "sha256:" in prompt


def test_source_ledger_prompt_includes_phase15_inputs_and_report_url(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    inputs = _phase15_inputs(workspace, deck_dir)
    config = BlueprintGenerationConfig(
        workspace=workspace,
        presentation_id="04_safety_alignment_privacy",
        presentation_type="topic-deep-dive",
        topic="04_safety_alignment_privacy",
        analysis_md=workspace / "analysis" / "01_demo_topic.md",
        source_corpus_inventory=inputs["source_corpus_inventory"],
        speaker_author_work_ledger=inputs["speaker_author_work_ledger"],
        source_repair_findings=inputs["source_repair_findings"],
        research_cluster_plan=inputs["research_cluster_plan"],
        final_report_url="https://drive.google.com/final-topic-4-report",
        provenance_inputs=(inputs["provenance"],),
    )

    records = _input_records(config, workspace)
    prompt = _build_prompt(config, records)

    assert "source_corpus_inventory" in prompt
    assert "speaker_author_work_ledger" in prompt
    assert "source_repair_findings" in prompt
    assert "research_cluster_plan" in prompt
    assert "final_report_url: https://drive.google.com/final-topic-4-report" in prompt
    assert "Agents in the Wild" in prompt


def test_source_ledger_generation_writes_hash_bound_blueprint(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    inputs = _phase15_inputs(workspace, deck_dir)
    output_json = deck_dir / "presentation_blueprint.json"
    receipt_path = deck_dir / "presentation_blueprint_receipt.json"
    config = BlueprintGenerationConfig(
        workspace=workspace,
        presentation_id="04_safety_alignment_privacy",
        presentation_type="topic-deep-dive",
        topic="04_safety_alignment_privacy",
        analysis_md=workspace / "analysis" / "01_demo_topic.md",
        source_corpus_inventory=inputs["source_corpus_inventory"],
        speaker_author_work_ledger=inputs["speaker_author_work_ledger"],
        source_repair_findings=inputs["source_repair_findings"],
        research_cluster_plan=inputs["research_cluster_plan"],
        final_report_url="https://drive.google.com/final-topic-4-report",
        provenance_inputs=(inputs["provenance"],),
        output_json=output_json,
        receipt_path=receipt_path,
        mode="source-ledger",
    )

    summary = generate_presentation_blueprint(config)

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "source-ledger"
    assert payload["final_report_url"] == "https://drive.google.com/final-topic-4-report"
    assert 11 <= payload["target_slide_count"] <= 14
    assert payload["duration_budget"]["target_minutes"] <= 38
    assert payload["cluster_coverage_justification"]["required"] is False
    assert payload["input_hashes"]["source_corpus_inventory.json"] == _sha256(inputs["source_corpus_inventory"])
    assert payload["input_hashes"]["speaker_author_work_ledger.json"] == _sha256(inputs["speaker_author_work_ledger"])
    assert payload["input_hashes"]["source_repair_findings.json"] == _sha256(inputs["source_repair_findings"])
    assert payload["input_hashes"]["research_cluster_plan.json"] == _sha256(inputs["research_cluster_plan"])
    assert payload["source_paper_plan"]
    assert payload["speaker_author_work_mention_plan"]
    assert all(slide["narration_beats"] for slide in payload["slides"])
    assert receipt["input_hashes"] == payload["input_hashes"]
    assert {record["kind"] for record in receipt["inputs"]} >= {
        "source_corpus_inventory",
        "speaker_author_work_ledger",
        "source_repair_findings",
        "research_cluster_plan",
        "provenance_input",
    }


def test_blueprint_generation_receipt_strips_prompt_source_content(tmp_path: Path) -> None:
    workspace, _deck_dir = _workspace(tmp_path)
    config = BlueprintGenerationConfig(
        workspace=workspace,
        presentation_id="01_demo_topic",
        presentation_type="topic-deep-dive",
        analysis_md=workspace / "analysis" / "01_demo_topic.md",
    )
    record = _input_records(config, workspace)[0]

    receipt_record = _receipt_input_record(record)

    assert "content" not in receipt_record
    assert receipt_record["content_included_chars"] > 0
    assert receipt_record["content_truncated"] is False


def test_dry_run_does_not_require_provider_imports_or_write_files(tmp_path: Path) -> None:
    workspace, deck_dir = _workspace(tmp_path)
    result = _run(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--output-json",
            str(deck_dir / "presentation_blueprint.json"),
            "--output-md",
            str(deck_dir / "presentation_blueprint.md"),
            "--receipt-path",
            str(deck_dir / "presentation_blueprint_receipt.json"),
            "--dry-run",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["mode"] == "dry-run"
    assert payload["provider_requirements"] == ["task:presentation.blueprint_generation"]
    assert not (deck_dir / "presentation_blueprint.json").exists()


def test_generation_rejects_legacy_presentations_lane(tmp_path: Path) -> None:
    workspace, _deck_dir = _workspace(tmp_path)
    mock_fixture = workspace / "fixtures" / "valid_blueprint.json"
    _write_text(mock_fixture, (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    result = _run(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--output-json",
            str(workspace / "presentations" / "01_demo_topic" / "presentation_blueprint.json"),
            "--mode",
            "mock",
            "--mock-fixture",
            str(mock_fixture),
        ]
    )

    assert result.returncode == 2
    assert "presentations/ is a legacy" in result.stderr
