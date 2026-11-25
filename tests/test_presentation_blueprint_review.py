from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline.presentation.blueprint import (
    BLUEPRINT_HASH_MISMATCH,
    validate_blueprint_review_file,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def _phase15_ready_payload(fixture_name: str = "valid_blueprint.json") -> dict[str, object]:
    payload = json.loads((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"))
    payload.update(
        {
            "final_report_url": "https://drive.google.com/final-topic-4-report",
            "input_hashes": {
                "source_corpus_inventory.json": "a" * 64,
                "speaker_author_work_ledger.json": "b" * 64,
                "source_repair_findings.json": "c" * 64,
                "research_cluster_plan.json": "d" * 64,
            },
            "duration_budget": {"target_minutes": 37, "hard_limit_minutes": 40},
            "cluster_coverage_justification": {"required": False},
            "source_paper_plan": [{"source_id": "SRC-001", "cluster_id": "cluster-001"}],
            "speaker_author_work_mention_plan": [
                {
                    "slide_id": "slide-001",
                    "speakers": ["Don Song"],
                    "authors": [],
                    "works": ["Agents in the Wild"],
                    "contribution_note": "Agent safety needs deployment-aware evidence.",
                }
            ],
        }
    )
    for slide in payload["slides"]:
        slide["narration_beats"] = [
            "Name the source-backed claim, explain the visual role, and connect it to the contribution."
        ]
    return payload


def _copy_blueprint(
    tmp_path: Path,
    fixture_name: str = "valid_blueprint.json",
    *,
    phase15_ready: bool = False,
) -> Path:
    blueprint = tmp_path / "deck" / "presentation_blueprint.json"
    blueprint.parent.mkdir(parents=True, exist_ok=True)
    if phase15_ready:
        blueprint.write_text(
            json.dumps(_phase15_ready_payload(fixture_name), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        blueprint.write_text((FIXTURE_ROOT / fixture_name).read_text(encoding="utf-8"), encoding="utf-8")
    return blueprint


def _write_receipt(blueprint: Path) -> Path:
    receipt = blueprint.parent / "presentation_blueprint_receipt.json"
    receipt.write_text(
        json.dumps({"blueprint_sha256": hashlib.sha256(blueprint.read_bytes()).hexdigest()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return receipt


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/presentations/review_presentation_blueprint.py", *args],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )


def test_review_writes_approval_token_only_for_passing_blueprint(tmp_path: Path) -> None:
    blueprint = _copy_blueprint(tmp_path, phase15_ready=True)
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"
    output_md = blueprint.parent / "blueprint_review.md"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--llm-judge-approved",
            "--llm-judge-artifact",
            "llm_judge_review.json",
            "--approve-for-draft-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "decks/01_demo_topic/blueprint_review.md",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(blueprint.read_bytes()).hexdigest()
    assert payload["status"] == "pass"
    assert payload["approval"]["status"] == "approved_for_draft_generation"
    assert payload["blueprint_sha256"] == expected_hash
    assert payload["approved_by"] == "operator"
    assert payload["approved_at"]
    assert payload["review_artifact"] == "decks/01_demo_topic/blueprint_review.md"
    assert set(payload["gate_types"]) == {"deterministic", "LLM-judge", "human"}
    assert payload["gate_types"] == {"LLM-judge": "pass", "deterministic": "pass", "human": "pass"}
    assert output_md.read_text(encoding="utf-8").startswith("# Presentation Blueprint Review")


def test_review_can_approve_for_deck_contract_generation_alias(tmp_path: Path) -> None:
    blueprint = _copy_blueprint(tmp_path, phase15_ready=True)
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--llm-judge-approved",
            "--approve-for-deck-contract-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "decks/04_safety_alignment_privacy/blueprint_review.md",
        ]
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["approval"]["status"] == "approved_for_deck_contract_generation"


def test_review_fails_invalid_blueprint_without_approval(tmp_path: Path) -> None:
    blueprint = _copy_blueprint(tmp_path, "invalid_missing_source_binding.json")
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--llm-judge-approved",
            "--approve-for-draft-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "blueprint_review.md",
        ]
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert "approval" not in payload
    assert any(finding["code"] == "BLUEPRINT_SOURCE_BINDING_MISSING" for finding in payload["findings"])


def test_review_cannot_approve_while_llm_judge_is_pending(tmp_path: Path) -> None:
    blueprint = _copy_blueprint(tmp_path, phase15_ready=True)
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--approve-for-draft-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "blueprint_review.md",
        ]
    )

    assert result.returncode == 1
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "fail"
    assert "approval" not in payload
    assert payload["gate_types"]["LLM-judge"] == "fail"


def test_review_without_human_approval_stays_nonblocking_but_unapproved(tmp_path: Path) -> None:
    blueprint = _copy_blueprint(tmp_path, phase15_ready=True)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(["--blueprint", str(blueprint), "--output-json", str(output_json)])

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert "approval" not in payload
    assert payload["gate_types"]["LLM-judge"] == "pending"
    assert payload["gate_types"]["human"] == "pending"
    assert any(finding["gate_type"] == "human" for finding in payload["findings"])


def test_review_fails_missing_source_report_contribution_visual_and_narration_gates(
    tmp_path: Path,
) -> None:
    payload = _phase15_ready_payload()
    payload.pop("final_report_url")
    payload.pop("source_paper_plan")
    payload.pop("speaker_author_work_mention_plan")
    payload["conclusion_synthesis"] = "Weak."
    payload["slides"][0].pop("visual_role")
    payload["slides"][0].pop("narration_beats")
    blueprint = tmp_path / "deck" / "presentation_blueprint.json"
    blueprint.parent.mkdir(parents=True, exist_ok=True)
    blueprint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--llm-judge-approved",
            "--approve-for-deck-contract-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "blueprint_review.md",
        ]
    )

    assert result.returncode == 1
    review = json.loads(output_json.read_text(encoding="utf-8"))
    messages = "\n".join(finding["message"] for finding in review["findings"])
    assert "missing source coverage" in messages
    assert "final report" in messages
    assert "speaker/author/work" in messages
    assert "weak conclusion" in messages
    assert "visual roles" in messages
    assert "narration beats" in messages
    assert "approval" not in review


def test_review_rejects_over_budget_blueprint_and_stale_approval(tmp_path: Path) -> None:
    payload = _phase15_ready_payload()
    payload["target_runtime_minutes"] = 41
    payload["duration_budget"]["target_minutes"] = 41
    blueprint = tmp_path / "deck" / "presentation_blueprint.json"
    blueprint.parent.mkdir(parents=True, exist_ok=True)
    blueprint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = _write_receipt(blueprint)
    output_json = blueprint.parent / "blueprint_review.json"

    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--llm-judge-approved",
            "--approve-for-deck-contract-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "blueprint_review.md",
        ]
    )

    assert result.returncode == 1
    over_budget = json.loads(output_json.read_text(encoding="utf-8"))
    assert "approval" not in over_budget
    assert any("40 minutes" in finding["message"] for finding in over_budget["findings"])

    payload["target_runtime_minutes"] = 37
    payload["duration_budget"]["target_minutes"] = 37
    blueprint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = _write_receipt(blueprint)
    result = _run(
        [
            "--blueprint",
            str(blueprint),
            "--blueprint-receipt",
            str(receipt),
            "--output-json",
            str(output_json),
            "--llm-judge-approved",
            "--approve-for-deck-contract-generation",
            "--approved-by",
            "operator",
            "--review-artifact",
            "blueprint_review.md",
        ]
    )
    assert result.returncode == 0, result.stderr

    payload["deck_thesis"] = "Changed after review to prove stale approval cannot pass."
    blueprint.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    stale = validate_blueprint_review_file(blueprint, output_json)
    assert BLUEPRINT_HASH_MISMATCH in {error.code for error in stale.errors}
