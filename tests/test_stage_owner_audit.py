from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conference_pipeline.stage_owner_audit import (
    StageOwnerRecord,
    audit_plan_owner_registry,
    audit_stage_owner_inventory,
    load_stage_owner_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_stage_owner_audit_accepts_reusable_output_owner():
    report = audit_stage_owner_inventory(
        [
            StageOwnerRecord(
                "presentation",
                "reusable_owner",
                ("scripts/presentations/generate_slides_agent.py",),
                production_output=True,
            )
        ],
        required_owner_refs={
            "presentation": ("scripts/presentations/generate_slides_agent.py",)
        },
    )

    assert report["status"] == "pass", report
    assert report["blockers"] == []


def test_stage_owner_audit_rejects_legacy_scaffold_production_output():
    report = audit_stage_owner_inventory(
        [
            StageOwnerRecord(
                "presentation",
                "legacy_scaffold",
                ("write_deck_package",),
                production_output=True,
            )
        ]
    )

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "production_output_uses_legacy_scaffold" in codes


def test_stage_owner_audit_reports_missing_required_owner_ref():
    report = audit_stage_owner_inventory(
        [
            StageOwnerRecord(
                "podcast",
                "reusable_owner",
                ("conference_pipeline.iclr2025_live.write_podcast_package",),
                production_output=True,
            )
        ],
        required_owner_refs={"podcast": ("scripts/generate_focused_podcast.py",)},
    )

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "stage_owner_required_ref_missing" in codes


def test_stage_owner_audit_loads_default_registry():
    registry = load_stage_owner_registry()

    assert "podcast.script_generation" in registry["tasks"]
    assert "presentation.draft_generation" in registry["tasks"]
    assert "presentation.svg_figure_generation" in registry["tasks"]
    assert "presentation.visual_qc" in registry["tasks"]


def test_stage_owner_inventory_can_use_registry_task_ids():
    report = audit_stage_owner_inventory(
        [
            StageOwnerRecord(
                "podcast",
                "reusable_owner",
                ("scripts/generate_focused_podcast.py",),
                production_output=True,
            )
        ],
        registry_task_ids_by_stage={"podcast": ("podcast.script_generation",)},
    )

    assert report["status"] == "pass", report


def test_plan_owner_registry_accepts_required_planning_owners(tmp_path: Path):
    registry = load_stage_owner_registry()
    terms: list[str] = []
    for task_id in registry["planning_required_task_ids"]:
        task = registry["tasks"][task_id]
        terms.extend(task["approved_owner_refs"])
        terms.extend(task["required_command_terms"])
    (tmp_path / "ROADMAP.md").write_text("\n".join(sorted(set(terms))), encoding="utf-8")

    report = audit_plan_owner_registry(tmp_path)

    assert report["status"] == "pass", report


def test_plan_owner_registry_blocks_missing_approved_owner(tmp_path: Path):
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/generate_focused_podcast.py\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "plan_owner_registry_approved_owner_missing" in codes


def test_plan_owner_registry_uses_custom_registry_diagnostic_profile(tmp_path: Path):
    registry = json.loads(
        (REPO_ROOT / "src" / "conference_pipeline" / "stage_owner_registry.json").read_text(
            encoding="utf-8"
        )
    )
    registry["scripts"]["verification.stage_owner_audit"]["diagnostic_profile"][
        "blocked_next_actions"
    ] = ["Use the custom registry remediation path."]
    plan_root = tmp_path / "plan"
    registry_path = tmp_path / "custom-registry" / "registry.json"
    registry_path.parent.mkdir()
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    plan_root.mkdir()
    (plan_root / "ROADMAP.md").write_text("", encoding="utf-8")

    report = audit_plan_owner_registry(
        plan_root,
        task_ids=("presentation.draft_generation",),
        registry_path=registry_path,
    )

    assert report["status"] == "blocked"
    assert "Use the custom registry remediation path." in report["diagnostic"]["next_actions"]


def test_plan_owner_registry_allows_rejected_owner_policy_list(tmp_path: Path):
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/presentations/generate_deck_contract.py\n"
        "--workspace\n--presentation-id\n--presentation-type\ntopic-deep-dive\n"
        "--topic\n--presentation-blueprint\ndecks/<id>/presentation_blueprint.json\n"
        "--blueprint-receipt\ndecks/<id>/presentation_blueprint_receipt.json\n"
        "--blueprint-review\ndecks/<id>/blueprint_review.json\n"
        "--output\ndecks/<id>/presentation_deck.json\n--receipt-output\n"
        "decks/<id>/presentation_deck_receipt.json\n\n"
        "Reject these as completion owners:\n"
        "- Reject `scripts/presentations/generate_slides_simple.py`\n"
        "- Reject `scripts/generate_slides_agentic.py`\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.deck_contract_generation",),
    )

    assert report["status"] == "pass", report


def test_plan_owner_registry_blocks_rejected_owner_live_command(tmp_path: Path):
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/presentations/generate_slides_agent.py\n"
        "--workspace\n--presentation-id\n--presentation-type\ntopic-deep-dive\n"
        "--topic\n--display-title\n--analysis-md\nanalysis/slide_briefs/<id>.md\n"
        "--output-dir\ndecks/<id>\n--output-slides-md\n"
        "decks/<id>/1_draft_slides.md\n"
        "**Live command:** `.venv/bin/python scripts/generate_slides_agentic.py --topic 02`\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    codes = {blocker["code"] for blocker in report["blockers"]}
    assert "plan_owner_registry_rejected_owner_used" in codes


def test_plan_owner_registry_blocks_rejected_owner_after_nearby_rejection(tmp_path: Path):
    registry = load_stage_owner_registry()
    task = registry["tasks"]["presentation.draft_generation"]
    (tmp_path / "ROADMAP.md").write_text(
        "\n".join(task["required_command_terms"])
        + "\nReject scripts/generate_slides_agentic.py as a completion owner.\n"
        + "python scripts/generate_slides_agentic.py --topic 02\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "plan_owner_registry_rejected_owner_used"
        for blocker in report["blockers"]
    )


def test_plan_owner_registry_scans_active_commands_in_historical_docs(tmp_path: Path):
    registry = load_stage_owner_registry()
    task = registry["tasks"]["presentation.draft_generation"]
    (tmp_path / "ROADMAP.md").write_text(
        "# Historical deprecated owner migration notes\n"
        + "\n".join(task["required_command_terms"])
        + "\npython scripts/generate_slides_agentic.py --topic 02\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "plan_owner_registry_rejected_owner_used"
        for blocker in report["blockers"]
    )


def test_plan_owner_registry_blocks_forbidden_command_in_policy_list(tmp_path: Path):
    registry = load_stage_owner_registry()
    task = registry["tasks"]["presentation.draft_generation"]
    (tmp_path / "ROADMAP.md").write_text(
        "\n".join(task["required_command_terms"])
        + "\nDeprecated completion owners:\n"
        + "- python scripts/generate_slides_agentic.py --topic 02\n",
        encoding="utf-8",
    )

    report = audit_plan_owner_registry(
        tmp_path,
        task_ids=("presentation.draft_generation",),
    )

    assert report["status"] == "blocked"
    assert any(
        blocker["code"] == "plan_owner_registry_rejected_owner_used"
        for blocker in report["blockers"]
    )


def test_stage_owner_audit_cli_writes_json_report(tmp_path: Path):
    (tmp_path / "ROADMAP.md").write_text(
        "scripts/generate_focused_podcast.py\nanalysis/<id>.md\n--fact-ranks\n"
        "audio/<id>/ranks.json\n--output-dir\naudio/<id>/focused_podcast\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "owner-report.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "conference_pipeline.stage_owner_audit",
            "--plan-root",
            str(tmp_path),
            "--task-id",
            "podcast.script_generation",
            "--json-output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
