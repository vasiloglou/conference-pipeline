from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any
import re

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
REPO_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
MIGRATION_NOTE = "Use decks/<presentation>/... for reusable presentation outputs; presentations/ is a legacy NeurIPS lane."
BLUEPRINT_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase24_{script_path.stem}",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_repo_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = REPO_SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase24_{script_path.stem}",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_repair_figure_detail_narration_uses_asset_inventory_and_updates_json(tmp_path: Path):
    module = load_presentation_script("repair_figure_detail_narration.py")
    lint_module = load_presentation_script(
        "lint_narration_visual_freshness.py",
        module_name="phase24_lint_narration_visual_freshness",
    )
    deck_dir = tmp_path / "decks" / "07_demo"
    deck_dir.mkdir(parents=True)
    (deck_dir / "video_build_notebooklm").mkdir()
    inventory = {
        "figure_specs": [
            {
                "slide_number": 1,
                "slide_title": "Protein-Ligand Binding",
                "depiction_type": "molecule_pocket",
                "entities": ["Protein pocket", "Ligand", "Binding site", "Validation gate"],
                "relationships": [
                    "Ligand docks into Protein pocket",
                    "Validation gate rejects unstable binding poses",
                ],
                "quantities": ["47 compounds entered wet-lab validation"],
                "semantic_goal": "Show the lab-grounded binding workflow from candidate molecule to validation gate",
                "must_depict": ["protein pocket with ligand inside", "validation gate after docking"],
                "must_not_do": ["Do not copy slide bullets into decorative boxes"],
            }
        ]
    }
    (deck_dir / "asset_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    (deck_dir / "5_draft_slides_with_images.md").write_text("## Slide 1: Demo\n", encoding="utf-8")
    (tmp_path / "decks" / "presentation_manifest.json").write_text(
        json.dumps({"presentations": [{"id": "07_demo"}]}),
        encoding="utf-8",
    )
    (deck_dir / "6_presenter_script.md").write_text(
        """# Presenter Script

### [SLIDE 1] Protein-Ligand Binding
**Time:** ~1 min | **Section:** Demo

**SAY:**
> Main narration.

**FIGURE:** Four colored boxes labeled Protein, Ligand, Binding, and Chart.

**TRANSITION:** "Next."

---

### [ZOOM 1] Protein-Ligand Binding — Figure Detail
**Time:** ~20s | **Section:** Demo

**SAY:**
> Let's zoom in on this diagram. The four boxes labeled Protein, Ligand, Binding, and Chart represent the idea.

**FIGURE:** The abstract visualization uses colored ellipses.

---
""",
        encoding="utf-8",
    )
    (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
        """# Notebook Script

### [SLIDE 100001] Protein-Ligand Binding — Figure Detail
**Time:** ~20s | **Section:** Demo

**DIALOGUE:**
- **Iapetus:** Let's zoom in on this diagram. It is four boxes labeled Protein, Ligand, Binding, and Chart.
- **Erinome:** The abstract visualization uses colored ellipses.

---
""",
        encoding="utf-8",
    )
    (deck_dir / "video_build_notebooklm" / "script.json").write_text(
        json.dumps(
            {
                "hosts": {"host_a": "Iapetus", "host_b": "Erinome"},
                "segments": [
                    {
                        "index": 1,
                        "speaker": "Iapetus",
                        "text": "Let's zoom in on this diagram. It is four boxes labeled Protein, Ligand, Binding, and Chart.",
                        "slide_number": 100001,
                        "slide_title": "Protein-Ligand Binding — Figure Detail",
                    },
                    {
                        "index": 2,
                        "speaker": "Erinome",
                        "text": "The abstract visualization uses colored ellipses.",
                        "slide_number": 100001,
                        "slide_title": "Protein-Ligand Binding — Figure Detail",
                    },
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 100001,
                            "title": "Protein-Ligand Binding — Figure Detail",
                            "turns": [
                                {
                                    "speaker": "Iapetus",
                                    "text": "Let's zoom in on this diagram.",
                                },
                                {
                                    "speaker": "Erinome",
                                    "text": "The abstract visualization uses colored ellipses.",
                                },
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    failed_lint = lint_module.lint_workspace(tmp_path / "decks")
    assert failed_lint["status"] == "failed"
    assert {
        finding["rule"] for finding in failed_lint["findings"]
    } >= {
        "presenter_bad_figure_detail_language",
        "notebooklm_bad_figure_detail_language",
    }

    report = module.repair_deck(deck_dir)

    presenter_text = (deck_dir / "6_presenter_script.md").read_text(encoding="utf-8")
    notebook_text = (deck_dir / "6_presenter_script_notebooklm_2host.md").read_text(
        encoding="utf-8"
    )
    notebook_json = json.loads(
        (deck_dir / "video_build_notebooklm" / "script.json").read_text(encoding="utf-8")
    )
    combined = presenter_text + notebook_text + json.dumps(notebook_json)

    assert report["presenter_zoom_blocks_repaired"] == 1
    assert report["notebooklm_blocks_repaired"] == 1
    assert report["notebooklm_json_slides_repaired"] == 1
    assert "Protein pocket" in combined
    assert "Ligand docks into Protein pocket" in combined
    assert "47 compounds entered wet-lab validation" in combined
    assert "four boxes labeled" not in combined.lower()
    assert "colored ellipses" not in combined.lower()
    assert "let's zoom in on this diagram" not in combined.lower()

    passed_lint = lint_module.lint_workspace(tmp_path / "decks")
    assert passed_lint["status"] == "passed"
    assert passed_lint["metrics"]["presenter_bad_figure_blocks"] == 0
    assert passed_lint["metrics"]["notebooklm_bad_figure_blocks"] == 0


def _write_editorial_gate_deck(deck_dir: Path, *, passing: bool) -> None:
    deck_dir.mkdir(parents=True)
    if passing:
        slides = """## Slide 1: Protein-Ligand Binding

- A pocket-specific binding model explains ligand fit.

<!--
FIGURE_NEED: Mechanism schematic of ligand docking into a protein pocket
-->

![Protein pocket binding mechanism](assets/generated/slide_001.svg)

## Slide 2: Wet-Lab Validation

- Validation filters unstable binding poses before synthesis.

<!--
FIGURE_NEED: Experiment workflow linking docking to validation
-->

![Wet-lab validation workflow](assets/generated/slide_002.svg)
"""
        presenter = """# Presenter

### [SLIDE 1] Protein-Ligand Binding
**SAY:**
> The protein pocket narrows the binding problem: the ligand only matters if its pose fits the pocket geometry.

### [SLIDE 2] Wet-Lab Validation
**SAY:**
> The validation gate then filters unstable poses before synthesis.
"""
        notebook = """# Notebook

### [SLIDE 1] Protein-Ligand Binding
**DIALOGUE:**
- **Iapetus:** The protein pocket is the anchor here.
- **Erinome:** The ligand pose has to survive the validation gate.

### [SLIDE 2] Wet-Lab Validation
**DIALOGUE:**
- **Iapetus:** The validation gate keeps the workflow grounded.
- **Erinome:** That is the experiment-facing constraint.
"""
        duration = 120
        storyboard = {
            "clusters": [
                {
                    "claim": "Binding quality depends on pocket-aware validation.",
                    "representative_papers": ["PocketBind"],
                    "retained_slides": [1, 2],
                    "merged_slides": [],
                    "dropped_slides": [],
                    "retained_rationale": "Slides 1 and 2 carry distinct mechanism and experiment roles.",
                    "merged_rationale": "No duplicate slides in this compact fixture.",
                    "dropped_rationale": "No drops needed for the fixture.",
                    "figure_roles": {"1": "mechanism", "2": "experiment"},
                    "narration_beats": {"1": "name pocket geometry", "2": "name validation gate"},
                }
            ]
        }
        (deck_dir / "editorial_storyboard.json").write_text(
            json.dumps(storyboard), encoding="utf-8"
        )
    else:
        repeated = "- Same claim about the same method and benchmark.\n"
        slides = "\n\n".join(
            f"""## Slide {index}: Repeated Claim {index}

{repeated}
- References: [Same Paper](https://example.com/paper), [Same Benchmark](https://example.com/bench)

<!--
FIGURE_NEED: Diagram showing the same boxes
-->

![Generated figure for slide {index}](assets/generated/slide_{index:03d}.svg)
"""
            for index in range(1, 4)
        )
        presenter = """# Presenter

### [SLIDE 1] Repeated Claim 1
**SAY:**
> This slide repeats a generic claim.

### [ZOOM 1] Repeated Claim 1 — Figure Detail
**SAY:**
> The diagram shows boxes.

### [SLIDE 2] Repeated Claim 2
**SAY:**
> This slide repeats a generic claim.

### [ZOOM 2] Repeated Claim 2 — Figure Detail
**SAY:**
> The diagram shows boxes.

### [SLIDE 3] Repeated Claim 3
**SAY:**
> This slide repeats a generic claim.

### [ZOOM 3] Repeated Claim 3 — Figure Detail
**SAY:**
> The diagram shows boxes.
"""
        notebook = """# Notebook

### [SLIDE 1] Repeated Claim 1
**DIALOGUE:**
- **Iapetus:** This repeats a generic claim.

### [SLIDE 100001] Repeated Claim 1 — Figure Detail
**DIALOGUE:**
- **Iapetus:** The diagram shows boxes.

### [SLIDE 2] Repeated Claim 2
**DIALOGUE:**
- **Iapetus:** This repeats a generic claim.

### [SLIDE 100002] Repeated Claim 2 — Figure Detail
**DIALOGUE:**
- **Iapetus:** The diagram shows boxes.

### [SLIDE 3] Repeated Claim 3
**DIALOGUE:**
- **Iapetus:** This repeats a generic claim.

### [SLIDE 100003] Repeated Claim 3 — Figure Detail
**DIALOGUE:**
- **Iapetus:** The diagram shows boxes.
"""
        duration = 2501

    (deck_dir / "5_draft_slides_with_images.md").write_text(slides, encoding="utf-8")
    (deck_dir / "6_presenter_script.md").write_text(presenter, encoding="utf-8")
    (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
        notebook, encoding="utf-8"
    )
    (deck_dir / "video_build").mkdir()
    (deck_dir / "video_build_notebooklm").mkdir()
    for build_dir in ("video_build", "video_build_notebooklm"):
        (deck_dir / build_dir / "script.json").write_text(
            json.dumps({"duration_seconds": duration, "segments": []}),
            encoding="utf-8",
        )
    (deck_dir / "asset_inventory.json").write_text(
        json.dumps(
            {
                "figure_specs": [
                    {
                        "slide_number": 1,
                        "entities": ["Protein pocket", "Ligand pose", "Validation gate"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_presentation_editorial_gate_fails_duration_zoom_redundancy_and_visual_repetition(
    tmp_path: Path,
):
    module = load_presentation_script("presentation_editorial_gate.py")
    deck_dir = tmp_path / "decks" / "01_demo"
    _write_editorial_gate_deck(deck_dir, passing=False)

    report = module.analyze_deck(deck_dir=deck_dir, max_zoom_count=1)
    rules = {finding["rule"] for finding in report["findings"]}

    assert report["status"] == "failed"
    assert {
        "editorial_storyboard_missing",
        "single_presenter_duration_over_budget",
        "notebooklm_duration_over_budget",
        "zoom_detail_density_over_budget",
        "redundant_slide_claim_clusters",
        "generic_figure_alt_text_repetition",
        "generic_figure_need_repetition",
        "central_figure_entities_missing_from_narration",
    } <= rules


def test_presentation_editorial_gate_passes_compressed_storyboard_with_aligned_figures(
    tmp_path: Path,
):
    module = load_presentation_script("presentation_editorial_gate.py")
    deck_dir = tmp_path / "decks" / "01_demo"
    _write_editorial_gate_deck(deck_dir, passing=True)

    report = module.analyze_deck(deck_dir=deck_dir)

    assert report["status"] == "passed"
    assert report["metrics"]["storyboard_present"] is True
    assert report["metrics"]["storyboard_cluster_count"] == 1
    assert report["metrics"]["storyboard_incomplete_cluster_count"] == 0
    assert report["metrics"]["central_figure_missing_mention_count"] == 0


def test_presentation_editorial_gate_fails_incomplete_storyboard(tmp_path: Path):
    module = load_presentation_script("presentation_editorial_gate.py")
    deck_dir = tmp_path / "decks" / "01_demo"
    _write_editorial_gate_deck(deck_dir, passing=True)
    (deck_dir / "editorial_storyboard.json").write_text(
        json.dumps({"clusters": [{"claim": "Too thin"}]}), encoding="utf-8"
    )

    report = module.analyze_deck(deck_dir=deck_dir)
    rules = {finding["rule"] for finding in report["findings"]}

    assert report["status"] == "failed"
    assert "editorial_storyboard_incomplete" in rules
    assert report["metrics"]["storyboard_incomplete_cluster_count"] == 1


def test_build_editorial_storyboard_writes_required_cluster_contract(tmp_path: Path):
    storyboard_module = load_presentation_script("build_editorial_storyboard.py")
    gate_module = load_presentation_script("presentation_editorial_gate.py")
    deck_dir = tmp_path / "decks" / "01_demo"
    _write_editorial_gate_deck(deck_dir, passing=True)
    (deck_dir / "editorial_storyboard.json").unlink()

    storyboard = storyboard_module.build_storyboard(deck_dir=deck_dir, max_zoom_count=1)
    (deck_dir / "editorial_storyboard.json").write_text(
        json.dumps(storyboard), encoding="utf-8"
    )
    report = gate_module.analyze_deck(deck_dir=deck_dir)

    assert storyboard["clusters"]
    assert storyboard["clusters"][0]["representative_papers"]
    assert storyboard["clusters"][0]["retained_rationale"]
    assert storyboard["clusters"][0]["figure_roles"]
    assert storyboard["clusters"][0]["narration_beats"]
    assert report["metrics"]["storyboard_incomplete_cluster_count"] == 0


def test_repair_slide_figure_metadata_uses_figure_specs(tmp_path: Path):
    module = load_presentation_script("repair_slide_figure_metadata.py")
    deck_dir = tmp_path / "decks" / "01_demo"
    _write_editorial_gate_deck(deck_dir, passing=False)
    (deck_dir / "asset_inventory.json").write_text(
        json.dumps(
            {
                "figure_specs": [
                    {
                        "slide_number": 1,
                        "depiction_type": "system_architecture",
                        "semantic_goal": "Show memory poisoning entering an agent tool loop",
                        "entities": ["Agent core", "Memory store", "Tool verifier"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = module.repair_markdown(
        deck_dir / "5_draft_slides_with_images.md",
        asset_inventory=deck_dir / "asset_inventory.json",
    )
    repaired = (deck_dir / "5_draft_slides_with_images.md").read_text(encoding="utf-8")

    assert report["alt_updates"] == 1
    assert "Generated figure for slide 1" not in repaired
    assert "Agent core" in repaired
    assert "memory poisoning" in repaired


def test_apply_editorial_storyboard_to_script_prunes_unselected_zoom(tmp_path: Path):
    module = load_presentation_script("apply_editorial_storyboard_to_script.py")
    script_md = tmp_path / "script.md"
    storyboard = tmp_path / "editorial_storyboard.json"
    script_md.write_text(
        """### [SLIDE 1] One
**SAY:**
> One.

### [ZOOM 1] One — Figure Detail
**SAY:**
> Keep this.

### [SLIDE 2] Two
**SAY:**
> Two.

### [ZOOM 2] Two — Figure Detail
**SAY:**
> Remove this.
""",
        encoding="utf-8",
    )
    storyboard.write_text(json.dumps({"selected_zoom_slides": [1]}), encoding="utf-8")

    report = module.apply_zoom_policy(script_md, storyboard=storyboard)
    repaired = script_md.read_text(encoding="utf-8")

    assert report["kept_zoom_count"] == 1
    assert report["removed_zoom_count"] == 1
    assert "[ZOOM 1]" in repaired
    assert "[ZOOM 2]" not in repaired


def test_phase13_editorial_preflight_summarizes_topic_and_executive_decks(tmp_path: Path):
    module = load_presentation_script("run_phase13_editorial_preflight.py")
    decks_dir = tmp_path / "decks"
    passing_topic = decks_dir / "01_passing_topic"
    failing_topic = decks_dir / "02_failing_topic"
    executive_short = decks_dir / "executive_summary_short"
    ignored_dir = decks_dir / "_qc_contact_sheets"
    _write_editorial_gate_deck(passing_topic, passing=True)
    _write_editorial_gate_deck(failing_topic, passing=False)
    _write_editorial_gate_deck(executive_short, passing=True)
    ignored_dir.mkdir(parents=True)

    report = module.build_preflight_report(decks_dir)
    markdown = module.render_markdown(report)

    assert report["status"] == "failed"
    assert report["summary"]["deck_count"] == 3
    assert report["summary"]["failed_decks"] == ["02_failing_topic"]
    assert report["summary"]["passed_deck_count"] == 2
    assert "editorial_storyboard_missing" in report["summary"]["hard_failure_rule_counts"]
    assert "_qc_contact_sheets" not in {deck["deck_id"] for deck in report["decks"]}
    assert "| 01_passing_topic | topic | passed | 0 | 0 |" in markdown
    assert "`02_failing_topic`" in markdown


def test_phase13_editorial_preflight_cli_writes_reports_and_can_fail_on_fail(
    tmp_path: Path,
):
    module = load_presentation_script("run_phase13_editorial_preflight.py")
    decks_dir = tmp_path / "decks"
    _write_editorial_gate_deck(decks_dir / "01_failing_topic", passing=False)
    report_out = tmp_path / "preflight.json"
    markdown_out = tmp_path / "preflight.md"

    exit_code = module.main(
        [
            "--decks-dir",
            str(decks_dir),
            "--report-out",
            str(report_out),
            "--markdown-out",
            str(markdown_out),
            "--fail-on-fail",
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    markdown = markdown_out.read_text(encoding="utf-8")
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["summary"]["failed_deck_count"] == 1
    assert "Phase 13 Editorial Preflight" in markdown


def test_record_human_review_receipt_requires_checklist_for_approval(
    tmp_path: Path,
):
    module = load_presentation_script("record_human_review_receipt.py")
    deck_dir = tmp_path / "decks" / "04_demo_topic"
    deck_dir.mkdir(parents=True)

    exit_code = module.main(
        [
            "--deck-dir",
            str(deck_dir),
            "--decision",
            "approved_for_bulk_rollout",
            "--reviewer",
            "operator",
        ]
    )

    assert exit_code == 2
    assert not (deck_dir / "human_review_receipt.json").exists()


def test_record_human_review_receipt_writes_approved_phase13_receipt(
    tmp_path: Path,
):
    module = load_presentation_script("record_human_review_receipt.py")
    deck_dir = tmp_path / "decks" / "04_demo_topic"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation.html").write_text("<html></html>", encoding="utf-8")
    (deck_dir / "presentation_video.mp4").write_bytes(b"video")
    produced_review = deck_dir / "presentation_review.json"
    produced_review.write_text(
        json.dumps(
            {
                "status": "passed",
                "production_ready": True,
                "aggregated_feedback": {"blockers": [], "warnings": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_review = tmp_path / "review.md"
    source_review.write_text("approved", encoding="utf-8")
    objective_evidence = tmp_path / "objective-evidence.json"
    objective_evidence.write_text(
        json.dumps(
            {
                "status": "passed",
                "deck_id": deck_dir.name,
                "blockers": [],
                "contact_sheet_path": str(tmp_path / "contact_sheet.png"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    canary_packet = tmp_path / "canary-packet.json"
    canary_packet.write_text(
        json.dumps(
            {
                "status": "ready_for_operator_review",
                "deck_id": deck_dir.name,
                "blockers": [],
                "objective_evidence_report": {
                    "status": "passed",
                    "contact_sheet_present": True,
                },
                "produced_artifact_review": {
                    "status": "passed",
                    "blocker_count": 0,
                    "stale_input_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--deck-dir",
            str(deck_dir),
            "--decision",
            "approved_for_bulk_rollout",
            "--reviewer",
            "operator",
            "--notes",
            "Canary approved after local playback.",
            "--source-review-md",
            str(source_review),
            "--canary-review-packet",
            str(canary_packet),
            "--objective-evidence-report",
            str(objective_evidence),
            "--produced-artifact-review",
            str(produced_review),
            "--all-checks-pass",
        ]
    )

    receipt = json.loads((deck_dir / "human_review_receipt.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert receipt["status"] == "accepted"
    assert receipt["decision"] == "approved_for_bulk_rollout"
    assert receipt["reviewer"] == "operator"
    assert receipt["notes"] == ["Canary approved after local playback."]
    assert {item["status"] for item in receipt["checklist"]} == {"pass"}
    assert "presentation.html" in receipt["artifacts_reviewed"]
    assert "presentation_video.mp4" in receipt["artifacts_reviewed"]
    assert receipt["review_evidence"]["source_review_md"]["sha256"]
    assert receipt["review_evidence"]["canary_review_packet"]["status"] == "ready_for_operator_review"
    assert receipt["review_evidence"]["canary_review_packet"]["objective_evidence_status"] == "passed"
    assert receipt["review_evidence"]["objective_evidence_report"]["status"] == "passed"
    assert receipt["review_evidence"]["produced_artifact_review"]["status"] == "passed"


def test_record_human_review_receipt_rejects_approval_without_canary_packet(
    tmp_path: Path,
):
    module = load_presentation_script("record_human_review_receipt.py")
    deck_dir = tmp_path / "decks" / "04_demo_topic"
    deck_dir.mkdir(parents=True)
    source_review = tmp_path / "review.md"
    source_review.write_text("approved", encoding="utf-8")
    objective_evidence = tmp_path / "objective-evidence.json"
    objective_evidence.write_text(
        json.dumps({"status": "passed", "blockers": []}) + "\n",
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--deck-dir",
            str(deck_dir),
            "--decision",
            "approved_for_bulk_rollout",
            "--reviewer",
            "operator",
            "--source-review-md",
            str(source_review),
            "--objective-evidence-report",
            str(objective_evidence),
            "--all-checks-pass",
        ]
    )

    assert exit_code == 2
    assert not (deck_dir / "human_review_receipt.json").exists()


def test_phase13_rollout_blocks_non_canary_without_approved_review(
    tmp_path: Path,
):
    module = load_presentation_script("run_phase13_rollout.py")
    workspace = tmp_path / "conferences" / "demo2026"
    (workspace / "decks" / "04_canary").mkdir(parents=True)
    (workspace / "decks" / "01_topic").mkdir()
    report_out = tmp_path / "rollout.json"

    exit_code = module.main(
        [
            "--workspace",
            str(workspace),
            "--deck-id",
            "01_topic",
            "--canary-deck-id",
            "04_canary",
            "--require-approved-canary",
            "--report-out",
            str(report_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "blocked_pending_canary_human_review"
    assert report["blocked_decks"] == ["01_topic"]
    assert report["canary_approved"] is False


def test_phase13_rollout_blocks_live_transform_without_provider_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    module = load_presentation_script("run_phase13_rollout.py")
    workspace = tmp_path / "conferences" / "demo2026"
    canary = workspace / "decks" / "04_canary"
    canary.mkdir(parents=True)
    report_out = tmp_path / "rollout.json"
    markdown_out = tmp_path / "rollout.md"

    exit_code = module.main(
        [
            "--workspace",
            str(workspace),
            "--deck-id",
            "04_canary",
            "--canary-deck-id",
            "04_canary",
            "--execute",
            "--enable-live-notebooklm-transform",
            "--env-file",
            str(tmp_path / "missing.env"),
            "--report-out",
            str(report_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    markdown = markdown_out.read_text(encoding="utf-8")
    assert exit_code == 2
    assert report["status"] == "blocked_missing_provider_env"
    assert report["provider_environment"]["env_file_exists"] is False
    assert report["provider_environment"]["has_google_or_gemini_key"] is False
    assert report["provider_environment"]["values_recorded"] is False
    assert "Secret values recorded: `False`" in markdown


def test_phase13_rollout_plans_pre_tts_commands_after_canary_approval(
    tmp_path: Path,
):
    module = load_presentation_script("run_phase13_rollout.py")
    workspace = tmp_path / "conferences" / "demo2026"
    canary = workspace / "decks" / "04_canary"
    deck_dir = workspace / "decks" / "01_topic"
    canary.mkdir(parents=True)
    deck_dir.mkdir()
    (canary / "human_review_receipt.json").write_text(
        json.dumps({"decision": "approved_for_bulk_rollout"}) + "\n",
        encoding="utf-8",
    )
    canary_packet = tmp_path / "canary_packet.json"
    canary_packet.write_text(
        json.dumps(
            {
                "status": "approved_for_bulk_rollout",
                "deck_id": "04_canary",
                "blockers": [],
                "human_review": {"approved": True},
                "objective_evidence_report": {
                    "status": "passed",
                    "contact_sheet_present": True,
                },
                "produced_artifact_review": {
                    "status": "passed",
                    "blocker_count": 0,
                    "stale_input_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (deck_dir / "5_draft_slides_with_images.md").write_text(
        "## Slide 1: Demo\n\n- Claim\n",
        encoding="utf-8",
    )
    (deck_dir / "6_presenter_script.md").write_text(
        "### [SLIDE 1] Demo\n\n**SAY:**\n> Spoken text.\n",
        encoding="utf-8",
    )
    (deck_dir / "asset_inventory.json").write_text(
        json.dumps({"figure_specs": []}) + "\n",
        encoding="utf-8",
    )
    report_out = tmp_path / "rollout.json"
    markdown_out = tmp_path / "rollout.md"

    exit_code = module.main(
        [
            "--workspace",
            str(workspace),
            "--deck-id",
            "01_topic",
            "--canary-deck-id",
            "04_canary",
            "--canary-review-packet",
            str(canary_packet),
            "--require-approved-canary",
            "--report-out",
            str(report_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    markdown = markdown_out.read_text(encoding="utf-8")
    steps = report["decks"][0]["steps"]
    commands = {step["name"]: step["command"] for step in steps}
    step_names = [step["name"] for step in steps]
    series_reviews = {step["name"]: step["command"] for step in report["series_reviews"]}
    assert exit_code == 0
    assert report["status"] == "planned"
    assert report["canary_approved"] is True
    assert report["canary_review_packet"]["approved"] is True
    assert "--stop-after-script-json" in commands["build_presenter_script_json"]
    assert "--stop-after-transform" in commands["transform_notebooklm_script_json"]
    assert {
        "lint_presenter_narration",
        "review_presenter_narration",
        "lint_notebooklm_narration",
        "review_notebooklm_narration",
    }.issubset(step_names)
    assert step_names.index("review_notebooklm_narration") < step_names.index("run_editorial_gate")
    assert "--review-out" in commands["review_presenter_narration"]
    assert "--json-out" in commands["review_notebooklm_narration"]
    assert any(
        "presentation_video.mp4" in token for token in commands["build_presenter_script_json"]
    )
    assert all(step["status"] == "planned" for step in steps)
    assert set(series_reviews) == {
        "review_presenter_narration_series",
        "review_notebooklm_narration_series",
    }
    assert "review_narration_series.py" in series_reviews["review_presenter_narration_series"][1]
    assert "--script-name" in series_reviews["review_presenter_narration_series"]
    assert "6_presenter_script.md" in series_reviews["review_presenter_narration_series"]
    assert any(
        "presenter_narration_series_review.json" in token
        for token in series_reviews["review_presenter_narration_series"]
    )
    assert "6_presenter_script_notebooklm_2host.md" in series_reviews["review_notebooklm_narration_series"]
    assert any(
        "narration_series_review.json" in token
        for token in series_reviews["review_notebooklm_narration_series"]
    )
    assert not (deck_dir / "presentation.html").exists()
    assert "Phase 13 Rollout Plan" in markdown
    assert "| 01_topic | topic | planned |" in markdown
    assert "## Series Reviews" in markdown


def test_phase13_rollout_blocks_approved_receipt_without_approved_packet(
    tmp_path: Path,
):
    module = load_presentation_script("run_phase13_rollout.py")
    workspace = tmp_path / "conferences" / "demo2026"
    canary = workspace / "decks" / "04_canary"
    deck_dir = workspace / "decks" / "01_topic"
    canary.mkdir(parents=True)
    deck_dir.mkdir()
    (canary / "human_review_receipt.json").write_text(
        json.dumps({"decision": "approved_for_bulk_rollout"}) + "\n",
        encoding="utf-8",
    )
    report_out = tmp_path / "rollout.json"

    exit_code = module.main(
        [
            "--workspace",
            str(workspace),
            "--deck-id",
            "01_topic",
            "--canary-deck-id",
            "04_canary",
            "--require-approved-canary",
            "--report-out",
            str(report_out),
        ]
    )

    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["status"] == "blocked_pending_canary_review_packet"
    assert report["canary_approved"] is True
    assert report["canary_review_packet"]["approved"] is False
    assert report["canary_review_packet"]["status"] == "not_provided"


def test_phase13_bulk_readiness_blocks_without_human_review(tmp_path: Path):
    module = load_presentation_script("verify_phase13_bulk_readiness.py")
    workspace = tmp_path / "conferences" / "demo2026"
    canary = workspace / "decks" / "04_canary"
    canary.mkdir(parents=True)
    contact_sheet = tmp_path / "contact_sheet.png"
    contact_sheet.write_text("contact", encoding="utf-8")
    objective_report = tmp_path / "objective.json"
    objective_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "deck_id": "04_canary",
                "blockers": [],
                "contact_sheet_path": str(contact_sheet),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "status": "ready_for_operator_review",
                "deck_id": "04_canary",
                "blockers": [],
                "human_review": {"approved": False, "evidence_errors": []},
                "objective_evidence_report": {
                    "status": "passed",
                    "contact_sheet_present": True,
                },
                "produced_artifact_review": {
                    "status": "passed",
                    "blocker_count": 0,
                    "stale_input_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(
        workspace=workspace,
        canary_deck_id="04_canary",
        canary_review_packet=packet,
        objective_evidence_report=objective_report,
        rollout_plan=None,
    )

    assert report["status"] == "blocked"
    assert "canary_human_review_not_approved" in report["blockers"]
    assert "canary_review_packet_not_approved" in report["blockers"]


def test_phase13_bulk_readiness_passes_after_approved_packet_and_rollout(tmp_path: Path):
    module = load_presentation_script("verify_phase13_bulk_readiness.py")
    workspace = tmp_path / "conferences" / "demo2026"
    canary = workspace / "decks" / "04_canary"
    canary.mkdir(parents=True)
    (canary / "human_review_receipt.json").write_text(
        json.dumps(
            {
                "status": "accepted",
                "decision": "approved_for_bulk_rollout",
                "review_evidence": {"source_review_md": {}, "canary_review_packet": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    contact_sheet = tmp_path / "contact_sheet.png"
    contact_sheet.write_text("contact", encoding="utf-8")
    objective_report = tmp_path / "objective.json"
    objective_report.write_text(
        json.dumps(
            {
                "status": "passed",
                "deck_id": "04_canary",
                "blockers": [],
                "contact_sheet_path": str(contact_sheet),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    packet = tmp_path / "packet.json"
    packet.write_text(
        json.dumps(
            {
                "status": "approved_for_bulk_rollout",
                "deck_id": "04_canary",
                "blockers": [],
                "human_review": {"approved": True, "evidence_errors": []},
                "objective_evidence_report": {
                    "status": "passed",
                    "contact_sheet_present": True,
                },
                "produced_artifact_review": {
                    "status": "passed",
                    "blocker_count": 0,
                    "stale_input_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rollout = tmp_path / "rollout.json"
    rollout.write_text(
        json.dumps(
            {
                "status": "planned",
                "canary_approved": True,
                "canary_review_packet": {"approved": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = module.build_report(
        workspace=workspace,
        canary_deck_id="04_canary",
        canary_review_packet=packet,
        objective_evidence_report=objective_report,
        rollout_plan=rollout,
    )
    markdown = module.render_markdown(report)

    assert report["status"] == "ready_for_bulk_rollout"
    assert report["blockers"] == []
    assert "Phase 13 Bulk Readiness" in markdown


def _write_phase13_completion_deck(deck_dir: Path) -> None:
    deck_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in (
        "presentation.html",
        "presentation.pdf",
        "presentation_video.mp4",
        "presentation_video_notebooklm.mp4",
        "6_presenter_script.md",
        "6_presenter_script_notebooklm_2host.md",
        "presentation_review.json",
    ):
        (deck_dir / relative_path).write_text("artifact", encoding="utf-8")
    (deck_dir / "presentation_review.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "aggregated_feedback": {"blockers": [], "warnings": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (deck_dir / "presentation_editorial_gate_report.json").write_text(
        json.dumps({"status": "warning", "finding_counts": {"warn": 1}}) + "\n",
        encoding="utf-8",
    )


def _write_phase13_series_reviews(decks_dir: Path) -> None:
    for filename in ("presenter_narration_series_review.json", "narration_series_review.json"):
        (decks_dir / filename).write_text(
            json.dumps({"status": "passed", "approve_for_tts": True, "finding_counts": {}}) + "\n",
            encoding="utf-8",
        )


def _phase13_required_validation_stages() -> list[dict[str, str]]:
    return [
        {"stage_id": "figures_structural_validation", "status": "succeeded"},
        {"stage_id": "figure_validation_receipt", "status": "succeeded"},
        {"stage_id": "revealjs_render_validation", "status": "succeeded"},
        {"stage_id": "visual_qc_validation", "status": "succeeded"},
        {"stage_id": "presenter_script_validation", "status": "succeeded"},
        {"stage_id": "video_generation_validation", "status": "succeeded"},
        {"stage_id": "pdf_export_validation", "status": "succeeded"},
    ]


def test_phase13_completion_verifier_blocks_missing_readiness_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("verify_phase13_completion.py")
    workspace = tmp_path / "conferences" / "demo2026"
    phase_dir = tmp_path / "phase13"
    _write_phase13_completion_deck(workspace / "decks" / "01_topic")
    phase_dir.mkdir()
    (phase_dir / "13-BULK-READINESS.json").write_text(
        json.dumps({"status": "blocked", "blockers": ["demo"]}) + "\n",
        encoding="utf-8",
    )
    (phase_dir / "13-ROLLOUT-PLAN.json").write_text(
        json.dumps({"status": "blocked_pending_canary_human_review"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)

    report = module.build_report(
        workspace=workspace,
        phase_dir=phase_dir,
        final_validation_summary=None,
    )

    assert report["status"] == "blocked"
    assert "bulk_readiness_not_ready" in report["blockers"]
    assert "bulk_rollout_not_passed" in report["blockers"]
    assert "final_strict_validation_not_succeeded" in report["blockers"]
    assert "final_strict_validation_required_stages_missing_or_failed" in report["blockers"]
    assert "series_narration_reviews_not_passed" in report["blockers"]


def test_phase13_completion_verifier_blocks_missing_required_validation_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("verify_phase13_completion.py")
    workspace = tmp_path / "conferences" / "demo2026"
    phase_dir = tmp_path / "phase13"
    final_summary = tmp_path / "summary.json"
    _write_phase13_completion_deck(workspace / "decks" / "01_topic")
    _write_phase13_series_reviews(workspace / "decks")
    phase_dir.mkdir()
    (phase_dir / "13-BULK-READINESS.json").write_text(
        json.dumps({"status": "ready_for_bulk_rollout", "blockers": []}) + "\n",
        encoding="utf-8",
    )
    (phase_dir / "13-ROLLOUT-PLAN.json").write_text(
        json.dumps({"status": "passed"}) + "\n",
        encoding="utf-8",
    )
    final_summary.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "halt_stage_id": None,
                "stages": [
                    stage
                    for stage in _phase13_required_validation_stages()
                    if stage["stage_id"] != "visual_qc_validation"
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)

    report = module.build_report(
        workspace=workspace,
        phase_dir=phase_dir,
        final_validation_summary=final_summary,
    )

    assert report["status"] == "blocked"
    assert "final_strict_validation_required_stages_missing_or_failed" in report["blockers"]
    assert report["strict_validation"]["missing_required_stages"] == ["visual_qc_validation"]


def test_phase13_completion_verifier_passes_after_all_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("verify_phase13_completion.py")
    workspace = tmp_path / "conferences" / "demo2026"
    phase_dir = tmp_path / "phase13"
    final_summary = tmp_path / "summary.json"
    _write_phase13_completion_deck(workspace / "decks" / "01_topic")
    _write_phase13_completion_deck(workspace / "decks" / "executive_summary_short")
    _write_phase13_series_reviews(workspace / "decks")
    phase_dir.mkdir()
    (phase_dir / "13-BULK-READINESS.json").write_text(
        json.dumps({"status": "ready_for_bulk_rollout", "blockers": []}) + "\n",
        encoding="utf-8",
    )
    (phase_dir / "13-ROLLOUT-PLAN.json").write_text(
        json.dumps({"status": "passed"}) + "\n",
        encoding="utf-8",
    )
    final_summary.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "halt_stage_id": None,
                "stages": _phase13_required_validation_stages(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)

    report = module.build_report(
        workspace=workspace,
        phase_dir=phase_dir,
        final_validation_summary=final_summary,
    )
    markdown = module.render_markdown(report)

    assert report["status"] == "complete"
    assert report["blockers"] == []
    assert report["deck_count"] == 2
    assert "Phase 13 Completion Verification" in markdown


def _write_canary_review_fixture(deck_dir: Path, validation_summary: Path) -> None:
    deck_dir.mkdir(parents=True)
    for relative_path in (
        "presentation_video.mp4",
        "presentation_video_notebooklm.mp4",
        "presentation.pdf",
        "presentation.html",
        "presentation_blueprint.json",
        "presentation_deck.json",
        "source_binding_report.json",
        "visual_plan.json",
        "render_spec.json",
        "delivery_single_speaker.json",
        "delivery_notebooklm.json",
        "video_sync_validation_single_speaker.json",
        "video_sync_validation_notebooklm.json",
        "editorial_storyboard.json",
        "presentation_review.json",
        "6_presenter_script.md",
        "6_presenter_script_notebooklm_2host.md",
    ):
        path = deck_dir / relative_path
        path.write_text("placeholder", encoding="utf-8")
    (deck_dir / "presentation_review.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "production_ready": True,
                "inputs": {
                    "requirements": {"path": "decks/04_canary/presentation_blueprint.json"},
                    "narration_script": {"path": "decks/04_canary/6_presenter_script.md"},
                    "artifacts": {},
                },
                "aggregated_feedback": {"blockers": [], "warnings": []},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (deck_dir / "presentation_editorial_gate_report.json").write_text(
        json.dumps(
            {
                "status": "warning",
                "finding_counts": {"warn": 1},
                "metrics": {
                    "presenter_estimated_duration_seconds": 1200,
                    "notebooklm_estimated_duration_seconds": 1300,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_summary.parent.mkdir(parents=True)
    validation_summary.write_text(
        json.dumps(
            {
                "halt_status": "halted",
                "halt_stage_id": "human_review_validation",
                "stages": [
                    {"stage_id": "video_generation_validation", "status": "succeeded"},
                    {"stage_id": "pdf_export_validation", "status": "succeeded"},
                    {"stage_id": "human_review_validation", "status": "failed"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_canary_objective_evidence_fixture(
    deck_dir: Path,
    report_path: Path,
    markdown_path: Path,
    *,
    status: str = "passed",
) -> None:
    contact_sheet = report_path.parent / "contact_sheet.png"
    contact_sheet.write_text("contact", encoding="utf-8")
    markdown_path.write_text("# Evidence\n", encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {
                "status": status,
                "deck_id": deck_dir.name,
                "blockers": [] if status == "passed" else ["demo_blocker"],
                "warnings": [],
                "contact_sheet_path": str(contact_sheet),
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_approved_human_review_receipt(
    *,
    deck_dir: Path,
    review_md: Path,
    objective_evidence_report: Path,
    produced_artifact_review: Path | None = None,
) -> None:
    produced_artifact_review = produced_artifact_review or deck_dir / "presentation_review.json"
    (deck_dir / "human_review_receipt.json").write_text(
        json.dumps(
            {
                "decision": "approved_for_bulk_rollout",
                "status": "accepted",
                "review_evidence": {
                    "source_review_md": {
                        "path": str(review_md),
                        "sha256": _sha256(review_md),
                    },
                    "objective_evidence_report": {
                        "path": str(objective_evidence_report),
                        "sha256": _sha256(objective_evidence_report),
                        "status": "passed",
                        "blockers": [],
                    },
                    "produced_artifact_review": {
                        "path": str(produced_artifact_review),
                        "sha256": _sha256(produced_artifact_review),
                        "status": "passed",
                        "blocker_count": 0,
                    },
                    "canary_review_packet": {
                        "path": str(review_md.with_name("packet.json")),
                        "sha256": "snapshot-sha256",
                        "status": "ready_for_operator_review",
                        "objective_evidence_status": "passed",
                        "objective_contact_sheet_present": True,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_revision_human_review_receipt(*, deck_dir: Path, review_md: Path) -> None:
    (deck_dir / "human_review_receipt.json").write_text(
        json.dumps(
            {
                "decision": "needs_revision_before_bulk_rollout",
                "status": "needs_revision",
                "review_evidence": {
                    "source_review_md": {
                        "path": str(review_md),
                        "sha256": _sha256(review_md),
                    },
                    "objective_evidence_report": None,
                    "canary_review_packet": None,
                },
                "notes": ["Slides need revision before bulk rollout."],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase13_canary_review_packet_ready_without_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)
    report_out = tmp_path / "packet.json"
    markdown_out = tmp_path / "packet.md"
    objective_evidence = tmp_path / "objective-evidence.md"
    objective_evidence_report = tmp_path / "objective-evidence.json"
    _write_canary_objective_evidence_fixture(
        deck_dir,
        objective_evidence_report,
        objective_evidence,
    )

    exit_code = module.main(
        [
            "--deck-dir",
            str(deck_dir),
            "--validation-summary",
            str(validation_summary),
            "--objective-evidence-md",
            str(objective_evidence),
            "--objective-evidence-report",
            str(objective_evidence_report),
            "--report-out",
            str(report_out),
            "--markdown-out",
            str(markdown_out),
        ]
    )

    packet = json.loads(report_out.read_text(encoding="utf-8"))
    markdown = markdown_out.read_text(encoding="utf-8")
    assert exit_code == 0
    assert packet["status"] == "ready_for_operator_review"
    assert packet["blockers"] == []
    assert "human_review_receipt_pending" in packet["warnings"]
    assert packet["strict_validation"]["status"] == "awaiting_human_review"
    assert packet["objective_evidence_md"] == str(objective_evidence)
    assert packet["objective_evidence_report"]["status"] == "passed"
    assert packet["objective_evidence_report"]["contact_sheet_present"] is True
    assert packet["produced_artifact_review"]["status"] == "passed"
    assert "Phase 13 Canary Review Packet" in markdown
    assert "Objective Evidence" in markdown


def test_phase13_canary_review_packet_blocks_missing_objective_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)

    packet = module.build_packet(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
    )

    assert packet["status"] == "blocked"
    assert "objective_evidence_missing_or_invalid" in packet["blockers"]


def test_phase13_canary_review_packet_blocks_failed_produced_artifact_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)
    objective_evidence = tmp_path / "objective-evidence.md"
    objective_evidence_report = tmp_path / "objective-evidence.json"
    _write_canary_objective_evidence_fixture(
        deck_dir,
        objective_evidence_report,
        objective_evidence,
    )
    (deck_dir / "presentation_review.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "aggregated_feedback": {"blockers": [{"comment": "Unreadable figure."}]},
                "inputs": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    packet = module.build_packet(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        objective_evidence_md=objective_evidence,
        objective_evidence_report=objective_evidence_report,
    )

    assert packet["status"] == "blocked"
    assert "produced_artifact_review_failed" in packet["blockers"]
    assert "produced_artifact_review_has_blockers" in packet["blockers"]


def test_phase13_canary_review_packet_approved_with_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)
    review_md = tmp_path / "human-review.md"
    review_md.write_text("# Human review\n", encoding="utf-8")
    objective_evidence = tmp_path / "objective-evidence.md"
    objective_evidence_report = tmp_path / "objective-evidence.json"
    _write_canary_objective_evidence_fixture(
        deck_dir,
        objective_evidence_report,
        objective_evidence,
    )
    _write_approved_human_review_receipt(
        deck_dir=deck_dir,
        review_md=review_md,
        objective_evidence_report=objective_evidence_report,
    )

    packet = module.build_packet(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        review_md=review_md,
        objective_evidence_md=objective_evidence,
        objective_evidence_report=objective_evidence_report,
    )

    assert packet["status"] == "approved_for_bulk_rollout"
    assert packet["human_review"]["approved"] is True
    assert packet["produced_artifact_review"]["blocker_count"] == 0
    assert "human_review_receipt_pending" not in packet["warnings"]


def test_phase13_canary_review_packet_records_revision_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)
    review_md = tmp_path / "human-review.md"
    review_md.write_text("# Human review\n\nNeeds revision.\n", encoding="utf-8")
    objective_evidence = tmp_path / "objective-evidence.md"
    objective_evidence_report = tmp_path / "objective-evidence.json"
    _write_canary_objective_evidence_fixture(
        deck_dir,
        objective_evidence_report,
        objective_evidence,
    )
    _write_revision_human_review_receipt(deck_dir=deck_dir, review_md=review_md)

    packet = module.build_packet(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        review_md=review_md,
        objective_evidence_md=objective_evidence,
        objective_evidence_report=objective_evidence_report,
    )

    assert packet["status"] == "needs_revision_before_bulk_rollout"
    assert packet["human_review"]["revision_required"] is True
    assert packet["human_review"]["approved"] is False
    assert "human_review_requested_revision" in packet["blockers"]
    assert "human_review_receipt_pending" not in packet["warnings"]


def test_phase13_canary_review_packet_blocks_stale_approval_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_review_packet.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_canary_review_fixture(deck_dir, validation_summary)
    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 1800.0)
    review_md = tmp_path / "human-review.md"
    review_md.write_text("# Human review\n", encoding="utf-8")
    objective_evidence = tmp_path / "objective-evidence.md"
    objective_evidence_report = tmp_path / "objective-evidence.json"
    _write_canary_objective_evidence_fixture(
        deck_dir,
        objective_evidence_report,
        objective_evidence,
    )
    _write_approved_human_review_receipt(
        deck_dir=deck_dir,
        review_md=review_md,
        objective_evidence_report=objective_evidence_report,
    )
    objective_evidence_report.write_text(
        json.dumps({"status": "passed", "deck_id": deck_dir.name, "blockers": []}) + "\n",
        encoding="utf-8",
    )

    packet = module.build_packet(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        review_md=review_md,
        objective_evidence_md=objective_evidence,
        objective_evidence_report=objective_evidence_report,
    )

    assert packet["status"] == "blocked"
    assert packet["human_review"]["decision_approved"] is True
    assert packet["human_review"]["approved"] is False
    assert "human_review_receipt_evidence_mismatch" in packet["blockers"]
    assert "human_review_receipt_objective_evidence_digest_mismatch" in packet["human_review"]["evidence_errors"]


def test_phase13_canary_objective_evidence_sums_slide_receipts(tmp_path: Path):
    module = load_presentation_script("build_phase13_canary_objective_evidence.py")
    slides_concat = tmp_path / "slides_concat.txt"
    slides_concat.write_text(
        "ffconcat version 1.0\n"
        "file 'slide_001.mp4'\n"
        "duration 1.25\n"
        "file 'slide_002.mp4'\n"
        "duration 2.50\n",
        encoding="utf-8",
    )

    assert module._sum_slides_concat(slides_concat) == pytest.approx(3.75)


def test_phase13_canary_objective_evidence_parses_signalstats():
    module = load_presentation_script("build_phase13_canary_objective_evidence.py")
    output = (
        "lavfi.signalstats.YMIN=24\n"
        "lavfi.signalstats.YAVG=45.9315\n"
        "lavfi.signalstats.YMAX=224\n"
    )

    assert module._parse_signalstats_output(output) == {
        "YMIN": 24.0,
        "YAVG": 45.9315,
        "YMAX": 224.0,
    }


def _write_objective_evidence_fixture(deck_dir: Path, validation_summary: Path) -> None:
    slide_id = "slide-001"
    asset_path = Path("figures") / "slide-001.svg"
    for relative_path in (
        "presentation_video.mp4",
        "presentation_video_notebooklm.mp4",
        "video_build/narration_track.m4a",
        "video_build_notebooklm/narration_track.m4a",
    ):
        path = deck_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder", encoding="utf-8")
    (deck_dir / "video_build" / "slides_concat.txt").write_text(
        "file 'a.mp4'\nduration 900\nfile 'b.mp4'\nduration 900\n",
        encoding="utf-8",
    )
    (deck_dir / "video_build_notebooklm" / "slides_concat.txt").write_text(
        "file 'a.mp4'\nduration 1000\nfile 'b.mp4'\nduration 1000\n",
        encoding="utf-8",
    )
    (deck_dir / asset_path).parent.mkdir(parents=True, exist_ok=True)
    (deck_dir / asset_path).write_text(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"640\" height=\"360\">"
        "<text x=\"40\" y=\"80\" font-size=\"24\">Readable canary figure</text>"
        "</svg>\n",
        encoding="utf-8",
    )
    structured_artifacts = {
        "presentation_blueprint.json": {
            "schema_version": 1,
            "topic_id": "04_canary",
            "approved_expansion": {"approved": True, "target_slide_count": 1},
        },
        "presentation_deck.json": {
            "schema_version": 1,
            "author": "Nikolaos Vasiloglou",
            "analysis_report_url": "https://drive.google.com/file/d/canary/view",
            "slides": [{"slide_id": slide_id, "title": "Canary slide"}],
        },
        "source_binding_report.json": {
            "schema_version": 1,
            "slides": [
                {
                    "slide_id": slide_id,
                    "visible_references": [{"source_id": "SRC-001", "label": "Source One"}],
                    "unresolved_sources": [],
                }
            ],
        },
        "visual_plan.json": {
            "schema_version": 1,
            "slides": [
                {
                    "slide_id": slide_id,
                    "figures": [
                        {
                            "asset_paths": [str(asset_path)],
                            "source_labels": ["Source One"],
                            "narration_hook": "The audience sees the readable canary figure and why it matters.",
                        }
                    ],
                }
            ],
        },
        "render_spec.json": {
            "schema_version": 1,
            "author": "Nikolaos Vasiloglou",
            "analysis_report_url": "https://drive.google.com/file/d/canary/view",
            "slides": [
                {
                    "slide_id": slide_id,
                    "title": "Canary slide",
                    "speaker_notes": "The audience sees the readable canary figure and why it matters.",
                }
            ],
        },
        "delivery_single_speaker.json": {
            "schema_version": 1,
            "max_voice_gap_seconds": 3,
            "segments": [{"slide_id": slide_id, "source_ids": ["SRC-001"]}],
        },
        "delivery_notebooklm.json": {
            "schema_version": 1,
            "max_voice_gap_seconds": 3,
            "timing_divergence_seconds": 2,
            "segments": [{"slide_id": slide_id, "source_ids": ["SRC-001"]}],
        },
        "video_sync_validation_single_speaker.json": {
            "schema_version": 1,
            "status": "pass",
            "errors": [],
        },
        "video_sync_validation_notebooklm.json": {
            "schema_version": 1,
            "status": "pass",
            "errors": [],
        },
    }
    for filename, payload in structured_artifacts.items():
        (deck_dir / filename).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    validation_summary.parent.mkdir(parents=True)
    validation_summary.write_text(
        json.dumps(
            {
                "halt_status": "halted",
                "halt_stage_id": "human_review_validation",
                "stages": [
                    {"stage_id": "video_generation_validation", "status": "succeeded"},
                    {"stage_id": "pdf_export_validation", "status": "succeeded"},
                    {"stage_id": "human_review_validation", "status": "failed"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_phase13_canary_objective_evidence_passes_with_matching_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_objective_evidence.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    frames_dir = tmp_path / "frames"
    _write_objective_evidence_fixture(deck_dir, validation_summary)

    def fake_duration(path: Path) -> float:
        return 2000.0 if "notebooklm" in str(path) else 1800.0

    def fake_extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("frame", encoding="utf-8")
        return True

    monkeypatch.setattr(module, "_ffprobe_duration", fake_duration)
    monkeypatch.setattr(module, "_extract_frame", fake_extract_frame)
    monkeypatch.setattr(
        module,
        "_signalstats",
        lambda path: {"YMIN": 24.0, "YAVG": 45.0, "YMAX": 224.0},
    )
    monkeypatch.setattr(
        module,
        "_make_contact_sheet",
        lambda frame_paths, output_path: output_path.write_text("sheet", encoding="utf-8") >= 0,
    )

    report = module.build_evidence(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        frames_dir=frames_dir,
    )
    markdown = module.render_markdown(report)

    assert report["status"] == "passed"
    assert report["blockers"] == []
    assert report["strict_validation"]["status"] == "awaiting_human_review"
    assert report["videos"]["single_presenter"]["mp4_slides_delta_seconds"] == pytest.approx(0.0)
    assert report["videos"]["notebooklm"]["mp4_audio_delta_seconds"] == pytest.approx(0.0)
    assert report["contact_sheet_path"].endswith("contact_sheet.png")
    assert "Phase 13 Canary Objective Evidence" in markdown
    assert "Tail YAVG" in markdown


def test_phase13_canary_objective_evidence_fails_duration_and_black_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_phase13_canary_objective_evidence.py")
    deck_dir = tmp_path / "decks" / "04_canary"
    validation_summary = tmp_path / "storage" / "summary.json"
    _write_objective_evidence_fixture(deck_dir, validation_summary)

    def fake_extract_frame(video_path: Path, timestamp: float, output_path: Path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("frame", encoding="utf-8")
        return True

    def fake_signalstats(path: Path) -> dict[str, float]:
        yavg = 0.5 if path.name.endswith("_tail.png") else 45.0
        return {"YMIN": 0.0, "YAVG": yavg, "YMAX": 60.0}

    monkeypatch.setattr(module, "_ffprobe_duration", lambda path: 2501.0)
    monkeypatch.setattr(module, "_extract_frame", fake_extract_frame)
    monkeypatch.setattr(module, "_signalstats", fake_signalstats)
    monkeypatch.setattr(module, "_make_contact_sheet", lambda frame_paths, output_path: True)

    report = module.build_evidence(
        deck_dir=deck_dir,
        validation_summary=validation_summary,
        frames_dir=tmp_path / "frames",
    )

    assert report["status"] == "failed"
    assert "single_presenter:video_duration_over_40_minutes" in report["blockers"]
    assert "notebooklm:video_duration_over_40_minutes" in report["blockers"]
    assert "single_presenter:tail_frame_black" in report["blockers"]
    assert "notebooklm:tail_frame_black" in report["blockers"]


def make_presentation_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "conferences" / "demo2026"
    presentation_dir = workspace / "decks" / "01_demo_topic"
    presentation_dir.mkdir(parents=True)
    (presentation_dir / "presentation_blueprint.json").write_text(
        (BLUEPRINT_FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (presentation_dir / "blueprint_review.json").write_text(
        (BLUEPRINT_FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (presentation_dir / "5_draft_slides_with_images.md").write_text(
        "## Slide 1: Demo\n\n- Point\n",
        encoding="utf-8",
    )
    (presentation_dir / "6_presenter_script.md").write_text(
        "### [SLIDE 1] Demo\n\n**SAY:**\nDemo narration.\n",
        encoding="utf-8",
    )
    (presentation_dir / "presentation.html").write_text("<html></html>\n", encoding="utf-8")
    (presentation_dir / "presentation_slide_mapping.json").write_text(
        '{"1": 0}\n',
        encoding="utf-8",
    )
    return workspace, presentation_dir


def test_md_to_revealjs_accepts_explicit_manifest_paths(tmp_path: Path):
    module = load_presentation_script("md_to_revealjs.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    slides_md = presentation_dir / "5_draft_slides_with_images.md"
    output_html = presentation_dir / "presentation.html"
    slide_mapping = presentation_dir / "presentation_slide_mapping.json"

    args = module.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--slides-md",
            str(slides_md),
            "--output-html",
            str(output_html),
            "--slide-mapping",
            str(slide_mapping),
        ]
    )

    assert Path(args.workspace).resolve() == workspace.resolve()
    assert args.presentation_id == "01_demo_topic"
    assert args.presentation_type == "topic-deep-dive"
    assert Path(args.slides_md).resolve() == slides_md.resolve()
    assert Path(args.output_html).resolve() == output_html.resolve()
    assert Path(args.slide_mapping).resolve() == slide_mapping.resolve()


def test_audit_presentation_links_allows_web_and_drive_links(tmp_path: Path):
    module = load_presentation_script("audit_presentation_links.py")
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation.html").write_text(
        "<a href='https://openreview.net/forum?id=abc'>Paper</a>"
        "<a href='https://drive.google.com/file/d/demo'>Source</a>",
        encoding="utf-8",
    )
    (deck_dir / "presentation.pdf").write_bytes(
        b"%PDF-1.4\n/URI (https://drive.google.com/file/d/demo)\n%%EOF\n"
    )

    payload = module.audit_workspace(
        workspace=workspace,
        decks_dir=workspace / "decks",
        conference_slug="demo2026",
    )

    assert payload["status"] == "passed"
    assert payload["issue_count"] == 0


def test_audit_presentation_links_blocks_local_html_and_pdf_links(tmp_path: Path):
    module = load_presentation_script("audit_presentation_links.py")
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "executive_summary"
    deck_dir.mkdir(parents=True)
    (deck_dir / "presentation.html").write_text(
        "<a href='../analysis/executive_summary.md'>Local source</a>",
        encoding="utf-8",
    )
    (deck_dir / "presentation.pdf").write_bytes(
        b"%PDF-1.4\n/URI (file:///Users/demo/project/analysis/executive_summary.md)\n%%EOF\n"
    )

    payload = module.audit_workspace(
        workspace=workspace,
        decks_dir=workspace / "decks",
        conference_slug="demo2026",
    )

    assert payload["status"] == "failed"
    assert {
        (issue["artifact"], issue["reason"])
        for issue in payload["issues"]
    } == {
        ("decks/executive_summary/presentation.html", "local_analysis_reference"),
        ("decks/executive_summary/presentation.pdf", "file_uri"),
    }


def test_executive_overview_source_link_contract_blocks_missing_drive_map():
    module = load_presentation_script("generate_executive_overview_decks.py")

    with pytest.raises(module.SourceLinkContractError, match="source_link_missing"):
        module.source_url_for(
            "analysis/executive_summary.md",
            {},
            require_external=True,
        )

    assert (
        module.source_url_for(
            "analysis/executive_summary.md",
            {},
            require_external=False,
        )
        == "../analysis/executive_summary.md"
    )


def test_executive_overview_source_link_contract_uses_drive_map():
    module = load_presentation_script("generate_executive_overview_decks.py")
    source_links = {
        "analysis/executive_summary.md": "https://drive.google.com/file/d/demo-id"
    }

    assert module.source_url_for(
        "../analysis/executive_summary.md",
        source_links,
        require_external=True,
    ) == "https://drive.google.com/file/d/demo-id"


def test_executive_overview_generation_writes_structured_render_contracts(tmp_path: Path):
    module = load_presentation_script("generate_executive_overview_decks.py")
    workspace = tmp_path / "iclr2026"
    analysis_dir = workspace / "analysis"
    decks_dir = workspace / "decks"
    analysis_dir.mkdir(parents=True)
    decks_dir.mkdir()
    (workspace / "config.yaml").write_text(
        "conference:\n"
        '  name: "TestConf 2026"\n'
        "urls:\n"
        '  virtual: "https://testconf.cc/virtual/2026"\n'
        '  papers: "https://testconf.cc/virtual/2026/papers.html"\n'
        '  workshops: "https://testconf.cc/virtual/2026/events/workshop"\n',
        encoding="utf-8",
    )
    topic_ids = [
        "01_reasoning_post_training",
        "02_agentic_systems",
        "03_evaluation_benchmarks_data",
        "04_safety_alignment_privacy",
        "05_generative_models",
        "06_multimodal_spatial_vlm",
        "07_ai_for_science",
        "08_theory_optimization_structure",
        "09_efficiency_systems_architectures",
        "10_interpretability_representation",
        "11_rl_robotics_world_models",
        "12_open_science_society_post_agi",
    ]
    manifest = {
        "topics": [
            {
                "id": topic_id,
                "title": topic_id.replace("_", " ").title(),
                "markdown_path": f"analysis/{topic_id}.md",
            }
            for topic_id in topic_ids
        ]
    }
    analysis_manifest = analysis_dir / "analysis_manifest.json"
    analysis_manifest.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    executive_summary = analysis_dir / "executive_summary.md"
    executive_summary.write_text(
        "\n".join(
            f"{index}) Signal {index}\n- Signal {index} changes the leadership operating model."
            for index in range(1, 9)
        )
        + "\n",
        encoding="utf-8",
    )
    trend_report = analysis_dir / "trend_report.md"
    trend_report.write_text("# Trend report\nConference trend synthesis.\n", encoding="utf-8")
    source_links = {
        "analysis/executive_summary.md": "https://drive.google.com/file/d/executive-summary",
        "analysis/trend_report.md": "https://drive.google.com/file/d/trend-report",
        **{
            f"analysis/{topic_id}.md": f"https://drive.google.com/file/d/{topic_id}"
            for topic_id in topic_ids
        },
    }
    source_map = decks_dir / "executive_source_link_map.json"
    source_map.write_text(json.dumps({"source_links": source_links}) + "\n", encoding="utf-8")

    result = module.generate(
        workspace=workspace,
        analysis_manifest=analysis_manifest,
        executive_summary=executive_summary,
        trend_report=trend_report,
        short_id="executive_summary_short",
        long_id="executive_summary_long",
        source_link_map=source_map,
        require_source_link_map=True,
    )

    assert result["status"] == "succeeded"
    assert result["structured_render_contract"]["status"] == "passed"
    for deck_id, expected_slides in (
        # Generic executive decks: short = title + events bookend + orientation
        # + 8 synthesis items + leadership + further reading; long adds topic
        # maps and source-backbone appendix slides (fixture has no surprises/
        # risks/trend sections).
        ("executive_summary_short", 13),
        ("executive_summary_long", 18),
    ):
        deck_dir = workspace / "decks" / deck_id
        assert not any((deck_dir / filename).exists() for filename in module.CHECKPOINT_FILES)
        for filename in (
            "presentation_frame.json",
            "presentation_blueprint.json",
            "presentation_deck.json",
            "source_binding_report.json",
            "visual_plan.json",
            "asset_inventory.json",
            "delivery_single_speaker.json",
            "delivery_notebooklm.json",
            "render_spec.json",
        ):
            assert (deck_dir / filename).is_file()
        render_spec = json.loads((deck_dir / "render_spec.json").read_text(encoding="utf-8"))
        deck_contract = json.loads((deck_dir / "presentation_deck.json").read_text(encoding="utf-8"))
        duration_contract = json.loads((deck_dir / "duration_contract.json").read_text(encoding="utf-8"))
        expected_duration = 900.0 if deck_id.endswith("_short") else 1800.0
        assert duration_contract["status"] == "passed"
        assert duration_contract["target_duration_seconds"] == expected_duration
        assert render_spec["delivery"]["single_speaker"]["total_duration_seconds"] == expected_duration
        assert render_spec["delivery"]["notebooklm"]["total_duration_seconds"] == expected_duration
        assert deck_contract["slides"][0]["timing_beats"][0]["seconds"] == 30.0
        assert render_spec["presentation_type"] == "executive-summary"
        assert len(render_spec["slides"]) == expected_slides
        assert render_spec["input_hashes"]["presentation_deck.json"]
        assert render_spec["input_hashes"]["source_binding_report.json"]
        assert render_spec["input_hashes"]["duration_contract.json"]
        assert render_spec["slides"][1]["slide_kind"] == "events"
        assert render_spec["slides"][-1]["slide_kind"] == "resources"
        assert deck_contract["slides"][1]["slide_kind"] == "events"
        assert deck_contract["slides"][-1]["slide_kind"] == "resources"
        point_count = next(
            item["bookend_contract"]["point_count"]
            for item in result["presentations"]
            if item["presentation_id"] == deck_id
        )
        assert len(render_spec["slides"][1]["events"]) <= point_count
        assert len(render_spec["slides"][1]["papers"]) >= point_count
        final_resources = render_spec["slides"][-1]["resources"]
        assert sum(1 for item in final_resources if item["kind"] == "topic_presentation") == 12
        assert sum(1 for item in final_resources if item["kind"] == "analysis_report") == 12
        assert result["presentation_link_contract"]["mode"] == "workspace-local"


def test_executive_presenter_say_avoids_bullet_stitch_templates():
    module = load_presentation_script("generate_executive_overview_decks.py")
    slide = module.SlideSpec(
        title="The Conference Moved To Systems",
        bullets=(
            "Treat frontier progress as model plus tools, memory, verifiers, evaluation, and governance.",
            "Measure operational readiness instead of relying on isolated benchmark improvements.",
            "Prioritize reliability, safety, and resource governance as first-order research levers.",
        ),
        refs=(),
        figure="system_stack",
        notes="Open with the central thesis. The conference signal is less about single model wins and more about composed systems.",
        narrative="Multiple topic reports converge on the stack as the new threat surface and investment unit.",
    )
    next_slide = module.SlideSpec(
        title="Eight Signals Drive The Story",
        bullets=(),
        refs=(),
        figure="radial_signals",
        notes="Use this as the orientation slide.",
    )

    say = module._compose_presenter_say(slide, index=1, total=8)
    transition = module._transition_to_next(slide, next_slide, index=1)

    assert "That connects directly to" not in say
    assert "The operational test is" not in say
    assert "Teams should treat frontier progress" in say
    assert "The release standard should be to measure operational readiness" in say
    assert "That sets up" not in transition


def test_executive_transition_pool_avoids_short_repeated_rotation():
    module = load_presentation_script("generate_executive_overview_decks.py")
    current = module.SlideSpec(
        title="Current",
        bullets=(),
        refs=(),
        figure="generic",
        notes="Current slide.",
    )
    transitions = [
        module._transition_to_next(
            current,
            module.SlideSpec(
                title=f"Slide {index}",
                bullets=(),
                refs=(),
                figure="generic",
                notes="Next slide.",
            ),
            index=index,
        )
        for index in range(1, 15)
    ]
    normalized = [re.sub(r"Slide \d+", "X", transition) for transition in transitions]

    assert len(set(normalized)) == len(normalized)
    assert not any("With that lens in place" in transition for transition in transitions)


def test_md_to_revealjs_preserves_figure_after_metadata_comment() -> None:
    module = load_presentation_script("md_to_revealjs.py")

    slides = module.parse_slides(
        """## Slide 1: Demo

- Point

<!--
SLIDE_TYPE: content
SPEAKER_NOTES: narration
-->

![Generated figure](assets/generated/slide_001.svg)
"""
    )

    assert len(slides) == 1
    assert "![Generated figure](assets/generated/slide_001.svg)" in slides[0]["content"]
    assert "SPEAKER_NOTES" not in slides[0]["content"]


def test_md_to_revealjs_keeps_inline_figure_when_zoom_disabled(tmp_path: Path) -> None:
    module = load_presentation_script("md_to_revealjs.py")
    generated_dir = tmp_path / "assets" / "generated"
    generated_dir.mkdir(parents=True)
    (generated_dir / "slide_001.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="80">'
        '<rect width="120" height="80" fill="#123456"/></svg>',
        encoding="utf-8",
    )

    slides = module.parse_slides(
        """## Slide 1: Demo

- Point

<!--
SLIDE_TYPE: content
FIGURE_ZOOM: no
SPEAKER_NOTES: narration
-->

![Generated figure](assets/generated/slide_001.svg)
"""
    )

    processed = module.process_content(slides[0]["content"], tmp_path)
    html = module.slide_to_html(
        {**slides[0], "content": processed},
        tmp_path,
        assets={"bug": "data:image/png;base64,AAA"},
    )

    assert 'class="slide-content-with-figure"' in html
    assert 'class="figure-float"' in html
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in html
    assert 'class="slide-figure-zoom"' not in html


def test_md_to_revealjs_reference_labels_use_explicit_ellipsis() -> None:
    module = load_presentation_script("md_to_revealjs.py")

    label = module._compact_reference_label("Combinatorial Brain-Inspired Reasoning")

    assert label == "Combinatorial B..."


def test_md_to_revealjs_reference_chips_use_readable_visible_names() -> None:
    module = load_presentation_script("md_to_revealjs.py")

    html = module._reference_chips_html(
        "[Combinatorial Brain-Inspired Reasoning](https://example.com/a), "
        "[Theoretical Analysis of RL](https://example.com/b)"
    )

    assert ">Combinatorial B...<" in html
    assert ">Theoretical Ana...<" in html
    assert "Combinatorial Brain-Inspired Reasoning" in html
    assert "Theoretical Analysis of RL" in html
    assert ">Ref 1<" not in html
    assert ">Ref 2<" not in html
    assert ">Theoretical Analysis of RL<" not in html


def test_md_to_revealjs_strips_internal_reference_keys() -> None:
    module = load_presentation_script("md_to_revealjs.py")

    content = module.process_content(
        "- Scaling reveals linear structure (paper_abstract:6jlc8m38gj)\n"
        "- Keep audience citation [source:internal_chunk_12]",
        Path("."),
    )

    assert "paper_abstract:" not in content
    assert "source:internal_chunk_12" not in content
    assert "Scaling reveals linear structure" in content


def test_md_to_revealjs_rejects_presentations_output_in_workspace_mode(tmp_path: Path):
    module = load_presentation_script("md_to_revealjs.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    legacy_output = workspace / "presentations" / "01_demo_topic" / "presentation.html"

    with pytest.raises(SystemExit) as excinfo:
        module.parse_args(
            [
                "--workspace",
                str(workspace),
                "--presentation-id",
                "01_demo_topic",
                "--presentation-type",
                "topic-deep-dive",
                "--slides-md",
                str(presentation_dir / "5_draft_slides_with_images.md"),
                "--output-html",
                str(legacy_output),
                "--slide-mapping",
                str(presentation_dir / "presentation_slide_mapping.json"),
            ]
        )

    assert str(excinfo.value) == MIGRATION_NOTE


def test_presenter_video_dry_run_prints_plan_without_media_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    module = load_presentation_script("build_presenter_video.py")
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    build_dir = presentation_dir / "video_build"
    output_video = presentation_dir / "presentation_video.mp4"
    side_effect_calls: list[str] = []

    for function_name in (
        "synthesize_audio",
        "capture_slides",
        "concat_audio",
        "build_slides_track",
        "mux_video",
        "run_cmd",
    ):
        monkeypatch.setattr(
            module,
            function_name,
            lambda *args, _name=function_name, **kwargs: side_effect_calls.append(_name),
        )

    result = module.main(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(presentation_dir / "6_presenter_script.md"),
            "--slides-html",
            str(presentation_dir / "presentation.html"),
            "--slide-mapping",
            str(presentation_dir / "presentation_slide_mapping.json"),
            "--build-dir",
            str(build_dir),
            "--output",
            str(output_video),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert side_effect_calls == []
    assert set(payload) == {
        "build_dir",
        "dry_run",
        "output",
        "presentation_id",
        "presentation_type",
        "provider_requirements",
        "blueprint",
        "blueprint_review",
        "blueprint_narration_beat_contract",
        "notebooklm_timing_map",
        "script_md",
        "slide_mapping",
        "slides_html",
    }
    assert payload["dry_run"] is True
    assert payload["script_md"] == str(presentation_dir / "6_presenter_script.md")
    assert payload["slides_html"] == str(presentation_dir / "presentation.html")
    assert payload["slide_mapping"] == str(presentation_dir / "presentation_slide_mapping.json")
    assert payload["build_dir"] == str(build_dir)
    assert payload["output"] == str(output_video)
    assert payload["provider_requirements"] == [
        "tool:ffmpeg",
        "tool:playwright",
        "env:GEMINI_API_KEY|GOOGLE_API_KEY",
    ]


def test_presenter_video_rejects_script_missing_title_slide(tmp_path: Path):
    module = load_presentation_script("build_presenter_video.py")
    mapping = tmp_path / "presentation_slide_mapping.json"
    mapping.write_text('{"0": 0, "1": 1}\n', encoding="utf-8")
    entries = [
        module.SlideNarration(
            index=1,
            slide_number=1,
            title="First Topic",
            say="Topic narration.",
            context=[],
            transition=None,
            narration="Topic narration.",
        )
    ]

    with pytest.raises(SystemExit, match=r"missing narration for mapped slide\(s\): 0"):
        module.validate_slide_mapping_coverage(entries=entries, slide_mapping=mapping)


def test_presenter_script_json_uses_only_spoken_say_and_transition(tmp_path: Path):
    module = load_presentation_script(
        "build_presenter_video.py",
        module_name="phase24_presenter_tts_contract",
    )
    script_md = tmp_path / "6_presenter_script.md"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] Demo",
                "",
                "**SAY:**",
                "> The spoken line explains the result.",
                "",
                "**CONTEXT** (not spoken -- for presenter reference):",
                "- Paper: Example Study (Ada, 2026) [arXiv]",
                "",
                "**REFS** (spoken-friendly citations for hosts to name-drop in the podcast dialogue):",
                "- Example Study (Ada, 2026)",
                "",
                '**TRANSITION:** "That sets up the reliability test."',
            ]
        ),
        encoding="utf-8",
    )

    entries = module.parse_presenter_script(script_md, max_context_items=4, context_prefix="For context")
    script_json = tmp_path / "script.json"
    module.write_script_json(entries, script_json, title="Demo")

    payload_text = script_json.read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    assert payload["segments"] == [
        {
            "speaker": "Presenter",
            "text": "The spoken line explains the result. That sets up the reliability test.",
        }
    ]
    for forbidden in ("**CONTEXT**", "**REFS**", "not spoken", "[arXiv]", "name-drop in the podcast"):
        assert forbidden not in payload_text
    assert payload["metadata"]["slides"][0]["context_count"] == 1


def test_presenter_video_stop_after_script_json_does_not_touch_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_presenter_video.py",
        module_name="phase24_presenter_stop_after_json",
    )
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    build_dir = presentation_dir / "video_build"
    output_video = presentation_dir / "presentation_video.mp4"
    side_effect_calls: list[str] = []

    for function_name in (
        "synthesize_audio",
        "capture_slides",
        "concat_audio",
        "build_slides_track",
        "mux_video",
    ):
        monkeypatch.setattr(
            module,
            function_name,
            lambda *args, _name=function_name, **kwargs: side_effect_calls.append(_name),
        )

    result = module.main(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(presentation_dir / "6_presenter_script.md"),
            "--slides-html",
            str(presentation_dir / "presentation.html"),
            "--slide-mapping",
            str(presentation_dir / "presentation_slide_mapping.json"),
            "--blueprint",
            str(presentation_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(presentation_dir / "blueprint_review.json"),
            "--build-dir",
            str(build_dir),
            "--output",
            str(output_video),
            "--stop-after-script-json",
        ]
    )

    assert result == 0
    assert side_effect_calls == []
    assert (build_dir / "script.json").is_file()
    assert not (build_dir / "audio_segments").exists()


def _assert_quicktime_safe_mux_command(cmd: list[str]) -> None:
    assert cmd[0] == "ffmpeg"
    assert "fps=25,tpad=stop_mode=clone:stop_duration=2" in cmd
    assert "-shortest" in cmd
    assert "+faststart" in cmd
    assert "-pix_fmt" in cmd
    assert "yuv420p" in cmd
    assert "-c:v" in cmd
    codec_index = cmd.index("-c:v") + 1
    assert cmd[codec_index] == "libx264"
    assert not any(
        item == "-c:v" and index + 1 < len(cmd) and cmd[index + 1] == "copy"
        for index, item in enumerate(cmd)
    )


def test_presenter_video_mux_reencodes_sparse_slide_track_for_quicktime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_presenter_video.py",
        module_name="phase24_presenter_quicktime_mux",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run_cmd", lambda cmd, **_kwargs: commands.append(cmd))

    module.mux_video(
        tmp_path / "slides_track.mp4",
        tmp_path / "narration.mp3",
        tmp_path / "presentation_video.mp4",
    )

    assert len(commands) == 1
    _assert_quicktime_safe_mux_command(commands[0])


def test_notebooklm_video_mux_reencodes_sparse_slide_track_for_quicktime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_quicktime_mux",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(module, "run_cmd", lambda cmd, **_kwargs: commands.append(cmd))

    module.mux_video(
        tmp_path / "slides_track.mp4",
        tmp_path / "narration.mp3",
        tmp_path / "presentation_video_notebooklm.mp4",
    )

    assert len(commands) == 1
    _assert_quicktime_safe_mux_command(commands[0])


def test_presenter_video_rejects_compact_outline_without_say(tmp_path: Path):
    module = load_presentation_script(
        "build_presenter_video.py",
        module_name="phase24_presenter_outline_gate",
    )
    script_md = tmp_path / "6_presenter_script.md"
    script_md.write_text(
        "\n".join(
            [
                "# Presenter Script: Executive Summary",
                "",
                "## Slide 1: The Conference Moved To Systems",
                "",
                "Open with the central thesis.",
                "",
                "Key points:",
                "- Treat frontier progress as systems.",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="without \\*\\*SAY:\\*\\* blocks"):
        module.parse_presenter_script(script_md, max_context_items=4, context_prefix="For context")


def test_notebooklm_duration_budget_accepts_single_host_script_json(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_budget",
    )
    script_json = tmp_path / "script.json"
    script_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker": "Presenter", "text": "alpha beta"},
                    {"speaker": "Presenter", "text": "gamma"},
                ]
            }
        ),
        encoding="utf-8",
    )

    budgets = module.load_char_budget_by_slide(script_json, scale=1.0)

    assert budgets == {1: 80, 2: 80}


def test_notebooklm_markdown_can_be_refreshed_from_script_json(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_markdown_refresh",
    )
    source_script = tmp_path / "6_presenter_script.md"
    source_script.write_text("### [SLIDE 1] Demo\n\n**SAY:**\nDemo.\n", encoding="utf-8")
    script_json = tmp_path / "script.json"
    script_json.write_text(
        json.dumps(
            {
                "hosts": {"host_a": "Iapetus", "host_b": "Erinome"},
                "source": {"script_md": str(source_script)},
                "segments": [
                    {
                        "index": 1,
                        "speaker": "Iapetus",
                        "text": "Actual generated dialogue.",
                        "slide_number": 1,
                        "slide_title": "Demo",
                    }
                ],
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 1,
                            "title": "Demo",
                            "time_line": None,
                            "transition": None,
                            "say": "Demo.",
                            "context": [],
                            "turns": [
                                {
                                    "speaker": "Iapetus",
                                    "text": "Actual generated dialogue.",
                                },
                                {
                                    "speaker": "Erinome",
                                    "text": "A grounded follow-up.",
                                },
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    stale_markdown = tmp_path / "6_presenter_script_notebooklm_2host.md"
    stale_markdown.write_text("Host A: Open with the central thesis.\n", encoding="utf-8")

    dialogues, host_a, host_b, loaded_source = module.load_dialogues_from_script_json(
        script_json
    )
    module.write_podcast_script_markdown(
        stale_markdown,
        source_script=loaded_source or source_script,
        host_a=host_a,
        host_b=host_b,
        generation_model="test-model",
        dialogues=dialogues,
    )
    module.validate_podcast_script_markdown(
        stale_markdown,
        host_a=host_a,
        host_b=host_b,
    )

    refreshed = stale_markdown.read_text(encoding="utf-8")
    assert "Host A: Open" not in refreshed
    assert "**Iapetus:** Actual generated dialogue." in refreshed
    assert "**Erinome:** A grounded follow-up." in refreshed


def test_notebooklm_stop_after_transform_refreshes_markdown_without_tts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_stop_after_transform",
    )
    source_script = tmp_path / "6_presenter_script.md"
    source_script.write_text("### [SLIDE 1] Demo\n\n**SAY:**\nDemo.\n", encoding="utf-8")
    slides_html = tmp_path / "presentation.html"
    slides_html.write_text("<html><body><section></section></body></html>\n", encoding="utf-8")
    build_dir = tmp_path / "video_build_notebooklm"
    build_dir.mkdir()
    script_json = build_dir / "script.json"
    script_json.write_text(
        json.dumps(
            {
                "hosts": {"host_a": "Iapetus", "host_b": "Erinome"},
                "source": {"script_md": str(source_script)},
                "metadata": {
                    "slides": [
                        {
                            "slide_number": 1,
                            "title": "Demo",
                            "time_line": None,
                            "transition": None,
                            "say": "Demo.",
                            "context": [],
                            "turns": [
                                {"speaker": "Iapetus", "text": "A clean narration line."},
                                {"speaker": "Erinome", "text": "A grounded closing observation."},
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    podcast_script_out = tmp_path / "6_presenter_script_notebooklm_2host.md"
    monkeypatch.setattr(
        module,
        "synthesize_audio_segments",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("TTS should not run")),
    )

    result = module.main(
        [
            "--script-md",
            str(source_script),
            "--slides-html",
            str(slides_html),
            "--build-dir",
            str(build_dir),
            "--podcast-script-out",
            str(podcast_script_out),
            "--skip-transform",
            "--stop-after-transform",
        ]
    )

    assert result == 0
    refreshed = podcast_script_out.read_text(encoding="utf-8")
    assert "**Iapetus:** A clean narration line." in refreshed
    assert "**Erinome:** A grounded closing observation." in refreshed
    assert not (build_dir / "audio_segments").exists()


def test_notebooklm_markdown_validation_rejects_stage_directions(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_markdown_gate",
    )
    script_md = tmp_path / "6_presenter_script_notebooklm_2host.md"
    script_md.write_text(
        "Host A: Open with the central thesis.\n"
        "Host B: Use this as the orientation slide.\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="missing host dialogue markers"):
        module.validate_podcast_script_markdown(
            script_md,
            host_a="Iapetus",
            host_b="Erinome",
        )


def test_notebooklm_rejects_script_missing_title_slide(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_title_missing",
    )
    mapping = tmp_path / "presentation_slide_mapping.json"
    mapping.write_text('{"0": 0, "1": 1}\n', encoding="utf-8")

    with pytest.raises(SystemExit, match=r"missing narration for mapped slide\(s\): 0"):
        module.validate_slide_mapping_coverage(
            slide_numbers={1},
            slide_mapping=mapping,
            source_label="Presenter script",
        )


def test_notebooklm_clear_slide_cache_removes_only_slide_pngs(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_slide_cache",
    )
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir()
    stale_slide = slides_dir / "slide_001.png"
    stale_zoom = slides_dir / "slide_002_zoom.png"
    keep_notes = slides_dir / "notes.txt"
    stale_slide.write_text("old", encoding="utf-8")
    stale_zoom.write_text("old", encoding="utf-8")
    keep_notes.write_text("keep", encoding="utf-8")

    removed = module.clear_slide_cache(slides_dir)

    assert removed == 2
    assert not stale_slide.exists()
    assert not stale_zoom.exists()
    assert keep_notes.exists()


def test_notebooklm_capture_wait_outlasts_reveal_slide_transition():
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_capture_wait",
    )

    args = module.parse_args([])

    assert args.wait_ms >= 2500


def test_notebooklm_parser_preserves_inline_transition_blocks(tmp_path: Path):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_inline_transition",
    )
    script_md = tmp_path / "6_presenter_script.md"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 1] First Topic",
                "",
                "**SAY:**",
                "First topic narration.",
                "",
                '**TRANSITION:** "Next, the deck turns to benchmark pressure."',
                "",
                "---",
                "",
                "### [SLIDE 2] Second Topic",
                "",
                "**SAY:** One-line narration also parses.",
            ]
        ),
        encoding="utf-8",
    )

    slides = module.parse_presenter_script(script_md, max_context_items=4)

    assert slides[0].transition == "Next, the deck turns to benchmark pressure."
    assert slides[1].say == "One-line narration also parses."


def test_notebooklm_podcastfy_prompt_requires_observational_endings():
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_prompt_endings",
    )
    slide = module.SlideSource(
        slide_number=1,
        title="Evidence Beats Hype",
        say="The strongest result is a measured benchmark drop after deployment.",
        context_items=["The paper reports the failure mode as verifier fragility."],
        transition="The next slide compares mitigation options.",
        figure_description=None,
        time_line=None,
        is_zoom=False,
        refs=None,
    )

    prompt = module.slide_prompt(
        slide,
        host_a="Iapetus",
        host_b="Erinome",
        conference_label="ICLR 2026",
        turns_per_slide=4,
        max_turn_chars=420,
        target_chars=None,
        char_budget_tolerance=0.12,
        prompt_style="podcastfy",
    )

    assert "Use questions sparingly" in prompt
    assert "Do not end with a question hook" in prompt
    assert "assertion, observation, implication, or bridge" in prompt
    assert "What should we pay closest attention to here?" in prompt
    assert '"Exactly."' in prompt
    assert '"Precisely."' in prompt


def test_notebooklm_slide_zero_alignment_feeds_timing_before_topic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_title_alignment",
    )
    slides_html = tmp_path / "presentation.html"
    slides_html.write_text(
        "\n".join(
            [
                "<section class='slide-title'></section>",
                "<section class='slide-content-with-figure'></section>",
                "<section class='slide-figure-zoom'></section>",
                "<section class='slide-content'></section>",
            ]
        ),
        encoding="utf-8",
    )
    script_md = tmp_path / "6_presenter_script.md"
    script_md.write_text(
        "\n".join(
            [
                "### [SLIDE 0] Deck Title",
                "",
                "**SAY:**",
                "Title narration.",
                "",
                "---",
                "",
                "### [SLIDE 1] First Topic",
                "",
                "**SAY:**",
                "First topic narration.",
                "",
                "---",
                "",
                "### [ZOOM 1] First Topic Detail",
                "",
                "**SAY:**",
                "Zoom narration.",
                "",
                "---",
                "",
                "### [SLIDE 2] Second Topic",
                "",
                "**SAY:**",
                "Second topic narration.",
            ]
        ),
        encoding="utf-8",
    )

    slides = module.parse_presenter_script(script_md, max_context_items=4)
    assert [slide.slide_number for slide in slides] == [
        0,
        1,
        module.ZOOM_SLIDE_OFFSET + 1,
        2,
    ]

    logical_to_rendered, strategy = module.build_logical_to_rendered_slide_map(
        logical_slide_numbers=[slide.slide_number for slide in slides],
        slides_html=slides_html,
        mode="auto",
        zoom_slide_class="slide-figure-zoom",
    )
    assert strategy == "zoom-aware(slide-figure-zoom)"
    assert logical_to_rendered == {
        0: [1],
        1: [2],
        module.ZOOM_SLIDE_OFFSET + 1: [3],
        2: [4],
    }

    title_slide = tmp_path / "slide_001.png"
    topic_slide = tmp_path / "slide_002.png"
    zoom_slide = tmp_path / "slide_003.png"
    second_slide = tmp_path / "slide_004.png"
    audio_paths = [
        tmp_path / "segment_001_01.mp3",
        tmp_path / "segment_002_01.mp3",
        tmp_path / "segment_003_01.mp3",
        tmp_path / "segment_004_01.mp3",
        tmp_path / "segment_005_01.mp3",
    ]
    for path in [title_slide, topic_slide, zoom_slide, second_slide, *audio_paths]:
        path.write_text("placeholder\n", encoding="utf-8")

    durations = {
        audio_paths[0]: 5.0,
        audio_paths[1]: 7.0,
        audio_paths[2]: 20.0,
        audio_paths[3]: 10.0,
        audio_paths[4]: 8.0,
    }
    monkeypatch.setattr(
        module,
        "probe_durations",
        lambda paths, *, workers: {path: durations[path] for path in paths},
    )

    segments = [
        module.Segment(index=1, slide_number=0, slide_title="Deck Title", speaker="A", text="Title a."),
        module.Segment(index=2, slide_number=0, slide_title="Deck Title", speaker="B", text="Title b."),
        module.Segment(index=3, slide_number=1, slide_title="First Topic", speaker="A", text="Topic."),
        module.Segment(
            index=4,
            slide_number=module.ZOOM_SLIDE_OFFSET + 1,
            slide_title="First Topic Detail",
            speaker="B",
            text="Zoom.",
        ),
        module.Segment(index=5, slide_number=2, slide_title="Second Topic", speaker="A", text="Second."),
    ]
    slides_concat = tmp_path / "slides_concat.txt"

    module.write_concat_files(
        segments=segments,
        slide_paths_by_logical_number={
            0: [title_slide],
            1: [topic_slide],
            module.ZOOM_SLIDE_OFFSET + 1: [zoom_slide],
            2: [second_slide],
        },
        audio_paths=audio_paths,
        audio_concat_path=tmp_path / "audio_concat.txt",
        slides_concat_path=slides_concat,
        duration_workers=1,
    )

    assert slides_concat.read_text(encoding="utf-8").splitlines() == [
        f"file '{title_slide.resolve().as_posix()}'",
        "duration 12.000000",
        f"file '{topic_slide.resolve().as_posix()}'",
        "duration 20.000000",
        f"file '{zoom_slide.resolve().as_posix()}'",
        "duration 10.000000",
        f"file '{second_slide.resolve().as_posix()}'",
        "duration 8.000000",
        f"file '{second_slide.resolve().as_posix()}'",
    ]


def test_notebooklm_video_infers_cached_speed_adjusted_audio_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="notebooklm_cached_audio_rate",
    )
    slide = tmp_path / "slide_001.png"
    audio_paths = [tmp_path / "segment_001_01.mp3", tmp_path / "segment_002_01.mp3"]
    narration_audio_raw = tmp_path / "narration_track_raw.m4a"
    narration_audio = tmp_path / "narration_track.m4a"
    for path in [slide, *audio_paths, narration_audio_raw, narration_audio]:
        path.write_text("placeholder\n", encoding="utf-8")

    durations = {
        audio_paths[0]: 60.0,
        audio_paths[1]: 60.0,
        narration_audio_raw: 120.0,
        narration_audio: 100.0,
    }
    monkeypatch.setattr(module, "probe_duration", lambda path: durations[path])
    monkeypatch.setattr(
        module,
        "probe_durations",
        lambda paths, *, workers: {path: durations[path] for path in paths},
    )

    inferred_rate = module.infer_reused_audio_speaking_rate(
        audio_paths=audio_paths,
        narration_audio=narration_audio,
        narration_audio_raw=narration_audio_raw,
        duration_workers=1,
    )
    assert inferred_rate == pytest.approx(1.2)

    slides_concat = tmp_path / "slides_concat.txt"
    module.write_concat_files(
        segments=[
            module.Segment(index=1, slide_number=1, slide_title="Topic", speaker="A", text="One."),
            module.Segment(index=2, slide_number=1, slide_title="Topic", speaker="B", text="Two."),
        ],
        slide_paths_by_logical_number={1: [slide]},
        audio_paths=audio_paths,
        audio_concat_path=tmp_path / "audio_concat.txt",
        slides_concat_path=slides_concat,
        duration_workers=1,
        speaking_rate=inferred_rate,
    )

    assert slides_concat.read_text(encoding="utf-8").splitlines() == [
        f"file '{slide.resolve().as_posix()}'",
        "duration 100.000000",
        f"file '{slide.resolve().as_posix()}'",
    ]


def test_presenter_script_rejects_incomplete_slide_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script(
        "generate_presenter_script.py",
        module_name="phase24_presenter_script_incomplete",
    )
    _workspace, presentation_dir = make_presentation_workspace(tmp_path)
    slides_md = presentation_dir / "5_draft_slides_with_images.md"
    slides_md.write_text(
        "\n".join(
            [
                "## Slide 1: First",
                "",
                "- First point",
                "",
                "---",
                "",
                "## Slide 2: Second",
                "",
                "- Second point",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (presentation_dir / "presentation_slide_mapping.json").write_text(
        '{"1": 0, "2": 1}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("PRESENTATION_GEMINI_SCRIPT_CONTENT_RETRIES", "1")
    monkeypatch.setattr(
        module,
        "call_gemini",
        lambda *args, **kwargs: "### [SLIDE 1] First\n\n**SAY:**\n> Only one slide.\n",
    )

    with pytest.raises(RuntimeError, match=r"missing slides \[2\]"):
        module.process_topic(
            "01_demo_topic",
            skip_dedup=True,
            topic_dir=presentation_dir,
            input_path=slides_md,
            slide_mapping_path=presentation_dir / "presentation_slide_mapping.json",
            output_path=presentation_dir / "6_presenter_script.md",
            output_notebooklm_path=presentation_dir / "6_presenter_script_notebooklm.md",
            topic_display_name="Demo Topic",
            conference_label=None,
        )


def test_presenter_video_concat_advances_audio_after_zoom_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = load_presentation_script("build_presenter_video.py")
    main_slide = tmp_path / "slide_001_main.png"
    zoom_slide = tmp_path / "slide_001_zoom.png"
    next_slide = tmp_path / "slide_002.png"
    first_audio = tmp_path / "segment_001_01.mp3"
    second_audio = tmp_path / "segment_002_01.mp3"
    for path in (main_slide, zoom_slide, next_slide, first_audio, second_audio):
        path.write_text("placeholder\n", encoding="utf-8")

    durations = {
        first_audio: 10.0,
        second_audio: 20.0,
    }
    monkeypatch.setattr(module, "probe_duration", lambda path: durations[path])

    slides_concat = tmp_path / "slides_concat.txt"
    module.write_concat_files(
        slide_items=[
            (main_slide, 0.7),
            (zoom_slide, 0.3),
            (next_slide, 1.0),
        ],
        audio_paths=[first_audio, second_audio],
        audio_concat_path=tmp_path / "audio_concat.txt",
        slides_concat_path=slides_concat,
    )

    assert slides_concat.read_text(encoding="utf-8").splitlines() == [
        f"file '{main_slide.resolve().as_posix()}'",
        "duration 7.000000",
        f"file '{zoom_slide.resolve().as_posix()}'",
        "duration 3.000000",
        f"file '{next_slide.resolve().as_posix()}'",
        "duration 20.000000",
        f"file '{next_slide.resolve().as_posix()}'",
    ]


def test_notebooklm_video_dry_run_does_not_import_google_genai(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    workspace, presentation_dir = make_presentation_workspace(tmp_path)
    build_dir = presentation_dir / "video_build_notebooklm"
    output_video = presentation_dir / "presentation_video_notebooklm.mp4"
    google_imports: list[str] = []
    original_import = builtins.__import__

    def track_provider_imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google" or name.startswith("google."):
            google_imports.append(name)
            raise AssertionError("NotebookLM dry-run imported google provider modules")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_provider_imports)
    module = load_presentation_script(
        "build_notebooklm_style_presenter_video.py",
        module_name="phase24_notebooklm_video",
    )

    result = module.main(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(presentation_dir / "6_presenter_script.md"),
            "--slides-html",
            str(presentation_dir / "presentation.html"),
            "--slide-mapping",
            str(presentation_dir / "presentation_slide_mapping.json"),
            "--build-dir",
            str(build_dir),
            "--output",
            str(output_video),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert google_imports == []
    assert set(payload) == {
        "build_dir",
        "conference_label",
        "dry_run",
        "host_a",
        "host_b",
        "output",
        "presentation_id",
        "presentation_type",
        "provider_requirements",
        "blueprint",
        "blueprint_review",
        "blueprint_narration_beat_contract",
        "notebooklm_timing_map",
        "script_md",
        "slide_mapping",
        "slides_html",
    }
    assert payload["dry_run"] is True
    assert payload["conference_label"] == "DEMO 2026"
    assert payload["script_md"] == str(presentation_dir / "6_presenter_script.md")
    assert payload["slides_html"] == str(presentation_dir / "presentation.html")
    assert payload["slide_mapping"] == str(presentation_dir / "presentation_slide_mapping.json")
    assert payload["build_dir"] == str(build_dir)
    assert payload["output"] == str(output_video)
    assert payload["host_a"] == "Iapetus"
    assert payload["host_b"] == "Erinome"
    assert payload["provider_requirements"] == [
        "tool:ffmpeg",
        "tool:playwright",
        "env:GEMINI_API_KEY|GOOGLE_API_KEY",
    ]


def test_tts_segment_watchdog_interrupts_stuck_provider_call() -> None:
    pytest.importorskip("google.genai")
    module = load_repo_script(
        "synthesize_audio_from_script.py",
        module_name="phase24_tts_watchdog",
    )

    started = time.monotonic()
    with pytest.raises(module.SegmentSynthesisTimeout, match="segment 9"):
        with module.segment_timeout(1, segment_index=9):
            time.sleep(10)

    assert time.monotonic() - started < 3


def test_tts_manifest_reuses_multi_speaker_alias_fingerprint(tmp_path: Path) -> None:
    module = load_repo_script(
        "synthesize_audio_from_script.py",
        module_name="phase24_tts_manifest_alias",
    )
    audio_path = tmp_path / "segment_001_01.mp3"
    audio_path.write_bytes(b"not-real-audio-but-cache-sized")
    text = module.prepare_text_for_tts("Iapetus: Hello. Erinome: Grounded reply.")
    speaker_voices = [("Iapetus", "Iapetus"), ("Erinome", "Erinome")]
    canonical = module.segment_fingerprint(
        text,
        audio_model=module.DEFAULT_AUDIO_MODEL,
        voice="Iapetus",
        speaker_voices=speaker_voices,
    )
    legacy = module.segment_fingerprint(
        text,
        audio_model=module.DEFAULT_AUDIO_MODEL,
        voice=module.DEFAULT_VOICE,
        speaker_voices=speaker_voices,
    )

    manifest = {
        "1": {
            "filename": audio_path.name,
            "fingerprint_sha256": legacy,
            "byte_count": audio_path.stat().st_size,
        }
    }

    assert module.reusable_segment_audio(
        audio_path,
        manifest_segments=manifest,
        segment_index=1,
        fingerprint=canonical,
        fingerprint_aliases=[legacy],
    )


def test_presenter_script_watchdog_interrupts_stuck_provider_call() -> None:
    module = load_presentation_script(
        "generate_presenter_script.py",
        module_name="phase24_script_watchdog",
    )

    started = time.monotonic()
    with pytest.raises(module.GeminiScriptTimeout):
        with module.gemini_script_timeout(1):
            time.sleep(10)

    assert time.monotonic() - started < 3


def test_presenter_script_skip_dedup_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_presentation_script(
        "generate_presenter_script.py",
        module_name="phase24_script_skip_dedup_env",
    )

    monkeypatch.setenv("PRESENTATION_SKIP_DEDUP", "yes")

    assert module._env_bool("PRESENTATION_SKIP_DEDUP") is True


def test_phase01_human_receipt_topic4_approval_does_not_unlock_bulk_rollout(tmp_path: Path) -> None:
    module = load_presentation_script(
        "record_human_review_receipt.py",
        module_name="phase01_record_human_review_receipt",
    )
    deck_dir = tmp_path / "04_safety_alignment_privacy"
    deck_dir.mkdir(parents=True)
    receipt = module.build_receipt(
        deck_dir=deck_dir,
        decision="approved_for_topic4_canary",
        reviewer="test_operator",
        notes=[],
        checklist=[{"item": "Slides look good", "status": "pass"}],
        source_review_md=None,
        canary_review_packet=None,
        objective_evidence_report=None,
    )
    assert receipt.get("approved_for_bulk_rollout") is not True, (
        "approved_for_topic4_canary must not set approved_for_bulk_rollout to true"
    )
    assert receipt["decision"] == "approved_for_topic4_canary"
