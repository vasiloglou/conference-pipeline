from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "presentations"
    / "apply_v22_editorial_contract.py"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "04_safety_alignment_privacy"
    phase_dir = tmp_path / "phase"
    deck_dir.mkdir(parents=True)
    phase_dir.mkdir()

    contribution = {
        "visible_label": "OpenAgentSafety - real-world agent safety framework",
        "paper_name": "OpenAgentSafety: A Comprehensive Framework For Evaluating Real-World AI Agent Safety",
        "authors": ["Sanidhya Vijayvargiya", "Maarten Sap"],
        "speakers": [],
        "system": "OpenAgentSafety",
        "dataset": "OpenAgentSafety 350+ multi-turn multi-user tasks",
        "role": "primary",
        "source_family": "OpenReview paper",
        "source_id": "paper_abstract:xggSxCFQbA",
        "paper_url": "https://openreview.net/forum?id=xggSxCFQbA",
        "openreview_url": "https://openreview.net/forum?id=xggSxCFQbA",
    }
    source_board = {
        "status": "pass",
        "payload": {
            "topic_id": "04_safety_alignment_privacy",
            "clusters": [
                {
                    "cluster_id": "agent-system-safety",
                    "title": "Agent and System Safety",
                    "story_contribution": "Agents must be evaluated as systems.",
                    "contributions": [contribution],
                }
            ],
        },
    }
    material_clusters = {
        "clusters": [
            {
                "cluster_id": "agent-system-safety",
                "title": "Agent and System Safety",
                "story_contribution": "Agents must be evaluated as systems.",
                "contributions": [contribution],
            }
        ]
    }
    section_spine = {
        "sections": [
            {
                "section_id": "agent-system-safety",
                "section_thesis": "Safety is now a deployed-system property.",
                "source_clusters": ["agent-system-safety"],
                "slide_roles": ["evidence"],
                "reference_set": ["paper_abstract:xggSxCFQbA"],
                "transition_purpose": "Moves from model-only refusal to deployed-system evidence.",
            }
        ]
    }
    storyboard_rows = {
        "rows": [
            {
                "slide_id": "s01",
                "viewer_facing_claim": "Agent safety has become a deployed-system evaluation problem.",
                "proof_object": "OpenAgentSafety benchmark contribution with named authors and dataset.",
                "visual_concept": "System boundary flow from agent task to tool use to safety outcome.",
                "named_source_contribution": (
                    "OpenAgentSafety [paper_abstract:xggSxCFQbA; "
                    "https://openreview.net/forum?id=xggSxCFQbA]"
                ),
                "narration_beat": "OpenAgentSafety makes the agent evaluation boundary concrete.",
                "reason_to_exist": "Grounds the systems-safety section in a named benchmark.",
            }
        ]
    }
    storyboard_review = {
        "status": "pass",
        "storyboard_sha256": "a" * 64,
        "approval": {"status": "approved_for_render_planning", "row_count": 1},
        "validator_summary": {"blockers": [], "rows_checked": 1},
    }
    cross_ai = {
        "reviewers": ["claude", "gemini", "codex"],
        "agreements": [],
        "disagreements": [],
        "accepted_changes": [],
        "required_repairs": [],
        "unresolved_blockers": [],
        "external_tool_blockers": [],
        "storyboard_review_sha256": "a" * 64,
        "gate_status": "approved_for_render_planning",
    }

    paths = {
        "workspace": workspace,
        "deck_dir": deck_dir,
        "blueprint": deck_dir / "presentation_blueprint.json",
        "brand": phase_dir / "brand.json",
        "source": phase_dir / "source.json",
        "clusters": phase_dir / "clusters.json",
        "sections": phase_dir / "sections.json",
        "rows": phase_dir / "rows.json",
        "visual_direction": phase_dir / "visual.md",
        "readthrough": phase_dir / "readthrough.md",
        "storyboard_review": phase_dir / "storyboard_review.json",
        "cross_ai": phase_dir / "cross_ai.json",
        "deck_out": deck_dir / "presentation_deck.json",
        "source_out": deck_dir / "source_binding_report.json",
        "section_out": deck_dir / "section_binding_report.json",
        "visual_out": deck_dir / "visual_plan.json",
        "inventory_out": deck_dir / "asset_inventory.json",
        "single_out": deck_dir / "delivery_single_speaker.json",
        "notebook_out": deck_dir / "delivery_notebooklm.json",
        "md_out": phase_dir / "05-EDITORIAL-COMPILE.md",
        "json_out": phase_dir / "05-EDITORIAL-COMPILE.json",
    }
    _write_json(
        paths["blueprint"],
        {
            "schema_version": 1,
            "presentation_id": "04_safety_alignment_privacy",
            "target_slide_count": 1,
            "source_plan": [],
        },
    )
    _write_json(paths["brand"], {"status": "blocked", "asset_sha256": "b" * 64})
    _write_json(paths["source"], source_board)
    _write_json(paths["clusters"], material_clusters)
    _write_json(paths["sections"], section_spine)
    _write_json(paths["rows"], storyboard_rows)
    _write_text(paths["visual_direction"], "Visual direction")
    _write_text(paths["readthrough"], "Narration readthrough")
    _write_json(paths["storyboard_review"], storyboard_review)
    _write_json(paths["cross_ai"], cross_ai)
    return paths


def _run(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--workspace",
            str(paths["workspace"]),
            "--presentation-id",
            "04_safety_alignment_privacy",
            "--presentation-type",
            "topic-deep-dive",
            "--topic",
            "04_safety_alignment_privacy",
            "--deck-dir",
            str(paths["deck_dir"]),
            "--blueprint",
            str(paths["blueprint"]),
            "--brand-lock-json",
            str(paths["brand"]),
            "--source-board-json",
            str(paths["source"]),
            "--material-clusters-json",
            str(paths["clusters"]),
            "--section-spine-json",
            str(paths["sections"]),
            "--storyboard-rows-json",
            str(paths["rows"]),
            "--visual-direction-md",
            str(paths["visual_direction"]),
            "--narration-readthrough-md",
            str(paths["readthrough"]),
            "--storyboard-review-json",
            str(paths["storyboard_review"]),
            "--cross-ai-synthesis-json",
            str(paths["cross_ai"]),
            "--output-deck-json",
            str(paths["deck_out"]),
            "--output-source-binding-json",
            str(paths["source_out"]),
            "--output-section-binding-json",
            str(paths["section_out"]),
            "--output-visual-plan-json",
            str(paths["visual_out"]),
            "--output-asset-inventory-json",
            str(paths["inventory_out"]),
            "--output-delivery-single-speaker",
            str(paths["single_out"]),
            "--output-delivery-notebooklm",
            str(paths["notebook_out"]),
            "--output-md",
            str(paths["md_out"]),
            "--output-json",
            str(paths["json_out"]),
            "--summary",
        ],
        capture_output=True,
        text=True,
    )


def test_v22_editorial_compile_blocks_unapproved_storyboard(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    payload = json.loads(paths["storyboard_review"].read_text(encoding="utf-8"))
    payload["approval"]["status"] = "needs_revision"
    _write_json(paths["storyboard_review"], payload)

    result = _run(paths)

    assert result.returncode == 2
    assert "storyboard_not_approved" in result.stderr


def test_v22_editorial_compile_blocks_missing_named_source_validator_summary(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    payload = json.loads(paths["clusters"].read_text(encoding="utf-8"))
    for cluster in payload["clusters"]:
        for contribution in cluster["contributions"]:
            contribution.pop("paper_url", None)
            contribution.pop("openreview_url", None)
    _write_json(paths["clusters"], payload)

    result = _run(paths)

    assert result.returncode == 2
    assert "missing_named_source_link_summary" in result.stderr


def test_v22_editorial_compile_blocks_raw_metadata_tag_claims(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    rows = json.loads(paths["rows"].read_text(encoding="utf-8"))
    rows["rows"][0]["viewer_facing_claim"] = "agent/system safety"
    _write_json(paths["rows"], rows)

    result = _run(paths)

    assert result.returncode == 2
    assert "raw_metadata_tag_visible_claim" in result.stderr


def test_v22_editorial_compile_preserves_people_system_dataset_links(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    source = json.loads(paths["source_out"].read_text(encoding="utf-8"))
    deck = json.loads(paths["deck_out"].read_text(encoding="utf-8"))
    visual = json.loads(paths["visual_out"].read_text(encoding="utf-8"))
    ref = source["slides"][0]["visible_references"][0]
    mention = deck["slides"][0]["named_paper_talk_speaker_author_mentions"][0]
    figure = visual["slides"][0]["figures"][0]
    assert ref["authors"] == ["Sanidhya Vijayvargiya", "Maarten Sap"]
    assert ref["systems"] == ["OpenAgentSafety"]
    assert ref["datasets"] == ["OpenAgentSafety 350+ multi-turn multi-user tasks"]
    assert ref["openreview_url"] == "https://openreview.net/forum?id=xggSxCFQbA"
    assert mention["reference_id"] == "paper_abstract:xggSxCFQbA"
    assert figure["model_route"] == "anthropic:claude-opus-4-8"
    assert figure["renderer_family"] in {"drawsvg", "graphviz", "altair-vl-convert", "pycairo"}


def test_v22_editorial_compile_preserves_explicit_unlisted_speech_token(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    rows = json.loads(paths["rows"].read_text(encoding="utf-8"))
    rows["rows"][0]["slide_id"] = "s02"
    rows["rows"][0]["named_source_contribution"] = (
        "Don Song, 'Opening Remarks: Agents in the Wild: Safety, Security, and Beyond,' "
        "ICLR 2026 Workshop AIWILD "
        "[speech_transcript:0d022da1ec8edbb2608ffb54:segment:00003]"
    )
    rows["rows"][0]["proof_object"] = (
        "Don Song's AIWILD cyber-risk framing with AI-enabled zero-day discovery evidence."
    )
    _write_json(paths["rows"], rows)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    source = json.loads(paths["source_out"].read_text(encoding="utf-8"))
    deck = json.loads(paths["deck_out"].read_text(encoding="utf-8"))
    visual = json.loads(paths["visual_out"].read_text(encoding="utf-8"))
    notebook = json.loads(paths["notebook_out"].read_text(encoding="utf-8"))
    ref = source["slides"][0]["visible_references"][0]
    assert ref["source_id"] == "speech_transcript:0d022da1ec8edbb2608ffb54:segment:00003"
    assert ref["reference_id"] == "speech_transcript:0d022da1ec8edbb2608ffb54:segment:00003"
    assert ref["title"] == "Opening Remarks: Agents in the Wild: Safety, Security, and Beyond"
    assert ref["speakers"] == ["Dawn Song"]
    assert ref["source_type"] == "talk"
    assert ref["source_family"] == "artifacts"
    assert ref["cluster_id"] == "agent-system-safety"
    assert ref["path"].endswith("talk_media_001_157d4e6078-raw.mp3")
    assert ref["source_url"] == "https://iclr.cc/virtual/2026/workshop/10000781"
    assert ref["url"] == "https://iclr.cc/virtual/2026/workshop/10000781"
    assert ref["authors"] == []
    assert "Phase 01 brand and creative brief" not in json.dumps(ref)
    generated = json.dumps([source, deck, visual, notebook])
    assert "Dawn Song" in generated
    assert "Don Song" not in generated


def test_v22_editorial_compile_writes_hash_bound_render_inputs(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    receipt = json.loads(paths["json_out"].read_text(encoding="utf-8"))
    contract = receipt["phase16_source_contract"]
    for key in (
        "source_board_sha256",
        "material_clusters_sha256",
        "section_spine_sha256",
        "storyboard_rows_sha256",
        "visual_direction_sha256",
        "narration_readthrough_sha256",
        "storyboard_review_sha256",
        "cross_ai_synthesis_sha256",
        "brand_lock_sha256",
    ):
        assert len(contract[key]) == 64
    assert receipt["output_hashes"]["presentation_deck.json"]
    assert receipt["counts"]["slide_count"] == 1
    deck = json.loads(paths["deck_out"].read_text(encoding="utf-8"))
    assert deck["blueprint_sha256"] == contract["blueprint_sha256"]
    assert deck["storyboard_sha256"] == contract["storyboard_rows_sha256"]
    assert deck["approved_expansion"]["approved"] is True
    assert deck["approved_expansion"]["target_slide_count"] == 1
    assert deck["approved_expansion"]["blueprint_sha256"] == contract["blueprint_sha256"]
    assert deck["approved_expansion"]["source_plan_sha256"] == contract["source_board_sha256"]
    assert "approved_expansion_justification" not in deck


def test_v22_editorial_compile_writes_delivery_contracts_linked_to_readthrough(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)

    result = _run(paths)

    assert result.returncode == 0, result.stderr
    single = json.loads(paths["single_out"].read_text(encoding="utf-8"))
    notebook = json.loads(paths["notebook_out"].read_text(encoding="utf-8"))
    assert single["delivery_variant"] == "single_speaker"
    assert notebook["delivery_variant"] == "notebooklm"
    assert single["segments"][0]["narration"] == "OpenAgentSafety makes the agent evaluation boundary concrete."
    assert notebook["source_grounding_checks"]["status"] == "pass"
    assert "Visually, the frame uses" in notebook["segments"][0]["narration"]
    assert "diagram about" not in notebook["segments"][0]["narration"]
    assert "the visual is doing the evidence check" not in notebook["segments"][0]["narration"]
    assert "Host A frames" not in notebook["segments"][0]["narration"]
    assert "Host B connects" not in notebook["segments"][0]["narration"]
    assert single["approve_for_tts"] is False
    assert "Phase 08" in single["approval_blocker"]


def test_v22_editorial_compile_generates_specific_notebooklm_visual_shapes(
    tmp_path: Path,
) -> None:
    cases = [
        (
            "s02",
            (
                "Annotated threat-surface diagram showing agent loop components--memory, "
                "tools, environment, GUI, RAG--with attack-vector labels at each component."
            ),
            "four-step agent loop",
            "three panels for memory, tools, and environment",
        ),
        (
            "s04",
            (
                "Source-bound benchmark coverage matrix: rows are A2ASecBench, MCP-SafetyBench, "
                "BLIND-ACT, DECEPTICON, and OpenAgentSafety; columns are protocol, tool, "
                "memory, environment, and GUI/action safety."
            ),
            "benchmark coverage matrix",
            "three panels for memory, tools, and environment",
        ),
        (
            "s16",
            (
                "Source-inference defense operating-point schematic: show undefended "
                "source-inference attack path, defended parameter-shuffling plus "
                "residue-number-system path, and an accuracy-preservation callout."
            ),
            "source-inference defense schematic",
            "parameter-shuffling plus.",
        ),
        (
            "s21",
            (
                "Seven-node radial diagram with each of the deck's topic clusters as a node "
                "connected to a central Trustworthy Deployment hub; each node carries a "
                "human-readable open question."
            ),
            "radial portfolio map",
            "each node carries.",
        ),
    ]

    for slide_id, visual_concept, expected, forbidden in cases:
        paths = _fixtures(tmp_path / slide_id)
        rows = json.loads(paths["rows"].read_text(encoding="utf-8"))
        rows["rows"][0]["slide_id"] = slide_id
        rows["rows"][0]["visual_concept"] = visual_concept
        _write_json(paths["rows"], rows)

        result = _run(paths)

        assert result.returncode == 0, result.stderr
        notebook = json.loads(paths["notebook_out"].read_text(encoding="utf-8"))
        narration = notebook["segments"][0]["narration"]
        assert expected in narration
        assert forbidden not in narration
