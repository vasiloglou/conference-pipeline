from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from conference_pipeline.presentation.presentation_frame import validate_presentation_frame_payload

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_topic_presentation_frames_test",
        REPO_ROOT / "scripts" / "presentations" / "prepare_topic_presentation_frames.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_topic_presentation_frames_writes_valid_frame_and_receipt(tmp_path: Path) -> None:
    module = _load_module()
    workspace = tmp_path / "conferences" / "demo2026"
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "analysis" / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "topics": [
                    {
                        "topic_id": "01_ai_for_science_physics_discovery",
                        "title": "AI for science, physics, and autonomous discovery",
                        "analysis_path": "analysis/01_ai_for_science_physics_discovery.md",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # Audience titles now come from the manifest title, with an optional
    # per-workspace override file (decks/topic_title_overrides.json) instead of
    # a hardcoded module dict.
    (workspace / "decks").mkdir(parents=True, exist_ok=True)
    (workspace / "decks" / "topic_title_overrides.json").write_text(
        json.dumps(
            {"01_ai_for_science_physics_discovery": "AI for science after fragile discovery loops"}
        ),
        encoding="utf-8",
    )
    (workspace / "analysis" / "01_ai_for_science_physics_discovery.md").write_text(
        "\n".join(
            [
                "# AI for science, physics, and autonomous discovery",
                "",
                "## Executive Summary",
                "",
                "The evidence shows autonomous scientific discovery becoming an infrastructure problem: "
                "models must survive physics constraints, corrupted measurements, specialist routing, "
                "and validation bottlenecks before they can be trusted in laboratories.",
                "",
                "## Key Findings",
                "",
                "- **Robust optimization matters.** Sequential discovery loops fail when measurements corrupt the surrogate.",
                "- **Specialist models matter.** Physics, biology, and medicine need different evidence interfaces.",
                "",
                "## Slide Brief",
                "",
                "- Slide seed: discovery loops need robust evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--workspace",
            str(workspace),
            "--run-id",
            "demo-run",
            "--conference-label",
            "DemoConf 2026",
            "--summary",
        ]
    )

    assert rc == 0
    frame_path = workspace / "decks" / "01_ai_for_science_physics_discovery" / "presentation_frame.json"
    frame = json.loads(frame_path.read_text(encoding="utf-8"))
    assert validate_presentation_frame_payload(frame).status == "pass"
    assert frame["audience_title"] == "AI for science after fragile discovery loops"
    assert frame["source_analysis_path"] == "analysis/01_ai_for_science_physics_discovery.md"
    assert "physics constraints" in frame["field_thesis"]
    assert len(frame["general_questions"]) >= 3

    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "demo-run"
            / "presentation-frame-prep.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "succeeded"
    assert receipt["written_count"] == 1
    assert receipt["failed_count"] == 0
    assert receipt["frames"][0]["presentation_frame_validation"]["status"] == "pass"


def test_prepare_topic_presentation_frames_preserves_existing_without_overwrite(
    tmp_path: Path,
) -> None:
    module = _load_module()
    workspace = tmp_path / "workspace"
    deck_dir = workspace / "decks" / "01_demo"
    deck_dir.mkdir(parents=True)
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "analysis" / "analysis_manifest.json").write_text(
        json.dumps({"topics": [{"topic_id": "01_demo", "title": "Demo topic"}]}),
        encoding="utf-8",
    )
    (workspace / "analysis" / "01_demo.md").write_text(
        "# Demo topic\n\n## Executive Summary\n\nNew analysis should not overwrite the frame.\n",
        encoding="utf-8",
    )
    original = {
        "schema_version": 1,
        "presentation_id": "01_demo",
        "topic_id": "01_demo",
        "retrieval_title": "Demo topic",
        "audience_title": "Existing audience title",
        "field_thesis": "Existing field thesis.",
        "general_questions": ["Question one?", "Question two?", "Question three?"],
        "source_analysis_path": "analysis/01_demo.md",
        "opening_anchor_policy": {"rules": ["Keep existing frame."]},
    }
    (deck_dir / "presentation_frame.json").write_text(
        json.dumps(original, indent=2) + "\n",
        encoding="utf-8",
    )

    rc = module.main(["--workspace", str(workspace), "--run-id", "demo-run"])

    assert rc == 0
    frame = json.loads((deck_dir / "presentation_frame.json").read_text(encoding="utf-8"))
    assert frame == original
    receipt = json.loads(
        (
            workspace
            / "storage"
            / "workflow-runs"
            / "demo-run"
            / "presentation-frame-prep.json"
        ).read_text(encoding="utf-8")
    )
    assert receipt["existing_count"] == 1
    assert receipt["written_count"] == 0
