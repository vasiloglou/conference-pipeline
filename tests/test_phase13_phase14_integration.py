from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from conference_pipeline.presentation.programmatic_figures import generate_programmatic_figures


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "presentations"
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "presentation_blueprint"


def load_presentation_script(script_name: str, module_name: str | None = None) -> Any:
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(
        module_name or f"phase14_integration_{script_path.stem}",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_live_blueprint_receipt(deck_dir: Path) -> None:
    blueprint = deck_dir / "presentation_blueprint.json"
    import hashlib

    write_text(
        deck_dir / "presentation_blueprint_receipt.json",
        json.dumps(
            {
                "blueprint_sha256": hashlib.sha256(blueprint.read_bytes()).hexdigest(),
                "mode": "live",
                "production_evidence": True,
                "provider_receipt_paths": ["storage/provider-calls/blueprint-generation/demo.json"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_phase15_summaries(phase_dir: Path) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, 12):
        write_text(phase_dir / f"15-{index:02d}-SUMMARY.md", f"# Summary {index}\n")


def write_phase15_owner_registry(path: Path, *, omit: str | None = None) -> None:
    required = [
        "presentation.source_corpus_inventory",
        "presentation.speaker_author_work_ledger",
        "presentation.repair_presentation_sources",
        "presentation.research_cluster_plan",
        "presentation.artifact_truth_validation",
        "presentation.video_sync_validation",
    ]
    scripts = {
        task_id: {"path": f"scripts/{task_id.replace('.', '_')}.py"}
        for task_id in required
        if task_id != omit
    }
    tasks = {task_id: {"status": "required"} for task_id in required if task_id != omit}
    task_bindings = {task_id: {"owner": task_id} for task_id in required if task_id != omit}
    write_text(
        path,
        json.dumps(
            {
                "schema_version": 1,
                "scripts": scripts,
                "tasks": tasks,
                "task_bindings": task_bindings,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_pre_repair_report(path: Path, *, status: str, deck_dir: Path) -> None:
    write_text(
        path,
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "checks": ["html", "pdf", "frames", "audio", "mp4", "source", "receipts"],
                "blockers": [
                    {
                        "code": "HTML_NO_IMAGE_OR_SVG",
                        "path": str(deck_dir / "presentation.html"),
                        "message": "pre-repair current Topic 4 artifact failed",
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def phase15_canary_args(
    *,
    workspace: Path,
    deck_id: str,
    phase_dir: Path,
    registry: Path,
    pre_repair_report: Path | None = None,
) -> list[str]:
    args = [
        "--workspace",
        str(workspace),
        "--deck-id",
        deck_id,
        "--phase15-canary-repair",
        "--phase15-phase-dir",
        str(phase_dir),
        "--stage-owner-registry",
        str(registry),
    ]
    if pre_repair_report is not None:
        args.extend(["--pre-repair-report", str(pre_repair_report)])
    return args


def test_phase15_canary_repair_guard_requires_topic4_fail_report_and_owner_registry(
    tmp_path: Path,
) -> None:
    rollout = load_presentation_script("run_phase13_rollout.py", "phase15_canary_rollout")
    workspace = tmp_path / "conferences" / "demo2026"
    topic4 = workspace / "decks" / "04_safety_alignment_privacy"
    other_topic = workspace / "decks" / "05_generative_models"
    topic4.mkdir(parents=True)
    other_topic.mkdir(parents=True)
    phase_dir = tmp_path / "phase15"
    registry = tmp_path / "stage_owner_registry.json"
    write_phase15_summaries(phase_dir)
    write_phase15_owner_registry(registry)
    pre_repair = topic4 / "artifact_truth_pre_repair_report.json"
    write_pre_repair_report(pre_repair, status="pass", deck_dir=topic4)

    args = rollout.parse_args(
        phase15_canary_args(
            workspace=workspace,
            deck_id="04_safety_alignment_privacy",
            phase_dir=phase_dir,
            registry=registry,
            pre_repair_report=pre_repair,
        )
    )
    targets = rollout._select_targets(
        decks_dir=workspace / "decks",
        deck_ids=args.deck_id,
        all_decks=args.all,
    )
    report = rollout.build_report(workspace=workspace, targets=targets, args=args)

    assert report["status"] == "blocked_pre_repair_artifact_truth_not_failed"
    assert report["phase15_canary_repair_gate"]["pre_repair_status"] == "pass"

    write_pre_repair_report(pre_repair, status="fail", deck_dir=topic4)
    args = rollout.parse_args(
        phase15_canary_args(
            workspace=workspace,
            deck_id="05_generative_models",
            phase_dir=phase_dir,
            registry=registry,
            pre_repair_report=pre_repair,
        )
    )
    targets = rollout._select_targets(
        decks_dir=workspace / "decks",
        deck_ids=args.deck_id,
        all_decks=args.all,
    )
    report = rollout.build_report(workspace=workspace, targets=targets, args=args)

    assert report["status"] == "blocked_phase15_canary_scope"
    assert report["phase15_canary_repair_gate"]["allowed_deck_id"] == "04_safety_alignment_privacy"

    write_phase15_owner_registry(registry, omit="presentation.artifact_truth_validation")
    args = rollout.parse_args(
        phase15_canary_args(
            workspace=workspace,
            deck_id="04_safety_alignment_privacy",
            phase_dir=phase_dir,
            registry=registry,
            pre_repair_report=pre_repair,
        )
    )
    targets = rollout._select_targets(
        decks_dir=workspace / "decks",
        deck_ids=args.deck_id,
        all_decks=args.all,
    )
    report = rollout.build_report(workspace=workspace, targets=targets, args=args)

    assert report["status"] == "blocked_missing_phase15_owner_commands"
    assert "presentation.artifact_truth_validation" in report["phase15_canary_repair_gate"]["missing_owner_commands"]


def test_conference_label_defaults_from_workspace_config_not_hardcoded_iclr(tmp_path: Path) -> None:
    """--conference-label must default to the invoking workspace's own
    config.yaml conference.name -- a prior bug hardcoded the argparse default
    to "ICLR 2026", so every other conference's rollout (including CVPR2026's
    once Phase 10/11 wires this script up) would have silently gotten ICLR's
    label on title slides and NotebookLM transformation."""

    rollout = load_presentation_script("run_phase13_rollout.py", "conference_label_default_rollout")

    workspace = tmp_path / "conferences" / "cvpr2026"
    workspace.mkdir(parents=True)
    write_text(
        workspace / "config.yaml",
        'conference:\n  name: "CVPR 2026"\n',
    )

    args = rollout.parse_args(["--workspace", str(workspace), "--deck-id", "01_some_topic"])

    assert args.conference_label == "CVPR 2026"

    # Explicit override still wins over the config-derived default.
    override_args = rollout.parse_args(
        [
            "--workspace",
            str(workspace),
            "--deck-id",
            "01_some_topic",
            "--conference-label",
            "Custom Label",
        ]
    )
    assert override_args.conference_label == "Custom Label"


def test_phase13_phase14_dry_contract_path_avoids_media_and_provider_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck_dir = workspace / "decks" / "01_demo_topic"
    write_text(deck_dir / "presentation_blueprint.json", (FIXTURE_ROOT / "valid_blueprint.json").read_text(encoding="utf-8"))
    write_live_blueprint_receipt(deck_dir)
    write_text(deck_dir / "blueprint_review.json", (FIXTURE_ROOT / "approved_blueprint_review.json").read_text(encoding="utf-8"))
    write_text(deck_dir / "source_pack_receipt.json", '{"schema_version": 1, "topic_id": "01_demo_topic"}\n')
    write_text(deck_dir / "ranked_papers.json", '{"papers": []}\n')
    write_text(deck_dir / "matched_papers.json", '{"papers": []}\n')
    write_text(deck_dir / "source_attributions.json", '{"sources": []}\n')

    executed_commands: list[list[str]] = []

    def record_command(cmd: list[str], *_args: object, **_kwargs: object) -> None:
        executed_commands.append(cmd)
        raise AssertionError(f"unexpected external command: {cmd}")

    slides_agent = load_presentation_script("generate_slides_agent.py", "phase14_integration_slides")
    monkeypatch.setattr(
        slides_agent,
        "generate_slides_with_claude",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("provider call")),
    )
    slides_args = slides_agent.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--display-title",
            "Demo Topic",
            "--topic",
            "01_demo_topic",
            "--from-blueprint",
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-receipt",
            str(deck_dir / "presentation_blueprint_receipt.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--output-dir",
            str(deck_dir),
            "--output-slides-md",
            str(deck_dir / "1_draft_slides.md"),
        ]
    )
    slides_agent.process_workspace_draft(slides_args)

    enrich = load_presentation_script("enrich_slides_with_papers.py", "phase14_integration_enrich")
    monkeypatch.setattr(enrich, "call_agent", lambda *_args, **_kwargs: {"new_slides": []})
    enrich_args = enrich.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--draft-slides",
            str(deck_dir / "1_draft_slides.md"),
            "--matched-papers",
            str(deck_dir / "matched_papers.json"),
            "--output-slides",
            str(deck_dir / "2_draft_slides_enriched_with_papers.md"),
            "--presentation-blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--slides",
            "2",
        ]
    )
    enrich.process_workspace_enrichment(enrich_args)

    inventory = generate_programmatic_figures(
        workspace,
        "01_demo_topic",
        input_slides=deck_dir / "2_draft_slides_enriched_with_papers.md",
        source_pack=deck_dir / "source_pack_receipt.json",
        ranked_papers=deck_dir / "ranked_papers.json",
        asset_inventory=deck_dir / "asset_inventory.json",
        output_slides=deck_dir / "5_draft_slides_with_images.md",
        generated_dir=deck_dir / "assets" / "generated",
        dry_run=True,
    )
    assert inventory["figures"]
    assert inventory["figures"][0]["figure_design_brief"]["slide_id"] == "slide-001"

    write_text(
        deck_dir / "6_presenter_script.md",
        """# Script

### [SLIDE 1] Demo
**SAY:**
> The blueprint narration beat names the failure chain and source evidence.

### [SLIDE 2] Demo
**SAY:**
> The blueprint narration beat now explains the dependency map.
""",
    )
    write_text(deck_dir / "6_presenter_script_notebooklm_2host.md", (deck_dir / "6_presenter_script.md").read_text(encoding="utf-8"))
    write_text(deck_dir / "presentation.html", "<html></html>")
    write_text(
        deck_dir / "presentation_slide_mapping.json",
        '{"slides": [{"markdown_slide": 1, "html_index": 0}, {"markdown_slide": 2, "html_index": 1}]}\n',
    )
    write_text(
        deck_dir / "editorial_storyboard.json",
        json.dumps(
            {
                "clusters": [
                    {
                        "claim": "Blueprints bind story, source, visual, and narration.",
                        "representative_papers": ["SRC-001"],
                        "retained_slides": [1, 2],
                        "merged_slides": [],
                        "dropped_slides": [],
                        "retained_rationale": "Both slides represent distinct blueprint gates.",
                        "merged_rationale": "The compact fixture has no duplicates.",
                        "dropped_rationale": "The compact fixture drops nothing.",
                        "figure_roles": {"1": "mechanism", "2": "workflow"},
                        "narration_beats": {"1": "failure chain", "2": "dependency map"},
                    }
                ]
            }
        ),
    )

    build_video = load_presentation_script("build_presenter_video.py", "phase14_integration_video")
    monkeypatch.setattr(build_video.subprocess, "run", record_command)
    video_args = build_video.parse_args(
        [
            "--workspace",
            str(workspace),
            "--presentation-id",
            "01_demo_topic",
            "--presentation-type",
            "topic-deep-dive",
            "--script-md",
            str(deck_dir / "6_presenter_script.md"),
            "--slides-html",
            str(deck_dir / "presentation.html"),
            "--slide-mapping",
            str(deck_dir / "presentation_slide_mapping.json"),
            "--build-dir",
            str(deck_dir / "video_build"),
            "--output",
            str(deck_dir / "presentation_video.mp4"),
            "--blueprint",
            str(deck_dir / "presentation_blueprint.json"),
            "--blueprint-review",
            str(deck_dir / "blueprint_review.json"),
            "--dry-run",
        ]
    )
    payload = build_video._dry_run_payload(
        video_args,
        script_md=deck_dir / "6_presenter_script.md",
        slides_html=deck_dir / "presentation.html",
        slide_mapping=deck_dir / "presentation_slide_mapping.json",
        build_dir=deck_dir / "video_build",
        output_video=deck_dir / "presentation_video.mp4",
    )
    assert payload["blueprint_narration_beat_contract"] == "blueprint narration beat"

    editorial_gate = load_presentation_script("presentation_editorial_gate.py", "phase14_integration_gate")
    report = editorial_gate.analyze_deck(
        deck_dir=deck_dir,
        slides_md=deck_dir / "2_draft_slides_enriched_with_papers.md",
        presenter_script_md=deck_dir / "6_presenter_script.md",
        notebooklm_script_md=deck_dir / "6_presenter_script_notebooklm_2host.md",
        asset_inventory=None,
        storyboard=deck_dir / "editorial_storyboard.json",
        deck_kind="topic",
    )
    assert report["metrics"]["blueprint_duration_gate_owner"] == "Phase 14"
    assert report["metrics"]["final_render_duration_gate_owner"] == "Phase 13 final rendered"

    forbidden = ("ffmpeg", "playwright", "synthesize_audio", "tts", "pdf", "mp4")
    flattened = " ".join(" ".join(command).lower() for command in executed_commands)
    assert not any(term in flattened for term in forbidden)
    assert not list(deck_dir.glob("*.mp4"))
    assert not list(deck_dir.glob("*.pdf"))
