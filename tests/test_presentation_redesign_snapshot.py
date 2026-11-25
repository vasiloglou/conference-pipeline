from __future__ import annotations

import json
from pathlib import Path

from scripts.presentations.snapshot_presentation_redesign import build_snapshot


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_snapshot_copies_only_control_artifacts_and_manifest(tmp_path: Path) -> None:
    workspace = tmp_path / "conferences" / "demo2026"
    deck = workspace / "decks" / "04_safety_alignment_privacy"
    run = workspace / "storage" / "workflow-runs" / "demo-run"
    _write(workspace / "decks" / "presentation_manifest.json", '{"decks": []}\n')
    _write(workspace / "storage" / "storage_manifest.json", '{"workspace": "demo"}\n')
    _write(run / "summary.json", '{"status": "partial"}\n')
    _write(deck / "1_draft_slides.md", "# Draft\n")
    _write(deck / "5_draft_slides_with_images.md", "# Images\n")
    _write(deck / "asset_inventory.json", '{"assets": []}\n')
    _write(deck / "human_review_receipt.json", '{"decision": "needs_revision"}\n')
    _write(deck / "phase13_canary_review_packet.json", '{"status": "needs_revision"}\n')

    _write(workspace / ".env", "SECRET=value\n")
    _write(workspace / "storage" / "provider-calls" / "call.json", '{"secret": true}\n')
    _write(deck / "presentation_video.mp4", "video bytes\n")
    _write(deck / "presentation.pdf", "pdf bytes\n")
    _write(deck / "screenshot.png", "screenshot bytes\n")
    _write(deck / "assets" / "generated" / "figure.svg", "<svg />\n")

    manifest = build_snapshot(
        workspace=workspace,
        presentation_id="04_safety_alignment_privacy",
        run_id="demo-run",
        snapshot_id="phase14-demo",
    )

    snapshot_root = workspace / manifest["snapshot_root"]
    manifest_path = snapshot_root / "snapshot_manifest.json"
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["bulk_regeneration_blocked"] is True
    assert loaded["presentation_id"] == "04_safety_alignment_privacy"
    assert "decks/04_safety_alignment_privacy/1_draft_slides.md" in loaded["source_paths"]
    assert "storage/workflow-runs/demo-run/summary.json" in loaded["source_paths"]
    for source_path in loaded["source_paths"]:
        assert source_path in loaded["sha256_by_path"]

    assert (snapshot_root / "decks" / "04_safety_alignment_privacy" / "1_draft_slides.md").exists()
    assert not (snapshot_root / ".env").exists()
    assert not (snapshot_root / "storage" / "provider-calls" / "call.json").exists()
    assert not (snapshot_root / "decks" / "04_safety_alignment_privacy" / "presentation_video.mp4").exists()
    assert not (snapshot_root / "decks" / "04_safety_alignment_privacy" / "presentation.pdf").exists()
    assert not (snapshot_root / "decks" / "04_safety_alignment_privacy" / "screenshot.png").exists()
    assert not (snapshot_root / "decks" / "04_safety_alignment_privacy" / "assets" / "generated" / "figure.svg").exists()


def test_snapshot_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside-summary.json"
    _write(outside, "{}\n")

    try:
        build_snapshot(
            workspace=workspace,
            presentation_id="demo",
            snapshot_id="snap",
            summary=outside,
        )
    except Exception as exc:
        assert "must stay inside the workspace" in str(exc)
    else:
        raise AssertionError("outside summary path should have been rejected")
