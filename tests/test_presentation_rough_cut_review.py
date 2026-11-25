from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline.presentation.rough_cut_review import (
    PACKET_TYPE,
    ROUGH_CUT_HUMAN_RECEIPT_NEEDS_REVISION,
    ROUGH_CUT_MECHANICAL_DIALOGUE,
    ROUGH_CUT_MISSING_MP4,
    ROUGH_CUT_MISSING_PDF_SCREENSHOTS,
    ROUGH_CUT_SCAFFOLD_PROSE_PRESENT,
    ROUGH_CUT_VIDEO_BLACK_FRAME,
    ROUGH_CUT_VIDEO_DESYNC,
    ROUGH_CUT_VISUAL_CLAIM_MISMATCH,
    build_rough_cut_packet,
    review_rough_cut_packet,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_good_inputs(deck_dir: Path, phase_dir: Path) -> dict[str, Path]:
    deck_dir.mkdir(parents=True, exist_ok=True)
    phase_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "presentation.html",
        "presentation.pdf",
        "render_contact_sheet.png",
        "presentation_video.mp4",
        "presentation_video_notebooklm.mp4",
    ]:
        (deck_dir / name).write_bytes(b"artifact")
    (deck_dir / "phase13_canary_frames").mkdir()
    (deck_dir / "phase13_canary_frames" / "contact_sheet.png").write_bytes(b"sheet")
    (deck_dir / "rendered_frames").mkdir()
    (deck_dir / "rendered_frames" / "slide_001.png").write_bytes(b"frame")
    _write_json(deck_dir / "render_spec.json", {"slides": [{"slide_id": "s01"}]})
    (deck_dir / "6_presenter_script.md").write_text("The proof object is visible.\n", encoding="utf-8")
    (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
        "- **Host A:** The source claim is visible.\n- **Host B:** The viewer-facing proof object is clear.\n",
        encoding="utf-8",
    )
    paths = {
        "brand": phase_dir / "brand.json",
        "source": phase_dir / "source.json",
        "storyboard": phase_dir / "storyboard.json",
        "readthrough": phase_dir / "readthrough.md",
        "artifact_truth": deck_dir / "artifact_truth_report.json",
        "single_sync": deck_dir / "video_sync_validation_single_speaker.json",
        "notebook_sync": deck_dir / "video_sync_validation_notebooklm.json",
    }
    _write_json(paths["brand"], {"status": "pass"})
    _write_json(paths["source"], {"status": "pass"})
    _write_json(paths["storyboard"], {"approval": {"status": "approved_for_render_planning"}})
    paths["readthrough"].write_text("Every visual phrase maps to a proof object.\n", encoding="utf-8")
    _write_json(paths["artifact_truth"], {"status": "pass", "blockers": []})
    _write_json(paths["single_sync"], {"status": "pass", "errors": []})
    _write_json(paths["notebook_sync"], {"status": "pass", "errors": []})
    return paths


def _build_packet(deck_dir: Path, phase_dir: Path, paths: dict[str, Path]):
    return build_rough_cut_packet(
        deck_dir=deck_dir,
        phase_dir=phase_dir,
        brand_lock_json=paths["brand"],
        source_board_json=paths["source"],
        storyboard_review_json=paths["storyboard"],
        narration_readthrough_md=paths["readthrough"],
        artifact_truth_json=paths["artifact_truth"],
        single_video_sync_json=paths["single_sync"],
        notebooklm_video_sync_json=paths["notebook_sync"],
    )


def _codes(packet_or_result) -> set[str]:
    if hasattr(packet_or_result, "blockers"):
        return {finding.code for finding in packet_or_result.blockers}
    return {finding.code for finding in packet_or_result.findings}


def test_rough_cut_packet_blocks_missing_pdf_screenshots(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    for frame in (deck_dir / "rendered_frames").glob("*.png"):
        frame.unlink()

    packet = _build_packet(deck_dir, phase_dir, paths)

    assert ROUGH_CUT_MISSING_PDF_SCREENSHOTS in _codes(packet)


def test_rough_cut_packet_blocks_missing_mp4s(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    (deck_dir / "presentation_video.mp4").unlink()

    packet = _build_packet(deck_dir, phase_dir, paths)

    assert ROUGH_CUT_MISSING_MP4 in _codes(packet)


def test_rough_cut_review_blocks_black_or_desynced_video_findings(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    _write_json(
        paths["single_sync"],
        {"status": "fail", "errors": [{"code": "VIDEO_FLAT_FRAME_LOW_ENTROPY", "path": "frame.png"}]},
    )
    _write_json(
        paths["notebook_sync"],
        {"status": "fail", "errors": [{"code": "PRESENTATION_SLIDE_SYNC_MISMATCH", "path": "video.mp4"}]},
    )

    packet = _build_packet(deck_dir, phase_dir, paths)

    assert ROUGH_CUT_VIDEO_BLACK_FRAME in _codes(packet)
    assert ROUGH_CUT_VIDEO_DESYNC in _codes(packet)


def test_rough_cut_review_blocks_visual_claim_mismatch(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    (deck_dir / "6_presenter_script.md").write_text("The diagram shows an unsupported claim.\n", encoding="utf-8")

    packet = _build_packet(deck_dir, phase_dir, paths)

    assert ROUGH_CUT_VISUAL_CLAIM_MISMATCH in _codes(packet)


def test_rough_cut_review_blocks_mechanical_notebooklm_dialogue(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "presentations" / "phase16" / "mechanical_notebooklm_script.md"
    (deck_dir / "6_presenter_script_notebooklm_2host.md").write_text(
        fixture.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    packet = _build_packet(deck_dir, phase_dir, paths)

    codes = _codes(packet)
    assert ROUGH_CUT_MECHANICAL_DIALOGUE in codes
    assert ROUGH_CUT_SCAFFOLD_PROSE_PRESENT in codes


def test_rough_cut_review_blocks_scaffold_prose(tmp_path: Path) -> None:
    deck_dir = tmp_path / "deck"
    phase_dir = tmp_path / "phase"
    paths = _write_good_inputs(deck_dir, phase_dir)
    (deck_dir / "6_presenter_script.md").write_text("Use the visual role to describe this slide.\n", encoding="utf-8")

    packet = _build_packet(deck_dir, phase_dir, paths)

    assert ROUGH_CUT_SCAFFOLD_PROSE_PRESENT in _codes(packet)


def test_rough_cut_review_blocks_needs_revision_receipt() -> None:
    packet = {
        "schema_version": 1,
        "packet_type": PACKET_TYPE,
        "status": "ready_for_operator_review",
        "blockers": [],
        "viewer_findings": [],
        "human_review": {
            "state": "needs_revision",
            "receipt_gate": "pending_human_review",
            "blocks_rough_cut": True,
        },
    }

    result = review_rough_cut_packet(packet)

    assert result.status == "blocked"
    assert ROUGH_CUT_HUMAN_RECEIPT_NEEDS_REVISION in _codes(result)


def test_rough_cut_cli_help_commands_work() -> None:
    env = {"PYTHONPATH": str(PROJECT_ROOT / "src")}
    for script in ["build_rough_cut_review_packet.py", "review_rough_cut.py"]:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "presentations" / script), "--help"],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode == 0, completed.stderr

