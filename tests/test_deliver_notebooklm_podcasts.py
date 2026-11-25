"""Tests for scripts/deliver_notebooklm_podcasts.py (curated Drive delivery)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from conference_pipeline import podcast_notebooklm as pn

from tests.test_podcast_notebooklm import EXEC_SHORT, TOPIC_A, TOPIC_B, make_workspace, passing_transcriber

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "deliver_notebooklm_podcasts.py"


def load_delivery_module():
    spec = importlib.util.spec_from_file_location("deliver_notebooklm_podcasts_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(tmp_path: Path, *, stage_status: str = "succeeded") -> tuple[Path, str, Path]:
    ws = make_workspace(tmp_path)
    summary = pn.run_stage(ws, deck_ids=None, engine=pn.FakeNotebookLMEngine(), mode="live", transcriber=passing_transcriber())
    assert summary["status"] == "succeeded"
    run_id = "test-delivery"
    receipt = ws / "storage" / "workflow-runs" / run_id / "stage-10-notebooklm-podcast.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"stage_id": "notebooklm-podcast", "status": stage_status}), encoding="utf-8")
    drive_root = tmp_path / "mounted-drive" / "TestConf 2030 Podcasts"
    drive_root.parent.mkdir(parents=True)
    return ws, run_id, drive_root


def test_dry_run_plans_every_episode_without_writing(tmp_path: Path) -> None:
    ws, run_id, drive_root = _seed(tmp_path)
    mod = load_delivery_module()
    receipt = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=True, receipt_path=None, require_stage_receipt=True)
    assert receipt["status"] == "planned" and receipt["expected_count"] == 3 == receipt["delivered_count"]
    assert {item["audio_action"] for item in receipt["items"]} == {"would_copy"}
    assert not drive_root.exists()
    assert Path(receipt["receipt_path"]).name == "notebooklm-podcast-delivery-receipt.json"


def test_live_delivery_names_episodes_writes_index_and_is_idempotent(tmp_path: Path) -> None:
    ws, run_id, drive_root = _seed(tmp_path)
    mod = load_delivery_module()
    receipt = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=False, receipt_path=None, require_stage_receipt=True)
    assert receipt["status"] == "succeeded", receipt["blockers"]
    names = sorted(p.name for p in drive_root.iterdir())
    assert names == [
        "00a - TestConf 2030 Executive Readout (short) - show notes.md",
        "00a - TestConf 2030 Executive Readout (short).m4a",
        "01 - Alpha Topic Systems - show notes.md",
        "01 - Alpha Topic Systems.m4a",
        "02 - Beta Topic Models - show notes.md",
        "02 - Beta Topic Models.m4a",
        "index.md",
    ]
    index = (drive_root / "index.md").read_text(encoding="utf-8")
    assert "| 01 | Alpha Topic Systems |" in index and "`01 - Alpha Topic Systems.m4a`" in index
    assert "Created by Ada Byron." in index and "read end to end by a small language model" in index
    by_deck = {item["deck_id"]: item for item in receipt["items"]}
    assert by_deck[TOPIC_A]["audio_sha256"] == pn.sha256_file(ws / "audio" / TOPIC_A / "notebooklm_podcast" / "podcast.m4a")
    assert by_deck[EXEC_SHORT]["prefix"] == "00a"
    assert receipt["secrets_copied"] is False

    again = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=False, receipt_path=None, require_stage_receipt=True)
    assert again["status"] == "succeeded"
    assert {item["audio_action"] for item in again["items"]} == {"skipped_identical"}


def test_delivery_refuses_without_succeeded_stage_receipt(tmp_path: Path) -> None:
    ws, run_id, drive_root = _seed(tmp_path, stage_status="blocked")
    mod = load_delivery_module()
    receipt = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=False, receipt_path=None, require_stage_receipt=True)
    assert receipt["status"] == "blocked"
    assert receipt["blockers"][0]["code"] == "notebooklm_delivery_stage_receipt_not_succeeded"
    assert not drive_root.exists()
    waived = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=True, receipt_path=None, require_stage_receipt=False)
    assert waived["status"] == "planned" and waived["stage_receipt_waived"] is True


def test_missing_podcast_blocks_delivery_as_incomplete(tmp_path: Path) -> None:
    ws, run_id, drive_root = _seed(tmp_path)
    (ws / "audio" / TOPIC_B / "notebooklm_podcast" / "podcast.m4a").unlink()
    mod = load_delivery_module()
    receipt = mod.deliver(workspace=ws, run_id=run_id, drive_root=drive_root, dry_run=False, receipt_path=None, require_stage_receipt=True)
    assert receipt["status"] == "blocked"
    codes = [b["code"] for b in receipt["blockers"]]
    assert "notebooklm_delivery_audio_missing" in codes and "notebooklm_delivery_incomplete" in codes
    assert receipt["delivered_count"] == 2
    assert not (drive_root / "index.md").exists(), "no index while incomplete"


def test_cli_exit_codes(tmp_path: Path) -> None:
    ws, run_id, drive_root = _seed(tmp_path)
    mod = load_delivery_module()
    assert mod.main(["--workspace", str(ws), "--run-id", run_id, "--drive-root", str(drive_root), "--dry-run"]) == 0
    assert mod.main(["--workspace", str(ws), "--run-id", "missing-run", "--drive-root", str(drive_root)]) == 2


def test_safe_title_turns_colons_into_spaced_dashes() -> None:
    mod = load_delivery_module()
    assert mod.safe_title("Human-Centric Perception: Motion, Avatars") == "Human-Centric Perception - Motion, Avatars"
    assert mod.safe_title("A/B: C?") == "A-B - C"
