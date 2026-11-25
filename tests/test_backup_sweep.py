from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conference_pipeline import workflow
from conference_pipeline.control_plane.artifacts import scheduled_backup_status


REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_to_gdrive.py"
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "run_scheduled_backup_sweep.py"


def _load_last_json(stdout: str) -> dict[str, object]:
    start = stdout.rfind('{\n  "cadence_label"')
    if start < 0:
        start = stdout.rfind("{\n")
    assert start >= 0, stdout
    payload = json.loads(stdout[start:])
    assert isinstance(payload, dict)
    return payload


def _write_workspace(root: Path, slug: str = "sample2027", conference_name: str = "Sample Conf 2027") -> Path:
    workspace = root / "conferences" / slug
    (workspace / "storage").mkdir(parents=True)
    (workspace / "analysis").mkdir(parents=True)
    (workspace / "analysis" / "sample.json").write_text('{"ok": true}\n', encoding="utf-8")
    (workspace / "config.yaml").write_text(
        (
            "conference:\n"
            f"  slug: \"{slug}\"\n"
            f"  name: \"{conference_name}\"\n"
            "storage:\n"
            "  manifest_path: \"storage/storage_manifest.json\"\n"
        ),
        encoding="utf-8",
    )
    (workspace / "storage" / "storage_manifest.json").write_text(
        json.dumps(
            {
                "artifact_families": [
                    {
                        "id": "analysis",
                        "path": "analysis",
                        "sync_profiles": {"public": False, "backup": True},
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace


def test_sync_to_gdrive_writes_machine_readable_receipt(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    target_root = tmp_path / "drive"
    target_root.mkdir()
    (target_root / "Sample Conf 2027-raw").mkdir()
    receipt_path = workspace / "storage" / "workflow-runs" / "phase-test" / "backup" / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SYNC_SCRIPT),
            "--conference-dir",
            str(workspace),
            "--profile",
            "backup",
            "--dry-run",
            "--remote-path",
            str(target_root / "Sample Conf 2027-raw"),
            "--receipt-path",
            str(receipt_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "succeeded"
    assert receipt["profile"] == "backup"
    assert receipt["dry_run_exit_code"] == 0
    assert receipt["write_exit_code"] is None
    assert receipt["digest_count"] > 0
    assert receipt["secrets_copied"] is False
    assert receipt["secret_like_paths"] == []


def test_sync_to_gdrive_refuses_secret_like_files(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    (workspace / "analysis" / ".env").write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
    target_root = tmp_path / "drive"
    target_root.mkdir()
    receipt_path = workspace / "storage" / "workflow-runs" / "phase-test" / "backup" / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            str(SYNC_SCRIPT),
            "--conference-dir",
            str(workspace),
            "--profile",
            "backup",
            "--dry-run",
            "--remote-path",
            str(target_root / "Sample Conf 2027-raw"),
            "--receipt-path",
            str(receipt_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "Refusing to sync secret-like files" in receipt["error"]
    assert "analysis/.env" in receipt["secret_like_paths"]
    assert receipt["secrets_copied"] is False


def test_scheduled_backup_sweep_writes_workspace_summary(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)
    target_root = tmp_path / "drive"
    target_root.mkdir()
    (target_root / "Sample Conf 2027-raw").mkdir()
    stale_write_path = workspace / "storage" / "workflow-runs" / "scheduled-backup" / "write-receipt.json"
    stale_write_path.parent.mkdir(parents=True, exist_ok=True)
    stale_write_path.write_text(
        json.dumps({"status": "succeeded", "digest_count": 99, "included_families": ["stale"]}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--workspace",
            str(workspace),
            "--target-root",
            str(target_root),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    summary_path = workspace / "storage" / "workflow-runs" / "scheduled-backup" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["workspace"] == str(workspace.resolve())
    assert summary["profile"] == "backup"
    assert summary["dry_run_exit_code"] == 0
    assert summary["write_exit_code"] is None
    assert summary["target"].endswith("Sample Conf 2027-raw")
    assert summary["digest_count"] == 1


def test_scheduled_backup_sweep_requires_target_root(tmp_path: Path) -> None:
    _write_workspace(tmp_path, "sample2027", "Sample Conf 2027")

    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--conferences-root",
            str(tmp_path / "conferences"),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    payload = _load_last_json(result.stdout)
    assert payload["status"] == "failed"
    assert "without --target-root" in payload["error"]


def test_scheduled_backup_sweep_skips_template_workspace(tmp_path: Path) -> None:
    _write_workspace(tmp_path, "sample2027", "Sample Conf 2027")
    template = _write_workspace(tmp_path, "TEMPLATE", "Template")
    target_root = tmp_path / "drive"
    target_root.mkdir()
    (target_root / "Sample Conf 2027-raw").mkdir()
    (target_root / "Template-raw").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SWEEP_SCRIPT),
            "--conferences-root",
            str(tmp_path / "conferences"),
            "--target-root",
            str(target_root),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    payload = _load_last_json(result.stdout)
    assert payload["workspace_count"] == 1
    assert all(item["workspace"] != str(template.resolve()) for item in payload["workspaces"])


def test_control_plane_reports_latest_scheduled_backup_status(tmp_path: Path) -> None:
    workspace_path = _write_workspace(tmp_path)
    summary_path = workspace_path / "storage" / "workflow-runs" / "scheduled-backup" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "generated_at": "2026-04-30T12:00:00Z",
                "included_families": ["analysis"],
                "skipped_families": [],
                "digest_count": 4,
                "target": "/Drive/sample2027-raw",
                "secrets_copied": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure_path = workspace_path / "storage" / "workflow-runs" / "scheduled-backup" / "dry-run-receipt.json"
    failure_path.write_text(
        json.dumps(
            {
                "status": "failed",
                "generated_at": "2026-04-30T13:00:00Z",
                "included_families": [],
                "skipped_families": ["analysis"],
                "digest_count": 0,
                "target": "/Drive/sample2027-raw",
                "secrets_copied": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolved = workflow.resolve_workspace(workspace_path)

    status = scheduled_backup_status(resolved)

    assert status["status"] == "failed"
    assert status["receipt_path"] == "storage/workflow-runs/scheduled-backup/dry-run-receipt.json"
    assert status["last_successful_run"] == "2026-04-30T12:00:00Z"
    assert status["last_successful_receipt_path"] == "storage/workflow-runs/scheduled-backup/summary.json"
    assert status["last_failure"] == "2026-04-30T13:00:00Z"
    assert status["last_failure_receipt_path"] == "storage/workflow-runs/scheduled-backup/dry-run-receipt.json"
    assert status["digest_count"] == 0
