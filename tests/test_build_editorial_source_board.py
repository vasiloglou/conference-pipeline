from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_CLI = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "build_editorial_source_board.py"
_FIXTURES = (
    pathlib.Path(__file__).parent
    / "fixtures"
    / "presentations"
    / "phase16"
    / "source_board_build_inputs.json"
)


def _fixture(key: str) -> dict:
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))[key]


def test_dry_run_exits_zero_with_would_write(tmp_path):
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(_fixture("valid_payload")), encoding="utf-8")
    out_json = tmp_path / "output.json"
    clusters_json = tmp_path / "clusters.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--source-board-json",
            str(payload_file),
            "--output-json",
            str(out_json),
            "--clusters-json",
            str(clusters_json),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "would_write" in data


def test_blocked_contribution_exits_one_with_findings(tmp_path):
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(_fixture("blocked_payload")), encoding="utf-8")
    out_json = tmp_path / "output.json"
    clusters_json = tmp_path / "clusters.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--source-board-json",
            str(payload_file),
            "--output-json",
            str(out_json),
            "--clusters-json",
            str(clusters_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    err = json.loads(result.stderr)
    assert "findings" in err


def test_valid_input_exits_zero_and_writes_output(tmp_path):
    payload_file = tmp_path / "payload.json"
    payload_file.write_text(json.dumps(_fixture("valid_payload")), encoding="utf-8")
    out_json = tmp_path / "output.json"
    clusters_json = tmp_path / "clusters.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--source-board-json",
            str(payload_file),
            "--output-json",
            str(out_json),
            "--clusters-json",
            str(clusters_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_json.exists()
    assert clusters_json.exists()
