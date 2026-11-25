from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_CLI = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "review_storyboard.py"
_FIXTURES = (
    pathlib.Path(__file__).parent
    / "fixtures"
    / "presentations"
    / "phase16"
    / "storyboard_build_inputs.json"
)


def _fixture(key: str):
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))[key]


def test_dry_run_exits_zero_with_would_write(tmp_path):
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("valid_rows")), encoding="utf-8")
    out_json = tmp_path / "approval.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--storyboard-json",
            str(storyboard_file),
            "--output-json",
            str(out_json),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "would_write" in data


def test_clean_storyboard_approved_for_render_planning(tmp_path):
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("valid_rows")), encoding="utf-8")
    out_json = tmp_path / "approval.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--storyboard-json",
            str(storyboard_file),
            "--output-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["approval"]["status"] == "approved_for_render_planning"


def test_bad_named_source_contribution_exits_one(tmp_path):
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("bad_source_rows")), encoding="utf-8")
    out_json = tmp_path / "approval.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--storyboard-json",
            str(storyboard_file),
            "--output-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    err = json.loads(result.stderr)
    assert "findings" in err
