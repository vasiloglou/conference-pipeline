from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

_CLI = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "presentations" / "build_storyboard.py"
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
    clusters_file = tmp_path / "clusters.json"
    clusters_file.write_text(json.dumps({"clusters": _fixture("valid_clusters")}), encoding="utf-8")
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("valid_rows")), encoding="utf-8")
    out_md = tmp_path / "output.md"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--clusters-json",
            str(clusters_file),
            "--storyboard-json",
            str(storyboard_file),
            "--output-markdown",
            str(out_md),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "would_write" in data


def test_scaffold_prose_exits_one(tmp_path):
    clusters_file = tmp_path / "clusters.json"
    clusters_file.write_text(json.dumps({"clusters": _fixture("valid_clusters")}), encoding="utf-8")
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("scaffold_prose_rows")), encoding="utf-8")
    out_md = tmp_path / "output.md"
    out_json = tmp_path / "output.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--clusters-json",
            str(clusters_file),
            "--storyboard-json",
            str(storyboard_file),
            "--output-markdown",
            str(out_md),
            "--output-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    err = json.loads(result.stderr)
    assert "findings" in err


def test_valid_rows_exits_zero_and_writes_output(tmp_path):
    clusters_file = tmp_path / "clusters.json"
    clusters_file.write_text(json.dumps({"clusters": _fixture("valid_clusters")}), encoding="utf-8")
    storyboard_file = tmp_path / "storyboard.json"
    storyboard_file.write_text(json.dumps(_fixture("valid_rows")), encoding="utf-8")
    out_md = tmp_path / "output.md"
    out_json = tmp_path / "output.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "--clusters-json",
            str(clusters_file),
            "--storyboard-json",
            str(storyboard_file),
            "--output-markdown",
            str(out_md),
            "--output-json",
            str(out_json),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_md.exists()
    assert out_json.exists()
