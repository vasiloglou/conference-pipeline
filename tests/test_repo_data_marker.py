"""v6.0 Phase 7 T5: the repo_data marker skips only when a named path is absent."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROBE = '''
import pytest

@pytest.mark.repo_data("pyproject.toml", "{absent}")
def test_marked_missing():
    assert True

@pytest.mark.repo_data("pyproject.toml", "tests/conftest.py")
def test_marked_present():
    assert True

def test_unmarked():
    assert True
'''

BARE = """import pytest


@pytest.mark.repo_data
def test_bare():
    pass
"""


def test_t5_marker_skips_naming_the_first_missing_path(tmp_path: Path) -> None:
    absent = "conferences/no-such-workspace/config.yaml"
    probe = REPO_ROOT / "tests" / f"_repo_data_probe_{tmp_path.name}.py"
    probe.write_text(PROBE.format(absent=absent), encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", str(probe)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        probe.unlink()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 1 skipped" in result.stdout, result.stdout
    assert f"repo_data: {absent} is not in this checkout" in result.stdout


def test_t5_marker_without_paths_is_a_usage_error(tmp_path: Path) -> None:
    probe = REPO_ROOT / "tests" / f"_repo_data_bare_{tmp_path.name}.py"
    probe.write_text(BARE, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(probe)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        probe.unlink()
    assert result.returncode != 0
    assert "repo_data needs at least one path" in result.stdout + result.stderr


TABLED = """
REPO_DATA = {"test_tabled_missing": ("pyproject.toml", "{absent}")}


def test_tabled_missing():
    assert True


def test_not_in_table():
    assert True
"""


def test_t5_a_module_table_feeds_the_same_rule(tmp_path: Path) -> None:
    absent = "conferences/no-such-workspace/storage/storage_manifest.json"
    probe = REPO_ROOT / "tests" / f"_repo_data_table_{tmp_path.name}.py"
    probe.write_text(TABLED.replace("{absent}", absent), encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs", str(probe)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        probe.unlink()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 skipped" in result.stdout, result.stdout
    assert f"repo_data: {absent} is not in this checkout" in result.stdout
