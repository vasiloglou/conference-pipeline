"""v6.0 Phase 7.1, PR 49 review pr-49-r2: one regression test per finding (F1-F3)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conference_pipeline.publishing.dataset_package import DatasetPackageError
from tests.test_hf_release_tree import FakeApi
from tests.test_phase71_review_r1 import _listed_bundle
from tests.test_pr49_review_r1 import _publish

MARKER = "codename-zephyr-fixture"


# --- F1: a hub file over the text-scan size still meets the env file's markers ---------


@pytest.mark.parametrize("size", [64, 2 * 1024 * 1024 + 1])
def test_f1_hub_env_markers_apply_to_large_files_too(tmp_path: Path, monkeypatch, size: int) -> None:
    from conference_pipeline.publishing.release_hub import hub_findings
    from tests.test_phase71_review_r3 import _hub

    monkeypatch.delenv("CONFERENCE_PIPELINE_PRIVATE_MARKERS", raising=False)
    hub = _hub(tmp_path)
    body = (MARKER + "\n").encode()
    (hub / "notes.txt").write_bytes(body + b"x" * max(0, size - len(body)))
    found = hub_findings(hub, secret_scanner=lambda paths: [], environ={"CONFERENCE_PIPELINE_PRIVATE_MARKERS": MARKER})
    assert any(f.startswith("private_marker: notes.txt") for f in found), found


# --- F2: GitHub refuses a present but invalid bundle id before anything is pushed --------


def _runner(calls: list):
    def runner(command, **kwargs):
        calls.append(command)
        if command[:3] == ["gh", "repo", "view"]:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        if "show" in command:
            return subprocess.CompletedProcess(command, 0, json.dumps({"bundle_id": "x-v1"}), "")
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, "0" * 40 + "\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    return runner


@pytest.mark.parametrize("bundle_id", ["x-v2.lock", "x..v2", "x-v2\n", 7])
def test_f2_github_refuses_an_invalid_bundle_id_before_any_push(tmp_path: Path, bundle_id) -> None:
    from conference_pipeline.publishing.public_repo import publish_github_repo

    target = tmp_path / "repo"
    target.mkdir()
    (target / "RELEASE.json").write_text(json.dumps({"bundle_id": bundle_id}), encoding="utf-8")
    calls: list = []
    result = publish_github_repo(target, "o/x-data", runner=_runner(calls))
    assert result["status"] != "succeeded"
    assert result["blocked_reason"] == "github_release_id_invalid"
    assert not any("push" in command for command in calls)


def test_f2_a_valid_bundle_id_still_publishes(tmp_path: Path) -> None:
    from conference_pipeline.publishing.public_repo import publish_github_repo

    target = tmp_path / "repo"
    target.mkdir()
    (target / "RELEASE.json").write_text(json.dumps({"bundle_id": "x-v2"}), encoding="utf-8")
    calls: list = []
    result = publish_github_repo(target, "o/x-data", runner=_runner(calls))
    assert result.get("blocked_reason") != "github_release_id_invalid", result
    assert ["git", "-C", str(target), "push", "origin", "refs/tags/x-v2"] in calls


# --- F3: the tag rule matches the whole string ------------------------------------------


@pytest.mark.parametrize("tag", ["x-v2\n", "x-v2\r", "x-v2\n\n", "\nx-v2"])
def test_f3_the_tag_rule_refuses_trailing_and_leading_newlines(tag: str) -> None:
    from conference_pipeline.publishing.release_layout import valid_tag

    git_ok = subprocess.run(["git", "check-ref-format", f"refs/tags/{tag}"], capture_output=True).returncode == 0
    assert git_ok is False
    assert valid_tag(tag) is False


def test_f3_hf_refuses_a_newline_bundle_id_before_any_mutation(tmp_path: Path) -> None:
    bundle = _listed_bundle(tmp_path, bundle_id="x-v2\n")
    api = FakeApi()
    with pytest.raises(DatasetPackageError, match="tag"):
        _publish(bundle, api)
    assert api.calls == []
