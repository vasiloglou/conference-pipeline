from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT / "conferences" / "neurips2025"

# These tests assert the ON-DISK layout of the NeurIPS 2025 workspace: audio,
# exports, cache and legacy lanes that are gitignored and populated only in
# the main checkout. A fresh worktree has the tracked files and none of the
# lanes, so the layout under test is simply absent there.
pytestmark = pytest.mark.skipif(
    not (WORKSPACE / "audio").is_dir(),
    reason="NeurIPS 2025 on-disk storage lanes not present in this checkout",
)
CONFIG_PATH = WORKSPACE / "config.yaml"
MANIFEST_PATH = WORKSPACE / "storage" / "storage_manifest.json"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def is_git_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(path.relative_to(REPO_ROOT))],
        cwd=REPO_ROOT,
        check=False,
    )
    return result.returncode == 0


def test_workspace_layout_contract_uses_lane_based_structure():
    config = load_config()
    storage = config["storage"]
    layout = storage["workspace_layout"]

    assert storage["manifest_path"] == "storage/storage_manifest.json"
    assert "data_dir" not in storage
    assert "presentations_dir" not in storage
    assert "logs_dir" not in storage

    assert layout["analysis"] == "analysis"
    assert layout["decks"] == "decks"
    assert layout["audio"] == "audio"
    assert layout["exports"]["root"] == "exports"
    assert layout["exports"]["deep_dive"] == "exports/deep-dive"
    assert layout["exports"]["exhaustive_extractions"] == "exports/exhaustive-extractions"
    assert layout["agent_work"]["root"] == "agent-work"
    assert layout["cache"]["root"] == "cache"
    assert layout["legacy"] == "legacy"


def test_workspace_root_contains_new_lanes_and_support_material():
    expected_dirs = {
        "analysis",
        "decks",
        "audio",
        "exports",
        "agent-work",
        "cache",
        "legacy",
        "storage",
        "assets",
        "scripts",
    }
    expected_files = {"README.md", "config.yaml"}
    old_dirs = {
        "presentations",
        "podcasts",
        "export_for_deep_dive",
        "exhaustive_extractions",
        "claude-autonomy",
        "codex-autonomy",
        "gpt_researcher",
        "notebook-lm-findings",
        "vector_store",
        "experimental_videos",
    }

    root_entries = {path.name for path in WORKSPACE.iterdir()}
    unexpected_old_dirs = {
        name
        for name in root_entries & old_dirs
        if not is_git_ignored(WORKSPACE / name)
    }

    assert expected_dirs.issubset(root_entries)
    assert expected_files.issubset(root_entries)
    assert unexpected_old_dirs == set()


def test_manifest_paths_resolve_after_relayout():
    manifest = load_manifest()
    manifest_paths = {family["path"] for family in manifest["artifact_families"]}

    assert {"agent-work", "cache", "exports", "legacy"} == set(manifest["inventory_exemptions"])

    exemptions = set(manifest["inventory_exemptions"])
    for family in manifest["artifact_families"]:
        # Skip existence check for paths under inventory-exempted directories
        # (exports, cache, agent-work, legacy) — these are gitignored/generated
        # and will not be present in a clean checkout.
        path = family["path"]
        top_level = path.split("/")[0]
        if top_level in exemptions:
            continue
        # Only git-class families are present in a clean checkout; backup, cache and
        # public families are ignored and Drive-backed by design (v6.0 Phase 5, D3).
        if family["storage_class"] != "git":
            continue
        resolved = WORKSPACE / path
        assert resolved.exists(), path

    assert "decks" in manifest_paths
    assert "audio" in manifest_paths
    assert "agent-work/claude" in manifest_paths
    assert "agent-work/codex" in manifest_paths
    assert "cache/vector-store" in manifest_paths
    assert "exports/deep-dive" in manifest_paths
    assert "exports/exhaustive-extractions" in manifest_paths
    assert "legacy/codex/paper-influence-archive" in manifest_paths
