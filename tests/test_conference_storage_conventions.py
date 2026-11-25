from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFERENCE_README = REPO_ROOT / "conferences" / "README.md"
TEMPLATE_DIR = REPO_ROOT / "conferences" / "TEMPLATE"
FUTURE_STORAGE_DOC = REPO_ROOT / "docs" / "operations" / "future-conference-storage.md"

README_PATHS = (
    CONFERENCE_README,
    TEMPLATE_DIR / "README.md",
)
CONFIG_PATHS = (
    TEMPLATE_DIR / "config.yaml",
)
MANIFEST_PATHS = (
    TEMPLATE_DIR / "storage" / "storage_manifest.json",
)
EXPECTED_LANES = {
    "analysis",
    "decks",
    "audio",
    "exports",
    "agent-work",
    "cache",
    "legacy",
}
EXPECTED_FAMILY_PATHS = {
    "analysis",
    "decks",
    "audio",
    "exports/deep-dive",
    # The exhaustive extraction owner writes storage/extractions/<run-id>/; the
    # historical exports/exhaustive-extractions path is NeurIPS 2025 only.
    "storage/extractions",
    "exports/notebooklm_export",
    "agent-work/claude",
    "agent-work/codex",
    "agent-work/gpt-researcher",
    "agent-work/notebook-lm",
    "cache/vector-store",
    "cache/experimental-videos",
    "legacy",
}
OBSOLETE_STORAGE_KEYS = {"data_dir", "presentations_dir", "logs_dir"}
OBSOLETE_PATHS = {
    "presentations",
    "podcasts",
    "export_for_deep_dive",
    "exhaustive_extractions",
    "exports/exhaustive-extractions",
    "claude-autonomy",
    "codex-autonomy",
    "gpt_researcher",
    "notebook-lm-findings",
    "vector_store",
    "experimental_videos",
}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def test_shared_readmes_teach_lane_based_workspace_layout():
    for readme_path in README_PATHS:
        content = readme_path.read_text(encoding="utf-8")

        for lane in EXPECTED_LANES:
            assert lane in content

        assert "workspace_layout" in content or readme_path == TEMPLATE_DIR / "README.md"
        assert "storage/storage_manifest.json" in content or readme_path == TEMPLATE_DIR / "README.md"
        assert "`data/`" not in content
        assert "`presentations/`" not in content


def test_starter_configs_define_workspace_layout_and_manifest_path():
    for config_path in CONFIG_PATHS:
        config = load_config(config_path)
        storage = config["storage"]
        workspace_layout = storage["workspace_layout"]

        assert storage["manifest_path"] == "storage/storage_manifest.json"
        assert OBSOLETE_STORAGE_KEYS.isdisjoint(storage)
        assert workspace_layout["analysis"] == "analysis"
        assert workspace_layout["decks"] == "decks"
        assert workspace_layout["audio"] == "audio"
        assert workspace_layout["exports"]["root"] == "exports"
        assert workspace_layout["exports"]["deep_dive"] == "exports/deep-dive"
        assert workspace_layout["exports"]["exhaustive_extractions"] == "exports/exhaustive-extractions"
        assert workspace_layout["exports"]["notebooklm"] == "exports/notebooklm_export"
        assert workspace_layout["agent_work"]["root"] == "agent-work"
        assert workspace_layout["cache"]["root"] == "cache"
        assert workspace_layout["legacy"] == "legacy"
        assert storage["support_material"]["storage"] == "storage"
        assert storage["support_material"]["assets"] == "assets"
        assert storage["support_material"]["scripts"] == "scripts"


def test_future_conference_docs_define_support_material_root_contract():
    content = FUTURE_STORAGE_DOC.read_text(encoding="utf-8")

    assert "support_material.storage" in content
    assert "`storage/storage_manifest.json`" in content
    assert normalize_whitespace(
        "`support_material.storage` is the root for `inventory.json`, "
        "`prune-verification/` evidence, and related cleanup receipts."
    ) in normalize_whitespace(content)
    assert "starter default remains `storage`" in content


def test_starter_manifests_are_sync_ready_and_lane_based():
    for manifest_path in MANIFEST_PATHS:
        manifest = load_manifest(manifest_path)
        manifest_paths = {family["path"] for family in manifest["artifact_families"] if "path" in family}

        assert manifest["schema_version"] == 1
        assert manifest["artifact_families"]
        assert EXPECTED_FAMILY_PATHS.issubset(manifest_paths)
        assert manifest_paths.isdisjoint(OBSOLETE_PATHS)
        assert {"agent-work", "cache", "exports", "legacy"} == set(manifest["inventory_exemptions"])
