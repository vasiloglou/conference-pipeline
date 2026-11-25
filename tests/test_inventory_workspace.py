from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def load_inventory_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "inventory_workspace.py"
    spec = importlib.util.spec_from_file_location("inventory_workspace", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_build_inventory_reports_expected_fields(tmp_path: Path):
    module = load_inventory_module()
    workspace = tmp_path / "workspace"
    write_bytes(workspace / "analysis" / "report.md", 2048)
    write_bytes(workspace / "presentations" / "deck.md", 4096)
    write_bytes(workspace / "vector_store" / "index.bin", 1024)
    write_bytes(workspace / "mystery_lab" / "notes.txt", 512)

    inventory = module.build_inventory(workspace, min_hotspot_mb=0.0001)

    assert inventory["workspace"] == "workspace"
    assert inventory["generated_at"]
    assert inventory["inventory_tool_version"]

    families = {item["path"]: item for item in inventory["artifact_families"]}
    assert "analysis" in families
    assert "presentations" in families
    assert "vector_store" in families

    analysis = families["analysis"]
    assert analysis["hotspots"]
    assert analysis["size_bytes"] > 0
    assert analysis["purpose"]
    assert analysis["producer"]
    assert analysis["classification_status"] == "classified"

    mystery = families["mystery_lab"]
    assert mystery["classification_status"] == "unclassified"


def test_excluded_patterns_and_symlinks_are_skipped(tmp_path: Path):
    module = load_inventory_module()
    workspace = tmp_path / "workspace"
    write_bytes(workspace / "analysis" / "report.md", 256)
    write_bytes(workspace / ".env", 128)
    write_bytes(workspace / "signing.key", 128)

    external_dir = tmp_path / "external"
    write_bytes(external_dir / "huge.bin", 8192)
    os.symlink(external_dir, workspace / "linked_dir")

    inventory = module.build_inventory(workspace, min_hotspot_mb=0.0001)
    family_paths = {item["path"] for item in inventory["artifact_families"]}

    assert ".env" not in family_paths
    assert "signing.key" not in family_paths
    assert "linked_dir" not in family_paths


def test_markdown_output_includes_table_and_inventory_reference(tmp_path: Path):
    module = load_inventory_module()
    workspace = tmp_path / "workspace"
    output_dir = tmp_path / "output"
    write_bytes(workspace / "analysis" / "report.md", 1024)

    inventory = module.build_inventory(workspace, min_hotspot_mb=0.0001)
    _, markdown_path = module.write_inventory_reports(inventory, output_dir)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "## Major Artifact Families" in markdown
    assert "inventory.json" in markdown
    assert "| `analysis` |" in markdown


def test_cli_accepts_conference_dir_and_explicit_outputs(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "inventory_workspace.py"
    workspace = tmp_path / "iclr2025"
    output_json = workspace / "storage" / "inventory.json"
    output_md = workspace / "storage" / "inventory.md"
    write_bytes(workspace / "analysis" / "report.md", 1024)
    (workspace / "config.yaml").write_text(
        'conference:\n  slug: "iclr2025"\n  name: "ICLR 2025"\n',
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--conference-dir",
            str(workspace),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert output_json.is_file()
    assert output_md.is_file()
    assert "ICLR 2025" in output_md.read_text(encoding="utf-8")


def test_cli_excludes_only_exact_output_paths(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent
    script_path = repo_root / "scripts" / "inventory_workspace.py"
    workspace = tmp_path / "iclr2025"
    output_json = workspace / "storage" / "inventory.json"
    output_md = workspace / "storage" / "inventory.md"
    write_bytes(workspace / "analysis" / "inventory.json", 1024)
    write_bytes(output_json, 2048)
    write_bytes(output_md, 2048)
    (workspace / "config.yaml").write_text(
        'conference:\n  slug: "iclr2025"\n  name: "ICLR 2025"\n',
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--conference-dir",
            str(workspace),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    inventory = json.loads(output_json.read_text(encoding="utf-8"))
    families = {item["path"]: item for item in inventory["artifact_families"]}

    assert families["analysis"]["size_bytes"] == 1024
    assert families["storage"]["size_bytes"] == 0
