#!/usr/bin/env python3
"""Generate a storage inventory snapshot for a conference workspace."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INVENTORY_TOOL_VERSION = "1.0"
DEFAULT_EXACT_EXCLUDES = {
    ".DS_Store",
    ".env",
    ".git",
    "__pycache__",
}
DEFAULT_GLOB_EXCLUDES = [
    ".env.*",
    "*.key",
]
DEFAULT_HOTSPOT_MB = 10.0

KNOWN_FAMILY_METADATA: dict[str, dict[str, str]] = {
    "README.md": {
        "purpose": "Workspace-level notes and historical operating guidance.",
        "producer": "manual-docs",
    },
    "analysis": {
        "purpose": "Conference analysis notes and derived writeups.",
        "producer": "analysis-workflow",
    },
    "agent-work": {
        "purpose": "Active autonomy and research sidecars grouped by tool.",
        "producer": "agent-workflow",
    },
    "audio": {
        "purpose": "Podcast scripts, audio renders, and other release-oriented audio outputs.",
        "producer": "podcast-pipeline",
    },
    "assets": {
        "purpose": "Static workspace assets and supporting reference material.",
        "producer": "manual-curation",
    },
    "cache": {
        "purpose": "Rebuildable caches and preview artifacts for active work.",
        "producer": "cache-workflow",
    },
    "claude-autonomy": {
        "purpose": "Autonomy outputs generated with Claude-driven workflows.",
        "producer": "claude-autonomy",
    },
    "codex-autonomy": {
        "purpose": "Autonomy outputs generated with Codex-driven workflows.",
        "producer": "codex-autonomy",
    },
    "config.yaml": {
        "purpose": "Conference-local configuration contract.",
        "producer": "manual-config",
    },
    "decks": {
        "purpose": "Presentation decks, scripts, figures, and rendered outputs.",
        "producer": "presentation-pipeline",
    },
    "exhaustive_extractions": {
        "purpose": "Exhaustive extraction runs and normalized extraction outputs.",
        "producer": "exhaustive-extraction",
    },
    "experimental_videos": {
        "purpose": "Experimental or preview video outputs that are not primary deliverables.",
        "producer": "video-experiments",
    },
    "export_for_deep_dive": {
        "purpose": "Prepared export bundles for deep-dive analysis workflows.",
        "producer": "deep-dive-export",
    },
    "exports": {
        "purpose": "Private export bundles and other downstream handoff packages.",
        "producer": "export-workflow",
    },
    "gpt_researcher": {
        "purpose": "GPT Researcher reports, cache, and supporting artifacts.",
        "producer": "gpt-researcher",
    },
    "legacy": {
        "purpose": "Explicit archive area for superseded or legacy conference artifacts.",
        "producer": "legacy-archive",
    },
    "notebook-lm-findings": {
        "purpose": "NotebookLM synthesized notes and downstream findings.",
        "producer": "notebooklm",
    },
    "podcasts": {
        "purpose": "Podcast scripts, audio renders, and packaging outputs.",
        "producer": "podcast-pipeline",
    },
    "presentations": {
        "purpose": "Presentation decks, scripts, figures, and rendering outputs.",
        "producer": "presentation-pipeline",
    },
    "scripts": {
        "purpose": "Conference-specific helper, debug, and maintenance scripts.",
        "producer": "manual-scripts",
    },
    "storage": {
        "purpose": "Workspace storage_manifest.json, policy, inventory, and verification receipts.",
        "producer": "storage-governance",
    },
    "vector_store": {
        "purpose": "Vector indexes and retrieval cache artifacts for analysis tools.",
        "producer": "vector-store-setup",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def human_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def sanitize_identifier(name: str) -> str:
    chars = []
    for char in name.lower():
        chars.append(char if char.isalnum() else "_")
    identifier = "".join(chars).strip("_")
    while "__" in identifier:
        identifier = identifier.replace("__", "_")
    return identifier or "artifact"


def is_excluded(name: str, exact_excludes: set[str], glob_excludes: list[str]) -> bool:
    if name in exact_excludes:
        return True
    return any(fnmatch.fnmatch(name, pattern) for pattern in glob_excludes)


def repo_relative_path(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def display_workspace_path(workspace: Path) -> str:
    candidates = [Path.cwd().resolve(), Path(__file__).resolve().parent.parent]
    for base in candidates:
        try:
            return workspace.relative_to(base).as_posix()
        except ValueError:
            continue
    return workspace.name


def read_conference_name(workspace: Path) -> str | None:
    config_path = workspace / "config.yaml"
    if not config_path.is_file():
        return None

    in_conference_section = False
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            in_conference_section = stripped == "conference:"
            continue
        if in_conference_section and stripped.startswith("name:"):
            return stripped.split(":", 1)[1].strip().strip("'\"") or None
    return None


def is_excluded_path(path: Path, excluded_paths: set[Path]) -> bool:
    return path.resolve() in excluded_paths


def path_size(
    path: Path,
    exact_excludes: set[str],
    glob_excludes: list[str],
    excluded_paths: set[Path] | None = None,
) -> int:
    excluded_paths = excluded_paths or set()
    if is_excluded_path(path, excluded_paths):
        return 0
    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        dirs[:] = [
            name
            for name in dirs
            if not is_excluded(name, exact_excludes, glob_excludes)
            and not is_excluded_path(root_path / name, excluded_paths)
            and not (root_path / name).is_symlink()
        ]
        for file_name in files:
            if is_excluded(file_name, exact_excludes, glob_excludes):
                continue
            file_path = root_path / file_name
            if is_excluded_path(file_path, excluded_paths) or file_path.is_symlink():
                continue
            total += file_path.stat().st_size
    return total


def list_hotspots(
    path: Path,
    workspace: Path,
    threshold_bytes: int,
    exact_excludes: set[str],
    glob_excludes: list[str],
    excluded_paths: set[Path] | None = None,
) -> list[dict[str, Any]]:
    if not path.is_dir():
        return []

    excluded_paths = excluded_paths or set()
    hotspots = []
    for child in sorted(path.iterdir()):
        if (
            is_excluded(child.name, exact_excludes, glob_excludes)
            or is_excluded_path(child, excluded_paths)
            or child.is_symlink()
        ):
            continue
        child_size = path_size(child, exact_excludes, glob_excludes, excluded_paths)
        if child_size < threshold_bytes:
            continue
        hotspots.append(
            {
                "path": repo_relative_path(child, workspace),
                "size_bytes": child_size,
                "size_human": human_size(child_size),
                "kind": "directory" if child.is_dir() else "file",
            }
        )
    hotspots.sort(key=lambda item: item["size_bytes"], reverse=True)
    return hotspots


def family_metadata(entry_name: str) -> dict[str, str]:
    if entry_name in KNOWN_FAMILY_METADATA:
        metadata = KNOWN_FAMILY_METADATA[entry_name]
        return {
            "purpose": metadata["purpose"],
            "producer": metadata["producer"],
            "classification_status": "classified",
        }
    return {
        "purpose": "Unclassified workspace artifact family that needs operator review.",
        "producer": "unknown",
        "classification_status": "unclassified",
    }


def build_inventory(
    workspace: Path,
    min_hotspot_mb: float = DEFAULT_HOTSPOT_MB,
    extra_excludes: list[str] | None = None,
    conference_name: str | None = None,
    excluded_paths: set[Path] | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    excluded_paths = {path.resolve() for path in (excluded_paths or set())}
    exact_excludes = set(DEFAULT_EXACT_EXCLUDES)
    glob_excludes = list(DEFAULT_GLOB_EXCLUDES)
    for extra in extra_excludes or []:
        if any(token in extra for token in "*?[]"):
            glob_excludes.append(extra)
        else:
            exact_excludes.add(extra)

    threshold_bytes = max(0, int(min_hotspot_mb * 1024 * 1024))
    artifact_families: list[dict[str, Any]] = []

    for entry in sorted(workspace.iterdir()):
        if (
            is_excluded(entry.name, exact_excludes, glob_excludes)
            or is_excluded_path(entry, excluded_paths)
            or entry.is_symlink()
        ):
            continue
        size_bytes = path_size(entry, exact_excludes, glob_excludes, excluded_paths)
        metadata = family_metadata(entry.name)
        artifact_families.append(
            {
                "id": sanitize_identifier(entry.name),
                "path": repo_relative_path(entry, workspace),
                "kind": "directory" if entry.is_dir() else "file",
                "size_bytes": size_bytes,
                "size_human": human_size(size_bytes),
                "purpose": metadata["purpose"],
                "producer": metadata["producer"],
                "classification_status": metadata["classification_status"],
                "hotspots": list_hotspots(
                    entry,
                    workspace,
                    threshold_bytes,
                    exact_excludes,
                    glob_excludes,
                    excluded_paths,
                ),
            }
        )

    artifact_families.sort(key=lambda item: item["size_bytes"], reverse=True)

    return {
        "schema_version": 1,
        "inventory_tool_version": INVENTORY_TOOL_VERSION,
        "generated_at": utc_now(),
        "workspace": display_workspace_path(workspace),
        "conference_name": conference_name or read_conference_name(workspace),
        "hotspot_threshold_mb": min_hotspot_mb,
        "excluded_names": sorted(exact_excludes),
        "excluded_globs": sorted(glob_excludes),
        "artifact_families": artifact_families,
    }


def render_hotspots(hotspots: list[dict[str, Any]]) -> str:
    if not hotspots:
        return "None above threshold"
    return "; ".join(f"`{item['path']}` ({item['size_human']})" for item in hotspots[:3])


def render_inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Workspace Inventory",
        "",
        f"- Workspace: `{inventory['workspace']}`",
    ]
    if inventory.get("conference_name"):
        lines.append(f"- Conference: `{inventory['conference_name']}`")
    lines.extend(
        [
            f"- Generated: `{inventory['generated_at']}`",
            f"- Tool version: `{inventory['inventory_tool_version']}`",
            f"- Hotspot threshold: `{inventory['hotspot_threshold_mb']:.1f} MB`",
            "- Machine-readable companion: `inventory.json`",
            "",
            "## Major Artifact Families",
            "",
            "## Top Family Hotspots",
            "",
            "| Path | Kind | Size | Purpose | Producer | Status | Representative Hotspots |",
            "|------|------|------|---------|----------|--------|-------------------------|",
        ]
    )

    for family in inventory["artifact_families"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{family['path']}`",
                    family["kind"],
                    family["size_human"],
                    family["purpose"],
                    family["producer"],
                    family["classification_status"],
                    render_hotspots(family["hotspots"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Snapshot Discipline",
            "",
            "Refresh this snapshot when artifact families, storage decisions, or materially significant hotspot structure changes.",
            "Do not recommit it for routine size drift alone.",
            "",
        ]
    )
    return "\n".join(lines)


def write_inventory_reports(
    inventory: dict[str, Any],
    output_dir: Path | None = None,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> tuple[Path, Path]:
    if output_dir is None and (output_json is None or output_md is None):
        raise ValueError("Provide output_dir or both output_json and output_md")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if output_json is None:
        assert output_dir is not None
        output_json = output_dir / "inventory.json"
    if output_md is None:
        assert output_dir is not None
        output_md = output_dir / "inventory.md"
    json_path = output_json
    markdown_path = output_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_inventory_markdown(inventory), encoding="utf-8")
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory a conference workspace for storage planning.")
    parser.add_argument("--workspace", "--conference-dir", dest="workspace", type=Path, help="Workspace directory to inventory.")
    parser.add_argument("--output-dir", type=Path, help="Directory for inventory outputs.")
    parser.add_argument("--output-json", type=Path, help="Path for machine-readable inventory JSON.")
    parser.add_argument("--output-md", type=Path, help="Path for human-readable inventory Markdown.")
    parser.add_argument(
        "--min-hotspot-mb",
        type=float,
        default=DEFAULT_HOTSPOT_MB,
        help="Minimum hotspot size to include in reports.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra exact names or glob patterns to exclude. Repeatable.",
    )
    args = parser.parse_args()
    if args.workspace is None:
        parser.error("one of --workspace or --conference-dir is required")
    if args.output_dir is None and (args.output_json is None or args.output_md is None):
        parser.error("provide --output-dir or both --output-json and --output-md")
    return args


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None
    output_json = args.output_json.resolve() if args.output_json else None
    output_md = args.output_md.resolve() if args.output_md else None

    if not workspace.exists() or not workspace.is_dir():
        raise SystemExit(f"Workspace does not exist or is not a directory: {args.workspace}")

    planned_output_json = output_json or (output_dir / "inventory.json" if output_dir else None)
    planned_output_md = output_md or (output_dir / "inventory.md" if output_dir else None)
    planned_output_paths = {
        output_path.resolve()
        for output_path in [planned_output_json, planned_output_md]
        if output_path is not None
    }

    inventory = build_inventory(
        workspace,
        min_hotspot_mb=args.min_hotspot_mb,
        extra_excludes=list(args.exclude),
        excluded_paths=planned_output_paths,
    )
    json_path, markdown_path = write_inventory_reports(
        inventory,
        output_dir=output_dir,
        output_json=output_json,
        output_md=output_md,
    )
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
