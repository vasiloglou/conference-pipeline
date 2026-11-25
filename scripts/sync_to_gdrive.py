#!/usr/bin/env python3
"""Conference-aware Google Drive sync helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROFILE_ENV_VARS = {
    "public": "GDRIVE_PUBLIC_PATH",
}
PROFILE_CHOICES = ("backup", "public")
DEFAULT_EXCLUDES = [
    ".DS_Store",
    "__pycache__/",
    "*.pyc",
    ".git/",
]
SECRET_EXCLUDES = [
    ".env",
    ".env.*",
    "cookies.json",
    "cookie.json",
    "*cookie*",
    "token.json",
    "tokens.json",
    "*access_token*",
    "*refresh_token*",
    "*auth_token*",
    "*bearer_token*",
    "*session_token*",
    "*id_token*",
    "*secret*",
    "*password*",
    "*api_key*",
    "*apikey*",
    "*credential*",
]
DEFAULT_MANIFEST_PATH = Path("storage/storage_manifest.json")
ERROR_NO_DESTINATION = "No destination configured for profile"
ERROR_NO_MANIFEST = "No storage manifest found for conference. Sync aborted for safety."
ERROR_DELETE_REQUIRES_REVIEW = "Refusing destructive sync without reviewed dry run"
ERROR_DELETE_RECEIPT_MISMATCH = "Refusing destructive sync because reviewed dry-run receipt does not match current plan"
ERROR_REMOTE_ROOT_MISSING = "Remote root does not exist:"
ERROR_SECRET_LIKE_FILES = "Refusing to sync secret-like files"
ERROR_BACKUP_REQUIRES_REMOTE_PATH = (
    "Backup sync requires an explicit --remote-path for the conference Drive leaf"
)


@dataclass(frozen=True)
class SyncConfig:
    conference_dir: Path
    config_path: Path
    profile: str
    remote_path: Path
    dry_run: bool
    delete: bool
    quiet: bool
    reviewed_dry_run: bool
    init_only: bool
    reviewed_dry_run_receipt: Path | None
    receipt_path: Path | None = None


@dataclass(frozen=True)
class ManifestFamily:
    family_id: str
    relative_path: Path | None
    path_glob: str | None
    sync_profiles: Mapping[str, bool]
    storage_class: str | None = None
    share_policy: str | None = None


@dataclass(frozen=True)
class SyncItem:
    family_id: str
    relative_path: Path
    source_path: Path
    destination_path: Path
    is_directory: bool


@dataclass(frozen=True)
class SyncPlan:
    config: SyncConfig
    manifest_path: Path
    families: tuple[ManifestFamily, ...]
    included_families: tuple[str, ...]
    excluded_families: tuple[str, ...]
    sync_items: tuple[SyncItem, ...]


@dataclass(frozen=True)
class DestinationStatus:
    target_root: Path
    target_root_status: str
    leaf_status: str


def get_local_size(path: Path) -> str:
    """Return a human-readable size for a directory."""
    try:
        result = subprocess.run(
            ["du", "-sh", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.split()[0]
    except Exception:
        pass
    return "unknown"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync a conference workspace to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/sync_to_gdrive.py --conference-dir conferences/neurips2025 --profile public --dry-run
    python3 scripts/sync_to_gdrive.py --config conferences/neurips2025/config.yaml --profile backup
    python3 scripts/sync_to_gdrive.py --conference-dir conferences/iclr2025 --profile public --remote-path /mounted/Drive/ICLR-2025
        """,
    )
    locator_group = parser.add_mutually_exclusive_group(required=True)
    locator_group.add_argument(
        "--conference-dir",
        type=Path,
        help="Path to a conference workspace directory such as conferences/neurips2025",
    )
    locator_group.add_argument(
        "--config",
        type=Path,
        help="Path to a conference config.yaml such as conferences/neurips2025/config.yaml",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        required=True,
        help="Sync profile to use: public or backup",
    )
    parser.add_argument(
        "--remote-path",
        type=Path,
        help="Override the resolved destination path for the selected profile",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without mutating the destination",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete remote files inside each included family destination that do not exist locally",
    )
    parser.add_argument(
        "--reviewed-dry-run",
        action="store_true",
        help="Acknowledge that the dry-run output has been reviewed before --delete",
    )
    parser.add_argument(
        "--reviewed-dry-run-receipt",
        type=Path,
        help=(
            "Path to a successful --dry-run --delete receipt to bind destructive "
            "--delete to the same manifest, target, and artifact snapshot."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress rsync progress output",
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help=(
            "Create the selected conference leaf/family directories on the mounted Drive "
            "target without copying artifact payloads."
        ),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help="Write a machine-readable JSON receipt for this sync attempt",
    )
    return parser.parse_args(argv)


def resolve_conference_paths(
    conference_dir: Path | None,
    config_path: Path | None,
) -> tuple[Path, Path]:
    """Resolve the workspace root and config path from one explicit locator."""
    if config_path is not None:
        resolved_config = config_path.resolve()
        if not resolved_config.exists() or not resolved_config.is_file():
            raise ValueError(f"Conference config does not exist: {resolved_config}")
        return resolved_config.parent, resolved_config

    assert conference_dir is not None
    resolved_dir = conference_dir.resolve()
    if not resolved_dir.exists():
        raise ValueError(f"Conference directory does not exist: {resolved_dir}")

    resolved_config = resolved_dir / "config.yaml"
    if not resolved_config.exists() or not resolved_config.is_file():
        raise ValueError(f"Conference config does not exist: {resolved_config}")
    return resolved_dir, resolved_config


def resolve_destination_path(
    profile: str,
    remote_path_override: Path | None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the destination path for one profile."""
    if remote_path_override is not None:
        return remote_path_override.expanduser().resolve()

    if profile == "backup":
        raise ValueError(ERROR_BACKUP_REQUIRES_REMOTE_PATH)

    env = environ if environ is not None else os.environ
    env_var = PROFILE_ENV_VARS.get(profile)
    if env_var is None:
        raise ValueError(f"{ERROR_NO_DESTINATION}: {profile}")
    value = env.get(env_var, "").strip()
    if not value:
        raise ValueError(f"{ERROR_NO_DESTINATION}: {profile} ({env_var})")
    return Path(value).expanduser().resolve()


def normalize_args(
    args: argparse.Namespace,
    environ: Mapping[str, str] | None = None,
) -> SyncConfig:
    conference_dir, config_path = resolve_conference_paths(args.conference_dir, args.config)
    remote_path = resolve_destination_path(args.profile, args.remote_path, environ=environ)
    return SyncConfig(
        conference_dir=conference_dir,
        config_path=config_path,
        profile=args.profile,
        remote_path=remote_path,
        dry_run=args.dry_run,
        delete=args.delete,
        quiet=args.quiet,
        reviewed_dry_run=args.reviewed_dry_run,
        init_only=args.init_only,
        reviewed_dry_run_receipt=args.reviewed_dry_run_receipt,
        receipt_path=args.receipt_path,
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read conference config: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Conference config is not a mapping: {path}")
    return data


def safe_relative_path(raw: str, *, label: str) -> Path:
    """Reject manifest paths that escape the conference workspace."""
    candidate = Path(raw.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay inside the conference workspace")
    return candidate


def resolve_manifest_path(conference_dir: Path, config_path: Path) -> Path:
    """Resolve the storage manifest path from config or the default fallback."""
    config = load_yaml_file(config_path)
    storage_config = config.get("storage", {})
    if storage_config is None:
        storage_config = {}
    if not isinstance(storage_config, dict):
        raise ValueError(f"Invalid storage config in {config_path}")

    candidates: list[Path] = []
    manifest_value = storage_config.get("manifest_path")
    if isinstance(manifest_value, str) and manifest_value.strip():
        manifest_relative_path = safe_relative_path(
            manifest_value,
            label="storage.manifest_path",
        )
        candidates.append((conference_dir / manifest_relative_path).resolve())

    fallback_path = (conference_dir / DEFAULT_MANIFEST_PATH).resolve()
    if fallback_path not in candidates:
        candidates.append(fallback_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
        if candidate.exists():
            raise ValueError(f"Storage manifest is not a file: {candidate}")

    raise ValueError(ERROR_NO_MANIFEST)


def load_manifest_families(manifest_path: Path) -> list[ManifestFamily]:
    """Load families from the manifest and validate sync profile fields."""
    try:
        manifest_data = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read storage manifest: {manifest_path}") from exc
    try:
        manifest = json.loads(manifest_data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed storage manifest: could not parse JSON in {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Malformed storage manifest: expected object in {manifest_path}")
    families_data = manifest.get("artifact_families")
    if not isinstance(families_data, list):
        raise ValueError(f"Malformed storage manifest: artifact_families missing in {manifest_path}")

    families: list[ManifestFamily] = []
    for family_data in families_data:
        if not isinstance(family_data, dict):
            raise ValueError(f"Malformed storage manifest entry in {manifest_path}")

        family_id = str(family_data.get("id", "unknown"))
        family_path = family_data.get("path")
        family_glob = family_data.get("path_glob")
        sync_profiles = family_data.get("sync_profiles")
        has_path = isinstance(family_path, str) and bool(family_path.strip())
        has_glob = isinstance(family_glob, str) and bool(family_glob.strip())
        if has_path == has_glob:
            raise ValueError(
                f"Malformed storage manifest: {family_id} must define exactly one of path or path_glob"
            )
        if not isinstance(sync_profiles, dict):
            raise ValueError(f"Malformed storage manifest: {family_id} missing sync_profiles")
        if not isinstance(sync_profiles.get("public"), bool) or not isinstance(
            sync_profiles.get("backup"), bool
        ):
            raise ValueError(
                f"Malformed storage manifest: {family_id} missing sync_profiles booleans"
            )

        relative_path = (
            safe_relative_path(family_path, label=f"{family_id}.path") if has_path else None
        )
        path_glob = str(family_glob).strip() if has_glob else None
        if path_glob is not None:
            safe_relative_path(path_glob, label=f"{family_id}.path_glob")
        families.append(
            ManifestFamily(
                family_id=family_id,
                relative_path=relative_path,
                path_glob=path_glob,
                sync_profiles={
                    "public": bool(sync_profiles["public"]),
                    "backup": bool(sync_profiles["backup"]),
                },
                storage_class=(
                    str(family_data["storage_class"])
                    if isinstance(family_data.get("storage_class"), str)
                    else None
                ),
                share_policy=(
                    str(family_data["share_policy"])
                    if isinstance(family_data.get("share_policy"), str)
                    else None
                ),
            )
        )
    return families


# storage_class values whose manifest families are allowed to be a deliberate,
# operator-placed symlink to an on-demand store outside the repository tree
# (for example storage/workshops/papers or storage/transcription-ocr pointing
# at a Google Drive mount). Everything else -- notably "git" and unset/None,
# which is what a family missing storage_class defaults to -- must never be a
# symlink escape, so containment for those stays symlink-following/strict.
DRIVE_BACKED_STORAGE_CLASSES = frozenset({"backup", "public", "cache"})


def _family_workspace_root(conference_dir: Path, workspace_root: Path, family: ManifestFamily) -> Path:
    """Return the containment root to check a family's source path against.

    Families explicitly declared as Drive/cache-backed (storage_class in
    DRIVE_BACKED_STORAGE_CLASSES) use lexical (os.path.abspath) containment,
    which normalizes ``.``/``..`` without following symlinks -- so their own
    deliberate on-demand symlink is admitted rather than rejected as
    "outside the workspace". Every other family (storage_class "git", or
    unset) keeps the original symlink-following ``.resolve()`` containment,
    so an unexpected symlink placed under a family that must never be one
    (for example a manifest-declared git-tracked family retargeted to an
    arbitrary external directory) still fails closed before its contents
    are synced to an external destination like Google Drive.
    """
    if family.storage_class in DRIVE_BACKED_STORAGE_CLASSES:
        return Path(os.path.abspath(str(conference_dir)))
    return workspace_root


def _lexical_or_strict_resolve(path: Path, *, lexical: bool) -> Path:
    return Path(os.path.abspath(str(path))) if lexical else path.resolve()


def resolve_family_sources(
    conference_dir: Path,
    workspace_root: Path,
    family: ManifestFamily,
) -> list[tuple[Path, Path]]:
    """Return ``(workspace-relative path, absolute source)`` pairs for a family."""
    lexical = family.storage_class in DRIVE_BACKED_STORAGE_CLASSES
    family_workspace_root = _family_workspace_root(conference_dir, workspace_root, family)
    sources: list[tuple[Path, Path]] = []
    if family.relative_path is not None:
        source_path = _lexical_or_strict_resolve(conference_dir / family.relative_path, lexical=lexical)
        if not source_path.is_relative_to(family_workspace_root):
            raise ValueError(f"{family.family_id}.path resolves outside the conference workspace")
        if source_path.exists():
            sources.append((family.relative_path, source_path))
        return sources

    assert family.path_glob is not None
    seen_sources: set[Path] = set()
    for path_glob in expand_brace_glob(family.path_glob):
        for candidate in sorted(conference_dir.glob(path_glob)):
            source_path = _lexical_or_strict_resolve(candidate, lexical=lexical)
            if source_path in seen_sources:
                continue
            if not source_path.is_relative_to(family_workspace_root):
                raise ValueError(
                    f"{family.family_id}.path_glob resolves outside the conference workspace"
                )
            if not source_path.exists():
                continue
            seen_sources.add(source_path)
            relative_path = source_path.relative_to(family_workspace_root)
            sources.append((relative_path, source_path))
    return sources


def expand_brace_glob(pattern: str) -> tuple[str, ...]:
    """Expand simple shell-style brace groups before passing patterns to Path.glob."""
    start = pattern.find("{")
    if start == -1:
        return (pattern,)
    end = pattern.find("}", start + 1)
    if end == -1:
        return (pattern,)

    prefix = pattern[:start]
    suffix = pattern[end + 1 :]
    options = [option for option in pattern[start + 1 : end].split(",") if option]
    if not options:
        return (pattern,)

    expanded: list[str] = []
    seen: set[str] = set()
    for option in options:
        for candidate in expand_brace_glob(f"{prefix}{option}{suffix}"):
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)
    return tuple(expanded)


def family_specificity(family: ManifestFamily) -> int:
    pattern = family.relative_path.as_posix() if family.relative_path is not None else family.path_glob or ""
    return len(pattern.split("/"))


def preferred_sync_item(
    existing: SyncItem,
    candidate: SyncItem,
    family_by_id: Mapping[str, ManifestFamily],
) -> SyncItem:
    existing_family = family_by_id[existing.family_id]
    candidate_family = family_by_id[candidate.family_id]
    existing_is_git = existing_family.storage_class == "git"
    candidate_is_git = candidate_family.storage_class == "git"
    if existing_is_git != candidate_is_git:
        return candidate if candidate_is_git else existing
    if family_specificity(candidate_family) > family_specificity(existing_family):
        return candidate
    return existing


def deduplicate_sync_items(
    items: Sequence[SyncItem],
    families: Sequence[ManifestFamily],
) -> tuple[SyncItem, ...]:
    """Keep each workspace path in one manifest family, preferring Git controls."""

    family_by_id = {family.family_id: family for family in families}
    by_path: dict[Path, SyncItem] = {}
    for item in items:
        current = by_path.get(item.relative_path)
        if current is None:
            by_path[item.relative_path] = item
            continue
        by_path[item.relative_path] = preferred_sync_item(current, item, family_by_id)
    return tuple(by_path.values())


def build_sync_plan(config: SyncConfig) -> SyncPlan:
    """Build the manifest-driven sync plan for the chosen profile."""
    manifest_path = resolve_manifest_path(config.conference_dir, config.config_path)
    families = load_manifest_families(manifest_path)
    workspace_root = config.conference_dir.resolve()

    included_families: list[str] = []
    excluded_families: list[str] = []
    sync_items: list[SyncItem] = []

    for family in families:
        enabled = family.sync_profiles[config.profile]
        if not enabled:
            excluded_families.append(family.family_id)
            continue

        sources = resolve_family_sources(config.conference_dir, workspace_root, family)
        if not sources:
            if config.init_only and family.relative_path is not None:
                included_families.append(family.family_id)
                placeholder_source = (config.conference_dir / family.relative_path).resolve()
                sync_items.append(
                    SyncItem(
                        family_id=family.family_id,
                        relative_path=family.relative_path,
                        source_path=placeholder_source,
                        destination_path=(config.remote_path / family.relative_path).resolve(),
                        is_directory=True,
                    )
                )
                continue
            excluded_families.append(f"{family.family_id} (missing locally)")
            continue

        included_families.append(family.family_id)
        for relative_path, source_path in sources:
            sync_items.append(
                SyncItem(
                    family_id=family.family_id,
                    relative_path=relative_path,
                    source_path=source_path,
                    destination_path=(config.remote_path / relative_path).resolve(),
                    is_directory=source_path.is_dir(),
                )
            )

    sync_items = list(deduplicate_sync_items(sync_items, families))
    included_families = sorted({item.family_id for item in sync_items})
    excluded_families.sort()
    sync_items.sort(key=lambda item: item.destination_path.as_posix())

    return SyncPlan(
        config=config,
        manifest_path=manifest_path,
        families=tuple(families),
        included_families=tuple(included_families),
        excluded_families=tuple(excluded_families),
        sync_items=tuple(sync_items),
    )


def build_rsync_command(
    sync_item: SyncItem,
    dry_run: bool,
    delete: bool,
    quiet: bool,
) -> list[str]:
    """Build the rsync command for one sync item."""
    command = ["rsync", "-a" if quiet else "-av"]
    if dry_run:
        command.append("--dry-run")
    if delete:
        command.append("--delete")
    if not quiet:
        command.append("--progress")
    for pattern in DEFAULT_EXCLUDES + SECRET_EXCLUDES:
        command.extend(["--exclude", pattern])

    if sync_item.is_directory:
        command.append(f"{sync_item.source_path}/")
        command.append(f"{sync_item.destination_path}/")
    else:
        command.append(str(sync_item.source_path))
        command.append(str(sync_item.destination_path))
    return command


def run_rsync(command: Sequence[str]) -> int:
    """Execute the rsync command and return its exit code."""
    print(f"Running: {' '.join(command)}", flush=True)
    print(flush=True)
    try:
        result = subprocess.run(command, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nSync interrupted by user", flush=True)
        return 130


def execute_sync_plan(plan: SyncPlan) -> int:
    """Run the prepared sync plan."""
    if plan.config.init_only:
        for sync_item in plan.sync_items:
            if plan.config.dry_run:
                continue
            if sync_item.is_directory:
                sync_item.destination_path.mkdir(parents=True, exist_ok=True)
            else:
                sync_item.destination_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            (
                "Init dry run complete - no directories created"
                if plan.config.dry_run
                else "Init complete - selected Drive directories are present"
            ),
            flush=True,
        )
        return 0

    if not plan.sync_items:
        print("No matching manifest families to sync.")
        return 0

    for sync_item in plan.sync_items:
        if not plan.config.dry_run:
            if sync_item.is_directory:
                sync_item.destination_path.mkdir(parents=True, exist_ok=True)
            else:
                sync_item.destination_path.parent.mkdir(parents=True, exist_ok=True)

        exit_code = run_rsync(
            build_rsync_command(
                sync_item=sync_item,
                dry_run=plan.config.dry_run,
                delete=plan.config.delete,
                quiet=plan.config.quiet,
            )
        )
        if exit_code != 0:
            return exit_code
    return 0


def iter_item_files(item: SyncItem) -> list[tuple[Path, Path]]:
    """Return workspace-relative file paths covered by one sync item."""
    if item.source_path.is_file():
        return [(item.relative_path, item.source_path)]
    if not item.source_path.is_dir():
        return []
    files: list[tuple[Path, Path]] = []
    for path in sorted(item.source_path.rglob("*")):
        if not path.is_file():
            continue
        files.append((item.relative_path / path.relative_to(item.source_path), path))
    return files


def build_artifact_snapshot(plan: SyncPlan) -> dict[str, Any]:
    """Build a compact metadata snapshot for stale-artifact checks.

    The snapshot hashes stable file metadata (workspace-relative path, byte size,
    and mtime) instead of file contents so frequent backup checkpoints stay fast.
    Restore verification remains the content-hash proof source.
    """
    aggregate = hashlib.sha256()
    total_file_count = 0
    total_size_bytes = 0
    latest_mtime_ns = 0
    items: list[dict[str, Any]] = []
    for item in plan.sync_items:
        item_hash = hashlib.sha256()
        item_file_count = 0
        item_size_bytes = 0
        item_latest_mtime_ns = 0
        for relative_path, file_path in iter_item_files(item):
            stat = file_path.stat()
            item_file_count += 1
            item_size_bytes += stat.st_size
            item_latest_mtime_ns = max(item_latest_mtime_ns, stat.st_mtime_ns)
            line = f"{relative_path.as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            item_hash.update(line.encode("utf-8"))
            aggregate.update(line.encode("utf-8"))
        total_file_count += item_file_count
        total_size_bytes += item_size_bytes
        latest_mtime_ns = max(latest_mtime_ns, item_latest_mtime_ns)
        items.append(
            {
                "family_id": item.family_id,
                "relative_path": item.relative_path.as_posix(),
                "destination_path": str(item.destination_path),
                "file_count": item_file_count,
                "size_bytes": item_size_bytes,
                "latest_mtime_ns": item_latest_mtime_ns,
                "snapshot_sha256": item_hash.hexdigest(),
            }
        )
    return {
        "file_count": total_file_count,
        "size_bytes": total_size_bytes,
        "latest_mtime_ns": latest_mtime_ns,
        "snapshot_sha256": aggregate.hexdigest(),
        "items": items,
    }


def count_plan_digests(plan: SyncPlan) -> int:
    """Count local files covered by a sync plan for receipt compatibility."""
    return int(build_artifact_snapshot(plan)["file_count"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_control_file_digests(plan: SyncPlan) -> list[dict[str, str]]:
    """Return content hashes only for selected exact-file Git control artifacts."""
    families = {family.family_id: family for family in plan.families}
    digests: list[dict[str, str]] = []
    for item in plan.sync_items:
        family = families[item.family_id]
        if (
            family.storage_class != "git"
            or family.path_glob is not None
            or item.is_directory
            or not item.source_path.is_file()
        ):
            continue
        digests.append(
            {
                "path": item.relative_path.as_posix(),
                "sha256": sha256_file(item.source_path),
            }
        )
    return sorted(digests, key=lambda entry: entry["path"])


def load_json_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{ERROR_DELETE_RECEIPT_MISMATCH}: could not read {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{ERROR_DELETE_RECEIPT_MISMATCH}: receipt is not an object")
    return payload


def validate_reviewed_dry_run_receipt(plan: SyncPlan, receipt_path: Path) -> None:
    """Bind destructive delete mode to a matching successful delete-preview receipt."""
    receipt = load_json_receipt(receipt_path)
    snapshot = build_artifact_snapshot(plan)
    expected = {
        "status": "succeeded",
        "profile": plan.config.profile,
        "target": str(plan.config.remote_path),
        "manifest_sha256": sha256_file(plan.manifest_path),
        "artifact_snapshot_sha256": snapshot["snapshot_sha256"],
        "artifact_file_count": snapshot["file_count"],
        "dry_run": True,
        "delete": True,
        "init_only": False,
        "dry_run_exit_code": 0,
        "secrets_copied": False,
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if receipt.get(key) != expected_value
    ]
    if list(receipt.get("included_families", [])) != list(plan.included_families):
        mismatches.append("included_families")
    if mismatches:
        raise ValueError(
            f"{ERROR_DELETE_RECEIPT_MISMATCH}: {', '.join(sorted(mismatches))}"
        )


def is_secret_like_path(path: Path) -> bool:
    name = path.name.lower()
    stem = path.stem.lower()
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {
        "cookies.json",
        "cookie.json",
        "token.json",
        "tokens.json",
        "secrets.json",
        "secret.json",
        "credentials.json",
        "credential.json",
        "password.json",
        "passwords.json",
        "api_key.json",
        "apikey.json",
    }:
        return True

    parts = [part for part in re.split(r"[^a-z0-9]+", stem) if part]
    if any(part in {"cookie", "cookies", "secret", "secrets", "password", "passwords"} for part in parts):
        return True
    if any(part in {"credential", "credentials"} for part in parts):
        return True
    if "apikey" in parts or ("api" in parts and "key" in parts):
        return True
    if any(part in {"token", "tokens"} for part in parts):
        return any(
            part in {"access", "auth", "bearer", "id", "oauth", "refresh", "session"}
            for part in parts
        )
    return False


def find_secret_like_paths(plan: SyncPlan) -> list[str]:
    hits: list[str] = []
    workspace_root = plan.config.conference_dir.resolve()
    for item in plan.sync_items:
        candidates: list[Path]
        if item.source_path.is_dir():
            candidates = [item.source_path] + [path for path in item.source_path.rglob("*") if path.is_file()]
        else:
            candidates = [item.source_path]
        for candidate in candidates:
            if not is_secret_like_path(candidate):
                continue
            try:
                hits.append(candidate.relative_to(workspace_root).as_posix())
            except ValueError:
                hits.append(str(candidate))
    return sorted(set(hits))


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_sync_receipt(
    path: Path,
    *,
    plan: SyncPlan | None,
    exit_code: int,
    status: str,
    error: str | None = None,
    secret_like_paths: Sequence[str] = (),
) -> None:
    """Write the backup/public sync receipt without persisting secret values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 3,
        "receipt_type": "conference_pipeline.gdrive_sync",
        "status": status,
        "profile": plan.config.profile if plan else None,
        "workspace": str(plan.config.conference_dir) if plan else None,
        "target": str(plan.config.remote_path) if plan else None,
        "manifest_path": str(plan.manifest_path) if plan else None,
        "manifest_sha256": sha256_file(plan.manifest_path) if plan else None,
        "included_families": list(plan.included_families) if plan else [],
        "skipped_families": list(plan.excluded_families) if plan else [],
        "dry_run": bool(plan.config.dry_run) if plan else None,
        "delete": bool(plan.config.delete) if plan else None,
        "init_only": bool(plan.config.init_only) if plan else None,
        "reviewed_dry_run_receipt": (
            str(plan.config.reviewed_dry_run_receipt)
            if plan and plan.config.reviewed_dry_run_receipt is not None
            else None
        ),
        "exit_code": exit_code,
        "dry_run_exit_code": exit_code if plan and plan.config.dry_run else None,
        "write_exit_code": exit_code if plan and not plan.config.dry_run else None,
        "digest_count": 0,
        "artifact_file_count": 0,
        "artifact_size_bytes": 0,
        "artifact_latest_mtime_ns": 0,
        "artifact_snapshot_sha256": None,
        "sync_items": [],
        "selected_control_file_digests": [],
        "bloat_control": {
            "policy_source": "storage/storage_manifest.json sync_profiles",
            "copy_mode": "rsync incremental",
            "cache_policy": "cache families sync only when manifest explicitly enables the selected profile",
        },
        "generated_at": utc_now(),
        "error": error,
        "secret_like_paths": list(secret_like_paths),
        "secrets_copied": status == "succeeded" and bool(secret_like_paths),
    }
    if plan is not None:
        snapshot = build_artifact_snapshot(plan)
        payload["digest_count"] = snapshot["file_count"]
        payload["artifact_file_count"] = snapshot["file_count"]
        payload["artifact_size_bytes"] = snapshot["size_bytes"]
        payload["artifact_latest_mtime_ns"] = snapshot["latest_mtime_ns"]
        payload["artifact_snapshot_sha256"] = snapshot["snapshot_sha256"]
        payload["sync_items"] = snapshot["items"]
        payload["selected_control_file_digests"] = selected_control_file_digests(plan)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def format_family_list(values: Sequence[str]) -> str:
    return ", ".join(values) if values else "(none)"


def prepare_destination_root(remote_path: Path, dry_run: bool) -> DestinationStatus:
    """Validate the mounted root and create the final conference leaf only in write mode."""
    target_root = remote_path.parent
    if not target_root.exists():
        raise ValueError(f"{ERROR_REMOTE_ROOT_MISSING} {target_root}")

    target_root_status = "reachable"
    if remote_path.exists():
        leaf_status = "existing"
    elif dry_run:
        leaf_status = "missing"
    else:
        remote_path.mkdir(parents=False, exist_ok=True)
        if not remote_path.exists():
            raise ValueError(f"Could not create destination leaf: {remote_path}")
        leaf_status = "created"

    return DestinationStatus(
        target_root=target_root,
        target_root_status=target_root_status,
        leaf_status=leaf_status,
    )


def render_preflight_report(
    plan: SyncPlan,
    workspace_size: str,
    destination_status: DestinationStatus,
) -> str:
    """Render the operator-facing preflight report."""
    return "\n".join(
        [
            "=" * 60,
            "Conference Workspace Sync",
            "=" * 60,
            f"Profile: {plan.config.profile}",
            f"Workspace: {plan.config.conference_dir}",
            f"Config: {plan.config.config_path}",
            f"Manifest: {plan.manifest_path}",
            f"Destination root: {plan.config.remote_path}",
            f"Target root status: {destination_status.target_root_status}",
            f"Leaf status: {destination_status.leaf_status}",
            f"Workspace size: {workspace_size}",
            f"Included families: {format_family_list(plan.included_families)}",
            f"Excluded families: {format_family_list(plan.excluded_families)}",
            f"Delete mode: {'enabled' if plan.config.delete else 'disabled'}",
            f"Dry run: {'enabled' if plan.config.dry_run else 'disabled'}",
            f"Init only: {'enabled' if plan.config.init_only else 'disabled'}",
            "=" * 60,
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan: SyncPlan | None = None
    secret_like_paths: list[str] = []
    try:
        config = normalize_args(args)
        if (
            config.delete
            and not config.dry_run
            and not config.reviewed_dry_run
            and config.reviewed_dry_run_receipt is None
        ):
            raise ValueError(ERROR_DELETE_REQUIRES_REVIEW)
        plan = build_sync_plan(config)
        if config.delete and not config.dry_run and config.reviewed_dry_run_receipt is not None:
            validate_reviewed_dry_run_receipt(plan, config.reviewed_dry_run_receipt)
        secret_like_paths = find_secret_like_paths(plan)
        if secret_like_paths:
            preview = ", ".join(secret_like_paths[:10])
            extra = "" if len(secret_like_paths) <= 10 else f", and {len(secret_like_paths) - 10} more"
            raise ValueError(f"{ERROR_SECRET_LIKE_FILES}: {preview}{extra}")
        destination_status = prepare_destination_root(config.remote_path, dry_run=config.dry_run)
    except ValueError as exc:
        print(f"Error: {exc}", flush=True)
        if args.receipt_path:
            write_sync_receipt(
                args.receipt_path,
                plan=plan,
                exit_code=1,
                status="failed",
                error=str(exc),
                secret_like_paths=secret_like_paths,
            )
        return 1

    local_size = get_local_size(config.conference_dir)

    print(render_preflight_report(plan, local_size, destination_status), flush=True)
    print(flush=True)

    exit_code = execute_sync_plan(plan)
    if config.receipt_path:
        write_sync_receipt(
            config.receipt_path,
            plan=plan,
            exit_code=exit_code,
            status="succeeded" if exit_code == 0 else "failed",
            secret_like_paths=secret_like_paths,
        )
    if exit_code == 0:
        print(
            "Sync complete!" if not config.dry_run else "Dry run complete - no changes made",
            flush=True,
        )
        return 0

    print("Sync failed!", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
