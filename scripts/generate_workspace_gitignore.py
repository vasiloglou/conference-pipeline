#!/usr/bin/env python3
"""Generate and verify conference workspace gitignore policy from storage manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.registry_diagnostics import attach_registry_tool_diagnostic  # noqa: E402

MANAGED_START = "# BEGIN conference-pipeline managed gitignore"
MANAGED_END = "# END conference-pipeline managed gitignore"
POLICY_VERSION = "manifest-storage-class-v1"
GIT_CLASS = "git"
IGNORED_CLASSES = {"public", "backup", "cache"}
ALLOWED_CLASSES = {GIT_CLASS, *IGNORED_CLASSES}
BUILTIN_TRACKED = {".gitignore"}
MAX_GIT_BLOB_BYTES = 95 * 1024 * 1024


@dataclass(frozen=True)
class Family:
    id: str
    storage_class: str
    path: str | None
    path_glob: str | None
    gitignore_exceptions: tuple[str, ...]
    tracked_samples: tuple[str, ...]

    @property
    def is_git(self) -> bool:
        return self.storage_class == GIT_CLASS

    @property
    def is_ignored(self) -> bool:
        return self.storage_class in IGNORED_CLASSES

    @property
    def display_path(self) -> str:
        return self.path if self.path is not None else str(self.path_glob)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        workspace=args.workspace,
        manifest_path=args.manifest,
        check_git=args.check,
    )

    if args.dry_run:
        print(report["generated_gitignore"], end="")

    if args.write:
        write_workspace_gitignore(
            Path(report["workspace_path"]),
            str(report["generated_gitignore"]),
        )
        report = build_report(
            workspace=args.workspace,
            manifest_path=args.manifest,
            check_git=args.check,
        )

    if args.receipt_path is not None:
        receipt = dict(report)
        receipt.pop("generated_gitignore", None)
        receipt["generated_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        args.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.json:
        printable = dict(report)
        if not args.include_generated_content:
            printable.pop("generated_gitignore", None)
        print(json.dumps(printable, indent=2, sort_keys=True))
    elif not args.dry_run:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key != "generated_gitignore"
                },
                indent=2,
                sort_keys=True,
            )
        )

    return 0 if report["status"] == "pass" else 1


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-generated-content", action="store_true")
    parser.add_argument("--receipt-path", type=Path)
    args = parser.parse_args(argv)
    if not args.write and not args.check and not args.dry_run:
        parser.error("choose at least one of --write, --check, or --dry-run")
    return args


def build_report(
    *,
    workspace: Path,
    manifest_path: Path | None = None,
    check_git: bool = False,
) -> dict[str, Any]:
    workspace_path = workspace.resolve()
    manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else workspace_path / "storage" / "storage_manifest.json"
    )
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    try:
        workspace_relative = workspace_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        blockers.append(
            {
                "code": "workspace_outside_repo",
                "workspace": str(workspace),
                "retryable": False,
            }
        )
        workspace_relative = workspace.as_posix()

    if not manifest.is_file():
        report = {
            "status": "blocked",
            "blockers": [
                {
                    "code": "storage_manifest_missing",
                    "manifest_path": display_path(manifest),
                    "retryable": True,
                }
            ],
            "warnings": warnings,
            "workspace": workspace_relative,
            "workspace_path": str(workspace_path),
            "manifest_path": display_path(manifest),
            "manifest_sha256": None,
            "policy_version": POLICY_VERSION,
            "generated_gitignore": "",
        }
        return attach_registry_tool_diagnostic(
            report,
            script_id="gitignore.workspace_policy",
            checked_inputs={
                "workspace": workspace_relative,
                "workspace_path": str(workspace_path),
                "manifest_path": display_path(manifest),
                "check_git": check_git,
            },
            accepted_evidence=[],
            blocked_next_actions=[
                "Create the storage manifest, then rerun generate_workspace_gitignore.py."
            ],
        )

    manifest_sha = sha256_file(manifest)
    try:
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report = {
            "status": "blocked",
            "blockers": [
                {
                    "code": "storage_manifest_unreadable",
                    "manifest_path": display_path(manifest),
                    "reason": str(exc),
                    "retryable": True,
                }
            ],
            "warnings": warnings,
            "workspace": workspace_relative,
            "workspace_path": str(workspace_path),
            "manifest_path": display_path(manifest),
            "manifest_sha256": manifest_sha,
            "policy_version": POLICY_VERSION,
            "generated_gitignore": "",
        }
        return attach_registry_tool_diagnostic(
            report,
            script_id="gitignore.workspace_policy",
            checked_inputs={
                "workspace": workspace_relative,
                "workspace_path": str(workspace_path),
                "manifest_path": display_path(manifest),
                "check_git": check_git,
            },
            accepted_evidence=["Storage manifest file exists and sha256 was computed."],
            blocked_next_actions=[
                "Repair the storage manifest JSON, then rerun generate_workspace_gitignore.py."
            ],
        )
    families = load_families(manifest_payload, blockers)
    blockers.extend(validate_manifest_workspace(manifest_payload, workspace_relative))
    blockers.extend(validate_family_conflicts(families))
    oversized_local_paths = oversized_local_gitignore_paths(workspace_path, families)
    generated = render_gitignore(
        workspace=workspace_relative,
        manifest_path=display_path(manifest),
        manifest_sha=manifest_sha,
        families=families,
        oversized_local_paths=oversized_local_paths,
    )

    gitignore_path = workspace_path / ".gitignore"
    if check_git:
        blockers.extend(validate_gitignore_file(gitignore_path, generated))
        blockers.extend(validate_git_tracking(workspace_path, families))
        blockers.extend(validate_git_file_size_limit(workspace_path, families))
        blockers.extend(validate_check_ignore(workspace_path, families))

    report = {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "warnings": warnings,
        "workspace": workspace_relative,
        "workspace_path": str(workspace_path),
        "manifest_path": display_path(manifest),
        "manifest_sha256": manifest_sha,
        "policy_version": POLICY_VERSION,
        "gitignore_path": display_path(gitignore_path),
        "generated_gitignore": generated,
        "oversized_local_gitignore_paths": oversized_local_paths,
        "families": [
            {
                "id": family.id,
                "path": family.path,
                "path_glob": family.path_glob,
                "storage_class": family.storage_class,
                "gitignore_exceptions": list(family.gitignore_exceptions),
                "tracked_samples": list(family.tracked_samples),
            }
            for family in families
        ],
    }
    return attach_registry_tool_diagnostic(
        report,
        script_id="gitignore.workspace_policy",
        checked_inputs={
            "workspace": workspace_relative,
            "workspace_path": str(workspace_path),
            "manifest_path": display_path(manifest),
            "check_git": check_git,
            "family_count": len(families),
            "oversized_local_gitignore_path_count": len(oversized_local_paths),
        },
        accepted_evidence=[
            "Storage manifest was parsed.",
            "Artifact families were normalized and rendered into the managed gitignore block.",
            "Oversized untracked local files outside ignored families were rendered as explicit gitignore entries.",
            "Workspace .gitignore and tracked-file policy were checked when --check was used.",
        ],
        pass_next_actions=[
            "Workspace gitignore policy is current; continue with planning or production proof."
        ],
        blocked_next_actions=[
            "Repair the cited manifest, gitignore, or tracked-file blocker, then rerun --check."
        ],
    )


def load_families(
    manifest: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> list[Family]:
    families: list[Family] = []
    for raw in manifest.get("artifact_families") or ():
        if not isinstance(raw, Mapping):
            blockers.append(
                {"code": "artifact_family_malformed", "retryable": False}
            )
            continue
        family_id = str(raw.get("id") or "")
        storage_class = str(raw.get("storage_class") or "")
        path = raw.get("path")
        path_glob = raw.get("path_glob")
        exceptions = tuple(str(item) for item in raw.get("gitignore_exceptions") or ())
        tracked_samples = tuple(str(item) for item in raw.get("tracked_samples") or ())
        if storage_class not in ALLOWED_CLASSES:
            blockers.append(
                {
                    "code": "artifact_family_unknown_storage_class",
                    "family_id": family_id,
                    "storage_class": storage_class,
                    "retryable": False,
                }
            )
        normalized_path = normalize_manifest_path(path, family_id, blockers)
        normalized_glob = normalize_manifest_glob(path_glob, family_id, blockers)
        if normalized_path is not None and normalized_glob is not None:
            blockers.append(
                {
                    "code": "artifact_family_path_and_glob_conflict",
                    "family_id": family_id,
                    "retryable": False,
                }
            )
        if normalized_path is None and normalized_glob is None:
            blockers.append(
                {
                    "code": "artifact_family_path_missing",
                    "family_id": family_id,
                    "retryable": False,
                }
            )
        if normalized_glob is not None and exceptions:
            blockers.append(
                {
                    "code": "artifact_family_glob_exceptions_unsupported",
                    "family_id": family_id,
                    "retryable": False,
                }
            )
        normalized_exceptions = tuple(
            normalize_exception_path(item, family_id, blockers) for item in exceptions
        )
        normalized_samples = tuple(
            normalize_manifest_path(item, family_id, blockers)
            for item in tracked_samples
        )
        families.append(
            Family(
                id=family_id,
                storage_class=storage_class,
                path=normalized_path,
                path_glob=normalized_glob,
                gitignore_exceptions=tuple(
                    item for item in normalized_exceptions if item is not None
                ),
                tracked_samples=tuple(item for item in normalized_samples if item is not None),
            )
        )
    return families


def normalize_manifest_path(
    value: object,
    family_id: str,
    blockers: list[dict[str, Any]],
) -> str | None:
    if value in (None, ""):
        return None
    raw = str(value)
    if any(char in raw for char in "*?[]"):
        blockers.append(
            {
                "code": "artifact_family_path_contains_glob",
                "family_id": family_id,
                "path": raw,
                "retryable": False,
            }
        )
        return None
    return normalize_posix_path(raw, "artifact_family_unsafe_path", family_id, blockers)


def normalize_manifest_glob(
    value: object,
    family_id: str,
    blockers: list[dict[str, Any]],
) -> str | None:
    if value in (None, ""):
        return None
    raw = str(value)
    if raw.startswith("/") or "\\" in raw:
        blockers.append(
            {
                "code": "artifact_family_unsafe_path_glob",
                "family_id": family_id,
                "path_glob": raw,
                "retryable": False,
            }
        )
        return None
    parts = PurePosixPath(raw).parts
    if ".." in parts:
        blockers.append(
            {
                "code": "artifact_family_unsafe_path_glob",
                "family_id": family_id,
                "path_glob": raw,
                "retryable": False,
            }
        )
        return None
    return raw.strip("/")


def normalize_exception_path(
    value: object,
    family_id: str,
    blockers: list[dict[str, Any]],
) -> str | None:
    normalized = normalize_posix_path(
        str(value),
        "artifact_family_unsafe_gitignore_exception",
        family_id,
        blockers,
    )
    if normalized is None:
        return None
    if normalized.endswith("/"):
        blockers.append(
            {
                "code": "artifact_family_directory_gitignore_exception",
                "family_id": family_id,
                "exception": str(value),
                "retryable": False,
            }
        )
    return normalized


def normalize_posix_path(
    raw: str,
    code: str,
    family_id: str,
    blockers: list[dict[str, Any]],
) -> str | None:
    if raw.startswith("/") or "\\" in raw:
        blockers.append(
            {"code": code, "family_id": family_id, "path": raw, "retryable": False}
        )
        return None
    stripped = raw.strip("/")
    parts = PurePosixPath(stripped).parts
    if not stripped or "." in parts or ".." in parts:
        blockers.append(
            {"code": code, "family_id": family_id, "path": raw, "retryable": False}
        )
        return None
    return PurePosixPath(stripped).as_posix()


def validate_manifest_workspace(
    manifest: Mapping[str, Any],
    workspace_relative: str,
) -> list[dict[str, Any]]:
    declared = str(manifest.get("workspace") or "")
    if declared != workspace_relative:
        return [
            {
                "code": "storage_manifest_workspace_mismatch",
                "declared_workspace": declared,
                "observed_workspace": workspace_relative,
                "retryable": False,
            }
        ]
    return []


def validate_family_conflicts(families: Iterable[Family]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    seen: dict[str, Family] = {}
    for family in families:
        key = family.path or f"glob:{family.path_glob}"
        previous = seen.get(key)
        if previous is not None and previous.storage_class != family.storage_class:
            blockers.append(
                {
                    "code": "artifact_family_storage_class_conflict",
                    "path": key,
                    "family_ids": [previous.id, family.id],
                    "storage_classes": [previous.storage_class, family.storage_class],
                    "retryable": False,
                }
            )
        seen[key] = family
    return blockers


def render_gitignore(
    *,
    workspace: str,
    manifest_path: str,
    manifest_sha: str,
    families: Iterable[Family],
    oversized_local_paths: Iterable[str] = (),
) -> str:
    lines = [
        f"{MANAGED_START}",
        "# Generated by scripts/generate_workspace_gitignore.py.",
        "# Edit storage/storage_manifest.json, then rerun this script.",
        f"# workspace: {workspace}",
        f"# policy_version: {POLICY_VERSION}",
        f"# manifest: {manifest_path}",
        f"# manifest_sha256: {manifest_sha}",
        "",
        "# Keep workspace control files trackable.",
        "!.gitignore",
    ]
    git_families = sorted(
        (family for family in families if family.is_git),
        key=lambda item: item.display_path,
    )
    ignored_families = sorted(
        (family for family in families if family.is_ignored),
        key=lambda item: item.display_path,
    )
    if git_families:
        lines.extend(["", "# Git-owned families."])
        for family in git_families:
            lines.extend(unignore_family_patterns(family))
    if ignored_families:
        lines.extend(["", "# Google Drive or cache-owned families."])
        for family in ignored_families:
            lines.extend(ignore_family_patterns(family))
    protected_git_patterns = [
        pattern
        for family in git_families
        if is_specific_git_family(family)
        for pattern in unignore_family_patterns(family)
    ]
    if protected_git_patterns:
        lines.extend(["", "# Git-owned file and glob families protected from payload ignores."])
        lines.extend(protected_git_patterns)
    lines.extend(
        [
            "",
            "# Local OS/editor metadata and Python bytecode cache.",
            ".DS_Store",
            "**/.DS_Store",
            "__pycache__/",
            "**/__pycache__/",
            "*.pyc",
        ]
    )
    oversized_paths = sorted(set(oversized_local_paths))
    if oversized_paths:
        lines.extend(
            [
                "",
                "# Oversized local files outside manifest-owned payload families.",
                "# Move durable payloads into public, backup, or cache families.",
            ]
        )
        for path in oversized_paths:
            lines.append(f"/{path}")
    lines.extend([MANAGED_END, ""])
    return "\n".join(lines)


def unignore_family_patterns(family: Family) -> list[str]:
    if family.path_glob:
        return [f"!/{path_glob}" for path_glob in expand_brace_glob(family.path_glob)]
    assert family.path is not None
    path = family.path
    if looks_like_file(path):
        return [f"!/{path}"]
    return [f"!/{path}/", f"!/{path}/**"]


def ignore_family_patterns(family: Family) -> list[str]:
    if family.path_glob:
        return [f"/{path_glob}" for path_glob in expand_brace_glob(family.path_glob)]
    assert family.path is not None
    path = family.path
    if looks_like_file(path):
        patterns = [f"/{path}"]
    elif family.gitignore_exceptions:
        # Keep a contents-level ignore so declared exceptions under this
        # directory can be re-included below.
        patterns = [f"/{path}/*"]
    else:
        # Ignore the family root entry itself as well as its contents, so a
        # symlink at this path (e.g. a Google Drive backup leaf) is ignored,
        # not just the files it would contain.
        patterns = [f"/{path}", f"/{path}/*"]
    for exception in family.gitignore_exceptions:
        exception_path = f"{path}/{exception}"
        patterns.extend(unignore_parent_chain(exception_path))
        patterns.append(f"!/{exception_path}")
    return patterns


def is_specific_git_family(family: Family) -> bool:
    if not family.is_git:
        return False
    if family.path_glob is not None:
        return True
    assert family.path is not None
    return looks_like_file(family.path)


def unignore_parent_chain(path: str) -> list[str]:
    parts = PurePosixPath(path).parts[:-1]
    patterns: list[str] = []
    current: list[str] = []
    for part in parts:
        current.append(part)
        patterns.append(f"!/{PurePosixPath(*current).as_posix()}/")
    return patterns


def looks_like_file(path: str) -> bool:
    return "." in PurePosixPath(path).name


def expand_brace_glob(pattern: str) -> tuple[str, ...]:
    """Expand simple shell-style brace groups into Git-compatible patterns."""
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


def write_workspace_gitignore(workspace: Path, generated: str) -> None:
    target = workspace / ".gitignore"
    existing = target.read_text(encoding="utf-8") if target.exists() else ""
    updated = replace_managed_block(existing, generated)
    target.write_text(updated, encoding="utf-8")


def replace_managed_block(existing: str, generated: str) -> str:
    if not existing.strip():
        return generated
    if MANAGED_START in existing and MANAGED_END in existing:
        before, rest = existing.split(MANAGED_START, 1)
        _, after = rest.split(MANAGED_END, 1)
        if not before.strip() and not after.strip():
            return generated
        updated = before.rstrip() + "\n\n" + generated.rstrip() + "\n" + after.lstrip()
        return updated if updated.endswith("\n") else updated + "\n"
    return existing.rstrip() + "\n\n" + generated


def validate_gitignore_file(gitignore_path: Path, generated: str) -> list[dict[str, Any]]:
    if not gitignore_path.is_file():
        return [
            {
                "code": "workspace_gitignore_missing",
                "path": display_path(gitignore_path),
                "retryable": True,
            }
        ]
    existing = gitignore_path.read_text(encoding="utf-8")
    expected = replace_managed_block(existing, generated)
    if existing != expected:
        return [
            {
                "code": "workspace_gitignore_stale",
                "path": display_path(gitignore_path),
                "retryable": True,
            }
        ]
    return []


def validate_git_tracking(workspace: Path, families: Iterable[Family]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    tracked = git_ls_files(workspace)
    for path in tracked:
        if path in BUILTIN_TRACKED:
            continue
        family = ignored_family_for_path(path, families)
        if family is None:
            continue
        if path in declared_exception_paths(family):
            continue
        blockers.append(
            {
                "code": "drive_or_cache_family_file_tracked_in_git",
                "path": f"{display_path(workspace)}/{path}",
                "family_id": family.id,
                "storage_class": family.storage_class,
                "retryable": True,
                "expected": (
                    "public, backup, and cache family payloads are ignored and "
                    "untracked unless declared as gitignore_exceptions"
                ),
                "observed": "git ls-files still reports this path as tracked",
                "next_action": (
                    "Move or untrack this payload before pushing; keep only small "
                    "control receipts or declared exceptions in Git."
                ),
            }
        )
    return blockers


def validate_git_file_size_limit(
    workspace: Path,
    families: Iterable[Family],
    *,
    max_bytes: int | None = None,
) -> list[dict[str, Any]]:
    if max_bytes is None:
        max_bytes = MAX_GIT_BLOB_BYTES
    blockers: list[dict[str, Any]] = []
    family_list = list(families)
    for path in git_ls_files(workspace):
        if path in BUILTIN_TRACKED:
            continue
        ignored = ignored_family_for_path(path, family_list)
        if ignored is not None and path not in declared_exception_paths(ignored):
            continue
        absolute_path = workspace / path
        try:
            byte_count = absolute_path.stat().st_size
        except OSError:
            continue
        if byte_count <= max_bytes:
            continue
        blockers.append(
            {
                "code": "git_owned_file_exceeds_size_limit",
                "path": f"{display_path(workspace)}/{path}",
                "byte_count": byte_count,
                "max_bytes": max_bytes,
                "retryable": True,
                "next_action": (
                    "Move this mutable or generated payload into a public, backup, "
                    "or cache artifact family and untrack it before pushing."
                ),
            }
        )
    return blockers


def oversized_local_gitignore_paths(
    workspace: Path,
    families: Iterable[Family],
    *,
    max_bytes: int | None = None,
) -> list[str]:
    """Return untracked oversized files that need explicit local ignore entries.

    Manifest-owned public/backup/cache families are already ignored by family rules.
    This catches new large files that appear in git-owned or unclassified areas before
    an operator has added the right manifest family.
    """

    if not workspace.exists():
        return []
    if max_bytes is None:
        max_bytes = MAX_GIT_BLOB_BYTES
    family_list = list(families)
    tracked = set(git_ls_files(workspace))
    oversized: list[str] = []
    for rel_path, absolute_path in iter_workspace_files_for_size_guard(
        workspace, family_list
    ):
        if rel_path in BUILTIN_TRACKED or rel_path in tracked:
            continue
        ignored = ignored_family_for_path(rel_path, family_list)
        if ignored is None:
            ignored = _deeper_ignored_family_for_path(rel_path, family_list)
        if ignored is not None and rel_path not in declared_exception_paths(ignored):
            continue
        try:
            byte_count = absolute_path.stat().st_size
        except OSError:
            continue
        if byte_count > max_bytes:
            oversized.append(rel_path)
    return sorted(oversized)


def _deeper_ignored_family_for_path(path: str, families: Iterable[Family]) -> Family | None:
    """Return an ignored family that owns ``path`` more deeply than any git family.

    ``family_specificity`` ranks any literal path above any glob, so the
    whole-tree ``storage`` git family outranks a deeper ignored glob family such
    as ``storage/papers/proceedings/*/{*.pdf,text}``. The rendered gitignore
    still ignores those files (the glob rule is written and ``storage`` is not
    re-included), so the oversized guard must not list them as explicit local
    entries: doing so made the managed block depend on which checkout holds the
    payload. Used only by the size guard; the tracking validators keep the
    literal-first ranking.
    """

    family_list = list(families)

    def depth(family: Family) -> int:
        return len((family.path or family.path_glob or "").split("/"))

    ignored = [f for f in family_list if f.is_ignored and family_matches_path(f, path)]
    if not ignored:
        return None
    best_ignored = max(ignored, key=depth)
    git = [f for f in family_list if f.is_git and family_matches_path(f, path)]
    if git and max(depth(f) for f in git) >= depth(best_ignored):
        return None
    return best_ignored


def iter_workspace_files_for_size_guard(
    workspace: Path,
    families: Iterable[Family],
) -> Iterable[tuple[str, Path]]:
    family_list = list(families)
    stack = [workspace]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        for entry in entries:
            try:
                rel_path = entry.relative_to(workspace).as_posix()
            except ValueError:
                continue
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if should_prune_size_guard_directory(rel_path, family_list):
                    continue
                stack.append(entry)
                continue
            if entry.is_file():
                yield rel_path, entry


def should_prune_size_guard_directory(path: str, families: Iterable[Family]) -> bool:
    ignored = ignored_family_for_path(path, families)
    if ignored is None:
        return False
    exception_paths = declared_exception_paths(ignored)
    if any(
        exception == path or exception.startswith(f"{path}/")
        for exception in exception_paths
    ):
        return False
    for family in families:
        if family.is_git and family.path is not None:
            if family.path == path or family.path.startswith(f"{path}/"):
                return False
    return True


def validate_check_ignore(workspace: Path, families: Iterable[Family]) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    for family in families:
        probes = policy_probes(family)
        for probe, should_ignore in probes:
            ignored = git_check_ignore(workspace / probe)
            if should_ignore and not ignored:
                blockers.append(
                    {
                        "code": "workspace_gitignore_expected_ignored_path_not_ignored",
                        "family_id": family.id,
                        "path": f"{display_path(workspace)}/{probe}",
                        "retryable": True,
                    }
                )
            if not should_ignore and ignored:
                blockers.append(
                    {
                        "code": "workspace_gitignore_expected_trackable_path_ignored",
                        "family_id": family.id,
                        "path": f"{display_path(workspace)}/{probe}",
                        "retryable": True,
                    }
                )
    return blockers


def policy_probes(family: Family) -> list[tuple[str, bool]]:
    if family.tracked_samples:
        return [(sample, family.is_ignored) for sample in family.tracked_samples]
    if family.path_glob:
        probe = expand_brace_glob(family.path_glob)[0].replace("*", "__policy_probe__")
        return [(probe, family.is_ignored)]
    assert family.path is not None
    path = family.path
    if family.is_git:
        probe = path if looks_like_file(path) else f"{path}/__policy_probe__.png"
        return [(probe, False)]
    probe = path if looks_like_file(path) else f"{path}/__policy_probe__.tmp"
    probes = [(probe, True)]
    for exception in family.gitignore_exceptions:
        probes.append((f"{path}/{exception}", False))
    return probes


def ignored_family_for_path(path: str, families: Iterable[Family]) -> Family | None:
    ignored_matches = [
        family
        for family in families
        if family.is_ignored and family_matches_path(family, path)
    ]
    if not ignored_matches:
        return None
    git_matches = [
        family
        for family in families
        if family.is_git and family_matches_path(family, path)
    ]
    best_ignored = max(ignored_matches, key=family_specificity)
    if git_matches:
        best_git = max(git_matches, key=family_specificity)
        if family_specificity(best_git) > family_specificity(best_ignored):
            return None
    return best_ignored


def family_matches_path(family: Family, path: str) -> bool:
    if family.path_glob:
        return any(fnmatch.fnmatch(path, path_glob) for path_glob in expand_brace_glob(family.path_glob))
    assert family.path is not None
    base = family.path
    return path == base or path.startswith(f"{base}/")


def family_specificity(family: Family) -> tuple[int, int]:
    """Rank families so a literal path always outranks a glob at equal or
    lesser depth (matching real gitignore's "more specific wins" intent).

    Plain segment-count alone ties a literal, non-wildcard git family (e.g.
    a single committed exception file) against an equally-deep wildcard glob
    family (e.g. `decks/**/*.{html,pdf,...}`), which made `ignored_family_for_path`
    fall back to the ignored family even though the real generated `.gitignore`
    text (via the "protected" re-echo of specific git families after the
    ignored block) correctly treats the literal path as trackable. See #REND-02
    ai4physics-talk `presentation.html` exception.
    """
    key = family.path or family.path_glob or ""
    is_literal = 0 if any(char in key for char in "*?[]") else 1
    return (is_literal, len(key.split("/")))


def declared_exception_paths(family: Family) -> set[str]:
    if family.path is None:
        return set()
    return {f"{family.path}/{exception}" for exception in family.gitignore_exceptions}


def git_ls_files(workspace: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", display_path(workspace)],
        cwd=REPO_ROOT,
        text=False,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    prefix = display_path(workspace).rstrip("/") + "/"
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8")
        paths.append(path[len(prefix) :] if path.startswith(prefix) else path)
    return paths


def git_check_ignore(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", display_path(path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
