#!/usr/bin/env python3
"""Conference-aware Google Drive symlink restore planner.

This tool is the *inverse* of ``scripts/sync_to_gdrive.py``. Sync pushes
existing local conference payloads UP to a mounted Google Drive backup leaf;
restore reads cleaned-up local paths and plans symlinks pointing DOWN to that
mounted leaf so the generated artifacts become browsable again without
rehydrating their large local payloads.

It resolves a conference slug to its ``conferences/<slug>`` workspace, parses
``DRIVE.md`` for the mounted leaf, reads ``storage/storage_manifest.json``,
selects eligible ``public``/``backup`` families, and emits a per-family JSON +
human plan of
``link`` / ``already_linked`` / ``missing_remote`` / ``unsafe_existing_local`` /
``skipped`` actions. It supports explicit ``--dry-run``, ``--apply``, and
``--verify`` modes; ``--receipt-path`` writes an optional JSON operation receipt.
"""

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST_PATH = Path("storage/storage_manifest.json")
DEFAULT_DRIVE_DOC = Path("DRIVE.md")

# Eligibility is keyed on storage_class only (D-05). public/backup families are
# linkable; git-owned control files and rebuildable cache families are skipped.
ELIGIBLE_STORAGE_CLASSES = frozenset({"public", "backup"})

ERROR_NO_MANIFEST = (
    "No storage manifest found for conference. Restore aborted for safety."
)
ERROR_DRIVE_LEAF_MISSING = "Mounted Drive leaf does not exist or is not a directory"
ERROR_NO_DRIVE_PATH = (
    "Could not resolve a mounted Drive leaf: provide --drive-path or add a "
    "'Mounted path, when available:' line to DRIVE.md"
)


@dataclass(frozen=True)
class ManifestFamily:
    """A single manifest artifact family (REUSED verbatim from sync_to_gdrive.py)."""

    family_id: str
    relative_path: Path | None
    path_glob: str | None
    sync_profiles: Mapping[str, bool]
    storage_class: str | None = None
    share_policy: str | None = None


@dataclass(frozen=True)
class RestoreConfig:
    """Normalized inputs for a restore-plan build."""

    conference_dir: Path
    config_path: Path
    drive_leaf: Path
    drive_leaf_name: str | None
    want_json: bool


# The complete LINK-05 action vocabulary. Every planned entry carries exactly one.
ACTION_LINK = "link"
ACTION_ALREADY_LINKED = "already_linked"
ACTION_MISSING_REMOTE = "missing_remote"
ACTION_UNSAFE_EXISTING_LOCAL = "unsafe_existing_local"
ACTION_SKIPPED = "skipped"
ACTION_VOCABULARY = (
    ACTION_LINK,
    ACTION_ALREADY_LINKED,
    ACTION_MISSING_REMOTE,
    ACTION_UNSAFE_EXISTING_LOCAL,
    ACTION_SKIPPED,
)


@dataclass(frozen=True)
class PlannedLink:
    """One planned symlink (or skip) carrying everything Phase 02 needs to execute.

    ``relative_path`` is the workspace-relative path; ``local_path`` is where the
    symlink would be created (``conference_dir / relative_path``); ``drive_path``
    is the symlink TARGET (``drive_leaf / relative_path``). For directory-level
    path families ``is_directory`` is True; for glob-derived leaves it is False.
    Skipped-family entries leave ``relative_path``/paths None and carry only the
    family id, storage class, action, and reason.
    """

    family_id: str
    storage_class: str | None
    relative_path: Path | None
    local_path: Path | None
    drive_path: Path | None
    is_directory: bool
    action: str
    reason: str


@dataclass(frozen=True)
class FamilyPlan:
    """All planned entries for one manifest family, grouped for per-family reporting."""

    family_id: str
    storage_class: str | None
    eligible: bool
    entries: tuple[PlannedLink, ...]


@dataclass(frozen=True)
class RestorePlan:
    """The complete dry-run restore plan, shaped for direct Phase 02 execution."""

    conference_dir: Path
    config_path: Path
    manifest_path: Path
    drive_leaf: Path
    drive_leaf_name: str | None
    families: tuple[FamilyPlan, ...]


@dataclass(frozen=True)
class ApplyResult:
    """Outcome of applying a restore plan to the local workspace."""

    applied_links: tuple[PlannedLink, ...]
    already_linked: tuple[PlannedLink, ...]
    blockers: tuple[PlannedLink, ...]
    hard_fault: Exception | None


@dataclass(frozen=True)
class VerificationResult:
    """Verification result for symlinks created by an apply run."""

    status: str
    links_checked: int
    unresolved_targets: tuple[Path, ...]
    git_control_violations: tuple[Path, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, mirroring sync_to_gdrive's argparse style (D-07).

    New surface vs. the sync tool: a positional conference slug XOR
    ``--conference-dir`` (required, mutually exclusive), ``--drive-path``
    (mounted-leaf override, replacing sync's ``--remote-path``), ``--json``,
    a required mode flag, and optional receipt output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Restore Google Drive symlinks for a conference workspace "
            "(inverse of sync_to_gdrive.py)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 scripts/restore_drive_symlinks.py iclr2026 --dry-run
    python3 scripts/restore_drive_symlinks.py iclr2026 --dry-run --json
    python3 scripts/restore_drive_symlinks.py iclr2026 --apply --receipt-path /tmp/restore-receipt.json
    python3 scripts/restore_drive_symlinks.py iclr2026 --verify --receipt-path /tmp/verify-receipt.json
    python3 scripts/restore_drive_symlinks.py --conference-dir conferences/iclr2026 --dry-run --json
    python3 scripts/restore_drive_symlinks.py iclr2026 --dry-run --drive-path "/Volumes/My Drive/ICLR 2026-raw"

Exactly one mode flag is required: --dry-run (read-only), --apply (creates
symlinks), or --verify (checks links).
        """,
    )
    locator_group = parser.add_mutually_exclusive_group(required=True)
    locator_group.add_argument(
        "conference",
        nargs="?",
        help="Conference slug or name such as iclr2026 (resolves conferences/<slug>)",
    )
    locator_group.add_argument(
        "--conference-dir",
        type=Path,
        help="Path to a conference workspace directory such as conferences/iclr2026",
    )
    parser.add_argument(
        "--drive-path",
        type=Path,
        help=(
            "Override the mounted Drive leaf parsed from DRIVE.md "
            "(replaces sync_to_gdrive's --remote-path)"
        ),
    )
    parser.add_argument(
        "--json",
        dest="want_json",
        action="store_true",
        help="Emit the machine-readable per-family plan as JSON",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan symlink restores without creating any symlinks (read-only)",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Create symlinks for all eligible planned entries",
    )
    mode_group.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Re-resolve created symlinks and check no Git-owned control files "
            "were replaced"
        ),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        help="Write a machine-readable JSON receipt for this run",
    )
    return parser.parse_args(argv)


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary (REUSED verbatim)."""
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
    """Reject manifest paths that escape the conference workspace (REUSED verbatim)."""
    candidate = Path(raw.strip())
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must stay inside the conference workspace")
    return candidate


def resolve_conference_paths(
    conference: str | None,
    conference_dir: Path | None,
) -> tuple[Path, Path]:
    """Resolve the workspace root and config path from a slug OR an explicit dir.

    Adapted from sync_to_gdrive.resolve_conference_paths: a bare slug resolves to
    ``PROJECT_ROOT/conferences/<slug>``; ``--conference-dir`` is honored as-is.
    Fails closed (ValueError) when the workspace or its config.yaml is missing.
    """
    if conference_dir is not None:
        resolved_dir = conference_dir.resolve()
    elif conference is not None:
        slug = conference.strip()
        if not slug:
            raise ValueError("Conference slug must not be empty")
        resolved_dir = (PROJECT_ROOT / "conferences" / slug).resolve()
    else:  # pragma: no cover - argparse mutually-exclusive group requires one.
        raise ValueError("A conference slug or --conference-dir is required")

    if not resolved_dir.exists() or not resolved_dir.is_dir():
        raise ValueError(f"Conference workspace does not exist: {resolved_dir}")

    resolved_config = resolved_dir / "config.yaml"
    if not resolved_config.exists() or not resolved_config.is_file():
        raise ValueError(f"Conference config does not exist: {resolved_config}")
    return resolved_dir, resolved_config


def resolve_manifest_path(conference_dir: Path, config_path: Path) -> Path:
    """Resolve the storage manifest path from config or the default fallback.

    REUSED verbatim from sync_to_gdrive (raises ERROR_NO_MANIFEST when absent),
    satisfying the manifest fail-closed contract (LINK-02).
    """
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
    """Load families from the manifest and validate fields (REUSED verbatim).

    Validates path-XOR-path_glob, sync_profiles booleans, and captures
    storage_class/share_policy. Fail-closed on malformed manifests (LINK-02).
    """
    try:
        manifest_data = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read storage manifest: {manifest_path}") from exc
    try:
        manifest = json.loads(manifest_data)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed storage manifest: could not parse JSON in {manifest_path}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Malformed storage manifest: expected object in {manifest_path}"
        )
    families_data = manifest.get("artifact_families")
    if not isinstance(families_data, list):
        raise ValueError(
            f"Malformed storage manifest: artifact_families missing in {manifest_path}"
        )

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
                f"Malformed storage manifest: {family_id} must define exactly one of "
                "path or path_glob"
            )
        if not isinstance(sync_profiles, dict):
            raise ValueError(
                f"Malformed storage manifest: {family_id} missing sync_profiles"
            )
        if not isinstance(sync_profiles.get("public"), bool) or not isinstance(
            sync_profiles.get("backup"), bool
        ):
            raise ValueError(
                f"Malformed storage manifest: {family_id} missing sync_profiles booleans"
            )

        relative_path = (
            safe_relative_path(family_path, label=f"{family_id}.path")
            if has_path
            else None
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


def expand_brace_glob(pattern: str) -> tuple[str, ...]:
    """Expand simple shell-style brace groups before passing patterns to glob.

    REUSED verbatim from sync_to_gdrive.py. Pure string transform, no I/O — turns
    ``decks/**/*.{html,png,svg,pdf,mp4,wav,json}`` into one ``**`` pattern per
    extension. Both the Drive-side and local-side resolvers use it.
    """
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
    """Count path segments to rank family specificity (PORTED from sync_to_gdrive).

    Mirrors sync_to_gdrive.family_specificity verbatim: a deeper path/glob is
    more specific. Used by the cross-family ownership guard (ME-01) when two
    skipped owners cover the same leaf and we want to name the most specific one.
    """
    pattern = (
        family.relative_path.as_posix()
        if family.relative_path is not None
        else family.path_glob or ""
    )
    return len(pattern.split("/"))


def _glob_to_regex(pattern: str) -> str:
    """Translate a POSIX-style glob (with ``**``) into an anchored regex.

    Pure string transform — no filesystem access (dry-run purity). ``**`` spans
    zero or more whole path segments when interior, or one or more segments when
    trailing; ``*`` and ``?`` stay within a single segment. The matching
    semantics agree with ``PurePath.full_match`` but run on Python >= 3.11.
    """
    segments = pattern.split("/")
    last_index = len(segments) - 1
    parts: list[str] = []
    for index, segment in enumerate(segments):
        is_last = index == last_index
        if segment == "**":
            # Trailing ``**`` requires at least one remaining segment; an
            # interior ``**`` may match zero or more whole segments.
            parts.append(".+" if is_last else "(?:[^/]+/)*")
            continue
        segment_regex = ""
        for char in segment:
            if char == "*":
                segment_regex += "[^/]*"
            elif char == "?":
                segment_regex += "[^/]"
            else:
                segment_regex += re.escape(char)
        parts.append(segment_regex)
        if not is_last:
            parts.append("/")
    return "^" + "".join(parts) + "$"


def _relative_path_covered_by_family(relative_path: Path, family: ManifestFamily) -> bool:
    """Return True when a workspace-relative leaf is owned by ``family``.

    Read-only string comparison only (no I/O). Ownership granularity mirrors the
    sync items sync_to_gdrive.resolve_family_sources would emit for ``family``,
    so the dedup is faithful to deduplicate_sync_items' exact-path keying: a
    ``path`` family emits ONE item for the path itself, so it owns only the
    EXACT path (not arbitrary descendants — a directory-level git family such as
    ``decks`` never suppresses the distinct leaf paths a glob family enumerates
    beneath it); a ``path_glob`` family emits one item per matching leaf, so it
    owns any leaf matching a brace-expanded sub-pattern. This is how the
    cross-family ownership guard (ME-01) decides whether a skipped git/cache
    family already claims a path an eligible family planned to link.
    """
    candidate = relative_path.as_posix()
    if family.relative_path is not None:
        return candidate == family.relative_path.as_posix()
    if family.path_glob is not None:
        for sub_pattern in expand_brace_glob(family.path_glob):
            if re.match(_glob_to_regex(sub_pattern), candidate):
                return True
    return False


def parse_drive_doc(drive_doc_path: Path) -> tuple[Path | None, str | None]:
    """Parse DRIVE.md for the mounted Drive leaf path and the Drive leaf name.

    Returns ``(mounted_path, drive_leaf_name)``; either may be None if absent.

    The mounted path lives on the line *after* ``Mounted path, when available:``
    and is backtick-quoted and CONTAINS SPACES (``.../My Drive/ICLR 2026-raw``),
    so this parser extracts the backtick-quoted span rather than splitting on
    whitespace. The path may also appear inline on the same line. The Drive leaf
    name is the backtick-quoted value after ``- Drive leaf:``.
    """
    if not drive_doc_path.is_file():
        return None, None

    try:
        lines = drive_doc_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None, None

    def first_backtick_span(text: str) -> str | None:
        start = text.find("`")
        if start == -1:
            return None
        end = text.find("`", start + 1)
        if end == -1:
            return None
        value = text[start + 1 : end].strip()
        return value or None

    mounted_path: Path | None = None
    drive_leaf_name: str | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        lowered = stripped.lower()

        if drive_leaf_name is None and lowered.startswith("- drive leaf:"):
            candidate = first_backtick_span(stripped)
            if candidate:
                drive_leaf_name = candidate

        if mounted_path is None and "mounted path, when available:" in lowered:
            # The backtick-quoted path may be inline or on the following line(s).
            after_colon = stripped.split(":", 1)[1] if ":" in stripped else ""
            candidate = first_backtick_span(after_colon)
            if candidate is None:
                for follow in lines[index + 1 : index + 4]:
                    candidate = first_backtick_span(follow)
                    if candidate:
                        break
            if candidate:
                mounted_path = Path(candidate).expanduser()

    return mounted_path, drive_leaf_name


def resolve_drive_leaf(
    conference_dir: Path,
    drive_path_override: Path | None,
) -> tuple[Path, str | None]:
    """Resolve and validate the mounted Drive leaf, failing closed if absent.

    Inverts sync_to_gdrive.prepare_destination_root: instead of validating a
    write root and creating the leaf, this STATS the leaf read-only and raises
    ValueError when it does not exist or is not a directory — even in dry-run
    (D-04 / LINK-02). It never creates the directory.
    """
    drive_doc_path = conference_dir / DEFAULT_DRIVE_DOC
    parsed_path, drive_leaf_name = parse_drive_doc(drive_doc_path)

    if drive_path_override is not None:
        candidate = drive_path_override.expanduser()
    elif parsed_path is not None:
        candidate = parsed_path
    else:
        raise ValueError(ERROR_NO_DRIVE_PATH)

    resolved = candidate.resolve()
    # Read-only stat only — fail closed, never create the directory (D-04).
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"{ERROR_DRIVE_LEAF_MISSING}: {resolved}")

    return resolved, drive_leaf_name


def normalize_args(args: argparse.Namespace) -> RestoreConfig:
    """Resolve workspace, config, and the mounted Drive leaf from parsed args."""
    conference_dir, config_path = resolve_conference_paths(
        getattr(args, "conference", None),
        args.conference_dir,
    )
    drive_leaf, drive_leaf_name = resolve_drive_leaf(conference_dir, args.drive_path)
    return RestoreConfig(
        conference_dir=conference_dir,
        config_path=config_path,
        drive_leaf=drive_leaf,
        drive_leaf_name=drive_leaf_name,
        want_json=bool(args.want_json),
    )


def _classify_local_target(local_path: Path, drive_target: Path) -> tuple[str, str]:
    """Decide the action+reason for one (local location, drive target) pair.

    Read-only ops only — no mutation. Ordering matters:
    1. missing_remote when the Drive target does not exist (REAL stat, D-02).
    2. already_linked when the local path is a symlink resolving to the target.
    3. unsafe_existing_local when the local path is a real file/non-empty dir.
    4. link when the local path is absent and the Drive target exists.
    """
    drive_exists = drive_target.exists()

    if local_path.is_symlink():
        # A symlink already present locally: equivalent only if it resolves to
        # the same computed Drive target. Evaluate before missing_remote so an
        # already-correct link is not misreported when the target is reachable.
        try:
            current_target = local_path.resolve()
        except OSError:
            current_target = None
        if current_target is not None and current_target == drive_target.resolve():
            return ACTION_ALREADY_LINKED, "local path already links to the Drive target"
        if not drive_exists:
            return (
                ACTION_MISSING_REMOTE,
                "Drive target does not exist (read-only stat)",
            )
        return (
            ACTION_UNSAFE_EXISTING_LOCAL,
            "local symlink points elsewhere; not overwriting in dry-run",
        )

    if not drive_exists:
        return ACTION_MISSING_REMOTE, "Drive target does not exist (read-only stat)"

    if local_path.exists():
        # A real (non-symlink) file or directory occupies the link location.
        if local_path.is_dir():
            try:
                non_empty = any(local_path.iterdir())
            except OSError:
                non_empty = True
            if non_empty:
                return (
                    ACTION_UNSAFE_EXISTING_LOCAL,
                    "real non-empty local directory present; refusing to shadow it",
                )
            # An empty real directory is still a real path, not an absent slot.
            return (
                ACTION_UNSAFE_EXISTING_LOCAL,
                "real (empty) local directory present; refusing to shadow it",
            )
        return (
            ACTION_UNSAFE_EXISTING_LOCAL,
            "real local file present; refusing to shadow it",
        )

    return ACTION_LINK, "local path absent; Drive target available to link"


def _plan_directory_family(
    config: RestoreConfig,
    family: ManifestFamily,
) -> list[PlannedLink]:
    """Plan ONE directory-level entry for a path family (D-01 path 1).

    symlink LOCATION = conference_dir / relative_path;
    symlink TARGET   = drive_leaf / relative_path (D-03). safe_relative_path is
    re-applied as a defensive workspace-escape guard (T-01-01).
    """
    assert family.relative_path is not None
    relative_path = safe_relative_path(
        family.relative_path.as_posix(),
        label=f"{family.family_id}.path",
    )
    # Do NOT resolve the link LOCATION: Path.resolve() dereferences an
    # already-correct symlink, which would make _classify_local_target's
    # is_symlink() check return False and starve the already_linked branch
    # (HI-01). Only the Drive TARGET is resolved for the equality comparison;
    # _classify_local_target resolves the symlink's own target internally.
    local_path = config.conference_dir / relative_path
    drive_target = (config.drive_leaf / relative_path).resolve()

    # Confirm the computed Drive target stays under the mounted leaf (T-01-02).
    if not drive_target.is_relative_to(config.drive_leaf):
        return [
            PlannedLink(
                family_id=family.family_id,
                storage_class=family.storage_class,
                relative_path=relative_path,
                local_path=local_path,
                drive_path=drive_target,
                is_directory=True,
                action=ACTION_SKIPPED,
                reason="computed Drive target escapes the mounted leaf",
            )
        ]

    action, reason = _classify_local_target(local_path, drive_target)
    return [
        PlannedLink(
            family_id=family.family_id,
            storage_class=family.storage_class,
            relative_path=relative_path,
            local_path=local_path,
            drive_path=drive_target,
            is_directory=True,
            action=action,
            reason=reason,
        )
    ]


def _plan_glob_family(
    config: RestoreConfig,
    family: ManifestFamily,
) -> list[PlannedLink]:
    """Plan FILE-LEVEL entries for a path_glob family by Drive-side enumeration.

    Inverts resolve_family_sources (D-02): after local cleanup the local payloads
    are gone, so we glob the MOUNTED DRIVE LEAF (read-only), keep is_file() hits,
    derive the workspace-relative path from each Drive hit, and plan one leaf link
    each. Glob families are ALWAYS file-leveled so a payload leaf inside a
    git-owned parent (presentation_payloads inside decks/) is never shadowed by a
    directory-level symlink (D-01 path 2 / specificity guard).
    """
    assert family.path_glob is not None
    entries: list[PlannedLink] = []
    seen: set[Path] = set()

    for sub_pattern in expand_brace_glob(family.path_glob):
        # safe_relative_path already validated the glob string at load time; it is
        # re-checked here as a defensive guard before globbing the leaf (T-01-01).
        safe_relative_path(sub_pattern, label=f"{family.family_id}.path_glob")
        try:
            hits = sorted(config.drive_leaf.glob(sub_pattern))
        except (OSError, ValueError):
            hits = []
        for drive_hit in hits:
            resolved_hit = drive_hit.resolve()
            if resolved_hit in seen:
                continue
            # Read-only stat: only file leaves become links (LINK-05).
            if not drive_hit.is_file():
                continue
            # Keep targets under the mounted leaf (T-01-02).
            if not resolved_hit.is_relative_to(config.drive_leaf):
                continue
            seen.add(resolved_hit)
            relative_path = resolved_hit.relative_to(config.drive_leaf)
            # Un-resolved link LOCATION so an already-correct symlink keeps
            # is_symlink() True inside _classify_local_target (HI-01); the
            # Drive TARGET (resolved_hit) is already resolved for comparison.
            local_path = config.conference_dir / relative_path
            action, reason = _classify_local_target(local_path, resolved_hit)
            entries.append(
                PlannedLink(
                    family_id=family.family_id,
                    storage_class=family.storage_class,
                    relative_path=relative_path,
                    local_path=local_path,
                    drive_path=resolved_hit,
                    is_directory=False,
                    action=action,
                    reason=reason,
                )
            )

    if not entries:
        # No Drive leaves matched: surface the family with an honest reason rather
        # than dropping it silently. The Drive leaf existence was already proven
        # (D-04), so an empty result means no backed-up payloads for this family.
        entries.append(
            PlannedLink(
                family_id=family.family_id,
                storage_class=family.storage_class,
                relative_path=None,
                local_path=None,
                drive_path=None,
                is_directory=False,
                action=ACTION_MISSING_REMOTE,
                reason="no Drive leaves matched the family glob (read-only)",
            )
        )

    entries.sort(
        key=lambda entry: entry.drive_path.as_posix() if entry.drive_path else ""
    )
    return entries


def _owning_skipped_family(
    relative_path: Path,
    skipped_owners: Sequence[ManifestFamily],
) -> ManifestFamily | None:
    """Return the skipped git/cache family that owns ``relative_path``, if any.

    Mirrors sync_to_gdrive's preference rule (preferred_sync_item): when more
    than one skipped owner covers the same leaf, the most specific (deepest)
    family wins so the recorded reason names the closest owner. Read-only.
    """
    best: ManifestFamily | None = None
    for owner in skipped_owners:
        if _relative_path_covered_by_family(relative_path, owner):
            if best is None or family_specificity(owner) > family_specificity(best):
                best = owner
    return best


def _reconcile_eligible_entries(
    entries: Sequence[PlannedLink],
    skipped_owners: Sequence[ManifestFamily],
) -> list[PlannedLink]:
    """Downgrade eligible-family leaves that a skipped git/cache family owns.

    Cross-family ownership guard (ME-01): an eligible public/backup glob can
    enumerate a Drive leaf whose workspace-relative path is declared by a
    skipped git/cache family (e.g. presentation_payloads' ``decks/**/*.json``
    overlapping git's presentation_manifest). sync_to_gdrive resolves this with
    deduplicate_sync_items/preferred_sync_item, always handing the path to the
    git-owned family. Here the git/cache family is skipped, so the eligible link
    is downgraded to ``skipped`` (rather than silently dropped) with a reason
    naming the owning family — keeping the plan auditable for Phase 02.
    """
    reconciled: list[PlannedLink] = []
    for entry in entries:
        if entry.relative_path is None or entry.action == ACTION_SKIPPED:
            reconciled.append(entry)
            continue
        owner = _owning_skipped_family(entry.relative_path, skipped_owners)
        if owner is None:
            reconciled.append(entry)
            continue
        reconciled.append(
            PlannedLink(
                family_id=entry.family_id,
                storage_class=entry.storage_class,
                relative_path=entry.relative_path,
                local_path=entry.local_path,
                drive_path=entry.drive_path,
                is_directory=entry.is_directory,
                action=ACTION_SKIPPED,
                reason=(
                    "owned by git/cache family "
                    f"{owner.family_id} (storage_class={owner.storage_class})"
                ),
            )
        )
    return reconciled


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_restore_plan(config: RestoreConfig) -> RestorePlan:
    """Build the dry-run symlink restore plan from the manifest and Drive leaf.

    Selection is keyed on storage_class (D-05): public/backup families are
    eligible; git and cache families are skipped with a storage_class reason
    (D-06 — no extra secret denylist). Path families plan one directory-level
    link; path_glob families plan Drive-enumerated file-level links. Eligible
    leaves whose workspace-relative path is owned by a skipped git/cache family
    are downgraded to ``skipped`` (cross-family ownership guard, ME-01). Every
    operation is read-only; the worktree is never mutated (LINK-05).
    """
    manifest_path = resolve_manifest_path(config.conference_dir, config.config_path)
    families = load_manifest_families(manifest_path)

    # Coverage of every SKIPPED (git/cache) family, collected up front so the
    # ownership guard can reconcile eligible leaves regardless of manifest order
    # (a skipped owner may be declared after the eligible family it overrides).
    skipped_owners = [
        family
        for family in families
        if family.storage_class not in ELIGIBLE_STORAGE_CLASSES
        and (family.relative_path is not None or family.path_glob is not None)
    ]

    family_plans: list[FamilyPlan] = []
    for family in families:
        eligible = family.storage_class in ELIGIBLE_STORAGE_CLASSES
        if not eligible:
            reason = f"skipped: storage_class={family.storage_class}"
            family_plans.append(
                FamilyPlan(
                    family_id=family.family_id,
                    storage_class=family.storage_class,
                    eligible=False,
                    entries=(
                        PlannedLink(
                            family_id=family.family_id,
                            storage_class=family.storage_class,
                            relative_path=None,
                            local_path=None,
                            drive_path=None,
                            is_directory=False,
                            action=ACTION_SKIPPED,
                            reason=reason,
                        ),
                    ),
                )
            )
            continue

        if family.relative_path is not None:
            entries = _plan_directory_family(config, family)
        else:
            entries = _plan_glob_family(config, family)
        entries = _reconcile_eligible_entries(entries, skipped_owners)

        family_plans.append(
            FamilyPlan(
                family_id=family.family_id,
                storage_class=family.storage_class,
                eligible=True,
                entries=tuple(entries),
            )
        )

    return RestorePlan(
        conference_dir=config.conference_dir,
        config_path=config.config_path,
        manifest_path=manifest_path,
        drive_leaf=config.drive_leaf,
        drive_leaf_name=config.drive_leaf_name,
        families=tuple(family_plans),
    )


def apply_restore_plan(plan: RestorePlan) -> ApplyResult:
    """Execute the planned symlinks: create for ACTION_LINK, no-op for
    ACTION_ALREADY_LINKED, record blockers for ACTION_UNSAFE_EXISTING_LOCAL /
    ACTION_MISSING_REMOTE. OSError during symlink creation is a hard fault (D-04).
    Skipped-family entries (ACTION_SKIPPED) are ignored silently. Per D-02, the
    plan is re-classified via _classify_local_target against live disk state
    before acting.
    """
    applied_links: list[PlannedLink] = []
    already_linked_list: list[PlannedLink] = []
    blockers_list: list[PlannedLink] = []

    for family_plan in plan.families:
        if not family_plan.eligible:
            continue
        for entry in family_plan.entries:
            if entry.action == ACTION_SKIPPED:
                continue
            if entry.local_path is None or entry.drive_path is None:
                blockers_list.append(
                    PlannedLink(
                        family_id=entry.family_id,
                        storage_class=entry.storage_class,
                        relative_path=entry.relative_path,
                        local_path=entry.local_path,
                        drive_path=entry.drive_path,
                        is_directory=entry.is_directory,
                        action=ACTION_MISSING_REMOTE,
                        reason=entry.reason,
                    )
                )
                continue
            if not entry.local_path.is_relative_to(plan.conference_dir):
                blockers_list.append(
                    PlannedLink(
                        family_id=entry.family_id,
                        storage_class=entry.storage_class,
                        relative_path=entry.relative_path,
                        local_path=entry.local_path,
                        drive_path=entry.drive_path,
                        is_directory=entry.is_directory,
                        action=ACTION_UNSAFE_EXISTING_LOCAL,
                        reason="local path escapes the conference workspace",
                    )
                )
                continue

            live_action, live_reason = _classify_local_target(
                entry.local_path,
                entry.drive_path,
            )
            live_entry = PlannedLink(
                family_id=entry.family_id,
                storage_class=entry.storage_class,
                relative_path=entry.relative_path,
                local_path=entry.local_path,
                drive_path=entry.drive_path,
                is_directory=entry.is_directory,
                action=live_action,
                reason=live_reason,
            )

            if live_action == ACTION_LINK:
                entry.local_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    entry.local_path.symlink_to(entry.drive_path)
                except OSError as exc:
                    print(
                        f"Fatal: symlink creation failed for {entry.local_path}: {exc}",
                        flush=True,
                    )
                    return ApplyResult(
                        applied_links=tuple(applied_links),
                        already_linked=tuple(already_linked_list),
                        blockers=tuple(blockers_list),
                        hard_fault=exc,
                    )
                applied_links.append(live_entry)
            elif live_action == ACTION_ALREADY_LINKED:
                already_linked_list.append(live_entry)
            elif live_action in {
                ACTION_UNSAFE_EXISTING_LOCAL,
                ACTION_MISSING_REMOTE,
            }:
                blockers_list.append(live_entry)

    return ApplyResult(
        applied_links=tuple(applied_links),
        already_linked=tuple(already_linked_list),
        blockers=tuple(blockers_list),
        hard_fault=None,
    )


def format_family_list(values: Sequence[str]) -> str:
    """Render a compact comma-joined family list (REUSED verbatim)."""
    return ", ".join(values) if values else "(none)"


def _path_str(path: Path | None) -> str | None:
    """Serialize a Path to a string (or pass through None) for JSON/report use."""
    return str(path) if path is not None else None


def _summarize_actions(family_plan: FamilyPlan) -> str:
    """Build a compact ``action=count`` summary for one family's entries."""
    counts: dict[str, int] = {}
    for entry in family_plan.entries:
        counts[entry.action] = counts.get(entry.action, 0) + 1
    return ", ".join(f"{action}={counts[action]}" for action in sorted(counts))


def _empty_action_counts() -> dict[str, int]:
    """Return a stable zero-filled action count map."""
    return {action: 0 for action in ACTION_VOCABULARY}


def git_ls_files_set(conference_dir: Path) -> set[str]:
    """Run git ls-files to enumerate git-tracked paths under conference_dir.
    Returns a set of POSIX-relative path strings (relative to conference_dir).
    On failure (git not available, non-zero exit) returns an empty set; the
    caller falls back to manifest-only git-family checks.
    """
    try:
        git_path = conference_dir.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        git_path = conference_dir
    git_path_str = git_path.as_posix()

    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", git_path_str],
            cwd=PROJECT_ROOT,
            text=False,
            capture_output=True,
            check=False,
        )
    except OSError:
        return set()
    if result.returncode != 0:
        return set()

    prefix = git_path_str.rstrip("/") + "/"
    paths: set[str] = set()
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8")
        paths.add(path[len(prefix) :] if path.startswith(prefix) else path)
    return paths


def verify_restore_result(
    created_links: Sequence[PlannedLink],
    plan: RestorePlan,
) -> VerificationResult:
    """Re-resolve each link created this run. Fail if any Drive target is
    unreachable or if any created link path falls within a git-class manifest
    family or is git-tracked. Receipt-scoped for speed (D-04 / D-08).
    """
    unresolved: list[Path] = []
    violations: list[Path] = []
    git_tracked = git_ls_files_set(plan.conference_dir)
    git_families = [
        family
        for family in load_manifest_families(plan.manifest_path)
        if family.storage_class == "git"
    ]

    for entry in created_links:
        if entry.local_path is None:
            continue
        try:
            resolved_target = entry.local_path.resolve()
        except OSError:
            unresolved.append(entry.local_path)
        else:
            if not resolved_target.exists():
                unresolved.append(entry.local_path)

        relative = entry.relative_path
        if relative is None:
            try:
                relative = entry.local_path.relative_to(plan.conference_dir)
            except ValueError:
                relative = None
        if relative is None:
            continue

        relative_posix = relative.as_posix()
        if relative_posix in git_tracked or any(
            _relative_path_covered_by_family(relative, family)
            for family in git_families
        ):
            violations.append(entry.local_path)

    status = "passed" if not unresolved and not violations else "failed"
    return VerificationResult(
        status=status,
        links_checked=len(list(created_links)),
        unresolved_targets=tuple(unresolved),
        git_control_violations=tuple(violations),
    )


def write_restore_receipt(
    path: Path,
    *,
    plan: RestorePlan | None,
    mode: str,
    exit_code: int,
    status: str,
    apply_result: ApplyResult | None = None,
    verification: VerificationResult | None = None,
    error: str | None = None,
) -> None:
    """Write the machine-readable restore receipt following sync_to_gdrive's
    serialization style. Successful leaves are summarized by count only (D-06);
    blockers are enumerated in full. The verification block is conditionally
    embedded when verification is not None (D-08).
    """
    family_counts: dict[str, dict[str, int]] = {}
    family_meta: dict[str, tuple[str | None, bool]] = {}
    totals = _empty_action_counts()

    if plan is not None:
        for family_plan in plan.families:
            family_counts[family_plan.family_id] = _empty_action_counts()
            family_meta[family_plan.family_id] = (
                family_plan.storage_class,
                family_plan.eligible,
            )
            if apply_result is None:
                for entry in family_plan.entries:
                    family_counts[family_plan.family_id][entry.action] += 1
                    totals[entry.action] += 1
            else:
                skipped_count = sum(
                    1 for entry in family_plan.entries if entry.action == ACTION_SKIPPED
                )
                family_counts[family_plan.family_id][ACTION_SKIPPED] = skipped_count
                totals[ACTION_SKIPPED] += skipped_count

    if apply_result is not None:
        for action, entries in (
            (ACTION_LINK, apply_result.applied_links),
            (ACTION_ALREADY_LINKED, apply_result.already_linked),
        ):
            for entry in entries:
                family_counts.setdefault(entry.family_id, _empty_action_counts())
                family_meta.setdefault(entry.family_id, (entry.storage_class, True))
                family_counts[entry.family_id][action] += 1
                totals[action] += 1
        for entry in apply_result.blockers:
            family_counts.setdefault(entry.family_id, _empty_action_counts())
            family_meta.setdefault(entry.family_id, (entry.storage_class, True))
            family_counts[entry.family_id][entry.action] += 1
            totals[entry.action] += 1

    blocker_entries: Sequence[PlannedLink] = ()
    if apply_result is not None:
        blocker_entries = apply_result.blockers
    elif plan is not None:
        blocker_entries = tuple(
            entry
            for family_plan in plan.families
            for entry in family_plan.entries
            if entry.action in {ACTION_UNSAFE_EXISTING_LOCAL, ACTION_MISSING_REMOTE}
        )

    families_payload = [
        {
            "family_id": family_id,
            "storage_class": family_meta.get(family_id, (None, False))[0],
            "eligible": family_meta.get(family_id, (None, False))[1],
            "action_counts": family_counts[family_id],
        }
        for family_id in sorted(family_counts)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.gdrive_symlink_restore",
        "status": status,
        "mode": mode,
        "applied": mode == "apply",
        "workspace": _path_str(plan.conference_dir) if plan else None,
        "drive_leaf": _path_str(plan.drive_leaf) if plan else None,
        "drive_leaf_name": plan.drive_leaf_name if plan else None,
        "manifest_path": _path_str(plan.manifest_path) if plan else None,
        "manifest_sha256": sha256_file(plan.manifest_path) if plan else None,
        "eligible_families": [f.family_id for f in plan.families if f.eligible]
        if plan
        else [],
        "skipped_families": [f.family_id for f in plan.families if not f.eligible]
        if plan
        else [],
        "families": families_payload,
        "totals": totals,
        "blockers": [
            {
                "family_id": entry.family_id,
                "local_path": _path_str(entry.local_path),
                "drive_path": _path_str(entry.drive_path),
                "action": entry.action,
                "reason": entry.reason,
            }
            for entry in blocker_entries
        ],
        "exit_code": exit_code,
        "generated_at": utc_now(),
        "error": error,
    }
    if verification is not None:
        payload["verification"] = {
            "status": verification.status,
            "links_checked": verification.links_checked,
            "unresolved_targets": [
                str(path) for path in verification.unresolved_targets
            ],
            "git_control_violations": [
                str(path) for path in verification.git_control_violations
            ],
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_restore_report(plan: RestorePlan) -> str:
    """Render the operator-facing per-family dry-run report (D-08 / LINK-03).

    Adapted from sync_to_gdrive.render_preflight_report: a header block followed
    by one row group per family showing family id, storage_class, local path,
    Drive path, action, and reason. Glob families summarize their resolved leaf
    LOCATION->TARGET entries (with a count) under the family row.
    """
    eligible_ids = [f.family_id for f in plan.families if f.eligible]
    skipped_ids = [f.family_id for f in plan.families if not f.eligible]

    lines: list[str] = [
        "=" * 70,
        "Conference Drive Symlink Restore Plan (dry-run)",
        "=" * 70,
        f"Workspace:        {plan.conference_dir}",
        f"Config:           {plan.config_path}",
        f"Manifest:         {plan.manifest_path}",
        f"Drive leaf name:  {plan.drive_leaf_name or '(unnamed)'}",
        f"Mounted Drive:    {plan.drive_leaf}",
        "Mode:             dry-run (no filesystem changes)",
        f"Eligible families: {format_family_list(eligible_ids)}",
        f"Skipped families:  {format_family_list(skipped_ids)}",
        "=" * 70,
    ]

    for family_plan in plan.families:
        lines.append("")
        lines.append(
            f"[{family_plan.family_id}] storage_class="
            f"{family_plan.storage_class or '(none)'} "
            f"({'eligible' if family_plan.eligible else 'skipped'})"
        )
        lines.append(f"  actions: {_summarize_actions(family_plan)}")

        # Directory-level path families and skipped families carry a single
        # entry; glob families carry many leaf entries summarized beneath a count.
        leaf_entries = [e for e in family_plan.entries if e.relative_path is not None]
        is_glob_family = (
            family_plan.eligible
            and not any(e.is_directory for e in family_plan.entries)
            and bool(leaf_entries)
        )

        if not family_plan.eligible:
            entry = family_plan.entries[0]
            lines.append(f"  action: {entry.action} | reason: {entry.reason}")
            continue

        if is_glob_family:
            lines.append(f"  resolved leaves: {len(leaf_entries)}")
            for entry in leaf_entries:
                lines.append(
                    f"    - {entry.action}: {entry.local_path} -> {entry.drive_path}"
                )
                lines.append(f"      reason: {entry.reason}")
            for entry in family_plan.entries:
                if entry.relative_path is None:
                    # Glob family that matched nothing (missing_remote summary).
                    lines.append(f"    - {entry.action}: (no leaves) | {entry.reason}")
        else:
            for entry in family_plan.entries:
                lines.append(
                    f"  local:  {entry.local_path}"
                    if entry.local_path is not None
                    else "  local:  (n/a)"
                )
                lines.append(
                    f"  drive:  {entry.drive_path}"
                    if entry.drive_path is not None
                    else "  drive:  (n/a)"
                )
                lines.append(f"  action: {entry.action} | reason: {entry.reason}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Dry run complete - no changes made")
    return "\n".join(lines)


def build_plan_json(plan: RestorePlan) -> dict[str, Any]:
    """Build a JSON-serializable dict of the plan (LINK-03 fields per family).

    Each family carries family_id, storage_class, eligible, an action summary, and
    a nested ``entries`` list whose items carry family_id, storage_class,
    local_path, drive_path, action, and reason. Glob families surface their
    resolved leaves in the same ``entries`` list. Path values are stringified.
    """
    families_json: list[dict[str, Any]] = []
    for family_plan in plan.families:
        action_counts: dict[str, int] = {}
        entries_json: list[dict[str, Any]] = []
        for entry in family_plan.entries:
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
            entries_json.append(
                {
                    "family_id": entry.family_id,
                    "storage_class": entry.storage_class,
                    "relative_path": _path_str(entry.relative_path),
                    "local_path": _path_str(entry.local_path),
                    "drive_path": _path_str(entry.drive_path),
                    "is_directory": entry.is_directory,
                    "action": entry.action,
                    "reason": entry.reason,
                }
            )
        families_json.append(
            {
                "family_id": family_plan.family_id,
                "storage_class": family_plan.storage_class,
                "eligible": family_plan.eligible,
                "action_counts": action_counts,
                "entry_count": len(entries_json),
                "entries": entries_json,
            }
        )

    return {
        "mode": "dry-run",
        "conference_dir": _path_str(plan.conference_dir),
        "config_path": _path_str(plan.config_path),
        "manifest_path": _path_str(plan.manifest_path),
        "drive_leaf": _path_str(plan.drive_leaf),
        "drive_leaf_name": plan.drive_leaf_name,
        "action_vocabulary": list(ACTION_VOCABULARY),
        "eligible_families": [f.family_id for f in plan.families if f.eligible],
        "skipped_families": [f.family_id for f in plan.families if not f.eligible],
        "families": families_json,
    }


def _mode_name(args: argparse.Namespace) -> str:
    """Return the selected CLI mode name for receipts."""
    if args.dry_run:
        return "dry-run"
    if args.apply:
        return "apply"
    if args.verify:
        return "verify"
    return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve inputs, execute the selected mode, and write optional receipts."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    plan: RestorePlan | None = None
    try:
        config = normalize_args(args)
        plan = build_restore_plan(config)
    except ValueError as exc:
        print(f"Error: {exc}", flush=True)
        if getattr(args, "receipt_path", None):
            write_restore_receipt(
                args.receipt_path,
                plan=None,
                mode=_mode_name(args),
                exit_code=1,
                status="failed",
                error=str(exc),
            )
        return 1

    apply_result: ApplyResult | None = None
    verification: VerificationResult | None = None
    receipt_status = "succeeded"
    exit_code = 0

    if args.dry_run:
        if config.want_json:
            print(json.dumps(build_plan_json(plan), indent=2), flush=True)
        else:
            print(render_restore_report(plan), flush=True)
    elif args.apply:
        apply_result = apply_restore_plan(plan)
        if apply_result.hard_fault is not None:
            print(f"Error: apply aborted - {apply_result.hard_fault}", flush=True)
            exit_code = 1
            receipt_status = "failed"
        else:
            verification = verify_restore_result(apply_result.applied_links, plan)
            if verification.status == "failed":
                print(
                    "Verification failed: "
                    f"{len(verification.unresolved_targets)} unresolved targets, "
                    f"{len(verification.git_control_violations)} git control violations",
                    flush=True,
                )
                exit_code = 1
                receipt_status = "failed"

            print(
                f"Apply complete: {len(apply_result.applied_links)} symlinks created, "
                f"{len(apply_result.already_linked)} already linked, "
                f"{len(apply_result.blockers)} blocked (see receipt for details).",
                flush=True,
            )
            for blocker in apply_result.blockers:
                print(
                    f"  BLOCKED [{blocker.action}] {blocker.local_path}: "
                    f"{blocker.reason}",
                    flush=True,
                )
    elif args.verify:
        planned_links = [
            entry
            for family_plan in plan.families
            if family_plan.eligible
            for entry in family_plan.entries
            if entry.action in {ACTION_LINK, ACTION_ALREADY_LINKED}
            and entry.local_path is not None
        ]
        verification = verify_restore_result(planned_links, plan)
        if verification.status == "failed":
            print(
                "Verification failed: "
                f"{len(verification.unresolved_targets)} unresolved targets, "
                f"{len(verification.git_control_violations)} git control violations",
                flush=True,
            )
            exit_code = 1
            receipt_status = "failed"
        else:
            print(
                f"Verification passed: {verification.links_checked} links checked.",
                flush=True,
            )

    if args.receipt_path:
        write_restore_receipt(
            args.receipt_path,
            plan=plan,
            mode=_mode_name(args),
            exit_code=exit_code,
            status=receipt_status,
            apply_result=apply_result,
            verification=verification,
        )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
