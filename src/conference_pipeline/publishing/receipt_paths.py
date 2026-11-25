"""Fail-closed receipt paths and durable same-directory JSON replacement."""

from __future__ import annotations

import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_MAX_TEMP_ATTEMPTS = 16


class ReceiptPathError(RuntimeError):
    """Stable, non-secret receipt path failure."""

    def __init__(self, code: str, label: str) -> None:
        super().__init__(f"{label}:{code}")
        self.code = code
        self.label = label


def _absolute_lexical(path: Path, *, base: Path | None = None) -> Path:
    candidate = path
    if not candidate.is_absolute():
        if base is None:
            raise ValueError("relative path requires a base")
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def workflow_receipt_root(conference_dir: Path) -> Path:
    """Return the lexical conference-local workflow receipt root."""
    return _absolute_lexical(conference_dir) / "storage" / "workflow-runs"


def _raise(code: str, label: str) -> None:
    raise ReceiptPathError(code, label)


def _validated_component(path: Path, *, leaf: bool, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _raise("missing_reviewed_receipt", label)
    if stat.S_ISLNK(metadata.st_mode):
        _raise("symlinked_receipt_component", label)
    if leaf:
        if not stat.S_ISREG(metadata.st_mode):
            _raise("receipt_not_regular", label)
    elif not stat.S_ISDIR(metadata.st_mode):
        _raise("receipt_parent_not_directory", label)
    return metadata


def _component_chain(root: Path, leaf: Path) -> list[Path]:
    relative = leaf.relative_to(root)
    chain = [root]
    current = root
    for part in relative.parts:
        current = current / part
        chain.append(current)
    return chain


def confine_workflow_receipt_path(
    conference_dir: Path,
    candidate: Path,
    *,
    project_root: Path,
    label: str,
    must_exist: bool = False,
) -> Path:
    """Confine a JSON receipt to one regular conference workflow-runs tree.

    Both lexical and resolved containment are checked. Existing symlinks are
    rejected at every component from the conference directory through the
    destination. Writable parent directories are created one component at a
    time and revalidated after creation.
    """

    project = _absolute_lexical(project_root)
    conference = _absolute_lexical(conference_dir, base=project)
    destination = _absolute_lexical(candidate, base=project)
    receipt_root = workflow_receipt_root(conference)

    try:
        conference.relative_to(project)
    except ValueError:
        _raise("conference_outside_project", label)
    try:
        destination.relative_to(receipt_root)
    except ValueError:
        _raise("outside_workflow_runs", label)
    if destination.suffix.lower() != ".json":
        _raise("receipt_not_json", label)

    _validated_component(conference, leaf=False, label=label)
    conference_chain = _component_chain(conference, destination)
    for index, component in enumerate(conference_chain):
        is_leaf = index == len(conference_chain) - 1
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            if must_exist:
                _raise("missing_reviewed_receipt", label)
            if is_leaf:
                break
            try:
                component.mkdir(mode=0o700)
            except FileExistsError:
                pass
            _validated_component(component, leaf=False, label=label)
            continue
        if stat.S_ISLNK(metadata.st_mode):
            _raise("symlinked_receipt_component", label)
        if is_leaf:
            if not stat.S_ISREG(metadata.st_mode):
                _raise("receipt_not_regular", label)
        elif not stat.S_ISDIR(metadata.st_mode):
            _raise("receipt_parent_not_directory", label)

    # Recheck the complete existing chain after any directory creation.
    for index, component in enumerate(conference_chain[:-1]):
        _validated_component(component, leaf=False, label=label)
    if destination.exists() or os.path.lexists(destination):
        _validated_component(destination, leaf=True, label=label)
    elif must_exist:
        _raise("missing_reviewed_receipt", label)

    try:
        conference_resolved = conference.resolve(strict=True)
        project_resolved = project.resolve(strict=True)
        receipt_root_resolved = receipt_root.resolve(strict=True)
        destination_resolved = destination.resolve(strict=False)
        conference_resolved.relative_to(project_resolved)
        receipt_root_resolved.relative_to(conference_resolved)
        destination_resolved.relative_to(receipt_root_resolved)
    except (FileNotFoundError, ValueError):
        _raise("outside_workflow_runs", label)
    return destination


def _parent_identity(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        _raise("receipt_parent_missing", label)
    if stat.S_ISLNK(metadata.st_mode):
        _raise("symlinked_receipt_component", label)
    if not stat.S_ISDIR(metadata.st_mode):
        _raise("receipt_parent_not_directory", label)
    return metadata


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def write_json_atomic_secure(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without following sibling temp symlinks."""

    destination = _absolute_lexical(path)
    parent = destination.parent
    label = "json_path"
    parent_before = _parent_identity(parent, label=label)
    parent_fd = -1
    temp_fd = -1
    temp_name: str | None = None
    temp_created = False
    # ``os.replace`` shares the rename implementation; CPython advertises the
    # dir-fd capability under ``os.rename`` on platforms such as macOS.
    use_dir_fds = os.open in os.supports_dir_fd and os.rename in os.supports_dir_fd

    parent_flags = os.O_RDONLY
    parent_flags |= getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, parent_flags)
        parent_pinned = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_pinned.st_mode) or not _same_identity(
            parent_before, parent_pinned
        ):
            _raise("receipt_parent_changed", label)

        open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        open_flags |= getattr(os, "O_NOFOLLOW", 0)
        for _attempt in range(_MAX_TEMP_ATTEMPTS):
            token = secrets.token_hex(16)
            temp_name = f".{destination.name}.{token}.tmp"
            try:
                if use_dir_fds:
                    temp_fd = os.open(temp_name, open_flags, 0o600, dir_fd=parent_fd)
                else:
                    temp_fd = os.open(parent / temp_name, open_flags, 0o600)
            except FileExistsError:
                continue
            temp_created = True
            break
        else:
            _raise("receipt_temp_collision_limit", label)

        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with os.fdopen(temp_fd, "w", encoding="utf-8") as handle:
            temp_fd = -1
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        if use_dir_fds:
            os.replace(
                temp_name,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        else:
            parent_now = _parent_identity(parent, label=label)
            if not _same_identity(parent_before, parent_now):
                _raise("receipt_parent_changed", label)
            os.replace(parent / temp_name, destination)
        temp_created = False
        try:
            os.fsync(parent_fd)
        except OSError:
            pass
    finally:
        if temp_fd >= 0:
            os.close(temp_fd)
        if temp_created and temp_name is not None:
            try:
                if parent_fd >= 0 and os.unlink in os.supports_dir_fd:
                    os.unlink(temp_name, dir_fd=parent_fd)
                else:
                    (parent / temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        if parent_fd >= 0:
            os.close(parent_fd)
