#!/usr/bin/env python3
"""Turn a workspace's Google Drive symlinks back into local directories.

The inverse of ``scripts/restore_drive_symlinks.py``: that tool replaces local
payloads with symlinks INTO a mounted Drive leaf; this one copies each linked
payload back to local disk and swaps the symlink for the real directory, so
the pipeline (lineage hashing, index migration, sandboxed sessions) can read
it. Drive keeps its copy and stays the backup leaf through
``storage/storage_manifest.json`` / ``scripts/sync_to_gdrive.py``.

Modes (dry run is the default and touches nothing):

* ``--dry-run``  list every symlink under the workspace whose target is a
  directory outside the repository, with file counts and bytes, and check the
  total against free local disk.
* ``--apply``    per link: copy the target into ``<name>.materializing`` next
  to the link (``rsync -a`` when available, resumable), verify that every file
  arrived with the same relative path and size (``--hash`` also compares
  SHA-256), then remove the symlink and rename the copy into place. A link
  whose copy fails verification is left untouched. Writes a JSON receipt that
  records each original target.
* ``--revert``   read that receipt and put each symlink back. The local copy is
  renamed to ``<name>.local-<utc>`` and never deleted.

Nothing is ever deleted: not on Drive, not locally.

Run it on the host (the Drive mount is not readable from a sandbox)::

    .venv/bin/python scripts/materialize_drive_symlinks.py --workspace conferences/icml2026
    .venv/bin/python scripts/materialize_drive_symlinks.py --workspace conferences/icml2026 --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPT_NAME = "drive-materialization-receipt.json"
MATERIALIZING_SUFFIX = ".materializing"
FREE_SPACE_MARGIN_BYTES = 20 * 1024**3
SCHEMA_VERSION = 1

Copier = Callable[[Path, Path], None]


@dataclass
class LinkedPayload:
    """One workspace symlink whose target is a directory outside the repo."""

    link: Path
    target: Path
    relative_link: str
    file_count: int = 0
    total_bytes: int = 0
    readable: bool = True
    error: str | None = None


@dataclass
class MaterializeResult:
    relative_link: str
    target: str
    status: str
    file_count: int = 0
    total_bytes: int = 0
    detail: str = ""
    mismatches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_link": self.relative_link,
            "target": self.target,
            "status": self.status,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "detail": self.detail,
            "mismatches": self.mismatches[:50],
        }


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{value} B"


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def find_linked_payloads(workspace: Path, *, repo_root: Path = PROJECT_ROOT) -> list[LinkedPayload]:
    """Every symlink under ``workspace`` pointing at a directory outside the repo.

    The walk never follows links, so a linked tree is reported once, at its link.
    """

    found: list[LinkedPayload] = []
    for dirpath, dirnames, filenames in os.walk(workspace, followlinks=False):
        for name in [*dirnames, *filenames]:
            candidate = Path(dirpath) / name
            if not candidate.is_symlink():
                continue
            target = Path(os.readlink(candidate))
            if not target.is_absolute():
                target = (candidate.parent / target).resolve()
            if _is_inside(target, repo_root):
                continue
            found.append(
                LinkedPayload(
                    link=candidate,
                    target=target,
                    relative_link=candidate.relative_to(workspace).as_posix(),
                )
            )
    found.sort(key=lambda item: item.relative_link)
    return found


def _raise(error: OSError) -> None:
    raise error


def _tree_listing(root: Path) -> dict[str, int]:
    """``{relative_path: size}`` for every regular file under ``root`` (links not followed).

    Raises on any unreadable directory: ``os.walk`` otherwise skips it
    silently, and an unreadable Drive mount would list as EMPTY -- which a
    copy of nothing would then "verify" against before swapping the link for
    an empty directory.
    """

    os.listdir(root)  # the root itself: os.walk reports its failure only via onerror
    listing: dict[str, int] = {}
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False, onerror=_raise):
        for name in filenames:
            path = Path(dirpath) / name
            if path.is_symlink():
                continue
            listing[path.relative_to(root).as_posix()] = path.stat().st_size
    return listing


def measure(payload: LinkedPayload) -> LinkedPayload:
    if not payload.target.is_dir():
        payload.readable = False
        payload.error = "target is not a readable directory"
        return payload
    try:
        listing = _tree_listing(payload.target)
    except OSError as error:
        payload.readable = False
        payload.error = f"{type(error).__name__}: {error}"
        return payload
    payload.file_count = len(listing)
    payload.total_bytes = sum(listing.values())
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rsync_or_copytree(source: Path, destination: Path) -> None:
    """Copy ``source``'s contents into ``destination`` (resumable with rsync)."""

    rsync = shutil.which("rsync")
    if rsync:
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run([rsync, "-a", "--partial", f"{source}/", f"{destination}/"], check=True)
        return
    shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)


def verify_copy(source: Path, destination: Path, *, hash_files: bool) -> list[str]:
    """Relative paths that are missing, extra, resized or (with hashing) altered."""

    expected = _tree_listing(source)
    actual = _tree_listing(destination)
    problems: list[str] = []
    for rel, size in sorted(expected.items()):
        if rel not in actual:
            problems.append(f"missing: {rel}")
        elif actual[rel] != size:
            problems.append(f"size: {rel} ({actual[rel]} != {size})")
        elif hash_files and _sha256(source / rel) != _sha256(destination / rel):
            problems.append(f"sha256: {rel}")
    for rel in sorted(set(actual) - set(expected)):
        problems.append(f"extra: {rel}")
    return problems


def materialize_one(
    payload: LinkedPayload, *, hash_files: bool, copier: Copier = rsync_or_copytree
) -> MaterializeResult:
    result = MaterializeResult(
        relative_link=payload.relative_link,
        target=str(payload.target),
        status="failed",
    )
    if not payload.link.is_symlink():
        result.status = "skipped"
        result.detail = "no longer a symlink"
        return result
    if not payload.target.is_dir():
        result.detail = "target is not a readable directory"
        return result
    staging = payload.link.with_name(payload.link.name + MATERIALIZING_SUFFIX)
    try:
        copier(payload.target, staging)
    except (OSError, subprocess.CalledProcessError) as error:
        result.detail = f"copy failed ({type(error).__name__}: {error}); staging kept at {staging.name} for resume"
        return result
    try:
        problems = verify_copy(payload.target, staging, hash_files=hash_files)
    except OSError as error:
        result.detail = f"verification could not read a tree ({type(error).__name__}: {error}); symlink left in place"
        return result
    listing = _tree_listing(staging)
    result.file_count = len(listing)
    result.total_bytes = sum(listing.values())
    if not listing and not problems:
        problems = ["empty: the target listed no files; refusing to swap a link for an empty directory"]
    if problems:
        result.status = "verify_failed"
        result.mismatches = problems
        result.detail = f"{len(problems)} mismatches; symlink left in place, staging kept at {staging.name}"
        return result
    original_target = os.readlink(payload.link)
    payload.link.unlink()
    try:
        staging.rename(payload.link)
    except OSError as error:
        os.symlink(original_target, payload.link)
        result.detail = f"swap failed ({error}); symlink restored"
        return result
    result.status = "materialized"
    result.detail = "verified by relative path + size" + (" + sha256" if hash_files else "")
    return result


def revert_one(entry: dict[str, Any], workspace: Path, stamp: str) -> MaterializeResult:
    link = workspace / entry["relative_link"]
    result = MaterializeResult(relative_link=entry["relative_link"], target=entry["target"], status="failed")
    if link.is_symlink():
        result.status = "skipped"
        result.detail = "already a symlink"
        return result
    if link.exists():
        backup = link.with_name(f"{link.name}.local-{stamp}")
        link.rename(backup)
        result.detail = f"local copy kept at {backup.name}"
    os.symlink(entry["target"], link)
    result.status = "reverted"
    return result


def default_receipt_path(workspace: Path) -> Path:
    return workspace / "storage" / DEFAULT_RECEIPT_NAME


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, required=True, help="conferences/<slug>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Measure and plan only (default).")
    mode.add_argument("--apply", action="store_true", help="Copy, verify and swap each link.")
    mode.add_argument("--revert", action="store_true", help="Put the links from the receipt back.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="Limit to these workspace-relative link paths (repeatable), e.g. storage/corpus.",
    )
    parser.add_argument("--hash", action="store_true", help="Also compare SHA-256 of every file (reads Drive twice).")
    parser.add_argument("--receipt-path", type=Path, help=f"Default: <workspace>/storage/{DEFAULT_RECEIPT_NAME}")
    parser.add_argument(
        "--ignore-free-space",
        action="store_true",
        help=f"Apply even when free disk is below the payload plus {human_bytes(FREE_SPACE_MARGIN_BYTES)}.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, *, copier: Copier = rsync_or_copytree) -> int:
    args = parse_args(argv)
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        print(f"workspace not found: {workspace}", file=sys.stderr)
        return 2
    receipt_path = (args.receipt_path or default_receipt_path(workspace)).resolve()
    stamp = utc_stamp()

    if args.revert:
        if not receipt_path.is_file():
            print(f"no receipt at {receipt_path}", file=sys.stderr)
            return 2
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        entries = [e for e in receipt.get("results", []) if e.get("status") == "materialized"]
        if args.only:
            entries = [e for e in entries if e["relative_link"] in set(args.only)]
        results = [revert_one(entry, workspace, stamp) for entry in entries]
        for item in results:
            print(f"{item.status:>10}  {item.relative_link}  {item.detail}")
        receipt.setdefault("reverts", []).append(
            {"reverted_at": stamp, "results": [item.to_dict() for item in results]}
        )
        write_receipt(receipt_path, receipt)
        return 0 if all(item.status in {"reverted", "skipped"} for item in results) else 1

    payloads = find_linked_payloads(workspace)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {p.relative_link for p in payloads}
        if unknown:
            print(f"not a Drive-linked path in this workspace: {sorted(unknown)}", file=sys.stderr)
            return 2
        payloads = [p for p in payloads if p.relative_link in wanted]
    if not payloads:
        print("no symlinks to directories outside the repository; nothing to do")
        return 0

    for payload in payloads:
        measure(payload)
    needed = sum(p.total_bytes for p in payloads if p.readable)
    free = shutil.disk_usage(workspace).free
    unreadable = [p for p in payloads if not p.readable]
    print(f"workspace: {workspace}")
    print(f"{'link':<36} {'files':>9} {'size':>11}  target")
    for p in payloads:
        size = human_bytes(p.total_bytes) if p.readable else "unreadable"
        print(f"{p.relative_link:<36} {p.file_count:>9} {size:>11}  {p.target}")
        if p.error:
            print(f"{'':<36} {'':>9} {'':>11}  ! {p.error}")
    print(f"total to copy: {human_bytes(needed)}; free local disk: {human_bytes(free)}")
    space_ok = free >= needed + FREE_SPACE_MARGIN_BYTES

    if not args.apply:
        if unreadable:
            print("some targets are unreadable here; run this on the host with the Drive mount available")
        if not space_ok:
            print(f"NOT ENOUGH SPACE: need the payload plus a {human_bytes(FREE_SPACE_MARGIN_BYTES)} margin")
        print("dry run: nothing changed. Re-run with --apply to materialize.")
        return 0 if not unreadable and space_ok else 1

    if unreadable:
        print("refusing to apply: some targets are unreadable (run on the host)", file=sys.stderr)
        return 1
    if not space_ok and not args.ignore_free_space:
        print("refusing to apply: not enough free disk (see --ignore-free-space)", file=sys.stderr)
        return 1

    results: list[MaterializeResult] = []
    for payload in payloads:
        print(f"materializing {payload.relative_link} ({human_bytes(payload.total_bytes)}) ...", flush=True)
        item = materialize_one(payload, hash_files=args.hash, copier=copier)
        print(f"{item.status:>14}  {item.relative_link}  {item.detail}")
        results.append(item)
    prior: dict[str, Any] = {}
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_text(encoding="utf-8"))
    kept = [
        entry
        for entry in prior.get("results", [])
        if entry.get("relative_link") not in {item.relative_link for item in results}
    ]
    write_receipt(
        receipt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generator": "scripts/materialize_drive_symlinks.py",
            "workspace": str(workspace),
            "applied_at": stamp,
            "hash_verified": bool(args.hash),
            "results": kept + [item.to_dict() for item in results],
            "reverts": prior.get("reverts", []),
        },
    )
    print(f"receipt: {receipt_path}")
    return 0 if all(item.status in {"materialized", "skipped"} for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
