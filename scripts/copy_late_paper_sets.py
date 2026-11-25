#!/usr/bin/env python3
"""Copy late paper sets into a conference workspace, with digests (v6.0 Phase 5, spec §6.3).

The workspace declares what to copy in ``storage/papers/late_copy_plan.json``:

    {"schema_version": 1,
     "source_label": "<where the source tree lives, as a path relative to its checkout>",
     "copy": ["<path relative to --source>", ...],      # directories or files: payloads, receipts
     "unit_files": ["units/<name>.jsonl", ...]}           # copied with their paths rewritten

Every copied file lands at ``storage/papers/late/<same relative path>``. Each one is hashed
at the source and again after the copy; a mismatch stops the run. A unit file's ``path``
fields (absolute paths under --source) are rewritten to workspace-relative
paths under ``storage/papers/late/``, and each rewritten path must name a copied file. The
source tree is only read. A rerun skips files whose copy already matches by size and
sha256. The receipt, ``storage/papers/late_copy_receipt.json``, records counts, bytes and
digests with relative paths only.

Exit codes: 0 copied (or planned, with --dry-run); 1 a digest mismatch or a unit path
that names no copied file; 2 usage or plan error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.publishing.receipt_paths import write_json_atomic_secure  # noqa: E402

PLAN = Path("storage") / "papers" / "late_copy_plan.json"
RECEIPT = Path("storage") / "papers" / "late_copy_receipt.json"
DEST = Path("storage") / "papers" / "late"


class CopyError(Exception):
    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(source: Path, rel: object) -> str:
    """A plan path: relative, no '..', resolving inside --source (review F3)."""
    text = str(rel or "")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise CopyError(f"plan paths must be relative to --source without '..': {text!r}", 2)
    try:
        (source / path).resolve().relative_to(source)
    except ValueError:
        raise CopyError(f"plan path escapes --source: {text!r}", 2) from None
    return path.as_posix()


def _files(source: Path, rel: str) -> list[Path]:
    root = source / rel
    if root.is_symlink():
        raise CopyError(f"plan entry is a symlink: {rel}", 2)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise CopyError(f"plan entry not found: {rel}", 2)
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CopyError(f"symlink inside a plan entry: {path.relative_to(source)}", 2)
        if path.is_file():
            out.append(path)
    return out


def _refuse_link(dst: Path) -> None:
    for node in (dst, *dst.parents):
        if node.is_symlink():
            raise CopyError(f"destination path goes through a symlink: {node}", 2)
        if node.name == "late":
            break


def _copy_verified(src: Path, dst: Path) -> tuple[str, bool]:
    _refuse_link(dst)
    src_sha = sha256_file(src)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == src_sha:
        return src_sha, False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if sha256_file(dst) != src_sha:
        raise CopyError(f"digest mismatch after copy: {dst}", 1)
    return src_sha, True


def _rewrite_unit_file(src: Path, dst: Path, workspace: Path, source: Path) -> dict[str, Any]:
    rows: list[str] = []
    rewritten = 0
    for number, line in enumerate(src.read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        unit = json.loads(line)
        if unit.get("path"):
            try:
                inside = Path(str(unit["path"])).resolve().relative_to(source)
            except ValueError:
                raise CopyError(f"{src.name}:{number}: unit path is not under --source", 1) from None
            relative = (DEST / inside).as_posix()
            if not (workspace / relative).is_file():
                raise CopyError(f"{src.name}:{number}: unit path names no copied file: {relative}", 1)
            unit["path"] = relative
            rewritten += 1
        # ASCII-escaped: a raw U+2028 inside a string would split the record for a
        # str.splitlines() reader (IF7).
        rows.append(json.dumps(unit, sort_keys=True))
    _refuse_link(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(row + "\n" for row in rows)
    dst.write_text(text, encoding="utf-8")
    return {
        "name": dst.name,
        "rows": len(rows),
        "paths_rewritten": rewritten,
        "source_sha256": sha256_file(src),
        "copy_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def run(source: Path, workspace: Path, *, dry_run: bool) -> dict[str, Any]:
    plan_path = workspace / PLAN
    if not plan_path.is_file():
        raise CopyError(f"no copy plan at {PLAN.as_posix()}", 2)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    destination = (workspace / DEST).resolve()
    for one, other in ((destination, source), (source, destination)):
        if one == other or other in one.parents:
            raise CopyError("--source and the workspace's storage/papers/late must not overlap", 2)
    copy_entries = [_safe_relative(source, rel) for rel in plan.get("copy") or []]
    unit_entries = [_safe_relative(source, rel) for rel in plan.get("unit_files") or []]
    entries: list[dict[str, Any]] = []
    total_files = total_bytes = copied = 0
    for rel in copy_entries:
        files = _files(source, rel)
        size = sum(path.stat().st_size for path in files)
        entry: dict[str, Any] = {"path": str(rel), "files": len(files), "bytes": size}
        if not dry_run:
            rows = []
            for path in files:
                relative = path.relative_to(source).as_posix()
                sha, did_copy = _copy_verified(path, workspace / DEST / relative)
                copied += did_copy
                rows.append(f"{relative}\0{sha}\0{path.stat().st_size}")
            entry["tree_sha256"] = hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()
        entries.append(entry)
        total_files += len(files)
        total_bytes += size
    units = []
    for rel in unit_entries:
        src = source / rel
        if not src.is_file():
            raise CopyError(f"unit file not found: {rel}", 2)
        units.append(
            {"name": src.name, "bytes": src.stat().st_size}
            if dry_run
            else _rewrite_unit_file(src, workspace / DEST / rel, workspace, source)
        )
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.late_paper_copy",
        "source_label": plan.get("source_label", ""),
        "destination": DEST.as_posix(),
        "dry_run": dry_run,
        "copy": entries,
        "unit_files": units,
        "totals": {"files": total_files, "bytes": total_bytes, "copied_this_run": copied},
        "source_modified": False,
    }
    if not dry_run:
        write_json_atomic_secure(workspace / RECEIPT, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="the source tree, e.g. <talk>/storage/augment")
    parser.add_argument("--workspace", type=Path, required=True, help="the conference workspace, conferences/<slug>")
    parser.add_argument("--dry-run", action="store_true", help="list files and bytes; copy nothing")
    args = parser.parse_args(argv)
    if not args.source.is_dir() or not args.workspace.is_dir():
        print("error: --source and --workspace must be directories", file=sys.stderr)
        return 2
    try:
        receipt = run(args.source.resolve(), args.workspace.resolve(), dry_run=args.dry_run)
    except CopyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.code
    totals = receipt["totals"]
    print(
        f"late paper sets: {len(receipt['copy'])} entries, {totals['files']} files, "
        f"{totals['bytes'] / 1e9:.2f} GB, {len(receipt['unit_files'])} unit files; "
        + ("dry run, nothing copied" if args.dry_run else f"{totals['copied_this_run']} copied this run")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
