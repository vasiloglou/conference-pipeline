#!/usr/bin/env python3
"""Build the fresh-history public export of the platform code and check it.

Exit 0: the export was built (or, with --check-only, checked) and every gate is clean.
Exit 1: a gate found something (private marker, gitleaks finding, broken docs link).
Exit 2: refused or misused: the output exists, the commit is unknown, the identity is a
private marker, or code_export.toml is missing at the commit.

It never pushes or adds a remote (v6.0 Phase 7; publishing is P8, with the owner's approval).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline._security.private_markers import builtin_rules, extra_rules_from_env, redact_text  # noqa: E402
from conference_pipeline.publishing.code_export import (  # noqa: E402
    CodeExportConfig,
    CodeExportError,
    build_code_export,
    check_code_export,
)
from conference_pipeline.publishing.receipt_paths import write_json_atomic_secure  # noqa: E402


def _fold(path: Path) -> tuple[str, ...]:
    return tuple(unicodedata.normalize("NFC", part).casefold() for part in path.parts)


def _inside(path: Path, directory: Path) -> bool:
    """Whether ``path`` is ``directory`` or below it, with names case- and NFC-folded."""
    folded, top = _fold(path), _fold(directory)
    return folded[: len(top)] == top


def _same_directory_above(receipt: Path, out: Path) -> bool:
    """Whether the receipt's directory, or one above it, is the export directory itself."""
    target = Path(os.path.abspath(receipt)).parent
    out_stat = os.stat(out)
    for directory in (target, *target.parents):
        try:
            found = os.stat(directory)
        except OSError:
            continue
        if (found.st_dev, found.st_ino) == (out_stat.st_dev, out_stat.st_ino):
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--source-commit", default="HEAD")
    parser.add_argument("--out", type=Path, help="export directory to create (must not exist)")
    parser.add_argument("--author-name")
    parser.add_argument("--author-email")
    parser.add_argument("--receipt-path", type=Path, help="default: storage/workflow-runs/code-export-<sha>-receipt.json")
    parser.add_argument("--dry-run", action="store_true", help="list the files and write nothing")
    parser.add_argument("--check-only", type=Path, metavar="OUT", help="run the gates over an existing export")
    args = parser.parse_args(argv)
    rules = builtin_rules() + extra_rules_from_env()
    # Every path is absolute from here on: the receipt writer needs one (review r1 F3).
    args.project_root = args.project_root.resolve()
    for name in ("out", "receipt_path", "check_only"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).absolute())

    def say(text: str) -> None:
        print(redact_text(text, rules))

    if args.check_only is not None:
        if not (args.check_only / ".git").is_dir():
            say(f"error: not an export repository: {args.check_only}")
            return 2
        findings = check_code_export(args.check_only)
        for item in findings:
            say(f"finding: {item['gate']} {item.get('path', '')} {item['detail']}".replace("  ", " "))
        say(f"code export check: {len(findings)} finding(s)")
        return 1 if findings else 0

    if not args.dry_run and (args.out is None or not args.author_name or not args.author_email):
        say("error: --out, --author-name and --author-email are required unless --dry-run")
        return 2
    receipt = args.receipt_path or (args.project_root / "storage" / "workflow-runs" / "code-export-receipt.json")
    # Compared where the bytes land: the export where the builder's filesystem calls put it,
    # the receipt where its writer puts it, ".." collapsed first and symlinks then followed
    # (pr-43-r1 F2, phase-7-r2 F2).
    # Names are compared case- and normalization-folded, since the macOS default volume
    # treats Export and export as one directory (phase-7-r4 F1); on a case-sensitive volume
    # this refuses a few receipt paths it need not, never one it should.
    out_real = args.out.resolve() if args.out is not None else None
    receipt_real = Path(os.path.abspath(receipt)).resolve()
    if out_real is not None and _inside(receipt_real, out_real):
        # A receipt written inside the export would change it after its gates (review r1 F2).
        say("refused: receipt_inside_export: the receipt must be written outside the export")
        return 2
    config = CodeExportConfig(
        project_root=args.project_root,
        source_commit=args.source_commit,
        out_dir=args.out or Path("/nonexistent"),
        author_name=args.author_name or "dry-run",
        author_email=args.author_email or "dry-run@example.invalid",
        dry_run=args.dry_run,
    )
    try:
        payload = build_code_export(config)
    except CodeExportError as exc:
        say(f"refused: {exc}")
        return 2
    if args.dry_run:
        for path in payload["files"]:
            print(path)
        say(f"code export dry run: {payload['file_count']} file(s) at {payload['source_commit'][:12]}; nothing written")
        return 0
    findings = check_code_export(args.out)
    payload["gates"] = {
        gate: sum(1 for item in findings if item["gate"] == gate) for gate in ("private_markers", "secrets", "docs_links")
    }
    if args.receipt_path is None:
        receipt = args.project_root / "storage" / "workflow-runs" / f"code-export-{payload['source_commit'][:12]}-receipt.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if _same_directory_above(receipt, args.out):
        # The export exists now, so the filesystem itself can say whether the receipt's
        # directory is the export or inside it, whatever its spelling (phase-7-r4 F1).
        say("refused: receipt_inside_export: the receipt must be written outside the export")
        return 2
    write_json_atomic_secure(receipt, payload)
    for item in findings:
        say(f"finding: {item['gate']} {item.get('path', '')} {item['detail']}".replace("  ", " "))
    say(
        f"code export: {payload['file_count']} file(s), export commit {payload['export_commit'][:12]} "
        f"from {payload['source_commit'][:12]}; {len(findings)} finding(s); receipt {receipt.name}"
    )
    print(json.dumps(payload["gates"], sort_keys=True))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
