#!/usr/bin/env python3
"""Fail-closed audit for private markers in the platform tree or a staged public tree.

A private marker is a home-directory path, a Google Drive account folder, a cloud
storage mount, a real email address, or code that reads secrets from a shell profile.
The operator's own literals (usernames, addresses) come from
``CONFERENCE_PIPELINE_PRIVATE_MARKERS`` in the environment (comma-separated, normally
set in the local ``.env``) and are never printed.

Run:
    python scripts/audit_private_markers.py                 # the platform tree, tracked files
    python scripts/audit_private_markers.py --tree <dir>    # a staged bundle or export
    python scripts/audit_private_markers.py --json out.json # also write the report

Exit codes: 0 no finding outside the allowlist; 1 findings; 2 usage or allowlist error.
Intentional generic fixtures are exempted in ``.private-markers-allowlist.toml`` with a
reason; that list is reviewed with every change to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline._security.private_markers import (  # noqa: E402
    AllowlistError,
    builtin_rules,
    extra_rules_from_env,
    load_allowlist,
    redact_text,
    scan_paths,
)

DEFAULT_ALLOWLIST = REPO_ROOT / ".private-markers-allowlist.toml"
# The tree the public code repository is exported from (plan D2 (a)): code, tests,
# docs, skills, the starter workspace and the root files.
PLATFORM_PATHS = (
    "src",
    "scripts",
    "tests",
    "docs",
    ".claude",
    "conferences/TEMPLATE",
    ":(top,glob)*",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root (tracked mode)")
    parser.add_argument("--paths", nargs="+", help="pathspecs to scan instead of the platform tree")
    parser.add_argument("--tree", type=Path, help="scan every file under this directory instead")
    parser.add_argument("--allowlist", type=Path, help="allowlist file (default: the repository's)")
    parser.add_argument("--json", type=Path, help="write the redacted report here")
    parser.add_argument("--no-env", action="store_true", help=f"ignore {'CONFERENCE_PIPELINE_PRIVATE_MARKERS'}")
    args = parser.parse_args(argv)
    rules = builtin_rules() + (() if args.no_env else extra_rules_from_env())

    def fail(message: str) -> int:
        # Diagnostics name paths the operator passed in; redact them like findings.
        print(f"error: {redact_text(message, rules)}", file=sys.stderr)
        return 2

    try:
        if args.allowlist is not None:
            allowlist = load_allowlist(args.allowlist)
        elif args.tree is None and DEFAULT_ALLOWLIST.is_file():
            allowlist = load_allowlist(DEFAULT_ALLOWLIST)
        else:
            allowlist = load_allowlist(None)
    except AllowlistError as exc:
        return fail(str(exc))

    if args.tree is not None:
        if not args.tree.is_dir():
            return fail(f"not a directory: {args.tree}")
        report = scan_paths(args.tree, ["."], rules, allowlist, tracked_only=False)
    else:
        report = scan_paths(args.root, args.paths or list(PLATFORM_PATHS), rules, allowlist, tracked_only=True)

    for finding in report.findings:
        print(f"{finding.path}:{finding.line} {finding.rule} {finding.excerpt}")
    for entry in report.stale_allowlist:
        print(redact_text(f"stale_allowlist: {entry.path} ({entry.rule}): {entry.reason}", rules))
    counts = ", ".join(f"{rule}={n}" for rule, n in report.counts().items()) or "none"
    print(
        f"private markers: {len(report.findings)} finding(s) ({counts}); "
        f"{report.suppressed} allowlisted; {report.files_scanned} file(s) scanned; "
        f"skipped {report.skipped or {}}"
    )
    if args.json is not None:
        args.json.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if report.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
