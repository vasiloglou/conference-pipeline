#!/usr/bin/env python3
"""Install the repo-local plan-conference-production skill into user skill dirs."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


SKILL_NAME = "plan-conference-production"
ALLOWED_TARGETS = ("codex", "claude")
COPIED_PATTERNS = ("SKILL.md", "agents/openai.yaml", "references/*.md", "scripts/*.py")


def skill_root() -> Path:
    return Path(__file__).resolve().parent.parent


def skill_hash(root: Path) -> str:
    return hashlib.sha256((root / "SKILL.md").read_bytes()).hexdigest()


def target_path(label: str) -> Path:
    home = Path.home()
    if label == "codex":
        return home / ".codex" / "skills" / SKILL_NAME
    if label == "claude":
        return home / ".claude" / "skills" / SKILL_NAME
    raise ValueError(f"unknown install target: {label}")


def files_to_copy(root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in COPIED_PATTERNS:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(files))


def verify_target(label: str, target: Path) -> Path:
    expected = target_path(label).resolve()
    resolved = target.resolve()
    parent = expected.parent.resolve()
    if resolved != expected:
        raise ValueError(f"unknown install target: {target}")
    if parent not in resolved.parents:
        raise ValueError(f"target escapes user skill parent: {target}")
    return resolved


def install(label: str, *, dry_run: bool) -> str:
    root = skill_root()
    target = verify_target(label, target_path(label))
    digest = skill_hash(root)
    if dry_run:
        return f"dry-run {label}: {target} skill_md_sha256={digest}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for source in files_to_copy(root):
        if source.is_symlink():
            raise ValueError(f"refusing symlink: {source}")
        relative = source.relative_to(root)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return f"installed {label}: {target} skill_md_sha256={digest}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Install {SKILL_NAME} skill.")
    parser.add_argument("--target", action="append", help='choices=("codex", "claude")')
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    targets = args.target or list(ALLOWED_TARGETS)
    unknown = [target for target in targets if target not in ALLOWED_TARGETS]
    if unknown:
        print("unknown install target: " + ", ".join(unknown), file=sys.stderr)
        return 2
    try:
        for target in targets:
            print(install(target, dry_run=args.dry_run))
        return 0
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
