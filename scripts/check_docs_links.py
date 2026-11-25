#!/usr/bin/env python3
"""Check local markdown links under docs/ for missing targets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

FENCE_RE = re.compile(r"^(?:```|~~~)")
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def strip_fenced_blocks(text: str) -> str:
    lines = text.splitlines()
    filtered = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        stripped = line.strip()
        if FENCE_RE.match(stripped):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif fence_marker == marker:
                in_fence = False
                fence_marker = ""
            continue
        if not in_fence:
            filtered.append(line)

    return "\n".join(filtered)


def normalize_target(target: str) -> str:
    target = target.strip().strip("<>")
    if not target:
        return ""

    if target.startswith(("http://", "https://", "mailto:", "tel:", "data:")):
        return ""

    if target.startswith("#"):
        return ""

    # Drop title portion for local paths: (path "title")
    if " " in target and not target.startswith("http"):
        target = target.split(" ", 1)[0]

    target = target.split("#", 1)[0].split("?", 1)[0]
    return target


def check_docs(root: Path) -> int:
    broken: list[tuple[Path, str]] = []

    for md_path in root.rglob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        filtered = strip_fenced_blocks(text)

        for match in LINK_RE.finditer(filtered):
            raw_target = match.group(1)
            target = normalize_target(raw_target)
            if not target:
                continue

            if target.startswith("/"):
                path = Path(".") / target.lstrip("/")
            else:
                path = (md_path.parent / target).resolve()

            if path.exists():
                continue

            broken.append((md_path, raw_target))

    if broken:
        print("Broken links found:")
        for md_path, target in broken:
            print(f"- {md_path}: {target}")
        return 1

    print("No broken local markdown links in docs/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check docs/ for broken local links.")
    parser.add_argument(
        "docs_root",
        nargs="?",
        default="docs",
        help="Docs root directory (default: docs)",
    )
    args = parser.parse_args()

    root = Path(args.docs_root)
    if not root.exists():
        print(f"Docs root not found: {root}")
        return 1

    return check_docs(root)


if __name__ == "__main__":
    sys.exit(main())
