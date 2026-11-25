#!/usr/bin/env python3
"""Inline traceability references into analysis topic documents."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Entry:
    quote: str
    source_title: str | None
    presentation_id: str | None
    slide: str | None
    timestamp: str | None
    paper_id: str | None
    paper_url: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inline addenda references into topic docs.")
    parser.add_argument(
        "--analysis-dir",
        default="conferences/neurips2025/analysis",
        help="Directory containing analysis docs.",
    )
    parser.add_argument(
        "--addenda-dir",
        default="conferences/neurips2025/analysis/traceability_addenda",
        help="Directory containing traceability addenda.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show intended changes without writing files.",
    )
    return parser.parse_args()


def parse_addendum(addendum_path: Path) -> tuple[Path | None, list[Entry]]:
    lines = addendum_path.read_text(encoding="utf-8").splitlines()
    source_doc = None
    entries: list[Entry] = []

    quote = None
    source_title = None
    presentation_id = None
    slide = None
    timestamp = None
    paper_id = None
    paper_url = None

    def flush() -> None:
        nonlocal quote, source_title, presentation_id, slide, timestamp, paper_id, paper_url
        if quote:
            entries.append(
                Entry(
                    quote=quote,
                    source_title=source_title,
                    presentation_id=presentation_id,
                    slide=slide,
                    timestamp=timestamp,
                    paper_id=paper_id,
                    paper_url=paper_url,
                )
            )
        quote = None
        source_title = None
        presentation_id = None
        slide = None
        timestamp = None
        paper_id = None
        paper_url = None

    for line in lines:
        if line.startswith("Source document:"):
            match = re.search(r"`([^`]+)`", line)
            if match:
                source_doc = Path(match.group(1))
            continue

        if line.startswith("## Entry "):
            flush()
            continue

        if line.startswith("Quote:"):
            match = re.search(r"Quote:\s*\"(.+)\"\s*$", line)
            if match:
                quote = match.group(1)
            continue

        if line.startswith("Source Title:"):
            source_title = value_or_none(line)
        elif line.startswith("Presentation ID:"):
            presentation_id = value_or_none(line)
        elif line.startswith("Slide:"):
            slide = value_or_none(line)
        elif line.startswith("Timestamp:"):
            timestamp = value_or_none(line)
        elif line.startswith("Paper ID:"):
            paper_id = value_or_none(line)
        elif line.startswith("Paper URL:"):
            paper_url = value_or_none(line)

    flush()
    return source_doc, entries


def value_or_none(line: str) -> str | None:
    value = line.split(":", 1)[1].strip()
    if value == "UNKNOWN":
        return None
    return value


def build_reference(entry: Entry) -> str | None:
    parts: list[str] = []
    if entry.source_title:
        parts.append(f"Source: {entry.source_title}")
    if entry.presentation_id:
        parts.append(f"Presentation ID: {entry.presentation_id}")
    if entry.slide:
        parts.append(f"Slide: {entry.slide}")
    if entry.timestamp:
        parts.append(f"Timestamp: {entry.timestamp}")
    if entry.paper_id:
        parts.append(f"Paper ID: {entry.paper_id}")
    if entry.paper_url:
        parts.append(f"URL: {entry.paper_url}")
    if not parts:
        return None
    return " [" + "; ".join(parts) + "]"


def inline_references(doc_path: Path, entries: list[Entry], dry_run: bool) -> int:
    content = doc_path.read_text(encoding="utf-8")
    updated = content
    changes = 0

    for entry in entries:
        reference = build_reference(entry)
        if not entry.quote or not reference:
            continue
        quoted = f"\"{entry.quote}\""
        if quoted not in updated:
            continue
        target = quoted + reference
        if target in updated:
            continue
        updated = updated.replace(quoted, target)
        changes += 1

    if changes and not dry_run:
        doc_path.write_text(updated, encoding="utf-8")
    return changes


def main() -> None:
    args = parse_args()
    addenda_dir = Path(args.addenda_dir)
    if not addenda_dir.exists():
        raise SystemExit(f"Addenda directory not found: {addenda_dir}")

    addenda_files = sorted(addenda_dir.glob("*_traceability_addendum.md"))
    total_changes = 0
    for addendum_path in addenda_files:
        source_doc, entries = parse_addendum(addendum_path)
        if not source_doc or not source_doc.exists():
            continue
        changes = inline_references(source_doc, entries, args.dry_run)
        total_changes += changes
        if changes:
            print(f"{source_doc}: added {changes} references")

    if args.dry_run:
        print(f"Dry run complete. Proposed changes: {total_changes}")
    else:
        print(f"Inline references added: {total_changes}")


if __name__ == "__main__":
    main()
