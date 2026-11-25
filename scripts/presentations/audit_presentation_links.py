#!/usr/bin/env python3
"""Audit presentation HTML/PDF outputs for local clickable links."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path


@dataclass(frozen=True)
class LinkIssue:
    artifact: str
    link: str
    reason: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() not in {"a", "link"}:
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(html.unescape(value.strip()))


def resolve_workspace_path(path: Path, workspace: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{label} must stay inside workspace: {resolved}"
        ) from exc
    return resolved


def workspace_relpath(path: Path, workspace: Path) -> str:
    return path.resolve().relative_to(workspace.resolve()).as_posix()


def local_link_reason(link: str, *, conference_slug: str | None = None) -> str | None:
    normalized = link.strip()
    lower = normalized.lower()
    if not normalized or lower.startswith(
        ("#", "mailto:", "tel:", "javascript:", "data:")
    ):
        return None
    if lower.startswith(("http://", "https://")):
        return None
    if lower.startswith("file:"):
        return "file_uri"
    if re.match(r"^[a-zA-Z]:[\\/]", normalized):
        return "windows_absolute_path"
    if normalized.startswith(("/Users/", "/Volumes/", "/private/", "/tmp/")):
        return "local_absolute_path"
    if "/Users/" in normalized:
        return "embedded_local_absolute_path"
    if normalized.startswith(("../analysis/", "analysis/")):
        return "local_analysis_reference"
    if "decks/analysis/" in normalized:
        return "bad_decks_analysis_reference"
    if conference_slug and f"conferences/{conference_slug}/" in normalized:
        return "repo_local_reference"
    if re.search(r"\.\./.+\.(md|pdf|html|json|txt|csv)(?:[?#].*)?$", normalized, re.I):
        return "relative_local_document_reference"
    return None


def html_links(path: Path) -> list[str]:
    parser = LinkParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser.links


def pdf_uri_links(path: Path) -> list[str]:
    text = path.read_bytes().decode("utf-8", errors="ignore")
    links = [
        html.unescape(match.group(1).strip())
        for match in re.finditer(r"/URI\s*\((.*?)\)", text)
    ]
    return links


def audit_artifact(
    path: Path,
    *,
    workspace: Path,
    conference_slug: str | None,
) -> list[LinkIssue]:
    if path.suffix == ".html":
        links = html_links(path)
    elif path.suffix == ".pdf":
        links = pdf_uri_links(path)
    else:
        links = []
    issues: list[LinkIssue] = []
    for link in links:
        reason = local_link_reason(link, conference_slug=conference_slug)
        if reason:
            issues.append(
                LinkIssue(
                    artifact=workspace_relpath(path, workspace),
                    link=link,
                    reason=reason,
                )
            )
    return issues


def audit_workspace(
    *,
    workspace: Path,
    decks_dir: Path,
    conference_slug: str | None,
) -> dict[str, object]:
    artifacts = sorted(decks_dir.glob("*/presentation.html")) + sorted(
        decks_dir.glob("*/presentation.pdf")
    )
    issues: list[LinkIssue] = []
    for artifact in artifacts:
        issues.extend(
            audit_artifact(
                artifact,
                workspace=workspace,
                conference_slug=conference_slug,
            )
        )
    return {
        "schema_version": 1,
        "status": "failed" if issues else "passed",
        "workspace": str(workspace),
        "decks_dir": workspace_relpath(decks_dir, workspace),
        "artifact_count": len(artifacts),
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--decks-dir", type=Path, default=Path("decks"))
    parser.add_argument("--conference-slug")
    parser.add_argument("--receipt-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    decks_dir = resolve_workspace_path(args.decks_dir, workspace, label="--decks-dir")
    receipt_output = (
        resolve_workspace_path(args.receipt_output, workspace, label="--receipt-output")
        if args.receipt_output
        else None
    )
    payload = audit_workspace(
        workspace=workspace,
        decks_dir=decks_dir,
        conference_slug=args.conference_slug or workspace.name,
    )
    if receipt_output:
        receipt_output.parent.mkdir(parents=True, exist_ok=True)
        receipt_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
