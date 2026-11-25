#!/usr/bin/env python3
"""Render a by-topic extraction file with dereferenced source text.

The Phase 14A by-topic JSON files are compact rollups. This script expands one
topic into a Markdown review file by joining each rollup item to its matching
by-artifact extraction record and the processed transcript/OCR text under
storage/transcription-ocr/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return sanitize_text(path.read_text(encoding="utf-8", errors="replace"))


def sanitize_text(text: str) -> str:
    """Keep Markdown readable when upstream OCR/transcripts contain controls."""
    return "".join(
        ch
        if ch in "\n\r\t" or ord(ch) >= 32
        else "\uFFFD"
        for ch in text
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def media_id_from_source_path(source_path: str) -> str | None:
    parts = Path(source_path).parts
    try:
        media_index = parts.index("media")
    except ValueError:
        return None
    if len(parts) <= media_index + 1:
        return None
    return parts[media_index + 1]


def build_by_artifact_index(extractions_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    index: dict[str, tuple[Path, dict[str, Any]]] = {}
    by_artifact_dir = extractions_root / "by-artifact"
    for path in sorted(by_artifact_dir.glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        source_path = payload.get("source_artifact_path")
        if isinstance(source_path, str) and source_path:
            index.setdefault(source_path, (path, payload))
    return index


def dereference_paths(workspace: Path, source_path: str) -> dict[str, Path | None]:
    media_id = media_id_from_source_path(source_path)
    if not media_id:
        return {"primary": None, "companion": None}

    media_dir = workspace / "storage" / "transcription-ocr" / "media" / media_id
    if "/audio/" in source_path:
        primary = media_dir / "transcript.txt"
        companion = media_dir / "text.txt"
    else:
        primary = media_dir / "text.txt"
        companion = media_dir / "transcript.txt"
    return {
        "primary": primary if primary.exists() else None,
        "companion": companion if companion.exists() else None,
    }


def excerpt_from_span(text: str, span: dict[str, Any] | None, *, context: int = 240) -> str:
    if not isinstance(span, dict):
        return ""
    try:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
    except (TypeError, ValueError):
        return ""
    if start < 0 or end <= start:
        return ""
    left = max(0, start - context)
    right = min(len(text), end + context)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + text[left:right].strip() + suffix


def fence_text(text: str) -> str:
    return "````text\n" + text.rstrip() + "\n````"


def md_escape_table(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def render_item(
    *,
    index: int,
    item: dict[str, Any],
    artifact_record: tuple[Path, dict[str, Any]] | None,
    workspace: Path,
) -> tuple[str, dict[str, int]]:
    stats = Counter()
    source_path = str(item.get("source_artifact_path") or "")
    media_id = media_id_from_source_path(source_path) or ""
    paths = dereference_paths(workspace, source_path)
    primary_path = paths["primary"]
    companion_path = paths["companion"]
    primary_text = read_text(primary_path) if isinstance(primary_path, Path) else ""
    companion_text = read_text(companion_path) if isinstance(companion_path, Path) else ""

    if primary_text:
        stats["primary_text_found"] += 1
    else:
        stats["primary_text_missing"] += 1
    if companion_text:
        stats["companion_text_found"] += 1

    artifact_path: Path | None = None
    artifact_payload: dict[str, Any] = {}
    if artifact_record:
        artifact_path, artifact_payload = artifact_record
        stats["by_artifact_found"] += 1
    else:
        stats["by_artifact_missing"] += 1

    evidence = artifact_payload.get("evidence") if artifact_payload else {}
    span = evidence.get("span") if isinstance(evidence, dict) else None
    span_excerpt = excerpt_from_span(primary_text, span)
    if span_excerpt:
        stats["span_resolved"] += 1

    quality = artifact_payload.get("quality_metadata") if artifact_payload else {}
    topic_scores = quality.get("topic_scores") if isinstance(quality, dict) else {}

    lines: list[str] = [
        f"## {index}. `{item.get('artifact_id')}`",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Media id | `{md_escape_table(media_id)}` |",
        f"| Source artifact path | `{md_escape_table(source_path)}` |",
        f"| Source artifact digest | `{md_escape_table(item.get('source_artifact_digest'))}` |",
        f"| Extraction method | `{md_escape_table(item.get('extraction_method'))}` |",
        f"| Confidence | `{md_escape_table(item.get('confidence'))}` |",
        f"| By-artifact record | `{rel(artifact_path) if artifact_path else 'missing'}` |",
        f"| Primary text | `{rel(primary_path) if isinstance(primary_path, Path) else 'missing'}` |",
        f"| Companion text | `{rel(companion_path) if isinstance(companion_path, Path) else 'missing'}` |",
        "",
    ]

    if artifact_payload:
        lines.extend(
            [
                "### Extraction Record",
                "",
                f"- Topic ids: `{json.dumps(artifact_payload.get('topic_ids', []), sort_keys=True)}`",
                f"- Input kind: `{quality.get('input_kind') if isinstance(quality, dict) else ''}`",
                f"- Match count: `{quality.get('match_count') if isinstance(quality, dict) else ''}`",
                f"- Text length: `{quality.get('text_length') if isinstance(quality, dict) else ''}`",
                f"- Topic scores: `{json.dumps(topic_scores, sort_keys=True) if isinstance(topic_scores, dict) else '{}'}`",
                f"- Evidence snippet: {json.dumps(evidence.get('snippet', '') if isinstance(evidence, dict) else '')}",
                f"- Evidence span: `{json.dumps(span, sort_keys=True) if isinstance(span, dict) else 'null'}`",
                "",
            ]
        )
    else:
        lines.extend(["### Extraction Record", "", "_No matching by-artifact record found._", ""])

    if span_excerpt:
        lines.extend(["### Span Dereference", "", fence_text(span_excerpt), ""])

    if primary_text:
        summary_label = f"Full primary text ({len(primary_text):,} chars)"
        lines.extend(
            [
                f"<details><summary>{summary_label}</summary>",
                "",
                fence_text(primary_text),
                "",
                "</details>",
                "",
            ]
        )
    else:
        lines.extend(["### Full primary text", "", "_Missing primary text._", ""])

    if companion_text:
        summary_label = f"Companion text ({len(companion_text):,} chars)"
        lines.extend(
            [
                f"<details><summary>{summary_label}</summary>",
                "",
                fence_text(companion_text),
                "",
                "</details>",
                "",
            ]
        )

    return "\n".join(lines), dict(stats)


def render_topic(workspace: Path, run_id: str, topic_id: str) -> str:
    extractions_root = workspace / "storage" / "extractions" / run_id
    by_topic_path = extractions_root / "by-topic" / f"{topic_id}.json"
    payload = read_json(by_topic_path)
    items = payload.get("items", [])
    artifact_index = build_by_artifact_index(extractions_root)

    method_counts = Counter(str(item.get("extraction_method") or "unknown") for item in items)
    source_kind_counts = Counter(
        "audio" if "/audio/" in str(item.get("source_artifact_path") or "") else "other"
        for item in items
    )
    total_stats = Counter()
    rendered_items: list[str] = []

    for index, item in enumerate(items, start=1):
        source_path = str(item.get("source_artifact_path") or "")
        rendered, stats = render_item(
            index=index,
            item=item,
            artifact_record=artifact_index.get(source_path),
            workspace=workspace,
        )
        rendered_items.append(rendered)
        total_stats.update(stats)

    lines: list[str] = [
        f"# Dereferenced Extraction Review: `{topic_id}`",
        "",
        f"Generated: {utc_now()}",
        f"Workspace: `{rel(workspace)}`",
        f"Run id: `{run_id}`",
        f"By-topic source: `{rel(by_topic_path)}`",
        "",
        "This file expands the by-topic extraction rollup into reviewable source text.",
        "For audio artifacts, the primary dereference is `transcript.txt`; where present, same-media poster/PDF OCR is included as companion `text.txt`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| By-topic items | {len(items)} |",
        f"| Declared artifact_count | {payload.get('artifact_count', '')} |",
        f"| By-artifact records found | {total_stats['by_artifact_found']} |",
        f"| By-artifact records missing | {total_stats['by_artifact_missing']} |",
        f"| Primary text files found | {total_stats['primary_text_found']} |",
        f"| Primary text files missing | {total_stats['primary_text_missing']} |",
        f"| Companion text files found | {total_stats['companion_text_found']} |",
        f"| Evidence spans resolved | {total_stats['span_resolved']} |",
        "",
        f"Extraction methods: `{json.dumps(dict(sorted(method_counts.items())), sort_keys=True)}`",
        "",
        f"Source kinds: `{json.dumps(dict(sorted(source_kind_counts.items())), sort_keys=True)}`",
        "",
        "## Items",
        "",
    ]
    lines.extend(rendered_items)
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "conferences" / "iclr2025")
    parser.add_argument("--run-id", default="v15-iclr2025-production")
    parser.add_argument("--topic-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = args.workspace.expanduser()
    if not workspace.is_absolute():
        workspace = PROJECT_ROOT / workspace
    workspace = workspace.resolve()
    output = args.output
    if output is None:
        output = (
            workspace
            / "storage"
            / "extractions"
            / args.run_id
            / "review"
            / f"{args.topic_id}-dereferenced.md"
        )
    elif not output.is_absolute():
        output = PROJECT_ROOT / output

    rendered = render_topic(workspace, args.run_id, args.topic_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(rel(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
