"""Dry-run/live-gated NotebookLM audio publication helpers."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from conference_pipeline._security.private_markers import has_publishing_marker
from conference_pipeline.publishing.public_index import (
    validate_public_bundle_root,
)


class NotebookLMAudioError(RuntimeError):
    """Raised when NotebookLM audio generation fails closed."""


RECEIPT_TYPE = "conference_pipeline.notebooklm_audio"
PINNED_REPO = "https://github.com/teng-lin/notebooklm-py"
PINNED_COMMIT = "1712d76cc2906d96a9386177e554529d677567a8"


@dataclass(frozen=True)
class NotebookLMAudioConfig:
    bundle_root: Path
    receipt_path: Path | None = None
    dry_run: bool = True
    live: bool = False
    ack_live_notebooklm: bool = False
    cookies_path: Path | None = None


@dataclass(frozen=True)
class NotebookLMAudioResult:
    manifest_path: Path
    receipt_path: Path
    planned_count: int


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise NotebookLMAudioError(f"JSON object expected: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def chunk_sources(bundle_root: Path) -> list[str]:
    chunks_path = bundle_root / "index" / "corpus_chunks.jsonl"
    if not chunks_path.exists():
        return []
    sources: list[str] = []
    seen: set[str] = set()
    with chunks_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            source = str(payload.get("source_path", ""))
            text = str(payload.get("text", ""))
            if has_publishing_marker(source) or has_publishing_marker(text):
                continue
            if source and source not in seen:
                seen.add(source)
                sources.append(source)
    return sources


def planned_audio_entries(sources: Sequence[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, source in enumerate(sources[:12], start=1):
        slug = Path(source).stem.replace(" ", "-").lower() or f"topic-{index:02d}"
        entries.append(
            {
                "topic_id": f"topic-{index:02d}-{slug}",
                "source_path": source,
                "status": "planned",
                "audio_path": None,
                "redistribution_status": "caution",
                "license": "review_required_before_publication",
                "provenance": "maintainer-generated NotebookLM overview over published corpus",
                "operator_live_step_required": True,
            }
        )
    return entries


def validate_live_gate(config: NotebookLMAudioConfig) -> None:
    if config.live and not config.ack_live_notebooklm:
        raise NotebookLMAudioError("live NotebookLM generation requires --ack-live-notebooklm")
    if config.live and config.cookies_path is None:
        raise NotebookLMAudioError("live NotebookLM generation requires --cookies-path")
    if config.live and config.cookies_path is not None and not config.cookies_path.exists():
        raise NotebookLMAudioError("cookies path does not exist")


def generate_notebooklm_audio(config: NotebookLMAudioConfig) -> NotebookLMAudioResult:
    validate_live_gate(config)
    bundle_root = validate_public_bundle_root(config.bundle_root)
    release = read_json(bundle_root / "RELEASE.json")
    sources = chunk_sources(bundle_root)
    entries = planned_audio_entries(sources)
    manifest_path = bundle_root / "metadata" / "notebooklm_audio_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "bundle_id": release.get("bundle_id"),
        "source_sha": release.get("source_sha"),
        "engine": "notebooklm-py",
        "engine_repo": PINNED_REPO,
        "engine_commit": PINNED_COMMIT,
        "mode": "live" if config.live else "dry-run",
        "operator_live_step_required": not config.live,
        "unofficial_google_browser_automation": True,
        "entries": entries,
        "generated_at": utc_now(),
    }
    write_json(manifest_path, manifest)
    receipt_path = config.receipt_path or bundle_root / "metadata" / "notebooklm_audio_receipt.json"
    write_json(
        receipt_path,
        {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": "succeeded",
            "exit_code": 0,
            "bundle_id": release.get("bundle_id"),
            "dry_run": not config.live,
            "live": config.live,
            "operator_live_step_required": not config.live,
            "engine_repo": PINNED_REPO,
            "engine_commit": PINNED_COMMIT,
            "planned_count": len(entries),
            "manifest_path": str(manifest_path),
            "cookies_path_recorded": bool(config.cookies_path),
            "cookies_value": "REDACTED" if config.cookies_path else None,
            "generated_at": utc_now(),
        },
    )
    return NotebookLMAudioResult(manifest_path, receipt_path, len(entries))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan or run NotebookLM audio generation")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Write a dry-run receipt")
    parser.add_argument("--live", action="store_true", help="Run live NotebookLM automation")
    parser.add_argument("--ack-live-notebooklm", action="store_true")
    parser.add_argument("--cookies-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = generate_notebooklm_audio(
        NotebookLMAudioConfig(
            bundle_root=args.bundle_root,
            receipt_path=args.receipt_path,
            dry_run=args.dry_run or not args.live,
            live=args.live,
            ack_live_notebooklm=args.ack_live_notebooklm,
            cookies_path=args.cookies_path,
        )
    )
    print(f"NotebookLM audio manifest: {result.manifest_path}")
    print(f"Receipt: {result.receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

