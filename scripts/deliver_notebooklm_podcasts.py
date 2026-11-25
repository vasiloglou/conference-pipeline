#!/usr/bin/env python3
"""Deliver per-deck NotebookLM podcasts to a curated, flat, shareable Drive folder.

The raw workspace mirror (``audio/**`` via the ``podcasts`` storage family)
still goes through ``scripts/sync_to_gdrive.py --profile backup``. This script
produces the *listener-facing* folder instead: one numbered ``.m4a`` per deck
(executive decks first), each deck's show notes, and an ``index.md``.

Rules:

* Refuses to run unless ``stage-10-notebooklm-podcast.json`` for ``--run-id``
  is ``succeeded`` (override with ``--no-require-stage-receipt``; the receipt
  records the waiver).
* Never deletes anything in the target folder; identical files (same sha256)
  are skipped, changed files are replaced atomically.
* Writes a sha-bound receipt under ``storage/workflow-runs/<run-id>/backup/``
  and exits non-zero unless every expected deck was delivered.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.conference_identity import conference_name  # noqa: E402
from conference_pipeline.podcast_notebooklm import (  # noqa: E402
    AUDIO_FILENAME,
    AUDIO_ROOT,
    DEFAULT_EXECUTIVE_DECK_IDS,
    MANIFEST_FILENAME,
    SHOW_NOTES_FILENAME,
    DeckRef,
    deck_scope,
    load_production_facts,
    output_dir,
    sha256_file,
    utc_now,
    write_json_atomic,
)

RECEIPT_TYPE = "conference_pipeline.notebooklm_podcast_delivery"
STAGE_RECEIPT_FILENAME = "stage-10-notebooklm-podcast.json"
DEFAULT_RECEIPT_NAME = "notebooklm-podcast-delivery-receipt.json"


def safe_title(text: str, *, limit: int = 110) -> str:
    text = re.sub(r"\s*:\s*", " - ", str(text))
    text = re.sub(r"[\\/*?\"<>|]+", "-", text)
    text = re.sub(r"\s+", " ", text).strip(" .-")
    return text[:limit].rstrip(" .-") or "untitled"


def episode_prefix(deck: DeckRef, executive_deck_ids: Sequence[str]) -> str:
    if deck.kind == "executive":
        try:
            letter = "abcdefghij"[list(executive_deck_ids).index(deck.id)]
        except (ValueError, IndexError):
            letter = "x"
        return f"00{letter}"
    return f"{int(deck.number or 0):02d}"


def episode_basename(deck: DeckRef, executive_deck_ids: Sequence[str]) -> str:
    title = deck.title
    if deck.kind == "executive":
        suffix = "long" if deck.id.endswith("long") else "short" if deck.id.endswith("short") else deck.id
        title = f"{title} ({suffix})"
    return f"{episode_prefix(deck, executive_deck_ids)} - {safe_title(title)}"


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _copy_if_changed(source: Path, destination: Path, *, dry_run: bool) -> tuple[str, str]:
    source_sha = sha256_file(source)
    if destination.is_file() and sha256_file(destination) == source_sha:
        return "skipped_identical", source_sha
    if dry_run:
        return "would_copy", source_sha
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, tmp)
    os.replace(tmp, destination)
    return "copied", source_sha


def build_index(
    conference_label: str,
    items: Sequence[dict[str, Any]],
    *,
    generated_at: str,
    creator: str | None = None,
    how_made: str | None = None,
) -> str:
    lines = [f"# {conference_label} podcasts", ""]
    if creator:
        lines += [f"Created by {creator}.", ""]
    if how_made:
        lines += ["## How these episodes were made", "", how_made, ""]
    lines += [f"Audio generated with Google NotebookLM Audio Overviews; delivered {generated_at}.", ""]
    lines += ["| # | Episode | Sources | Prompt | Audio file | sha256 |", "|---|---------|---------|--------|------------|--------|"]
    for item in items:
        lines.append(
            f"| {item['prefix']} | {item['title']} | {item.get('source_count') or ''} | {item.get('prompt_variant') or ''} | "
            f"`{item['audio_destination_name']}` | `{item.get('audio_sha256') or ''}` |"
        )
    lines += ["", "Each episode has a matching `- show notes.md` file listing the works discussed and the steering prompt.", ""]
    return "\n".join(lines)


def deliver(
    *,
    workspace: Path,
    run_id: str,
    drive_root: Path,
    dry_run: bool,
    receipt_path: Path | None,
    require_stage_receipt: bool,
    audio_root: str = AUDIO_ROOT,
    executive_deck_ids: Sequence[str] = DEFAULT_EXECUTIVE_DECK_IDS,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    conference_label = conference_name(workspace)
    run_dir = workspace / "storage" / "workflow-runs" / run_id
    receipt_path = receipt_path or run_dir / "backup" / DEFAULT_RECEIPT_NAME
    generated_at = utc_now()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": RECEIPT_TYPE,
        "status": "blocked",
        "dry_run": dry_run,
        "workspace": str(workspace),
        "run_id": run_id,
        "target": str(drive_root),
        "stage_receipt_path": str(run_dir / STAGE_RECEIPT_FILENAME),
        "stage_receipt_sha256": None,
        "stage_receipt_waived": not require_stage_receipt,
        "items": [],
        "index_path": None,
        "index_sha256": None,
        "expected_count": 0,
        "delivered_count": 0,
        "blockers": [],
        "secrets_copied": False,
        "generated_at": generated_at,
    }

    stage_receipt_path = run_dir / STAGE_RECEIPT_FILENAME
    stage_receipt = _load_json(stage_receipt_path)
    if stage_receipt is not None:
        receipt["stage_receipt_sha256"] = sha256_file(stage_receipt_path)
    if require_stage_receipt and (stage_receipt is None or stage_receipt.get("status") != "succeeded"):
        receipt["blockers"].append(
            {
                "code": "notebooklm_delivery_stage_receipt_not_succeeded",
                "detail": f"{stage_receipt_path.name} missing or not succeeded; run the notebooklm-podcast stage to completion first",
            }
        )
        _finish(receipt, receipt_path, dry_run)
        return receipt

    if not drive_root.parent.is_dir():
        receipt["blockers"].append({"code": "notebooklm_delivery_drive_root_parent_missing", "detail": str(drive_root.parent)})
        _finish(receipt, receipt_path, dry_run)
        return receipt
    if not dry_run:
        drive_root.mkdir(parents=True, exist_ok=True)

    decks = deck_scope(workspace, executive_deck_ids=executive_deck_ids)
    decks.sort(key=lambda d: episode_prefix(d, executive_deck_ids))
    receipt["expected_count"] = len(decks)
    delivered = 0
    for deck in decks:
        out_dir = output_dir(workspace, deck.id, audio_root=audio_root)
        audio = out_dir / AUDIO_FILENAME
        manifest = _load_json(out_dir / MANIFEST_FILENAME) or {}
        base = episode_basename(deck, executive_deck_ids)
        item: dict[str, Any] = {
            "deck_id": deck.id,
            "prefix": episode_prefix(deck, executive_deck_ids),
            "title": deck.title,
            "audio_source_path": str(audio.relative_to(workspace)),
            "audio_destination_name": f"{base}.m4a",
            "show_notes_destination_name": f"{base} - show notes.md",
            "source_count": len(manifest.get("sources") or []) or None,
            "prompt_variant": (manifest.get("prompt") or {}).get("variant_used"),
            "manifest_generation_status": (manifest.get("generation") or {}).get("status"),
            "audio_sha256": None,
            "audio_byte_count": None,
            "audio_action": None,
            "show_notes_action": None,
        }
        if not audio.is_file() or audio.stat().st_size <= 0:
            item["audio_action"] = "missing"
            receipt["blockers"].append({"code": "notebooklm_delivery_audio_missing", "deck_id": deck.id})
            receipt["items"].append(item)
            continue
        manifest_sha = (manifest.get("audio") or {}).get("sha256")
        actual_sha = sha256_file(audio)
        if manifest_sha and manifest_sha != actual_sha:
            item["audio_action"] = "sha_mismatch"
            receipt["blockers"].append({"code": "notebooklm_delivery_audio_sha_mismatch", "deck_id": deck.id})
            receipt["items"].append(item)
            continue
        action, sha = _copy_if_changed(audio, drive_root / item["audio_destination_name"], dry_run=dry_run)
        item["audio_action"], item["audio_sha256"], item["audio_byte_count"] = action, sha, audio.stat().st_size
        notes = out_dir / SHOW_NOTES_FILENAME
        if notes.is_file():
            item["show_notes_action"], _ = _copy_if_changed(notes, drive_root / item["show_notes_destination_name"], dry_run=dry_run)
        delivered += 1
        receipt["items"].append(item)

    receipt["delivered_count"] = delivered
    facts = load_production_facts(workspace, conference_label=conference_label)
    index_text = build_index(
        conference_label,
        [i for i in receipt["items"] if i.get("audio_sha256")],
        generated_at=generated_at,
        creator=facts.creator,
        how_made=facts.how_made(),
    )
    index_path = drive_root / "index.md"
    receipt["index_path"] = str(index_path)
    receipt["index_sha256"] = __import__("hashlib").sha256(index_text.encode("utf-8")).hexdigest()
    if not dry_run and not receipt["blockers"]:
        tmp = index_path.with_name(f".index.md.{os.getpid()}.tmp")
        tmp.write_text(index_text, encoding="utf-8")
        os.replace(tmp, index_path)
    if delivered != len(decks):
        receipt["blockers"].append({"code": "notebooklm_delivery_incomplete", "expected": len(decks), "delivered": delivered})
    _finish(receipt, receipt_path, dry_run)
    return receipt


def _finish(receipt: dict[str, Any], receipt_path: Path, dry_run: bool) -> None:
    receipt["status"] = "blocked" if receipt["blockers"] else ("planned" if dry_run else "succeeded")
    write_json_atomic(receipt_path, receipt)
    receipt["receipt_path"] = str(receipt_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deliver per-deck NotebookLM podcasts to a curated Drive folder.")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--drive-root", type=Path, required=True, help="Target folder on the mounted Drive, e.g. '<mount>/CVPR 2026 Podcasts'.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt-path", type=Path, default=None)
    parser.add_argument("--no-require-stage-receipt", action="store_true", help="Deliver even without a succeeded stage receipt (recorded as a waiver).")
    parser.add_argument("--audio-root", default=AUDIO_ROOT)
    parser.add_argument("--summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = deliver(
        workspace=args.workspace,
        run_id=args.run_id,
        drive_root=args.drive_root,
        dry_run=args.dry_run,
        receipt_path=args.receipt_path,
        require_stage_receipt=not args.no_require_stage_receipt,
        audio_root=args.audio_root,
    )
    if args.summary:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(f"{receipt['status']}: {receipt['delivered_count']}/{receipt['expected_count']} delivered -> {receipt['target']} (receipt {receipt['receipt_path']})")
    return 0 if receipt["status"] in ("succeeded", "planned") else 2


if __name__ == "__main__":
    raise SystemExit(main())
