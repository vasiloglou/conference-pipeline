#!/usr/bin/env python3
"""Reconcile ICML 2026 virtual-workshop media with the acquired corpus.

The ICML virtual-program event IDs are authoritative for workshop recordings.
This command fetches the public workshop listing, identifies its event cards,
then matches each non-cancelled event to an already-acquired SlidesLive audio
artifact in ``download_corpus.json``.  It does not download media or call a
provider.  A cancelled event is retained as explicit ``not_applicable``
coverage rather than treated as a missing recording.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_LISTING_URL = "https://icml.cc/virtual/2026/events/workshop"
WORKSHOP_MEDIA_INVENTORY = Path("storage/workshops/talk_media_inventory.json")
WORKSHOP_MEDIA_RECEIPT = Path("storage/workshops/talk_media_acquisition_receipt.json")
DOWNLOAD_CORPUS_INVENTORY = Path("storage/download-corpus/download_corpus.json")
RECONCILIATION_DIR = Path("storage/workshops/reconciliations")
WORKSHOP_AUDIO_PATH_RE = re.compile(
    r"^cache/acquisition-downloads/media_asset/workshop-(?P<event_id>\d+)/"
    r"slideslive_audio/raw\.mp3$"
)
REQUIRED_MEDIA_FIELDS = (
    "media_kind",
    "workshop_id",
    "workshop_title",
    "talk_id",
    "talk_title",
    "talk_speaker",
    "source_url",
    "media_url",
    "local_path",
    "sha256",
    "byte_count",
    "status",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class WorkshopEvent:
    event_id: str
    title: str
    detail_url: str


class WorkshopEventParser(HTMLParser):
    """Extract ICML virtual-program workshop cards without page-specific JS."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[WorkshopEvent] = []
        self._card: dict[str, str] | None = None
        self._card_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "div" and self._card is None:
            classes = attributes.get("class", "")
            if (
                "event-card" in classes
                and attributes.get("data-event-type", "").strip().lower() == "workshop"
            ):
                self._card = {
                    "event_id": attributes.get("data-event-id", "").strip(),
                    "detail_url": "",
                    "data_title": html.unescape(attributes.get("data-event-title", "").strip()),
                }
                self._card_depth = 1
                self._title_parts = []
                return
        if self._card is None:
            return
        if tag == "div":
            self._card_depth += 1
        if tag == "h3" and "event-title" in attributes.get("class", ""):
            self._in_title = True
        if tag == "a" and self._in_title and attributes.get("href"):
            self._card["detail_url"] = attributes["href"]

    def handle_data(self, data: str) -> None:
        if self._in_title and data.strip():
            self._title_parts.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if self._card is None:
            return
        if tag == "h3":
            self._in_title = False
        if tag != "div":
            return
        self._card_depth -= 1
        if self._card_depth:
            return
        event_id = self._card["event_id"]
        detail_url = self._card["detail_url"]
        title = html.unescape(" ".join(self._title_parts).strip()) or self._card["data_title"]
        if event_id and detail_url and title:
            self.events.append(WorkshopEvent(event_id=event_id, title=title, detail_url=detail_url))
        self._card = None
        self._title_parts = []


def parse_workshop_events(listing_html: str, *, listing_url: str) -> list[WorkshopEvent]:
    from urllib.parse import urljoin

    parser = WorkshopEventParser()
    parser.feed(listing_html)
    parser.close()
    events = [
        WorkshopEvent(
            event_id=event.event_id,
            title=event.title,
            detail_url=urljoin(listing_url, event.detail_url),
        )
        for event in parser.events
    ]
    seen: set[str] = set()
    unique: list[WorkshopEvent] = []
    for event in events:
        if event.event_id in seen:
            raise ValueError(f"duplicate ICML virtual workshop event id: {event.event_id}")
        seen.add(event.event_id)
        unique.append(event)
    if not unique:
        raise ValueError("ICML virtual workshop listing contained no workshop event cards")
    return unique


def acquired_audio_by_event(
    *, workspace: Path, download_corpus: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    for artifact in download_corpus.get("direct_download_artifacts", []):
        if not isinstance(artifact, dict) or artifact.get("status") != "succeeded":
            continue
        local_path = str(artifact.get("local_path") or "")
        match = WORKSHOP_AUDIO_PATH_RE.match(local_path)
        if not match:
            continue
        local_file = (workspace / local_path).resolve()
        try:
            local_file.relative_to(workspace.resolve())
        except ValueError as exc:
            raise ValueError(f"workshop audio path outside workspace: {local_path}") from exc
        if not local_file.is_file():
            raise ValueError(f"acquired workshop audio is missing: {local_path}")
        event_id = match.group("event_id")
        candidate = {
            "local_path": local_path,
            "sha256": str(artifact.get("sha256") or ""),
            "byte_count": int(artifact.get("byte_count") or local_file.stat().st_size),
            "media_url": str(artifact.get("source_url") or ""),
            "slideslive_id": str(artifact.get("slideslive_id") or ""),
            "content_type": str(artifact.get("content_type") or "audio/mpeg"),
        }
        if not candidate["sha256"] or not candidate["media_url"]:
            raise ValueError(f"acquired workshop audio lacks provenance: {local_path}")
        prior = resolved.get(event_id)
        if prior and prior != candidate:
            raise ValueError(f"multiple acquired audio artifacts for workshop {event_id}")
        resolved[event_id] = candidate
    return resolved


def build_reconciled_inventory(
    *,
    events: list[WorkshopEvent],
    audio_by_event: dict[str, dict[str, Any]],
    listing_url: str,
    listing_sha256: str,
    run_id: str,
    workspace: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: int(item.event_id)):
        event_is_cancelled = "cancelled" in event.title.lower()
        audio = audio_by_event.get(event.event_id)
        base = {
            "workshop_id": f"ICML.cc/2026/VirtualWorkshop/{event.event_id}",
            "workshop_title": event.title,
            "talk_id": f"virtual_workshop_{event.event_id}",
            "talk_title": event.title,
            "talk_speaker": None,
            "source_url": event.detail_url,
            "virtual_event_id": event.event_id,
        }
        if event_is_cancelled:
            records.append(
                {
                    **base,
                    "status": "cancelled",
                    "media_kind": "cancelled_event",
                    "media_url": None,
                    "local_path": None,
                    "sha256": None,
                    "byte_count": None,
                    "reason": "Official ICML virtual program marks this workshop as cancelled.",
                    "provider_calls_made": False,
                }
            )
            continue
        if audio is None:
            blocker = {
                **base,
                "code": "missing_acquired_workshop_audio",
                "reason": "A non-cancelled official workshop has no acquired local audio artifact.",
                "retryable": True,
            }
            blockers.append(blocker)
            records.append(
                {
                    **base,
                    "status": "blocked",
                    "media_kind": "blocked",
                    "media_url": None,
                    "local_path": None,
                    "sha256": None,
                    "byte_count": None,
                    "blocker_code": blocker["code"],
                    "reason": blocker["reason"],
                    "provider_calls_made": False,
                }
            )
            continue
        records.append(
            {
                **base,
                "status": "succeeded",
                "media_kind": "slideslive_audio",
                "media_url": audio["media_url"],
                "local_path": audio["local_path"],
                "sha256": audio["sha256"],
                "byte_count": audio["byte_count"],
                "content_type": audio["content_type"],
                "slideslive_id": audio["slideslive_id"],
                "retrieval_status": "reconciled_existing_acquisition",
                "provider_calls_made": False,
            }
        )
    succeeded_count = sum(record["status"] == "succeeded" for record in records)
    cancelled_count = sum(record["status"] == "cancelled" for record in records)
    return {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.icml2026_workshop_talk_media_inventory",
        "workspace": str(workspace),
        "run_id": run_id,
        "generated_at": utc_now(),
        "status": "succeeded" if not blockers else "blocked",
        "required_media_fields": list(REQUIRED_MEDIA_FIELDS),
        "workshop_count": len(events),
        "media_record_count": succeeded_count,
        "cancelled_event_count": cancelled_count,
        "blocker_count": len(blockers),
        "blocked_workshop_count": len(blockers),
        "source_listing_url": listing_url,
        "source_listing_sha256": listing_sha256,
        "records": records,
        "blockers": blockers,
        "boundaries": {
            "official_listing_reparsed": True,
            "audio_downloaded": False,
            "video_downloaded": False,
            "live_provider_calls_made": False,
            "secrets_copied": False,
        },
    }


def fetch_listing(url: str, *, timeout_seconds: float) -> str:
    request = Request(url, headers={"User-Agent": "conference-pipeline-icml2026-workshop-reconcile/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - user-authorized official URL.
        return response.read().decode("utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--listing-url", default=DEFAULT_LISTING_URL)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--write", action="store_true", help="Write the reconciled inventory and receipt.")
    parser.add_argument("--summary", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.resolve()
    corpus_path = workspace / DOWNLOAD_CORPUS_INVENTORY
    if not corpus_path.is_file():
        raise SystemExit(f"missing download corpus inventory: {corpus_path}")
    try:
        download_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid download corpus inventory: {exc}") from exc
    listing_html = fetch_listing(args.listing_url, timeout_seconds=max(1.0, args.timeout_seconds))
    events = parse_workshop_events(listing_html, listing_url=args.listing_url)
    listing_sha256 = hashlib.sha256(listing_html.encode("utf-8")).hexdigest()
    inventory = build_reconciled_inventory(
        events=events,
        audio_by_event=acquired_audio_by_event(workspace=workspace, download_corpus=download_corpus),
        listing_url=args.listing_url,
        listing_sha256=listing_sha256,
        run_id=args.run_id,
        workspace=workspace,
    )
    inventory_path = workspace / WORKSHOP_MEDIA_INVENTORY
    receipt_path = workspace / WORKSHOP_MEDIA_RECEIPT
    receipt = {
        "schema_version": 1,
        "receipt_type": "conference_pipeline.icml2026_workshop_talk_media_acquisition_receipt",
        "workspace": str(workspace),
        "run_id": args.run_id,
        "started_at": utc_now(),
        "completed_at": utc_now(),
        "status": inventory["status"],
        "inventory_path": str(WORKSHOP_MEDIA_INVENTORY),
        "inventory_sha256": None,
        "checks": {
            "official_workshop_event_count": inventory["workshop_count"],
            "media_record_count": inventory["media_record_count"],
            "cancelled_event_count": inventory["cancelled_event_count"],
            "blocker_count": inventory["blocker_count"],
            "source_listing_url": args.listing_url,
            "source_listing_sha256": listing_sha256,
            "existing_audio_reused": True,
        },
        "blockers": inventory["blockers"],
        "boundaries": inventory["boundaries"],
    }
    if args.write:
        if inventory_path.exists():
            backup = workspace / RECONCILIATION_DIR / f"{utc_now().replace(':', '').replace('-', '')}-talk_media_inventory.before_reconciliation.json"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inventory_path, backup)
        atomic_write_json(inventory_path, inventory)
        receipt["inventory_sha256"] = sha256_file(inventory_path)
        atomic_write_json(receipt_path, receipt)
    if args.summary:
        print(
            json.dumps(
                {
                    "status": inventory["status"],
                    "workshop_event_count": inventory["workshop_count"],
                    "media_record_count": inventory["media_record_count"],
                    "cancelled_event_count": inventory["cancelled_event_count"],
                    "blocker_count": inventory["blocker_count"],
                    "write": bool(args.write),
                    "inventory_path": str(inventory_path),
                    "receipt_path": str(receipt_path),
                },
                indent=2,
            )
        )
    return 0 if inventory["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
