"""Pluggable workshop-proceedings sources (conference-neutral).

A proceedings source is a public host that publishes a conference's archival
workshop papers independently of OpenReview (CVF Open Access for CVPR, ICCV
and WACV today; ECVA, ACL Anthology, PMLR and similar hosts can be added).
Each source owns a downloader that writes a ``workshop_paper_download_receipt``
and exposes that receipt to workshop coverage auditing through ``workshops``,
which normalizes the source's per-workshop entries.

Adding a source = one ``ProceedingsSource`` entry here, its downloader module
and script, a stage-owner registry entry, and tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProceedingsWorkshop:
    """One workshop as listed by a proceedings source."""

    source_id: str
    short_name: str
    title: str
    listing_url: str | None
    papers_listed: int
    papers_available: int


@dataclass(frozen=True)
class ProceedingsSource:
    source_id: str
    description: str
    receipt_relative: Path
    owner_script: str
    workshops: Callable[[Mapping[str, Any]], list[ProceedingsWorkshop]]


def _cvf_openaccess_workshops(receipt: Mapping[str, Any]) -> list[ProceedingsWorkshop]:
    rows: list[ProceedingsWorkshop] = []
    for entry in receipt.get("per_workshop", []) or []:
        if not isinstance(entry, Mapping) or not entry.get("cvf_short_name"):
            continue
        rows.append(
            ProceedingsWorkshop(
                source_id="cvf_openaccess",
                short_name=str(entry["cvf_short_name"]),
                title=str(entry.get("title") or ""),
                listing_url=entry.get("listing_url"),
                papers_listed=int(entry.get("paper_count_listed") or 0),
                papers_available=int(entry.get("available") or 0),
            )
        )
    return rows


SOURCES: dict[str, ProceedingsSource] = {
    "cvf_openaccess": ProceedingsSource(
        source_id="cvf_openaccess",
        description="CVF Open Access workshop proceedings (CVPR, ICCV, WACV)",
        receipt_relative=Path("storage/workshops/cvf_openaccess_download_receipt.json"),
        owner_script="scripts/download_cvf_openaccess_workshop_papers.py",
        workshops=_cvf_openaccess_workshops,
    ),
}


def load_proceedings_workshops(
    workspace: Path, receipts: Mapping[str, Mapping[str, Any]] | None = None
) -> tuple[list[ProceedingsWorkshop], list[str]]:
    """Normalized workshops from every proceedings source with a receipt on disk.

    ``receipts`` (source_id -> receipt) overrides disk reads, for tests.
    Returns ``(workshops, source_ids_loaded)``.
    """
    import json

    workshops: list[ProceedingsWorkshop] = []
    loaded: list[str] = []
    for source_id, source in SOURCES.items():
        receipt = (receipts or {}).get(source_id)
        if receipt is None and receipts is None:
            path = workspace / source.receipt_relative
            if path.is_file():
                try:
                    receipt = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    receipt = None
        if isinstance(receipt, Mapping):
            workshops.extend(source.workshops(receipt))
            loaded.append(source_id)
    return workshops, loaded
