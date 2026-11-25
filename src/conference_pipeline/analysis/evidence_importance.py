"""Importance signals that order equally-confident full-read evidence.

The full-read model gives most relevant units the same top confidence (hundreds
of papers per topic tie at the maximum), so which of them fill a topic's few
evidence slots used to be decided by chance. These signals rank a tie by what
the conference itself and the paid matching step already said matters:

1. **Program prominence.** The program committee's presentation tier (oral,
   then spotlight/highlight), read from the virtual-site calendar snapshots
   that source discovery already fetched. No new download.
2. **Matched-paper rank.** The topic's paper-matching rank, when the paper was
   matched.
3. **Talk corroboration.** A session talk (keynote, tutorial, workshop, oral
   session) in the topic's own pool names the paper.

Every signal is optional: with none available the scorer returns zeros and the
selection falls back to the digest draw unchanged.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from conference_pipeline.analysis.full_read_evidence_selection import ImportanceScorer, is_session_talk

PROMINENCE_TIERS: dict[str, int] = {"oral": 2, "spotlight": 1, "highlight": 1}
_CONTENT_BLOCK_RE = re.compile(
    r'<div class="content (%s)">(.*?)</div>' % "|".join(PROMINENCE_TIERS), re.S
)
_TAG_RE = re.compile(r"<[^>]+>")
_SCHEDULE_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_HANDLE_RE = re.compile(r"^[A-Za-z0-9][\w\-\+\^\$\.]{1,38}$")


def normalize_title(title: str) -> str:
    return _NORMALIZE_RE.sub(" ", str(title or "").lower()).strip()


@dataclass(frozen=True)
class ProgramProminence:
    tiers: Mapping[str, int] = field(default_factory=dict)
    sources: tuple[dict[str, Any], ...] = ()

    def tier(self, title: str) -> int:
        return int(self.tiers.get(normalize_title(title), 0))

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for tier in self.tiers.values():
            counts[str(tier)] = counts.get(str(tier), 0) + 1
        return {"titles_by_tier": counts, "sources": list(self.sources)}


def parse_prominence_calendar(text: str) -> dict[str, int]:
    """Map normalized titles to their tier from a virtual-site calendar page."""

    tiers: dict[str, int] = {}
    for match in _CONTENT_BLOCK_RE.finditer(text):
        title = html.unescape(" ".join(_TAG_RE.sub(" ", match.group(2)).split()))
        key = normalize_title(_SCHEDULE_PREFIX_RE.sub("", title))
        if key:
            tiers[key] = max(tiers.get(key, 0), PROMINENCE_TIERS[match.group(1)])
    return tiers


def _snapshot_bodies(workspace: Path) -> Iterable[Path]:
    inventory_path = workspace / "storage" / "source-discovery" / "source_discovery_inventory.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    seen: set[str] = set()
    bodies: list[Path] = []
    for record in inventory.get("source_records") or []:
        evidence = record.get("source_evidence") if isinstance(record, Mapping) else None
        metadata_path = str((evidence or {}).get("metadata_path") or "")
        if not metadata_path or metadata_path in seen:
            continue
        seen.add(metadata_path)
        body = (workspace / metadata_path).parent / "body.bin"
        if body.is_file():
            bodies.append(body)
    return sorted(bodies)


def load_program_prominence(workspace: Path) -> ProgramProminence:
    """Read presentation tiers from the calendar snapshots source discovery fetched."""

    tiers: dict[str, int] = {}
    sources: list[dict[str, Any]] = []
    for body in _snapshot_bodies(workspace):
        raw = body.read_bytes()
        found = parse_prominence_calendar(raw.decode("utf-8", "replace"))
        if not found:
            continue
        for key, tier in found.items():
            tiers[key] = max(tiers.get(key, 0), tier)
        sources.append(
            {
                "path": str(body.relative_to(workspace)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "title_count": len(found),
            }
        )
    return ProgramProminence(tiers=tiers, sources=tuple(sources))


def paper_handle(title: str) -> str:
    """A paper's short name ("PALM" in "PALM: Progress-Aware ..."), if it has one."""

    head, colon, _ = str(title or "").partition(":")
    head = head.strip()
    if not colon or not _HANDLE_RE.match(head):
        return ""
    if not any(char.isupper() or char.isdigit() for char in head):
        return ""
    return head


def _talk_text(items: Iterable[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for item in items:
        if item.get("input_kind") != "transcript_segment" or not is_session_talk(item):
            continue
        parts.append(str(item.get("summary") or ""))
        parts.extend(str(phrase) for phrase in (item.get("evidence_phrases") or []))
    return "\n".join(parts)


def _paper_hex(unit_id: str) -> str:
    return str(unit_id or "").partition(":")[2]


def build_importance_scorer(
    items: Iterable[Mapping[str, Any]],
    *,
    prominence: ProgramProminence,
    priority_ranks: Mapping[str, int] | None = None,
) -> ImportanceScorer:
    """Score (program tier, matched rank, talk corroboration) for a topic's pool."""

    pool = list(items)
    ranks = {str(key).partition(":")[2]: int(rank) for key, rank in (priority_ranks or {}).items()}
    talk_text = _talk_text(pool)
    corroborated: dict[str, bool] = {}

    def talk_names(handle: str) -> bool:
        if handle not in corroborated:
            corroborated[handle] = bool(
                re.search(r"(?<![\w-])" + re.escape(handle) + r"(?![\w-])", talk_text)
            )
        return corroborated[handle]

    def score(item: Mapping[str, Any]) -> tuple[float, float, float]:
        if item.get("input_kind") != "paper_abstract":
            return (0.0, 0.0, 0.0)
        title = str(item.get("paper_title") or "")
        rank = ranks.get(_paper_hex(str(item.get("unit_id") or "")))
        handle = paper_handle(title)
        return (
            float(prominence.tier(title)),
            1.0 / rank if rank else 0.0,
            1.0 if handle and talk_names(handle) else 0.0,
        )

    return score
