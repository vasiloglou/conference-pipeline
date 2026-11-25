"""Shared, order-independent selection of full-read evidence for a topic.

Every conference runner (CVPR2026, ICLR2026, ICML2026) carried its own copy of
the same selection: sort a topic's full-read items by ``confidence`` and keep
the first ``max_items_per_kind`` of each input kind. The full-read model emits
the same top confidence for most relevant units -- on the live CVPR2026 run,
27 to 738 transcript segments per topic tied at the maximum -- so the stable
sort fell back to the by-topic file order, which is alphabetical by unit id.
The "top" evidence was an alphabetical slice: every selected video id began
with ``-``, ``0``, ``1`` or ``2``, one recording could fill eight of ten slots,
and lane-prefixed unit ids (which sort last) could never be selected. Adding
transcripts to the corpus therefore did not change what the synthesis saw.

This is the same defect class as Finding D in
:mod:`conference_pipeline.analysis.source_pack_selection` (selection decided by
physical order instead of relevance) and is fixed the same way:

* **Order independence.** Ties on confidence are broken by a digest of the unit
  id, a uniform draw over equally-confident units, never by input order.
* **One source, one slot first.** Slots are dealt round-robin across distinct
  sources, so a single recording or paper cannot crowd out the rest; a source
  gets a second unit only once every other source has had its first.
* **Session-talk floor with backfill, never a hard reservation.** Transcript
  units that carry a talk identity come from talk-annotated session recordings
  (keynotes, tutorials, workshops, oral sessions). At least
  ``session_talk_floor`` of the transcript slots go to them when they exist;
  unused floor slots are backfilled from the remaining units.
* **Importance before chance.** An optional ``importance`` scorer orders
  equally-confident units (e.g. program orals, matched-paper rank) before the
  digest draw. It only ever reorders a confidence tie; it never lifts a
  less-confident unit over a more-confident one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

EVIDENCE_KINDS: tuple[str, ...] = ("transcript_segment", "paper_abstract")
SESSION_TALK_FLOOR_FRACTION = 0.5
_SEGMENT_MARKER = ":segment:"

ImportanceScorer = Callable[[Mapping[str, Any]], Sequence[float]]


def evidence_source_id(item: Mapping[str, Any]) -> str:
    """The recording or paper a unit was cut from."""

    unit_id = str(item.get("unit_id") or "")
    head, marker, _ = unit_id.rpartition(_SEGMENT_MARKER)
    return head if marker else unit_id


def is_session_talk(item: Mapping[str, Any]) -> bool:
    """Whether a transcript unit carries a talk identity from a session recording."""

    return any(str(item.get(key) or "").strip() for key in ("talk_id", "talk_title"))


def _confidence(item: Mapping[str, Any]) -> float:
    try:
        return float(item.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _tiebreak(item: Mapping[str, Any]) -> str:
    return hashlib.sha256(str(item.get("unit_id") or "").encode("utf-8")).hexdigest()


def _rank(
    items: Iterable[Mapping[str, Any]],
    importance: ImportanceScorer | None = None,
) -> list[Mapping[str, Any]]:
    def key(item: Mapping[str, Any]) -> tuple[Any, ...]:
        weights = tuple(-float(value) for value in importance(item)) if importance else ()
        return (-_confidence(item), weights, _tiebreak(item))

    return sorted(items, key=key)


def _deal_across_sources(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    taken_per_source: dict[str, int],
    importance: ImportanceScorer | None = None,
) -> list[Mapping[str, Any]]:
    """Deal ranked items round-robin across sources, best unit of each source first.

    A source that already holds ``n`` units joins at round ``n``, so it gets
    another only after every other source has had as many.
    """

    if limit <= 0:
        return []
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for item in _rank(items, importance):
        by_source.setdefault(evidence_source_id(item), []).append(item)
    already = dict(taken_per_source)
    last_round = max(
        (already.get(source, 0) + len(units) for source, units in by_source.items()), default=0
    )
    dealt: list[Mapping[str, Any]] = []
    for round_index in range(last_round):
        candidates = [
            units[round_index - already.get(source, 0)]
            for source, units in by_source.items()
            if 0 <= round_index - already.get(source, 0) < len(units)
        ]
        dealt += _rank(candidates, importance)[: limit - len(dealt)]
        if len(dealt) >= limit:
            break
    for item in dealt:
        source = evidence_source_id(item)
        taken_per_source[source] = taken_per_source.get(source, 0) + 1
    return dealt


def _select_kind(
    items: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    session_talk_floor: int,
    importance: ImportanceScorer | None = None,
) -> list[Mapping[str, Any]]:
    taken_per_source: dict[str, int] = {}
    ranked = _rank(items, importance)
    # The floor reorders within the confidence band that makes the cut; it never
    # lifts a less-confident unit over a more-confident one.
    cutoff = _confidence(ranked[min(limit, len(ranked)) - 1]) if ranked and limit > 0 else 0.0
    selected = _deal_across_sources(
        [item for item in items if is_session_talk(item) and _confidence(item) >= cutoff],
        limit=min(limit, session_talk_floor),
        taken_per_source=taken_per_source,
        importance=importance,
    )
    chosen = {str(item.get("unit_id") or "") for item in selected}
    selected += _deal_across_sources(
        [item for item in items if str(item.get("unit_id") or "") not in chosen],
        limit=limit - len(selected),
        taken_per_source=taken_per_source,
        importance=importance,
    )
    return _rank(selected, importance)


def select_full_read_evidence(
    items: Iterable[Mapping[str, Any]],
    *,
    max_items_per_kind: int,
    importance: ImportanceScorer | None = None,
) -> list[dict[str, Any]]:
    """Select a topic's full-read evidence; the result does not depend on input order.

    ``importance`` returns a tuple of weights (higher first) that orders units
    within a confidence tie before the digest draw.
    """

    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        by_kind.setdefault(str(item.get("input_kind") or "unknown"), []).append(item)
    selected: list[Mapping[str, Any]] = []
    for kind in EVIDENCE_KINDS:
        floor = (
            int(max_items_per_kind * SESSION_TALK_FLOOR_FRACTION)
            if kind == "transcript_segment"
            else 0
        )
        selected += _select_kind(
            by_kind.get(kind, []),
            limit=max_items_per_kind,
            session_talk_floor=floor,
            importance=importance,
        )
    if not selected:
        # No recognised kind: keep the historical fallback width.
        everything = [item for kind_items in by_kind.values() for item in kind_items]
        selected = _select_kind(
            everything,
            limit=max_items_per_kind * 2,
            session_talk_floor=0,
            importance=importance,
        )
    return [dict(item) for item in selected]
