"""Order-independent full-read evidence selection.

Live regression (CVPR 2026, 2026-09-20): hundreds of transcript segments per
topic tied at the top confidence, the selection sorted by confidence only, and
the stable sort kept the alphabetical by-topic file order. Every topic's
evidence was an alphabetical slice, and one recording could fill most slots.
"""

from __future__ import annotations

import random
from typing import Any

from conference_pipeline.analysis.full_read_evidence_selection import (
    evidence_source_id,
    select_full_read_evidence,
)


def _segment(video: str, index: int, *, confidence: float = 0.95, talk: str = "") -> dict[str, Any]:
    return {
        "unit_id": f"event_media_caption:{video}:segment:{index:05d}",
        "input_kind": "transcript_segment",
        "confidence": confidence,
        "talk_title": talk,
    }


def _abstract(paper: str, *, confidence: float = 0.95) -> dict[str, Any]:
    return {"unit_id": f"paper_abstract:{paper}", "input_kind": "paper_abstract", "confidence": confidence}


def _ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(item["unit_id"]) for item in items]


def test_selection_does_not_depend_on_input_order() -> None:
    items = [_segment(f"video{n:03d}", 0) for n in range(200)] + [
        _abstract(f"{n:04x}") for n in range(200)
    ]
    expected = _ids(select_full_read_evidence(items, max_items_per_kind=10))

    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    assert _ids(select_full_read_evidence(shuffled, max_items_per_kind=10)) == expected
    assert _ids(select_full_read_evidence(sorted(items, key=lambda i: i["unit_id"]), max_items_per_kind=10)) == expected


def test_confidence_ties_are_not_an_alphabetical_slice() -> None:
    items = [_segment(f"video{n:03d}", 0) for n in range(200)]
    selected = _ids(select_full_read_evidence(items, max_items_per_kind=10))

    alphabetical = sorted(_ids(items))[:10]
    assert selected != alphabetical
    assert len(set(selected) & set(alphabetical)) < 5


def test_one_recording_cannot_crowd_out_the_others() -> None:
    items = [_segment("-aaaaaaaaaa", n) for n in range(40)] + [
        _segment(f"video{n:03d}", 0) for n in range(20)
    ]
    selected = select_full_read_evidence(items, max_items_per_kind=10)

    assert len({evidence_source_id(item) for item in selected}) == 10


def test_slots_are_still_filled_when_sources_are_few() -> None:
    items = [_segment("only-one", n) for n in range(40)]

    assert len(select_full_read_evidence(items, max_items_per_kind=10)) == 10


def test_session_talks_get_a_floor_among_equally_confident_units() -> None:
    posters = [_segment(f"poster{n:03d}", 0) for n in range(300)]
    talks = [_segment(f"lane-z:session{n:02d}", 0, talk=f"Keynote {n}") for n in range(8)]
    selected = select_full_read_evidence(posters + talks, max_items_per_kind=10)

    assert sum(1 for item in selected if item["talk_title"]) == 5
    assert sum(1 for item in selected if not item["talk_title"]) == 5


def test_floor_is_backfilled_when_there_are_no_session_talks() -> None:
    posters = [_segment(f"poster{n:03d}", 0) for n in range(30)]

    assert len(select_full_read_evidence(posters, max_items_per_kind=10)) == 10


def test_floor_never_lifts_a_less_confident_talk_over_a_more_confident_unit() -> None:
    posters = [_segment(f"poster{n:03d}", 0, confidence=0.95) for n in range(30)]
    weak_talks = [_segment(f"session{n:02d}", 0, confidence=0.4, talk="Aside") for n in range(8)]
    selected = select_full_read_evidence(posters + weak_talks, max_items_per_kind=10)

    assert all(not item["talk_title"] for item in selected)


def test_higher_confidence_still_wins() -> None:
    items = [_abstract(f"{n:04x}", confidence=0.6) for n in range(50)] + [
        _abstract("best", confidence=0.99)
    ]
    selected = select_full_read_evidence(items, max_items_per_kind=3)

    assert selected[0]["unit_id"] == "paper_abstract:best"


def test_unrecognised_kinds_keep_the_fallback_width() -> None:
    items = [
        {"unit_id": f"slide_ocr:{n}", "input_kind": "slide_ocr", "confidence": 0.9} for n in range(50)
    ]

    assert len(select_full_read_evidence(items, max_items_per_kind=10)) == 20


def test_importance_orders_a_confidence_tie_before_the_draw() -> None:
    papers = [_abstract(f"{n:04x}") for n in range(300)]
    orals = {"paper_abstract:0007", "paper_abstract:00c8", "paper_abstract:012b"}

    def importance(item: dict[str, Any]) -> tuple[float, ...]:
        return (1.0 if item["unit_id"] in orals else 0.0,)

    selected = _ids(select_full_read_evidence(papers, max_items_per_kind=10, importance=importance))

    assert orals <= set(selected[:3])


def test_importance_never_lifts_a_less_confident_unit() -> None:
    items = [_abstract(f"{n:04x}", confidence=0.95) for n in range(20)] + [
        _abstract("weak-oral", confidence=0.5)
    ]

    def importance(item: dict[str, Any]) -> tuple[float, ...]:
        return (1.0 if item["unit_id"] == "paper_abstract:weak-oral" else 0.0,)

    selected = _ids(select_full_read_evidence(items, max_items_per_kind=10, importance=importance))

    assert "paper_abstract:weak-oral" not in selected


def test_importance_is_still_order_independent() -> None:
    papers = [_abstract(f"{n:04x}") for n in range(200)]

    def importance(item: dict[str, Any]) -> tuple[float, ...]:
        return (float(int(item["unit_id"][-1], 16) % 3),)

    expected = _ids(select_full_read_evidence(papers, max_items_per_kind=10, importance=importance))
    shuffled = list(papers)
    random.Random(3).shuffle(shuffled)
    assert _ids(select_full_read_evidence(shuffled, max_items_per_kind=10, importance=importance)) == expected
