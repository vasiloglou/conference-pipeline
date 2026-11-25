"""Importance signals for ordering equally-confident full-read evidence."""

from __future__ import annotations

import json
from pathlib import Path

from conference_pipeline.analysis.evidence_importance import (
    ProgramProminence,
    build_importance_scorer,
    load_program_prominence,
    paper_handle,
    parse_prominence_calendar,
)

CALENDAR = """
<div class="content oral"><a href="/virtual/x/oral/1">[9:15] PALM: Progress-Aware Policy Learning</a></div>
<div class="content poster"><a href="/virtual/x/poster/2">Some Poster: A Study</a></div>
<div class="content spotlight"><a href="/virtual/x/poster/3">Glow &amp; Tell: Lighting</a></div>
"""


def test_calendar_tiers_orals_above_spotlights_and_ignores_posters() -> None:
    tiers = parse_prominence_calendar(CALENDAR)

    assert tiers == {
        "palm progress aware policy learning": 2,
        "glow tell lighting": 1,
    }


def test_prominence_is_read_from_the_snapshots_source_discovery_fetched(tmp_path: Path) -> None:
    snapshot = tmp_path / "cache" / "source-discovery" / "schedule" / "snap-1"
    snapshot.mkdir(parents=True)
    (snapshot / "snapshot.json").write_text("{}", encoding="utf-8")
    (snapshot / "body.bin").write_text(CALENDAR, encoding="utf-8")
    inventory = tmp_path / "storage" / "source-discovery" / "source_discovery_inventory.json"
    inventory.parent.mkdir(parents=True)
    metadata_path = "cache/source-discovery/schedule/snap-1/snapshot.json"
    inventory.write_text(
        json.dumps({"source_records": [{"source_evidence": {"metadata_path": metadata_path}}] * 2}),
        encoding="utf-8",
    )

    prominence = load_program_prominence(tmp_path)

    assert prominence.tier("PALM: Progress-Aware Policy Learning") == 2
    assert prominence.tier("Some Poster: A Study") == 0
    assert [source["title_count"] for source in prominence.sources] == [2]


def test_missing_snapshots_leave_every_tier_at_zero(tmp_path: Path) -> None:
    assert load_program_prominence(tmp_path).tiers == {}


def test_paper_handle_only_takes_a_short_named_prefix() -> None:
    assert paper_handle("PALM: Progress-Aware Policy") == "PALM"
    assert paper_handle("ViT$^3$: Unlocking Test-Time Training") == "ViT$^3$"
    assert paper_handle("A Style is Worth One Code") == ""
    assert paper_handle("the model: lower-case prefixes are too generic") == ""


def test_scorer_ranks_tier_then_matched_rank_then_talk_corroboration() -> None:
    pool = [
        {"unit_id": "paper_abstract:aa", "input_kind": "paper_abstract", "paper_title": "PALM: Policy"},
        {"unit_id": "paper_abstract:bb", "input_kind": "paper_abstract", "paper_title": "Other: Thing"},
        {"unit_id": "paper_abstract:cc", "input_kind": "paper_abstract", "paper_title": "Quiet: Paper"},
        {
            "unit_id": "event_media_caption:v:segment:00001",
            "input_kind": "transcript_segment",
            "talk_title": "Keynote",
            "summary": "The speaker contrasts Other with earlier planners.",
        },
    ]
    scorer = build_importance_scorer(
        pool,
        prominence=ProgramProminence(tiers={"palm policy": 2}),
        priority_ranks={"paper:bb": 4},
    )

    assert scorer(pool[0]) == (2.0, 0.0, 0.0)
    assert scorer(pool[1]) == (0.0, 0.25, 1.0)
    assert scorer(pool[2]) == (0.0, 0.0, 0.0)
    assert scorer(pool[3]) == (0.0, 0.0, 0.0)


def test_a_poster_caption_does_not_count_as_talk_corroboration() -> None:
    pool = [
        {"unit_id": "paper_abstract:bb", "input_kind": "paper_abstract", "paper_title": "Other: Thing"},
        {
            "unit_id": "event_media_caption:v:segment:00001",
            "input_kind": "transcript_segment",
            "summary": "Other is presented by its authors.",
        },
    ]
    scorer = build_importance_scorer(pool, prominence=ProgramProminence())

    assert scorer(pool[0]) == (0.0, 0.0, 0.0)
