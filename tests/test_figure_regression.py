"""The vision judge as a regression signal, not a retry trigger (spec §7.7).

On the drawsvg route the pixel judge drove a redraw loop that never converged:
18 failing slides became 16, then 20 became 12, across two paid rounds. A
compiled figure cannot be improved that way -- the spec determines the pixels,
so re-running it reproduces them exactly -- which changes what the judge is
*for*. Its findings become a per-archetype health record: if `swimlane_sequence`
keeps drawing findings across decks, the fix belongs in the layout code.

These tests hold that contract: the join is correct, deck-contract defects are
not blamed on the layout, sampling is deterministic and spreads across
archetypes, and nothing here ever asks for a retry.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from conference_pipeline.figures.regression import (  # noqa: E402
    archetype_regression,
    load_manifests,
    sample_figures,
)


def _manifests(pairs: list[tuple[str, str]]) -> list[dict[str, object]]:
    return [
        {"figure_id": f"fig-{index}", "slide_id": slide_id, "archetype": archetype}
        for index, (slide_id, archetype) in enumerate(pairs, start=1)
    ]


def _render_spec(manifests: list[dict[str, object]]) -> dict[str, object]:
    return {"slides": [{"slide_id": manifest["slide_id"]} for manifest in manifests]}


def _qc(entries: list[tuple[int, list[tuple[str, str, str]]]]) -> dict[str, object]:
    return {
        "slides": [
            {
                "slide_number": number,
                "status": "fail" if issues else "pass",
                "issues": [
                    {"check_id": check, "severity": severity, "description": description}
                    for check, severity, description in issues
                ],
            }
            for number, issues in entries
        ]
    }


def test_findings_are_recorded_against_the_archetype_that_drew_them() -> None:
    manifests = _manifests([("sl_01", "flow"), ("sl_02", "flow"), ("sl_03", "bar_compare")])
    report = archetype_regression(
        visual_qc_report=_qc([
            (1, [("text_overlap", "blocker", "two labels overlap")]),
            (2, []),
            (3, [("axis_crowding", "warn", "ticks are crowded")]),
        ]),
        render_spec=_render_spec(manifests),
        manifests=manifests,
    )
    by_name = {row["archetype"]: row for row in report["archetypes"]}
    assert by_name["flow"]["figures_reviewed"] == 2
    assert by_name["flow"]["figures_with_findings"] == 1
    assert by_name["flow"]["finding_counts"] == {"text_overlap": 1}
    assert by_name["bar_compare"]["finding_counts"] == {"axis_crowding": 1}
    assert report["finding_total"] == 2


def test_the_judge_never_drives_a_retry() -> None:
    """The whole point of the record: it feeds a backlog, not a redraw loop."""

    manifests = _manifests([("sl_01", "flow")])
    report = archetype_regression(
        visual_qc_report=_qc([(1, [("text_overlap", "blocker", "overlap")])]),
        render_spec=_render_spec(manifests),
        manifests=manifests,
    )
    assert report["drives_retry"] is False


def test_a_slide_title_defect_is_not_blamed_on_the_layout() -> None:
    """A clipped slide title is a deck-contract problem, not a figure problem."""

    manifests = _manifests([("sl_01", "flow")])
    report = archetype_regression(
        visual_qc_report=_qc([(1, [("title", "blocker", "the slide title is clipped")])]),
        render_spec=_render_spec(manifests),
        manifests=manifests,
    )
    assert report["finding_total"] == 0
    assert report["archetypes"][0]["figures_with_findings"] == 0


def test_informational_findings_are_not_counted() -> None:
    manifests = _manifests([("sl_01", "flow")])
    report = archetype_regression(
        visual_qc_report=_qc([(1, [("note", "info", "a stylistic remark")])]),
        render_spec=_render_spec(manifests),
        manifests=manifests,
    )
    assert report["finding_total"] == 0


def test_a_failing_slide_with_no_compiled_figure_is_reported_not_dropped() -> None:
    """A bookend or a template slide has no figure; say so rather than lose it."""

    manifests = _manifests([("sl_01", "flow")])
    render_spec = {"slides": [{"slide_id": "sl_01"}, {"slide_id": "sl_99_bookend"}]}
    report = archetype_regression(
        visual_qc_report=_qc([(2, [("contrast", "blocker", "low contrast")])]),
        render_spec=render_spec,
        manifests=manifests,
    )
    assert report["unmatched_failing_slides"] == [2]
    assert report["finding_total"] == 0


def test_a_sample_spreads_across_archetypes_before_repeating_one() -> None:
    """Twelve looks at twelve archetypes says more than twelve looks at `flow`."""

    manifests = _manifests(
        [(f"sl_{i:02d}", "flow") for i in range(1, 9)]
        + [("sl_20", "bar_compare"), ("sl_21", "schematic_curve")]
    )
    chosen = sample_figures(manifests, size=3, seed="deck01")
    assert len(chosen) == 3
    assert {str(item["archetype"]) for item in chosen} == {"flow", "bar_compare", "schematic_curve"}


def test_sampling_is_deterministic() -> None:
    manifests = _manifests([(f"sl_{i:02d}", "flow") for i in range(1, 13)])
    first = [item["figure_id"] for item in sample_figures(manifests, size=4, seed="deck01")]
    second = [item["figure_id"] for item in sample_figures(manifests, size=4, seed="deck01")]
    assert first == second
    other = [item["figure_id"] for item in sample_figures(manifests, size=4, seed="deck02")]
    assert other != first, "a different deck should not always sample the same figures"


def test_an_unsampled_run_reviews_everything() -> None:
    manifests = _manifests([(f"sl_{i:02d}", "flow") for i in range(1, 6)])
    report = archetype_regression(
        visual_qc_report=_qc([]), render_spec=_render_spec(manifests), manifests=manifests
    )
    assert report["figures_reviewed"] == 5
    assert report["sampled"] is False


def test_a_sampled_run_says_that_it_sampled() -> None:
    manifests = _manifests([(f"sl_{i:02d}", "flow") for i in range(1, 11)])
    report = archetype_regression(
        visual_qc_report=_qc([]),
        render_spec=_render_spec(manifests),
        manifests=manifests,
        sample_size=4,
    )
    assert report["figures_reviewed"] == 4
    assert report["figures_available"] == 10
    assert report["sampled"] is True


def test_manifests_load_from_a_compiled_deck(tmp_path: Path) -> None:
    directory = tmp_path / "assets" / "compiled"
    directory.mkdir(parents=True)
    (directory / "a.manifest.json").write_text(
        json.dumps({"figure_id": "a", "slide_id": "sl_01", "archetype": "flow"}), encoding="utf-8"
    )
    (directory / "broken.manifest.json").write_text("{not json", encoding="utf-8")
    loaded = load_manifests(directory)
    assert [item["figure_id"] for item in loaded] == ["a"], "an unreadable manifest is skipped, not fatal"
    assert load_manifests(tmp_path / "missing") == []
