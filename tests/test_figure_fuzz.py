"""Two contracts, held over every archetype rather than every worked example.

1. **A compiled figure has no geometry defect.** Not "the examples are clean" --
   that was already true while thirteen real specs refused and nine more shipped
   with text the ladder had silently removed. Either the case compiles clean or
   it raises, with findings that say why.

2. **Every label the spec asked for is accounted for.** It was drawn, or moved
   somewhere it fits, or the remedy ladder deliberately trimmed it and said so.
   Before this contract a layout could measure a label, fail to place it, and
   drop it with no record in the SVG, the manifest, or the receipt -- which is
   how a scatter plot shipped with three unlabelled points and a 3-D scene
   shipped with none of its three labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conference_pipeline.figures.compiler import compile_figure
from conference_pipeline.figures.errors import FigureCompilerError
from conference_pipeline.figures.manifest import account_for_labels
from conference_pipeline.figures.spec import ARCHETYPES, content_labels, load_spec
from conference_pipeline.figures.tokens import FONT_ABSOLUTE_FLOOR
from conference_pipeline.presentation.svg_geometry import geometry_findings

from figure_fuzz_corpus import CORPUS, MAX_CASES  # noqa: E402

# Blocking on the compiler route for the reason `gate.BLOCKING_WARNINGS` gives:
# the compiler measured the text and chose the box, so it has no excuse.
BLOCKING_WARNINGS = {"text_overflow", "label_on_border", "text_outside_canvas"}

# The renderer's own legibility gate refuses a slide with more than this many
# text elements, so a fallback that turns labels into a list must respect it.
MAX_TEXT_ELEMENTS = 36

CASE_IDS = [case.case_id for case in CORPUS]


def _blocking(svg: str) -> list[str]:
    return [
        f"{f.check_id}: {f.message}"
        for f in geometry_findings(svg)
        if f.severity == "error" or f.check_id in BLOCKING_WARNINGS
    ]


@pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
def test_a_fuzzed_spec_compiles_clean_or_refuses_loudly(case) -> None:
    try:
        figure = compile_figure(case.payload)
    except FigureCompilerError as error:
        assert str(error), f"{case.case_id} refused without saying why"
        return
    assert _blocking(figure.svg) == [], case.case_id
    assert figure.svg.count("<text") <= MAX_TEXT_ELEMENTS, (
        f"{case.case_id} draws more text elements than the renderer will accept"
    )


@pytest.mark.parametrize("case", CORPUS, ids=CASE_IDS)
def test_a_fuzzed_spec_accounts_for_every_label(case) -> None:
    """Drawn, relocated, or trimmed by a named ladder rung. Never just gone."""

    try:
        figure = compile_figure(case.payload)
    except FigureCompilerError:
        return  # a refusal is the other contract's business
    manifest = figure.manifest
    unexplained = [
        entry for entry in manifest.labels_dropped
        if not str(entry.get("reason", "")).startswith("ladder:")
    ]
    assert unexplained == [], (
        f"{case.case_id} compiled while dropping {unexplained}; a placement failure "
        "must drive the ladder, not vanish"
    )


def test_the_corpus_covers_every_archetype() -> None:
    covered = {case.archetype for case in CORPUS}
    assert covered == set(ARCHETYPES), {"uncovered": sorted(set(ARCHETYPES) - covered)}


def test_the_corpus_stays_bounded() -> None:
    """A corpus that grows without limit stops being run."""

    assert len(CORPUS) <= MAX_CASES, len(CORPUS)


def test_the_corpus_is_deterministic() -> None:
    from figure_fuzz_corpus import build_corpus

    assert [case.case_id for case in build_corpus()] == CASE_IDS


def test_the_corpus_exercises_the_awkward_labels() -> None:
    """Unicode, RTL, CJK and an unbreakable token all reach some layout."""

    seen = " ".join(
        str(value)
        for case in CORPUS
        for value in content_labels(case.payload["content"])
    )
    for label in ("視覚言語モデル", "الرؤية الحاسوبية", "supercalifragilisticexpialidocious"):
        assert label in seen, label


def test_a_ladder_trim_names_the_text_it_removed() -> None:
    """The manifest used to record WHICH OPTION was set, never which text went."""

    payload = next(
        case.payload for case in CORPUS
        if case.archetype == "flow" and "-s1-" in case.case_id
    )
    spec = load_spec(payload)
    from conference_pipeline.figures.compiler import _render_once
    from conference_pipeline.figures.layouts._common import LayoutOptions

    doc = _render_once(spec, LayoutOptions(drop_edge_labels=True))
    trimmed = [o for o in doc.label_outcomes if o.reason == "ladder:drop_edge_labels"]
    assert trimmed, "dropping edge labels recorded nothing"
    assert all(o.text for o in trimmed)


def test_every_drawn_label_is_at_or_above_the_legibility_floor() -> None:
    import re

    for case in CORPUS[::37]:
        try:
            figure = compile_figure(case.payload)
        except FigureCompilerError:
            continue
        sizes = [float(v) for v in re.findall(r'font-size="([\d.]+)"', figure.svg)]
        assert sizes and min(sizes) >= FONT_ABSOLUTE_FLOOR, case.case_id


def test_label_accounting_matches_a_label_that_the_layout_composed() -> None:
    """A label counts as drawn when it is CONTAINED in a rendered run.

    Layouts routinely render "label (unit)" or "label - 55%", and an equality
    test would report every one of those as missing.
    """

    from conference_pipeline.figures.svgdoc import LabelOutcome

    spec = load_spec({
        "spec_version": "figure-spec/1", "figure_id": "f", "slide_id": "s",
        "archetype": "flow", "claim": "c", "takeaway": "t",
        "grounding": {"kind": "schematic"},
        "content": {"nodes": [{"id": "n", "label": "Latency", "kind": "process"}],
                    "edges": [{"from": "n", "to": "n"}]},
    }, strict=False)
    drawn, _, _, ignored = account_for_labels(
        spec, [LabelOutcome(text="Latency (ms)", status="drawn")]
    )
    assert drawn == ["Latency"] and ignored == []
