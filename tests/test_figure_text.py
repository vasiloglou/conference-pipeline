"""Measured text: wrapping, shrinking, and when a word may be broken.

A shipped figure rendered "Counterfactual retriever" as "Count- / erfactu- /
al" and passed every gate, because nothing in the compiler looked at
hyphenation. Two things were wrong: the wrapper was handed an 83px column
while the box it would draw was 140px wide, and having decided to break a
word it was willing to leave a two-letter fragment on a line of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from conference_pipeline.figures import tokens  # noqa: E402
from conference_pipeline.figures.gate import internal_findings  # noqa: E402
from conference_pipeline.figures.layouts.topology import MIN_NODE_WIDTH  # noqa: E402
from conference_pipeline.figures.svgdoc import SvgDoc  # noqa: E402
from conference_pipeline.figures.text import (  # noqa: E402
    MIN_WORD_FRAGMENT,
    fit,
    measure,
    wrap,
    wrap_with_breaks,
)


def test_a_word_that_fits_at_the_floor_is_never_broken() -> None:
    """Shrink two points before hyphenating: the reader notices one of those."""

    block = fit("Counterfactual retriever", max_width=MIN_NODE_WIDTH - 2 * tokens.NODE_PAD_X,
                size=tokens.FONT_NODE, min_size=tokens.FONT_FLOOR, bold=True)
    assert block.breaks == ()
    assert "Counterfactual" in block.lines


def test_a_runaway_token_still_breaks_but_never_into_runts() -> None:
    lines = wrap("supercalifragilisticexpialidocious", 20.0, 100.0)
    assert len(lines) > 1
    for piece in lines:
        assert len(piece.rstrip("-")) >= MIN_WORD_FRAGMENT, lines


def test_splitting_at_an_authors_own_hyphen_is_not_a_break() -> None:
    """"AVA-Bench" wrapping to "AVA-" and "Bench" is ordinary wrapping."""

    _, inserted = wrap_with_breaks("AVA-Bench aggregate score", 26.0, 90.0, bold=True)
    assert inserted == []


def test_the_gate_rejects_a_word_broken_into_fragments() -> None:
    doc = SvgDoc()
    doc.word_breaks.append(("Count-", "erfactu-", "al"))
    codes = {finding.check_id for finding in internal_findings(doc)}
    assert "hyphen_fragment" in codes


def test_a_clean_document_reports_no_hyphen_finding() -> None:
    doc = SvgDoc()
    doc.word_breaks.append(("supercalif-", "ragilistice-", "xpialidocious"))
    codes = {finding.check_id for finding in internal_findings(doc)}
    assert "hyphen_fragment" not in codes


@pytest.mark.parametrize("size", [16.0, 20.0, 26.0, 34.0])
def test_measurement_is_the_metric_the_checker_uses(size: float) -> None:
    from conference_pipeline.presentation.svg_geometry import estimate_text_width

    assert measure("Vision encoder", size) == estimate_text_width("Vision encoder", size)
