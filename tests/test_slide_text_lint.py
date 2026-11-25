"""The deterministic slide-text lint: string properties found without a model.

A vision model caught these on the ICML canary decks for ~1.5 cents a slide; each is
a property of the text, so the lint finds it for nothing before any screenshot is taken.
"""

from __future__ import annotations

from pathlib import Path

from conference_pipeline.presentation import slide_text_lint as lint
from conference_pipeline.presentation.structured_video import SVG_FRAGMENT_END_WORDS

WORDS = SVG_FRAGMENT_END_WORDS


def _deck(sections: list[str]) -> str:
    return "<!doctype html><html><body>" + "".join(sections) + "</body></html>"


def _slide(title: str, chips: list[str] = (), body: str = "", figure: str = "") -> str:
    chip_html = "".join(f'<a class="source-chip" data-source-id="s{i}" href="https://x.org/{i}">{c}</a>' for i, c in enumerate(chips))
    return (
        f'<section data-slide-id="sl_x"><h2>{title}</h2><p>{body}</p>'
        f'<div class="reference-chips" data-qc-role="source-chips"><span>Sources</span>{chip_html}</div>{figure}</section>'
    )


def _check_ids(findings: list[dict[str, str]]) -> set[str]:
    return {f["check_id"] for f in findings}


def _lint_one(html: str, **kwargs):
    (slide,) = lint.extract_slides(html, **kwargs)
    return lint.lint_slide(slide, fragment_words=WORDS)


def test_a_leaked_text_macro_name_is_an_error() -> None:
    findings = _lint_one(_deck([_slide("Speed", chips=["Pingzhi Li: textttFlashSchNet"])]))
    assert [(f["check_id"], f["severity"]) for f in findings] == [("leaked_latex_macro", "error")]
    assert "textttFlashSchNet" in findings[0]["evidence"]


def test_raw_latex_markup_is_an_error() -> None:
    assert "leaked_latex_macro" in _check_ids(_lint_one(_deck([_slide("Title", body=r"We use \texttt{X} here")])))
    assert "leaked_latex_macro" in _check_ids(_lint_one(_deck([_slide("Title", body=r"cost $\alpha^2$ units")])))


def test_a_raw_markdown_link_is_an_error() -> None:
    findings = _lint_one(_deck([_slide("What this talk draws on", body="ComplexMCP — , [OpenReview](https://openreview.net/forum?id=1)")]))
    assert "raw_markdown_link" in _check_ids(findings)


def test_raw_markdown_markers_are_an_error() -> None:
    assert "raw_markdown_marker" in _check_ids(_lint_one(_deck([_slide("T", body="the **key** result")])))
    assert "raw_markdown_marker" in _check_ids(_lint_one(_deck([_slide("T", body="call `run()` first")])))


def test_a_source_label_cut_off_on_a_connector_is_an_error() -> None:
    findings = _lint_one(_deck([_slide("T", chips=["4th Structured Probabilistic Inference &"])]))
    assert [f["check_id"] for f in findings] == ["dangling_label_end"]


def test_template_placeholders_are_an_error() -> None:
    for text in ("{{ takeaway }}", "[object Object]", "{% if x %}", "Lorem ipsum dolor"):
        assert "template_placeholder" in _check_ids(_lint_one(_deck([_slide("T", body=text)]))), text
    # a template that rendered nothing: the whole block is a null word
    assert "template_placeholder" in _check_ids(_lint_one(_deck([_slide("T", body="undefined")])))


def test_prose_that_mentions_undefined_or_nan_is_not_a_placeholder() -> None:
    body = "Undefined behaviour is not claimed; NaN handling and null pointers are out of scope. TODO items are tracked."
    assert _lint_one(_deck([_slide("T", body=body)])) == []


def test_ordinary_slide_text_raises_nothing() -> None:
    clean = _slide(
        "Rollouts compound error",
        chips=["Michael McCabe: Walrus", "Variant A", "Learning When to Act or Refuse", "Q&A on Safety"],
        body="Cost is $5 and $10; none of the above; the C++ compiler; undefined behaviour is not claimed here.",
    )
    assert _lint_one(_deck([clean])) == []


def test_a_dual_chip_checks_its_label_not_its_kind_tag() -> None:
    chip = (
        '<span class="source-chip source-chip-dual" data-source-id="a"><a href="https://x/1">Michael McCabe: Walrus</a>'
        '<a class="source-chip-poster" href="https://x/2">· poster</a></span>'
    )
    (slide,) = lint.extract_slides(_deck([f'<section><h2>T</h2><div data-qc-role="source-chips">{chip}</div></section>']))
    assert slide.chip_labels == ["Michael McCabe: Walrus"]
    assert lint.lint_slide(slide, fragment_words=WORDS) == []


def test_a_figure_repeating_the_slide_title_is_a_warning_not_an_error(tmp_path: Path) -> None:
    svg = (tmp_path / "fig.svg")
    svg.write_text('<svg><text>A research loop: propose, test, interpret</text><text>LLM agent</text></svg>')
    html = _deck([_slide("A research loop: propose, test, interpret", figure='<figure><img src="fig.svg"></figure>')])
    findings = _lint_one(html, html_dir=tmp_path)
    assert [(f["check_id"], f["severity"]) for f in findings] == [("duplicate_title_in_figure", "warning")]


def test_a_distinct_figure_heading_is_fine(tmp_path: Path) -> None:
    (tmp_path / "fig.svg").write_text("<svg><text>Which MLIP design route to pick</text></svg>")
    html = _deck([_slide("Interatomic potentials: faster MD", figure='<figure><img src="fig.svg"></figure>')])
    assert _lint_one(html, html_dir=tmp_path) == []


def test_findings_are_keyed_by_section_and_errors_sort_first(tmp_path: Path) -> None:
    html = _deck([
        _slide("Fine slide"),
        _slide("Broken", chips=["Someone: textttFoo"], figure='<figure><img src="f.svg"></figure>'),
    ])
    (tmp_path / "f.svg").write_text("<svg><text>Broken</text></svg>")
    deck = tmp_path / "presentation.html"
    deck.write_text(html)
    by_section = lint.lint_deck_html(deck, fragment_words=WORDS)
    assert list(by_section) == [1]
    assert [f["severity"] for f in by_section[1]] == ["error", "warning"]
    assert lint.summarize(by_section)["errors"] == 1 and lint.summarize(by_section)["warnings"] == 1


def test_svg_text_lines_join_tspans_and_skip_empty_text() -> None:
    assert lint.svg_text_lines('<svg><text><tspan>A</tspan> <tspan>B</tspan></text><text> </text></svg>') == ["A B"]
