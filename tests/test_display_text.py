"""Text that reaches a slide has to read as text: the shared clean-up rules.

Every case here is a defect that shipped on the ICML 2026 canary decks, or the
neighbour of one that a narrower cleaner would have missed.
"""

from __future__ import annotations

from conference_pipeline.presentation.display_text import (
    clean_display_text,
    ends_dangling,
    latex_to_text,
    repair_leaked_macro_names,
    strip_event_kind_prefix,
    strip_markdown,
    trim_dangling_end,
    unwrap_text_macros,
)
from conference_pipeline.presentation.structured_video import SVG_FRAGMENT_END_WORDS


def test_a_text_macro_is_unwrapped_not_left_to_leak_its_own_name() -> None:
    # The converter used to keep an unknown macro's NAME and drop its braces.
    assert clean_display_text(r"$\texttt{FlashSchNet}$: Fast and Accurate") == "FlashSchNet: Fast and Accurate"
    assert clean_display_text(r"$\texttt{FHAIM}$: Fully Homomorphic AIM") == "FHAIM: Fully Homomorphic AIM"
    assert "texttt" not in clean_display_text(r"$\texttt{Multi}^2$: Hierarchical Multi-Agent")


def test_macros_nest_and_work_outside_math_too() -> None:
    assert unwrap_text_macros(r"\textbf{\texttt{Core}} idea") == "Core idea"
    assert clean_display_text(r"The \texttt{FlashSchNet} model") == "The FlashSchNet model"


def test_symbols_and_scripts_still_convert() -> None:
    assert latex_to_text(r"\alpha") == "α"
    assert clean_display_text(r"$\oslash$ Source Models") == "⊘ Source Models"
    assert latex_to_text(r"\mathrm{Part}^{2}") == "Part²"


def test_a_leaked_macro_name_in_an_already_built_ledger_is_repaired_at_display_time() -> None:
    assert repair_leaked_macro_names("textttFlashSchNet: Fast and Accurate") == "FlashSchNet: Fast and Accurate"
    assert repair_leaked_macro_names("Pingzhi Li: textttFlashSchNet") == "Pingzhi Li: FlashSchNet"
    assert repair_leaked_macro_names("textbfCore and textitHint") == "Core and Hint"


def test_ordinary_words_that_merely_contain_a_macro_name_are_left_alone() -> None:
    for word in ("context", "textbook", "text analysis", "pretext", "emphasis", "mathematics", "Textual", "texture"):
        assert repair_leaked_macro_names(word) == word


def test_a_markdown_link_keeps_its_text_and_loses_its_url() -> None:
    raw = "*ComplexMCP* — , [OpenReview](https://openreview.net/forum?id=3u1M9wokD7); *SafeSearch*"
    assert strip_markdown(raw) == "ComplexMCP, OpenReview; SafeSearch"
    assert strip_markdown("see [the paper](https://x.org/a_(b)) now") == "see the paper now"
    assert strip_markdown("![diagram](a.png) caption") == "diagram caption"


def test_markdown_emphasis_and_stray_heading_markers_go() -> None:
    assert strip_markdown("**Bold** and `code` and ~~gone~~") == "Bold and code and gone"
    assert strip_markdown("## Heading text") == "Heading text"


def test_an_event_kind_prefix_is_the_pills_own_tag_not_part_of_the_name() -> None:
    assert strip_event_kind_prefix("Poster: Walrus: A Cross-Domain Model") == "Walrus: A Cross-Domain Model"
    assert strip_event_kind_prefix("Oral: Something") == "Something"
    assert strip_event_kind_prefix("Posterior Sampling: a title") == "Posterior Sampling: a title"


def test_a_label_may_not_end_on_a_connector() -> None:
    words = SVG_FRAGMENT_END_WORDS
    assert ends_dangling("4th Structured Probabilistic Inference &", words)
    assert ends_dangling("Calibrated approval of", words)
    assert ends_dangling("Cost and", words)
    assert trim_dangling_end("4th Structured Probabilistic Inference &", fragment_words=words) == (
        "4th Structured Probabilistic Inference"
    )


def test_a_label_that_ends_on_content_is_never_flagged() -> None:
    words = SVG_FRAGMENT_END_WORDS
    for label in ("Variant A", "Learning When to Act or Refuse", "Think Twice Before You Act",
                  "Attention Is All You Need", "C++ Compiler", "FlashSchNet", "Q&A"):
        assert not ends_dangling(label, words), label


def test_a_lone_star_is_content_not_markdown() -> None:
    assert strip_markdown("A* search and Q* learning") == "A* search and Q* learning"
    assert strip_markdown("significant at p < 0.05*") == "significant at p < 0.05*"
    assert strip_markdown("*ComplexMCP* and A* search") == "ComplexMCP and A* search"
