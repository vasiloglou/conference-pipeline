"""The exact defects that shipped on the ICML 2026 canary decks, fixed where they are made.

The lint finds them after the fact; these tests pin that the generation path no longer
produces them: the ledger's title cleaner, the chip labeller and the bookend slides.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from conference_pipeline.presentation import source_inventory as si
from conference_pipeline.presentation import structured_video as sv

ROOT = Path(__file__).resolve().parents[1]


def _bookends():
    path = ROOT / "scripts" / "presentations" / "add_bookend_slides.py"
    spec = importlib.util.spec_from_file_location("bookends_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_ledger_no_longer_turns_a_text_macro_into_a_leaked_word() -> None:
    # The raw OpenReview title; the ledger used to emit "textttFlashSchNet: Fast ...".
    cleaned = si._clean_display_text(r"$\texttt{FlashSchNet}$: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics")
    assert cleaned.startswith("FlashSchNet: Fast and Accurate")
    assert "texttt" not in cleaned


def test_a_chip_for_a_work_whose_ledger_title_already_leaked_reads_cleanly() -> None:
    label = sv._source_chip_label({
        "title": "textttFlashSchNet: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics",
        "works": ["textttFlashSchNet: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics"],
        "authors": ["Pingzhi Li"],
    })
    assert label == "Pingzhi Li: FlashSchNet"


def test_inline_math_in_a_title_keeps_its_name_without_the_macro() -> None:
    assert sv.strip_inline_math(r"$\texttt{Multi}^2$: Hierarchical Multi-Agent") == "Multi2: Hierarchical Multi-Agent"
    assert sv.strip_inline_math("Part$^{2}$GS") == "Part2GS"  # unchanged behaviour


def test_a_source_label_is_never_cut_on_an_ampersand() -> None:
    label = sv._compact_label("4th Structured Probabilistic Inference & Generative Modeling", limit=46)
    assert label == "4th Structured Probabilistic Inference"
    assert not label.endswith("&")


def test_the_event_kind_is_the_pills_tag_not_part_of_the_chip_name() -> None:
    label = sv._source_chip_label({
        "title": "Poster: Walrus: A Cross-Domain Foundation Model for Physics Simulation",
        "authors": ["Michael McCabe"],
    })
    # Before: "Michael McCabe: Poster: Walrus: A" -- the kind prefix ate the length budget
    # and the cut left a bare "A". The pill carries its own "· poster" tag.
    assert "Poster" not in label
    assert label.startswith("Michael McCabe: Walrus")
    assert not label.endswith((" A", ":"))


def test_the_resources_slide_never_shows_a_raw_markdown_link() -> None:
    note = "*ComplexMCP* — , [OpenReview](https://openreview.net/forum?id=3u1M9wokD7); *SafeSearch*"
    text = _bookends()._strip_md(note)
    assert "](" not in text and "http" not in text and "*" not in text
    assert text == "ComplexMCP, OpenReview; SafeSearch"


def test_the_resources_slide_repairs_a_leaked_macro_name_too() -> None:
    assert _bookends()._strip_md("textttFlashSchNet: Fast") == "FlashSchNet: Fast"


def test_existing_clean_behaviour_is_preserved() -> None:
    assert si._clean_display_text("identify sam- ples from unknown classes") == "identify samples from unknown classes"
    assert si._clean_display_text("short- and long-term") == "short- and long-term"
    assert _bookends()._strip_md("**bold** and `code`") == "bold and code"


def test_bookend_rows_already_in_a_render_spec_are_cleaned_when_rendered() -> None:
    """A render spec written before the cleaners were fixed still holds the raw strings;
    the renderer must show plain text anyway (re-running the chain would cost far more)."""

    rows = sv._render_link_list_items([
        {"name": "ComplexMCP", "detail": "*ComplexMCP* — , [OpenReview](https://openreview.net/forum?id=1); *SafeSearch*"},
        {"name": "textttFlashSchNet: Fast and Accurate", "detail": ""},
    ])
    assert "](" not in rows and "http" not in rows.split("<li>", 1)[1].split("</li>", 1)[0].replace("href", "")
    assert "*" not in rows
    assert "texttt" not in rows
    assert "FlashSchNet: Fast and Accurate" in rows
    assert "ComplexMCP, OpenReview; SafeSearch" in rows


def test_resource_rows_are_cleaned_when_rendered() -> None:
    html = sv._render_resources_section(
        {"resources": [{"title": "textttFlashSchNet: Fast", "note": "see [the paper](https://x.org/p)"}]},
        slide_id="sl_r", index=26, bug_src="", number_html="",
        deck_dir=Path("."), workspace=Path("."),
    )
    assert "texttt" not in html and "](" not in html
    assert "FlashSchNet: Fast" in html and "see the paper" in html


def test_a_star_that_is_part_of_a_name_survives_rendering() -> None:
    rows = sv._render_link_list_items([{"name": "A* search for planning", "detail": ""}])
    assert "A* search for planning" in rows


def test_a_ledger_section_header_note_is_not_one_works_claim() -> None:
    # ICML 2026 deck 02: split on its dash, this note put "Full-read abstract records and
    # paper resources: ComplexMCP" on the slide as a paper's name, with a link fragment.
    pattern = _bookends()._LEDGER_SECTION_NOTE_RE
    assert pattern.match("**Full-read abstract records and paper resources:** *ComplexMCP* — , [OpenReview](https://x)")
    assert not pattern.match("OpenSage — an Agent Development Kit that generates agents")
    assert not pattern.match("A single agent with iterative feedback contributes more than complexity.")


def test_a_tall_events_column_gets_the_compact_layout_so_it_clears_the_source_chips() -> None:
    # ICML 2026 deck 04: five events with two-line titles pushed the last author name
    # over the "Sources" chips. Deck 02's shorter column fit and keeps the default size.
    tall = [{"name": "Oral Session: Oral 5A LLM Training & Inference Efficiency", "detail": "x" * 110}] * 5
    short = [{"name": "Workshop: AI for Law Workshop", "detail": "x" * 70}] * 4
    assert sv._bookend_column_height(tall) > sv.BOOKEND_COLUMN_HEIGHT_PX
    assert sv._bookend_column_height(short) <= sv.BOOKEND_COLUMN_HEIGHT_PX


def test_provenance_id_gate_flags_ids_but_not_prose_with_a_colon():
    # ICML 2026 deck 06: "Position paper: static cues" refused the whole render.
    from conference_pipeline.presentation.structured_video import SVG_RAW_SOURCE_ID_RE
    assert SVG_RAW_SOURCE_ID_RE.search("paper:0CajQNVKyB")
    assert SVG_RAW_SOURCE_ID_RE.search("transcript:016fd...seg00000")
    assert not SVG_RAW_SOURCE_ID_RE.search("Position paper: static cues + priors stand in for motion")


def test_an_event_word_inside_a_hyphenated_title_is_not_an_event():
    # ICML 2026 deck 08: "Multi-Session Agentic Tasks" put a paper on the events rail.
    import importlib.util, sys
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/presentations/add_bookend_slides.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('add_bookend_slides_under_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert not module.EVENT_TEXT_RE.search('Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks')
    assert module.EVENT_TEXT_RE.search('Panel: AI Safety Evaluations')
    assert module.EVENT_TEXT_RE.search('Oral session 3A')
