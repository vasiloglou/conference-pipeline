"""Tests for extract_numbered_items' two supported executive-summary formats.

Live discovery: rebuilding the ICML 2026 executive-overview decks against the
Phase 14 recovery's refreshed executive_summary.md raised
"executive_summary_numbered_items_missing" -- the parser only recognized an
"N) Title\\n<body>" list, but stage-8 synthesis.executive_summary (both the
pre-recovery baseline and the refreshed version) actually emits
"**N. Title.** Body..." bold-numbered paragraphs. The parser had likely never
matched the real synthesis output format.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_module():
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "presentations"
        / "generate_executive_overview_decks.py"
    )
    spec = importlib.util.spec_from_file_location("generate_executive_overview_decks", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BOLD_FORMAT_TEXT = """# Executive Summary

## Cross-topic conclusions

**1. Constraint injection is displacing pure data-driven fitting.** The clearest signal is structure-preserving approximation across several fields.

**2. Efficiency has become an allocation problem, not a compression problem.** No single optimization dominates; gains come from co-design.

**3. Safety and alignment are lifecycle properties, not stable scalars.** Alignment can deteriorate under benign customization.

## Largest surprises

**1. This numbered list must not leak into the cross-topic items above.**
"""

N_PAREN_FORMAT_TEXT = """# Executive Summary

## Cross-topic conclusions

1) Constraint injection is displacing pure data-driven fitting
- The clearest signal is structure-preserving approximation across several fields.

2) Efficiency has become an allocation problem, not a compression problem
- No single optimization dominates; gains come from co-design.

## Largest surprises

1) This numbered list must not leak into the cross-topic items above
- unrelated body text
"""


def test_extract_numbered_items_parses_bold_markdown_format():
    module = load_module()
    items = module.extract_numbered_items(BOLD_FORMAT_TEXT)
    assert len(items) == 3
    titles = [title for title, _body in items]
    assert titles == [
        "Constraint injection is displacing pure data-driven fitting.",
        "Efficiency has become an allocation problem, not a compression problem.",
        "Safety and alignment are lifecycle properties, not stable scalars.",
    ]
    assert "structure-preserving approximation" in items[0][1]


def test_extract_numbered_items_bold_format_stops_before_next_heading():
    module = load_module()
    items = module.extract_numbered_items(BOLD_FORMAT_TEXT)
    assert not any("must not leak" in title for title, _body in items)


def test_extract_numbered_items_still_parses_n_paren_format():
    module = load_module()
    items = module.extract_numbered_items(N_PAREN_FORMAT_TEXT)
    assert len(items) == 2
    titles = [title for title, _body in items]
    assert titles == [
        "Constraint injection is displacing pure data-driven fitting",
        "Efficiency has become an allocation problem, not a compression problem",
    ]


def test_extract_numbered_items_n_paren_format_stops_before_next_heading():
    module = load_module()
    items = module.extract_numbered_items(N_PAREN_FORMAT_TEXT)
    assert not any("must not leak" in title for title, _body in items)


def test_extract_numbered_items_returns_empty_for_unnumbered_text():
    module = load_module()
    assert module.extract_numbered_items("# Executive Summary\n\nNo numbered list here.\n") == []


BINDING_DIGESTS_TEXT = """# Executive Summary

## Binding digests

- analysis_manifest_content_sha256: `50d481daf8ddfc133853fa714b417c1cd2af71ce2bda79bb779fd9df39cc7564`
- traceability_report_sha256: `044fa402788832848c7515d23a90e066bde5c96ff0667cdb3d089915f665c2f3`

## Orientation

This synthesis draws on twelve fully synthesized topic reports.

## Cross-topic conclusions

**1. Constraint injection is displacing pure data-driven fitting.** Body text here.
"""


def test_strip_binding_digests_section_removes_hash_block():
    module = load_module()
    stripped = module.strip_binding_digests_section(BINDING_DIGESTS_TEXT)
    assert "Binding digests" not in stripped
    assert "sha256" not in stripped
    assert "## Orientation" in stripped
    assert "## Cross-topic conclusions" in stripped


def test_strip_binding_digests_section_preserves_downstream_parsing():
    """Live discovery: without stripping this section, the "Leadership Actions"
    slide's fallback bullet (first_sentence(executive_text, ...), used when no
    "Recommended immediate actions" section exists) surfaced a literal
    "# Executive Summary ## Binding digests - analysis_manifest_content_sha256:
    50d481..." string on a rendered slide -- caught by the visual QC gate as
    illegible/raw-markdown content, blocking the whole presentation stage."""
    module = load_module()
    stripped = module.strip_binding_digests_section(BINDING_DIGESTS_TEXT)
    assert module.extract_numbered_items(stripped) == [
        (
            "Constraint injection is displacing pure data-driven fitting.",
            "Body text here.",
        )
    ]
    fallback_bullet = module.first_sentence(stripped, fallback="FALLBACK")
    assert "sha256" not in fallback_bullet
    assert "Binding digests" not in fallback_bullet


def test_first_sentence_stops_at_quote_closed_period():
    """Live discovery, round 3: first_sentence()'s regex required terminal
    punctuation to be followed directly by whitespace, so a sentence ending
    on a quoted phrase ('...outcome."') was skipped past in favor of the
    NEXT sentence's punctuation -- producing a "first sentence" 2-3x longer
    than intended, which then got mangled by downstream character-limit
    truncation into a garbled fragment (caught by visual QC on the ICML 2026
    executive-summary canary: "models "as." and similar mid-quote cutoffs)."""
    module = load_module()
    text = (
        'the report delivers a consequential message: alignment "can deteriorate '
        'under benign customization and become difficult to diagnose." Crucially, '
        '"greater reasoning capability is no longer assumed to improve safety."'
    )
    result = module.first_sentence(text, fallback="FALLBACK")
    assert result == (
        'the report delivers a consequential message: alignment "can deteriorate '
        'under benign customization and become difficult to diagnose."'
    )
    assert "Crucially" not in result


def test_first_sentence_still_stops_at_plain_period():
    module = load_module()
    text = "First sentence here. Second sentence should not be included."
    assert module.first_sentence(text, fallback="FALLBACK") == "First sentence here."


def test_svg_figure_bullet_lines_never_emits_ellipsis():
    """Live discovery: the executive figure cards render full synthesis
    sentences (up to ~480 chars) as wrapped list lines. wrap_svg_text(...,
    max_lines=2) truncated them mid-sentence with "..." even when the card had
    visible empty space -- the vision QC gate reads that as clipped text.
    svg_figure_bullet_lines shortens to a complete sentence/clause instead, so
    the rendered figure text is always a finished thought with no ellipsis."""
    module = load_module()
    long_bullet = (
        'the safety alignment governance report delivers perhaps the most '
        'consequential leadership message: alignment "can deteriorate under '
        'benign customization, be bypassed through optimization or adversarial '
        'fine-tuning, fail differently across languages, and become difficult '
        'to diagnose when an agent conceals failure behind an apparently '
        'successful outcome."'
    )
    lines = module.svg_figure_bullet_lines(long_bullet, wrap_chars=80, max_lines=4)
    assert len(lines) <= 4
    joined = " ".join(lines)
    assert "..." not in joined
    assert "…" not in joined
    # Ends on a complete word, not a dangling article/preposition.
    assert joined.split()[-1].strip(".,;:!?").lower() not in {"and", "the", "to", "of", "or", "a"}


def test_svg_figure_bullet_lines_keeps_short_bullet_whole():
    module = load_module()
    short_bullet = "Constraint injection is displacing pure data-driven fitting."
    lines = module.svg_figure_bullet_lines(short_bullet, wrap_chars=80, max_lines=4)
    assert " ".join(lines) == short_bullet
    assert "..." not in " ".join(lines)


def test_compact_signal_label_preserves_contrastive_ending():
    """Live discovery: the signal-grid SVG figure renders each label via
    wrap_svg_text(label, 40, max_lines=2) -- an 80-char capacity -- but
    compact_signal_label() hard-cut to the first 7 words whenever the label
    exceeded 52 chars, with no ellipsis. For this conference's frequent
    "X, not Y" contrastive titles, that silently dropped the clause carrying
    the actual meaning: "Safety and alignment are lifecycle properties, not
    stable scalars" (68 chars, 10 words) was cut to "...not", losing "stable
    scalars" entirely -- caught by visual QC as text_clipped."""
    module = load_module()
    title = "Safety and alignment are lifecycle properties, not stable scalars"
    result = module.compact_signal_label(title)
    assert result == title
    assert "stable scalars" in result


def test_compact_signal_label_still_trims_when_over_render_budget():
    module = load_module()
    title = (
        "This is a deliberately very long synthesis title that clearly exceeds "
        "the eighty character rendering budget of the signal grid figure by a lot"
    )
    result = module.compact_signal_label(title)
    assert len(result) <= 76
    assert result != title


def test_first_sentence_strips_paper_abstract_citation():
    """_content_sentences() already strips citation clutter before splitting
    into sentences; first_sentence() didn't, so a (paper_abstract:XXXX) token
    leaked verbatim into evidence_caption/narrative -- flagged by visual QC
    as an exposed internal reference artifact."""
    module = load_module()
    text = "This shows a result (paper_abstract:0xtMUL0eiF)."
    assert module.first_sentence(text, fallback="FALLBACK") == "This shows a result."


def test_first_sentence_strips_research_thread_id():
    """Same leak for the longer 'run_id:topic_id:hash' research-thread id
    format, e.g. (icml2026-v4.0-live-production-20260711:05_scaling_
    efficiency_foundations:248e4ee99a5c) -- not caught by the
    paper_abstract/analysis patterns, and long enough as a single
    unbreakable token to overflow a narrow caption column."""
    module = load_module()
    text = (
        "the report (icml2026-v4.0-live-production-20260711:05_scaling_efficiency_"
        'foundations:248e4ee99a5c) marks the shift "from scaling by uniform expansion '
        'toward scaling by selective allocation and systems co-design."'
    )
    result = module.first_sentence(text, fallback="FALLBACK")
    assert "icml2026" not in result
    assert "248e4ee99a5c" not in result
    assert result == (
        'the report marks the shift "from scaling by uniform expansion toward '
        'scaling by selective allocation and systems co-design."'
    )


def test_strip_binding_digests_section_is_noop_without_the_section():
    module = load_module()
    text = "# Executive Summary\n\n## Orientation\n\nNo digests here.\n"
    assert module.strip_binding_digests_section(text) == text


def test_section_paragraph_text_extracts_orientation_prose():
    """Live discovery, round 2: stripping Binding digests alone wasn't enough --
    first_sentence(executive_text, ...) still ran over the raw "# Executive
    Summary\\n\\n## Orientation\\n\\n..." text, and clean_markdown_text() doesn't
    strip heading marker characters, so the fallback bullet became
    "# Executive Summary ## Orientation This synthesis draws on...". Extracting
    just the Orientation section's own prose (heading-free) fixes it at the
    source rather than patching clean_markdown_text globally."""
    module = load_module()
    text = (
        "# Executive Summary\n\n"
        "## Orientation\n\n"
        "This synthesis draws on twelve topic reports. It covers several fields.\n\n"
        "## Cross-topic conclusions\n\n"
        "**1. Title.** Body.\n"
    )
    orientation = module._section_paragraph_text(text, r"^##\s+Orientation[^\n]*")
    assert orientation == "This synthesis draws on twelve topic reports. It covers several fields."
    assert "#" not in orientation


def test_section_paragraph_text_returns_empty_when_heading_missing():
    module = load_module()
    text = "# Executive Summary\n\n## Cross-topic conclusions\n\n**1. Title.** Body.\n"
    assert module._section_paragraph_text(text, r"^##\s+Orientation[^\n]*") == ""


def test_a_diagram_with_no_nodes_is_refused_not_drawn_as_a_bare_arrow() -> None:
    """CVPR 2026's first executive build shipped 48 figure cards that were an
    empty panel with one arrow: the decision and long-deck slides had no diagram
    spec, and the renderer drew the skeleton anyway. Every gate passed it."""

    import pytest

    module = load_module()
    with pytest.raises(ValueError, match="no nodes"):
        module._diagram_parts("comparison", [], "caption", "sources")
    with pytest.raises(ValueError, match="no nodes"):
        module._diagram_parts("pipeline", ["", "  "], "", "")
    # A real spec still renders.
    assert module._diagram_parts("pipeline", ["Input", "Encoder", "Output"], "A pipeline", "")


def test_executive_figures_render_through_the_same_compiler_as_topic_decks(tmp_path) -> None:
    """A conference had two live drawing paths: topic decks went through the
    figure compiler and its geometry gate, while executive figures were drawn
    by this script and never gated at all."""

    module = load_module()
    out = tmp_path / "fig.svg"
    ok = module.write_compiled_figure(
        out,
        title="Compression targets tokens and caches",
        diagram={
            "kind": "pipeline",
            "caption": "how compression is applied",
            "nodes": ["Input tokens", "Prune", "Quantize", "Decoder"],
        },
        slide_id="exec-s001",
    )
    assert ok and out.is_file()
    svg = out.read_text(encoding="utf-8")
    # The compiler's own document, not this script's hand-drawn card.
    assert 'viewBox="0 0 1280 720"' in svg
    assert "fc-title" in svg

    # A sketch with no nodes is refused rather than drawn as an empty card,
    # and the caller falls back instead of shipping nothing.
    assert not module.write_compiled_figure(
        tmp_path / "empty.svg", title="T", diagram={"kind": "pipeline", "nodes": []},
        slide_id="exec-s002",
    )
