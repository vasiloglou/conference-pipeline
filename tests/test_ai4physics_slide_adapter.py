"""Shared contract tests for Phase 40 (Reveal.js rendered delivery artifacts).

These 9 tests define the executable contract every later Phase 40 plan must
turn green (Nyquist "test-first" scaffolding, per 40-01-PLAN.md). The adapter
module under test (`scripts/presentations/adapt_ai4physics_slides_to_revealjs.py`)
does not exist yet as of this plan (40-01) -- it is authored by Plan 40-02.
Loading it is deferred to inside each test body via
`importlib.util.spec_from_file_location` (the same convention used throughout
`tests/test_presentation_script_cli.py`) so that pytest collection never
hard-errors: a missing module only fails at `exec_module()` time, inside the
test, not at collection time.

Locked adapter API (see 40-01-PLAN.md Task 2 / 40-RESEARCH.md):
    parse_authored_slides(md_text) -> list[dict]
        dict keys: slide_id, title, body, slide_kind, speaker_notes
    map_slide_kind_to_slide_type(slide_kind) -> str
    emit_canonical_markdown(slides) -> str
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = REPO_ROOT / "scripts" / "presentations"
DECK_DIR = REPO_ROOT / "cross-conference-talks" / "ai4physics2026" / "decks" / "ai4physics-talk"
SLIDES_MD = DECK_DIR / "slides.md"
NARRATIVE_MD = DECK_DIR / "narrative.md"
DELIVERY_MD = DECK_DIR / "DELIVERY.md"


def _load_module(script_name: str, module_name: str) -> Any:
    """Load a scripts/presentations/*.py module via spec_from_file_location.

    Mirrors `load_presentation_script()` in tests/test_presentation_script_cli.py.
    Deferred to call time (not import time) so a not-yet-existing script (the
    adapter, before Plan 40-02) fails inside the test body, not at collection.
    """
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_adapter() -> Any:
    return _load_module(
        "adapt_ai4physics_slides_to_revealjs.py",
        "phase40_adapt_ai4physics_slides_to_revealjs",
    )


def _load_renderer() -> Any:
    return _load_module("md_to_revealjs.py", "phase40_md_to_revealjs")


def _load_pdf_exporter() -> Any:
    return _load_module("export_revealjs_pdf.py", "phase40_export_revealjs_pdf")


def _load_polish() -> Any:
    return _load_module(
        "apply_ai4physics_revealjs_polish.py",
        "phase40_apply_ai4physics_revealjs_polish",
    )


def _render_real_deck(tmp_path: Path) -> dict[str, Any]:
    """Compose adapter + existing renderer on the real 49-slide deck.

    The intermediate canonical markdown is colocated with the real
    `slides.md` (same directory) so the deck's `../../assets/...`-relative
    figure references resolve exactly as they would in production --
    `md_to_revealjs.py` resolves image paths relative to the *input file's*
    directory (base_dir = input_path.parent; see 40-RESEARCH.md Pitfall 3).
    Without this, figures fail to inline and slide markers/zoom-companion
    counts silently diverge from real rendering behavior. The intermediate
    file is always removed afterward and `slides.md` itself is never
    touched. presentation.html and the slide-mapping JSON are written to
    tmp_path (isolated, disposable outputs).
    """
    adapter = _load_adapter()
    renderer = _load_renderer()

    md_text = SLIDES_MD.read_text(encoding="utf-8")
    slides = adapter.parse_authored_slides(md_text)
    canonical_text = adapter.emit_canonical_markdown(slides)

    canonical_path = DECK_DIR / "_phase40_test_canonical_slides.md"
    canonical_path.write_text(canonical_text, encoding="utf-8")
    try:
        output_html = tmp_path / "presentation.html"
        mapping_path = tmp_path / "presentation_slide_mapping.json"
        renderer.build_presentation(
            canonical_path,
            output_html,
            slide_mapping_path=mapping_path,
        )

        return {
            "slides": slides,
            "canonical_text": canonical_text,
            "html": output_html.read_text(encoding="utf-8"),
            "output_html_path": output_html,
            "mapping": json.loads(mapping_path.read_text(encoding="utf-8")),
        }
    finally:
        canonical_path.unlink(missing_ok=True)


def _delivery_slide_id_order() -> list[str]:
    """Parse DELIVERY.md Section 1's backtick-wrapped Slide ID column, in row order."""
    text = DELIVERY_MD.read_text(encoding="utf-8")
    return [
        slide_id
        for _, slide_id in re.findall(
            r"^\|\s*(\d+)\s*\|\s*`([a-z0-9-]+)`\s*\|", text, re.MULTILINE
        )
    ]


# ---------------------------------------------------------------------------
# REND-01: adapter unit tests (synthetic fixtures, no real-deck dependency)
# ---------------------------------------------------------------------------


def test_adapter_maps_slide_kind_to_slide_type():
    adapter = _load_adapter()

    assert adapter.map_slide_kind_to_slide_type("title") == "title"
    assert adapter.map_slide_kind_to_slide_type("section") == "section"
    assert adapter.map_slide_kind_to_slide_type("native-divider") == "section"
    for other_kind in ("references", "trend-findings", "cost-breakdown"):
        assert adapter.map_slide_kind_to_slide_type(other_kind) == "content"


def test_adapter_preserves_multiline_speaker_notes():
    adapter = _load_adapter()

    fixture = """<!-- slide_id: multiline-notes-demo -->
# Multiline Notes Demo

Body text for the slide.

<!--
slide_kind: references
speaker_notes: |
  First sentence of the note.
  Second sentence continues the thought.
  Third sentence closes it out.
-->
"""
    slides = adapter.parse_authored_slides(fixture)
    assert len(slides) == 1
    notes = slides[0]["speaker_notes"]
    assert "First sentence of the note." in notes
    assert "Second sentence continues the thought." in notes
    assert "Third sentence closes it out." in notes

    canonical = adapter.emit_canonical_markdown(slides)
    # Capture everything after "SPEAKER_NOTES:" up to the next top-level
    # KEY: line or the comment's close, so this assertion holds whether the
    # adapter flattens the note onto one physical line or preserves
    # continuation lines -- either way, nothing may be truncated at the
    # note's *first* line.
    notes_match = re.search(
        r"SPEAKER_NOTES:\s*(.*?)\s*(?:\n[A-Z_]+:|-->)", canonical, re.DOTALL
    )
    assert notes_match, "emitted canonical markdown must carry a SPEAKER_NOTES field"
    emitted_notes = notes_match.group(1)
    assert "First sentence of the note." in emitted_notes
    assert "Second sentence continues the thought." in emitted_notes
    assert "Third sentence closes it out." in emitted_notes


def test_adapter_handles_colon_in_free_text_value():
    adapter = _load_adapter()

    # A YAML parser raises `yaml.scanner.ScannerError: mapping values are not
    # allowed here` on this exact shape -- an unescaped ": " inside a plain
    # (non-block-scalar) free-text value -- see 40-RESEARCH.md Pitfall 1
    # (reproduced on blocks 30 and 38 of the real slides.md). This must not
    # raise, and the value must round-trip with the internal colon intact.
    fixture = """<!-- slide_id: colon-value-demo -->
# Colon Value Demo

Body text for the slide.

<!--
slide_kind: references
speaker_notes: The ratio is roughly 3: to 1 in favor of the structured approach.
-->
"""
    slides = adapter.parse_authored_slides(fixture)
    assert len(slides) == 1
    assert slides[0]["speaker_notes"].strip() == (
        "The ratio is roughly 3: to 1 in favor of the structured approach."
    )


# ---------------------------------------------------------------------------
# REND-02: end-to-end smoke tests against the real 49-slide deck
# ---------------------------------------------------------------------------


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md")
def test_end_to_end_produces_49_slide_mapping_entries(tmp_path: Path):
    rendered = _render_real_deck(tmp_path)
    assert len(rendered["mapping"]) == 49


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md")
def test_end_to_end_title_and_divider_markers(tmp_path: Path):
    rendered = _render_real_deck(tmp_path)
    html = rendered["html"]

    # Count `class="..."` HTML *attribute* occurrences, not CSS rule text --
    # CSS selectors in the template use `.slide-title { ... }` /
    # `.slide-section { ... }`, never the literal `class="` prefix, so this
    # substring only ever matches real <section> elements.
    assert html.count('class="slide-title') == 1

    # 1 `slide_kind: section` + 4 `slide_kind: native-divider` slides map to
    # the section-divider layout. The prefix match also covers the
    # `slide-section-with-figure` variant emitted for the one section slide
    # that carries an inline SVG (the renderer's existing, documented
    # behavior -- see md_to_revealjs.py `_render_section`).
    assert html.count('class="slide-section') == 5


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md")
def test_end_to_end_speaker_notes_present(tmp_path: Path):
    rendered = _render_real_deck(tmp_path)
    assert rendered["html"].count('<aside class="notes">') > 0


# ---------------------------------------------------------------------------
# REND-03: PDF export (skips gracefully when Playwright is unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md")
def test_pdf_export_page_count_matches_slides(tmp_path: Path):
    pytest.importorskip("playwright")
    import pypdf

    rendered = _render_real_deck(tmp_path)
    output_html = rendered["output_html_path"]
    output_pdf = tmp_path / "presentation.pdf"

    pdf_exporter = _load_pdf_exporter()
    pdf_exporter.export_pdf(output_html, output_pdf)

    assert output_pdf.is_file()
    assert output_pdf.stat().st_size > 0

    reader = pypdf.PdfReader(str(output_pdf))
    page_count = len(reader.pages)

    # Content/section slides with a single inline figure emit an extra
    # "zoom companion" <section> in the HTML (documented renderer behavior,
    # see md_to_revealjs.py module docstring). Reveal's print-pdf mode has no
    # CSS suppressing those sections, so the exported PDF renders every
    # <section> -- exclude the zoom-companion pages to compare 1:1 against
    # the 49 authored slides.
    zoom_companion_count = rendered["html"].count('class="slide-figure-zoom')
    assert page_count - zoom_companion_count == 49


# ---------------------------------------------------------------------------
# REND-04: narrative.md reconciliation (pure text assertions, no rendering)
# ---------------------------------------------------------------------------


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/narrative.md")
def test_narrative_reconciliation_no_dangling_cut_refs():
    text = NARRATIVE_MD.read_text(encoding="utf-8")

    headers = {
        slide_id: number
        for number, slide_id in re.findall(
            r"^## Slide (\S+)\s*—.*?\(`([a-z0-9-]+)`\)", text, re.MULTILINE
        )
    }
    assert headers.get("anticipated-direction") == "48", (
        "the anticipated-direction tail header must be renumbered to "
        "'Slide 48' after the two section-physics-* cuts collapsed the "
        "tail by two positions (see DELIVERY.md Section 1)"
    )
    assert headers.get("conference-knowledge-to-money") == "49", (
        "the conference-knowledge-to-money tail header must be renumbered "
        "to 'Slide 49'"
    )
    assert not re.search(r"^## Slide 50\b", text, re.MULTILINE), (
        "no 'Slide 50' header may remain; the delivered deck has only 49 "
        "slides"
    )

    note_match = re.search(r"\*\*Note on scope.*?\n\n", text, re.DOTALL)
    assert note_match, (
        "narrative.md must retain the 'Note on scope' cut-history block "
        "documenting the section-physics-* removal"
    )
    remainder = text[: note_match.start()] + text[note_match.end() :]
    assert "section-physics-frontiers" not in remainder
    assert "section-physics-trust" not in remainder
    assert "section-physics-frontiers" in note_match.group(0)
    assert "section-physics-trust" in note_match.group(0)


# ---------------------------------------------------------------------------
# DONE-01: cross-artifact slide-order consistency
# ---------------------------------------------------------------------------


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/slides.md")
def test_slide_order_consistent_across_artifacts(tmp_path: Path):
    rendered = _render_real_deck(tmp_path)

    authored_slide_ids = [slide["slide_id"] for slide in rendered["slides"]]
    delivery_slide_ids = _delivery_slide_id_order()

    assert len(authored_slide_ids) == 49
    assert len(delivery_slide_ids) == 49
    assert authored_slide_ids == delivery_slide_ids, (
        "slides.md's authored slide_id order must match DELIVERY.md "
        "Section 1's position table order exactly"
    )

    mapping_keys_in_order = sorted(int(key) for key in rendered["mapping"])
    assert mapping_keys_in_order == list(range(1, 50)), (
        "the slide-mapping JSON must key exactly the 49 sequential slide "
        "positions the adapter assigned, in order"
    )


def test_polish_strips_internal_provenance_captions():
    """The render polish removes internal figure-provenance / production
    scaffolding <text> captions (the Phase 38.5 footnotes baked into the SVGs:
    'Create-our-own ...', 'Source: conferences/...', repo `.md` references,
    'no paper figure cropped', 'not derived from the conference analysis')
    while leaving audience-facing figure annotations intact, idempotently
    (REND-02 delivery cleanliness)."""
    polish = _load_polish()

    strip = [
        # Create-our-own provenance prefix.
        '<text x="90" y="668" fill="#6a6a88">Create-our-own (metrics-pending '
        "slide). Sources: iclr2026/icml2026 corpus + transcription manifests. "
        "Some hours size-estimated; figures for the reconciliation pass.</text>",
        # Repo path + `.md` files + "exercise's own findings".
        '<text x="60" y="700" fill="#6a6a88">Source: conferences/iclr2026/'
        "ai4science_commercialization (THESIS.md, EXECUTIVE_BRIEF.md). Numbers "
        "are the exercise's own findings.</text>",
        # Figure-construction note + a repo `.md` reference.
        '<text x="60" y="710" fill="#6a6a88">Speaker\'s forward-looking thesis '
        "(speculative). Company examples cited in narrative.md; not derived from "
        "the conference analysis.</text>",
        # "no paper figure cropped" construction note.
        '<text x="90" y="682" fill="#6a6a88">Concept synthesized from dd03. '
        "Original redraw &#8212; no paper figure cropped.</text>",
    ]
    keep = [
        # Real figure annotations / audience-facing content -- must be preserved.
        '<text x="600" y="536" fill="#8a8aa5">Qwen 122B option wants a ~$5,000 '
        "Mac (speaker estimate).</text>",
        '<text x="10" y="20" fill="#8a8aa5">ML &amp; Physical Sciences panel '
        "&#183; asked from the stage</text>",
        '<text x="10" y="40" fill="#ffffff">hyperbolic PDEs</text>',
    ]
    html = "<svg>" + "".join(strip + keep) + "</svg>"

    cleaned, removed = polish.strip_provenance_captions(html)

    assert removed == len(strip)
    for marker in ("Create-our-own", "conferences/", "narrative.md", "cropped"):
        assert marker not in cleaned
    # Audience-facing annotations must survive.
    assert "Qwen 122B option wants a ~$5,000 Mac" in cleaned
    assert "asked from the stage" in cleaned
    assert "hyperbolic PDEs" in cleaned
    # Idempotent: re-running finds nothing left to strip.
    assert polish.strip_provenance_captions(cleaned)[1] == 0


# ---------------------------------------------------------------------------
# Presenter cheat sheet: lossless-reflow regression coverage
# ---------------------------------------------------------------------------


def _synthetic_reveal_html() -> str:
    """A minimal HTML body mirroring the real deck's end-of-body layout:
    a Reveal.initialize stub script followed by the single closing </body>."""
    return (
        "<html><head></head><body>\n"
        '<div class="reveal"><div class="slides">'
        '<section data-markdown-slide="1">One</section>'
        '<section data-markdown-slide="8">Eight</section>'
        '<section data-markdown-slide="8">Eight Zoom</section>'
        '<section data-markdown-slide="49">FortyNine</section>'
        "</div></div>\n"
        "<script>\n"
        "var hideControls = window.location.search.indexOf('hidecontrols') !== -1;\n"
        "Reveal.initialize({ slideNumber: !hideControls });\n"
        "</script>\n"
        "</body></html>"
    )


def test_polish_injects_single_slide_number_script():
    polish = _load_polish()

    html = polish.inject_slide_number(_synthetic_reveal_html())

    assert html.count('id="ai4physics-slide-number"') == 1
    script_index = html.index('id="ai4physics-slide-number"')
    body_close_index = html.index("</body>")
    assert script_index < body_close_index


def test_slide_number_script_references_data_markdown_slide():
    polish = _load_polish()

    html = polish.inject_slide_number(_synthetic_reveal_html())
    script_start = html.index('<script id="ai4physics-slide-number">')
    script_end = html.index("</script>", script_start) + len("</script>")
    block = html[script_start:script_end]

    assert "data-markdown-slide" in block
    assert "Reveal.configure" in block


def test_polish_slide_number_injection_idempotent():
    polish = _load_polish()

    once = polish.inject_slide_number(_synthetic_reveal_html())
    twice = polish.inject_slide_number(once)

    assert twice.count('id="ai4physics-slide-number"') == 1


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/presentation.html")
def test_committed_presentation_html_has_slide_number_script():
    """This test goes RED until the committed presentation.html is
    regenerated with the new polish step (Task 2 of the quick-260714-dsu
    plan) -- that is expected and is the Nyquist link between the two
    tasks."""
    html_path = DECK_DIR / "presentation.html"
    html = html_path.read_text(encoding="utf-8")

    assert html.count('id="ai4physics-slide-number"') == 1
    script_start = html.index('<script id="ai4physics-slide-number">')
    script_end = html.index("</script>", script_start) + len("</script>")
    assert "data-markdown-slide" in html[script_start:script_end]


def _load_cheatsheet_builder() -> Any:
    return _load_module(
        "build_ai4physics_presenter_cheatsheet.py",
        "phase40_build_ai4physics_presenter_cheatsheet",
    )


def test_cheatsheet_field_parser_handles_wrapped_and_qualified_headers():
    """A bold field header may (a) carry a parenthetical qualifier
    ("**Script (spoken with verified fills...):**") or (b) soft-wrap its
    descriptive text across two physical lines ("**Provenance (...long...\\n
    ...):**") -- both are real shapes present in narrative.md (slides 6/7/49).
    Either shape must still be recognized as a field boundary and normalized
    to its canonical name, or the field's body silently swallows every
    subsequent field (including Script) into one unparsed blob."""
    builder = _load_cheatsheet_builder()

    block = (
        "\n**Establishes:** The point of this slide.\n\n"
        "**Script (spoken with verified fills; keep the honesty flags):**\n\n"
        "> Sentence one. Sentence two.\n\n"
        "**Provenance (verified numbers; several manifests live in the "
        "populated main\ncheckout, gitignored on-disk here -- see note):**\n\n"
        "- A provenance bullet that must not leak into Script.\n\n"
        "**Figure/enrichment leads:** Not rendered by the cheat sheet.\n"
    )

    fields = builder._extract_fields(block)

    assert set(fields.keys()) == {
        "Establishes",
        "Script",
        "Provenance",
        "Figure/enrichment leads",
    }
    assert "Sentence one. Sentence two." in fields["Script"]
    # The Provenance field's content must never appear inside Script's body --
    # this is exactly the leak this test guards against.
    assert "provenance bullet" not in fields["Script"]


def test_cheatsheet_sentence_split_preserves_every_character():
    """Splitting a Script paragraph into one bullet per sentence must be a
    pure re-wrap: rejoining the bullets with spaces and collapsing whitespace
    must reproduce the original text exactly (no word added, removed, or
    altered -- including a trailing quotation mark on a sentence that ends
    mid-quoted-excerpt, which an earlier version of this splitter silently
    dropped)."""
    builder = _load_cheatsheet_builder()

    paragraph = (
        'And the precondition was said plainly: "at least with DFT '
        'simulation data, we are no longer in the low-data regime." When '
        "data is abundant, hard priors buy you less."
    )

    bullets = builder._script_to_bullets(paragraph)

    assert bullets[0].endswith('low-data regime."')  # closing quote preserved
    rejoined = re.sub(r"\s+", " ", " ".join(bullets)).strip()
    assert rejoined == re.sub(r"\s+", " ", paragraph).strip()


@pytest.mark.repo_data("cross-conference-talks/ai4physics2026/decks/ai4physics-talk/narrative.md")
def test_cheatsheet_zero_loss_across_real_deck_narrative(tmp_path: Path):
    """End-to-end regression: every one of the real narrative.md's 49 slides
    must retain a Script field, and reflowing it into bullets must not lose
    or alter any character (REND-04 companion-artifact integrity)."""
    builder = _load_cheatsheet_builder()

    narrative_text = NARRATIVE_MD.read_text(encoding="utf-8")
    matches = list(builder._SLIDE_BLOCK_RE.finditer(narrative_text))
    assert len(matches) == 49

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip()

    for i, m in enumerate(matches):
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(narrative_text)
        fields = builder._extract_fields(narrative_text[m.end():block_end])
        assert "Script" in fields, f"slide {m.group('num')} missing Script field"

        unwrapped = builder._unwrap_blockquote(fields["Script"])
        bullets = builder._script_to_bullets(unwrapped)
        assert norm(unwrapped) == norm(" ".join(bullets)), (
            f"slide {m.group('num')} lost or altered content during bulleting"
        )
